import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell # GRUCell, MultiRNNCell not used
from tensorflow.keras.layers import GRUCell, LSTMCell, Bidirectional, LSTM
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest
import numpy as np


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, item_fnum, num_cat, mu,
                 max_norm=None, multi_hot=False):
        '''
        feature_siez: (int) 10244
        eb_dim: (int) 16
        item_fnum: (int) 2
        num_cat: (int) as len(cate)==20 for 'ml-20m'
        '''
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            if multi_hot: # i.e. if MovieLens-20M, there're 20 genres (cates) instead of 5 topics (cates), num_cat==len(cid_dict)
                self.feat_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum + num_cat - 1], name='seq_feat_ph') # [None, 20, 2+20-1]
            else:
                self.feat_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum], name='seq_feat_ph') # not num_cat relevant [None, 20, 2]
            self.user_behavior_ph = tf.placeholder(tf.int32, [None, max_seq_len * num_cat], name='user_behavior_ph') # [None, max_behavior_len*5] or [None, max_b_l*20] for ml-20m
            self.behavior_len_ph = tf.placeholder(tf.int32, [None, num_cat], name='behavior_len_ph') # depends on the num of cate, cuz each cate has a behavior_seq
            self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
            self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
            self.items_div = tf.placeholder(tf.float32, [None, max_time_len, num_cat], name='items_div')

            self.lr = tf.placeholder(tf.float32, [])
            self.reg_lambda = tf.placeholder(tf.float32, [])
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.max_time_len = max_time_len
            self.hidden_size = hidden_size
            self.emb_dim = eb_dim
            self.item_fnum = item_fnum
            self.num_cat = num_cat
            self.mu = mu # the rel&div tradeoff param lambda, set in DCM, here it should only be a indicator that under which scenario the model RAPID has been trained
            self.max_grad_norm = max_norm

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim], # feature_size==10244, em_dim==16 [10245, 16]
                                           initializer=tf.truncated_normal_initializer)
            if multi_hot:
                self.cate_emb_mtx = tf.get_variable('cate_emb_mtx', [num_cat, eb_dim], # [20, 16]
                                                    initializer=tf.truncated_normal_initializer)

                item_multi_hot = tf.cast(tf.reshape(self.feat_ph, [-1, item_fnum + num_cat - 1])[:, 1:], tf.float32) # tf.reshape([None, 20, 21], [-1, 21])[:,1:]
                self.item_embed = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.feat_ph[:, :, 0]), [-1, eb_dim])
                self.cate_embed = tf.reshape(tf.matmul(item_multi_hot, self.cate_emb_mtx), [-1, eb_dim])
                self.cate_embed = self.cate_embed / (tf.reduce_sum(item_multi_hot, axis=-1, keepdims=True) + 1e-9)
                self.item_seq = tf.concat([self.item_embed, self.cate_embed], -1)

            else:
                self.item_seq = tf.gather(self.emb_mtx, self.feat_ph) # ([10245, 16], [None, 20, 2])

            self.user_seq = tf.gather(self.emb_mtx, self.user_behavior_ph) # emb_mtx==[10246,16], user_behavior_ph==[None,max_beh_len*20] for ml-20m
            # self.user_seq = [?, 100, 16]
            # tf.gather(): to select specific slices from a tensor along a specified axis, typically by using indices. used to gather embeddings or particular values based on index position from a larger tensor.
            # tf.gather(params, indices, axis) axis=0 by default -> the first dim. the indices shows which ele (row)'s embedding_mtx to extract

            self.item_seq = tf.reshape(self.item_seq, [-1, max_time_len, item_fnum * eb_dim]) # (self.item_seq[item_embed, cate_embed], [-1, 20, 32])

    def build_fc_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
        score = tf.nn.softmax(fc3)
        score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
        # output
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        self.y_pred = seq_mask * score

    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss # loss is just log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name: # if not biased nor 'embedding'(?) in the trainable_var
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def build_mseloss(self):
        self.loss = tf.losses.mean_squared_error(self.label_ph, self.y_pred)
        # regularization term
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm != None:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)


    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            #Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C) #⭐️
            #K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C) #⭐️
            #V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C) #⭐️
            Q = tf.keras.layers.Dense(num_units, activation=None)(queries)  # (N, T_q, C)
            K = tf.keras.layers.Dense(num_units, activation=None)(keys)  # (N, T_k, C)
            V = tf.keras.layers.Dense(num_units, activation=None)(keys)  # (N, T_k, C)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h) #⭐️
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h) 


            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False): # listwise relevance estimator 
        # inp (item_seq) has already been unstacked
        with tf.variable_scope(scope, reuse=reuse):
            bidirectional_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
            )
            #print(inp)
            #print(len(inp))
            # change inp to [batch_size, time_step, feature_dim]
            # stack again the unstacked tensor
            time_step = len(inp)
            # inp is a list of tensors with shape [?, 52]
            new_inp = tf.stack(inp, axis=1) # [batch_size, time_step, fea_dim]
            outputs, fw_h, fw_c, bw_h, bw_c = bidirectional_lstm(new_inp) # input tensor with shape [batch_size, time_steps, feature_dim]

            # final states for fw and bw cells
            state_fw = (fw_h, fw_c)
            state_bw = (bw_h, bw_c)
            '''
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw') # forward ⭐️
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw') # backward ⭐️
            # inp: [batch_size, feature_dim] * time_step
            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32') ⭐️
            '''
        return outputs, state_fw, state_bw

    def train(self, sess, batch_data, lr, reg_lambda, keep_prob=0.8): 
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.feat_ph: batch_data[0],
            self.label_ph: batch_data[1],
            self.seq_length_ph: batch_data[2],
            self.user_behavior_ph: batch_data[3], # 200/100 item feature_id for each user, max_behavior_len*num_cate -> 10/5*20
            self.items_div: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda, keep_prob=1, no_print=True):
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.feat_ph: batch_data[0],
            self.label_ph: batch_data[1],
            self.seq_length_ph: batch_data[2],
            self.user_behavior_ph: batch_data[3],
            self.items_div: batch_data[4],
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob
        })
        return pred.reshape([-1, self.max_time_len]).tolist(), pred.reshape([-1, self.max_time_len]).tolist(), \
               label.reshape([-1, self.max_time_len]).tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)


class RAPID(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len, item_fnum, num_cat, mu, map=False,
                 max_norm=None, multi_hot=False, mean_aggregate=False, pure_rnn=False):
        super(RAPID, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                    max_seq_len, item_fnum, num_cat, mu, max_norm, multi_hot)
        with tf.variable_scope('rapid'):
            self.istrain = tf.placeholder(tf.bool, [])
            self.map = map
            self.mean_aggregate = mean_aggregate
            self.item_div_seq = tf.concat([self.item_seq, self.items_div], 2) # [[-1, 20, 32], [None, 20, 20], 2] concatenates along 2-idx (the last idx) item_embed+cate_embed+items_div
            item_seq = tf.unstack(self.item_div_seq, max_time_len, 1) # unstack self.item_div_seq, now item_seq has shape [?,52] * 20 (max_time_len) -> split 3D tensor [?, 20, 52] along time axis
            
            # personalized diversity estimator
            if self.mean_aggregate:
                # reshaping + averaging along specific dims
                self.user_seq = tf.reshape(tf.reduce_mean(tf.reshape(self.user_seq, [-1, num_cat, max_seq_len, eb_dim]), axis=2), (-1, num_cat, eb_dim)) 

                # diversity gain
                attended_cat = self.multihead_attention(self.user_seq, self.user_seq)  # num_cat * emb_size
                attended_cat_bn = tf.reshape(tf.layers.batch_normalization(
                    inputs=attended_cat), [-1, self.num_cat * eb_dim]) # each topic has a user_behavior_seq
            else: # self.user_seq = [?, 100, 16]
                # reshape self.user_seq, split it into a structure that separates max_seq_len and eb_dim
                self.user_seq = tf.reshape(self.user_seq, [-1, max_seq_len, eb_dim]) # max_seq_len==10/5, eb_dim==16
                # self.user_seq = tf.unstack(self.user_seq, max_seq_len, 1) 
                length = tf.reshape(self.behavior_len_ph, [-1]) # flatten the tensor to ?*20
                # old version LSTMCell requires more manual input handling, e.g. tf.nn.dynamic_rnn or tf.nn.static_rnn to properly manage input seq
                # new version LSTMCell is intended to be wrapped in tf.keras.layers.RNN or tf.keras.layers.LSTM, can handle input shapes automatically. Uses tf.RaggedTensor 不规则 or seq masking automatically for variable-length seq.
                # use Keras approach for managing variable-length seq (automatic handling of sequence_length)
                
                # behave_outputs, _ = tf.nn.dynamic_rnn(LSTMCell(hidden_size), inputs=self.user_seq, sequence_length=length, dtype='float32', scope='behavior_gru') # error! ⭐️
                # hidden_size==32, self.user_seq==[?,5,16], behave_outputs==[?,5,32] 

                lstm_layer = tf.keras.layers.LSTM( # for learning the representation vector of each topic (last state)
                    hidden_size, return_sequences=True, return_state=False
                )
                # use masking to manage variable seq lengths
                # tf.keras.layers.LSTM automatically respects 'mask' if it's provided
                # 1. learn intra-topic user hehavior pattern
                mask = tf.sequence_mask(length, max_seq_len) # (?*20, 5/10)
                behave_outputs = lstm_layer(self.user_seq, mask=mask) # adding mask ignores paddings and irrelevant eles, lstm learns dependencies. mask tensor typically has the same shape as the seq
                
                self.user_seq = tf.reshape(behave_outputs[:,-1,:], (-1, num_cat, hidden_size))
                # self.bilstm(self.user_seq, hidden_size, scope='behave_bilstm')

                # 2. learn inter-topic correlations based on stacked topic-specific representation vecotrs (into a matrix)
                # diversity gain
                attended_cat = self.multihead_attention(self.user_seq, self.user_seq)  # num_cat * emb_size, using self-attention to learn the pref. distribution over topics based on aggregated topic-specific signas  
                #attended_cat_bn = tf.reshape(tf.layers.batch_normalization( # A: attended matrix #⭐️
                    #inputs=attended_cat), [-1, self.num_cat * hidden_size]) #⭐️
                attended_cat_bn = tf.reshape(tf.keras.layers.BatchNormalization()(attended_cat), [-1, self.num_cat * hidden_size]) #⭐️

            # relevance estimator (bilstm)
            with tf.variable_scope('rel'):                
                rel_outputs, _, _ = self.bilstm(item_seq, hidden_size, scope='rel_bilstm') # change dim of item_seq, evaluates the item_seq's (item_embed + cate_embed) rel
                # then no need to stack rel_outputs again
                # rel_outputs = tf.stack(rel_outputs, axis=1)
                seq_rel = tf.reshape(rel_outputs, (-1, max_time_len, hidden_size * 2)) # [-1, 20, hidden_size * 2]
                self.lstm_rel = seq_rel

            # further feed concatenated attended_cat_bn to MLP by non-linearly mapping
            # theta, personalized preference distribution in R^m over topics -> how likely a user is interested in each topic
            #div_pref = tf.expand_dims(tf.layers.dense(attended_cat_bn, self.num_cat, activation=tf.nn.relu), 1) #⭐️
            div_pref = tf.expand_dims(tf.keras.layers.Dense(self.num_cat, activation=tf.nn.relu)(attended_cat_bn), 1)
            
            div_add_item = tf.reduce_prod(1 - self.items_div, axis=1, keep_dims=True) # B * L * C
            self.remove_idx = self.idx_to_remove() # have a list with 19 ele repeats 20 times, just iterates the whole 20 ele and each time get one iteratively removed
            removed = tf.transpose(tf.gather(tf.transpose(1 - self.items_div, [1, 0, 2]), self.remove_idx), [1, 0, 2])
            removed = tf.reshape(removed, [-1, self.max_time_len, self.max_time_len - 1, self.num_cat]) # [-1, 20, 19, 20] for ml-20m, the seq is of num_cate, cuz for each cate there's one behavior_seq
            dev_removed = tf.reduce_prod(removed, axis=2) # becomes [-1, 20, 20] in the end, multiplies the 0-axis and 1-axis values along 2-axis (19)
            self.delta_gain = dev_removed - div_add_item # the change of the diversity score with and without R(i)
            self.div_gain = tf.multiply(self.delta_gain, div_pref) # theta * delta -> personlized div. gain of item R(i) in topic j, weighted by the user pref for topic j

            # concatenate rel + div
            if pure_rnn:
                seq_rel_inp = seq_rel
            else:
                seq_rel_inp = tf.concat([seq_rel, self.div_gain], axis=2) # rel + div
            

            # self.build_attraction()
            self.build_relevance_net(seq_rel_inp) # get final re-ranked score
            self.build_attraction()
            self.build_dcmloss() # optimize by reducing loss based on the generated click from DCM

    def idx_to_remove(self): # in the end have a len(max_time_len) i.e. 20 times repeated list ele without one ele, i.e. [19 ele] repeats 20 times
        ids = []
        for i in range(self.max_time_len):
            ids_all = list(range(self.max_time_len))
            del ids_all[i]
            ids.extend(ids_all)
        return ids

    def build_relevance_net(self, inp):
        #bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1') #⭐️
        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(inp) #⭐️
        #fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1') #⭐️
        fc1 = tf.keras.layers.Dense(200, activation=tf.nn.relu, name='fc1')(bn1) #⭐️
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        #fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2') #⭐️
        fc2 = tf.keras.layers.Dense(80, activation=tf.nn.relu, name='fc2')(dp1) #⭐️
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')

        # probablistic
        #mean = tf.layers.dense(dp2, 2, activation=None, name='fc3') #⭐️
        mean = tf.keras.layers.Dense(2, activation=None, name='fc3')(dp2) #⭐️
        var_dp2 = tf.nn.dropout(fc2, self.keep_prob, name='var_dp2')
        #var_fc3 = tf.layers.dense(var_dp2, 2, activation=None, name='var_fc3') #⭐️
        var_fc3 = tf.keras.layers.Dense(2, activation=None, name='var_fc3')(var_dp2) #⭐️
        var = tf.abs(tf.random_normal(tf.shape(var_fc3), 0, 1, tf.float32)) * var_fc3
        if self.map:
            score = tf.nn.softmax(mean)
        else: # map is set to False
            score = tf.cond(self.istrain, lambda: tf.nn.softmax(mean + var), lambda: tf.nn.softmax(mean))
        self.relevance = tf.reshape(score[:, :, 0], [-1, self.max_time_len])

    def build_termination_net(self, inp): 
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
        score = tf.nn.softmax(fc3)
        self.termination = tf.reshape(score[:, :, 0], [-1, self.max_time_len])

    def build_attraction(self):
        attraction_score = self.relevance
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        self.attraction = attraction_score * seq_mask

    def build_dcmloss(self):
        self.loss = tf.losses.log_loss(self.label_ph, self.attraction) # label_ph comes from DCM's generated user clicks
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss) # minimize loss while training 

    def train(self, sess, batch_data, lr, reg_lambda, keep_prob=0.8):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.feat_ph: batch_data[0],
            self.label_ph: batch_data[1],
            self.seq_length_ph: batch_data[2],
            self.user_behavior_ph: batch_data[3],
            self.behavior_len_ph: batch_data[-1],
            self.items_div: batch_data[4],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.istrain: True,
        })
        return loss

    def eval(self, sess, batch_data, reg_lambda, keep_prob=1, no_print=True):
        '''
        gives the output pred,term,label,loss in eval() in run_test.py using model.eval() (RAPID)
        the first and second output var. are the same (what's the meaning?)
        '''
        rel_div, rel, div, pred, label, loss = sess.run([self.relevance, self.lstm_rel, self.div_gain, self.attraction, self.label_ph, self.loss], feed_dict={ #⭐️
            self.feat_ph: batch_data[0],
            self.label_ph: batch_data[1], # from original batch_data
            self.seq_length_ph: batch_data[2],
            self.user_behavior_ph: batch_data[3],
            self.behavior_len_ph: batch_data[-1],
            self.items_div: batch_data[4],
            self.reg_lambda: reg_lambda,
            self.keep_prob: keep_prob,
            self.istrain: False,
        })
        #rel_div.reshape([-1, self.max_time_len]).tolist(), rel.reshape([-1, self.max_time_len]).tolist(), div.reshape([-1, self.max_time_len]).tolist(),
        return pred.reshape([-1, self.max_time_len]).tolist(), pred.reshape(
            [-1, self.max_time_len]).tolist(), label.reshape([-1, self.max_time_len]).tolist(), loss #⭐️

