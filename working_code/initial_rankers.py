import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle as pkl
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
#import lightgbm as lgb



class BaseModel(object):
    def __init__(self, eb_dim, feature_size, item_fnum, user_fnum, num_user, num_item, num_clusters, max_time_len, multi_hot=False):
        # max_time_len==20
        # item_fnum==2
        # num_clusters==20
        # reset graph
        tf.reset_default_graph()

        # input placeholders
        with tf.name_scope('inputs'):
            self.user_behavior_length_ph = tf.placeholder(tf.int32, [None, ], name='user_behavior_length_ph')
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_fnum], name='target_user_ph')
            if multi_hot:
                self.user_behavior_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum + num_clusters - 1], name='user_behavior_ph')
                self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum + num_clusters - 1], name='target_item_ph')
                self.item_fnum = item_fnum + num_clusters
            else:
                self.user_behavior_ph = tf.placeholder(tf.int32, [None, max_time_len, item_fnum], name='user_behavior_ph')
                self.target_item_ph = tf.placeholder(tf.int32, [None, item_fnum], name='target_item_ph')
                self.item_fnum = item_fnum
            self.label_ph = tf.placeholder(tf.int32, [None, ], name='label_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.emb_dim = eb_dim
            self.item_fnum = item_fnum
            self.num_user = num_user
            self.num_item = num_item

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [feature_size, eb_dim],
                                           initializer=tf.truncated_normal_initializer)

            if multi_hot:
                self.cate_emb_mtx = tf.get_variable('cate_emb_mtx', [num_clusters, eb_dim],
                                           initializer=tf.truncated_normal_initializer)
                item_multi_hot = tf.cast(self.target_item_ph[:, 1:], tf.float32)
                self.item_embed = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph[:, 0]), [-1, eb_dim])
                self.cate_embed = tf.reshape(tf.matmul(item_multi_hot, self.cate_emb_mtx), [-1, eb_dim])
                self.cate_embed = self.cate_embed / (tf.reduce_sum(item_multi_hot, axis=-1, keepdims=True) + 1e-9)
                self.target_item = tf.concat([self.item_embed, self.cate_embed], -1)

                usr_multi_hot = tf.cast(tf.reshape(self.user_behavior_ph, [-1, item_fnum + num_clusters - 1])[:, 1:], tf.float32)
                self.user_multi_hot = usr_multi_hot
                self.usr_item_embed = tf.reshape(tf.nn.embedding_lookup(self.emb_mtx, self.user_behavior_ph[:, :, 0]), [-1, eb_dim])
                self.usr_cate_embed = tf.reshape(tf.matmul(usr_multi_hot, self.cate_emb_mtx), [-1, eb_dim])
                self.usr_cate_embed = self.usr_cate_embed / (tf.reduce_sum(usr_multi_hot, axis=-1, keepdims=True) + 1e-9)
                self.user_seq = tf.concat([self.usr_item_embed, self.usr_cate_embed], -1)
            else:
                self.target_item = tf.nn.embedding_lookup(self.emb_mtx, self.target_item_ph)
                self.user_seq = tf.nn.embedding_lookup(self.emb_mtx, self.user_behavior_ph)

            self.target_item = tf.reshape(self.target_item, [-1, item_fnum * eb_dim])
            self.user_seq = tf.reshape(self.user_seq, [-1, max_time_len, item_fnum * eb_dim])
            self.target_user = tf.nn.embedding_lookup(self.emb_mtx, self.target_user_ph)
            self.target_user = tf.reshape(self.target_user, [-1, user_fnum * eb_dim])

    def build_fc_net(self, inp):
        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(inp)
        #bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        #fc1 = tf.keras.layers.Dense(bn1, 200, activation=tf.nn.relu, name='fc1')
        fc1 = tf.keras.layers.Dense(200, activation=tf.nn.relu, name='fc1')(bn1)
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        #fc2 = tf.keras.layers.Dense(dp1, 80, activation=tf.nn.relu, name='fc2')
        fc2 = tf.keras.layers.Dense(80, activation=tf.nn.relu, name='fc2')(dp1)

        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        #fc3 = tf.keras.layers.Dense(dp2, 2, activation=None, name='fc3')
        fc3 = tf.keras.layers.Dense(2, activation=None, name='fc3')(dp2)
        score = tf.nn.softmax(fc3)
        # output
        self.y_pred = tf.reshape(score[:, 0], [-1, ])

    def build_mlp_net(self, inp):
        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(inp)
        #bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        #fc1 = tf.keras.layers.Dense(bn1, 500, activation=tf.nn.relu, name='fc1')
        fc1 = tf.keras.layers.Dense(500, activation=tf.nn.relu, name='fc1')(bn1)
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        #fc2 = tf.keras.layers.Dense(dp1, 200, activation=tf.nn.relu, name='fc2')
        fc2 = tf.keras.layers.Dense(200, activation=tf.nn.relu, name='fc2')(dp1)
        dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
        #fc3 = tf.keras.layers.Dense(dp2, 80, activation=tf.nn.relu, name='fc3')
        fc3 = tf.keras.layers.Dense(80, activation=tf.nn.relu, name='fc3')(dp2)
        dp3 = tf.nn.dropout(fc3, self.keep_prob, name='dp3')
        #fc4 = tf.keras.layers.Dense(dp3, 2, activation=None, name='fc4')
        fc4 = tf.keras.layers.Dense(2, activation=None, name='fc4')(dp3)
        score = tf.nn.softmax(fc4)
        # output
        self.y_pred = tf.reshape(score[:, 0], [-1, ])

    def build_logloss(self):
        # loss
        self.log_loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        self.loss = self.log_loss
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
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

    def train(self, sess, batch_data, lr, reg_lambda):
        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.user_behavior_ph: batch_data[2],
            self.user_behavior_length_ph: batch_data[4],
            self.target_user_ph: np.array(batch_data[0]).reshape(-1, 1),
            # self.target_item_ph: np.array(batch_data[1]).reshape(-1, self.item_fnum),
            self.target_item_ph: np.array(batch_data[1]).reshape(-1, self.target_item_ph.shape[1]),
            self.label_ph: batch_data[3],
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8
        })
        
        # return self.user_behavior_ph.shape, np.array(batch_data[2]).shape, self.target_item_ph.shape, np.array(batch_data[1]).reshape(-1, self.target_item_ph.shape[1]).shape
        #return self.loss, self.train_step, len(batch_data[2]), len(batch_data[4]), np.array(batch_data[0]).reshape(-1,1).shape, np.array(batch_data[1]).reshape(-1,self.item_fnum).shape, len(batch_data[3]), lr, reg_lambda, 0.8
        return loss

    def eval(self, sess, batch_data, reg_lambda):
        # batch_data: [uid, iid, user_behavior, label, seq_len]
        pred, label, loss = sess.run([self.y_pred, self.label_ph, self.loss], feed_dict={
            self.user_behavior_ph: batch_data[2],
            self.user_behavior_length_ph: batch_data[4],
            self.target_user_ph: np.array(batch_data[0]).reshape(-1, 1),
            self.target_item_ph: batch_data[1],
            self.label_ph: batch_data[3],
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.
        })
        return pred.tolist(), label.tolist(), loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def save_pretrain(self, sess, path, multi_hot=False):
        if multi_hot:
            embed, cat_embed = sess.run([self.emb_mtx, self.cate_emb_mtx])
            with open(path, 'wb') as f:
                pkl.dump([embed, cat_embed], f)
        else:
            embed = sess.run([self.emb_mtx])
            with open(path, 'wb') as f:
                pkl.dump([embed], f)


class DIN(BaseModel):
    def __init__(self, eb_dim, feature_size, item_fnum, user_fnum, num_user, num_item, num_clusters, max_time_len, multi_hot):
        super(DIN, self).__init__(eb_dim, feature_size, item_fnum, user_fnum, num_user, num_item, num_clusters,
                                  max_time_len, multi_hot)
        mask = tf.sequence_mask(self.user_behavior_length_ph, max_time_len, dtype=tf.float32)
        _, user_behavior_rep = self.attention(self.user_seq, self.user_seq, self.target_item, mask)

        inp = tf.concat([user_behavior_rep, self.target_user, self.target_item], axis=1)

        # fc layer
        self.build_fc_net(inp)
        self.build_logloss()

    def attention(self, key, value, query, mask):
        # key: [B, T, Dk], query: [B, Dq], mask: [B, T]
        _, max_len, k_dim = key.get_shape().as_list()
        #query = tf.keras.layers.Dense(query, k_dim, activation=None)
        query = tf.keras.layers.Dense(k_dim, activation=None)(query)
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_len, 1])  # [B, T, Dk]
        kq_inter = queries * key
        atten = tf.reduce_sum(kq_inter, axis=2)

        mask = tf.equal(mask, tf.ones_like(mask))  # [B, T]
        paddings = tf.ones_like(atten) * (-2 ** 32 + 1)
        atten = tf.nn.softmax(tf.where(mask, atten, paddings))  # [B, T]
        atten = tf.expand_dims(atten, 2)

        res = tf.reduce_sum(atten * value, axis=1)
        return atten, res

