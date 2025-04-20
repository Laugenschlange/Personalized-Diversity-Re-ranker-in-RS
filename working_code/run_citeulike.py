import os
import pickle
import sys
import os
import tensorflow.compat.v1 as tf # tf 1.x
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time
import pickle as pkl
import numpy as np

from utils import load_data, evaluate, get_aggregated_batch, construct_list
from models import RAPID
from click_models import DCM

#@tf.function


def eval(model, sess, data_file, max_time_len, max_seq_len, reg_lambda, batch_size, num_cat, isrank, level):
    '''
    model: RAPID, has .train (train the model) .eval (evaluate on test set) 
    data_file: test_file, from 'data_{level}.data' [features, clicks, seq_len, user_behavior, items_div, uid, _ ]
    max_time_len: 20 by default
    max_seq_len: max_behavior_len (5|10)
    isrank: (bool) i.e. is_rerank in evaluate() in utils.py, is_rerank==True: re-rank the initial list and evaluate, otherwise just evaluate the initial ranking list
    '''
    preds = []
    terms = []
    labels = []
    losses = []
    users = [] 
    items = []
    items_div = []
    #seq_rel_inps = [] #⭐️
    #rels = [] #⭐️
    #divs = [] #⭐️

    data = load_data(data_file, click_model, num_cat, max_seq_len, test=True) # output: [feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm]. Generate data by using DCM, if test==True, appends all the data
    data_size = len(data[0])
    batch_num = data_size // batch_size # batch_num==3, batch_size==256
    print('eval', batch_size, batch_num)
    #if batch_num == 0:
    #    return None, None, None
    t = time.time()
    for batch_no in range(batch_num): # 3 times, batch_no.1/2/3, generates sess for RAPID model's re-ranking
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no) # data_batch: 7 * 256 * 20 * 21 ??? ⚡️what's the result mean? -> means batch_num==7, len(data)==256, len(data[0])==20, len(data[0][0])==21
        # data_batch already has repeated records 4 times for each uid, the input 'data' must already have been repeated 4 times 
        # data_batch: [items, x, x, x, items_div, x, x, users]
        #seq_rel_inp, rel, div, pred, term, label, loss = model.eval(sess, data_batch, reg_lambda, no_print=batch_no) # pred, term, label are included in data_batch⭐️ test div_gain, lstm_rel
        pred, term, label, loss = model.eval(sess, data_batch, reg_lambda, no_print=batch_no) # pred, term, label are included in data_batch⭐️

        preds.extend(pred) # get the preds from RAPID (model)
        terms.extend(term) # get terms from RAPID, **same as preds**
        labels.extend(label)
        losses.append(loss)
        #with tf.Session() as sess:
         #   div = sess.run(div)
          #  rel = sess.run(rel)
           # seq_rel_inp = sess.run(seq_rel_inp)
        #seq_rel_inps.extend(seq_rel_inp) #⭐️
        #rels.extend(rel) #⭐️
        #divs.extend(div) #⭐️
        users.extend(data_batch[-2]) # each time only takes the last ele (256-long, each ele with 20 uid)⭐️⭐️⭐️ changed index -1 into -2
        items.extend(data_batch[0]) # features
        items_div.extend(data_batch[4])
    
        # data_batch: [features, clicks, seq_len, user_behavior, items_div, uid, _ ]
    # in the end, users is of dim 3*256 (3 comes from batch_num, 256 comes from data_batch[-1])
    if not os.path.exists('logs_{}/{}/'.format(data_set_name, max_time_len)):
        os.makedirs('logs_{}/{}/'.format(data_set_name, max_time_len))
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(initial_rankers, model_type, max_behavior_len, level, hidden_size, mu, batch_size, lr, reg_lambda)

    '''
    ################ ----------TEST Generating Reco ---------- ################
    # save the generated reco
    if not os.path.exists('data/{}/reco/'.format(data_set_name)):
        os.makedirs('data/{}/reco/'.format(data_set_name))
        file_name = '{}_{}_{}_{}.pkl'.format(max_seq_len, hidden_size, mu, max_time_len)
    if isrank: # if RAPID's already trained
        pickle.dump([seq_rel_inps, rels, divs, preds, labels, users, items, losses, data], open('data/{}/reco/{}_{}_{}_{}_reranked_divgain.pkl'.format(data_set_name, max_seq_len, hidden_size, mu, max_time_len), 'wb'))
        #pickle.dump([preds, labels, users, items, losses, data], open('data/{}/reco/{}_{}_{}_{}_reranked_test.pkl'.format(data_set_name, max_seq_len, hidden_size, mu, max_time_len), 'wb'))
    else: # RAPID not trained yet
        pickle.dump([seq_rel_inps, rels, divs, preds, labels, users, items, losses, data], open('data/{}/reco/{}_{}_{}_{}_initial_divgain.pkl'.format(data_set_name, max_seq_len, hidden_size, mu, max_time_len), 'wb'))
        #pickle.dump([preds, labels, users, items, losses, data], open('data/{}/reco/{}_{}_{}_{}_initial_test.pkl'.format(data_set_name, max_seq_len, hidden_size, mu, max_time_len), 'wb'))
    ############### ------------´TEST ENDS -------------- ###################
    '''

    loss = sum(losses) / len(losses)
    #initlist = evaluate(labels, preds, terms, users, items, click_model, items_div, 5, isrank) # metrics@5 #⭐️
    #return initlist #⭐️
    res_low = evaluate(labels, preds, terms, users, items, click_model, items_div, 5, isrank) # metrics@5 #⭐️
    #uid, users_eva = evaluate(labels, preds, terms, users, items, click_model, items_div, 10, isrank) # metrics@5 #⭐️
    res_high = evaluate(labels, preds, terms, users, items, click_model, items_div, 10, isrank) # metrics@10 #⭐️

    print("EVAL TIME: %.4fs" % (time.time() - t))
    
    return loss, res_low, res_high #users, users_evaluate

def train(train_file, test_file, model_type, batch_size, feature_size, eb_dim, hidden_size, max_time_len, max_seq_len,
          item_fnum, num_cat, mu, max_norm, multi_hot, level):
    '''
    feature_size: (int) 10244
    eb_dim: (int) 16
    item_fnum: (int) 2
    num_cat: (int) 20
    '''
    tf.reset_default_graph()

    if model_type == 'RAPID':
        model = RAPID(feature_size, eb_dim, hidden_size, max_time_len,
                      max_seq_len, item_fnum, num_cat, mu, False, max_norm, multi_hot)
        
    else:
        print('No Such Model')
        exit()

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'ndcg_l': [],
        'utility_l': [],
        'ils_l': [],
        'diversity_l': [],
        'satisfaction_l': [],
        'mrr_l': [],
        'ndcg_h': [],
        'utility_h': [],
        'ils_h': [],
        'diversity_h': [],
        'satisfaction_h': [],
        'mrr_h': [],
    }
    if not os.path.exists('logs_{}/{}/{}'.format(data_set_name, max_time_len, mu)): # logs are more detailed with all param info
        os.makedirs('logs_{}/{}/{}'.format(data_set_name, max_time_len, mu))
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(initial_rankers, model_type, max_behavior_len, level, hidden_size, mu, batch_size, lr, reg_lambda) # add extra info for max_behavior_len
    log_save_path = 'logs_{}/{}/{}/{}_{}.metrics'.format(data_set_name, max_time_len, mu, model_name, level)

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process -> only needs 1 step for processing?
        step = 0
        
        vali_loss, res_l, res_h = eval(model, sess, test_file, max_time_len, max_seq_len, reg_lambda, batch_size, num_cat, False, level) #⭐️ not re-ranked yet
    
        # for not interrupting param-iteration
       #if vali_loss is None or res_l is None or res_h is None:
       #     return None, None, None

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(None)
        training_monitor['ndcg_l'].append(res_l[0])
        training_monitor['utility_l'].append(res_l[1])
        training_monitor['ils_l'].append(res_l[2])
        training_monitor['diversity_l'].append(res_l[3])
        training_monitor['satisfaction_l'].append(res_l[4])
        training_monitor['mrr_l'].append(res_l[5])
        training_monitor['ndcg_h'].append(res_h[0])
        training_monitor['utility_h'].append(res_h[1])
        training_monitor['ils_h'].append(res_h[2])
        training_monitor['diversity_h'].append(res_h[3])
        training_monitor['satisfaction_h'].append(res_h[4])
        training_monitor['mrr_h'].append(res_h[5])

        # initial list's performance from ranking stage

        #print("STEP %d  INTIAL RANKER | LOSS VALI: NULL NDCG@5: %.4f  UTILITY@5: %.4f  ILS@5: %.4f  "
        #      "DIVERSE@5: %.8f SATIS@5: %.8f MRR@5: %.8f | NDCG@10: %.4f  UTILITY@10: %.4f  ILS@10: %.4f  "
        #      "DIVERSE@10: %.8f SATIS@10: %.8f MRR@10: %.8f" % (
        #step, res_l[0], res_l[1], res_l[2], res_l[3], res_l[4], res_l[5], res_h[0], res_h[1], res_h[2], res_h[3], res_h[4], res_h[5]))


        early_stop = False
        
        data = load_data(train_file, click_model, num_cat, max_seq_len) # feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm
        data_size = len(data[0])
        batch_num = data_size // batch_size
        eval_iter_num = (data_size // 5) // batch_size
        print('train', data_size, batch_num)


        # begin eval process
        logs_list = [] # create a list to save the results, then only pick the last 3 as the best result
        for epoch in range(100):
            if early_stop:
               break
            for batch_no in range(batch_num): # batch_no changes over the for-loop
                data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no) # [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))] -> split the data into diff. batch_sizes
                # data_batch: the first batch_no for all rows, from index 0 to the end of data, then the next batch_no for all indices [[first batch_no], [seconde batch_no], ...] each batch_no block is len(data[i])==batch_no
                if early_stop:
                   break
                loss = model.train(sess, data_batch, lr, reg_lambda) # train RAPID using training set generated by load_data(train_file) + get_aggregated_batch() 
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step) # mean_loss
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    # re-ranked list with isrank==True
                    vali_loss, res_l, res_h = eval(model, sess, test_file, max_time_len, max_seq_len, reg_lambda, batch_size, num_cat, True, level) # re-ranked

                    training_monitor['train_loss'].append(train_loss)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['ndcg_l'].append(res_l[0])
                    training_monitor['utility_l'].append(res_l[1])
                    training_monitor['ils_l'].append(res_l[2])
                    training_monitor['diversity_l'].append(res_l[3])
                    training_monitor['satisfaction_l'].append(res_l[4])
                    training_monitor['mrr_l'].append(res_l[5])
                    training_monitor['ndcg_h'].append(res_h[0])
                    training_monitor['utility_h'].append(res_h[1])
                    training_monitor['ils_h'].append(res_h[2])
                    training_monitor['diversity_h'].append(res_h[3])
                    training_monitor['satisfaction_h'].append(res_h[4])
                    training_monitor['mrr_h'].append(res_h[5])
                    
                    print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f  NDCG@5: %.4f  UTILITY@5: %.4f  ILS@5: %.4f  "
                          "DIVERSE@5: %.8f SATIS@5: %.8f MRR@5: %.8f | NDCG@10: %.4f  UTILITY@10: %.4f  ILS@10: %.4f  "
                          "DIVERSE@10: %.8f SATIS@10: %.8f MRR@10: %.8f" % ( epoch,
                              step, train_loss, vali_loss, res_l[0], res_l[1], res_l[2], res_l[3], res_l[4],res_l[5],
                              res_h[0], res_h[1], res_h[2], res_h[3], res_h[4], res_h[5],))

                
                    if training_monitor['utility_l'][-1] > max(training_monitor['utility_l'][:-1]): # save model based on 'utility', only keep the best over the past
                        # save model (save each optimized state as a checkpoint)
                        model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(model_type, max_behavior_len, hidden_size, mu, level, batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name)):
                            os.makedirs('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name))
                        save_path = 'save_model_{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name)
                        model.save(sess, save_path)
                        pkl.dump([res_l[-1], res_h[-1]], open(log_save_path, 'wb')) # write the last record of res_l and res_h into logs file
                        print('model saved')

                    if len(training_monitor['vali_loss']) > 2 and epoch > 0:
                        if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                                training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                            early_stop = True
                        if (training_monitor['vali_loss'][-2] - training_monitor['vali_loss'][-1]) <= 0.001 and (
                                training_monitor['vali_loss'][-3] - training_monitor['vali_loss'][-2]) <= 0.001:
                            early_stop = True
                    logs_list.append([step, train_loss, vali_loss, res_l[0], res_l[1], res_l[2], res_l[3], res_l[4], res_l[5], res_h[0], res_h[1], res_h[2], res_h[3], res_h[4], res_h[5]]) # save each batch/epoch's value 

        # generate log
        with open('logs_{}/{}/{}/{}_{}.pkl'.format(data_set_name, max_time_len, mu, model_name, level), 'wb') as f: # for recording [train_loss, valid_loss, res_l[0-5], res_h[0-5]], under the same folder with .metrics in logs
            pkl.dump(training_monitor, f)
        return logs_list[-3:] # only return the last 3 records


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    data_dir = 'data/'
    # data_set_name = 'taobao'
    # data_set_name = 'ad'
    data_set_name = 'cite'
    #data_set_name = 'cite'
    multi_hot = True if data_set_name == 'cite' else False
    level = 'easy'
    n_topic = 20
    stat_dir = os.path.join(data_dir, data_set_name + f'/raw_data/data_{level}_{n_topic}.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    item_div_dir = os.path.join(processed_dir, f'diversity_{level}_{n_topic}.item')
    dcm_dir = os.path.join(data_dir, data_set_name + f'/dcm_{level}_{n_topic}.theta') # to store DCM's values
    item_fnum = 2
    user_fnum = 1
    # initial_rankers = 'svm'
    # initial_rankers = 'mart'
    initial_rankers = 'DIN'
    model_type = 'RAPID'
    max_time_len = 20 # time_step for inp in bilstm() in models.py
    max_behavior_len = 10 # [3, 5, 10] decides the len of the constructed user behavior seq by topic -> each topic has a user-speicific behavior seq.
    # ⭐️ set 10 for ml-20m
    #max_behavior_len = 10 #⭐️ 5 for reproduction
    num_clusters = 5 # for Taobao it's 5 (topics), the num of topics is already decided in preprocessing.py and is 5 by using GMM, if it's MovieLens, set num_clusters==len(cid_dict)
    reg_lambda = 1e-4 # regularization param
    lr = 1e-3 # [1e-5, 1e-4, 1e-3, 1e-2]
    embedding_size = 16
    batch_size = 256 # [256, 512, 1024] # 500 by run_initial_ranker.py
    hidden_size = 16 # [8, 16, 32, 64]
    #mu = 0.5 # rel-div-lambda, in DCM is called 'lmbd'
    mu = 1.0
    max_norm = None
   

    user_remap_dict, item_remap_dict, cat_remap_dict, cid_dict, feature_size = pkl.load(open(stat_dir, 'rb')) # uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id
    user_set = sorted(user_remap_dict.values()) # changed to .keys()
    item_set = sorted(item_remap_dict.values()) # changed to .keys()
    num_user, num_item = len(user_remap_dict), len(item_remap_dict)
    if data_set_name == 'cite': # for citeulike, num_cat should be decided based on cid_dict from the preprocessed data data.stat
        num_clusters = len(cid_dict) # 50 for citeulike
  
    # construct training files
    train_dir = os.path.join(processed_dir,  initial_rankers + f'.data.train.{max_behavior_len}.{level}.{n_topic}') # need to be processed by ``construct_list()`` using DIN.rankings.train.{max_behavior_len}, then saved in data.train.{max_behavior_len}
    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
        #print(train_lists)
    
    else:
        print('construct lists for training set')
        # ⭐️ changed names
        train_lists = construct_list(os.path.join(processed_dir, initial_rankers + f'.rankings.train.{max_behavior_len}.{level}.{n_topic}'), max_time_len, num_clusters, True, multi_hot) #⭐️ added 'max_behavior_len' info 
        pkl.dump(train_lists, open(train_dir, 'wb'))
        print('training set done')

    # construct test files
    test_dir = os.path.join(processed_dir, initial_rankers + f'.data.test.{max_behavior_len}.{level}.{n_topic}') # add {max_behavior_len} to distinguish with max_behavior_len==5
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
        #print(test_lists)

    else:
        print('construct lists for test set')
        test_lists = construct_list(os.path.join(processed_dir, initial_rankers + f'.rankings.test.{max_behavior_len}.{level}.{n_topic}'), max_time_len, num_clusters, False, multi_hot) #⭐️ added 'max_behavior_len' info, num_clusters==num_cat
        pkl.dump(test_lists, open(test_dir, 'wb'))
        print('test set done')

    click_model = DCM(max_time_len, num_clusters, user_set, item_set, item_div_dir, mu, dcm_dir) # construct click_model based on DCM, a global var.
   
    feat, click, seq_len, user_behavior, items_div, uid, _ = train_lists #!!! we dont have click info for citeulike !!!
    
    click_model.train(train_lists) # nothing's changed on train_lists after training click models
    # iterate param combinations
    #train(train_lists, test_lists, model_type, batch_size, feature_size, embedding_size, hidden_size, max_time_len,
    #                        max_behavior_len, item_fnum, num_clusters, mu, max_norm, multi_hot, level)

    for hidden_size in [16, 32, 64]: # [8, 16, 32, 64]
        if hidden_size == 16:
            lrlist = [1e-2]
        else:
            lrlist = [1e-5, 1e-4, 1e-3, 1e-2]
        for lr in lrlist: # [1e-5, 1e-4, 1e-3, 1e-2]
            if hidden_size==16 and lr==1e-4:
                ll = [1024]
            else:
                ll = [256, 512, 1024]
            for batch_size in ll: #[256, 512, 1024]: # [256, 512, 1024]
                rep_res = []
                #train(train_lists, test_lists, model_type, batch_size, feature_size, embedding_size, hidden_size, max_time_len,
                #            max_behavior_len, item_fnum, num_clusters, mu, max_norm, multi_hot, level) #⭐️


                rep_result1, rep_result2, rep_result3 = train(train_lists, test_lists, model_type, batch_size, feature_size, embedding_size, hidden_size, max_time_len,
                            max_behavior_len, item_fnum, num_clusters, mu, max_norm, multi_hot, level) #⭐️
                
                #rep_res.append([data_set_name, max_behavior_len, mu, hidden_size, lr, batch_size, max_time_len, model_type ,rep_result1])
                #rep_res.append([data_set_name, max_behavior_len, mu, hidden_size, lr, batch_size, max_time_len, model_type ,rep_result2])
                #rep_res.append([data_set_name, max_behavior_len, mu, hidden_size, lr, batch_size, max_time_len, model_type ,rep_result3])
                
                print(f"########################################## ---- h_s: {hidden_size}, lr: {lr}, b_s: {batch_size} is saved ########################################")
                #with open(f'citeulike_logs_{}/{}_{}_{}_{}_{}_{}.pkl'.format(data_set_name, max_behavior_len, mu, hidden_size, lr, batch_size, max_time_len, model_type), 'wb') as f:
                #        pkl.dump(rep_result, f)
                # Load existing data if the file exists
                if os.path.exists(f"citeulike_logs_early_stop_{max_behavior_len}_{level}_{int(mu)}.pkl"):
                    with open(f"citeulike_logs_early_stop_{max_behavior_len}_{level}_{int(mu)}.pkl", "rb") as f:
                        old_data = pickle.load(f)
                        rep_res.append(old_data)
                rep_res.append([data_set_name, max_behavior_len, mu, hidden_size, lr, batch_size, max_time_len, model_type, rep_result1, rep_result2, rep_result3])
                
                # Save the updated list (overwriting with the new combined data)
                with open(f"citeulike_logs_early_stop_{max_behavior_len}_{level}_{int(mu)}.pkl", "wb") as f:
                    pickle.dump(rep_res, f)
    #rep_res_df = pd.DataFrame(rep_res, columns=['dataset', 'max_beh_len', 'mu', 'hidden_size', 'lr', 'batch_size', 'max_time_len', 'model_type', 'rep_result'])
    #rep_res_df.to_pickle('rep_results.pkl')


    #print('###################################')
    #print(re_ranked)
