import os
import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import log_loss, roc_auc_score
import random
import time
import pickle as pkl
import numpy as np

from utils import get_aggregated_batch, rank, construct_behavior_data, dcm_theta
from initial_rankers import DIN#, LambdaMART # comment LambdaMART for this moment


def eval(model, sess, data, reg_lambda, batch_size):
    '''
    data: (csv) probably the ds that split_data created (training, val and test, uid-iid-cid-rel)
    sess:
    reg_lambda: (float) regularization param
    '''
    preds = [] # predictions
    labels = []
    losses = []

    data_size = len(data[0]) # which format is data? csv or df?
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)
    t = time.time()
    for batch_no in range(batch_num): # batch_no is the counter
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, label, loss = model.eval(sess, data_batch, reg_lambda)
        preds.extend(pred) # add each ele. from an iterable (list, tuple, or set) to the end of the list
        labels.extend(label)
        losses.append(loss) # add a single item to the end of the list e.g. [1,2,3,[4,5]]

    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses) # loss_avg

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, logloss, auc


def save_rank(model, sess, data, reg_lambda, batch_size, out_file, item_div_dir, user_history, max_behavior_len, multi_hot):
    preds = []
    labels = []
    users = []
    items = []

    data_size = len(data[0])
    batch_num = data_size // batch_size

    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        pred, label, loss = model.eval(sess, data_batch, reg_lambda)
        preds.extend(pred)
        labels.extend(label)
        users.extend(data_batch[0]) # 2:cid, 3:rel
        items.extend(data_batch[1])

    rank(users, items, preds, labels, out_file, item_div_dir, user_history, max_behavior_len, multi_hot)


def train(train_file, val_file, test_file, eb_dim, feature_size,
          item_fnum, user_fnum, num_user, num_item, num_clusters, lr, reg_lambda, batch_size,
          max_time_len, item_div_dir, user_history, max_behavior_len, processed_dir, pt_dir, multi_hot):
    tf.reset_default_graph()

    if model_type == 'DIN':
        model = DIN(eb_dim, feature_size, item_fnum, user_fnum, num_user, num_item, num_clusters, max_time_len, multi_hot)
    else:
        print('WRONG MODEL TYPE')
        exit(1)

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'logloss': [],
        'auc': []
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, logloss, auc = eval(model, sess, test_file, reg_lambda, batch_size)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(vali_loss)
        training_monitor['logloss'].append(logloss)
        training_monitor['auc'].append(auc)

        print("STEP %d  LOSS TRAIN: NULL | LOSS VALI: %.4f  LOGLOSS: %.4f  AUC: %.4f" % (
        step, vali_loss, logloss, auc))
        early_stop = False
        data_size = len(train_file[0])
        batch_num = data_size // batch_size
        eval_iter_num = (data_size // 5) // batch_size
        print('train', data_size, batch_num)

        # begin training process
        for epoch in range(10):
            if early_stop:
                break
            for batch_no in range(batch_num):
                data_batch = get_aggregated_batch(train_file, batch_size=batch_size, batch_no=batch_no)
                if early_stop:
                    break
                loss = model.train(sess, data_batch, lr, reg_lambda)
                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, logloss, auc = eval(model, sess, test_file, reg_lambda, batch_size)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['logloss'].append(logloss)
                    training_monitor['auc'].append(auc)

                    print("STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f  LOGLOSS: %.4f  AUC: %.4f" % (
                    step, train_loss, vali_loss, logloss, auc))
                    if training_monitor['auc'][-1] > max(training_monitor['auc'][:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name)):
                            os.makedirs('save_model_{}/{}/{}/'.format(data_set_name, max_time_len, model_name))
                        save_path = 'save_model_{}/{}/{}/ckpt'.format(data_set_name, max_time_len, model_name)
                        model.save(sess, save_path)
                        # create ranking list using vali_file and test_file
                        save_rank(model, sess, val_file, reg_lambda, batch_size, processed_dir + model_type + '.rankings.train.' + str(max_behavior_len),
                                  item_div_dir, user_history, max_behavior_len, multi_hot)
                        save_rank(model, sess, test_file, reg_lambda, batch_size, processed_dir + model_type + '.rankings.test.' + str(max_behavior_len),
                                  item_div_dir, user_history, max_behavior_len, multi_hot)
                        print('initial lists saved')

                    if len(training_monitor['vali_loss']) > 2 and epoch > 0:
                        if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                                training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                            early_stop = True
                        if (training_monitor['vali_loss'][-2] - training_monitor['vali_loss'][-1]) <= 0.001 and (
                                training_monitor['vali_loss'][-3] - training_monitor['vali_loss'][-2]) <= 0.001:
                            early_stop = True

        # model.save_pretrain(sess, pt_dir, multi_hot)
        # generate log
        if not os.path.exists('logs_{}/{}/'.format(data_set_name, max_time_len)):
            os.makedirs('logs_{}/{}/'.format(data_set_name, max_time_len))
        model_name = '{}_{}_{}_{}'.format(model_type, batch_size, lr, reg_lambda)

        with open('logs_{}/{}/{}.pkl'.format(data_set_name, max_time_len, model_name), 'wb') as f:
            pkl.dump(training_monitor, f)


def get_data(dataset, embed_dir, multi_hot=False):
    target_user, target_item, user_behavior, label, seq_length = dataset
    embeddings = pkl.load(open(embed_dir, 'rb'))
    if multi_hot:
        embeddings, cate_embed = embeddings
    else:
        embeddings = embeddings[0][0]
    records = []
    # print(embeddings)

    for feat_i, click_i, uid_i in zip(target_item, label, target_user):
        # remove all 0s list
        if feat_i[0]:
            itm_emd = embeddings[feat_i[0]]
            if multi_hot:
                cat_emd = np.matmul(np.array(feat_i[1:]).reshape(1, -1), cate_embed) / sum(feat_i[1:])
                cat_emd = np.reshape(cat_emd, -1)
            else:
                cat_emd = embeddings[feat_i[1]]

            record_i = [click_i, uid_i] + itm_emd.tolist() + cat_emd.tolist() + feat_i
            records.append(record_i)
    records = np.array(sorted(records, key=lambda k: k[1]))
    # print(len(target_item[0]))
    return records[:, :-len(target_item[0])], records[:, -len(target_item[0]):].tolist()


def train_mart(train_file, val_file, test_file, embed_dir, item_div_dir, user_history, max_behavior_len, processed_dir,
               multi_hot, tree_num=300, lr=0.05, tree_type='lgb'):
    training_data, train_iids = get_data(train_file, embed_dir, multi_hot)
    test_data, test_iids = get_data(test_file, embed_dir, multi_hot)
    model = LambdaMART(training_data, tree_num, lr, tree_type)
    model.fit()
    if not os.path.exists('save_model_{}/{}'.format(data_set_name, 'mart')):
        os.makedirs('save_model_{}/{}'.format(data_set_name, 'mart'))
    model.save('save_model_{}/{}/{}_{}_{}'.format(data_set_name, 'mart', tree_num, lr, tree_type))
    test_pred = model.predict(test_data)
    rank(list(map(int, test_data[:, 1].tolist())), test_iids, test_pred, list(map(int, test_data[:, 0].tolist())),
         processed_dir + model_type +'.rankings.test', item_div_dir, user_history, max_behavior_len, multi_hot)
    logloss = log_loss(list(map(int, test_data[:, 0].tolist())), test_pred)
    auc = roc_auc_score(list(map(int, test_data[:, 0].tolist())), test_pred)
    print('mart logloss:', logloss, 'auc:', auc)

    val_data, val_iids = get_data(val_file, embed_dir, multi_hot)
    val_pred = model.predict(val_data)
    rank(list(map(int, val_data[:, 1].tolist())), val_iids, val_pred, list(map(int, val_data[:, 0].tolist())),
         processed_dir + model_type + '.rankings.train', item_div_dir, user_history, max_behavior_len, multi_hot)


def save_svm_file(dataset, out_file):
    svm_rank_fout = open(out_file, 'w')
    for i, record in enumerate(dataset):
        feats = []
        for j, v in enumerate(record[2:]):
            feats.append(str(j + 1) + ':' + str(v))
        line = str(int(record[0])) + ' qid:' + str(int(record[1])) + ' ' + ' '.join(feats) + '\n'
        svm_rank_fout.write(line)
    svm_rank_fout.close()


def train_svm(train_file, val_file, test_file, embed_dir, item_div_dir, user_history, max_behavior_len, processed_dir,
               multi_hot, c=2):
    svm_dir = processed_dir + 'svm'
    if not os.path.exists(svm_dir):
        os.makedirs(svm_dir)
    training_data, train_iids = get_data(train_file, embed_dir, multi_hot)
    save_svm_file(training_data, svm_dir + '/train.txt')
    test_data, test_iids = get_data(test_file, embed_dir, multi_hot)
    save_svm_file(test_data, svm_dir + '/test.txt')
    val_data, val_iids = get_data(val_file, embed_dir, multi_hot)
    save_svm_file(val_data, svm_dir + '/valid.txt')

    # train SVMrank model
    command = 'SVMrank/svm_rank_learn -c ' + str(c) + ' ' + svm_dir + '/train.txt ' + svm_dir + '/model.dat'
    os.system(command)

    # test the train set left, generate initial rank for context feature and examination
    # SVM_rank_path+svm_rank_classify remaining_train_set_path output_model_path output_prediction_path
    command = 'SVMrank/svm_rank_classify ' + svm_dir + '/test.txt ' + svm_dir + '/model.dat ' + svm_dir + '/test.predict'
    os.system(command)
    command = 'SVMrank/svm_rank_classify ' + svm_dir + '/valid.txt ' + svm_dir + '/model.dat ' + svm_dir + '/valid.predict'
    os.system(command)

    test_fin = open(svm_dir + '/test.predict', 'r')
    test_pred = list(map(float, test_fin.readlines()))
    test_fin.close()
    rank(list(map(int, test_data[:, 1].tolist())), test_iids, test_pred, list(map(int,test_data[:, 0].tolist())),
         processed_dir + model_type + '.rankings.test', item_div_dir, user_history, max_behavior_len, multi_hot)
    logloss = log_loss(list(map(int, test_data[:, 0].tolist())), test_pred)
    auc = roc_auc_score(list(map(int, test_data[:, 0].tolist())), test_pred)
    print('mart logloss:', logloss, 'auc:', auc)

    val_fin = open(svm_dir + '/valid.predict', 'r')
    val_pred = list(map(float, val_fin.readlines()))
    val_fin.close()
    rank(list(map(int, val_data[:, 1].tolist())), val_iids, val_pred, list(map(int,val_data[:, 0].tolist())),
         processed_dir + model_type + '.rankings.train', item_div_dir, user_history, max_behavior_len, multi_hot)


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    data_dir = 'data/'
    # data_set_name = 'ad'
    data_set_name = 'ml-20m'
    multi_hot = True if data_set_name == 'ml-20m' else False
    stat_dir = os.path.join(data_dir, data_set_name + '/raw_data/data.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    dcm_dir = os.path.join(data_dir, data_set_name + '/dcm.theta')
    item_div_dir = os.path.join(processed_dir, 'diversity.item')
    pt_dir = os.path.join(processed_dir, 'pretrain')
    model_type = 'DIN'
    # model_type = 'svm'
    # model_type = 'mart'
    item_fnum = 2
    user_fnum = 1
    max_time_len = 20
    max_behavior_len = 5 #⭐️
    #max_behavior_len = 5
    num_clusters = 5
    reg_lambda = 1e-4
    lr = 1e-3   # DIN
    # lr = 5e-2   # mart
    embedding_size = 16
    batch_size = 256 #500
    tree_num = 20
    tree_type = 'lgb'
    c = 2

    user_remap_dict, item_remap_dict, cat_remap_dict, cate_dict, feature_size = pkl.load(open(stat_dir, 'rb'))
    train_file, val_file, test_file, user_profile_dict, cat_dict = pkl.load(open(os.path.join(processed_dir, 'data.data'), 'rb'))
    user_set = sorted(user_remap_dict.values())
    #print(len(user_set))
    #print(user_set)
    item_set = sorted(item_remap_dict.values())
    num_user, num_item, num_cate = len(user_remap_dict), len(item_remap_dict), len(cate_dict)
    if data_set_name == 'ml-20m':
        num_clusters = num_cate

    behavioral_train_file = construct_behavior_data(train_file, user_profile_dict, max_time_len, multi_hot)
    behavioral_val_file = construct_behavior_data(val_file, user_profile_dict, max_time_len, multi_hot)
    behavioral_test_file = construct_behavior_data(test_file, user_profile_dict, max_time_len, multi_hot)
    if model_type == 'DIN':
        '''
        test = train(behavioral_train_file, behavioral_val_file, behavioral_test_file, embedding_size, feature_size, \
              item_fnum, user_fnum, num_user, num_item, num_clusters, lr, reg_lambda, batch_size, max_time_len, item_div_dir, \
              user_profile_dict, max_behavior_len, processed_dir, pt_dir, multi_hot)
        print(test)
        '''
        dcm_theta(user_profile_dict, item_div_dir, num_clusters, user_set, dcm_dir, multi_hot)
        train(behavioral_train_file, behavioral_val_file, behavioral_test_file, embedding_size, feature_size, \
              item_fnum, user_fnum, num_user, num_item, num_clusters, lr, reg_lambda, batch_size, max_time_len, item_div_dir, \
              user_profile_dict, max_behavior_len, processed_dir, pt_dir, multi_hot)
        
    elif model_type == 'mart':
        train_mart(behavioral_train_file, behavioral_val_file, behavioral_test_file, pt_dir, item_div_dir,
              user_profile_dict, max_behavior_len, processed_dir, multi_hot, tree_num, lr, tree_type)
    elif model_type == 'svm':
        train_svm(behavioral_train_file, behavioral_val_file, behavioral_test_file, pt_dir, item_div_dir,
              user_profile_dict, max_behavior_len, processed_dir, multi_hot, c)
    else:
        print('No Such Model')


