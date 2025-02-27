import os
import pickle as pkl
from datetime import date
import random
import csv 
from collections import defaultdict
# a built-in dict type, simplifies handling cases where we need to provide a default value for non-existent keys in a dict.
# with a regular dict, we would typically check whether a key exists before accessing or modifying, with defaultdict this's handled automatically
# it automatically assign a value to a non-existent key when using defaultdict, dont need to worry about raising a KeyError 
import numpy as np
import pandas as pd
from itertools import islice


def stat_data(raw_dir, cate_dir, stat_dir, diver_dict):
    """
    for CiteULike
    Dataset description:
    user has only one single feature -> user_id
    after preprocessing of the dataset, now each user has on average 37 articles in the library, ranging from 10 to 403, 93% users have fewer than 100 articles
    the articles were added to CiteULike bet. 2004 and 2010, each article on average appears in 12 users' libraries, ranging from 1 to 321, 97% articles appear in fewer than 40 libraries
    citations.dat: (0-based indexing, item-item), [0]: 3 2 485 3284 means to this 0-id node, there're 3 edges linked to node 0 and their ids are 2, 485, 3284
    item-tag.dat: (0-based indexing, item-category), [0]: 17 4276 32443 37837 3378 7650 44590 42810 28819 43806 3805 25497 23779 42140 12234 37386 30698 43503 means node 0 item has 17 tags, their ids are...
    raw-data.csv: (1-based indexing, item), the info about articles (doc-id, title, abstract, citeulike-id etc.)
    tags.dat: (0-based indexing, category), sorted by tag-id
    users.dat: (0-based indexing, user-item), rating matrix (collected articles)
    
    raw_dir: users.dat (user-item rating matrix), user's individual library
    cate_dir: item-tag.dat
    stat_dir: an input file to save the statistics of the data
    diver_dict: also an input file to save the embeddings of iid based on cates (to learn diversity?)
    """
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}

    uid_set = set()
    iid_set = set()
    cid_set = set()
    date_set = set() # what's this for?

    user_item = defaultdict(list) # similar to user.dat, where basically just rating matrix kept (collected items of each user), just with the num of collected item out 
    item_cate = defaultdict(None)
    cate_num = defaultdict(int)
    filter_cate_num = defaultdict(int)
    rating_num = defaultdict(int)
    pos, neg = 0, 0

    with open(cate_dir, 'r', encoding='utf-8') as r:
        '''
        item-tag.dat
        '''
        for iid, row in enumerate(r):
            values = list(map(int, row.strip().split()))
            cates = values[1:] # remove the first value (num of cates)
            item_cate[iid] = cates
            for cate in cates:
                cate_num[cate] += 1 # cate-id-based indexing

    print('num of cate:', len(cate_num))
    for k, v in cate_num.items():
        print('cate:', k, '  num:', v)


    with open(raw_dir, 'r', encoding='utf-8') as r:
        '''
        users' collected articles
        '''
        for uid, row in enumerate(r):
            values = list(map(int, row.strip().split()))
            collected_art = values[1:] # each user's collected articles
            #user_item[uid].append([values[0]) # keep the same structure, first the num of collected articles
            for art in collected_art:
                rating_num[art] += 1 # count each article's collected state
                user_item[uid].append(art) # appends user's collected articles
            total_art = 0
            for art in user_item.values():
                total_art += len(art)

    '''
    for k, v in rating_num.items(): # k is the index of rating_num -> the id of the article
        print('article-id:', k, ' num of collecting users:', v, ' percent among all articles collecting:', v * 1.0 / total_art)
    print('num of user:', len(user_item))
    print('num of item:', len(item_cate))
    '''
    # stats = [0, 0, 0, 0]
    # for uid in user_item.keys():
    #     if len(user_item[uid]) > 150:
    #         stats[0] += 1
    #     elif len(user_item[uid]) > 100:
    #         stats[1] += 1
    #     elif len(user_item[uid]) > 50:
    #         stats[2] += 1
    #     else:
    #         stats[3] += 1
    # user_num = len(user_item.keys())
    # print('item per user > 150: ', stats[0], stats[0] * 1.0 / user_num)
    # print('100 < item per user < 150: ', stats[1], stats[1] * 1.0 / user_num)
    # print('50 < item per user < 100: ', stats[2], stats[2] * 1.0 / user_num)
    # print('item per user < 50: ', stats[3], stats[3] * 1.0 / user_num)

    # no need of further filtering, the data has already been processed with only users that with more than 10 references in the library left
    # just keep the part of creating uid_set, iid_set
    #filter_pos, filter_neg = 0, 0
    for uid in user_item.keys():
            uid_set.add(uid)
            for item in user_item[uid]:
                iid_set.add(item) # add item_id, because iid_set if of 'set' format, even duplicated values are added, still only the unique remained

    for iid in iid_set: # iterate all iid in iid_set to get their cates
        cates = item_cate[iid] # a list of cate
        for cate in cates:
            filter_cate_num[cate] += 1 # organized according to cate, count their nums
    
    print('AFTER FILTER \nnum of cate:', len(filter_cate_num))
    for k, v in filter_cate_num.items():
        print('cate:', k, '  num:', v)
    #print('# total record:', filter_pos + filter_neg, '  when pos>=4, pos vs neg', filter_pos * 1.0 / filter_neg)
    
    
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(filter_cate_num.keys()) # keys are cate names, values are counts


    print('num user', len(uid_list))
    print('num item', len(iid_list))
    

    feature_id = 1 # starts from 1 to calculate the num of whole features in uid, iid and cid
    for uid in uid_list:
        uid_remap_dict[uid] = feature_id
        feature_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = feature_id
        feature_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = feature_id
        feature_id += 1
    # count total feature num incl. user, item and cates
    print('total original feature number: {}'.format(feature_id)) 

    cid_dict = {}
    for i, v in enumerate(cid_list): # (cate), i=index, v=value=cate
        cid_dict[v] = i # create a dict with cate as keys/index and index as value
    # print(cid_dict)
    iid_cate_map = {}
    for iid in iid_set:
        cates = item_cate[iid] # which cates can this iid have
        # print(generate_cate_multi_hot(cates.split('|'), cid_dict))
        # iid_remap_dict has iid as keys, the index as values
        # by doint this, iid_cate_map and iid_remap_dict are cooresponding
        # create an iid_cate_map -> an embedding of iid in terms of cates
        iid_cate_map[iid_remap_dict[iid]] = generate_cate_multi_hot(cates, cid_dict)

    with open(stat_dir, 'wb') as f:
        # serialize the info as a byte stream and save them in f
        # save the list of uid, iid and cid and cid_dict (dont see the diff.) and total feature num
        pkl.dump([uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id], f)
    with open(diver_dict, 'wb') as f:
        # to save embedding of iid based on cate
        pkl.dump(iid_cate_map, f) # [feature_id] = item_matrix_based_on_cid_dict -> represents topic coverage of each item / topic distribution
    print('======= statistic done ============')
    return uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id


def generate_cate_multi_hot(cates, cate_dict):
    multi_hot = np.zeros(len(cate_dict)) # first create the right size of zero-matrix
    for i in cates:
        # here shows the meaning of separately creating a cid_dict
        # just to match the index between multi_hot and cate_dict 
        # with cate as key/indicator 
        # which cates this iid has, assign 1 to this position in multi_hot
        multi_hot[cate_dict[i]] = 1
    return multi_hot


def split_data(in_file, statistics, diver_dir, out_file):
    '''
    split data for training, validation and test
    in_file: users.dat
    spliting_ratio: 2:3:4:1 (user_profile_data:trainig:validation:test)
    '''
    user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, _ = pkl.load(open(statistics, 'rb'))
    user_set = set(user_remap_dict.keys())
    records = []
    with open(raw_dir, 'r', encoding='utf-8') as r:
        '''
        users' collected articles
        '''
        for uid, row in enumerate(r):
            values = list(map(int, row.strip().split()))
            collected_art = values[1:] # each user's collected articles
            if uid in user_set:
                records.append([uid, collected_art]) # records only contain uid + items, no ratings

    rec_num = len(records)
    random.shuffle(records)
    user_profile_data, train_data, val_data, test_data = records[:int(rec_num*0.2)], \
      records[int(rec_num*0.2):int(rec_num*0.5)], records[int(rec_num*0.5):int(rec_num*0.9)], records[int(rec_num*0.9):]

    cat_dict = pkl.load(open(diver_dir, 'rb')) # iid_cate_map, cate matrix based on iid

    user_profile_dict, train_dict, val_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    train_set, val_set, test_set = [], [], []
    for user, items in user_profile_data: # from records [uid, collected_art(iids)]
        #if rel: # if pos (~ that user likes this recommendation) -> as user behavior data to learn
            uid = user_remap_dict[user] # uid_remap_dict, uses uid as index, feature_id as value -> uid is actually feature_id 
            for item in items:
                iid = item_remap_dict[item] # ⭐️ item is actually the true iid, iid is just the feature_id of this iid?
                cid = cat_dict[iid] # iid is the feature_id, cid is the rep matrix of this iid w.r.t cate
                ft = [iid] # feature
                ft.extend(cid)
                user_profile_dict[uid].append(ft) # based on user, records a list of [feature_id of iid, rep. matrix of iid]

    for user, items in train_data:
        uid = user_remap_dict[user]
        for item in items:
            iid = item_remap_dict[item]
            cid = cat_dict[iid]
            train_set.append([uid, iid, cid])
            train_dict[uid].append(iid)

    for user, items in test_data:
        uid = user_remap_dict[user]
        for item in items:
            iid = item_remap_dict[item]
            cid = cat_dict[iid]
            test_set.append([uid, iid, cid])
            test_dict[uid].append(iid)

    for user, items in val_data:
        uid = user_remap_dict[user]
        for item in items:
            iid = item_remap_dict[item]
            cid = cat_dict[iid]
            val_set.append([uid, iid, cid])
            val_dict[uid].append(iid)

    # print('train data', train_set[100])
    # print('user behavior', user_profile_dict[100])
    with open(out_file, 'wb') as f:
        pkl.dump([train_set, val_set, test_set, user_profile_dict, cat_dict], f)
    print(' =============data split done=============')
    return train_set, val_set, test_set, user_profile_dict, cat_dict
    
if __name__ == '__main__':
    # parameters
    random.seed(1234)
    data_dir = 'data/'
    data_set_name = 'cite'
    num_clusters = 5
    raw_dir = os.path.join(data_dir, data_set_name + '/raw_data/users.dat')
    cate_dir = os.path.join(data_dir, data_set_name + '/raw_data/item-tag.dat')
    stat_dir = os.path.join(data_dir, data_set_name + '/raw_data/data.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    diver_dir = os.path.join(processed_dir, 'diversity.item') # to learn diversity

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if os.path.isfile(stat_dir):
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = pkl.load(open(stat_dir, 'rb')) # uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id
        print('loaded stat file')
    else:
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = stat_data(raw_dir, cate_dir,
                                                                                             stat_dir, diver_dir)
        #stat_data(raw_dir, cate_dir, stat_dir, diver_dir)
    # user_set = sorted(user_remap_dict.values())
    # item_set = sorted(item_remap_dict.values())
    # num_user, num_item, num_cat = len(user_remap_dict), len(item_remap_dict), len(cat_remap_dict)
    #print(f'cid_dict: {cid_list}')
    processed_data_dir = os.path.join(processed_dir, 'data.data')
    
    if os.path.isfile(processed_data_dir):
        train_file, val_file, test_file, user_profile_dict, cat_dict = pkl.load(open(processed_dir + '/data.data', 'rb'))
        print('loaded data for initial rankers')
    else:
        train_file, val_file, test_file, user_profile_dict, cat_dict = split_data(raw_dir, stat_dir, diver_dir, processed_data_dir)

