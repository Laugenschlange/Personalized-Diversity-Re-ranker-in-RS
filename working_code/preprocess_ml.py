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


def stat_data(raw_dir, cate_dir, stat_dir, diver_dict):
    """
    for MovieLens
    Dataset description:
    user has only one single feature -> user_id
    genome files: contains tag relevance scores for movies. a dense matrix. 
        each movie in the genome has a value for **every** tag in the genome
        tag genome was computed using ML on user-contributed content incl. tags, ratings and textual reviews
    genome-scores.csv: contains movie-tag relevance, movieId, tagId, relevance
    genome-tags.csv: contains tag description for the tag Ids. tagId, tag
    links.csv: movieId, imdbId, tmdbId
    movies.csv: movieId, title, genres (genres are a pip-separated list)
    ratings.csv: userId, movieId, rating, timestamp (seconds)
    tags.csv: (one tag applied to one movie by one user) userId, movieId, tag, timestamp
        (tags are user-generated metadata about movies)
    raw_dir: ratings.csv (userId, movieId, rating, timestamp)
    cate_dir: movies.csv (movieId, title, genres)
    stat_dir: an input file to save the statistics of the data
    diver_dict: also an input file to save the embeddings of iid based on cates (diversity.item)
    """
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}

    uid_set = set()
    iid_set = set()
    cid_set = set()
    date_set = set()

    user_item = defaultdict(list)
    item_cate = defaultdict(None)
    cate_num = defaultdict(int)
    filter_cate_num = defaultdict(int)
    rating_num = defaultdict(int)
    pos, neg = 0, 0

    with open(cate_dir, 'r', encoding='utf-8') as r:
        csv_reader = csv.reader(r)
        header = next(csv_reader)
        for row in csv_reader:
            iid, title, cates = row
            item_cate[iid] = cates
            for cate in cates.split('|'):
                cate_num[cate] += 1

    print('num of cate:', len(cate_num))
    for k, v in cate_num.items():
        print('cate:', k, '  num:', v)


    with open(raw_dir, 'r', encoding='utf-8') as r:
        csv_reader = csv.reader(r)
        header = next(csv_reader)
        for row in csv_reader:
            uid, iid, rating, ts = row
            # d = date.fromtimestamp(float(ts))
            # date_set.add(ts)
            rating_num[rating] += 1
            rel = 1 if float(rating) > 4 else 0 # define only > 4 is positive
            if rel:
                pos += 1
            else:
                neg += 1
            user_item[uid].append([iid, rel]) # appends interacted item_id and pref

    print('# total record:', pos + neg, '  when pos>=4, pos vs neg', pos * 1.0 / neg)
    for k, v in rating_num.items(): # k is the rating value
        print('rating:', k, ' num:', v, ' percent:', v * 1.0 / (pos + neg))
    print('num of user:', len(user_item))
    print('num of item:', len(item_cate))

    # filter out users who interact with less than 50 items (or 200?)
    filter_pos, filter_neg = 0, 0
    for uid in user_item.keys():
        if len(user_item[uid]) > 200: # means that the user has to have interacted with more than 200 items?
            uid_set.add(uid)
            for item in user_item[uid]:
                iid_set.add(item[0]) # add item_id
                # print(item[0], item_cate[item[0]])
                if item[1]: # if it's 1 -> pos
                    filter_pos += 1
                else:
                    filter_neg += 1

    for iid in iid_set: # iterate all iid in iid_set
        cates = item_cate[iid]
        for cate in cates.split('|'):
            filter_cate_num[cate] += 1 # organized according to cate, count their nums

    print('AFTER FILTER \nnum of cate:', len(filter_cate_num))
    for k, v in filter_cate_num.items():
        print('cate:', k, '  num:', v)
    print('# total record:', filter_pos + filter_neg, '  when pos>=4, pos vs neg', filter_pos * 1.0 / filter_neg)


    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(filter_cate_num.keys()) # keys are cate names, values are counts


    print('num user', len(uid_list))
    print('num item', len(iid_list))
    print('num cat', len(cid_list))

    feature_id = 1
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
        # iid_remap_dict has iid as keys, the index as values
        # by doint this, iid_cate_map and iid_remap_dict are cooresponding
        # create an iid_cate_map -> an embedding of iid in terms of cates
        iid_cate_map[iid_remap_dict[iid]] = generate_cate_multi_hot(cates.split('|'), cid_dict)

    with open(stat_dir, 'wb') as f: # what is stat_dir? A pre-created file? -- seems so
        # serialize the info as a byte stream and save them in f
        # save the list of uid, iid and cid and cid_dict (dont see the diff.) and total feature num
        pkl.dump([uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id], f)
    with open(diver_dict, 'wb') as f:
        # to save embedding of iid based on cate
        pkl.dump(iid_cate_map, f)
    print('======= statistic done ============')
    return uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id


def generate_cate_multi_hot(cates, cate_dict): # for MovieLens-20M
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
    in_file: (csv) ratings.csv
    spliting_ratio: 2:3:4:1 (user_profile_data:trainig:validation:test)
    out_file: data.data
    '''
    user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, _ = pkl.load(open(statistics, 'rb')) # [user, iid, cid,
    user_set = set(user_remap_dict.keys())
    records = []
    with open(in_file, 'r', encoding='utf-8') as r:
        csv_reader = csv.reader(r)
        header = next(csv_reader)
        for row in csv_reader:
            usr, itm, rating, ts = row
            if usr in user_set: # record relevant users' info
                rel = 1 if float(rating) > 4 else 0 # mark records with ratings > 4 as positive, otherwise negative -> in the end als 'label'
                records.append([usr, itm, rel])

    rec_num = len(records)
    random.shuffle(records) # no time order
    user_profile_data, train_data, val_data, test_data = records[:int(rec_num*0.2)], \
      records[int(rec_num*0.2):int(rec_num*0.5)], records[int(rec_num*0.5):int(rec_num*0.9)], records[int(rec_num*0.9):]

    cat_dict = pkl.load(open(diver_dir, 'rb'))

    user_profile_dict, train_dict, val_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    train_set, val_set, test_set = [], [], []
    for user, item, rel in user_profile_data:
        if rel: # if pos (~ that user likes this recommendation) -> as user behavior data to learn
            uid, iid = user_remap_dict[user], item_remap_dict[item]
            cid = cat_dict[iid] # the multi_hot encoding for mutli_hot==True
            ft = [iid]
            ft.extend(cid)
            user_profile_dict[uid].append(ft) # based on user, records iid, cid

    for user, item, rel in train_data:
        uid, iid = user_remap_dict[user], item_remap_dict[item]
        cid = cat_dict[iid]
        train_set.append([uid, iid, cid, rel])
        train_dict[uid].append(iid)

    for user, item, rel in test_data:
        uid, iid = user_remap_dict[user], item_remap_dict[item]
        cid = cat_dict[iid]
        test_set.append([uid, iid, cid, rel])
        test_dict[uid].append(iid)

    for user, item, rel in val_data:
        uid, iid = user_remap_dict[user], item_remap_dict[item]
        cid = cat_dict[iid]
        val_set.append([uid, iid, cid, rel])
        val_dict[uid].append(iid)

    # print('train data', train_set[100])
    # print('user behavior', user_profile_dict[100])
    with open(out_file, 'wb') as f:
        pkl.dump([train_set, val_set, test_set, user_profile_dict, cat_dict], f) # [uid, iid, cid, rel], {uid: [iid, cate_multi]} -> user_profile_dict only contains positive info
    print(' =============data split done=============')
    return train_set, val_set, test_set, user_profile_dict, cat_dict


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    # start_date = date(2017, 11, 25)
    # end_date = date(2017, 12, 3)
    data_dir = 'data/'
    data_set_name = 'ml-20m'
    num_clusters = 5
    raw_dir = os.path.join(data_dir, data_set_name + '/raw_data/ratings.csv')
    cate_dir = os.path.join(data_dir, data_set_name + '/raw_data/movies.csv')
    stat_dir = os.path.join(data_dir, data_set_name + '/raw_data/data.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    diver_dir = os.path.join(processed_dir, 'diversity.item') # to learn diversity?

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if os.path.isfile(stat_dir):
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = pkl.load(open(stat_dir, 'rb')) # uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id
        print('loaded stat file')
    else:
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = stat_data(raw_dir, cate_dir,
                                                                                             stat_dir, diver_dir)


    processed_data_dir = os.path.join(processed_dir, 'data.data')
    if os.path.isfile(processed_data_dir):
        train_file, val_file, test_file, user_profile_dict, cat_dict = pkl.load(open(processed_dir + '/data.data', 'rb'))
        print('loaded data for initial rankers')
    else:
        train_file, val_file, test_file, user_profile_dict, cat_dict = split_data(raw_dir, stat_dir, diver_dir, processed_data_dir)
