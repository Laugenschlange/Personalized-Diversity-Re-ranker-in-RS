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
from collections import defaultdict
from scipy.sparse import csr_matrix # compressed sparse row
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity




def tag_topic_assign(tag_dir, item_tag_dir, item_tag_matrix_dir, item_topic_dir, level, n_topic):
    # load item_tag_matrix and topic_df
    if os.path.isfile(item_tag_matrix_dir):
        item_tag_df = pd.read_pickle(item_tag_matrix_dir)
        print('loaded item_tag_matrix.pkl')
        #topic_df = pd.read_pickle("topic_df.pkl")
    else:
        df_tag = pd.read_csv(tag_dir, header=None, names=['tag_name']) 
        tags_dict = df_tag.tag_name.to_dict()
        with open(item_tag_dir, 'r') as f:
            item_tag_data = [line.strip().split(' ') for line in f]
        item_tag_data = [list(map(int, row[1:])) for row in item_tag_data] # ignore the first col, cuz it's the total num of tags of this item
        item_tag_dict = {item_id: tag_ids for item_id, tag_ids in enumerate(item_tag_data)} # index starts from 0
    
        all_tags = sorted(set(tag_id for tags in item_tag_dict.values() for tag_id in tags)) # 46390 cates
        # initialize matrix as a dict of dict
        item_tag_matrix = defaultdict(lambda: {tag: 0 for tag in all_tags})
    
        # fill the matrix 
        for item_id, tags in item_tag_dict.items():
            if tags: # if tags is not empty
                for tag in tags:
                    item_tag_matrix[item_id][tag] = 1
            else: # if the item has no tags, still assign it all 0 for each tag
                for tag in all_tags:
                    item_tag_matrix[item_id][tag] = 0
        item_tag_df = pd.DataFrame.from_dict(item_tag_matrix, orient='index').fillna(0)
        # rename cols to tag names
        item_tag_df.columns = [tags_dict[tag_id] for tag_id in item_tag_df.columns]
        # save df into .pickle format
        item_tag_df.to_pickle("item_tag_matrix.pkl")
        print("item_tag_matrix.pkl saved")

    if os.path.isfile(item_topic_dir):
        dat_format = pd.read_csv(item_topic_dir)
        print(f'loaded item-topic_{level}_{n_topic}.dat')
    else:
        # compress the sparse matrix into less sparse one
        compressed_matrix = csr_matrix(item_tag_df.values)
        #### Apply LDA (NMF not applied, cuz it's mostly deterministic and less general, flexible)
        # assumes each article is a mix of latent topics and each topic is a mix of tags
        # unlike NMF, LDA is probabilistic (works with distributions rather than just factorizing)
        n_topics=n_topic
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        W_lda = lda_model.fit_transform(compressed_matrix) # item-topic-matrix
        H_lda = lda_model.components_ # topic-tag-matrix
    
        article_topic_lda_df = pd.DataFrame(W_lda, index=item_tag_df.index, columns=[f"Topic {i+1}" for i in range(n_topics)])
        article_topic_lda_df = article_topic_lda_df.sort_index() # resort values by index/item_id
        topic_tag_lda_df = pd.DataFrame(H_lda, columns=item_tag_df.columns, index=[f"Topic {i+1}" for i in range(n_topics)])
        article_topic_lda_df.to_pickle(f"data/cite/raw_data/item_topic_lda_{level}_{n_topic}.pkl")
        topic_tag_lda_df.to_pickle(f"data/cite/raw_data/topic_tag_lda_{level}_{n_topic}.pkl")
        print(f"item_topic_lda_{level}_{n_topic}.pkl and topic_tag_lda.pkl saved")
    
        # set prob threshold, filter certain topics for each item
        prob_threshold = 0.1
        # each item should have at least one topic that its probability is over threshold, otherwise there's something wrong with tag-topic-assignment
        # add a check to ensure each item has at least one topic with its probability over threshold
        def get_topics(x):
            topics_above_threshold = [i for i, value in enumerate(x) if value > prob_threshold]
            return topics_above_threshold if topics_above_threshold else None # if there's no assigned topic, return None
        
        top_topics_df = article_topic_lda_df.apply(get_topics, axis=1)
        num_items_below_threshold = (article_topic_lda_df.max(axis=1) < prob_threshold).sum()
        
        dat_format = top_topics_df.apply(lambda x: ' '.join(map(str, x)) if x is not None else 'None')
        #dat_format.columns = ['item_id', 'topics']
        dat_format.to_csv(f'data/cite/raw_data/item-topic_{level}_{n_topic}.dat', sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar=' ')
        print(f"File 'item-topic_{level}_{n_topic}.dat' saved.")
        print("cate_dir is ready.")
        
    

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
    cate_dir: item-topic.dat
    stat_dir: an input file to save the statistics of the data
    diver_dict: diversity.item, also an input file to save the embeddings of iid based on cates (to learn each item's cate_expression)
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
    none_topic_num = 0
    pos, neg = 0, 0

    with open(cate_dir, 'r', encoding='utf-8') as r:
        '''
        item-topic.dat (for citeulike, to reduce the dim of category)
        '''
        for iid, row in enumerate(r):
            if row.strip() != 'None': 
                values = list(map(int, row.strip().split()))
                cates = values[:] # remove the first value for item-tag.dat but not for item-topic.dat
                item_cate[iid] = cates
                for cate in cates:
                    cate_num[cate] += 1 # cate-id-based indexing
            else:
                item_cate[iid] = None

    print('num of cate:', len(cate_num))
    for k, v in cate_num.items():
        print('cate:', k, '  num:', v)


    with open(raw_dir, 'r', encoding='utf-8') as r:
        '''
        users.dat: users' collected articles
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

    # no need of further filtering, the data has already been processed with only users that with more than 10 references in the library left
    # just keep the part of creating uid_set, iid_set
    for uid in user_item.keys(): # only keep users that with more than 200 interactions
        #if len(user_item[uid]) > 200: # means that the user has to have interacted with more than 200 items?
            uid_set.add(uid)
            for item in user_item[uid]:
                iid_set.add(item) # add item_id, because iid_set if of 'set' format, even duplicated values are added, still only the unique remained

    for iid in iid_set: # iterate all iid in iid_set to get their cates
        cates = item_cate[iid] # a list of cate
        if cates: # if not None
            for cate in cates:
                filter_cate_num[cate] += 1 # organized according to cate, count their nums
        else:
            none_topic_num += 1
    # filter out these items that with no topics:

    print('AFTER FILTER \nnum of cate:', len(filter_cate_num))
    for k, v in filter_cate_num.items():
        print('cate:', k, '  num:', v)
    #print('# total record:', filter_pos + filter_neg, '  when pos>=4, pos vs neg', filter_pos * 1.0 / filter_neg)
    
    
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(filter_cate_num.keys()) # keys are cate names, values are counts


    print('num user', len(uid_list))
    print('num item', len(iid_list))
    print('num items that without any topic', none_topic_num)
    

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
        # iid_remap_dict has iid as keys, the index as values
        # by doint this, iid_cate_map and iid_remap_dict are cooresponding
        # create an iid_cate_map -> an embedding of iid in terms of cates
        iid_cate_map[iid_remap_dict[iid]] = generate_cate_multi_hot(cates, cid_dict)

    with open(stat_dir, 'wb') as f: # what is stat_dir? A pre-created file? -- seems so
        # serialize the info as a byte stream and save them in f
        # save the list of uid, iid and cid and cid_dict (dont see the diff.) and total feature num
        pkl.dump([uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id], f) # cid_dict: {cid: idx}
    with open(diver_dict, 'wb') as f:
        # to save embedding of iid based on cate
        pkl.dump(iid_cate_map, f) # [feature_id] = item_matrix_based_on_cid_dict -> represents topic coverage of each item / topic distribution
    print('======= statistic done ============')
    return uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id


def generate_cate_multi_hot(cates, cate_dict):
    multi_hot = np.zeros(len(cate_dict)) # first create the right size of zero-matrix
    if cates:
        for i in cates:
            # here shows the meaning of separately creating a cid_dict
            # just to match the index between multi_hot and cate_dict 
            # with cate as key/indicator 
            # which cates this iid has, assign 1 to this position in multi_hot
            multi_hot[cate_dict[i]] = 1

    return multi_hot


def split_data(in_file, statistics, diver_dir, out_file, level):
    '''
    split data for training, validation and test
    in_file: users.dat
    spliting_ratio: 2:3:4:1 (user_profile_data:trainig:validation:test)
    out_file: data.data.{level}
    '''
    user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, _ = pkl.load(open(statistics, 'rb')) # user_remap_dict: {uid: fea_id}, cat_remap_dict: {cid: fea_id}, cid_list: {cid: idx}
    iid_cate_map = pkl.load(open(diver_dir, 'rb'))
    user_set = set(user_remap_dict.keys()) # uids
    records = []
    with open(raw_dir, 'r', encoding='utf-8') as r:
        '''
        users' collected articles
        '''
        embeddings = np.array([iid_cate_map[fea_id] for iid, fea_id in item_remap_dict.items()])
        simi_matrix = cosine_similarity(embeddings)
        top_simi_iid = {} # for saving each iid's top-s iids

        # get similarity scores for each item pair
        for iid in item_remap_dict.keys(): # iid
            sim_scores = list(enumerate(simi_matrix[iid]))
            filtered_scores = [
                (other_iid, score) for other_iid, score in sim_scores
                if other_iid != iid #all_iid[other_idx] not in collected_art and 
            ] # except itm itself
            if level=='hard': # hard takes the top 5 similar articles as negative
                top_s = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:5] # for each iid select the most similar 5 iids as negative
            if level=='easy': # easy takes the top 5 dissimilart articles as negative
                top_s = sorted(filtered_scores, key=lambda x: x[1], reverse=False)[:5] # for each iid select the most dissimilar 5 iids as negative
            top_s_iid = [(i, score) for i, score in top_s]
            top_simi_iid[iid] = top_s_iid
            
        for uid, row in enumerate(r):
            values = list(map(int, row.strip().split()))
            collected_art = values[1:] # each user's collected article ids
            
            all_recs = set()
            for art in collected_art:
                ##### compute each article-pari's similarity in terms of cates based on: 1. Jaccard similarity 2. Cosine similarity
                # add positive info
                if uid in user_set:
                    records.append([uid, art, 1]) # manually add rating=1 as positive info
                
                # add negative info
                for itm, _ in top_simi_iid[art]:
                    all_recs.add(itm)
            final_recs = all_recs - set(collected_art)
            for art_neg in final_recs:
                if uid in user_set:
                    records.append([uid, art_neg, 0]) # manually add rating=0 as negative info
        
    # shuffle 
    rec_num = len(records)
    random.shuffle(records)
    user_profile_data, train_data, val_data, test_data = records[:int(rec_num*0.2)], \
      records[int(rec_num*0.2):int(rec_num*0.5)], records[int(rec_num*0.5):int(rec_num*0.9)], records[int(rec_num*0.9):]

    cat_dict = pkl.load(open(diver_dir, 'rb')) # iid_cate_map, cate matrix based on iid

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

  
    with open(out_file, 'wb') as f:
        pkl.dump([train_set, val_set, test_set, user_profile_dict, cat_dict], f)
    print(' =============data split done=============')
    return train_set, val_set, test_set, user_profile_dict, cat_dict
    
if __name__ == '__main__':
    # parameters
    random.seed(1234)
    # start_date = date(2017, 11, 25)
    # end_date = date(2017, 12, 3)
    data_dir = 'data/'
    data_set_name = 'cite'
    #num_clusters = 5
    level = 'hard'
    n_topic = 20
    raw_dir = os.path.join(data_dir, data_set_name + '/raw_data/users.dat')
    tag_dir = os.path.join(data_dir, data_set_name + '/raw_data/tags.dat')
    item_tag_dir = os.path.join(data_dir, data_set_name + '/raw_data/item-tag.dat') # ⭐️ use item-topic.dat instead 
    item_tag_matrix_dir = os.path.join(data_dir, data_set_name + '/raw_data/item_tag_matrix.pkl')
    article_topic_lda_dir = os.path.join(data_dir, data_set_name + f"/raw_data/item_topic_lda_{level}_{n_topic}.pkl")
    topic_tag_lda_dir = os.path.join(data_dir, data_set_name + f"/raw_data/topic_tag_lda_{level}_{n_topic}.pkl")
    cate_dir = os.path.join(data_dir, data_set_name + f'/raw_data/item-topic_{level}_{n_topic}.dat')
    stat_dir = os.path.join(data_dir, data_set_name + f'/raw_data/data_{level}_{n_topic}.stat')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')
    diver_dir = os.path.join(processed_dir, f'diversity_{level}_{n_topic}.item') # to get each item's cate distribution


    tag_topic_assign(tag_dir, item_tag_dir, item_tag_matrix_dir, cate_dir, level, n_topic)
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if os.path.isfile(stat_dir):
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = pkl.load(open(stat_dir, 'rb')) # uid_remap_dict, iid_remap_dict, cid_remap_dict, cid_dict, feature_id
        print('loaded stat file')
    else:
        user_remap_dict, item_remap_dict, cat_remap_dict, cid_list, feature_size = stat_data(raw_dir, cate_dir,
                                                                                             stat_dir, diver_dir)
        #stat_data(raw_dir, cate_dir, stat_dir, diver_dir)
     processed_data_dir = os.path.join(processed_dir, f'data_{level}_{n_topic}.data') # ⭐️⭐️⭐️⭐️⭐️⭐️⭐️ need to add level param too 
    
    if os.path.isfile(processed_data_dir):
        train_file, val_file, test_file, user_profile_dict, cat_dict = pkl.load(open(processed_dir + f'/data_{level}_{n_topic}.data', 'rb'))
        print('loaded data for initial rankers')
    else:
        train_file, val_file, test_file, user_profile_dict, cat_dict = split_data(raw_dir, stat_dir, diver_dir, processed_data_dir, level)

