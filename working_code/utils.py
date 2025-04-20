import pickle

import numpy as np
import pickle as pkl
from collections import defaultdict
import random
from sklearn.metrics.pairwise import euclidean_distances

# from diversity_baselines import MMR, DPP

def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_batch(data, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


def get_aggregated_batch(data, batch_size, batch_no):
    return [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))]


def padding_list(seq, max_len):
    seq_length = min(len(seq), max_len)
    if len(seq) < max_len:
        seq += [np.zeros_like(np.array(seq[0])).tolist()] * (max_len - len(seq))
    return seq[:max_len], seq_length


def construct_behavior_data(data, user_history, max_len, multi_hot=False): # here user_history is only the 0.2 part of the whole data.data, no intersection with train/val/test set
    target_user, target_item, user_behavior, label, seq_length = [], [], [], [], []
    for d in data: # from train/test_file
        uid, iid, cid, lb, = d # lb=relevance, 1 if rating>4
        if uid in user_history:
            target_user.append(uid)
            if multi_hot:
                ft = [iid]
                ft.extend(cid)
            else:
                ft = [iid, cid]
            target_item.append(ft)
            user_list, length = padding_list(user_history[uid], max_len)
            user_behavior.append(user_list)
            label.append(lb)
            seq_length.append(length)
    return target_user, target_item, user_behavior, label, seq_length


def get_user_div_history(user_history, max_behavior_len, items_div, multi_hot=False): # already specified uid in input, user_history is now the list of user's liked items and its cate_multi -> only positive
    '''
    generate separate user history for learning theta in diver
    items belong to the category that has the largest prob
    '''
    num_cat = len(items_div[user_history[0][0]]) # user_history[0][0] is the iid
    div_dict = {cid: [] for cid in range(num_cat)}
    user_div = []
    for iid in user_history: # it's actually uid
        if iid[0] == 0:
            continue # this user has no behavior history? (at least after filtering)
        if multi_hot:
            for i, v in enumerate(iid[1:]):
                if v:
                    div_dict[i].append(iid[0])
        else:
            itm_div_class = np.argmax(np.array(items_div[iid[0]])) #items: [iid, cid]
            div_dict[itm_div_class].append(iid[0])
    for cid, div_hist in div_dict.items():
        if len(div_hist) < max_behavior_len:
            div_hist += [0] * (max_behavior_len - len(div_hist))
        user_div.extend(div_hist[:max_behavior_len])
    assert len(user_div) == num_cat * max_behavior_len
    return user_div


def dcm_theta(user_profile_dict, item_div_dir, num_clusters, user_set, out_file, multi_hot=False):
    '''
    to write user-item div into file 'out_file' (dcm.theta)
    '''
    count_cat = {uid: np.zeros(num_clusters) for uid in user_set} # first create a shell for user embeddings based on num_clusters and user_set
    theta = {}
    items_div = pkl.load(open(item_div_dir, 'rb'))
    for uid, user_list in user_profile_dict.items(): # {uid: [iid, cate_multi_hot], [iid, cate_multi_hot]...}
        for itm in user_list: # [iid, cate_multi_hot]
            iid = itm[0]
            if iid == 0: # iid is now the list of [iid, cate_multi]
                continue
            if multi_hot:
                count_cat[uid] += np.array(itm[1:]) # concatenate the cate_multi for of each record for each user -> to count the num of cate (repetable)
            else:
                itm_div_class = np.argmax(np.array(items_div[iid])) # pick out the cluster that has the most div_value for this iid?
                count_cat[uid][itm_div_class] += 1 # now has created an embedding that with uid as keys, with the num of each cluster has popped up in items_div as value for its new value
    for uid, cats in count_cat.items():
        if multi_hot:
            theta[uid] = normalize(cats) # the theta (diversity degree) of one user is acquired by normalizing her num of cates
        else:
            theta[uid] = normalize(cats)
    with open(out_file, 'wb') as f:
        pkl.dump(theta, f) # saved items diversity embeddings for each uid
    print('dcm theta saved')


def rank(users, items, preds, labels, out_file, item_div_dir, user_history, max_behavior_len, multi_hot=False):
    '''
    for ranking-stage, creating initial list for RAPID (DIN.rankings.training/test)
    '''
    items_div = pkl.load(open(item_div_dir, 'rb')) # diversity.item [iid, cate_multi_hot]
    rankings = defaultdict(list)
    with open(out_file, 'w') as fout:
        for uid, iid, pred, lb in zip(users, items, preds, labels): # lb=label
            rankings[uid].append((iid, pred, lb)) # create initial ranking list for each uid
        for uid, user_list in rankings.items():
            if len(user_list) >= 3: # limit the seq of each user up to 3
                user_list = sorted(user_list, key=lambda x: x[1], reverse=True) # sort by predictions. moved this to load_data()
                for itm in user_list: # [iid, pred, lable]
                    hist_u = list(map(str, get_user_div_history(user_history[uid], max_behavior_len, items_div, multi_hot))) # user_history is user_profile_dict {uid: [iid, cate_multi]...}
                    ft, p, l = itm # feature is item feature [iid, multi_cate]
                    if multi_hot:
                        i, c = ft[0], list(map(int, ft[1:])) # [iid, cate_multi]
                        div_i = list(map(str, items_div[i]))
                        fout.write(','.join([str(l), str(p), str(uid), str(i)] + list(map(str, c)) + div_i + hist_u))
                    else:
                        i, c = ft
                        div_i = list(map(str, items_div[i]))
                        fout.write(','.join([str(l), str(p), str(uid), str(i), str(c)] + div_i + hist_u))
                    fout.write('\t')
                fout.write('\n')


def get_last_click_pos(my_list):
    '''
    get the index of the last positive click (1)
    my_list: (list) should be a int-list filled with 0/1
    '''
    if sum(my_list) == 0 or sum(my_list) == len(my_list): # if all the values are 0 -> all non-clicked | all are 1 -> all clicked
        return len(my_list) - 1 # return the last index
    return max([index for index, el in enumerate(my_list) if el]) # if el==1, return the index of last pos-valued record


def construct_list(data_dir, max_time_len, num_cat, is_train, multi_hot=False):
    '''
    constuct training and test files
    is_train: (bool) decides train(True)/test(False)
    multi_hot: (bool) 'True' for ml-20m and citeulike | 'False' for ad
    '''
    if multi_hot:
        num_sample = 20 if is_train else 4 # need to change for citeulike
    else:
        num_sample = 50 if is_train else 10
    feat, click, seq_len, user_behavior, items_div, uid, pred = [], [], [], [], [], [], []
    with open(data_dir, 'r') as f: # read DIN.rankings.train/test.{max_behavior_len}
        for line in f:
            items = line.strip().split('\t')

            uid_i, feat_i, click_i, user_i, div_i, pred_i = [], [], [], [], [], []
            for itm in items:
                itm = itm.strip().split(',')
                click_i.append(int(itm[0]))
                pred_i.append(float(itm[1]))
                uid_i.append(int(itm[2]))
                if multi_hot:
                    feat_i.append(list(map(int, list(map(float, itm[3:4 + num_cat]))))) # []
                    div_i.append(normalize(list(map(float, itm[4 + num_cat:4 + 2*num_cat]))).tolist())
                    user_i.append(list(map(int, itm[4 + 2*num_cat:])))
                else:
                    feat_i.append(list(map(int, map(float,itm[3:5]))))
                    div_i.append(list(map(float, itm[5:5 + num_cat])))
                    user_i.append(list(map(int, itm[5 + num_cat:])))
            rankings = list(zip(click_i, pred_i, feat_i, div_i, user_i))
            if len(rankings) > max_time_len: # if the number of candidates beyond the max_time_len
                for i in range(num_sample): # as long as it's within num_sample (multi_hot: 20 for training, 4 for testing | not multi_hot: 50 for training, 10 for testing)
                    cand = random.sample(rankings, max_time_len) # randomly select candidates
                    click_i, pred_i, feat_i, div_i, user_i = zip(*cand)
                    seq_len_i = len(feat_i)
                    sorted_idx = sorted(range(len(pred_i)), key=lambda k: pred_i[k], reverse=True)
                    pred_i = np.array(pred_i)[sorted_idx].tolist()
                    click_i = np.array(click_i)[sorted_idx].tolist()
                    feat_i = np.array(feat_i)[sorted_idx].tolist()
                    div_i = np.array(div_i)[sorted_idx].tolist()

                    feat.append(feat_i[:max_time_len]) # only extracts the first max_time_len's features i.e. 20
                    user_behavior.append(user_i[0])
                    click.append(click_i[:max_time_len])
                    items_div.append(div_i[:max_time_len])
                    seq_len.append(min(max_time_len, seq_len_i))
                    uid.append(uid_i[0])
                    pred.append(pred_i[:max_time_len])
            else: # fill the rest of max_time_len with [0,0,...] -> could cause the wrong item's feat_id '0'
                click_i, pred_i, feat_i, div_i, user_i = zip(*rankings)
                seq_len_i = len(feat_i)
                sorted_idx = sorted(range(len(pred_i)), key=lambda k: pred_i[k], reverse=True)
                pred_i = np.array(pred_i)[sorted_idx].tolist()
                click_i = np.array(click_i)[sorted_idx].tolist()
                feat_i = np.array(feat_i)[sorted_idx].tolist()
                div_i = np.array(div_i)[sorted_idx].tolist()
                
                feat.append(feat_i + [np.zeros_like(np.array(feat_i[0])).tolist()] * (max_time_len - seq_len_i)) # [array]*(20-len(feat_i))
                #print(feat_i)
                #print(feat_i + [np.zeros_like(np.array(feat_i[0])).tolist()] * (max_time_len - seq_len_i))
                user_behavior.append(user_i[0])
                click.append(click_i + [0] * (max_time_len - seq_len_i))
                items_div.append(div_i + [np.zeros_like(np.array(div_i[0])).tolist()] * (max_time_len - seq_len_i))
                seq_len.append(seq_len_i)
                uid.append(uid_i[0])
                pred.append(pred_i + [-1e9] * (max_time_len - seq_len_i))

    return feat, click, seq_len, user_behavior, items_div, uid, pred


def load_data(data, click_model, num_cate, max_hist_len, test=False):
    '''
    load data via click_model DCM
    '''
    feat, click, seq_len, user_behavior, items_div, uid, _ = data
    feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm = [], [], [], [], [], [], []
    i = 0
    for feat_i, click_i, seq_len_i, user_i, div_i, uid_i in zip(feat, click, seq_len, user_behavior, items_div, uid): # len(user_behavior) should be 100 if len(max_behavior_len)==5, should be 200 if max_beh_len==10
        behav_len = process_seq(user_i, num_cate, max_hist_len) # len(user_i) == len(num_cate) * len(max_hist_len)
        item_list_i = [itm[0] for itm in feat_i]
        label_cm_i = click_model.generate_clicks(uid_i, item_list_i, len(item_list_i)) # the label data also come from click_model
        if not test:
            if sum(label_cm_i) == len(label_cm_i) or sum(label_cm_i) == 0: # either the user has clicked all the showed items so far or has clicked no one -> skip the following, no appending
                continue
        # only appends the meaningful features, if test=True, then appends all data
        feat_cm.append(feat_i)
        user_behavior_cm.append(user_i)
        behav_len_cm.append(behav_len)
        label_cm.append(label_cm_i) # the label now decided by DCM
        items_div_cm.append(div_i)
        seq_len_cm.append(seq_len_i)
        uid_cm.append(uid_i)

    return feat_cm, label_cm, seq_len_cm, user_behavior_cm, items_div_cm, uid_cm, behav_len_cm


def get_hist_len(seq):
    length = 0
    while length < len(seq) and seq[length] > 0:
        length += 1
    return length

def process_seq(seq, num_cate, seq_len):
    len_list = []
    seq = np.reshape(np.array(seq), [num_cate, seq_len]) # num_cate==20, seq_len==5 if max_behavior_len==5, num_cate==20, seq_len==10 if max_behavior_len==10
    for idx in range(num_cate):
        len_list.append(get_hist_len(seq[idx]))
    return len_list

def rerank(attracts, terms):
    '''
    for re-ranking stage, creating re-ranked list
    
    input:
    attracts: (list) list of preds
    terms: (list) list of terms (prob.? it's actually the same of preds)
    output:
    sorted reco list with index values
    '''
    val = np.array(attracts) * np.array(np.ones_like(terms)) # multiplies preds * 1-array
    return sorted(range(len(val)), key=lambda k: val[k], reverse=True)


def evaluate(labels, preds, terms, users, items, click_model, items_div, scope_number, is_rerank): # len(users)==768 768 * [20], each row is a 20-long(20 uid) seq, and each unique seq will repeat 4 times???
    '''
    scope_number: (int) 5/10 (might be ndcg@k's this 'k', top-k)
    items: features from train/test_file (e.g. [2122,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0] -> item_id + num_cat
    '''
    ndcg, utility, ils, diversity, satisfaction, mrr = [], [], [], [], [], []
    for label, pred, term, uid, init_list, item_div in zip(labels, preds, terms, users, items, items_div): # len(uid)==20 [6, 1, 6, 2, 0, 0, 4, 2, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 0, 0] 
        # each tuple contains one ele from each of the original seq in zip(), if the seq in zip() are of unequal lengths, zip stops after the shortest seq ends
        # each var. only contains the corresponding seq's ele
        # items makes init_list e.g. [6474, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        init_list = np.array([item[0] for item in init_list]) # why only takes the first ele of each item list? -- i got it, it's the item_id e.g. 6474, in the end init_list looks like [6474 4116 7792 4720 4286 2445 5138 469 9391 6596 1063 6750 3344 3710 9658 7343 6620 4731 9674  593]
        seq_len = get_last_click_pos(init_list) # get the last positive click's index, but why??? does it mean anything here based on the case that init_list is just a list of item_id? -- seems that it shouldnt be item_id
        # 19, cuz it's 0-starting index with len(init_list)==20
        item_div = np.array(item_div)


        if is_rerank:
            # rerank list
            final = rerank(pred, term)
            final_list = init_list[final]
        else:
            final = list(range(len(pred))) # evaluate initial rankers
            final_list = init_list[final]
        #return init_list, item_div, pred, term, final, final_list #⭐️

        click = np.array(label)[final].tolist() # reranked labels
        gold = sorted(range(len(click)), key=lambda k: click[k], reverse=True) # optimal list for ndcg
        item_div_final = item_div[final]

        ideal_dcg, dcg, util, rr = 0, 0, 0, 0 # ndcg is acquired by using ideal_dcg
        scope_number = min(scope_number, seq_len+1) # ⭐️choose scope_number or seq_len as the final index, cuz the click_value after seq_len are all 0 -> useless anyway. But if it's about how many items shouldnt it be seq_len+1? ⭐️ changed to seq_len+1
        scope_final = final[:scope_number] 
        scope_gold = gold[:scope_number]
        scope_div = item_div_final[:scope_number]

        for _i, _g in zip(range(1, scope_number + 1), scope_gold): # scope_number==5/10
            dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
            ideal_dcg += (pow(2, click[_g]) - 1) / (np.log2(_i + 1))
        _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.

        #return uid, users #⭐️ uid: [20], users: [768, 20] 768/4=192 unique uid

        new_click = click_model.generate_click_prob(uid, final_list, scope_number) # ⚡️error! uid should be a single int, yeah, should take data_batch[-2] in run_test.py's eval() as input which is exactly uid but not the index -1
        rr = 1. / (np.argmax(np.array(new_click)) + 1)

        ndcg.append(_ndcg)
        utility.append(sum(new_click)) # the sum of click_prob of all items?
        mrr.append(rr)
        ils.append(np.sum(euclidean_distances(scope_div, scope_div)) / (scope_number * (scope_number - 1) / 2))
        diversity.append(np.sum(1 - np.prod(1 - scope_div, axis=0)))
        satisfaction.append(click_model.generate_satisfaction(uid, final_list, scope_number))
    return np.mean(np.array(ndcg)), np.mean(np.array(utility)), np.mean(np.array(ils)), np.mean(
        np.array(diversity)), np.mean(np.array(satisfaction)), np.mean(np.array(mrr)), \
        [ndcg, utility, diversity, satisfaction, mrr]
