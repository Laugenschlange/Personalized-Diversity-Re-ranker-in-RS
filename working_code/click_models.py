import os
import numpy as np
import pickle as pkl


from utils import get_last_click_pos, softmax


class DCM(object):
    '''
    read dcm.theta (user & cate) file and get a click prob for each candidate item
    '''
    def __init__(self, max_time_len, num_cat, user_set, item_set, item_div_path, lmbd, theta_file):
        self.attr = {iid: 0. for iid in item_set} # initialize the attributes
        self.term = np.zeros(max_time_len)
        self.theta = pkl.load(open(theta_file, 'rb')) # get its theta (dcm.theta) (user&cate) value for each uid -> each user's normalized(num_of_cate)
        print(len(self.theta))
        self.max_time_len = max_time_len
        self.num_cat = num_cat
        self.user_set = user_set
        self.item_set = item_set
        self.items_div = pkl.load(open(item_div_path, 'rb')) # [item, cate_multi_hot]
        self.lmbd = lmbd

    def train(self, train_lists): # [feature, click, seq_len, user_behavior, item_div, uid, _]
        # updates DCM's self.attr
    
        feat, click, seq_len, user_behavior, items_div, uid, _ = train_lists

        attr_nominator = {iid: 0. for iid in self.item_set} 
        attr_denominator = {iid: 0. for iid in self.item_set} # expects feat_id, not iid

        term_nominator  = {iid: 0. for iid in range(self.max_time_len)} 
        term_denominator = {iid: 0. for iid in range(self.max_time_len)}

        for seq_feat, seq_click, user_hist, u, item_div in zip(feat, click, user_behavior, uid, items_div):
            last_click = get_last_click_pos(seq_click) 
            for i in range(last_click + 1):
                if seq_feat[i][0] != 0: # only if it's a valid item feat_id
                    attr_denominator[seq_feat[i][0]] += 1
                    if seq_click[i] == 1: # if pos
                        attr_nominator[seq_feat[i][0]] += 1
                        term_denominator[i] += 1
                        if i == last_click:
                            term_nominator [i] += 1

        # relevance = num clicks / num impressions
        for iid, val in attr_denominator.items(): # [iid, value]
            if val != 0.:
                self.attr[iid] = attr_nominator[iid] / float(val) # gives the weights of this iid in the true user-click history, num_clicked/num_occurred 

        # term = num click as last click at this pos / num click at this pos
        for pos, val in term_denominator.items(): # computes rank-relevant values
            if val != 0.:
                self.term[pos] = term_nominator [pos] / float(val)
                print(term_nominator [pos], float(val), self.term[pos])
        self.term[-1] = 1

    def generate_click_prob(self, uid, input_list, seq_len):
        theta = self.theta[uid] # user's personal div. preference -> normalized(num_of_cate)
        input_list = input_list[:seq_len] # item's feat_ids
        attr_list = []
        cum_prod = np.ones(self.num_cat) # a full 1 cat_matrix
        for idx, itm in enumerate(input_list): # index, feat_id
            if itm == 0:
                attr = 0
            else:
                delta_gain = cum_prod * self.items_div[itm] # diversity gain
                cum_prod *= 1 - np.array(self.items_div[itm]) # irrelevant
                attr = self.lmbd * self.attr[itm] + (1 - self.lmbd) * np.sum(theta * delta_gain) # !!!!!!!!
            attr_list.append(attr)
        if seq_len < self.max_time_len:
            attr_list += [0] * (self.max_time_len - seq_len)
        return attr_list # a list of click prob of each item 

    def generate_clicks(self, uid, input_list, seq_len):
        attr_list = self.generate_click_prob(uid, input_list, seq_len)
        clicks = []
        for attr in attr_list:
            clicks.append(1 if np.random.rand() < attr else 0)
        return clicks

    def generate_satisfaction(self, uid, input_list, seq_len):
        theta = self.theta[uid]
        input_list = input_list[:seq_len]
        attr_list = []
        cum_prod = np.ones(self.num_cat)
        for idx, itm in enumerate(input_list):
            if itm == 0:
                attr = 0
            else:
                delta_gain = cum_prod * self.items_div[itm]
                cum_prod *= 1 - np.array(self.items_div[itm])
                attr = self.lmbd * self.attr[itm] + (1 - self.lmbd) * np.sum(theta * delta_gain)
            attr_list.append(attr)

        attrs = np.array(attr_list)
        terms = softmax(self.term[:seq_len])
        return 1 - np.prod(1 - attrs * terms)