import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import statistics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, \
accuracy_score, roc_auc_score, roc_curve, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import copy
from scipy.spatial import distance
import itertools
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import csv


# Read ComE node embs per timestep [id, emb]

folder = os.listdir('ComE_per_timestep/embs')
path = 'ComE_per_timestep/embs'
ComE_id_embs = []
for file in folder:
    ComE_id_embs.append(np.genfromtxt(os.path.join(path, file), dtype=None).tolist())

# Read ComE labels per timestep

folder = os.listdir('ComE_per_timestep/labels_pred')
path = 'ComE_per_timestep/labels_pred'
ComE_lbls = []
for file in folder:
    ComE_lbls.append(np.genfromtxt(os.path.join(path, file), dtype=None).tolist())

# Node ids per timestep

node_ids = []
for step in ComE_id_embs:
    tmp = [id_emb[0] for id_emb in step]
    node_ids.append(tmp)

# [Node_id, clr] per timestep

id_clr = []
for i in range(len(node_ids)):
    tmp = {}
    for ind,node in enumerate(node_ids[i]):
        tmp[node] = ComE_lbls[i][ind]
    id_clr.append(tmp)

# Clustered nodes per timestep

clustered_nodes_init = []
for ind,i in enumerate(id_clr):
    clrids_uniq = set(i.values())
    d = {}
    for clrid in clrids_uniq:
        d[clrid] = [k for k in i.keys() if i[k] == clrid]
    clustered_nodes_init.append(d)

clustered_nodes = []
for s in clustered_nodes_init:
    per_step = []
    for k,v in sorted(s.items()):
        per_step.append(v)
    clustered_nodes.append(per_step)
                    
# ------------------------------ READ FEATURES -------------------------------

# ComE FEATURES

folder = os.listdir('ComE_features_per_timestep/')
path = 'ComE_features_per_timestep/'
id_ComE_feats_clr = []
id_ComE_feats_out = []
id_ComE_feats_gbl = []
id_ComE_feats_clrout = []
id_ComE_feats_clrgbl = []
id_ComE_feats_all = []
id_ComE_distmed = []
for file in folder:
    df_ComE = pd.read_csv(os.path.join(path,file), names=['node_id', \
        'distin_med_eucl', 'distin_med_cos', 'distin_med_l1',\
        'distout_med_eucl', 'distout_med_cos', 'distout_med_l1',\
        'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg',\
        'distin_cos_max', 'distin_cos_min', 'distin_cos_avg',\
        'distin_l1_max', 'distin_l1_min', 'distin_l1_avg',\
        'distout_eucl_max', 'distout_eucl_min', 'distout_eucl_avg',\
        'distout_cos_max', 'distout_cos_min', 'distout_cos_avg',\
        'distout_l1_max', 'distout_l1_min', 'distout_l1_avg', \
        'dist_glob_max_eucl', 'dist_glob_min_eucl', 'dist_glob_avg_eucl', \
        'dist_glob_max_cos', 'dist_glob_min_cos', 'dist_glob_avg_cos', \
        'dist_glob_max_l1', 'dist_glob_min_l1',  'dist_glob_avg_l1'], skiprows=1)
    df_ComE_clr = df_ComE[['node_id', 'distin_med_eucl', \
                          'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg']]
    df_ComE_out = df_ComE[['node_id', 'distout_med_eucl', \
                        'distout_eucl_max', 'distout_eucl_min', 'distout_eucl_avg']]
    df_ComE_gbl = df_ComE[['node_id', 'distout_med_eucl', \
                        'dist_glob_max_eucl', 'dist_glob_min_eucl', 'dist_glob_avg_eucl']]
    df_ComE_clrout = df_ComE[['node_id', 'distin_med_eucl', 'distout_med_eucl', \
                          'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg', \
                        'distout_eucl_max', 'distout_eucl_min', 'distout_eucl_avg']]
    df_ComE_clrgbl = df_ComE[['node_id', 'distin_med_eucl', \
                          'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg', \
                        'dist_glob_max_eucl', 'dist_glob_min_eucl', 'dist_glob_avg_eucl']]
    df_ComE_all = df_ComE[['node_id', 'distin_med_eucl', 'distout_med_eucl', \
                          'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg', \
                        'distout_eucl_max', 'distout_eucl_min', 'distout_eucl_avg', \
                        'dist_glob_max_eucl', 'dist_glob_min_eucl', 'dist_glob_avg_eucl']]
    df_ComE_distmed = df_ComE[['node_id', 'distin_med_eucl', 'distout_med_eucl']]
    df_ComE_clr_lst = df_ComE_clr.values.tolist()
    df_ComE_out_lst = df_ComE_out.values.tolist()
    df_ComE_gbl_lst = df_ComE_gbl.values.tolist()
    df_ComE_clrout_lst = df_ComE_clrout.values.tolist()
    df_ComE_clrgbl_lst = df_ComE_clrgbl.values.tolist()
    df_ComE_all_lst = df_ComE_all.values.tolist()
    df_ComE_distmed_lst = df_ComE_distmed.values.tolist()
    id_ComE_feats_clr.append(df_ComE_clr_lst)
    id_ComE_feats_out.append(df_ComE_out_lst)
    id_ComE_feats_gbl.append(df_ComE_gbl_lst)
    id_ComE_feats_clrout.append(df_ComE_clrout_lst)
    id_ComE_feats_clrgbl.append(df_ComE_clrgbl_lst)
    id_ComE_feats_all.append(df_ComE_all_lst)
    id_ComE_distmed.append(df_ComE_distmed_lst)
#sort by node id
for i in id_ComE_feats_clr:
    i.sort()
for i in id_ComE_feats_out:
    i.sort()
for i in id_ComE_feats_gbl:
    i.sort()
for i in id_ComE_feats_clrout:
    i.sort()
for i in id_ComE_feats_clrgbl:
    i.sort()
for i in id_ComE_feats_all:
    i.sort()
for i in id_ComE_distmed:
    i.sort()

# Classic FEATURES

folder = os.listdir('classic_features_per_timestep/classic_features')
path = 'classic_features_per_timestep/classic_features'
id_classic_clr = []
id_classic_gbl = []
id_classic_all = []
id_classic_nodeg = []
id_classic_deg = []
for file in folder:
    df_classic = pd.read_csv(os.path.join(path,file), names=['node_id', \
    'degree', 'betweenness', 'closeness', 'eigenvector', \
    'degree_ntwk', 'betweenness_ntwk', 'closeness_ntwk', 'eigenvector_ntwk'], \
    skiprows=1)
    df_classic_clr = df_classic[['node_id', \
                    'degree', 'betweenness', 'closeness', 'eigenvector']]
    df_classic_gbl = df_classic[['node_id', \
    'degree_ntwk', 'betweenness_ntwk', 'closeness_ntwk', 'eigenvector_ntwk']]
    df_classic_nodeg = pd.read_csv(os.path.join(path,file), names=['node_id', \
    'betweenness', 'closeness', 'eigenvector', \
    'betweenness_ntwk', 'closeness_ntwk', 'eigenvector_ntwk'], \
    skiprows=1)
    df_classic_deg = df_classic[['node_id', 'degree', 'degree_ntwk']]
    df_classic_all_lst = df_classic.values.tolist()
    df_classic_clr_lst = df_classic_clr.values.tolist()
    id_classic_gbl_lst = df_classic_gbl.values.tolist()
    id_classic_nodeg_lst = df_classic_nodeg.values.tolist()
    id_classic_deg_lst = df_classic_deg.values.tolist()
    id_classic_all.append(df_classic_all_lst)
    id_classic_clr.append(df_classic_clr_lst)
    id_classic_gbl.append(id_classic_gbl_lst)
    id_classic_nodeg.append(id_classic_nodeg_lst)
    id_classic_deg.append(id_classic_deg_lst)
#sort by node id
for i in id_classic_all:
    i.sort()
for i in id_classic_clr:
    i.sort()
for i in id_classic_gbl:
    i.sort()
for i in id_classic_nodeg:
    i.sort()
for i in id_classic_deg:
    i.sort()

id_combo_ComE_clrout_classic_all = []
for ind,s in enumerate(id_ComE_feats_clrout):
    temp = []
    for inx,row in enumerate(s):
        tmp = row[:]
        tmp.extend(id_classic_all[ind][inx][1:])
        temp.append(tmp)
    id_combo_ComE_clrout_classic_all.append(temp)

#-------------------------------- MATCHING ------------------------------------

# [clr_x_tn, clr_y_tn+1, common_nodes_tn_tn+1]
matching = []
a = 0
while a<len(clustered_nodes)-1:
    matching_two = []
    for indcurr,clrcurr in enumerate(clustered_nodes[a]):
        tmp = []
        for indnxt,clrnxt in enumerate(clustered_nodes[a+1]):
            num_of_common = len(list(set(clrcurr)&set(clrnxt)))
            tmp.append([indcurr,indnxt,num_of_common])
        tmp_max = max(item[-1] for item in tmp)
        for t in tmp:
            if t[-1] == tmp_max:
                maxtmp = t
        matching_two.append(maxtmp)
    matching.append(matching_two)
    a += 1

#--------------------------------- CHAINS -------------------------------------
#************ SCD
# 2-chain
def twoChain_scd(features):
    two_chain_scd = []
    for ind,step in enumerate(matching[:-1]):
        per_step = []
        for inx,clr in enumerate(step):
            for nodeid in clustered_nodes[ind][clr[0]]:
                tmp = [nodeid]
                for idfeatures in features[ind]:
                    if nodeid == idfeatures[0]:
                        tmp.extend(idfeatures[1:])
                if nodeid in clustered_nodes[ind+1][clr[1]]:
                    tmp.append(0)#stay
                    for idfeatures in features[ind+1]:
                        if nodeid == idfeatures[0]:
                            tmp.extend(idfeatures[1:])
                    for cl in matching[ind+1]:
                        if nodeid in clustered_nodes[ind+1][cl[0]]:
                            if nodeid in clustered_nodes[ind+2][cl[1]]:
                                tmp.append(0)#stay
                                break
                            elif nodeid in node_ids[ind+2]:
                                tmp.append(1)#move
                                break
                            else:
                                tmp.append(2)#drop
                                break
                elif nodeid in node_ids[ind+1]:
                    tmp.append(1)#move
                    for idfeatures in features[ind+1]:
                        if nodeid == idfeatures[0]:
                            tmp.extend(idfeatures[1:])
                    for cl in matching[ind+1]:
                        if nodeid in clustered_nodes[ind+1][cl[0]]:
                            if nodeid in clustered_nodes[ind+2][cl[1]]:
                                tmp.append(0)#stay
                                break
                            elif nodeid in node_ids[ind+2]:
                                tmp.append(1)#move
                                break
                            else:
                                tmp.append(2)#drop
                                break
                else:
                    tmp.extend([-1]*(len(features[0][0][1:])+1))
                    tmp.append(2)#drop
                per_step.append(tmp)
        two_chain_scd.append(per_step)
    return(two_chain_scd)

def chains_scd(prev_chain_scd, features, a):
    curr_chain_scd = copy.deepcopy(prev_chain_scd[:-1])
    for ind,step in enumerate(curr_chain_scd):
        for row in step:
            if row[-1] == 0 or row[-1] == 1:
                for idfeatures in features[ind+2+a]:
                    if row[0] == idfeatures[0]:
                        row.extend(idfeatures[1:])
                for cl in matching[ind+2+a]:
                    if row[0] in clustered_nodes[ind+2+a][cl[0]]:
                        if row[0] in clustered_nodes[ind+3+a][cl[1]]:
                            row.append(0)#stay
                            break
                        elif row[0] in node_ids[ind+3+a]:
                            row.append(1)#move
                            break
                        else:
                            row.append(2)#drop
                            break
            else:
                row[-1:-1] = [-1]*(len(features[0][0][1:])+1)#add -1*(#feats + ev)
    return(curr_chain_scd)

# ----------------------------------------------------------------------------

#************ SL
def chains_sl(chainsSCD):
    chainsSL = copy.deepcopy(chainsSCD)
    for row in chainsSL:
        if row[-1] == 2:
            row[-1] = 1
    return(chainsSL)

# ----------------------------------------------------------------------------

#************ SC
def chains_sc(chainsSCD):
    chainsSC = []
    for row in chainsSCD:
        if row[-1] != 2:
            chainsSC.append(row)
    return(chainsSC)

# ----------------------------------------------------------------------------

def per_chain_all_chains_scd(feats):
    two_chain_scd = twoChain_scd(feats)
    three_chain_scd = chains_scd(two_chain_scd, feats, 0)
    four_chain_scd = chains_scd(three_chain_scd, feats, 1)
    two_chain_scd = [row for s in two_chain_scd for row in s]#flat
    three_chain_scd = [row for s in three_chain_scd for row in s]#flat
    four_chain_scd = [row for s in four_chain_scd for row in s]#flat
    # merge chains
    all_chains_scd = []
    all_chains_scd.append(two_chain_scd)
    all_chains_scd.append(three_chain_scd)
    all_chains_scd.append(four_chain_scd)
    all_chains_scd = [row for chain in all_chains_scd for row in chain]
    return(two_chain_scd, three_chain_scd, four_chain_scd, all_chains_scd)

# ----------------------------------------------------------------------------

# CHAINS ----------------------
# ComE
# clrout
two_chain_ComE_clrout_scd, three_chain_ComE_clrout_scd, four_chain_ComE_clrout_scd, \
    chains_ComE_clrout_scd = per_chain_all_chains_scd(id_ComE_feats_clrout)
# per chain
two_chain_ComE_clrout_sl = chains_sl(two_chain_ComE_clrout_scd)
two_chain_ComE_clrout_sc = chains_sc(two_chain_ComE_clrout_scd)
three_chain_ComE_clrout_sl = chains_sl(three_chain_ComE_clrout_scd)
three_chain_ComE_clrout_sc = chains_sc(three_chain_ComE_clrout_scd)
four_chain_ComE_clrout_sl = chains_sl(four_chain_ComE_clrout_scd)
four_chain_ComE_clrout_sc = chains_sc(four_chain_ComE_clrout_scd)
# SL
chains_ComE_clrout_sl = chains_sl(chains_ComE_clrout_scd)
# SC
chains_ComE_clrout_sc = chains_sc(chains_ComE_clrout_scd)

# clr
two_chain_ComE_clr_scd, three_chain_ComE_clr_scd, four_chain_ComE_clr_scd, \
    chains_ComE_clr_scd = per_chain_all_chains_scd(id_ComE_feats_clr)
# per chain
two_chain_ComE_clr_sl = chains_sl(two_chain_ComE_clr_scd)
two_chain_ComE_clr_sc = chains_sc(two_chain_ComE_clr_scd)
three_chain_ComE_clr_sl = chains_sl(three_chain_ComE_clr_scd)
three_chain_ComE_clr_sc = chains_sc(three_chain_ComE_clr_scd)
four_chain_ComE_clr_sl = chains_sl(four_chain_ComE_clr_scd)
four_chain_ComE_clr_sc = chains_sc(four_chain_ComE_clr_scd)
# SL
chains_ComE_clr_sl = chains_sl(chains_ComE_clr_scd)
# SC
chains_ComE_clr_sc = chains_sc(chains_ComE_clr_scd)

# out
two_chain_ComE_out_scd, three_chain_ComE_out_scd, four_chain_ComE_out_scd, \
    chains_ComE_out_scd = per_chain_all_chains_scd(id_ComE_feats_out)
# per chain
two_chain_ComE_out_sl = chains_sl(two_chain_ComE_out_scd)
two_chain_ComE_out_sc = chains_sc(two_chain_ComE_out_scd)
three_chain_ComE_out_sl = chains_sl(three_chain_ComE_out_scd)
three_chain_ComE_out_sc = chains_sc(three_chain_ComE_out_scd)
four_chain_ComE_out_sl = chains_sl(four_chain_ComE_out_scd)
four_chain_ComE_out_sc = chains_sc(four_chain_ComE_out_scd)
# SL
chains_ComE_out_sl = chains_sl(chains_ComE_out_scd)
# SC
chains_ComE_out_sc = chains_sc(chains_ComE_out_scd)

# ----------------------------------------------------------------------------

# Classic
#clr
# SCD
two_chain_classic_clr_scd, three_chain_classic_clr_scd, four_chain_classic_clr_scd, \
    chains_classic_clr_scd = per_chain_all_chains_scd(id_classic_clr)
# per chain
two_chain_classic_clr_sl = chains_sl(two_chain_classic_clr_scd)
two_chain_classic_clr_sc = chains_sc(two_chain_classic_clr_scd)
three_chain_classic_clr_sl = chains_sl(three_chain_classic_clr_scd)
three_chain_classic_clr_sc = chains_sc(three_chain_classic_clr_scd)
four_chain_classic_clr_sl = chains_sl(four_chain_classic_clr_scd)
four_chain_classic_clr_sc = chains_sc(four_chain_classic_clr_scd)
# SL
chains_classic_clr_sl = chains_sl(chains_classic_clr_scd)
# SC
chains_classic_clr_sc = chains_sc(chains_classic_clr_scd)

#gbl
# SCD
two_chain_classic_gbl_scd, three_chain_classic_gbl_scd, four_chain_classic_gbl_scd, \
    chains_classic_gbl_scd = per_chain_all_chains_scd(id_classic_gbl)
# per chain
two_chain_classic_gbl_sl = chains_sl(two_chain_classic_gbl_scd)
two_chain_classic_gbl_sc = chains_sc(two_chain_classic_gbl_scd)
three_chain_classic_gbl_sl = chains_sl(three_chain_classic_gbl_scd)
three_chain_classic_gbl_sc = chains_sc(three_chain_classic_gbl_scd)
four_chain_classic_gbl_sl = chains_sl(four_chain_classic_gbl_scd)
four_chain_classic_gbl_sc = chains_sc(four_chain_classic_gbl_scd)
# SL
chains_classic_gbl_sl = chains_sl(chains_classic_gbl_scd)
# SC
chains_classic_gbl_sc = chains_sc(chains_classic_gbl_scd)

#all
# SCD
two_chain_classic_all_scd, three_chain_classic_all_scd, four_chain_classic_all_scd, \
    chains_classic_all_scd = per_chain_all_chains_scd(id_classic_all)
# per chain
two_chain_classic_all_sl = chains_sl(two_chain_classic_all_scd)
two_chain_classic_all_sc = chains_sc(two_chain_classic_all_scd)
three_chain_classic_all_sl = chains_sl(three_chain_classic_all_scd)
three_chain_classic_all_sc = chains_sc(three_chain_classic_all_scd)
four_chain_classic_all_sl = chains_sl(four_chain_classic_all_scd)
four_chain_classic_all_sc = chains_sc(four_chain_classic_all_scd)
# SL
chains_classic_all_sl = chains_sl(chains_classic_all_scd)
# SC
chains_classic_all_sc = chains_sc(chains_classic_all_scd)


# CV - StratifiedKfold
def classify_sc(chain):
    chain = shuffle(np.array(chain))
    X = [i[1:-1] for i in chain.tolist()]
    Y = [i[-1] for i in chain.tolist()]
    X = np.array(X)
    Y = np.array(Y)
    print('Y_dataset:', Counter(Y))
    skf = StratifiedKFold(n_splits=5)
    clf = RandomForestClassifier(n_estimators=200)
    scores_zeros = []
    scores_ones = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        #scale X
        X_flat = [i for row in X_train for i in row if i!=-1 and i!=-2]
        X_mean = np.mean(X_flat)
        X_std = np.std(X_flat)
        for lst in X_train:
            for ind,i in enumerate(lst):
                if i!=-1 and i!=-2:
                    lst[ind] = (i - X_mean)/X_std
        for lst in X_test:
            for ind,i in enumerate(lst):
                if i!=-1 and i!=-2:
                    lst[ind] = (i - X_mean)/X_std
        sm = SMOTE(sampling_strategy='auto', k_neighbors=3)
        X_rs,y_rs = sm.fit_resample(X_train, y_train)
        clf.fit(X_rs, y_rs)
        clf_pred = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, clf_pred)
        classif_report = classification_report(y_test, clf_pred)
        #metrics for zeros
        prec_zeros = precision_score(y_test, clf_pred, pos_label=0)
        rec_zeros = recall_score(y_test, clf_pred, pos_label=0)
        f1_zeros = f1_score(y_test, clf_pred, pos_label=0)
        #all metrics for zeros together
        tmp = [prec_zeros,rec_zeros,f1_zeros]
        scores_zeros.append(tmp)
        #metrics for ones
        prec_ones = precision_score(y_test, clf_pred, pos_label=1)
        rec_ones = recall_score(y_test, clf_pred, pos_label=1)
        f1_ones = f1_score(y_test, clf_pred, pos_label=1)
        #all metrics for ones together
        temp = [prec_ones,rec_ones,f1_ones]
        scores_ones.append(temp)
        print(conf_matrix, '\n', classif_report)
    scores_zeros = np.array(scores_zeros)
    scores_ones = np.array(scores_ones)
    scores_zeros_means = scores_zeros.mean(axis=0)
    scores_ones_means = scores_ones.mean(axis=0)
    print('prec,rec,f1 0s mean', scores_zeros_means, '\n', \
          'prec,rec,f1 1s mean', scores_ones_means)
    return(scores_zeros_means, scores_ones_means)

scores_zeros_sc, scores_ones_sc = classify_sc(two_chain_classic_clr_sc)

# stratified kfold 

def Classification(chain):
    chain = shuffle(np.array(chain))
    X = [i[1:-1] for i in chain.tolist()]
    Y = [i[-1] for i in chain.tolist()]
    # padding
    longest = len(max(X,key=len))
    for row in X:
        while len(row)<longest:
            row.append(-99)
    #######
    X = np.array(X)
    Y = np.array(Y)
    print('Y_dataset:', Counter(Y))
    skf = StratifiedKFold(n_splits=5)
    clf = RandomForestClassifier()
    precisions = []
    recalls = []
    f1s = []
    f1s_macro = []
    f1s_micro = []
    accuracies = []
    fold = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_train_cnt = pd.DataFrame([Counter(y_train)]).transpose()
        y_train_cnt.sort_index(inplace=True)
        print('y_train:', y_train_cnt)
        clf.fit(X_train, y_train)
        clf_pred = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, clf_pred)
        classif_report = classification_report(y_test, clf_pred, output_dict=True)
        #accuracy
        accuracy = accuracy_score(y_test, clf_pred)
        #precision per label
        prec = precision_score(y_test, clf_pred, average=None)
        #recall per label
        rec = recall_score(y_test, clf_pred, average=None)
        #f1 per label
        f1 = f1_score(y_test, clf_pred, average=None)
        #f1 macro
        f1_macro = f1_score(y_test, clf_pred, average='macro')
        #f1 micro
        f1_micro = f1_score(y_test, clf_pred, average='micro')
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        f1s_macro.append(f1_macro)
        f1s_micro.append(f1_micro)
        accuracies.append(accuracy)
        fold += 1
        cr = pd.DataFrame(classif_report).transpose()
        cm = pd.DataFrame(conf_matrix)
        with open('results_details.csv','a') as f:
            f.write('Fold: '+str(fold)+'\n')
            f.write('f1 macro: '+str(f1_macro)+'\n')
            f.write('f1 micro: '+str(f1_micro)+'\n')
        with open('results_details.csv','a') as f:
            f.write('Train set labels distribution'+'\n')
        y_train_cnt.to_csv('results_details.csv', mode='a', header=None)
        cr.to_csv('results_details.csv', mode='a')
        with open('results_details.csv','a') as f:
            f.write('Confusion Matrix'+'\n')
        cm.to_csv('results_details.csv', mode='a')
        with open('results_details.csv', 'a') as f:
            f.write('\n')
            f.close()
    precisions_mean = np.array(precisions).mean(axis=0)
    precisions_mean = pd.DataFrame(precisions_mean)
    recalls_mean = np.array(recalls).mean(axis=0)
    recalls_mean = pd.DataFrame(recalls_mean)
    f1s_mean = np.array(f1s).mean(axis=0)
    f1s_mean = pd.DataFrame(f1s_mean)
    f1_macro_mean = pd.DataFrame([np.array(f1s_macro).mean()])
    f1_macro_mean.columns = ['f1 macro']
    f1_macro_mean.transpose().to_csv('results.csv', mode='a', header=False)
    f1_micro_mean = pd.DataFrame([np.array(f1s_micro).mean()])
    f1_micro_mean.columns = ['f1 micro']
    f1_micro_mean.transpose().to_csv('results.csv', mode='a', header=False)
    accuracies_mean = pd.DataFrame([np.array(accuracies).mean()])
    accuracies_mean.columns = ['accuracy']
    accuracies_mean.transpose().to_csv('results.csv', mode='a', header=False)
    metrics = pd.concat([precisions_mean, recalls_mean, f1s_mean], axis=1)
    metrics.columns = ['precision','recall', 'f1']
    metrics.to_csv('results.csv', mode='a')
    with open('results.csv','a') as fd:
        fd.write('\n')
    print('prec 0-1-2 mean', precisions_mean, '\n', \
          'rec 0-1-2 mean', recalls_mean, '\n', \
              'f1 0-1-2 mean', f1s_mean)

# default: 2 chain
# ComE features

# clrout
# 2 chain
with open('results_details.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_sc'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_sc'+'\n')
Classification(two_chain_ComE_clrout_sc)

with open('results_details.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_sl'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_sl'+'\n')
Classification(two_chain_ComE_clrout_sl)

with open('results_details.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_scd'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_ComE_clrout_scd'+'\n')
Classification(two_chain_ComE_clrout_scd)


# Classic features

# all
# 2 chain
with open('results_details.csv','a') as fd:
    fd.write('two_chain_classic_all_sc'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_classic_all_sc'+'\n')
Classification(two_chain_classic_all_sc)

with open('results_details.csv','a') as fd:
    fd.write('two_chain_classic_all_sl'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_classic_all_sl'+'\n')
Classification(two_chain_classic_all_sl)

with open('results_details.csv','a') as fd:
    fd.write('two_chain_classic_all_scd'+'\n')
with open('results.csv','a') as fd:
    fd.write('two_chain_classic_all_scd'+'\n')
Classification(two_chain_classic_all_scd)