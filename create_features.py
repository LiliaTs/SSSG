import numpy as np
import os
import csv
import networkx as nx
from scipy.spatial import distance
import copy
import pandas as pd
import community as community_louvain
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans



#---------------------------- READ FILES --------------------------------------

# Read edgelists per timestep

folder = os.listdir('dblp/edgelists_per_timestep')
path = 'dblp/edgelists_per_timestep'
edgelists = []
for file in folder:
    edgelists.append(np.genfromtxt(os.path.join(path, file), dtype=None).tolist())

#****************************** louvain to find number of clusters
embs = []
for i in ComE_id_embs:
    x = [list(line[1:]) for line in i]
    embs.append(x)
nmis = []
for i in range(5):
    G = nx.Graph()
    G.add_edges_from(edgelists[i])
    partition = community_louvain.best_partition(G)
    num_cls_edglst = []
    for v in partition.values():
        num_cls_edglst.append(v)
    value, count = np.unique(num_cls_edglst, return_counts=True)
    if i==0 or i==3:
        kmeans = KMeans(n_clusters=17, random_state=0)
        pred_y = kmeans.fit_predict(embs[i])
        nmi = normalized_mutual_info_score(num_cls_edglst,pred_y)
        nmis.append(nmi)
    else:
        kmeans = KMeans(n_clusters=16, random_state=0)
        pred_y = kmeans.fit_predict(embs[i])
        nmi = normalized_mutual_info_score(num_cls_edglst,pred_y)
        nmis.append(nmi)
    print(sorted(counts))
    print(len(values))
#*****************************************************************

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

# Read ComE means per timestep

folder = os.listdir('ComE_per_timestep/means')
path = 'ComE_per_timestep/means'
ComE_means = []
for file in folder:
    ComE_means.append(np.genfromtxt(os.path.join(path, file), dtype=None).tolist())

#------------------------------------------------------------------------------

# Node ids per timestep

node_ids = []
for step in ComE_id_embs:
    tmp = [id_emb[0] for id_emb in step]
    node_ids.append(tmp)

# [Node_id: clr_id] per timestep

id_clr = []
for i in range(len(node_ids)):
    tmp = {}
    for ind,node in enumerate(node_ids[i]):
        tmp[node] = ComE_lbls[i][ind]
    id_clr.append(tmp)

# Clustered nodes per timestep

clustered_nodes = []
for ind,i in enumerate(id_clr):
    clrids_uniq = set(i.values())
    d = {}
    for clrid in clrids_uniq:
        d[clrid] = [k for k in i.keys() if i[k] == clrid]
    clustered_nodes.append(d)

# Edges per cluster

clr_edges = []
for ind,step in enumerate(clustered_nodes):
    per_step = []
    for key,val in sorted(step.items()):
        per_clr = []
        for edge in edgelists[ind]:
            if edge[0] in val:
                if edge[1] in val:
                    per_clr.append(edge)
        per_step.append(per_clr)
    clr_edges.append(per_step)

#save list with cluster edges to file

for ind,i in enumerate(clr_edges,start=15):
    with open('cluster_edges_per_timestep/clr_edges_'+str(ind)+'.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerows(i)

# clustered_id_embs

clrd_embs = []
for ind,step in enumerate(clustered_nodes):
    per_step = []
    for key,val in sorted(step.items()):
        per_clr = []
        for node_id in val:
            for tpl in ComE_id_embs[ind]:
                if node_id == tpl[0]:
                    emb = list(tpl[1:])
            per_clr.append(emb)
        per_step.append(per_clr)
    clrd_embs.append(per_step)

# ****************************************************************************
# ******************************** FEATURES **********************************

# ***************************** ComE FEATURES ********************************

# ------------------- DIST from MED in and out of cluster --------------------
start = time.time()
dst_med = []
for ind,i in enumerate(clrd_embs[:-1]):
    per_step = []
    for indj,j in enumerate(i):
        rest_clrs_med = [med for indmed,med in enumerate(ComE_means[ind]) if indmed != indj]
        per_clr = []
        for emb in j:
            per_node = []
            tmp_eucl = distance.euclidean(emb,ComE_means[ind][indj])
            tmp_cos = distance.cosine(emb,ComE_means[ind][indj])
            tmp_l1 = np.linalg.norm(emb - np.array(ComE_means[ind][indj]), ord=1)
            per_node.append(tmp_eucl)
            per_node.append(tmp_cos)
            per_node.append(tmp_l1)
            tmp_out_eucl = []
            tmp_out_cos = []
            tmp_out_l1 = []
            for m in rest_clrs_med:
                tmp_out_eucl.append(distance.euclidean(emb,m))
                tmp_out_cos.append(distance.cosine(emb,m))
                tmp_out_l1.append(np.linalg.norm(np.array(emb) - np.array(m), ord=1))
            per_node.append(min(tmp_out_eucl))
            per_node.append(min(tmp_out_cos))
            per_node.append(min(tmp_out_l1))
            per_clr.append(per_node)
        per_step.append(per_clr)
    dst_med.append(per_step)

# ---------------- CLUSTER-TO-CLUSTER DISTANCES | FEATURES -------------------

# eucl, cos, L1
distances_in_eucl = []
distances_in_cos = []
distances_in_l1 = []
for i in clrd_embs[:-1]:
    tmp_eucl = []
    tmp_cos = []
    tmp_l1 = []
    for j in i:
        dist_eucl = distance.cdist(j, j)
        tmp_eucl.append(dist_eucl)
        dist_cos = distance.cdist(j, j, 'cosine')
        tmp_cos.append(dist_cos)
        dist_l1 = distance.cdist(j, j, 'cityblock')
        tmp_l1.append(dist_l1)
    distances_in_eucl.append(tmp_eucl)
    distances_in_cos.append(tmp_cos)
    distances_in_l1.append(tmp_l1)

# extract features from distance matrices
def dist_feats(dist):
    distances_in_feats = []
    for i in dist:
        per_step = []
        for clr in i:
            per_clr = []
            for edge in clr:
                rm_zero = edge[edge != 0]
                dist_max = max(rm_zero)
                dist_min = min(rm_zero)
                dist_avg = sum(rm_zero) / len(rm_zero)
                tmp = [dist_max]
                tmp.append(dist_min)
                tmp.append(dist_avg)
                per_clr.append(tmp)
            per_step.append(per_clr)
        distances_in_feats.append(per_step)
    return(distances_in_feats)

distances_in_eucl_feats = dist_feats(distances_in_eucl)
distances_in_cos_feats = dist_feats(distances_in_cos)
distances_in_l1_feats = dist_feats(distances_in_l1)

# ------------------- CLUSTER TO REST CLUSTERS DISTANCES ---------------------

# eucl, cos, L1

distances_out_eucl = []
distances_out_cos = []
distances_out_l1 = []
for i in clrd_embs[:-1]:
    all_clrs = pd.DataFrame(np.vstack(i))
    tmp_eucl = []
    tmp_cos = []
    tmp_l1 = []
    for j in i:
        j_arr = np.array(j)
        rest_clrs = all_clrs[~all_clrs.isin(j_arr).all(1)]
        rest_clrs_arr = np.array(rest_clrs)
        dist_eucl = distance.cdist(j_arr,rest_clrs_arr)
        dist_cos = distance.cdist(j_arr, rest_clrs_arr, 'cosine')
        dist_l1 = distance.cdist(j_arr, rest_clrs_arr, 'cityblock')
        tmp_eucl.append(dist_eucl)
        tmp_cos.append(dist_cos)
        tmp_l1.append(dist_l1)
    distances_out_eucl.append(tmp_eucl)
    distances_out_cos.append(tmp_cos)
    distances_out_l1.append(tmp_l1)

distances_out_eucl_feats = dist_feats(distances_out_eucl)
distances_out_cos_feats = dist_feats(distances_out_cos)
distances_out_l1_feats = dist_feats(distances_out_l1)

# ------------------ CLUSTER TO NETWORK DISTANCES - GLOBAL -------------------

# eucl, cos, L1

distances_out_glb_eucl = []
distances_out_glb_cos = []
distances_out_glb_l1 = []
for i in clrd_embs[:-1]:
    all_clrs = np.vstack(i)
    tmp_eucl = []
    tmp_cos = []
    tmp_l1 = []
    for j in i:
        j_arr = np.array(j)
        dist_eucl = distance.cdist(j_arr,all_clrs)
        dist_cos = distance.cdist(j_arr, all_clrs, 'cosine')
        dist_l1 = distance.cdist(j_arr, all_clrs, 'cityblock')
        tmp_eucl.append(dist_eucl)
        tmp_cos.append(dist_cos)
        tmp_l1.append(dist_l1)
    distances_out_glb_eucl.append(tmp_eucl)
    distances_out_glb_cos.append(tmp_cos)
    distances_out_glb_l1.append(tmp_l1)

distances_out_glb_eucl_feats = dist_feats(distances_out_glb_eucl)
distances_out_glb_cos_feats = dist_feats(distances_out_glb_cos)
distances_out_glb_l1_feats = dist_feats(distances_out_glb_l1)

end = time.time()
print(end - start)
# --------------------- add all ComE features together -----------------------

ComE_features = []
for ind,step in enumerate(clustered_nodes[:-1]):
    per_step = []
    for key,val in sorted(step.items()):
        per_clr = []
        for inx,nodeid in enumerate(val):
            per_node = [nodeid]
            per_node.extend(dst_med[ind][key][inx])
            per_node.extend(distances_in_eucl_feats[ind][key][inx])
            per_node.extend(distances_in_cos_feats[ind][key][inx])
            per_node.extend(distances_in_l1_feats[ind][key][inx])
            per_node.extend(distances_out_eucl_feats[ind][key][inx])
            per_node.extend(distances_out_cos_feats[ind][key][inx])
            per_node.extend(distances_out_l1_feats[ind][key][inx])
            per_node.extend(distances_out_glb_eucl_feats[ind][key][inx])
            per_node.extend(distances_out_glb_cos_feats[ind][key][inx])
            per_node.extend(distances_out_glb_l1_feats[ind][key][inx])
            per_step.append(per_node)
    ComE_features.append(per_step)
for i in ComE_features:
    i.sort()

feature_names = ['node_id', 'distin_med_eucl', 'distin_med_cos', 'distin_med_l1',\
         'distout_med_eucl', 'distout_med_cos', 'distout_med_l1',\
        'distin_eucl_max', 'distin_eucl_min', 'distin_eucl_avg',\
        'distin_cos_max', 'distin_cos_min', 'distin_cos_avg',\
        'distin_l1_max', 'distin_l1_min', 'distin_l1_avg',\
        'distout_eucl_max', 'distout_eucl_min', 'distout_eucl_avg',\
        'distout_cos_max', 'distout_cos_min', 'distout_cos_avg',\
        'distout_l1_max', 'distout_l1_min', 'distout_l1_avg', \
        'dist_glob_max_eucl', 'dist_glob_min_eucl', 'dist_glob_avg_eucl', \
        'dist_glob_max_cos', 'dist_glob_min_cos', 'dist_glob_avg_cos', \
        'dist_glob_max_l1', 'dist_glob_min_l1',  'dist_glob_avg_l1']

#save to file
for ind,i in enumerate(ComE_features):
    with open('ComE_features_per_timestep/ComE_features_'+str(ind)+'.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(feature_names)
        wr.writerows(i)


#**************************** CLASSIC FEATURES ********************************

# ---------------------------- CLUSTER FEATURES ------------------------------

start = time.time()
degrees = []
betwenness = []
closeness = []
eigenvector = []
for step in clr_edges[:-1]:
    print(clr_edges.index(step))
    per_step_deg = []
    per_step_btw = []
    per_step_cls = []
    per_step_eigen = []
    for clr in step:
        print(step.index(clr))
        G = nx.Graph()
        G.add_edges_from(clr)
        deg_clr = list(G.degree())
        btw_clr = list(sorted(nx.betweenness_centrality(G).items()))
        print('btw completed!')
        cls_clr = list(sorted(nx.closeness_centrality(G).items()))
        print('cls completed!')
        eigen_clr = list(sorted(nx.eigenvector_centrality(G).items()))
        per_step_deg.append(deg_clr)
        per_step_btw.append(btw_clr)
        per_step_cls.append(cls_clr)
        per_step_eigen.append(eigen_clr)
    degrees.append(per_step_deg)
    betwenness.append(per_step_btw)
    closeness.append(per_step_cls)
    eigenvector.append(per_step_eigen)
end = time.time()
print(end - start)

# add all classic cluster cc features together

classic_feats_clr_cc = []
for ind,step in enumerate(degrees):
    per_step = []
    for inx,clr in enumerate(step):
        for tpl in clr:
            tmp = list(tpl)
            for tplbtw in betwenness[ind][inx]:
                if tpl[0] == tplbtw[0]:
                    tmp.append(tplbtw[1])
            for tplcls in closeness[ind][inx]:
                if tpl[0] == tplcls[0]:
                    tmp.append(tplcls[1])
            for tpleig in eigenvector[ind][inx]:
                if tpl[0] == tpleig[0]:
                    tmp.append(tpleig[1])
            per_step.append(tmp)
    classic_feats_clr_cc.append(per_step)
# sort by node id
for i in classic_feats_clr_cc:
    i.sort()

# save to file

feature_names = ['node_id', 'degree', 'betweenness', 'closeness', 'eigenvector']

for ind,i in enumerate(classic_feats_clr_cc):
    with open('classic_features_per_timestep/classic_clr_cc_features/classic_clr_cc_features'+str(ind)+'.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(feature_names)
        wr.writerows(i)

# ---------------------------- NETWORK FEATURES ------------------------------

start_ntwk = time.time()
degrees_ntwk = []
betwenness_ntwk = []
closeness_ntwk = []
eigenvector_ntwk = []
for step in edgelists[:-1]:
    print(edgelists.index(step))
    G_ntwk = nx.Graph()
    G_ntwk.add_edges_from(step)
    deg_step_ntwk = list(G_ntwk.degree())
    btw_step_ntwk = list(sorted(nx.betweenness_centrality(G_ntwk).items()))
    print('btw_ntwk completed!')
    cls_step_ntwk = list(sorted(nx.closeness_centrality(G_ntwk).items()))
    print('cls_ntwk completed!')
    eigen_step_ntwk = list(sorted(nx.eigenvector_centrality(G_ntwk).items()))
    degrees_ntwk.append(deg_step_ntwk)
    betwenness_ntwk.append(btw_step_ntwk)
    closeness_ntwk.append(cls_step_ntwk)
    eigenvector_ntwk.append(eigen_step_ntwk)
end_ntwk = time.time()
print(end_ntwk - start_ntwk)

# add all classic network features together

classic_ntwk_features_cc = []
for ind,step in enumerate(degrees_ntwk):
    per_step = []
    for tpl in step:
        tmp = list(tpl)
        for tplbtw in betwenness_ntwk[ind]:
            if tpl[0] == tplbtw[0]:
                tmp.append(tplbtw[1])
        for tplcls in closeness_ntwk[ind]:
            if tpl[0] == tplcls[0]:
                tmp.append(tplcls[1])
        for tpleig in eigenvector_ntwk[ind]:
            if tpl[0] == tpleig[0]:
                tmp.append(tpleig[1])
        per_step.append(tmp)
    classic_ntwk_features_cc.append(per_step)
# sort by node_id
for i in classic_ntwk_features_cc:
    i.sort()
    
feature_names = ['node_id', 'degree_ntwk', 'betweenness_ntwk', 'closeness_ntwk', 'eigenvector_ntwk']

for ind,i in enumerate(classic_feats_clr_cc):
    with open('classic_features_per_timestep/classic_ntwk_features_cc/classic_ntwk_features_cc'+str(ind)+'.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(feature_names)
        wr.writerows(i)

# -----------------------------------------------------------------------------

# classic clr and network features together

classic_features = []
for ind,step in enumerate(classic_ntwk_features_cc):
    per_step = []
    for features_ntwk in step:
        tmp = features_ntwk[:]
        for features_clr in classic_feats_clr_cc[ind]:
            if features_ntwk[0] == features_clr[0]:
                tmp.extend(features_clr[1:])
        per_step.append(tmp)
    classic_features.append(per_step)
# add 0 to node that there are no classic clr features
for step in classic_features:
    for features in step:
        if len(features) == 5:
            features.extend([0]*4)
# change order per line: classic clr feats, classic ntwk feats
for step in classic_features:
    for idx,features in enumerate(step):
        step[idx] = features[0:1] + features[5:] + features[1:5]

feature_names = ['node_id', 'degree', 'betweenness', 'closeness', 'eigenvector', \
        'degree_ntwk', 'betweenness_ntwk', 'closeness_ntwk', 'eigenvector_ntwk']

for ind,i in enumerate(classic_features):
    with open('classic_features_per_timestep/classic_features/classic_features'+str(ind)+'.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(feature_names)
        wr.writerows(i)
