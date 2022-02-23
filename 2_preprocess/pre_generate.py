import numpy as np
import sys
import os
import re
sys.path.append("/home/zx52/pathLen/common_ml")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=180)
np.core.arrayprint._line_width = 180
from parse_net_new import mainParse
from parse_net_new import generateFeaturesLabels
#from parse_net import reverseGraph
#from parse_net import mergeGraph
from parse_net_new import reverseGraph
from parse_net_new import mergeGraph
import random
from multiprocessing import Process

def getStd(n):
    if len(n) == 0:
        return 0
    else:
        return np.std(n)


def sumFeats(features):
    clus_n = 9
    sumFeat = []
    for start in range(12, 12 + clus_n):
        cols = list(range(start, 12 + 7*clus_n, clus_n))
        sumFeat.append(features[:, cols].sum(axis=1))
    sumFeat = np.array(sumFeat).T
    return np.concatenate([features, sumFeat], axis=1)

def getClusterFromEdges(edge_file, size_part):
    with open (edge_file) as f:
        head = f.readline().split()
        edges, nodes = int(head[0]), int(head[1])
    n_parts = max(round(nodes / size_part), 2)

    print ('debug:', edge_file + '.part.' + str(n_parts))
    with open (edge_file + '.part.' + str(n_parts)) as f:
        clusters = f.readlines()
    clusters = np.array([int(c.split()[0]) for c in clusters]) 
    return clusters


def getAll():
    raw_data = './dataT_folder/'
    data_folder = raw_data
    prev_Lnode = './edgeT_node_folder'
    clus_folder  = prev_Lnode 

    prev_folder = './clusT_folder'
    clus_noNode_folder  = prev_folder + '_' + str(500)
    clus_noNode_folder2 = prev_folder + '_' + str(1000)
    clus_noNode_folder3 = prev_folder + '_' + str(2000)


    data_files = [(data_folder + "/" + fl) for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    data_files.sort()

    clus_files = [(clus_folder + "/" + fl[:-4] + '.hgr') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    clus_files.sort()

    clus_noNode_files = [(clus_noNode_folder + "/" + fl[:-4] + '.npy') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    clus_noNode_files.sort()

    clus_noNode_files2 = [(clus_noNode_folder2 + "/" + fl[:-4] + '.npy') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    clus_noNode_files2.sort()

    clus_noNode_files3 = [(clus_noNode_folder3 + "/" + fl[:-4] + '.npy') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    clus_noNode_files3.sort()


    print ('data_files', data_files)

    features_l, net_labels_l, edges_l, edgesAttr_l = [], [], [], []
    random.seed(0)
    for si in range(len(data_files)):
        data_file = data_files[si]
        clus_file = clus_files[si]

        cluster  = getClusterFromEdges(clus_file, 100)
        cluster2 = getClusterFromEdges(clus_file, 200)
        cluster3 = getClusterFromEdges(clus_file, 300)
        cluster4 = getClusterFromEdges(clus_file, 500)

        cluster5 = getClusterFromEdges(clus_file, 1000)
        cluster6 = getClusterFromEdges(clus_file, 2000)
        cluster7 = getClusterFromEdges(clus_file, 3000)

        print ('cluster, cluster2', cluster.shape, cluster2.shape)

        cluster_noNode = np.load(clus_noNode_files[si])
        cluster_noNode2 = np.load(clus_noNode_files2[si])
        cluster_noNode3 = np.load(clus_noNode_files3[si])
        print ('cluster_noNode, cluster_noNode2', cluster_noNode.shape, cluster_noNode2.shape)

        cluster_all = [cluster, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7]
        cluster_all_noN = [cluster_noNode, cluster_noNode2, cluster_noNode3]

        print ('cluster_all.shape', len(cluster_all))

        print (data_file)
        design_net = data_file.rsplit('/', 1)[1].rsplit('_', 2)[0]

        nets_list, adj_lists0 = mainParse(data_file)
        features, labels = generateFeaturesLabels(nets_list, 'libInfo/area.csv', 'libInfo/fanin.csv')

        print ('check labels', round(np.median(labels), 3))
        if np.median(labels) < 0.0001:
            print ('#########This netlist is not placed!! Skipped!!')
            with open (skip_file, 'a') as f:
                f.write(data_file + '\n')
            print ()
            continue


        rev_list = reverseGraph(adj_lists0)
        adj_lists = mergeGraph(adj_lists0.copy(), rev_list)

        free_list = []
        in_port, out_port = [], []
        for key in range(len(adj_lists)):
            outs = adj_lists0[key]
            ins = rev_list[key]
            outs_outs = [len(adj_lists0[o]) for o in outs]
            outs_ins  = [len(rev_list[o])   for o in outs]
            ins_outs  = [len(adj_lists0[o]) for o in ins]
            ins_ins   = [len(rev_list[o])   for o in ins]
            free_list.append([int(len(outs) == 0), len(outs), np.sum(outs_outs), getStd(outs_outs),
                                                   np.sum(outs_ins),  getStd(outs_ins), 
                                                   np.sum(ins_outs),  getStd(ins_outs), 
                                                   np.sum(ins_ins),   getStd(ins_ins)])

                                     # 2       # 10                # 16 + 12 = 28
        print ('data_file', data_file, clus_file)

        fanoutsSum = []
        for i in range(len(adj_lists)):
            fanoutsSum.append( sum([features[ai][0] for ai in adj_lists0[i]]) + features[i][0])
        new_feat = np.array([fanoutsSum, len(adj_lists)//10000* np.ones(len(adj_lists))])
        features = np.concatenate([features, np.array(free_list), new_feat.T], axis=1)
        print ('features', features.shape)

        thres = 300
        for k, v in adj_lists0.items():
            if len(v) > thres:
                print ('large fanout reduced to 200', len(v))
                adj_lists0[k] = random.sample(v, thres)
    
        edges, edgesAttr = [], []
        for i in range(len(adj_lists0)):
            groups, nodes = [], []
            if len(rev_list[i]) > 0:
                for ri in rev_list[i]:
                    groups.append([ri] + list(adj_lists0[ri]))
                    nodes.append(ri)

            if len(adj_lists0[i]) > 0:
                for ai in adj_lists0[i]:
                    groups.append([ai] + list(adj_lists0[ai]))
                    nodes.append(ai)
            
            if len(nodes) <= 1:
                continue

            features_all = groupPair(cluster_all, nodes, groups) 
            features_allN = groupPairN(cluster_all_noN, nodes, i) 
            combined = np.concatenate([features_all, features_allN], axis=1)

            for ni, n in enumerate(nodes):
                # src to target
                edges.append([n, i])
                edgesAttr.append(combined[ni])

        edges = np.transpose(np.array(edges))
        edgesAttr = np.transpose(edgesAttr)
        print (edges.shape)
        print (edgesAttr.shape)

        print ('edges, edgesAttr shape', edges.shape, edgesAttr.shape)
        print (si, 'size:', features.shape, edges.shape)

        features_l.append(features)
        net_labels_l.append(labels)
        edges_l.append(edges)
        edgesAttr_l.append(edgesAttr)
        print ()
    return features_l, net_labels_l, edges_l, edgesAttr_l



def groupPairN(cluster_all_noN, nodes, ii):
    features_all = None
    for clusters in cluster_all_noN:
        lii = clusters[ii]

        features = [0]* len(nodes)
        s1, s2 = 0, 0
        for i in range(len(nodes)):
            li = clusters[nodes[i]]

            s1 += float(lii != li)
            for j in range(len(nodes)):
                if i != j:
                    lj = clusters[nodes[j]]
                    s2 += float(li != lj)

            features[i] = [s1, s2, s2/(len(nodes)-1)]

        features = np.array(features)
        if features_all is None:
            features_all  = features
        else:
            features_all = np.concatenate([features_all, features], axis=1)

    clus_n = 3
    sumTrain = []
    for start in range(clus_n):
        cols = list(range(start, 3*clus_n, clus_n))
        sumTrain.append(features_all[:, cols].sum(axis=1))
    sumTrain = np.array(sumTrain).T
    features_all = np.concatenate([features_all, sumTrain], axis=1)
    #print ('features_all', features_all.shape)
    features_all = np.array(features_all)
    return features_all



def groupPair(cluster_all, nodes, groups): 

    if len(nodes) != len(groups):
        sys.exit('Length mismatch!!')

    features_all = None
    for clusters in cluster_all:
        sd1, sd2 = dict(), dict()

        features = [0]* len(nodes)

        for i in range(len(nodes)):
            li = clusters[groups[i]]
            ci = set(li)
            ni = clusters[nodes[i]]

            for j in range(i+1, len(nodes)):
                lj = clusters[groups[j]]
                cj = set(lj)
                nj = clusters[nodes[j]]
                        
                inter = ci.intersection(cj)
                pi = len([i for i in li if i not in inter]) / len(li)
                pj = len([i for i in lj if i not in inter]) / len(lj)

                s1 = float(ni != nj)
                s2 = (pi + pj) / 2
                addDict(sd1, i, s1)
                addDict(sd1, j, s1)
                addDict(sd2, i, s2)
                addDict(sd2, j, s2)

        for i in range(len(nodes)):
            features[i] = [sum(sd1[i]), sum(sd2[i]), \
                           sum(sd1[i])/len(sd1[i]), sum(sd2[i])/ len(sd1[i])]

            if len(sd1[i]) + 1 != len(nodes):
                print (len(sd1[i]), len(nodes))
                sys.exit('length mismatch!!')

        features = np.array(features)
        if features_all is None:
            features_all  = features
        else:
            features_all = np.concatenate([features_all, features], axis=1)

    clus_n = 4 
    sumTrain = []
    for start in range(clus_n):  
        cols = list(range(start, 7*clus_n, clus_n))
        sumTrain.append(features_all[:, cols].sum(axis=1))
    sumTrain = np.array(sumTrain).T
    features_all = np.concatenate([features_all, sumTrain], axis=1)
    features_all = np.array(features_all)
    return features_all


def addDict(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def mainProc():
    features_l, net_labels_l, edges_l, edgesAttr_l = getAll()

    import pickle
    with open("output/data_Graph.pickle","wb") as f:
        pickle.dump([features_l, net_labels_l, edges_l, edgesAttr_l],f)


mainProc()

