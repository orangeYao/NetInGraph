import numpy as np
import sys
import os
import re
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=180)
np.core.arrayprint._line_width = 180
from parse_net_new import mainParse
from parse_net_new import reverseGraph
import networkx as nx
import random
import time
import os
from scipy import stats


raw_data = './dataT_folder'
def getAll(size_part):

    data_folder = raw_data
    edge_folder = './edgeT_node_folder' 
    clus_folder = './clusT_Lnode_more' + '_' + str(size_part)

    data_files = [(data_folder + "/" + fl) for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    data_files.sort()

    edge_files = [(edge_folder + "/" + fl[:-4] + '.hgr') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    edge_files.sort()

    clus_files = [(clus_folder + "/" + fl[:-4] ) for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    clus_files.sort()

    if not os.path.exists(clus_folder):
        os.makedirs(clus_folder)

    random.seed(0)

    for di in range(len(data_files)):
        data_file = data_files[di]
        edge_file = edge_files[di]
        clus_file = clus_files[di]

        print ('clus_file', clus_file + '.npy')
        if os.path.exists(clus_file + '.npy'):
            print ('clus_file already exist, continue!!\n')
            continue

        print ('data_file', data_file)
        print ('edge_file', edge_file)

        with open (edge_file) as f:
            head = f.readline().split()
            edges, nodes = int(head[0]), int(head[1])

        n_parts = max(round(nodes / size_part), 2)

        with open (edge_file + '.part.' + str(n_parts)) as f:
            clusters = f.readlines()
        clusters = np.array([int(c.split()[0]) for c in clusters])

        cluster_nums, Lcluster_nums, diff_Lcluster_nums2 = [], [], []
        diff_Lcluster_nums3 = []
        diff_num2_unorm = []
        diff_Lcluster_nums2_i = []
        diff_Lcluster_nums2_a = []

        diff_Lcluster_contr = []
        diff_Lcluster_Lcontr = []

        netlist, adj_lists0 = mainParse(data_file)
        thres = 200
        for k, v in adj_lists0.items():
            if len(v) > thres:
                print ('large fanout reduced to 200', len(v))
                adj_lists0[k] = random.sample(v, thres)
        rev_list = reverseGraph(adj_lists0)

        for i in range(len(adj_lists0)):
            if len(adj_lists0[i]) == 0:
                cluster_nums.append(1)
                Lcluster_nums.append(1)
                diff_Lcluster_nums2.append(0)
                diff_Lcluster_nums3.append(0)

                diff_Lcluster_contr.append(0)
                diff_num2_unorm.append(0)

                diff_Lcluster_nums2_i.append(0)
                diff_Lcluster_nums2_a.append(0)

                diff_Lcluster_Lcontr.append(0)
                continue

            tmp = np.array(list(adj_lists0[i].copy())) 
            tmp = np.append(tmp, i)
            cluster_nums.append(len(set(clusters[tmp])) )

            groups, Lgroups = [], []

            if len(rev_list[i]) > 0:
                tmp = np.append(tmp, np.array(list(rev_list[i].copy())) )
                groups.append(list(rev_list[i]) + [i])
                
                L = []
                for ri in rev_list[i]:
                    ril  = list(adj_lists0[ri]).copy()
                    ril.remove(i)
                    #L.append (list(adj_lists0[ri]) + [ri])
                    L.append (ril + [ri, i])
                Lgroups.append(L)

            out_out = []
            for ai in adj_lists0[i]:
                L = []
                out_out += adj_lists0[ai]
                groups.append(list(adj_lists0[ai]) + [ai])
                L.append(list(adj_lists0[ai]) + [ai])

                for ri in rev_list[ai]:
                    if ri != i:
                        ril  = list(adj_lists0[ri]).copy()
                        ril.remove(ai)
                        #L.append (list(adj_lists0[ri]) + [ri])
                        L.append (ril + [ri, ai])
                Lgroups.append(L)


            if len(out_out) > 0:
                tmp = np.append(tmp, np.array(out_out) )
            Lcluster_nums.append(len(set(clusters[tmp])) )

            s2, s3, contr = groupPair(clusters, groups)

            diff_Lcluster_nums2.append(round(2* sum(s2) / max(len(groups)-1, 1), 3))
            diff_Lcluster_nums3.append(round(2* sum(s3) / max(len(groups)-1, 1), 3))
            diff_Lcluster_contr.append(sum(contr))
            if len(s2) == 0:
                diff_num2_unorm.append(0)
            else:
                diff_num2_unorm.append(round(2* max(s2), 3))


            s2i, s2a, contr = LgroupPair(clusters, Lgroups)

            diff_Lcluster_nums2_i.append(round(2* sum(s2i) / max(len(groups)-1, 1), 3))
            diff_Lcluster_nums2_a.append(round(2* sum(s2a) / max(len(groups)-1, 1), 3))
            diff_Lcluster_Lcontr.append(sum(contr))

            ##############################

        X_train = np.array([cluster_nums, Lcluster_nums, diff_Lcluster_nums2,
                            diff_Lcluster_nums3, diff_Lcluster_contr, 
                            diff_num2_unorm, 
                            diff_Lcluster_nums2_i, 
                            diff_Lcluster_nums2_a,
                            diff_Lcluster_Lcontr]).T

        print ('X_train', X_train.shape)
        print ()
        np.save(clus_file, X_train)

def groupPair(clusters, groups):
    s2, s3, contr = [], [], []
    l = len(groups)
    for i in range(l):
        li = clusters[groups[i]]
        ci = set(li)
        contr.append(1 / len(ci))

        for j in range(i+1, l):
            lj = clusters[groups[j]]
            cj = set(lj)

            inter = ci.intersection(cj)
            pi = len([i for i in li if i not in inter]) / len(li)
            pj = len([i for i in lj if i not in inter]) / len(lj)
            s2.append((pi + pj) / 2)
            s3.append( 1 - max((1-pi), (1-pj)) )
    return s2, s3, contr


def LgroupPair(clusters, groups):
    s2i = []
    s2a = []
    contr = []
    l = len(groups)

    for i in range(l):
        contri = 0 
        for gi in groups[i]:
            li = clusters[gi]
            ci = set(li)
            contri += 1 / len(ci)

        contr.append(contri)
        for j in range(i+1, l):
            sl, sl2 = [], []
            for gi in groups[i]:
                for gj in groups[j]:
                    li = clusters[gi]
                    ci = set(li)
                    lj = clusters[gj]
                    cj = set(lj)
                    sl.append(min(len(ci-cj), len(cj-ci)) )

                    inter = ci.intersection(cj)
                    pi = len([i for i in li if i not in inter]) / len(li)
                    pj = len([i for i in lj if i not in inter]) / len(lj)
                    sl2.append((pi + pj) / 2)
            s2i.append(min(sl2))
            s2a.append(max(sl2))
    return s2i, s2a, contr


import pickle

size_parts = [100, 200, 300, 500, 1000, 2000, 3000]

for size_part in size_parts:
    print ('size_part!!', size_part)
    getAll(size_part)


