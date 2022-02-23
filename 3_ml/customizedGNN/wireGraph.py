import numpy as np
import sys
import os
import re
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=180)
np.core.arrayprint._line_width = 180
sys.path.append("../../2_preprocess/") # to load parse_net_new.py
from parse_net_new import mainParse
from parse_net_new import generateFeaturesLabels
from parse_net_new import reverseGraph
from parse_net_new import mergeGraph
from modelWireGraph import run_cora
import pickle


def loadData(raw_data, data_folder):
    designs = os.listdir(raw_data)
    designs = [d.rsplit('.', 1)[0] for d in designs]
    designs.sort()
    print (designs)

    features_l, net_labels_l, adj_lists_l = [], [], []

    f = open(data_folder + "data_Graph.pickle","rb")
    features_ll, net_labels_ll, adj_lists_ll, edgesAttr_ll = pickle.load(f)
    f.close()

    for i in range(len(features_ll)):
        features_ll[i] = features_ll[i][:, :-1]
    return features_ll, net_labels_ll, adj_lists_ll, edgesAttr_ll, designs


raw_data = '../../2_preprocess/dataT_folder/'
data_folder = '../../2_preprocess/output/'
features_ll, net_labels_ll, adj_lists_ll, edgesAttr_ll, designs = loadData(raw_data, data_folder)


incr = 1
print ('start training !!\n')
for j in range(0, len(designs)):
    features_ltra, net_labels_ltra, adj_lists_ltra = [], [], []
    edgesAttr_ltra = []
    features_ltest, net_labels_ltest, adj_lists_ltest = [], [], []
    edgesAttr_ltest = []

    for k in range(len(designs)):
        if k < j or k >= j+incr:
            features_ltra.append(features_ll[k])
            net_labels_ltra.append(net_labels_ll[k])
            adj_lists_ltra.append(adj_lists_ll[k])
            edgesAttr_ltra.append(edgesAttr_ll[k])
        else:
            features_ltest.append(features_ll[k])
            net_labels_ltest.append(net_labels_ll[k])
            adj_lists_ltest.append(adj_lists_ll[k])
            edgesAttr_ltest.append(edgesAttr_ll[k])

    design = designs[j: j+incr]

    print ('## Design')
    print ('design_name', design)
    print ('features_ltra', len(features_ltra), len(net_labels_ltra))
    print ('len(adj_lists_ltra)', len(adj_lists_ltra))
    print ('len(adj_lists_ltest)', len(adj_lists_ltest))
    print ('start training!!')
    run_cora(features_ltra, net_labels_ltra, adj_lists_ltra, edgesAttr_ltra, 
             features_ltest, net_labels_ltest, adj_lists_ltest, edgesAttr_ltest, design)
    print ()
    print ()

