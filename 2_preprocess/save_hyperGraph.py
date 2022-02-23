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
from parse_net_new import mergeGraph
import networkx as nx
import random
import time
import os


raw_data = './dataT_folder'

def getAll():
    data_folder = raw_data 
    edge_folder = './edgeT_folder' 

    data_files = [(data_folder + "/" + fl) for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    data_files.sort()

    edge_files = [(edge_folder + "/" + fl[:-4] + '.hgr') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    edge_files.sort()

    if not os.path.exists(edge_folder):
        os.makedirs(edge_folder)

    random.seed(0)
    for di in range(len(data_files)):
        data_file = data_files[di]
        edge_file = edge_files[di]

        print ('data_file', data_file)
        print ('edge_file', edge_file)
        if (os.path.exists(edge_file)):
            print ('Already finished')
            continue

        #features, labels, remove_list, adj_lists0 = mainParse(data_file)
        netlist, adj_lists0 = mainParse(data_file)

        rev_list = reverseGraph(adj_lists0)
        adj_lists = mergeGraph(adj_lists0.copy(), rev_list)

        hyperGraph = []
        for i in range(len(rev_list)):
            if len(rev_list[i]) == 0:
                #print (i+1)
                continue 

            tmp = np.array(list(rev_list[i].copy()))
            tmp = np.append(tmp, i)
            hyperGraph.append(tmp + 1)

        with open(edge_file, 'w') as fw:
            fw.write(str(len(hyperGraph)) + ' ' + str(len(rev_list)) + '\n')

            for edge in hyperGraph:
                for n in edge:
                    fw.write(str(n) + ' ')
                fw.write('\n')
        print ()

designs = os.listdir(raw_data)
designs = [d.rsplit('.', 1)[0] for d in designs]
print (designs)

getAll()


