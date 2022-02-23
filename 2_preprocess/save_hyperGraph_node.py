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
import random
import time
import os

raw_data = './dataT_folder'

def getAll():
    data_folder = raw_data
    edge_folder = './edgeT_node_folder'

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

        netlist, adj_lists0 = mainParse(data_file)

        hyperGraph = []
        for i in range(len(adj_lists0)):
            if len(adj_lists0[i]) == 0:
                continue 

            tmp = np.array(list(adj_lists0[i].copy()))
            tmp = np.append(tmp, i)
            hyperGraph.append(tmp + 1)

        with open(edge_file, 'w') as fw:
            fw.write(str(len(hyperGraph)) + ' ' + str(len(adj_lists0)) + '\n')

            for edge in hyperGraph:
                for n in edge:
                    fw.write(str(n) + ' ')
                fw.write('\n')

getAll()

