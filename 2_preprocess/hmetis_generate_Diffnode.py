import numpy as np
import sys
import os
import re
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=180)
np.core.arrayprint._line_width = 180
import random
import time
import os
from scipy import stats


raw_data = './dataT_folder'
def getAll(size_part):

    data_folder = raw_data
    edge_folder = './edgeT_node_folder'

    data_files = [(data_folder + "/" + fl) for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    data_files.sort()

    edge_files = [(edge_folder + "/" + fl[:-4] + '.hgr') for fl in os.listdir(data_folder)
                          if os.path.isfile(os.path.join(data_folder, fl))]
    edge_files.sort()

    random.seed(0)

    hmetis = '/home/zx52/hmetis/hmetis-2.0pre1/Linux-x86_64/hmetis2.0pre1'
    UBfactor = '5'
    print ('here')

    for di in range(len(data_files)):
        data_file = data_files[di]
        edge_file = edge_files[di]

        print ('data_file', data_file)
        print ('edge_file', edge_file)
        print ()

        with open (edge_file) as f:
            head = f.readline().split()
            edges, nodes = int(head[0]), int(head[1])

        n_parts = max(round(nodes / size_part), 2)

        if os.path.exists(edge_file + '.part.' + str(n_parts)):
            print ('edge_file.part already exists:', edge_file + '.part.' + str(n_parts))
            continue

        cmd = hmetis + '  ' + edge_file + '  ' + str(n_parts) + '  -ufactor=' + UBfactor + ' -nruns=1'
        os.system(cmd)
        print (cmd)


import pickle
size_parts = [100, 200, 300, 500, 1000, 2000, 3000]

for size_part in size_parts:
    print ('size_part!!', size_part)
    getAll(size_part)



