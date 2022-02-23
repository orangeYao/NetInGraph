import numpy as np
import sys
import os
import re
sys.path.append("/home/zx52/pathLen/common_ml")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=180)
np.core.arrayprint._line_width = 180
import random
import time
import os


raw_data = './dataT_folder'
def getAll(size_part):

    data_folder = raw_data
    edge_folder = './edgeT_folder'
    clus_folder = './clusT_folder' + '_' + str(size_part)

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

    hmetis = '/home/zx52/hmetis/hmetis-2.0pre1/Linux-x86_64/hmetis2.0pre1'
    UBfactor = '5'

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

        cmd = hmetis + '  ' + edge_file + '  ' + str(n_parts) + '  -ufactor=' + UBfactor + ' -nruns=1' # correct one
        os.system(cmd)
        print (cmd)

        with open (edge_file + '.part.' + str(n_parts)) as f:
            clusters = f.readlines()
        clusters = np.array([int(c.split()[0]) for c in clusters])

        np.save(clus_file, clusters)

        cmd = 'rm  ' + edge_file + '.part.' + str(n_parts)
        os.system(cmd)


import pickle

size_parts = [500, 1000, 2000]

for size_part in size_parts:
    print ('size_part!!', size_part)
    getAll(size_part)


