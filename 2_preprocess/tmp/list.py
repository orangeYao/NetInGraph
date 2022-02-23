
import glob
import os

dirs = glob.glob('/home/zx52/Net2_TCAD/preprocess_l/clusT*')

print (dirs)

for d in dirs:
    dl = d.split('/')[-1]
    f = glob.glob(d + '/b14.npy')[0]

    print ('mkdir ' + dl)
    print ('cp ' + f + ' ' + dl)
    os.system('mkdir ' + dl)
    os.system('cp ' + f + ' ' + dl)
    print ()

