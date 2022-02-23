from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import os
from torch.nn import init
from torch.autograd import Variable
import copy

import numpy as np
import random
from plot_accuracy import main_plot
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score
import pickle
import time

def getTensor(adj_lists, feat_data, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = labels.shape[0]
    feat_data = torch.FloatTensor(feat_data)
    labels = torch.FloatTensor(labels)
    edges = torch.LongTensor(adj_lists)
    return edges.to(device), feat_data.to(device), labels.to(device)


def run_cora_helper(feat_data, labels, adj_lists, Net, 
                    feat_data_t, labels_t, adj_lists_t, 
                    design_names, optimizer_f, outFile):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = feat_data[0].shape[1]
    print ('size', size )
    model = Net(size).to(device)


    if optimizer_f[0] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_f[1])
    elif optimizer_f[0] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_f[1], momentum=optimizer_f[2])
    else:
        print ('unknown optimizer!!')
        exit()

    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True)
    lossF = nn.MSELoss()
    best_acc = [0 for i in range(len(adj_lists_t))]

    random.seed(0)
    for epoch in range(250):

        print ('epoch', epoch)
        train_r, train_roc, pred_l = [], [], []
        lossAll = 0

        lists = list(range(len(adj_lists)))

        for i in range(len(adj_lists)):
            edges, feat_data_c, labels_c = getTensor(adj_lists[i], feat_data[i], labels[i] )
            pred, lloss, eloss = model(feat_data_c, edges)
            optimizer.zero_grad()

            loss = lossF(pred, labels_c)
            loss.backward()
            optimizer.step()

            print (round(float(loss.data.cpu().numpy()), 2))
            lossAll += float(loss.data.cpu().numpy())

            del edges
            del feat_data_c
            del labels_c

            #if epoch % 10 == 1 and epoch > 1:
            if epoch % 10 == 1:
                train_output = pred.data.cpu().numpy()
                y_train_90 = labels[i] > np.percentile(labels[i], 90) 
                y_train_95 = labels[i] > np.percentile(labels[i], 95)
                r = stats.pearsonr(labels[i], train_output)[0]
                train_r.append(r)
                train_roc.append(round(roc_auc_score(y_train_90, train_output)*100))
            del pred
        scheduler.step(lossAll)
        print ('lossAll', round(lossAll, 2))
        print ()


        #if epoch % 10 == 1 and epoch > 1:
        if epoch % 10 == 1:
            assert len(train_roc) > 0
            print (epoch, 'train_r roc', round(sum(train_r)/ len(train_r), 3), sum(train_roc)/ len(train_roc))

            print ('\n=train_eval=\n', epoch)
            evaluate(adj_lists_t, feat_data_t, labels_t, model, device, design_names, best_acc, outFile)
            print ('\n=eval=\n', epoch)
            model.eval()
            evaluate(adj_lists_t, feat_data_t, labels_t, model, device, design_names, best_acc, outFile)
            model.train()

        
def evaluate(adj_lists_t, feat_data_t, labels_t, model, device, design_names, best_acc, outFile):
    test_r, test_roc = [], []
    pred_l, label_l = [], []

    for i in range(len(adj_lists_t)):
        st = time.time()
        edges, feat_data_c, labels_c = getTensor(adj_lists_t[i], feat_data_t[i], 
                                                              labels_t[i] )

        pred, _, _ = model(feat_data_c, edges)

        lossF = nn.MSELoss()
        loss = lossF(pred, labels_c)
        print (loss.data.cpu().numpy())
        del edges
        del feat_data_c
        del labels_c

        test_output = pred.data.cpu().numpy()
        pred_l.append(test_output)
        label_l.append(labels_t[i])
        y_test_90 = labels_t[i] > np.percentile(labels_t[i], 90)
        y_test_95 = labels_t[i] > np.percentile(labels_t[i], 95)
        print ('roc_test', round(roc_auc_score(y_test_90, test_output)*100, 1))
        r = stats.pearsonr(labels_t[i], test_output)[0]
        test_r.append(r)
        test_roc.append(round(roc_auc_score(y_test_90, test_output)*100, 1))
        del pred
        et = time.time()
        print ('runtime of', design_names[i], et - st)

    assert len(test_roc) > 0

    if not os.path.exists(outFile):
        os.makedirs(outFile)

    for i in range(len(adj_lists_t)):
        if test_roc[i] >= best_acc[i]:
            best_acc[i] = test_roc[i]
            print ('best_acc', design_names[i], best_acc[i], pred_l[i].shape)
            with open (outFile + design_names[i] + '_' + str(best_acc[i]), 'wb') as f:
                pickle.dump(pred_l[i], f)
            with open (outFile + design_names[i], 'wb') as f:
                pickle.dump(pred_l[i], f)
            with open (outFile + design_names[i] + '_Label', 'wb') as f:
                pickle.dump(label_l[i], f)
    print ()

