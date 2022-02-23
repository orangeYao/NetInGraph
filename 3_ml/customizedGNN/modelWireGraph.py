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

from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GATConv
import torch.nn.functional as F
from math import ceil
from cgOwn import CGConvOwn

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop
        self.in_channels = in_channels

        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = GATConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.convE = CGConvOwn(13, 44)
        self.bn4 = torch.nn.BatchNorm1d(13)

        self.conv6 = GCNConv(13, out_channels)
        self.bn6 = torch.nn.BatchNorm1d(out_channels)


    def bn(self, i, x):
        #batch_size = 1
        num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        return x

    def forward(self, x, adj, edgesAttr):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x.squeeze()

        ce = self.convE(x0[:, :13], adj, edgesAttr)
        xe = self.bn(4, F.relu(ce) )

        c1 = self.conv1(x0, adj)
        x1 = self.bn(1, F.relu(c1) )

        c2 = self.conv2(x1, adj)
        x2 = self.bn(2, F.relu(c2) )

        c3 = self.conv3(x2, adj)
        x3 = self.bn(3, F.relu(c3) )

        c6 = self.conv6(xe, adj)
        x6 = self.bn(6, F.relu(c6) )

        x = torch.cat([x1, x2, x3, xe, x6], dim=-1)
        return x

class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()

        self.check = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hid_size = 32
        self.gnn1_embed = GNN(input_size, hid_size, hid_size, add_loop=True, lin=False)

        self.gnn_all_embed = GNN(510, 2*2 *hid_size - hid_size, hid_size, lin=False)
        print ()
        self.final1 = torch.nn.Linear(748, hid_size*2)
        self.final2 = torch.nn.Linear(hid_size*2, 1)
        self.final00 = torch.nn.Linear(141, hid_size)
        self.final0 = torch.nn.Linear(hid_size, 1)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x, adj, edgesAttr):
        x = x.view(1, x.shape[0], x.shape[1])
        adj0 = adj

        x00 = x.squeeze()
        x = self.gnn1_embed(x, adj, edgesAttr)
        x0s = x
        x0 = x.squeeze()

        x = self.final00(x0)
        x = self.final0(x)
        l1, e1 = None, None

        x = F.relu(x).squeeze()
        return x, l1, e1 


def run_cora(feat_data, labels, adj_lists, edgesAttr, 
             feat_data_t, labels_t, adj_lists_t, edgesAttr_t, 
             design_id, netType, save_name='outs2/'):
    from trainGS import run_cora_helper
    run_cora_helper(feat_data, labels, adj_lists, edgesAttr, Net, 
                    feat_data_t, labels_t, adj_lists_t, edgesAttr_t, design_id,
                    optimizer_f=['Adam', 0.002, 0.9], save_name=save_name)


