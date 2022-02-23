import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class CGConvOwn(MessagePassing):
    def __init__(self, channels, dim, aggr='add', bias=True, **kwargs):
        super(CGConvOwn, self).__init__(aggr=aggr, **kwargs)
        self.out_channels = channels
        self.dim = dim

        self.lin_f  = Linear(dim + channels*2, (dim + channels)*2, bias=bias)
        self.lin_f2 = Linear((dim + channels)*2, channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        #self.lin_f2.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        #print ('own edge_attr', edge_attr.shape)
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.relu(self.lin_f2(F.relu(self.lin_f(z))) )

    def update(self, aggr_out, x):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.out_channels, self.dim)

