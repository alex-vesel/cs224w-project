import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

# A GAT layer. This class defines the attention logic
class GATLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GATLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        self.lin_l = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = self.lin_l

        self.att_l = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_l.weight)
        torch.nn.init.xavier_uniform_(self.lin_r.weight)
        torch.nn.init.xavier_uniform_(self.att_l)
        torch.nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):

        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, size=size, x=x_r, alpha=(alpha_l, alpha_r))
        out = out.view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_j * alpha.view(-1, self.heads, 1)
        del alpha

        return x_j

    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

        return out

# This class defines a Graph Attention Network
# https://arxiv.org/abs/1710.10903
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads, dropout, encoder, emb=False):
        super(GAT, self).__init__()
        conv_model = self.build_conv_model()
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (num_layers >= 1), 'Number of layers is not >=1'
        for l in range(num_layers-1):
            self.convs.append(conv_model(heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(heads * hidden_dim, hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = dropout
        self.num_layers = num_layers

        self.encoder = encoder
        self.emb = emb

    def build_conv_model(self):
        return GATLayer

    def forward(self, x, edge_index):
        x = self.encoder.encode(x)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return softmax(x, dim=1)