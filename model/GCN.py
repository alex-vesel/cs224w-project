import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# this class implements a Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, encoder=None, return_logits=False):
        super(GCN, self).__init__()

        # encoder to generate embeddings from raw text
        self.encoder = encoder

        # A list of GCNConv layers
        convlist = [GCNConv(input_dim, hidden_dim)]
        convlist += [GCNConv(hidden_dim, hidden_dim) for i in range(num_layers - 2)]
        convlist += [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convlist)

        # A list of 1D batch normalization layers
        bnslist = [torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers - 1)]
        self.bns = torch.nn.ModuleList(bnslist)

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax(dim=1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_logits = return_logits

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, input, adj_t):
        # input = torch.zeros_like(input)
        x = self.encoder.encode(input)

        for i in range(len(self.convs) - 1):
          x = self.convs[i](x, adj_t)
          x = self.bns[i](x)
          x = F.relu(x)
          x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        if self.return_logits:
          return x
        out = self.softmax(x).float()

        return out