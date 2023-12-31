import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GraphSAGE as pygGraphSAGE

# this class implements a GraphSAGE by wrapping PyG's implementation
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, encoder=None, return_logits=False):
        super(GraphSAGE, self).__init__()
        
        self.encoder = encoder

        # Create GraphSAGE
        self.graph_sage = pygGraphSAGE(
            in_channels=input_dim, 
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, adj_t):
        # input = torch.zeros_like(input)
        x = self.encoder.encode(input)

        out = self.graph_sage(x, adj_t)
        out = self.softmax(out).float()

        return out