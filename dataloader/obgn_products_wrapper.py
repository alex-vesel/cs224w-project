import os
import torch
import pandas as pd
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import DATA_PATH


class ObgnProductsWrapper():
    def __init__(self, text=False, subset_size=None):
        self.data_path = os.path.join(DATA_PATH, "ogbn_products")

        # load dataset
        dataset = PygNodePropPredDataset(name='ogbn-products', root=self.data_path, transform=T.ToSparseTensor())
        if text:
            # load text
            node_text = pd.read_csv(os.path.join(self.data_path, "raw", "node-feat-text.csv.gz"), header=None).values
            dataset.data.x = node_text

        # select a subset of the dataset for testing purposes
        if subset_size is not None:
            subset_indices = torch.arange(subset_size)
            dataset.data.x = dataset.data.x[subset_indices]
            dataset.data.y = dataset.data.y[subset_indices]
            mask = (dataset.data.edge_index[0] < subset_size) & (dataset.data.edge_index[1] < subset_size)
            # mask = torch.zeros_like(mask)
            dataset.data.edge_index = dataset.data.edge_index[:, mask]
            dataset.data.num_nodes = subset_size

        self.graph = dataset[0]

        if text:
            self.graph.x = [desc[0] for desc in self.graph.x]

        # get split
        self.split_idx = dataset.get_idx_split()
        if subset_size is not None:
            self.split_idx['train'] = torch.arange(subset_size)


    def to(self, device):
        self.graph = self.graph.to(device)
        for key in self.split_idx:
            self.split_idx[key] = self.split_idx[key].to(device)
    

if __name__ == "__main__":
    dataset = ObgnProductsWrapper(split='train', text=True)