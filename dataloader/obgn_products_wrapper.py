import os
import torch
import pandas as pd
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import DATA_PATH


class ObgnProductsWrapper():
    def __init__(self, split, text=False):
        self.data_path = os.path.join(DATA_PATH, "ogbn_products")

        # load dataset
        import IPython; IPython.embed(); exit(0)
        dataset = PygNodePropPredDataset(name='ogbn-products', root=self.data_path, transform=T.ToSparseTensor())
        if text:
            # load text
            node_text = pd.read_csv(os.path.join(self.data_path, "raw", "node-feat-text.csv.gz"), header=None).values
            dataset.data.x = node_text
        self.graph = dataset[0]

        # get split
        split_idx = dataset.get_idx_split()
        self.split_idx = split_idx[split]


    def to(self, device):
        self.graph = self.graph.to(device)
        self.split_idx = self.split_idx.to(device)
    

if __name__ == "__main__":
    dataset = ObgnProductsWrapper(split='train', text=True)