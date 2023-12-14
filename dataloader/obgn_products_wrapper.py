import os
import torch
import pandas as pd
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import sys
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import DATA_PATH, ENCODER_NAME, TRAIN_SIZE, VAL_SIZE

# this class loads the ogbn-products graph data and merges ndoe features
# depending on the specified text encoder embeddings
class ObgnProductsWrapper():
    def __init__(self, text=False, subset_size=None):
        self.data_path = os.path.join(DATA_PATH)

        # load dataset
        dataset = PygNodePropPredDataset(name='ogbn-products', root=self.data_path, transform=T.ToSparseTensor())
        if text:
            # load text
            encoder_name = ENCODER_NAME.replace("/", "_")
            node_text = np.load(os.path.join(DATA_PATH, "ogbn_products", "raw", f"node-feat-{encoder_name}.npy"))
            dataset.data.x = torch.tensor(node_text)
            dataset.data.x = dataset.data.x.float()

        # select a subset of the dataset for testing purposes
        if subset_size is not None:
            subset_indices = torch.arange(subset_size)
            dataset.data.x = dataset.data.x[subset_indices, :]
            dataset.data.y = dataset.data.y[subset_indices]
            mask = (dataset.data.edge_index[0] < subset_size) & (dataset.data.edge_index[1] < subset_size)
            mask = torch.zeros_like(mask)
            dataset.data.edge_index = dataset.data.edge_index[:, mask]
            dataset.data.num_nodes = subset_size

        self.graph = dataset[0]

        # get split
        self.split_idx = dataset.get_idx_split()
        if subset_size is not None:
            train_size = int(TRAIN_SIZE * subset_size)
            val_size = int(VAL_SIZE * subset_size)
            test_size = int((1 - TRAIN_SIZE - VAL_SIZE) * subset_size)
            self.split_idx['train'] = torch.arange(train_size)
            self.split_idx['val'] = train_size + torch.arange(val_size)
            self.split_idx['test'] = train_size + val_size + torch.arange(test_size)

    # put graph and splits onto specified device
    def to(self, device):
        self.graph = self.graph.to(device)
        for key in self.split_idx:
            self.split_idx[key] = self.split_idx[key].to(device)
    

if __name__ == "__main__":
    dataset = ObgnProductsWrapper(split='train', text=True)