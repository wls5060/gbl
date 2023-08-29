import gc
import torch
import dgl
import dgl.function as fn
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from basedata import BaseData, BaseDataSet
from utils import to_undirected
import numpy as np

class Ogbn(BaseDataSet):
    def __init__(self, name, root, split) :
        super(Ogbn, self).__init__(name, root)
        
    def process(self):
        dataset = PygNodePropPredDataset(self.name, self.root)
        data = dataset[0]
        print(data)

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        
        features, labels = data.x.numpy().astype(np.float32), data.y.to(torch.long).squeeze()
        print(features, '\n', labels)
        
        N = data.num_nodes
        print(N)

        row, col = to_undirected(data.edge_index)
        weight = torch.ones(len(row))
        
        self.g = BaseData(row, col, weight, N, features, labels)
        
test = Ogbn("ogbn-arxiv","dataset/dataset/","official")
print(test.g)