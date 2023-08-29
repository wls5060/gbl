import gc
import torch
import dgl
import dgl.function as fn
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from basedata import BaseData, BaseDataSet

class Ogbn(BaseDataSet):
    def __init__(self, name, root, split) :
        super(Ogbn, self).__init__(name, root)
        
    def process(self):
        dataset = PygNodePropPredDataset(self.name, self.root)
        data = dataset[0]
        print(data)
        # features, labels = data.x.numpy().astype(np.float32), data.y.to(torch.long).squeeze(1)
        # split_idx = dataset.get_idx_split()
        # features, labels 
test = Ogbn("ogbn-arxiv","dataset/dataset/","official")
