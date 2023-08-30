import torch
import scipy.sparse as sp
import torch.nn as nn
import transformation
import operators
class BaseModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(BaseModel, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.operator = None
        self.transformation = None
    def propagation(self, adj, x, dataset):
        raise NotImplementedError
    def forward(self, idx, device):
        raise NotImplementedError

class SGC(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, hops, r):
        super(SGC, self).__init__(in_channels, hidden_channels, out_channels, dropout)

        self.hops = hops
        self.r = r
        self.operator = operators.LaplacianOperator(hops, r)
        self.transformation = transformation.LogisticRegression(in_channels, out_channels)
    
    def propagation(self, adj, x, dataset):
        self.operator.propagation(adj, x, dataset)
        self.feats = torch.FloatTensor(torch.load(f'./dataset/dataset/{dataset}_{self.hops}.pt'))
    
    def forward(self, idx, device):
        feats = self.feats[idx].to(device)
        output = self.transformation(feats)
        return output
    