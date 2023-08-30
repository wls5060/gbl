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
    
class SGC(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, hops, r):
        super(BaseModel, self).__init__(in_channels, hidden_channels, out_channels, dropout)

        self.operator = LaplacianOperator(hops, r)
test = BaseModel(1,1,1,1)