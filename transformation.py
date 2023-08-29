import torch
import scipy.sparse as sp
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, nfeat, nclass):
        super(LogisticRegression, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
