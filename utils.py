import numpy as np
import os
import pickle as pkl
import scipy.sparse as sp
import ssl
import sys
import torch
import urllib
import random
def to_undirected(edge_index):
    row, col = edge_index
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)
    return new_edge_index

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False