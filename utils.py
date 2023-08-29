import numpy as np
import os
import pickle as pkl
import scipy.sparse as sp
import ssl
import sys
import torch
import urllib
def to_undirected(edge_index):
    row, col = edge_index
    new_row = torch.hstack((row, col))
    new_col = torch.hstack((col, row))
    new_edge_index = torch.stack((new_row, new_col), dim=0)

    return new_edge_index