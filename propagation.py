import torch
import scipy.sparse as sp
class BasePropagation:
    def __init__(self, result, adj):
        self.adj = adj
        self.result = result @ adj
'''
def sgc_precompute(features, adj, hops):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    print(precompute_time)
    return features, precompute_time
def sgc_precompute_1(features, adj, hops):
    t = perf_counter()
    for i in range(degree):
        features = adj @ features
    precompute_time = perf_counter()-t
    print(precompute_time)
    return features, precompute_time
'''