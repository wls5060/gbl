import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import torch
class BaseOperator:
    def __init__(self, hops):
        self.hops = hops
    
    def adj_normalized(self, adj):
        pass

class LaplacianOperator(BaseOperator):
    def __init__(self, hops, r = 0.5):
        super(LaplacianOperator, self).__init__(hops)
        self.r = r
        self.adj = None

    def adj_normalized(self, adj):
        adj = adj + sp.eye(adj.shape[0])
        
        d = np.array(adj.sum(1)).flatten()
        d_inv_left = np.power(d, self.r - 1)
        d_inv_left[np.isinf(d_inv_left)] = 0
        d_mat_inv_left = sp.diags(d_inv_left)
        
        d_inv_right=np.power(d, -self.r)
        d_inv_right[np.isinf(d_inv_right)] = 0
        d_mat_inv_right = sp.diags(d_inv_right)

        adj = d_mat_inv_left @ adj @ d_mat_inv_right

        return adj
    
    def propagation(self, adj, feature, name):
        self.adj = self.adj_normalized(adj)

        x = feature

        # saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
        torch.save(x, f'./dataset/dataset/{name}_0.pt')
        for i in range(self.hops):
            x = self.adj @ x
            # saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
            torch.save(x, f'./dataset/dataset/{name}_{i + 1}.pt')