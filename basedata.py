import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

class sparse_Adj:
    def __init__(self, row, col, weight, N) :

        if(not isinstance(row, (list, np.ndarray, Tensor))) or (not isinstance(col, (list, np.ndarray, Tensor))) or (
                not isinstance(weight, (list, np.ndarray, Tensor))):
            raise TypeError("Row, col and edge_weight must be a list, np.ndarray or Tensor!")
        if(not isinstance(row, Tensor)) :
            row = torch.Tensor(row)
        if(not isinstance(col, Tensor)) :
            col = torch.Tensor(col)
        if(not isinstance(weight, Tensor)) :
            weight = torch.FloatTensor(weight)
        self.__row = row
        self.__col = col
        self.__weight = weight
        
        self.Adj = csr_matrix((weight, (row, col)), shape=(N, N))

class Node:
    def __init__(self, N, x = None, y = None) :
        if(x is not None):
            if(isinstance(x, np.ndarray)) :
                x = torch.FloatTensor(x)
            elif not isinstance(x, Tensor) :
                raise TypeError("x must be a np.ndarray or Tensor!")
        self.__x = x

        if(y is not None):
            if(isinstance(y, np.ndarray)) :
                y = torch.Tensor(x)
            elif not isinstance(y, Tensor) :
                raise TypeError("y must be a np.ndarray or Tensor!")
        self.__y = y

        self.__N = N

class BaseData:
    def __init__(self, row, col, weight, N, x = None, y = None):
        self.__Adj = sparse_Adj(row, col, weight, N)
        self.__Node = Node(N, x, y)
class BaseDataSet:
    def __init__(self, name, root):
        self.name = name
        self.root = root
        self.data = None
        self.train_idx, self.val_idx, self.test_idx = None, None, None
        self.process()
    def process() :
        pass
if __name__=="__main__":
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])

    test = sparse_Adj(row,col,data,3)

    print(test.Adj.toarray())