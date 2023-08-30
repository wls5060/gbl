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
    return new_edge_index

def batch_train(model, feats, labels, loss_fcn, optimizer, train_loader, evaluator, dataset):
    model.train()
    device = labels.device
    total_loss = 0
    iter_num = 0
    y_true = []
    y_pred = []
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        output_att = model(batch_feats)
        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1))
        L1 = loss_fcn(output_att, labels[batch].long())
        loss_train = L1
        total_loss = loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_pred, dim=0), torch.cat(y_true))
    return loss, acc
