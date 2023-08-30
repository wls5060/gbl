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

def batch_train(model, train_len, labels, loss_fcn, optimizer, train_loader, device):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    labels = labels.to(device)
    for batch in train_loader:
        train_output = model.forward(batch, device)
        loss_train = loss_fcn(train_output, labels[batch])

        y_pred = train_output.argmax(dim=-1)
        correct_num += y_pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / train_len
    return loss_train, acc_train

def batch_evaluate(model, val_len, val_loader, test_len, test_loader, labels, device):
    model.eval()
    correct_num_val = 0 
    correct_num_test = 0
    labels = labels.to(device)
    for batch in val_loader:
        val_output = model.forward(batch, device)
        y_pred = val_output.argmax(dim=-1)
        # pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += y_pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / val_len

    for batch in test_loader:
        test_output = model.forward(batch, device)
        y_pred = test_output.argmax(dim=-1)
        # pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += y_pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / test_len

    return acc_val.item(), acc_test.item()