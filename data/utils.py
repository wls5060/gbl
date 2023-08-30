import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred import NodePropPredDataset, Evaluator
from ogbn import Ogbn
import torch.nn.functional as F
from utils import to_undirected
import gc



def prepare_label_emb(args, graph, labels, n_classes, train_node_nums, label_teacher_emb=None):
    if label_teacher_emb == None:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[:train_node_nums] = F.one_hot(labels[:train_node_nums].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.FloatTensor(y)
    else:
        print("use teacher label")
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_node_nums:] = label_teacher_emb[train_node_nums:]
        y[:train_node_nums] = F.one_hot(labels[:train_node_nums].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.FloatTensor(y)
    graph.row_norm()
    for hop in range(args.label_num_hops):
        y = torch.sparse.mm(graph, y)
    return y


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset(name, device, args, return_nid=False):
    """
    Load dataset and move graph and features to device
    """
    if name in ["ogbn-products","ogbn-papers100M", "ogbn-arxiv"]:
        graph = Ogbn(name=name, root=args.root, split="offical")
    else:
        raise RuntimeError("{} is not supported".format(name))
    train_nid = graph.train_idx
    val_nid =  graph.val_idx
    test_nid = graph.test_idx
    # assert np.max(train_nid.numpy()) <= np.min(val_nid.numpy())
    # assert np.max(val_nid) <= np.min(test_nid)
    train_node_nums = len(train_nid)
    valid_node_nums=len(val_nid)
    test_node_nums=len(test_nid)
    evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {graph.num_nodes}\n"
          f"# Edges: {graph.num_edges}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {graph.num_classes}\n")
    if not return_nid:
        return graph, train_node_nums, valid_node_nums, test_node_nums, evaluator
    train_nid = torch.LongTensor(train_nid)
    val_nid = torch.LongTensor(val_nid)
    test_nid = torch.LongTensor(test_nid)
    return graph, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    data = load_dataset(args.dataset, device, args)

    graph, train_node_nums, valid_node_nums, test_node_nums, evaluator = data
    if args.dataset == 'ogbn-papers100M':
        graph.edge_index = to_undirected(graph.edge_index)

    # move to device
    feats=[]
    for i in range(args.num_hops+1):
        if args.giant:
            print(f"load feat_{i}_giant.pt")
            feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}_giant.pt"))
        else:
            print(f"load feat_{i}.pt")
            feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}.pt"))
    in_feats=feats[0].shape[1]

    if args.dataset == 'ogbn-products':
        return graph, feats, graph.y, in_feats, graph.num_classes, \
               train_node_nums, valid_node_nums, test_node_nums, evaluator