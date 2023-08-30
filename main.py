import argparse
import propagation
import models
from ogbn import Ogbn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import gc
import time
import utils

def main(args) :
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.model == "sgc":

        data = Ogbn(args.dataset, args.root, "official")
        if(args.batch_size != 0) :
            train_loader = DataLoader(data.train_idx, batch_size = args.batch_size, shuffle = True)
            val_loader = DataLoader(data.val_idx, batch_size = args.eval_batch_size, shuffle = False)
            test_loader = DataLoader(data.test_idx, batch_size = args.eval_batch_size, shuffle = False)
        model = models.SGC(1, args.hidden, 1, args.dropout, args.hops, args.r)
        optimizer = Adam(model.parameters(), lr=args.lr)

    # propagation
    model.operator.propagation(data.g.Adj(), data.g.X(), args.dataset)
    for epoch in range(args.epoch) :
        gc.collect()
        start = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--root", type=str, default="./")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--model", type=str, default="sgc")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=5000)
    args = parser.parse_args()
    main(args)