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
        print(data.g.X().shape[1], data.num_classes)
        model = models.SGC(data.g.X().shape[1], args.hidden, data.num_classes, args.dropout, args.hops, args.r).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
    
    if args.model == "sign":
        data = Ogbn(args.dataset, args.root, "official")
        if(args.batch_size != 0) :
            train_loader = DataLoader(data.train_idx, batch_size = args.batch_size, shuffle = True)
            val_loader = DataLoader(data.val_idx, batch_size = args.eval_batch_size, shuffle = False)
            test_loader = DataLoader(data.test_idx, batch_size = args.eval_batch_size, shuffle = False)
        print(data.g.X().shape[1], data.num_classes)
        model = models.SIGN(data.g.X().shape[1] * (args.num_layers + 1), args.hidden, data.num_classes, args.dropout, args.hops, args.r, args.num_layers).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
    # propagation
    model.propagation(data.g.Adj(), data.g.X(), args.dataset)
    loss_fcn = nn.CrossEntropyLoss()
    best_val = 0.
    best_test = 0.
    
    for epoch in range(args.epoch) :
        gc.collect()
        start = time.time()
        loss, acc = utils.batch_train(model, len(data.train_idx), data.g.Y(), loss_fcn, optimizer, train_loader, device)
        acc_val, acc_test = utils.batch_evaluate(model, len(data.val_idx), val_loader,
                                                        len(data.test_idx), test_loader,
                                                        data.g.Y(), device)
        end = time.time()
        print("epoch : {}, Times:{:.4f}, Train_loss{:.4f}, Train_acc{:.4f},\
         Val_acc{:.4f}, Test_acc{:.4f}".format(epoch, end - start, loss, acc, acc_val, acc_test))
        

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
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()
    main(args)

    # python main.py --dataset ogbn-arxiv --root dataset/dataset/ --gpu 0 --model sign --batch_size 5000 --hidden 2048 --dropout 0.5