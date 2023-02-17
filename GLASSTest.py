import argparse
import functools
import json
import random
import time

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import yaml
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, lr_scheduler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_sparse import SparseTensor

import datasets
from impl import models, SubGDataset, train, metrics, utils, config
from impl.models import GLASSConv, GCN, DGCNN, SIGNNet
from impl.utils import extract_enclosing_subgraphs
import warnings

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
# Node feature settings. 
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')
parser.add_argument('--num_hops', type=int, default=0)
parser.add_argument('--m', type=int, default=1)
parser.add_argument('--M', type=int, default=5)
parser.add_argument('--samples', type=float, default=0)
parser.add_argument('--num_powers', type=int, default=1)
parser.add_argument('--views', type=int, default=1)
parser.add_argument('--pool', type=str, default="mean")

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())


    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset, baseG, trn_loader, val_loader, tst_loader
    global max_deg, output_channels, loader_fn, tloader_fn
    baseG = datasets.load_dataset(args.dataset)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)
    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    # baseG.to(config.device)
    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{64}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{64}.pt",
                         map_location=torch.device('cpu')).detach()
        trn_dataset.x = emb
        val_dataset.x = emb
        tst_dataset.x = emb

    N = trn_dataset.x.shape[0]
    E = trn_dataset.edge_index.size()[-1]
    sparse_adj = SparseTensor(
        row=trn_dataset.edge_index[0], col=trn_dataset.edge_index[1],
        value=torch.arange(E, device="cpu"),
        sparse_sizes=(N, N))
    rw_kwargs = {"rw_m": args.m, "rw_M": args.M, "rw_samples": args.samples, "sparse_adj": sparse_adj,
                 "edge_index": trn_dataset.edge_index,
                 "device": config.device,
                 "data": trn_dataset}

    A = ssp.csr_matrix(
        (trn_dataset.edge_attr.cpu().numpy(),
         (trn_dataset.edge_index[0].cpu().numpy(), trn_dataset.edge_index[1].cpu().numpy())),
        shape=(trn_dataset.x.shape[0], trn_dataset.x.shape[0])
    )
    trn_list = extract_enclosing_subgraphs(
        trn_dataset.pos, A, trn_dataset.x, trn_dataset.y, args.num_hops, args.num_powers, views=args.views, rw_kwargs=rw_kwargs, edge_index=trn_dataset.edge_index)
    val_list = extract_enclosing_subgraphs(
        val_dataset.pos, A, val_dataset.x, val_dataset.y, args.num_hops, args.num_powers, rw_kwargs=rw_kwargs, edge_index=val_dataset.edge_index)
    tst_list = extract_enclosing_subgraphs(
        tst_dataset.pos, A, tst_dataset.x, tst_dataset.y, args.num_hops, args.num_powers, rw_kwargs=rw_kwargs, edge_index=tst_dataset.edge_index)
    trn_loader = DataLoader(trn_list, batch_size=32, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_list, batch_size=32, num_workers=0, shuffle=True, drop_last=True)
    tst_loader = DataLoader(tst_list, batch_size=32, num_workers=0, shuffle=True, drop_last=True)
    # choice of dataloader
    if args.use_maxzeroone:

        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:

        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn.
    '''
    synthetic = True

    emb = baseG.x
    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        emb = nn.Embedding.from_pretrained(emb, freeze=False)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool
    }
    if pool1 in pool_fn_fn and pool2 in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool1]()
        pool_fn2 = pool_fn_fn[pool2]()
    else:
        raise NotImplementedError

    gnn = SIGNNet(hidden_channels=hidden_dim, num_layers=conv_layer, powers=args.num_powers, train_dataset=trn_dataset,
                  pool=pool_fn_fn[args.pool](), output_channels=output_channels).to(config.device)
    parameters = list(gnn.parameters())
    total_params = sum(p.numel() for param in parameters for p in param)
    print(f'Total number of parameters is {total_params}')
    return gnn


def test(pool1="size",
         pool2="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         resi=0.7):
    '''
    Test a set of hyperparameters in a task.
    Args:
        jk: whether to use Jumping Knowledge Network.
        z_ratio: see GLASSConv in impl/model.py. A hyperparameter of GLASS.
        resi: the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()

    outs = []
    run_times = []
    trn_time = []
    inference_time = []
    preproc_times = []
    for repeat in range(args.repeat):
        start_time = time.time()
        set_seed(repeat + 1)
        print(f"repeat {repeat}")
        start_pre = time.time()
        split()
        num_div = tst_dataset.y.shape[0] / batch_size
        # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
        if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
            num_div /= 5
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio,
                         aggr)
        # trn_loader = loader_fn(trn_dataset, batch_size)
        # val_loader = tloader_fn(val_dataset, batch_size)
        # tst_loader = tloader_fn(tst_dataset, batch_size)
        end_pre = time.time()
        preproc_times.append(end_pre - start_pre)
        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        val_score = 0
        tst_score = 0
        early_stop = 0
        for i in range(300):
            t1 = time.time()
            trn_score, loss = train.train(optimizer, gnn, trn_loader, score_fn, loss_fn, device=config.device)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 50:
                score, _ = train.test(gnn,
                                      val_loader,
                                      score_fn,
                                      loss_fn=loss_fn, device=config.device)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    inf_start = time.time()
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                    print(f"Best so far- val {val_score:.4f} tst {tst_score:.4f}")
                elif score >= val_score - 1e-5:
                    inf_start = time.time()
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                    print(f"Best so far- val {val_score:.4f} tst {tst_score:.4f}")
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        inf_start = time.time()
                        test = train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn, device=config.device)
                        inf_end = time.time()
                        inference_time.append(inf_end - inf_start)
                        print(
                            f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {score:.4f} tst {test[0]:.4f}",
                            flush=True)
                        print(f"Best so far- val {val_score:.4f} tst {tst_score:.4f}")
            if val_score >= 1 - 1e-5:
                early_stop += 1
            if early_stop > 30:
                break
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"Total run time: {run_time}")
        print(
            f"end: epoch {i}, train time {sum(trn_time):.2f} s, train {trn_score:.4f} val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score * 100)
    print(f"Time for {args.dataset} dataset")
    print(f"Average run time: {np.average(run_times):.3f} with std {np.std(run_times):.3f}")
    print(f"Average preprocessing time: {np.average(preproc_times):.3f} with std {np.std(preproc_times):.3f}")
    print(f"Average train time: {np.average(trn_time):.3f} with std {np.std(trn_time):.3f}")
    print(f"Average inference time: {np.average(inference_time):.3f} with std {np.std(inference_time):.3f}")
    tst_average = np.average(outs)
    tst_error = np.std(outs) / np.sqrt(len(outs))
    print(
        f"average {tst_average :.3f} error {tst_error :.3f}"
    )
    exp_results = {}
    exp_results[f"{args.dataset}"] = {
        "results": {
            "Test Accuracy": f"{tst_average:.2f} error {tst_error:.2f}",
            "Avg runtime": f"{np.average(run_times):.2f} with std {np.std(run_times):.2f}",
            "Avg preprocessing time": f"{np.average(preproc_times):.2f} with std {np.std(preproc_times):.2f}",
            "Avg train time": f"{np.average(trn_time):.2f} with std {np.std(trn_time):.3f}",
            "Avg inference time": f"{np.average(inference_time):.2f} with std {np.std(inference_time):.2f}",
        },
    }
    with open(f"{args.dataset}_pos_powers_{args.num_powers}_views_{args.views}_samples_{args.samples}_m_{args.m}_M_{args.M}_results.json", 'w') as output_file:
        json.dump(exp_results, output_file)


print(args)
# read configuration
with open(f"compl-config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
test(**(params))
