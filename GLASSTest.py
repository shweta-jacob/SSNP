import argparse
import functools
import itertools
import random
import time

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from GNNSeg import GsDataset, GsDataloader
from torch_geometric.utils import k_hop_subgraph

import datasets
from GNNSeg import GNN, GConv, MyGINConv
from impl import models, SubGDataset, train, metrics, utils, config
from impl.models import MLP

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')
parser.add_argument('--pool1', type=str, default='mean')
parser.add_argument('--pool2', type=str, default='mean')
# Node feature settings. 
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')
parser.add_argument('--num_hops', type=int, default=1)

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


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def extract_neighborhood(dataset_split):
    edge_weight = Tensor.cpu(dataset_split[2])
    A = ssp.csr_matrix(
        (edge_weight, (Tensor.cpu(dataset_split[1][0]), Tensor.cpu(dataset_split[1][1]))),
        shape=(dataset_split[0].shape[0], dataset_split[0].shape[0])
    )

    comp = []
    for idx, nodes in enumerate(dataset_split[3]):
        # remove padding from subgraph node list
        subgraph_nodes = list(filter(lambda node: node != -1, nodes.tolist()))
        visited = set(subgraph_nodes)
        fringe = set(subgraph_nodes)
        neighborhood = []
        for dist in range(1, args.num_hops + 1):
            fringe = neighbors(fringe, A)
            fringe = fringe - visited
            neighborhood.append(list(fringe))
            visited = visited.union(fringe)
            if len(fringe) == 0:
                break
        comp.append(torch.Tensor(list(itertools.chain.from_iterable(neighborhood))))

    comp = pad_sequence(comp, batch_first=True, padding_value=-1).to(torch.int64)
    return comp.to(config.device)

def split_GNN_seg(trn_dataset, val_dataset, tst_dataset):
    def todata(x, edge_index, edge_attr, centre, hop, y):
        node, edge, inv, edge_mask = k_hop_subgraph(centre,
                                                    hop,
                                                    edge_index,
                                                    relabel_nodes=True)
        if node.shape[0] == 0:
            print("empty", centre)
        npos = torch.zeros_like(node, device=node.device)
        npos[inv] = 1
        if not torch.any(npos):
            print("empty", centre)
        return Data(x[node], edge, edge_attr[edge_mask], y=y, pos=npos)

    # def tocompdatalist(gd, hop):
    #     return [
    #         todata(gd.x, gd.edge_index, gd.edge_attr,
    #                gd.comp[i][gd.comp[i] >= 0], hop, gd.y[i])
    #         for i in range(len(gd))
    #     ]

    def tosubdatalist(gd, hop):
        return [
            todata(gd.x, gd.edge_index, gd.edge_attr,
                   gd.pos[i][gd.pos[i] >= 0], hop, gd.y[i])
            for i in range(len(gd))
        ]

    global sub_trn_dataset, sub_val_dataset, sub_tst_dataset, comp_trn_dataset, comp_val_dataset, comp_tst_dataset, seg_loader_fn, seg_tloader_fn

    sub_trn_dataset = GsDataset(tosubdatalist(trn_dataset, 0))
    sub_val_dataset = GsDataset(tosubdatalist(val_dataset, 0))
    sub_tst_dataset = GsDataset(tosubdatalist(tst_dataset, 0))

    # comp_trn_dataset = GsDataset(tocompdatalist(trn_dataset, 0))
    # comp_val_dataset = GsDataset(tocompdatalist(val_dataset, 0))
    # comp_tst_dataset = GsDataset(tocompdatalist(tst_dataset, 0))

    def tfunc(ds, bs, shuffle=False, drop_last=False):
        return GsDataloader(ds, bs, shuffle=shuffle, drop_last=drop_last)

    def seg_loader_fn(ds, bs):
        return tfunc(ds, bs)

    def seg_tloader_fn(ds, bs):
        return tfunc(ds, bs, False, False)

def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset
    global max_deg, output_channels, loader_fn, tloader_fn
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
    baseG.to(config.device)
    # split data
    train = baseG.get_split("train")
    valid = baseG.get_split("valid")
    test = baseG.get_split("test")

    train_comp = extract_neighborhood(train)
    valid_comp = extract_neighborhood(valid)
    test_comp = extract_neighborhood(test)

    trn_dataset = SubGDataset.GDataset(*train, train_comp)
    val_dataset = SubGDataset.GDataset(*valid, valid_comp)
    tst_dataset = SubGDataset.GDataset(*test, test_comp)

    split_GNN_seg(trn_dataset=trn_dataset, val_dataset=val_dataset, tst_dataset=tst_dataset)
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
            return SubGDataset.GDataloader(ds, bs, shuffle=False)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=False)


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
    conv = models.EmbZGConv(hidden_dim,
                            hidden_dim,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(models.GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout),
                            gn=True)

    sub_conv = GConv(hidden_dim,
                     hidden_dim,
                     hidden_dim,
                     conv_layer,
                     max_deg=max_deg,
                     conv=functools.partial(GCNConv, add_self_loops=False),
                     activation=nn.ELU(inplace=True),
                     dropout=dropout)

    # comp_conv = GConv(hidden_dim,
    #                   hidden_dim,
    #                   hidden_dim,
    #                   conv_layer,
    #                   max_deg=max_deg,
    #                   conv=functools.partial(GCNConv, add_self_loops=False),
    #                   activation=nn.ELU(inplace=True),
    #                   dropout=dropout)


    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
        sub_conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
        # comp_conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    sub_gnn = GNN(sub_conv).to(config.device)
    # comp_gnn = GNN(comp_conv).to(config.device)

    mlp = MLP(input_channels=3 * hidden_dim * (conv_layer), hidden_channels=2 * hidden_dim,
              output_channels=output_channels, num_layers=4)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool,
        "sort": models.SortPool
    }
    if pool1 in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool1]()
    else:
        raise NotImplementedError

    if pool2 in pool_fn_fn:
        pool_fn2 = pool_fn_fn[pool2]()
    else:
        raise NotImplementedError

    gnn = models.GLASS(conv, sub_gnn, torch.nn.ModuleList([mlp]), torch.nn.ModuleList([pool_fn1, pool_fn2]),
                       hidden_dim, output_channels, conv_layer, pool1, pool2).to(config.device)
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

    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        start_time = time.time()
        set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio,
                         aggr)
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)

        sub_trn_loader = seg_loader_fn(sub_trn_dataset, batch_size)
        sub_val_loader = seg_tloader_fn(sub_val_dataset, batch_size)
        sub_tst_loader = seg_tloader_fn(sub_tst_dataset, batch_size)

        # comp_trn_loader = seg_loader_fn(comp_trn_dataset, batch_size)
        # comp_val_loader = seg_tloader_fn(comp_val_dataset, batch_size)
        # comp_tst_loader = seg_tloader_fn(comp_tst_dataset, batch_size)

        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []
        for i in range(300):
            t1 = time.time()
            loss = train.train(optimizer, gnn, trn_loader, sub_trn_loader, loss_fn,
                               device=config.device)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 100 / num_div:
                score, _ = train.test(gnn,
                                      val_loader,
                                      sub_val_loader,
                                      score_fn,
                                      loss_fn=loss_fn, device=config.device)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          sub_tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                elif score >= val_score - 1e-5:
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          sub_tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, sub_tst_loader, score_fn, loss_fn=loss_fn, device=config.device)[0]:.4f}",
                            flush=True)
            if val_score >= 1 - 1e-5:
                early_stop += 1
            if early_stop > 100 / num_div:
                break
        print(
            f"end: epoch {i + 1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        end_time = time.time()
        outs.append(tst_score)
        time_taken = end_time - start_time
        print(f'Time taken: {time_taken}')
    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )


print(args)
# read configuration
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

params.update({'pool1': args.pool1, 'pool2': args.pool2})

print("params", params, flush=True)
split()
test(**(params))
