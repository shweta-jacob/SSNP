import argparse
import functools
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, lr_scheduler
from torch_geometric.utils import to_networkx

import datasets
from artificial import graph1, graph3, graph4, graph5, graph2, graph7, graph8, graph9
from impl import models, SubGDataset, train, metrics, utils, config
from impl.models import SpectralNet
from impl.plain_gnn_models import GLASS, EmbZGConv, MeanPool, MaxPool, AddPool, SizePool, GLASSConv

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

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

args = parser.parse_args()
config.set_device(args.device)


def set_seed(seed: int, f):
    print("seed ", seed, file=f)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    f = open('output.log', 'w')
    set_seed(0, f)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
train_subgraph_assignment, val_subgraph_assignment, test_subgraph_assignment = None, None, None
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
    global trn_dataset, val_dataset, tst_dataset, baseG
    global train_subgraph_assignment, val_subgraph_assignment, test_subgraph_assignment
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
        # baseG.setNodeIdFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    # import networkx as nx
    # from matplotlib import pylab as pl
    #
    # class0 = []
    # class1 = []
    # class2 = []
    # for idx, subgraph in enumerate(trn_dataset.pos):
    #     G = to_networkx(baseG, to_undirected=True)
    #     # print(trn_dataset.y[idx])
    #     k = G.subgraph(subgraph.tolist())
    #     density = nx.density(k)
    #     if trn_dataset.y[idx] == 0:
    #         class0.append(density)
    #     elif trn_dataset.y[idx] == 1:
    #         class1.append(density)
    #     elif trn_dataset.y[idx] == 2:
    #         class2.append(density)
    #     # print(density)
    #     # pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
    #     # pl.figure()
    #     # nx.draw_networkx(k, pos=pos)
    #     #
    #     # pl.show()
    #
    # class0 = []
    # class1 = []
    # class2 = []
    # for idx, subgraph in enumerate(val_dataset.pos):
    #     G = to_networkx(baseG, to_undirected=True)
    #     print(val_dataset.y[idx])
    #     k = G.subgraph(subgraph.tolist())
    #     density = nx.density(k)
    #     if val_dataset.y[idx] == 0:
    #         class0.append(density)
    #     elif val_dataset.y[idx] == 1:
    #         class1.append(density)
    #     elif val_dataset.y[idx] == 2:
    #         class2.append(density)
    #     print(density)
    #     # pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
    #     # pl.figure()
    #     # nx.draw_networkx(k, pos=pos)
    #     #
    #     # pl.show()
    #
    # class0 = []
    # class1 = []
    # class2 = []
    # for idx, subgraph in enumerate(tst_dataset.pos):
    #     G = to_networkx(baseG, to_undirected=True)
    #     print(tst_dataset.y[idx])
    #     k = G.subgraph(subgraph.tolist())
    #     density = nx.density(k)
    #     if tst_dataset.y[idx] == 0:
    #         class0.append(density)
    #     elif tst_dataset.y[idx] == 1:
    #         class1.append(density)
    #     elif tst_dataset.y[idx] == 2:
    #         class2.append(density)
    #     print(density)
    #     # pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
    #     # pl.figure()
    #     # nx.draw_networkx(k, pos=pos)
    #     #
    #     # pl.show()
    train_subgraph_assignment = torch.zeros((trn_dataset.pos.shape[0], trn_dataset.x.shape[0])).to(config.device)
    val_subgraph_assignment = torch.zeros((val_dataset.pos.shape[0], val_dataset.x.shape[0])).to(config.device)
    test_subgraph_assignment = torch.zeros((tst_dataset.pos.shape[0], tst_dataset.x.shape[0])).to(config.device)
    for idx, pos in enumerate(trn_dataset.pos):
        for node in pos:
            if node != -1:
                train_subgraph_assignment[idx][node] = 1
    for idx, pos in enumerate(val_dataset.pos):
        for node in pos:
            if node != -1:
                val_subgraph_assignment[idx][node] = 1
    for idx, pos in enumerate(tst_dataset.pos):
        for node in pos:
            if node != -1:
                test_subgraph_assignment[idx][node] = 1
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


def buildModel(f, hidden_dim1, hidden_dim2, conv_layer, dropout, jk, pool, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn. 
    '''
    input_channels = hidden_dim1
    if args.use_nodeid:
        input_channels = 64

    num_clusters1 = 500
    num_clusters2 = 200
    average_nodes = int(trn_dataset.x.size(0))
    print(f"Average number of nodes in graph: {average_nodes}")
    # print(f'Number of clusters in each layer: {num_clusters1}, {num_clusters2}', file=f)
    gnn = SpectralNet(input_channels,
                      hidden_dim1,
                      hidden_dim2,
                      output_channels,
                      conv_layer,
                      average_nodes,
                      num_clusters1,
                      num_clusters2,
                      max_deg=max_deg,
                      activation=nn.ELU(inplace=True),
                      jk=jk).to(config.device)

    conv = EmbZGConv(hidden_dim1,
                            hidden_dim2,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout),
                            gn=True)
    pool_fn_fn = {
        "mean": MeanPool,
        "max": MaxPool,
        "sum": AddPool,
        "size": SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError
    plain_gnn = GLASS(conv,
                       torch.nn.ModuleList([pool_fn1])).to(config.device)
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_64.pt")
        emb = torch.load(f"./Emb/{args.dataset}_64.pt",
                         map_location=torch.device('cpu')).detach()

    #     # plain_gnn.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
        gnn.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
        gnn.input_emb.to(config.device)
        # plain_gnn.input_emb.to(config.device)
    # emb = torch.load(f"./subgraph_emb/cut_ratio_emb.pt",
    #                  map_location=torch.device('cpu')).detach()
    # gnn.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
    # gnn.input_emb.to(config.device)
    from prettytable import PrettyTable
    # https://stackoverflow.com/a/62508086
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    ensemble = models.Ensemble(plain_gnn, gnn, hidden_dim2, output_channels)
    count_parameters(ensemble)
    # ensemble.input_emb = emb
    return ensemble


def test(f,
         pool="size",
         aggr="mean",
         hidden_dim1=64,
         hidden_dim2=64,
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
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size
    # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        set_seed(1, f)
        print(f"repeat {repeat}", file=f)
        split()
        gnn = buildModel(f, hidden_dim1, hidden_dim2, conv_layer, dropout, jk, pool, z_ratio,
                         aggr)
        # trn_loader = loader_fn(trn_dataset, batch_size)
        # val_loader = tloader_fn(val_dataset, batch_size)
        # tst_loader = tloader_fn(tst_dataset, batch_size)
        optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=1e-4)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)
        val_score = 0
        tst_score = 0
        train_score = 0
        early_stop = 0
        trn_time = []
        training_losses = []
        epochs = []
        prev_classification_loss = 0
        prev_clustering_loss = 0
        for i in range(20000):
            t1 = time.time()
            train_score, loss, classification_loss, clustering_loss = train.train(optimizer, gnn, trn_dataset,
                                                                     train_subgraph_assignment, score_fn, loss_fn,
                                                                     prev_classification_loss, prev_clustering_loss)
            prev_classification_loss = classification_loss
            prev_clustering_loss = clustering_loss
            trn_time.append(time.time() - t1)
            if i % 10 == 0:
                training_losses.append(loss.detach().numpy())
                epochs.append(i)
            scd.step(loss)

            if i >= 1:
                score, _ = train.test(f,gnn,
                                      val_dataset,
                                      val_subgraph_assignment,
                                      score_fn,
                                      loss_fn=loss_fn)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = train.test(f,gnn,
                                          tst_dataset,
                                          test_subgraph_assignment,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} train {train_score:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True, file=f)
                elif score >= val_score - 1e-5:
                    score, _ = train.test(f,gnn,
                                          tst_dataset,
                                          test_subgraph_assignment,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} train {train_score:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True, file=f)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} train {train_score:.4f} val {score:.4f} tst {train.test(f, gnn, tst_dataset, test_subgraph_assignment, score_fn, loss_fn=loss_fn)[0]:.4f}",
                            flush=True, file=f)
            if val_score >= 1 - 1e-5:
                early_stop += 1
            # if early_stop > 1000:
            #     break
        print(
            f"end: epoch {i + 1}, train time {sum(trn_time):.2f} s, train {train_score:.4f} val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True, file=f)
        outs.append(tst_score)
        figure(figsize=(8, 6))
        plt.plot(np.array(epochs), np.array(training_losses))
        # plt.xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
        plt.xlabel("Epochs")
        plt.ylabel("Training loss")
        plt.show()
    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}", file=f
    )


with open('output.log', 'w') as out_file:
    print(args, file=out_file)
    # read configuration
    with open(f"config/{args.dataset}.yml") as f:
        params = yaml.safe_load(f)

    print("params", params, flush=True, file=out_file)
    split()
    test(out_file, **params)
f.close()
