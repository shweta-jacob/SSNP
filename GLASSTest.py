import argparse
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from networkx.algorithms.community import modularity
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, lr_scheduler, SGD
from torch_geometric.utils import to_networkx

import datasets
from artificial import graph1, graph3, graph4, graph5, graph2, graph7, graph8, graph9
from impl import models, SubGDataset, train, metrics, utils, config
from impl.models import SpectralNet

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

# baseG = datasets.load_dataset(args.dataset)
baseG = graph7.load_dataset()

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
    global trn_dataset, val_dataset, tst_dataset
    global train_subgraph_assignment, val_subgraph_assignment, test_subgraph_assignment
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
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))

    train_subgraph_assignment = torch.zeros((trn_dataset.pos.shape[0], trn_dataset.x.shape[0])).to(config.device)
    val_subgraph_assignment = torch.zeros((val_dataset.pos.shape[0], val_dataset.x.shape[0])).to(config.device)
    test_subgraph_assignment = torch.zeros((tst_dataset.pos.shape[0], tst_dataset.x.shape[0])).to(config.device)
    # a[torch.arange(a.size(0)).unsqueeze(1), index] = 1
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
    # val_subgraph_assignment = train_subgraph_assignment
    # test_subgraph_assignment = train_subgraph_assignment
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

        def loader_fn(ds, sa, bs):
            return SubGDataset.GDataloader(ds, sa, bs)

        def tloader_fn(ds, sa, bs):
            return SubGDataset.GDataloader(ds, sa, bs, shuffle=True)


def buildModel(hidden_dim1, hidden_dim2, conv_layer, dropout, jk, pool, z_ratio, aggr):
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
    # if args.use_nodeid:
    #     input_channels = 64

    average_nodes = int(trn_dataset.x.size(0))
    print(f"Average number of nodes in graph: {average_nodes}")
    gnn = SpectralNet(input_channels,
                      hidden_dim1,
                      hidden_dim2,
                      output_channels,
                      conv_layer,
                      max_deg=max_deg,
                      activation=nn.ELU(inplace=True),
                      jk=jk).to(config.device)

    # if args.use_nodeid:
    #     print("load ", f"./Emb/{args.dataset}_64.pt")
    #     emb = torch.load(f"./Emb/{args.dataset}_64.pt",
    #                      map_location=torch.device('cpu')).detach()
    #     gnn.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)
    #     gnn.input_emb.to(config.device)
    return gnn


def draw_clustering_on_g(nx_graph, s):
    # only support k = 2.
    color_map = []
    threshold = 0.75
    communities = [[], [], []]
    for node in nx_graph:
        score = s[node]
        if float(score[0]) >= threshold:
            color = 'red'
            communities[0].append(node)
        elif float(score[1]) >= threshold:
            color = 'blue'
            communities[1].append(node)
        else:
            color = 'grey'
            communities[2].append(node)
        color_map.append(color)

    nx.draw(nx_graph, node_color=color_map, with_labels=True, pos=nx.spring_layout(nx_graph))
    plt.show()
    mod = modularity(nx_graph, communities=communities)
    print(f"Modularity: {mod}")


def test(pool="size",
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
        num_div /= 30

    outs = []
    nx_graph = to_networkx(baseG.to(config.device), to_undirected=True, remove_self_loops=False)
    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        gnn = buildModel(hidden_dim1, hidden_dim2, conv_layer, dropout, jk, pool, z_ratio,
                         aggr)
        split()
        trn_loader = loader_fn(trn_dataset, train_subgraph_assignment, batch_size)
        val_loader = tloader_fn(val_dataset, val_subgraph_assignment, batch_size)
        tst_loader = tloader_fn(tst_dataset, test_subgraph_assignment, batch_size)
        optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=1e-4)
        # optimizer = SGD(gnn.parameters(), lr=lr, weight_decay=1e-4, momentum=0.8)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5,
                                             patience=10)
        val_score = 0
        train_scores = []
        val_scores = []
        tst_scores = []
        tst_score = 0
        train_score = 0
        early_stop = 0
        trn_time = []
        epochs = []
        s = None
        for i in range(50):
            t1 = time.time()
            train_score, loss, s = train.train(optimizer, gnn, trn_loader, score_fn, loss_fn)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 1:
                score, val_loss = train.test(gnn,
                                             val_loader,
                                             score_fn,
                                             loss_fn=loss_fn)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, tst_loss = train.test(gnn,
                                                 tst_loader,
                                                 score_fn,
                                                 loss_fn=loss_fn)
                    tst_score = score
                    val_scores.append(val_loss)
                    tst_scores.append(tst_loss)
                    train_scores.append(loss)
                    epochs.append(i + 1)
                    print(
                        f"iter {i + 1} loss {loss:.4f} train {train_score:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                elif score >= val_score - 1e-5:
                    score, tst_loss = train.test(gnn,
                                                 tst_loader,
                                                 score_fn,
                                                 loss_fn=loss_fn)
                    tst_score = max(score, tst_score)
                    val_scores.append(val_loss)
                    tst_scores.append(tst_loss)
                    train_scores.append(loss)
                    epochs.append(i + 1)
                    print(
                        f"iter {i + 1} loss {loss:.4f} train {train_score:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    print(
                        f"iter {i + 1} loss {loss:.4f} train {train_score:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)[0]:.4f}",
                        flush=True)
                    val_scores.append(val_loss)
                    tst_scores.append(tst_loss)
                    train_scores.append(loss)
                    epochs.append(i + 1)
            if val_score >= 1 - 1e-5:
                early_stop += 1
            # if early_stop > 100/num_div:
            #     break
        print(
            f"end: train time {sum(trn_time):.2f} s, train {train_score:.4f} val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score)
        figure(figsize=(8, 6))
        # plt.xlim([1, 200])
        plt.plot(np.array(epochs), np.array(train_scores))
        plt.plot(np.array(epochs), np.array(val_scores))
        plt.plot(np.array(epochs), np.array(tst_scores))
        # plt.xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(['Train', 'Validation', 'Test'])
        plt.show()

        # s, nx_graph
        draw_clustering_on_g(nx_graph, s)

    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )


print(args)
# read configuration
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
split()
test(**params)
