import argparse
import functools
import itertools
import json
import os.path
import random
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import yaml
from ray import tune
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import MLP, GCNConv
from torch_sparse import SparseTensor

import datasets
from impl import models_hybrid, SubGDataset_hybrid, train_hybrid, metrics, utils, config
import warnings

from impl.models_hybrid import COMGraphConv

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None
loss_fn = None


def set_seed(seed: int):
    print("seed = ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


def split(args, hypertuning=False):
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset1, trn_dataset2, trn_dataset3, trn_dataset4, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn
    global row, col
    baseG = datasets.load_dataset(args.dataset, hypertuning)
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
    N = baseG.x.shape[0]
    E = baseG.edge_index.size()[-1]
    sparse_adj = SparseTensor(
        row=baseG.edge_index[0], col=baseG.edge_index[1],
        value=torch.arange(E, device="cpu"),
        sparse_sizes=(N, N))
    row, col, _ = sparse_adj.csr()
    baseG.to(config.device)
    # split data
    trn_dataset1 = SubGDataset_hybrid.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset_hybrid.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset_hybrid.GDataset(*baseG.get_split("test"))
    trn_dataset1.sample_pos_comp_train(m=args.m, M=args.M, nv=args.nv,
                                 device=config.device, row=row, col=col, dataset=args.dataset)
    val_dataset.sample_pos_comp_test(m=args.m, M=args.M, device=config.device,
                                row=row, col=col, dataset=args.dataset)
    tst_dataset.sample_pos_comp_test(m=args.m, M=args.M, device=config.device,
                                row=row, col=col, dataset=args.dataset)

    trn_dataset1 = trn_dataset1.to(config.device)
    val_dataset = val_dataset.to(config.device)
    tst_dataset = tst_dataset.to(config.device)
    # choice of dataloader
    if args.use_maxzeroone:

        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset_hybrid.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:

        def loader_fn(ds, bs, seed):
            return SubGDataset_hybrid.GDataloader(ds, bs, seed=seed)

        def tloader_fn(ds, bs, seed):
            return SubGDataset_hybrid.GDataloader(ds, bs, shuffle=True, seed=seed)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio, aggr, args=None, hypertuning=False):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn.
    '''
    conv = functools.partial(COMGraphConv, aggr=aggr, dropout=dropout)
    if args.use_gcn_conv:
        conv = functools.partial(GCNConv, add_self_loops=False)
    conv = models_hybrid.COMGraphLayerNet(hidden_dim,
                                   hidden_dim,
                                   conv_layer,
                                   max_deg=max_deg,
                                   activation=nn.ELU(inplace=True),
                                   jk=jk,
                                   dropout=dropout,
                                   conv=conv,
                                   gn=True)

    # use pretrained node embeddings.
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        path_to_emb = f"Emb/{args.dataset}_{hidden_dim}.pt"
        if hypertuning:
            path_to_emb = os.path.join('/media/nvme/sjacob/extended-GLASS/', path_to_emb)
        emb = torch.load(path_to_emb, map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    num_rep = 1
    in_channels = hidden_dim * (1) * num_rep if jk else hidden_dim
    if args.model == 0:
        in_channels = hidden_dim * (conv_layer) * num_rep if jk else hidden_dim
    if args.model == 2 and not args.diffusion:
        # if MLP mixing is enabled, num_rep is 1 throughout, else it becomes 2
        num_rep = 2
        in_channels = hidden_dim * (conv_layer) * num_rep if jk else hidden_dim

    mlp = MLP(channel_list=[in_channels, output_channels], dropout=[0], norm=None, act=None)
    # mlp = nn.Linear(hidden_dim * (1) * num_rep if jk else hidden_dim,
    #                 output_channels)

    pool_fn_fn = {
        "mean": models_hybrid.MeanPool,
        "max": models_hybrid.MaxPool,
        "sum": models_hybrid.AddPool,
        "size": models_hybrid.SizePool
    }
    if pool1 in pool_fn_fn and pool2 in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool1]()
        pool_fn2 = pool_fn_fn[pool2]()
        if args.model == 1 or args.model == 2:
            pooling_layers = torch.nn.ModuleList([pool_fn1, pool_fn2])
        else:
            pooling_layers = torch.nn.ModuleList([pool_fn1])
    else:
        raise NotImplementedError

    if args.use_mlp:
        num_rep = 1
        in_channels = hidden_dim * (1) * num_rep if jk else hidden_dim
        if args.model == 0:
            in_channels = hidden_dim * num_rep if jk else hidden_dim
        if args.model == 2 and not args.diffusion:
            # if MLP mixing is enabled, num_rep is 1 throughout, else it becomes 2
            num_rep = 2
            in_channels = hidden_dim * num_rep if jk else hidden_dim

        mlp = MLP(channel_list=[in_channels, output_channels], dropout=[0], norm=None, act=None)
        gnn = models_hybrid.COMGraphMLPMasterNet(preds=torch.nn.ModuleList([mlp]), pools=pooling_layers, model_type=args.model,
                                       hidden_dim=hidden_dim, max_deg=max_deg, diffusion=args.diffusion).to(
            config.device)
    else:
        gnn = models_hybrid.COMGraphMasterNet(conv, torch.nn.ModuleList([mlp]), pooling_layers, args.model, hidden_dim, conv_layer,
                                   args.diffusion).to(config.device)

    print("-" * 64)
    print("GNN Architecture is as follows ->")
    print(gnn)
    print("-" * 64)

    parameters = list(gnn.parameters())
    total_params = sum(p.numel() for param in parameters for p in param)

    print(f'Total number of parameters is {total_params}')
    print("-" * 64)

    return gnn


def test(pool1="size",
         pool2="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0,
         batch_size=None,
         resi=0,
         hypertuning=False,
         args=None):
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
    vals = []
    run_times = []
    trn_time = []
    inference_time = []
    preproc_times = []
    wamrup = {"ppi_bp": 50, "hpo_metab": 50, "hpo_neuro": 50, "em_user": 10}

    for repeat in range(args.repeat):
        start_time = time.time()
        if not hypertuning:
            set_seed(repeat + 1)
        print(f"repeat = {repeat}")

        tst_average = np.average(outs)
        tst_error = np.std(outs) / np.sqrt(len(outs))
        print(f"Average so far for {repeat} runs: {tst_average :.3f} ± {tst_error :.3f}")

        start_pre = time.time()
        split(args, hypertuning)
        # we set batch_size = tst_dataset.y.shape[0] // num_div.
        num_div = tst_dataset.y.shape[0] / batch_size
        # we use num_div to calculate the number of iteration per epoch and count the number of iteration.
        if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
            num_div /= 5

        print(f"Warmup and early stop steps are set to 100/{num_div}  = {100 / num_div}")
        print("-" * 64)
        early_stop = {"ppi_bp": 100 / num_div, "hpo_metab": 50, "hpo_neuro": 50, "em_user": 10}
        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio,
                         aggr, args, hypertuning)
        nve = args.nve
        trn_loader1 = sample_views(args, nve, trn_dataset1, batch_size, repeat + 1)
        trn_loader2 = sample_views(args, nve, trn_dataset1, batch_size, repeat + 1)
        trn_loader3 = sample_views(args, nve, trn_dataset1, batch_size, repeat + 1)
        trn_loader4 = sample_views(args, nve, trn_dataset1, batch_size, repeat + 1)
        val_loader = tloader_fn(val_dataset, batch_size, repeat + 1)
        tst_loader = tloader_fn(tst_dataset, batch_size, repeat + 1)
        end_pre = time.time()
        preproc_times.append(end_pre - start_pre)
        optimizer = Adam(gnn.parameters(), lr=lr)

        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)

        warmup = 50
        if args.use_mlp:
            warmup = 0
        val_score = 0
        tst_score = 0
        early_stop = 0
        print(f"Warm up for {100 / num_div} steps in progress...")
        for i in range(args.epochs):
            t1 = time.time()
            trn_loader = random.choice([trn_loader1, trn_loader2, trn_loader3, trn_loader4])

            trn_score, loss = train_hybrid.train(optimizer, gnn, trn_loader, score_fn, loss_fn, device=config.device,
                                          row=row, col=col, run=repeat + 1, epoch=i)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= warmup:
                score, _ = train_hybrid.test(gnn,
                                      val_loader,
                                      score_fn,
                                      loss_fn=loss_fn, device=config.device, row=row, col=col, run=repeat + 1, epoch=i)

                if score > val_score:
                    early_stop = 0
                    val_score = score
                    inf_start = time.time()
                    score, _ = train_hybrid.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device, row=row, col=col, run=repeat + 1,
                                          epoch=i)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                    print(f"Best picked so far- val: {val_score:.4f} tst: {tst_score:.4f}, early stop: {early_stop} \n")
                    if hypertuning:
                        tune.report(loss=loss, val_accuracy=val_score, test_accuracy=tst_score)
                elif score >= val_score - 1e-5:
                    inf_start = time.time()
                    score, _ = train_hybrid.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn, device=config.device, row=row, col=col, run=repeat + 1,
                                          epoch=i)
                    inf_end = time.time()
                    inference_time.append(inf_end - inf_start)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                    print(f"Best picked so far- val: {val_score:.4f} tst: {tst_score:.4f}, early stop: {early_stop} \n")
                    if hypertuning:
                        tune.report(loss=loss, val_accuracy=val_score, test_accuracy=tst_score)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        inf_start = time.time()
                        test = train_hybrid.test(gnn, tst_loader, score_fn, loss_fn=loss_fn, device=config.device,
                                          row=row, col=col, run=repeat + 1, epoch=i)
                        inf_end = time.time()
                        inference_time.append(inf_end - inf_start)
                        print(
                            f"iter {i} loss {loss:.4f} train {trn_score:.4f} val {score:.4f} tst {test[0]:.4f}",
                            flush=True)
                        print(
                            f"Best picked so far- val: {val_score:.4f} tst: {tst_score:.4f}, early stop: {early_stop} \n")
            if val_score >= 1 - 1e-5:
                early_stop += 1
            if not args.use_mlp:
                if early_stop > (100 / num_div):
                    print("Patience exhausted. Early stopping.")
                    break
        end_time = time.time()
        run_time = end_time - start_time
        run_times.append(run_time)
        print(f"Total run time: {run_time}")
        print(
            f"end: epoch {i}, train time {sum(trn_time):.2f} s, train {trn_score:.4f} val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score)
        vals.append(val_score)
    print(f"Time for {args.dataset} dataset and model {args.model}")
    print(f"Average run time: {np.average(run_times):.3f} ± {np.std(run_times):.3f}")
    print(f"Average preprocessing time: {np.average(preproc_times):.3f} ± {np.std(preproc_times):.3f}")
    print(f"Average train time: {np.average(trn_time):.3f} ± {np.std(trn_time):.3f}")
    print(f"Average inference time: {np.average(inference_time):.3f} ± {np.std(inference_time):.3f}")
    tst_average = np.average(outs)
    tst_error = np.std(outs) / np.sqrt(len(outs))
    val_average = np.average(vals)
    val_error = np.std(vals) / np.sqrt(len(vals))
    print(
        f"Test Accuracy {tst_average :.3f} ± {tst_error :.3f}"
    )
    print(
        f"Val Accuracy {val_average :.3f} ± {val_error :.3f}"
    )
    exp_results = {}
    exp_results[f"{args.dataset}_model{args.model}"] = {
        "results": {
            "Test Accuracy": f"{tst_average:.3f} ± {tst_error:.3f}",
            "Val Accuracy": f"{val_average:.3f} ± {val_error:.3f}",
            "Avg runtime": f"{np.average(run_times):.3f} ± {np.std(run_times):.3f}",
            "Avg preprocessing time": f"{np.average(preproc_times):.3f} ± {np.std(preproc_times):.3f}",
            "Avg train time": f"{np.average(trn_time):.3f} ± {np.std(trn_time):.3f}",
            "Avg inference time": f"{np.average(inference_time):.3f} ± {np.std(inference_time):.3f}",
        },
    }
    results_json = f"{args.dataset}_model{args.model}_results.json"
    if args.model == 2:
        results_json = f"{args.dataset}_model{args.model}_m_{args.m}_M_{args.M}_results.json"
        if args.diffusion:
            results_json = f"{args.dataset}_model{args.model}_m_{args.m}_M_{args.M}_with_diff_results.json"
    with open(results_json, 'w') as output_file:
        json.dump(exp_results, output_file)


def sample_views(args, nve, trn_dataset1, batch_size, repeat):
    trn_dataset = trn_dataset1
    selected_views = random.sample(range(0, args.nv), nve)
    selected_pos = [trn_dataset1.pos_temp[i] for i in selected_views]
    selected_comp = [trn_dataset1.comp_temp[i] for i in selected_views]
    selected_y = [trn_dataset1.y_temp[i] for i in selected_views]
    trn_dataset.pos = torch.stack(list(itertools.chain.from_iterable(selected_pos)), dim=0)
    trn_dataset.comp = pad_sequence(list(itertools.chain.from_iterable(selected_comp)), batch_first=True,
                                    padding_value=-1).to(
        torch.int64)
    if args.dataset == "hpo_neuro":
        trn_dataset.y = torch.vstack(list(itertools.chain.from_iterable(selected_y)))
    elif args.dataset == "em_user":
        trn_dataset.y = torch.Tensor(list(itertools.chain.from_iterable(selected_y)))
    else:
        trn_dataset.y = torch.Tensor(list(itertools.chain.from_iterable(selected_y))).to(torch.int64)
    trn_dataset = trn_dataset.to(config.device)
    return loader_fn(trn_dataset, batch_size, repeat)


def ray_tune_run_helper(config, argument_class, device):
    argument_class.m = config['m']
    argument_class.M = config['M']
    argument_class.samples = config['samples']
    argument_class.diffusion = config['diffusion']
    argument_class.device = device

    run_helper(argument_class, hypertuning=True)


def run_helper(argument_class, hypertuning=False):
    config.set_device(argument_class.device)

    if argument_class.use_seed:
        if not hypertuning:
            set_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

    baseG = datasets.load_dataset(argument_class.dataset, hypertuning)

    global trn_dataset, val_dataset, tst_dataset
    global max_deg, output_channels
    global score_fn, loss_fn

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

    loader_fn = SubGDataset_hybrid.GDataloader
    tloader_fn = SubGDataset_hybrid.GDataloader

    print("-" * 64)
    print("User input args", "->")
    print(argument_class)

    # read configuration
    path_to_config = f"compl-config/{argument_class.dataset}.yml"
    if hypertuning:
        path_to_config = os.path.join('/media/nvme/sjacob/extended-GLASS/', path_to_config)
    with open(path_to_config) as f:
        params = yaml.safe_load(f)
    print("-" * 64)

    print("Loaded YAML", "->")
    pprint(params)
    print("-" * 64)

    params.update({'args': argument_class,
                   'hypertuning': hypertuning})
    test(**(params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ppi_bp')
    # Node feature settings.
    # deg means use node degree. one means use homogeneous embeddings.
    # nodeid means use pretrained node embeddings in ./Emb
    parser.add_argument('--use_deg', action='store_true')
    parser.add_argument('--use_one', action='store_true')
    parser.add_argument('--use_nodeid', action='store_true')
    # model 0 means use subgraph emb. model 1 means use complement emb. model 2 means use both subgraph and complement.
    parser.add_argument('--model', type=int, default=0)
    # node label settings
    parser.add_argument('--use_maxzeroone', action='store_true')
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--M', type=int, default=0)
    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--nv', type=int, default=1)
    parser.add_argument('--nve', type=int, default=1)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_seed', action='store_true')

    parser.add_argument('--use_gcn_conv', action='store_true')
    parser.add_argument('--use_mlp', action='store_true')

    args = parser.parse_args()
    run_helper(args)
