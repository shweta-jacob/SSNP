# Taken from: https://github.com/Xi-yuanWang/GLASS/blob/6f865380b340f38833ff27fdf03a1257fb6ac26b/GNNSeg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphNorm, GINConv

from impl import utils

'''
Dataset and Dataloader class for segregated subgraph
'''


class GsDataset(InMemoryDataset):
    '''
    designed for GNN-seg.
    '''

    def __init__(self, datalist):
        self.datalist = datalist
        super(GsDataset, self).__init__()
        self.data, self.slices = self.collate(self.datalist)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]

    def to(self, device):
        self.data = self.data.to(device)
        for i in self.slices:
            self.slices[i] = self.slices[i].to(device)
        return self


class GsDataloader(DataLoader):
    '''
    dataloader for GsDataset
    '''

    def __init__(self, Gsdataset, batch_size=64, shuffle=True, drop_last=True):
        super(GsDataloader, self).__init__(Gsdataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           drop_last=drop_last)

    def __iter__(self):
        self.iter = super(GsDataloader, self).__iter__()
        return self

    def __next__(self):
        batch = next(self.iter)
        x = batch.x
        ei = batch.edge_index
        ea = batch.edge_attr
        pos = utils.batch2pad(batch.batch)
        y = batch.y
        return x, ei, ea, pos, y


'''
Models for segregated subgraph
'''


class GConv(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 **kwargs):
        super(GConv, self).__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=False)
        self.convs = nn.ModuleList()
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        self.gns = nn.ModuleList(
            [GraphNorm(hidden_channels) for layer in range(num_layers - 1)])
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for gn in self.gns:
            gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        # convert integer input to vector node features.
        x = self.input_emb(x).reshape(x.shape[0], -1)
        xs = []
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        return torch.cat(xs, dim=-1)


class GNN(torch.nn.Module):
    def __init__(self, conv):
        super(GNN, self).__init__()
        self.mods = nn.ModuleList()
        self.mods.append(conv)

    def forward(self, x, edge_index, edge_weight, subG_node):
        def pos2sp(pos, n_node: int):
            coord_2 = torch.arange(pos.shape[0]).reshape(
                -1,
                1)[:,
                      torch.zeros(pos.shape[1], dtype=torch.int64)].to(pos.device)
            coord = torch.stack([coord_2.flatten(), pos.flatten()])
            coord = coord[:, coord[1] >= 0]
            vec_pos = torch.sparse_coo_tensor(coord,
                                              torch.ones(coord.shape[1],
                                                         device=pos.device),
                                              size=(pos.shape[0], n_node),
                                              device=pos.device)
            return vec_pos

        embs = []
        for _ in range(x.shape[1]):
            emb = self.mods[0](x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                               edge_index, edge_weight)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        sp = pos2sp(subG_node, emb.shape[0])
        emb = sp @ emb
        return emb


class MyGINConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyGINConv, self).__init__()

        self.conv = GINConv(nn.Linear(in_channels, out_channels), 0, False)

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight):
        return self.conv(x, edge_index)


seg_loader_fn = None
seg_tloader_fn = None
sub_trn_dataset = None
sub_val_dataset = None
sub_tst_dataset = None
comp_trn_dataset = None
comp_val_dataset = None
comp_tst_dataset = None


# def test(hidden_dim=64, conv_layer=8, dropout=0.3, lr=1e-3, batch_size=160):
#     def set_seed(seed: int):
#         print("seed ", seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     trn_loader = loader_fn(trn_dataset, batch_size)
#     val_loader = tloader_fn(val_dataset, batch_size)
#     tst_loader = tloader_fn(tst_dataset, batch_size)
#     outs = []
#     for _ in range(args.repeat):
#         print(f"repeat {_}")
#         set_seed(_)
#         gnn = buildModel(hidden_dim, conv_layer, dropout)
#         optimizer = Adam(gnn.parameters(), lr=lr)
#         scd = lr_scheduler.ReduceLROnPlateau(optimizer,
#                                              factor=0.7,
#                                              min_lr=5e-5)
#         val_score = 0
#         early_stop = 0
#         tst_score = 0
#         for i in range(500):
#             loss = train.train(optimizer, gnn, trn_loader, loss_fn)
#             scd.step(loss)
#             if i % 5 == 0:
#                 score, _ = train.test(gnn,
#                                       val_loader,
#                                       score_fn,
#                                       loss_fn=loss_fn)
#                 early_stop += 1
#                 if score > val_score:
#                     val_score = score
#                     score, _ = train.test(gnn,
#                                           tst_loader,
#                                           score_fn,
#                                           loss_fn=loss_fn)
#                     tst_score = score
#                     print(
#                         f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
#                         flush=True)
#                     early_stop /= 2
#                 elif score >= val_score - 1e-5:
#                     score, _ = train.test(gnn,
#                                           tst_loader,
#                                           score_fn,
#                                           loss_fn=loss_fn)
#                     tst_score = max(score, tst_score)
#                     print(
#                         f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
#                         flush=True)
#                     early_stop /= 2
#                 else:
#                     print(
#                         f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)[0]:.4f}",
#                         flush=True)
#                 if early_stop > 10:
#                     break
#         print(f"end: val {val_score:.4f} tst {tst_score:.4f}", flush=True)
#         outs.append(tst_score)
#     print("tst scores", outs)
#     print(np.average(outs), np.std(outs) / np.sqrt(len(outs)))
#     return np.average(outs)


best_hyperparams = {
    'density': {
        'conv_layer': 1,
        'dropout': 0.4,
        'hidden_dim': 16
    },
    'component': {
        'conv_layer': 1,
        'dropout': 0.0,
        'hidden_dim': 16
    },
    'coreness': {
        'conv_layer': 1,
        'dropout': 0.3,
        'hidden_dim': 16
    },
    'cut_ratio': {
        'conv_layer': 1,
        'dropout': 0.1,
        'hidden_dim': 4
    },
    'hpo_neuro': {
        'conv_layer': 1,
        'dropout': 0.4,
        'hidden_dim': 64
    },
    'ppi_bp': {
        'conv_layer': 8,
        'dropout': 0.4,
        'hidden_dim': 64
    },
    'hpo_metab': {
        'conv_layer': 1,
        'dropout': 0.1,
        'hidden_dim': 64
    },
    'em_user': {
        'conv_layer': 1,
        'dropout': 0.4,
        'hidden_dim': 64
    }
}

