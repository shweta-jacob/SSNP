import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, global_add_pool, MLP
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.transforms import SIGN

from .utils import pad2batch


class MySIGN(SIGN):

    def __call__(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(data.num_nodes, data.num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        assert data.x is not None
        xs = [data.x]
        for i in range(1, self.K + 1):
            xs += [adj_t @ xs[-1]]
        setattr(data, f'x', torch.cat(xs, dim=-1))

        return data

class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


# class MLP(nn.Module):
#     '''
#     Multi-Layer Perception.
#     Args:
#         tail_activation: whether to use activation function at the last layer.
#         activation: activation function.
#         gn: whether to use GraphNorm layer.
#     '''
#     def __init__(self,
#                  input_channels: int,
#                  hidden_channels: int,
#                  output_channels: int,
#                  num_layers: int,
#                  dropout=0,
#                  tail_activation=False,
#                  activation=nn.ReLU(inplace=True),
#                  gn=False):
#         super().__init__()
#         modlist = []
#         self.seq = None
#         if num_layers == 1:
#             modlist.append(nn.Linear(input_channels, output_channels))
#             if tail_activation:
#                 if gn:
#                     modlist.append(GraphNorm(output_channels))
#                 if dropout > 0:
#                     modlist.append(nn.Dropout(p=dropout, inplace=True))
#                 modlist.append(activation)
#             self.seq = Seq(modlist)
#         else:
#             modlist.append(nn.Linear(input_channels, hidden_channels))
#             for _ in range(num_layers - 2):
#                 if gn:
#                     modlist.append(GraphNorm(hidden_channels))
#                 if dropout > 0:
#                     modlist.append(nn.Dropout(p=dropout, inplace=True))
#                 modlist.append(activation)
#                 modlist.append(nn.Linear(hidden_channels, hidden_channels))
#             if gn:
#                 modlist.append(GraphNorm(hidden_channels))
#             if dropout > 0:
#                 modlist.append(nn.Dropout(p=dropout, inplace=True))
#             modlist.append(activation)
#             modlist.append(nn.Linear(hidden_channels, output_channels))
#             if tail_activation:
#                 if gn:
#                     modlist.append(GraphNorm(output_channels))
#                 if dropout > 0:
#                     modlist.append(nn.Dropout(p=dropout, inplace=True))
#                 modlist.append(activation)
#             self.seq = Seq(modlist)
#
#     def forward(self, x):
#         return self.seq(x)


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError


class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 dropout=0.2):
        super().__init__()
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.adj @ x_
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=GLASSConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     **kwargs))
        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        # convert integer input to vector node features.
        # x = self.input_emb(x).reshape(x.shape[0], -1)
        # x = self.emb_gn(x)
        # xs = []
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # # pass messages at each layer.
        # for layer, conv in enumerate(self.convs[:-1]):
        #     x = conv(x, edge_index, edge_weight)
        #     xs.append(x)
        #     if not (self.gns is None):
        #         x = self.gns[layer](x)
        #     x = self.activation(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, edge_index, edge_weight)
        # xs.append(x)
        #
        # if self.jk:
        #     x = torch.cat(xs, dim=-1)
        #     if not (self.gns is None):
        #         x = self.gns[-1](x)
        #     return x
        # else:
        #     x = xs[-1]
        #     if not (self.gns is None):
        #         x = self.gns[-1](x)
        #     return x
        # x = torch.cat(xs, dim=-1)
        return x


class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args: 
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                 pools: nn.ModuleList, model_type, hidden_dim):
        super().__init__()
        self.conv = conv
        self.operator_diff = MLP(channel_list=[hidden_dim*3, hidden_dim*2, hidden_dim],
              act_first=True, act="ELU", dropout=[0.5, 0.5])
        self.preds = preds
        self.pools = pools
        self.model_type = model_type

    def NodeEmb(self, x, edge_index, edge_weight):
        # embs = []
        # for _ in range(x.shape[1]):
        #     emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
        #                     edge_index, edge_weight)
        #     embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        # emb = torch.cat(embs, dim=1)
        # emb = torch.mean(emb, dim=1)
        emb = self.operator_diff(x)
        return emb

    def Pool(self, emb, subG_node, pool):
        if self.model_type == 0:
            batch, pos = pad2batch(subG_node)
            emb_subg = emb[pos]
            emb = pool[0](emb_subg, batch)
        elif self.model_type == 1:
            graph_emb = torch.sum(emb, dim=0)
            all_graph_embs = graph_emb.repeat(len(subG_node), 1)
            batch, pos = pad2batch(subG_node)
            emb_subg = emb[pos]
            emb_subg = pool[0](emb_subg, batch)
            emb = torch.sub(all_graph_embs, emb_subg)
        else:
            graph_emb = torch.sum(emb, dim=0)
            all_graph_embs = graph_emb.repeat(len(subG_node), 1)
            batch, pos = pad2batch(subG_node)
            emb_subg = emb[pos]
            emb_subg = pool[0](emb_subg, batch)
            emb_comp = torch.sub(all_graph_embs, emb_subg)
            emb = torch.cat([emb_subg, emb_comp], dim=-1)

        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight)
        emb = self.Pool(emb, subG_node, self.pools)
        return self.preds[id](emb)


# models used for producing node embeddings.


class MyGCNConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for pretrained GNNs.
    Args:
        aggr: the aggregation method.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean"):
        super().__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)

    def reset_parameters(self):
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.trans_fn(x_)
        x = self.activation(x)
        x = self.adj @ x
        x = self.gn(x)
        x = torch.cat((x, x_), dim=-1)
        x = self.comb_fn(x)
        return x


class EmbGConv(torch.nn.Module):
    '''
    combination of some message passing layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
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
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        x = F.dropout(self.input_emb(x.reshape(-1)),
                      p=self.dropout,
                      training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]


class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        emb = emb[subG_node]
        emb = torch.mean(emb, dim=1)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)
