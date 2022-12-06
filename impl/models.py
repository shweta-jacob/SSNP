from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.nn import GCNConv, dense_mincut_pool
from torch_geometric.nn import aggr
from torch_geometric.nn.norm import GraphNorm

from .utils import pad2batch


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


class MLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout: float = 0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


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
    deg = torch.sparse.sum(adj, dim=(1,)).to_dense().flatten()
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
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)

        x = self.activation(self.trans_fns[0](x_))
        # pass messages.
        x = self.adj @ x
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

    def forward(self, x, edge_index, edge_weight, z=None):
        # z is the node label.
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        else:
            mask = (z > 0.5).reshape(-1, 1)
        # convert integer input to vector node features.
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
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


# class AddPool(PoolModule):
#     def __init__(self, trans_fn=None):
#         super().__init__(global_add_pool, trans_fn)
#
#
# class MaxPool(PoolModule):
#     def __init__(self, trans_fn=None):
#         super().__init__(global_max_pool, trans_fn)
#
#
# class MeanPool(PoolModule):
#     def __init__(self, trans_fn=None):
#         super().__init__(global_mean_pool, trans_fn)


# class SizePool(AddPool):
#     def __init__(self, trans_fn=None):
#         super().__init__(trans_fn)
#
#     def forward(self, x, batch):
#         if x is not None:
#             if self.trans_fn is not None:
#                 x = self.trans_fn(x)
#         x = GraphSizeNorm()(x, batch)
#         return self.pool_fn(x, batch)


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                 pools: nn.ModuleList):
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
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


# models used for producing node embeddings.
class Ensemble(torch.nn.Module):
    def __init__(self,
                 plain_gnn,
                 spectral_gnn,
                 hidden_channels2,
                 output_channels,):
        super().__init__()
        self.input_emb = None
        self.plain_gnn = plain_gnn
        self.spectral_gnn = spectral_gnn
        self.hidden_channels2 = hidden_channels2
        self.k = 20
        self.preds = torch.nn.ModuleList([MLP(input_channels=(self.hidden_channels2 * 2 + self.k),
                                              hidden_channels=hidden_channels2, output_channels=output_channels,
                                              num_layers=4, dropout=0.5)])
    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, x2, edge_index, edge_weight, pos, subgraph_assignment):
        # subgraph_emb = self.plain_gnn(x, edge_index, edge_weight, pos)
        subgraph_emb = self.Pool(self.input_emb, pos, self.plain_gnn.pools[0])
        cont_labels, mc_loss, o_loss, sub_loss, ent_loss = self.spectral_gnn(x, edge_index, edge_weight, pos,
                                                                             subgraph_assignment)
        return self.preds[0](torch.cat([subgraph_emb, cont_labels],
                                       dim=-1)), mc_loss, o_loss, 0, ent_loss


class SpectralNet(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels1,
                 hidden_channels2,
                 output_channels,
                 num_layers,
                 average_nodes,
                 num_clusters1,
                 num_clusters2,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      input_channels,
                                      scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(input_channels)
        self.num_clusters1 = 100
        self.num_clusters2 = 70
        self.num_clusters3 = 50
        self.num_clusters4 = 20
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels1))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels2))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels2))
        self.convs = nn.ModuleList()
        self.jk = jk
        self.conv1 = GLASSConv(in_channels=input_channels, out_channels=hidden_channels1, activation=activation,
                               **kwargs)
        self.conv2 = GLASSConv(in_channels=hidden_channels1, out_channels=hidden_channels2, activation=activation,
                               **kwargs)
        self.conv3 = GLASSConv(in_channels=hidden_channels2, out_channels=hidden_channels2, activation=activation,
                               **kwargs)
        self.conv4 = GLASSConv(in_channels=hidden_channels2, out_channels=hidden_channels2, activation=activation,
                               **kwargs)
        self.activation = activation
        self.dropout = dropout
        self.mlp1 = Linear(hidden_channels1, self.num_clusters1)
        self.mlp2 = Linear(hidden_channels2, self.num_clusters2)
        self.mlp3 = Linear(hidden_channels2, self.num_clusters3)
        self.mlp4 = Linear(hidden_channels2, self.num_clusters4)
        self.num_layers = num_layers
        self.k1 = 5
        self.k2 = 5
        self.k3 = 5
        self.k4 = 5
        self.global_sort1 = aggr.SortAggregation(k=self.k1)
        self.global_sort2 = aggr.SortAggregation(k=self.k2)
        self.global_sort3 = aggr.SortAggregation(k=self.k3)
        self.global_sort4 = aggr.SortAggregation(k=self.k4)

        # self.preds = torch.nn.ModuleList([MLP(input_channels=(self.k1 + self.k2 + self.k3 + self.k4),
        #                                       hidden_channels=2 * hidden_channels2, output_channels=output_channels,
        #                                       num_layers=4, dropout=0.5)])

        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        # for conv in self.convs:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        # if not (self.gns is None):
        #     for gn in self.gns:
        #         gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, pos, subgraph_assignment):
        # Propagate node feats
        x1 = self.input_emb(x).reshape(x.shape[0], -1)
        x1 = self.emb_gn(x1)
        # x = self.convs[-1](x, edge_index, edge_weight)
        # x = self.activation(x)
        x = self.conv1(x1, edge_index, edge_weight)
        x = self.bns[0](x)
        x = self.activation(x)

        # Cluster assignments (logits)
        s = self.mlp1(x)
        ent_loss1 = (-torch.softmax(s, dim=-1) * torch.log(torch.softmax(s, dim=-1) + 1e-15)).sum(dim=-1).mean()
        print(f"Entropy loss: {ent_loss1}")
        l = torch.transpose(subgraph_assignment, 0, 1)
        subgraph_to_cluster1 = F.normalize(torch.transpose(torch.softmax(s, dim=-1), 0, 1), p=1,
                                           dim=1) @ l
        if len(edge_index[0]) == 0:
            adj = torch.zeros(x.shape[0], x.shape[0])
        else:
            adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.shape[0])
        out, out_adj, mc_loss1, o_loss1 = dense_mincut_pool(x, adj, s)
        out = out.reshape(self.num_clusters1, self.hidden_channels1)
        # # Motif adj matrix - not sym. normalised
        # motif_adj = torch.mul(torch.matmul(adj, adj), adj)
        # motif_out_adj = torch.matmul(torch.matmul(torch.transpose(s, 0, 1), motif_adj), s)
        #
        # # Higher order cut
        # diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        # d_flat = torch.einsum("ijk->ij", motif_adj)
        # d = _rank3_diag(d_flat)
        # diag_SDS = (torch.einsum(
        #     "ijk->ij", torch.matmul(torch.matmul(torch.transpose(s, 0, 1), d), s)) +
        #             1e-15)
        # ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        # ho_mincut_loss1 = 1 / self.num_clusters1 * torch.mean(ho_mincut_loss)
        # subgraph_mc_loss1 = 0
        # for idx, subgraph_nodes in enumerate(pos):
        #     edge_index_s, edge_attr_s = utils.subgraph(subgraph_nodes, edge_index, edge_weight)
        #     if len(edge_attr_s):
        #         adj_s = utils.to_dense_adj(edge_index_s, edge_attr=edge_attr_s, max_num_nodes=len(x1))
        #         adj_s = adj_s.unsqueeze(0) if adj_s.dim() == 2 else adj_s
        #         out_adj_s = torch.matmul(torch.matmul(torch.softmax(s, dim=-1).transpose(0, 1), adj_s), torch.softmax(s, dim=-1))
        #         # MinCut regularization.
        #         mincut_num = _rank3_trace(out_adj_s)
        #         d_flat = torch.einsum('ijk->ij', adj_s)
        #         d = _rank3_diag(d_flat)
        #         mincut_den = _rank3_trace(
        #             torch.matmul(torch.matmul(s.transpose(0, 1), d), s))
        #         mincut_loss = -(mincut_num / mincut_den)
        #         mincut_loss = torch.mean(mincut_loss)
        #         subgraph_mc_loss1 += mincut_loss
        #
        # subgraph_mc_loss1 = subgraph_mc_loss1/len(pos)
        # preds = []
        embs = []
        for idx, subgraph in enumerate(pos):
            r = subgraph_to_cluster1[:, idx]
            r = r.sort(descending=True)[0]
            # x = torch.cat([out, r.reshape(self.num_clusters1, 1)], dim=-1)
            # x = out * r.reshape(self.num_clusters, 1)
            # pooled_features = x[x[:, -1].sort(descending=True)[1]]
            # # pooled_features = x
            # pooled_features = pooled_features.reshape(1, self.num_clusters1 * (self.hidden_channels1 + 1))  # [num_graphs, 1, k * hidden]
            # embs.append(pooled_features)
            # x = self.global_sort1(x)
            # x = x.reshape(self.k1 * (self.hidden_channels1 + 1))
            embs.append(r[:self.k1])
        emb1 = torch.stack(embs, dim=0)
        # emb1 = emb.reshape(len(pos), self.k1 * (self.hidden_channels1 + 1))

        new_adj = out_adj.reshape(self.num_clusters1, self.num_clusters1)
        updated_edge_index = new_adj.nonzero().t().contiguous()
        all_edge_weights = torch.flatten(new_adj)
        updated_edge_weight = all_edge_weights[torch.nonzero(all_edge_weights)].reshape(updated_edge_index[0].shape, )
        x = self.conv2(out, updated_edge_index, updated_edge_weight.detach())
        x = self.bns[1](x)
        x = self.activation(x)

        # Cluster assignments (logits)
        s = self.mlp2(x)
        ent_loss2 = (-torch.softmax(s, dim=-1) * torch.log(torch.softmax(s, dim=-1) + 1e-15)).sum(dim=-1).mean()
        subgraph_to_cluster2 = F.normalize(torch.transpose(torch.softmax(s, dim=-1), 0, 1), p=1,
                                           dim=1) @ subgraph_to_cluster1
        if len(updated_edge_index[0]) == 0:
            adj = torch.zeros(x.shape[0], x.shape[0])
        else:
            adj = utils.to_dense_adj(updated_edge_index, edge_attr=updated_edge_weight, max_num_nodes=x.shape[0])
        out, out_adj, mc_loss2, o_loss2 = dense_mincut_pool(x, adj, s)
        out = out.reshape(self.num_clusters2, self.hidden_channels2)
        # # Motif adj matrix - not sym. normalised
        # motif_adj = torch.mul(torch.matmul(adj, adj), adj)
        # motif_out_adj = torch.matmul(torch.matmul(torch.transpose(s, 0, 1), motif_adj), s)
        #
        # # Higher order cut
        # diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        # d_flat = torch.einsum("ijk->ij", motif_adj)
        # d = _rank3_diag(d_flat)
        # diag_SDS = (torch.einsum(
        #     "ijk->ij", torch.matmul(torch.matmul(torch.transpose(s, 0, 1), d), s)) +
        #             1e-15)
        # ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        # ho_mincut_loss2 = 1 / self.num_clusters2 * torch.mean(ho_mincut_loss)
        # subgraph_mc_loss2 = 0
        # for idx, subgraph_nodes in enumerate(pos):
        #     edge_index_s, edge_attr_s = utils.subgraph(subgraph_nodes, updated_edge_index, updated_edge_weight)
        #     if len(edge_attr_s):
        #         adj_s = utils.to_dense_adj(edge_index_s, edge_attr=edge_attr_s, max_num_nodes=len(x1))
        #         adj_s = adj_s.unsqueeze(0) if adj_s.dim() == 2 else adj_s
        #         out_adj_s = torch.matmul(torch.matmul(torch.softmax(s, dim=-1).transpose(0, 1), adj_s),
        #                                  torch.softmax(s, dim=-1))
        #         # MinCut regularization.
        #         mincut_num = _rank3_trace(out_adj_s)
        #         d_flat = torch.einsum('ijk->ij', adj_s)
        #         d = _rank3_diag(d_flat)
        #         mincut_den = _rank3_trace(
        #             torch.matmul(torch.matmul(s.transpose(0, 1), d), s))
        #         mincut_loss = -(mincut_num / mincut_den)
        #         mincut_loss = torch.mean(mincut_loss)
        #         subgraph_mc_loss2 += mincut_loss
        #
        # subgraph_mc_loss2 = subgraph_mc_loss2 / len(pos)
        # preds = []
        embs = []
        for idx, subgraph in enumerate(pos):
            r = subgraph_to_cluster2[:, idx]
            r = r.sort(descending=True)[0]
            # x = torch.cat([out, r.reshape(self.num_clusters2, 1)], dim=-1)
            # x = out * r.reshape(self.num_clusters, 1)
            # pooled_features = x[x[:, -1].sort(descending=True)[1]]
            # # pooled_features = x
            # pooled_features = pooled_features.reshape(1, self.num_clusters2 * (
            #             self.hidden_channels2 + 1))  # [num_graphs, 1, k * hidden]
            # embs.append(pooled_features)
            # x = self.global_sort2(x)
            # x = x.reshape(self.k2 * (self.hidden_channels2 + 1))
            embs.append(r[:self.k2])
        emb2 = torch.stack(embs, dim=0)
        # emb2 = emb.reshape(len(pos), self.k2 * (self.hidden_channels2 + 1))

        new_adj = out_adj.reshape(self.num_clusters2, self.num_clusters2)
        updated_edge_index = new_adj.nonzero().t().contiguous()
        all_edge_weights = torch.flatten(new_adj)
        updated_edge_weight = all_edge_weights[torch.nonzero(all_edge_weights)].reshape(updated_edge_index[0].shape, )
        x = self.conv3(out, updated_edge_index, updated_edge_weight.detach())
        x = self.bns[2](x)
        x = self.activation(x)

        # Cluster assignments (logits)
        s = self.mlp3(x)
        ent_loss3 = (-torch.softmax(s, dim=-1) * torch.log(torch.softmax(s, dim=-1) + 1e-15)).sum(dim=-1).mean()
        subgraph_to_cluster3 = F.normalize(torch.transpose(torch.softmax(s, dim=-1), 0, 1), p=1,
                                           dim=1) @ subgraph_to_cluster2
        if len(updated_edge_index[0]) == 0:
            adj = torch.zeros(x.shape[0], x.shape[0])
        else:
            adj = utils.to_dense_adj(updated_edge_index, edge_attr=updated_edge_weight, max_num_nodes=x.shape[0])
        out, out_adj, mc_loss3, o_loss3 = dense_mincut_pool(x, adj, s)
        out = out.reshape(self.num_clusters3, self.hidden_channels2)
        # # Motif adj matrix - not sym. normalised
        # motif_adj = torch.mul(torch.matmul(adj, adj), adj)
        # motif_out_adj = torch.matmul(torch.matmul(torch.transpose(s, 0, 1), motif_adj), s)
        #
        # # Higher order cut
        # diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        # d_flat = torch.einsum("ijk->ij", motif_adj)
        # d = _rank3_diag(d_flat)
        # diag_SDS = (torch.einsum(
        #     "ijk->ij", torch.matmul(torch.matmul(torch.transpose(s, 0, 1), d), s)) +
        #             1e-15)
        # ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        # ho_mincut_loss3 = 1 / self.num_clusters3 * torch.mean(ho_mincut_loss)
        # subgraph_mc_loss3 = 0
        # for idx, subgraph_nodes in enumerate(pos):
        #     edge_index_s, edge_attr_s = utils.subgraph(subgraph_nodes, updated_edge_index, updated_edge_weight)
        #     if len(edge_attr_s):
        #         adj_s = utils.to_dense_adj(edge_index_s, edge_attr=edge_attr_s, max_num_nodes=len(x1))
        #         adj_s = adj_s.unsqueeze(0) if adj_s.dim() == 2 else adj_s
        #         out_adj_s = torch.matmul(torch.matmul(torch.softmax(s, dim=-1).transpose(0, 1), adj_s),
        #                                  torch.softmax(s, dim=-1))
        #         # MinCut regularization.
        #         mincut_num = _rank3_trace(out_adj_s)
        #         d_flat = torch.einsum('ijk->ij', adj_s)
        #         d = _rank3_diag(d_flat)
        #         mincut_den = _rank3_trace(
        #             torch.matmul(torch.matmul(s.transpose(0, 1), d), s))
        #         mincut_loss = -(mincut_num / mincut_den)
        #         mincut_loss = torch.mean(mincut_loss)
        #         subgraph_mc_loss2 += mincut_loss
        #
        # subgraph_mc_loss2 = subgraph_mc_loss2 / len(pos)
        # preds = []
        embs = []
        for idx, subgraph in enumerate(pos):
            r = subgraph_to_cluster3[:, idx]
            r = r.sort(descending=True)[0]
            # x = torch.cat([out, r.reshape(self.num_clusters3, 1)], dim=-1)
            # x = out * r.reshape(self.num_clusters, 1)
            # pooled_features = x[x[:, -1].sort(descending=True)[1]]
            # # pooled_features = x
            # pooled_features = pooled_features.reshape(1, self.num_clusters2 * (
            #             self.hidden_channels2 + 1))  # [num_graphs, 1, k * hidden]
            # embs.append(pooled_features)
            # x = self.global_sort3(x)
            # x = x.reshape(self.k3 * (self.hidden_channels2 + 1))
            embs.append(r[:self.k3])
        emb3 = torch.stack(embs, dim=0)
        # emb3 = emb.reshape(len(pos), self.k3 * (self.hidden_channels2 + 1))

        new_adj = out_adj.reshape(self.num_clusters3, self.num_clusters3)
        updated_edge_index = new_adj.nonzero().t().contiguous()
        all_edge_weights = torch.flatten(new_adj)
        updated_edge_weight = all_edge_weights[torch.nonzero(all_edge_weights)].reshape(updated_edge_index[0].shape)
        x = self.conv4(out, updated_edge_index, updated_edge_weight.detach())

        # Cluster assignments (logits)
        s = self.mlp4(x)
        ent_loss4 = (-torch.softmax(s, dim=-1) * torch.log(torch.softmax(s, dim=-1) + 1e-15)).sum(dim=-1).mean()
        subgraph_to_cluster4 = F.normalize(torch.transpose(torch.softmax(s, dim=-1), 0, 1), p=1,
                                           dim=1) @ subgraph_to_cluster3
        if len(updated_edge_index[0]) == 0:
            adj = torch.zeros(x.shape[0], x.shape[0])
        else:
            adj = utils.to_dense_adj(updated_edge_index, edge_attr=updated_edge_weight, max_num_nodes=x.shape[0])
        out, out_adj, mc_loss4, o_loss4 = dense_mincut_pool(x, adj, s)
        out = out.reshape(self.num_clusters4, self.hidden_channels2)
        # # Motif adj matrix - not sym. normalised
        # motif_adj = torch.mul(torch.matmul(adj, adj), adj)
        # motif_out_adj = torch.matmul(torch.matmul(torch.transpose(s, 0, 1), motif_adj), s)
        #
        # # Higher order cut
        # diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        # d_flat = torch.einsum("ijk->ij", motif_adj)
        # d = _rank3_diag(d_flat)
        # diag_SDS = (torch.einsum(
        #     "ijk->ij", torch.matmul(torch.matmul(torch.transpose(s, 0, 1), d), s)) +
        #             1e-15)
        # ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        # ho_mincut_loss4 = 1 / self.num_clusters4 * torch.mean(ho_mincut_loss)
        # subgraph_mc_loss4 = 0
        # for idx, subgraph_nodes in enumerate(pos):
        #     edge_index_s, edge_attr_s = utils.subgraph(subgraph_nodes, updated_edge_index, updated_edge_weight)
        #     if len(edge_attr_s):
        #         adj_s = utils.to_dense_adj(edge_index_s, edge_attr=edge_attr_s, max_num_nodes=len(x1))
        #         adj_s = adj_s.unsqueeze(0) if adj_s.dim() == 2 else adj_s
        #         out_adj_s = torch.matmul(torch.matmul(torch.softmax(s, dim=-1).transpose(0, 1), adj_s),
        #                                  torch.softmax(s, dim=-1))
        #         # MinCut regularization.
        #         mincut_num = _rank3_trace(out_adj_s)
        #         d_flat = torch.einsum('ijk->ij', adj_s)
        #         d = _rank3_diag(d_flat)
        #         mincut_den = _rank3_trace(
        #             torch.matmul(torch.matmul(s.transpose(0, 1), d), s))
        #         mincut_loss = -(mincut_num / mincut_den)
        #         mincut_loss = torch.mean(mincut_loss)
        #         subgraph_mc_loss2 += mincut_loss
        #
        # subgraph_mc_loss2 = subgraph_mc_loss2 / len(pos)
        # preds = []
        embs = []
        for idx, subgraph in enumerate(pos):
            r = subgraph_to_cluster4[:, idx]
            r = r.sort(descending=True)[0]
            # x = torch.cat([out, r.reshape(self.num_clusters4, 1)], dim=-1)
            # x = out * r.reshape(self.num_clusters, 1)
            # pooled_features = x[x[:, -1].sort(descending=True)[1]]
            # # pooled_features = x
            # pooled_features = pooled_features.reshape(1, self.num_clusters2 * (
            #             self.hidden_channels2 + 1))  # [num_graphs, 1, k * hidden]
            # embs.append(pooled_features)
            # x = self.global_sort4(x)
            # x = x.reshape(self.k4 * (self.hidden_channels2 + 1))
            embs.append(r[:self.k4])
        emb4 = torch.stack(embs, dim=0)
        # emb4 = emb.reshape(len(pos), self.k4 * (self.hidden_channels2 + 1))

        return torch.cat([emb1, emb2, emb3, emb4],
                         dim=-1), mc_loss1 + mc_loss2 + mc_loss3 + mc_loss4, o_loss1 + o_loss2 + o_loss3 + o_loss4, 0, ent_loss1 +  ent_loss2 + ent_loss3 + ent_loss4


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
