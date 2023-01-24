import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric import utils
from torch_geometric.nn import GCNConv, dense_mincut_pool, global_mean_pool, GINConv
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
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def _weights_init(self, lin_layer):
        torch.nn.init.xavier_uniform_(lin_layer.weight.data)
        if lin_layer.bias is not None:
            lin_layer.bias.data.fill_(0.0)

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        # pass messages.
        x = self.adj @ x
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
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

class SpectralNet(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels1,
                 hidden_channels2,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 gn=True,
                 jk=False,
                 conv=GLASSConv,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      input_channels,
                                      scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(input_channels)
        self.num_clusters1 = 2
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2
        self.bns = torch.nn.ModuleList()
        self.bns.append(GraphNorm(hidden_channels1))
        self.bns.append(GraphNorm(hidden_channels1))
        self.gn = GraphNorm(hidden_channels1*3)
        self.convs = nn.ModuleList()
        self.jk = jk
        self.convs.append(
            conv(in_channels=input_channels,
                 out_channels=hidden_channels1,
                 activation=activation,
                 **kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels1,
                     out_channels=hidden_channels1,
                     activation=activation,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels1))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels1))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None
        self.activation = activation
        self.dropout = dropout
        self.mlp1 = Linear(3 * hidden_channels1, self.num_clusters1)
        self.num_layers = num_layers
        self.k1 = 2
        self.gn1 = GraphNorm(self.k1 * hidden_channels1)
        self.lin = Linear(self.k1, hidden_channels1)

        self.preds = torch.nn.ModuleList([MLP(input_channels=self.k1 * hidden_channels1 * 3,
                                              hidden_channels=hidden_channels2, output_channels=output_channels,
                                              num_layers=2, dropout=0.5)])

        self.reset_parameters()
        self._weights_init(self.mlp1)

    def _weights_init(self, lin_layer):
        torch.nn.init.xavier_uniform_(lin_layer.weight.data)
        if lin_layer.bias is not None:
            lin_layer.bias.data.fill_(0.0)

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, pos, subgraph_assignment):
        # Propagate node feats
        init_embs = []
        mask = (torch.zeros(
            (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)
        init_embs.append(x)
        init_embs = torch.stack(init_embs, dim=0).reshape(x.shape[0], -1)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, mask)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        x = torch.cat(xs, dim=-1)
        x = self.gn(x)

        # Cluster assignments (logits)
        s = self.mlp1(x)
        s = torch.softmax(s / 0.1, dim=-1)
        ent_loss1 = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
        l = torch.transpose(subgraph_assignment, 0, 1)
        transposed_s = torch.transpose(s, 0, 1)
        subgraph_to_cluster1 = transposed_s @ l
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.shape[0])
        out, out_adj, mc_loss1, o_loss1 = dense_mincut_pool(x, adj, s, temp=0.1)
        out = out.reshape(self.num_clusters1, self.hidden_channels1 * 3)

        cluster_sizes = torch.sum(s, dim=0)
        sorted_cluster_indices1 = cluster_sizes.sort(descending=True)[1]

        batch, subg_nodes = pad2batch(pos)
        emb = x[subg_nodes]
        emb = global_mean_pool(emb, batch)

        comp_embs = []
        node_embs = []
        for idx, row in enumerate(subgraph_assignment):
            node_emb = row.reshape(x.shape[0], 1) * x
            # r = subgraph_to_cluster1[:, idx]
            cluster_emb = torch.transpose(torch.softmax(s, dim=-1), 0, 1) @ node_emb
            node_emb = cluster_emb[sorted_cluster_indices1]
            # cluster_emb = torch.cat([cluster_emb, r.reshape(self.num_clusters1, 1)], dim=-1)
            # cluster_emb = self.global_sort1(cluster_emb)
            node_embs.append(node_emb.reshape(self.k1 * self.hidden_channels1 * 3))
            comp_emb = out - cluster_emb
            comp_emb = comp_emb[sorted_cluster_indices1]
            comp_embs.append(comp_emb.reshape(self.k1 * self.hidden_channels1 * 3))

        emb1 = torch.stack(node_embs, dim=0)
        # emb1 = self.gn1(emb1)
        emb2 = torch.stack(comp_embs, dim=0)

        return self.preds[0](torch.cat([emb1], dim=-1)), init_embs, emb, mc_loss1, o_loss1, ent_loss1


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
