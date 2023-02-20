import math
import random

from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from GLASSTest import set_seed


class GDataset:
    '''
    A class to contain splitted data.
    Args:
        x : node feature
        pos : the node set of target subgraphs. 
            For example, [[0, 1, 2], [6, 7, -1]] means two subgraphs containing nodes 0, 1, 2 and 6, 7 respectively.
        y : the target of subgraphs.
    '''

    def __init__(self, x, edge_index, edge_attr, pos, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.comp = pos
        self.num_nodes = x.shape[0]

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx]

    def sample_pos_comp(self, samples, m, M, stoch, views=1, device=0):
        if not stoch:
            print("Setting up non-stochastic data")
            N = self.x.shape[0]
            E = self.edge_index.size()[-1]
            sparse_adj = SparseTensor(
                row=self.edge_index[0].to(device), col=self.edge_index[1].to(device),
                value=torch.arange(E, device=device),
                sparse_sizes=(N, N))
            row, col, _ = sparse_adj.csr()
            y = []
            batch_comp_nodes = []
            subgraph_nodes_list = []
            for idx, graph_nodes in enumerate(self.pos):
                y_val = self.y[idx]
                updated_graph_nodes = graph_nodes[graph_nodes != -1].tolist()
                for i in range(views):
                    if samples:
                        updated_graph_node_list = updated_graph_nodes
                        if len(updated_graph_node_list) < math.ceil(samples * len(updated_graph_node_list)):
                            samples_required = len(updated_graph_node_list)
                        else:
                            samples_required = math.ceil(samples * len(updated_graph_node_list))
                        starting_for_rw = random.sample(updated_graph_node_list, k=samples_required)
                    else:
                        starting_for_rw = updated_graph_nodes
                    starting_nodes = torch.tensor(starting_for_rw, dtype=torch.long)
                    start = starting_nodes.repeat(M).to(device)
                    node_ids = torch.ops.torch_cluster.random_walk(row, col, start, m, 1, 1)[0]
                    rw_nodes = torch.unique(node_ids.flatten()).tolist()
                    subgraph_nodes = set(updated_graph_nodes).intersection(set(rw_nodes))
                    complement_nodes = set(rw_nodes).difference(subgraph_nodes)

                    batch_comp_nodes.append(torch.Tensor(list(complement_nodes)))
                    subgraph_nodes_list.append(torch.Tensor(list(subgraph_nodes)))
                    y.append(y_val)

            self.pos = pad_sequence(subgraph_nodes_list, batch_first=True, padding_value=-1).to(torch.int64)
            self.comp = pad_sequence(batch_comp_nodes, batch_first=True, padding_value=-1).to(torch.int64)
            self.y = torch.Tensor(y).to(torch.int64)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.comp = self.comp.to(device)
        self.y = self.y.to(device)
        return self


class GDataloader(DataLoader):
    '''
    Dataloader for GDataset
    '''

    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False, seed=-1):
        super(GDataloader,
              self).__init__(torch.arange(len(Gdataset)).to(Gdataset.x.device),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
        set_seed(seed)
        self.Gdataset = Gdataset

    def get_x(self):
        return self.Gdataset.x

    def get_ei(self):
        return self.Gdataset.edge_index

    def get_ea(self):
        return self.Gdataset.edge_attr

    def get_pos(self):
        return self.Gdataset.pos

    def get_comp(self):
        return self.Gdataset.comp

    def get_y(self):
        return self.Gdataset.y

    def __iter__(self):
        self.iter = super(GDataloader, self).__iter__()
        return self

    def __next__(self):
        perm = next(self.iter)
        return self.get_x(), self.get_ei(), self.get_ea(), self.get_pos(
        )[perm], self.get_comp()[perm], self.get_y()[perm]


class ZGDataloader(GDataloader):
    '''
    Dataloader for GDataset. 
    Args:
        z_fn: assigning node label for each batch.
    '''

    def __init__(self,
                 Gdataset,
                 batch_size=64,
                 shuffle=True,
                 drop_last=False,
                 z_fn=lambda x, y: torch.zeros(
                     (x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(ZGDataloader, self).__init__(Gdataset, batch_size, shuffle,
                                           drop_last)
        self.z_fn = z_fn

    def __next__(self):
        perm = next(self.iter)
        tpos = self.get_pos()[perm]
        return self.get_x(), self.get_ei(), self.get_ea(), tpos, self.z_fn(
            self.get_x(), tpos), self.get_y()[perm]
