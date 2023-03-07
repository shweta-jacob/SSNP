import math
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu

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

    def neighbors(self, fringe, A, outgoing=True):
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

    def sample_pos_comp(self, num_hops, A=None):
        print("Setting up non-stochastic data")
        batch_comp_nodes = []
        samples = 1.0
        max_nodes_per_hop = None
        for idx, graph_nodes in enumerate(self.pos):
            center = graph_nodes[graph_nodes != -1].tolist()
            nodes = center
            visited = set(center)
            fringe = set(center)
            for dist in range(1, num_hops + 1):
                fringe = self.neighbors(fringe, A)
                fringe = fringe - visited
                visited = visited.union(fringe)
                if samples < 1.0:
                    fringe = random.sample(fringe, int(samples * len(fringe)))
                if max_nodes_per_hop is not None:
                    if max_nodes_per_hop < len(fringe):
                        fringe = random.sample(fringe, max_nodes_per_hop)
                if len(fringe) == 0:
                    break
                nodes = nodes + list(fringe)

                complement_nodes = set(nodes).difference(center)
                batch_comp_nodes.append(torch.Tensor(list(complement_nodes)))

        self.comp = pad_sequence(batch_comp_nodes, batch_first=True, padding_value=-1).to(torch.int64)

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
