from torch_geometric.data import Data
import torch
from torch.utils.data import DataLoader


class GDataset:
    '''
    A class to contain splitted data.
    Args:
        x : node feature
        pos : the node set of target subgraphs. 
            For example, [[0, 1, 2], [6, 7, -1]] means two subgraphs containing nodes 0, 1, 2 and 6, 7 respectively.
        y : the target of subgraphs.
    '''

    def __init__(self, x, edge_index, edge_attr, pos, y, comp):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.num_nodes = x.shape[0]
        self.comp = comp

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx], self.comp[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.comp = self.comp.to(device)
        return self


class GDataloader(DataLoader):
    '''
    Dataloader for GDataset
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False):
        super(GDataloader,
              self).__init__(torch.arange(len(Gdataset)).to(Gdataset.x.device),
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
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
        )[perm], self.get_comp()[perm], self.get_y()[perm],


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
