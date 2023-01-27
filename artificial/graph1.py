import torch
from torch.nn.utils.rnn import pad_sequence
import networkx as nx
import datasets

def load_dataset():
    num_node = 11
    x = torch.empty((num_node, 1, 0))

    rawedge = nx.read_edgelist(f"./artificial/graph1/edgelist.txt").edges
    edge_index = torch.tensor([[int(i[0]), int(i[1])]
                                       for i in rawedge]).t()

    train_sub_G = [[1, 2, 3, 4], [6, 7, 8, 9], [0, 1, 6, 7], [5, 6, 1, 2]]
    val_sub_G = [[5, 9, 3, 4], [5, 6, 7, 9]]
    test_sub_G = val_sub_G
    train_sub_G_label = torch.Tensor([0, 0, 1, 1])
    val_sub_G_label = torch.Tensor([1, 0])
    test_sub_G_label = val_sub_G_label

    mask = torch.cat(
        (torch.zeros(len(train_sub_G_label), dtype=torch.int64),
         torch.ones(len(val_sub_G_label), dtype=torch.int64),
         2 * torch.ones(len(test_sub_G_label))),
        dim=0)

    label = torch.cat(
                    (train_sub_G_label, val_sub_G_label, test_sub_G_label))
    pos = pad_sequence(
                [torch.tensor(i) for i in train_sub_G + val_sub_G + test_sub_G],
                batch_first=True,
                padding_value=-1)
    baseG = datasets.BaseGraph(x, edge_index, torch.ones(edge_index.shape[1]), pos,
                               label.to(torch.float), mask)

    return baseG