import torch
from networkx import number_strongly_connected_components
from torch_geometric.utils import to_networkx


def set_labels_to_connected_components(baseG):
    G = to_networkx(baseG)
    components = number_strongly_connected_components(G)
    print(f"Number of connected components in dataset: {components}")
    y = []
    for subgraph in baseG.pos:
        k = G.subgraph(subgraph.tolist())
        subgraph_components = number_strongly_connected_components(k)
        y.append(subgraph_components)
        # print(subgraph_components)
    old_val = list(set(y))
    new_val = list(range(len(old_val)))
    new_dict = dict(zip(old_val, new_val))
    y = [new_dict[value] for value in y]
    baseG.y = torch.Tensor(y)
    return baseG
