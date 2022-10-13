import networkx as nx
import torch
from matplotlib import pylab as pl
from networkx import number_connected_components
from torch_geometric.utils import to_networkx


def set_labels(baseG):
    G = to_networkx(baseG, to_undirected=True)
    components = number_connected_components(G)
    print(f"Number of connected components in dataset: {components}")
    y = []
    for idx, subgraph in enumerate(baseG.pos):
        # draw_subgraph(G, subgraph)
        k = G.subgraph(subgraph.tolist())
        subgraph_components = number_connected_components(k)
        subgraph_density = nx.density(k)
        subgraph_has_bridge = nx.has_bridges(k)
        subgraph_clustering = nx.average_clustering(k)
        cycles = nx.cycle_basis(k)
        subgraph_bridges = 0
        if subgraph_has_bridge:
            subgraph_bridges = 1
        subgraph_cycles = len(cycles)
        y.append([subgraph_density, subgraph_clustering, subgraph_components, subgraph_bridges, subgraph_cycles])
        # print(subgraph_components)
    labels = torch.Tensor(y)
    return labels


def draw_subgraph(G, subgraph):
    # G = to_networkx(G, to_undirected=True)
    k = G.subgraph(subgraph.tolist())
    pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
    pl.figure()
    nx.draw_networkx(k, pos=pos)
    pl.show()
