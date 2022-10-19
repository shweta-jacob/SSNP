import itertools
import scipy.sparse as ssp

import networkx as nx
import torch
from matplotlib import pylab as pl, pyplot as plt
from networkx import number_connected_components
from torch import Tensor
from torch_geometric.utils import to_networkx


def neighbors(fringe, A, outgoing=True):
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


def extract_neighborhood(dataset_split, subgraph):
    edge_weight = dataset_split.edge_attr
    A = ssp.csr_matrix(
        (edge_weight, (Tensor.cpu(dataset_split.edge_index[0]), Tensor.cpu(dataset_split.edge_index[1]))),
        shape=(dataset_split.x.shape[0], dataset_split.x.shape[0])
    )

    visited = set(subgraph)
    fringe = set(subgraph)
    neighborhood = []
    for dist in range(1, 2):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        neighborhood.append(list(fringe))
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
    return torch.Tensor(list(itertools.chain.from_iterable(neighborhood)))


def set_labels(baseG):
    G = to_networkx(baseG, to_undirected=True)
    # nx.write_edgelist(G, "test.csv", delimiter=" ", data=False)

    components = number_connected_components(G)
    print(f"Number of connected components in dataset: {components}")
    y = []
    for idx, subgraph in enumerate(baseG.pos):
        # G = to_networkx(baseG, to_undirected=True)
        subgraph = torch.Tensor(list(filter(lambda node: node != -1, subgraph.tolist())))
        comp = extract_neighborhood(baseG, subgraph)
        draw_subgraph(baseG, subgraph, torch.unique(torch.cat((subgraph, comp))))
    #     k = G.subgraph(subgraph.tolist())
    #     subgraph_components = number_connected_components(k)
    #     subgraph_density = nx.density(k)
    #     subgraph_has_bridge = nx.has_bridges(k)
    #     subgraph_clustering = nx.average_clustering(k)
    #     cycles = nx.cycle_basis(k)
    #     subgraph_bridges = 0
    #     if subgraph_has_bridge:
    #         subgraph_bridges = 1
    #     subgraph_cycles = len(cycles)
    #     y.append([subgraph_density, subgraph_clustering, subgraph_components, subgraph_bridges, subgraph_cycles])
    #     # print(subgraph_components)
    # labels = torch.Tensor(y)
    return y


def draw_subgraph(G, subgraph, comp):
    G = to_networkx(G, to_undirected=True)
    k = G.subgraph(comp.tolist())
    pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
    plt.figure(figsize=(50, 50))
    color_map = ['red' if node in subgraph else 'blue' for node in comp]
    nx.draw_networkx(k, pos=pos, node_color=color_map)
    plt.show()
