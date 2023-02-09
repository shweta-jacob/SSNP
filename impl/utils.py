import networkx as nx
import torch_geometric
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import scipy.sparse as ssp
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import k_hop_subgraph as org_k_hop_subgraph


def batch2pad(batch):
    '''
    The j-th element in batch vector is i if node j is in the i-th subgraph.
    The i-th row of pad matrix contains the nodes in the i-th subgraph.
    batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    '''
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni >= 0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    '''
    pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    '''
    batch = torch.arange(pad.shape[0])
    batch = batch.reshape(-1, 1)
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]
    batch = batch.to(pad.device).flatten()
    pos = pad.flatten()
    idx = pos >= 0
    return batch[idx], pos[idx]


@torch.jit.script
def MaxZOZ(x, pos):
    '''
    produce max-zero-one label
    x is node feature
    pos is a pad matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph.
    -1 is padding value.
    '''
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
    pos = pos.flatten()
    # pos[pos >= 0] removes -1 from pos
    tpos = pos[pos >= 0].to(z.device)
    z[tpos] = 1
    return z


def draw_graph(graph):
    # helps draw a graph object and save it as a png file
    f = plt.figure(1, figsize=(48, 48))
    nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph))
    plt.show()  # check if same as in the doc visually
    f.savefig("input_graph.pdf", bbox_inches='tight')


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


def k_hop_subgraph(center, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None, rw_kwargs=None):
    debug = False  # set True manually to debug using matplotlib and gephi
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    if not rw_kwargs:
        nodes = center
        dists = [0] * len(center)
        visited = set(center)
        fringe = set(center)
        for dist in range(1, num_hops + 1):
            if not directed:
                fringe = neighbors(fringe, A)
            else:
                out_neighbors = neighbors(fringe, A)
                in_neighbors = neighbors(fringe, A_csc, False)
                fringe = out_neighbors.union(in_neighbors)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if sample_ratio < 1.0:
                fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes + list(fringe)
            dists = dists + [dist] * len(fringe)

        subgraph = A[nodes, :][:, nodes]

        # Remove target link between the subgraph.
        # subgraph[0, 1] = 0
        # subgraph[1, 0] = 0

        # if node_features is not None:
        #     node_features = node_features[nodes]

        ones = center.tolist()
        zeros = list(set(fringe) - set(center.tolist()))

        sub_nodes_arranged = ones + zeros
        node_features = node_features[sub_nodes_arranged] if hasattr(node_features, 'size') else None
        one_label = torch.ones(size=[len(ones), ]).to(torch.long)
        zero_label = torch.zeros(size=[len(zeros), ]).to(torch.long)

        # Calculate node labeling.
        z_revised = torch.cat([one_label, zero_label], dim=0)

        return nodes, subgraph, dists, node_features, y, z_revised
    else:
        # Start of core-logic for S.C.A.L.E.D.
        rw_m = rw_kwargs['rw_m']
        rw_M = rw_kwargs['rw_M']
        sparse_adj = rw_kwargs['sparse_adj']
        edge_index = rw_kwargs['edge_index']
        device = rw_kwargs['device']
        data_org = rw_kwargs['data']

        if rw_kwargs.get('unique_nodes'):
            nodes = rw_kwargs.get('unique_nodes')[(center)]
        else:
            row, col, _ = sparse_adj.csr()
            starting_nodes = torch.tensor(center, dtype=torch.long, device=device)
            start = starting_nodes.repeat(rw_M)
            rw = torch.ops.torch_cluster.random_walk(row, col, start.cpu(), rw_m, 1, 1)[0]
            if debug:
                from networkx import write_gexf
                draw_graph(to_networkx(data_org))
                write_gexf(torch_geometric.utils.to_networkx(data_org), path='gephi.gexf')
            nodes = torch.unique(rw.flatten()).tolist()

        rw_set = nodes
        # import torch_geometric
        # edge_index_new, edge_attr_new = torch_geometric.utils.subgraph(subset=rw_set, edge_index=edge_index,
        #                                                                relabel_nodes=True)
        # subgraph api is same as org_k_hop_subgraph

        sub_nodes, sub_edge_index, mapping, _ = org_k_hop_subgraph(rw_set, 0, edge_index, relabel_nodes=True,
                                                                   num_nodes=data_org.num_nodes)

        # src_index = rw_set.index(src)
        # dst_index = rw_set.index(dst)
        # mapping_list = mapping.tolist()
        # src, dst = mapping_list[src_index], mapping_list[dst_index]
        # Remove target link from the subgraph.
        # mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        # mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        # sub_edge_index_revised = sub_edge_index[:, mask1 & mask2]

        ones = starting_nodes.tolist()
        zeros = list(set(rw_set) - set(starting_nodes.tolist()))
        # old_new_node_ids = {subnode: counter for counter, subnode in enumerate(sub_nodes.tolist())}
        # ones = [old_new_node_ids[node_id] for node_id in ones]
        # zeros = [old_new_node_ids[node_id] for node_id in zeros]

        sub_nodes_arranged = ones + zeros
        x = data_org.x[sub_nodes_arranged] if hasattr(data_org.x, 'size') else None
        one_label = torch.ones(size=[len(ones), ]).to(torch.long)
        zero_label = torch.zeros(size=[len(zeros), ]).to(torch.long)

        # Calculate node labeling.
        z_revised = torch.cat([one_label, zero_label], dim=0)
        data_revised = Data(x=x, z=z_revised,
                            edge_index=sub_edge_index, y=y, node_id=torch.LongTensor(rw_set),
                            num_nodes=len(rw_set), edge_weight=torch.ones(sub_edge_index.shape[-1]))
        # end of core-logic for S.C.A.L.E.D.
        return data_revised


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='zo', K=1):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    csr_subgraph = adj
    csr_shape = csr_subgraph.shape[0]
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    adj_t = SparseTensor(row=u, col=v,
                         sparse_sizes=(csr_shape, csr_shape))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    subgraph_features = node_features
    subgraph = adj_t

    assert subgraph_features is not None

    powers_of_a = [subgraph]
    for _ in range(K - 1):
        powers_of_a.append(subgraph @ powers_of_a[-1])

    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    if node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                node_id=node_ids, num_nodes=num_nodes)
    return data


def extract_enclosing_subgraphs(pos, A, x, y, num_hops, node_label='zo',
                                ratio_per_hop=1.0, max_nodes_per_hop=None,
                                directed=False, A_csc=None, rw_kwargs=None, edge_index=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []

    for idx, center in enumerate(tqdm(pos.tolist())):
        if not rw_kwargs['rw_m']:
            # tmp = k_hop_subgraph(list(filter(lambda pos: pos != -1, center)), num_hops, A, ratio_per_hop,
            #                      max_nodes_per_hop, node_features=x, y=y[idx],
            #                      directed=directed, A_csc=A_csc)

            subset, edge_index, inv, edge_mask = org_k_hop_subgraph(list(filter(lambda pos: pos != -1, center)),
                                                                    num_hops=1,
                                                                    edge_index=edge_index,
                                                                    num_nodes=x.shape[0])
            u, v = edge_index
            u, v = torch.LongTensor(u), torch.LongTensor(v)
            adj_t = SparseTensor(row=u, col=v,
                                 sparse_sizes=(x.shape[0], x.shape[0]))

            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

            subset = set(subset.tolist())
            ones = set(filter(lambda pos: pos != -1, center))
            zeros = subset.difference(ones)
            subgraph_nodes = list(ones) + list(zeros)

            subgraph_features = x[subgraph_nodes]
            subgraph = adj_t

            K = 3  # how many powers?

            powers_of_a = [subgraph]
            for _ in range(K - 1):
                powers_of_a.append(subgraph @ powers_of_a[-1])

            x_a = torch.cat([torch.ones(size=[len(ones), 1]), torch.zeros(size=[len(zeros), 1])])
            x_b = subgraph_features
            subg_x = torch.hstack([x_a, x_b])

            center_indices = [idx for idx in range(len(ones))]
            all_x = [subg_x[center_indices]]
            for index, power_of_a in enumerate(powers_of_a):
                all_x.append((power_of_a @ subg_x)[center_indices])

            x_revised = torch.cat(all_x, dim=-1)
            data = Data(x=x_revised, y=y[idx])

        else:
            data = k_hop_subgraph(list(filter(lambda pos: pos != -1, center)), num_hops, A, ratio_per_hop,
                                  max_nodes_per_hop, node_features=x, y=y[idx],
                                  directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
        draw = False
        if draw:
            draw_graph(to_networkx(data))
        data_list.append(data)

    return data_list
