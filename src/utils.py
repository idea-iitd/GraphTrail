"""Helper functions"""
from tqdm import tqdm
import torch_geometric as pyg
import torch
import networkx as nx
from sympy.logic.boolalg import Or, And, Not
from sympy.parsing.sympy_parser import parse_expr


def preprocess_dataset(dataset):
    dataset_ = []
    for graph in dataset:
        try:
            edge_attr=graph.edge_attr.argmax(dim=1)
        except:
            edge_attr = torch.ones(graph.edge_index.shape[1])
        data = pyg.data.Data(
            x=graph.x.argmax(dim=1),
            edge_index=graph.edge_index,
            y=graph.y,
            edge_attr=edge_attr,
        )
        dataset_.append(data)
    return dataset_

def graph_from_dfs_code(dfs_code):
    G = nx.DiGraph()
    dfs_code = dfs_code.split(" ")
    expec_root = True
    id = 0
    par = {}
    curr = -1
    par[-1] = -2
    i = 0
    while(i < len(dfs_code)):
        ch = dfs_code[i]
        if(expec_root):
            G.add_node(id, attr=int(ch))
            curr = id
            expec_root = False
            par[id] = id
            id += 1
            i += 1
        else:
            if(ch == '$'):
                curr = par[curr]
                i += 1
                continue
            ch_nxt = dfs_code[i+1]
            par[id] = curr
            G.add_node(id, attr=int(ch_nxt))
            G.add_edge(curr, id, attr=int(ch))
            curr = id
            id += 1
            i += 2
    return G.reverse()

def int_to_onehot(attr: int, num_features: int):
    one_hot = [0 for __ in range(num_features)]
    one_hot[attr] = 1.
    return one_hot

def nx_to_pyg(
        ctree: nx.digraph,
        num_node_features: int,
        num_edge_features: int
) -> pyg.data.Data:
    for node in ctree.nodes:
        ctree.nodes[node]["attr"] = int_to_onehot(
            ctree.nodes[node]["attr"], num_node_features
        )
    for edges in ctree.edges:
        ctree.edges[edges]["attr"] = int_to_onehot(
            ctree.edges[edges]["attr"], num_edge_features
        )
    if len(ctree.edges) == 0:
        graph_pyg = pyg.utils.from_networkx(
            ctree, group_node_attrs=["attr"], group_edge_attrs=None
        )
    else:
        graph_pyg = pyg.utils.from_networkx(
            ctree, group_node_attrs=["attr"], group_edge_attrs=["attr"]
        )

    return graph_pyg

def simplify_expression(str_exp):
    expression = parse_expr(str_exp)
    simplified_expr = expression.simplify()
    return simplified_expr

def getVariables(expr):
    return expr.atoms()

def _label_from_feat(feat):
    """Convert a node/edge feature to a single integer label.

    PyG graphs often store one-hot vectors; downstream code expects one
    integer ``attr``. Handles None, scalars, tensors (via argmax), and lists
    produced by ``to_networkx``.
    """
    if feat is None:
        return 0
    if isinstance(feat, (int, float)):
        return int(feat)
    if hasattr(feat, "argmax"):
        return int(feat.argmax())
    if isinstance(feat, list):
        return feat.index(max(feat))
    return int(feat)

def _normalize_graph_attrs(G):
    """Rewrite a ``to_networkx`` graph to use integer ``attr`` on nodes/edges.

    Replaces ``x`` and ``edge_attr`` fields with ``attr`` via
    ``_label_from_feat``, which isomorphism checks and plotting rely on.
    """
    for node in G.nodes:
        x = G.nodes[node].pop("x", None)
        if x is not None:
            G.nodes[node]["attr"] = _label_from_feat(x)
    for u, v in G.edges:
        edge_attr = G.edges[u, v].pop("edge_attr", None)
        if edge_attr is not None:
            G.edges[u, v]["attr"] = _label_from_feat(edge_attr)
        elif "attr" not in G.edges[u, v]:
            G.edges[u, v]["attr"] = 0
    return G

def pyg_to_nx(data):
    """Convert a PyG ``Data`` object to an undirected NetworkX graph.

    Uses ``pyg.utils.to_networkx``, then normalizes node/edge features to
    integer ``attr`` labels for subgraph comparison and visualization.
    """
    node_attrs = ["x"] if data.x is not None else None
    edge_attrs = ["edge_attr"] if getattr(data, "edge_attr", None) is not None else None
    G = pyg.utils.to_networkx(
        data,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        to_undirected=True,
    )
    return _normalize_graph_attrs(G)

def extract_l_hop_neighborhood(data, center, hops):
    subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(
        center,
        hops,
        data.edge_index,
        num_nodes=data.num_nodes,
        relabel_nodes=True,
    )
    sub_kwargs = {"x": data.x[subset], "edge_index": edge_index}
    if getattr(data, "edge_attr", None) is not None:
        sub_kwargs["edge_attr"] = data.edge_attr[edge_mask]
    return pyg_to_nx(pyg.data.Data(**sub_kwargs))

def most_frequent_isomorph(subgraphs):
    """Return the most common L-hop neighborhood for a ctree.

    Clusters graphs by ``nx.is_isomorphic`` (matching node/edge ``attr``),
    then returns a representative from the largest cluster and its size.
    Used when many training nodes produce the same ctree but different
    local subgraph shapes.
    """
    if not subgraphs:
        raise ValueError("No subgraphs to compare")
    clusters = []
    for g in subgraphs:
        matched = False
        for cluster in clusters:
            if nx.is_isomorphic(
                g,
                cluster[0],
                node_match=lambda a, b: a.get("attr") == b.get("attr"),
                edge_match=lambda a, b: a.get("attr") == b.get("attr"),
            ):
                cluster.append(g)
                matched = True
                break
        if not matched:
            clusters.append([g])
    clusters.sort(key=len, reverse=True)
    return clusters[0][0], len(clusters[0])

def get_node_mapping(dataset_name):
    if dataset_name == "MUTAG":
        return {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    if dataset_name == "Mutagenicity":
        return {
            0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br",
            7: "S", 8: "P", 9: "I", 10: "Na", 11: "K", 12: "Li", 13: "Ca",
        }
    return None

def plot_subgraph(G, path, dataset_name, title=None):
    import matplotlib.pyplot as plt

    TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
    node_mapping = get_node_mapping(dataset_name)
    edge_colors = None
    if dataset_name in ["MUTAG", "Mutagenicity"]:
        colors = ["green", "black", "blue", "red"]
        edge_colors = [colors[G.edges[e]["attr"]] for e in G.edges()]

    labeldict = None
    if node_mapping is not None:
        labeldict = {i: node_mapping[G.nodes[i]["attr"]] for i in G.nodes}
    elif dataset_name == "NCI1":
        labeldict = {i: G.nodes[i]["attr"] for i in G.nodes}

    plt.figure(figsize=(8, 6))
    if title is not None:
        plt.title(title)
    if dataset_name in TU_DATASETS:
        nx.draw_kamada_kawai(
            G,
            labels=labeldict,
            with_labels=labeldict is not None,
            node_color="#FD5D02",
            width=1.75,
            edge_color=edge_colors,
        )
    else:
        nx.draw_kamada_kawai(
            G,
            node_color="#FD5D02",
            width=1.75,
        )
    plt.savefig(path)
    plt.close()
