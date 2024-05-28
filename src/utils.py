"""Helper functions"""
from tqdm import tqdm
import torch_geometric as pyg
import torch
import networkx as nx
from sympy.logic.boolalg import Or, And, Not
from sympy.parsing.sympy_parser import parse_expr

def removeH(dataset):
    modified_dataset = []
    for dt in tqdm(dataset, colour = 'green'):
        graph_nx = pyg.utils.to_networkx(
            data= dt,
            to_undirected=True,
            node_attrs=["x"],
            edge_attrs=["edge_attr"],
        )
        # remove node in nx
        nodes_to_remove = [node for node, data in graph_nx.nodes(data=True) if np.argmax(data['x']) == 3]
        graph_nx.remove_nodes_from(nodes_to_remove)
        data = pyg.utils.from_networkx(graph_nx)
        data.y = dt.y
        modified_dataset.append(data)
        return modified_dataset

def preprocess_dataset(dataset):
    dataset_ = []
    for graph in dataset:
        try:
            edge_attr=graph.edge_attr.argmax(dim=1)
        except:
            edge_attr = torch.ones(graph.edge_index.shape[1])
        data = pyg.data.Data(
            x=graph.x.argmax(dim=1),
            id=torch.arange(graph.num_nodes),
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

def dfs(G):
    G_new = nx.Graph()
    for i in range(len(G.nodes)):
        G_new.add_node(G.nodes[i]['attr'], attr=G.nodes[i]['attr'])
    for e in G.edges:
        src,dest = e
        src = G.nodes[src]['attr']
        dest = G.nodes[dest]['attr']
        G_new.add_edge(src,dest, attr = G.edges[e]['attr'])
    return G_new
