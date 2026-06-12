"""Map formula ctrees to their most frequent isomorphic L-hop training subgraphs."""
from argparse import ArgumentParser
from os import makedirs
from pickle import dump, load
from time import process_time

import torch_geometric as pyg

import utils

N_HOPS = 3

parser = ArgumentParser(description='Map formula ctrees to subgraphs and plot them.')
parser.add_argument('--name', type=str, required=True,
                    choices=["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"])
parser.add_argument('--arch', type=str, required=True, choices=['GCN', 'GIN', 'GAT'])
parser.add_argument('--pool', type=str, default='add', choices=['add', 'mean', 'max'])
parser.add_argument('--size', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=45)
parser.add_argument('-k', type=int, default=200)
parser.add_argument('-s', '--sample', type=float,
                    choices=[0.05, 0.25, 0.5, 0.75, 1.0], default=1.0)
parser.add_argument('--hops', type=int, default=N_HOPS,
                    help='L-hop neighborhood radius (must match gen_ctree N_HOPS)')
args = parser.parse_args()

FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"
PLOT_FOLDER = (
    f"../plots/{args.name}/{args.arch}/{args.pool}/{args.size}/"
    f"{args.seed}/sample{args.sample}/subgraphs"
)
makedirs(PLOT_FOLDER, exist_ok=True)

TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
if args.name in TU_DATASETS:
    dataset = pyg.datasets.TUDataset(root="../data/", name=args.name)
elif args.name == "BAMultiShapesDataset":
    dataset = pyg.datasets.BAMultiShapesDataset(root=f"../data/{args.name}")

with open(f"{FOLDER}/unique_ctree_codes.pkl", "rb") as file:
    unique_ctree_codes = load(file)
with open(f"{FOLDER}/ctree_node_sources.pkl", "rb") as file:
    ctree_node_sources = load(file)
with open(f"{FOLDER}/formula_meta_sample{args.sample}.pkl", "rb") as file:
    formula_meta = load(file)

start_time = process_time()

topk_indices = formula_meta["shap_indices"][-formula_meta["k"]:]
subgraph_meta = {}

for var_idx in formula_meta["formula_var_indices"]:
    ctree_idx = topk_indices[var_idx]
    dfs_code = unique_ctree_codes[ctree_idx]
    sources = ctree_node_sources.get(dfs_code, [])
    if not sources:
        print(f"x{var_idx}: no sources for ctree, skipping")
        continue

    neighborhoods = []
    graph_cache = {}
    for dataset_idx, node_idx in sources:
        if dataset_idx not in graph_cache:
            graph_cache[dataset_idx] = dataset[dataset_idx]
        neighborhoods.append(
            utils.extract_l_hop_neighborhood(
                graph_cache[dataset_idx], node_idx, args.hops
            )
        )

    best_subgraph, frequency = utils.most_frequent_isomorph(neighborhoods)
    plot_path = f"{PLOT_FOLDER}/{var_idx}_subgraph.png"
    utils.plot_subgraph(
        best_subgraph,
        plot_path,
        args.name,
        title=f"x{var_idx}",
    )
    subgraph_meta[var_idx] = {
        "dfs_code": dfs_code,
        "ctree_idx": ctree_idx,
        "frequency": frequency,
        "num_sources": len(sources),
    }
    print(
        f"x{var_idx}: {frequency}/{len(sources)} isomorph votes, "
        f"saved {plot_path}"
    )

with open(f"{FOLDER}/subgraph_meta_sample{args.sample}.pkl", "wb") as file:
    dump(subgraph_meta, file)

end_time = process_time()
print(f"[TIME] gen_subgraphs: {end_time - start_time} s")
