"""
1. Identify the unique computation trees.
2. Create the concept vector for each graph.
3. Compute the graph embedding of the unique computatation trees.
"""
from argparse import ArgumentParser
from os import makedirs
from pickle import load, dump
from time import process_time

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import gnn
import utils
from pygcanl import canonical


parser = ArgumentParser(description='Process some parameters.')
parser.add_argument('--name', type=str, required=True,
                    help='Dataset name (case-sensitive) as passed to pyg',
                    choices=["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"],)
parser.add_argument('--arch', type=str,
                    choices=['GCN', 'GIN', 'GAT'], help='GNN model architecutre', required=True)
parser.add_argument('--pool', type=str, default='add', choices=['add', 'mean', 'max'],
                    help='Graph pooling layer.')
parser.add_argument('--size', type=float, default=1.0,
                    help='Fraction of training data to be used.')
parser.add_argument('--seed', type=int, default=45,
                    help='Seed used for train-val-test split.')
args = parser.parse_args()

torch.manual_seed(args.seed)
FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"
# makedirs(FOLDER, exist_ok=True)


# * ----- Data
TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
if args.name in TU_DATASETS:
    dataset = pyg.datasets.TUDataset(root=f"../data/", name=args.name)
    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_EDGE_FEATURES = dataset.num_edge_features
    if args.name == "NCI1":
        NUM_EDGE_FEATURES = 2
elif args.name == "BAMultiShapesDataset":
    dataset = pyg.datasets.BAMultiShapesDataset(root=f"../data/{args.name}")
    NUM_NODE_FEATURES = 10
    NUM_EDGE_FEATURES = 2

with open(f"{FOLDER}/train_indices.pkl", "rb") as file:
    train_indices = load(file)
with open(f"{FOLDER}/test_indices.pkl", "rb") as file:
    test_indices = load(file)
with open(f"{FOLDER}/val_indices.pkl", "rb") as file:
    val_indices = load(file)

train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]
val_dataset = dataset[val_indices]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# * ----- Indicator vectors of training graphs
start_time = process_time()

processed_dataset = utils.preprocess_dataset(
    [dataset[i] for i in train_indices])
list_of_dfs_id_codes = canonical(processed_dataset, 3)

dict_dfs_id_codes = {}
all_ctree_codes = []
graph_cnt_lst = []

str1 = 'dfs_code \ id_code'

for l in list_of_dfs_id_codes:
    temp = []
    d = {}
    for s in l:
        key = s.split('/')[0]
        val = s.split('/')[1]
        temp.append(key)
        dict_dfs_id_codes[key] = val
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    graph_cnt_lst.append(d)
    all_ctree_codes.append(temp)

# Calculated in train, used directly in val, test.
unique_ctree_codes = list(dict_dfs_id_codes.keys())

with open(f"{FOLDER}/dict_dfs_id_codes.pkl", "wb") as file:
    dump(dict_dfs_id_codes, file)
with open(f"{FOLDER}/unique_ctree_codes.pkl", "wb") as file:
    dump(unique_ctree_codes, file)

cnt_ind_vec = []
for g_dict in graph_cnt_lst:
    temp = []
    for ct in unique_ctree_codes:
        if ct in g_dict:
            temp.append(g_dict[ct])
        else:
            temp.append(0)
    cnt_ind_vec.append(temp)
cnt_ind_vec = np.array(cnt_ind_vec)

with open(f"{FOLDER}/cnt_ind_vec.pkl", "wb") as file:
    dump(cnt_ind_vec, file)


# * ----- Indicator vectors of validation graphs
processed_dataset_val = utils.preprocess_dataset(
    [dataset[i] for i in val_indices])
list_of_dfs_id_codes_val = canonical(processed_dataset_val, 3)

dict_dfs_id_codes_val = {}
all_ctree_codes_val = []
graph_cnt_lst_val = []

for l in list_of_dfs_id_codes_val:
    temp = []
    d = {}
    for s in l:
        key = s.split('/')[0]
        val = s.split('/')[1]
        temp.append(key)
        dict_dfs_id_codes_val[key] = val
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    graph_cnt_lst_val.append(d)
    all_ctree_codes_val.append(temp)

cnt_ind_vec_val = []
for g_dict in graph_cnt_lst_val:
    temp = []
    for ct in unique_ctree_codes:
        if ct in g_dict:
            temp.append(g_dict[ct])
        else:
            temp.append(0)
    cnt_ind_vec_val.append(temp)
cnt_ind_vec_val = np.array(cnt_ind_vec_val)

with open(f"{FOLDER}/cnt_ind_vec_val.pkl", "wb") as file:
    dump(cnt_ind_vec_val, file)


# * ----- Indicator vectors of test graphs
processed_dataset_test = utils.preprocess_dataset(
    [dataset[i] for i in test_indices])
list_of_dfs_id_codes_test = canonical(processed_dataset_test, 3)

dict_dfs_id_codes_test = {}
all_ctree_codes_test = []
graph_cnt_lst_test = []

for l in list_of_dfs_id_codes_test:
    temp = []
    d = {}
    for s in l:
        key = s.split('/')[0]
        val = s.split('/')[1]
        temp.append(key)
        dict_dfs_id_codes_test[key] = val
        if key in d:
            d[key] += 1
        else:
            d[key] = 1
    graph_cnt_lst_test.append(d)
    all_ctree_codes_test.append(temp)

cnt_ind_vec_test = []
for g_dict in graph_cnt_lst_test:
    temp = []
    for ct in unique_ctree_codes:
        if ct in g_dict:
            temp.append(g_dict[ct])
        else:
            temp.append(0)
    cnt_ind_vec_test.append(temp)
cnt_ind_vec_test = np.array(cnt_ind_vec_test)

with open(f"{FOLDER}/cnt_ind_vec_test.pkl", "wb") as file:
    dump(cnt_ind_vec_test, file)


# * ----- Ctree Embeddings
model = eval(f"gnn.{args.arch.upper().upper()}_{args.name}(pooling='{args.pool}')")
model.load_state_dict(torch.load(f"{FOLDER}/model.pt", map_location="cpu"))
model.eval()

ctree_embeddings = []
with torch.no_grad():
    for ct in tqdm(unique_ctree_codes, desc="Ctree embeddings", colour="green"):
        ctree_nx = utils.graph_from_dfs_code(ct)
        ctree_pyg = utils.nx_to_pyg(
            ctree_nx,
            NUM_NODE_FEATURES,
            NUM_EDGE_FEATURES,
        )
        if args.arch.upper() == 'GCN' or args.arch.upper() == 'GIN':
            __, ctree_node_embs, __ = model(
                ctree_pyg.x,
                ctree_pyg.edge_index,
                batch=None
            )
        elif args.name in TU_DATASETS and args.name != "NCI1":
            __, ctree_node_embs, __ = model(
                ctree_pyg.x,
                ctree_pyg.edge_index,
                edge_attr=ctree_pyg.edge_attr,
                batch=None
            )
        else:
            __, ctree_node_embs, __ = model(
                ctree_pyg.x,
                ctree_pyg.edge_index,
                batch=None
            )

        ctree_embeddings.append(ctree_node_embs[0].tolist())

ctree_embeddings = np.array(ctree_embeddings)
with open(f"{FOLDER}/ctree_embeddings.pkl", "wb") as file:
    dump(ctree_embeddings, file)

end_time = process_time()
print(f"[TIME] gen_ctree: {end_time - start_time} s.ms")
