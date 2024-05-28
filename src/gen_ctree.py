"""Identify the unique computation trees and create the concept vectors."""
from argparse import ArgumentParser
from os import makedirs
from pickle import load, dump

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import gnn
import utils
from pygcanl import canonical


parser = ArgumentParser(description='Process some parameters.')
parser.add_argument('--name', type=str, help='Name parameter', required=True)
parser.add_argument('--arch', type=str,
                    choices=['GCN', 'GIN', 'GAT'], help='GNN model architecutre', required=True)
parser.add_argument('--pool', type=str, default='add',
                    help='Model parameter')
parser.add_argument('--size', type=float, default=1.0,
                    help='Size parameter')
parser.add_argument('--seed', type=int, default=45,
                    help='Seed parameter')
args = parser.parse_args()

torch.manual_seed(args.seed)
FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"
makedirs(FOLDER, exist_ok=True)


# * ----- Data
TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1", "PROTEINS", "Mutagenicity_noH"]
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


# * ----- Train indicator vectors
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

# Calculated in train, used directly in test.
unique_ctree_codes = list(dict_dfs_id_codes.keys())

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


# * ----- Test indicator vectors
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

test_ind_vec = []
for g_dict in graph_cnt_lst_test:
    temp = []
    for ct in unique_ctree_codes:
        if ct in g_dict:
            temp.append(g_dict[ct])
        else:
            temp.append(0)
    test_ind_vec.append(temp)
test_ind_vec = np.array(test_ind_vec)

model = eval(
    f"gnn.{args.arch.upper().upper()}_{args.name}(pooling='{args.pool}')")
model.eval()
try:
    model.load_state_dict(torch.load(f"{FOLDER}/model.pt", map_location="cpu"))
except FileNotFoundError:
    print("[ERROR] Couldn't find model weights.")
    exit(1)


# * ----- Embeddings
try:
    with open(f"{FOLDER}/ctrees_embeddings.pkl", "rb") as file:
        ctree_embeddings = load(file)
except:
    ctree_embeddings = []
    with torch.no_grad():
        for ct in tqdm(unique_ctree_codes):
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
    with open(f"{FOLDER}/ctrees_embeddings.pkl", "wb") as file:
        dump(ctree_embeddings, file)

with open(f"{FOLDER}/shap_ind_vec.pkl", "wb") as file:
    dump(cnt_ind_vec, file)
