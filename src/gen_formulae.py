"""Generate formulae over the ctrees identified by gen_shap.py"""
from pysrModel import PySrModelBoolean  # * Import this first!

import shutil
import warnings
from argparse import ArgumentParser
from os import makedirs
from pickle import load

import numpy as np
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
import networkx as nx
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import gnn
import utils
from pygcanl import canonical

sns.set()
sns.set_style("white")


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
PLOT_FOLDER = f"../plots/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"
makedirs(FOLDER, exist_ok=True)
makedirs(PLOT_FOLDER, exist_ok=True)


# * ----- Data
TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
if args.name in TU_DATASETS and args.name != "NCI1":
    dataset = pyg.datasets.TUDataset(root="../data/", name=args.name)
elif args.name == "BAMultiShapesDataset":
    dataset = pyg.datasets.BAMultiShapesDataset(root=f"../data/{args.name}")

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


# * ----- Computation trees
processed_dataset = utils.preprocess_dataset(
    [dataset[i] for i in train_indices])
list_of_dfs_id_codes = canonical(processed_dataset, 3)

dict_dfs_id_codes = {}
all_ctree_codes = []
graph_cnt_lst = []

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

model = eval(f"gnn.{args.arch}_{args.name}(pooling='{args.pool}')")
model.eval()
try:
    model.load_state_dict(torch.load(f"{FOLDER}/model.pt", map_location="cpu"))
except FileNotFoundError:
    print("[ERROR] Couldn't find model weights.")
    exit(1)


def predict(loader):
    model.eval()
    predictions = []
    probabilities = []
    for data in loader:
        data = data

        if args.arch == 'GCN' or args.arch == 'GIN':
            out, __, __ = model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        elif args.name in TU_DATASETS and args.name != "NCI1":
            out, __, __ = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch
            )
        else:
            out, __, __ = model(
                x=data.x,
                edge_index=data.edge_index,
                batch=data.batch
            )
        pred = out.argmax(dim=1)
        prob = torch.nn.functional.softmax(out, dim=1)
        predictions += pred.tolist()
        probabilities += prob.tolist()
    return predictions, probabilities


train_pred, train_prob = predict(train_loader)
test_pred, test_prob = predict(test_loader)

pysr_weights = []
for prob in train_prob:
    pysr_weights.append(max(prob))

with open(f"{FOLDER}shap_ind_vec.pkl", "rb") as file:
    cnt_ind_vec = load(file)

try:
    with open(f"{FOLDER}shap_values.pkl", "rb") as file:
        shap_values = load(file)
    shap_imp = np.abs(shap_values).mean(axis=0)
except FileNotFoundError:
    print("[ERROR] Couldn't find Shape Values.")
    exit(1)

indices = np.argsort(shap_imp)

x_train = cnt_ind_vec[:, indices[-200:]]
x_test = test_ind_vec[:, indices[-200:]]
x_train_bin = []
for lst in x_train:
    temp = []
    for i in lst:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)
    x_train_bin.append(temp)

x_test_bin = []
for lst in x_test:
    temp = []
    for i in lst:
        if i > 0:
            temp.append(1)
        else:
            temp.append(0)
    x_test_bin.append(temp)
warnings.filterwarnings('ignore')

shutil.rmtree('./HallOfFame', ignore_errors=True)
makedirs('./HallOfFame')
pysrmodelB = PySrModelBoolean(
    x_train_bin, x_test_bin, train_pred, test_pred, pysr_weights)
pysrmodelB.fit()
train_acc = round(pysrmodelB.get_train_acc(), 3)
test_acc = round(pysrmodelB.get_test_acc(), 3)
equation = pysrmodelB.model.get_best().equation
equation = utils.simplify_expression(equation)
print(f"{args.name} - {args.seed} - {args.size}")
print("TRAIN ACCURACY : ", train_acc)
print("TEST ACCURACY : ", test_acc)
print("EQUATION : ", equation)
variables_eq = utils.getVariables(equation)

del pysrmodelB


if args.name == "MUTAG":
    l = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
elif args.name == "Mutagenicity":
    l = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
         7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
elif args.name == "BAMultiShapesDataset":
    l = {1: '*'}

clr = ['g', 'k', 'r', 'skyblue']

for v in variables_eq:
    if str(v)[0] != "x":
        continue
    v = int(str(v)[1:])
    code_dfs = unique_ctree_codes[indices[-200:][v]]
    code_id = dict_dfs_id_codes[code_dfs]

    if args.name == 'BAMultiShapesDataset':
        G = utils.graph_from_dfs_code(code_id)
    else:
        G = utils.graph_from_dfs_code(code_dfs)

    G = G.reverse()

    if args.name != "NCI1":
        edge_colours = [clr[G.edges[edge]['attr']] for edge in G.edges()]
    labeldict = {}

    for i in range(len(G.nodes)):
        if args.name in TU_DATASETS and args.name != "NCI1":
            labeldict[i] = l[G.nodes[i]['attr']]
        elif args.name == "NCI1":
            labeldict[i] = G.nodes[i]['attr']
        else:
            labeldict[i] = '*'

    plt.figure(figsize=(6, 4))
    plt.title(v)

    if args.name == "NCI1":
        nx.draw_planar(G, labels=labeldict, with_labels=True,
                       node_color="#f2780d")
    elif args.name in TU_DATASETS:
        nx.draw_planar(G, labels=labeldict, with_labels=True,
                       node_color="#f2780d", edge_color=edge_colours, width=3)
    else:
        nx.draw_kamada_kawai(utils.dfs(G), node_color="#FD5D02",
                             width=1.75, style="solid")  # 9f67e5, #f2780d

    plt.savefig(f"{PLOT_FOLDER}/{v}_ctree.png")
    # ------------------------

    G = utils.graph_from_dfs_code(code_id)
    G = G.reverse()
    edge_colours = [clr[G.edges[edge]['attr']] for edge in G.edges()]

    labeldict = {}

    for i in range(len(G.nodes)):
        if args.name in TU_DATASETS and args.name != "NCI1":
            labeldict[i] = G.nodes[i]['attr']
        elif args.name == "NCI1":
            labeldict[i] = G.nodes[i]['attr']
        else:
            labeldict[i] = '*'

    plt.figure(figsize=(10, 5))
    plt.title(v)
    if args.name == "NCI1":
        nx.draw_planar(G, labels=labeldict, with_labels=True,
                       node_color="#f2780d")
    elif args.name in TU_DATASETS:
        nx.draw_planar(G, labels=labeldict, with_labels=True,
                       edge_color=edge_colours, width=3, node_color="#f2780d")
    else:
        nx.draw_kamada_kawai(utils.dfs(G), node_color="#f2780d",
                             width=1.75, style="dotted")  # 9f67e5, #f2780d

    plt.savefig(f"{PLOT_FOLDER}/{v}_ctree_id.png")
    # ------------------------

    G = utils.graph_from_dfs_code(code_id)
    G = G.reverse()

    labeldict = {}
    G = utils.dfs(G)
    edge_colours = [clr[G.edges[edge]['attr']] for edge in G.edges()]
    for i in G.nodes:
        if args.name in TU_DATASETS and args.name != "NCI1":
            labeldict[i] = G.nodes[i]['attr']
        elif args.name == "NCI1":
            labeldict[i] = G.nodes[i]['attr']
        else:
            labeldict[i] = '*'

    plt.figure(figsize=(7, 5))
    plt.title(v)
    if args.name == "NCI1":
        nx.draw_planar(G, labels=labeldict, with_labels=True,
                       node_color="#f2780d")
    elif args.name in TU_DATASETS:
        nx.draw_kamada_kawai(G, labels=labeldict, with_labels=True,
                             edge_color=edge_colours, width=3, node_color="#f2780d")
    else:
        nx.draw_kamada_kawai(utils.dfs(G), node_color="#f2780d",
                             width=1.75, style="dotted")  # 9f67e5, #f2780d

    plt.savefig(f"{PLOT_FOLDER}/{v}_structure.png")
