"""Generate formulae over the ctrees identified by gen_shap.py"""
from pysr import PySRRegressor # Import this first!

import warnings
from argparse import ArgumentParser
from os import makedirs
from pickle import dump, load
from shutil import rmtree
from time import process_time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader

import gnn
import utils

sns.set()
sns.set_style("white")
warnings.filterwarnings('ignore')

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
parser.add_argument('-k', type=int, default=200, help="k in top-k ctrees based on their shapley values")
parser.add_argument('-s', '--sample', type=float, choices=[0.05, 0.25, 0.5, 0.75, 1.0], default=1.0)
parser.add_argument('-c', type=float, default=1.0)
args = parser.parse_args()

torch.manual_seed(args.seed)
FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"
PLOT_FOLDER = f"../plots/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/sample{args.sample}"
makedirs(FOLDER, exist_ok=True)

rmtree(PLOT_FOLDER, ignore_errors=True)
makedirs(PLOT_FOLDER, exist_ok=True)


# * ----- Data
TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
if args.name in TU_DATASETS:
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

with open(f"{FOLDER}/cnt_ind_vec.pkl", "rb") as file:
    cnt_ind_vec = load(file)
with open(f"{FOLDER}/cnt_ind_vec_val.pkl", "rb") as file:
    cnt_ind_vec_val = load(file)
with open(f"{FOLDER}/cnt_ind_vec_test.pkl", "rb") as file:
    cnt_ind_vec_test = load(file)

if args.sample != 1.0:
    ORIG = FOLDER
    # replace args.size with args.sample
    FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.sample}/{args.seed}/"

with open(f"{FOLDER}/dict_dfs_id_codes.pkl", "rb") as file:
    dict_dfs_id_codes = load(file)
with open(f"{FOLDER}/unique_ctree_codes.pkl", "rb") as file:
    unique_ctree_codes = load(file)

with open(f"{FOLDER}shap_values.pkl", "rb") as file:
    shap_values = load(file)
    shap_imp = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(shap_imp)


if args.sample != 1.0:
    FOLDER = ORIG

model = eval(f"gnn.{args.arch}_{args.name}(pooling='{args.pool}')")
model.eval()
model.load_state_dict(torch.load(f"{FOLDER}/model.pt", map_location="cpu"))


def predict(loader):
    model.eval()
    predictions = []
    probabilities = []
    for data in loader:
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
val_pred, val_prob = predict(val_loader)
test_pred, test_prob = predict(test_loader)

pysr_weights = [max(prob) for prob in train_prob]

x_train = cnt_ind_vec[:, indices[- args.k:]]
x_val = cnt_ind_vec_val[:, indices[- args.k:]]
x_test = cnt_ind_vec_test[:, indices[- args.k:]]

x_train_bin = []
x_val_bin = []
x_test_bin = []
for set_ in ["train", "val", "test"]:
    for lst in eval(f"x_{set_}"):
        temp = []
        for i in lst:
            if i > 0:
                temp.append(1)
            else:
                temp.append(0)
        eval(f"x_{set_}_bin.append(temp)")


# * ----- Symbolic Regression
start_time = process_time()

pysrmodel = PySRRegressor(
    unary_operators = ["Not(x) = (x <= zero(x)) * one(x)"],
    binary_operators = [
        "And(x, y) = ((x > zero(x)) & (y > zero(y))) * one(x)",
        "Or(x, y)  = ((x > zero(x)) | (y > zero(y))) * one(x)",
        "Xor(x, y) = (((x > 0) & (y <= 0)) | ((x <= 0) & (y > 0))) * 1f0",
    ],
    extra_sympy_mappings = {
        "Not": lambda x: sympy.Piecewise((1.0, (x <= 0)), (0.0, True)),
        "And": lambda x, y: sympy.Piecewise((1.0, (x > 0) & (y > 0)), (0.0, True)),
        "Or":  lambda x, y: sympy.Piecewise((1.0, (x > 0) | (y > 0)), (0.0, True)),
        "Xor": lambda x, y: sympy.Piecewise((1.0, (x > 0) ^ (y > 0)), (0.0, True)),
    },

    elementwise_loss = "loss(prediction, target) = sum(prediction != target)",
    model_selection="accuracy",

    complexity_of_variables=args.c,
    complexity_of_operators={'Not': args.c, 'And': args.c, 'Or': args.c, 'Xor': args.c},

    select_k_features = min(args.k, 10),
    weights = pysr_weights,

    batch_size = 32,

    # Paperwork
    temp_equation_file = True,
    delete_tempfiles = True,

    # Determinism
    procs=0,
    deterministic=True,
    multithreading=False,
    random_state=0,
    warm_start=False,
)

def cal_pysr_acc(X, Y, index=None):
    Y = np.array(Y)
    Y_pred = pysrmodel.predict(X, index=index)
    assert Y.shape == Y_pred.shape , "Shape mismatch!"
    return (Y_pred == Y).sum() / len(Y)


pysrmodel.fit(x_train_bin, train_pred)
print(pysrmodel)

selected_ctrees = pysrmodel.selection_mask_

df_equations = pysrmodel.equations.drop(["sympy_format", "lambda_format"], axis=1)
# Add a column for accuracy.
df_equations["acc"] = 1 - df_equations["loss"]
# Re-arrange columns to have "acc" as the second column.
cols = df_equations.columns.tolist()
cols.insert(1, cols.pop(-1))
df_equations = df_equations[cols]
# Round values.
for col in ["acc", "loss", "score"]:
    df_equations[col] = df_equations[col].round(4)


# Find the equation that performs the best on the validation set
best_val_acc = 0
print("\nValidation accuracies:")
for j in range(pysrmodel.equations_.shape[0]):
    # PySR sometimes fails to evaluate certain formulae
    # it usually happens when C is set to a small value.
    # We've been unable to identify when and why it happens
    try:
        __ = pysrmodel.predict(x_train_bin, index=j)
        __ = pysrmodel.predict(x_test_bin, index=j)
        pysr_val_pred = pysrmodel.predict(x_val_bin, index=j)
    except ValueError:
        print(f"{j}: failed")
        continue
    val_acc = (pysr_val_pred == val_pred).sum() / len(val_pred)
    print(f"{j}: {val_acc}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_index = j
print("Best equation index:", best_index)


# * ----- Metrics
pysr_train_pred = pysrmodel.predict(x_train_bin, index=best_index)
pysr_test_pred = pysrmodel.predict(x_test_bin, index=best_index)
pysr_train_pred = torch.LongTensor(pysr_train_pred)
pysr_test_pred = torch.LongTensor(pysr_test_pred)


"""
# Best
equation = pysrmodel.get_best().equation
print(f"\nName:{args.name} - Seed:{args.seed} - Size:{args.size}")
print("=" * 50)
print("Equation:", equation)

train_acc = round(cal_pysr_acc(x_train_bin, train_pred), 3)
test_acc  = round(cal_pysr_acc(x_test_bin, test_pred), 3)

equation = utils.simplify_expression(equation)
print("Simplified equation:", equation)
print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
"""

# Best based on val set
equation = pysrmodel.get_best(index=best_index).equation
print()
print("=" * 50)
print("Equation:", equation)
print("C =", args.c)

train_acc = round(cal_pysr_acc(x_train_bin, train_pred, index=best_index), 3)
test_acc  = round(cal_pysr_acc(x_test_bin, test_pred, index=best_index), 3)

equation = utils.simplify_expression(equation)
print("Simplified equation:", equation)
print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)


# * ----- Save stuff to disk
# Save equations
df_equations.to_csv(f"{FOLDER}/equations_sample{args.sample}.csv", index=True)
del df_equations

# Save predictions
torch.save(torch.LongTensor(train_pred), f"{FOLDER}/gnn_train_pred.pt")
torch.save(torch.LongTensor(test_pred), f"{FOLDER}/gnn_test_pred.pt")
# torch.save(pysr_train_pred, f"{FOLDER}/pysr_train_pred_sample{args.sample}.pt")
torch.save(pysr_test_pred, f"{FOLDER}/pysr_test_pred_sample{args.sample}.pt")
del train_pred, test_pred, pysr_train_pred, pysr_test_pred

# Save pysrmodel
with open(f"{FOLDER}/pysrmodel_sample{args.sample}.pkl", "wb") as file:
    dump(pysrmodel, file)
del pysrmodel

end_time = process_time()
print(f"[TIME] gen_formulae: {end_time - start_time} s.ms")


# * ----- Visualize the computation trees present in the forumulae.
variables_eq = utils.getVariables(equation)

node_mapping = None
if args.name == "MUTAG":
    node_mapping = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
elif args.name == "Mutagenicity":
    node_mapping = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
                    7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}

FIGSIZE = (8, 6)
NODESIZE = ...
EDGE_WIDTH = 1.75
NODE_COLOR = "#FD5D02"
colors = ['green', 'black', 'blue', 'red'] # aromatic, single, double, triple

# for v in variables_eq:
#     if str(v)[0] != "x":
#         continue
#     v = int(str(v)[1:])
for v in selected_ctrees:
    code_dfs = unique_ctree_codes[indices[- args.k:][v]]
    code_id = dict_dfs_id_codes[code_dfs]

    # * ----- Ctree using node attributes
    if args.name == 'BAMultiShapesDataset':
        ctree = utils.graph_from_dfs_code(code_id)
    else:
        ctree = utils.graph_from_dfs_code(code_dfs)
    ctree = ctree.reverse()

    edge_colors = None
    if args.name in ["MUTAG", "Mutagenicity"]:
        edge_colors = [colors[ctree.edges[edge]['attr']] for edge in ctree.edges()]

    labeldict = {}
    for i in range(len(ctree.nodes)):
        if args.name in ["MUTAG", "Mutagenicity"]:
            labeldict[i] = node_mapping[ctree.nodes[i]['attr']]
        elif args.name == "NCI1":
            labeldict[i] = ctree.nodes[i]['attr']

    plt.figure(figsize=FIGSIZE)
    plt.title(v)
    if args.name == "NCI1":
        nx.draw_planar(
            ctree,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH
        )
    elif args.name in TU_DATASETS:
        nx.draw_planar(
            ctree,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH,
            edge_color=edge_colors,
        )
    else:
        nx.draw_planar(ctree, node_color=NODE_COLOR, width=EDGE_WIDTH)

    plt.savefig(f"{PLOT_FOLDER}/{v}_ctree.png")


    # * ----- Ctree using node ids
    ctree_id = utils.graph_from_dfs_code(code_id)
    ctree_id = ctree_id.reverse()

    labeldict = None
    if args.name in TU_DATASETS:
        labeldict = {}
        for i in range(len(ctree_id.nodes)):
            labeldict[i] = ctree_id.nodes[i]['attr']

    plt.figure(figsize=FIGSIZE)
    plt.title(v)
    if args.name == "NCI1":
        nx.draw_planar(
            ctree_id,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH,
        )
    elif args.name in TU_DATASETS:
        nx.draw_planar(
            ctree_id,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH,
            edge_color=edge_colors,
        )
    else:
        nx.draw_planar(ctree_id, node_color=NODE_COLOR, width=EDGE_WIDTH)

    plt.savefig(f"{PLOT_FOLDER}/{v}_ctree_id.png")


    # * ----- Ctree to subgraph
    G = utils.dfs(ctree=ctree, ctree_id=ctree_id, node_mapping=node_mapping)
    edge_colors = [colors[G.edges[edge]['attr']] for edge in G.edges()]

    labeldict = None
    if args.name in TU_DATASETS:
        labeldict = {}
        for i in G.nodes:
            labeldict[i] = G.nodes[i]['attr']

    plt.figure(figsize=FIGSIZE)
    plt.title(v)
    if args.name == "NCI1":
        nx.draw_kamada_kawai(
            G,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH,
        )
    elif args.name in TU_DATASETS:
        nx.draw_kamada_kawai(
            G,
            labels=labeldict,
            with_labels=True,
            node_color=NODE_COLOR,
            width=EDGE_WIDTH,
            edge_color=edge_colors,
        )
    else:
        nx.draw_kamada_kawai(G, node_color=NODE_COLOR, width=EDGE_WIDTH)

    plt.savefig(f"{PLOT_FOLDER}/{v}_structure.png")
