"""Train GNNs to be explained."""
from argparse import ArgumentParser
from copy import deepcopy
from pickle import load

import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader

import gnn
import gnn_eigsearch


parser = ArgumentParser(description='Process some parameters.')
parser.add_argument('--name', type=str, required=True,
                    help='Dataset name (case-sensitive) as passed to pyg',
                    choices=["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"],)
parser.add_argument('--arch', type=str,
                    choices=['GCN', 'GIN', 'GAT', 'EIG'], help='GNN model architecutre', required=True)
parser.add_argument('--pool', type=str, default='add', choices=['add', 'mean', 'max'],
                    help='Graph pooling layer.')
parser.add_argument('--size', type=float, default=1.0,
                    help='Fraction of training data to be used.')
parser.add_argument('--seed', type=int, default=45,
                    help='Seed used for train-val-test split.')
args = parser.parse_args()


# * ----- Data
torch.manual_seed(args.seed)
FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"

TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]
if args.name in TU_DATASETS:
    dataset = pyg.datasets.TUDataset(
        root=f"../data", name=args.name)
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


# * ----- Model
if args.arch == "EIG":
    if args.name == "MUTAG":
        model = gnn_eigsearch.GIN(num_features=7, num_classes=2, num_layers=3, hidden=128)
    elif args.name == "Mutagenicity":
        model = gnn_eigsearch.GIN(num_features=14, num_classes=2, num_layers=3, hidden=64)
    elif args.name == "NCI1":
        model = gnn_eigsearch.GIN(num_features=37, num_classes=2, num_layers=3, hidden=64)
    else:
        print("Invalid dataset for eigsearch")
        exit(1)
else:
    model = eval(f"gnn.{args.arch.upper()}_{args.name}(pooling='{args.pool}')")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
device = "cpu"
model = model.to(device)


def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.arch.upper() == 'GCN' or args.arch.upper() == 'GIN':
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
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            if args.arch.upper() in ['GCN', 'GIN', 'EIG']:
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
        loss = criterion(out, data.y).item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        acc = correct / len(loader.dataset)
    return loss, acc


def plot_curves(losses_train, losses_val, accs_train, accs_val):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(range(len(losses_train)), losses_train, color="C0")
    axes[0].plot(range(len(losses_val)), losses_val, color="C1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(range(len(accs_train)), accs_train, color="C0", label="Train")
    axes[1].plot(range(len(accs_val)), accs_val, color="C1", label="Valid")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[1].legend()
    fig.savefig(f"{FOLDER}/training.png", dpi=150, bbox_inches="tight")


loss_val_best = float("inf")
acc_val_best = 0.0
epoch_best = 0
weights_best = None

patience = 100
warmup = 90
max_epochs = 1000

losses_train = []
losses_val = []
accs_train = []
accs_val = []

for epoch in range(max_epochs):
    train()
    loss_train, acc_train = test(train_loader)
    loss_val, acc_val = test(val_loader)

    losses_train.append(loss_train)
    losses_val.append(loss_val)
    accs_train.append(acc_train)
    accs_val.append(acc_val)

    print(f"Epoch {epoch:03d}"
            f" | Loss train: {loss_train:.4f}, Loss val: {loss_val:.4f}"
            f" | Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")
    if epoch % 20 == 0:
        plot_curves(losses_train, losses_val, accs_train, accs_val)

    # store best results
    if (epoch > warmup) and (acc_val >= acc_val_best):
        epoch_best = epoch
        loss_val_best = loss_val
        acc_val_best = acc_val
        weights_best = deepcopy(model.state_dict())

    # check stopping criteria
    if (epoch > warmup) and ((epoch - epoch_best) > patience):
        print(f"\nStopping early at {epoch}")
        break


model.load_state_dict(weights_best)
torch.save(model.state_dict(), f"{FOLDER}/model.pt")

weights_best = f"{FOLDER}/model.pt"
model.load_state_dict(torch.load(weights_best))
model.eval()


__, test_acc = test(test_loader)
__, train_acc = test(train_loader)
__, val_acc = test(val_loader)
print()
print(f"Name:{args.name}, Architecture:{args.arch}, Seed:{args.seed}, Pooling:{args.pool}")
# print(f"Best epoch: {epoch_best}")
print(f"Train acc: {round(train_acc, 4)}")
print(f"Val acc: {round(val_acc, 4)}")
print(f"Test acc: {round(test_acc, 4)}")

with open(f"{FOLDER}/model_acc.log", "w") as file:
    file.write(FOLDER)
    file.write(f"\nTrain acc: {round(train_acc, 4)}")
    file.write(f"\nVal acc: {round(val_acc, 4)}")
    file.write(f"\nTest acc: {round(test_acc, 4)}")
