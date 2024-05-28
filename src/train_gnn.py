"""Train GNNs to be explained."""
from argparse import ArgumentParser
from copy import deepcopy
from pickle import load

import torch
import torch_geometric as pyg
from torch_geometric.loader import DataLoader

import gnn


parser = ArgumentParser(description='Process some parameters.')
parser.add_argument('--name', type=str, default="MUTAG",
                    help='Name parameter of pyg dataset', required=True)
parser.add_argument('--arch', type=str,
                    choices=['GCN', 'GIN', 'GAT'], help='GNN model architecutre', required=True)
parser.add_argument('--pool', type=str, default='add', help='Model parameter')
parser.add_argument('--size', type=float, default=1.0, help='Size parameter')
parser.add_argument('--seed', type=int, default=45, help='Seed parameter')
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
model = eval(f"gnn.{args.arch.upper()}_{args.name}(pooling='{args.pool}')")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
device = "cpu"
model = model.to(device)


def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
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
        optimizer.zero_grad()


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
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
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


best_acc = 0
best_state_dict = model.state_dict()
best_epoch = 0
for epoch in range(1, 1001):
    if epoch-best_epoch > 100 and epoch > 100:
        print("early stopped")
        break
    train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    if val_acc > best_acc:
        best_acc = val_acc
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = epoch
        print("New model saved.")
    print(f'#{epoch:03d} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}')
print(f"Best val acc: {best_acc:.4f}")
model.load_state_dict(best_state_dict)
torch.save(model.state_dict(), f"{FOLDER}/model.pt")

test_acc = test(test_loader)
train_acc = test(train_loader)
val_acc = test(val_loader)
print(f"{args.name} {args.arch.upper()} {args.seed} {args.pool}")
print(f"Test acc: {round(test_acc, 4)}")
print(f"Train acc: {round(train_acc,4)}")
print(f"Val acc: {round(val_acc,4)}")
