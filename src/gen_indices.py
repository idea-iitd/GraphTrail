"""Generate training, validation, and test indices for all datasets."""
from copy import deepcopy
from os import makedirs
from pickle import dump

import torch
import torch_geometric as pyg
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
TU_DATASETS = ["MUTAG", "Mutagenicity", "NCI1"]

for name in ["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"]:
    if name == "NCI1":
        seeds = [45, 1225, 1983]
    else:
        seeds = [45, 357, 796]

    if name in TU_DATASETS:
        dataset = pyg.datasets.TUDataset(root=f"../data/", name=name)
    elif name == "BAMultiShapesDataset":
        dataset = pyg.datasets.BAMultiShapesDataset(root=f"../data/{name}")

    for seed in seeds:
        # train-test split
        train_indices, test_indices = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            train_size=0.8,
            random_state=seed,
            shuffle=True,
            stratify=dataset.y
        )
        # train-val split
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=0.125,
            train_size=0.875,
            random_state=seed,
            shuffle=True,
            stratify=dataset.y[train_indices]
        )
        train_indices_1 = deepcopy(train_indices)

        for size in [0.05, 0.25, 0.5, 0.75, 1.0]:
            if size == 1.0:
                train_indices_small = deepcopy(train_indices_1)
            else:
                train_indices_small, __ = train_test_split(
                    train_indices,
                    train_size=size,
                    random_state=seed,
                    shuffle=True,
                    stratify=dataset.y[train_indices]
                )
            for arch in ["GCN", "GIN", "GAT"]:
                for pool in ["add", "mean", "max"]:
                    FOLDER = f"../data/{name}/{arch}/{pool}/{size}/{seed}/"
                    makedirs(FOLDER, exist_ok=True)
                    with open(f"{FOLDER}/train_indices.pkl", "wb") as file:
                        dump(train_indices, file)
                    with open(f"{FOLDER}/test_indices.pkl", "wb") as file:
                        dump(test_indices, file)
                    with open(f"{FOLDER}/val_indices.pkl", "wb") as file:
                        dump(val_indices, file)
    print(f"Generated indices for {name}!")
