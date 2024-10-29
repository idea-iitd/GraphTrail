"""Compute the Shapley values of the computation trees identified in gen_ctree.py"""
import multiprocess as mp
from argparse import ArgumentParser
from pickle import load, dump

import numpy as np
import shap
import torch

import gnn

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
parser.add_argument('--procs', type=int, default=1, help='Number of processes to run in parallel.')
args = parser.parse_args()

FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"

with open(f"{FOLDER}/ctree_embeddings.pkl", "rb") as file:
    ctree_embeddings = load(file)

with open(f"{FOLDER}/cnt_ind_vec.pkl", "rb") as file:
    cnt_ind_vec = load(file)

model = eval(f"gnn.{args.arch.upper()}_{args.name}(pooling='{args.pool}')")
model.load_state_dict(torch.load(f"{FOLDER}/model.pt", map_location="cpu"))
model.eval()


def get_graph_embedding(row):
    embd = np.zeros((ctree_embeddings.shape[1]))
    freq = 0
    max_embd = embd
    for i, cnt in enumerate(row):
        freq += cnt
        embd += cnt * ctree_embeddings[i]
        max_embd = np.maximum(max_embd, cnt * ctree_embeddings[i])

    if args.pool == 'add':
        return embd
    elif args.pool == 'mean':
        if freq == 0:
            return embd
        return embd/freq
    elif args.pool == 'max':
        return max_embd
    return embd


def f(ind_vectors):
    embedings = np.apply_along_axis(
        get_graph_embedding, axis=1, arr=ind_vectors)
    embedings = torch.Tensor(embedings)
    with torch.no_grad():
        model.eval()
        out = model.fc(embedings)
        # Probability of class 1
        out = torch.nn.functional.softmax(out, dim=1)[:, 1]
    return out.numpy()


def calculate_shap(shap_arr, chunk_num):
    z = np.zeros((1, shap_arr[0].shape[0]))
    explainer = shap.KernelExplainer(f, z)
    shap_values = explainer.shap_values(X=shap_arr, gc_collect=True, silent=True)
    print(f"Chunk {chunk_num} done!")
    return shap_values


args.procs = min(args.procs, len(cnt_ind_vec))
chunk_size = len(cnt_ind_vec) // args.procs
remainder  = len(cnt_ind_vec) %  args.procs

# Doesn't contain zero and len(cnt_ind_vec)
chunks = [chunk_size for __ in range(args.procs)]

# Distribute the remainder
for j in range(remainder):
    chunks[j] += 1

# Take the cumulative sum to get the indices.
indices = [0] + np.cumsum(chunks).tolist()

# Divide the indicator vectors into chunks.
chunked_ind_vectors = []
for i in range(len(indices) - 1):
    idx_start = indices[i]
    idx_end = indices[i + 1]
    chunked_ind_vectors.append(cnt_ind_vec[idx_start: idx_end])

print("Chunk size:", chunks)
print("#Chunks:", len(chunked_ind_vectors))
print()

with mp.Pool(args.procs) as p:
    results = p.starmap(calculate_shap, zip(chunked_ind_vectors, range(len(chunked_ind_vectors))))

shap_values = np.concatenate(results, axis=0)

with open(f"{FOLDER}/shap_values.pkl", "wb") as file:
    dump(shap_values, file)
