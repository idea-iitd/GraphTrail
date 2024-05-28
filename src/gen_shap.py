"""Compute the Shapley values of the computation trees identified in gen_ctree.py"""
from argparse import ArgumentParser
import multiprocess as mp
from pickle import load, dump

import numpy as np
import shap
import torch

import gnn

parser = ArgumentParser(description='Process some parameters.')
parser.add_argument('--name', type=str,
                    help='Name parameter', required=True)
parser.add_argument('--arch', type=str,
                    choices=['GCN', 'GIN', 'GAT'], help='GNN model architecutre', required=True)
parser.add_argument('--pool', type=str, default='add',
                    help='Model parameter')
parser.add_argument('--size', type=float, default=1.0,
                    help='Size parameter')
parser.add_argument('--seed', type=int, default=45,
                    help='Seed parameter')
parser.add_argument('--procs', type=int, default=8, help='Number of processes to run in parallel.')
args = parser.parse_args()

FOLDER = f"../data/{args.name}/{args.arch}/{args.pool}/{args.size}/{args.seed}/"

with open(f"{FOLDER}/ctrees_embeddings.pkl", "rb") as file:
    ctree_embeddings = load(file)

with open(f"{FOLDER}/shap_ind_vec.pkl", "rb") as file:
    cnt_ind_vec_shap = load(file)

model = eval(f"gnn.{args.arch.upper()}_{args.name}(pooling='{args.pool}')")
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


def calculate_shap(shap_arr):
    z = np.zeros((1, shap_arr[0].shape[0]))
    explainer = shap.KernelExplainer(f, z)
    shap_values = explainer.shap_values(X=shap_arr, gc_collect=True)
    return shap_values


def main():
    import os
    if os.path.exists(f"{FOLDER}/shap_values.pkl"):
        print("Shap compuation already done")
        exit(0)
    chunk_size = len(cnt_ind_vec_shap) // args.procs
    remainder = len(cnt_ind_vec_shap) % args.procs

    if remainder:
        chunks = [cnt_ind_vec_shap[i *
                                   chunk_size: (i + 1) * chunk_size] for i in range(args.procs-1)]
        chunks.append(cnt_ind_vec_shap[(args.procs-1) * chunk_size:])
    else:
        chunks = [cnt_ind_vec_shap[i *
                                   chunk_size: (i + 1) * chunk_size] for i in range(args.procs)]

    with mp.Pool(args.procs) as p:
        results = p.map(calculate_shap, chunks)

    shap_values = np.concatenate(results, axis=0)

    with open(f"{FOLDER}/shap_values.pkl", "wb") as file:
        dump(shap_values, file)


if __name__ == '__main__':
    main()
