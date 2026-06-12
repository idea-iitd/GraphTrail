# GraphTrail: Translating GNN Predictions into Human-Interpretable Logical Rules

This is the official repository of [GraphTrail](https://openreview.net/pdf?id=fzlMza6dRZ) accepted at NeurIPS 2024.

To cite our work, use:
```
@inproceedings{armgaan2024graphtrail,
  title={GraphTrail: Translating GNN Predictions into Human-Interpretable Logical Rules},
  author={Burouj Armgaan and Manthan Dalmia and Sourav Medya and Sayan Ranu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=fzlMza6dRZ}
}
```

![GraphTrail's Pipeline](pipeline.png)

# Environment

`GraphTrail.yml` is a frozen GPU export (PyTorch 1.13.1, CUDA 11.7). Create the environment with:
```bash
conda env create -f GraphTrail.yml
conda activate GraphTrail

cd src/
rm -rf pygcanl/build/ pygcanl/pygcanl.egg-info/ pygcanl/*.so
pip install -e pygcanl --no-build-isolation
```

If `conda env create` fails, install step by step instead:
```bash
cd src/

conda create -n GraphTrail python=3.10 -y
conda activate GraphTrail
conda clean -a -y

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
conda install -c conda-forge shap multiprocess -y
conda install networkx matplotlib seaborn ipykernel ipywidgets -y

pip cache purge
pip install pysr

conda install conda-forge::boost gxx_linux-64 -y

# Pin MKL/NumPy to avoid PyTorch 1.13.1 import errors on newer conda defaults
conda install mkl=2023.1.0 intel-openmp=2023.1.0 "numpy<2" -y

rm -rf pygcanl/build/ pygcanl/pygcanl.egg-info/ pygcanl/*.so
pip install -e pygcanl --no-build-isolation
```

For a CPU-only setup, replace the PyTorch line with:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -y
```

# Run the code
```bash
cd src/

# Generate training, validation, and test indices for all datasets.
python gen_indices.py

# Train GNN if not already trained.
python train_gnn.py -h
python train_gnn.py ...

# Identify the unique computation trees and create the concept vectors.
python gen_ctree.py -h
python gen_ctree.py ...

# Compute the Shapley values of the computation trees identified in gen_ctree.py
python gen_shap.py -h
python gen_shap.py --name MUTAG --arch GIN --procs 8

# Parallel chunking (--procs): gen_shap.py splits the training indicator
# vectors into equal chunks and runs one SHAP KernelExplainer per chunk in
# parallel (multiprocess.Pool). Wall time scales roughly linearly with the
# number of processes, up to the number of training graphs, e.g. --procs 8
# is ~8x faster than the default --procs 1. For batch runs over many configs,
# see gen_shap.sh (uses --procs 72 on a 72-core machine).

# Generate formulae over the ctrees identified by gen_shap.py
# You will see some Julia installation on your first run.
python gen_formulae.py -h
python gen_formulae.py ...

# Map formula ctrees to training subgraphs and save plots.
python gen_subgraphs.py -h
python gen_subgraphs.py ...
```

## Example
```bash
cd src/

python gen_indices.py

python train_gnn.py --name MUTAG --arch GIN

python gen_ctree.py --name MUTAG --arch GIN

python gen_shap.py --name MUTAG --arch GIN --procs 8

python gen_formulae.py --name MUTAG --arch GIN

python gen_subgraphs.py --name MUTAG --arch GIN

# Shapley values are expensive; use --procs to parallelize across CPU cores.
python gen_shap.py --name MUTAG --arch GIN --procs 72
```

### Speeding up `gen_shap.py` with chunking

`gen_shap.py` splits the training indicator vectors into `--procs` chunks and runs one `shap.KernelExplainer` per chunk in parallel (`multiprocess.Pool`). With `--procs 1` (the default), all training graphs are explained sequentially.

Pass `--procs N` to process `N` chunks at once. Speedup is close to linear in `N`, up to the number of training graphs (each graph is assigned to exactly one chunk). For example, `--procs 72` on a 72-core machine launches 72 explainer jobs in parallel instead of one.

For batch runs over many dataset/arch/pool/size/seed combinations, use [`src/gen_shap.sh`](src/gen_shap.sh), which pins jobs to cores 0–71 and sets `--procs 72`:

```bash
cd src/
chmod +x gen_shap.sh
./gen_shap.sh
```

Timing for each run is written to `data/.../gen_shap_2.log` (stderr from `time`).

# Data
The code will generate some intermediate files and save them under the following directory structure:
```bash
data
├── BAMultiShapesDataset
│   ├── GAT
│   │   ├── add
│   │   │   ├── 0.05
│   │   │   │   ├── 357
│   │   │   │   │   ├── test_indices.pkl
│   │   │   │   │   ├── train_indices.pkl
│   │   │   │   │   └── val_indices.pkl
│   │   │   │   ├── 45
│   │   │   │   │   ├── test_indices.pkl
│   │   │   │   │   ├── train_indices.pkl
│   │   │   │   │   └── val_indices.pkl
│   │   │   │   └── 796
│   │   │   │       ├── test_indices.pkl
│   │   │   │       ├── train_indices.pkl
│   │   │   │       └── val_indices.pkl
```
