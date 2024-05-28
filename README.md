# GraphTrail
GrpahTrail: Translating GNN Predictions into Human-Interpretable Logical Rules

NOTE: All commands should be run from `src/`

# Environment
```bash
conda env create -f GraphTrail.yml
```

In case you have some issues with the above command, use the following instead:
```bash
cd src/

conda create -n GraphTrail -y
conda activate GraphTrail
conda clean -a -y

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -y
conda install pyg -c pyg -y
conda install -c conda-forge shap multiprocess -y
conda install networkx matplotlib seaborn ipykernel ipywidgets -y

pip cache purge
pip install pysr

conda install conda-forge::boost -y
conda install gxx_linux-64 -y

# if files and folder are present
rm -r pygcanl/build/
rm -r pygcanl.egg-info/
rm pygcanl/*.so
pip install -e pygcanl
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
pythopn gen_shap.py -h
python gen_shap.py ...

# Generate formulae over the ctrees identified by gen_shap.py
# You will see some Julia installation on your first run.
python gen_formulae.py -h
python gen_formulae.py ...
```

## Example
```python
cd src/

python gen_indices.py

python train_gnn.py --name MUTAG --arch GIN

python gen_ctree.py --name MUTAG --arch GIN

python gen_shap.py --name MUTAG --arch GIN

python gen_formula.py --name MUTAG --arch GIN
```

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
