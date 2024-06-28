"""
This script creates a tmux session for each script.
Each script runs on one core.
Ignore the "can't find tmux..." message.
Run "bash kill_tmux.sh" to kill the generated tmux sessions.
"""
from subprocess import run

def manage_tmux_sessions(sessions: dict):
    """Create multiple tmux sessions and run commands in them."""
    for session_name, commands in sessions.items():
        # Kill old tmux session.
        run(["tmux", "kill-session", "-t", session_name])
        # Create a new tmux session.
        run(["tmux", "new-session", "-d", "-s", session_name])

        # Run the command in the tmux session.
        for command in commands:
            run(["tmux", "send-keys", "-t", session_name, command, "C-m"])


DATASETS = ["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"]
ARCHS = ["GAT", "GCN", "GIN"]
POOLS = ["add", "max", "mean"]
SIZES = [0.05, 0.25, 0.5, 0.75, 1.0]

# Use variables like this to customize.
DATASETS = ["NCI1"]
SIZES = [1.0]

sample = 1.0 # * Use ctrees and shapley values from this split of the training set.

CORES = 48
start_core = 48
core = start_core
sessions = {}
for dataset in DATASETS:
    for arch in ARCHS:
        for pool in POOLS:
            for size in SIZES:
                if dataset == "NCI1":
                    SEEDS = [45, 1225, 1983]
                else:
                    SEEDS = [45, 357, 796]
                for seed in SEEDS:
                    session_name = f"graphtrail_{dataset}_{arch}_{pool}_size{int(size * 100)}_seed{seed}"
                    log_file_prefix = f"../data/{dataset}/{arch}/{pool}/{size}/{seed}/"
                    conda = "conda activate GraphTrail"

                    # Training set remains the same for the GNN.
                    # Size reduces for the explainer.
                    train_gnn = f"taskset -c {core} python train_gnn.py --name {dataset}"\
                                f" --arch {arch} --pool {pool} --size 1.0 --seed {seed}"

                    gen_ctree = f"taskset -c {core} python gen_ctree.py --name {dataset}"\
                                f" --arch {arch} --pool {pool} --size {size} --seed {seed}"\
                                f"  > {log_file_prefix}/gen_ctree_1.log"\
                                f" 2> {log_file_prefix}/gen_ctree_2.log"
                    
                    # "gen_shap" requires multiple cores per instance. Therefore it isn't included
                    # here and has its own script "gen_shap.sh"

                    gen_formulae =  f"taskset -c {core} python gen_formulae.py"\
                                    f" --name {dataset} --arch {arch} --pool {pool}"\
                                    f" --size {size} --seed {seed} --sample {sample}"\
                                    f"  > {log_file_prefix}/gen_formulae_1_sample{sample}.log"\
                                    f" 2> {log_file_prefix}/gen_formulae_2_sample{sample}.log"

                    # * Change accordingly.
                    command = gen_formulae
                    sessions[session_name] = [conda, command]
                    core = start_core + ((core + 1) % CORES)
manage_tmux_sessions(sessions)
