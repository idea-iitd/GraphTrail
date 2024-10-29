"""
This script creates a tmux session for each script.
Each script runs on one core.
Ignore the "can't find tmux..." message that you'll see upon running this script.
Run "bash kill_tmux.sh" to kill the generated tmux sessions.
"""
from subprocess import run
from torch_geometric import datasets


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


DATASETS = ["BAMultiShapesDataset", "NCI1", "MUTAG", "Mutagenicity"]
ARCHS = ["GAT", "GCN", "GIN"]
POOLS = ["add", "max", "mean"]
SIZES = [0.05, 0.25, 0.5, 0.75, 1.0]

# Use variables like this to customize.
DATASETS = ["NCI1"]
ARCHS = ["GIN"]
POOLS = ["mean", "max"]
SIZES = [0.5]
C = 1.0

# * Use ctrees and shapley values from this split of the training set.
# * If set to 1.0, it'll use the shapley values from the same training set as the SIZES.
samples = [1.0]

TOTAL_CORES = 96
START_CORE = 0
IN_HAND_CORES = TOTAL_CORES - START_CORE
core = START_CORE
counter = 0
sessions = {}
for sample in samples:
    for dataset in DATASETS:
        for arch in ARCHS:
            for pool in POOLS:
                for size in SIZES:
                    if sample != 1.0 and sample > size:
                        raise "Sample should not be greater than size."
                    if dataset == "NCI1":
                        SEEDS = [45, 1225, 1983]
                    else:
                        SEEDS = [45, 357, 796]
                    for seed in SEEDS:
                        session_name = f"graphtrail_{dataset}_{arch}_{pool}_size{int(size * 100)}"\
                                    f"_seed{seed}_sample{int(sample * 100)}"
                        log_file_prefix = f"../data/{dataset}/{arch}/{pool}/{size}/{seed}/"
                        conda = "conda activate GraphTrail"

                        # Training set remains the same for the GNN.
                        # Size reduces for the explainer.
                        train_gnn = f"taskset -c {core} python train_gnn.py --name {dataset}"\
                                    f" --arch {arch} --pool {pool} --size {size} --seed {seed}"

                        gen_ctree = f"taskset -c {core} python gen_ctree.py --name {dataset}"\
                                    f" --arch {arch} --pool {pool} --size {size} --seed {seed}"\
                                    f"  > {log_file_prefix}/gen_ctree_1.log"\
                                    f" 2> {log_file_prefix}/gen_ctree_2.log"
                        
                        gen_ctree_eig = f"taskset -c {core} python gen_ctree_eig.py --name {dataset}"\
                                    f" --arch {arch} --pool {pool} --size {size} --seed {seed}"\
                                    f"  > {log_file_prefix}/gen_ctree_1.log"\
                                    f" 2> {log_file_prefix}/gen_ctree_2.log"

                        # "gen_shap" requires multiple cores per instance. Therefore it isn't included
                        # here and has its own script "gen_shap.sh"

                        gen_formulae = f"taskset -c {core} python gen_formulae.py"\
                            f" --name {dataset} --arch {arch} --pool {pool}"\
                            f" --size {size} --seed {seed} --sample {sample} -c {C}"\
                            f"  > {log_file_prefix}/gen_formulae_1_sample{sample}.log"\
                            f" 2> {log_file_prefix}/gen_formulae_2_sample{sample}.log"
                        
                        gen_formulae_eig = f"taskset -c 0-32 python gen_formulae_eig.py"\
                            f" --name {dataset} --arch {arch} --pool {pool}"\
                            f" --size {size} --seed {seed} --sample {sample}"\
                            f"  > {log_file_prefix}/gen_formulae_1_sample{sample}.log"\
                            f" 2> {log_file_prefix}/gen_formulae_2_sample{sample}.log"

                        # * Change accordingly.
                        command = gen_formulae

                        sessions[session_name] = [f"{conda} && {command}"]
                        core = START_CORE + ((counter + 1) % IN_HAND_CORES)
                        counter += 1

manage_tmux_sessions(sessions)
