"""
Copy trained GNNs from size 1.0 to other sizes.
Since the training set remains the same for the GNN across all sizes, copy the one from 1.0.
"""
import os
import shutil


DATASETS = ["BAMultiShapesDataset", "MUTAG", "Mutagenicity", "NCI1"]
ARCHS = ["GAT", "GCN", "GIN"]
POOLS = ["add", "max", "mean"]
for dataset in DATASETS:
    for arch in ARCHS:
        for pool in POOLS:
            # Define the main directory and the target subdirectory
            main_directory = f"../data/{dataset}/{arch}/{pool}/"
            target_directory = "1.0"

            # Get the list of directories in the main directory
            directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

            # Loop through each directory
            for directory in directories:
                # Skip the target directory
                if directory == target_directory:
                    continue
                
                # Get the subdirectories of the target directory
                target_subdirectories = [os.path.join(main_directory, target_directory, sub) for sub in os.listdir(os.path.join(main_directory, target_directory)) if os.path.isdir(os.path.join(main_directory, target_directory, sub))]

                # Get the subdirectories of the current directory
                current_subdirectories = [os.path.join(main_directory, directory, sub) for sub in os.listdir(os.path.join(main_directory, directory)) if os.path.isdir(os.path.join(main_directory, directory, sub))]

                # Loop through each pair of subdirectories
                for target_subdir, current_subdir in zip(target_subdirectories, current_subdirectories):
                    # Get the list of files in the target subdirectory
                    files = [f for f in os.listdir(target_subdir) if os.path.isfile(os.path.join(target_subdir, f)) and "model" in f]

                    # Copy each file to the corresponding subdirectory
                    for file in files:
                        src_path = os.path.join(target_subdir, file)
                        dst_path = os.path.join(current_subdir, file)
                        shutil.copy(src_path, dst_path)
                        print(f"Copied {src_path:30} to {dst_path:30}")
                    print()
