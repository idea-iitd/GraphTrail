# Run a single instances of gen_shap.py across multiple cores one after the other.
datasets=("BAMultiShapesDataset" "MUTAG" "Mutagenicity" "NCI1")
archs=("GAT" "GCN" "GIN")
pools=("add" "max" "mean")
sizes=(0.05 0.25 0.5 0.75 1.0)

# Use variables like this to run one at a time.
datasets=("Mutagenicity")
sizes=(1.0)

for dataset in "${datasets[@]}"; do
    for arch in "${archs[@]}"; do
        for pool in "${pools[@]}"; do
            for size in "${sizes[@]}"; do

                if [[ $dataset == "NCI1" ]]
                then
                    seeds=(45 1225 1983)
                else
                    seeds=(45 357 796)
                fi

                for seed in "${seeds[@]}"; do
                    log_file_prefix="../data/${dataset}/${arch}/${pool}/${size}/${seed}"

                    gen_shap="taskset -c 0-71 python gen_shap.py\
                            --name ${dataset} --arch ${arch} --pool ${pool}\
                            --size ${size} --seed ${seed} --procs 72"

                    log_1="${log_file_prefix}/gen_shap_1.log"
                    log_2="${log_file_prefix}/gen_shap_2.log"

                    echo $gen_shap
                    { time $gen_shap ; } > $log_1 2> $log_2
                done
            done
        done
    done
done
