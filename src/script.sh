cpu=0

for dataset in "BAMultiShapesDataset" "NCI1" "MUTAG" "Mutagenicity" ; do
    # iterate over the seeds
    seeds=(45 357 796)
    if [ "$dataset" == "NCI1" ]; then
        seeds=(45 1225 1983)
    fi
    for seed in "${seeds[@]}"; do
        # iterate over the sizes
        command="python sample.py --name ${dataset} --seed ${seed} --size 0.25 --mode confidentSample"
        log="../logs/${dataset}_seed${seed}_size${size}.log"
        echo "$command &> $log"
        echo "$command" | xargs -P 1 -I CMD sh -c "taskset -c ${cpu} CMD" &
        ((cpu++))
    done
done
wait
