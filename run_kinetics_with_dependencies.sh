#!/bin/bash

modality=$1
model_name=$2
accumulate_batches=$3

get_hostname () {
    hostname=$(echo $(hostname -f) | awk -F "." '{print $2}')
}

get_batch_size () {
    if [ "$modality" == "audio" ]; then
        train_bs=64
        val_bs=16
    else
        if [ "$model_name" == "vivit" ] || [ "$model_name" == "video_mae" ]; then
            train_bs=64
            val_bs=8
        else
            train_bs=32
            val_bs=4
        fi
    fi
    if [ "$batch_size_amplifier" == "" ]; then
        batch_size_amplifier=1
    fi
    train_bs=$(( train_bs*batch_size_amplifier ))
}

get_num_nodes () {
    if [ "$modality" == "audio" ]; then
        num_nodes=4
    else
        num_nodes=8
    fi
}

get_num_gpus_per_node () {
    if [ "$modality" == "audio" ]; then
        gpus_per_node=4
    else
        if [ "$hostname" == "alpha" ]; then
            gpus_per_node=8
        elif [ "$hostname" == "capella" ]; then
            gpus_per_node=4
        fi
    fi
}

get_accumulate_batches () {
    if [ "$accumulate_batches" == "" ]; then
        accumulate_batches=1
    fi
}

continue=0
get_hostname
get_accumulate_batches
get_batch_size
get_num_nodes
get_num_gpus_per_node
ntasks_per_node="$gpus_per_node"

echo hostname "$hostname"
echo modality "$modality"
echo model name "$model_name"
echo train batch size "$train_bs"
echo val batch size "$val_bs"
echo num nodes "$num_nodes"
echo num gpus per node "$gpus_per_node"
echo accumulate batches "$accumulate_batches"
echo batch size ampligier "$batch_size_amplifier"

master_job_id=0
sbatch_output=$(sbatch --nodes $num_nodes \
    --gres gpu:"$gpus_per_node" \
    --ntasks-per-node "$ntasks_per_node" \
    --exclude c147,c145 \
    --job-name train_on_kinetics_model_"$model_name"_modality_"$modality" \
    run_kinetics_train_script.sh \
        "$modality" \
        "$model_name" \
        "$train_bs" \
        "$val_bs" \
        "$num_nodes" \
        "$continue" \
        "$master_job_id" \
        "$accumulate_batches")

dependency_job_id=$(echo "$sbatch_output" | awk '{print $4}')
master_job_id="$dependency_job_id"

echo $sbatch_output
echo "job id = "$dependency_job_id

continue=1
num_iters=2

echo now continue is $continue

for iter in $(seq "$num_iters"); 
do  
    echo iter num = "$iter"
    
    sbatch_output=$(sbatch --nodes $num_nodes \
        --gres gpu:"$gpus_per_node" \
        --ntasks-per-node "$ntasks_per_node" \
        --exclude c147,c145 \
        --job-name train_on_kinetics_model_"$model_name"_modality_"$modality" \
        --dependency=afterany:${dependency_job_id} \
        run_kinetics_train_script.sh \
            "$modality" \
            "$model_name" \
            "$train_bs" \
            "$val_bs" \
            "$num_nodes" \
            "$continue" \
            "$master_job_id" \
            "$accumulate_batches")
    
    dependency_job_id=$(echo "$sbatch_output" | awk '{print $4}')
    
    echo $sbatch_output
    echo "job id = "$dependency_job_id
done