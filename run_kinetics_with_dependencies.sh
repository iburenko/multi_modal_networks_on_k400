#!/bin/bash

modality=$1
model_name=$2
lr=$3
residual_block=$4

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
        gpus_per_node=4
    fi
}

continue=0
get_batch_size
get_num_nodes
get_num_gpus_per_node
ntasks_per_node="$gpus_per_node"

echo modality "$modality"
echo model name "$model_name"
echo lr "$lr"
echo residual block "$residual_block"
echo train batch size "$train_bs"
echo val batch size "$val_bs"
echo num nodes "$num_nodes"
echo num gpus per node "$gpus_per_node"

master_job_id=0
sbatch_output=$(sbatch --nodes $num_nodes \
    --gres gpu:"$gpus_per_node" \
    --ntasks-per-node "$ntasks_per_node" \
    --exclude c77 \
    --job-name train_on_kinetics_model_"$model_name"_modality_"$modality"_lr_"$lr"_res_block_"$residual_block" \
    run_kinetics_train_script.sh "$modality" "$model_name" "$lr" "$train_bs" "$val_bs" "$num_nodes" "$continue" "$master_job_id" "$residual_block")

dependency_job_id=$(echo "$sbatch_output" | awk '{print $4}')
master_job_id="$dependency_job_id"

echo $sbatch_output
echo "job id = "$dependency_job_id

continue=1
num_iters=1

echo now continue is $continue

for iter in $(seq "$num_iters"); 
do  
    echo iter num = "$iter"
    
    sbatch_output=$(sbatch --nodes $num_nodes \
        --gres gpu:"$gpus_per_node" \
        --ntasks-per-node "$ntasks_per_node" \
        --exclude c77 \
        --job-name train_on_kinetics_model_"$model_name"_modality_"$modality"_lr_"$lr"_res_block_"$residual_block" \
        --dependency=afterany:${dependency_job_id} \
        run_kinetics_train_script.sh "$modality" "$model_name" "$lr" "$train_bs" "$val_bs" "$num_nodes" "$continue" "$master_job_id" "$residual_block")
    
    dependency_job_id=$(echo "$sbatch_output" | awk '{print $4}')
    
    echo $sbatch_output
    echo "job id = "$dependency_job_id
done