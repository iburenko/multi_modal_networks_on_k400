#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=720Gb
#SBATCH --output=/data/cat/ws/ilbu282f-kinetics_dataset/experiments/slurm_logs/%x_%j.out
#SBATCH --account=p_scads_nlp

export HF_HOME=/data/horse/ws/ilbu282f-mm_landscape/hf_home

modality="$1"
model_name="$2"
train_bs="$3"
val_bs="$4"
num_nodes="$5"
continue="$6"
master_job_id="$7"
accumulate_batches="$8"

module purge
ml release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch-bundle/2.1.2-CUDA-12.1.1

. /data/cat/ws/ilbu282f-capella_gpu_py_env/bin/activate
echo $(which python)

srun python train_kinetics_with_warmup.py \
    --modality "$modality" \
    --model-name "$model_name" \
    --train-bs "$train_bs" \
    --val-bs "$val_bs" \
    --num-nodes "$num_nodes" \
    --continue-training "$continue" \
    --master-job-id "$master_job_id" \
    --accumulate-batches "$accumulate_batches"