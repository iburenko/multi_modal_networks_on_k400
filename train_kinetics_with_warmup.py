from os import listdir as ls, environ, makedirs
from argparse import ArgumentParser
from pathlib import Path
import random
import yaml

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np

from model import R3D
from datamodule import KineticsDataModule
from conf_utils import put_args_into_yaml
from io_utils import (
    print_current_time,
    get_last_checkpoint, get_experiment_folder_name
)

torch.set_float32_matmul_precision("high")


def main(args):
    conf = yaml.safe_load(open("config.yaml"))
    seed = conf["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    conf = put_args_into_yaml(args, conf, job_id)
    job_id = environ["SLURM_JOB_ID"]
    modality = args.modality
    model_name = args.model_name
    num_nodes = args.num_nodes
    continue_training = args.continue_training
    master_job_id = args.master_job_id
    epochs = conf["optimization"]["epochs"]
    residual_block = conf["model"]["residual_block"]
    print("conf", conf)
    print("Start main!", flush=True)
    print_current_time()
    seed_everything(seed, workers=True)
    LitR3D = R3D(conf)
    kinetics_datamodule = KineticsDataModule(conf)
    
    save_dir = Path("/data/cat/ws/ilbu282f-kinetics_dataset/experiments/")
    model_name_with_res_block = f"{model_name}_{residual_block}"
    tb_log_dir = save_dir.joinpath("tb_logs", model_name_with_res_block, modality)
    ckpt_modality_model_path = save_dir.joinpath("checkpoints", model_name_with_res_block, modality)
    makedirs(ckpt_modality_model_path, exist_ok=True)
    experiment_folder = get_experiment_folder_name(job_id, master_job_id, ckpt_modality_model_path)
    ckpt_path = ckpt_modality_model_path.joinpath(experiment_folder)
    
    logger = TensorBoardLogger(tb_log_dir, name=experiment_folder)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=1,
        mode="max",
        monitor="val_acc",
        filename="k400_" + model_name_with_res_block + "_{epoch:02d}_{val_acc:.2f}_job_id_" + str(job_id)
        )
    if continue_training:
        last_ckpt = get_last_checkpoint(ckpt_path)
    else:
        last_ckpt = None

    trainer = L.Trainer(
        accelerator="cuda",
        devices=-1,
        num_nodes=num_nodes,
        num_sanity_val_steps=0,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=0,
        logger=logger,
        precision="bf16-mixed",
        accumulate_grad_batches=conf["optimization"]["accumulate_batches"]
        )
    trainer.fit(model=LitR3D, datamodule=kinetics_datamodule, ckpt_path=last_ckpt)

if __name__ == "__main__":
    print(environ["SLURM_JOB_ID"])
    print("start!", flush=True)
    print_current_time()
    print("available devices =", torch.cuda.device_count(), flush=True)
    parser = ArgumentParser()
    parser.add_argument("--modality", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--train-bs", type=int)
    parser.add_argument("--val-bs", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--continue-training", type=int)
    parser.add_argument("--master-job-id", type=int)
    parser.add_argument("--residual-block", type=str)
    args = parser.parse_args()
    print(args)
    main(args)