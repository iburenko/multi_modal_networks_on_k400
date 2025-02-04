from os import listdir as ls, environ, makedirs
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import random
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics as tm
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info
from einops import rearrange
import numpy as np

from create_json_for_dataset import KineticDataset
from models import get_model
from conf_utils import put_args_into_yaml

torch.set_float32_matmul_precision("high")

def print_current_time():
    print(datetime.now().strftime("%H:%M:%S %Y-%m-%d"), flush=True)

def get_last_checkpoint(ckpt_path):
    model_name = sorted([elem for elem in ls(ckpt_path) if "val_acc" in elem])[-1]
    print("We use the last checkpoint",model_name, flush=True)
    return ckpt_path.joinpath(model_name)

def get_experiment_number(ckpt_path):
    all_experiments = sorted(ls(ckpt_path))
    if len(all_experiments) == 0:
        experiment_num = str(0).zfill(4)
    else:
        last_experiment = all_experiments[-1]
        last_experiment_num = int(last_experiment.split("_")[-1])
        experiment_num = str(last_experiment_num + 1).zfill(4)
    return experiment_num

def get_experiment_folder_name(job_id, master_job_id, ckpt_modality_model_path):
    if master_job_id == 0:
        experiment_number = get_experiment_number(ckpt_modality_model_path)
        folder_name = f"experiment_{job_id}_{experiment_number}"
    else:
        folder_name = [elem for elem in ls(ckpt_modality_model_path) if str(master_job_id) in elem][0]
    return folder_name

class R3D(L.LightningModule):
    def __init__(self, modality, models, lr, warmup, epochs, model_name, num_classes=400, conf=None):
        super().__init__()
        if args is not None:
            self.save_hyperparameters(conf)
            self.residual_block = conf["model"]["residual_block"]
        else:
            self.save_hyperparameters(ignode=["models"])
            self.residual_block = "bottleneck"
        self.modality = modality
        self.video_model, self.audio_model = models
        self.model_name = model_name
        self.lr = lr
        self.warmup = warmup
        self.epochs = epochs
        self.accuracy = tm.classification.Accuracy(task="multiclass", num_classes=num_classes)
        if self.modality == "rgb_audio":
            audio_feat_dim = 2048
            if model_name in ["vivit", "video_mae"]:
                video_feat_dim = 768
            elif model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
                if self.residual_block == "bottleneck":
                    video_feat_dim = 2048
                else:
                    video_feat_dim = 512
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(audio_feat_dim + video_feat_dim, video_feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(video_feat_dim, video_feat_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(video_feat_dim, num_classes)
            )
        else:
            self.fusion = torch.nn.Identity()

    def reshape_input(self, video, split):
        if split == "train":
            if self.model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
                return rearrange(video, 'b t c h w -> b c t h w')
        else:
            if self.model_name in ["video_mae", "vivit"]:
                return rearrange(video, 'b s t c h w -> (b s) t c h w')
            elif self.model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
                return rearrange(video, 'b s t c h w -> (b s) c t h w')
        
    def forward_video(self, video):
        if self.model_name in ["video_mae", "vivit"]:
            if self.modality == "rgb_audio":
                video_model_output = self.video_model(pixel_values=video, output_hidden_states=True)
                last_hidden_state = video_model_output.hidden_states[-1]
                if self.model_name == "vivit":
                    return last_hidden_state[:, 0]
                else:
                    return last_hidden_state.mean(1)
            else:
                video_model_output = self.video_model(pixel_values=video)
                return video_model_output.logits
        elif self.model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
            return self.video_model(video)

    def forward(self, video, audio):
        if self.modality == "audio":
            return self.audio_model(audio)
        elif self.modality == "rgb":
            return self.forward_video(video)
        elif self.modality == "rgb_audio":
            video_feats = self.forward_video(video)
            audio_feats = self.audio_model(audio)
            concated_feats = torch.hstack([video_feats.squeeze(), audio_feats.squeeze()])
            return self.fusion(concated_feats)

    def training_step(self, batch, batch_idx):
        video, audio, labels = batch
        if self.modality != "audio":
            video = self.reshape_input(video, "train")
        audio = audio.unsqueeze(1)
        logits = self(video, audio)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss.item(), on_step=True, sync_dist=True)
        # max_logits, pred = logits.max(1)
        # print(batch_idx, pred.eq(labels).sum().item(), loss.item())
        # print(max_logits)
        # print(pred)
        # print(labels)
        # print('+'*80)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # **************************************************
        # test with multiple snippets
        video, audio, labels = batch
        batch_size = video.shape[0]
        if self.modality != "audio":
            video = self.reshape_input(video, "validation")
        audio = rearrange(audio, 'b s h w -> (b s) h w')
        audio = audio.unsqueeze(1)
        logits = self(video, audio)
        logits = rearrange(logits, '(b s) d -> b s d', b=batch_size, s=10)
        logits = logits.mean(axis=1)
        _, pred = logits.max(1)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss.item(), on_step=True, sync_dist=True)
        self.accuracy(pred, labels)
        # print(batch_idx, pred.eq(labels).sum().item(), loss.item())
        # print(max_logits)
        # print(pred)
        # print(labels)
        # print('+'*80)
        # **************************************************

    @rank_zero_info
    def on_train_epoch_start(self):
        print_current_time()
        print(f"Current training epoch {self.trainer.current_epoch} starts!", flush=True)

    @rank_zero_info
    def on_train_epoch_end(self):
        print_current_time()
        print(f"Current training epoch {self.trainer.current_epoch} finishes!", flush=True)

    @rank_zero_info
    def on_validation_epoch_start(self):
        print_current_time()
        print(f"Current validation epoch {self.trainer.current_epoch} starts!", flush=True)
    
    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        self.log("val_acc", acc, sync_dist=True)
        print_current_time()
        print(f"Current validation epoch {self.trainer.current_epoch} finisehs!", flush=True)
        print("="*80, flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if self.warmup:
            warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=self.lr, total_iters=10, verbose=True
            )
            drop_down_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[(i + 2) * 10 for i in range(self.epochs // 10)],
                verbose=True
                )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [warm_up_scheduler, drop_down_scheduler]
            )
        else:
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, 
            #     milestones=[(i + 1) * 10 for i in range(self.epochs // 10)],
            #     verbose=True
            #     )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[15, 45],
                verbose=True
                )
        return [optimizer], [scheduler]

def main(args):
    conf = yaml.safe_load(open("config.yaml"))
    seed = conf["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    job_id = environ["SLURM_JOB_ID"]
    modality = args.modality
    model_name = args.model_name
    lr = args.lr
    warmup = conf["optimization"]["warmup"]
    epochs = conf["optimization"]["epochs"]
    train_bs = args.train_bs
    val_bs = args.val_bs
    num_nodes = args.num_nodes
    continue_training = args.continue_training
    master_job_id = args.master_job_id
    conf = put_args_into_yaml(args, conf, job_id)
    print("conf", conf)
    T = conf["model"]["T"]
    print("Start main!", flush=True)
    print_current_time()
    seed_everything(seed, workers=True)
    residual_block = conf["model"]["residual_block"]
    models = get_model(modality, model_name, T, residual_block)
    LitR3D = R3D(modality, models, lr, warmup, epochs, model_name, conf=conf)
    train_dataset = KineticDataset("train", modality, T=T)
    val_dataset = KineticDataset("val", modality, T=T)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_bs, 
        shuffle=False, 
        num_workers=conf["training"]["train_num_workers"], 
        pin_memory=True, 
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_bs, 
        shuffle=False,
        num_workers=conf["training"]["val_num_workers"],
        pin_memory=True, 
        drop_last=True
    )
    
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
    trainer.fit(model=LitR3D, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=last_ckpt)

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