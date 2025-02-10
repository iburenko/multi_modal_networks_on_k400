import torch
import torch.nn.functional as F
import lightning as L
from pytorch_lightning.utilities import rank_zero_only
import torchmetrics as tm
from einops import rearrange

from io_utils import print_current_time
from backbones import get_model


class R3D(L.LightningModule):
    def __init__(self, conf):
        super().__init__()
        
        assert conf is not None
        def set_val(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    setattr(self, key, set_val(val))
                else:
                    setattr(self, key, val)
        set_val(conf)

        self.save_hyperparameters(conf)

        models = get_model(
            self.modality,
            self.model_name,
            self.T,
            self.residual_block,
            pretrained=self.pretrained,
            scale_invariant=self.scale_invariant
        )
        self.video_model, self.audio_model = models
        
        if self.modality == "rgb_audio":
            audio_feat_dim = 2048
            if self.model_name in ["vivit", "video_mae"]:
                video_feat_dim = 768
            elif self.model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
                if self.residual_block == "bottleneck":
                    video_feat_dim = 2048
                else:
                    video_feat_dim = 512
            if self.scale_invariant:
                norm_layer = torch.nn.BatchNorm1d(video_feat_dim, affine=False)
            else:
                norm_layer = torch.nn.Identity()
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(audio_feat_dim + video_feat_dim, video_feat_dim, bias=False),
                norm_layer,
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(video_feat_dim, video_feat_dim, bias=False),
                norm_layer,
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(video_feat_dim, self.num_classes, bias=False)
            )
        else:
            self.fusion = torch.nn.Identity()
        
        if self.scale_invariant:
            if self.modality == "audio":
                W = self.audio_model.fc.weight.data
                self.audio_model.fc.weight.data.copy_(self.linear_norm * W / W.norm())
                for p in self.audio_model.fc.parameters():
                    p.requires_grad = False
            elif self.modality == "rgb":
                W = self.video_model.linear.weight.data
                self.video_model.linear.weight.data.copy_(self.linear_norm * W / W.norm())
                for p in self.video_model.linear.parameters():
                    p.requires_grad = False
            else:
                W = self.fusion[-1].weight.data
                self.fusion[-1].weight.data.copy_(self.linear_norm * W / W.norm())
                for p in self.fusion[-1:].parameters():
                    p.requires_grad = False

        self.train_accuracy = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes)

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
        _, pred = logits.max(1)
        loss = F.cross_entropy(logits, labels)
        self.train_accuracy(pred, labels)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        self.val_accuracy(pred, labels)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True)

    @rank_zero_only
    def on_train_epoch_start(self):
        print_current_time()
        print(f"Current training epoch {self.trainer.current_epoch} starts!", flush=True)

    @rank_zero_only
    def on_train_epoch_end(self):
        print_current_time()
        print(f"Current training epoch {self.trainer.current_epoch} finishes!", flush=True)

    @rank_zero_only
    def on_validation_epoch_start(self):
        print_current_time()
        print(f"Current validation epoch {self.trainer.current_epoch} starts!", flush=True)
    
    @rank_zero_only
    def on_validation_epoch_end(self):
        print_current_time()
        print(f"Current validation epoch {self.trainer.current_epoch} finishes!", flush=True)

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
            if self.modality == "audio":
                first_milestone = 10
                second_milestone = 30
            else:
                first_milestone = 15
                second_milestone = 40
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=[first_milestone, second_milestone],
                verbose=True
                )
        return [optimizer], [scheduler]