from torch import nn
from torchvision.models.video.resnet import (
    BasicBlock, Bottleneck, 
    BasicStem, R2Plus1dStem,
    Conv3DSimple, Conv3DNoTemporal, 
    Conv2Plus1D,
    _video_resnet
)
from timm import create_model
from transformers import (
    VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEModel,
    VivitConfig, VivitForVideoClassification, VivitModel
)
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18

from resnet_si import make_resnet34k

def get_video_encoder(model_name, T, residual_block, num_classes, modality):
    if model_name in ["video_mae", "vivit"]:
        return get_transformer_encoder(model_name, T, num_classes, modality)
    elif model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
        return get_cnn_encoder(model_name, residual_block, num_classes)
    
def get_transformer_encoder(model_name, T, num_classes, modality):
    if model_name == "video_mae":
        if modality == "rgb_audio":
            num_classes = -1
            video_mae_config = VideoMAEConfig(num_labels=num_classes)
            video_mae = VideoMAEModel(video_mae_config)
        elif modality == "rgb":
            video_mae_config = VideoMAEConfig(num_labels=num_classes)
            video_mae = VideoMAEForVideoClassification(video_mae_config)
        else:
            raise ValueError("modality should be wither rgb_audio or rgb!")
        return video_mae
    elif model_name == "vivit":
        if modality == "rgb_audio":
            num_classes = -1
            use_mean_pooling = True
            vivit_config = VivitConfig(
                num_frames=T, 
                num_labels=num_classes, 
                use_mean_pooling=use_mean_pooling
                )
            vivit = VivitModel(vivit_config, add_pooling_layer=False)
        else:
            use_mean_pooling = False
            vivit_config = VivitConfig(num_frames=T, num_labels=num_classes, use_mean_pooling=use_mean_pooling)
            vivit = VivitForVideoClassification(vivit_config)
        return vivit

def get_cnn_encoder(model_name, residual_block, num_classes):
    if residual_block == "basic":
        res_block = BasicBlock
    elif residual_block == "bottleneck":
        res_block = Bottleneck
    if model_name == "r2plus1_18":
        video_encoder = r2plus1d_18()
    elif model_name == "r2plus1_34":
        video_encoder = _video_resnet(
            res_block,
            [Conv2Plus1D] * 4,
            [3, 4, 6, 3],
            R2Plus1dStem,
            weights=None,
            progress=False
        )
    elif model_name == "r2plus1_50":
        video_encoder = _video_resnet(
            res_block,
            [Conv2Plus1D] * 4,
            [3, 4, 6, 3],
            R2Plus1dStem,
            weights=None,
            progress=None
        )
    elif model_name == "mc3_18":
        video_encoder = mc3_18()
    elif model_name == "mc3_34":
        video_encoder = _video_resnet(
            res_block,
            [Conv3DSimple] + [Conv3DNoTemporal] * 3,
            [3, 4, 6, 3],
            BasicStem,
            weights=None,
            progress=False
        )
    elif model_name == "mc3_50":
        video_encoder = _video_resnet(
            res_block,
            [Conv3DSimple] + [Conv3DNoTemporal] * 3,
            [3, 4, 6, 3],
            BasicStem,
            weights=None,
            progress=False
        )
    elif model_name == "r3d_18":
        video_encoder = r3d_18()
    elif model_name == "r3d_34":
        video_encoder = _video_resnet(
            res_block,
            [Conv3DSimple] *4,
            [3, 4, 6, 3],
            BasicStem,
            weights=None,
            progress=False
        )
    elif model_name == "r3d_50":
        video_encoder = _video_resnet(
            res_block,
            [Conv3DSimple] * 4,
            [3, 4, 6, 3],
            BasicStem,
            weights=None,
            progress=False
        )
    elif model_name == "r3d_si_34":
        video_encoder = make_resnet34k(num_classes=num_classes)
    else:
        raise NotImplementedError("Choose a correct model name!")
    return video_encoder

def get_model(modality, model_name, T, residual_block, num_classes=400):
    video_encoder, audio_encoder = None, None
    if modality == "audio":
        audio_encoder = create_model("resnet50", in_chans=1, pretrained=False, num_classes=num_classes)
    elif modality == "rgb":
        video_encoder = get_video_encoder(model_name, T, residual_block, num_classes, modality)
    elif modality == "rgb_audio":
        audio_encoder = create_model("resnet50", in_chans=1, pretrained=False, num_classes=num_classes)
        audio_encoder = nn.Sequential(*list(audio_encoder.children())[:-1])
        video_encoder = get_video_encoder(model_name, T, residual_block, num_classes, modality)
        if model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
            video_encoder = nn.Sequential(*list(video_encoder.children())[:-1])
    return video_encoder, audio_encoder