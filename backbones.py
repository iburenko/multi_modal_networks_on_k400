from functools import partial

import torch
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

from resnet_si import make_resnet34k, make_resnet34k_3d

def get_video_encoder(model_name, T, residual_block, num_classes, modality, pretrained, scale_invariant):
    if model_name in ["video_mae", "vivit"]:
        return get_transformer_encoder(model_name, T, num_classes, modality)
    elif model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
        cnn_encoder = get_cnn_encoder(model_name, residual_block, num_classes)
        if pretrained:
            assert "34" in model_name
            rn18_pretrained = r3d_18(weights=True)
            cnn_encoder = copy_weights(cnn_encoder, rn18_pretrained, scale_invariant)
        return cnn_encoder

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
        video_encoder = make_resnet34k_3d(num_classes=num_classes)
    else:
        raise NotImplementedError("Choose a correct model name!")
    return video_encoder

def copy_weights(rn34, rn18, scale_invariant):
    ## Initialise the weights of Resnet34 3D CNN using the layers
    #+ of the Resnet18 3D Cnn.
    #+ NotaBene: This is a manual job. WORKS ONLY FOR ResNet34!

    def generate_nosie(W):
        W_mean, W_std = W.mean(), W.std() / 3
        return torch.normal(mean=0, std=W_std.item(), size=tuple(W.shape))

    def update_weights(target_layer, source_layer, scale_invariant):
        for conv in ['conv1', 'conv2']:
            W = getattr(source_layer, conv)[0].weight
            W_nosie = generate_nosie(W)
            if not scale_invariant:
                target_conv = getattr(target_layer, conv)[0]
            else:
                target_conv = getattr(target_layer, conv)
            target_conv.weight.data.copy_(W + W_nosie)

    def update_si_layer(layer_name, rn34, rn18):
        if layer_name == "layer1":
            extra_iters = 1
        elif layer_name == "layer2":
            extra_iters = 2
        elif layer_name == "layer3":
            extra_iters = 4
        else:
            extra_iters = 1
        rn34_layer = getattr(rn34, layer_name)
        rn18_layer = getattr(rn18, layer_name)
        if layer_name in ["layer2", "layer3", "layer4"]:
            source_downsample_conv = rn18_layer[0].downsample[0]
            target_downsample_conv = rn34_layer[0].shortcut[1]
            target_downsample_conv.load_state_dict(source_downsample_conv.state_dict())
        for i in range(2):
            rn34_block = rn34_layer[i]
            rn18_block = rn18_layer[i]
            for conv in ["conv1", "conv2"]:
                source_layer = getattr(rn18_block, conv)[0]
                target_layer = getattr(rn34_block, conv)
                target_layer.load_state_dict(source_layer.state_dict())
        for i in range(extra_iters):
            rn34_block = rn34_layer[i + 2]
            rn18_block = rn18_layer[1]
            update_weights(rn34_block, rn18_block, scale_invariant=True)

    if not scale_invariant:
        rn34.load_state_dict(rn18.state_dict(), strict=False)
        update_weights(rn34.layer1[2], rn18.layer1[1], scale_invariant=False)
        for i in range(2):
            update_weights(rn34.layer2[i + 2], rn18.layer2[1], scale_invariant=False)
        for i in range(4):
            update_weights(rn34.layer3[i + 2], rn18.layer3[1], scale_invariant=False)
        update_weights(rn34.layer4[2], rn18.layer4[1], scale_invariant=False)
    else:
        rn34.conv1.load_state_dict(rn18.stem[0].state_dict())
        for layer_num in range(1, 5):
            layer_name = "layer" + str(layer_num)
            update_si_layer(layer_name, rn34, rn18)
    return rn34

def get_model(
        modality,
        model_name,
        T,
        residual_block,
        num_classes=400,
        pretrained=False,
        scale_invariant=False,
    ):
    video_encoder, audio_encoder = None, None
    if scale_invariant:
        model_name_split = model_name.split("_")
        model_name_split.insert(1, "si")
        model_name = "_".join(model_name_split)
    if modality == "audio":
        if scale_invariant:
            audio_encoder = make_resnet34k(num_classes=num_classes)
        else:
            audio_encoder = create_model("resnet34", in_chans=1, pretrained=False, num_classes=num_classes)
            audio_encoder.fc = nn.Linear(2048, num_classes, bias=False)
    elif modality == "rgb":
        video_encoder = get_video_encoder(model_name, T, residual_block, num_classes, modality, pretrained, scale_invariant)
    elif modality == "rgb_audio":
        if scale_invariant:
            audio_encoder = make_resnet34k(num_classes=num_classes)
        else:
            audio_encoder = create_model("resnet34", in_chans=1, pretrained=False, num_classes=num_classes)
        audio_encoder = nn.Sequential(*list(audio_encoder.children())[:-1])
        video_encoder = get_video_encoder(model_name, T, residual_block, num_classes, modality, pretrained, scale_invariant)
        if model_name.split("_")[0] in ["r2plus1", "mc3", "r3d"]:
            video_encoder = nn.Sequential(*list(video_encoder.children())[:-1])
    return video_encoder, audio_encoder