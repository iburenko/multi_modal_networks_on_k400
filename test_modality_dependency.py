from argparse import ArgumentParser
from itertools import product

import torch
from torch import nn

def get_norm_layer(scale_invariant, video_feat_dim):
    if scale_invariant:
        norm_layer = nn.BatchNorm1D(video_feat_dim)
    else:
        norm_layer = nn.Identity()
    return norm_layer

def create_fusion_layer(video_feat_dim, audio_feat_dim, num_classes, norm_layer):
    fusion = torch.nn.Sequential(
        torch.nn.Linear(audio_feat_dim + video_feat_dim, video_feat_dim, bias=False),
        norm_layer,
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(video_feat_dim, video_feat_dim, bias=False),
        norm_layer,
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(video_feat_dim, num_classes, bias=False)
    )
    return fusion

def validate(fusion, video_embeddings, audio_embeddings, x, y, labels):
    video_embeddings = x * video_embeddings
    audio_embeddings = y * audio_embeddings
    logits = fusion(torch.hstack([video_embeddings, audio_embeddings]))
    _, pred = logits.max(1)
    acc = (pred == labels).sum() / len(labels)
    return acc

def main(args):
    audio_feat_dim = args.audio_feat_dim
    video_feat_dim = args.video_feat_dim
    num_classes = args.num_classes
    scale_invariant = args.scale_invariant
    norm_layer = get_norm_layer(scale_invariant, video_feat_dim)
    fusion = create_fusion_layer(
        video_feat_dim, 
        audio_feat_dim, 
        num_classes, 
        norm_layer
        )

    video_embeddings = torch.randn(100, 512)
    audio_embeddings = torch.randn(100, 512)
    labels = torch.randint(0, 401, (100,))

    num_points = 25
    mesh_acc = torch.zeros(num_points, num_points)

    x_linspace = torch.linspace(-1, 2, num_points)
    y_linspace = torch.linspace(-1, 2, num_points)
    grid = torch.meshgrid(x_linspace, y_linspace)
    for i, j in product(range(num_points), range(num_points)):
        x = grid[0][i][j]
        y = grid[1][i][j]
        mesh_acc[i, j] = validate(fusion, video_embeddings, audio_embeddings, x, y, labels)
        print(x, y, mesh_acc[i, j])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio-feat-dim", type=int, default=512)
    parser.add_argument("--video-feat-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=400)
    parser.add_argument("--scale-invariant", action="store_true")
    args = parser.parse_args()
    main(args)