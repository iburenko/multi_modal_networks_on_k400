## ResNet18 for CIFAR
## Based on: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
## copied from https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/resnet18.py

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class PreActBlock3D(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock3D, self).__init__()
        # affine and bias are False, see https://arxiv.org/pdf/1910.07454.pdf
        self.bn1 = nn.BatchNorm3d(in_planes, affine=False)
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, affine=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            # https://arxiv.org/pdf/1910.07454.pdf: "we add an additional normalizaiton layer in the downsample before downsampling"
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_planes, affine=False),
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        downsample = self.downsample(out) if hasattr(self, 'downsample') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += downsample
        return out

class PreActResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, init_channels=64, linear_norm=10.0, linear_bias=False):
        super(PreActResNet3D, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv3d(3, c, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm3d(8*c*block.expansion, affine=False)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool3d(4)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(8*c*block.expansion, num_classes, bias=linear_bias)  

        # Custom initialization: just set the norm higher
        # Move this part to backbones.py
        # if linear_norm > 0:
        #     alpha = linear_norm
        #     W = self.linear.weight.data
        #     self.linear.weight.data = alpha * W / W.norm()

        # Freeze the parameters in the last FC layer: https://arxiv.org/pdf/1910.07454.pdf
        #for n, param in self.linear.named_parameters():
        #    param.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        # affine and bias are False, see https://arxiv.org/pdf/1910.07454.pdf
        self.bn1 = nn.BatchNorm2d(in_planes, affine=False)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            # https://arxiv.org/pdf/1910.07454.pdf: "we add an additional normalizaiton layer in the downsample before downsampling"
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes, affine=False),
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        downsample = self.downsample(out) if hasattr(self, 'downsample') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += downsample
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, in_chans=3, num_classes=10, init_channels=64, linear_norm=10.0, linear_bias=False):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(in_chans, c, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(8*c*block.expansion, affine=False)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(8*c*block.expansion, num_classes, bias=linear_bias)  

        # Custom initialization: just set the norm higher
        # Move this part to backbones.py
        # if linear_norm > 0:
        #     alpha = linear_norm
        #     W = self.linear.weight.data
        #     self.linear.weight.data = alpha * W / W.norm()

        # Freeze the parameters in the last FC layer: https://arxiv.org/pdf/1910.07454.pdf
        #for n, param in self.linear.named_parameters():
        #    param.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def make_resnet18k_3d(k=64, num_classes=10, *args, **kwargs) -> PreActResNet3D:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet3D(PreActBlock3D, [2, 2, 2, 2], num_classes=num_classes, init_channels=k, *args, **kwargs)

def make_resnet34k_3d(k=64, num_classes=10, *args, **kwargs) -> PreActResNet3D:
    ''' Returns a ResNet34 with width parameter k. (k=64 is standard ResNet34)'''
    return PreActResNet3D(PreActBlock3D, [3, 4, 6, 3], num_classes=num_classes, init_channels=k, *args, **kwargs)

def make_resnet18k(k=64, num_classes=10, *args, **kwargs) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, init_channels=k, *args, **kwargs)

def make_resnet34k(k=64, num_classes=10, *args, **kwargs) -> PreActResNet:
    ''' Returns a ResNet18 with width parameter k. (k=64 is standard ResNet18)'''
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes, init_channels=k, *args, **kwargs)