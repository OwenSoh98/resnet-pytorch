from turtle import left
from typing import List
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
# from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, input: int, output: int, stride: int, downsample: bool):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1x1 = nn.Conv2d(input, output, kernel_size=1, stride=2, bias=False)
        self.left_conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_left = self.left_conv(x)

        if self.downsample:
            x = self.conv1x1(x)
        
        x_left += x
        
        return F.relu(x_left)


class Resnet(nn.Module):
    def __init__(self, input_size: int, num_class: int, num_blocks: List):
        super(Resnet, self).__init__()
        self.input_size = input_size
        self.classes = num_class
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, num_blocks[0], downsample=False)
        self.layer2 = self.make_layer(64, 128, num_blocks[1], downsample=True)
        self.layer3 = self.make_layer(128, 256, num_blocks[2], downsample=True)
        self.layer4 = self.make_layer(256, 512, num_blocks[3], downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)


    def make_layer(self, input, output, num_blocks, downsample):
        """Create ResNet layers"""
        layers = []

        """TODO: OPTIMIZE DOWNSAMPLING CODE"""
        if downsample:
            stride = 2
        else:
            stride = 1
        
        for i in range(num_blocks):
            layers.append(ResidualBlock(input, output, stride, downsample))

            if downsample:
                input = output
                stride = 1
            downsample = False
        
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(input_size: int, num_class: int):
    return Resnet(input_size, num_class, [2, 2, 2, 2])


def ResNet34(input_size: int, num_class: int):
    return Resnet(input_size, num_class, [3, 4, 6, 3])