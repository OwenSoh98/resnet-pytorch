import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, input, output, downsample):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1x1 = nn.Conv2d(input, output, kernel_size=1, stride=1, bias=False)
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=output),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=output),
        )


    def forward(self, x):
        x_left = self.left(x)

        if self.downsample:
            self.conv1x1(x)
        
        x_left += x
        return F.relu(x)


class Resnet34(nn.Module):
    def __init__(self, input_size, classes):
        super(Resnet34, self).__init__()
        self.input_size = input_size
        self.classes = classes

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(128, 128, 2)
        self.layer3 = self.make_layer(256, 256, 2)
        self.layer4 = self.make_layer(512, 512, 2)
        

    def make_layer(self, input, output, num_blocks):
        layers = []

        for i in range(num_blocks):
            layers.append(ResidualBlock(input, output))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x

model = Resnet34(224, 10)
# input = torch.rand(1, 3, 416, 416)
# output = model(input)
summary(model, (3, 224, 224))