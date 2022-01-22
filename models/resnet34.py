from turtle import left
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, input, output, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1x1 = nn.Conv2d(input, output, kernel_size=1, stride=2, bias=False)
        self.left_conv = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=output),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=output),
        )


    def forward(self, x):
        x_left = self.left_conv(x)

        if self.downsample:
            x = self.conv1x1(x)
        
        x_left += x
        
        return F.relu(x_left)


class Resnet34(nn.Module):
    def __init__(self, input_size, num_class):
        super(Resnet34, self).__init__()
        self.input_size = input_size
        self.classes = num_class

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, 3, downsample=False)
        self.layer2 = self.make_layer(64, 128, 4, downsample=True)
        self.layer3 = self.make_layer(128, 256, 6, downsample=True)
        self.layer4 = self.make_layer(256, 512, 3, downsample=True)
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


    def layer_shape(self, name, x):
        print(name)
        print(x.shape)
        print('\n')


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# model_size = 224
# num_class = 10
# model = Resnet34(model_size, num_class).cuda()
# input = torch.rand(4, 3, 224, 224).cuda()
# output = model(input)
# summary(model, (3, 224, 224))