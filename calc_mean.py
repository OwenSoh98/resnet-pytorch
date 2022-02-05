from random import shuffle
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import pandas as pd

class Calc_Mean():
    def __init__(self):
        self.train_path = './CIFAR-10/train'
        self.batch_size = 8192
        self.imgsz = 32
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.train_len = 0
        self.train_loader= self.get_data()
        self.mean, self.std = self.get_mean_std(self.train_loader)
        print(self.mean)
        print(self.std)
        self.save_mean_std()

    def train_transform(self):
        """ Specify train transformations here"""
        return transforms.Compose([
            transforms.Resize([self.imgsz, self.imgsz]),
            transforms.ToTensor()
        ])
    
    def get_dataloader(self, path, transform):
        """ Returns dataloader of dataset"""
        dataset = datasets.ImageFolder(path, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        return dataloader, len(dataset)

    def get_data(self):
        """ Returns Train dataloaders """
        train_dataloader, self.train_len = self.get_dataloader(self.train_path, transform=self.train_transform())
        # train_dataloader, self.train_len = self.get_dataloader(self.train_path, transform=None)
        return train_dataloader
    
    def save_mean_std(self):
        mean = pd.DataFrame(self.mean, columns=['mean'])
        std = pd.DataFrame(self.std, columns=['std'])
        data = pd.concat([mean, std], axis=1)
        data.to_csv('cifar10_mean_std_rgb.csv', index=False)

    def get_mean_std(self, loader):
        mean = 0
        std = 0
        for idx, (x, y) in enumerate(loader):
            x = x.to(self.device)
            x = x.view(x.shape[0], x.shape[1], -1)
            mean += torch.sum(torch.mean(x, axis=2), axis=0).cpu().detach().numpy()
            std += torch.sum(torch.std(x, axis=2), axis=0).cpu().detach().numpy()

        return mean/self.train_len, std/self.train_len


Calc_Mean()
