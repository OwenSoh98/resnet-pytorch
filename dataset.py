from random import shuffle
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np

class Classification_Dataset():
    def __init__(self, train_path, test_path, batch_size, shuffle, train_ratio):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_len = 0
        self.val_len = 0
        self.test_len = 0
        self.train_loader, self.val_loader, self.test_loader = self.get_data(train_ratio)

    def transforms(self):
        """ Specify transformations here"""
        return transforms.Compose([
            transforms.Resize([32, 32]),
            # transforms.RandomCrop([24, 24]),
            transforms.RandomCrop([32, 32], padding=4),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def ToTensor(self):
        """ Specify transformations here"""
        return transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor()
        ])
    
    def get_dataloader(self, dataset):
        """ Returns dataloader of dataset"""
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    
    def split_trainval(self, train_ratio):
        """ Split train dataset into train/val dataset """
        train_dataset = datasets.ImageFolder(self.train_path, self.transforms())
        self.train_len = int(train_ratio * len(train_dataset))
        self.val_len = len(train_dataset) - self.train_len
        return torch.utils.data.random_split(train_dataset, [self.train_len, self.val_len])


    def get_data(self, train_ratio):
        """ Returns Train, Val, Test dataloaders """
        train_dataset, val_dataset = self.split_trainval(train_ratio)
        train_dataloader = self.get_dataloader(train_dataset)
        val_dataloader = self.get_dataloader(val_dataset)

        test_dataset = datasets.ImageFolder(self.test_path, self.ToTensor())
        self.test_len = len(test_dataset)
        test_dataloader = self.get_dataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
