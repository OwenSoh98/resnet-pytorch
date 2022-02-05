from random import shuffle
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import pandas as pd

class Classification_Dataset():
    def __init__(self, train_path, val_path, test_path, mean_std_path, imgsz, batch_size, shuffle):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.mean, self.std = self.parse_data(mean_std_path, ['mean', 'std'])

        self.imgsz = imgsz
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_len = 0
        self.val_len = 0
        self.test_len = 0

        self.train_loader, self.val_loader, self.test_loader = self.get_data()

    def train_transform(self):
        """ Specify train transformations here"""
        return transforms.Compose([
            transforms.Resize([self.imgsz, self.imgsz]),
            transforms.RandomCrop([32, 32], padding=4),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def testval_transform(self):
        """ Specify test/val transformations here"""
        return transforms.Compose([
            transforms.Resize([self.imgsz, self.imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def parse_data(self, file, columns):
        """ Parse csv file """
        csv_file = pd.read_csv(file)
        data = pd.DataFrame(csv_file, columns=columns).to_numpy()
        return data[:,0], data[:,1]
    
    def get_dataloader(self, path, transform):
        """ Returns dataloader of dataset"""
        dataset = datasets.ImageFolder(path, transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader, len(dataset)

    def get_data(self):
        """ Returns Train, Val, Test dataloaders """
        train_dataloader, self.train_len = self.get_dataloader(self.train_path, transform=self.train_transform())
        val_dataloader, self.val_len = self.get_dataloader(self.val_path, transform=self.testval_transform())
        test_dataloader, self.test_len = self.get_dataloader(self.test_path, transform=self.testval_transform())

        return train_dataloader, val_dataloader, test_dataloader
