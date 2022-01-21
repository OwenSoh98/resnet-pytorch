from random import shuffle
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np

class Classification_Dataset():
    def __init__(self, train_path, test_path, batch_size, shuffle):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_dataset = self.get_dataloader(train_path)
        self.test_dataset = self.get_dataloader(test_path)


    def transforms(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])
    

    def get_dataloader(self, dataset_path):
        dataset = datasets.ImageFolder(dataset_path, transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    
    def split_train(self, train_ratio):
        train_size = int(train_ratio * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size
        return torch.utils.data.random_split(self.train_dataset, [train_size, test_size])


cifar10 = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/test', 32, False)
train, val = cifar10.split_train(0.8)
print(len(train)*32)
