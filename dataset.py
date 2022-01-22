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
        dataset = datasets.ImageFolder(dataset_path, self.transforms())
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    
    def split_train(self, train_ratio):
        train_size = int(train_ratio * len(self.train_dataset))
        test_size = len(self.train_dataset) - train_size
        return torch.utils.data.random_split(self.train_dataset, [train_size, test_size])


    def get_dataset(self, train_ratio):
        # train, val = self.split_train(train_ratio)
        # return train, val, self.test_dataset
        return self.train_dataset, self.test_dataset


# cifar10 = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/test', batch_size=32, shuffle=False)
# train, val, test = cifar10.get_dataset(0.8)
