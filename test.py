import torch
from torchvision import datasets, transforms
import numpy as np


data_transform = transforms.Compose([
            # transforms.Resize([32, 32]),
            # transforms.RandomCrop([24, 24]),
            # transforms.RandomCrop([32, 32], padding=4),
            # transforms.RandomRotation(15),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

# obtain training indices that will be used for validation
train_data = datasets.ImageFolder('./CIFAR-10/train', transform=data_transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = datasets.SubsetRandomSampler(train_idx)
valid_sampler = datasets.SubsetRandomSampler(valid_idx)