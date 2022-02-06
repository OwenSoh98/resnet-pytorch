import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from dataset import *
from models.resnet34 import *
from models.resnet18cifar10 import *
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt

class Train_Model():
    def __init__(self):
        self.batch_size = 1024
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.dataset = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/val', './CIFAR-10/test', 'cifar10_mean_std.csv', imgsz=32, batch_size=self.batch_size, shuffle=True)
        self.num_class = 10
        self.epochs = 50
        self.lr = 0.01
        self.weight_decay = 0.0005
        self.momentum = 0.9

        self.modelpath = './results/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '/'
        self.model = Resnet_Cifar10(input_size=32, num_class=self.num_class, n=2).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20*len(self.dataset.train_loader), 40*len(self.dataset.train_loader)], gamma=0.1)
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.1, epochs=self.epochs, steps_per_epoch=len(self.dataset.train_loader), pct_start=0.3, 
        #                     anneal_strategy="cos", div_factor=0.1/self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.05, epochs=self.epochs, steps_per_epoch=len(self.dataset.train_loader), pct_start=0.4, 
                            anneal_strategy="linear", div_factor=0.05/self.lr, final_div_factor=self.lr/0.0005, three_phase=True)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.train()

    def train_epoch(self, loader):
        """ Train for 1 epoch """
        self.model.train()
        lr = self.scheduler.get_last_lr()

        for idx in range(len(self.dataset.train_loader)):
            self.optimizer.step()
            self.scheduler.step()

        return lr[0]
    def train(self):
        """ Train model """
        lrs = []

        for i in range(self.epochs):
            lrs.append(self.train_epoch(self.dataset.train_loader))
        
        epochs = range(1, self.epochs+1)
        print(lrs)
        plt.plot(epochs, lrs)
        plt.show()


Train_Model()
