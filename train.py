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

class Train_Model():
    def __init__(self):
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.dataset = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/val', './CIFAR-10/test', 'cifar10_mean_std.csv', imgsz=32, batch_size=1024, shuffle=True)
        self.num_class = 10
        self.epochs = 150
        self.lr = 1e-2
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.modelpath = './results/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '/'
        self.model = Resnet_Cifar10(input_size=32, num_class=self.num_class, n=2).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 100], gamma=0.1)
        
        self.train()

    def create_directory(self, path):
        """ Create directory """
        if not os.path.exists(path):
            os.makedirs(path)

    def save_csv(self, csv_file, row):
        """ Write to CSV file"""
        with open(csv_file, 'a', newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(row)

        writeFile.close()

    def save_headers(self, csv_file):
        """ Write CSV file header """
        fieldnames = ['epoch', 'validation loss', 'validation loss', 'validation accuracy']
        self.save_csv(csv_file, fieldnames)

    def train_epoch(self):
        """ Train for 1 epoch """
        self.model.train()
        total_loss = 0

        for idx, (x, y) in enumerate(self.dataset.train_loader):
            # print(x)
            y_pred = self.model(x.to(self.device))
            y = y.to(self.device)
            loss = self.loss_function(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.cpu().detach().numpy()

        return total_loss / self.dataset.train_len
    
    def train(self):
        """ Train model """
        self.create_directory(self.modelpath)
        csv_file = os.path.join(self.modelpath, 'training_results.csv')
        self.save_headers(csv_file)
        
        last_val_acc = 0
        for i in range(self.epochs):
            print('Epoch: {}/{}'.format(i+1, self.epochs))
            train_loss = self.train_epoch()

            val_loss, val_TP = self.eval(self.dataset.val_loader)
            val_acc = val_TP / self.dataset.val_len

            print('Train Loss: {}, Val Loss {}, Val Accuracy {}'.format(train_loss, val_loss, val_acc))

            row = [i+1, train_loss, val_loss, val_acc]
            self.save_csv(csv_file, row)

            if val_acc > last_val_acc:
                last_val_acc = val_acc
                print('New highest accuracy detected. Checkpoint saved.')
                self.save()

    def save(self):
        """ Save model """
        PATH = os.path.join(self.modelpath, 'model.pt')
        torch.save(self.model.state_dict(), PATH)

    def eval(self, loader):
        """ Evaluate the loss and TP """
        self.model.eval()
        total_loss = 0
        TP = 0

        for idx, (x, y) in enumerate(loader):
            y_pred = self.model(x.to(self.device))
            y = y.to(self.device)
            total_loss += self.loss_function(y_pred, y).cpu().detach().numpy()

            y_pred = F.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            TP += ((y == y_pred).int().sum()).cpu().detach().numpy()

        return total_loss / self.dataset.test_len, TP


Train_Model()
