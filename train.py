from torch import nn
from torch import optim
from dataset import *
from models.resnet34 import *


class Train_Model():
    def __init__(self, epochs):
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.dataset = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/test', batch_size=32, shuffle=False)
        # self.train_loader, self.val_loader, self.test_loader = self.dataset.get_dataset(0.8)
        self.train_loader, self.test_loader = self.dataset.get_dataset(0.8)

        self.model = Resnet34(input_size=32, num_class=10).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()

        self.epochs = epochs
        self.lr = 1e-4
        self.weight_decay = 5e-4
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

        self.train()

    def train_epoch(self):
        self.model.train()
        for idx, (x, y) in enumerate(self.train_loader):
            y_pred = self.model(x.to(self.device))
            loss = self.loss_function(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        for i in range(self.epochs):
            print('Epoch:{}/{}'.format(i+1, self.epochs))
            self.train_epoch()

Train_Model(20)
