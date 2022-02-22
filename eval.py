import torch
from dataset import *
from models.resnet import *
from models.resnet18cifar10 import *

class Eval_Model():
    def __init__(self):
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.dataset = Classification_Dataset('./CIFAR-10/train', './CIFAR-10/val', './CIFAR-10/test', 'cifar10_mean_std.csv', imgsz=32, batch_size=1024, shuffle=True)
        self.num_class = 10
        self.model_path = './results/22-02-2022-18-28-03/best.pt'
        self.model = ResNet18(input_size=32, num_class=self.num_class).to(self.device)

        self.load_model()
        self.eval_test()
    
    def load_model(self):
        """ Load model """
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])
    
    def eval(self, loader):
        """ Evaluate the loss and TP """
        self.model.eval()
        TP = 0

        for idx, (x, y) in enumerate(loader):
            y_pred = self.model(x.to(self.device))
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)

            y = y.to(self.device)

            TP += ((y == y_pred).int().sum()).cpu().detach().numpy()

        return TP
    
    def eval_test(self):
        """ Evaluate Test set """
        test_TP = self.eval(self.dataset.test_loader)
        test_acc = test_TP / self.dataset.test_len

        print('Test Accuracy {}'.format(test_acc))

Eval_Model()
