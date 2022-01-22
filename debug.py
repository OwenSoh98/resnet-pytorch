import enum
import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('./CIFAR-10/train', transforms)
test_dataset = datasets.ImageFolder('./CIFAR-10/test', transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

for idx, (x, y) in enumerate(trainloader):
    print(idx)

# for i in range(1):
#     i = 4
#     img = np.array(image_batch[i,:,:,:])
#     img = np.reshape(img, (32, 32, 3))
#     img = np.moveaxis(image_batch[i].numpy(), 0, 2)
#     img = img
#     print(img)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#     plt.imshow(img)
#     plt.title(labels[i])
# plt.show()
# for i in range(1):
#     img = image_batch[i].numpy()
#     print(img.shape)
#     img = np.moveaxis(image_batch[i].numpy(), 0, 2)
#     print('-------------------------------------------------')
#     print(img.shape)
#     # plt.subplot(2, 4, i+1)
#     plt.imshow(img)
#     plt.title(labels[i])

# img = cv2.imread('./CIFAR-10/train/airplane/0000.jpg')
# print(img)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# plt.show()