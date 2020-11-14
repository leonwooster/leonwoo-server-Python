import matplotlib.pyplot as plt  # one of the best graphics library for python

import os
import time

from typing import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision

# points = [10,20,30,45]
# npArray = np.array(points)
# print(npArray)

# m = nn.ReLU()
# input = torch.randn(2)
# output = m(input)
# print(output)


# ## ignoring a value
# a, _, b = (1, 2, 3) # a = 1, b = 3
# print(a, b)

# a, *_, b = (7, 6, 5, 4, 3, 2, 1)
# print(a, b)

# for _ in range(5):
#     print(_)


# #If you have a long digits number, you can separate the group of digits as you like for better understanding.
# ## different number systems
# ## you can also check whether they are correct or not by coverting them into integer using "int" method
# million = 1_000_000
# binary = 0b_0010
# octa = 0o_64
# hexa = 0x_23_ab

# print(million)
# print(binary)
# print(octa)
# print(hexa)    


# x = torch.tensor([1, 2, 3, 4])
# torch.unsqueeze(x, -1)
# print(x)

m = nn.BatchNorm2d(100)
#m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)

ct= 1
for x in range(output):        
    print("Count: ", ct," ", x)
    ct += 1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images    
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))   
