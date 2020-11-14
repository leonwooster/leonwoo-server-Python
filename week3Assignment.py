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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self._body = nn.Sequential(
            # First convolution Layer
            # input size = (32, 32), output size = (28, 28)
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5),
            nn.BatchNorm2d(10),
            # ReLU activation
            nn.ReLU(inplace=True),
            # Max pool 2-d
            nn.MaxPool2d(kernel_size=2),
            
            # Second convolution layer
            # input size = (14, 14), output size = (10, 10)
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # output size = (5, 5)
        )
        
        # Fully connected layers
        self._head = nn.Sequential(
            # First fully connected layer
            # in_features = total number of weight in last conv layer = 16 * 5 * 5
            nn.Linear(in_features=16 * 5 * 5, out_features=120), 
            
            # ReLU activation
            nn.ReLU(inplace=True),
            
            # second fully connected layer
            # in_features = output of last linear layer = 120 
            nn.Linear(in_features=120, out_features=84), 
            
            # ReLU activation
            nn.ReLU(inplace=True),
            
            # Third fully connected layer. It is also output layer
            # in_features = output of last linear layer = 84
            # and out_features = number of classes = 10 (MNIST data 0-9)
            nn.Linear(in_features=84, out_features=10)
        )
        
    def forward(self, x):
       # apply feature extractor
        x = self._body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        x = x.view(x.size()[0], -1)
        # apply classification head
        x = self._head(x)
        return x            

'Calculate mean and standard deviation for image normalization'
def get_mean_std_train_data(data_root):
    
    'inputs[0] = R, inputs[1] = G, inputs[2] = B'
    'inputs[0][31][31] = R channel at row 31 and column 31 (e.g. tensor(0.4824))'
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)    
    print('%d training samples.' % len(train_set))
    
    #placeholder for 3 dim 
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])    

    loader = torch.utils.data.DataLoader(train_set, batch_size=200, num_workers=2)
    h , w = 0, 0        
    batch_end = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chsum = None
    for batch_idx, (inputs, targets) in enumerate(loader): #inputs = 3 dim image data/matrix
        inputs = inputs.to(device)
        batch_end = batch_idx
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3) #[B,C,W,H]            
            chsum = inputs.sum(dim=(0,2,3), keepdim=True) #first dim = batch, second dim= RGB, third = row, fourth = col
        else:
            chsum += inputs.sum(dim=(0,2,3), keepdim=True)

    mean = chsum/len(train_set)/h/w
    print("Batch total: %d" % batch_end) #250 batches for 50,000 images with each batch 200 images
    print("mean_o: %s" % mean)

    chsum = None    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0,2,3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0,2,3), keepdim=True)
    std = torch.sqrt(chsum/(len(train_set) * h * w - 1))
    print("std_o: %s" % std)

    mean = torch.reshape(mean, (1,3))
    std = torch.reshape(std, (1,3))

    return mean, std

def get_data(batch_size, data_root, num_workers=1):
    
    try:
        mean, std = get_mean_std_train_data(data_root)
        assert len(mean) == len(std) == 3
    except:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        
    
    train_test_transforms = transforms.Compose([
        # this re-scale image tensor values between 0-1. image_tensor /= 255
        transforms.ToTensor(),
        # subtract mean and divide by variance.
        transforms.Normalize(mean, std)
    ])
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=False, download=False, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader

@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)    

if __name__ == "__main__":
    my_model = MyModel()
    print(my_model)
    mean, std = get_mean_std_train_data("data")
    print(mean, "\n", std)

#input("Press Key to exit")