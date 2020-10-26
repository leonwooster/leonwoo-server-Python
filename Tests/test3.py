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

tensor1 = torch.randn(3, 4)
print(tensor1)
tensor2 = torch.randn(4)
print(tensor2)
result = torch.matmul(tensor1, tensor2)
print(result)
print(result.size())