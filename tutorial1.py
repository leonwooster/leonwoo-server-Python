import torch
from torch._C import dtype

from torch.nn.parameter import Parameter

x = torch.randn(5,3)
print(x)

w = torch.Tensor(5, 3)
p = Parameter(w)
print(w)
print(p)


