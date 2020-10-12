import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        
        # define linear layer (WX + B)
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x):
        # calculate WX + B
        x = self.linear(x)
        
        # sigmoid activation (prediction probability of class 1)
        predictions = torch.sigmoid(x)
        return predictions