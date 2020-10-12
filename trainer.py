import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, criterion, optimizer, epoch_num):
        self.model = model
        
        # loss function
        self.criterion = criterion
        
        # Optimizer
        self.optimizer = optimizer
        
        # num of epochs
        self.epoch_num = epoch_num

    def fit(self, inputs, targets):
        """
        Updating model trainable parameters in loop for given number of epochs
        """
        
        # set model in train state. 
        # Why it (and model.eval()) is important, 
        # we will see when we will be training a deep neural network.
        self.model.train()
        
        # run train loop for given epochs
        for _ in range(self.epoch_num):
            
            # reset previously calculated gradient to zero
            self.optimizer.zero_grad()
            
            # predict probability of class '1'
            preds = self.model(inputs)
            
            # get loss
            loss = self.criterion(preds, targets)
            
            # calculate gradients
            loss.backward()
            
            # update parameters with gradient
            self.optimizer.step()

    def predict(self, inputs):
        
        # set model in train state. 
        self.model.eval()
        # temporarily set requires_grad flag to false
        with torch.no_grad():
            # probability of class one prediction
            preds = self.model(inputs)
        return preds