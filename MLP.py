import torch.nn as nn
import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.nn.modules import linear
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

class MLP(torch.nn.Module):
    def __init__(self, train_loader, validation_loader, loss_function):
        super().__init__()

        self.train_loader = train_loader
        self.loss_function = loss_function
        self.validation_loader = validation_loader
        self.optimizer = None # to be set later

        #build model using sequential container
        self.model = nn.Sequential(
            #add input layer
            nn.Linear(28*28, 512),
            #add RELu activation
            nn.ReLU(),
            #add another layer
            nn.Linear(512,512),
            #add ReLU activation
            nn.ReLU(),
            #add output layer
            nn.Linear(512, 10)            
        )

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, x):
        #forward pass
        return self.model(x)

    'This is the Training routine which does the following:'
    '1. It takes batches of data from train dataloader'
    '2. Prepares the input data in the form that can be fed to the network, i.e. it flattens the 28x28 image to a single 784 dimensional vector before passing it to the network.'
    '3. The training data is passed through the network'
    '4. Compute the cross entropy loss using the predicted output and the training labels'
    '5. Remove previous gradients using optimizer.zero_grad'
    '6. Compute Gradients using the backward function'
    '7. Update the weights using the optimizer.step function and repeat until all the data is passed through the network.'
    def train(self):

        if(self.optimizer == None):
            raise Exception("Optimizer is None, please set one before call train.")

        self.model.train() #set model to train mode

        running_loss = 0
        running_correct = 0

        for(x_train, y_train) in self.train_loader:
            #forward pass
            #flatten the image since the input to the network is a 784 dimensional vector
            x_train = x_train.view(x_train.shape[0], -1)
            #compute predicted y by passing x to the model
            y = self.model(x_train)

            #compute and print loss
            loss = self.loss_function(y, y_train)
            running_loss += loss.item()

            #compute accurary
            y_pred = y.argmax(dim=1)
            correct = torch.sum(y_pred == y_train)
            running_correct += correct

            #zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()            

            #calculate gradient using backward pass
            loss.backward()

            #update model parameters(weights)
            self.optimizer.step()

        return running_loss/len(self.train_loader), running_correct.item()/len(self.train_loader.dataset)

    'We use the validation loader to pass batches of data through the network for performing validation on unseen data.'
    'Note that there is only forward pass and no backward pass during validation.'
    def val(self):
        self.model.eval() #set model to eval mode

        running_loss = 0
        running_correct = 0

        with torch.no_grad():
            for(x_val, y_val) in self.validation_loader:
                #forward pass
                #flatten the image since the input to the network is a 784 dimensional vector
                x_val = x_val.view(x_val.shape[0], -1)

                #compute raw score by passing x to the model
                y = self.model(x_val)

                #score to probability using softmax
                prob = nn.functional.softmax(y, dim=1)

                #compute accurary
                y_pred = prob.argmax(dim=1)

                correct = torch.sum(y_pred==y_val)
                running_correct += correct

                #compute and print loss
                loss = self.loss_function(y, y_val)
                running_loss += loss.item()
        return running_loss/len(self.validation_loader), running_correct.item()/len(self.validation_loader.dataset)




