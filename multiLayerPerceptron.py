from torch import optim
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import torch
from torch.optim import optimizer

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from MLP import MLP

#Get reproducable result
torch.manual_seed(0)

#Training Set
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

#validation dataset
validation_dataset = datasets.MNIST('./data',train=False, transform=transforms.ToTensor())

#bath size: How many images are used to calculate the gradient
batch_size = 32

#train  dataloader
train_loader= DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

#validation loader
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

'We first instantiate a MLP model using the MLP class defined above.'
'We then specify the Cross Entropy loss for doing classification. This will be used for calculating the loss over each batch.'
'Finally, we specify the optimizer which we have chosen to be SGD in this case with a Learning rate of 0.1.'
#training params
num_epochs = 20

#define loss function
loss_function = nn.CrossEntropyLoss()

#construct model
model = MLP(train_loader, validation_loader, loss_function)

#define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)
model.setOptimizer(optimizer)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

'Main Routine that calls the training and validation functions. '
'We keep track of the loss of each epoch so that we can plot it to visualize the progressive change in loss over epochs.'
print("Starting training......")
for ep in range(num_epochs):
    train_loss, train_acc = model.train()
    val_loss, val_acc = model.val()
    print("Epoch: {}, Train Loss = {:.3f}, Train Acc = {:.3f} , Val Loss = {:.3f}, Val Acc = {:.3f}".
          format(ep, train_loss, train_acc, val_loss, val_acc))
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

'Plot the Loss & Accuracy curves'
plt.figure(figsize=[20,8])
plt.subplot(121)
plt.plot(train_loss_history,'r')
plt.plot(val_loss_history,'b')
plt.title("Loss Curve")

plt.subplot(122)
plt.plot(train_acc_history,'r')
plt.plot(val_acc_history,'b')
plt.title("Accuracy Curve")

'Perform Inference'
images, labels = next(iter(validation_loader))
plt.imshow(images[0][0],'gray')

images.resize_(images.shape[0], 1, 784)
score = model(images[0,:])
prob = nn.functional.softmax(score[0], dim=0)
y_pred =  prob.argmax()
print("Predicted class {} with probability {}".format(y_pred, prob[y_pred]))