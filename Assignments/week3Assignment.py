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

if __name__ == "__main__":
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            # convolution layers
            self._body = nn.Sequential(
                # input size = (32, 32), output size = (28, 28)
                nn.Conv2d(in_channels=3, out_channels=50, kernel_size=2),
                nn.BatchNorm2d(50),                
                nn.ReLU(inplace=True),                
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # input size = (14, 14), output size = (10, 10)
                nn.Conv2d(in_channels=50, out_channels=100, kernel_size=2),
                nn.BatchNorm2d(100),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=100, out_channels=200, kernel_size=2),
                nn.BatchNorm2d(200),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                #size(2x2)
                nn.Conv2d(in_channels=200, out_channels=500, kernel_size=2),
                nn.BatchNorm2d(500),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Fully connected layers
            self._head = nn.Sequential(
                nn.Linear(in_features=500, out_features=300), 
                nn.ReLU(inplace=True),

                nn.Linear(in_features=300, out_features=100), 
                nn.ReLU(inplace=True),
                
                nn.Linear(in_features=100, out_features=84),                
                nn.ReLU(inplace=True),
                
                nn.Linear(in_features=84, out_features=10)
            )
            
        def forward(self, x):
        # apply feature extractor
            x = self._body(x)
            # flatten the output of conv layers
            # dimension should be batch_size * number_of weight_in_last conv_layer
            x = x.view(x.size()[0], -1)
            #x = x.view(-1, 16 * 5 * 5)
            # apply classification head
            x = self._head(x)
            return x            
    'Calculate mean and standard deviation for image normalization'
    def get_mean_std_train_data(data_root):
        
        'inputs[0][0] = R, inputs[0][1] = G, inputs[0][2] = B'
        'inputs[0][0][31][31] = Batch 0, R channel at row 31 and column 31 (e.g. tensor(0.4824))'
        '32 x 32 image size'
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

        mean = torch.reshape(mean, (-1,))
        std = torch.reshape(std, (-1,))  
        mean = mean.cpu().numpy()
        std = std.cpu().numpy() 

        lenMean = len(mean)     
        lenStd = len(std)

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

    @dataclass
    class TrainingConfiguration:
        '''
        Describes configuration of the training process
        '''
        batch_size: int = 16  # amount of data to pass through the network at each forward-backward iteration
        epochs_count: int = 4  # number of times the whole dataset will be passed through the network
        learning_rate: float = 0.001  # determines the speed of network's weights update
            
        log_interval: int = 100  # how many batches to wait between logging training status
        test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
        data_root: str = "../resource/lib/publicdata/images"  # folder to save data
        num_workers: int = 20  # number of concurrent processes using to prepare data
        device: str = 'cuda'  # device to use for training.
        # update changed parameters in blow coding block.
        # Please do not change "data_root" 
        
        momentum: float = 0.9
        eps: float = 1e-05

    def setup_system(system_config: SystemConfiguration) -> None:
        torch.manual_seed(system_config.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
            torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

    def train(
        #train_config: TrainingConfiguration, model: nn.Module, optimizer: torch.optim.Optimizer,
        train_config: TrainingConfiguration, model: nn.Module, optimizer: torch.optim.Adam,
        train_loader: torch.utils.data.DataLoader, epoch_idx: int
    ) -> None:
        
        # change model in training mood
        model.train()
        
        # to get batch loss
        batch_loss = np.array([])
        
        # to get batch accuracy
        batch_acc = np.array([])
            
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # clone target
            indx_target = target.clone()
            # send data to device (its is medatory if GPU has to be used)
            data = data.to(train_config.device)
            # send target to device
            target = target.to(train_config.device)

            # reset parameters gradient to zero
            optimizer.zero_grad()
            
            # forward pass to the model
            output = model(data)
            
            # cross entropy loss
            loss = F.cross_entropy(output, target)
            
            # find gradients w.r.t training parameters
            loss.backward()
            # Update parameters using gardients
            optimizer.step()
            
            batch_loss = np.append(batch_loss, [loss.item()])
            
            # Score to probability using softmax
            prob = F.softmax(output, dim=1)
                
            # get the index of the max probability
            pred = prob.data.max(dim=1)[1]  
                            
            # correct prediction
            correct = pred.cpu().eq(indx_target).sum()
                
            # accuracy
            acc = float(correct) / float(len(data))
            
            batch_acc = np.append(batch_acc, [acc])

            if batch_idx % train_config.log_interval == 0 and batch_idx > 0:              
                print(
                    'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                        epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
                    )
                )
                
        epoch_loss = batch_loss.mean()
        epoch_acc = batch_acc.mean()
        return epoch_loss, epoch_acc

    def validate(
        train_config: TrainingConfiguration,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
    ) -> float:
        model.eval()
        test_loss = 0
        count_corect_predictions = 0
        for data, target in test_loader:
            indx_target = target.clone()
            data = data.to(train_config.device)
            
            target = target.to(train_config.device)
            
            output = model(data)
            # add loss for each mini batch
            test_loss += F.cross_entropy(output, target).item()
            
            # Score to probability using softmax
            prob = F.softmax(output, dim=1)
            
            # get the index of the max probability
            pred = prob.data.max(dim=1)[1] 
            
            # add correct prediction count
            count_corect_predictions += pred.cpu().eq(indx_target).sum()

        # average over number of mini-batches
        test_loss = test_loss / len(test_loader)  
        
        # average over number of dataset
        accuracy = 100. * count_corect_predictions / len(test_loader.dataset)
        
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, count_corect_predictions, len(test_loader.dataset), accuracy
            )
        )
        return test_loss, accuracy/100.0

    def save_model(model, device, model_dir='models', model_file_name='cifar10_cnn_model.pt'):
        

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, model_file_name)

        # make sure you transfer the model to cpu.
        if device == 'cuda':
            model.to('cpu')

        # save the state_dict
        torch.save(model.state_dict(), model_path)
        
        if device == 'cuda':
            model.to('cuda')
        
        return    

    def main(system_configuration=SystemConfiguration(), training_configuration=TrainingConfiguration()):
        
        # system configuration
        setup_system(system_configuration)

        # batch size
        batch_size_to_set = training_configuration.batch_size
        # num_workers
        num_workers_to_set = training_configuration.num_workers
        # epochs
        epoch_num_to_set = training_configuration.epochs_count

        # if GPU is available use training config, 
        # else lowers batch_size, num_workers and epochs count
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            num_workers_to_set = 2

        # data loader
        train_loader, test_loader = get_data(
            batch_size=training_configuration.batch_size,
            data_root=training_configuration.data_root,
            num_workers=num_workers_to_set
        )
        
        # Update training configuration
        training_configuration = TrainingConfiguration(
            device=device,
            num_workers=num_workers_to_set
        )

        # initiate model
        model = MyModel()
            
        # send model to device (GPU/CPU)
        model.to(training_configuration.device)

        # optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_configuration.learning_rate,
            momentum=training_configuration.momentum
        )

        best_loss = torch.tensor(np.inf)
        best_accuracy = torch.tensor(0)
        
        # epoch train/test loss
        epoch_train_loss = np.array([])
        epoch_test_loss = np.array([])
        
        # epch train/test accuracy
        epoch_train_acc = np.array([])
        epoch_test_acc = np.array([])
        
        # trainig time measurement
        t_begin = time.time()
        for epoch in range(training_configuration.epochs_count):
            
            train_loss, train_acc = train(training_configuration, model, optimizer, train_loader, epoch)
            
            epoch_train_loss = np.append(epoch_train_loss, [train_loss])
            
            epoch_train_acc = np.append(epoch_train_acc, [train_acc])

            elapsed_time = time.time() - t_begin
            speed_epoch = elapsed_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * training_configuration.epochs_count - elapsed_time
            
            print(
                "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                    elapsed_time, speed_epoch, speed_batch, eta
                )
            )

            if epoch % training_configuration.test_interval == 0:
                current_loss, current_accuracy = validate(training_configuration, model, test_loader)
                
                epoch_test_loss = np.append(epoch_test_loss, [current_loss])
            
                epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])
                
                if current_loss < best_loss:
                    best_loss = current_loss
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    print('Accuracy improved, saving the model.\n')
                    save_model(model, device)
                
                    
        print("Total time: {:.2f}, Best Loss: {:.3f}, Best Accuracy: {:.3f}".format(time.time() - t_begin, best_loss, 
                                                                                    best_accuracy))
        
        return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc

    model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = main()

    # Plot loss
    plt.rcParams["figure.figsize"] = (10, 6)
    x = range(len(epoch_train_loss))


    plt.figure
    plt.plot(x, epoch_train_loss, color='r', label="train loss")
    plt.plot(x, epoch_test_loss, color='b', label="validation loss")
    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot accuracy
    plt.rcParams["figure.figsize"] = (10, 6)
    x = range(len(epoch_train_loss))


    plt.figure
    plt.plot(x, epoch_train_acc, color='r', label="train accuracy")
    plt.plot(x, epoch_test_acc, color='b', label="validation accuracy")
    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc='center right')
    plt.title('Training and Validation Accuracy')
    plt.show()

    # initialize the model
    cnn_model = MyModel()

    models = 'models'

    model_file_name = 'cifar10_cnn_model.pt'

    model_path = os.path.join(models, model_file_name)

    # loading the model and getting model parameters by using load_state_dict
    cnn_model.load_state_dict(torch.load(model_path))


    def prediction(model, train_config, batch_input):
        
        # send model to cpu/cuda according to your system configuration
        model.to(train_config.device)
        
        # it is important to do model.eval() before prediction
        model.eval()

        data = batch_input.to(train_config.device)

        output = model(data)

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]
        
        # get the index of the max probability
        pred_index = prob.data.max(dim=1)[1]
        
        return pred_index.cpu().numpy(), pred_prob.cpu().numpy()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size = 5
    train_config = TrainingConfiguration()

    if torch.cuda.is_available():
        train_config.device = "cuda"
    else:
        train_config.device = "cpu"
        
        

    # load test data without image transformation
    test = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=train_config.data_root, train=False, download=False, 
                    transform=transforms.functional.to_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
        )

    try:
        mean, std = get_mean_std_train_data("data")
        assert len(mean) == len(std) == 3
    except:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    # load testdata with image transformation
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    test_trans = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=train_config.data_root, train=False, download=False, transform=image_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
        )

    for data, _ in test_trans:
        # pass the loaded model
        pred, prob = prediction(cnn_model, train_config, data)
        break
        

    plt.rcParams["figure.figsize"] = (3, 3)
    for images, label in test:
        for i, img in enumerate(images):
            img = transforms.functional.to_pil_image(img)
            plt.imshow(img)
            plt.gca().set_title('Pred: {0}({1:0.2}), Label: {2}'.format(classes[pred[i]], prob[i], classes[label[i]]))
            plt.show()
        break    

    #input("Press Key to exit")