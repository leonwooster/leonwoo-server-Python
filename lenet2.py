import matplotlib.pyplot as plt  # one of the best graphics library for python
import os

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchvision import datasets, transforms
from lenetNetwork import LeNet5
from lenetConfig import TrainingConfiguration

def prediction(model, train_config, batch_input):
    
    # send model to cpu/cuda according to your system configuration
    model.to(train_config.device)
    
    # it is important to do model.eval() before prediction
    model.eval()

    data = batch_input.to(train_config.device)

    output = model(data)

    # get probability score using softmax
    prob = F.softmax(output, dim=1)

    # get the max probability
    pred_prob = prob.data.max(dim=1)[0]
    
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()

if __name__ == "__main__":
    # initialize the model
    lenet5_mnist = LeNet5()

    # loading the model and getting model parameters by using load_state_dict
    models = 'models'
    model_file_name = 'lenet5_mnist.pt'
    model_path = os.path.join(models, model_file_name)

    lenet5_mnist.load_state_dict(torch.load(model_path))

    batch_size = 5
    train_config = TrainingConfiguration()

    if torch.cuda.is_available():
        train_config.device = "cuda"
    else:
        train_config.device = "cpu"

    # load test data without image transformation
    test = torch.utils.data.DataLoader(
        datasets.MNIST(root=train_config.data_root, train=False, download=True, 
                    transform=transforms.functional.to_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
        )

    # load testdata with image transformation
    image_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    test_trans = torch.utils.data.DataLoader(
        datasets.MNIST(root=train_config.data_root, train=False, download=True, transform=image_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
        )

    for data, _ in test_trans:
        # pass the loaded model
        pred, prob = prediction(lenet5_mnist, train_config, data)
        break
        

    plt.rcParams["figure.figsize"] = (3, 3)
    for images, _ in test:
        for i, img in enumerate(images):
            img = transforms.functional.to_pil_image(img)
            plt.imshow(img, cmap='gray')
            plt.gca().set_title('Prediction: {0}, Prob: {1:.2}'.format(pred[i], prob[i]))
            plt.show()
        break