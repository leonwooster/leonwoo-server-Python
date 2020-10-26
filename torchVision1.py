# Import torch and torchvision modules
import torch
from torchvision import models

import wget

from PIL import Image
import matplotlib.pyplot as plt

# Specify image transformations
from torchvision import transforms
from torchvision.utils import save_image

print(dir(models))

# Load alexnet model
alexnet = models.alexnet(pretrained=True)

print(alexnet)

# Put our model in eval mode
alexnet.eval()


# Download classes text file
#wget.download("https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt")

# Load labels
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

print(classes[:5])

#wget.download("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/800px-Grosser_Panda.JPG")

img = Image.open("cock2.jpg")
plt.imshow(img)

transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                # Ensure images are of size 224x224
 transforms.ToTensor(),                     # Convert the image to float tensor of range [0,1]
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                # Normalize data with the mean
 std=[0.229, 0.224, 0.225]                  # Normalize data with the std
 )])

# Apply the transform to the input image
img_t = transform(img)

save_image(img_t, "panda.jpg")

# Create a mini-batch 
batch_t = torch.unsqueeze(img_t, 0)

# Carry out inference
out = alexnet(batch_t)
print(out.shape)
print(out)

_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
for idx in indices[0][:5]:
    print("Class:{}, Class Name: {}, Confidence: {:.4f}%".format(idx,classes[idx], percentage[idx].item()))

#k = input('Press Enter to exit')