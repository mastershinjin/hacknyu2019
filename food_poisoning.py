import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 224
epochs = 400
data_path = 'data'
model_path = ''

class Classifier(nn.Module):
    """VGG tensor to binary classification"""
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)


loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def image_loader(image_name):
    """convert PIL image to model-compatible tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# set up for training
cnn = models.vgg19(pretrained=True).features.to(device).eval()

classifier = Classifier()
# load from file
if(model_path):
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

loss_f = nn.MSELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# load images
img_list = []
output_list = []
with open(os.path.join(data_path, "train.csv"), 'r') as f:
    data = f.read().split('\n')
    for entry in data:
        vals = entry.split(',')
        img_list.append(os.path.join(data_path, vals[0]))
        output_list.append(torch.Tensor([int(vals[1])]).squeeze())

# TRAIN
print("Started training...")
total_loss = 0

for i in range(epochs):
    # choose random image
    idx = random.randint(0, len(output_list)-1)
    inputs = image_loader(img_list[idx])
    outputs = output_list[idx]

    # run iteration
    optimizer.zero_grad()

    cnn_output = cnn(inputs)
    outputs = classifier(cnn_output.view(1, -1))

    loss = loss_f(inputs, outputs)
    loss.backward()

    optimizer.step()

    # save model
    if i % 100 == 0:
        torch.save(classifier.state_dict(), os.path.join(data_path, f"models/classifier-{i}-{loss}"))

    print(f"Epoch: {i}\tLoss: {loss}")
