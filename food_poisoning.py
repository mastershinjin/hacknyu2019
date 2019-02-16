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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def image_loader(image_name):
    """Convert PIL image to model-compatible tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def load_from_csv(fname):
    """Load data from csv"""
    input_list = []
    output_list = []
    with open(os.path.join(data_path, "train.csv"), 'r') as f:
        data = f.read().split('\n')
        for entry in data:
            vals = entry.split(',')
            input_list.append(os.path.join(data_path, vals[0]))
            output_list.append(torch.Tensor([int(vals[1])]).unsqueeze(1))

    return input_list, output_list

# set up for training
cnn = models.vgg19(pretrained=True).features.to(device).eval()

classifier = Classifier()
# load from file
if(model_path):
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

loss_f = nn.MSELoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

# load images
train_list, target_list = load_from_csv('train.csv')

# === TRAIN ===
print("Started training...")
total_loss = 0
losses = []

for i in range(epochs):
    # choose random image
    idx = random.randint(0, len(train_list)-1)
    inputs = image_loader(train_list[idx])
    target = target_list[idx]

    # run iteration
    optimizer.zero_grad()

    outputs = classifier(cnn(inputs).view(1, -1))

    loss = loss_f(outputs, target)
    total_loss += loss.item()
    losses.append(loss.item())
    
    loss.backward()

    optimizer.step()

    # save model
    if i % 100 == 99:
        torch.save(classifier.state_dict(), os.path.join(data_path, f"models/classifier-{i}-{loss}"))

    print(f"Epoch: {i}\tLoss: {loss}\tAvg. Loss: {total_loss / (i+1)}")

# === TEST ===
with torch.no_grad():
    test_inputs, test_outputs = load_from_csv('test.csv')
    
    print("Started testing...")
    correct = 0

    for idx in range(len(test_inputs)):
        # choose random image
        inputs = image_loader(img_list[idx])
        target = output_list[idx]

        outputs = classifier(cnn(inputs).view(1, -1))

        res = round(outputs.item())
        if res == target.item():
            correct += 1

    print("Correct: {correct}\tPercent: {float(correct) / len(test_inputs)}")
