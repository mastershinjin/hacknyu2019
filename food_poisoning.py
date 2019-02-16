import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import sys
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 224
epochs = 400
data_path = 'data'
model_path = 'models/classifier-299'
is_train = False
is_test = False

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
    with open(os.path.join(data_path, f"{fname}"), 'r') as f:
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
if model_path:
    classifier.load_state_dict(torch.load(model_path))
    start = int(model_path.split('-')[1])
    classifier.eval()

def train():
    # === TRAIN ===
    print("Started training...")

    train_list, target_list = load_from_csv('train.csv')
    loss_f = nn.MSELoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    total_loss = 0
    losses = []

    for i in range(start, epochs):
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
            torch.save(classifier.state_dict(), f"models/classifier-{i}")

        print(f"Epoch: {i}\tLoss: {round(loss.item(), 2)}\tAvg. Loss: {round(total_loss / (i+1), 2)}")

# === TEST ===
def test():
    with torch.no_grad():
        test_inputs, test_outputs = load_from_csv('test.csv')

        if model_path:
            classifier.load_state_dict(torch.load(model_path))
            classifier.eval()
        
        print("Started testing...")
        correct = 0

        for idx in range(len(test_inputs)):
            # choose random image
            inputs = image_loader(test_inputs[idx])
            target = test_outputs[idx]

            outputs = classifier(cnn(inputs).view(1, -1))

            res = round(outputs.item())
            if res == target.item():
                correct += 1

            print(f"Progress {round(float(idx) / len(test_inputs), 2)}\tPredict: {round(outputs.item(), 2)}\tActual: {round(target.item(), 2)}")

        print(f"Correct: {correct}\tPercent: {float(correct) / len(test_inputs)}")

# === RUN ===
def run(path):
    
    if model_path:
            classifier.load_state_dict(torch.load(model_path))
            classifier.eval()

    im = image_loader(path)
    outputs = classifier(cnn(im).view(1, -1))
    print("rotten" if outputs.item() >= 0.5 else "not rotten")

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 2 and sys.argv[1] == "test":
        if len(sys.argv) == 3 and sys.argv[2]:
            model_path = sys.argv[2]
        test()
    elif len(sys.argv) >= 2 and sys.argv[1] == "run":
        if len(sys.argv) == 4 and sys.argv[3]:
            model_path = sys.argv[3]
        run(sys.argv[2])
    else:
        print("Usage:")
        print("To train:\tfood_poisoning.py train")
        print("To test:\tfood_poisoning.py test [model path]")
        print("To run:\t\tfood_poisoning.py run [image path] [model path]")
