import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

import copy
from time import time

# Change pretrained model download directory, so it doesn't
# download every time the runtime restarts
os.environ["TORCH_HOME"] = "./models/pretrained"

class kWTA(nn.Module):
    def __init__(self, sr):
        super(kWTA, self).__init__()
        self.sr = sr

    # Paper's forward implementation
    def forward(self, x):
        tmpx = x.view(x.shape[0], -1)
        size = tmpx.shape[1]
        k = int(self.sr * size)

        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x

def activation_to_kwta(model, old_activation, sr=0.2):
    for child_name, child in model.named_children():
        if isinstance(child, old_activation):
            setattr(model, child_name, kWTA(sr))
        else:
            activation_to_kwta(child, old_activation, sr)

# We have four types of models stored here:
# - pretrained: Default models from pytorch, trained on ImageNet
# - relu: Fine-tuned models for CIFAR10
# - kwta_0_1: Models using kwta activation with sparsity=0.1 for CIFAR10
# - kwta_0_2: Models using kwta activation with sparsity=0.2 for CIFAR10
models = {"pretrained": {}, "relu": {}, "kwta-0.1": {}, "kwta-0.2": {}}

# Download and load pretrained models (trained for ImageNet dataset)
models["pretrained"]["resnet"] = torchvision.models.resnet18(pretrained=True)
models["pretrained"]["alexnet"] = torchvision.models.alexnet(pretrained=True)
#models["pretrained"]["densenet"] = torchvision.models.densenet121(pretrained=True)
#models["pretrained"]["wide_resnet"] = torchvision.models.wide_resnet50_2(pretrained=True)
#models["pretrained"]["vgg"] = torchvision.models.vgg11(pretrained=True)
#models["pretrained"]["squeezenet"] = torchvision.models.squeezenet1_1(pretrained=True)

## ResNet ReLU
models["relu"]["resnet"] = copy.deepcopy(models["pretrained"]["resnet"])
models["relu"]["resnet"].fc.out_features = 10
if os.path.isfile("./models/relu/ResNet18.pth"):
    models["relu"]["resnet"].load_state_dict(torch.load("./models/relu/ResNet18.pth"))

## ResNet kWTA 0.1
models["kwta-0.1"]["resnet"] = copy.deepcopy(models["relu"]["resnet"])
activation_to_kwta(models["kwta-0.1"]["resnet"], kWTA, sr=0.1)
if os.path.isfile("./models/kwta-0.1/ResNet18.pth"):
    models["kwta-0.1"]["resnet"].load_state_dict(torch.load("./models/kwta-0.1/ResNet18.pth"))

## ResNet kWTA 0.2
models["kwta-0.2"]["resnet"] = copy.deepcopy(models["kwta-0.1"]["resnet"])
activation_to_kwta(models["kwta-0.2"]["resnet"], nn.ReLU, sr=0.2)
if os.path.isfile("./models/kwta-0.2/ResNet18.pth"):
    models["kwta-0.2"]["resnet"].load_state_dict(torch.load("./models/kwta-0.2/ResNet18.pth"))

## AlexNet ReLU
models["relu"]["alexnet"] = copy.deepcopy(models["pretrained"]["alexnet"])
models["relu"]["alexnet"].classifier[-1].out_features = 10
if os.path.isfile("./models/relu/AlexNet.pth"):
    models["relu"]["alexnet"].load_state_dict(torch.load("./models/relu/AlexNet.pth"))

## AlexNet kWTA 0.1
models["kwta-0.1"]["alexnet"] = copy.deepcopy(models["relu"]["alexnet"])
activation_to_kwta(models["kwta-0.1"]["alexnet"], kWTA, sr=0.1)
if os.path.isfile("./models/kwta-0.1/AlexNet.pth"):
    models["kwta-0.1"]["alexnet"].load_state_dict(torch.load("./models/kwta-0.1/AlexNet.pth"))

## AlexNet kWTA 0.2
models["kwta-0.2"]["alexnet"] = copy.deepcopy(models["kwta-0.1"]["alexnet"])
activation_to_kwta(models["kwta-0.2"]["alexnet"], nn.ReLU, sr=0.2)
if os.path.isfile("./models/kwta-0.2/AlexNet.pth"):
    models["kwta-0.2"]["alexnet"].load_state_dict(torch.load("./models/kwta-0.2/AlexNet.pth"))

MEAN = 0
VAR = 1
INPUT_SIZE = 224

def performEpoch(loader, model, opt=None, device=None, use_tqdm=True):
    totalAccuracy, totalError, totalLoss = 0., 0., 0.
    if opt is None: #Test
        model.eval()
    else: #Train
        model.train()

    if use_tqdm:
        pbar = tqdm(total=len(loader))

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)


        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, Y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        max_vals, max_indices = torch.max(model(X),1)
        totalAccuracy = (max_indices == Y).sum().item() / max_indices.size()[0]
        totalError += (yp.max(dim=1)[1] != Y).sum().item()
        totalLoss += loss.item() * X.shape[0]

        if use_tqdm:
            pbar.update(1)

    return totalAccuracy, totalError / len(loader.dataset), totalLoss / len(loader.dataset)

# Trains the model in-place, and saves after every epoch to save_path.
# Only trains 1 epoch by default
def train(model, save_path, lr=0.1, epochs=1, batchSize=64):
    transform_train = T.Compose(
        [T.RandomCrop(32, padding=4),
         T.RandomHorizontalFlip(),
         T.Resize(INPUT_SIZE),
         T.ToTensor(),
         T.Normalize((MEAN,MEAN,MEAN), (VAR,VAR,VAR))])
    transform_test = T.Compose(
        [T.Resize(INPUT_SIZE),
         T.ToTensor(),
         T.Normalize((MEAN,MEAN,MEAN), (VAR,VAR,VAR))])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchSize, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start = time()

    for epoch in range(epochs):
        train_acc, train_err, train_loss = performEpoch(trainloader, model, optimizer, device=device, use_tqdm=True)
        print("[TRAIN] Epoch:", epoch + 1, ", Accuracy:", train_acc, ", Error:", train_err, ", Loss:", train_loss)

        # save checkpoint after every epoch
        torch.save(model.state_dict(), save_path)

        test_acc, test_err, test_loss = performEpoch(testloader, model, device=device, use_tqdm=True)
        print("[TEST] Epoch:", epoch + 1, ", Accuracy:", test_acc, ", Error:", test_err, ", Loss:", test_loss)

    end = time()
    print("Finished Training. Time Elapsed=%d" % (end - start))

train(models["relu"]["resnet"], lr=0.01, epochs=2, save_path="./models/relu/ResNet18.pth", batchSize=32)
