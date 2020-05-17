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

# Change pretrained model download directory, so it doesn't download every time the runtime restarts
os.environ["TORCH_HOME"] = "./models/pretrained"

class kWTA(nn.Module):
    def __init__(self, sr):
        super(kWTA, self).__init__()
        self.sr = sr

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

MEAN = 0
VAR = 1
INPUT_SIZE = 224

def loadDataset(datasetName, batchSize):
    transformTrain = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize((MEAN, MEAN, MEAN), (VAR, VAR, VAR))])
    transformTest = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize((MEAN, MEAN, MEAN), (VAR, VAR, VAR))])
    if datasetName == "cifar10":
        trainSet = datasets.CIFAR10(root="./data", train=True, download=True, transform=transformTrain)
        testSet = datasets.CIFAR10(root="./data", train=False, download=True, transform=transformTest)
    else:
        raise NotImplementedError

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=True)
    return trainLoader, testLoader

def loadModel(modelName, usekWTA=False, kWTAsr=0.2, path="", pretrained=False):
    if modelName == "resnet18":
        if pretrained:
            model = torchvision.models.resnet18(pretrained=True).fc.out_features = 10
        else:
            model = torchvision.models.resnet18(pretrained=False)
    elif modelName == "alexnet":
        if pretrained:
            model = torchvision.models.alexnet(pretrained=True).classifier[-1].out_features = 10
        else:
            model = torchvision.models.alexnet(pretrained=False)
    else:
        raise NotImplementedError

    if usekWTA:
        activation_to_kwta(model, kWTA, sr=kWTAsr)

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    return model


def performEpoch(loader, model, opt=None, device=None, use_tqdm=True):
    totalAccuracy, totalError, totalLoss = 0., 0., 0.
    if opt is None: #Test
        model.eval()
    else: #Train
        model.train()

    if use_tqdm: #Show progress bar
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
def train(model, save_path, datasetName, lr=0.1, epochs=1, batchSize=64):

    trainloader, testloader = loadDataset(datasetName, batchSize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start = time()

    for epoch in range(epochs):
        train_acc, train_err, train_loss = performEpoch(trainloader, model, optimizer, device=device, use_tqdm=True)
        print("[TRAIN] Epoch:", epoch + 1, ", Accuracy:", train_acc, ", Error:", train_err, ", Loss:", train_loss)

        test_acc, test_err, test_loss = performEpoch(testloader, model, device=device, use_tqdm=True)
        print("[TEST] Epoch:", epoch + 1, ", Accuracy:", test_acc, ", Error:", test_err, ", Loss:", test_loss)

        # save checkpoint after every epoch
        torch.save(model.state_dict(), save_path)

    end = time()
    print("Finished Training. Time Elapsed=%d" % (end - start))

if __name__ == "__main__":
    model = loadModel("resnet18")
    train(model, datasetName="cifar10", lr=0.01, epochs=2, save_path="./models/relu/ResNet18.pth", batchSize=32)
