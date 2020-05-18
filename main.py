import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as T

from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from tqdm import tqdm

import copy
from time import time

# Change pretrained model download directory, so it doesn't download every time the runtime restarts
os.environ["TORCH_HOME"] = "./Models/Pretrained"
dataPath = "./Data"
modelsPath = "./Models"
resultsPath = "./Results/"

myModels = ["ResNet18", "AlexNet"]
myDatasets = ["CIFAR-10", "SVHN"]
myActivationFunctions = ["ReLU", "k-WTA"]

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

def setActivationkWTA(model, old_activation, sr=0.2):
    for child_name, child in model.named_children():
        if isinstance(child, old_activation):
            setattr(model, child_name, kWTA(sr))
        else:
            setActivationkWTA(child, old_activation, sr)

mean = 0
var = 1
inputSize = 224

def loadDataset(datasetName, batchSize, mean, var, inputSize):
    transformTrain = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(inputSize),
        T.ToTensor(),
        T.Normalize((mean, mean, mean), (var, var, var))])
    transformTest = T.Compose([
        T.Resize(inputSize),
        T.ToTensor(),
        T.Normalize((mean, mean, mean), (var, var, var))])
    if datasetName == "CIFAR-10":
        trainSet = datasets.CIFAR10(root=dataPath, train=True, download=True, transform=transformTrain)
        testSet = datasets.CIFAR10(root=dataPath, train=False, download=True, transform=transformTest)
    elif datasetName == "SVHN":
        trainSet = datasets.SVHN(root=dataPath, split="train", download=True, transform=transformTrain)
        testSet = datasets.SVHN(root=dataPath, split="test", download=True, transform=transformTest)
    else:
        print("Dataset not present!")
        raise NotImplementedError

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=True)
    return trainLoader, testLoader

def loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel):
    if modelName == "ResNet18":
        if pretrainedModel:
            model = torchvision.models.resnet18(pretrained=True).fc.out_features = 10
        else:
            model = torchvision.models.resnet18(pretrained=False)
    elif modelName == "AlexNet":
        if pretrainedModel:
            model = torchvision.models.alexnet(pretrained=True).classifier[-1].out_features = 10
        else:
            model = torchvision.models.alexnet(pretrained=False)
    else:
        print("Model not present")
        raise NotImplementedError

    model.info = {"name": modelName, "activationFunction": activationFunction, "kWTAsr": kWTAsr}

    if activationFunction == "k-WTA":
        setActivationkWTA(model, kWTA, sr=kWTAsr)

    if os.path.isfile(loadPath):
        model.load_state_dict(torch.load(loadPath))

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

# Trains the model in-place, and saves after every epoch to savePath.
def train(modelName, datasetName, activationFunction, epochs, batchSize, kWTAsr, lr, savePath, loadPath, pretrainedModel, getPlot, mean, var, inputSize):

    trainloader, testloader = loadDataset(datasetName, batchSize, mean, var, inputSize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start = time()

    accValuesTrain = []
    lossValuesTrain = []
    errorValuesTrain = []
    accValuesTest = []
    lossValuesTest = []
    errorValuesTest = []
    for epoch in range(epochs):
        train_acc, train_err, train_loss = performEpoch(trainloader, model, optimizer, device=device, use_tqdm=True)
        print("\n[TRAIN] Epoch:", epoch + 1, ", Accuracy:", train_acc, ", Error:", train_err, ", Loss:", train_loss)

        test_acc, test_err, test_loss = performEpoch(testloader, model, device=device, use_tqdm=True)
        print("[TEST] Epoch:", epoch + 1, ", Accuracy:", test_acc, ", Error:", test_err, ", Loss:", test_loss + "\n")

        # save checkpoint after every epoch
        if savePath != "":
            torch.save(model.state_dict(), savePath)

        accValuesTrain.append(train_acc)
        lossValuesTrain.append(train_loss)
        errorValuesTrain.append(train_err)
        accValuesTest.append(test_acc)
        lossValuesTest.append(test_loss)
        errorValuesTest.append(test_err)

        print(accValuesTrain)

    end = time()
    print("Finished Training. Time Elapsed=%d" % (end - start))

    if getPlot:
        #Plot
        fig = plt.figure(figsize=(15, 4))
        gs = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        plotTitle = "Model: " + model.info["name"] + ", Dataset: " + datasetName + ", Activation Function: " + model.info["activationFunction"]
        if model.info["activationFunction"] == "k-WTA":
            title += " with Sparcity Ratio " + str(model.info["kWTAsr"])
        fig.suptitle(plotTitle)
        ax1.set_title("Accuracy")
        ax2.set_title("Loss")
        ax3.set_title("Error")
        ax1.plot(accValuesTrain, label="Train")
        ax1.plot(accValuesTest, label="Test")
        ax1.legend(loc = "upper left")

        ax2.plot(lossValuesTrain, label="Train")
        ax2.plot(lossValuesTest, label="Test")
        ax2.legend(loc = "upper left")

        ax3.plot(errorValuesTrain, label="Train")
        ax3.plot(errorValuesTest, label="Test")
        ax3.legend(loc = "upper left")

        plotPath = resultsPath + modelName + "_" + datasetName + "_" + model.info["activationFunction"]
        if model.info["activationFunction"] == "k-WTA":
            plotPath += "_" + str(model.info["kWTAsr"])
        plotPath += ".png"
        plt.savefig(plotPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-WTA")
    parser.add_argument("--model", type = str, help = "Model: " + ", ".join(myModels), required = True)
    parser.add_argument("--dataset", type = str, help = "Dataset: " + ", ".join(myDatasets), required = True)
    parser.add_argument("--activation", type = str, help = "Activation function: " + ", ".join(myActivationFunctions), required = True)
    parser.add_argument("--epochs", type = int, help = "Epoch number", required = False, default=80)
    parser.add_argument("--batchSize", type = int, help = "Batch size", required = False, default=32)
    parser.add_argument("--kWTAsr", type = float, help = "k-WTA sparsity ratio", required = False, default=0.2)
    parser.add_argument("--lr", type = float, help = "Learning rate", required = False, default=0.01)
    parser.add_argument("--saveModel", type = bool, help = "Save model", required = False, default=True)
    parser.add_argument("--loadModel", type = bool, help = "Load model", required = False, default=False)
    parser.add_argument("--pretrainedModel", type = bool, help = "Use pretrained model", required = False, default=False)
    parser.add_argument("--getPlot", type = bool, help = "Plot the graph", required = False, default=True)
    parser.add_argument("--mean", type = int, help = "Mean", required = False, default=0)
    parser.add_argument("--var", type = int, help = "Var", required = False, default=1)
    parser.add_argument("--inputSize", type = int, help = "Input size", required = False, default=224)
    args = parser.parse_args()
    if args.model not in myModels:
        print("Invalid model %s" % args.model)
        exit(1)
    if args.dataset not in myDatasets:
        print("Invalid dataset %s" % args.dataset)
        exit(1)
    if args.activation not in myActivationFunctions:
        print("Invalid activation function %s" % args.activation)
        exit(1)

    #Parameters
    modelName = args.model
    datasetName = args.dataset
    activationFunction = args.activation
    epochs = args.epochs
    batchSize = args.batchSize
    kWTAsr = args.kWTAsr
    lr = args.lr
    saveModel = args.saveModel
    restoreModel = args.loadModel
    pretrainedModel = args.pretrainedModel
    getPlot = args.getPlot
    mean = args.mean
    var = args.var
    inputSize = args.inputSize

    savePath = ""
    if saveModel:
        savePath = modelsPath + "/" + modelName + "_" + datasetName + "_" + activationFunction
        if activationFunction == "k-WTA":
            savePath += "_" + str(kWTAsr)
        savePath += ".pth"

    loadPath = ""
    if restoreModel:
        loadPath = modelsPath + "/" + modelName + "_" + datasetName + "_" + activationFunction
        if activationFunction == "k-WTA":
            loadPath += "_" + str(kWTAsr)
        loadPath += ".pth"

    train(modelName, datasetName, activationFunction, epochs, batchSize, kWTAsr, lr, savePath, loadPath, pretrainedModel, getPlot, mean, var, inputSize)
