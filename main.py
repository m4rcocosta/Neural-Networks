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

import foolbox

import copy
from time import time
import datetime

# Change pretrained model download directory, so it doesn't download every time the runtime restarts
os.environ["TORCH_HOME"] = "./Models/Pretrained"
dataPath = "./Data/"
modelsPath = "./Models/"
resultsPath = "./Results/"

tasks = ["train", "test", "attack"]
myModels = ["ResNet18", "AlexNet"]
myDatasets = ["CIFAR-10", "SVHN"]
myActivationFunctions = ["ReLU", "k-WTA"]
myAttacks = ["PGD", "Deepfool"]

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

    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=True)
    return trainLoader, testLoader

def loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel):
    if modelName == "ResNet18":
        if pretrainedModel:
            model = torchvision.models.resnet18(pretrained=True)
            model.fc.out_features = 10
        else:
            model = torchvision.models.resnet18(pretrained=False)
    elif modelName == "AlexNet":
        if pretrainedModel:
            model = torchvision.models.alexnet(pretrained=True)
            model.classifier[-1].out_features = 10
        else:
            model = torchvision.models.alexnet(pretrained=False)

    if activationFunction == "k-WTA":
        setActivationkWTA(model, kWTA, sr=kWTAsr)

    if os.path.isfile(loadPath):
        model.load_state_dict(torch.load(loadPath))

    return model

def performEpoch(loader, model, opt=None, device=None):
    totalCorrect, totalError, totalLoss = 0., 0., 0.
    #correct = 0
    #total = 0
    if opt is None: #Test
        model.eval()
    else: #Train
        model.train()

    pbar = tqdm(total=len(loader))

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)


        outputs = model(X)
        loss = nn.CrossEntropyLoss()(outputs, Y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        #_, predicted = torch.max(outputs.data, 1)
        #total += Y.size(0)
        #correct += (predicted == Y).sum().item()
        totalCorrect += (outputs.max(dim=1)[1] == Y).sum().item()
        totalError += (outputs.max(dim=1)[1] != Y).sum().item()
        totalLoss += loss.item() * X.shape[0]

        pbar.update(1)

    #print("New accuracy: %.1f %%" % (100 * correct / total))
    return totalCorrect / len(loader.dataset), totalError / len(loader.dataset), totalLoss / len(loader.dataset)

# Trains the model in-place, and saves after every epoch to savePath.
def train(modelName, datasetName, activationFunction, epochs, batchSize, kWTAsr, lr, savePath, loadPath, pretrainedModel, getPlot, getResultTxt, mean, var, inputSize):

    trainLoader, testLoader = loadDataset(datasetName, batchSize, mean, var, inputSize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel)

    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    resultModelPath = resultsPath + modelName
    resultModelPath += "_" + datasetName + "_" + activationFunction
    if activationFunction == "k-WTA":
        resultModelPath += "_" + str(kWTAsr)
    if pretrainedModel:
        resultModelPath += "_Pretrained"
    resultFilePath = resultModelPath + ".txt"
    plotPath = resultModelPath + ".png"

    if getResultTxt:
        resultFile = open(resultFilePath, "w")

    accValuesTrain = []
    lossValuesTrain = []
    errorValuesTrain = []
    accValuesTest = []
    lossValuesTest = []
    errorValuesTest = []

    timesPerEpoch = []

    start = time()
    for epoch in range(epochs):

        #Train
        trainStart = time()
        train_acc, train_err, train_loss = performEpoch(trainLoader, model, optimizer, device=device)
        trainEnd = time()
        timesPerEpoch.append(trainEnd-trainStart)
        trainTime = str(datetime.timedelta(seconds=round(trainEnd-trainStart)))
        print("\n[TRAIN] Epoch: " + str(epoch + 1) + ", Accuracy :" + str(train_acc) + ", Error: " + str(train_err) + ", Loss: " + str(train_loss) + ", Time elapsed: " + trainTime)

        #Test
        testStart = time()
        test_acc, test_err, test_loss = performEpoch(testLoader, model, device=device)
        testEnd = time()
        testTime = str(datetime.timedelta(seconds=round(testEnd-testStart)))
        print("[TEST] Epoch: " + str(epoch + 1) + ", Accuracy: " + str(test_acc) + ", Error: " + str(test_err) + ", Loss: " + str(test_loss) + ", Time elapsed: " + testTime)

        if getResultTxt:
            print("\n[TRAIN] Epoch: " + str(epoch + 1) + ", Accuracy :" + str(train_acc) + ", Error: " + str(train_err) + ", Loss: " + str(train_loss) + ", Time elapsed: " + trainTime, file = resultFile)
            print("[TEST] Epoch: " + str(epoch + 1) + ", Accuracy: " + str(test_acc) + ", Error: " + str(test_err) + ", Loss: " + str(test_loss) + ", Time elapsed: " + testTime, file = resultFile)

        # save checkpoint after every epoch
        if savePath != "":
            torch.save(model.state_dict(), savePath)

        accValuesTrain.append(train_acc)
        lossValuesTrain.append(train_loss)
        errorValuesTrain.append(train_err)
        accValuesTest.append(test_acc)
        lossValuesTest.append(test_loss)
        errorValuesTest.append(test_err)

    end = time()
    totalTime = str(datetime.timedelta(seconds=round(end-start)))
    timesPerEpoch = np.array(timesPerEpoch)
    averageTimePerEpoch = str(datetime.timedelta(seconds=round(timesPerEpoch.sum()/len(timesPerEpoch))))
    print("Finished Training. Total time Elapsed: " + totalTime + ", average time per epoch: " + averageTimePerEpoch)
    if getResultTxt:
        print("Time Elapsed: " + totalTime + ", average time per epoch: " + averageTimePerEpoch, file = resultFile)
        resultFile.close()

    if getPlot:
        #Plot
        fig = plt.figure(figsize=(15, 4))
        gs = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        plotTitle = "Model: " + modelName
        if pretrainedModel:
            plotTitle += " Pretrained"
        plotTitle += ", Dataset: " + datasetName + ", Activation Function: " + activationFunction
        if activationFunction == "k-WTA":
            plotTitle += " with Sparcity Ratio " + str(kWTAsr)
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

        plt.savefig(plotPath)

def test(modelName, datasetName, activationFunction, batchSize, kWTAsr, loadPath, pretrainedModel, mean, var, inputSize):

    if os.path.isfile(loadPath):
        trainLoader, testLoader = loadDataset(datasetName, batchSize, mean, var, inputSize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel)
        model.to(device)

        start = time()
        test_acc, test_err, test_loss = performEpoch(testLoader, model, device=device)
        end = time()
        timeElapsed =  str(datetime.timedelta(seconds=round(end-start)))
        print("Model " + modelName + "trained on " + datasetName + " with activation function " + activationFunction)
        print("Accuracy: " + str(test_acc) + ", Error: " + str(test_err) + ", Loss: " + str(test_loss))
        print("Time Elapsed: " + timeElapsed)

    else:
        print("Model doesn't exist! Train the model first...")

def testAdversial(modelName, datasetName, activationFunction, batchSize, kWTAsr, loadPath, pretrainedModel, mean, var, inputSize, attackType, attackBatches):

    if os.path.isfile(loadPath):
        trainLoader, testLoader = loadDataset(datasetName, batchSize, mean, var, inputSize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = loadModel(modelName, activationFunction, kWTAsr, loadPath, pretrainedModel)
        model.to(device)

        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        # variables to keep track of foolbox attack's stats
        robust_acc_sum = 0

        descBar = tqdm(bar_format="{desc}")
        descBar.set_description("Minibatch 0: n.d.")
        descBar.refresh()
        pbar = tqdm(total=attackBatches)
        for j, data in enumerate(testLoader, 0):
            if j >= attackBatches:
                break

            images, labels = data[0].to(device), data[1].to(device)

            fmodel = foolbox.PyTorchModel(model, bounds=(0,1))

            # Using parameters specified in paper's appendix D.1
            if attackType == "PGD":
                attack_fn = foolbox.attacks.LinfPGD(steps=40, random_start=True, rel_stepsize=0.003)
            elif attackType == "Deepfool":
                attack_fn = foolbox.attacks.LinfDeepFoolAttack(steps=20, candidates=10)
            epsilons = [0.031] # value used in the paper
            _, _, success = attack_fn(fmodel, images, labels, epsilons=epsilons)

            robust_accuracy = 1 - success.double().mean(axis=-1)
            robust_acc_sum += robust_accuracy

            barText = "Minibatch %d: %.2f%%" % (j+1, 100*robust_accuracy.item())
            descBar.set_description(barText)
            descBar.refresh()

            pbar.update(1)

            #print("[Minibatch: %d] Accuracy: %.2f %%" % (j+1, 100*robust_accuracy.item()))

        #print("Robustness Accuracy: ", 100 * robust_acc_sum.item() / attackBatches, "%")
        barText = "Robustness Accuracy: %.2f%%" % (100 * robust_acc_sum.item() / attackBatches)
        descBar.set_description(barText)
        descBar.refresh()

    else:
        print("Model doesn't exist! Train the model first...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-WTA")
    parser.add_argument("--task", type = str, help = "Task: " + ", ".join(tasks), required = True)
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
    parser.add_argument("--log", type = bool, help = "Log file", required = False, default=True)
    parser.add_argument("--mean", type = int, help = "Mean", required = False, default=0)
    parser.add_argument("--var", type = int, help = "Var", required = False, default=1)
    parser.add_argument("--inputSize", type = int, help = "Input size", required = False, default=224)
    parser.add_argument("--attack", type = str, help = "Attack: " + ", ".join(myAttacks), required = False, default = "")
    parser.add_argument("--batches", type = int, help = "Attack batshes number", required = False, default=20)
    args = parser.parse_args()
    if args.task not in tasks:
        print("Invalid task %s" % args.task)
        exit(1)
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
    task = args.task
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
    getResultTxt = args.log
    mean = args.mean
    var = args.var
    inputSize = args.inputSize
    attackType = args.attack
    attackBatches = args.batches

    if task == "attack" and attackType not in myAttacks:
        print("Invalid attack type %s" % args.attack)
        exit(1)

    savePath = ""
    if saveModel:
        savePath = modelsPath +  modelName + "_" + datasetName + "_" + activationFunction
        if activationFunction == "k-WTA":
            savePath += "_" + str(kWTAsr)
        if pretrainedModel:
            savePath += "_Pretrained"
        savePath += ".pth"

    loadPath = ""
    if restoreModel:
        loadPath = modelsPath +  modelName + "_" + datasetName + "_" + activationFunction
        if activationFunction == "k-WTA":
            loadPath += "_" + str(kWTAsr)
        if pretrainedModel:
            loadPath += "_Pretrained"
        loadPath += ".pth"

    if task == "train":
        train(modelName, datasetName, activationFunction, epochs, batchSize, kWTAsr, lr, savePath, loadPath, pretrainedModel, getPlot, getResultTxt, mean, var, inputSize)
    elif task == "test":
        test(modelName, datasetName, activationFunction, batchSize, kWTAsr, loadPath, pretrainedModel, mean, var, inputSize)
    elif task == "attack":
        testAdversial(modelName, datasetName, activationFunction, batchSize, kWTAsr, loadPath, pretrainedModel, mean, var, inputSize, attackType, attackBatches)
