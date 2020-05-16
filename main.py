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

import foolbox

from PIL import Image

import copy
from time import time

# Change pretrained model download directory, so it doesn't
# download every time the runtime restarts
os.environ['TORCH_HOME'] = './models/pretrained'

# We have four types of models stored here:
# - pretrained: Default models from pytorch, trained on ImageNet
# - relu: Fine-tuned models for CIFAR10
# - kwta_0_1: Models using kwta activation with sparsity=0.1 for CIFAR10
# - kwta_0_2: Models using kwta activation with sparsity=0.2 for CIFAR10
models = {'pretrained': {}, 'relu': {}, 'kwta_0_1': {}, 'kwta_0_2': {}}

# Download and load pretrained models (trained for ImageNet dataset)
models['pretrained']['resnet'] = torchvision.models.resnet18(pretrained=True)
models['pretrained']['densenet'] = torchvision.models.densenet121(pretrained=True)
models['pretrained']['wide_resnet'] = torchvision.models.wide_resnet50_2(pretrained=True)
models['pretrained']['vgg'] = torchvision.models.vgg11(pretrained=True)
models['pretrained']['alexnet'] = torchvision.models.alexnet(pretrained=True)
models['pretrained']['squeezenet'] = torchvision.models.squeezenet1_1(pretrained=True)

## AlexNet ReLU
#models['relu']['alexnet'] = copy.deepcopy(models['pretrained']['alexnet'])
#models['relu']['alexnet'].classifier[-1].out_features = 10
#models['relu']['alexnet'].load_state_dict(torch.load('./models/relu/AlexNet.pth'))

## AlexNet kWTA 0.2
#models['kwta-0.2']['alexnet'] = copy.deepcopy(models['relu']['alexnet'])
#activation_to_kwta(models['kwta-0.2']['alexnet'], nn.ReLU, sr=0.2)
#models['kwta-0.2']['alexnet'].load_state_dict(torch.load('./models/kwta-0.2/AlexNet.pth'))

## AlexNet kWTA 0.1
#models['kwta-0.1']['alexnet'] = copy.deepcopy(models['kwta-0.2']['alexnet'])
#activation_to_kwta(models['kwta-0.1']['alexnet'], kWTA, sr=0.1)
#models['kwta-0.1']['alexnet'].load_state_dict(torch.load('./models/kwta-0.1/AlexNet.pth'))


## ResNet ReLU
#models['relu']['resnet'] = copy.deepcopy(models['pretrained']['resnet'])
#models['relu']['resnet'].fc.out_features = 10
#models['relu']['resnet'].load_state_dict(torch.load('./models/relu/ResNet18.pth'))

## ResNet kWTA 0.2
#models['kwta-0.2']['resnet'] = copy.deepcopy(models['relu']['resnet'])
#activation_to_kwta(models['kwta-0.2']['resnet'], nn.ReLU, sr=0.2)
#models['kwta-0.2']['resnet'].load_state_dict(torch.load('./models/kwta-0.2/ResNet18.pth'))

## ResNet kWTA 0.1
#models['kwta-0.1']['resnet'] = copy.deepcopy(models['kwta-0.2']['resnet'])
#activation_to_kwta(models['kwta-0.1']['resnet'], kWTA, sr=0.1)
#models['kwta-0.1']['resnet'].load_state_dict(torch.load('./models/kwta-0.1/ResNet18.pth'))

MEAN = 0
VAR = 1
INPUT_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

# Trains the model in-place, and saves after every epoch to save_path.
# Only trains 1 epoch by default
def train(model, save_path, lr=0.01, epochs=1):
    model = model.to(device) # use CUDA
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            # had to add clipping to fix exploding gradients:
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # print statistics
            running_loss += loss.item() / 200
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

        # save checkpoint after every epoch
        torch.save(model.state_dict(), save_path)

    print('Finished Training')

# Test the model
def test(net):
    net = net.to(device)
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.1f %%' % (
        100 * correct / total))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

start = time()

train(torchvision.models.resnet18(pretrained=True), lr=0.001, epochs=1, save_path='./models/relu/ResNet18_bench.pth')

end = time()
print("Time Elapsed=%d" % (end - start))

# Track time for benchmarking
start = time()

test(models['relu']['resnet'])

end = time()
print("Time Elapsed=%d" % (end - start))
