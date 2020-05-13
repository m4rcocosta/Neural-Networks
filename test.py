
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import json
import time
import sys
import copy

from kWTA import models
from kWTA import activation
from kWTA import attack
from kWTA import training
from kWTA import utilities
from kWTA import densenet
from kWTA import resnet
from kWTA import wideresnet

norm_mean = 0
norm_var = 1
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])
cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(cifar_train, batch_size = 128, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = 100, shuffle=True)

# ReLU ResNet 18
device = torch.device('cuda:0')
model = resnet.ResNet18().to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
eps = 0.031
for ep in range(80):
    if ep == 50:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.01
    train_err, train_loss = training.epoch(train_loader, model, opt, device=device, use_tqdm=True)
    test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)
    adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20,
        use_tqdm=True, epsilon=eps, randomize=True, alpha=0.003, n_test=1000)
    print('epoch', ep, 'train err', train_err, 'test err', test_err, 'adv_err', adv_err)
    torch.save(model.state_dict(), 'models/resnet18_cifar.pth')
