#!/bin/bash

# Train nets
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --lr 0.1
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --lr 0.1
python main.py --task train --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --lr 0.1

python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --lr 0.1
python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --lr 0.1
python main.py --task train --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --lr 0.1

python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --lr 0.1
python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --lr 0.1
python main.py --task train --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --lr 0.1

python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --epochs 10 --lr 0.01
python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --lr 0.1
python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --lr 0.1
python main.py --task train --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --lr 0.1

# Test Adversarial
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack PGD --batches 20 --loadModel True

python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack PGD --batches 20 --loadModel True

python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack PGD --batches 20 --loadModel True

python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack PGD --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack PGD --batches 20 --loadModel True

python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack Deepfool --batches 20 --loadModel True

python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model ResNet18 --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack Deepfool --batches 20 --loadModel True

python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation ReLU --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset CIFAR-10 --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack Deepfool --batches 20 --loadModel True

python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation ReLU --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.2 --pretrained True --attack Deepfool --batches 20 --loadModel True
python main.py --task attack --model AlexNet --dataset SVHN --batchSize 64 --activation k-WTA --kWTAsr 0.1 --pretrained True --attack Deepfool --batches 20 --loadModel True
