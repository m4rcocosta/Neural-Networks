# Neural Networks
Project for Neural Network course

<a href="https://www.dis.uniroma1.it/"><img src="http://www.dis.uniroma1.it/sites/default/files/marchio%20logo%20eng%20jpg.jpg" width="1000"></a>

## Team
* Andrea Antonini <a href="https://github.com/AndreaAntonini"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="20"></a>

* Marco Costa <a href="https://github.com/marcocosta96"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/1024px-Octicons-mark-github.svg.png" width="20"></a>
<a href="https://www.linkedin.com/in/marco-costa-ecs"><img src="https://www.tecnomagazine.it/tech/wp-content/uploads/2013/05/linkedin-aggiungere-immagini.png" width="20"></a>

## Requirements
- pytorch
- torchvision
- numpy
- matplotlib
- tqdm
- requests
- foolbox

## Usage
* Run the script with:
    * > python main.py
* Specify the required parameters:
    * > --task [train|test|testAdversial] #specify the task executed
    * > --model [ResNet18|AlexNet] #choose the model
    * > --dataset [CIFAR-10|SVGH] #choose the dataset
    * > --activation [ReLU|k-WTA] #choose the activation function
    * > --attack [PGD|Deepfool] #choose the attack type (Needed only with *--task testAdversial*)
* Change the default parameters if needed:
    * > --epochs (default *80*) #choose the epoch number in train
    * > --batchSize (default *32*) #choose the batch size
    * > --kWTAsr (default *0.2*) #choose the sparsity ratio when using k-WTA activation function
    * > --lr (default *0.01*) #choose the learning rate
    * > --saveModel (default *True*) #choose whether to save the model
    * > --loadModel (default *False*) #choose whether to load the model
    * > --pretrainedModel (default *False*) #choose whether to load a pretrained model
    * > --getPlot (default *True*) #choose whether to plot a graph showing the train
    * > --log (default *True*) #choose whether to print a log file
    * > --mean (default *0*) #choose the mean
    * > --var (default *1*) #choose the variance
    * > --inputSize (default *224*) #choose the input size, useful for pretrained models which need 224x224 images
    * > --batches (default *20*) #choose the batch number while testing adversarial attacks

Example:
> python main.py --task train --model ResNet18 --dataset CIFAR-10 --activation ReLU --batchSize 64
