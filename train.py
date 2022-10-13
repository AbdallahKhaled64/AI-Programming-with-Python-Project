import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import os
import fullcode

p = argparse.ArgumentParser(description='Train.py')
p.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
p.add_argument('--gpu', dest="gpu", action="store", default="gpu")
p.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
p.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
p.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
p.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
p.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
p.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = p.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, v_loader, testloader = fullcode.load_data(where)


model, optimizer, criterion = fullcode.nn_setup(structure,dropout,hidden_layer1,lr,power)


fullcode.train_network(model, optimizer, criterion, epochs, trainloader, power)


fullcode.save_checkpoint(path,structure,hidden_layer1,dropout,lr)
print("The Model is trained")