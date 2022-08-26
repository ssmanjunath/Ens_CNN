import os 
import argparse
import torch.optim as optim
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torchsummary import summary
from PIL import Image
from ema import EMA
from mnist import mnist
from transforms import Randomrotate
from modelm.modelm3 import ModelM3
from modelm.modelm5 import ModelM5
from modelm.modelm7 import ModelM7

def run(seed = 0, kernelsize = 5, epochs = 150, logdir = "tmp"):
    # random seed
    SEED = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    
    #variables
    kernel_size = kernelsize
    num_epochs = epochs

    #log path
    if not os.path.exists(f"../logs/{logdir}"):
        os.makedirs(f"../logs/{logdir}") 
    OUTPUT_FILE = str(f"../logs/{logdir}/log{SEED:03d}.out")
    MODEL_FILE = str(f"../logs/{logdir}/model{SEED:03d}.pth")
    
    #device to run nn
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print('Warning: CPU is used for training')
        exit(0)
    
    #Data Augmentation
    transform = transforms.Compose([Randomrotate(20,seed =SEED),
                transforms.RandomAffine(0,translate=(0.2,0.2))])
    
    #Load dataset
    test_dataset = mnist(training=False,transform=None)
    train_dataset = mnist(training=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 120, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle= False)

    #model selection 
    if kernel_size ==3:
        model = ModelM3.to(device=device)
    elif kernel_size==5:
        model = ModelM5.to(device=device)
    else:
        model = ModelM7.to(device=device)
    
    #Hyperparameter optimisation
    ema = EMA(model,decay=0.999)
    optimiser = optim.Adam(model.parameters() ,lr=0.001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser,gamma=0.98)

    #out file 
    f = open(OUTPUT_FILE,'W')
    f.close()
    
    #global var
    g_step = 0
    max_correct = 0

    #training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_corr = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data,target = data.to(device), target.to(device,torch.int64)
            optimiser.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            train_pred = output.argmax(dim =1, Keepdim = True)
            train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(output,target,reduction='sum').item()
            loss.backward()
            optimiser.step()
            g_step +=1
            ema(model,g_step)
            