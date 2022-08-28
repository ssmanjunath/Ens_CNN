from email.policy import default
import sys
import os
import argparse
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from PIL import Image
from ema import EMA
from mnist import mnist
from transforms import RandomRotation
from modelm.modelm3 import ModelM3
from modelm.modelm5 import ModelM5
from modelm.modelm7 import ModelM7

def run(seed=0 , epochs=150, Kernel_size=5, logdir='tmp'):
    

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda'if use_cuda else 'gpu')
    if use_cuda == False:
        print('Warning! CPU will be used for training')
        exit(0)
    
    
    test_dataset = mnist(training=False,transform=None)
    
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle=False)

    if Kernel_size == 3:
        model1 = ModelM3.to(device)
    elif Kernel_size == 5:
        model1 = ModelM5.to(device)
    elif Kernel_size == 7:
        model1 = ModelM7.to(device)
    
    model1.load_state_dict(torch.load("../logs/%s/model%03d.pth"%(logdir,seed)))

    model1.eval()

    test_loss = 0 
    correct = 0
    wrong_images = []

    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model1(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            pred = output.argmax(dim =1, keep_dim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            wrong_images.extend(np.nonzero(~pred.eq(target.view_as(pred)).cpu().numpy())[0]+(100*batch_idx)
            
    np.savetxt("../logs/%s/wrong%03d.txt"%(logdir,seed),wrong_images,fmt="%d")
    print(len(wrong_images), wrong_images)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="modelM5")
    p.add_argument("--seed",default=0,type=int)
    p.add_argument("--kernel_size",default=5,type=int)
    p.add_argument("--trials",default=30,type=int)
    args = p.parse_args()
    for i in range(args.trials):
        run(seed = args.seed + i,
            kernel_size = args.kernel_size,
            logdir = args.logdir)
