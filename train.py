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
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 *(train_corr)/len(train_loader.dataset)

        model.eval()
        ema.assign(model)
        train_loss = 0
        correct = 0
        total_pred = np.zeros(0)
        total_target = np.zeros(0)
        with torch.no_grad():
            with data, target in test_loader:
                data, target = data.to(device), target.to(device,dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output,target,reduction='sum').item()
                pred = output.argmax(dim=1,keep_dim = True)
                total_pred = np.append(total_pred,pred.cpu().numpy())
                total_target = np.append(total_target,target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            if (max_correct < correct):
                torch.save(model.state_dict(),MODEL_FILE)
                max_correct = correct
                print('best accuracy! Current images:%5d'%correct)
        ema.resume(model)

        #output

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct/len(test_loader.dataset)
        best_test_accuracy = 100 * max_correct/len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy, best_test_accuracy))
        
        f = open(OUTPUT_FILE, 'a')
        f.write(" %3d %12.6f %9.3f %12.6f %9.3f %9.3f\n"%(epoch, train_loss, train_accuracy, test_loss, test_accuracy, best_test_accuracy))
        f.close()

        #update Lr
        lr_scheduler.step()

        #main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--trials", default=15, type=int)
    p.add_argument("--epochs", default=150, type=int)    
    p.add_argument("--kernel_size", default=5, type=int)    
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="temp")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    for i in range(args.trials):
        run(p_seed = args.seed + i,
            p_epochs = args.epochs,
            p_kernel_size = args.kernel_size,
            p_logdir = args.logdir)
