import os 
import argparse
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
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
    

