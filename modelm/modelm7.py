import torch
import torch.nn as nn
import torch.nn.functional as f

class ModelM5(nn.Module):
    
    def __init__(self):
        super(ModelM5,self).__init__()
        self.conv1 = nn.Conv2d(1,48,7,bias=False)
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48,96,7,bias=False)
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96,144,7,bias=False)
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144,192,7,bias=False)
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fcn1 = nn.Linear(3072,10,bias=False)
        self.fcn1_bn = nn.BatchNorm2d(10)

    def get_logits(self,x):
        x = (x-0.5)*2.0
        conv1 = f.relu(self.conv1_bn(self.conv1(x)))
        conv2 = f.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = f.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = f.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0,2,3,1),1)
        logits = self.fcn1_bn(self.fcn1(flat1))
        return logits

    def forward(self,x):
        logits = self.get_logits(x)
        return f.log_softmax(logits,1)
