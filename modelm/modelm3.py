import torch
import torch.nn as nn
import torch.nn.functional as f

class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,48,3,bias=False)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48,64,3,bias=False)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,80,3,bias=False)
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5 = nn.Conv2d(80,96,3,bias=False)
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96,112,3,bias=False)
        self.conv6_bn = nn.BatchNorm2d(112)
        self.conv7 = nn.Conv2d(112,128,3,bias=False)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128,144,3,bias=False)
        self.conv8_bn = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144,160,3,bias=False)
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160,176,3,bias=False)
        self.conv10_bn = nn.BatchNorm2d(176)
        self.fcn1 = nn.Linear(11264,10,bias=False)
        self.fcn1_bn = nn.BatchNorm2d(10)

    def get_logits(self,x):
        x = (x-0.5)*2.0
        conv1 = f.relu(self.conv1_bn(self.conv1(x)))
        conv2 = f.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = f.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = f.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = f.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = f.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = f.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = f.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = f.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = f.relu(self.conv10_bn(self.conv10(conv9)))
        flat1 = torch.flatten(conv10.permute(0,2,3,1),1)
        logit = self.fcn1_bn(self.fcn1(flat1))
        return logit

    def forward(self,x):
        logits = self.get_logits(x)
        return f.log_softmax(logits,1)