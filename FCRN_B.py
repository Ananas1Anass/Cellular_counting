#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:11:25 2022

@author: anass
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# FCRN-B
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #1 input image channel,6 output channels,2x2 square convolution
        #kernel
        self.conv1=nn.Conv2d(1,32,3)
        self.conv2=nn.Conv2d(32,64,3)

        self.conv3=nn.Conv2d(64,128,3)
        self.conv4=nn.Conv2d(128,256,5)
        self.conv5=nn.Conv2d(256,256,5)

        self.upsample1=nn.Upsample(size=50,scale_factor=2,mode='bilinear')

    def forward(self,x):
        #Max pooling over a (2,2) window
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(F.relu(self.conv1(x)),2,stride=3)
        x=F.relu(self.conv3(x))
        #1,32,100,100
        x=F.max_pool2d(F.relu(self.conv4(x)),2,stride=5)
        x=F.relu(self.conv5(x))
        x=F.relu(self.conv4(x))
        x=F.relu(F.conv2d(self.upsample1(x)))
        return x
net = Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(len(params))
criterion=nn.CrossEntropyLoss()
lr=0.01
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
mse_loss=nn.MSELoss()
# output test
#output=mse_loss(my_input,my_target)
