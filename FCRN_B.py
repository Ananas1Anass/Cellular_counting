#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:23:15 2022

@author: anass
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import SquareDataset as SD
import numpy as np
# FCRN-B
class Net(nn.Module):

        def __init__(self):
            super(Net,self).__init__()
            #1 input image channel,6 output channels,2x2 square convolution
            #kernel
            self.conv1=nn.Conv2d(1,32,3,padding=1)
            self.conv2=nn.Conv2d(32,64,3,padding=1)
            self.conv3=nn.Conv2d(64,128,3,padding=1)
            self.conv4=nn.Conv2d(128,256,5,padding=2)
            self.conv5= nn.Conv2d(256,256,5,padding=2)
            self.conv6= nn.Conv2d(256,1,5,padding=2)

            self.upsample1=nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample2=nn.Upsample(scale_factor=2, mode='bilinear')
            
        def forward(self,x):
            #Max pooling over a (2,2) window
            x=F.relu(self.conv1(x))
            x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
            x=F.relu(self.conv3(x))
            x=F.max_pool2d(F.relu(self.conv4(x)),(2,2))
            x=F.relu(self.conv5(self.upsample1(x)))
            x=self.conv6(self.upsample2(x))
            return x
net = Net()
params = list(net.parameters())
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
epochs = 10
dataset_train = SD.SquareDataset('/home/anass/Documents/projet_thematique/train/' )

optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
loss = nn.MSELoss()
print(type(loss))
dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
is_gpu=torch.cuda.is_available()
for i in range(epochs):
    for input_,gth in dataloader:
            if is_gpu:
               input_, gth = input_.cuda(), gth.cuda()
            optimizer.zero_grad()
            
            output = net(input_)
            l= loss(output,gth)
            print(l.item)
            l.backward()
            optimizer.step()

    print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{loss}')
