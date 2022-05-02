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
class Net(nn.Module):
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
        self.conv4=nn.Conv2d(256,256,5)
        self.conv4=nn.Conv2d(256,256,5)

        self.upsample1=nn.Upsample(scale_factor=1, mode='nearest')
        self.upsample2=nn.Upsample(scale_factor=1, mode='nearest')
        self.upsample3=nn.Upsample(scale_factor=8, mode='nearest')
        
    def forward(self,x):
        #Max pooling over a (2,2) window
        x=F.relu(F.conv1())
        x=F.MaxPool2d(F.relu(F.conv2()))
        x=F.relu(F.conv3())
        x=F.MaxPool2d(F.relu(F.conv4()))
        x=F.relu(F.conv2d(F.upsample1()))
        x=F.relu(F.conv2d(F.upsample2()))
        x=F.relu(F.conv2d(F.upsample1()))
        return x
net = Net()
params = list(net.parameters())
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
epochs = 5
dataset_train = SD.SquareDataset('/home/anass/Documents/projet_thematique/train/' )


dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
is_gpu=torch.cuda.is_available()
for i in range(epochs):
    for input_,ground in dataloader:
            if is_gpu:
                input_, ground = input_.cuda(), ground.cuda()
            optimizer.zero_grad()
            
            output = net(input_)
            loss = nn.MSELoss()
            loss.backward()
            output= loss(input_,ground)
            optimizer.step()
            output.backward()
    print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{loss}')
