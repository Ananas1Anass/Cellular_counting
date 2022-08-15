#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:23:15 2022

@author: anass
"""
import torch
import gc
import numpy as np
import torch.nn as nn
import torchplot as plttorch
import torch.nn.functional as F
import SquareDataset as SD
from matplotlib import pyplot as plt
gc.collect()
torch.cuda.empty_cache()

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
            x=F.relu(self.conv5(x))
            x=F.relu(self.conv5(self.upsample1(x)))
            x=self.conv6(self.upsample2(x))
            return x
net = Net()

params = list(net.parameters())
if torch.cuda.is_available():
    net = net.cuda()
epochs = 100
dataset_validation = SD.SquareDataset('/home/anass/Documents/projet_thematique/validation/' )
dataset_train = SD.SquareDataset('/home/anass/Documents/projet_thematique/train/' )


optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
#ADAM
loss = nn.MSELoss()
print(type(loss))
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=False)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=16, shuffle=False)
is_gpu=torch.cuda.is_available()
T_train=[]

for i in range(epochs):
    loss_cout=0
    for input_,gth,grd_sum in dataloader_train:
        #for input_,gth,ground_img in dataloader_train:

        
            if is_gpu:
               input_, gth = input_.cuda(), gth.cuda()
            optimizer.zero_grad()
            
            output = net(input_)
            l= loss(output,gth)
            #loss_cout+=l
            print(l.item())
            T_train.append(l.item())
            #T.sort()
            #T.reverse()
            l.backward()
            optimizer.step()
    print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{l}')
    plt.plot(T_train,'r')
    plt.show()
T_validation=[]
calcul=[]
error=[]
y=torch.empty((2,3))
for i in range(epochs):
    loss_cout=0
    for input_,gth,grd_sum in dataloader_validation:
   # for input_,gth,ground_img in dataloader_validation:
        
            if is_gpu:
               input_, gth = input_.cuda(), gth.cuda()
            optimizer.zero_grad()
            
            output = net(input_)
            l= loss(output,gth)
            #loss_cout+=l
            T_validation.append(l.item())
            #T.sort()
            #T.reverse()
            #l = ground_img.detach().tolist()
            #grd_sum=ground_img
            #error.append((int(output.detach().cpu().sum().numpy())-)**2)
            print(l.item())
            print(grd_sum)
            l.backward()
            optimizer.step()
    print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{l}')
    plt.savefig(f'/home/anass/Documents/projet_thematique/validation/validation_{i}')
    #plt.plot(error,'b')
plt.plot(error,'*')
plt.show()


plt.show()
state = {
        'epoch': epochs,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
}
savepath='checkpoint.t7'
torch.save(state, "/home/anass/Documents/projet_thematique/MODEL/model1.pt")
