# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:42:47 2022

@author: Taha & Anass
"""


#code_final

import numpy as np
from scipy import signal 
import cv2
from random import randrange, uniform
import matplotlib.pyplot as plt
 
sigma = 30

min_scale = 0.08
max_scale = 0.1
patch = cv2.imread('C:/Users/Taha/Desktop/RANDOM/square.png',0)
scale = uniform(min_scale, max_scale)
patch_h, patch_w = patch.shape
p = cv2.resize(patch, (int(patch_w * scale), int(patch_h * scale)))
p_h, p_w = p.shape
h, w = 100, 100
def gene_coord(amt):
    l=[]    
    min_scale = 0.08
    max_scale = 0.1
    patch = cv2.imread('C:/Users/Taha/Desktop/RANDOM/square.png',0)
    scale = uniform(min_scale, max_scale)
    patch_h, patch_w = patch.shape
    p = cv2.resize(patch, (int(patch_w * scale), int(patch_h * scale)))
    p_h, p_w = p.shape
    h, w = 100, 100
    for i in range(amt):
        l.append(randrange(w - p_w))
        l.append(randrange(h - p_h))
    return l



def patch_img(img, l, amt):
    for i in range(amt):  
        x=l[2*i]
        y=l[2*i+1]
        seg = img[y: y + p_h, x: x + p_w]
        seg[:] = cv2.bitwise_or(seg, p)
        


def Gaussian2D_v1(coords,  # x1 and y1 coordinates for each image.
          amplitude=1,  # Highest intensity in image.
          xo=0,  # x-coordinate of peak centre.
          yo=0,  # y-coordinate of peak centre.
          sigma_x=1,  # Standard deviation in x.
          sigma_y=1,  # Standard deviation in y.
          rho=0,  # Correlation coefficient.
          offset=0):  # Offset from zero (background radiation).
    x1, y1 = coords

    xo = float(xo)
    yo = float(yo)

    # Create covariance matrix
    mat_cov = [[sigma_x**2, rho * sigma_x * sigma_y],
               [rho * sigma_x * sigma_y, sigma_y**2]]
    mat_cov = np.asarray(mat_cov)
    # Find its inverse
    mat_cov_inv = np.linalg.inv(mat_cov)

    # PB We stack the coordinates along the last axis
    mat_coords = np.stack((x1 - xo, y1 - yo), axis=-1)

    G = amplitude * np.exp(-0.5*np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                    mat_cov_inv),
                                          mat_coords[..., np.newaxis])) + offset
    return G.squeeze()
    






coords = np.meshgrid(np.arange(0, 100), np.arange(0, 100))
#    print(coords)
# model_1 = Gaussian2D_v1(coords,
#                         amplitude=20,
#                         xo=32,
#                         yo=32,
#                         sigma_x=6,
#                         sigma_y=3,
#                         rho=0.8,
#                         offset=20)
# plt.figure(figsize=(5, 5)).add_axes([0, 0, 1, 1])
# plt.contourf(model_1)
     
        

        
[h,w]=patch.shape
[hpix,wpix]=[h*0.1,w*0.1]
#print(hpix,wpix)

for i in range(5):
    Gau = np.zeros((100,100))
    img = np.zeros((100, 100), dtype="uint8")
    for j in range(1,5): 
         l=gene_coord(j)
         patch_img(img, l, j)
         for k in range(j):
             temp = Gaussian2D_v1(coords,amplitude=200,xo=l[2*k]+wpix/2,yo=l[2*k+1]+hpix/2,sigma_x=1,sigma_y=1,rho=0,offset=0)
             Gau +=temp/temp.sum()
 

             cv2.imwrite(f'C:/Users/Taha/Desktop/RANDOM/input/image_in_{j}_{i}.png',img)
             cv2.imwrite(f'C:/Users/Taha/Desktop/RANDOM/gth/image_gth_{j}_{i}.png',1000*Gau)

         cv2.waitKey(0) 
        
        
#meshgrid !!!!!!