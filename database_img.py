#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:33:41 2022

@author: anass
"""

import numpy as np
import cv2
from random import randrange, uniform
from scipy import ndimage
import os 

def patch_img(img, patch, amt):
    h, w = img.shape
    min_scale = 0.008
    max_scale = 0.01
    for _ in range(amt):
        patch_h, patch_w = patch.shape
        scale = uniform(min_scale, max_scale)
        p = cv2.resize(patch, (int(patch_w * scale), int(patch_h * scale)))
        p_h, p_w = p.shape
        x = randrange(w - p_w)
        y = randrange(h - p_h)
        seg = img[y: y + p_h, x: x + p_w]
        seg[:] = cv2.bitwise_or(seg, p)

patch = cv2.imread('square.png', 0)
for i in range(100):
    for j in range(1,11):
        img = np.zeros((100, 100), dtype="uint8")
        patch_img(img, patch,j)
        cv2.imwrite(f'/home/anass/Documents/projet_thematique/RANDOM8/image{j}_{i}.png',img)
        cv2.waitKey(0)  
