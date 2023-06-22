# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:04:17 2023

@author: rupok
"""

import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

img = plt.imread("Lena.jpg",'gray')

#cv.imshow("original",img)

plt.imshow(img,'gray')

mn = img.min()
mx = img.max()

R = img.shape[0]
C = img.shape[1]
T = img.shape[2]

'''op = img
for x in range(R):
    for y in range(C):
        for z in range(T):
        op[x][y][z] = ((img[x][y][z]-mn)/(mx-mn))*255
'''
#cv.imshow("final",img)

cv.waitKey()
cv.destroyAllWindows()
    
    