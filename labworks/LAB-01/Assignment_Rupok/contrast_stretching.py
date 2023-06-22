# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:24:43 2023

@author: rupok
"""

import cv2 as cv
import numpy as np


img = cv.imread("Lena.jpg",cv.IMREAD_GRAYSCALE)

cv.imshow("Original",img)

mn = img.min()

mx = img.max()

op = img

#print(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j]>=0 and img[i][j]<=mn:
            op[i][j] = 0
        elif(img[i][j]>mn and img[i][j]<=mx):
            op[i][j] = ((img[i][j] - mn)/(mx-mn))*255
        else:
            op[i][j] = 255
cv.imshow("fianl",op)
cv.waitKey()
cv.destroyAllWindows()