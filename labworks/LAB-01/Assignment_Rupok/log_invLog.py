# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 04:27:38 2023

@author: rupok
"""

import cv2
import numpy as np

img1  = cv2.imread("einstein.jpg")

img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 

cv2.imshow("original",img)

R = img.shape[0]
C = img.shape[1]

factor = 1

for x in range(R):
    for y in range(C):
        img[x][y] = np.exp((img[x][y])**factor)-1

#print(img)


cv2.imshow("final",img)

cv2.waitKey()
cv2.destroyAllWindows()