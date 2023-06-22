# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:15:21 2023

@author: Taslima Joty
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img= cv2.imread('lena.jpg',0)
cv2.imshow("Input",img)
plt.show()
L=256

histr=cv2.calcHist([img],[0],None,[256],[0,255])
plt.title("Built In")
plt.plot(histr)
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,256])
plt.show()

img2=np.zeros_like(img)
h,w=img.shape
Tpixel=h*w
hist=np.zeros(256)
for i in range (h):
    for j in range (w) :
        x=img[i][j]
        hist[x]+=1
new_hist=hist
new_hist=new_hist/Tpixel
#plt.plot(new_hist)
#plt.show()
cdf = np.zeros(256)
cdf[0]=new_hist[0]
s=new_hist
sm=0.0
print(cdf)
for i in range(1,256):
    sm=sm+new_hist[i]
    cdf[i]=sm
    s[i]=round((L-1)*cdf[i])

print(s)
plt.subplot(1,3,2)
plt.title("CDF Image Histogram")
plt.plot(cdf)
plt.show()
for i in range(h):
    for j in range(w):
        x=img[i][j]
        img2[i][j]=s[x]
print(img2)
cv2.imshow("Output",img2)
plt.show()
plt.subplot(1,3,3)
plt.title("Output Image Histogram")
plt.hist(img2.ravel(),256,[0,256])
plt.show()
 
cv2.waitKey(0)
cv2.destroyAllWindows()       