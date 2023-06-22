# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:12:04 2023

@author: USER
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def gauss(sigma, mean):
    gauss_array=np.zeros(256,dtype=np.float32)
    for i in range(256):
        r=i-mean
        r=(r**2)/(sigma**2)
        r=math.exp(-r)/(sigma*math.sqrt(2*math.pi))
        gauss_array[i]=r;
    return gauss_array
    
img=cv2.imread("./histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input Image",img)

L=256
height, width = img.shape

input_hist = cv2.calcHist([img],[0],None, [256], [0,256])
pdf = input_hist/(height*width)
cdf = np.zeros(256, dtype=np.float32)
cdf[0]=pdf[0]
s=pdf

for i in range(1, 256):
    cdf[i] = cdf[i-1] + pdf[i]
    s[i]=round((L-1)*cdf[i]) 
        
eq_org_img=np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        x=img[i][j]
        eq_org_img[i][j]=s[x]


plt.subplot(2,2,1)
plt.title("Input Image")
plt.hist(img.ravel(),256,[0,256])

plt.subplot(2,2,2)
plt.title("Equalized Image")
plt.hist(eq_org_img.ravel(),256,[0,256])

sigma_a=32
mean_a=120
gauss_a=gauss(sigma_a, mean_a)

sigma_b=18
mean_b=186
gauss_b=gauss(sigma_b,mean_b)

final_gauss=gauss_a+gauss_b
target_pdf=np.zeros(256)
target_pdf=final_gauss/final_gauss.sum()
print(target_pdf.sum())

plt.subplot(2,2,3)
plt.title("Target Histogram")
plt.plot(gauss_a)
plt.plot(gauss_b)
plt.plot(final_gauss)
plt.show()


G=np.zeros(256)
target_cdf=np.zeros(256)

target_cdf[0]=target_pdf[0]

for i in range(1,256):
    target_cdf[i]=target_cdf[i-1]+target_pdf[i]
    #print(G[i])
    G[i]=round((L-1)*target_cdf[i])

mapping=np.zeros(256,dtype=np.int64)
for i in range(256):
    x=np.searchsorted(G, s[i])
    # x=min(x,255)
    if x>0 and abs(s[i]-G[x-1])<abs(G[x]-s[i]):
        x=x-1
    mapping[int(s[i])]=x

final_matching_image=np.zeros_like(img)
for i in range(eq_org_img.shape[0]):
    for j in range(eq_org_img.shape[1]):
        x=eq_org_img[i][j]
        final_matching_image[i][j]=mapping[x]

plt.subplot(2,2,4)
plt.title("Matching Histogram")      
plt.hist(final_matching_image.ravel(),256,[0,256])
plt.show()
cv2.imshow("Output Image",final_matching_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
