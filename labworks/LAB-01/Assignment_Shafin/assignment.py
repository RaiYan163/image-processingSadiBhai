# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:04:57 2023

@author: USER
"""
import numpy as np
import cv2
import math


def gamma(img):
    c=255/8# s=clog(1+r)
    gm=2.5 #s=c*r^gamma
    height=img.shape[0]
    width=img.shape[1]
    output=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            output[i][j]=c*math.pow(img[i][j],gm)
    
    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output = np.round(output).astype(np.uint8)
    
    cv2.imshow("gamma",output)
    
    
def inverseLog(img):
    c=255/math.log2(1+255)
    height=img.shape[0]
    width=img.shape[1]
    output=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            output[i][j]=math.pow(2, img[i][j]/c)-1
    
    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output = np.round(output).astype(np.uint8)
    cv2.imshow("inverseLog",output)

def contrast_stretching(img):
    
    height=img.shape[0]
    width=img.shape[1]
    xmin=np.min(img)
    xmax=np.max(img)
    output=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            output[i][j]=((img[i][j]-xmin)*255)/(xmax-xmin)
            
    output = np.round(output).astype(np.uint8)        
    cv2.imshow("Contrast Stretching",output)
    

def gaussian(img, ksize, sigma):
    height=img.shape[0]
    width=img.shape[1]
    kernel=np.zeros((ksize,ksize),dtype='float32')
    k=ksize//2
    gconst=sigma**2
    gconst=2*math.pi*gconst
    
    for i in range(-k,k+1):
        for j in range(-k,k+1):
            gpower=((i*i)+(j*j))/(2*sigma**2)
            kernel[i+k][j+k]=math.exp(-gpower)/gconst
    
    print(kernel)
    output=np.zeros((height,width))
    bordered_output=cv2.copyMakeBorder(img, k,k,k,k,cv2.BORDER_REPLICATE)
    
    for i in range(height):
        for j in range(width):
            for x in range(ksize):
                for y in range(ksize):
                    output[i][j]+=bordered_output[i+x-k][j+y-k]*kernel[ksize-x-1][ksize-y-1]
    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output=np.round(output).astype(np.uint8)
    cv2.imshow("gaussian",output)


def median(img,s):
    height = img.shape[0]
    width = img.shape[1]
    output = np.zeros((height-s+1,width-s+1))
    for i in range(height-s+1):
        for j in range(width-s+1):
            a=[]
            for x in range(s):
                for y in range(s):
                    a.append(img[i+x][j+y])
            a.sort()
            output[i][j]=a[s*s//2]
    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output=np.round(output).astype(np.uint8)    
    cv2.imshow("median",output)


def mean(img,s):
    kernel = np.ones((s,s))
    height = img.shape[0]
    width = img.shape[1]
    output = np.zeros((height-s+1,width-s+1),dtype=np.float32)
    for x in range(height-s+1):
        for y in range(width-s+1):
            total = 0.0
            for i in range(s):
                for j in range(s):
                    total = total + kernel[i][j]*img[x+i][y+j]
            output[x][y]= total/np.sum(kernel)
    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output=np.round(output).astype(np.uint8)
    cv2.imshow("mean",output)
        

def sobel(img):
    height=img.shape[0]
    width= img.shape[1]
    sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    output = np.zeros((height,width),dtype =np.float32)
    x_output = np.zeros((height,width),dtype =np.float32)
    y_output = np.zeros((height,width),dtype =np.float32)
    k_h=sobel_x.shape[0]
    k_w=sobel_x.shape[1]
    h=k_h//2
    w=k_w//2
    for x in range(h,height-h):
        for y in range(w,width-w):
            for i in range(k_h):
                for j in range(k_w):
                    x_output[x][y] += sobel_x[k_h-i-1][k_w-j-1]*img[x+i-h][y+j-w]
                    y_output[x][y] += sobel_y[k_h-i-1][k_w-j-1]*img[x+i-h][y+j-w]
            output[x][y] = np.sqrt((x_output[x][y])**2+(y_output[x][y])**2)

    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output=np.round(output).astype(np.uint8)    
    cv2.imshow("sobel",output)
    cv2.normalize(x_output,x_output,0,255,cv2.NORM_MINMAX)
    x_output=np.round(x_output).astype(np.uint8)    
    cv2.imshow("x_sobel",x_output)
    cv2.normalize(y_output,y_output,0,255,cv2.NORM_MINMAX)
    y_output=np.round(y_output).astype(np.uint8)    
    cv2.imshow("y_sobel",y_output)

img=cv2.imread(r"D:\Academics\4-1\2.Lab\CSE 4128 (Image Processing Lab)\LAB-01\Assignment_Sadi\noise.png",cv2.IMREAD_GRAYSCALE)
#gamma(img)
#inverseLog(img)
#contrast_stretching(img)
gaussian(img, 3, 1)
#median(img,3)
#mean(img, 3)
#sobel(img)
cv2.imshow("Actual",img)
cv2.waitKey(0)
cv2.destroyAllWindows()