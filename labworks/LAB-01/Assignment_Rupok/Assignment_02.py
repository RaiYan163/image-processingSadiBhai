# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:16:57 2023

@author: rupok
"""

import numpy as np
import cv2


def gaussian(m,n,sigma):
    gaussian = np.zeros((m,n))
    m = m//2
    n = n//2
    
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            
            x1 = (2*np.pi * sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
            
            gaussian[x+m,y+n] = (1/x1) * x2
    
    return gaussian
    
def conv(img,filt):
    S = img.shape
    F = filt.shape

    R = S[0] + F[0] -1
    C = S[1] + F[1] -1

    Z = np.zeros((R,C))


    for x in range(S[0]):
        for y in range(S[1]):
            Z[x+int((F[0]-1)/2), y+int((F[1]-1)/2)] = img[x,y]


    for i in range(S[0]):
        for j in range(S[1]):
            k = Z[i:i+F[0],j:j+F[1]]
            sum = 0
            for x in range(F[0]):
                for y in range (F[1]):
                        sum += (k[x][y] * filt[-x][-y])

            img[i,j] = sum
    return img

def median(img,filt):
    S = img.shape
    F = filt.shape

    R = S[0] + F[0] -1
    C = S[1] + F[1] -1

    Z = np.zeros((R,C))


    for x in range(S[0]):
        for y in range(S[1]):
            Z[x+int((F[0]-1)/2), y+int((F[1]-1)/2)] = img[x,y]


    for i in range(S[0]):
        for j in range(S[1]):
            k = Z[i:i+F[0],j:j+F[1]]
            v = []
            for x in range(F[0]):
                for y in range (F[1]):
                    v.append(k[x][y])
            v.sort()
            img[i,j] = v[int(len(v)/2)]
            v.clear()
    return img

img = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original",img) 

#plt.imshow(img,'gray')
#plt.show()

print("Enter 1: For mean \nEnter 2: For median \nEnter 3: For gaussian \nEnter 4: For laplacian\nEnter 5: For Sobel\n")

p = int(input("Enter a number between 1 to 5:\n"))

if(p==1):
    filt = np.array([
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,1]])/25
    img = conv(img, filt)
    
elif p==2:
    filt = np.array([[0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0]]) 
    img = median(img,filt)

elif p==3:
    sigma = int(input("Enter sigma value: "))
    m = 5*sigma
    if m%2==0:
        m += 1
    filt = gaussian(m,m,sigma)
    img = conv(img, filt)
    
elif p==4:
    filt = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]])
    img = conv(img,filt)
elif p==5:
    filt = np.array([
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]])
    img = conv(img, filt)
    

#print(filt)


        
cv2.imshow("final",img)

#plt.imshow(img,'gray')
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

