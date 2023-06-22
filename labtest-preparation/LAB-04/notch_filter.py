# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:18:59 2023

@author: rupok
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import cv2 as cv




img = cv.imread('input.jpg',cv.IMREAD_GRAYSCALE)


f1 = plt.figure(1)
plt.title("Input Image")
plt.imshow(img,'gray')


D0 = int(input("Enter a radious: \n"))

x = int(input("Enter X axis value: \n"))
y = int(input("Enter Y axis value: \n"))



M = img.shape[0]
N = img.shape[1]

a = M - x
b = N - y

p = 1




filter = np.zeros((M,N),np.float32)

for i in range(M):
    for j in range(N):
       
        if i>=M//2 and j>=M//2:
            D = math.sqrt((i-x)**2 + (j-y)**2)
        if i<M//2 and j<M//2:
            D = math.sqrt((i-a)**2 + (j-b)**2) 
       
        if D<=D0:    
            filter[i][j] = 0.0
        else:
            filter[i][j] = 1.0


'''filter2 = np.zeros((M,N),np.float32)
for i in range(M):
    for j in range(N):
        D = math.sqrt((i-a)**2 + (j-b)**2)
       
        if D<=D0:    
            filter2[i][j] = 0.0
        else:
            filter2[i][j] = 1.0'''
            
f1 = plt.figure(2)
plt.title("Filter Image")
plt.imshow(filter,'gray')


#filter = filter1 * filter2



f = np.fft.fft2(img)
shift = np.fft.fftshift(f)
mag = np.abs(shift)
angle = np.angle(shift)

f1 = plt.figure(3)
plt.title("input image spectrum")
plt.imshow(np.log(mag),'gray')

pp = np.log(mag)


mag = mag * filter


inv = np.multiply(mag,np.exp(1j * angle))

inv_shift = np.fft.ifftshift(inv)

out = np.real(np.fft.ifft2(inv_shift))

f1 = plt.figure(4)
plt.title("input image spectrum with filter")
plt.imshow(pp * filter,'gray')

f1 = plt.figure(5)

plt.title("Ouput Image")
plt.imshow(out,'gray')
plt.show()        
        
        
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


cv.waitKey()
cv.destroyAllWindows()