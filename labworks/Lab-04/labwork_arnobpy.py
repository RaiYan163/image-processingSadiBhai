# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:42:23 2023

@author: NLP Lab
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc

Do = 3
n = 2

def point_op(filter, uk, vk):
    M = filter.shape[0]
    N = filter.shape[1]
    H = np.ones((M,N), np.float32)
    for u in range(M):
        for v in range(N):
            H[u, v] = 1.0
            dk = ((u - uk)**2 + (v - vk)**2)**(0.5)
            d_k = ((u - (M-uk))**2 + (v - (N-vk))**2)**(0.5)
            if dk == 0 or d_k == 0:
                H[u, v] = 0.0
                continue
            H[u, v] = (1/(1+((Do/dk)**(2*n)))) * (1/(1+((Do/d_k)**(2*n))))
    return H

def butter(filter, points):
    ret = np.ones(filter.shape, np.float32)
    for u,v in points:
        ret *= point_op(filter, u, v)
    return ret


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('two_noise.jpeg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)




print(img.shape)
filter = np.ones(img.shape, float)


points = [(200, 200)]

notch = butter(filter, points)


cv2.imshow("notch filter", notch)

result= magnitude_spectrum_ac * notch

## phase add
#final_result = np.multiply(result, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("Inverse transform",img_back_scaled)



cv2.waitKey(0)
cv2.destroyAllWindows() 