# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:21:26 2023

@author: USER
"""

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import math
from copy import deepcopy as dpc

input_img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Input", input_img)
cv2.imwrite("input.jpg", input_img)


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range(img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = ((img_inp[i][j] - inp_min) / (inp_max - inp_min)) * 255
    return np.array(img_inp, dtype="uint8")


def Gaussian_noise(sigma, img):
    noise = np.zeros_like(img, np.float32)
    constant = 2 * math.pi * (sigma**2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            power = -(i * i + j * j)
            power /= 2 * (sigma**2)
            val = math.exp(power)
            noise[i, j] = val / constant
    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
    noise = noise.astype(np.uint8)
    return noise


def Homomorphic_Filter(img):
    gh = 1.8
    gl = 0.1
    c = 5
    D0 = 10
    M = img.shape[0]
    N = img.shape[1]
    h_filter = np.zeros(img.shape)
    for i in range(M):
        for j in range(N):
            dk = np.sqrt((i - M // 2) ** 2 + (j - N // 2) ** 2)
            power = -c * ((dk**2) / (D0**2))
            h_filter[i, j] = (gh - gl) * (1 - np.exp(power)) + gl
    return h_filter


noise = np.zeros_like(input_img, np.float32)

sigma_a = 100
sigma_b = 70

noise = Gaussian_noise(sigma_a, input_img)

noise_flip = Gaussian_noise(sigma_b, input_img)

noise_flip = np.flip(noise_flip, 0)
noise_flip = np.flip(noise_flip, 1)

noise = noise + noise_flip

cv2.imshow("Noise", noise)

noisy_img = cv2.add(input_img, noise)
cv2.imshow("Noisy_Image", noisy_img)
# cv2.imwrite("Noisy.jpg", noisy_img)
img = dpc(noisy_img)

# Homomorphic Filter
h_filter = np.zeros(img.shape)
h_filter = Homomorphic_Filter(img)
# plt.imshow(np.log(np.abs(h_filter)), "gray")
# plt.show()


image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)


ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac * h_filter, np.exp(1j * ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_scaled)

cv2.imshow("Inverse transform", img_back_scaled)
# cv2.imwrite("output.jpg", img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
