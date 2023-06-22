import numpy as np
import cv2 as cv

img = cv.imread('./rubiks_cube.png', 0)
cv.imshow('input', img)

def get_spatial_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), np.float32)
    k = ksize // 2
    const = 2 * np.pi * sigma * sigma
    for i in range (-k, k + 1):
        for j in range (-k, k + 1):
            kernel[k + i][k + j] = np.exp(-(i * i + j * j) / 2 * sigma * sigma) / const
    return kernel

def get_range_kernel(img, ksize, x, y, sigma):
    Ip = img[x][y]
    kernel = np.zeros((ksize, ksize), np.float32)
    const = np.sqrt(2 * np.pi) * sigma
    k = ksize // 2
    for i in range (-k, k + 1):
        for j in range (-k, k + 1):
            Iq = img[x + i][y + j]
            kernel[k + i][k + j] = np.exp(-(Ip - Iq) ** 2) / const
    return kernel

ksize = 5
k = ksize // 2
spatial_kernel = get_spatial_kernel(ksize, sigma = 4)
H, W = img.shape
# borderedImg = np.zeros(img.shape, np.float32)
borderedImg = cv.copyMakeBorder(img, k, k, k, k, cv.BORDER_CONSTANT)
output = np.zeros(img.shape, np.float32)

for x in range (H):
    for y in range (W):
        range_kernel = get_range_kernel(borderedImg, ksize, x + k, y + k, sigma = 0.4)
        kernel = spatial_kernel * range_kernel
        kernel /= np.sum(kernel)
        for i in range (ksize):
            for j in range (ksize):
                output[x][y] += borderedImg[x + i][y + j] * kernel[ksize - 1 - i][ksize - 1 - j]

cv.normalize(output, output, 0, 255, cv.NORM_MINMAX)
output = np.round(output).astype(np.uint8)
cv.imshow('output', output)

cv.waitKey(0)
cv.destroyAllWindows()