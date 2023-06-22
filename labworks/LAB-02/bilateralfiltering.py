import numpy as np
import cv2 as cv

img = cv.imread('lena.jpg', 0)
# img = cv.imread('rubiks_cube.png', 0)

def get_spatial_kernel(ksize, sigma):
    const = 2 * np.pi * sigma * sigma
    k = ksize // 2
    kernel = np.zeros((ksize, ksize), np.float32)
    for i in range (-k, k + 1):
        for j in range (-k, k + 1):
            kernel[k + i][k + j] = np.exp(-(i * i + j * j) / 2 * sigma * sigma) / const
    return kernel

def get_range_kernel(img, x, y, ksize, sigma):
    const = np.sqrt(2 * np.pi) * sigma
    k = ksize // 2
    Ip = img[x][y]
    kernel = np.zeros((ksize, ksize), np.float32)
    for i in range (-k, k + 1):
        for j in range (-k, k + 1):
            Iq = img[x + i][y + j]
            kernel[k + i][k + j] = np.exp(-((Ip - Iq) ** 2) / 2 * sigma * sigma) / const
    return kernel

ksize = 5
k = ksize // 2
borderedInput = cv.copyMakeBorder(img, k, k, k, k, cv.BORDER_REPLICATE)
output = np.zeros((img.shape), np.float32)


spatial_kernel = get_spatial_kernel(ksize, 4)

for x in range (img.shape[0]):
    for y in range(img.shape[1]):
        range_kernel = get_range_kernel(borderedInput, x + k, y + k, ksize, sigma = 0.4)
        kernel = spatial_kernel * range_kernel
        kernel /= np.sum(kernel)
        for i in range (ksize):
            for j in range (ksize):
                output[x][y] += borderedInput[x + i][y + j] * kernel[ksize - 1 - i][ksize - 1 - j]

cv.normalize(output, output, 0, 255, cv.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv.imshow('Lena', img)
cv.imshow('Bilateral Output', output)
cv.waitKey(0)
cv.destroyAllWindows()
