import numpy as np
import cv2 as cv

img = cv.imread('./labfiles/lena.jpg', 0)
cv.imshow('input', img)

def get_gaussian_kernel(h, w, sigma):
    const = 2 * np.pi * sigma * sigma
    kh = h // 2
    kw = w // 2
    kernel = np.zeros((h, w))
    for i in range (-kh, kh + 1):
        for j in range (-kw, kw + 1):
            kernel[kh + i][kw + j] = np.exp(-(i * i + j * j) / 2 * sigma * sigma) / const    
    print(kernel)
    return kernel

kernel = get_gaussian_kernel(h = 7, w = 5, sigma = 1)

h, w = img.shape
kh, kw = kernel.shape

# specified center of the kernel
# p, q = (3, 2) 
p = kh // 2
q = kw // 2

# calculate padding amount
top = kernel.shape[0] - p - 1
bottom = p
left = kernel.shape[1] - q - 1
right = q

padimg = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT)
output = np.zeros(img.shape)

for x in range (h):
    for y in range (w):
        for i in range (kh):
            for j in range (kw):
                output[x][y] += padimg[x + i][y + j] * kernel[kh - 1 - i][kw - 1 - j]

cv.normalize(output, output, 0, 255, cv.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv.imshow('output', output)

cv.waitKey(0)
cv.destroyAllWindows()