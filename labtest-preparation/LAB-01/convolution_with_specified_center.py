import numpy as np
import cv2 as cv

img = cv.imread('./labfiles/lena.jpg', 0)
cv.imshow('input', img)

kernel = np.ones((5, 5))

h, w = img.shape
kh, kw = kernel.shape

# specified center of the kernel
p, q = (2, 1) 

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