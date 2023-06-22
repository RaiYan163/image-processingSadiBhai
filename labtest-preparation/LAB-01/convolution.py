import numpy as np
import cv2 as cv

img = cv.imread('./labfiles/lena.jpg', 0)

cv.imshow("input", img)

kernel = np.ones((5, 5))
print(kernel)

k = kernel.shape[0] // 2

padimg = cv.copyMakeBorder(img, k, k, k, k, cv.BORDER_CONSTANT)
cv.imshow("padded image", padimg)

output = np.zeros(img.shape)
h, w = img.shape
kh, kw = kernel.shape

for x in range (h):
    for y in range (w):
        for i in range (kernel.shape[0]):
            for j in range (kernel.shape[1]):
                output[x][y] += padimg[x + i][y + j] * kernel[kh - 1 - i][kw - 1 - j]

cv.normalize(output, output, 0, 255, cv.NORM_MINMAX)
output = np.round(output).astype(np.uint8)
print(output.shape)

cv.imshow('output', output)

cv.waitKey(0)
cv.destroyAllWindows()