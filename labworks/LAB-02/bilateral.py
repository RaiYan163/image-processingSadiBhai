import numpy as np
import cv2 as cv
import math

path = './rubiks_cube.png'
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.imshow('Input', img)

dim = 3
k = dim // 2
sigma = 1

spatial_filter = np.zeros((dim, dim), dtype = np.float32)

for i in range (-k, k + 1):
    for j in range (-k, k + 1):
        spatial_filter[i + k][j + k] = np.exp(-(i * i + j * j) / 2 * (sigma ** 2)) / (2 * math.pi * (sigma ** 2))

padImg = cv.copyMakeBorder(img, k, k, k, k, cv.BORDER_CONSTANT) / 255

output = np.zeros(img.shape, dtype = np.float32)

h = img.shape[0]
w = img.shape[1]
range_filter = np.zeros((dim, dim), dtype = np.float32)
res = np.zeros(img.shape, dtype = np.float32)

for i in range (h):
    for j in range (w):
        sum = 0
        for x in range (-k, k + 1):
            for y in range (-k, k + 1):
                distance = int(padImg[i + (k + x)][j + (k + y)] - padImg[i + k][j + k])
                range_filter[x + k][y + k] = np.exp(-(distance ** 2) / (2 * (sigma ** 2))) / (2 * math.pi * (sigma ** 2))
        final_filter = np.multiply(spatial_filter, range_filter)
        norm = np.sum(final_filter)
        
        for x in range (dim):
            for y in range (dim):
                res[i][j] += padImg[i + x][j + y] * final_filter[dim - x - 1][dim - y - 1]
        res[i][j] /= norm

bilFilter = cv.bilateralFilter(res, dim, 1, 1)

cv.imshow('function', bilFilter)
cv.normalize(res, res, 0, 255, cv.NORM_MINMAX)
res = np.round(res).astype(np.uint8)
cv.imshow('Bilateral Filtered', res)
cv.waitKey(0)
cv.destroyAllWindows()
print(gf)
