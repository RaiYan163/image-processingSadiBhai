import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

def Gaussian(sigma, mean):
    g = np.zeros(256, dtype = np.float32)
    for i in range (256):
        f = i - mean
        p = -(f ** 2) / (sigma ** 2)
        res = math.exp(p) / (sigma * math.sqrt(2 * math.pi))
        g[i] = res
    return g

sigma1 = 25
mean1 = 100
pdf1 = Gaussian(sigma1, mean1)

sigma2 = 30
mean2 = 180
pdf2 = Gaussian(sigma2, mean2)

final = pdf1 + pdf2
target = np.zeros(256)
target = final / final.sum()

plt.subplot(2, 2, 1)
plt.title('Specified Probability Distribution')
plt.plot(pdf1)
plt.plot(pdf2)
plt.plot(final)
plt.show()

img = cv.imread('./histogram.jpg', cv.IMREAD_GRAYSCALE) 
cv.imshow('Input Image', img)

L = 256
M, N = img.shape

hist = cv.calcHist([img],[0],None, [256], [0,255])
pdf = hist / (M * N)
cdf = np.zeros(256, dtype = np.float32)
cdf[0] = pdf[0]
s = pdf

for i in range (1, 256):
    cdf[i] += cdf[i - 1] + pdf[i]
    s[i] = round((L - 1) * cdf[i])

eq_img = np.zeros_like(img)
for i in range (M):
    for j in range(N):
        x = img[i][j]
        eq_img[i][j] = s[x]

plt.subplot(2, 2, 2)
plt.title('Input Histogram')
plt.hist(img.ravel(), 256, [0, 255])

plt.subplot(2, 2, 3)
plt.title('Equalized Histogram')
plt.hist(eq_img.ravel(), 256, [0, 255])


G = np.zeros(256)
target_cdf = np.zeros(256)
target_cdf[0] = target[0]

for i in range(1, 256):
    target_cdf[i] = target_cdf[i-1] + target[i]
    G[i] = round((L-1) * target_cdf[i])

map = np.zeros(256, dtype = np.int64)
for i in range (256):
    x = np.searchsorted(G, s[i])
    if x > 0 and abs(s[i] - G[x-1]) < abs(G[x] - s[i]):
        x = x-1
    map[int(s[i])] = x

final_image = np.zeros_like(img)
for i in range(eq_img.shape[0]):
    for j in range(eq_img.shape[1]):
        x = eq_img[i][j]
        final_image[i][j] = map[x]

plt.subplot(2,2,4)
plt.title('Matched Histogram')    
plt.hist(final_image.ravel(), 256, [0, 255])
plt.show()

cv.imshow('Matched Final Image', final_image)
cv.imwrite('./Matched Final Image.jpg', final_image)

cv.waitKey(0)
cv.destroyAllWindows()