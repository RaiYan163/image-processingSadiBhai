import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

img = cv.imread('lena.jpg', 0)
cv.imshow('Input Image', img)
L = 256

# h = cv.calcHist([img], [0], None, [256], [0, 255])
# plt.title('Built In')
# plt.plot(h)


plt.figure(figsize = (10, 4))
plt.subplot(1, 3, 1)
plt.title('Input Image Histogram')
plt.hist(img.ravel(), 256, [0, 255])
plt.show()

H = img.shape[0]
W = img.shape[1]

MN = H * W

freq = np.zeros(256)

for i in range (H):
    for j in range (W):
        x = img[i][j]
        freq[x] += 1
        
hist = freq / MN


cdf = hist
s = np.zeros(cdf.size, dtype=np.float32)
s[0] = round((L - 1) * cdf[0])

for i in range (1, 256):
    cdf[i] += cdf[i - 1]
    s[i] = round((L - 1) * cdf[i])

print(s)
plt.subplot(1, 3, 2)
plt.title('CDF')
plt.plot(cdf)
plt.show()

out = np.zeros(img.shape)
for i in range (H):
    for j in range (W):
        x = img[i][j]
        out[i][j] = s[x]

print(out)
cv.imshow('Output', out)
plt.show()
plt.subplot(1, 3, 3)
plt.title('Output Image Histogram')
plt.hist(out.ravel(), 256, [0, 255])
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
