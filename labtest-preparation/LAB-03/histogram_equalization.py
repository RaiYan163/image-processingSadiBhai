import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./labfiles/color_img.jpg', 1)

b, g, r = cv.split(img)

plt.figure(figsize = (25, 25))

L = 256
H, W = b.shape
MN = H * W

blue_hist = cv.calcHist([b], [0], None, [256], [0, 255])
green_hist = cv.calcHist([g], [0], None, [256], [0, 255])
red_hist = cv.calcHist([r], [0], None, [256], [0, 255])

blue_prob = blue_hist / MN
green_prob = green_hist / MN
red_prob = red_hist / MN

plt.subplot(3, 3, 1)
plt.title('Blue Channel Histogram')
plt.hist(b.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 2)
plt.title('Green Channel Histogram')
plt.hist(g.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 3)
plt.title('Red Channel Histogram')
plt.hist(r.ravel(), 256, [0, 255])


cdf_blue = blue_prob
cdf_green = green_prob
cdf_red = red_prob

s_blue = np.zeros(256)
s_green = np.zeros(256)
s_red = np.zeros(256)

s_blue[0] = np.round((L - 1) * cdf_blue[0])
s_green[0] = np.round((L - 1) * cdf_green[0])
s_red[0] = np.round((L - 1) * cdf_red[0])

for i in range (1, 256):
    cdf_blue[i] += cdf_blue[i - 1]
    cdf_green[i] += cdf_green[i - 1]
    cdf_red[i] += cdf_red[i - 1]
    
    s_blue[i] = np.round((L - 1) * cdf_blue[i])
    s_green[i] = np.round((L - 1) * cdf_green[i])
    s_red[i] = np.round((L - 1) * cdf_red[i])
    
for i in range (H):
    for j in range (W):
        x = b[i][j]
        y = g[i][j]
        z = r[i][j]
        
        b[i][j] = s_blue[x]
        g[i][j] = s_green[y]
        r[i][j] = s_red[z]

plt.subplot(3, 3, 4)
plt.title('blue')
plt.hist(b.ravel(), 256, [0, 255])
# plt.show()

plt.subplot(3, 3, 5)
plt.title('blue')
plt.hist(g.ravel(), 256, [0, 255])
# plt.show()

plt.subplot(3, 3, 6)
plt.title('blue')
plt.hist(r.ravel(), 256, [0, 255])
plt.show()

merge = cv.merge((b, g, r))
cv.imshow('merge', merge)

cv.imshow('input', img)
cv.waitKey(0)
cv.destroyAllWindows()