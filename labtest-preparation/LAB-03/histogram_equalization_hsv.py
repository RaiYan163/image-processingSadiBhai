import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./labfiles/color_img.jpg', 1)
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h, s, v = cv.split(hsv_img)
H, W = v.shape
MN = H * W
L = 256

v_hist = cv.calcHist([v], [0], None, [256], [0, 255])
v_prob = v_hist / MN
cdf_v = v_prob

s_v = np.zeros(256)
cdf_v[0] = np.round((L - 1) * v_prob[0])

for i in range (1, 256):
    cdf_v[i] += cdf_v[i - 1]
    s_v[i] = np.round((L - 1) * cdf_v[i])

for i in range (H):
    for j in range (W):
        x = v[i][j]
        v[i][j] = s_v[x]
      
plt.title('histogram')
plt.hist(v.ravel(), 256, [0, 255])
plt.show()

output = cv.merge((h, s, v))
cv.imshow('output', output)

cv.waitKey(0)
cv.destroyAllWindows()