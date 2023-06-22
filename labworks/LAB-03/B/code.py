import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

#----------------------------------------RGB---------------------------------------------#

img = cv.imread('./color_img.jpg')
b, g, r = cv.split(img)
L = 256
H, W = b.shape
MN = H * W

blue_hist = cv.calcHist([b], [0], None, [256], [0, 255])
green_hist = cv.calcHist([g], [0], None, [256], [0, 255])
red_hist = cv.calcHist([r], [0], None, [256], [0, 255])

blue_prob = blue_hist / MN
green_prob = green_hist / MN
red_prob = red_hist / MN

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

merge = cv.merge((b, g, r))
         

#----------------------------------------HSV---------------------------------------------#

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_img)

v_hist = cv.calcHist([v], [0], None, [256], [0, 255])

v_prob = v_hist / MN

s_v = np.zeros(256)
cdf_v = v_prob
s_v[0] = np.round((L - 1) * cdf_v[0])

for i in range (1, 256):
    cdf_v[i] += cdf_v[i - 1]
    s_v[i] = np.round((L - 1) * cdf_v[i])


for i in range (H):
    for j in range (W):
        x = v[i][j]
        v[i][j] = s_v[x]
        
output = cv.merge((h, s, v))