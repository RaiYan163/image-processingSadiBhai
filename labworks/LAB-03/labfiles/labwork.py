import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

img = cv.imread('./color_img.jpg')

b, g, r = cv.split(img)

cv.imshow('Before Equalizing Blue', b)
cv.imwrite('./print/before_equalizing_blue.jpg', b)
cv.imshow('Before Equalizing Green', g)
cv.imwrite('./print/before_equalizing_green.jpg', g)
cv.imshow('Before Equalizing Red', r)
cv.imwrite('./print/before_equalizing_red.jpg', r)

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
plt.savefig('Blue Channel Histogram.jpg')
plt.hist(b.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 2)
plt.title('Green Channel Histogram')
plt.savefig('Green Channel Histogram.jpg')
plt.hist(g.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 3)
plt.title('Red Channel Histogram')
plt.savefig('Red Channel Histogram.jpg')
plt.hist(r.ravel(), 256, [0, 255])
plt.show()

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
plt.title('Blue Channel Equalized Histogram')
plt.savefig('./print/Blue Channel Histogram.jpg')
plt.hist(b.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 5)
plt.title('Green Channel Equalized Histogram')
plt.savefig('./print/Green Channel Equalized Histogram.jpg')
plt.hist(g.ravel(), 256, [0, 255])
plt.show()

plt.subplot(3, 3, 6)
plt.title('Red Channel Equalized  Histogram')
plt.savefig('./print/Red Channel Equalized Histogram.jpg')
plt.hist(r.ravel(), 256, [0, 255])
plt.show()

cv.imshow('Equalized Blue', b)
cv.imshow('Equalized Green', g)
cv.imshow('Equalized Red', r)

cv.imwrite('./print/after_equalizing_blue.jpg', b)
cv.imwrite('./print/after_equalizing_green.jpg', g)
cv.imwrite('./print/after_equalizing_red.jpg', r)

merge = cv.merge((b, g, r))
cv.imshow('Merged', merge)
cv.imwrite('./print/after_equalizing_merged.jpg', merge)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV Input Image", hsv_img)
cv.imwrite('./print/hsv_input_image.jpg', hsv_img)
plt.plot()

h, s, v = cv.split(hsv_img)
cv.imshow("Before Equalizing v channel", v)
cv.imwrite('./print/before_equalizing_v_channel.jpg', v)
plt.plot()

plt.subplot(3, 3, 7)
plt.title('V channel histogram - Before Equalization')
plt.savefig('./print/V channel histogram - Before Equalization.jpg')
plt.hist(v.ravel(), 256, [0, 255])
plt.show()

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

plt.subplot(3, 3, 8)
plt.title('V channel histogram - After Equalization')
plt.hist(v.ravel(), 256, [0, 255])
plt.show()
plt.savefig('./print/V channel histogram - After Equalization.jpg')

cv.imshow("Equalized v channel image", v)
cv.imwrite('./print/after_equalizing_v_channel.jpg', v)
plt.plot()

output = cv.merge((h, s, v))
cv.imshow('Final Merged Image-HSV Equalized', output)
cv.imwrite('./print/after_equalizing_merge.jpg', output)

cv.waitKey(0)
cv.destroyAllWindows()