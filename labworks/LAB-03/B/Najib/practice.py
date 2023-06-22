import numpy as np
import cv2
import matplotlib.pyplot as plt
import math



def HistFunc(img,channel):
    hist = cv2.calcHist([img], [channel], None, [256], [0, 255])
    return hist



def EqualizeFunc(img,hist,L=256):
    H,W = img.shape
    
    cdf = hist
    s = np.zeros(cdf.size, dtype=np.float32)
    s[0] = np.round((L - 1) * cdf[0])
    
    for i in range (1, 256):
        cdf[i] += cdf[i - 1]
        s[i] = np.round((L - 1) * cdf[i])
    
    out = np.zeros_like(img)
    for i in range (H):
        for j in range (W):
            x = img[i][j]
            out[i][j] = s[x]

    return out


path = r'D:\Academics\4-1\2.Lab\CSE 4128 (Image Processing Lab)\LAB-03\B\Najib\color_img.jpg'
img = cv2.imread(path,1)
cv2.imshow('Input Image', img)
L = 256


b,g,r = cv2.split(img)
cv2.imshow("Blue Channel",b)
cv2.imshow("Green Channel",g)
cv2.imshow("Red Channel",r)
# =============================================================================
# 
hist_blue = cv2.calcHist([b], [0], None, [256], [0, 255])
hist_green = cv2.calcHist([g], [0], None, [256], [0, 255])
hist_red = cv2.calcHist([r], [0], None, [256], [0, 255])
# 
# 
#print(hist_blue)
# 
plt.figure(figsize = (10, 4))
plt.subplot(1, 3, 1)
plt.title('Input blue Image Histogram')
plt.hist(hist_blue, 256, [0, 255])
plt.show()



plt.subplot(1, 3, 2)
plt.title('Input green Image Histogram')
plt.hist(hist_green, 256, [0, 255])
plt.show()


plt.subplot(1, 3, 3)
plt.title('Input red Image Histogram')
plt.hist(hist_red, 256, [0, 255])
plt.show()


out_blue = EqualizeFunc(b, hist_blue,256)
out_green = EqualizeFunc(g, hist_green,256)
out_red = EqualizeFunc(r, hist_red,256)

cv2.imshow("Output Blue Image",out_blue)
cv2.imshow("Output Green Image",out_green)
cv2.imshow("Output Red Image",out_red)
plt.figure(figsize = (10, 4))


plt.subplot(1, 3, 1)
plt.title('Output Blue Image Histogram')
plt.hist(out_blue.ravel(), 256, [0, 255])
plt.show()
plt.subplot(1, 3, 2)
plt.title('Output Green Image Histogram')
plt.hist(out_green.ravel(), 256, [0, 255])
plt.show()
plt.subplot(1, 3, 3)
plt.title('Output Red Image Histogram')
plt.hist(out_red.ravel(), 256, [0, 255])
plt.show()

output = cv2.merge([out_blue,out_green,out_red])
cv2.imshow("Output",output)
# =============================================================================



h_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV image",h_img)
h,s,v = cv2.split(h_img)

cv2.imshow("HSV_v image",v)
hist_v = cv2.calcHist([v], [0], None, [256], [0, 255])
out_v = EqualizeFunc(v, hist_v,256)
cv2.imshow("what the fuck",out_v)
plt.plot()
plt.title('Output Red Image Histogram')
plt.hist(out_v.ravel(), 256, [0, 255])
plt.show()
img[::2]
h_output = cv2.merge([h,s,out_v])
cv2.imshow("merged",h_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
