import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("eye.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histr = cv2.calcHist([img],[0],None,[256],[0,255])
plt.plot(histr)



plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

img2 = cv2.equalizeHist(img)


plt.subplot(1, 2, 2)
plt.title("output Image Histogram")
plt.hist(img2.ravel(),256,[0,255])


cv2.imshow("output",img2)

plt.show()

## Convert image from RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.subplot(2, 2, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'b')

# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

plt.subplot(2, 2, 4)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'g')


# Convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)



cv2.waitKey(0)
cv2.destroyAllWindows()