import numpy as np
import cv2

import matplotlib.pyplot as plt

img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('INPUT',img)
#out=img.copy()
out = np.zeros((512,512), dtype=np.uint8)
print(img.max())
print(img.min())

#cv2.imshow('output image',out)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i,j)
        out.itemset((i,j),255-a)
        
cv2.imshow('output image',out)

cv2.imshow('output image',img)


blur = cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)#(3, 3) is the kernel size, and 0 is the standard deviation value.
#cv2.imshow('output image1',blur)
blur = cv2.GaussianBlur(img,(7,7),0)#(3, 3) is the kernel size, and 0 is the standard deviation value.
#cv2.imshow('output image12',blur)
median = cv2.medianBlur(img,5)
kernel2 =(1/16)* np.array([[-1,2,1],
                    [2,4,2],
                    [1,2,1]], np.float32)
  

edge = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
#cv2.imshow("edge",edge)
print(edge)


cv2.waitKey(0)
cv2.destroyAllWindows()





#cv2.normalize(src,des, 0, 255, cv2.NORM_MINMAX)
#des = np.round(src).astype(np.uint8)