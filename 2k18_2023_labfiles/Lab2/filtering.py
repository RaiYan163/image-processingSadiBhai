import numpy as np
import cv2


img = cv2.imread('pic.png',cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)


#dst = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
blur = cv2.GaussianBlur(img,(7,7),3,cv2.BORDER_DEFAULT)
cv2.imshow('output image1',blur)



#sigma_color = float - Sigma for grey or color value. 
#For large sigma_color values the filter becomes closer to gaussian blur.
#sigma_spatial: float. Standard ev. for range distance.
#Increasing this smooths larger features.


#cv2.bilateralFilter ( src, dst, d, sigmaColor,sigmaSpace, borderType = BORDER_DEFAULT )
bilateral = cv2.bilateralFilter(img, -1, 75, 2)
cv2.imshow('output image2',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()





#cv2.normalize(src,des, 0, 255, cv2.NORM_MINMAX)
#s = np.round(s).astype(np.uint8)