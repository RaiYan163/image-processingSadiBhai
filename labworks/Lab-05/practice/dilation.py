import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./img/hand.jpg', 0)
original = img
plt.imshow(img, 'gray')

_, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
img = img // 255
plt.imshow(img, 'gray')

def dilate(img, kernel):
    rows, cols = img.shape
    krows, kcols = kernel.shape

    pad_rows = krows // 2
    pad_cols = kcols // 2
    
    padded_img = cv.copyMakeBorder(img, pad_rows, pad_rows, pad_cols, pad_cols, cv.BORDER_CONSTANT)
    
    output = np.zeros_like(img)
    
    for i in range (rows):
        for j in range (cols):
            
            roi = padded_img[i:i + krows, j:j + kcols]
            
            result = np.zeros_like(roi)
            h, w = result.shape

            for x in range (h):
                for y in range (w):
                    result[x, y] = roi[x, y] & kernel[x, y]
            
            if np.any(result):
                output[i][j] = 255
    
    return output

kernel = np.ones((5, 5), np.uint8)
dilated_img = dilate(img, kernel)
cv.imshow('Original Image', original)
cv.imshow('Dilated Image', dilated_img)

cv.waitKey(0)
cv.destroyAllWindows()