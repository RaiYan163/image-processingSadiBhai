import numpy as np
import cv2 as cv
import math

def Tapestry():
    img = cv.imread("./../img/tapestry_input.png", 1)
    a = 5
    tx, ty = 30, 30

    M = img.shape[0] // 2
    N = img.shape[1] // 2

    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            u = i + a *  np.sin((2 * math.pi / tx) * (i - M))
            v = j + a * np.sin((2 * math.pi / ty) * (j - N))
            u = np.round(u).astype(np.uint32)
            v = np.round(v).astype(np.uint32)
            for k in range(3):
                if 0 <= u < img.shape[0] and 0 <= v < img.shape[1]:
                    output[i, j, k] = img[u, v, k]
                else:
                    output[i, j, k] = 0

    cv.imshow('Tapestry Input Image', img)
    cv.imshow('Tapestry Output Image', output)
    cv.imwrite('TapestryOutput.jpg', output)

def Ripple():
    img = cv.imread("./../img/ripple_input.jpg", 1)
    ax = 10
    ay = 10
    tx, ty = 20, 20

    # ax = 10
    # ay = 15
    # tx, ty = 50, 70

    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            u = i + ax * np.sin((2 * np.pi * j) / tx)
            v = j + ay * np.sin((2 * np.pi * i) / ty)
            u = np.round(u).astype(np.uint32)
            v = np.round(v).astype(np.uint32)
            for k in range(3):
                if 0 <= u < img.shape[0] and 0 <= v < img.shape[1]:
                    output[i, j, k] = img[u, v, k]
                else:
                    output[i, j, k] = 0

    cv.imshow('Ripple Input Image', img)
    cv.imshow('Ripple Output Image', output)
    cv.imwrite('RippleOutput.jpg', output)    

Tapestry()
Ripple()

cv.waitKey()
cv.destroyAllWindows()
