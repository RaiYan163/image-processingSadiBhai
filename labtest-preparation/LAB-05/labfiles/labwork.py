import cv2 as cv
import numpy as np

img = cv.imread('./argho.png', 0)
cv.imshow('Input', img)

print(img)

k1 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 0]
    ], np.uint8) * 255

k2 = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 1]
    ], np.uint8) * 255

k3 = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0]
    ], np.uint8) * 255

w = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    ], np.uint8) * 255

rate = 50
k1 = cv.resize(k1, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
k2 = cv.resize(k2, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
k3 = cv.resize(k3, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
w = cv.resize(w, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)



# cv.imshow('K1', k1)
# cv.imshow('K2', k2)
# cv.imshow('K3', k3)
# cv.imshow('W', w)


d1 = w - k1
d2 = w - k2
d3 = w - k3

complement = cv.bitwise_not(img)
print(complement)

k = np.ones((50, 50))

res1 = cv.erode(img, k1, iterations = 1)
res2 = cv.erode(complement, d1, iterations = 1)
final1 = cv.bitwise_and(res1, res2)

final1 = cv.dilate(final1, k, iterations = 1)
cv.imshow('FINAL 1', final1)

res3 = cv.erode(img, k2, iterations = 1)
res4 = cv.erode(complement, d2, iterations = 1)
final2 = cv.bitwise_and(res3, res4)
final2 = cv.dilate(final2, k, iterations = 1)
# cv.imshow('FINAL 2', final2)

res5 = cv.erode(img, k3, iterations = 1)
res6 = cv.erode(complement, d3, iterations = 1)
final3 = cv.bitwise_and(res5, res6)
final3 = cv.dilate(final3, k, iterations = 1)
# cv.imshow('FINAL 3', final3)


# cv.imwrite('./../img/K1.jpg', k1)
# cv.imwrite('./../img/K2.jpg', k2)
# cv.imwrite('./../img/K3.jpg', k3)
# cv.imwrite('./../img/Output_1.jpg', final1)
# cv.imwrite('./../img/Output_2.jpg', final2)
# cv.imwrite('./../img/Output_3.jpg', final3)

cv.waitKey(0)
cv.destroyAllWindows()