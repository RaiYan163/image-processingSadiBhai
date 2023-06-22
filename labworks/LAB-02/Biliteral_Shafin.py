import numpy as np
import cv2
import math

def spatial_gaussian_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype="float32")
    k = ksize // 2
    gconst = sigma**2
    gconst = 2 * math.pi * gconst
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            kernel[i + k][j + k] = (
                math.exp(-((i * i) + (j * j)) / (2 * sigma**2))
            ) / gconst
    return kernel


def range_gaussian_kernel(img, x, y, ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype="float32")
    gconst = np.sqrt(2 * math.pi) * sigma
    k = ksize // 2
    Ip = img[x][y]
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            Iq = img[x + i][y + j]
            kernel[i + k][j + k] = (
                math.exp(-((Ip - Iq) ** 2) / (2 * sigma * sigma))
            ) / gconst
    return kernel


def multiplication(sp_kernel, rng_kernel, ksize):
    final_kernel = np.zeros((ksize, ksize), dtype="float32")
    for i in range(ksize):
        for j in range(ksize):
            final_kernel[i][j] = sp_kernel[i][j] * rng_kernel[i][j]

    summ = final_kernel.sum()
    final_kernel = final_kernel / summ
    return final_kernel


img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
ksize = 5
sigma = 5
height = img.shape[0]
width = img.shape[1]
output = np.zeros((height, width), dtype="float32")
k = ksize // 2
bordered_output = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_REPLICATE)

sp_kernel = spatial_gaussian_kernel(ksize,  sigma = 4)

for x in range(height):
    for y in range(width):
        rng_kernel = range_gaussian_kernel(bordered_output, x + k, y + k, ksize, 0.4)
        # kernel = multiplication(sp_kernel, rng_kernel, ksize)
        kernel = sp_kernel * rng_kernel
        kernel /= np.sum(kernel)
        for i in range(ksize):
            for j in range(ksize):
                output[x][y] += (
                    bordered_output[x + i][y + j] * kernel[ksize - 1 - i][ksize - 1 - j]
                )

# output=output*255
cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv2.imshow("Actual", img)
cv2.imshow("Bilateral", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
