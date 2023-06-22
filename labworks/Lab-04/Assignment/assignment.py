import numpy as np
import cv2 as cv
import math

img = cv.imread('./lena.jpg', cv.IMREAD_GRAYSCALE)
cv.imshow("Input", img)

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)
    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j] - inp_min) / (inp_max - inp_min)) * 255)
    return np.array(img_inp, dtype='uint8')

def IlluminationPatternGenerator(img, x, y, sigma):
    pattern = np.zeros_like(img, np.float32)
    for i in range (x):
        for j in range (y):
            constant = 2 * math.pi * (sigma ** 2)
            result = -(i * i + j * j)
            result /= 2 * (sigma ** 2)
            numerator = math.exp(result)
            pattern[i, j] = numerator / constant
    return pattern

pattern1 = IlluminationPatternGenerator(img, 400, 400, 100)
pattern2 = IlluminationPatternGenerator(img, 500, 500, 85)

pattern2 = np.flip(pattern2, 0)
pattern2 = np.flip(pattern2, 1)

pattern = pattern1 + pattern2
pattern = cv.normalize(pattern, None, 0, 255, cv.NORM_MINMAX)
pattern = pattern.astype(np.uint8)


cv.imshow("Pattern", pattern)
cv.imwrite("Pattern.jpg", pattern)

img = img.astype(np.uint32)
corrupted_img = img + pattern
corrupted_img = corrupted_img.astype(np.float32)
corrupted_img = cv.normalize(corrupted_img, None, 0, 255, cv.NORM_MINMAX)
corrupted_img = corrupted_img.astype(np.uint8)

# img = img.astype(np.uint8)
# corrupted_img = cv.add(img, pattern)

cv.imshow("Corrupted Image", corrupted_img)
cv.imwrite("./Corrupted_Image.jpg", corrupted_img)


def getHomomorphicKernel():
    gh = 1.4
    gl = 0.5
    D0 = 8
    c = 3
    M = img.shape[0]
    N = img.shape[1]
    kernel = np.zeros(img.shape)
    for i in range (M):
        for j in range (N):
            dk = np.sqrt((i - M // 2) ** 2 + (j - N // 2) ** 2)
            power = -c * ((dk ** 2) / (D0 ** 2))
            kernel[i, j] = (gh - gl) * (1 - np.exp(power)) + gl
    return kernel


kernel = getHomomorphicKernel()

ft = np.fft.fft2(corrupted_img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

cv.imshow("Magnitude Spectrum", magnitude_spectrum_scaled)
cv.imwrite("./Magnitude_Spectrum.jpg", magnitude_spectrum_scaled)

ang = np.angle(ft_shift)
output = np.multiply(magnitude_spectrum_ac * kernel, np.exp(1j * ang))

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(output)))
img_back_scaled = min_max_normalize(img_back)

cv.imshow("Homomorphic Filtered Image", img_back_scaled)
cv.imwrite("./Homomorphic_Filtered_Image.jpg", img_back_scaled)

cv.waitKey(0)
cv.destroyAllWindows()