import numpy as np
import cv2
import matplotlib.pyplot as plt


width, height = 512, 512

img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

# angle = float(input("Angle: "))
# gammaL = float(input("Gamma Low: "))
# gammaH = float(input("Gamma High: "))
# c = float(input("C: "))
# D0 = float(input("D0: "))

angle = 30
gammaL = 0.2
gammaH = 1.4
c = 10
D0 = 10


M = img.shape[0]
N = img.shape[1]


angle = np.deg2rad(angle)

x = np.linspace(-1, 1, width)
y = np.linspace(-1, 1, height)
xx, yy = np.meshgrid(x, y)

grad_dir = np.array([np.cos(angle), np.sin(angle)])
illum_pattern = grad_dir[0] * xx + grad_dir[1] * yy

illum_pattern -= illum_pattern.min()
illum_pattern /= illum_pattern.max()

corrupt_img = np.multiply(img, illum_pattern)

corrupt_img = cv2.normalize(corrupt_img, None, 0, 1, cv2.NORM_MINMAX)

corrupt_img = np.log1p(corrupt_img)

f = np.fft.fft2(corrupt_img)
shift = np.fft.fftshift(f)
mag = np.abs(shift)
angle = np.angle(shift)

H = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        u = (i - M // 2) ** 2
        v = (j - N // 2) ** 2
        r = np.exp(-((c * (u + v)) / (2 * D0**2)))
        r = (gammaH - gammaL) * (1 - r) + gammaL
        H[i][j] = r


mag = mag * H

op = np.multiply(mag, np.exp(1j * angle))

inv = np.fft.ifftshift(op)

inv = np.real(np.fft.ifft2(inv))

inv = np.exp(inv) - 1

inv = cv2.normalize(inv, None, 0, 1, cv2.NORM_MINMAX)

f1 = plt.figure(1)
plt.imshow(np.log(np.abs(shift)), "gray")
f1 = plt.figure(2)
plt.imshow(illum_pattern, cmap="gray")
plt.title("Illumination Pattern")
# f1 = plt.figure(3)
# plt.imshow(corrupt_img, cmap='gray')
# plt.title('Corrupted Image')
# f1 = plt.figure(4)
# plt.imshow(inv, cmap='gray')
# plt.title('Output')
# f1 = plt.figure(5)
# plt.imshow(img, cmap='gray')
# plt.title('Input')
plt.show()
cv2.imshow("output", inv)
cv2.imshow("Corrupt", corrupt_img)
cv2.imshow("Input", img)
cv2.waitKey()
cv2.destroyAllWindows()
