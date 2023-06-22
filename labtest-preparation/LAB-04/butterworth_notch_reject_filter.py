import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

img = cv.imread('./labfiles/two_noise.jpeg', 0)
cv.imshow('input', img)

def min_max_normalize(img):
    h, w = img.shape
    inp_min = np.min(img)
    inp_max = np.max(img)

    for i in range (h):
        for j in range (w):
            img[i][j] = ((img[i][j] - inp_min) / (inp_max - inp_min)) * 255
    return np.array(img, np.uint8)

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

seed_img = magnitude_spectrum_scaled
point_list = []
x = None
y = None

def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int (round(x))
        y = int (round(y))
        point_list.append((x, y))


plt.title('Select Seed Pixel')
im = plt.imshow(seed_img, cmap = 'gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block = True)
print(point_list)

def ButterWorthKernel(img, point_list, D0, n):
    M, N = img.shape
    H = np.ones_like(img, np.float32)
    for uk, vk in point_list:
        for u in range (M):
            for v in range (N):
                # Dk = (u - M // 2 - (vk - M // 2)) ** 2 + (v - N // 2 - (uk - N // 2)) ** 2
                Dk = (u - vk) ** 2 + (v - uk) ** 2
                Dk = math.sqrt(Dk)
                # D_k = (u - M // 2 + (vk - M // 2)) ** 2 + (v - N // 2 + (uk - N // 2)) ** 2
                D_k = (u - M + vk) ** 2 + (v - N + uk) ** 2
                D_k = math.sqrt(D_k)
                if(Dk == 0.0 or D_k == 0.0):
                    H[u, v] = 0
                    continue
                H[u, v] *= (1 / (1 + (D0 / Dk)) ** (2 * n)) * (1 / (1 + (D0 / D_k)) ** (2 * n))
    return H

H = ButterWorthKernel(img, point_list, D0 = 1, n = 2)
cv.imshow('Filter', H)

# output_shift = ft_shift * H
# output = np.fft.ifftshift(output_shift)
# output = np.fft.ifft2(output).real
# output = min_max_normalize(output)
# cv.imshow('output', output)

Final = magnitude_spectrum_ac * H
ang = np.angle(ft_shift)
final_result = np.multiply(Final, np.exp(1j * ang))

final_result_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
final_result_back_scaled = min_max_normalize(final_result_back)
cv.imshow("Final Output",final_result_back_scaled)

cv.waitKey(0)
cv.destroyAllWindows()