import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
matplotlib.use('TkAgg')

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('./two_noise.jpeg', 0)
cv2.imshow('Input', img_input)
img = dpc(img_input)
image_size = img.shape[0] * img.shape[1]
# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

point_list=[]
img = magnitude_spectrum_scaled
# img[4:6, 5:7] = 1
# click and seed point set up
x = None
y = None

# The mouse coordinate system and the Matplotlib coordinate system are different, handle that
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))


X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)
print(point_list)

def ButterworthFilter(img, point_list, D, n):
    M = img.shape[0]
    N = img.shape[1]
    H = np.ones_like(img.shape, np.float32)
    for uk, vk in point_list:
        for u in range (M):
            for v in range (N):
                Dk = (u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2
                Dk = math.sqrt(Dk)
                D_k = (u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2
                D_k = math.sqrt(D_k)
                if(Dk == 0.0 or D_k == 0.0):
                    H[u, v] = 0.0
                    continue
                H[u, v] *= (1 / (1 + (D / Dk)) ** (2 * n)) * (1 / (1 + (D / D_k)) ** (2 * n))
                    
    return H 

def point_op(filter, uk, vk):
    Do = 2
    D = 2
    n = 2
    M = filter.shape[0]
    N = filter.shape[1]
    H = np.ones((M,N), np.float32)
    for u in range(M):
        for v in range(N):
            H[u, v] = 1.0
            # Dk = (u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2
            # Dk = math.sqrt(Dk)
            # D_k = (u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2
            # D_k = math.sqrt(D_k)
            dk = ((u - uk)**2 + (v - vk)**2)**(0.5)
            d_k = ((u - (M-uk))**2 + (v - (N-vk))**2)**(0.5)
            if dk == 0 or d_k == 0:
                H[u, v] = 0.0
                continue
            H[u, v] = (1/(1+((Do/dk)**(2*n)))) * (1/(1+((Do/d_k)**(2*n))))
            # H[u, v] = (1/(1+((Do/Dk)**(2*n)))) * (1/(1+((Do/D_k)**(2*n))))
            # H[u, v] = (1 / (1 + (D / Dk)) ** (2 * N)) * (1 / (1 + (D / D_k)) ** (2 * N))
    return H

def butter(filter, points):
    ret = np.ones(filter.shape, np.float32)
    for u,v in points:
        ret *= point_op(filter, u, v)
    return ret

filter = np.ones(img.shape, np.float64)
H = butter(filter, point_list)

# H = ButterworthFilter(img_input, point_list, 2, 2)

cv2.imshow('Filter', H)
cv2.imwrite('./Filter.jpeg', H)

Final = magnitude_spectrum_ac * H

ang = np.angle(ft_shift)
final_result = np.multiply(Final, np.exp(1j * ang))


final_result_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
final_result_back_scaled = min_max_normalize(final_result_back)

cv2.imshow("Final Output",final_result_back_scaled)
cv2.imwrite("Final_Output.jpeg",final_result_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows() 