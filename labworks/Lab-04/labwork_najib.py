import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
import matplotlib
from math import sqrt,ceil
matplotlib.use('TkAgg')

img_input = cv2.imread('./two_noise.jpeg', 0)
img = dpc(img_input)
image_size = img.shape[0] * img.shape[1]

point_list=[]
x = None
y = None

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

X = np.zeros_like(img_input)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img_input, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block = True)
print(point_list)


def FilterFunc(uk,vk,d0,n,img):
    M,N = img.shape
    Hrn = np.ones((M,N),np.float32)
    for u in range(M):
        for v in range(N):
            dk = sqrt(((u-(M/2)-uk))**2 + ((v-(M/2)-vk))**2)
            dk_ = sqrt(((u-(M/2)+uk))**2 + ((v-(M/2)+vk))**2)
            if(dk == 0.0 or dk_ == 0.0):
                Hrn[u][v] = 0.0
            else:
                Hrn[u][v] = ((1/(1+((d0/dk))**(2*n)))*(1/(1+((d0/dk_))**(2*n))))
    return Hrn

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')


ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)


coOr = co_ordinate(magnitude_spectrum_scaled)

u1 = coOr[0]
v1 = coOr[1]

u2 = coOr[2]
v2 = coOr[3]

print(u1,v1)
print(u2,v2)

d0 = 10
n = 100


M, N = img.shape


filter1 = FilterFunc(u1, v1, d0, n, img)
filter2 = FilterFunc(u2, v2, d0, n, img)
f_filter = filter1 * filter2


magnitude_spectrum_ac = magnitude_spectrum_scaled * f_filter
cv2.imshow("final filter",f_filter)

ang = np.angle(ft_shift)

final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

filtered_spectrum = np.fft.ifftshift(final_result)
img_back = np.abs(np.fft.ifft2(filtered_spectrum))

cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("Inverse transform",img_back)


cv2.waitKey(0)
cv2.destroyAllWindows() 
