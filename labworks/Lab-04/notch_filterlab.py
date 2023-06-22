# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:52:40 2023

@author: itxpa
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp


img = cv.imread('two_noise.jpeg',cv.IMREAD_GRAYSCALE)

point_list=[]


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
print(point_list)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)


M = img.shape[0]
N = img.shape[1]

D0 = 5

x = N//2

y = N//2

a = M - x
b = N - y
p = 1

filter = np.zeros((M,N),np.float32)
f = np.fft.fft2(img)
shift = np.fft.fftshift(f)
mag = np.abs(shift)
angle = np.angle(shift)

f1 = plt.figure(3)
plt.title("input image spectrum")
plt.imshow(np.log(mag),'gray')

pp = np.log(mag)


