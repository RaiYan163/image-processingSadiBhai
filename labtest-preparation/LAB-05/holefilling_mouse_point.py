import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('./labfiles/bubbles.jpg', 0)
cv.imshow('input', img)
seed_img = img
# img = img // 255
print(img)

point_list = []
x = None
y = None

def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        point_list.append((y, x))
plt.title('Select Seed Pixel')
im = plt.imshow(seed_img, cmap = 'gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block = True)
print(point_list)


complement = 255 - img

kernel = np.zeros((5, 5), np.uint8)
kernel[0, 0] = kernel[0, 2] = kernel[2, 0] = kernel[2, 2] = 0
kernel[:, 1] = kernel[1, :] = 1

plt.imshow(kernel)

fill = np.zeros_like(img)
for (x, y) in point_list:
    fill[x, y] = 255

while True:
    prev = fill
    output = cv.dilate(fill, kernel, iterations = 1)
    fill = np.bitwise_and(output, complement)
    # plt.imshow(fill)
    # plt.show()
    if np.array_equal(fill, prev):
        break

output = np.bitwise_or(img, fill)
cv.imshow('output', output)

cv.waitKey(0)
cv.destroyAllWindows()