import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

point_list=[]

def getPoints(img):
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
            point_list.append((x,y))


    X = np.zeros_like(img)
    plt.title("Please select seed pixel from the input")
    im = plt.imshow(img, cmap='gray')
    im.figure.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    
    return point_list
    

img = cv2.imread('./two_noise.jpeg', 0)
img_h, img_w = img.shape

plt.imshow(img, 'gray')
plt.title('Input image')
plt.show()

F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

magnitude = np.log(np.abs(F))

plt.imshow(magnitude, 'gray')
plt.title('Magnitude')
plt.show()

magnitude_shift = np.log(np.abs(F_shift))

plt.imshow(magnitude_shift, 'gray')
plt.title('Magnitude after shift')
plt.show()

pts=getPoints(magnitude_shift)
print(pts)

filter_butter = np.zeros((img_h, img_w), dtype=np.float32)

n =input("Take value n :")
d0 = input("Take radius d0 :")
n=np.uint8(n)
d0 = np.uint8(d0)

v=[]
u=[]
for i in range (len(pts)):
    g,h=pts[i]
    v.append(g)
    u.append(h)
print(v,u)

center_i, center_j = img_h//2, img_w//2

for i in range(img_h):
    for j in range(img_w):
        prod = 1
        for k in range(len(v)):
            duv = np.sqrt((i - center_i - (u[k]-center_i))**2 + (j - center_j - (v[k]-center_j))**2)
            dmuv = np.sqrt((i - center_i + (u[k]-center_i))**2 + (j - center_j + (v[k]-center_j))**2)
            prod *= (1 / (1 + (d0 / duv)**(2*n))) * (1 / (1 + (d0 / dmuv)**(2*n)))
        filter_butter[i, j] = prod

plt.imshow(filter_butter, cmap='gray')
plt.title('Notch filter')
plt.show()

G_shift = F_shift * filter_butter

G = np.fft.ifftshift(G_shift)
output = np.fft.ifft2(G).real

plt.imshow(output, cmap='gray')
plt.title('Output')
plt.show()