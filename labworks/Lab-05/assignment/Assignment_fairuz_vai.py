import numpy as np
import cv2 as cv
import math

def Tapestry(inpImg, a_x, a_y, xc, yc, tau_x, tau_y):
    height = inpImg.shape[0]
    width = inpImg.shape[1]
    outImg = np.zeros_like(inpImg)
    
    for x in range(width):
        for y in range(height):
            xx = x+a_x*(math.sin( ((2*math.pi)/tau_x)*(x-xc)))
            yy = y+a_y*(math.sin( ((2*math.pi)/tau_y)*(y-yc)))
            
            if 0 <= xx < width and 0 <= yy < height:
               x0 = int(xx)
               y0 = int(yy)
               if x0+1 < width:
                   x1=x0+1
               else:
                   x1=x0
               if y0+1 < height:
                   y1=y0+1
               else:
                   y1=y0
               
               f00 = inpImg[y0, x0]
               f01 = inpImg[y0, x1]
               f10 = inpImg[y1, x0]
               f11 = inpImg[y1, x1]
               
               dx = xx - x0
               dy = yy - y0
               
               outImg[y, x] = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f01 + (1 - dx) * dy * f10 + dx * dy * f11
            else:
               outImg[y, x] = 0  
    
    return outImg


def Ripple(img, a_x, a_y, tau_x, tau_y):
    height = img.shape[0]
    width = img.shape[1]
    outImg = np.zeros_like(img)

    for x in range(width):
        for y in range(height):
            xx = x + a_x * math.sin((2 * math.pi * y) / tau_x)
            yy = y + a_y * math.sin((2 * math.pi * x) / tau_y)

            if 0 <= xx < width and 0 <= yy < height:
                x0 = int(x_prime)
                y0 = int(y_prime)
                
                if x0+1 < width:
                    x1=x0+1
                else:
                    x1=x0
                if y0+1 < height:
                    y1=y0+1
                else:
                    y1=y0
                    
                f00 = img[y0, x0]
                f01 = img[y0, x1]
                f10 = img[y1, x0]
                f11 = img[y1, x1]

                dx = xx - x0
                dy = yy - y0

                outImg[y, x] = (1 - dx) * (1 - dy) * f00 + dx * (1 - dy) * f01 + (1 - dx) * dy * f10 + dx * dy * f11
            else:
                outImg[y, x] = 0  # Handle out-of-bounds cases

    return outImg

img = cv.imread('./../img/input2.png')
cv.imshow("Input Image", img)

out = Tapestry(img, 5, 5, 0, 0, 30, 30)
cv.imshow("Output Image", out)

img = cv.imread('./../img/input1.jpg')
cv.imshow("Input Image", img)

out = Ripple(img, 10, 15, 50, 70)
cv.imshow("Output Image", out)



cv.waitKey()
cv.destroyAllWindows()