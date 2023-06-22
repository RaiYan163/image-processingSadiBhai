import cv2
import numpy as np

def homomorphic_filter(image, gamma_l, gamma_h, c, d0):
    # Convert the image to floating point representation
    image_float = np.float64(image)

    # Take the logarithm of the image
    log_image = np.log(image_float)

    # Perform Fourier Transform
    f = np.fft.fft2(log_image)

    # Center the spectrum
    fshift = np.fft.fftshift(f)

    # Define the high-pass filter
    rows, cols = image.shape
    x = np.arange(-cols/2, cols/2, 1)
    y = np.arange(-rows/2, rows/2, 1)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D**2 / d0**2))) + gamma_l

    # Apply the filter to the spectrum
    filtered_spectrum = fshift * H

    # Perform inverse Fourier Transform
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))

    # Take the exponential of the image
    filtered_image = np.exp(np.real(filtered_image))

    # Convert the image back to the original data type
    filtered_image = np.uint8(filtered_image)

    return filtered_image

# Read the input image
image = cv2.imread('./../corrupt_image.jpeg', 0)  # Read as grayscale

# Set the parameters for the homomorphic filter
gamma_l = 0.3
gamma_h = 1.5
c = 1
d0 = 10

# Apply the homomorphic filter
filtered_image = homomorphic_filter(image, gamma_l, gamma_h, c, d0)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
