import cv2
import numpy as np
from scipy import signal


img=np.array([[1,2,3],[4,5,6]])
kernel=np.array([[10,20],[30,40]])

H, W = img.shape
kh, kw = kernel.shape

output_row = kh + H - 1
output_col = kw + W - 1

pad_top = output_row - kh
pad_right = output_col - kw


zero_pad_kernel = np.pad(kernel, ((output_row - kh, 0), (0, output_col - kw)), 'constant', constant_values = 0)
# zero_pad_kernel = cv2.copyMakeBorder(kernel, pad_top, 0, 0, pad_right, cv2.BORDER_CONSTANT)

f = np.zeros((output_row, output_col, W))
for i in range(output_row):
    ft = np.zeros((output_col, W))
    for j in range(kw):
        for l in range(W):
            ft[j + l][l] = zero_pad_kernel[output_row - 1 - i][j]
    f[i] = ft

no_of_matrix, fh, fw = f.shape
print(f.shape)

double_blocked = np.zeros((fh * no_of_matrix, H * fw))

for t in range(H):
    for f_matrix_no in range(no_of_matrix - t):
        for i in range(fh):
            for j in range(fw):
                double_blocked[(f_matrix_no + t) * fh + i][(fw * t) + j] = f[f_matrix_no][i][j]

print(double_blocked)

def matrix_to_vector (input):
    input_h , input_w = input.shape
    output_vector = np.zeros(input_h * input_w, dtype=input.dtype)
    input = np.flipud(input)
    for i, row in enumerate(input):
        st = i * input_w
        nd = st + input_w
        output_vector [st : nd] = row
    return output_vector

img_col_vector = matrix_to_vector(img)

result_col = np.matmul(double_blocked, img_col_vector)

def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype = input.dtype)
    for i in range(output_h):
        st = i * output_w
        nd = st + output_w
        output[i, :] = input[st : nd]
    output = np.flipud(output)
    return output

output = vector_to_matrix(result_col, (output_row, output_col))
output = np.round(output).astype(np.uint32)
print(output)


result = signal.convolve2d (img, kernel, "full")
print(result)
# output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
# output = np.round(output).astype(np.uint8)
# cv2.imshow("convolution.jpg", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
