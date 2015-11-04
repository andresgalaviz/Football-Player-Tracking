import numpy as np
np.set_printoptions(threshold=np.nan)
import numpy.linalg as la
import cv2
import cv2.cv as cv
import math
import matplotlib.pyplot as plt
import time

# Count runtime

start_time = time.time()

# Greyscale intensity levels

n_greyscale_levels = 256

# Sobel kernels

sobel_kernel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_kernel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# is the pixel at the image border?
# Edge strength detector ignores border pixels

def is_border_pixel(row, col, n_rows, n_cols):
    if (row == 0 or row == n_rows - 1):
	return True
    if (col == 0 or col == n_cols - 1):
	return True
    return False

# Convolve an image with a 3x3 kernel

def convolve(img, kernel):  
    img_shape = img.shape
    n_rows = img_shape[0]
    n_cols = img_shape[1]
    result = np.zeros([n_rows, n_cols])
    for i in range(0, n_rows - 1):
	for j in range(0, n_cols - 1):
	    if (not is_border_pixel(i, j, n_rows, n_cols)):
		upper_left = img[i - 1, j - 1] * kernel[0, 0]
		left = img[i, j - 1] * kernel[1, 0]
		lower_left = img[i + 1, j - 1] * kernel[2, 0]
		upper_centre = img[i - 1, j] * kernel[0, 1]
		centre = img[i, j] * kernel[1, 1]
		lower_centre = img[i + 1, j] * kernel[2, 1]
		upper_right = img[i - 1, j + 1] * kernel[0, 2]
		right = img[i, j + 1] * kernel[1, 2]
		lower_right = img[i + 1, j + 1] * kernel[2, 2]
		result[i, j] = upper_left + left + lower_left + upper_centre + centre + lower_centre + upper_right + right + lower_right
    return result

# returns a 2d gaussian kernel

def gauss_kernels(size,sigma=1):
    if size<3:
	size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
    	kernel = kernel/kernel_sum
    return kernel

# Compute the harris corner response given the matrix W

def harris_corner_response(W, k=0.06):
    detW = la.det(W)
    traceW = np.trace(W)
    response = detW - (k * traceW**2)
    return response

# Compute the harris corner response on every pixel of the image
# return the responses in a 2D array

def detect_corners(img):
    gx = convolve(img, sobel_kernel_h)
    gy = convolve(img, sobel_kernel_v)
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy
    size = 3
    sigma = 1
    kernel = gauss_kernels(size, sigma)
    W_xx = convolve(I_xx, kernel)
    W_xy = convolve(I_xy, kernel)
    W_yy = convolve(I_yy, kernel)
    img_shape = img.shape
    n_rows = img_shape[0]
    n_cols = img_shape[1]
    corners = np.zeros([n_rows, n_cols])
    for i in range(0, n_rows - 1):
	for j in range(0, n_cols - 1):
	    W = np.matrix([[W_xx[i, j], W_xy[i, j]], [W_xy[i, j], W_yy[i, j]]])
	    corners[i, j] = harris_corner_response(W)
    return corners

# Overlay the corners on top of the image

def show_corners(img, corners, corners_threshold_factor=0.1):
    corners_max = np.max(corners)
    print corners_max
    corners_threshold = corners_threshold_factor * corners_max
    y, x = np.where(corners >= corners_threshold)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.hold(True)
    plt.scatter(x, y, color='blue')
    plt.show()

def main():
    # Read the .mp4 video using OpenCV Python API cv2.VideoCapture

    cap = cv2.VideoCapture("football_right.mp4")

    # Print the frame width, frame height, frames per second 
    # and frame count of the input video using cap.get

    fwidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CV_CAP_PROP_FPS)
    fcount = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

    print "Frame width: " + str(fwidth) + "\nFrame height: " + str(fheight) + "\nFrames per second: " + str(fps) + "\nFrame count: " + str(fcount)

    _,img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = detect_corners(img)
    show_corners(img, corners)

    cap.release()

main()
print("--- Done! %s seconds ---" % (time.time() - start_time))
