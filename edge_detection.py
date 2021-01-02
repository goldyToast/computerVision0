# This file creates a function that performs the Sobel Edge detector

import numpy as np
from numba import jit

import filters
import utils
import sys


# Takes one Numpy array and applies a Sobel edge detector
def sobel_detector(image, sigma=1):
    # Step 1: Smooth the image with a Gaussian
    gauss = filters.gauss1D(sigma)
    gauss = gauss[np.newaxis]
    
    # Doing two 1D gaussian filters saves time over doing one 2D filter (2n vs n^2)
    im = utils.convolve2d(image, gauss)
    im = utils.convolve2d(im, gauss.T)

    # Step 2: Apply the Sobel Edge detector
    im = central_difference(im)

    return im


# Apply Central differencing filters in both the x and y directions then average the result
def central_difference(image):

    central_y = filters.central_y()
    central_x = filters.central_x()

    x = utils.convolve2d(image, central_y)
    y = utils.convolve2d(image, central_x)
    return utils.combine_arrays(x,y)



# Gaussian allows the option of running a gaussian beforehand to smooth out noise *even more* if desired
def marr_hildreth_detector(image, log, gaussian = 0):
    if gaussian:
        gauss = filters.gauss1D(2)
        gauss = gauss[np.newaxis]
        image = utils.convolve2d(image, gauss)
        image = utils.convolve2d(image, gauss.T)

    np.set_printoptions(threshold=sys.maxsize)
    
    edges = marr_hildreth_convolution(image, log)

    return edges

    
# Convolves an image with the filter, making adjustments if the filter has even-parity dimensions
# Accelerate with jit
@jit(nopython=True)
def marr_hildreth_convolution(image, filter):
    # Return array, image dimensions, filter dimensions
    ret = np.zeros(image.shape)
    i_w, i_h = image.shape
    f_w, f_h = filter.shape

    # Parity is 1 if its even, as we will use this value to adjust the centre/indices
    x_parity = 1 if f_w % 2 == 0 else 0
    y_parity = 1 if f_h % 2 == 0 else 0

    # Offset: Distance from the centre of the filter to the edge
    # Centre: Index of the filter's centre adjusted for even parity
    offset_x, offset_y = f_w // 2, f_h // 2
    centre_x, centre_y = f_w // 2 - x_parity, f_h // 2 - y_parity
    
    # Loop through original image's indices
    for x in range(centre_x, i_w - offset_x):
        for y in range(centre_y, i_h - offset_y):
            val = 0

            # Apply filter to the pixels around the image, adjusting the indices if the dimension is odd
            for i in range(-offset_x + x_parity, offset_x + 1):
                for j in range(-offset_y + y_parity, offset_y + 1):
                    val = val + image[x + i, y + j] * filter[centre_x - i + x_parity, centre_y - j + y_parity]
            ret[x,y] = val

    w, h = ret.shape

    fin = np.zeros_like(ret)
    for i in range(w - 1):
        for j in range(h - 1):
            curr = np.sign(ret[i,j])
            if  curr != np.sign(ret[i+1,j]) or curr != np.sign(ret[i,j+1]):
                fin[i,j] = 255
    
    return fin