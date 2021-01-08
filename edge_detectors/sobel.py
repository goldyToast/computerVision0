import numpy as np
from numba import jit

import filters
import utils


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