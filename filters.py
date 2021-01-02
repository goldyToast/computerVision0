import numpy as np

import utils

# Just assorted pre-defined filters returned by calling a function to create the numpy array
# IMPORTANT NOTE: Most fitlers are returned as 2D even if they're 1D for consistency in convolution

# Box filter
def box_filter(size):
    ret = np.ones((size,size))
    return ret / ret.sum()

# Returns a central differencing filter (x direction)
def central_x():
    return np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

# Returns a central differencing filter (y direction)
def central_y():
    return np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


# Returns a 1D Gaussian filter
def gauss1D(sigma):
    half = int(np.ceil(6*sigma)) // 2
    size = half * 2 + 1
    
    filter = np.arange(-half,half+1)
    
    gaussian = lambda x: np.exp(-x**2 / (2 * sigma**2))
    vec = np.vectorize(gaussian)

    filter = vec(filter)

    return filter / filter.sum()


def gaussian_2D(size, sigma=1):
    gauss_1D = gauss1D(size,sigma=sigma)
    gauss_2D = np.zeros((size,size))
    return  gauss_2D


# Returns a 2D Laplacian of a Gaussian Filter
# Set round to true if rounding the float64's is desired
#   - Accuracy will drop HEAVILY if rounding is used
def lap_of_gauss(sigma, round=False):
    centre = int(np.ceil(6*sigma)) // 2
    size = centre * 2 + 1

    log = np.zeros((size,size))

    front = -1 / (2 * np.pi * (sigma ** 4))
    sigma2 = sigma ** 2

    for i in range(centre + 1):
        for j in range(centre + 1):
            # Intermitent calculations
            squared_sum = i ** 2 + j ** 2
            temp_1 = squared_sum / sigma2
            temp_2 = -(squared_sum / (2 * sigma2))

            # Actual LoG calculation using the intermitent calculations
            val = front * (2 - temp_1) * np.exp(temp_2)

            # Assign to all possible locations
            log[centre + i, centre + j] = val
            log[centre + i, centre - j] = val
            log[centre - i, centre + j] = val
            log[centre - i, centre - j] = val
    if round:
        return np.around((log / log.sum()),2)
    return (log / log.sum())
