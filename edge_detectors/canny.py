import numpy as np
from numba import jit

import filters
import utils


def canny_detector(image, low, high, sigma=1):
    # Step 1: Smooth image with Gaussian to remove any noise
    gauss = filters.gauss1D(sigma)[np.newaxis]
    im = utils.convolve2d(image, gauss)
    im = utils.convolve2d(im, gauss.T)

    # Step 2: Get directional derivatives of the smoothed image
    derivative = np.array([[-1,1]])
    im_x = canny_convolution(im, derivative)
    im_y = canny_convolution(im, derivative.T)

    # Step 3: Create array consisting of gradient direction/magnitude
    width, height = image.shape
    gradient = np.empty((width, height, 4), dtype=np.float64)

    gradient[:,:,0] = im_x
    gradient[:,:,1] = im_y
    gradient[:,:,2] = gradient_magnitude(im_x, im_y)
    gradient[:,:,3] = gradient_direction(im_x, im_y)

    # Step 4: Non-Max Suppression
    suppressed = non_max_suppression(gradient)
    # sup_2 = suppressed[:,:,2].astype('uint8')
    # sup_2 = sup_2 * int(255 / np.max(sup_2) - 1)

    # Step 5: Thresholding
    thresholded = thresholds(suppressed, low, high)
    
    # Step 6: Hysteresis
    final = canny_hysteresis(thresholded)
    return final


# NEED IT TO RETURN A FLOAT IN THIS INSTANCE
#   TODO: MAKE NORMAL CONVOLUTION ABLE TO RETURN FLOATS
@jit(nopython=True)
def canny_convolution(image, filter):
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
    
    return ret


def gradient_magnitude(im_x, im_y):
    assert im_x.shape == im_y.shape
    
    x_squared = np.square(im_x)
    y_squared = np.square(im_y)
    magnitude = np.sqrt(x_squared + y_squared)
    return magnitude


# Get the direction of the gradient
def gradient_direction(im_x, im_y):
    dirs = -np.arctan2(im_y, im_x)
    vec = np.vectorize(lambda x: (x + 3.14) if x < 0 else x)
    dirs = np.around(vec(dirs), 2)
    return dirs


@jit(nopython=True)
def non_max_suppression(gradient):

    # GET WHERE MAGNITUDE IS NON ZERO
    indices = np.nonzero(gradient[:,:,2])

    ret = np.copy(gradient)

    for (x,y) in zip(*indices):
        to_check = find_adjacents_np(gradient, x, y)

        good = 0
        for i in range(to_check.shape[0]):
            i,j = x + to_check[i,0], y + to_check[i,1]

            if gradient[x,y,2] < gradient[int(i),int(j),2]:
                ret[x,y,2] = 0
                ret[x,y,1] = 0
                ret[x,y,0] = 0

    return ret


# Returns list of coordinates to check
@jit(nopython=True)
def find_adjacents_np(gradient, i, j):
    dir = gradient[i,j,3]

    # If either of the derivatives are 0, we just need to check the pixels perpendicular to that derivative
    if dir == 0.0:
        return np.array([[1,0], [-1,0]]).astype(np.float64)
    if dir == 1.57:
        return np.array([[0,1], [0,-1]]).astype(np.float64)
    if dir == 0.79:
        return np.array([[1,-1], [-1,1]]).astype(np.float64)
    if dir == 2.36:
        return np.array([[1,1], [-1,-1]]).astype(np.float64)

    if dir < 0.79:
        return np.array([[1,0], [1,-1], [-1,0], [-1,1]]).astype(np.float64)
    elif dir < 1.57:
        return np.array([[0,1], [1,-1], [0,-1], [-1,1]]).astype(np.float64)
    elif dir < 2.36:
        return np.array([[0,1], [-1,-1], [0,-1], [1,1]]).astype(np.float64)
    else:
        return np.array([[1,0], [-1,-1], [-1,0], [1,1]]).astype(np.float64)


@jit(nopython=True)
def thresholds(suppressed, low, high):
    x, y = suppressed[:,:,2].shape
    ret = np.zeros((x,y), dtype=np.uint8)

    for i in range(x):
        for j in range(y):
            curr = suppressed[i,j,2]

            if curr >= high:
                ret[i,j] = 255
            elif curr >= low:
                ret[i,j] = 125

    return ret

@jit(nopython=True)
def canny_hysteresis(edges):
    x, y = edges.shape
    final_edges = np.zeros((x,y), dtype=np.uint8)

    for i in range(x):
        for j in range(y):
            # If its a weak edge
            if edges[i,j] == 1:
                l = min(0,i-1)
                r = max(x, i+2)
                t = min(0,j-1)
                b = max(y,j+2)
                for p in range(-1,2):
                    for q in range(-1,2):
                        if edges[i+p,j+q] >= 255:
                            final_edges[i,j] = 255
            if edges[i,j] == 255:
                final_edges[i,j] = 255
    return final_edges