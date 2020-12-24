import filters
import numpy as np
from PIL import Image, ImageDraw
from numba import jit


# Converts an image to intensity, applies filter, and returns resulting image
def apply_filter_intensity(image, filter):
    im = np.asarray(image)

    result = convolve(im, filter)
    return result.astype('uint8')


def apply_filter(image, filter):
    im = np.asarray(image)
    
    if im.ndim == 2:
        result = convolve(im, filter)
    else:
        w,h,d = im.shape

        result = np.zeros_like(im)
        for i in range(d):
            result[:,:,i] = convolve(im[:,:,i], filter)

    return Image.fromarray(result.astype('uint8'))


# Convolves an image with the filter, making adjustments if the filter has even-parity dimensions
# Accelerate with jit
@jit(nopython=True)
def convolve(image, filter):
    # Ensure both are 2D arrays
    assert image.ndim == 2
    assert filter.ndim == 2

    # Return array, image dimensions, filter dimensions
    ret = np.zeros_like(image)
    i_w, i_h = image.shape
    f_w, f_h = filter.shape

    # Parity is 1 if its even, as we will use this value to adjust the centre/indices
    x_parity = 1 if f_w % 2 == 0 else 0
    y_parity = 1 if f_h % 2 == 0 else 0

    # Offset: Distance from the centre of the filter to the edge
    # Centre: Index of the filter's centre adjusted for even parity
    offset_x, offset_y = f_w // 2, f_h // 2
    centre_x, centre_y = f_w // 2 - x_parity, f_h // 2 - y_parity
    
    # Loop through original image's indices, adjusted for NO PADDING
    for x in range(offset_x, i_w - offset_x):
        for y in range(offset_y, i_h - offset_y):
            val = 0

            # Apply filter to the pixels around the image, adjusting the indices if the dimension is odd
            for i in range(-offset_x + x_parity, offset_x + 1):
                for j in range(-offset_y + y_parity, offset_y + 1):
                    val = val + filter[centre_x + i, centre_y + j] * image[x - i, y - j]

            # Clamp the result into the range of 0-255
            ret[x,y] = max(0, min(val, 255))
    
    return ret

