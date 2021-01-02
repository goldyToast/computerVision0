import filters
import numpy as np
from PIL import Image, ImageDraw
from numba import jit
import cv2 as cv


@jit(nopython=True)
def combine_arrays(image1, image2):
    # Ensure both are 2D arrays | Ensure both are the same shape
    assert image1.ndim == 2
    assert image2.ndim == 2
    assert image1.shape == image2.shape

    n, d = image1.shape
    ret = np.zeros_like(image1)

    for i in range(n):
        for j in range(d):
            ret[i,j] = (image1[i,j] + image2[i,j]) / 2

    return ret


#  Sets up convolution w/ prechecks and modifies arrays if need be
def convolve2d(image, filter, padding="none"):
    # Ensure both are 2D arrays
    assert image.ndim == 2
    assert filter.ndim == 2

    paddings_options = ["none", "zero"]
    
    if padding not in paddings_options:
        raise ValueError("Invalid Padding Type: Supported padding types are: %s" % paddings_options)

    if padding == "none":
        return convolve_helper(image, filter)
    elif padding == "zero":
        i_w, i_h = image.shape
        f_w, f_h = filter.shape

        # Used to index where the image gets placed | Also must add 2*(f_v) + 1 to either side of the v dimension for padding
        x, y = f_w - 1, f_h - 1

        # Create new array with extra zeros added as padding | Place the image into the new padded version
        im = np.zeros((2 * x + i_w, 2 * y + i_h))
        im[x : x + i_w, y : y + i_h] = image

        res = convolve_helper(im, filter)

        r_x = f_w // 2 - (f_w + 1) % 2
        r_y = f_h // 2 - (f_h + 1) % 2

        # Remove columns and rows guaranteed to be all 0s
        return res[r_x:r_x + i_w + x, r_y:r_y + i_h + y]


# Convolves an image with the filter, making adjustments if the filter has even-parity dimensions
# Accelerate with jit
@jit(nopython=True)
def convolve_helper(image, filter):
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
    
    # Loop through original image's indices
    for x in range(centre_x, i_w - offset_x):
        for y in range(centre_y, i_h - offset_y):
            val = 0

            # Apply filter to the pixels around the image, adjusting the indices if the dimension is odd
            for i in range(-offset_x + x_parity, offset_x + 1):
                for j in range(-offset_y + y_parity, offset_y + 1):
                    val = val + image[x + i, y + j] * filter[centre_x - i + x_parity, centre_y - j + y_parity]


            # Clamp the result into the range of 0-255
            ret[x,y] = max(0, min(val, 255))
    
    return ret



def make_gaussian(image, min_size=50, scale=0.75):
    assert image.ndim == 2 or image.ndim == 3

    # Get the images width/weight, taking into account whether its an Intensity image or not
    if image.ndim == 2:
        w,h = image.shape
    else:
        w,h,d = image.shape

    # Create return list
    ret = [image]

    # Get Sigma to calculate the Gaussian filter | Calculate Gaussian filter ahead of time to save computation
    sigma = 1 / (2 * scale)
    gauss = filters.gauss1D(sigma)
    gauss = gauss[np.newaxis]


    while (w >= min_size) and (h >= min_size):
        # Get the last image in the pyramid
        im = ret[-1]

        # Calculate the width/height of the next image in the pyramid
        w, h = int(w * scale) , int(h * scale)

        # Perform blur/subsample based on whether it has 2 or 3 dimensions
        if image.ndim == 2:
            # If there is only 2 dimension we can blur/subsample immediately
            result = blur_and_scale(im, w, h, gauss)
        else:
            # If there are 3 dimensions we have to blur/subsample per channel
            result = np.zeros((w, h, d))
            for i in range(d):
                temp = im[:,:,i]

                result[:,:,i] = blur_and_scale(temp, w, h, gauss)
                print(result.shape)

        ret.append(result)

    return ret


# Take's a numpy array (convolve should handle if its RGB/L/RGBA)
def blur_and_scale(image, new_width, new_height, gauss):
    assert image.ndim == 2

    # Blur (taking advantage of Separability)
    im = convolve2d(image, gauss)
    im = convolve2d(im, gauss.T)

    # Resize the array using Bicubic interpolation
    im2 = cv.resize(im, dsize=(new_width, new_height), interpolation=cv.INTER_CUBIC)

    return im2


def visualize_pyramid(pyramid, save_name):
    if pyramid[0].ndim != 2:
        d = pyramid[0].shape[2]

    fun = lambda x: x.shape[1]
    width = sum(list(map(fun,pyramid)))
    height = pyramid[0].shape[0]


    if pyramid[0].ndim == 2:
        ret = np.zeros((height,width))
    else:
        ret = np.zeros((height,width,d))

    print(ret.shape)
    curr_x = 0

    for im in pyramid:
        w, h = im.shape[1],im.shape[0]
        end = curr_x + w
        print("Shape: ", im.shape)
        print("Curr_x: ", curr_x)
        print("W: ", w)
        print("end: ", end)
        print("H: ", h)
        print(ret[0:h,curr_x:end].shape)
        print()


        ret[0:h,curr_x:end] = im
        curr_x += w


    im = Image.fromarray(ret.astype('uint8'))
    im.save(str(save_name) + '.png', 'png')















# POTENTIALLY DEPRECATED

# Converts an image to intensity, applies filter, and returns resulting image
def apply_filter_intensity(image, filter):
    #image = image.convert("L")
    im = np.asarray(image)

    result = convolve_helper(im, filter, padding="zero")
    return result.astype('uint8')


def apply_filter(image, filter):
    im = np.asarray(image)
    
    if im.ndim == 2:
        result = convolve2d(im, filter)
    else:
        w,h,d = im.shape

        result = np.zeros_like(im)
        for i in range(d):
            result[:,:,i] = convolve2d(im[:,:,i], filter)

    return Image.fromarray(result.astype('uint8'))


# Resizing function specifically made for gaussian pyramids -> will still use OpenCV
def resize(array, ratio):
    w,h = array.shape

    w0,h0 = int(np.floor(w * ratio)), int(np.floor(h * ratio))
    ret = np.zeros((w0,h0))
    print(ret.shape)

    for i in range(w0):
        for j in range(h0):
            ret[i,j] = array[int(i // ratio), int(j // ratio)]
    
    return ret