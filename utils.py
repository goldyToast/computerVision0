import numpy as np
from PIL import Image, ImageDraw


# STARTING WITH IGNORING PADDING


# def init_convolution(image, filter):
#     assert filter.ndim == 2

#     c_image = np.zeros_like(image)

#     if (image.ndim == 1):
#         # Do normal convlution on just this dimension
#     else:
#         n,d,z = image.shape
#         for dim in range(z):
#             # c_image[:,:,dim] = convolve

                
#     return c_image

# Converts an image to intensity, convolves the filter over it, then returns it
def filter_image(image, filter):
    im = image.convert('L')
    im = np.asarray(im)

    result = convolve_odd(im, filter)
    return Image.fromarray(result.astype('uint8'))


def convolve_odd(image, filter):
    assert image.ndim == 2
    assert filter.ndim == 2

    ret = np.zeros_like(image)
    w, h = image.shape
    fx, fy = filter.shape
    
    for x in range(fx // 2, w - fx // 2):
        for y in range(fy // 2, h - fy // 2):
            val = 0

            # the range will have to be adjusted for differences between odd and even
            for i in range(0 - (fx // 2), (fx // 2) + 1):
                for j in range(0 - (fy // 2), (fy // 2) + 1):
                    val = val + filter[fx // 2 - i,fy // 2 - j] * image[x + i, y + j]
            ret[x,y] = max(0, min(val, 255))
    
    return ret

