import cv2
import numpy as np
from numba import njit, prange
import math

@njit(parallel=True)
def NNupscaling(src, scale=4):
    height, width, channels = src.shape
    new_height, new_width = int(height * scale), int(width * scale)
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in prange(new_height):
        for j in range(new_width):
            h = min(int(round(i / scale)), height - 1)
            w = min(int(round(j / scale)), width - 1)
            new_img[i, j] = src[h, w]

    return new_img

@njit(parallel=True)
def bilinear_scaling(src, scale=4):
    height, width, channels = src.shape
    new_height = int(height * scale)
    new_width = int(width * scale)

    dst = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in prange(new_height):
        for j in range(new_width):
            x = i / scale
            y = j / scale
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)

            dx, dy = x - x1, y - y1

            for c in range(channels):
                top = (1 - dx) * src[x1, y1, c] + dx * src[x2, y1, c]
                bot = (1 - dx) * src[x1, y2, c] + dx * src[x2, y2, c]
                dst[i, j, c] = np.uint8((1 - dy) * top + dy * bot)

    return dst

@njit
def cubic_kernel(x):
    a = -0.5
    abs_x = abs(x)

    if abs_x <= 1:
        return (a + 2) * abs_x**3 - (a + 3) * abs_x**2 + 1
    elif 1 < abs_x < 2:
        return a * abs_x**3 - 5 * a * abs_x**2 + 8 * a * abs_x - 4 * a
    return 0

@njit(parallel=True)
def bicubic_interpolation(image, scale):
    height, width, channels = image.shape
    new_height, new_width = int(height * scale), int(width * scale)
    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in prange(new_height):
        for j in range(new_width):
            x = i / scale
            y = j / scale
            x1, y1 = math.floor(x), math.floor(y)

            for c in range(channels):
                value = 0.0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        xm = max(0, min(x1 + m, height - 1))
                        yn = max(0, min(y1 + n, width - 1))
                        
                        weight_x = cubic_kernel(x - (x1 + m))
                        weight_y = cubic_kernel(y - (y1 + n))

                        value += image[xm, yn, c] * weight_x * weight_y

                value = 0 if value < 0 else (255 if value > 255 else value)
                new_img[i, j, c] = np.uint8(value)


    return new_img