import cv2
import numpy as np
from numba import jit


def resize(img: np.array, factor: int) -> np.array:
    img_resized = cv2.resize(
        img, (int(img.shape[0] / factor), int(img.shape[1] / factor))
    )
    return img_resized


def convert_to_8bit(img: np.array) -> np.array:
    img_rescaled = img / np.amax(img) * 255.0
    img_rescaled = img_rescaled.astype(np.ubyte)
    return img_rescaled


@jit(nopython=True, parallel=True)
def gaussian_rescale(img: np.array, bitness=11, stdev_bound=2) -> np.array:
    img_0_1 = img / ((2 ** bitness) - 1)
    mean = img_0_1.mean()
    stdev = img_0_1.std()
    img_8_bit = img_0_1 + (0.5 - mean)
    new_min, new_max = 0.5 - stdev_bound * stdev, 0.5 + stdev_bound * stdev
    img_8_bit = img_8_bit - new_min
    img_8_bit = img_8_bit / new_max
    img_8_bit = img_8_bit * 255
    return img_8_bit
