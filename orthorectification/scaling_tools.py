import cv2
import numpy as np


def resize(img: np.array, factor: int) -> np.array:
    img_resized = cv2.resize(
        img, (int(img.shape[0] / factor), int(img.shape[1] / factor))
    )
    return img_resized


def convert_to_8bit(img: np.array) -> np.array:
    img_rescaled = img / np.amax(img) * 255.0
    img_rescaled = img_rescaled.astype(np.ubyte)
    return img_rescaled


def gaussian_rescale(img: np.array, bitness=11, stdev_bound=3) -> np.array:
    img_float = img.astype(np.float32)
    img_0_1 = img_float / ((2 ** bitness) - 1)
    mean = img_0_1.mean()
    stdev = img_0_1.std()
    img_8_bit = img_0_1 + (0.5 - mean)
    new_min, new_max = 0.5 - stdev_bound * stdev, 0.5 + stdev_bound * stdev
    img_8_bit = img_8_bit - new_min
    img_8_bit = img_8_bit / new_max
    img_8_bit = img_8_bit * 255
    img_8_bit[np.where(img_8_bit > 255)] = 255
    img_8_bit[np.where(img_8_bit < 0)] = 0
    return img_8_bit
