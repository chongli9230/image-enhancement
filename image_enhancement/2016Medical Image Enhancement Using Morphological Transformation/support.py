from skimage import filters, io
import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt


def get_ROI(raw_image):
    """
    Get the ROI of the input raw image.
    :param raw_image: grayscale image.
    :return: grayscale image.
    """

    float_image = np.float32(raw_image)
    shape_height, shape_width = raw_image.shape
    #shape_height, shape_width, channels = raw_image.shape

    otsu_threshold = filters.threshold_otsu(float_image)
    otsu_mask = float_image < otsu_threshold
    int_mask = np.ones_like(float_image) * otsu_mask
    kernel = np.ones((5, 5), np.int64)
    gradient = cv2.morphologyEx(int_mask, cv2.MORPH_GRADIENT, kernel)
    gradient_mask = gradient > 0

    coordinate_w = np.array([[y for y in range(shape_width)] for _ in range(shape_height)], dtype=np.int64)
    coordinate_h = np.array([[x for _ in range(shape_width)] for x in range(shape_height)], dtype=np.int64)
    coordinate_w = coordinate_w[gradient_mask]
    coordinate_h = coordinate_h[gradient_mask]

    min_h, min_w = np.min(coordinate_h), np.min(coordinate_w)
    max_h, max_w = np.max(coordinate_h), np.max(coordinate_w)
    print(min_h, min_w, max_h, max_w )
    #result = raw_image[min_h:max_h, min_w:max_w]
    #result = raw_image[0:288, 0:384]
    result = raw_image[min_h:966, min_w:1225]
    return result


def post_propossing(image, gamma):
    float_image = (image/(np.max(image) - np.min(image))) ** (1.0 / gamma)
    int_image = np.int64(float_image * (np.max(image) - np.min(image)))
    int_image[int_image < 0] = 0
    int_image[int_image > 65535] = 65535
    return int_image
    #return 65535 - int_image



