import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import colorsys
import random
from PIL import Image

##########################################################################
# Pre processing tools
##########################################################################


# TODO: consummate the annotation of this module.


def raw_read(file_path):
    """
    Read data form .raw file.
    :param file_path: file path of the raw data.
    :return: [height, width] ndarray.
    """
    #raw_image = np.fromfile(file_path, dtype=np.uint16)
    #return np.reshape(raw_image, [2292, 2804])  
    tf1 = cv2.imread(file_path)
    tf1 = cv2.cvtColor(tf1, cv2.COLOR_BGR2GRAY)
    #tf1 = np.fromfile(file_path, dtype=np.uint8)
    #return np.reshape(tf1, [288, 384])
    return np.reshape(tf1, [966, 1225])
    


def pre_processing(img, gamma):
    """
    Pre process the image data.
    :param img: origin image.
    :param gamma: gamma correction factor.
    :return: pre-processed image.
    """
    img = set_min_max_window_with_gamma_correction(img, gamma)
    img = deal_with_out_boundary(img)
    return complementary(img)


def set_min_max_window_with_gamma_correction(img, gamma):
    """
    Set min-max-window and perform gamma correction on image data.
    :param img: input image data.
    :param gamma: gamma correction factor.
    :return: output image.
    """
    min_value = np.min(img)
    max_value = np.max(img)
    return 255. * ((img / (max_value - min_value)) ** (1.0 / gamma))


def deal_with_out_boundary(img):
    """
    Deal with the outlier.
    :param img: input image.
    :return: image without outlier.
    """
    img[img > 255.] = 255.
    img[img < 0.] = 0.
    return img


def complementary(img):
    """
    get complementary image.
    :param img: input image.
    :return: complementary image.
    """
    return 255. - img


def find_roi(img):
    """
    Find the region of interest of origin image.
    :param img: input image.
    :return: [x1, x2, y1, y2] are the top-left and bottom-right coordinates respectively.
    """
    img_ = copy.deepcopy(img / 65535. * 255.)
    img_ = img_.astype(np.uint8)
    blur = cv2.GaussianBlur(img_, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    horizontal_indicies = np.where(np.any(th, axis=0))[0]
    vertical_indicies = np.where(np.any(th, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    return img[y1: y2 + 1, x1: x2 + 1]


def prime_factor_decomposition(num):
    """
    Prime factor decomposition of a number.
    :param num: input number.
    :return: all prime factors of input number.
    """
    primes = [2]
    for i in range(3, num):
        if i * i > num:
            break
        flag = True
        for j in primes:
            if i % j == 0:
                flag = False
                break
            if j * j > i:
                break
        if flag:
            primes.append(i)
    factor = []
    for i in primes:
        if i * i > num:
            break
        while num % i == 0:
            factor.append(i)
            num = num // i
    if num != 1:
        factor.append(num)
    return factor

##########################################################################
# Visualize tools
##########################################################################


def density_along_center_line(image):
    """
    Density along center line.
    :param image: input image.
    :return:
    """
    height, width = image.shape
    center_line = image[height // 2, ...]
    x = list(range(width))
    return x, center_line


def visualize_center_line_density(images, titles=None):
    """
    Visualize center line density.
    :param images: images to be shown.
    :param titles: titles of input images.
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)
    colors = random_colors(num_images)

    if num_images % 3 != 0:
        num_rows = num_images // 3 + 1
    else:
        num_rows = num_images // 3

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 3

    fig = plt.figure()

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows + 1, num_cols, i + 1)
        ax.imshow(image, interpolation='nearest', cmap='gray')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)

        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])

        x, center_line = density_along_center_line(image)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(x, center_line, color=colors[i], label=title)
        ax.legend(loc='upper left')
        ax.set_label('density')


def visualize_images(images, titles=None):
    """
    Show all images with respective titles.
    :param images:
    :param titles:
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)

    if num_images % 3 != 0:
        num_rows = num_images // 3 + 1
    else:
        num_rows = num_images // 3

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 3

    fig = plt.figure()

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(image, 'gray')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


def show_histogram(images, titles=None, bins=255):
    """
    Show the histogram of images
    :param images: input images
    :param titles:
    :param bins:
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)

    if num_images % 2 != 0:
        num_rows = num_images // 2 + 1
    else:
        num_rows = num_images // 2

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 2

    fig = plt.figure()

    ns = []
    binses = []

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        n, bins, _ = ax.hist(np.reshape(image, [-1]), bins=bins)
        ax.plot(bins[:-1], n, color='r')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)
        ns.append(n)
        binses.append(bins)
    return ns, binses


def deal_with_inputs(ipt1, ipt2):
    if isinstance(ipt1, tuple):
        ipt1 = list(ipt1)

    if not isinstance(ipt1, list):
        ipt1 = [ipt1]

    num_ipt1 = len(ipt1)

    if isinstance(ipt2, tuple):
        ipt2 = list(ipt2)

    if not isinstance(ipt2, list):
        ipt2 = [ipt2]

    if not ipt2:
        ipt2 = [None] * num_ipt1

    num_ipt2 = len(ipt2)

    if num_ipt2 > num_ipt1:
        ipt2 = ipt2[:num_ipt1]
    elif num_ipt2 < num_ipt1:
        ipt2 += (num_ipt1 - num_ipt2) * [None]

    return ipt1, ipt2, num_ipt1, num_ipt2


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def main():
    origin_image = raw_read('./demo_img/4.raw')
    x1, x2, y1, y2 = find_roi(origin_image)
    origin_image = origin_image[y1: y2+1, x1: x2+1]
    plt.imshow(origin_image, 'gray')
    plt.show()


if __name__ == '__main__':
    main()
