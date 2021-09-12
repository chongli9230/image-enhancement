#-*- coding : utf-8 -*-

from PIL import Image, ImageOps
import numpy as np
import cv2

from contrast import ImageContraster


img = cv2.imread("F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/7.tif")
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#cv2.imshow('50',img)
# equalize the histogram of the Y channel
icter = ImageContraster()
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "HE")
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "AHE", window_size = 32, affect_size = 16)
he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "CLAHE", blocks = 8, threshold = 10.0)
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "Bright")

#icter.plot_images(img, he_eq_img)
#img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_yuv[:,:,0] = he_eq_img
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
 
cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)
cv2.imwrite("./2b_enhanced7.tif", img_output)
cv2.waitKey(0)


"""
# read image
#img = Image.open("car.jpg")
img = Image.open("2.tif")
print(np.array(img).shape)


# contraster
icter = ImageContraster()


# HE  2.自己实现的直方图均衡化HE
he_eq_img = icter.enhance_contrast(img, method = "HE")
icter.plot_images(img, he_eq_img)


# AHE   3.自适应直方图均衡化AHE
ahe_eq_img = icter.enhance_contrast(img, method = "AHE", window_size = 32, affect_size = 16)
icter.plot_images(img, ahe_eq_img)

# CLAHE   4.限制对比度自适应直方图均衡化CLAHE
clahe_eq_img = icter.enhance_contrast(img, method = "CLAHE", blocks = 8, threshold = 10.0)
icter.plot_images(img, clahe_eq_img)

# Local Region Stretch   5.自适应局部区域伸展直方图均衡化Local Region Stretch HE
lrs_eq_img = icter.enhance_contrast(img, method = "Bright")
icter.plot_images(img, lrs_eq_img)
"""

