#-*- coding : utf-8 -*-

from PIL import Image, ImageOps
import numpy as np
import cv2
from contrast import ImageContraster


img = cv2.imread("F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/1.tif",1)   #BGR   
img_trans = img.copy()

#####直方图均衡化
icter = ImageContraster()
he_eq_img = icter.enhance_contrast(img_trans[:,:,1], method = "HE")
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "AHE", window_size = 32, affect_size = 16)
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "CLAHE", blocks = 8, threshold = 10.0)
#he_eq_img = icter.enhance_contrast(img_yuv[:,:,0], method = "Bright")
he_eq_img = np.float32(he_eq_img)
img_trans[:,:,1] = he_eq_img


cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_trans)
cv2.imwrite("./result/1Color input image.tif", img)
cv2.imwrite("./result/1Histogram equalized.tif", img_trans)
cv2.waitKey(0)




