from skimage import data,filters,img_as_ubyte
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import cv2


path1 = "./result/USM/192usm8c25.png"
path2 = "./result/USM/192usm8cpp8.png"

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
ycrcb_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
ycrcb_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
y1 = ycrcb_image1[:,:, 0]*1.0
y2 = ycrcb_image2[:,:, 0]*1.0
#导向滤波
y=abs(y1-y2)
m = y.max()
#img = img1 - img2
print(y)
print(m)


