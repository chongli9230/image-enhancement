from skimage import data,filters,img_as_ubyte
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from guidef import *
from canny import *
import cv2
import time
img_path = "F:/JZYY/code/ImageEnhancement/2018Endoscope_image_enhancement/result/myresult/"
img_name = "2.png"
#img_name2 = "video_134.jpg"
img = cv2.imread(img_path + img_name) #BGR, HWC
#img2 = cv2.imread(img_path + img_name2) #BGR, HWC
# img = img.astype(np.float)
"""
def min_box(image,kernel_size=15):
    min_image = sfr.minimum(image,disk(kernel_size))  #skimage.filters.rank.minimum()返回图像的局部最小值

    return min_image

def calculate_dark(image):
    if not isinstance(image,np.ndarray):
        raise ValueError("input image is not numpy type")  #手动抛出异常
    dark = np.minimum(image[:,:,0],image[:,:,1],image[:,:,2]).astype(np.float32) #取三个通道的最小值来获取暗通道
    dark = min_box(dark,kernel_size=15)
    return dark/255

dark_img = calculate_dark(img)
"""

#rgb通道
r = img[:,:, 0]
g = img[:,:, 1]
b = img[:,:, 2]

#r2 = img2[:,:, 0]
#g2 = img2[:,:, 1]
#b2 = img2[:,:, 2]

w = 0.4
winsize = 13
img[:,:, 0] = process2(r)
img[:,:, 1] = process2(g)
img[:,:, 2] = process2(b)

img = np.uint8(np.clip(img, 0, 255))
img_enhance = img


"""
start_Real = time.time() 
#ycrcb通道
ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#ycrcb_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)

y = ycrcb_image[:,:, 0]
cr = ycrcb_image[:,:, 1]
cb = ycrcb_image[:,:, 2]

y2 = ycrcb_image2[:,:, 0]
cr2 = ycrcb_image2[:,:, 1]
cb2 = ycrcb_image2[:,:, 2]

#导向滤波
w = 0.5
winsize = 13
eet = process(y, y2, winsize, w)
end_End = time.time()
print((end_End - start_Real))

new_y = process2(y)
new_cr = cr
new_cb = cb      
#new_cb = enhance2(cb)   cb通道增强

ycrcb_image[:, :, 0] = new_y
ycrcb_image[:, :, 1] = new_cr
ycrcb_image[:, :, 2] = new_cb

ycrcb_image = np.uint8(np.clip(ycrcb_image, 0, 255))
img_enhance = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCR_CB2BGR)
"""
"""
#hsv通道增强
img_hsv = cv2.cvtColor(img_enhance, cv2.COLOR_BGR2HSV)
s = img_hsv[:, :, 1]
#s = s/255.
new_s = enhance2(s)
img_hsv[:, :, 1] = new_s
img_enhance = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
"""
"""
#I-scan

ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y = ycrcb_image[:,:, 0]
w = 1
eet = process(ycrcb_image[:,:,0], 30, w)
mask = (eet < 30).astype(np.float32)
img[:,:,1] = img[:,:,1]* (1-mask)+ img[:,:,1]*mask*0.90
img[:,:,2] = img[:,:,2]* (1-mask)+ img[:,:,2]*mask*0.90
img = img.astype(np.uint8)

"""
#cv2.imwrite('./result/myresult4/6k3bright30m9.png', eet)
#cv2.imwrite('./result/guidef/test/6outrgb2.png', img)
#cv2.imshow("image1",y)
#cv2.imshow("image2",img_enhance)
cv2.imwrite('./result/myresult/2sl6.png', img_enhance)
cv2.waitKey(0)
cv2.destroyAllWindows()


