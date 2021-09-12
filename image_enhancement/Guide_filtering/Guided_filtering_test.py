from skimage import data,filters,img_as_ubyte
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from Guided_filtering import *
import cv2

img_path = "F:/JZYY/code/ImageEnhancement/2018Endoscope_image_enhancement/result/myresult/"
img_name = "2.png"

img = cv2.imread(img_path + img_name) #BGR, HWC
# img = img.astype(np.float)

ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y = ycrcb_image[:,:, 0]
cr = ycrcb_image[:,:, 1]
cb = ycrcb_image[:,:, 2]

#导向滤波
eet = process(ycrcb_image[:,:,0], 50, 0.5)
new_y = eet
new_cr = cr
new_cb = cb      

ycrcb_image[:, :, 0] = new_y
ycrcb_image[:, :, 1] = new_cr
ycrcb_image[:, :, 2] = new_cb

ycrcb_image = np.uint8(np.clip(ycrcb_image, 0, 255))
img_enhance = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCR_CB2BGR)

cv2.imwrite("./result/2g_550.png", img_enhance)
cv2.imshow("image1", img_enhance)
cv2.waitKey(0)
cv2.destroyAllWindows()