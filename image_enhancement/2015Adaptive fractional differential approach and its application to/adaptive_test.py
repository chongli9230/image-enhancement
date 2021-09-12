from scipy import optimize
from sympy import *
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
from adaptive import *
import cv2


#file_name = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/55.tif"
file_name = "F:/JZYY/pic/pic/2.png"
img = Image.open(file_name)
img_arr = np.array(img)

#0.5, 0.8, 0.5* V[i,j]
v = 1
if len(img_arr.shape) == 2:
    channel_num = 1
elif len(img_arr.shape) == 3:
    channel_num = img_arr.shape[2]

if channel_num == 1:
    # gray image
    arr = calculate(img_arr, v)
    img_res = Image.fromarray(arr)
elif channel_num == 3 or channel_num == 4:
    # RGB image or RGBA image(such as png)
    rgb_arr = [None] * 3
    rgb_img = [None] * 3
    # process dividely
    for k in range(3):
        rgb_arr[k] = calculate(img_arr[:,:,k], v)
        rgb_img[k] = Image.fromarray(rgb_arr[k]).convert('L')
    
    img_res = Image.merge("RGB", tuple(rgb_img))

#plt.figure(1)
#plt.imshow(img)
#plt.figure(2)
plt.imshow(img_res)
plt.savefig("./result/2enhanced.png")
plt.show()




