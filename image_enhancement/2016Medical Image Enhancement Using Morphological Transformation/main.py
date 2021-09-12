import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import cv2
import time
import support
import utils
import math
import trans
import MT

#MT.process1('6.tif')


img = Image.open("./demo_img/6.tif")
print(np.array(img).shape)
img_arr = np.array(img)
rgb_arr = [None] * 3
rgb_img = [None] * 3
# process dividely
for k in range(3):
    rgb_arr[k] = MT.process2(img_arr[:,:,k],'6.tif')
    rgb_img[k] = Image.fromarray(np.uint8(rgb_arr[k]))

img_res = Image.merge("RGB", tuple(rgb_img))

img_res.save("./demo_img/6_out3.tif")

