from scipy import optimize
from sympy import *
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
from adaptive import *
import cv2
import time


#imgpath = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
imgpath = "F:/JZYY/pic/pic/"
imgname = "5.png"
img = cv2.imread(imgpath + imgname)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

start_Real = time.time()

#0.5, 0.6, 0.8, 0.5* V[i,j]
v = 0.6

arr = calculate(img_yuv[:,:,0], v)
img_yuv[:,:,0] = arr 
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

end_End = time.time()
print((end_End - start_Real))

cv2.imshow('Color input image', img)
cv2.imshow('result', img_output)
cv2.imwrite("F:/JZYY/result/resultcomp/5ada.png", img_output)
cv2.waitKey(0)



