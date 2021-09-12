import tifffile as tif
import cv2 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 灰度转化为RGB
def gray2rgb(src,src_gray):
    B = src[:,:,0]
    G = src[:,:,1]
    R = src[:,:,2]
    # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
    g = src_gray[:]
    p = 0.2989; q = 0.5870; t = 0.1140
    B_new = (g-p*R-q*G)/t
    B_new = np.uint8(B_new)
    src_new = np.zeros((src.shape)).astype("uint8")
    src_new[:,:,0] = B_new
    src_new[:,:,1] = G
    src_new[:,:,2] = R

    return src_new

######调用函数

def color(file_path1, file_path2):
    src= cv2.imread(file_path1)
    src_gray0 = cv2.imread(file_path2)      
    src_gray = cv2.cvtColor(src_gray0, cv2.COLOR_BGR2GRAY)  
    print(src.shape)
    print(src_gray.shape)

    src_new = gray2rgb(src,src_gray)

    cv2.imshow("input", src)
    cv2.imshow("output", src_gray)
    cv2.imshow("result", src_new)
    cv2.imwrite("./demo_img/6_color.tif", src_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
