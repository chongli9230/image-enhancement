from skimage import data,filters,img_as_ubyte
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import cv2

def guideFilter(I, p, winSize, eps, s):

    #输入图像的高、宽
    h, w = I.shape[:2]
    
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    
    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    
    small_a = var_small_I / (var_small_I + eps)
    small_b = mean_small_I - small_a*mean_small_I
    
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_CUBIC)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_CUBIC)
    
    q = mean_a*I + mean_b
    
    return q
#s采样比例(4),eps 是调整图的模糊程度与边缘检测精度的参数
#s=4, eps=0.12, winSize=13

def process(grayimage, r, w):
    ########## Guided filtering
    #0.12 
    eps = 0.12
    winSize = (r, r)       #类似卷积核（数字越大，磨皮效果越好）
    I = grayimage/255.0       #将图像归一化
    p =I
    s = 0.5 #步长
    Filter_img = guideFilter(I, p, winSize, eps,s)
    # 保存导向滤波结果
    Filter_img = Filter_img  * 255         #(0,1)->(0,255)

    baseimage= np.uint8(np.clip(Filter_img, 0, 255))

    #w = 0.5, 0.8
    result_d = cv2.addWeighted(grayimage, 1+w, baseimage, -w,0)  
    return result_d
        
 