import numpy as np
from os import path
from math import sqrt
import time
import cv2

def guideFilter2(I, p, winSize, eps, s):
    
    #输入图像的高、宽
    h, w = I.shape[:2]
    
    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))
    
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(p, size, interpolation=cv2.INTER_CUBIC)
    
    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))
    
    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    
    #p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)
    
    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)
    
    mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)
    
    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式
    
    #协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
   
    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I
    
    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)
    
    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)
    
    q = mean_a*I + mean_b
    
    return q
#s采样比例(4),eps 是调整图的模糊程度与边缘检测精度的参数
#s=4, eps=0.12, winSize=13

def process(grayimage):
    ########## Guided filtering
    eps = 0.12
    winSize = (30,30)       #类似卷积核（数字越大，磨皮效果越好）
    I = grayimage/255.0       #将图像归一化
    p =I
    s = 1 #步长
    guideFilter_img = guideFilter2(I, p, winSize, eps,s)

    # 保存导向滤波结果
    guideFilter_img = guideFilter_img  * 255         #(0,1)->(0,255)
    
    guideFilter_img[guideFilter_img  > 255] = 255    #防止像素溢出
    guideFilter_img = np.round(guideFilter_img )
    baseimage = guideFilter_img.astype(np.uint8)       #亮度层

    #detailimage = grayimage - baseimage         #细节层
    #detailimage = cv2.subtract(grayimage, baseimage)
    
    w = 1
    result_d = cv2.addWeighted(grayimage, 1+w, baseimage, -w,0)  
    return result_d  


if __name__ == "__main__":
    #path = "./test_data/eet"
    #path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
    path = "F:/JZYY/code/2018Endoscope_image_enhancement/result/myresult/"
    img = cv2.imread(path + "2.png") #bgr
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    start_Real = time.time()    

    eet = process(yuv[:,:,0])
    yuv[:,:,0] = eet
    bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.uint8)

    end_End = time.time()
    print((end_End - start_Real))

    cv2.imshow("image", bgr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./result/myresult/2myd30.png', bgr_img)    