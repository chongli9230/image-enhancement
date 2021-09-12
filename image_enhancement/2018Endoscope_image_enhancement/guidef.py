from skimage import data,filters,img_as_ubyte
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from guidef import *
import cv2
import math

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

def get_sobeledge(img):
    """
    x = cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=1)
    y = cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return absX, absY
    
    """
    """
    con = 1 / 24
    kernel = np.array([
        [con, con, con, con, con ],
        [con, con, con, con, con ],
        [con, con, -1, con, con ],
        [con, con, con, con, con ],
        [con, con, con, con, con ],
    ])
    """
    """
    con = 1 / 48
    kernel = np.array([
        [con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con ],
        [con, con, con, -1, con, con, con ],
        [con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con ],
    ])
    
    con = 1 / 80
    kernel = np.array([
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, -1, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
        [con, con, con, con, con, con, con, con, con ],
    ])  
    """
    """
    #9*9
    #1, 1/2, 1/14    20oooooo   2
    #1, 1/2, 1/28    18ooooo  1
    #1, 1/2, 1/56    17oo    3
    #
    #1/56, 1/2, 1    60o

    #1, 1/2, 1/6, 1/14   24p
    #1, 1/2, 1/28, 1/28   20pp
    con1 = 1     #8
    con2 =  1 / 2    #16
    con3 = 1 / 28      #24
    con =  1 /28     #32
    kernel = np.array([
        [con, con, con, con, con, con, con, con, con ],
        [con, con3, con3, con3, con3, con3, con3, con3, con ],
        [con, con3, con2, con2, con2, con2, con2, con3, con ],
        [con, con3, con2, con1, con1, con1, con2, con3, con ],
        [con, con3, con2, con1, -18, con1, con2, con3, con ],
        [con, con3, con2, con1, con1, con1, con2, con3, con ],
        [con, con3, con2, con2, con2, con2, con2, con3, con ],
        [con, con3, con3, con3, con3, con3, con3, con3, con ],
        [con, con, con, con, con, con, con, con, con ],
    ])  
    """
    
    """
    #17 * 17
    con = 1 / 288    
    kernel = np.array([
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, -1, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
        [con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con, con],
    ])       
    """

     
    """
    x = cv2.filter2D(img, cv2.CV_16S, kernel)
    print(x)
    x = x  * 255         #(0,1)->(0,255)
    #双阈值图像


    """
    
    # Laplacian of Gaussian
    kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16,-2, -1,],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0],
    ])  
    
    """
    kernel = np.array([
        [-2, -4, -4, -4, -2],
        [-4, 0, -8, 0, -4],
        [-4, 8, 24, 8, -4],
        [-4, 0, -8, 0, -4],
        [-2, -4, -4, -4, -2],
    ])
    """
    """
    #Laplace
    kernel = np.array([
        [0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]
        ])
    """
    
    kernel = np.array([
        [-1, 0, -1], 
        [0, 4, 0], 
        [-1, 0, -1]
        ])
      
    
    cv2.imshow("image4", img)
    x = cv2.filter2D(img, cv2.CV_16S, kernel)
    
    TL = 0
    TH = 255
    #关键在这两个阈值的选择
    # TL = 0.4*np.max(img2)
    # TH = 0.5*np.max(img2)
    x = x + 30
    x[x  > TH] = 255
    x[x  < TL] = 0
    
    return x


def new(img):
    k1 = np.array([
        [-1, -2, -1], 
        [0, 0, 0], 
        [1, 2, 1]
        ])

    k2 = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
        ])    

    k3 = np.array([
        [-2, -1, 0], 
        [-1, 0, 1], 
        [0, 1, 2]
        ])

    k4 = np.array([
        [0, 1, 2], 
        [-1, 0, 1], 
        [-2, -1, 0]
        ])

    k5 = np.array([
        [-1, -1, -1], 
        [-1, 8, -1], 
        [-1, -1, -1]
        ])

    x1 = cv2.filter2D(img, cv2.CV_16S, k1)
    x2 = cv2.filter2D(img, cv2.CV_16S, k2)
    x3 = cv2.filter2D(img, cv2.CV_16S, k3)
    x4 = cv2.filter2D(img, cv2.CV_16S, k4)
    x5 = cv2.filter2D(img, cv2.CV_16S, k5)
    g = abs(x1)+abs(x2)+abs(x3)+abs(x4)
    Re = g/4 *x5
    return Re

def process2(img):
    #I = img/255.0 
    Re = new(img)
    #Filter_img = Re  * 255 
    Re = np.uint8(np.clip(Re , 0, 255))
    cv2.imshow("i1",Re)
    result = cv2.addWeighted(img, 1, Re, 0.08, 0)
    return result

def process(grayimage, grayimage2, r, w):
    ########## Guided filtering
    
    #0.12 
    eps = 0.12
    winSize = (r, r)       #类似卷积核（数字越大，磨皮效果越好）
    I = grayimage/255.0       #将图像归一化
    p = grayimage2/255.0
    s = 1 #步长
    Filter_img = guideFilter2(I, p, winSize, eps,s)
    # 保存导向滤波结果
    Filter_img = Filter_img  * 255         #(0,1)->(0,255)
    #cv2.imwrite('./result/guidef/test/6in.png', grayimage)
    #cv2.imwrite('./result/guidef/test/6yout.png', Filter_img)

    """
    #sobel 边缘检测
    x_edge, y_edge = get_sobeledge(grayimage)
    Filter_img = cv2.addWeighted(x_edge,w,y_edge,w,0)
    cv2.imshow("image3", Filter_img)
    """
    
    #Filter_img = get_sobeledge(grayimage)
    
    Filter_img[Filter_img  > 255] = 255    #防止像素溢出
    Filter_img = np.round(Filter_img )
    baseimage = Filter_img.astype(np.uint8)       #亮度层

      #show
    #cv2.imshow("image3", baseimage)
    #cv2.imwrite('./result/myresult2/6lygray.png', baseimage)
    #detailimage = grayimage - baseimage         #细节层
    #detailimage = cv2.subtract(grayimage, baseimage)
    #w = 0.5, 0.8
    #result_d = cv2.addWeighted(grayimage, 1+w, baseimage, -w,0)  
    #result_d = cv2.addWeighted(grayimage, 1, baseimage, w,0)  
    #result_d = cv2.addWeighted(grayimage, -w, baseimage, w,0) 
    return baseimage 

    #return baseimage
        
        
def enhance(g):
    #g = g/255.
    ave = np.mean(g)
    print('ave', ave)
    d = (3*(ave**2) - 2*(ave**3)) / (2*(ave**3) - 3*(ave**2) + 1)
    print("d",d)
    h, w = g.shape[0], g.shape[1]
    new_g = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if g[i,j] == 0:
                continue
            new_g[i,j] = d*(( 1./g[i,j] - 1.)**2) + 1.
            new_g[i,j] = 1./new_g[i,j]
            # print("s:", g[i,j], "new:", new_g[i,j])
    return new_g

def enhance2(g):
    h, w = g.shape[0], g.shape[1]
    g = g/255.0
    new_g = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if g[i,j] == 0:
                continue
            new_g[i,j] = 1.245/(1.0+(0.545/g[i,j])**2.35) 
    return new_g*255.0
    
