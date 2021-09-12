import numpy as np
from os import path
from math import sqrt
import cv2

class Aindane(object):
    """Implementation of AINDANE
    Adaptive and integrated neighborhood-dependent approach for nonlinear enhancement of color images
    Attributes:
        img_bgr: The image to be processed, read by cv2.imread
        img_gray: img_bgr converted to gray following NTSC
    """

    _EPS = 1e-6  # eliminate divide by zero error in I_conv/I

    def __init__(self, img):
        """
        :param path_to_img : full path to the image file
        """
        self.img_bgr = img
        if self.img_bgr is None:
            raise Exception("cv2.imread failed! Please check if the path is valid")

        self.img_gray = img

        # to collect parameters
        self.z = None
        self.c = None
        self.p = None

    def _ale(self):
        """ale algorithm in SubSection 3.1 of the paper.
        Basically just the implementation of the following formula:
            In_prime = f(In, z)
        Calculates In and z, then return In_prime
        :return In_prime:
        """

        # Calculate In
        In = self.img_gray / 255.0  # 2d array, equation 2

        # Calculate z   searchsorted（a，v）函数是判断v在a中哪两个a[n-1],a[n]之间，并返回n-1
        cdf = cv2.calcHist([self.img_gray], [0], None, [256], [0, 256]).cumsum()
        L = np.searchsorted(cdf, 0.1 * self.img_gray.shape[0] * self.img_gray.shape[1], side='right')
        L_as_array = np.array([L])  # L as array, for np.piecewise
        z_as_array = np.piecewise(L_as_array,
                         [L_as_array <= 50,
                          50 < L_as_array <= 150,
                          L_as_array > 150
                          ],
                         [0, (L-50) / 100.0, 1]
                         )
        z = z_as_array[0]  # take the value out of array

        self.z = z

        # Result In_prime = f(In, z)
        In_prime = 0.5 * (In**(0.75*z+0.25) + (1-In)*0.4*(1-z) + In**(2-z))
        return In_prime

    def _ace(self, In_prime, c=5):
        """ace algorithm in SubSection 3.2 of the paper
        Implementation of:
            S = f(In_prime, E(P()))
        :param In_prime:
        :param c:
        :return S:
        """

        # image freq shift
        img_freq = np.fft.fft2(self.img_gray)
        img_freq_shift = np.fft.fftshift(img_freq)

        # gaussian freq shift
        sigma = sqrt(c**2 / 2)
        _gaussian_x = cv2.getGaussianKernel(
            int(round(sigma*3)),  # size of gaussian: 3*sigma(0.99...)
            int(round(sigma))  # cv2 require sigma to be int
        )
        gaussian = (_gaussian_x * _gaussian_x.T) / np.sum(_gaussian_x * _gaussian_x.T)  # normalize
        gaussian_freq_shift = np.fft.fftshift(
            np.fft.fft2(gaussian, self.img_gray.shape)  # gaussian kernel padded with 0, extend to image.shape
        )

        # "image freq shift" * "gaussian freq shift"
        image_fm = img_freq_shift * gaussian_freq_shift
        I_conv = np.real(np.fft.ifft2(np.fft.ifftshift(image_fm)))  # equation 6

        sigma_I = np.array([np.std(self.img_gray)])  # std of I,to an array, for np.piecewise
        P = np.piecewise(sigma_I,
                         [sigma_I <= 3,
                          3 < sigma_I < 10,
                          sigma_I >= 10
                          ],
                         [3, 1.0 * (27 - 2 * sigma_I) / 7, 1]
                         )[0]  # take the value out of array

        self.c = c
        self.p = P

        E = ((I_conv + self._EPS) / (self.img_gray + self._EPS)) ** P
        S = 255 * np.power(In_prime, E)
        return S

    def aindane(self):
        """The algorithm put in a whole
        """

        In_prime = self._ale()
        S = self._ace(In_prime, c=240)
        #return self._color_restoration(S, lambdaa=[1, 1, 1])
        return S


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
    winSize = (5,5)       #类似卷积核（数字越大，磨皮效果越好）
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
    detailimage = cv2.subtract(grayimage, baseimage)

 
    #细节层归一化后加强
    #D = detailimage/255.0      
    #enhanced_detailimage = cv2.multiply(D, D)
    #enhanced_detailimage = cv2.multiply(detailimage, detailimage)
    enhanced_detailimage = detailimage  * 1.5
    #enhanced_detailimage = enhanced_detailimage  * 255         #(0,1)->(0,255)
    enhanced_detailimage[enhanced_detailimage > 255] = 255    #防止像素溢出 
    enhanced_detailimage = np.round(enhanced_detailimage)
    enhanced_detailimage = enhanced_detailimage.astype(np.uint8)     


    #亮度层加强
    
    ain =  Aindane(baseimage)   #
    enhanced_baseimage = np.clip(ain.aindane(), 0, 255)
    enhanced_baseimage = np.round(enhanced_baseimage)
    enhanced_baseimage = enhanced_baseimage.astype(np.uint8) 


    #temp = cv2.resize(enhanced_baseimage,None,fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC) 

    #亮度层和细节层混合
    #enhanced_grayimage = cv2.add(enhanced_detailimage, grayimage)
    enhanced_grayimage = cv2.add(enhanced_detailimage, enhanced_baseimage)
    #enhanced_grayimage = enhanced_detailimage + baseimage


    return enhanced_grayimage