import cv2
import numpy as np
# =============================================================
# class: sharpening
#   sharpens the image
# =============================================================
class Sharpening:
    def __init__(self, Y, clip=255,type="uint8"):
        self.Y = Y
        self.clip = clip
        self.type = type
    def usm_no_black_edge(self,sharp, maxgradient=100):#采用的代码部分
#         w = sharp/2
#         blurImg = cv2.GaussianBlur(self.Y, (7,7),5)
#         self.Y = cv2.addWeighted(self.Y, 1 + w, blurImg, -w, 0)
#         sigma = sharp * 0.05
#         if sharp > 0 and sharp <=5:
#             self.Y = cv2.GaussianBlur(self.Y, (3, 3), 0.3)
#         else:
#             self.Y = cv2.GaussianBlur(self.Y, (3, 3), sigma)
#         return self.Y
        # w = 5
        temp = self.Y
        w = sharp/2.
        sigma = sharp * 0.05
        
        kernel_size = 5
        # Generate 2-D gaussian kernel.
        kernel_G = cv2.getGaussianKernel(kernel_size,kernel_size)
        kernel_G = kernel_G * np.transpose(kernel_G)
        print(kernel_G )
        # Merge addweighted into kernel
        original = np.zeros((kernel_size,kernel_size))
        print(original.shape) 
        original[kernel_size//2][kernel_size//2] = 1+w
        kernel_G = original - kernel_G * w
        print(original)
        print(kernel_G )
        self.Y = cv2.filter2D(self.Y, -1, kernel=kernel_G)
        
        #blurImg = cv2.GaussianBlur(temp, (5,5), 10)  
        #self.Y = cv2.addWeighted(temp, 1+w, blurImg, -w,0)
        
        if sharp > 0 and sharp <=5:
            self.Y = cv2.GaussianBlur(self.Y, (3, 3), 0.3)
        else:
            self.Y = cv2.GaussianBlur(self.Y, (3, 3), sigma)
        
        #return self.Y
        #检验梯度变化
        x = cv2.Sobel(self.Y, cv2.CV_16S, 1, 0, ksize=1)   #原图self.Y
        y = cv2.Sobel(self.Y, cv2.CV_16S, 0, 1, ksize=1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        gra = cv2.addWeighted(absX, 1, absY, 1, 0)
        
        gra_weight = (gra - maxgradient )/(255.0 - maxgradient)
        mask = (gra > maxgradient).astype(np.float32)
        #self.Y = self.Y* (1-mask) + temp* mask
        self.Y = self.Y* (1-mask) + temp* mask* gra_weight + self.Y* mask* (1 - gra_weight) 
        
        return self.Y