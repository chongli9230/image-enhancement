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
    def usm(self,sharp):#采用的代码部分
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
        w = sharp/2.
        sigma = sharp * 0.05

        kernel_size = 5
        # Generate 2-D gaussian kernel.
        kernel_G = cv2.getGaussianKernel(kernel_size,5)
        
        kernel_G = kernel_G * np.transpose(kernel_G)
        
        # Merge addweighted into kernel
        original = np.zeros((kernel_size,kernel_size)) 
        original[kernel_size//2][kernel_size//2] = 1+w
        kernel_G = original - kernel_G * w

        self.Y = cv2.filter2D(self.Y, -1, kernel=kernel_G)
        if sharp > 0 and sharp <=5:
            self.Y = cv2.GaussianBlur(self.Y, (3, 3), 0.3)
        else:
            self.Y = cv2.GaussianBlur(self.Y, (3, 3), sigma)
        return self.Y
        