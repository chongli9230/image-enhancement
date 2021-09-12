import time
import cv2
import numpy as np

def gammaTranform(image,gamma,c=1):
    h, w = image.shape[0],image.shape[1]
    new_img = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = c*pow(image[i, j], gamma)
    cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)
    return new_img

#img_path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
#img_path2 = "F:/JZYY/pic/CVC-ClinicDB/Original/"
img_path = "F:/JZYY/pic/pic/"
#img_path2 = "F:/JZYY/pic/CVC-ClinicDB/Original/"
img_name = "4.png"
img = cv2.imread(img_path + img_name)

start = time.time()
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
channelsHSV = cv2.split(imgHSV)
channelsHSV[2] = gammaTranform(channelsHSV[2],gamma=0.4) # 只在V通道，即灰度图上进行处理
channels = cv2.merge(channelsHSV)
result = cv2.cvtColor(channels, cv2.COLOR_HSV2BGR)
print(time.time()-start)
cv2.imwrite("./result/spies/"+img_name, result)
"""
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""