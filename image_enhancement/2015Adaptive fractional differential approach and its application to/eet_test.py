import sys
import os
sys.path.append("../")
from eet import *
import time

if __name__ == "__main__":
    #path = "./test_data/eet"
    #path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
    path = "F:/JZYY/code/2018Endoscope_image_enhancement/result/myresult/"
    img = cv2.imread(os.path.join(path, "2.png")) #bgr
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    start_Real = time.time()    

    sharpness = 8
    eet = Sharpening(yuv[:,:,0])
    yuvimg_eet = eet.usm(sharp=sharpness)
    yuv[:,:,0] = yuvimg_eet
    bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.uint8)

    end_End = time.time()
    print((end_End - start_Real))

    cv2.imwrite("F:/JZYY/result/2usm.png", bgr_img)
    #cv2.imwrite(os.path.join(path, 'result.jpg'), bgr_img)