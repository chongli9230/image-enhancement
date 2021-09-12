import sys
import os
sys.path.append("../")
from eet2 import *
import time

if __name__ == "__main__":
    #path = "F:/JZYY/pic/blood-vessel/"
    path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB"
    #path = "./result/USM"
    img = cv2.imread(os.path.join(path, "192.tif")) #bgr
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    start_Real = time.time()    

    sharpness = 8
    maxgra = 25
    eet = Sharpening(yuv[:,:,0])
    yuvimg_eet = eet.usm_no_black_edge(sharp=sharpness, maxgradient=maxgra)
    
    yuv[:,:,0] = np.uint8(np.clip(yuvimg_eet, 0, 255))
    bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.uint8)

    end_End = time.time()
    print((end_End - start_Real))

    cv2.imshow("image",bgr_img)
    #cv2.imshow("1",yuv[:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./result/USM/192usm8c25.png", bgr_img)
    #cv2.imwrite(os.path.join(path, 'result.jpg'), bgr_img)