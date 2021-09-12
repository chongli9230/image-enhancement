import sys
import os
sys.path.append("../")
from eet import *

if __name__ == "__main__":
    path = "./test_data/eet"
    img = cv2.imread(os.path.join(path, "ISO12233_gt.jpg")) #bgr
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    sharpness = 5
    eet = Sharpening(yuv[:,:,0])
    yuvimg_eet = eet.usm(sharp=sharpness)
    yuv[:,:,0] = yuvimg_eet
    bgr_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).astype(np.uint8)
    cv2.imwrite(os.path.join(path, 'result.jpg'), bgr_img)