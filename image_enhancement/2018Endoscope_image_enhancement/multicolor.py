import numpy as np
import cv2

def enhance(g):
    # g = g/255.
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

if "__main__" == __name__:
    img_path = "F:/JZYY/code/2018Endoscope_image_enhancement/result/myresult/"
    #img_path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
    img_name = "6.png"
    img = cv2.imread(img_path + img_name) #BGR, HWC
    # img = img.astype(np.float)
    img = img/255.
    b = img[:,:, 0]
    g = img[:,:, 1]
    r = img[:,:, 2]
    new_g = enhance(g)
    new_r = r + new_g - g
    new_b = b + new_g - g
    img[:, :, 0] = new_b
    img[:, :, 1] = new_g
    img[:, :, 2] = new_r
    img = img*255.
    img = np.uint8(np.clip(img, 0, 255))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img_hsv[:, :, 1]
    s = s/255.
    new_s = enhance(s)
    img_hsv[:, :, 1] = new_s*255.
    img_enhance = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("image1",img)
    cv2.imshow("image2",img_enhance)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./result/myresult/6mul.png', img_enhance)