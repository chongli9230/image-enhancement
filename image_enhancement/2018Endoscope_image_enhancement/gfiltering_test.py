import numpy as np
import time
import cv2
from gfiltering import process


if __name__ == '__main__':
    
    img_path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
    img_path2 = "F:/JZYY/pic/CVC-ClinicDB/Original/"
    img_name = "121.tif"
    #19, 49, 43，139
    img_path3 = "F:/JZYY/pic/blood-vessel/"
    img_name3 = "2.png"


    image = cv2.imread(img_path3 + img_name3)
    """   #1
    image[:,:,0] = process(image[:,:,0])
    image[:,:,1] = process(image[:,:,1])
    image[:,:,2] = process(image[:,:,2])

    
    cv2.imshow("image",image)
    #cv2.imwrite("./result/3_" + img_name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """

    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #grayimage = cv2.resize(grayimage,None,fx=0.8,fy=0.8,interpolation=cv2.INTER_CUBIC) 

    enhanced_grayimage = process(grayimage)
    """ #2
    # b,g,r 
    temp =cv2.divide(enhanced_grayimage, grayimage)
    B =  cv2.multiply(image[:,:,0], temp) 
    G =  cv2.multiply(image[:,:,1], temp) 
    R =  cv2.multiply(image[:,:,2], temp) 


    res = cv2.merge([B,G,R]) 
    
    """
    # 3
    lambdaa=[1, 1, 1]
    EPS = 1e-6
    S_restore = np.zeros(image.shape)
    for j in range(3):  # b,g,r
        S_restore[..., j] = enhanced_grayimage * (1.0 * image[..., j] / (grayimage + EPS)) * lambdaa[j]

    res = np.clip(S_restore, 0, 255).astype('uint8')
    
    cv2.imshow("image",res)
    #cv2.imwrite("./result/g" + img_name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    












