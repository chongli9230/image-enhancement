import numpy as np
import cv2

def enhance(gr):
    h, w = gr.shape[0], gr.shape[1]
    gr = gr/255
    new_gr = np.zeros((h,w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if gr[i,j] == 0:
                continue
            new_gr[i,j] = 1.245/(1.0+(0.545/gr[i,j])**2.35) 
    return new_gr*255



if __name__ == '__main__':
    img_path = "F:/JZYY/pic/ETIS-LaribPolypDB/ETIS-LaribPolypDB/"
    img_name = "97.tif"

    img_path2 = "F:/JZYY/pic/blood-vessel/"
    img_name2 = "2.png"
    
    img_path3 = "F:/JZYY/code/2018Endoscope_image_enhancement/result/myresult/"
    img_name3 = "2myy.png"

    img = cv2.imread(img_path2 + img_name2)
    
    
    b = img[:,:, 0]
    g = img[:,:, 1]
    r = img[:,:, 2]

    #bg  r*0.9

    G = g
    B = enhance(b)
    R = r

    B = np.clip(B, 0, 255).astype('uint8')
    G = np.clip(G, 0, 255).astype('uint8')
    R = np.clip(R, 0, 255).astype('uint8')

    res = cv2.merge([B,G,R]) 
    img = np.uint8(np.clip(res, 0, 255))
    
    
    cv2.imshow("image",img)
    #cv2.imwrite('./result/myresult/2mycon.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
