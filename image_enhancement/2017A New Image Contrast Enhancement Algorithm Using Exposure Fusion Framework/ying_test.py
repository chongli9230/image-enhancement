import cv2
import sys
import imageio
from ying import Ying_2017_CAIP
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy, scipy.misc, scipy.signal

def main():
    img_path = "F:/JZYY/pic/pic/"
    #img_path2 = "F:/JZYY/pic/CVC-ClinicDB/Original/"
    img_name = "4.png"
    #img_name = sys.argv[1]
    img = imageio.imread(img_path + img_name)

    result = Ying_2017_CAIP(img)
   
    plt.imshow(result)
    imageio.imsave("./result/ying/ying_"+img_name,result)
    plt.show()

if __name__ == '__main__':
    main()