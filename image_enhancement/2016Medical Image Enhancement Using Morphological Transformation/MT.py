"""
This project is used to enhance the contrast of X-ray image. It is an implementation of Morphological Transformation.
For detail, please refer to 'Medical Image Enhancement Using Morphological Transformation'.

Last Modified: Jan 1st, 2019

vxallset@outlook.com

All rights reserved.
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import support
import utils
import math
import trans

def PSNR(img1, img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def process1(file_name):
    print('Start processing...')

    start_time = time.time()

    #file_name = '4.raw'
    file_path = './demo_img/' + file_name
    img_save_path = './demo_img/'

    #raw_image = utils.raw_read(file_path)
    #roi_image = support.get_ROI(raw_image)
    tf1 = cv2.imread(file_path)

    YUV_image = cv2.cvtColor(tf1, cv2.COLOR_BGR2YUV)
    roi_image = YUV_image[:,:,0]
    #print(roi_image.shape)
    kernel_size = 28
    #6-28   26.27

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ABH_img = cv2.morphologyEx(roi_image, cv2.MORPH_BLACKHAT, kernel)
    ATH_img = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel)

    A_q = roi_image + ATH_img - ABH_img

    result = support.post_propossing(A_q, 2.2)
    end_time = time.time()
    print('Finished processing, time elapsed: {}s.'.format(end_time - start_time))

    original_image_complement_with_gamma = support.post_propossing(roi_image, 2.2)
    plt.imsave(img_save_path + file_name[:-4] + '_original.png', original_image_complement_with_gamma, cmap='gray')
    plt.imsave(img_save_path + file_name[:-4] + '_enhanced.png', result, cmap='gray')

    psnr = PSNR(original_image_complement_with_gamma, result)
    print(psnr)

    utils.visualize_center_line_density((original_image_complement_with_gamma, result),
                                        titles=['Original Image', 'Morphological Transformation Enhanced Image'])
    plt.show()


    #trans.color("./demo_img/6.tif", "./demo_img/6_enhanced.png")
    #trans.color("./demo_img/5.tif", "./demo_img/5_enhanced.png")


    YUV_image[:,:,0] = result
    im = cv2.cvtColor(YUV_image, cv2.COLOR_YUV2BGR)

    cv2.imshow("out", im)
    cv2.imwrite("./demo_img/6_out2.tif", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def process2(roi_image,file_name):
    print('Start processing...')
    start_time = time.time()

    img_save_path = './demo_img/'
    #print(roi_image.shape)
    kernel_size = 28
    #6-28   26.27

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ABH_img = cv2.morphologyEx(roi_image, cv2.MORPH_BLACKHAT, kernel)
    ATH_img = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel)

    A_q = roi_image + ATH_img - ABH_img

    result = support.post_propossing(A_q, 2.2)
    end_time = time.time()
    print('Finished processing, time elapsed: {}s.'.format(end_time - start_time))

    original_image_complement_with_gamma = support.post_propossing(roi_image, 2.2)
    plt.imsave(img_save_path + file_name[:-4] + '_original.png', original_image_complement_with_gamma, cmap='gray')
    plt.imsave(img_save_path + file_name[:-4] + '_enhanced.png', result, cmap='gray')

    psnr = PSNR(original_image_complement_with_gamma, result)
    print(psnr)

    utils.visualize_center_line_density((original_image_complement_with_gamma, result),
                                        titles=['Original Image', 'Morphological Transformation Enhanced Image'])
    #plt.show()


    #trans.color("./demo_img/6.tif", "./demo_img/6_enhanced.png")
    #trans.color("./demo_img/5.tif", "./demo_img/5_enhanced.png")

    return result
