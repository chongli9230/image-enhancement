from PIL import Image
import numpy as np
import pylab as plt
import pywt

class FusionDWB():
    """ Image Fusion based wavelet  """

    def __init__(self, imageNames = None, zt=2, ap=2, mp=0):
        self._imageNames = imageNames
        self._images = []
        self._fusionImage = None
        self._zt = zt   # level num
        self._ap = ap   # 0-average, 1-min, 2-max
        self._mp = mp   # 0-average, 1-min, 2-max

    def _load_images(self):
        for name in self._imageNames:
            self._images.append(np.array(Image.open(name)))

    def fusion(self):
        self._load_images()
        coeffss = []
        for image in self._images:
            coeffss.append(pywt.wavedec2(image, 'db1', level=self._zt))
        # low pass
        if self._mp == 0:
            cAF = coeffss[0][0]
            for coeffs in coeffss[1:]:
                cAF += coeffs[0]
            cAF = cAF/len(coeffs)
        # high pass
        if self._ap == 2:
            hipassF  = coeffss[0][1:]
            for coeffs in coeffss[1:]:   # every image
                for idxLevel, HVDs in enumerate(coeffs[1:]):   # every level
                    for idxDirec, HVD in enumerate(HVDs):
                        maxMap = hipassF[idxLevel][idxDirec] < HVD
                        hipassF[idxLevel][idxDirec][maxMap] = HVD[maxMap]

        coeffsFusion = [cAF,] + hipassF
        self._fusionImage = pywt.waverec2(coeffsFusion, 'db1')
        return self._fusionImage

    def plot(self):
        plt.figure(0)
        plt.gray()
        plt.subplot(131)
        plt.imshow(self._images[0])
        plt.subplot(132)
        plt.imshow(self._images[1])
        plt.subplot(133)
        new_im = Image.fromarray(self._fusionImage.astype(np.uint8))
        plt.imshow(new_im)
        plt.savefig("./result/dwb8.jpg")
        plt.show()


if __name__ == '__main__':
    IMAGEPATH = "F:/JZYY/code/image_fusion/pic/"
    imLists = [IMAGEPATH+"8y.tif",IMAGEPATH+"8k.tif"]
    #imLists = [IMAGEPATH+"5y.png",IMAGEPATH+"5k.png"]
    #imLists = [IMAGEPATH+"5y.jpg",IMAGEPATH+"5k.jpg"]
    fu = FusionDWB(imLists)
    fu.fusion()
    fu.plot()