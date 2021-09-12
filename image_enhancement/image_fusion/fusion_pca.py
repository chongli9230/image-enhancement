from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte



class FusionPCA():
    """ Image fusion based PCA"""
    def __init__(self, imageNames):
        self._imageNames = imageNames
        self._images = []
        self._fusionImage = None

    def _load_images(self):
        for name in self._imageNames:
            self._images.append(np.array(Image.open(name)))
    def fusion(self):
        self._load_images()
        imageSize = self._images[0].size
        # Todo: for more than two images
        allImage = np.concatenate((self._images[0].reshape(1, imageSize), self._images[1].reshape(1, imageSize)), axis=0)
        covImage = np.cov(allImage)
        D, V = np.linalg.eig(covImage)
        if D[0] > D[1]:
            a = V[:,0] / V[:,0].sum()
        else:
            a = V[:,1] / V[:,1].sum()
        self._fusionImage = self._images[0]*a[0] + self._images[1]*a[1]
        return self._fusionImage


    def plot(self):
        plt.figure()
        plt.gray()
        plt.subplot(1,3,1)
        plt.imshow(self._images[0])
        plt.subplot(1,3,2)
        plt.imshow(self._images[1])
        plt.subplot(1,3,3)
        new_im = Image.fromarray(self._fusionImage.astype(np.uint8))
        plt.imshow(new_im)
        plt.savefig("./result/pca8.jpg")        
        plt.show()

if __name__ == '__main__':
    IMAGEPATH = "F:/JZYY/code/image_fusion/pic/"
    imLists = [IMAGEPATH+"8y.tif",IMAGEPATH+"8k.tif"]
    #imLists = [IMAGEPATH+"6y.jpg",IMAGEPATH+"6k.jpg"]
    
    fu = FusionPCA(imLists)
    fu.fusion()
    fu.plot()