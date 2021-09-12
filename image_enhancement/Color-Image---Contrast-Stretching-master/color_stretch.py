"""
Created on Tue Oct 30 11:16:51 2018

@author: Binish125
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class stretcher:
    def __init__(self):
        self.img=[];
        self.newImage=[];
        self.pix_number=0;
        self.redOnly=[];
        self.blueOnly=[];
        self.greenOnly=[];
        self.L=256;
        
        
    def stretch(self,input_image):
        self.img=cv2.imread(input_image);
        self.pix_number=np.ma.count(self.img);
        number_of_cols=int(self.img[1].size/3)
        number_of_rows=(int(np.ma.count(self.img)/self.img[1].size))
        self.newImage=np.random.randint(0,1,size=(number_of_rows,number_of_cols,3))
        
        self.redOnly=np.random.randint(0, 1, size=(number_of_rows, number_of_cols))
        self.greenOnly=np.random.randint(0, 1, size=(number_of_rows, number_of_cols))
        self.blueOnly=np.random.randint(0, 1, size=(number_of_rows, number_of_cols))
        
        for i in range(number_of_rows):
            for j in range(number_of_cols):
                self.blueOnly[i][j]=self.img[i][j][0]
                self.greenOnly[i][j]=self.img[i][j][1]
                self.redOnly[i][j]=self.img[i][j][2]

        x,a,d=plt.hist(self.blueOnly.ravel(),256,[0,256],label='x')
        y,b,e=plt.hist(self.greenOnly.ravel(),256,[0,256],label='y')
        z,c,f=plt.hist(self.redOnly.ravel(),256,[0,256],label='z')
        
        arr=np.vstack([x,y,z])
        k=np.zeros(3)
        sk=np.zeros(3)
        k[0]=np.sum(x)
        k[1]=np.sum(y)
        k[2]=np.sum(z)
        
        prk_list=np.empty((3,len(x)));
        sk_list=np.empty((3,len(x)));
        last_list=np.empty((3,len(x)));
                
        for n in range(3):
            for i in range(len(x)):
                prk=arr[n][i]/k[n]
                sk[n]+=prk
                last=(self.L-1)*sk[n]
                rem= int(last % last if last!=0 else 0)
                last=int(last+1 if rem >=0.5 else last)
                prk_list[n][i]=prk
                sk_list[n][i]=sk[n]
                last_list[n][i]=last
                
        for i in range(number_of_rows):
            for j in range(number_of_cols):
                num_red=self.redOnly[i][j]
                if num_red != last_list[2][num_red]:
                    self.redOnly[i][j]=last_list[2][num_red]
                num_green=self.greenOnly[i][j]
                if num_green != last_list[1][num_green]:
                    self.greenOnly[i][j]=last_list[1][num_green]
                num_blue=self.blueOnly[i][j]
                if num_blue != last_list[0][num_blue]:
                    self.blueOnly[i][j]=last_list[0][num_blue]
                self.newImage[i][j][0]=self.blueOnly[i][j]
                self.newImage[i][j][1]=self.greenOnly[i][j]
                self.newImage[i][j][2]=self.redOnly[i][j]
                
        cv2.imwrite("output_data/output.jpg",self.newImage)

    def plotHistogram(self):
        #needs to be fixed
         plt.hist(self.newImage.ravel(),256,[0,256])
         
    def showImage(self):
        newImage=cv2.imread("output_data/output.jpg");
        cv2.imshow("Output-Image",newImage);
        cv2.imshow("Input-Image",self.img);
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        

st=stretcher();
st.stretch('image_data/a1.tif');
st.plotHistogram();
st.showImage();