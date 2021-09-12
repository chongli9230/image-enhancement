from scipy import optimize
from sympy import *
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt



def calculate(img, v):
    height, wide = img.shape   #966 1225
    #print(img.type)
    V = np.zeros(shape=(height ,wide))

    if(v == 1):
        M = np.zeros(shape=(height ,wide))
        Mint = np.zeros(shape=(height ,wide))

        for i in range(1, height-1):
            for j in range(1, wide-1):
                M[i,j] = abs((8*img[i,j]-img[i-1,j-1]-img[i-1,j]-img[i-1,j+1]-img[i,j-1]-img[i,j+1]
                            -img[i+1,j-1]-img[i+1,j]-img[i+1,j+1])/8)
                Mint[i,j] = int(M[i,j])

        maxM = int(np.amax(M))
        minM = int(np.amin(M))

        nums = np.zeros(maxM+1)
        for i in range(1, height-1):
            for j in range(1, wide-1):
                temp = int(Mint[i,j])
                nums[temp] += 1

        bestvar = bestt = bestu = bestu1 = bestu2 =0
        for t in range(0, maxM+1):
            u1 = u2 = w1 = w2 = 0
            for everynum in range(len(nums)):
                if(everynum > t):
                    w2 += nums[everynum]/(height * wide)
                else:
                    w1 += nums[everynum]/(height * wide)
            for i in range(0,t+1):
                u1 += i*nums[i]/(height * wide)
            if(w1 != 0):
                u1 = u1/w1
            for i in range(t+1,maxM+1):
                u2 += i*nums[i]/(height * wide)
            if(w2 != 0):
                u2 = u2/w2
            #总平均值
            u = u1*w1 + u2*w2
            #方差
            var = w1*(u1-u)*(u1-u) + w2*(u2-u)*(u2-u)
            if(var > bestvar):
                bestvar = var
                bestt = t
                bestu = u    #总平均梯度
                bestu1 = u1    #纹理
                bestu2 = u2    #边缘

        #print(bestt,bestu,bestu1,bestu2)
        v1 = (bestu2-bestu)/bestu2     #边缘
        v2 = (bestu-bestu1)/bestu      #纹理
        print("v1: "+str(v1))
        print("v2: "+str(v2))
        
        for i in range(0, height):
            for j in range(0, wide):
                if(M[i,j] <= 2):
                    V[i,j] = 0
                elif ((M[i,j] < bestt and M[i,j]/bestt < v2)):
                    V[i,j] = M[i,j]/bestt
                elif ((M[i,j] < bestt and M[i,j]/bestt >= v2)):
                    V[i,j] = v2
                elif ((M[i,j] >= bestt and (M[i,j]-bestt)/M[i,j] < v1)):
                    V[i,j] = v1
                elif ((M[i,j] >= bestt and (M[i,j]-bestt)/M[i,j] >= v1)):
                    V[i,j] = (M[i,j]-bestt)/M[i,j]
                    
        
        minV = np.amin(V)
        maxV = np.amax(V)

    newimg = np.zeros(shape=(height ,wide))
    for i in range(0, height):
        for j in range(0, wide):
            if(i==0 or i==1 or j==0 or j==1 or i==height-1 or i==height-2 or j==wide-1 or j==wide-2):
                newimg[i,j] = img[i,j]
            else:
                #0.5,  0.8, 0.5* V[i,j]
                if(v == 1):
                    v = 0.5* V[i,j]
                a0 = 1
                a1 = -v
                a2 = (-v)*(-v+1)/2
                temp_1=(8+4*v*v-12*v)

                temp = a2*(int(img[i-2][j-2])+int(img[i-2][j])+int(img[i-2][j+2])+int(img[i][j-2])+int(img[i][j+2])+int(img[i+2][j-2])+int(img[i+2][j])+int(img[i+2][j+2]))
                temp+= a1*(int(img[i-1][j-1])+int(img[i-1][j])+int(img[i-1][j+1])+int(img[i][j-1])+int(img[i][j+1])+int(img[i+1][j-1])+int(img[i+1][j])+int(img[i+1][j+1]))
                temp+= 8*a0*img[i][j]
                temp = temp/ temp_1
                if(temp<0):
                    temp = 0
                elif(temp>255):
                    temp = 255
                newimg[i,j] = int(temp)

    return newimg


