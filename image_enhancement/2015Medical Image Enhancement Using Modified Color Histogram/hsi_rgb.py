import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections


def rgb_hsi(rgb_image):
    # 保存原始图像的行列数
    rows = int(rgb_image.shape[0])
    cols = int(rgb_image.shape[1])
    # 图像复制
    hsi_image = rgb_image.copy()
    # 通道拆分
    b = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    r = rgb_image[:, :, 2]
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            if den == 0:
                hsi_h = 0
            else:
                theta = float(np.arccos(num / den))
                if b[i, j] <= g[i, j]:
                    hsi_h = theta
                else:
                    hsi_h = 2*np.pi - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                hsi_s = 0
            else:
                hsi_s = 1 - 3*min_RGB/sum

            hsi_h = hsi_h/(2*np.pi)
            hsi_i = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_image[i, j, 0] = hsi_h*255
            hsi_image[i, j, 1] = hsi_s*255
            hsi_image[i, j, 2] = hsi_i*255
    return hsi_image


def hsi_rgb(hsi_image):
    # 保存原始图像的行列数
    rows = np.shape(hsi_image)[0]
    cols = np.shape(hsi_image)[1]
    # 对原始图像进行复制
    rgb_image = hsi_image.copy()
    # 对图像进行通道拆分
    hsi_h = hsi_image[:, :, 0]
    hsi_s = hsi_image[:, :, 1]
    hsi_i = hsi_image[:, :, 2]
    # 把通道归一化到[0,1]
    hsi_h = hsi_h / 255.0
    hsi_s = hsi_s / 255.0
    hsi_i = hsi_i / 255.0
    B, G, R = hsi_h, hsi_s, hsi_i
    for i in range(rows):
        for j in range(cols):
            hsi_h[i, j] *= 360
            if 0 <= hsi_h[i, j] < 120:
                B = hsi_i[i, j] * (1 - hsi_s[i, j])
                R = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
                    (60 - hsi_h[i, j]) * np.pi / 180))
                G = 3 * hsi_i[i, j] - (R + B)
            elif 120 <= hsi_h[i, j] < 240:
                hsi_h[i, j] = hsi_h[i, j] - 120
                R = hsi_i[i, j] * (1 - hsi_s[i, j])
                G = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
                    (60 - hsi_h[i, j]) * np.pi / 180))
                B = 3 * hsi_i[i, j] - (R + G)
            elif 240 <= hsi_h[i, j] <= 300:
                hsi_h[i, j] = hsi_h[i, j] - 240
                G = hsi_i[i, j] * (1 - hsi_s[i, j])
                B = hsi_i[i, j] * (1 + (hsi_s[i, j] * np.cos(hsi_h[i, j] * np.pi / 180)) / np.cos(
                    (60 - hsi_h[i, j]) * np.pi / 180))
                R = 3 * hsi_i[i, j] - (G + B)
            rgb_image[i, j, 0] = B * 255
            rgb_image[i, j, 1] = G * 255
            rgb_image[i, j, 2] = R * 255
    return rgb_image


# 计算灰度图的直方图
def draw_histogram(grayscale):
    # 对图像进行通道拆分
    hsi_i = grayscale[:, :, 2]
    color_key = []
    color_count = []
    color_result = []
    histogram_color = list(hsi_i.ravel())  # 将多维数组转换成一维数组
    color = dict(collections.Counter(histogram_color))  # 统计图像中每个亮度级出现的次数
    color = sorted(color.items(), key=lambda item: item[0])  # 根据亮度级大小排序
    for element in color:
        key = list(element)[0]
        count = list(element)[1]
        color_key.append(key)
        color_count.append(count)
    for i in range(0, 256):
        if i in color_key:
            num = color_key.index(i)
            color_result.append(color_count[num])
        else:
            color_result.append(0)
    color_result = np.array(color_result)
    return color_result


def histogram_equalization(histogram_e, lut_e, image_e):
    sum_temp = 0
    cf = []
    for i in histogram_e:
        sum_temp += i
        cf.append(sum_temp)
    for i, v in enumerate(lut_e):
        lut_e[i] = int(255.0 * (cf[i] / sum_temp) + 0.5)
    equalization_result = lut_e[image_e]
    return equalization_result


x = []
for i in range(0, 256):  # 横坐标
    x.append(i)

# 原图及其直方图
rgb_image = cv2.imread("./2.tif")
cv2.imshow('rgb', rgb_image)
histogram = draw_histogram(rgb_image)
plt.bar(x, histogram)  # 绘制原图直方图
plt.savefig('./imgs/before_histogram.png')
plt.show()

# rgb转hsi
hsi_image = rgb_hsi(rgb_image)
cv2.imshow('hsi_image', hsi_image)
cv2.imwrite('./imgs/hsi_result.png', hsi_image)

# hsi在亮度分量上均衡化
histogram_1 = draw_histogram(hsi_image)
lut = np.zeros(256, dtype=hsi_image.dtype)  # 创建空的查找表
result = histogram_equalization(histogram_1, lut, hsi_image)  # 均衡化处理
cv2.imshow('his_color_image', result)
cv2.imwrite('./imgs/his_color.png', result)  # 保存均衡化后图片

# hsi转rgb
image_equ = cv2.imread(r'./imgs/his_color.png')  # 读取图像
rgb_result = hsi_rgb(image_equ)
cv2.imshow('rgb_image', rgb_result)
cv2.imwrite('./imgs/gbr_result.png', rgb_result)

rgb = cv2.imread("./imgs/gbr_result.png")
histogram_2 = draw_histogram(rgb)
plt.bar(x, histogram_2)
plt.savefig('./imgs/after_histogram.png')
plt.show()

plt.plot(x, lut)  # 绘制灰度级变换曲线图
plt.savefig('./imgs/Grayscale_transformation_curve.png')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
