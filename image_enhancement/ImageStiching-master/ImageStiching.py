﻿from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("./pic/4k.jpg")
imageB = cv2.imread("./pic/4y.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()