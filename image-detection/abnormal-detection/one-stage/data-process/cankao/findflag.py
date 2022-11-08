import cv2
import numpy as np

## 像素值矩阵
img_bgr = cv2.imread('../findflag.jpg')   #读取图像
cv2.imshow('原始图像', img_bgr)            #显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()