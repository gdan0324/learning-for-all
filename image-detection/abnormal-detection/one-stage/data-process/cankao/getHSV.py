import cv2
import numpy as np

## 图像类型转换
img_bgr = cv2.imread('../../train/01/Images/01_0001.jpg')  
# img_bgr = cv2.imread('../findflag.jpg')                  # 读取彩色图像
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)      # 转化为 HSV 格式
thresh1 = np.array([0, 10, 10])                       # 目标红旗的低阈值
thresh2 = np.array([10, 255, 255])                      # 目标红旗的高阈值
img_flag = cv2.inRange(img_hsv, thresh1, thresh2)       # 获取红旗部分

cv2.imshow('原始图像', img_bgr)                          #显示图像           
cv2.imshow('红旗图像', img_flag)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()