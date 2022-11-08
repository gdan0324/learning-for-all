# coding:UTF-8
# v1--读取与显示图像
import cv2
import numpy as np

# train\02\Images\02_0035.PNG
## 图像类型转换
img_bgr = cv2.imread('../../train/02/Images/02_0035.PNG')                  # 读取彩色图像
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)      # 转化为 HSV 格式
thresh1 = np.array([0, 120, 120])                       # 目标红旗的低阈值
thresh2 = np.array([10, 255, 255])                      # 目标红旗的高阈值
img_flag = cv2.inRange(img_hsv, thresh1, thresh2)       # 获取红旗部分
    
## 形态学滤波
img_morph = img_flag.copy()                             # 复制图像
cv2.erode(img_morph, (3,3), img_morph, iterations= 3)   # 腐蚀运算
cv2.dilate(img_morph, (3,3), img_morph, iterations= 3)  # 膨胀运算


## 获取图像特征
cnts, _ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 获取图像轮廓
cnts_sort = sorted(cnts, key= cv2.contourArea, reverse= True) # 将轮廓包含面积从大到小排列
box = cv2.minAreaRect(cnts_sort[0])                     # 选取包含最大面积的轮廓，并得出最小外接矩形
points = np.int0(cv2.boxPoints(box))                    # 获得该矩形的四个定点

cen_v = (points[0,0] + points[2,0]) / 2                 # 得出横向中心
cen_h = (points[0,1] + points[2,1]) / 2                 # 得出纵向中心
rows, cols = img_bgr.shape[:2]
print('彩色图像大小: (' + str(cols) + ', ' + str(rows) + ')')
print('目标中心位置: (' + str(cen_h) + ', ' + str(cen_v) + ')')
print('目标中心位置: (' + str(points[0,0] ) + ', ' + str(points[2,0]) + ')')

cv2.drawContours(img_bgr, [points], -1, (255,0,0), 2)         # 在原图上绘制轮廓

## 显示图像  
cv2.imshow('框选图像', img_bgr)                                    
cv2.imshow('红旗图像', img_flag)
cv2.imshow('滤波图像', img_morph)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()