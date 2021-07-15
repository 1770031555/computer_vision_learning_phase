# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:25
# Target  =
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)

x1, y1 = 200, 200
x2, y2 = 400, 400

cv2.rectangle(img, (x1, y1), (x2, y2),
			  (0, 0, 255), 5)
cv2.namedWindow("rectangle image", 0) # 0 代表绘制的图像可以使用鼠标
cv2.imshow("rectangle shape", img)
cv2.waitKey(0)
# cv2.rectangle 5 代表绘制空心的矩形 0 代表绘制实心的矩形

