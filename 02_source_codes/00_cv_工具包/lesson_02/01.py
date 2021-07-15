# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:20
# Target  =
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)
x, y = 500, 400
R = 60

cv2.circle(img, (x, y), R, (0,0,255), -1)
cv2.namedWindow("circle")
cv2.imshow("circle", img)
cv2.waitKey(0)
# -1表示绘制实心圆形 4 表示绘制空心圆形


