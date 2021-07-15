# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:29
# Target  =
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)
cv2.line(img, (200, 100), (700, 600), (0, 0, 255), 50)
# 50代表绘制的线条的thickness
cv2.namedWindow("line image", 0)
cv2.imshow("line image", img)
cv2.waitKey(0)


