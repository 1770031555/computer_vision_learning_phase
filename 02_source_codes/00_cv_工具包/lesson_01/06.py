# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:11
# Target  =
import cv2
import numpy as np

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)

# cv2.namedWindow("source image", 0)
# cv2.imshow("source image", img)
# cv2.waitKey(0)
# print(img.shape)

B, G, R = cv2.split(img)
# cv2.namedWindow("Blue", 0)
# cv2.imshow("Blue", B)
# cv2.waitKey(500)

# cv2.namedWindow("Green", 0)
# cv2.imshow("Green", G)
# cv2.waitKey(500)

# cv2.namedWindow("Red", 0)
# cv2.imshow("Red", R)
# cv2.waitKey(0)

image_merge = cv2.merge([B, G, R])
cv2.namedWindow("merge", 0)
cv2.imshow("merge image", image_merge)
cv2.waitKey(0)

