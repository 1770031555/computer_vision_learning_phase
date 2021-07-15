# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:02
# Target  =
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)
# 直接显示对应的图像
# cv2.namedWindow("source image", 0)
# cv2.imshow("source Image", img)
# cv2.waitKey(0)

# image resize 时候，宽度在前高度在后
# img_r = cv2.resize(img, (125, 256), interpolation=cv2.INTER_LINEAR)
# cv2.namedWindow("resize image")
# cv2.imshow("resized image", img_r)
# cv2.waitKey(0)
# print("INFO : shape of img_r {}".format(
# 	img_r.shape
# ))



