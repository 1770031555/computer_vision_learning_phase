# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午4:55
# Target  =
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)
# cv2.namedWindow("image", 0)
# cv2.imshow("source image", img)
# cv2.waitKey(0)
# print("INFO : image shape : {}".format(img.shape))

# 对原始图像进行裁剪
image_crop = img[400:800, 630:900, :]
# x1, y1 = 630, 400 ; x2, y2 = 900, 800
cv2.namedWindow("image_crop", 0)
cv2.imshow("image crop", image_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

