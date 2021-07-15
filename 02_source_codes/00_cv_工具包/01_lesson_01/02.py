# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午4:34
# Target  =
import cv2
print(cv2.__version__)

image_path = "00_data/02.jpg"
img = cv2.imread(image_path)
# img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.namedWindow("image", 0)
# cv2.imshow("image", img)
# cv2.imshow("image_convert", img_convert)
# cv2.waitKey(0)

# 对图像进行翻转
# image_flip = cv2.flip(img, 1)
# cv2.namedWindow("image_flip_LR", 0)
# cv2.imshow("image flip", image_flip)
# cv2.waitKey(0)

# 图像左右翻转
# image_flip = cv2.flip(img, 0)
# cv2.imshow("image flip UD", image_flip)
# cv2.waitKey(0)

image_flip = cv2.flip(img, -1)
cv2.imshow("imageFLIP", image_flip)
cv2.waitKey(5000)

cv2.destroyAllWindows()
