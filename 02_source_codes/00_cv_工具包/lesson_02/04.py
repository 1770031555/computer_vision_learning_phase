# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午5:31
# Target  =
import numpy as np
import cv2

image_path = "00_data/1.jpg"

img = cv2.imread(image_path)

cv2.putText(img, "AAAA",
			(5, 55), cv2.FONT_HERSHEY_SIMPLEX,
			1.5, (0, 0, 255),
			7)
cv2.namedWindow("text image", 0)
cv2.imshow("AAA", img)
cv2.waitKey(0)

