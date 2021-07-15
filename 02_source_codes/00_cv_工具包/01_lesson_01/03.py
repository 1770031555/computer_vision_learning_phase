# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 下午4:44
# Target  =
import cv2

from PIL import Image
import numpy as np

image_path = "00_data/1.jpg"
image_cv = cv2.imread(image_path)

# image_pil = np.array(Image.open(image_path))
# print(image_cv) # [[[238 219 214]
# print('------')
# print(image_pil) # [[[214 219 238]

if __name__ == '__main__':
	image_height = 480
	image_width = 640

	img = np.zeros([image_height, image_width, 3],
				   dtype=np.float)
	cv2.namedWindow("black", 1)
	cv2.imshow("black", img)
	cv2.waitKey(500)
	# generate one black image
	img[:, :, 0].fill(1.0)
	cv2.namedWindow("B", 1)
	cv2.imshow("b", img)
	cv2.waitKey(500)
	# 将 0 通道进行填充，即形成了蓝色
	img[:, :, 1].fill(255.0)
	cv2.namedWindow("green", 0)
	cv2.imshow("green", img)
	cv2.waitKey(500)
	# 将 1 通道进行填充，即形成了绿色
	img[:, :, 2].fill(1.0)
	cv2.namedWindow("red", 0)
	cv2.imshow("red", img)
	cv2.waitKey(500)
	# 将 2 通道进行填充

	cv2.destroyAllWindows()




