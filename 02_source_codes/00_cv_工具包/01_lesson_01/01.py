# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 上午10:20
# Target  =
import os, codecs, time
import numpy as np

import cv2
print(cv2.__version__)
from PIL import Image

def PIL_read_image(image_path : str) -> np.ndarray:
	if not os.access(image_path, os.F_OK):
		raise RuntimeError("ERROR : {} path does not exists".format(
			image_path
		))
	pil_img = Image.open(image_path)
	# pil_img_numpy = np.array(pil_img)
	return pil_img

def opencv_read_image(image_path):
	return cv2.imread(image_path)


def image_shape(cv2_img):
	return cv2_img.shape


def cv2_convert_img(cv2_img):
	convert_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
	return convert_img


def display_img(cv2_img):
	cv2.namedWindow("image", 0)
	cv2.imshow("image", cv2_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return None


if __name__ == '__main__':
	# 1. 从地址中读取对应的图像
	path = "00_data/1.jpg"
	img = opencv_read_image(image_path=path)
	print(img)
	cv2.namedWindow("source image", 0)
	cv2.imshow("image", img)
	cv2.waitKey(0)
