# -*- coding:utf-8 -*-
r'''
图像使用过程的一般方法
'''
import os, json, codecs, time, glob
from PIL import Image
from pprint import pprint
import numpy as np
from tqdm import tqdm

# TODO 1 求训练数据的 MEAN 和 STD
# Image.open 出来的图像的通道是 RGB 的顺序 ， 这和OpenCV读取的通道BGR的顺序刚好相反
TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/dataset/cifar100/trainval"
folder_list = [os.path.join(TRAIN_FOLDER_PATH, _) for _ in os.listdir(TRAIN_FOLDER_PATH)]

means = [0, 0, 0]
stdevs = [0, 0, 0]
image_count = 0
for folder in tqdm(folder_list):
	image_file_list = glob.glob(os.path.join(folder, "*.png"))
	for image_file in image_file_list:
		img = np.array(Image.open(image_file)) / 255.
		# resize((target_image_width, target_image_height))
		for i in range(3):
			means[i] += img[:, :, i].mean()
			stdevs[i] += img[:, :, i].std()
		image_count += 1

# print(image_count)
# print(means)
# means.reverse()
# print(means)
# print(stdevs)
# stdevs.reverse()
# print(stdevs)

means = np.asarray(means) / image_count
stdevs = np.asarray(stdevs) / image_count

print(means)
print(stdevs)



