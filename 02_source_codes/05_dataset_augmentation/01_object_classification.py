# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/9 : 下午3:05
# Target  =
'''
Test usage for data augmentation task
RandomCrop : 按照给定高度和宽度进行随机剪切，width裁剪后的宽度，height裁剪后的高度, p随机裁剪的概率
HorizontalFlip : 按照y轴进行随机的旋转
RandomBrightnessContrast : 亮度调整
Rotate : 角度调整
'''
import json
import os, glob, time, codecs
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import albumentations as A
current_file_path = os.path.abspath('.')
transform = A.Compose([
	A.CenterCrop(width=70, height=70, p=0.3),
	# A.RandomCrop(width=256, height=256, p=0.5),
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(p=0.2),
	A.Rotate()
])

# image = cv2.imread("cat.png")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# use opencv to load image

def generate_dataset(target_image_path_json_file:str, generate_image_folder:str, amount=None, target_image_size=(None, None)) -> None:
	r"""
	:param target_image_path_json_file: JSON-file include the target image path and the label of the image
	:param generate_image_folder: Folder storage the generate files
	:param amount: generate how many image based on one source image
	:return: None
	"""
	with codecs.open(filename=target_image_path_json_file, mode='r', encoding='utf-8') as fr:
		target_image_labels = json.load(fr)
	fa = codecs.open(filename='final_output.txt', mode='a', encoding='utf-8')
	if not os.access(generate_image_folder, os.F_OK):
		print("INFO : {} does not exists, make folder now".format(generate_image_folder))
		os.makedirs(generate_image_folder)
	if amount==None:
		amount = 10
	for (img_path, img_label) in target_image_labels:
		if not os.access(img_path, os.F_OK):
			print("ERROR : {} does not exists".format(img_path))
			continue
		if target_image_size[0]!=None and target_image_size[1]!=None:
			# img = Image.open(img_path).resize((self.target_image_width, self.target_image_height),
			# 								  Image.ANTIALIAS).convert('RGB')
			image = Image.open(img_path).resize((target_image_size[0], target_image_size[1]), Image.ANTIALIAS).convert('RGB')
		else:
			image = Image.open(img_path).convert('RGB')
		image = np.array(image)
		img_base_name = os.path.basename(img_path).replace(".png", "_")
		for i in range(amount):
			transformed = transform(image=image)
			transformed_image = transformed['image']
			transformed_image = Image.fromarray(transformed_image)
			# transformed_image.show()
			target_image_path = os.path.join(generate_image_folder, img_base_name + str(i) + ".png")
			transformed_image.save(target_image_path)
			fa.write(json.dumps(dict(
				image_path=os.path.join(current_file_path, target_image_path),
				image_label=img_label,
			), ensure_ascii=False))
			fa.write('\r\n')


if __name__ == '__main__':
	apple_folder_path = "/home/liuyang/Desktop/02_CV/03_study/03_data_augmendation/01_my_codes/data/apple"
	bee_folder_path = "/home/liuyang/Desktop/02_CV/03_study/03_data_augmendation/01_my_codes/data/bee"
	all_data = list()
	files_list = glob.glob(os.path.join(apple_folder_path, "*.png"))
	for f in files_list:
		all_data.append([f, 'apple'])
	files_list = glob.glob(os.path.join(bee_folder_path, "*.png"))
	for f in files_list:
		all_data.append([f, "bee"])
	with codecs.open(filename='01.json', mode='w',
					 encoding='utf-8') as fw:
		json.dump(obj=all_data, fp=fw, ensure_ascii=False, indent=4)

	generate_dataset(target_image_path_json_file="01.json",
					 generate_image_folder="data_2",
					 target_image_size=[100, 100])











