# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/5 : 上午10:27
# Target  =
'''
Dataset utils for training and testing
'''
import os, time, json, codecs, datetime, glob
import random

import cv2
import numpy as np
from PIL import Image
from pprint import pprint
import torch
import xml.etree.cElementTree as ET
import albumentations as A


TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/train_dataset.json"
TEST_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/test_dataset.json"
INDEX_LABEL_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/label2int.json"
# 设置文件路径
with codecs.open(filename=INDEX_LABEL_PATH, mode='r', encoding='utf-8') as fr:
	INT2LABEL = json.load(fr)

def get_xml_msg(xml_file_path):
	list_x = []
	tree = ET.parse(xml_file_path)
	root = tree.getroot()
	for Object in root.findall("object"):
		name = Object.find("name").text
		bndbox = Object.find("bndbox")
		xmin = np.float((bndbox.find('xmin').text))
		xmax = np.float((bndbox.find('xmax').text))
		ymin = np.float((bndbox.find('ymin').text))
		ymax = np.float((bndbox.find('ymax').text))
		xyxy = [xmin, ymin, xmax, ymax]
		list_x.append([name, xyxy])
	return list_x


def make_dataset(image_folder_path: str, need_data_amount=None) -> list:
	r"""
	load dataset from image folder
	"""
	response_list = []
	# image_files_list = []
	# image_int_labels_list = []
	# image_bboxs_list = []
	with codecs.open(filename=image_folder_path, mode='r', encoding='utf-8') as fr:
		D = json.load(fr)
		for d in D:
			img_file_path = d[0]
			xml_file_path = d[1]
			if os.access(img_file_path, os.F_OK) and os.access(xml_file_path, os.F_OK):
				list_x_ = get_xml_msg(xml_file_path=xml_file_path)
				choose_idx = random.randint(0, int(len(list_x_)-1))
				list_x = list_x_[choose_idx]
				int_label = int(xml_file_path.split('/')[-2].split('-')[0])
				# image_files_list.append(img_file_path)
				# image_int_labels_list.append(int_label)
				# image_bboxs_list.append(list_x[1]) # [xmin, ymin, xmax, ymax]
				response_list.append(dict(
					img_path = img_file_path,
					int_label = int_label,
					bbox = list_x[1]
				))
	if need_data_amount:
		return response_list[:need_data_amount]
	else:
		return response_list

	# dataset = []
	# dataset_labels_dict = dict()
	# target_image_label = 0.
	# if not os.path.exists(image_folder_path):
	# 	raise RuntimeError('ERROR : {} does not exists'.format(image_folder_path))
	# if not os.path.isdir(image_folder_path):
	# 	raise RuntimeError('ERROR : {} is not one folder'.format(image_folder_path))
	#
	# image_folder_list = [os.path.join(image_folder_path, _) for _ in os.listdir(image_folder_path)]
	# for img_folder in image_folder_list:
	# 	img_label = os.path.basename(img_folder)
	# 	for img_path in glob.glob(os.path.join(img_folder, "*.png")):
	# 		img_target_label = float(LABEL2INT[img_label])
	# 		dataset.append(dict(
	# 			path=img_path,
	# 			label=img_label,
	# 			img_target_label=img_target_label
	# 		))
	# 		if img_label not in dataset_labels_dict.keys():
	# 			dataset_labels_dict.update({img_label: 0})
	# 		dataset_labels_dict[img_label] += 1
	# if False:
	# 	print('#####################################')
	# 	print("load dataset from {}".format(image_folder_path))
	# 	pprint(dataset_labels_dict)
	# 	print("dataset amount : {}".format(np.array(list(dataset_labels_dict.values())).sum()))
	# 	print('#####################################')
	#
	# if need_data_amount == None:
	# 	return dataset
	# else:
	# 	return dataset[:need_data_amount]


# if __name__ == '__main__':
# 	# 	# 计算对应有多少个类别
# 	# 	dataset, dataset_labels_dict = make_dataset(TRAIN_FOLDER_PATH)
# 	# 	dataset_labels_dict_KEYS = list(dataset_labels_dict.keys())
# 	# 	with codecs.open(filename='01.json', mode='w', encoding='utf-8') as fw:
# 	# 		json.dump(obj={J:I for I, J in enumerate(dataset_labels_dict_KEYS)},
# 	# 				  fp=fw, ensure_ascii=False, indent=4)
# 	# 	time.sleep(1000)
# 	# 计算训练数据的 MEAN STD
# 	# dataset = make_dataset(TRAIN_FOLDER_PATH)
# 	...
# else:
# 	...

def img_agu_crop(img_):
	r"""
	data augmentation via img crop
	"""
	scale_ = 5
	xmin = max(0, random.randint(0, scale_))
	ymin = max(0, random.randint(0, scale_))
	xmax = min(img_.shape[1]-1, img_.shape[1]-random.randint(0, scale_))
	ymax = min(img_.shape[0]-1, img_.shape[0]-random.randint(0, scale_))
	return img_[ymin : ymax, xmin : xmax , : ]


class CustomeImageDataset(torch.utils.data.Dataset):
	def __init__(self,
				 data_folder,
				 transform=None,
				 target_transform=None,
				 target_image_height=None,
				 target_image_width=None,
				 need_data_amount=None):
		super(CustomeImageDataset, self).__init__()
		self.data_folder = data_folder
		self.transform = transform
		self.target_transform = target_transform
		self.data_set = make_dataset(data_folder, need_data_amount=need_data_amount)
		self.idxs = [_ for _ in range(len(self.data_set))]
		self.target_image_height = target_image_height
		self.target_image_width = target_image_width

		self.transform_A = A.Compose([
			A.HorizontalFlip(p=0.5),
			A.RandomBrightnessContrast(p=0.2),
			A.RandomBrightness(p=0.2),
			A.RandomGamma(p=0.2),
			A.RandomShadow(p=0.2),
			A.Rotate(p=0.5)
		])

	def __getitem__(self, idx):
		img_path = self.data_set[idx]['img_path']
		int_label = self.data_set[idx]['int_label']
		bbox = self.data_set[idx]['bbox']
		# img_label = torch.eye(10)[img_label, :]
		# 将对应的标签数据转换为 one_hot 形式
		img = cv2.imread(img_path)
		xmin, ymin, xmax, ymax = bbox
		xmin = int(np.clip(xmin, 0, img.shape[1]-1))
		ymin = int(np.clip(ymin, 0, img.shape[0]-1))
		xmax = int(np.clip(xmax, 0, img.shape[1]-1))
		ymax = int(np.clip(ymax, 0, img.shape[0]-1))
		img = img[ymin:ymax , xmin:xmax , : ]
		if random.random()>0.5:
			img = img_agu_crop(img_=img)
		transformed = self.transform_A(image=img)
		transformed_img = transformed['image']
		image = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
		if self.target_image_height and self.target_image_width:
			img = Image.fromarray(image).resize((self.target_image_width, self.target_image_height),
											  Image.ANTIALIAS).convert('RGB')
		else:
			img = Image.fromarray(image).convert('RGB')
		# convert转换图像为 RGB 或者 L 形式
		if self.transform:
			img = self.transform(img)
		return img, int_label

	def __len__(self):
		return len(self.data_set)

	def __repr__(self):
		return "Data_set folder : {}\r\nData_set length : {}". \
			format(self.data_folder, self.__len__())
