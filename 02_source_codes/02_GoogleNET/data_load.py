# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/5 : 上午10:27
# Target  =
'''
Dataset utils for training and testing
'''
import os, time, json, codecs, datetime, glob
import numpy as np
from PIL import Image
from pprint import pprint
import torch

TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/dataset/cifar100/trainval"
TEST_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/dataset/cifar100/test"
LABELS_INDEX_PATH = "/home/liuyang/Desktop/02_CV/03_study/dataset/cifar100/labels_index.json"
# 设置文件路径
with codecs.open(filename=LABELS_INDEX_PATH, mode='r', encoding='utf-8') as fr:
	LABEL2INT = json.load(fr)


def make_dataset(image_folder_path: str, need_data_amount=None) -> list:
	r"""
	load dataset from image folder
	"""
	dataset = []
	dataset_labels_dict = dict()
	target_image_label = 0.
	if not os.path.exists(image_folder_path):
		raise RuntimeError('ERROR : {} does not exists'.format(image_folder_path))
	if not os.path.isdir(image_folder_path):
		raise RuntimeError('ERROR : {} is not one folder'.format(image_folder_path))

	image_folder_list = [os.path.join(image_folder_path, _) for _ in os.listdir(image_folder_path)]
	for img_folder in image_folder_list:
		img_label = os.path.basename(img_folder)
		for img_path in glob.glob(os.path.join(img_folder, "*.png")):
			img_target_label = float(LABEL2INT[img_label])
			dataset.append(dict(
				path=img_path,
				label=img_label,
				img_target_label=img_target_label
			))
			if img_label not in dataset_labels_dict.keys():
				dataset_labels_dict.update({img_label: 0})
			dataset_labels_dict[img_label] += 1
	if False:
		print('#####################################')
		print("load dataset from {}".format(image_folder_path))
		pprint(dataset_labels_dict)
		print("dataset amount : {}".format(np.array(list(dataset_labels_dict.values())).sum()))
		print('#####################################')

	if need_data_amount == None:
		return dataset
	else:
		return dataset[:need_data_amount]


if __name__ == '__main__':
	# 	# 计算对应有多少个类别
	# 	dataset, dataset_labels_dict = make_dataset(TRAIN_FOLDER_PATH)
	# 	dataset_labels_dict_KEYS = list(dataset_labels_dict.keys())
	# 	with codecs.open(filename='01.json', mode='w', encoding='utf-8') as fw:
	# 		json.dump(obj={J:I for I, J in enumerate(dataset_labels_dict_KEYS)},
	# 				  fp=fw, ensure_ascii=False, indent=4)
	# 	time.sleep(1000)
	# 计算训练数据的 MEAN STD
	# dataset = make_dataset(TRAIN_FOLDER_PATH)
	...
else:
	...


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

	def __getitem__(self, idx):
		img_path = self.data_set[idx]['path']
		img_label = self.data_set[idx]['label']
		img_target_label = self.data_set[idx]['img_target_label']
		# img_label = torch.eye(10)[img_label, :]
		# 将对应的标签数据转换为 one_hot 形式
		if self.target_image_height and self.target_image_width:
			img = Image.open(img_path).resize((self.target_image_width, self.target_image_height),
											  Image.ANTIALIAS).convert('RGB')
		else:
			img = Image.open(img_path).convert('RGB')
		# convert转换图像为 RGB 或者 L 形式
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			img_target_label = self.target_transform(img_target_label)
		return img_path, img_label, img_target_label, img

	def __len__(self):
		return len(self.data_set)

	def __repr__(self):
		return "Data_set folder : {}\r\nData_set length : {}". \
			format(self.data_folder, self.__len__())
