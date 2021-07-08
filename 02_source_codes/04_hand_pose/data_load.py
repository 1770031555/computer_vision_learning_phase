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

# 设置文件路径


def make_dataset(image_folder_path: str, need_data_amount=None) -> list:
	r"""
	load dataset from image folder
	"""
	generate_data_summary_list = []
	# file_list = []
	# hand_anno_list = []
	idx = 0
	for file in json.load(codecs.open(filename=image_folder_path, mode='r', encoding='utf-8')):
		if ".jpg" in file:
			img_path = os.path.join(image_folder_path, file)
			label_path = img_path.replace(".jpg", ".json")
			if not os.path.exists(label_path):
				print('----')
				continue

			with codecs.open(filename=label_path, mode='r', encoding='utf-8') as fr:
				hand_dict_ = json.load(fr)
			if len(hand_dict_)==0:
				continue
			hand_dict_ = hand_dict_['info']
			if len(hand_dict_)>0:
				for msg in hand_dict_:
					bbox = msg['bbox']
					pts = msg['pts']
					# file_list.append(img_path)
					# hand_anno_list.append((bbox, pts))
					generate_data_summary_list.append(dict(
						image_path = img_path,
						image_label = dict(
							bbox=bbox, pts=pts
						)
					))
					idx += 1
	with codecs.open(filename='generate_train_test_dataset.json',
					 mode='w', encoding='utf-8') as fw:
		json.dump(obj=generate_data_summary_list, fp=fw, ensure_ascii=False, indent=4)

	if need_data_amount == None:
		return generate_data_summary_list
	else:
		return generate_data_summary_list[:need_data_amount]


# if __name__ == '__main__':
# 	# 将手势识别的对应的数据集进行切割，分别装在对应的训练数据集和测试数据集
# 	# FOLDER_PATH = '/home/liuyang/99_保存数据/handpose_datasets_v1-2021-01-31/handpose_datasets_v1'
# 	# make_dataset(FOLDER_PATH)
# 	# 生成对应的总的数据的图像位置和对应label的标签之后，进行数据集的切分
# 	with codecs.open(filename='generate_train_test_dataset.json', mode='r', encoding='utf-8') as fr:
# 		D = json.load(fr)
# 	train_data_images_path = [J['image_path'] for I, J in enumerate(D) if I%10!=0]
# 	test_data_images_path = [J['image_path'] for I, J in enumerate(D) if I%10==0]
# 	with codecs.open(filename='train_data_image_path.json', mode='w', encoding='utf-8') as fw:
# 		json.dump(obj=train_data_images_path, fp=fw, ensure_ascii=False, indent=4)
# 	with codecs.open(filename='test_data_image_path.json', mode='w', encoding='utf-8') as fw:
# 		json.dump(obj=test_data_images_path, fp=fw, ensure_ascii=False, indent=4)
# else:
# 	...


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
		img_path = self.data_set[idx]['image_path']
		img_label = self.data_set[idx]['image_label']
		bbox, pts = img_label['bbox'], img_label['pts']
		x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		# img_target_label = self.data_set[idx]['img_target_label']
		# img_label = torch.eye(10)[img_label, :]
		# 将对应的标签数据转换为 one_hot 形式
		pts_ = []
		for i in range(21):
			x_, y_ = pts[str(i)]['x'], pts[str(i)]['y']
			x_ += x1
			y_ += y1
			pts_.append(x_/self.target_image_width)
			pts_.append(y_/self.target_image_height)
		pts_ = np.array(pts_)
		if self.target_image_height and self.target_image_width:
			img = Image.open(img_path).resize((self.target_image_width, self.target_image_height),
											  Image.ANTIALIAS).convert('RGB')
		else:
			img = Image.open(img_path).convert('RGB')
		# convert转换图像为 RGB 或者 L 形式
		if self.transform:
			img = self.transform(img)
		return img_path, pts_, img
		# if self.target_transform:
		# 	img_target_label = self.target_transform(img_target_label)
		# return img_path, img_label, img_target_label, img

	def __len__(self):
		return len(self.data_set)

	def __repr__(self):
		return "Data_set folder : {}\r\nData_set length : {}". \
			format(self.data_folder, self.__len__())
