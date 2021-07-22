# -*- coding:utf-8 -*-
"""
utils for training pytorch
"""
import os, codecs, json, time, datetime, re
import os, json, codecs, time, glob
from PIL import Image
from pprint import pprint
import numpy as np
from tqdm import tqdm
import xml.etree.cElementTree as ET

def best_acc_weights(weights_folder):
	'''
	return the best acc .pth file in given folder, if no best
	acc weights found , return empty string
	'''
	files = os.listdir(weights_folder)
	if len(files)==0:
		return ''
	regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
	best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
	if len(best_files) == 0:
		return ''

	best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
	return best_files[-1]


# TODO 1 求训练数据的 MEAN 和 STD
# Image.open 出来的图像的通道是 RGB 的顺序 ， 这和OpenCV读取的通道BGR的顺序刚好相反
def calculate_mean_std(train_folder_PATH):
	folder_list = [os.path.join(train_folder_PATH, _) for _ in os.listdir(train_folder_PATH)]
	means = [0, 0, 0]
	stdevs = [0, 0, 0]
	image_count = 0
	for folder in tqdm(folder_list):
		image_file_list = glob.glob(os.path.join(folder, "*.jpg"))
		for image_file in image_file_list:
			img = np.array(Image.open(image_file)) / 255.
			# resize((target_image_width, target_image_height))
			for i in range(3):
				means[i] += img[:, :, i].mean()
				stdevs[i] += img[:, :, i].std()
			image_count += 1
	means = np.asarray(means) / image_count
	stdevs = np.asarray(stdevs) / image_count
	return means, stdevs

# if __name__ == '__main__':
# 	TRAIN_FOLDER_PATH = "/home/liuyang/99_保存数据/01_计算机视觉数据集相关/03_数据分类_斯坦福狗/01_斯坦福狗数据集/datasets/train"
# 	mean, stdevs = calculate_mean_std(TRAIN_FOLDER_PATH)
# 	print(mean) [0.47652145 0.45168954 0.39113665]
# 	print(stdevs) [0.23417899 0.22931373 0.22738224]



from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
	r'''
	warm up training learning rate scheduler
	'''
	def __init__(self, optimizer, total_iters, last_epoch=-1):
		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
	# get subfolders in net_weights
	folders = os.listdir(net_weights)
	# filter out empty folders
	folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
	if len(folders) == 0:
		return ''
	# sort folders by folder created time
	folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
	return folders[-1]

def most_recent_weights(weights_folder):
	"""
		return most recent created weights file
		if folder is empty return empty string
	"""
	weight_files = os.listdir(weights_folder)
	if len(weights_folder) == 0:
		return ''

	regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

	# sort files by epoch
	weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

	return weight_files[-1]


def last_epoch(weights_folder):
	weight_file = most_recent_weights(weights_folder)
	if not weight_file:
		raise Exception('no recent weights were found')
	resume_epoch = int(weight_file.split('-')[1])

	return resume_epoch


def generate_json_file_from_folder(train_data_folder, test_data_folder):
	r'''
	go through all the files in this folder and generate train img path and labels
	'''
	train_dataset = []
	test_dataset = []
	label_dict = dict()

	folder_list = [os.path.join(train_data_folder, _) for _ in os.listdir(train_data_folder)]
	for folder in folder_list:
		folder_base_name = os.path.basename(folder)
		label_int = folder_base_name.split('-')[0]
		label_string = folder_base_name.split('-')[1]
		if label_int not in label_dict.keys():
			label_dict.update({label_int : label_string})
		else:
			if label_string != label_dict[label_int]:
				print('ERROR')
		img_files_list = glob.glob(os.path.join(folder, "*.jpg"))
		for img_file in img_files_list:
			xml_file = img_file.replace("jpg", "xml")
			if os.access(xml_file, os.F_OK):
				train_dataset.append([img_file, xml_file])

	folder_list = [os.path.join(test_data_folder, _) for _ in os.listdir(train_data_folder)]
	for folder in folder_list:
		folder_base_name = os.path.basename(folder)
		label_int = folder_base_name.split('-')[0]
		label_string = folder_base_name.split('-')[1]
		if label_int not in label_dict.keys():
			label_dict.update({label_int : label_string})
		else:
			if label_string != label_dict[label_int]:
				print('ERROR')
		img_files_list = glob.glob(os.path.join(folder, "*.jpg"))
		for img_file in img_files_list:
			xml_file = img_file.replace("jpg", "xml")
			if os.access(xml_file, os.F_OK):
				test_dataset.append([img_file, xml_file])

	with codecs.open(filename="dataset/label2int.json",
					 mode='w', encoding='utf-8') as fw:
		json.dump(obj=label_dict, fp=fw, ensure_ascii=False, indent=4)

	return train_dataset, test_dataset

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

def check_name_xml():
	with codecs.open(filename='dataset/label2int.json', mode='r', encoding='utf-8') as fr:
		label_string_list = list(json.load(fr).values())
	all_data_set = []
	with codecs.open(filename='dataset/train_dataset.json', mode='r', encoding='utf-8') as fr:
		train_data = json.load(fr)
	with codecs.open(filename='dataset/test_dataset.json', mode='r', encoding='utf-8') as fr:
		test_data = json.load(fr)
	all_data_set.extend(train_data)
	all_data_set.extend(test_data)
	for d in all_data_set:
		xml_path = d[1]
		list_x = get_xml_msg(xml_file_path=xml_path)
		for [name, xyxy] in list_x:
			if name not in label_string_list:
				print(name)

# if __name__ == '__main__':
# 	TRAIN_FOLDER_PATH = "/home/liuyang/99_保存数据/01_计算机视觉数据集相关/03_数据分类_斯坦福狗/01_斯坦福狗数据集/datasets/train"
# 	TEST_FOLDER_PATH = "/home/liuyang/99_保存数据/01_计算机视觉数据集相关/03_数据分类_斯坦福狗/01_斯坦福狗数据集/datasets/test"
#
# 	num_classes_1 = len(os.listdir(TRAIN_FOLDER_PATH))
# 	with codecs.open(filename='dataset/label2int.json', mode='r', encoding='utf-8') as fr:
# 		D = json.load(fr)
# 	num_classes_2 = len(D)
# 	print(num_classes_1)
# 	print(num_classes_2)
#
# 	train_dataset, test_dataset = generate_json_file_from_folder(TRAIN_FOLDER_PATH, TEST_FOLDER_PATH)
# 	# print(train_dataset)
# 	# print('----')
# 	# print(test_dataset)
# 	# print(len(train_dataset))
# 	# print(len(test_dataset))
#
# 	with codecs.open(filename='dataset/train_dataset.json',
# 					 mode='w', encoding='utf-8') as fw:
# 		json.dump(obj=train_dataset, fp=fw, ensure_ascii=False, indent=4)
# 	with codecs.open(filename='dataset/test_dataset.json',
# 					 mode='w', encoding='utf-8') as fw:
# 		json.dump(obj=test_dataset, fp=fw, ensure_ascii=False, indent=4)
#

def set_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def get_acc(output, label):
	total = output.shape[0]
	_, pred_label = output.max(1)
	num_correct = (pred_label == label).sum().item()
	return num_correct / float(total)






