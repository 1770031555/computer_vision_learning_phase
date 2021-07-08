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
	# folder_list = [os.path.join(train_folder_PATH, _) for _ in os.listdir(train_folder_PATH)]
	means = [0, 0, 0]
	stdevs = [0, 0, 0]
	image_count = 0
	# for folder in tqdm(folder_list):
	image_file_list = glob.glob(os.path.join(train_folder_PATH, "*.jpg"))
	for image_file in tqdm(image_file_list):
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
# 	IMAGE_FOLDER = '/home/liuyang/99_保存数据/handpose_datasets_v1-2021-01-31/handpose_datasets_v1'
# 	means, stdevs = calculate_mean_std(IMAGE_FOLDER)
# 	print(means) [0.5279996  0.46980743 0.47215794]
# 	print(stdevs) [0.23443894 0.22663836 0.2268186 ]


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

# pil_img = transforms.ToPILImage()(images[0].cpu()) * 255
# pil_img = transforms.Compose([
# 	transforms.Normalize(mean=-np.array(G_settings.CIFAR100_TRAIN_MEAN) / np.array(G_settings.CIFAR100_TRAIN_STD),
# 						 std=1 / np.array(G_settings.CIFAR100_TRAIN_STD)),
# 	transforms.ToPILImage()
# ])(images[0].cpu())
# pil_img.show()
# print(img_label[0])
# print(img_path[0])
# print(outputs[0].cpu().argmax().tolist())
# time.sleep(1000)
# TOSO : convert the tensor type image to PIL Image





