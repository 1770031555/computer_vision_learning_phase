# -*- coding:utf-8 -*-
# VGG network for training cifar-100 dataset
'''
The main script for training network
Model subclass from models.py
utils subclass from training_utils.py
dataloader from data_load.py
'''
import codecs, datetime
import json
import os, glob, time
import numpy as np
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch import trunc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import global_settings as G_settings
from data_load import CustomeImageDataset
from training_utils import WarmUpLR, most_recent_folder, best_acc_weights, last_epoch, most_recent_weights

TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/02_hand_pose/02_my_codes/dataset/train_data_image_path.json"
TEST_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/02_hand_pose/02_my_codes/dataset/test_data_image_path.json"
TRAIN_FOLDER_PATH_SMALL = "/home/liuyang/Desktop/02_CV/03_study/02_hand_pose/02_my_codes/handpose_datasets"


train_data_dataset = CustomeImageDataset(
	data_folder=TRAIN_FOLDER_PATH,
	transform=transforms.Compose([
		# transforms.RandomCrop(32, padding=4),
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomRotation(15),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=G_settings.CIFAR100_TRAIN_MEAN,
			std=G_settings.CIFAR100_TRAIN_STD
		),
	]),
	target_image_width=G_settings.TARGET_IMAGE_WIDTH,
	target_image_height=G_settings.TARGET_IMAGE_HEIGHT
	# target_transform=transforms.Compose([
	# 	transforms.ToTensor()
	# ])
)
# print(train_data_dataset)
train_loader = DataLoader(train_data_dataset, batch_size=6, shuffle=True)

test_data_dataset = CustomeImageDataset(
	data_folder=TEST_FOLDER_PATH,
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=G_settings.CIFAR100_TRAIN_MEAN,
							 std=G_settings.CIFAR100_TRAIN_STD)
	]),
	target_image_width=G_settings.TARGET_IMAGE_WIDTH,
	target_image_height=G_settings.TARGET_IMAGE_HEIGHT,
)
test_loader = DataLoader(test_data_dataset, batch_size=6, shuffle=False)

# if __name__ == '__main__':
# 	img_path, labels, img = next(iter(train_loader))
# 	# print(img)
# 	print(labels)
# 	print(len(labels))
# 	print(labels[0])
# 	print(len(labels[0]))
# 	time.sleep(1000)
# TODO : load dataset 加载对应的训练数据和预测数据

from ResNet import resnet50
net = resnet50(num_classes=42, img_size=256,dropout_factor = 0.3)
if G_settings.GPU_FLAG:
	net = net.cuda()
# TODO : 搭建对应的神经网络

def train(epoch):
	start = time.time()
	net.train()
	# 当网络中存在DropOut或者BN层的时候，需要加上 model.train() 或者 model.eval()
	for batch_index, (img_path, labels, img) in enumerate(train_loader):
		if G_settings.GPU_FLAG:
			labels = torch.tensor(labels, dtype=torch.float, device=G_settings.DEVICE)
			images = img.cuda()
		else:
			labels = labels
			images = img

		optimizer.zero_grad()
		outputs = net(images)
		loss = loss_function(outputs, labels)

		loss.backward()
		optimizer.step()

		n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

		last_layer = list(net.children())[-1]
		for name, para in last_layer.named_parameters():
			if 'weight' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
			if 'bias' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
		if batch_index%10==0:
			print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
				loss.item(),
				optimizer.param_groups[0]['lr'],
				epoch=epoch,
				trained_samples=batch_index * G_settings.BATCH_SIZE + len(images),
				total_samples=len(train_loader.dataset)
			))

		#update training loss for each iteration
		writer.add_scalar('Train/loss', loss.item(), n_iter)

		if epoch <= G_settings.WARMUP_training_phase:
			warmup_scheduler.step()

	for name, param in net.named_parameters():
		layer, attr = os.path.splitext(name)
		attr = attr[1:]
		writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
	finish = time.time()
	print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
	start_time = time.time()
	net.eval()
	test_loss = 0.0 # cost function error
	correct = 0.0
	for (img_path, labels, img) in test_loader:
		if G_settings.GPU_FLAG:
			images = img.cuda()
			# labels = img_target_label.cuda()
			labels = torch.tensor(labels, dtype=torch.float, device=G_settings.DEVICE)
		else:
			images = img
			labels = labels
		outputs = net(images)
		loss = loss_function(outputs, labels.float())
		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()
	finish_time = time.time()
	if G_settings.GPU_FLAG:
		print('GPU INFO.....')
		print(torch.cuda.memory_summary(), end='')
	print('Evaluating Network.....')
	print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
		epoch,
		test_loss / len(test_loader.dataset),
		correct.float() / len(test_loader.dataset),
		finish_time - start_time
	))
	print()

	#add informations to tensorboard
	if tb:
		writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
		writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

	return correct.float() / len(test_loader.dataset)
# TODO : establish the train phase and test phase



loss_function = torch.nn.MSELoss()
optimizer = optim.SGD(
	params=net.parameters(),
	lr=G_settings.LEARNING_RATE,
	momentum=G_settings.LEARNING_MOMENTUM,
	weight_decay=5e-4
)
# weight_decay 权值衰减的目的是为了防止过拟合，weight_decay是放在正则项regularization
# 前面的系数，正则项一般表示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响
# momentum：加速收敛的作用
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
												 milestones=G_settings.MILESTONES,
												 gamma=0.2)
iter_per_epoch = G_settings.ITER_PER_EPOCH
warmup_scheduler = WarmUpLR(optimizer=optimizer,
							total_iters=iter_per_epoch*1)
# 只在第一个epoch内进行学习率的warmup

if os.access(G_settings.FINETUNE_MODEL_PATH, os.F_OK):
	torch_load = torch.load(f=G_settings.FINETUNE_MODEL_PATH,
							map_location=G_settings.DEVICE)
	net.load_state_dict(torch_load)
	print("INFO : load fine-tune model from {}".format(G_settings.FINETUNE_MODEL_PATH))
else:
	print("ERROR : could not load fine-tune model from {}".format(
		G_settings.FINETUNE_MODEL_PATH
	))
# TODO : load model from findtune model


checkpoint_path = os.path.join(G_settings.CHECKPOINT_PATH,
							   G_settings.NET_NAME,
							   G_settings.TIME_NOW)
if not os.path.exists(checkpoint_path):
	os.mkdir(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


writer = SummaryWriter(log_dir=os.path.join(
	G_settings.LOG_DIR, G_settings.NET_NAME, G_settings.TIME_NOW
))
input_tensor = torch.Tensor(1, 3, 256, 256)
if G_settings.GPU_FLAG:
	input_tensor = input_tensor.cuda()
writer.add_graph(net, input_tensor)

best_acc = 0.0


if __name__ == '__main__':
	for epoch in range(1, G_settings.EPOCHS + 1):
		if epoch > G_settings.WARMUP_training_phase:
			train_scheduler.step(epoch)

		train(epoch)
		acc = eval_training(epoch)

		# start to save best performance model after learning rate decay to 0.01
		if epoch>G_settings.MILESTONES[1] and best_acc<acc:
			weights_path = checkpoint_path.format(net=G_settings.NET_NAME,
												  epoch=epoch,
												  type='best')
			print("INFO : saving weights file to {}".format(weights_path))
			torch.save(net.state_dict(), weights_path)
			best_acc = acc
			continue

		if not epoch%G_settings.SAVE_EPOCH:
			weights_path = checkpoint_path.format(net=G_settings.NET_NAME,
												  epoch=epoch,
												  type='regular')
			print("INFO : saving weights file to {}".format(weights_path))
			torch.save(net.state_dict(), weights_path)
	writer.close()




