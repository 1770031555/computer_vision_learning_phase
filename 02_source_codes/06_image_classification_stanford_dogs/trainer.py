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
import os, glob, time, sys
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
from training_utils import WarmUpLR, most_recent_folder, best_acc_weights, last_epoch, most_recent_weights, set_learning_rate, get_acc
from loss.focal_loss import FocalLoss

TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/train_dataset.json"
TEST_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/test_dataset.json"
LABEL_2_INT_PATH = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/dataset/label2int.json"

train_data_dataset = CustomeImageDataset(
	data_folder=TRAIN_FOLDER_PATH,
	transform=transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(15),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=G_settings.CIFAR100_TRAIN_MEAN,
			std=G_settings.CIFAR100_TRAIN_STD
		),
	]),
	target_image_width=G_settings.TARGET_IMAGE_WIDTH,
	target_image_height=G_settings.TARGET_IMAGE_HEIGHT,
	# target_transform=transforms.Compose([
	# 	transforms.ToTensor()
	# ])
	need_data_amount=None
)
# print(train_data_dataset)
train_loader = DataLoader(train_data_dataset, batch_size=G_settings.BATCH_SIZE, shuffle=True)

test_data_dataset = CustomeImageDataset(
	data_folder=TEST_FOLDER_PATH,
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=G_settings.CIFAR100_TRAIN_MEAN,
							 std=G_settings.CIFAR100_TRAIN_STD)
	]),
	target_image_width=G_settings.TARGET_IMAGE_WIDTH,
	target_image_height=G_settings.TARGET_IMAGE_HEIGHT,
	need_data_amount=None
)
test_loader = DataLoader(test_data_dataset, batch_size=G_settings.BATCH_SIZE, shuffle=False)

# if __name__ == '__main__':
# 	A = next(iter(train_loader))
# 	print(A)
# 	time.sleep(1000)
# TODO : load dataset 加载对应的训练数据和预测数据

from ResNet import resnet50
net = resnet50(num_classes=G_settings.NUM_CLASSES)
if G_settings.GPU_FLAG:
	net = net.cuda()
if os.access(G_settings.finetune_model_path, os.F_OK):
	ckpt = torch.load(G_settings.finetune_model_path, map_location=G_settings.DEVICE)
	net.load_state_dict(ckpt)
	print("INFO Load finetune model from {}".format(
		G_settings.finetune_model_path
	))
# 加载对应的net模型
# TODO : 搭建对应的神经网络








#
# def train(epoch):
# 	start = time.time()
# 	net.train()
# 	# 当网络中存在DropOut或者BN层的时候，需要加上 model.train() 或者 model.eval()
# 	for batch_index, (img_path, img_label, img_target_label, img) in enumerate(train_loader):
# 		if G_settings.GPU_FLAG:
# 			# labels = img_target_label.cuda()
# 			labels = torch.tensor(img_target_label, dtype=torch.long, device=G_settings.DEVICE)
# 			images = img.cuda()
# 		else:
# 			labels = img_target_label
# 			images = img
#
# 		optimizer.zero_grad()
# 		outputs = net(images)
# 		loss = loss_function(outputs, labels)
# 		# pil_img = transforms.ToPILImage()(images[0].cpu()) * 255
# 		# pil_img = transforms.Compose([
# 		# 	transforms.Normalize(mean=-np.array(G_settings.CIFAR100_TRAIN_MEAN) / np.array(G_settings.CIFAR100_TRAIN_STD),
# 		# 						 std=1 / np.array(G_settings.CIFAR100_TRAIN_STD)),
# 		# 	transforms.ToPILImage()
# 		# ])(images[0].cpu())
# 		# pil_img.show()
# 		# print(img_label[0])
# 		# print(img_path[0])
# 		# print(outputs[0].cpu().argmax().tolist())
# 		# time.sleep(1000)
# 		# convert the tensor type image to PIL Image
# 		loss.backward()
# 		optimizer.step()
#
# 		n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
#
# 		last_layer = list(net.children())[-1]
# 		for name, para in last_layer.named_parameters():
# 			if 'weight' in name:
# 				writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
# 			if 'bias' in name:
# 				writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
#
# 		print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
# 			loss.item(),
# 			optimizer.param_groups[0]['lr'],
# 			epoch=epoch,
# 			trained_samples=batch_index * G_settings.BATCH_SIZE + len(images),
# 			total_samples=len(train_loader.dataset)
# 		))
#
# 		#update training loss for each iteration
# 		writer.add_scalar('Train/loss', loss.item(), n_iter)
#
# 		if epoch <= G_settings.WARMUP_training_phase:
# 			warmup_scheduler.step()
#
# 	for name, param in net.named_parameters():
# 		layer, attr = os.path.splitext(name)
# 		attr = attr[1:]
# 		writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
# 	finish = time.time()
# 	print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
#
# @torch.no_grad()
# def eval_training(epoch=0, tb=True):
# 	start_time = time.time()
# 	net.eval()
# 	test_loss = 0.0 # cost function error
# 	correct = 0.0
# 	for (img_path, img_label, img_target_label, img) in test_loader:
# 		if G_settings.GPU_FLAG:
# 			images = img.cuda()
# 			# labels = img_target_label.cuda()
# 			labels = torch.tensor(img_target_label, dtype=torch.long, device=G_settings.DEVICE)
# 		else:
# 			images = img
# 			labels = img_target_label
# 		outputs = net(images)
# 		loss = loss_function(outputs, labels)
# 		test_loss += loss.item()
# 		_, preds = outputs.max(1)
# 		correct += preds.eq(labels).sum()
# 	finish_time = time.time()
# 	if G_settings.GPU_FLAG:
# 		print('GPU INFO.....')
# 		print(torch.cuda.memory_summary(), end='')
# 	print('Evaluating Network.....')
# 	print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
# 		epoch,
# 		test_loss / len(test_loader.dataset),
# 		correct.float() / len(test_loader.dataset),
# 		finish_time - start_time
# 	))
# 	print()
#
# 	#add informations to tensorboard
# 	if tb:
# 		writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
# 		writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)
#
# 	return correct.float() / len(test_loader.dataset)
# # TODO : establish the train phase and test phase


if G_settings.LOSS_FUNCTION=="focal_loss":
	loss_function = FocalLoss(num_class=G_settings.NUM_CLASSES)
else:
	loss_function = torch.nn.CrossEntropyLoss()
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


checkpoint_path = os.path.join(G_settings.CHECKPOINT_PATH,
							   G_settings.NET_NAME,
							   G_settings.TIME_NOW)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}--{loss}.pth')



def trainer():
	if G_settings.LOG_FLAG:
		sys.stdout = G_settings.F_LOG
	print("INFO : length of train datasets {}".format(
		train_loader.__len__()
	))

	step = 0
	idx = 0

	# 变量初始化
	best_loss = np.inf
	loss_mean = 0. # 损失均值
	loss_idx = 0. # 损失计算计数器
	flag_change_lr_cnt = 0 # 学习率更新计数器
	init_lr = G_settings.LEARNING_RATE

	loss_val_list = []

	for epoch in range(0, G_settings.EPOCHS):
		print("\r\n epoch = {} ----->>>>>".format(epoch))
		net.train()
		# 学习率更新策略
		if loss_mean!=0.:
			if best_loss>(loss_mean/loss_idx):
				flag_change_lr_cnt = 0
				best_loss = (loss_mean / loss_idx)
			else:
				flag_change_lr_cnt += 1
				if flag_change_lr_cnt>10:
					init_lr = init_lr * G_settings.LEARNING_RATE_DECAY
					set_learning_rate(optimizer=optimizer, lr=init_lr)
					flag_change_lr_cnt = 0
		loss_mean = 0. # 损失均值
		loss_idx = 0. # 损失计算计数器
		for i, (imgs_, labels_) in enumerate(train_loader):
			if G_settings.GPU_FLAG:
				imgs_ = imgs_.cuda()
				labels_ = labels_.cuda()
			output = net(imgs_.float())
			loss = loss_function(output, labels_)
			loss_mean += loss.item()
			loss_idx += 1
			if i%100==0:
				loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
				print('INFO:{loc_time}-{net_name}-epoch [{current_epoch}/{all_epoch}]({current_step}/{all_step})\r\n'.format(
					loc_time=loc_time, net_name=G_settings.NET_NAME, current_epoch=epoch,
					all_epoch=G_settings.EPOCHS, current_step=i, all_step=train_loader.__len__()
				))
			if i%10==0:
				acc = get_acc(output=output, label=labels_)
				print("mean loss : %.6f , loss : %.6f  "%(loss_mean / loss_idx, loss.item()), \
					  " acc : %.4f"%acc, " lr : %.5f"%init_lr, " bs : ", G_settings.BATCH_SIZE, \
					  " img_size %s * %s"%(G_settings.TARGET_IMAGE_WIDTH, G_settings.TARGET_IMAGE_HEIGHT), "  best_loss : %.4f"%best_loss)
			# 计算梯度
			loss.backward()
			# 优化起对模型的参数更新
			optimizer.step()
			# 优化器梯度清零
			optimizer.zero_grad()
			step += 1

		# 每间隔多少个epoch之后自动保存一次模型
		if (epoch%G_settings.SAVE_EPOCH)==0 and (epoch>5):
			checkpoint_path_ = checkpoint_path.format(net=G_settings.NET_NAME, epoch=epoch, type="regular", loss="0.")
			torch.save(obj=net.state_dict(), f=checkpoint_path_)
		if epoch%G_settings.TEST_INTERNAL_train_phase==0:
			net.eval()
			loss_val = tester()
			if len(loss_val_list)>0:
				if loss_val<np.min(loss_val_list):
					checkpoint_path_ = checkpoint_path.format(net=G_settings.NET_NAME, epoch=epoch, type="best", loss=str(loss_val))
					torch.save(obj=net.state_dict(), f=checkpoint_path_)
			loss_val_list.append(loss_val)


def tester():
	print("\r\n------------------------>>> tester loss")
	loss_val = []
	with torch.no_grad():
		for i, (imgs_, labels_) in enumerate(test_loader):
			if G_settings.GPU_FLAG:
				imgs_ = imgs_.cuda()
				labels_ = labels_.cuda()
			output = net(imgs_)
			loss = loss_function(input=output, target=labels_)
			loss_val.append(loss.item())
		print("loss validation : {}".format(
			np.mean(loss_val)
		))
	return np.mean(loss_val)



if __name__ == '__main__':
	# 1.将LOG日志写入对应的地址
	writer = SummaryWriter(log_dir=os.path.join(
		G_settings.LOG_DIR, G_settings.NET_NAME, G_settings.TIME_NOW
	))
	# input_tensor = torch.Tensor(1, 3, G_settings.TARGET_IMAGE_HEIGHT, G_settings.TARGET_IMAGE_WIDTH)
	# if G_settings.GPU_FLAG:
	# 	input_tensor = input_tensor.cuda()
	# writer.add_graph(net, input_tensor)
	# writer.add_graph(net)

	# 开始训练，将测试函数嵌入到训练函数中
	trainer()

	# best_acc = 0.0
	# for epoch in range(1, G_settings.EPOCHS + 1):
	# 	if epoch>G_settings.WARMUP_training_phase:
	# 		train_scheduler.step(epoch)
	# 	# training procedure
	# 	net.train()
	# 	for batch_index, (images, labels) in enumerate(train_loader):
	# 		if epoch<=G_settings.WARMUP_training_phase:
	# 			warmup_scheduler.step()
	# 		images = images.cuda()
	# 		labels = labels.cuda()






	# step = 0
	# idx = 0
	# # 变量初始化
	# best_loss = np.inf
	# loss_mean = 0.
	# loss_idx = 0.
	# flag_change_lr_cnt = 0
	# init_lr = G_settings.LEARNING_RATE
	# epochs_loss_dict = {}
	#
	# for epoch in range(1, G_settings.EPOCHS + 1):
	# 	if epoch > G_settings.WARMUP_training_phase:
	# 		train_scheduler.step(epoch)
	# 	train(epoch)
	# 	acc = eval_training(epoch)
	#
	# 	# start to save best performance model after learning rate decay to 0.01
	# 	if epoch>G_settings.MILESTONES[1] and best_acc<acc:
	# 		weights_path = checkpoint_path.format(net=G_settings.NET_NAME,
	# 											  epoch=epoch,
	# 											  type='best')
	# 		print("INFO : saving weights file to {}".format(weights_path))
	# 		torch.save(net.state_dict(), weights_path)
	# 		best_acc = acc
	# 		continue
	#
	# 	if not epoch%G_settings.SAVE_EPOCH:
	# 		weights_path = checkpoint_path.format(net=G_settings.NET_NAME,
	# 											  epoch=epoch,
	# 											  type='regular')
	# 		print("INFO : saving weights file to {}".format(weights_path))
	# 		torch.save(net.state_dict(), weights_path)
	# writer.close()




