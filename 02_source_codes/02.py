# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/6/30 : 下午1:49
# Target  = MNIST数据集实验
import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import trunc
from torchvision import datasets, transforms
# from torchvision.io import read_image
from torch.utils.data import DataLoader

print(torch.__version__)

from PIL import Image

BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置对应的超参数

TRAIN_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/01_mono/02_source_codes/monodepth2-master/01/data/Mnist-image/train"
TEST_FOLDER_PATH = "/home/liuyang/Desktop/02_CV/01_mono/02_source_codes/monodepth2-master/01/data/Mnist-image/test"
# 设置文件路径

def make_dataset(image_folder_path: str) -> list:
	r"""
	load dataset from image folder
	"""
	dataset = []
	if not os.path.exists(image_folder_path):
		raise RuntimeError('ERROR : {} does not exists'.format(image_folder_path))
	if not os.path.isdir(image_folder_path):
		raise RuntimeError('ERROR : {} is not one folder'.format(image_folder_path))

	image_folder_list = [os.path.join(image_folder_path, _) for _ in os.listdir(image_folder_path)]
	for img_folder in image_folder_list:
		for img_path in glob.glob(os.path.join(img_folder, "*.png")):
			img_file_name = os.path.basename(img_path)
			img_label = int(img_file_name[0])
			dataset.append(dict(
				path=img_path,
				label=img_label
			))
	return dataset[:1000]


class CustomeImageDataset(torch.utils.data.Dataset):
	def __init__(self, data_folder, transform=None, target_transform=None):
		super(CustomeImageDataset, self).__init__()
		self.data_folder = data_folder
		self.transform = transform
		self.target_transform = target_transform
		self.data_set = make_dataset(data_folder)
		self.idxs = [_ for _ in range(len(self.data_set))]

	def __getitem__(self, idx):
		img_path = self.data_set[idx]['path']
		img_label = int(self.data_set[idx]['label'])
		# img_label = torch.eye(10)[img_label, :]
		# 将对应的标签数据转换为 one_hot 形式
		img = Image.open(img_path).resize((28, 28),Image.ANTIALIAS).convert('L')
		# convert转换图像为 RGB 或者 L 形式
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			img_label = self.target_transform(img_label)
		return img_path, img_label, img

	def __len__(self):
		return len(self.data_set)

	def __repr__(self):
		return "Data_set folder : {}\r\nData_set length : {}".\
			format(self.data_folder,self.__len__())


train_data_dataset = CustomeImageDataset(
	data_folder=TRAIN_FOLDER_PATH,
	transform=transforms.Compose([
		transforms.ToTensor()
	]),
	# target_transform=transforms.Compose([
	# 	transforms.ToTensor()
	# ])
)
print(train_data_dataset)
train_loader = DataLoader(train_data_dataset, batch_size=6, shuffle=True)

test_data_dataset = CustomeImageDataset(
	data_folder=TEST_FOLDER_PATH,
	transform=transforms.Compose([
		transforms.ToTensor()
	]),
)
print(test_data_dataset)
test_loader = DataLoader(test_data_dataset, batch_size=6, shuffle=False)
# 加载对应的训练数据和预测数据

class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
		self.fc1 = nn.Linear(in_features=20 * 10 * 10, out_features=500)
		self.fc2 = nn.Linear(in_features=500, out_features=10)

	def forward(self, X):
		in_size = X.size(0)
		out = self.conv1(X)
		out = F.relu(out)
		out = F.max_pool2d(input=out, kernel_size=2, stride=2)
		out = self.conv2(out)
		out = F.relu(out)
		out = out.view(in_size, -1)
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.log_softmax(out, dim=1)
		return out


# 搭建对应的神经网络
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(
	params=model.parameters(),
	lr=LEARNING_RATE
)

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (img_path, label, data) in enumerate(train_loader):
		data = data.to(device)
		label = label.to(device, dtype=torch.int64)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, label)
		loss.backward()
		optimizer.step()
		if (batch_idx + 1) % 30 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0.
	correct = 0.
	with torch.no_grad():
		for img_path, label, data in test_loader:
			data, label = data.to(device), label.to(device, dtype=torch.int64)
			output = model(data)
			test_loss += F.nll_loss(output, label,
									reduction='sum').item()
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(label.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
	for epoch in range(1, EPOCHS + 1):
		train(model, DEVICE, train_loader, optimizer, epoch)
		test(model, DEVICE, test_loader)

else:
	...
