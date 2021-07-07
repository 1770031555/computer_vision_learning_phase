# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/6 : 下午2:27
# Target  =
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
	def __init__(self,
				 input_channels,
				 n1x1,
				 n3x3_reduce,
				 n3x3,
				 n5x5_reduce,
				 n5x5, pool_proj):
		super(Inception, self).__init__()
		self.b1 = nn.Sequential(
			nn.Conv2d(in_channels=input_channels, out_channels=n1x1, kernel_size=1),
			nn.BatchNorm2d(num_features=n1x1),
			nn.ReLU(inplace=True),
		)

		self.b2 = nn.Sequential(
			nn.Conv2d(in_channels=input_channels, out_channels=n3x3_reduce, kernel_size=1),
			nn.BatchNorm2d(n3x3_reduce),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=n3x3_reduce, out_channels=n3x3, kernel_size=3, padding=1),
			nn.BatchNorm2d(n3x3),
			nn.ReLU(inplace=True)
		)

		self.b3 = nn.Sequential(
			nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
			nn.BatchNorm2d(n5x5_reduce),
			nn.ReLU(inplace=True),
			nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
			nn.BatchNorm2d(n5x5, n5x5),
			nn.ReLU(inplace=True),
			nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
			nn.BatchNorm2d(n5x5),
			nn.ReLU(inplace=True)
		)

		self.b4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(in_channels=input_channels, out_channels=pool_proj, kernel_size=1),
			nn.BatchNorm2d(pool_proj),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)



class GoogleNet(nn.Module):
	def __init__(self, num_class=100):
		super(GoogleNet, self).__init__()
		self.prelayer = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			# num_features input this func , and got the mean and stddev
			nn.ReLU(inplace=True),
			# inplace 是是否进行原地操作，比如X = X +5就是对X进行原地操作，这样能够起到降低内存的作用
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=192),
			nn.ReLU(inplace=True),
		)

		self.a3 = Inception(input_channels=192,
							n1x1=64,
							n3x3_reduce=96,
							n3x3=128,
							n5x5_reduce=16,
							n5x5=32,
							pool_proj=32)
		self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
		self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
		self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
		self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
		self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

		self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
		self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.dropout = nn.Dropout(p=0.4)
		self.linear = nn.Linear(1024, num_class)

	def forward(self, x):
		x = self.prelayer(x)
		x = self.maxpool(x)
		x = self.a3(x)
		x = self.b3(x)

		x = self.maxpool(x)

		x = self.a4(x)
		x = self.b4(x)
		x = self.c4(x)
		x = self.d4(x)
		x = self.e4(x)

		x = self.maxpool(x)

		x = self.a5(x)
		x = self.b5(x)

		#"""It was found that a move from fully connected layers to
		#average pooling improved the top-1 accuracy by about 0.6%,
		#however the use of dropout remained essential even after
		#removing the fully connected layers."""
		x = self.avgpool(x)
		x = self.dropout(x)
		x = x.view(x.size()[0], -1)
		x = self.linear(x)
		return x

def googlenet():
	return GoogleNet()



