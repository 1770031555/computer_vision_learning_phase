# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
		super(BasicBlock, self).__init__()

		self.residual_function = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(num_features=out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=out_channels * BasicBlock.expansion),
		)

		self.shortcut = nn.Sequential()

		if stride!=1 or in_channels!=BasicBlock.expansion*out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(num_features=out_channels * BasicBlock.expansion),
			)

	def forward(self, x):
		return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, stride=1):
		super(BottleNeck, self).__init__()
		self.residual_function = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(num_features=out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels*BottleNeck.expansion, kernel_size=1, bias=False),
			nn.BatchNorm2d(num_features=out_channels * BottleNeck.expansion),
		)

		self.shortcut = nn.Sequential()

		if stride!=1 or in_channels!=out_channels*BottleNeck.expansion:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels*BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
				nn.BatchNorm2d(num_features=out_channels * BottleNeck.expansion),
			)
	def forward(self, x):
		nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class RESNET(nn.Module):
	def __init__(self, block, num_block, num_classes=100):
		super(RESNET, self).__init__()
		self.in_channels = 64

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(inplace=True),
		)
		self.conv2_x = self._make_layers(block, out_channels=64, num_blocks=num_block[0], stride=1)
		self.conv3_x = self._make_layers(block, 128, num_block[1], 2)
		self.conv4_x = self._make_layers(block, 256, num_block[2], 2)
		self.conv5_x = self._make_layers(block, 512, num_block[3], 2)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

	def _make_layers(self, block, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_channels, out_channels, stride))
			self.in_channels = out_channels * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		output = self.conv1(x)
		output = self.conv2_x(output)
		output = self.conv3_x(output)
		output = self.conv4_x(output)
		output = self.conv5_x(output)
		output = self.avg_pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)
		return output


def resnet34():
	return RESNET(BasicBlock, [3,4,6,3])








