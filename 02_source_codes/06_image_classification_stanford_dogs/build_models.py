# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/5 : 上午10:24
# Target  =
import torch
import torch.nn as nn

from global_settings import DROPOUT_RATIO

net_cfg = {
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class VGG16(torch.nn.Module):
	def __init__(self, features, num_class=100):
		super(VGG16, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(512, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(DROPOUT_RATIO),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(DROPOUT_RATIO),
			nn.Linear(4096, num_class)
		)

	def forward(self, X):
		output = self.features(X)
		output = output.view(output.size()[0], -1)
		output = self.classifier(output)
		return output


def make_layers(cfg, batch_norm=False):
	layers = []
	input_channels = 3
	for l in cfg:
		if l == "M":
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			continue
		layers += [nn.Conv2d(in_channels=input_channels,
							 out_channels=l,
							 kernel_size=3,
							 padding=1)]
		if batch_norm:
			layers += [nn.BatchNorm2d(l)]
		layers += [nn.ReLU(inplace=True)]
		input_channels = l
	return nn.Sequential(*layers)


def vgg16_bn():
	return VGG16(make_layers(cfg=net_cfg['VGG16'], batch_norm=True))
