# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/16 : 上午10:29
# Target  =
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from ResNet import resnet50
import global_settings as G_settings

model_ = resnet50(num_classes=G_settings.NUM_CLASSES)
if G_settings.GPU_FLAG:
	model_ = model_.to(G_settings.DEVICE)
model_.eval()
# 设置为前向推断模式

if os.access(G_settings.finetune_model_path, os.F_OK):
	ckpt = torch.load(f=G_settings.finetune_model_path,
					  map_location=G_settings.DEVICE)
	model_.load_state_dict(ckpt)
	print("INFO : load model from {}".format(
		G_settings.finetune_model_path
	))
else:
	raise RuntimeError("ERROR : could find trained model")

image_file_path = "./02113799_333.jpg"
img = Image.open(image_file_path).crop((201, 126, 334, 323)).resize((
	G_settings.TARGET_IMAGE_WIDTH, G_settings.TARGET_IMAGE_HEIGHT
), Image.ANTIALIAS).convert("RGB")
img.show()
img = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
		mean=G_settings.CIFAR100_TRAIN_MEAN,
		std=G_settings.CIFAR100_TRAIN_STD
	),
])(img)

img = img.unsqueeze(0)
output = model_(img.cuda().float())
total = output.shape[0]
_, pred_label = output.max(1)
print(pred_label)
# output = F.softmax(output, dim=1)
# output = output[0]
# output = output.cpu().detach().numpy()
# output = np.array(output)
# max_index = np.argmax(output)
# print(output)
# print(max_index)




