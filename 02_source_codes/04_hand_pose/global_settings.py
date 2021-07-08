# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/5 : 上午10:28
# Target  =
import torch
import os, time, datetime

BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-2
LEARNING_MOMENTUM = 0.9
DROPOUT_RATIO = 0.2
WARMUP_training_phase = 1
CIFAR100_TRAIN_MEAN = [0.5279996,  0.46980743, 0.47215794]
CIFAR100_TRAIN_STD = [0.23443894, 0.22663836, 0.2268186 ]
GPU_FLAG = True
if GPU_FLAG:
	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
	DEVICE = torch.device("cpu")
TARGET_IMAGE_HEIGHT = 256
TARGET_IMAGE_WIDTH = 256
MILESTONES = [60, 120, 160]
ITER_PER_EPOCH = 2000
LOG_DIR = "runs"
#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
CURRENT_FILE_PATH = os.path.abspath(".")
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, LOG_DIR)):
	os.mkdir(os.path.join(CURRENT_FILE_PATH, LOG_DIR))
FINETUNE_MODEL_PATH = "/path/to/finetune/model/path"
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
CHECKPOINT_PATH = 'checkpoint'
NET_NAME = "ResNET"
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME)):
	os.makedirs(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME))
# 设置对应的超参数




