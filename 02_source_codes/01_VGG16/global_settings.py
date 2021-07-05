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
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
GPU_FLAG = True
if GPU_FLAG:
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	DEVICE = torch.device("cpu")

MILESTONES = [60, 120, 160]
ITER_PER_EPOCH = 2000
LOG_DIR = "runs"
#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
CURRENT_FILE_PATH = os.path.abspath(".")
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, LOG_DIR)):
	os.mkdir(os.path.join(CURRENT_FILE_PATH, LOG_DIR))
RESUME_FLAG = False
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
CHECKPOINT_PATH = 'checkpoint'
NET_NAME = "VGG16"
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME)):
	os.makedirs(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME))
# 设置对应的超参数




