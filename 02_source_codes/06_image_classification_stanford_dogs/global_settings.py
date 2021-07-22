# -*- coding:utf-8 -*-
"""
global settings for trainer and inference
"""
import codecs
import torch
import os, time, datetime
import argparse

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
NET_NAME = "ResNET"

NUM_CLASSES = 120
BATCH_SIZE = 10
EPOCHS = 200
ITER_PER_EPOCH = 2000
SAVE_EPOCH = 4
TEST_INTERNAL_train_phase = 5
# 每训练多少轮之后进行一次测试

LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 0.96
LEARNING_MOMENTUM = 0.9
DROPOUT_RATIO = 0.2
WARMUP_training_phase = 2
LOSS_FUNCTION = "focal_loss"

CIFAR100_TRAIN_MEAN = [0.47652145, 0.45168954, 0.39113665]
CIFAR100_TRAIN_STD = [0.23417899, 0.22931373, 0.22738224]
TARGET_IMAGE_WIDTH = 256
TARGET_IMAGE_HEIGHT = 256


GPU_FLAG = True
if GPU_FLAG:
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	DEVICE = torch.device("cpu")

MILESTONES = [10, 15, 18]

LOG_DIR = "runs"
CURRENT_FILE_PATH = os.path.abspath(".")
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, LOG_DIR, TIME_NOW)):
	os.makedirs(os.path.join(CURRENT_FILE_PATH, LOG_DIR, TIME_NOW))

CHECKPOINT_PATH = 'checkpoint'
if not os.path.exists(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME,TIME_NOW)):
	os.makedirs(os.path.join(CURRENT_FILE_PATH, CHECKPOINT_PATH, NET_NAME,TIME_NOW))

finetune_model_path = "/home/liuyang/Desktop/02_CV/03_study/05_image_classification/01_stanford_dogs/checkpoint/ResNET/Wednesday_21_July_2021_18h_36m_17s/ResNET-196-regular--0..pth"
if not os.access(finetune_model_path, os.F_OK):
	print("Could not load fine tune model from {}".format(
		finetune_model_path
	))
# 设置对应的超参数

RESUME_FLAG = False

LOG_FLAG = False
if LOG_FLAG:
	if not os.access(os.path.join(CURRENT_FILE_PATH, "log"), os.F_OK):
		os.makedirs(os.path.join(CURRENT_FILE_PATH, "log"))
	LOG_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "log", NET_NAME + TIME_NOW + ".log")
	F_LOG = codecs.open(filename=LOG_FILE_PATH, mode="a+", encoding='utf-8')


