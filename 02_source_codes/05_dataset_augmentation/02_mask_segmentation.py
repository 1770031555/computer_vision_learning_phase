# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/9 : 下午4:41
# Target  =
import albumentations as A
import cv2

transform = A.Compose([
	A.RandomCrop(width=256, height=256),
	A.HorizontalFlip(p=0.5),
	A.RandomBrightnessContrast(p=0.2),
])

image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



mask = cv2.imread("/path/to/mask.png")
transformed = transform(image=image, mask=mask)
transformed_image = transformed['image']
transformed_mask = transformed['mask']

# For instance segmentation, you sometimes need to read multiple masks per image. Then
# you create a list that contains all the masks.
mask_1 = cv2.imread("/path/to/mask_1.png")
mask_2 = cv2.imread("/path/to/mask_2.png")
mask_3 = cv2.imread("/path/to/mask_3.png")
masks = [mask_1, mask_2, mask_3]
transformed = transform(image=image, masks=masks)
transformed_image = transformed['image']
transformed_masks = transformed['masks']


