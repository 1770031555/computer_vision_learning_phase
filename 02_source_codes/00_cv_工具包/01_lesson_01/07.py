# -*- coding:utf-8 -*-
# Author  = liuyang
# Time    = 2021/7/15 : 上午10:55
# Target  =
import cv2

cap = cv2.VideoCapture("00_data/1.mp4")
while True:
	ret, img = cap.read()
	# if ret==True:
	# 	cv2.namedWindow("video", 0)
	# 	cv2.imshow("video", img)
	# 	key = cv2.waitKey(33)
	# 	if key==27:
	# 		break
	# else:
	# 	break
	#
	# cap.release()
	# cv2.destroyAllWindows()

