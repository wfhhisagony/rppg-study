#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2023/9/20 17:50
# @Author  : lqh
# @python-version 3.10
# @File    : test_yuface.py
# @Software: PyCharm
"""
import cv2
from yuface import detect
img = cv2.imread('face2.jpg')
confs, bboxes, landmarks = detect(img, conf=0.5)
for conf, bbox, landmark in zip(confs, bboxes, landmarks):
    # 框出人脸
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
    # 标出置信度
    cv2.putText(img, str(conf), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 标出人脸的5个特征
    for i in range(5):
        cv2.circle(img, (landmark[2*i], landmark[2*i+1]), 2, (0, 255, 0), 1)
cv2.imshow("cropped face", img)

# 图img
# img = cv2.imread('face2.jpg')

# img: numpy.ndarray, shape=(H, W, 3), dtype=uint8, BGR
# conf: float, confidence threshold, default=0.5, range=[0.0, 1.0]
# confs, bboxes, landmarks = detect(img, conf=0.5)

# confs: numpy.ndarray, shape=(N,), dtype=uint16, confidence
# bboxes: numpy.ndarray, shape=(N, 4), dtype=uint16, bounding box (XYWH)
# landmarks: numpy.ndarray, shape=(N, 10), dtype=uint16, landmarks (XYXYXYXYXY)
# for conf, bbox, landmark in zip(confs, bboxes, landmarks):
#     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
#     cv2.putText(img, str(conf), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#     for i in range(5):
#         cv2.circle(img, (landmark[2*i], landmark[2*i+1]), 2, (0, 255, 0), 1)
# cv2.imwrite('result.jpg', img)

# 视频
# video_file_path = r"./video/one_min_video.mp4"
# vc = cv2.VideoCapture(video_file_path)
# ret, frame = vc.read()
#
# while ret:
#     cv2.waitKey(33)
#     confs, bboxes, landmarks = detect(frame, conf=0.5)
#     for conf, bbox, landmark in zip(confs, bboxes, landmarks):
#         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
#     cv2.imshow("vc", frame)
#     ret, frame = vc.read()

# def crop_face_resize(ori, fs, width=72, height=72):
#     confs, bboxes, landmarks = detect(ori, conf=0.5)
#     resized_faceClip_frame = None
#     if len(bboxes) > 0:
#         bbox = bboxes[0]  # 只取一个
#         cv2.rectangle(ori, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
#         resized_faceClip_frame = ori[max(bbox[1], 0):min(bbox[1] + bbox[3], ori.shape[0]),
#                                  max(bbox[0], 0):min(bbox[0] + bbox[2], ori.shape[1])]
#         resized_faceClip_frame = cv2.resize(resized_faceClip_frame, (width, height), interpolation=cv2.INTER_AREA)
#
#     return resized_faceClip_frame
#
#
# # 实时摄像头  截取面部图像
# vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# isOpenedFlag = vc.isOpened()
# if isOpenedFlag:
#     ret, frame = vc.read()
#     while ret:
#
#         cv2.waitKey(33)
#         # frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)  # 要保证是RGB的顺序，这和算法有关
#         resized_faceClip_frames = crop_face_resize(frame, fs=30)
#         resized_faceClip_frames = resized_faceClip_frames.astype(np.uint8)
#         cv2.imshow("vc", frame)
#         cv2.imshow("crop face", resized_faceClip_frames)
#         ret, frame = vc.read()
#
# vc.release()
# cv2.destroyAllWindows()
