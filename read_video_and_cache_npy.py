#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2023/9/16 19:51
# @Author  : lqh
# @python-version 3.10
# @File    : read_video_and_cache_npy.py
# @Software: PyCharm
"""
import cv2
import numpy as np
import os
from math import ceil
from yuface import detect

# haar_classifier_path = r"./haarcascade_frontalface_default.xml"  # use yuface instead


def getVCInfo(vc: cv2.VideoCapture):
    videoInfoDic = dict()
    isOpenedFlag = vc.isOpened()
    vc_width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    vc_height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vc_FS = vc.get(cv2.CAP_PROP_FPS)
    vc_frameCount = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    videoInfoDic['width'] = vc_width
    videoInfoDic['height'] = vc_height
    videoInfoDic['FS'] = vc_FS
    videoInfoDic['frameCount'] = vc_frameCount
    print(videoInfoDic)
    return isOpenedFlag, videoInfoDic


def face_detection(frame, use_larger_box=False, larger_box_coef=1.5, scaleFactor=None, minNeighbors=None):
    """Face detection on a single frame.
    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    # 注意路径

    confs, bboxes, landmarks = detect(frame, conf=0.5) # 使用yuface进行人脸检测

    if len(bboxes) < 1:
        # print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(bboxes) >= 2:
        face_box_coor = np.argmax(confs, axis=0)  # 找出置信度最大的下标
        face_box_coor = bboxes[face_box_coor]  # 获取最大的位置
        # print("Warning: More than one faces are detected(Only cropping the biggest one.)")  # 只检测一个人脸
    else:
        face_box_coor = bboxes[0]  # 只有一个，就取出该列表的这个元素
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])  # 向左上角缩减
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])  # 向左上角缩减
        face_box_coor[2] = larger_box_coef * face_box_coor[2]  # 向右下角扩张
        face_box_coor[3] = larger_box_coef * face_box_coor[3]  # 向右下角扩张
    return face_box_coor


def crop_face_resize(frames, exceptionSize, detection_freq=30, width=72, height=72, widthPercent=0.96, heightPercent=0.8):
    """Crop face and resize frames."""
    # Face Cropping
    num_dynamic_det = ceil(frames.shape[0] / detection_freq)  # 事实上detection_freq为1时检测最频繁
    face_region_all = []
    frame_num = frames.shape[0]
    for idx in range(num_dynamic_det):
        iidx = detection_freq * idx
        face_box_coor = face_detection(frames[iidx])
        while (face_box_coor[2] == exceptionSize[0] or face_box_coor[1] == exceptionSize[
            1]) and iidx != frame_num:  # 防止使用错误的矩形坐标
            if len(face_region_all) != 0:
                face_box_coor = face_region_all[-1]
            else:
                iidx += 1
                face_box_coor = face_detection(frames[iidx])
        face_region_all.append(face_box_coor)
    face_region_all = np.asarray(face_region_all, dtype='int')  # 这里获取的face_region_all其实是矩形坐标的数组，且是经过采样后的帧的矩形坐标
    face_region_median = np.median(face_region_all, axis=0).astype('int')
    # 只取框中心宽度90%
    to_delete_width = int(face_region_median[2] * ((1 - widthPercent) / 2))
    face_region_median[0] += to_delete_width
    face_region_median[2] = face_region_median[2] - 2 * to_delete_width

    to_delete_height = int(face_region_median[3] * ((1-heightPercent) / 2))
    face_region_median[1] += to_delete_height
    face_region_median[3] = face_region_median[3] - 2 * to_delete_height

    face_region = face_region_median
    # Frame Resizing
    resized_faceClip_frames = np.zeros((frames.shape[0], height, width, 3))  # 彩色图像
    for i in range(0, frames.shape[0]):
        frame = frames[i]

        frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_faceClip_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return face_region_median, resized_faceClip_frames


def read_and_cache(video_file_path, cache_fileName_faceClipFrames, cache_fileName_faceBox):
    vc = cv2.VideoCapture(video_file_path)
    vc.set(cv2.CAP_PROP_POS_MSEC, 0)  # 设置视频从最初开始播放
    isOpenFlag, videoInfoDic = getVCInfo(vc)
    frames = list()
    ret, frame = vc.read()
    print("press any key to continue")
    cv2.imshow("vc", frame)
    cv2.waitKey(0)  # 按任意键继续
    print("Go on")
    while ret:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)  # 要保证是RGB的顺序，这和算法有关
        frames.append(frame)
        ret, frame = vc.read()

    frames = np.asarray(frames)
    exceptionSize = (videoInfoDic['height'], videoInfoDic['width'])
    face_region_median, resized_faceClip_frames = crop_face_resize(frames, exceptionSize, int(videoInfoDic['FS']))
    np.save(cache_fileName_faceClipFrames, resized_faceClip_frames)
    np.save(cache_fileName_faceBox, face_region_median)
    print("finish cache data! press any key to continue")
    cv2.waitKey(0)
    print("Go on")
    vc.release()
    cv2.destroyAllWindows()
    return face_region_median, resized_faceClip_frames
