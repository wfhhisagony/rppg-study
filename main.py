#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2023/9/13 16:58
# @Author  : lqh
# @python-version 3.10
# @File    : main2.py
# @Software: PyCharm
"""
import os
from ICA_POH import ICA_POH
from POS_WANG import POS_WANG
import cv2
import numpy as np
from read_video_and_cache_npy import read_and_cache, getVCInfo
from post_process import calculate_metric_per_video
# import tracemalloc

video_file_path = r"./video/one_min_video5.mp4"

dirStr, _ = os.path.splitext(video_file_path)
video_file_name = dirStr.split("\\")[-1]
video_file_name = video_file_name.split("/")[-1]
cache_fileName_faceClipFrames = f"cached_faceClipFrames_{video_file_name}.npy"
cache_fileName_faceBox = f"cached_faceBox_{video_file_name}.npy"


def unsupervised_predict(fs, face_clip_frames, calc_seconds=15, method="POS_WANG"):
    predict_hr_fft_all = []
    frame_num = face_clip_frames.shape[0]
    if method == "POS_WANG":
        print("use POS_WANG method")
        BVP = POS_WANG(face_clip_frames, fs)  # 具体算法函数见下面
        LP = 0.75
        HP = 3
    elif method == "ICA_PHO":
        print("use ICA_PHO method")
        BVP = ICA_POH(face_clip_frames, fs)
        LP = 0.75
        HP = 2.5
    else:
        print(f"method error. No such method: {method}")
        raise Exception

    window_frame_size = int(fs * calc_seconds)  # 假如说视频FS为35，则35 * 15，表示15s检测一次

    for i in range(0, frame_num, window_frame_size):
        BVP_window = BVP[i:i + window_frame_size]

        if len(BVP_window) < 9:
            print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
            continue
        if method=="POS_WANG":
            pre_fft_hr = calculate_metric_per_video(BVP_window, diff_flag=False, fs=fs, use_bandpass=True, LP=LP, HP=HP, hr_method="FFT")
        elif method=="ICA_PHO":
            pre_fft_hr = calculate_metric_per_video(BVP_window, diff_flag=False, fs=fs, use_bandpass=True, LP=LP, HP=HP,
                                                    hr_method="peak detection")
        else:
            raise Exception
        predict_hr_fft_all.append(pre_fft_hr)  # 获取预测的心率
    return predict_hr_fft_all

def prepocess(video_file_path, cache_fileName_faceClipFrames, cache_fileName_faceBox, isCached=False):
    if not isCached:
        face_region_median, resized_faceClip_frames = read_and_cache(video_file_path, cache_fileName_faceClipFrames,
                                                                     cache_fileName_faceBox)
    else:
        face_region_median, resized_faceClip_frames = load_npy(cache_fileName_faceClipFrames, cache_fileName_faceBox)
    return face_region_median, resized_faceClip_frames


def show_processed_video(video_file_path, face_region_median, resized_faceClip_frames, calc_seconds, method="POS_WANG"):
    vc = cv2.VideoCapture(video_file_path)
    isOpenedFlag, videoInfoDic = getVCInfo(vc)
    if not isOpenedFlag:
        print("video open failed")
        return
    FS = int(videoInfoDic['FS'])
    window_frame_size = int(FS * calc_seconds)
    predict_hr_fft_all = unsupervised_predict(FS, resized_faceClip_frames, calc_seconds, method)
    print(predict_hr_fft_all)
    resized_faceClip_frames = resized_faceClip_frames.astype(np.uint8)
    cnt = 0
    x, y, w, h = face_region_median
    ret, frame = vc.read()
    while ret:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if cnt // window_frame_size < len(predict_hr_fft_all):
            cv2.putText(frame, f"HR : {predict_hr_fft_all[cnt // window_frame_size]}",
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        cv2.imshow("video", frame)
        cv2.imshow("clip face", resized_faceClip_frames[cnt])
        ret, frame = vc.read()
        cnt += 1
        cv2.waitKey(1)
    print("press any key to continue")
    cv2.waitKey(0)
    vc.release()
    cv2.destroyAllWindows()
    return predict_hr_fft_all, videoInfoDic

def load_npy(cache_fileName_faceClipFrames, cache_fileName_faceBox):
    face_region_median = np.load(cache_fileName_faceBox)
    resized_faceClip_frames = np.load(cache_fileName_faceClipFrames)
    return face_region_median, resized_faceClip_frames


if __name__ == '__main__':

    method = "POS_WANG"  # POS_WANG  or ICA_PHO
    isCached = False
    calc_seconds = 10  # 每次将10s内的所有帧做一次算法检测
    # tracemalloc.start()

    face_region_median, resized_faceClip_frames = prepocess(video_file_path, cache_fileName_faceClipFrames,
                                                            cache_fileName_faceBox, isCached)
    show_processed_video(video_file_path, face_region_median, resized_faceClip_frames, calc_seconds, method)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')

    # print("[ Top 15 ]")
    # for stat in top_stats[:15]:
    #     print(stat)
