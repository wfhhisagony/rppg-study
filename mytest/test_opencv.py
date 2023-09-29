#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2023/9/14 10:40
# @Author  : lqh
# @python-version 3.10
# @File    : test_opencv.py
# @Software: PyCharm
"""

# import cv2
# import time
# cnt = 0
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# flag = cap.isOpened()
# ret, frame = cap.read()
# while flag:
#     ret, frame = cap.read()
#     cv2.imshow("ccc", frame)  # 按很短的时间不断显示1帧就成为了视频
#     k = cv2.waitKey(1) & 0xFF  # 等待按键的时间(加入这个才不会卡住)
#     cnt += 1
#     if cnt % 100 == 0:
#         print("width:",  cap.get(3) )
#         print("height:", cap.get(4) )
#     if k == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# import cv2
#
# image_path = r"C:\Users\shigua\Desktop\Snipaste_2023-09-14_09-13-02.png"  # 图片路径
# img = cv2.imread(image_path)
# cv2.imshow('window_name', img)  # 显示图片,[图片窗口名字，图片]
# # 等待一个按键
# cv2.waitKey(0)  # 无限期显示窗口，同时开始处理各种事件
#
# b, g, r = cv2.split(img)
# cv2.imshow("Blue_1", b)
# cv2.imshow("Green_1", g)
# cv2.imshow("Red_1", r)
# cv2.waitKey(0)  # 无限期显示窗口
# cv2.destroyAllWindows() # 删除所有窗口
# isOk = cv2.imwrite(r"./blue_img.jpg", b)  # 保存图片
# if isOk:
#     print("save success")


# import cv2
#
# # 创建一个窗口 名字叫做Window
# cv2.namedWindow('Window', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
#
#
# # 打开默认摄像头
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#
# # # 摄像头的IP地址,http://用户名：密码@IP地址：端口/
# # ip_camera_url = 'http://admin:admin@192.168.1.101:8081/'
# # # 创建一个VideoCapture
# # cap = cv2.VideoCapture(ip_camera_url)
#
# print('摄像头是否开启： {}'.format(cap.isOpened()))
#
# # 显示缓存数
# print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
# # 设置缓存区的大小
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#
# # 调节摄像头分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # 设置FPS
# print('setfps', cap.set(cv2.CAP_PROP_FPS, 25))
# print(cap.get(cv2.CAP_PROP_FPS))
#
# while (True):
#     # 逐帧捕获
#     ret, frame = cap.read()  # 第一个参数返回一个布尔值（True/False），代表有没有读取到图片；第二个参数表示截取到一帧的图片
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('Window', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 当一切结束后，释放VideoCapture对象
# cap.release()
# cv2.destroyAllWindows()

# cv2.VideoCapture.get(3)  视频流的帧宽度
# cv2.VideoCapture.get(4)  视频流的帧高度
# while flag:
#     ret, frame = cap.read()
#
#     cv2.imshow("Capture_Paizhao", frame)
#     k = cv2.waitKey(1) & 0xFF


# import cv2
#
# vc = cv2.VideoCapture('test.mp4')  # 视频路径
#
# # 获取原视频的基本属性
# fps = vc.get(cv2.CAP_PROP_FPS)
# width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
# frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)
# duration = frame_count / fps
#
# isVCOpened = vc.isOpened()
# ret = False
# if vc.isOpened():
#     # 检查是否打开正确，正确则读取第一帧
#     ret, frame = vc.read()
# while ret:
#     ret, frame = vc.read()
#     if ret is False:
#         break
#     else:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result', gray)
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break
#
# vc.release()
# cv2.destroyAllWindows()

import cv2

img_file_path = r"./lemonade.jpg"
img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
x1, y1 = 25, 25
x2, y2 = 50, 50
drawed_img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("drawed", drawed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()