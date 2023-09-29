#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2023/9/23 17:04
# @Author  : lqh
# @python-version 3.10
# @File    : settingWindow.py
# @Software: PyCharm
"""
import settingWindowDesign

from PyQt5 import QtCore
import sys
from functools import partial
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from main import prepocess, show_processed_video
import os


class MyWorkThread(QObject):
    result_ready = pyqtSignal(str)

    def __init__(self, mainWindow=None):
        super().__init__()
        self.mainWindow = mainWindow

    def do_work(self):
        dirStr, _ = os.path.splitext(self.mainWindow.str_lineEdit_path)
        video_file_name = dirStr.split("\\")[-1]
        video_file_name = video_file_name.split("/")[-1]
        cache_fileName_faceClipFrames = f"cached_faceClipFrames_{video_file_name}.npy"
        cache_fileName_faceBox = f"cached_faceBox_{video_file_name}.npy"
        face_region_median, resized_faceClip_frames = prepocess(self.mainWindow.str_lineEdit_path,
                                                                cache_fileName_faceClipFrames,
                                                                cache_fileName_faceBox,
                                                                self.mainWindow.bool_radioButton_isCached)
        predict_hr_fft_all, videoInfoDic = show_processed_video(self.mainWindow.str_lineEdit_path, face_region_median,
                                                                resized_faceClip_frames,
                                                                self.mainWindow.int_spinBox_window_size,
                                                                self.mainWindow.str_comboBox_method)
        resStr =  (f"video_info:{str(videoInfoDic)}\n"
                   f"HR_list:{str(predict_hr_fft_all)}\n")
        self.result_ready.emit(resStr)


class My_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = settingWindowDesign.Ui_MainWindow()
        self.ui.setupUi(self)

        self.str_comboBox_method = None
        self.int_spinBox_window_size = 0
        self.str_lineEdit_path = ""
        self.bool_radioButton_isCached = False

        self.thread = QThread()
        # 实例化做事的类
        self.work_thread = MyWorkThread(self)
        # moveToThread方法把实例化线程移到Thread管理
        self.work_thread.moveToThread(self.thread)
        # 线程开始执行之前，从相关线程发射信号
        self.thread.started.connect(self.work_thread.do_work)
        # 接收子线程信号发来的数据
        self.work_thread.result_ready.connect(self.operate_result)
        # 线程执行完成关闭线程
        self.thread.finished.connect(self.thread_stop)

        self.ui.pushButton_run.clicked.connect(partial(self.my_on_pushButton_run_clicked))
        self.ui.toolButton_open.clicked.connect(self.on_toolButton_open_triggered)

        self.refreshInfo()

    def refreshInfo(self):
        self.str_comboBox_method = self.ui.comboBox_method.currentText()
        self.int_spinBox_window_size = self.ui.spinBox_window_size.value()
        self.str_lineEdit_path = self.ui.lineEdit_path.text()
        self.bool_radioButton_isCached = self.ui.radioButton_isCached.isChecked()

    def logSettingInfo(self):
        settingInfo = "\n" + (f"method_name:{self.str_comboBox_method}\n"
                              f"window_size:{self.int_spinBox_window_size}s\n"
                              f"video_path:{self.str_lineEdit_path}\n"
                              f"use_cache:{self.bool_radioButton_isCached}\n")
        self.ui.textEdit_logs.append(settingInfo)

    def log(self, resStr):
        self.ui.textEdit_logs.append("\n" + resStr)

    def thread_stop(self):
        # 退出线程
        print("thread finished")
        self.thread.exit()

    def my_on_pushButton_run_clicked(self):
        self.refreshInfo()
        self.logSettingInfo()
        if self.thread.isRunning() == True:
            print("thread is already running...")
        else:
            self.ui.pushButton_run.setDisabled(True)
            self.log("Running...")
            self.thread.start()

    @QtCore.pyqtSlot()
    def on_toolButton_open_triggered(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          os.getcwd(),
                                                          "Video Files (*.mp4);;All Files (*)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(fileName1, filetype)
        self.ui.lineEdit_path.setText(fileName1)

    def operate_result(self, resStr):
        self.ui.textEdit_logs.append("\n" + resStr)
        self.log("\n" + "Finish process!")
        self.ui.pushButton_run.setDisabled(False)
        self.thread.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = My_MainWindow()
    window.show()
    sys.exit(app.exec_())
