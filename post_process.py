"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):  # 基于平滑先验的去趋势处理
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """
        Calculate heart rate based on PPG using Fast Fourier transform (FFT).
        返回一个标量
    """
    ppg_signal = np.expand_dims(ppg_signal, 0)  # shape由(frame_num,) 到 (1, frame_num)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak


def calculate_metric_per_video(predictions,  fs=35, diff_flag=False, use_bandpass=True, LP=0.75, HP=2.5, hr_method="FFT"):
    """Calculate video-level HR"""
    predictions = _detrend(predictions, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz  Hz意思是每秒多少次
        # equals [45, 150] beats per min
        [b, a] = butter(1, [LP / fs * 2, HP / fs * 2], btype='bandpass')  # 制作带通滤波器
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    if hr_method == "FFT":
        hr_pred = _calculate_fft_hr(predictions, fs=fs, low_pass=LP, high_pass=HP)
    elif hr_method == "peak detection":
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
    else:
        raise Exception
    return hr_pred
