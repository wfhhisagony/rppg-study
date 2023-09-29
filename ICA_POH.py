"""ICA
Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762
"""
import math
import numpy as np

from scipy import signal, linalg, sparse


def detrend(input_signal, lambda_value):
    "去趋势的平滑处理"
    signal_length = input_signal.shape[0]  # 获取行数，而其实知道列数也就1列
    # observation matrix
    H = np.identity(signal_length)  # 单位矩阵（对角线全1，其他全0）
    ones = np.ones(signal_length)  # 行向量，列数为signal_length，元素全为1
    minus_twos = -2 * np.ones(signal_length)  # 全部为-2
    diags_data = np.array([ones, minus_twos, ones])  # 会变成3行分别为 ones、minus_tows、ones
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                       (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def ICA_POH(frames, FS=35, LPF=0.75, HPF=2.5):  # frames为frames_clip  (1500, 72, 72, 3)
    # Cut off frequency.
    RGB = process_video(frames)  # 计算每一帧(72,72,3)中(72,72,1)的单通道所有像素的平均值

    NyquistF = 1 / 2 * FS  # 奈奎斯特采样频率 以15Hz的采样频率对信号进行插值处理    采样频率为源信号频率的两倍
    BGRNorm = np.zeros(RGB.shape)  # 存储归一化后的三通道
    Lambda = 100  # 平滑参数为100
    for c in range(3):
        # 使用基于平滑先验的方法进行去趋势处理
        BGRDetrend = detrend(RGB[:, c], Lambda)  # :, c表示所有行的第c列，而你知道RGB是有frame num行的，有三列
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(
            BGRDetrend)  # 归一化，np.mean和np.std不指定axis时对所有元素取平均值、标准差
    # 获取独立源 X=S，3 x frameNum  每一列都是y(t)的独立源x(t)
    _, S = ica(np.mat(BGRNorm).H, 3)  # np.mat.H返回矩阵的共轭转置，这里把 frameNum x 3 => 3 x frameNum，且是mat类型，使用下标访问时要用[i, j]

    # select BVP Source
    MaxPx = np.zeros((1, 3))  # 1 x 3矩阵
    for c in range(3):  # 3个通道
        FF = np.fft.fft(S[c, :])  # 计算一维离散傅里叶变换  FF保持S[c,:]的形状 1 x frameNum
        # F的shape为  1 x frameNum
        F = np.arange(0, FF.shape[1]) / FF.shape[
            1] * FS * 60  # np.arange是左闭右开，生成一个行向量[0, 1, 2, ..., frameNum-1] ，然后使其归一化后 *FS * 60（表示一分钟） 就是为了获得FS*60的各个等分分点
        FF = FF[:, 1:]  # shape(1, frameNum-1)  dtype complex128
        FF = FF[0]  # shape(frameNum-1,)
        N = FF.shape[0]  # frameNum-1
        Px = np.abs(FF[:math.floor(N / 2)])  # 取前一半各个元素的长度 shape(N/2,) Px的元素都大于0
        Px = np.multiply(Px, Px)  # 这是各个元素对应相乘  shape(N/2,)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF  # 和上面的F类似
        Px = Px / np.sum(Px, axis=0)  # Px每个元素都除以其所有元素之和，即算出每个元素的权重
        MaxPx[0, c] = np.max(Px)  # 取出权重最大的那个
    MaxComp = np.argmax(MaxPx)  # 为一个标量，表明矩阵中最大数的小标位置。相当于所有行排成一行，然后找下标。
    BVP_I = S[MaxComp, :]  # 获取该行所有的 (三个通道中最大的那个)
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')  # 带通滤波器，截断不在范围内的频率。返回B,A分别是多项式的分子和分母
    # signal.filtfilt应用线性数字滤波器两次，一次向前，一次向后。组合滤波器的相位为零，滤波器阶数为原滤波器阶数的两倍。  这里3x2=6
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))  # shape(1,frameNum)

    BVP = BVP_F[0]  # shape(frameNum,)
    return BVP


def process_video(frames):
    RGB = list()
    for frame in frames:
        pixel_sum = np.sum(np.sum(frame, axis=0),
                           axis=0)  # 各个通道分别将所有像素都加起来求平均，但这里没有分离各个通道，RGB是三通道的 RGB = [[1,2,3], [1,2,3], ...(frame num)]
        RGB.append(pixel_sum / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


def ica(X, Nsources=3, Wprev=0):  # 主要使用jade算法，其他的包括：进行了输入检测
    # 之前使用mat.H转置，就是因为要把X看作一个个列向量x(t)的。注意这里还是把X看作源Y，x(t)看作源信号y(t)。
    nRows = X.shape[0]  # 3
    nCols = X.shape[1]  # frameNum
    # 下面两个if都是检查输入是否规范
    if nRows > nCols:  # 说明至少要收集3帧数据，在不满足的条件下，就要转换回去变为 frameNum x 3，总要保证 row < col
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):  # 上面传入的Nsources是3，而 min(x,x)=3，也就是说该if不满足，事实上也不希望该if满足
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources,
                      Wprev)  # 传入的: X 是 3 x frameNum的， Nsources = 3是通道数  Wprev = 0  返回的: W是x(t) = Wy(t)的W，Zhat是x(t)【独立信号】
    W = np.linalg.pinv(Winv)  # W其实没有用  np.linalg.pinv是计算矩阵的伪逆
    return W, Zhat


def jade(X, m, Wprev):
    "联合近似特征矩阵对角化算法"
    "返回酉矩阵S，类似于正交矩阵"
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))  # 计算特征向量和特征值
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S
