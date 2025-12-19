#引用作者：CSDN用户“宋体的微软雅黑（hsyxxyg）”
#时间：2020年6月18日
#脚本任务：生成投影系统矩阵，并利用此矩阵进行ART重建。

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from numba import jit
# 1. 设置字体族
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 ['SimHei']
# 2. 解决负号‘-’显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

@jit(nopython=True, fastmath=True)
def CalSystemMatrix(theta, pictureSize, projectionNum, delta):
    squareChannels = np.power(pictureSize, 2)
    totalPorjectionNum = len(theta) * projectionNum
    gridNum = np.zeros((totalPorjectionNum, 2 * pictureSize))
    gridLen = np.zeros((totalPorjectionNum, 2 * pictureSize))
    t = np.arange(-(pictureSize - 1) / 2, (pictureSize - 1) / 2+1)
    for loop1 in range(len(theta)):
        for loop2 in range(pictureSize):
            u = np.zeros((2 * pictureSize))
            v = np.zeros((2 * pictureSize))
            th = theta[loop1]
            if th == 90:
            #如果计算的结果超出了网格的范围，则立刻开始计算下一个射束
                if ((t[loop2] >= pictureSize / 2 * delta) or (t[loop2] <= - pictureSize / 2 * delta)):
                    continue
                kout = pictureSize * np.ceil(pictureSize/2 - t[loop2]/delta)
                kk = np.arange(kout - (pictureSize -1 ), kout+1)
                u[0:pictureSize] = kk
                v[0:pictureSize] = np.ones(pictureSize) * delta
                
            elif th==0:
                if (t[loop2] >= pictureSize / 2 * delta) or (t[loop2] <= -pictureSize / 2 * delta):
                    continue
                kin = np.ceil(pictureSize/2 + t[loop2] / delta)
                kk = np.arange(kin, (kin + pictureSize * pictureSize), step=pictureSize)
                u[0:pictureSize] = kk
                v[0:pictureSize] = np.ones(pictureSize) * delta

            else:
                if th>90:
                    th_temp = th - 90
                elif th<90:
                    th_temp = 90 - th
                th_temp = th_temp * np.pi / 180
                #计算束线的斜率和截距
                b = t / np.cos(th_temp)
                m = np.tan(th_temp)
                y1d = -(pictureSize / 2) * delta * m + b[loop2]
                y2d = (pictureSize / 2) * delta * m + b[loop2]
                #if (y1d < -pictureSize / 2 * delta and y2d < -pictureSize/2 * delta) or (y1d > pictureSize / 2 * delta and y2d > -pictureSize / 2 * delta):
                if (y1d < -pictureSize / 2 * delta and y2d < -pictureSize/2 * delta) or (y1d > pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    continue
                if (y1d <= pictureSize / 2 * delta and  y1d >= -pictureSize/2 * delta and y2d > pictureSize / 2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin / delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = pictureSize / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d <= pictureSize/2 * delta and y1d >= -pictureSize/2 * delta and y2d >= -pictureSize/2 * delta and y2d < pictureSize/2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin/delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = y2d
                    #1:
                    #2:xout = (yout - b[loop2]) / m
                    kout = pictureSize * np.floor(pictureSize/2 - yout/delta) + pictureSize

                elif (y1d < - pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    yin = - pictureSize / 2  * delta
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = pictureSize / 2 * delta
                    #error: xout = (yout / b[loop2])/m
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d < - pictureSize / 2 * delta and y2d >= -pictureSize / 2 * delta and y2d < pictureSize / 2 * delta):
                    yin = -pictureSize / 2 * delta 
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = y2d
                    kout = pictureSize * np.floor(pictureSize / 2 - yout / delta) + pictureSize
                else:
                    continue
                #计算第i条射束穿过的像素的编号和长度
                k = kin
                c = 0
                d2 = d1 + m * delta
                while k >= 1 and k <= squareChannels:
                    if d1 >= 0 and d2 > delta:
                        u[c] = k
                        v[c] = (delta - d1) * np.sqrt(np.power(m, 2) + 1) / m
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            d1 = d1 - delta
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 == delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k>pictureSize and k != kout:
                            k = k - pictureSize + 1
                            d1 = 0
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 < delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k!=kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 >= 0 and d2 <= delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k != kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 > delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1) / m
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            #k = k + 1
                            d1 = -delta + d1
                            d2 = d1 + m * delta
                        else:
                            break
                    else:
                        print(d1, d2, "error!!!")
                    c = c + 1
                #如果投影角度小于90度，应该利用投影射线关于y轴的对称性计算出权重因子向量。
                if th < 90:
                    u_temp = np.zeros(2 * pictureSize)
                    if u.any() == 0:
                        continue
                    indexULTZero = np.where(u>0)
                    for innerloop in range(len(u[indexULTZero])):
                        r = np.mod(u[innerloop], pictureSize)
                        if r == 0:
                            u_temp[innerloop] = u[innerloop] - pictureSize
                        else:
                            u_temp[innerloop] = u[innerloop] - 2 * r + pictureSize
                    u = u_temp
            gridNum[loop1 * projectionNum + loop2, :] = u
            gridLen[loop1 * projectionNum + loop2, :] = v
    return gridNum, gridLen

def DiscreteRadonTransform(image, theta):
    projectionNum = len(image[0])
    thetaLen = len(theta)
    radontansformRes = np.zeros((projectionNum, thetaLen), dtype='float64')
    for s in range(len(theta)):
        rotation = ndimage.rotate(image, -theta[s], reshape=False).astype('float64')
        radontansformRes[:, s] = sum(rotation)
    return radontansformRes

@jit(nopython=True, fastmath=True)
def ART_reconstruction(gridNum, gridLen, projectionValue, theta, pictureSize, pictureSizeSquare, irt_Num, lam, F):
    for c in range(irt_Num):
        for loop1 in range(len(theta)):
            for loop2 in range(pictureSize):
                u = gridNum[loop1 * pictureSize + loop2, :]
                v = gridLen[loop1 * pictureSize + loop2, :]
                if np.all(u == 0):
                    continue
                w = np.zeros(pictureSizeSquare, dtype=np.float64)
                for i in range(len(u)):
                    if u[i] > 0:
                        idx = int(u[i] - 1)
                        if idx < pictureSizeSquare:
                            w[idx] = v[i]
                PP = np.dot(w, F)
                w_sq = 0.0
                for i in range(len(w)):
                    w_sq += w[i] * w[i]
                if w_sq == 0:
                    continue
                alpha = (projectionValue[loop2, loop1] - PP) / w_sq
                for i in range(len(F)):
                    F[i] += lam * alpha * w[i]
        for i in range(len(F)):
            if F[i] < 0:
                F[i] = 0
    return F

# 定义图片尺寸参数
pictureSize = np.int64(256)

# 创建双高斯分布截面
x, y = np.mgrid[0:pictureSize, 0:pictureSize]
center_x = pictureSize // 2
center_y = pictureSize // 2
    
# 第一个高斯峰
sigma1 = pictureSize * 0.15
gaussian1 = np.exp(-((x - center_x + pictureSize*0.2)**2 + (y - center_y)**2) / (2 * sigma1**2))
    
# 第二个高斯峰
sigma2 = pictureSize * 0.2
gaussian2 = np.exp(-((x - center_x - pictureSize*0.2)**2 + (y - center_y)**2) / (2 * sigma2**2))
    
# 合并两个高斯峰并归一化
double_gaussian = gaussian1 + 0.6 * gaussian2
double_gaussian /= double_gaussian.max()

# 使用双高斯分布作为输入数据
image = double_gaussian.astype(np.float64)

# 定义旋转角度
theta = np.linspace(0, 180, 60, dtype=np.float64)

# 使用离散Radon变换获取投影值
projectionValue = DiscreteRadonTransform(image, theta)

# 定义其他参数：探测器的道数，平移步长，最大迭代次数，驰豫因子
projectionNum = np.int64(256)
pictureSizeSquare = pictureSize * pictureSize
delta = np.int64(1)
irt_Num = np.int64(10)
lam = np.float64(0.25)

#计算投影矩阵
gridNum, gridLen = CalSystemMatrix(theta, pictureSize, projectionNum, delta)

dfgridNum = pd.DataFrame(gridNum)
dfgridLen = pd.DataFrame(gridLen)
#可以将系统矩阵存储到文件中,以后直接使用。
dfgridNum.to_csv("gridNum.csv", header=False, index=False)
dfgridLen.to_csv("gridLen.csv", header=False, index=False)

#gridNum = np.array(pd.read_csv("gridNum1.csv"), header=None)
#gridLen = np.array(pd.read_csv("gridLen1.csv"), header=None)
#存储重建获得的图像的矩阵
F = np.zeros((pictureSize*pictureSize, ))

irt_Num = 10
lam = 0.25
c = 0


#开始迭代过程
print("开始ART重建迭代...")
import time
start_time = time.time()
F = ART_reconstruction(gridNum, gridLen, projectionValue, theta, pictureSize, pictureSizeSquare, irt_Num, lam, F)
end_time = time.time()
print(f"ART重建完成，耗时: {end_time - start_time:.2f}秒")

F = F.reshape(pictureSize, pictureSize).conj()

# 计算MSE误差
mse = np.mean((image - F)**2)
print(f"重建MSE误差: {mse:.6f}")

# 绘制对比图：原始图像、重建结果和投影数据
plt.figure(figsize=(16, 6))

# 显示原始双高斯分布
plt.subplot(1, 3, 1)
orig_plot = plt.imshow(image, cmap="viridis")
plt.title("原始双高斯分布")
plt.colorbar(orig_plot, label="像素值")

# 显示投影数据（sinogram）
plt.subplot(1, 3, 2)
proj_plot = plt.imshow(projectionValue, cmap="viridis")
plt.title("投影数据 (Sinogram)")
plt.colorbar(proj_plot, label="投影强度")

# 显示重建结果
plt.subplot(1, 3, 3)
recon_plot = plt.imshow(F, cmap="viridis")
plt.title(f"ART重建结果 (MSE: {mse:.6f})")
plt.colorbar(recon_plot, label="重建值")

plt.tight_layout()
plt.savefig("double_gaussian_comparison.png", dpi=300)
plt.show()