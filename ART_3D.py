# 三维ART重建算法实现
# 将双高斯分布扩展到三维空间，z方向为高度值

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 三维离散Radon变换（简化版）
def DiscreteRadonTransform3D(volume, theta):
    """对三维体数据进行离散Radon变换，获取投影值。
    优化：修正投影维度顺序，添加边界处理，确保与ART重建函数保持一致。
    """
    projectionNum = volume.shape[0]
    thetaLen = len(theta)
    
    # 三维投影结果：(x方向, z方向, 角度)，与ART重建函数的索引顺序保持一致
    projectionValue = np.zeros((projectionNum, projectionNum, thetaLen), dtype='float64')
    
    for s in range(thetaLen):
        # 对每个切片进行旋转和投影
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z]
            # 旋转切片，使用最近邻插值减少模糊，添加边界模式处理
            rotation = ndimage.rotate(slice_2d, -theta[s], reshape=False, 
                                     mode='constant', cval=0.0).astype('float64')
            # 在x方向积分（投影）
            projectionValue[:, z, s] += np.sum(rotation, axis=1)  # 修正索引顺序为(loop2, loop3, loop1)
    
    return projectionValue

# 计算MSE误差的函数
def calculate_mse(original, reconstructed):
    """
    计算原始数据和重建数据之间的均方误差(MSE)
    
    参数:
        original: 原始三维体数据
        reconstructed: 重建的三维体数据
        
    返回:
        mse: 均方误差值
    """
    mse = np.mean((original - reconstructed) ** 2)
    return mse

# 创建三维双高斯分布数据的函数
def create_3d_double_gaussian(pictureSize, random_params=True, seed=None):
    """
    创建三维双高斯分布数据
    
    参数:
        pictureSize: 三维体数据的尺寸
        random_params: 是否使用随机参数
        seed: 随机种子（用于可重复的随机结果）
        
    返回:
        volume_3d: 三维双高斯分布数据
        double_gaussian_2d: 中间切片的二维双高斯分布
    """
    if random_params:
        if seed is not None:
            np.random.seed(seed)
        
        # 随机中心点位置
        center_x = np.random.randint(pictureSize // 3, 2 * pictureSize // 3)
        center_y = np.random.randint(pictureSize // 3, 2 * pictureSize // 3)
        
        # 随机高斯峰位置偏移 - 优化：增加两个峰分开的概率
        # 第一个峰的偏移量
        offset1_x = np.random.uniform(-pictureSize*0.25, pictureSize*0.25)
        offset1_y = np.random.uniform(-pictureSize*0.25, pictureSize*0.25)
        
        # 第二个峰的偏移量与第一个峰相反，并且距离更大
        # 使用更大的偏移范围，确保两个峰更可能分开
        offset2_x = -offset1_x * np.random.uniform(1.2, 1.8)
        offset2_y = -offset1_y * np.random.uniform(1.2, 1.8)
        
        # 限制最大偏移量，确保峰不会移出图像边界
        max_offset = pictureSize * 0.35
        offset1_x = np.clip(offset1_x, -max_offset, max_offset)
        offset1_y = np.clip(offset1_y, -max_offset, max_offset)
        offset2_x = np.clip(offset2_x, -max_offset, max_offset)
        offset2_y = np.clip(offset2_y, -max_offset, max_offset)
        
        # 随机标准差
        sigma1 = np.random.uniform(pictureSize*0.1, pictureSize*0.25)
        sigma2 = np.random.uniform(pictureSize*0.1, pictureSize*0.25)
        
        # 随机权重
        weight2 = np.random.uniform(0.4, 0.8)
        
        # 随机z方向衰减因子
        z_sigma = np.random.uniform(pictureSize*0.15, pictureSize*0.3)
    else:
        # 默认参数
        center_x = pictureSize // 2
        center_y = pictureSize // 2
        
        offset1_x = pictureSize * 0.2
        offset2_x = -pictureSize * 0.2
        offset1_y = 0
        offset2_y = 0
        
        sigma1 = pictureSize * 0.15
        sigma2 = pictureSize * 0.2
        
        weight2 = 0.6
        z_sigma = pictureSize * 0.2
    
    x, y = np.mgrid[0:pictureSize, 0:pictureSize]
    
    # 第一个高斯峰（在x-y平面）
    gaussian1 = np.exp(-((x - center_x + offset1_x)**2 + (y - center_y + offset1_y)**2) / (2 * sigma1**2))
    
    # 第二个高斯峰（在x-y平面）
    gaussian2 = np.exp(-((x - center_x + offset2_x)**2 + (y - center_y + offset2_y)**2) / (2 * sigma2**2))
    
    # 合并两个高斯峰并归一化
    double_gaussian_2d = gaussian1 + weight2 * gaussian2
    double_gaussian_2d /= double_gaussian_2d.max()
    
    # 创建三维体数据：z方向为高度值
    volume_3d = np.zeros((pictureSize, pictureSize, pictureSize), dtype=np.float64)
    
    for z in range(pictureSize):
        # 计算z方向的衰减因子：形成高斯分布
        z_factor = np.exp(-((z - pictureSize//2)**2) / (2 * z_sigma**2))
        volume_3d[:, :, z] = double_gaussian_2d * z_factor
    
    # 调试输出：显示高斯峰参数
    print(f"高斯峰参数：")
    print(f"  中心位置: ({center_x}, {center_y})")
    print(f"  第一个峰偏移: ({offset1_x:.2f}, {offset1_y:.2f}), 标准差: {sigma1:.2f}")
    print(f"  第二个峰偏移: ({offset2_x:.2f}, {offset2_y:.2f}), 标准差: {sigma2:.2f}")
    print(f"  两峰之间的距离: {np.sqrt((offset2_x - offset1_x)**2 + (offset2_y - offset1_y)**2):.2f}")
    print(f"  第二峰权重: {weight2:.2f}")
    print(f"  z方向标准差: {z_sigma:.2f}")
    
    return volume_3d, double_gaussian_2d

@jit(nopython=True, fastmath=True)
def ART_reconstruction_3D(gridNum, gridLen, projectionValue, theta, pictureSize, lam, F):
    """三维ART重建算法，优化z方向索引映射和边界处理"""
    pictureSizeCube = pictureSize * pictureSize * pictureSize
    projectionNum = pictureSize
    
    # 遍历所有投影角度
    for loop1 in range(len(theta)):
        # 遍历所有投影线
        for loop2 in range(projectionNum):
            for loop3 in range(projectionNum):  # 添加z方向的投影线
                # 获取当前投影线的系统矩阵信息
                u = gridNum[loop1 * projectionNum + loop2, :]
                v = gridLen[loop1 * projectionNum + loop2, :]
                
                if np.all(u == 0):
                    continue
                
                # 计算权重向量
                w = np.zeros(pictureSizeCube, dtype=np.float64)
                for i in range(len(u)):
                    if u[i] > 0:
                        # 将2D索引转换为3D索引
                        idx_2d = int(u[i] - 1)
                        # 系统矩阵中的像素编号是按行优先(x, y)存储的，x是行号，y是列号
                        # 与原始数据的(x, y)保持一致，不需要反转y坐标
                        y = pictureSize - 1 -(idx_2d // pictureSize)
                        x = (idx_2d % pictureSize)
                  
                        # 在z方向上，当前投影线对应loop3位置
                        z = loop3
                        
                        # 添加边界检查，确保索引有效
                        if 0 <= x < pictureSize and 0 <= y < pictureSize and 0 <= z < pictureSize:
                            idx_3d = (x * pictureSize + y) * pictureSize + z
                            if idx_3d < pictureSizeCube:
                                w[idx_3d] = v[i]
                
                # 计算投影值
                PP = np.dot(w, F)
                
                # 计算权重平方和
                w_sq = 0.0
                for i in range(len(w)):
                    w_sq += w[i] * w[i]
                
                if w_sq == 0:
                    continue
                
                # 确保索引在有效范围内
                if (0 <= loop3 < projectionValue.shape[0] and 
                    0 <= loop2 < projectionValue.shape[1] and 
                    0 <= loop1 < projectionValue.shape[2]):
                    # 获取真实投影值
                    true_projection = projectionValue[loop2, loop3, loop1]  # 修正索引顺序
                    
                    # 更新重建值，添加正则化项防止数值不稳定
                    alpha = (true_projection - PP) / (w_sq + 0.0001)
                    
                    # 限制alpha的范围，防止过度更新
                    alpha = min(max(alpha, -100), 100)
                    
                    for i in range(len(F)):
                        F[i] += lam * alpha * w[i]
    
    # 非负约束
    for i in range(len(F)):
        if F[i] < 0:
            F[i] = 0
    
    return F

@jit(nopython=True, fastmath=True)
def CalSystemMatrix(theta, pictureSize, projectionNum, delta):
    """计算投影系统矩阵，优化用于3D重建。
    增强：添加更严格的边界检查，优化计算步骤，提高稳定性。
    """
    squareChannels = np.power(pictureSize, 2)
    totalPorjectionNum = len(theta) * projectionNum
    gridNum = np.zeros((totalPorjectionNum, 2 * pictureSize))
    gridLen = np.zeros((totalPorjectionNum, 2 * pictureSize))
    
    # 预计算t数组，避免重复计算
    t = np.arange(-(pictureSize - 1) / 2, (pictureSize - 1) / 2 + 1)
    
    for loop1 in range(len(theta)):
        for loop2 in range(projectionNum):
            u = np.zeros((2 * pictureSize))
            v = np.zeros((2 * pictureSize))
            th = theta[loop1]
            
            # 处理特殊角度（0度和90度）
            if th == 90:
                if not ((-pictureSize / 2 * delta <= t[loop2] <= pictureSize / 2 * delta)):
                    continue
                    
                kout = pictureSize * np.ceil(pictureSize/2 - t[loop2]/delta)
                # 确保kout在有效范围内
                kout = max(1, min(kout, squareChannels))
                start = max(1, kout - (pictureSize - 1))
                end = min(squareChannels, kout + 1)
                count = int(end - start + 1)
                
                if count > 0:
                    u[0:count] = np.arange(start, end + 1)
                    v[0:count] = np.ones(count) * delta
                
            elif th == 0:
                if not ((-pictureSize / 2 * delta <= t[loop2] <= pictureSize / 2 * delta)):
                    continue
                    
                kin = np.ceil(pictureSize/2 + t[loop2] / delta)
                # 确保kin在有效范围内
                kin = max(1, min(kin, pictureSize))
                start = kin
                end = min(pictureSize * pictureSize, kin + (pictureSize - 1) * pictureSize)
                kk = np.arange(start, end + 1, step=pictureSize)
                
                if len(kk) > 0:
                    u[0:len(kk)] = kk
                    v[0:len(kk)] = np.ones(len(kk)) * delta

            else:
                # 处理其他角度
                if th > 90:
                    th_temp = th - 90
                elif th < 90:
                    th_temp = 90 - th
                else:
                    th_temp = 0
                    
                th_temp = th_temp * np.pi / 180
                
                # 计算束线的斜率和截距，添加稳定性检查
                cos_th = np.cos(th_temp)
                if cos_th == 0:
                    continue
                    
                b = t / cos_th
                m = np.tan(th_temp)
                
                y1d = -(pictureSize / 2) * delta * m + b[loop2]
                y2d = (pictureSize / 2) * delta * m + b[loop2]
                
                # 检查射线是否完全在图像外
                if (y1d < -pictureSize / 2 * delta and y2d < -pictureSize/2 * delta) or \
                   (y1d > pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    continue
                    
                # 计算射线与图像边界的交点
                if (y1d <= pictureSize / 2 * delta and y1d >= -pictureSize/2 * delta and y2d > pictureSize / 2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin / delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = pictureSize / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d <= pictureSize/2 * delta and y1d >= -pictureSize/2 * delta and 
                      y2d >= -pictureSize/2 * delta and y2d < pictureSize/2 * delta):
                    yin = y1d
                    d1 = yin - np.floor(yin/delta) * delta
                    kin = pictureSize * np.floor(pictureSize / 2 - yin / delta) + 1
                    yout = y2d
                    kout = pictureSize * np.floor(pictureSize/2 - yout/delta) + pictureSize

                elif (y1d < - pictureSize / 2 * delta and y2d > pictureSize / 2 * delta):
                    yin = - pictureSize / 2  * delta
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = pictureSize / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kout = np.ceil(xout / delta) + pictureSize / 2

                elif (y1d < - pictureSize / 2 * delta and 
                      y2d >= -pictureSize / 2 * delta and y2d < pictureSize / 2 * delta):
                    yin = -pictureSize / 2 * delta 
                    xin = (yin - b[loop2]) / m
                    d1 = pictureSize / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]
                    kin = pictureSize * (pictureSize - 1) + pictureSize / 2 + np.ceil(xin / delta)
                    yout = y2d
                    kout = pictureSize * np.floor(pictureSize / 2 - yout / delta) + pictureSize
                else:
                    continue
                    
                # 确保kin和kout在有效范围内
                kin = max(1, min(kin, squareChannels))
                kout = max(1, min(kout, squareChannels))
                
                # 计算第i条射束穿过的像素的编号和长度
                k = kin
                c = 0
                d2 = d1 + m * delta
                max_iter = 2 * pictureSize  # 添加最大迭代次数限制，避免死循环
                iter_count = 0
                
                while (1 <= k <= squareChannels) and (c < 2 * pictureSize) and (iter_count < max_iter):
                    iter_count += 1
                    
                    if d1 >= 0 and d2 > delta:
                        u[c] = k
                        v[c] = (delta - d1) * np.sqrt(m**2 + 1) / max(m, 1e-10)  # 避免除以零
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            d1 = d1 - delta
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 == delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(m**2 + 1)
                        if k > pictureSize and k != kout:
                            k = k - pictureSize + 1
                            d1 = 0
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 >= 0 and d2 < delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(m**2 + 1)
                        if k != kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 >= 0 and d2 <= delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(m**2 + 1) / max(m, 1e-10)  # 避免除以零
                        if k != kout:
                            k = k + 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break
                    elif d1 <= 0 and d2 > delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(m**2 + 1) / max(m, 1e-10)  # 避免除以零
                        if k > pictureSize and k != kout:
                            k = k - pictureSize
                            d1 = -delta + d1
                            d2 = d1 + m * delta
                        else:
                            break
                    else:
                        break  # 未知情况，跳出循环
                    
                    c = c + 1
                    
                # 对于th < 90度，利用投影射线关于y轴的对称性
                if th < 90:
                    if not np.all(u == 0):  # 只有当u不为全零时才处理
                        u_temp = np.zeros(2 * pictureSize)
                        # 找出所有大于0的索引
                        for i in range(len(u)):
                            if u[i] > 0:
                                # 计算原始像素位置
                                pixel_pos = int(u[i])
                                row = (pixel_pos - 1) // pictureSize
                                col = (pixel_pos - 1) % pictureSize
                                
                                # 关于y轴的对称位置
                                new_col = pictureSize - 1 - col
                                
                                # 转换回像素编号
                                new_pixel_pos = row * pictureSize + new_col + 1
                                
                                # 确保新位置在有效范围内
                                if 1 <= new_pixel_pos <= pictureSize * pictureSize:
                                    u_temp[i] = new_pixel_pos
                        u = u_temp
            
            # 将计算结果存储到输出矩阵中
            gridNum[loop1 * projectionNum + loop2, :] = u
            gridLen[loop1 * projectionNum + loop2, :] = v
    
    return gridNum, gridLen

# 主程序
if __name__ == "__main__":
    print("开始三维ART重建...")
    
    # 定义体数据尺寸参数
    pictureSize = np.int64(32)  # 较小的尺寸以便快速计算
    
    # 创建三维双高斯分布数据
    print("创建三维双高斯分布...")
    # 使用随机参数生成双高斯分布，不指定seed以获得真正的随机性
    volume_3d, double_gaussian_2d = create_3d_double_gaussian(pictureSize, random_params=True, seed=None)
    
    # 定义旋转角度
    theta = np.linspace(0, 180, 20, dtype=np.float64)  # 减少角度数量以便快速计算
    
    # 使用离散Radon变换获取投影值
    print("计算三维Radon变换...")
    projectionValue = DiscreteRadonTransform3D(volume_3d, theta)
    
    # 定义其他参数
    projectionNum = pictureSize  # 投影数量应与图片尺寸一致
    pictureSizeSquare = pictureSize * pictureSize
    delta = np.int64(1)
    irt_Num = np.int64(20)  # 减少迭代次数防止过拟合
    lam = np.float64(0.2)  # 减小松弛因子防止振荡
    
    # 计算投影矩阵
    print("计算投影系统矩阵...")
    gridNum, gridLen = CalSystemMatrix(theta, pictureSize, projectionNum, delta)
    print(f"投影系统矩阵计算完成，形状: gridNum={gridNum.shape}, gridLen={gridLen.shape}")
    
    # 初始化重建结果
    pictureSizeCube = pictureSize * pictureSize * pictureSize
    F = np.zeros((pictureSizeCube,), dtype=np.float64)
    print("重建结果初始化完成")
    
    # 开始ART重建
    print("开始ART重建迭代...")
    import time
    start_time = time.time()
    
    # 存储所有迭代步的MSE值
    mse_values = []
    
    try:
        # 在主程序中控制迭代次数
        for iteration in range(irt_Num):
            # 执行一次ART重建迭代
            F = ART_reconstruction_3D(gridNum, gridLen, projectionValue, theta, pictureSize, lam, F)
            
            # 将当前重建结果转换为三维体数据
            temp_reconstructed = F.reshape(pictureSize, pictureSize, pictureSize)
            
            # 应用与最终结果相同的平滑处理，确保MSE计算的一致性
            temp_reconstructed = ndimage.gaussian_filter(temp_reconstructed, sigma=1.0)
            temp_reconstructed = ndimage.median_filter(temp_reconstructed, size=2)
            temp_reconstructed = ndimage.gaussian_filter(temp_reconstructed, sigma=(0.8, 0.8, 1.2))
            
            # 计算当前迭代的MSE
            current_mse = calculate_mse(volume_3d, temp_reconstructed)
            mse_values.append(current_mse)
            
            # 打印当前迭代信息
            print(f"迭代 {iteration+1}/{irt_Num}, MSE: {current_mse:.6f}")
    except Exception as e:
        print(f"重建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    print(f"ART重建完成，耗时: {end_time - start_time:.2f}秒")
    
    # 将最终重建结果转换为三维体数据
    reconstructed_volume = F.reshape(pictureSize, pictureSize, pictureSize)
    
    # 应用增强平滑处理，解决分层问题
    print("应用增强平滑处理...")
    
    # 1. 应用高斯滤波到整个3D体积数据，平滑高频噪声
    reconstructed_volume = ndimage.gaussian_filter(reconstructed_volume, sigma=1.0)  # 适当增大sigma值
    
    # 2. 应用中值滤波去除椒盐噪声，保持边缘
    reconstructed_volume = ndimage.median_filter(reconstructed_volume, size=2)  # 小尺寸中值滤波
    
    # 3. 应用3D平滑算法，重点处理z方向的分层问题
    # 使用更精细的高斯滤波，针对不同方向设置不同的sigma
    reconstructed_volume = ndimage.gaussian_filter(reconstructed_volume, sigma=(0.8, 0.8, 1.2))  # z方向sigma更大
    
    # 计算最终MSE误差
    print("计算最终MSE误差...")
    final_mse = calculate_mse(volume_3d, reconstructed_volume)
    print(f"原始数据与重建数据的最终MSE误差: {final_mse:.6f}")
    
    # 打印MSE变化情况
    print(f"MSE从初始值 {mse_values[0]:.6f} 下降到最终值 {final_mse:.6f}")
    print(f"MSE下降百分比: {(1 - final_mse/mse_values[0]) * 100:.2f}%")
    
    # 可视化结果
    print("可视化结果...")
    
    # 1. 显示原始双高斯分布的中间切片
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.imshow(double_gaussian_2d, cmap='viridis')
    plt.title('原始双高斯分布 (中间切片)')
    plt.colorbar(label='像素值')
    
    # 2. 显示重建后的中间切片
    plt.subplot(132)
    plt.imshow(reconstructed_volume[:, :, pictureSize//2], cmap='viridis')
    plt.title('重建结果 (中间切片)')
    plt.colorbar(label='重建值')
    
    # 3. 显示投影数据（sinogram）
    # plt.subplot(133)
    # plt.imshow(projectionValue[:, :, theta.shape[0]//2], cmap='viridis')
    # plt.title('投影数据 (中间角度)')
    # plt.colorbar(label='投影强度')
    
    plt.tight_layout()
    plt.savefig('3d_double_gaussian_comparison.png', dpi=300)
    
    # 创建3D表面图对比
    fig = plt.figure(figsize=(16, 8))
    
    # 绘制原始双高斯分布的3D表面
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.arange(pictureSize), np.arange(pictureSize))
    ax1.plot_surface(X, Y, double_gaussian_2d, cmap='viridis', alpha=0.7)
    ax1.set_title('原始双高斯分布 3D 表面')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (高度)')
    
    # 绘制重建后的双高斯分布的3D表面
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, reconstructed_volume[:, :, pictureSize//2], cmap='viridis', alpha=0.7)
    ax2.set_title('重建双高斯分布 3D 表面')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (高度)')
    
    plt.tight_layout()
    plt.savefig('3d_double_gaussian_surface.png', dpi=300)
    
    # 绘制MSE随迭代次数变化的图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(mse_values) + 1), mse_values, 'b-', linewidth=2, marker='o', markersize=5)
    plt.title('MSE随迭代次数变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差 (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, mse in enumerate(mse_values):
        plt.text(i+1, mse, f'{mse:.6f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('mse_vs_iteration.png', dpi=300)
    
    # 保存后关闭所有图形，避免在某些环境下程序无响应
    plt.show()
    # plt.close('all')
    
    print("三维ART重建完成！")
