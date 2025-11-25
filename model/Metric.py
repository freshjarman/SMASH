"""
Metric.py - 不确定性量化与校准评估指标模块

本模块实现了 SMASH 项目的评估指标，主要用于衡量模型预测的**不确定性质量**。
核心思想是：一个好的概率预测模型不仅要预测准确，还要对自己的不确定性有正确的认知。

核心概念：
=========

什么是校准（Calibration）？
--------------------------
校准衡量的是预测概率与实际频率的一致性。

例如：如果模型说"我有80%的置信度认为事件会发生在区间A内"，
那么在大量预测中，应该恰好有80%的真实事件落在区间A内。

- 校准良好：预测的置信区间与真实覆盖率一致
- 过度自信：置信区间太窄，真实覆盖率低于预测
- 过度保守：置信区间太宽，真实覆盖率高于预测

主要函数：
=========

1. time_intervals(t, target_levels)
-----------------------------------
计算时间预测的置信区间。

输入：
    t: 采样得到的时间预测值（多个样本）
    target_levels: 目标置信水平列表，如 [0.5, 0.6, 0.7, 0.8, 0.9]

处理流程：
    1. 使用核密度估计（KDE）拟合时间样本的概率密度
    2. 计算累积分布函数（CDF）
    3. 找到对应各置信水平的区间边界
    
返回：
    intervals: 各置信水平对应的区间 [(left, right), ...]
    t_pdf: 拟合的概率密度函数
    x: 采样网格点
    t_pdf_values: 网格点上的概率密度值

注意：时间置信区间的左边界固定为0（因为时间间隔非负）

2. loc_level(loc, target_levels)
--------------------------------
计算空间位置预测的置信等高线水平。

输入：
    loc: 采样得到的空间位置预测值，shape: (n_samples, 2)
    target_levels: 目标置信水平列表

处理流程：
    1. 使用2D核密度估计拟合空间样本的联合概率密度
    2. 计算使得区域内概率等于目标值的等高线水平
    
返回：
    levels: 各置信水平对应的概率密度阈值
    loc_pdf: 拟合的2D概率密度函数
    x, y: 网格坐标
    loc_pdf_values: 网格点上的概率密度值

判断点是否在置信区域内：loc_pdf(point) >= level

3. ECELoss (Expected Calibration Error)
---------------------------------------
期望校准误差，用于评估分类预测的校准程度。

公式：
    ECE = Σ (n_b / N) * |acc_b - conf_b|
    
其中：
    - n_b: 第b个bin中的样本数
    - acc_b: 第b个bin中的准确率
    - conf_b: 第b个bin中的平均置信度

工作原理：
    1. 将预测置信度分成n_bins个区间（如10个）
    2. 对每个区间计算：平均置信度 vs 实际准确率
    3. 加权求和得到总体校准误差

ECE = 0 表示完美校准
ECE 越大表示校准越差

4. get_calibration_score()
--------------------------
主评估函数，计算完整的校准分数。

输入：
    time_all: 所有时间采样结果的列表
    loc_all: 所有空间采样结果的列表  
    mark_all: 所有事件类型采样结果的列表（可选）
    time_gt: 真实时间值
    loc_gt: 真实空间位置（可能包含事件类型）
    target_levels: 评估的置信水平，默认 [0.5, 0.6, 0.7, 0.8, 0.9]

返回 7 个指标：
    [0] CS_time: 时间校准分数（各置信水平）
    [1] CS_loc: 空间校准分数（各置信水平）
    [2] calibration_time: 时间的实际覆盖计数
    [3] calibration_loc: 空间的实际覆盖计数
    [4] ece: 事件类型的ECE值
    [5] correct_list: 各bin的正确预测数
    [6] num_list: 各bin的样本数

校准分数解读：
=============

理想情况：
    如果 target_level = 0.8，则80%的真实值应该落在80%置信区间内
    calibration_time/loc 应该等于 0.8 * 样本数
    CS (Calibration Score) = |实际覆盖数 - 期望覆盖数| 应该趋近于0

计算示例：
    假设有100个样本，target_level = 0.8
    期望：80个真实值落在80%置信区间内
    实际：75个落在区间内
    CS = |75 - 80| = 5

使用场景：
=========

1. 模型选择：选择校准分数更低的模型
2. 超参数调优：调整采样参数优化不确定性估计
3. 可信度评估：评估模型预测的可信程度
4. 风险控制：在高风险场景中确保预测置信度可靠

代码示例：
=========

# 评估模型的校准性能
calibration_results = get_calibration_score(
    time_all=sampled_times,      # 采样的时间列表
    loc_all=sampled_locs,        # 采样的位置列表
    mark_all=sampled_marks,      # 采样的类型列表
    time_gt=real_times,          # 真实时间
    loc_gt=real_locs             # 真实位置
)

cs_time = calibration_results[0]  # 时间校准分数
cs_loc = calibration_results[1]   # 空间校准分数

# 校准分数越小越好
print(f"时间校准: {cs_time.mean():.4f}")
print(f"空间校准: {cs_loc.mean():.4f}")

依赖：
=====
- scipy.stats.gaussian_kde: 核密度估计
- numpy: 数值计算
- torch: 张量操作

注意事项：
=========
1. KDE需要足够的样本数（时间>=2，空间>=3）
2. 采样数量影响校准评估的稳定性
3. target_levels 通常选择 0.5-0.9 区间的多个值
4. 对于时间，置信区间左边界固定为0（时间间隔非负）

最后更新：2025年11月
"""

import torch
from scipy.stats import gaussian_kde
import numpy as np


def time_intervals(t, target_levels):

    # Time distribution
    # Estimate the PDF using KDE
    # Convert to numpy if it's a tensor and ensure it's 1D
    if hasattr(t, 'numpy'):
        t_np = t.numpy().flatten()
    else:
        t_np = np.array(t).flatten()

    # Check if we have enough samples
    if len(t_np) < 2:
        # If too few samples, create a simple interval based on min/max
        t_min, t_max = t_np.min(), t_np.max()
        intervals = [(t_min, t_max) for _ in target_levels]
        return intervals, None, None, None

    t_pdf = gaussian_kde(t_np)
    x = np.linspace(min(t_np), max(t_np), 100)  # 采样网格
    t_pdf_values = t_pdf(x)
    t_pdf_values = t_pdf_values.reshape(x.shape)  # x 上的概率密度值

    def find_credible_intervals(x, pdf_values, target_levels):
        dx = x[1] - x[0]
        cumulative = np.cumsum(pdf_values) * dx

        intervals = []
        for target in target_levels:
            left_idx = np.where(cumulative >= 0)[0][0]
            right_idx = np.where(cumulative >= min(max(cumulative), (target)))[0][0]

            intervals.append((0, x[right_idx]))  # TPP中SMASH论文中默设了t上的置信区间的左值为0

        return intervals

    intervals = find_credible_intervals(x, t_pdf_values, target_levels)
    return intervals, t_pdf, x, t_pdf_values


def loc_level(loc, target_levels):
    """
    计算空间位置预测的置信区域密度阈值。
    
    核心思想：
    ---------
    给定一组采样点（模型的多次预测），使用核密度估计(KDE)拟合一个
    2D概率密度函数，然后找到各置信水平对应的密度阈值。
    
    置信区域定义：
    - 80%置信区域 = 使得该区域包含80%的概率质量（概率密度累积）
    - 如果真实点的密度 >= 阈值，说明它落在置信区域内
    
    Parameters:
    -----------
    loc : array-like, shape (n_samples, 2)
        采样得到的空间位置预测值
        例如：300个采样点 [[x1,y1], [x2,y2], ...]
        
    target_levels : array-like
        目标置信水平列表，如 [0.5, 0.6, 0.7, 0.8, 0.9]
        
    Returns:
    --------
    levels : list
        各置信水平对应的概率密度阈值
        注意：返回顺序是 target_levels[::-1]（即对应[0.9, 0.8, ..., 0.5]）；源代码bad design遗留问题
        
    loc_pdf : scipy.stats.gaussian_kde
        拟合的2D核密度估计函数，可调用 loc_pdf(point) 获取任意点的密度
        
    x, y : ndarray
        100x100的网格坐标矩阵
        
    loc_pdf_values : ndarray, shape (100, 100)
        网格点上的概率密度值
        
    使用示例：
    ---------
    >>> loc_samples = np.random.randn(300, 2)  # 300个采样点
    >>> target_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    >>> levels, loc_pdf, _, _, _ = loc_level(loc_samples, target_levels)
    >>> 
    >>> # 判断真实点是否在80%置信区域内
    >>> real_point = np.array([[0.1, 0.2]])
    >>> in_80_region = loc_pdf(real_point.T) >= levels[::-1][4]  # levels按逆序排列
    """

    # ===== 步骤1: 数据预处理 =====
    # 将输入转换为numpy数组
    if hasattr(loc, 'numpy'):
        loc_np = loc.numpy()
    else:
        loc_np = np.array(loc)

    # 检查样本数量是否足够进行2D KDE
    # KDE需要至少3个点来估计2D分布的协方差矩阵
    if loc_np.shape[0] < 3:
        # 样本太少，返回默认值（无法进行有意义的密度估计）
        levels = [0.0 for _ in target_levels]
        return levels, None, None, None, None

    # ===== 步骤2: 核密度估计 (KDE) =====
    # 使用高斯核将离散采样点转换为连续的概率密度函数，输出一个函数 pdf(x,y) 返回任意位置的概率密度
    # 注意：gaussian_kde期望输入shape为 (n_dims, n_samples)，所以要转置
    loc_pdf = gaussian_kde(loc_np.T)  # loc_np.T: shape (2, n_samples)

    # ===== 步骤3: 创建评估网格 =====
    # 在采样点的范围内创建100x100的网格
    x = np.linspace(min(loc_np[:, 0]), max(loc_np[:, 0]), 100)  # x轴100个点
    y = np.linspace(min(loc_np[:, 1]), max(loc_np[:, 1]), 100)  # y轴100个点
    x, y = np.meshgrid(x, y)  # 创建网格，各为100x100矩阵

    # ===== 步骤4: 计算网格点上的概率密度 =====
    # 将网格坐标展平并堆叠成 (2, 10000) 的形状
    # vstack后: [[x1,x2,...,x10000], [y1,y2,...,y10000]]
    loc_pdf_values = loc_pdf(np.vstack([x.ravel(), y.ravel()]))
    loc_pdf_values = loc_pdf_values.reshape(x.shape)  # 恢复成100x100矩阵

    # ===== 步骤5: 计算置信水平对应的密度阈值 =====
    def find_contour_levels(grid, target_levels):
        """
        找到各置信水平对应的概率密度阈值。
        
        原理：
        -----
        对于X%的置信区域，我们要找一个密度阈值level，使得
        所有密度>level的区域积分等于X%。
        
        实现方法：
        ---------
        1. 将所有网格点的密度值从小到大排序
        2. 计算累积和（近似积分）
        3. 找到累积和达到(1-X%)的位置，该位置的密度就是阈值
        
        为什么是 (1-target)?
        -------------------
        - 我们要找密度高于阈值的区域（高密度区域）
        - 如果目标是80%置信区域，则80%概率在高密度区域
        - 剩余20%概率在低密度区域
        - 所以阈值是累积到20%时的密度值
        """
        # 将网格展平并排序（从小到大）
        sorted_grid = np.sort(grid.ravel())

        # 计算总"概率质量"（网格上密度值的总和，作为归一化基准）
        total = sorted_grid.sum()

        # 计算累积和（从最低密度开始累加）
        cumulative = sorted_grid.cumsum()

        levels = []
        # 注意：这里用 [::-1] 反转顺序，所以返回的levels也是反序的
        for target in target_levels[::-1]:
            # 找到累积和首次超过 (1-target)*total 的位置
            # 例如：target=0.8 时，找累积和>=0.2*total的第一个位置
            idx = np.where(cumulative >= (1 - target) * total)[0][0]
            levels.append(sorted_grid[idx])

        return levels

    levels = find_contour_levels(loc_pdf_values, target_levels)
    return levels, loc_pdf, x, y, loc_pdf_values


class ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model. 期望校准误差 (Expected Calibration Error)
    工作原理：
    1. 将预测置信度划分为 n_bins 个equally-sized区间（如 [0,0.1), [0.1,0.2), ...）
    2. 对每个区间，计算 |平均置信度 - 实际准确率|，i.e., l1_norm
    3. 按样本数加权求和
    公式：ECE = Σ (n_b / N) × |conf_b - acc_b|

    示例：
    ------
    假设 Bin [0.7, 0.8) 中有 30 个样本：
    - 平均置信度 = 0.75
    - 实际准确率 = 0.70（30个中有21个预测正确）
    - 该 Bin 的贡献 = (30/N) × |0.75 - 0.70| = (30/N) × 0.05
    """

    def __init__(self, n_bins: int = 15):
        """
        :param n_bins: number of confidence interval bins.
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels, mode):
        """
        计算 ECE
        
        Parameters:
        -----------
        probs : Tensor
            预测概率
            - mode='sample': shape (batch, n_samples)，每个样本是预测的类别
            - mode='probs': shape (batch, n_classes)，每个类别的概率
            
        labels : Tensor, shape (batch,)
            真实标签（类别索引）
            
        mode : str
            'sample': probs 是多次采样的类别结果，置信度 = 众数出现频率
            'probs': probs 是概率分布，置信度 = 最大概率值
            
        Returns:
        --------
        ece : Tensor
            期望校准误差（未归一化，需除以样本总数）
        correct_list : list
            每个 bin 中的正确预测数
        num_list : list
            每个 bin 中的样本数
        """
        if mode == 'sample':
            # 采样模式：从多次采样中统计
            # probs shape: (batch, n_samples)，每个值是预测的类别
            predictions = torch.mode(probs, 1)[0]  # 众数作为最终预测
            # 置信度 = 众数出现次数 / 总采样次数
            confidences = probs.eq(predictions.unsqueeze(-1)).sum(-1) / probs.size(1)
        else:
            # 概率模式：直接从概率分布获取
            # probs shape: (batch, n_classes)
            confidences, predictions = torch.max(probs, 1)  # 最大概率及其索引

        # 计算每个预测是否正确
        accuracies = predictions.eq(labels)  # shape: (batch,)

        correct_list = []  # 每个 bin 的正确数
        num_list = []  # 每个 bin 的样本数
        ece = torch.zeros(1, device=labels.device)

        # 遍历每个置信度区间
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # 找出置信度落在当前区间 (bin_lower, bin_upper] 的样本
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            num_in_bin = in_bin.float().sum()  # 该区间的样本数

            if num_in_bin.item() > 0:
                # 该区间的实际准确率
                accuracy_in_bin = accuracies[in_bin].float().mean()
                # 该区间的平均置信度
                avg_confidence_in_bin = confidences[in_bin].mean()
                # 累加 ECE：|置信度 - 准确率| × 样本数
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * num_in_bin
                correct_list.append(accuracy_in_bin * num_in_bin)
                num_list.append(num_in_bin)
            else:
                correct_list.append(0.)
                num_list.append(0.)

        return ece, correct_list, num_list


def get_calibration_score(time_all,
                          loc_all,
                          mark_all,
                          time_gt,
                          loc_gt,
                          target_levels=np.linspace(0.5, 0.9, 5),
                          model='ddSMTPP'):
    ece, correct_list, num_list = 0, [], []
    if loc_gt.size(-1) == 3:
        mark_gt = loc_gt[:, 0]
        loc_gt = loc_gt[:, 1:]
        if mark_all is not None:
            if model == 'DSTPP':
                mark_scores = torch.max(torch.cat(mark_all, 1), dim=-1)[1]
                mode = 'sample'
            else:
                mark_scores = torch.cat(mark_all, 1).mean(1)
                mode = 'probs'
            # mark_scores /= mark_scores.sum(-1,keepdim=True)
            eceloss = ECELoss(n_bins=10)
            ece, correct_list, num_list = eceloss(mark_scores, (mark_gt - 1).long(), mode=mode)

    time_samples = torch.cat(time_all, 1)
    loc_samples = torch.cat(loc_all, 1)
    # print(loc_samples.size())

    calibration_time = torch.zeros(len(target_levels))
    calibration_loc = torch.zeros(len(target_levels))
    for t, loc, t_g, loc_g in zip(time_samples, loc_samples, time_gt, loc_gt):
        intervals, t_pdf, _, _ = time_intervals(t, target_levels)
        calibration_time += (t_g >= torch.tensor([intervals[i][0]
                                                  for i in range(len(intervals))])) & (t_g <= torch.tensor(
                                                      [intervals[i][1] for i in range(len(intervals))]))
        levels, loc_pdf, _, _, _ = loc_level(loc, target_levels)
        calibration_loc += loc_pdf(loc_g) >= np.array(levels[::-1])

    CS_time = torch.abs(calibration_time - target_levels * len(time_samples))
    CS_loc = torch.abs(calibration_loc - target_levels * len(time_samples))

    return [CS_time, CS_loc, calibration_time, calibration_loc, ece, torch.tensor(correct_list), torch.tensor(num_list)]
