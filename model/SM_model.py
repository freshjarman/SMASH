"""
SM_model.py - 分数匹配（Score Matching）核心模型模块

本模块实现了 SMASH 项目的核心生成模型，基于分数匹配（Score Matching）方法
进行时空事件的联合建模与生成。这是整个项目最核心的技术实现。

核心概念：
=========

什么是分数匹配（Score Matching）？
---------------------------------
分数（Score）是概率密度函数对数的梯度：s(x) = ∇_x log p(x)
分数匹配通过学习这个梯度函数来隐式地学习数据分布，而无需计算归一化常数。

优势：
    - 避免了计算复杂的配分函数（partition function），即归一化常数
    - 可以通过 Langevin 动力学从学到的分布中采样
    - 适合处理连续分布的时空数据

数据范围变换：
=============
模型内部使用 [-1, 1] 范围进行计算，外部数据使用 [0, 1] 范围：

    normalize_to_neg_one_to_one():   [0,1] → [-1,1]  (输入模型前)
    unnormalize_to_zero_to_one():    [-1,1] → [0,1]  (输出模型后)

注意：当数据维度为4（含事件类型）时，第2维(index=1)的mark不做变换，保持原值。

主要类：
========

1. SinusoidalPosEmb (正弦位置编码)
----------------------------------
将标量值转换为高维向量表示，灵感来自 Transformer。
用于编码连续的时间/噪声步信息。

公式：PE(pos, 2i) = sin(pos / 10000^(2i/d))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

2. ScoreMatch_module (分数匹配网络)
-----------------------------------
核心的分数估计网络，用于估计数据分布的分数函数。

网络结构：
    - linears_temporal: 时间分支，处理时间维度 (1 → num_units)
    - linears_spatial: 空间分支，处理空间维度 (2 → num_units)
    - cond_temporal/spatial/joint: 条件信息注入层
    - output_intensity: 输出时间强度函数 λ(t)
    - output_score: 输出空间分数

关键方法：
    get_intensity(t, cond): 
        计算时间强度函数 λ(t|history)
        用于描述事件在时间 t 发生的瞬时概率
    
    get_score_loc(x, cond):
        计算空间位置的分数 ∇_loc log p(loc|t, history)
        使用注意力机制融合时间和空间信息
    
    get_score(x, cond):
        计算完整分数（时间+空间）
        时间分数通过强度函数的自动微分获得
        score_t = ∂log λ(t)/∂t - λ(t)  (源自点过程的分数推导)
    
    get_score_mark(x, mark, cond):
        带事件类型的分数计算
        额外返回mark type的概率分布（他除了返回score_t & score_loc外，还返回了score_mark但实际就是mark_probs: P(k∣t,history)）

3. SMASH (采样器)
-----------------
基于分数匹配的采样器，使用 Langevin 动力学生成样本。

Langevin 动力学采样公式：
    x_{t+1} = x_t + (ε/2) * score(x_t) + √ε * z,  z ~ N(0, I)

关键参数：
    sigma: (sigma_time, sigma_loc) 噪声尺度，控制扰动强度
    langevin_step: Langevin 采样步长 ε, 每一步的更新幅度
    sampling_timesteps: 总采样步数, 控制生成质量与速度
    n_samples: 每个条件生成的样本数（用于不确定性估计）

关键方法：
    sample_from_last(batch_size, step, is_last, cond, last_sample):
        从上一状态继续采样
        - step: 本次执行的采样步数
        - is_last: 是否是最后一轮（需要额外去噪修正）
        - last_sample: 上一轮的采样结果（支持渐进式采样）
        
        最后一步使用 Tweedie 公式进行去噪：
        x_final = x + σ² * score(x)
    
    p_losses(x_start, noise, cond):
        计算去噪分数匹配（DSM）损失
        损失函数：L = E[||score(x+σz) - (x_start-x)/σ²||²]
    
    p_losses_mark(x_start, noise, cond):
        带事件类型的损失，包含：
        - 去噪分数匹配损失（时间+空间）
        - 交叉熵损失（事件类型分类）

    forward(img, cond):
        前向传播，计算训练损失
        输入先转换到 [-1,1] 范围

4. Model_all (模型包装器)
------------------------
整合 Transformer 编码器和 SMASH 解码器的包装类。
    - transformer: 编码历史序列，生成条件向量
    - decoder: SMASH 采样器，基于条件生成下一事件

训练与推理流程：
===============

训练阶段：
    1. Transformer 编码历史序列 → 条件向量 cond
    2. 对真实下一事件 (t, x, y) 加噪：x_noisy = x + σ * noise
    3. 网络预测分数 score(x_noisy | cond); 如果有mark，还要基于THP输出的intensity计算类别概率
    4. 计算去噪分数匹配损失；如果有mark还要加上交叉熵损失项
    5. 反向传播更新模型参数

推理阶段：
    1. Transformer 编码历史序列 → 条件向量 cond
    2. 初始化随机噪声 x ~ N(0, I)
    3. Langevin 采样迭代：
       for t in range(sampling_timesteps):
           z ~ N(0, I)
           x = x + (ε/2) * score(x|cond) + √ε * z

        PS: 分段采样（比如每次250步共8段）中，每次都都把上次的结果归一化到[-1,1]再开始继续采样
    4. 最后去噪修正：x_final = x + σ² * score(x)
    5. 反归一化得到预测的 (时间间隔, x坐标, y坐标)

超参数说明：
===========
    sigma_time, sigma_loc: 噪声尺度
        - 越大：探索范围越广，但收敛慢
        - 越小：精度高，但可能陷入局部
        - 典型值：0.05
    
    langevin_step: Langevin 步长 ε
        - 越大：收敛快，但可能不稳定
        - 越小：稳定，但需要更多步数
        - 典型值：0.005
    
    sampling_timesteps: 采样总步数
        - 越多：质量越好，但推理越慢
        - 典型值：500
    
    loss_lambda: 事件类型损失权重
        - 平衡时空损失和类型分类损失
        - 典型值：0.5

    loss_lambda2: 空间损失相对时间损失的权重
        - [1, loss_lambda2, loss_lambda2] 分别对应 [时间, x, y]
        - 典型值：1.0

数学背景：
=========
时间点过程的分数：
    对于强度函数 λ(t)，时间的分数为：
    score_t = ∂log λ(t)/∂t - λ(t)
    
    这来自于点过程的密度函数：
    p(t) = λ(t) * exp(-∫₀ᵗ λ(s)ds)

去噪分数匹配（Denoising Score Matching）：
    目标：学习 score(x) = ∇_x log p(x)
    通过学习去噪来隐式学习分数：
    L = E_{x~p(x), z~N(0,I)} [||s_θ(x+σz) - (-z/σ)||²]
    等价于：||s_θ(x_noisy) - (x_clean - x_noisy)/σ²||²

依赖关系：
=========
本模块被 app.py 调用：
    1. ScoreMatch_module 作为分数估计网络
    2. SMASH 包装器进行训练和采样
    3. Model_all 整合完整的编码-解码流程

最后更新：2025年11月
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


def normalize_to_neg_one_to_one(img):
    img_new = img * 2 - 1
    if img.size(-1) == 4:
        img_new[:, :, 1] = img[:, :, 1]  # keep mark-dim (index=1) unchanged
    return img_new


def unnormalize_to_zero_to_one(img):
    img_new = (img + 1) * 0.5
    if img.size(-1) == 4:
        img_new[:, :, 1] = img[:, :, 1]
    return img_new


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScoreMatch_module(nn.Module):

    def __init__(self, dim, num_units=64, self_condition=False, condition=True, cond_dim=0, num_types=1):

        super(ScoreMatch_module, self).__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        self.cond_dim = cond_dim

        sinu_pos_emb = SinusoidalPosEmb(num_units)
        fourier_dim = num_units
        self.num_types = num_types

        time_dim = num_units

        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        self.linears_spatial = nn.ModuleList([
            nn.Linear(2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        ])

        self.linears_temporal = nn.ModuleList([
            nn.Linear(1, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
        ])

        self.output_intensity = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(),
                                              nn.Linear(num_units, num_types), nn.Softplus(beta=1))
        self.output_score = nn.Sequential(nn.Linear(num_units * 2, num_units), nn.ReLU(), nn.Linear(num_units, 2))

        self.linear_t = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, num_units),
                                      nn.ReLU(), nn.Linear(num_units, 2))

        self.linear_s = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, num_units),
                                      nn.ReLU(), nn.Linear(num_units, 2))

        self.cond_all = nn.Sequential(nn.Linear(cond_dim * 3 if num_types == 1 else cond_dim * 4, num_units), nn.ReLU(),
                                      nn.Linear(num_units, num_units))

        self.cond_temporal = nn.ModuleList(
            [nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units)])

        self.cond_spatial = nn.ModuleList(
            [nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units)])

        self.cond_joint = nn.ModuleList(
            [nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units),
             nn.Linear(cond_dim, num_units)])

    def get_intensity(self, t, cond):
        x_temporal = t

        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:, :, :hidden_dim], cond[:, :, hidden_dim:2 *
                                                                                           hidden_dim], cond[:, :, (
                                                                                               2 * hidden_dim):(
                                                                                                   3 * hidden_dim
                                                                                               )], cond[:, :,
                                                                                                        3 * hidden_dim:]

        cond = self.cond_all(cond)

        for idx in range(3):
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)

            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal +
                                                         cond_mark) if self.num_types > 1 else cond_temporal)

            x_temporal += cond_joint_emb + cond_temporal_emb
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_temporal = self.linears_temporal[-1](x_temporal)

        pred = self.output_intensity(x_temporal)
        return pred

    def get_score_loc(self, x, cond):
        x_spatial, x_temporal = x[:, :, 1:], x[:, :, :1]

        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:, :, :
                                                                  hidden_dim], cond[:, :, hidden_dim:2 *
                                                                                    hidden_dim], cond[:, :,
                                                                                                      2 * hidden_dim:3 *
                                                                                                      hidden_dim], cond[:, :,
                                                                                                                        3
                                                                                                                        *
                                                                                                                        hidden_dim:]

        cond = self.cond_all(cond)

        alpha_s = F.softmax(self.linear_s(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)
        alpha_t = F.softmax(self.linear_t(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)

        for idx in range(3):
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal +
                                                         cond_mark) if self.num_types > 1 else cond_temporal)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial += cond_joint_emb + cond_spatial_emb
            x_temporal += cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        x_output_t = x_temporal * alpha_t[:, :1, :] + x_spatial * alpha_t[:, 1:2, :]
        x_output_s = x_temporal * alpha_s[:, :1, :] + x_spatial * alpha_s[:, 1:2, :]

        pred = self.output_score(torch.cat((x_output_t, x_output_s), dim=-1))
        return pred

    def get_score(self, x, cond=None, sample=True):
        t = torch.autograd.Variable(x[:, :, :1], requires_grad=True)

        intensity = self.get_intensity(t, cond)
        intensity_log = (intensity + 1e-10).log()

        intensity_grad_t = torch.autograd.grad(intensity_log.sum(), t, retain_graph=True, create_graph=sample)[0]
        score_t = intensity_grad_t - intensity
        score_loc = self.get_score_loc(x, cond)

        return torch.cat((score_t, score_loc), -1)

    def get_score_mark(self, x, mark, cond=None, sample=True):
        """ Get score with mark information. 除了返回score_t & score_loc外，还返回了score_mark，但实际就是mark_probs: P(k∣t,history)）"""

        t = torch.autograd.Variable(x[:, :, :1], requires_grad=True)

        intensity = self.get_intensity(t, cond)
        # intensity_total = intensity.sum(-1)
        mark_onehot = F.one_hot(mark.long(), num_classes=self.num_types)  # batch*(len-1)*num_samples*num_types
        # print(mark_onehot.size(),intensity.size())
        intensity_mark = (mark_onehot * intensity).sum(-1)
        intensity_mark_log = (intensity_mark + 1e-10).log()

        intensity_grad_t = torch.autograd.grad(intensity_mark_log.sum(), t, retain_graph=True, create_graph=sample)[0]
        score_t = intensity_grad_t - intensity.sum(-1, keepdim=True)
        score_loc = self.get_score_loc(x, cond)
        score_mark = intensity / (intensity.sum(-1).unsqueeze(-1) + 1e-10)

        return torch.cat((score_t, score_loc), -1), score_mark


class SMASH(nn.Module):
    """
    SMASH: Score-based Model for Sequence Modeling with Auxiliary Sampling and Heterogeneity
    This class implements a score-based generative model for sequence data, supporting both standard and marked (heterogeneous) data. 
    It provides methods for sampling, loss computation, and denoising objectives, with configurable noise, sampling, and loss parameters.
    Args:
        model (nn.Module): The neural network model used for score estimation.
        sigma (tuple or list): Standard deviations for noise injection (for different sequence parts).
        seq_length (int): Length of the input sequences.
        num_noise (int, optional): Number of noise samples per input. Default is 50.
        sampling_timesteps (int, optional): Number of timesteps for sampling. Default is 500.
        langevin_step (float, optional): Step size for Langevin dynamics. Default is 0.05.
        n_samples (int, optional): Number of samples (channels) per batch. Default is 300.
        sampling_method (str, optional): Sampling method to use ('normal' supported). Default is 'normal'.
        num_types (int, optional): Number of types for marked data. Default is 1 (unmarked).
        loss_lambda (float, optional): Weight for the mark loss term. Default is 1.
        loss_lambda2 (float, optional): Weight for the denoising loss term (for marked data). Default is 1.
        smooth (float, optional): Label smoothing factor for mark loss. Default is 0.0.
    Attributes:
        model (nn.Module): The underlying score model.
        channels (int): Number of channels (samples).
        num_noise (int): Number of noise samples.
        self_condition (bool): Whether the model uses self-conditioning.
        is_marked (bool): Whether the model handles marked data.
        num_types (int): Number of mark types.
        loss_lambda (float): Mark loss weight.
        loss_lambda2 (Tensor): Denoising loss weight (as tensor).
        smooth (float): Label smoothing factor.
        seq_length (int): Sequence length.
        sampling_timesteps (int): Number of sampling timesteps.
        sigma (Tensor): Noise standard deviations.
        langevin_step (float): Langevin step size.
        n_samples (int): Number of samples.
        sampling_method (str): Sampling method.
    Methods:
        sample_from_last(batch_size=16, step=100, is_last=False, cond=None, last_sample=None):
            Generates samples from the model, optionally conditioned on previous samples and marks.
        p_losses(x_start, noise=None, cond=None):
            Computes the denoising score matching loss for unmarked data.
        p_losses_mark(x_start, noise=None, cond=None):
            Computes the denoising and mark prediction loss for marked data.
        get_obj_denoise(x_start, x, score):
            Computes the denoising objective (score matching loss).
        get_obj_mark(x_mark, score_mark, smooth=0.0):
            Computes the cross-entropy loss for mark prediction with optional label smoothing.
        forward(img, cond, *args, **kwargs):
            Computes the total loss for a batch, dispatching to the appropriate loss function based on data type.
    Note:
        - The model expects input data to be normalized to [-1, 1] and provides utilities for normalization/denormalization.
        - Marked data refers to sequences with categorical labels (marks) for each sample.
    """

    def __init__(self,
                 model,
                 sigma,
                 seq_length,
                 num_noise=50,
                 sampling_timesteps=500,
                 langevin_step=0.05,
                 n_samples=300,
                 sampling_method='normal',
                 num_types=1,
                 loss_lambda=1,
                 loss_lambda2=1,
                 smooth=0.0):
        super(SMASH, self).__init__()
        self.model = model
        self.channels = n_samples
        self.num_noise = num_noise
        self.self_condition = self.model.self_condition
        self.is_marked = num_types > 1
        self.num_types = num_types
        self.loss_lambda = loss_lambda
        self.loss_lambda2 = torch.tensor([1., loss_lambda2, loss_lambda2]).cuda()
        self.smooth = smooth

        self.seq_length = seq_length
        self.sampling_timesteps = sampling_timesteps
        self.sigma = torch.tensor([sigma[0], sigma[1], sigma[1]]).cuda()
        self.langevin_step = langevin_step
        self.n_samples = n_samples
        self.sampling_method = sampling_method

    def sample_from_last(self, batch_size=16, step=100, is_last=False, cond=None, last_sample=None):
        seq_length, channels = 3, self.channels
        shape = (batch_size, channels, seq_length)
        e = self.langevin_step
        n_samples = self.n_samples

        if not self.is_marked:
            if last_sample is not None:
                x = normalize_to_neg_one_to_one(last_sample[0])
            else:
                x = torch.randn([*shape], device=cond.device)

            sqrt_e = math.sqrt(e)

            if self.sampling_method == 'normal':
                for _ in range(step):
                    z = torch.randn_like(x)
                    score = self.model.get_score(x, cond, False)
                    x = x + 0.5 * e * score.detach() + sqrt_e * z
                    x.clamp_(-1., 1.)

            if is_last:
                score = self.model.get_score(x, cond, False)
                x_final = x + self.sigma**2 * score.detach()
            else:
                x_final = x
            x.clamp_(-1., 1.)

            x_final.required_grads = False

            img = unnormalize_to_zero_to_one(x_final)
            return (img.detach(), None)
        else:
            if last_sample is not None:
                x, score_mark = last_sample
                x = normalize_to_neg_one_to_one(x)  # 分段采样中每次都都把上次的结果归一化到[-1,1]
                mark = torch.multinomial(score_mark.reshape(-1, self.num_types) + 1e-10, 1,
                                         replacement=False).reshape(batch_size, n_samples)  # batch*len-1*num_samples

            else:
                x = 0.5 * torch.randn([*shape], device=cond.device)
                mark = torch.multinomial(torch.ones(self.num_types).cuda(), batch_size * n_samples,
                                         replacement=True).reshape(batch_size,
                                                                   n_samples).cuda()  # batch*len-1*num_samples

            # torch.ones(self.num_types).cuda()
            sqrt_e = math.sqrt(e)

            if self.sampling_method == 'normal':
                for s in range(step):
                    z = torch.randn_like(x)
                    score, score_mark = self.model.get_score_mark(x, mark, cond, False)
                    x = x + 0.5 * e * score.detach() + sqrt_e * z
                    x.clamp_(-1., 1.)
                    mark = torch.multinomial(score_mark.detach().reshape(-1, self.num_types) + 1e-10,
                                             1,
                                             replacement=False).reshape(batch_size,
                                                                        n_samples)  # batch*len-1*num_samples
                    # if s >10:
                    #     return 0

            if is_last:
                score, _ = self.model.get_score_mark(x, mark, cond, False)
                x_final = x + self.sigma**2 * score.detach()
                _, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
                mark = torch.multinomial(score_mark.detach().reshape(-1, self.num_types) + 1e-10, 1,
                                         replacement=False).reshape(batch_size, n_samples)  # batch*len-1*num_samples
                for s in range(200):
                    z = torch.randn_like(x)
                    score, score_mark = self.model.get_score_mark(x_final, mark, cond, False)
                    x_final[:, :, 1:] = x_final[:, :, 1:] + 0.5 * e * score.detach()[:, :, 1:] + sqrt_e * z[:, :, 1:]
            else:
                x_final = x
            x_final.clamp_(-1., 1.)

            x_final.required_grads = False

            img = unnormalize_to_zero_to_one(x_final)
            return (img.detach(), score_mark.detach())

    def p_losses(self, x_start, noise=None, cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start.repeat(1, self.num_noise, 1)))

        # noise sample
        x = x_start + self.sigma * noise

        score = self.model.get_score(x, cond)

        loss = self.get_obj_denoise(x_start, x, score)

        return loss.mean()

    def p_losses_mark(self, x_start, noise=None, cond=None):
        x_mark = x_start[:, :, 1]
        x_start = torch.cat((x_start[:, :, :1], x_start[:, :, 2:]), -1)
        noise = default(noise, lambda: torch.randn_like(x_start.repeat(1, self.num_noise, 1)))

        # noise sample
        x = x_start + self.sigma * noise

        score, score_mark = self.model.get_score_mark(x, x_mark - 1, cond)

        loss = self.get_obj_denoise(x_start, x, score)
        loss *= self.loss_lambda2
        loss_mark = self.get_obj_mark(x_mark, score_mark, smooth=self.smooth)

        return loss.mean() + self.loss_lambda * loss_mark.mean()

    def get_obj_denoise(self, x_start, x, score):
        target = (x_start - x) / self.sigma**2
        # print('t',target[0][0])
        # print('s',score[0][0])
        obj = 0.5 * (score - target)**2
        obj *= self.sigma**2
        # print(obj.mean(0).mean(0))
        return obj

    def get_obj_mark(self, x_mark, score_mark, smooth=0.0):
        truth = x_mark - 1
        one_hot = F.one_hot(truth.long(), num_classes=self.num_types).float()
        one_hot = one_hot * (1 - smooth) + (1 - one_hot) * smooth / self.num_types
        log_prb = (score_mark + 1e-10).log()
        obj = -(one_hot * log_prb).sum(dim=-1)
        return obj

    def forward(self, img, cond, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        img = normalize_to_neg_one_to_one(img)

        if not self.is_marked:
            loss = self.p_losses(img, cond=cond, *args, **kwargs)
        else:
            loss = self.p_losses_mark(img, cond=cond, *args, **kwargs)

        return loss


class Model_all(nn.Module):

    def __init__(self, transformer, decoder):
        super(Model_all, self).__init__()
        self.transformer = transformer
        self.decoder = decoder
