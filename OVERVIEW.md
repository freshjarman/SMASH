正在收集工作区信息# SMASH Project Overview

## 项目简介

SMASH (Score Matching-based pseudolikelihood estimation of neural mArked Spatio-temporal point proceSs with uncertainty quantification) 是一个用于标记时空点过程的神经网络模型。该项目基于 Score Matching 方法，通过 Langevin dynamics 进行采样，并提供不确定性量化能力。

**核心论文思想**：使用 Score Matching 避免传统 TPP 中复杂的归一化常数计算，通过迭代采样生成事件的时间和空间位置。

---

## 项目结构

```
SMASH/
├── app.py                    # 主训练/测试脚本
├── app_sampletime.py         # 采样时间测试脚本
├── model/                    # 模型核心代码
│   ├── SM_model.py          # Score Matching 核心模型
│   ├── Models.py            # Transformer 编码器
│   ├── Dataset.py           # 数据加载器
│   ├── Metric.py            # 评估指标（校准分数、ECE、MAE）
│   ├── Layers.py            # Encoder Layer
│   ├── SubLayers.py         # Multi-Head Attention 等
│   └── Modules.py           # Scaled Dot-Product Attention
├── dataset/                 # 数据集
│   ├── Earthquake/
│   ├── crime/
│   └── football/
├── scripts/                 # 训练脚本
│   ├── train_earthquake.sh
│   ├── train_crime.sh
│   └── train_football.sh
└── ModelSave/              # 模型保存路径
```

---

## 核心模型架构

### 1. 整体流程图

```
输入序列 → Transformer编码器 → 条件编码 → Score Matching模块 → Langevin采样 → 输出预测
```

### 2. 关键组件

#### 2.1 `Transformer_ST` - 时空分离编码器

**位置**：Models.py

**核心思想**：将时间、空间、标记（mark）**分别编码**，然后拼接作为条件。

```python
# 关键代码片段
class Encoder_ST(nn.Module):
    def __init__(self, ...):
        # 三个独立的编码分支
        self.event_emb_temporal = nn.Sequential(...)  # 时间编码
        self.event_emb_loc = nn.Sequential(...)       # 空间编码
        self.event_emb_mark = nn.Embedding(...)       # 标记编码（仅3维数据）
        
        # 三个独立的Transformer层
        self.layer_stack = nn.ModuleList([...])           # 联合特征
        self.layer_stack_temporal = nn.ModuleList([...])  # 时间特征
        self.layer_stack_loc = nn.ModuleList([...])       # 空间特征
        self.layer_stack_mark = nn.ModuleList([...])      # 标记特征（可选）
```

**输出格式**：
- 如果 `dim=2`：`[temporal_enc, spatial_enc, joint_enc]` (3×hidden_dim)
- 如果 `dim=3`：`[temporal_enc, spatial_enc, joint_enc, mark_enc]` (4×hidden_dim)

**实验要点**：
- `d_model`（条件维度）通过 `--cond_dim` 控制，不同数据集最优值不同
- `d_rnn = d_model * 4` 用于 RNN 层增强序列建模能力

---

#### 2.2 `ScoreMatch_module` - 核心得分估计网络

**位置**：SM_model.py

**核心功能**：估计 $\nabla_x \log p(x|cond)$（score function）

**关键设计**：
1. **时间分支** (`linears_temporal`)：预测时间的 score
2. **空间分支** (`linears_spatial`)：预测空间的 score
3. **自适应融合**：通过 `linear_t` 和 `linear_s` 动态加权时空信息

```python
# 核心前向传播
def get_score_loc(self, x, cond):
    x_spatial, x_temporal = x[:,:,1:], x[:,:,:1]
    
    # 提取条件编码的不同部分
    cond_temporal = cond[:,:,:hidden_dim]
    cond_spatial = cond[:,:,hidden_dim:2*hidden_dim]
    cond_joint = cond[:,:,2*hidden_dim:3*hidden_dim]
    cond_mark = cond[:,:,3*hidden_dim:]  # 如果 num_types > 1
    
    # 自适应权重（关键创新点）
    alpha_s = F.softmax(self.linear_s(cond), dim=-1)
    alpha_t = F.softmax(self.linear_t(cond), dim=-1)
    
    # 加权融合时空特征
    x_output_t = x_temporal * alpha_t[:,:1,:] + x_spatial * alpha_t[:,1:2,:]
    x_output_s = x_temporal * alpha_s[:,:1,:] + x_spatial * alpha_s[:,1:2,:]
```

**标记处理**（仅 `dim=3` 时）：
```python
def get_intensity(self, t, cond):
    # 预测各类别的强度函数
    pred = self.output_intensity(x_temporal)  # → [batch, 1, num_types]
    return pred

def get_score_mark(self, x, mark, cond):
    # 计算给定 mark 的 score
    intensity_mark = (mark_onehot * intensity).sum(-1)
    score_mark = intensity / intensity.sum(-1)  # 归一化为概率
    return score, score_mark
```

---

#### 2.3 `SMASH` - Langevin 采样器

**位置**：SM_model.py

**核心算法**：Langevin Dynamics

$$
x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t | cond) + \sqrt{\epsilon} z, \quad z \sim \mathcal{N}(0, I)
$$

**关键参数**：
- `sigma`：噪声标准差（`sigma_time`, `sigma_loc`）
- `langevin_step`：Langevin 步长 $\epsilon$
- `sampling_timesteps`：总采样步数
- `per_step`：每次迭代的步数（用于渐进采样）

**渐进式采样流程**：
```python
def sample_from_last(self, batch_size, step, is_last, cond, last_sample):
    # 1. 初始化或从上次结果继续
    if last_sample is not None:
        x = normalize_to_neg_one_to_one(last_sample[0])
    else:
        x = torch.randn([batch_size, n_samples, seq_length])
    
    # 2. Langevin 迭代
    for _ in range(step):
        z = torch.randn_like(x)
        score = self.model.get_score(x, cond, False)
        x = x + 0.5 * e * score.detach() + sqrt_e * z
        x.clamp_(-1., 1.)
    
    # 3. 最后一步：额外精修
    if is_last:
        score = self.model.get_score(x, cond, False)
        x_final = x + self.sigma**2 * score.detach()
    
    return unnormalize_to_zero_to_one(x_final)
```

**实验技巧**：
- **训练时**：`n_samples=1`, `per_step=1`（快速迭代）
- **测试时**：`n_samples=30`, `per_step=250`（高质量采样）
- **分步采样**：避免一次性生成导致内存溢出

---

### 3. 损失函数

**位置**：`SMASH.p_losses_mark`

**组成**：
```python
loss = loss_score + λ * loss_mark
```

1. **Score Matching 损失**（Denoising Score Matching）：
   $$
   L_{score} = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2} \| s_\theta(x_0 + \sigma \epsilon, cond) - \frac{x_0 - (x_0 + \sigma \epsilon)}{\sigma^2} \|^2 \sigma^2 \right]
   $$

2. **标记分类损失**（交叉熵 + Label Smoothing）：
   ```python
   def get_obj_mark(self, x_mark, score_mark, smooth=0.0):
       one_hot = F.one_hot(truth.long(), num_classes=self.num_types)
       one_hot = one_hot * (1 - smooth) + (1 - one_hot) * smooth / self.num_types
       obj = -(one_hot * (score_mark + 1e-10).log()).sum(dim=-1)
   ```

**超参数**：
- `loss_lambda`：控制 mark 损失权重（通常 0.5）
- `loss_lambda2`：控制时空损失的相对权重（通常 1.0）
- `smooth`：Label Smoothing 系数（Crime 数据集用 0.1）

---

## 数据处理

### 数据格式

**输入数据**（dataset）：
- 原始格式：`[时间, 标记, 经度, 维度]`（dim=3）或 `[时间, 经度, 维度]`（dim=2）

**预处理步骤**（`data_loader`）：
1. **时间差分**：将绝对时间转为相对时间间隔
   ```python
   if opt.log_normalization:
       delta_t = math.log(max(t_i - t_{i-1}, 1e-4))
   else:
       delta_t = t_i - t_{i-1}
   ```

2. **标记偏移**：`mark → mark + 1`（为 padding 保留 0）

3. **归一化**：
   ```python
   x_norm = (x - MIN) / (MAX - MIN)
   ```

**关键函数**：
- [`normalization`](app.py)：训练前归一化
- [`denormalization`](app.py)：评估前反归一化
- [`Batch2toModel`](app.py)：移除 padding 并重塑张量

---

## 训练流程

### 完整训练命令（以 Earthquake 为例）

```bash
cd scripts
bash train_earthquake.sh
```

**脚本内容**（train_earthquake.sh）：
```bash
dataset=Earthquake
sigma_time=0.2
sigma_loc=0.25
samplingsteps=2000
langevin_step=0.005
cond_dim=16
seed=2
loss_lambda=0.5

# 训练
python app.py \
  --dim 3 \
  --dataset ${dataset} \
  --mode train \
  --sigma_time ${sigma_time} \
  --sigma_loc ${sigma_loc} \
  --cond_dim ${cond_dim} \
  --loss_lambda ${loss_lambda} \
  --seed ${seed} \
  --batch_size 32 \
  --total_epochs 150 \
  --n_samples 1 \
  --per_step 1

# 测试
python app.py \
  --dim 3 \
  --dataset ${dataset} \
  --mode test \
  --samplingsteps ${samplingsteps} \
  --langevin_step ${langevin_step} \
  --n_samples 30 \
  --per_step 250 \
  --weight_path ${save_path}model_best.pkl
```

### 关键超参数对比

| 数据集       | sigma_time | sigma_loc | cond_dim | loss_lambda | samplingsteps |
|------------|-----------|-----------|----------|-------------|---------------|
| Earthquake | 0.2       | 0.25      | 16       | 0.5         | 2000          |
| Crime      | 0.3       | 0.03      | 64       | 0.5         | 1000          |
| Football   | 0.2       | 0.1       | 32       | 0.5         | 2000          |

**调参建议**：
1. **`sigma_time/loc`**：控制噪声水平，需根据数据归一化后的范围调整
2. **`cond_dim`**：越大模型容量越强，但训练越慢（16-64）
3. **`langevin_step`**：测试时的步长，通常 0.005-0.01
4. **`samplingsteps`**：测试采样总步数，越多越准但越慢（1000-2000）

---

## 评估指标

### 1. MAE（平均绝对误差）

**计算方式**（app.py L322-327）：
```python
gen_time = torch.cat(sampled_seq_temporal_all, 1).mean(1, keepdim=True)
mae_temporal = torch.abs(real_time - gen_time).sum() / total_num

gen_loc = torch.cat(sampled_seq_spatial_all, 1).mean(1)
mae_spatial = torch.sqrt(((real_loc[:,-2:] - gen_loc)**2).sum(-1)).sum() / total_num
```

### 2. Calibration Score（校准分数）

**核心思想**：检验预测分布是否与真实分布一致

**实现**（Metric.py）：
- **时间校准**：使用 KDE 估计 PDF，计算置信区间覆盖率
  ```python
  def time_intervals(t, target_levels=[0.5, 0.6, 0.7, 0.8, 0.9]):
      t_pdf = gaussian_kde(t.numpy().flatten())
      # 计算不同置信水平下的区间
      intervals = [(0, threshold) for threshold in cumulative_cdf]
  ```

- **空间校准**：使用 2D KDE 估计等高线
  ```python
  def loc_level(loc, target_levels):
      loc_pdf = gaussian_kde(loc.T)
      # 计算不同置信水平下的密度阈值
  ```

**输出**：
- `CS_time/loc`：各置信水平的校准偏差
- `cs2_time/loc`：实际覆盖的样本数

### 3. ECE（期望校准误差）

**用于标记分类**（Metric.py L63-89）：
```python
class ECELoss(nn.Module):
    def forward(self, probs, labels):
        # 将置信度分成 10 个 bins
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += |avg_confidence - accuracy| * num_in_bin
```

---

## 实验复现步骤

### 环境配置

```bash
# 1. 创建虚拟环境
conda create -n smash python=3.11
conda activate smash

# 2. 安装依赖
pip install torch==2.0.1 numpy scipy tqdm
```

### 训练新模型

```bash
# 1. 准备数据（已包含在 dataset/ 中）

# 2. 修改脚本（如需调参）
vim scripts/train_earthquake.sh

# 3. 运行训练
cd scripts
bash train_earthquake.sh
```

**训练监控**：
- 每 10 个 epoch 评估一次验证集
- Early stopping：验证集损失 50 个 epoch 不下降则停止
- 最佳模型保存：`model_best.pkl`

### 仅测试已有模型

```bash
python app.py \
  --dim 3 \
  --dataset Earthquake \
  --mode test \
  --sigma_time 0.2 \
  --sigma_loc 0.25 \
  --samplingsteps 2000 \
  --langevin_step 0.005 \
  --n_samples 30 \
  --per_step 250 \
  --weight_path ./ModelSave/your_model/model_best.pkl
```

### 测试采样速度

使用 app_sampletime.py：
```bash
python app_sampletime.py --mode test --dataset Earthquake --weight_path ...
```

**输出示例**：
```
One batch: Generation time for step 0-250: 12.3456 seconds
Total Test time (150 sequences): 123.45 seconds
```

---

## 关键代码段速查

### 1. 三层循环结构（采样+评估）

**位置**：app.py L271-357

```python
# 外层：分步骤采样（避免内存溢出）
while current_step < opt.samplingsteps:
    # 中层：遍历测试集
    for idx, batch in enumerate(testloader):
        # 内层：多次采样（获取统计分布）
        for i in range(int(300 / opt.n_samples)):
            sampled_seq, score_mark = Model.decoder.sample_from_last(
                batch_size=event_time_non_mask.shape[0],
                step=opt.per_step,
                is_last=is_last,
                cond=enc_out_non_mask,
                last_sample=last_sample[idx][i] if last_sample else None
            )
```

**为什么需要三层循环？**
1. **外层**：渐进式生成（如 2000 步分 8 次完成，每次 250 步）
2. **中层**：处理整个测试集的所有 batch
3. **内层**：对同一输入采样多次（如 30 次）以估计不确定性

### 2. 数据归一化/反归一化

```python
# 归一化（训练前）
def normalization(x, MAX, MIN):
    return (x - MIN) / (MAX - MIN)

# 反归一化（评估时）
def denormalization(x, MAX, MIN, log_normalization=False):
    if log_normalization:
        return torch.exp(x.detach().cpu() * (MAX - MIN) + MIN)
    else:
        return x.detach().cpu() * (MAX - MIN) + MIN
```

### 3. 学习率 Warmup

```python
def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num

# 使用
if itr < warmup_steps:
    lr = LR_warmup(1e-3, warmup_steps, itr)
else:
    lr = 1e-3 - (1e-3 - 5e-5) * (itr - warmup_steps) / opt.total_epochs
```

---

## 常见问题

### Q1: 如何添加新数据集？

1. 准备数据格式：`[[time, mark, lng, lat], ...]`
2. 保存为 pickle：`data_train.pkl`, `data_val.pkl`, `data_test.pkl`
3. 放入 `dataset/your_dataset/`
4. 修改 `get_args()` 添加数据集选项
5. 创建对应的训练脚本

### Q2: 训练不收敛怎么办？

1. 检查数据归一化范围（应在 [0, 1]）
2. 调整 `sigma_time/loc`（降低噪声）
3. 增大 `cond_dim`（增强模型容量）
4. 检查 `loss_lambda`（平衡时空和标记损失）

### Q3: 采样太慢怎么办？

- **训练时**：保持 `n_samples=1`, `per_step=1`
- **快速测试**：减少 `samplingsteps`（如 500）和 `n_samples`（如 10）
- **最终评估**：使用完整配置（2000 步 × 30 样本）

### Q4: 如何可视化结果？

生成的样本保存在：
```python
'./samples/test_{dataset}_{model}_sigma_{sigma_time}_{sigma_loc}_steps_{samplingsteps}_log_{log_normalization}.pkl'
```

加载并可视化：
```python
import pickle
sampled_record, gt_record = pickle.load(open('samples/...', 'rb'))
# sampled_record: [num_batches][num_samples](tensor, score_mark)
# gt_record: [num_batches](时间, 标记, 经度, 维度)
```

---

## 重要提示

1. **随机种子**：使用 `setup_init(args)` 确保可复现性
2. **GPU 内存**：测试时建议 batch_size=4-32（根据 GPU 显存调整）
3. **Early Stopping**：验证集损失连续 50 epoch 不下降会自动停止
4. **模型保存**：每 10 个 epoch 保存一次，最佳模型额外保存为 `model_best.pkl`

---

## 参考文献

- 原始论文：SMASH (Score Matching-based Spatio-Temporal Point Process)
- 基础实现：[DSTPP](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)
- 技术文档：[zread.ai/zichongli5/SMASH](https://zread.ai/zichongli5/SMASH)