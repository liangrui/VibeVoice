# 第三章：扩散头与调度器分析

> 本章深入分析 VibeVoice 的扩散头（`modular_vibevoice_diffusion_head.py`）和扩散调度器（`dpm_solver.py`、`timestep_sampler.py`），包括 DiT 风格架构、AdaLN-Zero 初始化、v-prediction 原理、DPM-Solver 算法以及损失感知的时间步采样。

---

## 3.1 扩散头整体架构

VibeVoice 的扩散头采用 DiT（Diffusion Transformer）风格，约 300 行代码。

```
VibeVoiceDiffusionHead
├── noisy_images_proj: Linear(latent_size, hidden_size)
├── cond_proj: Linear(hidden_size, cond_dim)
├── t_embedder: TimestepEmbedder
│   └── sinusoidal_embedding → MLP(256 → hidden_size → hidden_size)
├── layers: N × HeadLayer
│   └── adaLN_modulation: SiLU → Linear(cond_dim, 3*embed_dim)
│       → shift, scale, gate = modulation params
│       → x = x + gate * FFN(modulate(RMSNorm(x), shift, scale))
└── final_layer: FinalLayer
    └── adaLN_modulation: SiLU → Linear(cond_dim, 2*hidden_size)
        → shift, scale = modulation params
        → x = Linear(modulate(RMSNorm(x), shift, scale))
```

---

## 3.2 TimestepEmbedder

### 3.2.1 模块定义

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
```

### 3.2.2 前向传播

```python
def forward(self, t):
    x = sinusoidal_embedding(t, self.frequency_embedding_size)
    return self.mlp(x)
```

### 3.2.3 正弦位置编码

```python
def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

将标量时间步 `t` 映射为 `dim` 维向量，使用与 Transformer 位置编码相同的正弦-余弦公式。

---

## 3.3 HeadLayer

### 3.3.1 模块定义

```python
class HeadLayer(nn.Module):
    def __init__(self, hidden_size, cond_dim):
        self.norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_size),
        )
```

### 3.3.2 前向传播

```python
def forward(self, x, c):
    shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
    x = x + gate * self.mlp(modulate(self.norm(x), shift, scale))
    return x
```

其中 `modulate(x, shift, scale) = x * (1 + scale) + shift`。

### 3.3.3 AdaLN 调制机制

AdaLN（Adaptive Layer Normalization）是条件注入的核心：

1. 条件向量 `c` 通过 `adaLN_modulation` 网络生成三组参数：`shift`、`scale`、`gate`
2. `shift` 和 `scale` 调制归一化后的特征：`modulate(RMSNorm(x), shift, scale) = RMSNorm(x) * (1 + scale) + shift`
3. `gate` 控制残差连接的强度：`x = x + gate * mlp_output`

这种设计使得条件信息能够灵活地控制每一层的特征变换。

---

## 3.4 FinalLayer

### 3.4.1 模块定义

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, cond_dim, out_dim):
        self.norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size),
        )
```

### 3.4.2 前向传播

```python
def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
    x = self.linear(modulate(self.norm(x), shift, scale))
    return x
```

FinalLayer 与 HeadLayer 的差异：
- 只生成 `shift` 和 `scale`（无 `gate`），因为最后一层不需要残差连接
- 输出维度为 `out_dim`（= latent_size），而非 `hidden_size`

---

## 3.5 AdaLN-Zero 初始化

这是扩散头最关键的设计之一：

### 3.5.1 初始化策略

```python
# HeadLayer: adaLN_modulation 最后一层初始化为 0
nn.init.zeros_(self.adaLN_modulation[-1].weight)
nn.init.zeros_(self.adaLN_modulation[-1].bias)

# FinalLayer: adaLN_modulation 和 linear 都初始化为 0
nn.init.zeros_(self.adaLN_modulation[-1].weight)
nn.init.zeros_(self.adaLN_modulation[-1].bias)
nn.init.zeros_(self.linear.weight)
nn.init.zeros_(self.linear.bias)
```

### 3.5.2 初始化效果分析

**训练初期**：
- HeadLayer：`adaLN_modulation(c) = 0` → `shift=0, scale=0, gate=0`
  - `modulate(RMSNorm(x), 0, 0) = RMSNorm(x) * (1 + 0) + 0 = RMSNorm(x)`
  - `x = x + 0 * mlp_output = x`（恒等映射）
- FinalLayer：`adaLN_modulation(c) = 0`, `linear = 0`
  - `output = 0`

**意义**：
- 训练初期，扩散头输出全零，不会干扰 LLM 的文本训练
- 随着训练进行，AdaLN 参数逐渐学习非零值，扩散头逐步发挥作用
- 这种"零初始化"策略在深度学习中广泛使用（如 GPT-2 的残差层、DiT 等），有助于训练稳定性

---

## 3.6 条件注入机制

### 3.6.1 条件组合

```python
c = self.cond_proj(condition) + self.t_embedder(timestep)
```

条件 = LLM 隐状态投影 + 时间步嵌入

LLM 隐状态通过 `cond_proj` 投影到 `cond_dim` 维度，与时间步嵌入相加后作为统一条件。

### 3.6.2 条件注入流程

```
LLM hidden_state (hidden_size)
  → cond_proj → (cond_dim)
  → + timestep_embedding (cond_dim)
  → c (cond_dim)
  → AdaLN modulation for each layer
```

### 3.6.3 为什么使用加法而非拼接？

- 加法更高效：不需要增加 HeadLayer 的输入维度
- 语义对齐：LLM 隐状态和时间步嵌入在 `cond_dim` 空间中对齐
- 实验验证：DiT 论文中加法和拼接效果相当，加法更简洁

---

## 3.7 v-prediction 原理

### 3.7.1 三种预测目标

| 预测类型 | 目标 | 公式 |
|---------|------|------|
| epsilon-prediction | 预测噪声 ε | `ε_θ(x_t, t)` |
| x-prediction | 预测原始数据 x_0 | `x_θ(x_t, t)` |
| v-prediction | 预测 v | `v = α_t * ε - σ_t * x_0` |

其中 `α_t` 和 `σ_t` 是噪声调度的参数，满足 `α_t² + σ_t² = 1`。

### 3.7.2 v-prediction 的优势

1. **高信噪比稳定性**：在信噪比较高时（t 接近 0），epsilon-prediction 的目标噪声 ε 被信号 x_0 淹没，难以学习。v-prediction 通过 `α_t * ε - σ_t * x_0` 的组合，在高信噪比时主要预测 -x_0，在低信噪比时主要预测 ε，自动适应。
2. **与 DPM-Solver 配合**：v-prediction 的输出可以直接用于 DPM-Solver 的步进公式，无需额外转换。
3. **训练稳定性**：v-prediction 的目标范数在不同时间步上更均匀，有利于训练。

### 3.7.3 从 v 预测 x_0

```python
# v-prediction 目标
v = alpha_t * noise - sigma_t * original_sample

# 从 v 和 x_t 预测 x_0
x0 = alpha_t * v - sigma_t * noise_pred
# 其中 noise_pred 可以从 v 和 x_t 推导
```

---

## 3.8 扩散头整体前向传播

```python
def forward(self, latent, timestep, condition):
    # 1. 投影条件
    c = self.cond_proj(condition) + self.t_embedder(timestep)

    # 2. 投影输入
    x = self.noisy_images_proj(latent)

    # 3. 逐层处理
    for layer in self.layers:
        x = layer(x, c)

    # 4. 最终输出
    x = self.final_layer(x, c)

    return x  # 预测的 v 值
```

**输入**：
- `latent`：(B, 1, vae_dim) — 加噪的语音隐向量
- `timestep`：(B,) — 扩散时间步
- `condition`：(B, 1, hidden_size) — LLM 输出的条件特征

**输出**：
- (B, 1, vae_dim) — 预测的 v 值

---

## 3.9 DPM-Solver 调度器（`dpm_solver.py`）

基于 diffusers 的 `DPMSolverMultistepScheduler`，约 800 行代码。

### 3.9.1 支持的算法

| 算法 | 类型 | 说明 |
|------|------|------|
| `dpmsolver++` | 确定性 | DPM-Solver++，最常用 |
| `sde-dpmsolver++` | 随机 | 添加随机噪声的 DPM-Solver++ |

### 3.9.2 支持的阶数

- **1 阶**：等价于 DDIM
- **2 阶**：使用二阶 Runge-Kutta 方法
- **3 阶**：使用三阶 Runge-Kutta 方法（默认，精度最高）

### 3.9.3 噪声调度

**余弦调度**（VibeVoice 默认）：
```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

余弦调度的优势：
- 噪声增加更平滑，避免线性调度的早期信噪比骤降
- 在高信噪比区域（接近原始数据）有更细的时间步划分

### 3.9.4 Sigma 调度

**Karras sigmas**：
```
σ = (σ_max^(1/ρ) + t/(T-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
```
- `ρ` 控制时间步的分布密度
- 较大的 `ρ` 在高信噪比区域放置更多时间步

**LU lambdas**：基于对数均匀分布的 lambda 调度。

### 3.9.5 推理步数

- 训练步数：1000 步
- 推理步数：20 步（DPM-Solver 高阶方法加速 50 倍）

---

## 3.10 时间步采样器（`timestep_sampler.py`）

### 3.10.1 LossAwareTimestepSampler

基于损失的时间步采样器，在训练时动态调整时间步的采样分布：

```python
class LossAwareTimestepSampler:
    def __init__(self, num_timesteps, ...):
        self.loss_ema = torch.zeros(num_timesteps)  # 每个时间步的指数移动平均损失

    def sample(self, batch_size):
        # 根据损失分布采样时间步（损失高的时间步被更频繁地采样）
        probs = F.softmax(self.loss_ema / temperature, dim=0)
        timesteps = torch.multinomial(probs, batch_size, replacement=True)
        return timesteps

    def update(self, timesteps, losses):
        # 更新指数移动平均损失
        self.loss_ema[timesteps] = (1 - beta) * self.loss_ema[timesteps] + beta * losses.detach()
```

### 3.10.2 设计理念

- **均匀采样的问题**：不同时间步的损失差异很大，均匀采样可能导致在"简单"时间步上浪费训练资源
- **损失感知采样**：损失高的时间步被更频繁地采样，使训练更高效
- **温度参数**：`temperature` 控制采样分布的尖锐程度。高温 → 更均匀，低温 → 更集中在高损失时间步
- **指数移动平均**：使用 EMA 平滑损失估计，避免单次异常损失导致采样分布剧烈变化

---

## 3.11 扩散训练流程

### 3.11.1 加噪

```python
timesteps = torch.randint(0, num_train_timesteps, (B,))
noise = torch.randn_like(acoustic_tokens)
noisy_speech = scheduler.add_noise(acoustic_tokens, noise, timesteps)
```

### 3.11.2 预测

```python
model_pred = prediction_head(noisy_speech, timesteps, condition)
```

### 3.11.3 计算目标

```python
v_target = scheduler.get_v(acoustic_tokens, noise, timesteps)
# v = alpha_t * noise - sigma_t * original_sample
```

### 3.11.4 损失

```python
diffusion_loss = F.mse_loss(model_pred, v_target)
```

---

## 3.12 扩散推理流程（DPM-Solver）

```python
def sample_speech_tokens(condition, cfg_scale=3.0):
    # 初始化随机噪声
    speech = torch.randn(B, 1, vae_dim)

    # 设置时间步
    scheduler.set_timesteps(num_inference_steps=20)
    timesteps = scheduler.timesteps

    for t in timesteps:
        # 扩散头预测 v
        v_pred = prediction_head(speech, t, condition)

        # DPM-Solver 步进
        speech = scheduler.step(v_pred, t, speech).prev_sample

    return speech
```

### 3.12.1 Classifier-Free Guidance（CFG）

```python
# 正负条件各一份
combined_speech = cat([speech, speech])
combined_condition = cat([positive_condition, negative_condition])

# 扩散头预测
eps = prediction_head(combined_speech, t, combined_condition)
cond_eps, uncond_eps = eps.chunk(2)

# CFG 引导
guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

CFG 通过对比有条件和无条件的预测结果，增强条件对生成结果的控制力。`cfg_scale` 越大，生成结果越符合条件，但多样性降低。

---

## 3.13 总结

VibeVoice 的扩散系统设计精良：

1. **DiT 风格架构**：AdaLN-Zero 初始化 + 条件注入，训练稳定
2. **v-prediction**：比 epsilon-prediction 更稳定，与 DPM-Solver 配合良好
3. **DPM-Solver**：20 步推理，50 倍加速
4. **损失感知采样**：动态调整时间步分布，训练更高效
5. **CFG**：增强条件控制，提高生成质量
