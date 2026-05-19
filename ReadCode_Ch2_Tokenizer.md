# 第二章：语音分词器深度分析

> 本章深入分析 VibeVoice 的语音分词器（`modular_vibevoice_tokenizer.py`），这是项目中最复杂的模块（约 1200 行），实现了完整的音频编解码器，包括 ConvNeXt 风格的 Block1D、流式卷积缓存机制、以及多种采样策略。

---

## 2.1 整体架构

```
VibeVoiceAcousticTokenizerModel
├── encoder: TokenizerEncoder
│   ├── downsample_layers[0]: stem (SConv1d)
│   ├── downsample_layers[1-6]: 逐级下采样
│   ├── stages[0-5]: Block1D × depths[i]
│   ├── norm: ConvRMSNorm / Identity
│   └── head: SConv1d → (B, vae_dim, T')
│
└── decoder: TokenizerDecoder
    ├── upsample_layers[0]: stem (SConv1d)
    ├── upsample_layers[1-6]: 逐级上采样 (SConvTranspose1d)
    ├── stages[0-5]: Block1D × depths[i]
    ├── norm: ConvRMSNorm / Identity
    └── head: SConv1d → (B, 1, T)
```

**核心设计特点**：
- 编码器和解码器结构对称
- 使用连续隐空间（codebook_size = -1），而非离散 VQ 码本
- Acoustic Tokenizer 具有完整的编解码能力
- Semantic Tokenizer 仅使用编码器（无解码器）

---

## 2.2 编码器详细结构

### 2.2.1 下采样路径

以 Acoustic Tokenizer 为例（vae_dim=64）：

```
输入: (B, 1, T) @ 24kHz
  → stem: SConv1d(1, 32, k=7) → (B, 32, T)
  → stage0: 3×Block1D(32) + downsample: SConv1d(32, 64, k=16, s=8) → (B, 64, T/8)   @ 3kHz
  → stage1: 3×Block1D(64) + downsample: SConv1d(64, 128, k=10, s=5) → (B, 128, T/40)  @ 600Hz
  → stage2: 3×Block1D(128) + downsample: SConv1d(128, 256, k=10, s=5) → (B, 256, T/200) @ 120Hz
  → stage3: 3×Block1D(256) + downsample: SConv1d(256, 512, k=8, s=4) → (B, 512, T/800) @ 30Hz
  → stage4: 3×Block1D(512) + downsample: SConv1d(512, 1024, k=4, s=2) → (B, 1024, T/1600) @ 15Hz
  → stage5: 3×Block1D(1024) + downsample: SConv1d(1024, 2048, k=4, s=2) → (B, 2048, T/3200) @ 7.5Hz
  → stage6: 8×Block1D(2048)
  → head: SConv1d(2048, 64, k=7) → (B, 64, T/3200) @ 7.5Hz
```

### 2.2.2 压缩比计算

```
encoder_ratios = [8, 5, 5, 4, 2, 2]
compress_ratio = 8 × 5 × 5 × 4 × 2 × 2 = 3200
frame_rate = 24000 / 3200 = 7.5 Hz
```

即每秒音频仅产生 7.5 个 token，60 分钟音频产生 2700 个 token。

### 2.2.3 各阶段维度变化

| 阶段 | 输入维度 | 输出维度 | 下采样率 | 累计压缩 | 帧率 |
|------|---------|---------|---------|---------|------|
| stem | 1 | 32 | 1 | 1 | 24000 Hz |
| stage0 | 32 | 64 | 8 | 8 | 3000 Hz |
| stage1 | 64 | 128 | 5 | 40 | 600 Hz |
| stage2 | 128 | 256 | 5 | 200 | 120 Hz |
| stage3 | 256 | 512 | 4 | 800 | 30 Hz |
| stage4 | 512 | 1024 | 2 | 1600 | 15 Hz |
| stage5 | 1024 | 2048 | 2 | 3200 | 7.5 Hz |
| stage6 | 2048 | 2048 | 1 | 3200 | 7.5 Hz |
| head | 2048 | 64 | 1 | 3200 | 7.5 Hz |

---

## 2.3 Block1D 结构（ConvNeXt 风格）

### 2.3.1 模块定义

```python
class Block1D(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, mixer_layer="depthwise_conv"):
        self.dwconv = DepthwiseConv1d(dim, kernel_size=7, padding=3)
        self.norm = ConvRMSNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 升维 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)   # 降维
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(layer_scale_init_value * torch.ones(dim))
```

### 2.3.2 前向传播

```
x → dwconv(norm(x)) * γ → +residual → pwconv2(act(pwconv1(norm(x)))) * γ_ffn → +residual
```

更详细的展开：
```python
def forward(self, x):
    # 分支1：Depthwise Conv + Layer Scale
    residual = x
    x = self.norm(x)
    x = self.dwconv(x)
    x = x * self.gamma
    x = residual + x

    # 分支2：FFN + Layer Scale
    residual = x
    x = self.norm(x)
    x = self.pwconv1(x)  # 升维 4x
    x = self.act(x)       # GELU 激活
    x = self.pwconv2(x)   # 降维
    x = x * self.gamma_ffn
    x = residual + x

    return x
```

### 2.3.3 设计要点

1. **Layer Scale**（`layer_scale_init_value=1e-6`）：深层网络稳定训练的关键技术。初始化为极小值，训练初期 Block1D 近似恒等映射，随训练逐步学习残差。
2. **Depthwise Conv**：每个输入通道独立卷积，参数量从 O(C² × K) 降为 O(C × K)，大幅减少计算量。
3. **FFN 升维 4 倍**：标准 Transformer FFN 的扩展比例，提供足够的非线性表达能力。
4. **ConvRMSNorm**：1D 卷积友好的 RMS 归一化，在时间维度上归一化。

---

## 2.4 流式卷积层

### 2.4.1 SConv1d（Streaming Conv1d）

```python
class SConv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, ...):
        self.conv = nn.Conv1d(cin, cout, kernel_size, stride, ...)
        self.context_size = kernel_size - 1  # 需要的上下文长度
```

**非流式前向传播**：
```python
def forward(self, x, cache=None):
    if cache is None:
        return self.conv(x)
```

**流式前向传播**：
```python
def forward(self, x, cache=None):
    if cache is not None:
        key = (self.layer_id, self.sample_idx)
        prev = cache.cache.get(key, torch.zeros(...))
        x = cat([prev, x], dim=-1)       # 拼接缓存 + 当前输入
        cache.cache[key] = x[..., -self.context_size:]  # 更新缓存
    return self.conv(x)
```

**工作原理**：
1. 从缓存获取上一 chunk 的尾部 `context_size` 个样本
2. 拼接缓存 + 当前输入，保证卷积的边界正确性
3. 执行卷积
4. 更新缓存（保留尾部 `context_size` 个样本供下一个 chunk 使用）

### 2.4.2 SConvTranspose1d（Streaming ConvTranspose1d）

```python
class SConvTranspose1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, ...):
        self.conv = nn.ConvTranspose1d(cin, cout, kernel_size, stride, ...)
        self.context_size = kernel_size - stride  # 转置卷积的上下文
```

**流式转置卷积前向传播**：
1. 从缓存获取历史输入
2. 拼接历史 + 当前输入
3. 执行转置卷积
4. 只返回当前输入对应的输出部分（裁剪掉历史输入产生的输出）
5. 更新缓存（保留尾部 `context_size` 个输入）

**转置卷积的流式处理比普通卷积更复杂**：因为转置卷积的输出不仅取决于当前输入，还受到相邻输入的影响。需要精确裁剪输出，确保拼接后的音频无缝。

---

## 2.5 流式缓存机制

### 2.5.1 VibeVoiceTokenizerStreamingCache

```python
class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}  # Dict[(layer_id, sample_idx), Tensor]
        self.sample_idx = 0

    def advance_sample_idx(self):
        self.sample_idx += 1
```

- 每个 `(layer_id, sample_idx)` 对应一个缓存条目
- `layer_id` 标识网络中的具体卷积层
- `sample_idx` 标识当前处理的音频 chunk
- 编码器和解码器共享同一个缓存对象

### 2.5.2 长音频流式编码（>60s）

```python
def encode_long_audio(self, audio, sample_rate=24000):
    max_chunk_length = 60 * sample_rate  # 60秒分段
    cache = VibeVoiceTokenizerStreamingCache()
    acoustic_means = []

    for i, start in enumerate(range(0, len(audio), max_chunk_length)):
        chunk = audio[start:start + max_chunk_length]
        cache.advance_sample_idx()  # 递增 chunk 索引

        encoder_output = self.acoustic_tokenizer.encode(chunk, cache=cache)
        acoustic_means.append(encoder_output.mean)

    # 拼接所有段的 mean 后统一采样
    all_means = torch.cat(acoustic_means, dim=-1)
    # 从拼接的 mean 中采样
    acoustic_tokens = all_means + fix_std * torch.randn_like(all_means)
    return acoustic_tokens
```

**关键设计决策**：
- 分段编码使用共享缓存，保证跨段边界的连续性
- Acoustic Tokenizer：先收集各段 mean，拼接后统一采样（保持全局一致性）
- Semantic Tokenizer：直接使用各段 mean（Semantic 不需要解码，无需采样）
- 避免卷积溢出：当序列长度 > 2^32 时 int32 索引会溢出

### 2.5.3 为什么 60 秒分段？

1. **内存限制**：超长序列的卷积中间激活占用大量 GPU 内存
2. **数值稳定性**：int32 索引在序列长度超过 2^32 时会溢出
3. **缓存效率**：60 秒的分段大小在内存和计算效率之间取得平衡

---

## 2.6 采样策略

### 2.6.1 VibeVoiceTokenizerEncoderOutput

编码器输出包含 `mean` 和 `std` 两个张量，采样策略由 `std_dist_type` 控制：

**`fix` 模式**（Acoustic Tokenizer 默认）：
```python
x = mean + fix_std * randn  # fix_std = 0.5
```
- 固定标准差，不使用编码器输出的 std
- 简单稳定，避免 std 估计不准导致的采样问题

**`gaussian` 模式**：
```python
x = mean + (randn * std / 0.8) * randn
```
- 使用编码器输出的 std，但除以 0.8（经验值）
- 随机标准差，提供更大的采样多样性
- 除以 0.8 是为了补偿 std 的低估倾向

**`none` 模式**（Semantic Tokenizer 使用）：
```python
x = mean  # 直接返回 mean，不采样
```
- Semantic Tokenizer 仅编码不解码，不需要随机性
- 确定性输出保证语义特征的一致性

### 2.6.2 为什么 Acoustic 和 Semantic 使用不同采样策略？

- **Acoustic Tokenizer**：需要解码为音频波形，采样引入的随机性使生成的语音更自然、更多样
- **Semantic Tokenizer**：仅提供语义特征给 LLM，不需要解码，确定性输出更利于 LLM 的理解

---

## 2.7 解码器结构

### 2.7.1 上采样路径

与编码器对称（以 Acoustic Tokenizer 为例）：

```
输入: (B, 64, T/3200) @ 7.5Hz
  → stem: SConv1d(64, 2048, k=7) → (B, 2048, T/3200)
  → stage0: 8×Block1D(2048)
  → upsample: SConvTranspose1d(2048, 1024, k=4, s=2) → (B, 1024, T/1600) @ 15Hz
  → stage1: 3×Block1D(1024)
  → upsample: SConvTranspose1d(1024, 512, k=4, s=2) → (B, 512, T/800) @ 30Hz
  → stage2: 3×Block1D(512)
  → upsample: SConvTranspose1d(512, 256, k=8, s=4) → (B, 256, T/200) @ 120Hz
  → stage3: 3×Block1D(256)
  → upsample: SConvTranspose1d(256, 128, k=10, s=5) → (B, 128, T/40) @ 600Hz
  → stage4: 3×Block1D(128)
  → upsample: SConvTranspose1d(128, 64, k=10, s=5) → (B, 64, T/8) @ 3kHz
  → stage5: 3×Block1D(64)
  → upsample: SConvTranspose1d(64, 32, k=16, s=8) → (B, 32, T) @ 24kHz
  → stage6: 3×Block1D(32)
  → head: SConv1d(32, 1, k=7) → (B, 1, T) @ 24kHz
```

### 2.7.2 解码器的流式处理

解码器同样支持流式处理，使用 SConvTranspose1d 的缓存机制：
- 每次解码一个语音 latent（7.5 Hz 帧率）
- 输出约 3200 个音频样本（0.133 秒 @ 24kHz）
- 通过流式缓存保证跨帧边界的连续性

### 2.7.3 编码器-解码器对称性

| 属性 | 编码器 | 解码器 |
|------|--------|--------|
| depths | "3-3-3-3-3-3-8" | "8-3-3-3-3-3-3" |
| ratios | [8,5,5,4,2,2] | [2,2,4,5,5,8] |
| kernel_sizes | "7-16-10-10-8-4-4" | "7-4-4-8-10-10-16" |
| 卷积类型 | SConv1d (下采样) | SConvTranspose1d (上采样) |

depths 和 ratios 完全反转，形成镜像对称结构。

---

## 2.8 ConvRMSNorm

```python
class ConvRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, T)
        norm = torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        x = x * norm * self.weight.unsqueeze(-1)
        return x
```

- 在通道维度上计算 RMS（Root Mean Square）
- 相比 LayerNorm，不需要计算均值，计算更高效
- 适合卷积层使用，在时间维度上广播

---

## 2.9 DepthwiseConv1d

```python
class DepthwiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3):
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
```

- `groups=dim` 实现深度可分离卷积
- 每个输入通道独立卷积，不跨通道混合
- 参数量：`dim × kernel_size`，远小于标准卷积的 `dim × dim × kernel_size`

---

## 2.10 总结

VibeVoice 的语音分词器是一个精心设计的音频编解码器，核心特点包括：

1. **超低帧率**：7.5 Hz（3200:1 压缩比），使长音频处理成为可能
2. **ConvNeXt 风格**：Block1D + Layer Scale + Depthwise Conv，兼顾效率和性能
3. **连续隐空间**：不使用 VQ 码本，避免离散化损失
4. **流式设计**：SConv1d/SConvTranspose1d 的缓存机制支持实时编解码
5. **双分词器互补**：Acoustic（可解码，声学细节）+ Semantic（仅编码，语义信息）
6. **灵活采样**：fix/gaussian/none 三种策略适应不同场景
