# VibeVoice 项目代码深度分析

## 一、项目概览

**VibeVoice** 是微软开源的前沿语音 AI 模型家族，包含 TTS（文本转语音）、ASR（自动语音识别）和 Streaming TTS（实时流式语音合成）三大核心能力。项目基于 Qwen2 大语言模型作为文本理解骨干，结合连续语音分词器（Acoustic/Semantic Tokenizer）和扩散头（Diffusion Head），实现了语音-文本的统一建模。

### 核心创新点
- **超低帧率连续分词器**：7.5 Hz 帧率，大幅降低长序列计算开销
- **Next-Token Diffusion 框架**：LLM 理解文本上下文 + 扩散头生成高保真声学细节
- **60 分钟单次 ASR**：无需分片，保持全局说话人一致性和语义连贯性
- **流式 TTS**：~300ms 首音频延迟，支持流式文本输入

---

## 二、代码结构总览

```
VibeVoice/
├── vibevoice/                    # 核心库
│   ├── modular/                  # 模型架构（配置、建模、分词器、扩散头）
│   │   ├── configuration_vibevoice.py           # TTS/ASR 配置类
│   │   ├── configuration_vibevoice_streaming.py # Streaming 配置类
│   │   ├── modeling_vibevoice.py                # TTS 模型
│   │   ├── modeling_vibevoice_asr.py            # ASR 模型
│   │   ├── modeling_vibevoice_streaming.py      # Streaming 模型定义
│   │   ├── modeling_vibevoice_streaming_inference.py  # Streaming 推理逻辑
│   │   ├── modular_vibevoice_tokenizer.py       # 语音分词器（编码器+解码器）
│   │   ├── modular_vibevoice_diffusion_head.py  # 扩散头（DiT 风格）
│   │   ├── modular_vibevoice_text_tokenizer.py  # 文本分词器封装
│   │   └── streamer.py                          # 音频流式输出
│   ├── processor/                # 数据处理器（音频、文本、ASR、流式）
│   │   ├── vibevoice_processor.py               # TTS 处理器
│   │   ├── vibevoice_asr_processor.py           # ASR 处理器
│   │   ├── vibevoice_streaming_processor.py     # Streaming 处理器
│   │   ├── vibevoice_tokenizer_processor.py     # 分词器处理器
│   │   └── audio_utils.py                       # 音频工具（归一化等）
│   ├── schedule/                 # 扩散调度器
│   │   ├── dpm_solver.py                        # DPM-Solver 调度器
│   │   └── timestep_sampler.py                  # 时间步采样器
│   ├── configs/                  # 模型配置文件
│   │   ├── qwen2.5_1.5b_64k.json               # 1.5B 模型配置
│   │   └── qwen2.5_7b_32k.json                 # 7B 模型配置
│   └── scripts/                  # 辅助脚本
├── vllm_plugin/                  # vLLM 推理加速插件
│   ├── __init__.py              # 注册入口
│   ├── model.py                 # vLLM 多模态模型（1251行）
│   ├── inputs.py                # 音频输入映射器
│   ├── tools/
│   │   └── generate_tokenizer_files.py         # 分词器文件生成工具
│   ├── scripts/
│   │   ├── start_server.py                     # 一键部署脚本
│   │   └── gradio_asr_demo_api_video.py        # Gradio ASR 演示
│   └── tests/
│       ├── test_api.py                         # API 基础测试
│       └── test_api_auto_recover.py            # 自动恢复测试
├── demo/                         # 演示与推理脚本
│   ├── vibevoice_asr_inference_from_file.py    # ASR 文件推理（581行）
│   ├── vibevoice_asr_gradio_demo.py            # ASR Gradio 演示（1268行）
│   ├── vibevoice_realtime_demo.py              # 流式 TTS 实时演示
│   ├── realtime_model_inference_from_file.py   # 流式 TTS 文件推理（311行）
│   ├── web/
│   │   └── app.py                              # FastAPI + WebSocket 实时 TTS（517行）
│   ├── voices/                                 # 预置说话人音色
│   └── text_examples/                          # 文本示例
├── finetuning-asr/               # ASR LoRA 微调
│   ├── lora_finetune.py         # 微调训练脚本（538行）
│   ├── inference_lora.py        # LoRA 推理脚本（234行）
│   └── toy_dataset/             # 示例数据集
├── docs/                         # 文档
├── Figures/                      # 论文图表
└── pyproject.toml                # 项目配置
```

---

## 三、设计理念与架构原理

### 3.1 统一语音-文本建模架构

VibeVoice 的核心设计理念是**将语音视为一种可与文本交替的"语言"**，通过统一的 Transformer 架构同时处理文本和语音：

```
文本 Token ──→ Embedding ──→ ┐
                              ├──→ Qwen2 LLM ──→ 文本生成 / 语音条件
语音 Token ──→ Connector ──→ ┘
```

**关键设计决策**：
1. **双分词器设计**：Acoustic Tokenizer（64维，可编解码）+ Semantic Tokenizer（128维，仅编码），分别捕获声学细节和语义信息
2. **SpeechConnector**：`fc1 → RMSNorm → fc2`（无激活函数），将语音特征投影到 LLM 隐空间
3. **分词器帧率**：7.5 Hz（压缩比 3200:1），即每秒音频仅产生 7.5 个 token

### 3.2 三大模型变体的架构差异

| 特性 | VibeVoice-TTS (1.5B) | VibeVoice-ASR (7B) | VibeVoice-Streaming (0.5B) |
|------|----------------------|---------------------|---------------------------|
| LLM 骨干 | Qwen2-1.5B | Qwen2-7B | Qwen2-0.5B（分层） |
| 分词器 | Acoustic + Semantic | Acoustic + Semantic | 仅 Acoustic |
| 扩散头 | ✓ | ✗ | ✓ |
| 生成方向 | 文本→语音 | 语音→文本 | 文本→语音（流式） |
| LLM 分层 | 单一 | 单一 | 下层文本编码 + 上层 TTS 生成 |
| EOS 检测 | 无 | 标准 EOS | BinaryClassifier |
| 配置类 | VibeVoiceConfig | VibeVoiceConfig | VibeVoiceStreamingConfig |
| model_type | vibevoice | vibevoice | vibevoice_streaming |

### 3.3 Next-Token Diffusion 机制

这是 VibeVoice 最核心的创新——将 LLM 的自回归生成与扩散模型结合：

**训练阶段**：
1. LLM 处理文本 + 语音 token 序列
2. 在语音 token 位置，LLM 输出隐状态作为条件
3. 对语音 latent 加噪，训练 Diffusion Head 预测噪声（v-prediction）
4. 损失 = CE Loss（文本）+ MSE Loss（扩散）

**推理阶段（TTS）**：
1. LLM 自回归生成文本 token
2. 遇到语音位置时，LLM 输出条件特征
3. Diffusion Head 从随机噪声出发，迭代去噪生成语音 latent
4. Acoustic Tokenizer 解码器将 latent 转为音频波形

---

## 四、模块详细分析

### 4.1 配置系统

#### 4.1.1 VibeVoiceConfig（`configuration_vibevoice.py`）

**设计模式**：组合配置（Composition Config），`is_composition = True`

```
VibeVoiceConfig
├── acoustic_tokenizer_config: VibeVoiceAcousticTokenizerConfig
├── semantic_tokenizer_config: VibeVoiceSemanticTokenizerConfig
├── decoder_config: Qwen2Config
└── diffusion_head_config: VibeVoiceDiffusionHeadConfig
```

**VibeVoiceAcousticTokenizerConfig 关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| vae_dim | 64 | 隐空间维度 |
| fix_std | 0.5 | 固定采样标准差 |
| std_dist_type | "gaussian" | 采样分布类型（fix/gaussian/none） |
| encoder_ratios | [8,5,5,4,2,2] | 下采样比率（乘积=3200） |
| encoder_depths | "3-3-3-3-3-3-8" | 每阶段 Block 数 |
| mixer_layer | "depthwise_conv" | 混合层类型 |
| layer_scale_init_value | 1e-6 | Layer Scale 初始值 |
| norm_type | "conv_rms_norm" | 归一化类型 |
| sample_rate | 24000 | 采样率 |
| codebook_size | -1 | 码本大小（-1表示连续） |
| encoder_kernel_sizes | "7-16-10-10-8-4-4" | 编码器卷积核大小 |
| encoder_dilations | "1-1-1-1-1-1-1" | 编码器膨胀率 |
| decoder_ratios | [2,2,4,5,5,8] | 解码器上采样比率 |
| decoder_depths | "8-3-3-3-3-3-3" | 解码器每阶段 Block 数 |
| decoder_kernel_sizes | "7-4-4-8-10-10-16" | 解码器卷积核大小 |

**VibeVoiceSemanticTokenizerConfig 关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| vae_dim | 128 | 隐空间维度（比 Acoustic 更高） |
| fix_std | 0 | 不使用固定标准差 |
| std_dist_type | "none" | 不采样，直接返回 mean |
| 其余参数 | 同 Acoustic | 共享编码器结构 |

**VibeVoiceDiffusionHeadConfig 关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| prediction_type | "v_prediction" | v-prediction 模式 |
| ddpm_beta_schedule | "cosine" | 余弦噪声调度 |
| ddpm_num_steps | 1000 | 训练扩散步数 |
| ddpm_num_inference_steps | 20 | 推理扩散步数 |
| head_num_layers | 8 | HeadLayer 层数 |
| head_embed_dim | 512 | 隐层维度 |
| head_cond_dim | 1536 | 条件维度 |
| ddpm_clip_sample | False | 是否裁剪采样值 |

**VibeVoiceConfig 自身参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| acoustic_vae_dim | 64 | Acoustic 隐空间维度 |
| semantic_vae_dim | 128 | Semantic 隐空间维度 |
| speech_scaling_factor | 1.0 | 语音缩放因子（运行时计算） |
| speech_bias_factor | 0.0 | 语音偏移因子（运行时计算） |
| use_speech_scaling | False | 是否使用语音缩放 |
| use_speech_bias | False | 是否使用语音偏移 |

**配置的 `to_dict()` 方法**：处理 `torch.dtype` 序列化问题，将 dtype 对象转为字符串，修复 GitHub Issue #199。

**`get_text_config()` 方法**：返回 `decoder_config`，兼容 transformers >= 4.57 的缓存机制。

**`num_hidden_layers` 属性**：代理到 `decoder_config.num_hidden_layers`，兼容 transformers >= 4.57。

#### 4.1.2 VibeVoiceStreamingConfig（`configuration_vibevoice_streaming.py`）

与 VibeVoiceConfig 的关键差异：
- **无 semantic_tokenizer_config**：Streaming 模型仅使用 Acoustic Tokenizer
- **新增 tts_backbone_num_hidden_layers**：指定上层 TTS 生成 Transformer 层数
- **sub_configs 仅包含**：acoustic_tokenizer_config、decoder_config、diffusion_head_config

```python
class VibeVoiceStreamingConfig(PretrainedConfig):
    model_type = "vibevoice_streaming"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
```

**LLM 分层设计**：
- `decoder_config.num_hidden_layers` = 总层数
- `tts_backbone_num_hidden_layers` = 上层 TTS 层数
- 下层文本编码层数 = 总层数 - tts_backbone_num_hidden_layers

#### 4.1.3 模型配置文件对比

**1.5B 模型（`qwen2.5_1.5b_64k.json`）**：
- decoder: Qwen2-1.5B, 28层, hidden_size=1536, 64K 上下文
- Acoustic Tokenizer: vae_dim=64, fix_std=0.5, std_dist_type="gaussian"
- Semantic Tokenizer: vae_dim=128, fix_std=0, std_dist_type="none"
- Diffusion Head: head_num_layers=8, head_embed_dim=512, head_cond_dim=1536

**7B 模型（`qwen2.5_7b_32k.json`）**：
- decoder: Qwen2-7B, 28层, hidden_size=3584, 32K 上下文
- Acoustic Tokenizer: vae_dim=64, fix_std=0.5, std_dist_type="gaussian"
- Semantic Tokenizer: vae_dim=128, fix_std=0, std_dist_type="none"
- Diffusion Head: head_num_layers=8, head_embed_dim=512, head_cond_dim=3584

### 4.2 语音分词器（`modular_vibevoice_tokenizer.py`）

这是项目中最复杂的模块，约 1200 行代码，实现了完整的音频编解码器。

#### 4.2.1 整体架构

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

#### 4.2.2 编码器详细结构

**下采样路径**（以 Acoustic Tokenizer 为例）：
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

**压缩比计算**：
```
encoder_ratios = [8, 5, 5, 4, 2, 2]
compress_ratio = 8 × 5 × 5 × 4 × 2 × 2 = 3200
frame_rate = 24000 / 3200 = 7.5 Hz
```

即每秒音频仅产生 7.5 个 token，60 分钟音频产生 2700 个 token。

#### 4.2.3 Block1D 结构（ConvNeXt 风格）

```python
class Block1D(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, ...):
        self.dwconv = DepthwiseConv1d(dim, kernel_size=7, padding=3)
        self.norm = ConvRMSNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 升维 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)   # 降维
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(layer_scale_init_value * torch.ones(dim))
```

**前向传播**：
```
x → dwconv(norm(x)) * γ → +residual → pwconv2(act(pwconv1(norm(x)))) * γ_ffn → +residual
```

- 使用 Layer Scale（`layer_scale_init_value=1e-6`）实现深层网络稳定训练
- Depthwise Conv 大幅减少参数量
- FFN 升维 4 倍后降回原维度

#### 4.2.4 流式卷积层

**SConv1d（Streaming Conv1d）**：

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
    # 流式模式
    key = (self.layer_id, self.sample_idx)
    prev = cache.cache.get(key, torch.zeros(...))
    x = cat([prev, x], dim=-1)
    cache.cache[key] = x[..., -self.context_size:]
    return self.conv(x)
```

**SConvTranspose1d（Streaming ConvTranspose1d）**：

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

#### 4.2.5 流式缓存机制

**VibeVoiceTokenizerStreamingCache**：

```python
class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}  # Dict[(layer_id, sample_idx), Tensor]
        self.sample_idx = 0

    def advance_sample_idx(self):
        self.sample_idx += 1
```

- 每个 (layer_id, sample_idx) 对应一个缓存条目
- `sample_idx` 在每次处理新 chunk 时递增
- 编码器和解码器共享同一个缓存对象

**长音频流式编码**（>60s）：
- 按 60s 分段（`max_chunk_length = 60 * sample_rate`）
- 每段独立编码，使用共享缓存
- 所有段的 mean 拼接后统一采样
- 避免卷积溢出（当序列长度 > 2^32 时 int32 索引会溢出）

#### 4.2.6 采样策略

**VibeVoiceTokenizerEncoderOutput**：
- `fix` 模式：`x = mean + fix_std * randn`（固定标准差，Acoustic 默认 fix_std=0.5）
- `gaussian` 模式：`x = mean + (randn * std/0.8) * randn`（随机标准差，除以 0.8 是经验值）
- `none` 模式：直接返回 mean（Semantic Tokenizer 使用，因为 Semantic 仅编码不解码）

#### 4.2.7 解码器结构

**上采样路径**（与编码器对称）：
```
输入: (B, 64, T/3200) @ 7.5Hz
  → stem: SConv1d(64, 2048, k=7) → (B, 2048, T/3200)
  → stage0: 8×Block1D(2048)
  → upsample: SConvTranspose1d(2048, 1024, k=4, s=2) → (B, 1024, T/1600) @ 15Hz
  → stage1: 3×Block1D(1024)
  → upsample: SConvTranspose1d(1024, 512, k=4, s=2) → (B, 512, T/800) @ 30Hz
  → ... (对称结构)
  → stage5: 3×Block1D(64)
  → upsample: SConvTranspose1d(64, 32, k=16, s=8) → (B, 32, T) @ 24kHz
  → stage6: 3×Block1D(32)
  → head: SConv1d(32, 1, k=7) → (B, 1, T) @ 24kHz
```

### 4.3 扩散头（`modular_vibevoice_diffusion_head.py`）

**架构**：DiT（Diffusion Transformer）风格，约 300 行代码。

#### 4.3.1 整体结构

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

#### 4.3.2 TimestepEmbedder

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        x = sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(x)
```

使用正弦位置编码将标量时间步映射为向量，再通过 MLP 投影到 hidden_size。

#### 4.3.3 HeadLayer

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

**前向传播**：
```python
def forward(self, x, c):
    shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
    x = x + gate * self.mlp(modulate(self.norm(x), shift, scale))
    return x
```

其中 `modulate(x, shift, scale) = x * (1 + scale) + shift`。

#### 4.3.4 FinalLayer

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

**前向传播**：
```python
def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
    x = self.linear(modulate(self.norm(x), shift, scale))
    return x
```

#### 4.3.5 关键设计

1. **AdaLN-Zero 初始化**：
   - HeadLayer 的 `adaLN_modulation` 最后一层权重初始化为 0
   - FinalLayer 的 `adaLN_modulation` 和 `linear` 权重初始化为 0
   - 训练初期，HeadLayer 输出 gate=0（恒等映射），FinalLayer 输出 0
   - 这使得训练初期扩散头相当于零输出，不会干扰 LLM 的文本训练

2. **条件注入**：
   ```python
   c = condition + timestep_embedding  # 条件 = LLM隐状态 + 时间步嵌入
   ```
   通过 AdaLN 调制每一层，实现条件控制。

3. **v-prediction**：
   预测 `v = α_t * ε - σ_t * x_0`，比 epsilon-prediction 在高信噪比时更稳定。
   与 DPM-Solver 配合使用，推理仅需 20 步。

4. **整体前向传播**：
   ```python
   def forward(self, latent, timestep, condition):
       c = self.cond_proj(condition) + self.t_embedder(timestep)
       x = self.noisy_images_proj(latent)
       for layer in self.layers:
           x = layer(x, c)
       x = self.final_layer(x, c)
       return x  # 预测的 v 值
   ```

### 4.4 TTS 模型（`modeling_vibevoice.py`）

**VibeVoiceForConditionalGeneration** 是 TTS 训练和推理的核心模型。

#### 4.4.1 模型初始化

```python
class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel):
    def __init__(self, config):
        # 子模型
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceAcousticTokenizerModel(config.semantic_tokenizer_config)
        self.language_model = Qwen2ForCausalLM(config.decoder_config)
        self.prediction_head = VibeVoiceDiffusionHead(config.diffusion_head_config)

        # 连接器
        self.acoustic_connector = SpeechConnector(
            config.acoustic_tokenizer_config.vae_dim,
            config.decoder_config.hidden_size
        )
        self.semantic_connector = SpeechConnector(
            config.semantic_tokenizer_config.vae_dim,
            config.decoder_config.hidden_size
        )

        # 噪声调度器
        self.noise_scheduler = DPMShellSolverMultistepScheduler(...)

        # 语音缩放因子（运行时计算）
        self.speech_scaling_factor = config.speech_scaling_factor
        self.speech_bias_factor = config.speech_bias_factor
        self.use_speech_scaling = config.use_speech_scaling
        self.use_speech_bias = config.use_speech_bias
```

#### 4.4.2 SpeechConnector

```python
class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))
```

注意：**无激活函数**！这是一个纯线性变换 + 归一化，设计理念是让语音特征尽可能无损地投影到 LLM 隐空间。

#### 4.4.3 训练前向传播

```python
def forward(self, input_ids, speech_tensors, acoustic_input_mask, acoustic_loss_mask, labels, ...):
    # 1. 文本嵌入
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)

    # 2. 语音特征编码
    acoustic_encoder_outputs = self.acoustic_tokenizer.encode(speech_tensors)
    acoustic_tokens = acoustic_encoder_outputs.sample()  # (B, vae_dim, T')
    acoustic_tokens = acoustic_tokens.transpose(1, 2)    # (B, T', vae_dim)

    semantic_encoder_outputs = self.semantic_tokenizer.encode(speech_tensors)
    semantic_tokens = semantic_encoder_outputs.mean      # (B, T', vae_dim_sem)
    semantic_tokens = semantic_tokens.transpose(1, 2)

    # 3. 语音特征缩放
    if self.use_speech_scaling:
        acoustic_tokens = (acoustic_tokens + self.speech_bias_factor) * self.speech_scaling_factor

    # 4. 连接器投影
    acoustic_features = self.acoustic_connector(acoustic_tokens)
    semantic_features = self.semantic_connector(semantic_tokens)

    # 5. 替换语音位置嵌入
    inputs_embeds[acoustic_input_mask] = acoustic_features + semantic_features

    # 6. LLM 前向传播
    outputs = self.language_model(inputs_embeds=inputs_embeds, ...)

    # 7. CE 损失（文本部分）
    logits = outputs.logits
    ce_loss = CrossEntropyLoss(shifted_logits, shifted_labels)

    # 8. 扩散损失（语音部分）
    condition = outputs.hidden_states[acoustic_loss_mask]
    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,))
    noise = torch.randn_like(acoustic_tokens)
    noisy_speech = self.noise_scheduler.add_noise(acoustic_tokens, noise, timesteps)
    model_pred = self.prediction_head(noisy_speech, timesteps, condition)
    v_target = self.noise_scheduler.get_v(acoustic_tokens, noise, timesteps)
    diffusion_loss = F.mse_loss(model_pred, v_target)

    return CausalLMOutput(loss=ce_loss, diffusion_loss=diffusion_loss)
```

#### 4.4.4 语音特征缩放

```python
# 首次遇到语音数据时计算
if self.use_speech_scaling and self.speech_scaling_factor == 1.0:
    self.speech_scaling_factor = 1 / acoustic_tokens.std().item()
if self.use_speech_bias and self.speech_bias_factor == 0.0:
    self.speech_bias_factor = -acoustic_tokens.mean().item()
```

- 支持分布式训练时通过 `all_reduce` 同步
- 归一化后语音特征的方差为 1，均值为 0
- 这有助于扩散头的训练稳定性

#### 4.4.5 生成流程

TTS 生成使用 transformers 的 `generate()` 方法，通过 `prepare_inputs_for_generation` 钩子实现自回归生成：

```python
def prepare_inputs_for_generation(self, input_ids, ...):
    # 首次调用：包含语音输入
    if past_key_values is None:
        # 编码语音，构建完整输入
        ...
    # 后续调用：仅传入新 token
    else:
        input_ids = input_ids[:, -1:]
    return {"input_ids": input_ids, "past_key_values": past_key_values, ...}
```

### 4.5 ASR 模型（`modeling_vibevoice_asr.py`）

**VibeVoiceASRForConditionalGeneration** 是纯编码-解码模型（无扩散头）。

#### 4.5.1 模型初始化

```python
class VibeVoiceASRForConditionalGeneration(VibeVoicePreTrainedModel):
    def __init__(self, config):
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceAcousticTokenizerModel(config.semantic_tokenizer_config)
        self.language_model = Qwen2ForCausalLM(config.decoder_config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)

        self.acoustic_connector = SpeechConnector(...)
        self.semantic_connector = SpeechConnector(...)
```

注意：ASR 模型**没有** `prediction_head`（扩散头），因为 ASR 只需要文本生成。

#### 4.5.2 语音编码

```python
def encode_speech(self, speech_tensors, ...):
    # 短音频（<=60s）：直接编码
    if audio_length <= max_chunk_length:
        acoustic_tokens = self.acoustic_tokenizer.encode(audio).sample()
        semantic_tokens = self.semantic_tokenizer.encode(audio).mean
    # 长音频（>60s）：流式分段编码
    else:
        cache = VibeVoiceTokenizerStreamingCache()
        acoustic_means, semantic_means = [], []
        for chunk in chunks:
            acoustic_out = self.acoustic_tokenizer.encode(chunk, cache=cache)
            acoustic_means.append(acoustic_out.mean)
            semantic_out = self.semantic_tokenizer.encode(chunk, cache=cache)
            semantic_means.append(semantic_out.mean)
        # 拼接所有段的 mean 后统一采样
        acoustic_tokens = sample(cat(acoustic_means))
        semantic_tokens = cat(semantic_means)  # Semantic 直接用 mean

    # 连接器投影
    acoustic_features = self.acoustic_connector(acoustic_tokens)
    semantic_features = self.semantic_connector(semantic_tokens)
    return acoustic_features + semantic_features
```

**长音频处理的关键设计**：
- 分段编码使用共享缓存，保证跨段边界的连续性
- Acoustic Tokenizer：先收集各段 mean，拼接后统一采样（保持全局一致性）
- Semantic Tokenizer：直接使用各段 mean（Semantic 不需要解码，无需采样）

#### 4.5.3 训练前向传播

```python
def forward(self, input_ids, speech_tensors, acoustic_input_mask, labels, ...):
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)
    speech_features = self.encode_speech(speech_tensors)
    inputs_embeds[acoustic_input_mask] = speech_features
    outputs = self.language_model(inputs_embeds=inputs_embeds)
    logits = self.lm_head(outputs[0])
    loss = CrossEntropyLoss(logits, labels)
    return CausalLMOutput(loss=loss)
```

#### 4.5.4 生成时语音输入处理

```python
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, ...):
    if past_key_values is None:
        # 第一次前向传播：编码语音并注入
        speech_features = self.encode_speech(speech_tensors)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        inputs_embeds[acoustic_input_mask] = speech_features
        return {"inputs_embeds": inputs_embeds, ...}
    else:
        # 后续前向传播：利用 KV Cache，只传入最新 token
        input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, ...}
```

遵循 Qwen2-VL 模式：语音输入仅在第一次前向传播时传入，后续利用 KV Cache。

### 4.6 流式 TTS 模型

这是架构最复杂的变体，由两个文件组成：
- `modeling_vibevoice_streaming.py`：模型定义
- `modeling_vibevoice_streaming_inference.py`：推理逻辑

#### 4.6.1 模型定义

```python
class VibeVoiceStreamingModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        # 下层 LLM（文本编码）
        self.language_model = Qwen2ForCausalLM(config.decoder_config)
        # 修改：只使用部分层，norm 设为 Identity
        self.language_model.model.layers = self.language_model.model.layers[:num_lm_layers]
        self.language_model.model.norm = nn.Identity()

        # 上层 TTS LLM（语音生成）
        self.tts_language_model = Qwen2ForCausalLM(config.decoder_config)
        # 修改：只使用部分层，不使用 embed_tokens
        self.tts_language_model.model.layers = self.tts_language_model.model.layers[num_lm_layers:]
        self.tts_language_model.model.embed_tokens = None

        # 输入类型嵌入（区分文本和语音）
        self.tts_input_types = nn.Embedding(2, hidden_size)
        # 0 = 语音位置, 1 = 文本位置

        # 语音分词器（仅 Acoustic）
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(...)
        self.acoustic_connector = SpeechConnector(...)

        # 扩散头
        self.prediction_head = VibeVoiceDiffusionHead(...)
        self.noise_scheduler = DPMShellSolverMultistepScheduler(...)

        # EOS 分类器
        self.tts_eos_classifier = BinaryClassifier(hidden_size)
```

#### 4.6.2 BinaryClassifier

```python
class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # sigmoid 后 > 0.5 则结束
```

#### 4.6.3 推理流程（`modeling_vibevoice_streaming_inference.py`）

**核心常量**：
```python
TTS_TEXT_WINDOW_SIZE = 5     # 每次输入 5 个文本 token
TTS_SPEECH_WINDOW_SIZE = 6   # 每次生成 6 个语音 token
```

**generate 方法完整流程**：

```python
def generate(self, tts_lm_input_ids, tts_text_ids, audio_streamer, ...):
    # 1. 预填充 voice prompt（编码说话人音色）
    lm_outputs = self.language_model(tts_lm_input_ids, use_cache=True)
    lm_cache = lm_outputs.past_key_values
    tts_cache = init_tts_cache()

    # 2. 负条件编码（用于 CFG）
    neg_outputs = self.tts_language_model(neg_input_ids, ...)
    neg_condition = neg_outputs.last_hidden_state[:, -1:]

    # 3. 主生成循环
    num_text_windows = len(tts_text_ids) // TTS_TEXT_WINDOW_SIZE
    for i in range(num_text_windows):
        # 3a. 文本窗口输入
        text_window = tts_text_ids[i*TTS_TEXT_WINDOW_SIZE : (i+1)*TTS_TEXT_WINDOW_SIZE]

        # 3b. 下层 LM 编码文本
        lm_outputs = self.language_model(text_window, past_key_values=lm_cache, use_cache=True)
        lm_cache = lm_outputs.past_key_values
        lm_hidden = lm_outputs.last_hidden_state

        # 3c. 上层 TTS LM 编码（注入 LM 隐状态，标记为文本类型）
        type_embed = self.tts_input_types(torch.ones_like(text_window))  # type=1 文本
        tts_input = lm_hidden + type_embed
        tts_outputs = self.tts_language_model(
            inputs_embeds=tts_input,
            past_key_values=tts_cache,
            use_cache=True
        )
        tts_cache = tts_outputs.past_key_values
        positive_condition = tts_outputs.last_hidden_state[:, -1:]

        # 3d. 生成 6 个语音 token
        for j in range(TTS_SPEECH_WINDOW_SIZE):
            # 扩散采样（CFG）
            speech_latent = self.sample_speech_tokens(
                positive_condition=positive_condition,
                negative_condition=neg_condition,
                cfg_scale=cfg_scale
            )

            # 解码为音频
            audio = self.acoustic_tokenizer.decode(
                speech_latent.transpose(1, 2),
                cache=acoustic_cache
            )
            audio_streamer.put(audio)  # 流式输出

            # 语音 token 反馈到 TTS LM
            acoustic_embed = self.acoustic_connector(speech_latent)
            type_embed = self.tts_input_types(torch.zeros(...))  # type=0 语音
            tts_input = acoustic_embed + type_embed
            tts_outputs = self.tts_language_model(
                inputs_embeds=tts_input,
                past_key_values=tts_cache,
                use_cache=True
            )
            tts_cache = tts_outputs.past_key_values
            positive_condition = tts_outputs.last_hidden_state[:, -1:]

            # EOS 检测
            eos_logit = torch.sigmoid(self.tts_eos_classifier(positive_condition))
            if eos_logit > 0.5:
                finished = True
                break

    audio_streamer.end()
```

#### 4.6.4 Classifier-Free Guidance 采样

```python
def sample_speech_tokens(self, positive_condition, negative_condition, cfg_scale=3.0):
    # 初始化随机噪声
    speech = torch.randn(B, 1, acoustic_vae_dim)

    # 获取 DPM-Solver 时间步
    timesteps = self.noise_scheduler.timesteps

    for t in timesteps:
        # 正负条件各一份
        combined_speech = cat([speech, speech])
        combined_condition = cat([positive_condition, negative_condition])
        combined_t = cat([t, t])

        # 扩散头预测
        eps = self.prediction_head(combined_speech, combined_t, combined_condition)
        cond_eps, uncond_eps = eps.chunk(2)

        # CFG 引导
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        # DPM-Solver 步进
        speech = self.noise_scheduler.step(guided_eps, t, speech).prev_sample

    return speech[:B]  # 只取正条件部分
```

### 4.7 音频流式输出（`streamer.py`）

#### 4.7.1 AudioStreamer

```python
class AudioStreamer:
    def __init__(self, batch_size):
        self.queues = [Queue() for _ in range(batch_size)]
        self._stopped = [False] * batch_size

    def put(self, audio_chunk, batch_idx=0):
        self.queues[batch_idx].put(audio_chunk)

    def end(self, batch_idx=0):
        self.queues[batch_idx].put(None)  # 哨兵值
        self._stopped[batch_idx] = True

    def __iter__(self):
        while True:
            chunk = self.queues[0].get()
            if chunk is None:
                break
            yield chunk
```

#### 4.7.2 AsyncAudioStreamer

```python
class AsyncAudioStreamer:
    def __init__(self, batch_size, loop=None):
        self.queues = [asyncio.Queue() for _ in range(batch_size)]
        self.loop = loop or asyncio.get_event_loop()

    def put(self, audio_chunk, batch_idx=0):
        self.loop.call_soon_threadsafe(
            self.queues[batch_idx].put_nowait, audio_chunk
        )

    async def __aiter__(self):
        while True:
            chunk = await self.queues[0].get()
            if chunk is None:
                break
            yield chunk
```

- 使用 `loop.call_soon_threadsafe` 实现线程安全的音频放入
- 适用于 FastAPI/WebRTC 等异步场景

### 4.8 数据处理器

#### 4.8.1 VibeVoiceProcessor（TTS）

**处理播客脚本的完整流程**：

1. **脚本解析**：`Speaker X: text` 格式
2. **构建 token 序列**：system prompt + voice input + text input + speech output
3. **语音位置标记**：在语音位置插入特殊 token

**特殊 Token 映射**（TTS 复用 Qwen2-VL 视觉 token）：

| 用途 | TTS Token | ASR Token |
|------|-----------|-----------|
| 语音开始 | `<|vision_start|>` | `<|object_ref_start|>` |
| 语音结束 | `<|vision_end|>` | `<|object_ref_end|>` |
| 语音填充 | `<|vision_pad|>` | `<|box_start|>` |
| 填充 | `<|image_pad|>` | `<|image_pad|>` |

**语音 token 序列结构**：
```
<|vision_start|> <|vision_pad|> × N <|vision_end|>
```
其中 N = 音频时长 × 7.5（帧率）

**语音样本处理**：
- 加载音频文件 → 归一化 → 填充到相同长度
- 创建 `speech_input_mask` 标记 `<|vision_pad|>` 位置

#### 4.8.2 VibeVoiceASRProcessor（ASR）

**处理 ASR 输入的流程**：

1. **音频加载**：支持 ffmpeg 和 soundfile
2. **重采样**：到 24kHz
3. **dB FS 归一化**：使用 AudioNormalizer
4. **构建聊天模板**：
   ```
   <|im_start|>system
   You are a speech recognition assistant.<|im_end|>
   <|im_start|>user
   <|object_ref_start|><|box_start|> × N <||object_ref_end|>
   The audio lasts for {duration:.1f} seconds.
   Please transcribe the audio into text.
   Context info: {context_info}<|im_end|>
   <|im_start|>assistant
   ```
5. **创建 `acoustic_input_mask`**：标记 `<|box_start|>` 位置
6. **热词支持**：通过 `context_info` 参数传入自定义热词

**音频时长计算**：
```python
num_speech_tokens = int(audio_duration * 7.5)  # 7.5 Hz 帧率
```

#### 4.8.3 VibeVoiceStreamingProcessor（流式 TTS）

**特殊之处**：
- `__call__` 故意未实现（`raise NotImplementedError`）
- 必须使用 `process_input_with_cached_prompt` 方法
- 接受预计算的 KV Cache（`cached_prompt`），避免重复编码 voice prompt
- 输出包含 `tts_lm_input_ids` 和 `tts_text_ids` 两套 token 序列

**`process_input_with_cached_prompt` 方法**：
```python
def process_input_with_cached_prompt(self, text, voice_prompt, cached_prompt=None, ...):
    # 构建 voice prompt token 序列
    voice_ids = tokenize(voice_prompt)
    # 构建 TTS 文本 token 序列
    tts_text_ids = tokenize(text)
    # 构建 TTS LM 输入 ID（包含 voice prompt + 文本）
    tts_lm_input_ids = cat([voice_ids, text_start_ids])

    return VibeVoiceStreamingProcessorOutput(
        tts_lm_input_ids=tts_lm_input_ids,
        tts_text_ids=tts_text_ids,
        ...
    )
```

#### 4.8.4 VibeVoiceTokenizerProcessor

分词器的统一封装，处理文本分词和特殊 token 的添加/移除。

#### 4.8.5 AudioNormalizer

```python
class AudioNormalizer:
    def __init__(self, target_dBFS=-25):
        self.target_dBFS = target_dBFS

    def __call__(self, audio):
        audio = tailor_dB_FS(audio, self.target_dBFS)
        audio = avoid_clipping(audio)
        return audio
```

- `tailor_dB_FS`：调整音频到目标 dB FS
- `avoid_clipping`：防止削波失真

### 4.9 扩散调度器（`dpm_solver.py`）

基于 diffusers 的 `DPMSolverMultistepScheduler`，约 800 行代码。

**支持的算法**：
- `dpmsolver++`：确定性 DPM-Solver++
- `sde-dpmsolver++`：随机 DPM-Solver++

**支持的阶数**：1、2、3 阶

**噪声调度**：
- `cosine` beta 调度（VibeVoice 默认）
- `linear` beta 调度
- `scaled_linear` beta 调度

**sigma 调度**：
- Karras sigmas：`σ = (σ_max^(1/ρ) + t/(T-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ`
- LU lambdas：基于对数均匀分布

**v-prediction 模式**：
```python
# v-prediction 目标
v = alpha_t * noise - sigma_t * original_sample

# 从 v 预测 x_0
x0 = alpha_t * v - sigma_t * noise_pred
```

### 4.10 时间步采样器（`timestep_sampler.py`）

**LossAwareTimestepSampler**：基于损失的时间步采样器

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

### 4.11 文本分词器（`modular_vibevoice_text_tokenizer.py`）

**VibeVoiceASRTextTokenizerFast**：基于 Qwen2 的快速文本分词器

```python
class VibeVoiceASRTextTokenizerFast(PreTrainedTokenizerFast):
    # 添加 ASR 特殊 token
    special_tokens = {
        "object_ref_start": "<|object_ref_start|>",
        "object_ref_end": "<|object_ref_end|>",
        "box_start": "<|box_start|>",
    }
```

### 4.12 vLLM 插件

#### 4.12.1 注册入口（`__init__.py`）

```python
def register():
    # 注册配置
    AutoConfig.register("vibevoice", VibeVoiceConfig)
    # 注册分词器
    AutoTokenizer.register(VibeVoiceConfig, fast_tokenizer_class=VibeVoiceASRTextTokenizerFast)
    # 注册 vLLM 模型
    ModelRegistry.register_model("VibeVoiceForCausalLM", "vibevoice")
```

通过 `pyproject.toml` 的入口点自动加载：
```toml
[project.entry-points."vllm.general_plugins"]
vibevoice = "vllm_plugin:register"
```

#### 4.12.2 模型封装（`model.py`）

**VibeVoiceForCausalLM**（1251行）：

```python
class VibeVoiceForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, config, ...):
        self.audio_encoder = VibeVoiceAudioEncoder(config)
        self.model = Qwen2Model(config.decoder_config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

**VibeVoiceAudioEncoder**：
```python
class VibeVoiceAudioEncoder(nn.Module):
    def __init__(self, config):
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(...)
        self.semantic_tokenizer = VibeVoiceAcousticTokenizerModel(...)
        self.acoustic_connector = SpeechConnector(...)
        self.semantic_connector = SpeechConnector(...)

    def forward(self, audio, ...):
        # 支持流式编码长音频
        acoustic_tokens = self.acoustic_tokenizer.encode(audio).sample()
        semantic_tokens = self.semantic_tokenizer.encode(audio).mean
        return self.acoustic_connector(acoustic_tokens) + self.semantic_connector(semantic_tokens)
```

**VibeVoiceMultiModalProcessor**：
```python
class VibeVoiceMultiModalProcessor:
    def __call__(self, prompt, audio_info, ...):
        # 将 <|AUDIO|> 占位符扩展为语音 token 序列
        num_tokens = int(audio_duration * 7.5)
        audio_tokens = "<|object_ref_start|>" + "<|box_start|>" * num_tokens + "<|object_ref_end|>"
        prompt = prompt.replace("<|AUDIO|>", audio_tokens)
        return prompt
```

**权重映射**：
```python
# vLLM 权重名 → VibeVoice 权重名
weight_mappings = {
    "audio_encoder.acoustic_tokenizer.": "model.acoustic_tokenizer.",
    "audio_encoder.semantic_tokenizer.": "model.semantic_tokenizer.",
    "audio_encoder.acoustic_connector.": "model.acoustic_connector.",
    "audio_encoder.semantic_connector.": "model.semantic_connector.",
}
```

#### 4.12.3 音频输入映射（`inputs.py`）

```python
class VibeVoiceAudioInputMapper:
    def __call__(self, audio_input):
        # 支持多种输入格式
        if isinstance(audio_input, str):  # 文件路径
            audio = load_audio_with_ffmpeg(audio_input)
        elif isinstance(audio_input, bytes):  # 字节数据
            audio = decode_audio_bytes(audio_input)
        elif isinstance(audio_input, np.ndarray):  # numpy 数组
            audio = audio_input

        # 归一化
        audio = AudioNormalizer()(audio)

        # 时长限制
        max_duration = int(os.environ.get("VIBEVOICE_MAX_AUDIO_DURATION", 3660))
        if len(audio) / 24000 > max_duration:
            audio = audio[:max_duration * 24000]

        return torch.tensor(audio, dtype=torch.float32)
```

**FFmpeg 补丁**：替换 vLLM 默认的 AudioMediaIO，确保 24kHz 重采样。

#### 4.12.4 分词器文件生成工具（`tools/generate_tokenizer_files.py`）

```python
def generate_tokenizer_files(model_path, output_dir):
    # 1. 从 Qwen2.5 模型下载基本 tokenizer 文件
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 添加 VibeVoice 音频特殊 token
    special_tokens = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 3. 更新 tokenizer_config.json 和 tokenizer.json
    # 4. 生成 added_tokens.json 和 special_tokens_map.json
    # 5. 适配自定义聊天模板以支持音频输入
    tokenizer.save_pretrained(output_dir)
```

#### 4.12.5 一键部署脚本（`scripts/start_server.py`）

功能包括：
1. 安装系统依赖（FFmpeg 等）
2. 安装 Python 包和下载模型
3. 生成 tokenizer 文件
4. 启动单个或多个 vLLM 服务器实例（支持数据并行）
5. 支持使用 nginx 进行负载均衡
6. 优雅关闭服务并处理进程信号

### 4.13 演示脚本

#### 4.13.1 ASR 文件推理（`vibevoice_asr_inference_from_file.py`，581行）

支持批量推理：
- 从 JSONL 数据集加载音频和文本
- 支持单样本和批量处理
- 输出 WER/CER 评估指标
- 支持热词（context_info）

#### 4.13.2 ASR Gradio 演示（`vibevoice_asr_gradio_demo.py`，1268行）

功能丰富的 Web 界面：
- 上传/录制音频
- 实时转写
- 音频片段提取
- 流式输出
- 支持 MP3 和 WAV 格式
- 热词输入

#### 4.13.3 流式 TTS 文件推理（`realtime_model_inference_from_file.py`，311行）

- 从文本文件读取内容
- 支持说话人音色映射
- 输出音频文件和生成指标（延迟、RTF 等）

#### 4.13.4 Web 实时 TTS（`web/app.py`，517行）

FastAPI + WebSocket 实时 TTS 服务：
- `StreamingTTSService` 类封装模型加载和语音生成
- `/stream` WebSocket 端点提供实时音频流
- 支持配置参数：CFG 缩放、推理步骤、语音预设
- 语音缓存机制
- 线程安全的并发请求处理

### 4.14 微调（`finetuning-asr/`）

#### 4.14.1 LoRA 微调脚本（`lora_finetune.py`，538行）

```python
# 使用 PEFT 进行参数高效微调
model = VibeVoiceASRForConditionalGeneration.from_pretrained(base_model)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 只微调注意力层
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 训练循环
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

#### 4.14.2 LoRA 推理脚本（`inference_lora.py`，234行）

```python
# 加载基础模型 + LoRA 权重
model = VibeVoiceASRForConditionalGeneration.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # 合并权重

# 推理
result = model.generate(...)
```

---

## 五、关键实现细节

### 5.1 Token 压缩比计算

```
encoder_ratios = [8, 5, 5, 4, 2, 2]
compress_ratio = 8 × 5 × 5 × 4 × 2 × 2 = 3200
frame_rate = 24000 / 3200 = 7.5 Hz
```

即每秒音频仅产生 7.5 个 token，60 分钟音频产生 2700 个 token。

### 5.2 特殊 Token 映射

| 用途 | TTS Token | ASR Token |
|------|-----------|-----------|
| 语音开始 | `<|vision_start|>` | `<|object_ref_start|>` |
| 语音结束 | `<|vision_end|>` | `<|object_ref_end|>` |
| 语音填充 | `<|vision_pad|>` | `<|box_start|>` |
| 填充 | `<|image_pad|>` | `<|image_pad|>` |

复用 Qwen2-VL 的视觉 token 作为语音 token，避免修改词表。

### 5.3 语音特征归一化

```python
# 首次遇到语音数据时计算
scaling_factor = 1 / std(audio_tokens)  # 归一化方差
bias_factor = -mean(audio_tokens)        # 归一化均值

# 应用归一化
audio_features = (audio_tokens + bias_factor) * scaling_factor
```

支持分布式训练时通过 `all_reduce` 同步。

### 5.4 流式 TTS 的窗口机制

```
TTS_TEXT_WINDOW_SIZE = 5    # 每次输入 5 个文本 token
TTS_SPEECH_WINDOW_SIZE = 6  # 每次生成 6 个语音 token
```

文本和语音交替输入，实现流式文本输入和实时语音输出。

每个窗口的延迟分析：
- 5 个文本 token 对应 5/7.5 = 0.667 秒语音
- 6 个语音 token 对应 6/7.5 = 0.8 秒语音
- 首音频延迟 ≈ 语音编码 + 首窗口 LLM + 扩散采样 + 音频解码 ≈ 300ms

### 5.5 EOS 检测

Streaming TTS 使用 `BinaryClassifier` 预测是否结束：
```python
class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, 1)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x  # sigmoid后 > 0.5 则结束
```

### 5.6 Transformers 版本兼容

代码中大量处理了 transformers >= 4.57 的兼容性问题：

**MockCacheLayer**：
```python
class MockCacheLayer:
    """为新版 DynamicCache 提供 layers 接口"""
    def __init__(self):
        self.self_attn = None
        self.cross_attn = None
        self.is_updated = {}
```

**_ensure_cache_has_layers**：
```python
def _ensure_cache_has_layers(cache, num_layers):
    """确保缓存对象具有所需属性"""
    if not hasattr(cache, 'layers'):
        cache.layers = [MockCacheLayer() for _ in range(num_layers)]
```

**_init_cache_for_generation**：
```python
def _init_cache_for_generation(model, batch_size, dtype, ...):
    """根据 transformers 版本选择缓存初始化方式"""
    if hasattr(model, '_get_cache'):
        return model._get_cache('dynamic', batch_size, ...)
    else:
        return DynamicCache()
```

### 5.7 分布式训练支持

语音特征缩放因子的分布式同步：
```python
if torch.distributed.is_initialized():
    # 在所有 GPU 间同步缩放因子
    torch.distributed.all_reduce(scaling_factor, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(bias_factor, op=torch.distributed.ReduceOp.AVG)
```

### 5.8 vLLM 自动恢复机制

`test_api_auto_recover.py` 实现了模型重复输出的自动恢复：
1. 检测重复模式（连续相同 token）
2. 自动重试，调整采样参数（temperature、top_p）
3. 保存已处理的有效内容
4. 在重试时恢复上下文

---

## 六、数据流图

### 6.1 ASR 推理数据流

```
音频文件
  → ffmpeg 加载 → 24kHz 重采样 → dB归一化
  → VibeVoiceASRProcessor: 构建 input_ids + acoustic_input_mask
  → AcousticTokenizer.encode → sample → acoustic_connector
  → SemanticTokenizer.encode → mean → semantic_connector
  → 合并嵌入替换 acoustic_input_mask 位置
  → Qwen2 LLM 自回归生成
  → lm_head → token IDs → 解码为文本
```

### 6.2 TTS 训练数据流

```
播客脚本 + 语音样本
  → VibeVoiceProcessor: 解析脚本，构建 token 序列
  → AcousticTokenizer.encode → sample → acoustic_connector
  → SemanticTokenizer.encode → mean → semantic_connector
  → 替换语音位置嵌入
  → Qwen2 LLM 前向传播
  → CE Loss (文本) + Diffusion Loss (语音)
```

### 6.3 流式 TTS 推理数据流

```
文本 + 说话人音色
  → VibeVoiceStreamingProcessor: 构建 input_ids + tts_text_ids
  → 预填充 voice prompt (缓存 KV Cache)
  → 循环:
      → 文本窗口 → language_model → hidden_states
      → tts_language_model(注入 hidden_states, type=text)
      → 扩散采样(positive + negative condition, CFG)
      → AcousticTokenizer.decode(流式缓存) → 音频块
      → audio_streamer.put(音频块)
      → tts_language_model(注入 acoustic_embed, type=speech)
      → EOS 检测
  → 拼接所有音频块 → 最终音频
```

### 6.4 vLLM ASR 服务数据流

```
HTTP 请求 (音频文件/bytes)
  → VibeVoiceAudioInputMapper: 加载 → 归一化 → 时长限制
  → VibeVoiceMultiModalProcessor: <|AUDIO|> → 语音 token 序列
  → VibeVoiceAudioEncoder: 双分词器编码 → 连接器投影
  → Qwen2Model: 自回归生成
  → lm_head → token IDs → 解码为文本
  → HTTP 响应 (转写文本)
```

---

## 七、模块间依赖关系

```
                    ┌─────────────────────┐
                    │  configuration_      │
                    │  vibevoice.py        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐  ┌─────▼──────┐  ┌──────▼──────────────┐
    │ modular_vibe-  │  │ modeling_  │  │ configuration_vibe-  │
    │ voice_tokenizer│  │ vibevoice  │  │ voice_streaming.py   │
    └───────┬────────┘  └─────┬──────┘  └──────────┬──────────┘
            │                 │                     │
            │     ┌───────────┼───────────┐        │
            │     │           │           │        │
    ┌───────▼─────▼──┐ ┌─────▼─────┐ ┌───▼────────▼────────┐
    │ modular_vibe-  │ │ modeling_ │ │ modeling_vibevoice_  │
    │ voice_diffu-  │ │ vibevoice │ │ streaming.py         │
    │ sion_head.py  │ │ _asr.py   │ └──────────┬───────────┘
    └───────┬────────┘ └───────────┘            │
            │                                    │
    ┌───────▼────────┐              ┌───────────▼───────────┐
    │ schedule/      │              │ modeling_vibevoice_   │
    │ dpm_solver.py  │              │ streaming_inference.py│
    └────────────────┘              └───────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │ processor/                                           │
    │ ├── vibevoice_processor.py      ← TTS 数据处理       │
    │ ├── vibevoice_asr_processor.py  ← ASR 数据处理       │
    │ ├── vibevoice_streaming_processor.py ← 流式数据处理  │
    │ ├── vibevoice_tokenizer_processor.py ← 分词器处理    │
    │ └── audio_utils.py             ← 音频工具            │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │ vllm_plugin/                                         │
    │ ├── __init__.py    ← 注册入口                        │
    │ ├── model.py       ← 模型封装（依赖 modular/）       │
    │ └── inputs.py      ← 音频输入映射（依赖 processor/） │
    └──────────────────────────────────────────────────────┘
```

---

## 八、总结

VibeVoice 是一个设计精良的语音 AI 系统，其核心创新在于：

1. **统一架构**：ASR 和 TTS 共享分词器和 LLM 骨干，仅在上层任务头有差异
2. **超低帧率**：7.5 Hz 分词器使长音频处理成为可能（60分钟仅 2700 token）
3. **Next-Token Diffusion**：巧妙结合 LLM 的上下文理解能力和扩散模型的生成能力
4. **流式设计**：从分词器（流式卷积缓存）到生成流程（窗口机制）全面支持流式处理
5. **工程化**：vLLM 插件、Transformers 集成、版本兼容、一键部署等工程细节处理完善
6. **可扩展性**：LoRA 微调支持、热词机制、多说话人支持
