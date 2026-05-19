# 第五章：流式 TTS 模型与推理分析

> 本章深入分析 VibeVoice 的流式 TTS 模型（`modeling_vibevoice_streaming.py` + `modeling_vibevoice_streaming_inference.py`），包括 LLM 分层设计、窗口机制、CFG 采样、EOS 检测、音频流式输出等核心机制。

---

## 5.1 流式 TTS 的设计目标

- **低延迟**：首音频延迟 ~300ms
- **流式输入**：支持逐句输入文本
- **流式输出**：音频逐块生成，边生成边播放
- **实时性**：生成速度 >= 播放速度（RTF < 1）

---

## 5.2 模型定义（VibeVoiceStreamingModel）

### 5.2.1 整体架构

```
VibeVoiceStreamingModel
├── language_model: Qwen2 (下层，文本编码)
│   └── num_hidden_layers = total - tts_backbone_num_hidden_layers
│   └── norm = Identity (不使用最终归一化)
│
├── tts_language_model: Qwen2 (上层，TTS 生成)
│   └── num_hidden_layers = tts_backbone_num_hidden_layers
│   └── embed_tokens 未使用（接收 language_model 的隐状态）
│
├── tts_input_types: Embedding(2, hidden_size)
│   └── 0 = 语音位置, 1 = 文本位置
│
├── acoustic_tokenizer + acoustic_connector
├── prediction_head (Diffusion Head)
├── noise_scheduler (DPM-Solver)
└── tts_eos_classifier (BinaryClassifier)
```

### 5.2.2 模型初始化

```python
class VibeVoiceStreamingModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        # 下层 LLM（文本编码）
        self.language_model = Qwen2ForCausalLM(config.decoder_config)
        self.language_model.model.layers = self.language_model.model.layers[:num_lm_layers]
        self.language_model.model.norm = nn.Identity()  # 不使用最终归一化

        # 上层 TTS LLM（语音生成）
        self.tts_language_model = Qwen2ForCausalLM(config.decoder_config)
        self.tts_language_model.model.layers = self.tts_language_model.model.layers[num_lm_layers:]
        self.tts_language_model.model.embed_tokens = None  # 不使用嵌入层

        # 输入类型嵌入
        self.tts_input_types = nn.Embedding(2, hidden_size)

        # 语音分词器（仅 Acoustic）
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(...)
        self.acoustic_connector = SpeechConnector(...)

        # 扩散头
        self.prediction_head = VibeVoiceDiffusionHead(...)
        self.noise_scheduler = DPMShellSolverMultistepScheduler(...)

        # EOS 分类器
        self.tts_eos_classifier = BinaryClassifier(hidden_size)
```

### 5.2.3 LLM 分层设计

**为什么将 LLM 分为两层？**

1. **职责分离**：下层专注文本理解，上层专注语音生成
2. **效率优化**：下层处理文本时可以使用 KV Cache，避免重复计算
3. **训练灵活性**：可以冻结下层，只微调上层
4. **推理优化**：下层和上层可以并行处理不同的窗口

**下层 LLM 的特殊处理**：
- `norm = Identity`：不使用最终归一化，因为隐状态需要直接传递给上层
- 只使用部分 Transformer 层

**上层 TTS LLM 的特殊处理**：
- `embed_tokens = None`：不使用嵌入层，直接接收下层的隐状态
- 通过 `tts_input_types` 嵌入区分文本和语音输入

---

## 5.3 输入类型嵌入

```python
self.tts_input_types = nn.Embedding(2, hidden_size)
# 0 = 语音位置, 1 = 文本位置
```

### 5.3.1 使用方式

```python
# 文本输入
type_embed = self.tts_input_types(torch.ones(*text_shape, dtype=torch.long))  # type=1
tts_input = lm_hidden + type_embed

# 语音输入
type_embed = self.tts_input_types(torch.zeros(*speech_shape, dtype=torch.long))  # type=0
tts_input = acoustic_embed + type_embed
```

### 5.3.2 设计意义

上层 TTS LLM 需要区分输入是来自下层的文本隐状态还是来自连接器的语音特征。类型嵌入让模型能够学习不同的处理策略：
- 文本输入：理解语义，准备生成条件
- 语音输入：维护语音上下文，指导后续生成

---

## 5.4 BinaryClassifier EOS 检测

### 5.4.1 模块定义

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

### 5.4.2 为什么使用 BinaryClassifier 而非标准 EOS？

- **流式场景**：标准 EOS token 需要模型在 token 级别预测结束，但流式 TTS 的结束点是语音级别
- **灵活性**：BinaryClassifier 基于隐状态判断是否结束，比 token 级预测更灵活
- **训练**：使用 BCE Loss，标签为 0（继续）或 1（结束）

### 5.4.3 推理时的 EOS 判断

```python
eos_logit = torch.sigmoid(self.tts_eos_classifier(positive_condition))
if eos_logit > 0.5:
    finished = True
```

---

## 5.5 推理流程详解

### 5.5.1 核心常量

```python
TTS_TEXT_WINDOW_SIZE = 5     # 每次输入 5 个文本 token
TTS_SPEECH_WINDOW_SIZE = 6   # 每次生成 6 个语音 token
```

### 5.5.2 窗口机制

每个窗口的延迟分析：
- 5 个文本 token 对应 5/7.5 = 0.667 秒语音
- 6 个语音 token 对应 6/7.5 = 0.8 秒语音
- 生成速度 > 播放速度（0.8s 语音 / ~0.1s 计算），满足实时性

### 5.5.3 generate 方法完整流程

```python
def generate(self, tts_lm_input_ids, tts_text_ids, audio_streamer,
             cfg_scale=3.0, num_inference_steps=20, ...):
    # ========================================
    # Phase 1: 预填充 voice prompt
    # ========================================
    lm_outputs = self.language_model(tts_lm_input_ids, use_cache=True)
    lm_cache = lm_outputs.past_key_values
    tts_cache = init_tts_cache()

    # ========================================
    # Phase 2: 编码负条件（用于 CFG）
    # ========================================
    neg_input_ids = torch.zeros_like(tts_text_ids[:1])  # 空文本
    neg_lm_outputs = self.language_model(neg_input_ids, use_cache=True)
    neg_lm_hidden = neg_lm_outputs.last_hidden_state
    neg_type_embed = self.tts_input_types(torch.ones_like(neg_input_ids))
    neg_tts_input = neg_lm_hidden + neg_type_embed
    neg_tts_outputs = self.tts_language_model(inputs_embeds=neg_tts_input)
    neg_condition = neg_tts_outputs.last_hidden_state[:, -1:]

    # ========================================
    # Phase 3: 主生成循环
    # ========================================
    num_text_windows = len(tts_text_ids) // TTS_TEXT_WINDOW_SIZE
    acoustic_cache = VibeVoiceTokenizerStreamingCache()
    finished = False

    for i in range(num_text_windows):
        if finished:
            break

        # 3a. 文本窗口输入
        text_window = tts_text_ids[i*TTS_TEXT_WINDOW_SIZE : (i+1)*TTS_TEXT_WINDOW_SIZE]

        # 3b. 下层 LM 编码文本
        lm_outputs = self.language_model(
            text_window,
            past_key_values=lm_cache,
            use_cache=True
        )
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
            if finished:
                break

            # 扩散采样（CFG）
            speech_latent = self.sample_speech_tokens(
                positive_condition=positive_condition,
                negative_condition=neg_condition,
                cfg_scale=cfg_scale,
                num_inference_steps=num_inference_steps
            )

            # 解码为音频
            audio = self.acoustic_tokenizer.decode(
                speech_latent.transpose(1, 2),
                cache=acoustic_cache
            )
            audio_streamer.put(audio)  # 流式输出

            # 语音 token 反馈到 TTS LM
            acoustic_embed = self.acoustic_connector(speech_latent)
            type_embed = self.tts_input_types(torch.zeros(1, 1, dtype=torch.long))  # type=0 语音
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

    audio_streamer.end()
```

---

## 5.6 Classifier-Free Guidance 采样

### 5.6.1 sample_speech_tokens 方法

```python
def sample_speech_tokens(self, positive_condition, negative_condition, cfg_scale=3.0, num_inference_steps=20):
    B = positive_condition.shape[0]
    acoustic_vae_dim = self.config.acoustic_vae_dim

    # 初始化随机噪声
    speech = torch.randn(B, 1, acoustic_vae_dim, device=positive_condition.device)

    # 设置 DPM-Solver 时间步
    self.noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = self.noise_scheduler.timesteps

    for t in timesteps:
        # 正负条件各一份
        combined_speech = torch.cat([speech, speech])
        combined_condition = torch.cat([positive_condition, negative_condition])
        combined_t = torch.cat([t.expand(B), t.expand(B)])

        # 扩散头预测
        eps = self.prediction_head(combined_speech, combined_t, combined_condition)
        cond_eps, uncond_eps = eps.chunk(2)

        # CFG 引导
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        # DPM-Solver 步进
        speech = self.noise_scheduler.step(guided_eps, t, speech).prev_sample

    return speech[:B]  # 只取正条件部分
```

### 5.6.2 CFG 原理

CFG 通过对比**有条件预测**和**无条件预测**的差异来增强条件控制：

```
guided_prediction = unconditional + cfg_scale * (conditional - unconditional)
```

- `cfg_scale = 1.0`：等价于普通条件生成
- `cfg_scale > 1.0`：增强条件的影响，生成更符合文本的语音
- `cfg_scale < 1.0`：减弱条件的影响，增加多样性

VibeVoice 默认 `cfg_scale = 3.0`，这是一个较强的引导强度。

### 5.6.3 负条件的构建

```python
neg_input_ids = torch.zeros_like(tts_text_ids[:1])  # 空文本
```

负条件使用空文本（全零 token），代表"无条件"的情况。这样 CFG 引导的对比是"有文本条件"vs"无文本条件"。

---

## 5.7 音频流式输出

### 5.7.1 AudioStreamer

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

**设计特点**：
- 每个 batch 样本一个独立队列，支持批量生成
- 使用 `None` 作为哨兵值标记流结束
- 提供 `__iter__` 接口，可以用 `for chunk in streamer` 逐块获取

### 5.7.2 AsyncAudioStreamer

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

**设计特点**：
- 使用 `asyncio.Queue` 替代 `Queue`
- `loop.call_soon_threadsafe` 实现线程安全的音频放入（生成线程 → 异步主线程）
- 适用于 FastAPI/WebRTC 等异步场景

---

## 5.8 延迟分析

### 5.8.1 首音频延迟

```
首音频延迟 = voice_prompt编码 + 首窗口LLM + 扩散采样 + 音频解码
           ≈ 100ms          + 50ms       + 100ms     + 50ms
           ≈ 300ms
```

### 5.8.2 稳态延迟

```
稳态每窗口延迟 = LLM编码 + 扩散采样×6 + 音频解码×6
               ≈ 10ms   + 60ms       + 30ms
               ≈ 100ms

每窗口生成音频 = 6/7.5 = 0.8s
RTF = 0.1/0.8 = 0.125 << 1  （远快于实时）
```

### 5.8.3 延迟优化策略

1. **KV Cache**：voice prompt 只编码一次，后续复用
2. **窗口机制**：小窗口（5+6）减少单次计算量
3. **DPM-Solver**：20 步推理，比 DDPM 1000 步快 50 倍
4. **流式解码**：Acoustic Tokenizer 支持逐帧解码

---

## 5.9 数据流图

```
文本 + 说话人音色
  → VibeVoiceStreamingProcessor: 构建 input_ids + tts_text_ids
  → 预填充 voice prompt (缓存 KV Cache)
  → 循环:
      → 文本窗口(5 tokens) → language_model(下层) → hidden_states
      → + type_embed(text=1) → tts_language_model(上层) → condition
      → 扩散采样(positive + negative condition, CFG, 20步)
      → AcousticTokenizer.decode(流式缓存) → 音频块(0.8s)
      → audio_streamer.put(音频块)
      → acoustic_connector(speech_latent) + type_embed(speech=0)
      → tts_language_model(上层) → 更新 condition
      → EOS 检测 (BinaryClassifier)
  → audio_streamer.end()
  → 拼接所有音频块 → 最终音频
```

---

## 5.10 总结

流式 TTS 是 VibeVoice 架构最复杂的变体，其核心设计包括：

1. **LLM 分层**：下层文本编码 + 上层语音生成，职责分离
2. **窗口机制**：5 文本 + 6 语音交替，实现流式输入输出
3. **类型嵌入**：区分文本和语音输入，让上层 TTS LM 学习不同处理策略
4. **CFG 采样**：增强文本条件对语音生成的控制
5. **BinaryClassifier EOS**：隐空间级别的结束检测
6. **流式输出**：AudioStreamer/AsyncAudioStreamer 支持实时播放
7. **低延迟**：首音频 ~300ms，RTF << 1
