# 第六章：处理器、vLLM 插件与部署分析

> 本章深入分析 VibeVoice 的数据处理器（processor/）、vLLM 推理加速插件、演示脚本和微调流程，覆盖从数据处理到模型部署的完整工程链路。

---

## 6.1 数据处理器总览

VibeVoice 提供三种处理器，分别对应三种模型变体：

| 处理器 | 模型变体 | 特殊 Token | 输出 |
|--------|---------|-----------|------|
| VibeVoiceProcessor | TTS | `<\|vision_pad\|>` | input_ids + speech_input_mask + speech_tensors |
| VibeVoiceASRProcessor | ASR | `<\|box_start\|>` | input_ids + acoustic_input_mask + speech_tensors |
| VibeVoiceStreamingProcessor | Streaming TTS | 自定义 | tts_lm_input_ids + tts_text_ids |

---

## 6.2 VibeVoiceProcessor（TTS）

### 6.2.1 处理播客脚本的完整流程

1. **脚本解析**：`Speaker X: text` 格式
2. **构建 token 序列**：system prompt + voice input + text input + speech output
3. **语音位置标记**：在语音位置插入特殊 token
4. **语音样本处理**：加载 → 归一化 → 填充

### 6.2.2 特殊 Token 映射

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

### 6.2.3 语音样本处理

```python
def process_speech(self, speech_samples):
    # 加载音频文件
    audios = [load_audio(path) for path in speech_samples]
    # 归一化
    audios = [AudioNormalizer()(audio) for audio in audios]
    # 填充到相同长度
    max_len = max(len(a) for a in audios)
    audios = [np.pad(a, (0, max_len - len(a))) for a in audios]
    return torch.tensor(audios)
```

---

## 6.3 VibeVoiceASRProcessor（ASR）

### 6.3.1 处理 ASR 输入的流程

1. **音频加载**：支持 ffmpeg 和 soundfile
2. **重采样**：到 24kHz
3. **dB FS 归一化**：使用 AudioNormalizer
4. **构建聊天模板**
5. **创建 `acoustic_input_mask`**
6. **热词支持**

### 6.3.2 聊天模板格式

```
<|im_start|>system
You are a speech recognition assistant.<|im_end|>
<|im_start|>user
<|object_ref_start|><|box_start|> × N <|object_ref_end|>
The audio lasts for {duration:.1f} seconds.
Please transcribe the audio into text.
Context info: {context_info}<|im_end|>
<|im_start|>assistant
```

### 6.3.3 音频时长计算

```python
num_speech_tokens = int(audio_duration * 7.5)  # 7.5 Hz 帧率
```

### 6.3.4 热词支持

```python
def __call__(self, audio, context_info=None, ...):
    if context_info:
        prompt += f"\nContext info: {context_info}"
```

热词通过 `context_info` 参数传入，添加到 user message 中。这允许用户指定领域专有名词、人名等，提高转写准确率。

---

## 6.4 VibeVoiceStreamingProcessor（流式 TTS）

### 6.4.1 特殊之处

- `__call__` 故意未实现（`raise NotImplementedError`）
- 必须使用 `process_input_with_cached_prompt` 方法
- 接受预计算的 KV Cache（`cached_prompt`），避免重复编码 voice prompt
- 输出包含 `tts_lm_input_ids` 和 `tts_text_ids` 两套 token 序列

### 6.4.2 process_input_with_cached_prompt 方法

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

### 6.4.3 为什么需要两套 token 序列？

- `tts_lm_input_ids`：用于下层 LLM 的输入，包含 voice prompt + 文本
- `tts_text_ids`：仅包含需要生成语音的文本 token，按 TTS_TEXT_WINDOW_SIZE=5 分窗口

---

## 6.5 VibeVoiceTokenizerProcessor

分词器的统一封装，处理文本分词和特殊 token 的添加/移除。

---

## 6.6 AudioNormalizer

```python
class AudioNormalizer:
    def __init__(self, target_dBFS=-25):
        self.target_dBFS = target_dBFS

    def __call__(self, audio):
        audio = tailor_dB_FS(audio, self.target_dBFS)
        audio = avoid_clipping(audio)
        return audio
```

- `tailor_dB_FS`：调整音频到目标 dB FS（默认 -25）
- `avoid_clipping`：防止削波失真

**为什么需要归一化？**
- 不同来源的音频响度差异很大
- 统一响度有助于模型学习稳定的语音特征
- -25 dB FS 是语音处理的常用目标响度

---

## 6.7 vLLM 插件

### 6.7.1 注册入口（`__init__.py`）

```python
def register():
    AutoConfig.register("vibevoice", VibeVoiceConfig)
    AutoTokenizer.register(VibeVoiceConfig, fast_tokenizer_class=VibeVoiceASRTextTokenizerFast)
    ModelRegistry.register_model("VibeVoiceForCausalLM", "vibevoice")
```

通过 `pyproject.toml` 的入口点自动加载：
```toml
[project.entry-points."vllm.general_plugins"]
vibevoice = "vllm_plugin:register"
```

### 6.7.2 模型封装（`model.py`，1251行）

**VibeVoiceForCausalLM**：

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
        acoustic_tokens = self.acoustic_tokenizer.encode(audio).sample()
        semantic_tokens = self.semantic_tokenizer.encode(audio).mean
        return self.acoustic_connector(acoustic_tokens) + self.semantic_connector(semantic_tokens)
```

**VibeVoiceMultiModalProcessor**：

```python
class VibeVoiceMultiModalProcessor:
    def __call__(self, prompt, audio_info, ...):
        num_tokens = int(audio_duration * 7.5)
        audio_tokens = "<|object_ref_start|>" + "<|box_start|>" * num_tokens + "<|object_ref_end|>"
        prompt = prompt.replace("<|AUDIO|>", audio_tokens)
        return prompt
```

**权重映射**：

```python
weight_mappings = {
    "audio_encoder.acoustic_tokenizer.": "model.acoustic_tokenizer.",
    "audio_encoder.semantic_tokenizer.": "model.semantic_tokenizer.",
    "audio_encoder.acoustic_connector.": "model.acoustic_connector.",
    "audio_encoder.semantic_connector.": "model.semantic_connector.",
}
```

### 6.7.3 音频输入映射（`inputs.py`）

```python
class VibeVoiceAudioInputMapper:
    def __call__(self, audio_input):
        if isinstance(audio_input, str):      # 文件路径
            audio = load_audio_with_ffmpeg(audio_input)
        elif isinstance(audio_input, bytes):   # 字节数据
            audio = decode_audio_bytes(audio_input)
        elif isinstance(audio_input, np.ndarray):  # numpy 数组
            audio = audio_input

        audio = AudioNormalizer()(audio)

        max_duration = int(os.environ.get("VIBEVOICE_MAX_AUDIO_DURATION", 3660))
        if len(audio) / 24000 > max_duration:
            audio = audio[:max_duration * 24000]

        return torch.tensor(audio, dtype=torch.float32)
```

**FFmpeg 补丁**：替换 vLLM 默认的 AudioMediaIO，确保 24kHz 重采样。

### 6.7.4 分词器文件生成工具（`tools/generate_tokenizer_files.py`）

```python
def generate_tokenizer_files(model_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    special_tokens = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.save_pretrained(output_dir)
```

### 6.7.5 一键部署脚本（`scripts/start_server.py`）

功能包括：
1. 安装系统依赖（FFmpeg 等）
2. 安装 Python 包和下载模型
3. 生成 tokenizer 文件
4. 启动单个或多个 vLLM 服务器实例（支持数据并行）
5. 支持使用 nginx 进行负载均衡
6. 优雅关闭服务并处理进程信号

---

## 6.8 演示脚本

### 6.8.1 ASR 文件推理（`vibevoice_asr_inference_from_file.py`，581行）

支持批量推理：
- 从 JSONL 数据集加载音频和文本
- 支持单样本和批量处理
- 输出 WER/CER 评估指标
- 支持热词（context_info）

### 6.8.2 ASR Gradio 演示（`vibevoice_asr_gradio_demo.py`，1268行）

功能丰富的 Web 界面：
- 上传/录制音频
- 实时转写
- 音频片段提取
- 流式输出
- 支持 MP3 和 WAV 格式
- 热词输入

### 6.8.3 流式 TTS 文件推理（`realtime_model_inference_from_file.py`，311行）

- 从文本文件读取内容
- 支持说话人音色映射
- 输出音频文件和生成指标（延迟、RTF 等）

### 6.8.4 Web 实时 TTS（`web/app.py`，517行）

FastAPI + WebSocket 实时 TTS 服务：
- `StreamingTTSService` 类封装模型加载和语音生成
- `/stream` WebSocket 端点提供实时音频流
- 支持配置参数：CFG 缩放、推理步骤、语音预设
- 语音缓存机制
- 线程安全的并发请求处理

---

## 6.9 微调（`finetuning-asr/`）

### 6.9.1 LoRA 微调脚本（`lora_finetune.py`，538行）

```python
model = VibeVoiceASRForConditionalGeneration.from_pretrained(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 只微调注意力层
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

**LoRA 配置说明**：
- `r=8`：LoRA 秩为 8，在参数效率和表达能力之间取得平衡
- `lora_alpha=16`：缩放因子，实际缩放 = alpha/r = 2
- `target_modules=["q_proj", "v_proj"]`：只微调注意力的 Q 和 V 投影
- `lora_dropout=0.05`：5% 的 dropout 防止过拟合

### 6.9.2 LoRA 推理脚本（`inference_lora.py`，234行）

```python
model = VibeVoiceASRForConditionalGeneration.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # 合并权重

result = model.generate(...)
```

**merge_and_unload**：将 LoRA 权重合并到基础模型中，推理时无额外开销。

---

## 6.10 vLLM 自动恢复机制

`test_api_auto_recover.py`（638行）实现了模型重复输出的自动恢复：

1. **重复模式检测**：检测连续相同 token 的模式
2. **自动重试**：调整采样参数（temperature、top_p）打破循环
3. **内容保存**：保存已处理的有效内容
4. **上下文恢复**：在重试时恢复已转写的内容

---

## 6.11 文本分词器（`modular_vibevoice_text_tokenizer.py`）

```python
class VibeVoiceASRTextTokenizerFast(PreTrainedTokenizerFast):
    special_tokens = {
        "object_ref_start": "<|object_ref_start|>",
        "object_ref_end": "<|object_ref_end|>",
        "box_start": "<|box_start|>",
    }
```

基于 Qwen2 的快速文本分词器，添加了 ASR 特殊 token。

---

## 6.12 数据流图

### 6.12.1 vLLM ASR 服务数据流

```
HTTP 请求 (音频文件/bytes)
  → VibeVoiceAudioInputMapper: 加载 → 归一化 → 时长限制
  → VibeVoiceMultiModalProcessor: <|AUDIO|> → 语音 token 序列
  → VibeVoiceAudioEncoder: 双分词器编码 → 连接器投影
  → Qwen2Model: 自回归生成
  → lm_head → token IDs → 解码为文本
  → HTTP 响应 (转写文本)
```

### 6.12.2 Web 实时 TTS 数据流

```
WebSocket 连接
  → 接收文本 + 语音预设
  → VibeVoiceStreamingProcessor: 构建 tts_lm_input_ids + tts_text_ids
  → StreamingTTSService.generate():
      → 文本窗口 → LM → TTS LM → 扩散采样 → 音频解码
      → audio_streamer.put(音频块)
  → WebSocket 发送音频块
  → 客户端播放
```

---

## 6.13 总结

VibeVoice 的工程化设计完善：

1. **三种处理器**：针对 TTS/ASR/Streaming 的差异化数据处理
2. **vLLM 插件**：通过注册机制无缝集成，支持多模态推理服务化
3. **一键部署**：从依赖安装到服务启动的完整自动化
4. **LoRA 微调**：参数高效微调，适配特定领域
5. **多种演示**：Gradio/Web/FastAPI 覆盖不同使用场景
6. **自动恢复**：处理模型重复输出等异常情况
7. **音频归一化**：统一的响度处理确保推理稳定性
