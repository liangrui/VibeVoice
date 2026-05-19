# VibeVoice TTS 使用指南

## 一、模型总览

VibeVoice 提供三种 TTS 模型变体：

| 模型 | 路径 | LLM 骨干 | 上下文 | 最大时长 | 模型大小 | 状态 |
|------|------|---------|--------|---------|---------|------|
| VibeVoice-1.5B | `microsoft/VibeVoice-1.5B` | Qwen2.5-1.5B | 64K | ~90 分钟 | ~5.4 GB | HuggingFace 公开 |
| VibeVoice-Large | `aoi-ot/VibeVoice-Large` | Qwen2.5-7B/9B | 32K | ~45 分钟 | ~17.6 GB | 微软已下架，社区镜像可用 |
| VibeVoice-Realtime-0.5B | `microsoft/VibeVoice-Realtime-0.5B` | Qwen2.5-0.5B（分层） | - | 实时流式 | ~2.0 GB | HuggingFace 公开 |

### 选择建议

| 场景 | 推荐模型 |
|------|---------|
| 长播客（>45分钟） | VibeVoice-1.5B（64K上下文） |
| 短对话/高质量 | VibeVoice-Large（更好的韵律和稳定性） |
| 12/16GB 显卡 | VibeVoice-1.5B 或 Large + NF4量化 |
| 中文语音 | VibeVoice-Large（更稳定） |
| 实时流式 | VibeVoice-Realtime-0.5B |

---

## 二、流式 TTS（VibeVoice-Realtime-0.5B）

这是项目中有完整推理脚本的 TTS 方式，支持实时流式输出，首音频延迟 ~300ms。

### 2.1 安装

```bash
pip install -e ".[vllm]"
```

### 2.2 命令行推理

```bash
python demo/realtime_model_inference_from_file.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --txt_path demo/text_examples/1p_vibevoice.txt \
    --speaker_name Frank \
    --output_dir ./outputs \
    --cfg_scale 1.5
```

### 2.3 Python 代码示例

```python
import torch
import copy
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)

# 1. 加载处理器和模型
model_path = "microsoft/VibeVoice-Realtime-0.5B"
processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)
model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()
model.set_ddpm_inference_steps(num_steps=5)

# 2. 加载说话人音色（预计算的 KV Cache）
voice_path = "demo/voices/streaming_model/en-Frank_man.pt"
with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
    cached_prompt = torch.load(voice_path, map_location="cuda", weights_only=True)

# 3. 准备输入
text = "Hello, welcome to the VibeVoice demo. This is a streaming text-to-speech example."
inputs = processor.process_input_with_cached_prompt(
    text=text,
    cached_prompt=cached_prompt,
    padding=True,
    return_tensors="pt",
    return_attention_mask=True,
)

# 4. 移到 GPU
for k, v in inputs.items():
    if torch.is_tensor(v):
        inputs[k] = v.to("cuda")

# 5. 生成语音
outputs = model.generate(
    **inputs,
    cfg_scale=1.5,
    tokenizer=processor.tokenizer,
    generation_config={"do_sample": False},
    verbose=True,
    all_prefilled_outputs=copy.deepcopy(cached_prompt),
)

# 6. 保存音频
processor.save_audio(
    outputs.speech_outputs[0],
    output_path="output.wav",
)
```

### 2.4 可用的说话人音色

项目预置了 25 种音色（`demo/voices/streaming_model/`）：

| 语言 | 音色名 | 性别 |
|------|--------|------|
| 英文 | Frank, Mike, Carter, Davis | 男 |
| 英文 | Emma, Grace | 女 |
| 英文 | Samuel | 男（印度口音） |
| 中文 | Spk0 | 女 |
| 中文 | Spk1 | 男 |
| 日文 | Spk0 | 男 |
| 日文 | Spk1 | 女 |
| 韩文 | Spk0 | 女 |
| 韩文 | Spk1 | 男 |
| 法文 | Spk0 | 男 |
| 法文 | Spk1 | 女 |
| 德文 | Spk0 | 男 |
| 德文 | Spk1 | 女 |
| 意大利文 | Spk0 | 女 |
| 意大利文 | Spk1 | 男 |
| 荷兰文 | Spk0 | 男 |
| 荷兰文 | Spk1 | 女 |
| 波兰文 | Spk0 | 男 |
| 波兰文 | Spk1 | 女 |
| 葡萄牙文 | Spk0 | 女 |
| 葡萄牙文 | Spk1 | 男 |
| 西班牙文 | Spk0 | 女 |
| 西班牙文 | Spk1 | 男 |

### 2.5 Web 实时演示

```bash
# 启动 FastAPI + WebSocket 服务
python demo/vibevoice_realtime_demo.py \
    --model_path microsoft/VibeVoice-Realtime-0.5B \
    --port 3000
```

然后在浏览器中访问 `http://localhost:3000`，可以实时输入文本并听到流式语音输出。

### 2.6 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cfg_scale` | 1.5 | CFG 引导强度，越大越符合文本，但多样性降低 |
| `num_steps` | 5 | 扩散推理步数，越少越快但质量略降 |
| `attn_implementation` | flash_attention_2 | 注意力实现，SDPA 兼容但质量可能降低 |
| `torch_dtype` | bfloat16 | GPU 用 bfloat16，MPS/CPU 用 float32 |

---

## 三、标准 TTS（VibeVoice-1.5B）

> ⚠️ 官方文档中标注 "Installation and Usage: Disabled due to widespread misuse"，但模型权重仍在 HuggingFace 上公开。

标准 TTS 模型支持**长篇多说话人播客生成**，最长可达 90 分钟，最多 4 个说话人。

### 3.1 Python 代码示例

```python
import torch
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# 1. 加载模型
model_path = "microsoft/VibeVoice-1.5B"
processor = VibeVoiceProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()

# 2. 准备播客脚本（多说话人格式）
script = """
Speaker 1: Welcome to today's podcast about AI technology.
Speaker 2: Thanks for having me! I'm excited to discuss this topic.
Speaker 1: Let's start with the basics. What is VibeVoice?
Speaker 2: VibeVoice is a novel framework for generating expressive, long-form conversational audio.
"""

# 3. 处理输入
inputs = processor(
    text=script,
    speech_samples=["voice_speaker1.wav", "voice_speaker2.wav"],
    padding=True,
    return_tensors="pt",
)

# 4. 生成语音（使用 transformers 的 generate 方法）
outputs = model.generate(
    **inputs.to("cuda"),
    max_new_tokens=4096,
)

# 5. 保存音频
processor.save_audio(outputs.speech_outputs[0], output_path="podcast.wav")
```

### 3.2 播客脚本格式

标准 TTS 使用 `Speaker X: text` 格式：

```
Speaker 1: Hello everyone, welcome to our tech podcast.
Speaker 2: Today we're discussing the future of AI speech synthesis.
Speaker 1: That's right. Let me introduce our guest speaker.
Speaker 3: Hi, I'm glad to be here.
```

- 每行一个说话人的发言
- `Speaker X` 中的数字对应不同的说话人
- 最多支持 4 个不同说话人
- 说话人音色通过 `speech_samples` 参数传入对应的音频样本

---

## 四、VibeVoice-Large（7B/9B 高质量模型）

### 4.1 基本信息

| 属性 | VibeVoice-Large | VibeVoice-1.5B（对比） |
|------|-----------------|----------------------|
| LLM 骨干 | Qwen2.5-7B/9B | Qwen2.5-1.5B |
| 上下文长度 | 32K | 64K |
| 最大生成长度 | ~45 分钟 | ~90 分钟 |
| 模型大小 | ~17.6 GB | ~5.4 GB |
| 说话人数 | 最多 4 人 | 最多 4 人 |
| 语言 | 英文、中文 | 英文、中文 |

### 4.2 历史背景

`microsoft/VibeVoice-Large` 最初由微软在 HuggingFace 上公开发布（MIT 协议），但在 **2025年9月5日** 微软删除了官方 GitHub 仓库和 HuggingFace 上的 Large 模型权重（与删除 `modeling_vibevoice_inference.py` 推理脚本是同一轮 RAI 清理）。社区在删除前一天（2025-09-04）做了完整备份，目前可通过以下镜像获取：

- **Modelscope**: `microsoft/VibeVoice-Large`（社区重新上传）
- **HuggingFace 镜像**: `aoi-ot/VibeVoice-Large`
- **整合包**: `AEmotionStudio/vibevoice-models`（包含 `tts-large/` 子目录）

### 4.3 与 1.5B 的关键差异

1. **更大的 LLM 骨干** → 更好的韵律和说话人一致性，语音质量更高
2. **更短的上下文**（32K vs 64K）→ 最大生成长度减半（45分钟 vs 90分钟）
3. **3倍模型大小和显存** → 需要 NF4 量化才能在 12/16GB 显卡上运行
4. **更稳定** → 中文语音更稳定，意外背景音乐/BGM 出现概率更低
5. **更强的涌现能力** → 更容易展现自发唱歌等涌现行为

### 4.4 使用方式

VibeVoice-Large 的使用方式与 1.5B 完全相同（相同的 `Speaker N: text` 脚本格式、相同的语音克隆流程），只是模型路径不同：

```python
import torch
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# 加载 Large 模型（从社区镜像）
model_path = "aoi-ot/VibeVoice-Large"  # 或本地路径
processor = VibeVoiceProcessor.from_pretrained(model_path)

# 大模型建议使用 bfloat16 + flash_attention_2
model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

# 准备播客脚本
script = """
Speaker 1: Welcome to today's podcast about AI technology.
Speaker 2: Thanks for having me! I'm excited to discuss this topic.
Speaker 1: Let's start with the basics. What is VibeVoice?
Speaker 2: VibeVoice is a novel framework for generating expressive, long-form conversational audio.
"""

# 处理输入
inputs = processor(
    text=script,
    speech_samples=["voice_speaker1.wav", "voice_speaker2.wav"],
    padding=True,
    return_tensors="pt",
)

# 生成语音
outputs = model.generate(
    **inputs.to("cuda"),
    max_new_tokens=4096,
)

# 保存音频
processor.save_audio(outputs.speech_outputs[0], output_path="podcast.wav")
```

### 4.5 显存不足时的量化方案

对于 12GB/16GB 显卡，可以使用 bitsandbytes NF4 量化（~8GB 工作集）：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = VibeVoiceForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
)
```

---

## 五、标准 TTS vs 流式 TTS 对比

| 特性 | 标准TTS (1.5B/Large) | 流式TTS (0.5B) |
|------|----------------------|----------------|
| 模型类 | `VibeVoiceForConditionalGeneration` | `VibeVoiceStreamingForConditionalGenerationInference` |
| 最大长度 | ~90分钟(1.5B) / ~45分钟(Large) | 实时流式 |
| 说话人数 | 最多4人 | 单人 |
| 输入格式 | 播客脚本 `Speaker X: text` | 纯文本 |
| 音色来源 | 语音样本文件 (.wav) | 预计算 KV Cache (.pt) |
| 输出方式 | 一次性生成 | 流式逐块输出 |
| 首音频延迟 | 较高（需等全部文本编码完成） | ~300ms |
| 官方推理脚本 | 已禁用 | ✓ 完整推理脚本 |
| 扩散推理步数 | 20步 | 5步 |
| CFG Scale | - | 1.5（默认） |

---

## 六、官方使用建议

根据 `docs/vibevoice-tts.md` 的 FAQ 和 Tips：

### 6.1 中文语音不稳定

- 即使是中文文本也建议使用**英文标点**（仅逗号和句号）
- 推荐使用 **Large 模型**，"Using the Large model variant, which is considerably more stable"
- 如果语速过快，将文本拆分为多个 turn，使用相同的 speaker label

### 6.2 意外背景音乐/BGM

- 如果 voice prompt 包含背景音乐，生成语音更可能出现 BGM
- Large 模型更稳定，"The Large model is more stable and has a lower probability of generating unexpected background music"
- 某些引导词（"Welcome to"、"Hello"、"However"）可能触发 BGM
- 这是模型的内容感知特性，非 bug

### 6.3 唱歌能力

- 训练数据**不包含任何音乐数据**
- 唱歌是模型的**涌现能力**（emergent capability）
- Large 模型比 1.5B 更容易展现唱歌行为
- 唱歌可能走调

### 6.4 跨语言迁移

- 模型具有跨语言迁移能力（如保留口音）
- 但性能不稳定，可能需要多次采样才能获得满意结果

### 6.5 情感控制

- 社区用户 [PsiPi](https://huggingface.co/PsiPi) 发现了情感控制的方法
- 详见 [HuggingFace Discussion #12](https://huggingface.co/microsoft/VibeVoice-1.5B/discussions/12)

---

## 七、风险与限制

- 高质量合成语音可能被滥用于伪造、欺诈或传播虚假信息
- 仅支持英文和中文，其他语言可能产生意外输出
- 不处理背景噪声、音乐或其他音效
- 不支持重叠语音生成
- 微软不建议在商业或实际应用中使用，仅供研究和开发
- 使用 AI 生成内容时应披露其来源
