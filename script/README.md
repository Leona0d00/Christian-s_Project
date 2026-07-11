# 🎙 实时中文语音转文字

基于 [FunASR](https://github.com/modelscope/FunASR) `paraformer-zh-streaming` 模型的实时中文语音转文字应用，提供低延迟流式识别、字幕校正和现代化 GUI 界面。

## ✨ 功能特性

- **实时流式识别** — 边说话边转写，低延迟（CUDA 下 ~480ms）
- **字幕自动校正** — 内置去重算法 + ct-punc 标点恢复模型
- **现代化 GUI** — 自定义主题，录音状态指示器，计时器
- **设备自适应** — CUDA 自动选用低延迟配置，CPU 自动降级
- **边缘部署就绪** — 预留 ONNX 导出接口，4 种预设部署配置
- **即插即用** — 自动检测麦克风和计算设备

## 📋 环境要求

- Python ≥ 3.8
- CUDA ≥ 11.0（可选，用于 GPU 加速）
- 麦克风设备

### 快速安装

```bash
# 创建 conda 环境
conda create -n christian python=3.10 -y
conda activate christian

# 安装依赖
pip install -r requirements.txt
```

依赖包括 `funasr`、`torch`、`pyaudio`、`modelscope` 等，详见 `requirements.txt`。

## 🚀 使用方式

### 启动 GUI 应用

```bash
python script/main.py
```

启动后：
1. 选择麦克风和计算设备（CPU / CUDA）
2. 点击 **开始识别** 开始录音
3. 实时字幕出现在文本区域
4. 点击 **停止识别** 结束并显示标点校正后的完整文本
5. 点击 **清空** 清除文本区域

### 运行测试

```bash
python script/Test.py
```

使用 `test.wav` 模拟实时音频流，执行 8 组共 26 项功能测试。

## 🏗 项目结构

```
.
├── script/
│   ├── main.py                  # GUI 主程序
│   ├── Test.py                  # 功能测试脚本
│   └── Transcribe/              # 核心模块
│       ├── __init__.py          # 包入口，导出所有公共接口
│       ├── _config.py           # 配置常量（采样率、chunk 尺寸、预设等）
│       ├── _core.py             # 核心 ASR 推理 (process_chunk)
│       ├── _pipeline.py         # 实时转录流水线 (RealTimeTranscriber)
│       ├── _corrector.py        # 字幕校正器（去重 + 标点恢复）
│       ├── _audio.py            # 麦克风采集与 WAV 文件处理
│       └── _edge.py             # 边缘部署接口（设备检测、ONNX 导出）
├── requirements.txt             # Python 依赖
├── test.wav                     # 测试用音频文件
└── README.md
```

## 📖 API 概览

### 核心处理 — `process_chunk()`

独立于外部状态的核心 ASR 方法，可直接被测试脚本调用：

```python
from script.Transcribe import process_chunk
import numpy as np

cache = {}
audio = np.zeros(9600, dtype=np.float32)  # 600ms @ 16kHz

result = process_chunk(audio, cache, is_final=False)
print(result["text"])  # 识别文本
```

### 实时转录 — `RealTimeTranscriber`

完整的流水线整合：

```python
from script.Transcribe import RealTimeTranscriber

t = RealTimeTranscriber(device="cuda", chunk_size=[0, 5, 3])
t.start()

while t.is_running():
    for text in t.get_results():
        print(text, end="", flush=True)

final = t.stop()  # 标点校正后的完整文本
print(f"\n最终: {final}")
```

### 字幕校正 — `SubtitleCorrector`

```python
from script.Transcribe import SubtitleCorrector

corrector = SubtitleCorrector()
display = corrector.correct_chunk("欢迎大家")  # 自动去重
final = corrector.finalize(apply_punctuation=True)  # 标点恢复
```

### 边缘部署

```python
from script.Transcribe import (
    list_available_devices,   # 检测可用设备
    select_device,           # 自动选择设备
    get_edge_deployment_config,  # 获取预设部署配置
    list_edge_profiles,      # 列出所有预设
    get_optimization_guide,  # 部署优化指南
)
```

## ⚙ 延迟配置

| 模式 | chunk 配置 | 延迟 (E2E) | 适用场景 |
|------|-----------|-----------|---------|
| CUDA 低延迟 | `[0, 5, 3]` | ~480ms | RTX 4060+ GPU |
| CPU 低延迟 | `[0, 8, 4]` | ~480ms | 中端 CPU |
| CPU 标准 | `[0, 10, 5]` | ~600ms | 低端 CPU |
| ONNX 边缘 | `[0, 10, 5]` | ~600ms | 树莓派/Jetson |

GUI 会自动根据设备选择合适的配置。

## 🧪 测试

```bash
python script/Test.py
```

8 组测试覆盖：模型加载、单块识别、多块流式识别、cache 持久性、音频重采样、麦克风检测、字幕校正、边缘部署接口。

## 🔧 已知问题

- 首次运行需从 ModelScope 下载模型（~2GB），需要网络连接
- Windows 终端建议设置 `PYTHONIOENCODING=utf-8` 避免编码问题
- PyAudio 在部分 Linux 发行版需额外安装 `portaudio` 开发库

## 📄 许可

MIT License
