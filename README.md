# 🎙 实时中文语音转文字

本项目包含两个版本的中文实时语音转文字实现：

| 分支 | 语言 | ASR 引擎 | 界面 | 适用场景 |
|------|------|----------|------|----------|
| **[main](../../tree/main)** | Python | FunASR paraformer-zh-streaming | tkinter GUI | 桌面应用，快速原型 |
| **[Sherpa](../../tree/Sherpa)** | C | Sherpa-ONNX Zipformer-zh | CLI + DLL | 边缘部署，嵌入式系统 |

---

## Main 分支 (Python / FunASR) ← 当前分支

**核心特点**: 基于 FunASR paraformer-zh-streaming 模型，提供现代化 GUI 桌面应用。

- 实时流式识别，CUDA 下 ~480ms 低延迟
- 字幕自动校正（去重 + ct-punc 标点恢复模型）
- tkinter GUI（录音按钮、设备选择、计时器、状态指示）
- 设备自适应（CUDA / CPU 自动降级）
- 预留 ONNX 导出接口，4 种边缘部署预设配置
- 即插即用，自动检测麦克风和计算设备

📖 详细文档：[script/README.md](./script/README.md)

### 快速开始

```bash
# 创建 conda 环境
conda create -n christian python=3.10 -y
conda activate christian
pip install -r requirements.txt

# 启动 GUI
python script/main.py

# 运行测试
python script/Test.py
```

### 项目结构

```
.
├── script/
│   ├── main.py                  # GUI 主程序
│   ├── Test.py                  # 功能测试脚本 (8 组 26 项)
│   ├── README.md                # script 模块详细文档
│   └── Transcribe/              # 核心模块
│       ├── _core.py             # 核心 ASR 推理 (process_chunk)
│       ├── _pipeline.py         # 实时转录流水线
│       ├── _corrector.py        # 字幕校正器（去重 + 标点恢复）
│       ├── _audio.py            # 麦克风采集与 WAV 文件处理
│       └── _edge.py             # 边缘部署接口
├── requirements.txt             # Python 依赖
├── test.wav                     # 测试用音频文件
└── README.md
```

### API 概览

```python
from script.Transcribe import process_chunk, RealTimeTranscriber, SubtitleCorrector

# 核心处理
cache = {}
result = process_chunk(audio_chunk, cache, is_final=False)

# 实时转录
t = RealTimeTranscriber(device="cuda")
t.start()
final = t.stop()  # 标点校正后的完整文本

# 字幕校正
c = SubtitleCorrector()
display = c.correct_chunk("欢迎大家")  # 自动去重
```

---

## Sherpa 分支 (C / Sherpa-ONNX)

**核心特点**: 纯 C 实现，无 Python 运行时依赖，专为边缘部署设计。

- 基于 sherpa-onnx 预编译 C API，无需 PyTorch
- 流式 Zipformer-transducer 模型 (INT8 量化，~200MB)
- CLI 命令行工具 + DLL 导出（C/C++/Python 可调用）
- 内存占用 ~500MB，处理速度 12.8× 实时

📖 详细文档：切换到 Sherpa 分支后查看 `new-script/README.md`

### 快速开始

```bash
git checkout Sherpa
cd new-script
bash setup.sh      # 下载依赖 + 模型
mingw32-make       # 编译
./build/transcribe_cli.exe -v ../test.wav   # 运行
mingw32-make test  # 测试
```

---

## 版本选择指南

| 需求 | 推荐版本 |
|------|----------|
| 桌面 GUI，快速体验 | **main** (Python) |
| 边缘设备部署 (树莓派/Jetson/嵌入式) | **Sherpa** (C) |
| 低内存、无 GPU 环境 | **Sherpa** (C) |
| 需要 Python 生态集成 | **main** (Python) |
| 集成到 C/C++ 项目中 | **Sherpa** (C) |
| 研究/实验/自定义模型 | **main** (Python) |

## 📄 许可

MIT License
