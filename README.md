# 🎙 实时中文语音转文字

本项目包含两个版本的中文实时语音转文字实现：

| 分支 | 语言 | ASR 引擎 | 界面 | 适用场景 |
|------|------|----------|------|----------|
| **[main](../../tree/main)** | Python | FunASR paraformer-zh-streaming | tkinter GUI | 桌面应用，快速原型 |
| **[Sherpa](../../tree/Sherpa)** | C | Sherpa-ONNX Zipformer-zh | CLI + DLL | 边缘部署，嵌入式系统 |

---

## Sherpa 分支 (C / Sherpa-ONNX) ← 当前分支

**核心特点**: 纯 C 实现，无 Python 运行时依赖，专为边缘部署设计。

- 基于 sherpa-onnx 预编译 C API，无需 PyTorch
- 流式 Zipformer-transducer 模型 (INT8 量化)
- 字幕去重校正 + 可选标点恢复模型
- 导出 DLL 供外部 C/C++/Python 调用
- 内存占用 ~300MB (含 INT8 模型，远小于 PyTorch 版本)

📖 详细文档：[new-script/README.md](./new-script/README.md)

### 快速开始

```bash
# 下载依赖 (Git Bash 或 Linux)
cd new-script && bash setup.sh

# 编译 (Windows: mingw32-make, Linux: make)
mingw32-make

# 运行
./build/transcribe_cli.exe -v ../test.wav

# 测试
mingw32-make test
```

---

## Main 分支 (Python / FunASR)

**核心特点**: 基于 FunASR 的 paraformer-zh-streaming 模型，提供 GUI 桌面应用。

- 实时流式识别 (~480ms CUDA 延迟)
- 字幕自动校正 (去重 + ct-punc 标点恢复)
- tkinter GUI 界面 (录音按钮、设备选择、计时器)
- Python conda 环境，依赖 PyTorch

📖 详细文档：参见 main 分支的 README.md

### 快速开始

```bash
git checkout main
conda run -n christian python script/main.py   # 启动 GUI
conda run -n christian python script/Test.py   # 运行测试
```

---

## 版本选择指南

| 需求 | 推荐版本 |
|------|----------|
| 桌面 GUI，快速体验 | main (Python) |
| 边缘设备部署 (树莓派/Jetson/嵌入式) | Sherpa (C) |
| 低内存、无 GPU 环境 | Sherpa (C) |
| 需要 Python 生态集成 | main (Python) |
| 集成到 C/C++ 项目中 | Sherpa (C) |
| 研究/实验/自定义模型 | main (Python) |

---

## 📄 许可

MIT License
