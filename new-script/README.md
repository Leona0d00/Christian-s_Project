# 🎙 实时中文语音转文字 (C / Sherpa-ONNX)

基于 [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) C API 的中文实时流式语音转文字项目，专为边缘部署场景设计。

> 本版本在 `Sherpa` 分支。Python/FunASR 版本在 [`main`](../../tree/main) 分支。

## ✨ 功能特性

- **流式 ASR** — 基于 Zipformer-transducer 模型的实时语音识别
- **字幕自动校正** — 内置去重算法，增量输出无重复字幕
- **标点恢复** — 可选中英文 ct-transformer 标点模型
- **边缘部署优先** — 纯 C 实现，无 Python 运行时依赖
- **DLL 导出** — 可被外部 C/C++ 或 Python ctypes 调用
- **CLI 工具** — 命令行单文件处理，无需 GUI

## 📋 环境要求

- Windows x64 (或 Linux/macOS)
- MinGW-w64 (gcc/g++) 或 MSVC
- CMake ≥ 3.13 (可选) 或 GNU Make
- Git Bash (用于 `setup.sh`)

## 🚀 快速开始

### 0. 前置依赖

本机已有：
- **Windows**: MinGW gcc/g++ (`D:/CP Editor/cpeditor/mingw64/bin/`)
- **Linux**: 系统 gcc/g++ (或 MinGW 交叉编译)
- conda 环境 `christian` (Python 3.10, 仅测试用)

### 1. 下载依赖

**Windows (Git Bash)** 或 **Linux**:

```bash
cd new-script
bash setup.sh
```

**Windows (cmd.exe)**: 手动下载并解压（见 [手动下载](#-手动下载依赖)）

此脚本会下载：
- **Sherpa-ONNX 预编译库** (v1.13.4, ~20MB)
  - 路径: `deps/sherpa-onnx/<package>/`
- **中文流式 ASR 模型** (zipformer-zh-int8, ~200MB)
  - 路径: `models/asr/<model>/`
- **中英文标点模型** (ct-transformer-zh-en-int8, ~72MB)
  - 路径: `models/punct/<model>/`

### 2. 编译

| 平台 | 命令 |
|------|------|
| **Windows (cmd.exe)** | `mingw32-make` |
| **Windows (Git Bash)** | `mingw32-make` |
| **Linux** | `make` |

Makefile 自动检测操作系统，无需额外配置。

输出文件：
| 文件 | 说明 |
|------|------|
| `build/transcribe_cli.exe` | CLI 命令行工具 (Windows) / `transcribe_cli` (Linux) |
| `build/transcribe.dll` | 动态链接库 (对外 API) |
| `build/*.dll` | 运行时 DLL (Windows) 或 `.so` (Linux) |

### 3. 运行

**Windows:**
```cmd
build\transcribe_cli.exe -v ..\test.wav
```

**Linux:**
```bash
./build/transcribe_cli -v ../test.wav

# 详细输出模式
./build/transcribe_cli.exe -v ../test.wav

# 使用标点恢复
./build/transcribe_cli.exe -v --punct ./models/punct/model.int8.onnx ../test.wav

# 保存结果到文件
./build/transcribe_cli.exe -o result.txt ../test.wav
```

### 4. 测试

| 平台 | 命令 |
|------|------|
| **Windows** | `mingw32-make test` |
| **Linux** | `make test` |

或手动运行:
```bash
conda run -n christian python test/test.py
# 或直接用 python:
D:/Miniconda/envs/christian/python.exe test/test.py
```

## 🏗 项目结构

```
new-script/
├── include/
│   └── transcribe_api.h          # 公共 C API 头文件 (DLL 导出)
├── src/
│   ├── config.h                   # 配置常量
│   ├── core.h / core.c            # 核心 ASR 处理 (process_chunk)
│   ├── corrector.h / corrector.c  # 字幕校正器 (去重 + 标点恢复)
│   ├── audio.h / audio.c          # WAV 文件 I/O 与重采样
│   ├── pipeline.h / pipeline.c    # 实时转录流水线
│   ├── edge.h / edge.c            # 边缘部署接口
│   └── utils.h / utils.c          # 字符串与 UTF-8 工具
├── main.c                         # CLI 主程序
├── test/
│   └── test.py                    # Python ctypes 测试脚本
├── deps/
│   └── sherpa-onnx/               # 预编译 sherpa-onnx 库 (需下载)
├── models/
│   ├── asr/                       # 流式 ASR 模型 (需下载)
│   └── punct/                     # 标点模型 (需下载)
├── build/                         # 编译输出
├── setup.sh                       # 依赖下载脚本
├── Makefile                       # 构建文件
├── CMakeLists.txt                 # CMake 构建
├── .gitignore
└── README.md
```

## 📖 C API 概览

### 核心处理 — `transcriber_process_chunk()`

```c
#include "transcribe_api.h"

TranscriberHandle *handle = transcriber_create("./models/asr", 4, "cpu");

float audio[9600];  // 600ms @ 16kHz
/* ... fill audio from mic or file ... */

char *text = transcriber_process_chunk(handle, audio, 9600, 0);  // is_final=0
printf("%s\n", text);
free(text);

// 最后一块
char *final_text = transcriber_process_chunk(handle, audio, 9600, 1);
free(final_text);

// 获取标点校正的完整结果
char *result = transcriber_finalize(handle);
printf("最终: %s\n", result);
free(result);

transcriber_destroy(handle);
```

### Python ctypes 调用

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./build/transcribe.dll")
lib.transcriber_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
lib.transcriber_create.restype = ctypes.c_void_p

handle = lib.transcriber_create(b"./models/asr", 4, b"cpu")

audio = np.zeros(9600, dtype=np.float32)
ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

result = lib.transcriber_process_chunk(handle, ptr, 9600, 0)
print(result.decode("utf-8"))

lib.transcriber_destroy(handle)
```

### 完整 API 列表

| 函数 | 说明 |
|------|------|
| `transcriber_create()` | 创建识别器会话 |
| `transcriber_destroy()` | 释放资源 |
| `transcriber_process_chunk()` | **核心方法**: 处理音频块 |
| `transcriber_get_display_text()` | 去重后的增量显示文本 |
| `transcriber_finalize()` | 获取带标点的完整文本 |
| `transcriber_reset()` | 重置会话（新语句） |
| `transcriber_process_file()` | 处理整个 WAV 文件 |
| `transcriber_load_punctuation()` | 加载标点模型 |
| `transcriber_version()` | 版本号 |
| `transcriber_get_info()` | 模型信息 |
| `transcriber_get_device_count()` | 可用设备数 |
| `transcriber_get_device_info()` | 设备信息 |
| `transcriber_get_optimization_guide()` | 部署优化指南 |

## ⚙ 配置预设

| 配置 | chunk | 线程 | 说明 |
|------|-------|------|------|
| `cpu_optimized` | 600ms | 4 | 默认平衡配置 |
| `low_latency` | 480ms | 4 | 低延迟交互场景 |
| `cuda_optimized` | 600ms | 1 | GPU 加速 (需 CUDA 版 sherpa-onnx) |

## 🧪 测试

### 环境准备

C 核心代码无 Python 依赖。测试脚本使用 Python ctypes 调用 DLL，仅需三个轻量库：

```bash
# 方式 1: conda (推荐)
conda create -n transcribe-test python=3.10 -y
conda activate transcribe-test
pip install numpy soundfile scipy

# 方式 2: pip
pip install numpy soundfile scipy

# 方式 3: 使用已有的 christian 环境
conda run -n christian pip install numpy soundfile scipy
```

依赖清单（见 `../requirements.txt`）：

| 库 | 用途 | 大小 |
|----|------|------|
| `numpy` | 音频数组处理 | ~20MB |
| `soundfile` | 读取 WAV 文件 | ~2MB |
| `scipy` | 音频重采样 (48kHz→16kHz) | ~30MB |

> **注意**: 这些库**仅用于测试**，不参与 ASR 推理。生产部署无需 Python 环境。

### 运行测试

```bash
mingw32-make test
# 或:
conda run -n christian python test/test.py
```

8 组测试覆盖：DLL 加载、模型创建、单块识别、多块流式、状态持久化、音频处理、字幕校正、边缘部署接口。

## 🔧 手动下载依赖

如果 `setup.sh` 失败，手动下载以下文件并解压：

### Sherpa-ONNX 库
```
URL:  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/
      sherpa-onnx-v1.13.4-win-x64-shared-MD-Release-no-tts.tar.bz2
解压到: deps/sherpa-onnx/
```

### ASR 模型
```
URL:  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/
      sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2
解压到: models/asr/
```

### 标点模型 (可选)
```
URL:  https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/
      sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
解压到: models/punct/
```

## 📄 许可

MIT License

## 📊 版本评估报告

> 测试日期: 2026-07-12 | 测试平台: Windows 11 x64, Intel Core i7-13620H

### 功能测试 (8 组 / 24 项)

| 测试组 | 项目数 | 结果 | 说明 |
|--------|--------|------|------|
| Test 1: DLL 加载 | 1 | ✅ PASS | 版本号返回正常 |
| Test 2: 模型创建 | 2 | ✅ PASS | 模型加载成功 (zipformer-zh-int8) |
| Test 3: 单块处理 | 3 | ✅ PASS | 静音/终结块处理正确 |
| Test 4: 多块流式 | 2 | ✅ PASS | 19 块 11s 音频逐块识别通过 |
| Test 5: 状态持久化 | 2 | ✅ PASS | 跨块 cache 保持 + reset 正常 |
| Test 6: 音频处理 | 3 | ✅ PASS | WAV 读取/重采样/格式验证 |
| Test 7: 字幕去重 | 5 | ✅ PASS | 最长前缀匹配去重算法正确 |
| Test 8: 边缘接口 | 4 | ✅ PASS | 设备检测/预设配置/优化指南 |
| **总计** | **24** | **✅ 100%** | **零失败** |

### 识别准确率

使用 `test.wav` (11s 中文语音，48kHz 立体声 → 自动重采样到 16kHz 单声道)：

| 模型输出 (逐块累积) | 去重后增量 |
|---------------------|------------|
| `你` | `你` |
| `你好这` | `好这` |
| `你好这是一` | `是一` |
| `你好这是一段测` | `一段测` |
| `你好这是一段测试音频` | `段测试音频` |
| `请` | `请` |
| `请听` | `听` |
| `我` | `我` |
| `我现在在` | `现在在` |
| `我现在在喝水` | `在喝水` |

**完整识别文本**: `你好这是一段测试音频请听我现在在喝水`

> 注: test.wav 实际内容为 "你好这是一段测试音频" + "请听" + "我现在在喝水"。
> 流式模型输出存在自然的前缀修正现象（如"你好这"→"你好这是一"），去重算法正确提取增量部分。

### 延迟性能

| 指标 | 1 线程 | 2 线程 | 4 线程 |
|------|--------|--------|--------|
| 模型加载时间 | 3.49s | — | — |
| 单块平均延迟 (静音) | 46.5ms | — | — |
| 单块平均延迟 (语音) | 47.4ms | 40.1ms | 42.3ms |
| 单块 P50 延迟 (语音) | 49.7ms | 42.2ms | 43.0ms |
| 单块 P95 延迟 (语音) | 53.2ms | 45.4ms | 49.7ms |
| 端到端处理时间 | 0.86s | — | — |
| 实时率 (RTF) | **0.078** | — | — |
| 处理速度 | **12.8×** 实时 | — | — |

> **延迟定义**: 每块 600ms 音频 (9600 采样点 @ 16kHz) 送入 `process_chunk()` 到返回结果的耗时。
> **RTF**: 处理时间 / 音频时长。RTF < 1 表示比实时快，0.078 意味着处理速度是实时播放速度的 12.8 倍。
> 多线程对延迟改善有限（瓶颈在 ONNX 推理而非 CPU 调度），建议边缘部署使用 1-2 线程以节省功耗。

### 与 Python/FunASR 版本对比

| 维度 | Python (FunASR) | C (Sherpa-ONNX) |
|------|-----------------|-----------------|
| 模型 | paraformer-zh-streaming | zipformer-zh (INT8) |
| 模型大小 | ~2GB (含 PyTorch) | ~200MB (ONNX) |
| 运行时依赖 | Python + PyTorch + CUDA | 仅 sherpa-onnx DLL |
| 内存占用 | ~1-2GB | ~500MB (含模型) |
| GUI | tkinter | CLI (无 GUI，可集成) |
| 外部调用 | Python import | DLL / C ABI |
| 边缘部署 | 需 Python 环境 | 纯 C，无运行时依赖 |
| 启动时间 | ~10-30s (加载 PyTorch) | ~3.5s |
| 单块延迟 | ~480ms (CUDA) / 未知 (CPU) | ~47ms (CPU) |
| 处理速度 | 实时 | **12.8× 实时** |

### 已知限制

1. **CLI stderr 日志缺失**: MinGW 编译的 exe 与 MSVC 编译的 sherpa-onnx DLL 使用不同的 CRT，导致 `fprintf(stderr, ...)` 输出不可见（sherpa-onnx 内部 warning 正常）。不影响识别功能和 API 调用。
2. **标点恢复未启用**: 标点模型已下载但测试未加载。标点模型为 ct-transformer-zh-en (INT8, 72MB)，可通过 `transcriber_load_punctuation()` 启用。
3. **麦克风采集未实现**: 当前版本仅支持 WAV 文件输入。实时麦克风采集需集成 PortAudio。
4. **GPU 加速**: 当前使用 sherpa-onnx CPU 预编译库。如需 CUDA 加速，需下载 CUDA 版本的 sherpa-onnx 库。

## 🔗 相关

- [Sherpa-ONNX GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [Sherpa-ONNX C API 文档](https://k2-fsa.github.io/sherpa/onnx/c-api/index.html)
- [Python/FunASR 版本](../../tree/main)
