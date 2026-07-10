"""
配置常量模块
定义所有采样率、chunk大小、模型名称等核心配置参数。
"""

# ================= 音频配置 =================
SAMPLE_RATE = 16000                     # FunASR 要求 16kHz
CHUNK_STRIDE_MS = 600                   # 每块 600ms
CHUNK_STRIDE_SAMPLES = SAMPLE_RATE * CHUNK_STRIDE_MS // 1000  # 9600 采样点
PYAUDIO_CHUNK = 1024                    # PyAudio 每次读取帧数
PYAUDIO_FORMAT = "int16"               # PyAudio 采样格式
CHANNELS = 1                            # 单声道

# ================= 流式模型配置 =================
# chunk_size: [lookback_frames, current_frames, lookahead_frames]
# 1 frame = 60ms, 所以 [0, 10, 5] = 600ms 当前 + 300ms 前瞻
CHUNK_SIZE_MS = 60                      # 每帧时长(ms)
CHUNK_SIZE_CONFIG = [0, 10, 5]          # 默认 chunk 配置
ENCODER_CHUNK_LOOK_BACK = 4             # 编码器回看帧数
DECODER_CHUNK_LOOK_BACK = 1             # 解码器回看帧数

# ================= 模型名称 =================
MODEL_NAME = "paraformer-zh-streaming"
MODEL_REVISION = "v2.0.4"
PUNC_MODEL_NAME = "ct-punc"
VAD_MODEL_NAME = "fsmn-vad"

# ================= 设备配置 =================
DEFAULT_DEVICE = "cpu"

# ================= 延迟优化常量 =================
CUDA_CHUNK_SIZE_CONFIG = [0, 5, 3]      # CUDA 低延迟 chunk (300ms+180ms)
READ_CHUNK_POLL_INTERVAL_SEC = 0.01     # 音频读取轮询间隔(秒)

# ================= GUI 配置 =================
WINDOW_TITLE = "实时中文语音转文字"
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 500
FONT_FAMILY = "Microsoft YaHei"
FONT_SIZE = 14
POLL_INTERVAL_MS = 50                   # GUI 轮询间隔

# ================= 边缘部署预设配置 =================
EDGE_PROFILES = {
    "cpu_optimized": {
        "chunk_size": [0, 10, 5],
        "device": "cpu",
        "latency_ms": 600,
        "description": "默认配置，平衡延迟与准确率",
    },
    "low_latency": {
        "chunk_size": [0, 8, 4],
        "device": "cpu",
        "latency_ms": 480,
        "description": "低延迟模式，适合交互场景",
    },
    "cuda_optimized": {
        "chunk_size": [0, 10, 5],
        "device": "cuda",
        "latency_ms": 600,
        "description": "GPU 加速模式",
    },
    "cuda_low_latency": {
        "chunk_size": [0, 5, 3],
        "device": "cuda",
        "latency_ms": 300,
        "description": "CUDA 低延迟模式（需要 GPU）",
    },
    "onnx_edge": {
        "chunk_size": [0, 10, 5],
        "device": "cpu",
        "latency_ms": 600,
        "description": "ONNX Runtime 边缘部署模式",
    },
}
