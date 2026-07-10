"""
Transcribe — 实时中文语音转文字模块

提供完整的实时语音识别能力：
- process_chunk(): 核心 ASR 处理方法（独立可测）
- RealTimeTranscriber: 完整的实时转录流水线
- SubtitleCorrector: 字幕校正（去重 + 标点恢复）
- MicrophoneStream: 麦克风音频采集
- 边缘部署相关配置和接口

Usage:
    from script.Transcribe import RealTimeTranscriber

    transcriber = RealTimeTranscriber(device="cpu")
    transcriber.start()
    # ... 从队列读取结果 ...
    transcriber.stop()
"""

# ===== 核心处理（最重要：供外部测试脚本直接调用） =====
from ._core import process_chunk, get_model_info, reset_model

# ===== 音频采集 =====
from ._audio import (
    MicrophoneStream,
    list_microphones,
    auto_detect_microphone,
    read_wav_file,
    resample_audio,
    stereo_to_mono,
)

# ===== 字幕校正 =====
from ._corrector import SubtitleCorrector, reset_punc_model

# ===== 转录流水线 =====
from ._pipeline import RealTimeTranscriber

# ===== 边缘部署 =====
from ._edge import (
    list_available_devices,
    select_device,
    get_edge_deployment_config,
    list_edge_profiles,
    export_onnx,
    get_optimization_guide,
)

# ===== 配置常量 =====
from ._config import (
    SAMPLE_RATE,
    CHUNK_STRIDE_MS,
    CHUNK_STRIDE_SAMPLES,
    CHUNK_SIZE_CONFIG,
    CUDA_CHUNK_SIZE_CONFIG,
    DEFAULT_DEVICE,
    EDGE_PROFILES,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    POLL_INTERVAL_MS,
    READ_CHUNK_POLL_INTERVAL_SEC,
)

__all__ = [
    # Core
    "process_chunk",
    "get_model_info",
    "reset_model",
    # Audio
    "MicrophoneStream",
    "list_microphones",
    "auto_detect_microphone",
    "read_wav_file",
    "resample_audio",
    "stereo_to_mono",
    # Corrector
    "SubtitleCorrector",
    "reset_punc_model",
    # Pipeline
    "RealTimeTranscriber",
    # Edge
    "list_available_devices",
    "select_device",
    "get_edge_deployment_config",
    "list_edge_profiles",
    "export_onnx",
    "get_optimization_guide",
    # Config
    "SAMPLE_RATE",
    "CHUNK_STRIDE_MS",
    "CHUNK_STRIDE_SAMPLES",
    "CHUNK_SIZE_CONFIG",
    "CUDA_CHUNK_SIZE_CONFIG",
    "DEFAULT_DEVICE",
    "EDGE_PROFILES",
    "WINDOW_TITLE",
    "WINDOW_WIDTH",
    "WINDOW_HEIGHT",
    "POLL_INTERVAL_MS",
    "READ_CHUNK_POLL_INTERVAL_SEC",
]
