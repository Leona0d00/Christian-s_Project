"""
音频采集与处理模块
提供麦克风流式采集、WAV 文件读取、音频重采样等基础功能。
"""

import logging
import threading
import time
import numpy as np
from typing import Optional, List, Callable

from ._config import (
    SAMPLE_RATE,
    CHUNK_STRIDE_SAMPLES,
    PYAUDIO_CHUNK,
    CHANNELS,
    PYAUDIO_FORMAT,
    READ_CHUNK_POLL_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)


# ================= 麦克风设备枚举 =================

def list_microphones() -> List[dict]:
    """
    列出所有可用的音频输入设备。

    Returns:
        list[dict]: 每个元素包含 index, name, max_input_channels, default_sample_rate
                    如果未找到设备或 PyAudio 不可用，返回空列表
    """
    devices = []
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        default_host_api = p.get_default_host_api_info()
        default_input_index = default_host_api.get("defaultInputDevice", None)

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info.get("name", f"Device {i}"),
                    "max_input_channels": info.get("maxInputChannels", 0),
                    "default_sample_rate": int(info.get("defaultSampleRate", SAMPLE_RATE)),
                    "is_default": (i == default_input_index),
                })
        p.terminate()
    except ImportError:
        logger.warning("PyAudio 未安装，无法枚举麦克风设备。")
    except Exception as e:
        logger.warning(f"枚举麦克风设备时出错: {e}")

    return devices


def auto_detect_microphone() -> Optional[dict]:
    """
    自动检测并返回默认输入设备。

    Returns:
        dict 或 None: 默认设备信息，如果没有可用设备则返回 None
    """
    devices = list_microphones()
    if not devices:
        return None

    # 优先返回系统默认设备
    for d in devices:
        if d.get("is_default"):
            return d

    # 否则返回第一个可用设备
    return devices[0]


# ================= 麦克风流式采集 =================

class MicrophoneStream:
    """
    麦克风实时音频流采集器。

    使用 PyAudio 阻塞模式从麦克风采集音频，
    在后台线程中运行，自动处理 int16→float32 转换。

    Usage:
        stream = MicrophoneStream()
        stream.start()
        while stream.is_running():
            chunk = stream.read_chunk()
            if chunk is not None:
                process_chunk(chunk, cache)
        stream.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_STRIDE_SAMPLES,
        device_index: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: 采样率，默认 16000
            chunk_samples: 每次 read_chunk 返回的采样点数，默认 9600 (600ms)
            device_index: PyAudio 设备索引，None 则使用系统默认
        """
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device_index = device_index

        self._p: Optional[object] = None       # PyAudio 实例
        self._stream: Optional[object] = None  # PyAudio Stream
        self._buffer = np.array([], dtype=np.float32)
        self._is_running = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        打开麦克风并开始采集。

        Returns:
            bool: 是否成功启动
        """
        if self._is_running:
            logger.warning("麦克风流已在运行中。")
            return True

        try:
            import pyaudio
            self._p = pyaudio.PyAudio()

            # 确定格式
            if PYAUDIO_FORMAT == "int16":
                pyaudio_format = pyaudio.paInt16
            elif PYAUDIO_FORMAT == "float32":
                pyaudio_format = pyaudio.paFloat32
            else:
                pyaudio_format = pyaudio.paInt16

            self._stream = self._p.open(
                format=pyaudio_format,
                channels=CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=PYAUDIO_CHUNK,
            )
            self._is_running = True
            self._buffer = np.array([], dtype=np.float32)
            logger.info(
                f"麦克风流已启动 (device={self.device_index}, "
                f"sr={self.sample_rate}, chunk={self.chunk_samples})"
            )
            return True

        except ImportError:
            logger.error("PyAudio 未安装，无法使用麦克风。请执行: pip install pyaudio")
            return False
        except OSError as e:
            logger.error(f"无法打开麦克风设备 (index={self.device_index}): {e}")
            return False
        except Exception as e:
            logger.error(f"启动麦克风流失败: {e}")
            return False

    def stop(self):
        """停止麦克风采集并释放资源。"""
        self._is_running = False

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"关闭音频流时出错: {e}")
                self._stream = None

            if self._p is not None:
                try:
                    self._p.terminate()
                except Exception as e:
                    logger.warning(f"终止 PyAudio 时出错: {e}")
                self._p = None

        logger.info("麦克风流已停止。")

    def read_chunk(self) -> Optional[np.ndarray]:
        """
        读取一个音频块（阻塞，直到积累足够的采样点）。

        Returns:
            np.ndarray 或 None: float32 数组，长度为 chunk_samples。
                               如果流未运行，返回 None。
        """
        if not self._is_running or self._stream is None:
            return None

        try:
            # 读取原始数据
            data = self._stream.read(PYAUDIO_CHUNK, exception_on_overflow=False)
            # int16 → float32 归一化
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            with self._lock:
                self._buffer = np.concatenate([self._buffer, audio])

            # 检查是否积累够了
            if len(self._buffer) >= self.chunk_samples:
                with self._lock:
                    chunk = self._buffer[:self.chunk_samples].copy()
                    self._buffer = self._buffer[self.chunk_samples:]
                return chunk

            return None  # 还没攒够

        except Exception as e:
            logger.error(f"读取音频数据时出错: {e}")
            return None

    def read_chunk_blocking(self, timeout_ms: float = 2000) -> Optional[np.ndarray]:
        """
        阻塞式读取一个音频块，直到攒够数据或超时。

        Args:
            timeout_ms: 超时时间（毫秒）

        Returns:
            np.ndarray 或 None: 超时返回 None
        """
        start = time.time()
        while time.time() - start < timeout_ms / 1000:
            chunk = self.read_chunk()
            if chunk is not None:
                return chunk
            time.sleep(READ_CHUNK_POLL_INTERVAL_SEC)
        return None

    def is_running(self) -> bool:
        """返回流是否正在运行。"""
        return self._is_running and self._stream is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ================= 音频文件读取与重采样 =================

def read_wav_file(filepath: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    读取 WAV 文件并转换为 16kHz 单声道 float32 格式。

    自动处理：
    - 重采样（任意采样率 → 16kHz）
    - 立体声 → 单声道
    - 任意位深度 → float32

    Args:
        filepath: WAV 文件路径
        target_sr: 目标采样率，默认 16000

    Returns:
        np.ndarray: 1D float32 数组，范围 [-1.0, 1.0]
    """
    import soundfile as sf

    audio, sr = sf.read(filepath, dtype="float32")

    # 立体声 → 单声道
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 重采样
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)

    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    音频重采样。

    Args:
        audio: 1D numpy 数组
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        np.ndarray: 重采样后的音频
    """
    if orig_sr == target_sr:
        return audio.copy()

    try:
        from scipy.signal import resample
        import scipy
        # scipy 版本兼容处理
        if hasattr(scipy.signal, 'resample'):
            num_samples = int(len(audio) * target_sr / orig_sr)
            return resample(audio, num_samples).astype(np.float32)
    except ImportError:
        pass

    # 降级方案：使用 librosa
    try:
        import librosa
        return librosa.resample(
            audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr
        ).astype(np.float32)
    except ImportError:
        pass

    # 最终降级方案：线性插值
    logger.warning("scipy 和 librosa 均不可用，使用线性插值重采样。")
    old_len = len(audio)
    new_len = int(old_len * target_sr / orig_sr)
    old_indices = np.linspace(0, old_len - 1, old_len)
    new_indices = np.linspace(0, old_len - 1, new_len)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    将立体声音频转换为单声道（取左右声道平均值）。

    Args:
        audio: 形状 (N, 2) 的数组

    Returns:
        np.ndarray: 1D 数组
    """
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio
