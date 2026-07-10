"""
实时转录流水线模块
将麦克风采集、核心处理、字幕校正串联为完整的实时转录流水线。
"""

import logging
import threading
import queue
import numpy as np
from typing import Optional, Callable

from ._config import (
    CHUNK_STRIDE_SAMPLES,
    CHUNK_SIZE_CONFIG,
    DEFAULT_DEVICE,
    SAMPLE_RATE,
)
from ._core import process_chunk, get_model_info, reset_model
from ._audio import MicrophoneStream, auto_detect_microphone, list_microphones
from ._corrector import SubtitleCorrector

logger = logging.getLogger(__name__)


class RealTimeTranscriber:
    """
    实时语音转录器。

    整合了麦克风采集、流式 ASR 推理和字幕校正的完整流水线。
    在后台线程中运行采集→推理→校正循环，
    通过线程安全队列将结果传递给 GUI 线程。

    Usage:
        transcriber = RealTimeTranscriber(device="cpu")
        transcriber.start()
        # ... 等待用户停止 ...
        while transcriber.is_running():
            results = transcriber.get_results()
            for text in results:
                print(text, end="")
            time.sleep(0.1)
        final_text = transcriber.stop()
        print("\\n最终文本:", final_text)
    """

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        device_index: Optional[int] = None,
        on_text_callback: Optional[Callable[[str], None]] = None,
        chunk_size: Optional[list] = None,
    ):
        """
        Args:
            device: 计算设备 "cpu" 或 "cuda"
            device_index: 麦克风设备索引，None 表示自动检测
            on_text_callback: 可选回调函数，每识别到新文本时调用
            chunk_size: chunk 配置，默认 [0, 10, 5]
        """
        self.device = device
        self.device_index = device_index
        self.on_text_callback = on_text_callback
        self.chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE_CONFIG

        # 内部组件（在 start() 中初始化）
        self._stream: Optional[MicrophoneStream] = None
        self._corrector = SubtitleCorrector()
        self._cache: dict = {}
        self._results_queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 模型预加载标志
        self._model_ready = False

    def start(self) -> bool:
        """
        启动实时转录。

        1. 预加载模型（首次调用时会下载）
        2. 打开麦克风
        3. 启动后台处理线程

        Returns:
            bool: 是否成功启动
        """
        if self._thread and self._thread.is_alive():
            logger.warning("转录已在运行中。")
            return True

        # 预加载模型
        try:
            logger.info("正在预加载模型...")
            # 触发模型加载
            dummy = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
            test_cache = {}
            process_chunk(dummy, test_cache, is_final=False, device=self.device)
            self._model_ready = True
            logger.info("模型预加载完成。")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model_ready = False
            return False

        # 预加载标点模型（避免首次 stop() 时冷启动延迟）
        try:
            logger.info("正在预加载标点模型...")
            from ._corrector import _get_punc_model
            _punc_device = "cuda" if self.device == "cuda" else "cpu"
            _get_punc_model(device=_punc_device)
            logger.info("标点模型预加载完成。")
        except Exception as e:
            logger.warning(f"标点模型预加载失败（将在首次 stop 时重试）: {e}")

        # 自动检测麦克风
        if self.device_index is None:
            mic = auto_detect_microphone()
            if mic:
                self.device_index = mic["index"]
                logger.info(f"自动选择麦克风: {mic['name']} (index={mic['index']})")
            else:
                logger.warning("未检测到麦克风设备，将只能使用文件输入模式。")

        # 打开麦克风
        self._stream = MicrophoneStream(
            sample_rate=SAMPLE_RATE,
            chunk_samples=CHUNK_STRIDE_SAMPLES,
            device_index=self.device_index,
        )

        if not self._stream.start():
            logger.error("无法打开麦克风。")
            self._stream = None
            return False

        # 重置状态
        self._cache = {}
        self._corrector.reset()
        self._stop_event.clear()
        self._results_queue = queue.Queue()

        # 启动后台线程
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="transcriber-thread",
        )
        self._thread.start()
        logger.info("实时转录已启动。")
        return True

    def stop(self) -> str:
        """
        停止转录并返回最终校正文本。

        1. 发送停止信号
        2. 等待后台线程结束
        3. 刷新最终缓存
        4. 返回标点校正后的完整文本

        Returns:
            str: 最终校正文本
        """
        if not self.is_running():
            return ""

        logger.info("正在停止转录...")
        self._stop_event.set()

        # 等待后台线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("后台线程未在超时内停止。")

        # 关闭麦克风
        if self._stream:
            self._stream.stop()
            self._stream = None

        # 刷新最终缓存
        final_text = ""
        try:
            dummy = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
            result = process_chunk(
                dummy,
                self._cache,
                is_final=True,
                chunk_size=self.chunk_size,
                device=self.device,
            )
            if result.get("text"):
                self._corrector.correct_chunk(result["text"])
        except Exception as e:
            logger.warning(f"刷新最终缓存失败: {e}")

        # 获取标点校正后的完整文本
        final_text = self._corrector.finalize(apply_punctuation=True, device=self.device)

        self._thread = None
        logger.info("转录已停止。")
        return final_text

    def is_running(self) -> bool:
        """返回转录器是否正在运行。"""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    def get_results(self) -> list:
        """
        获取队列中所有待处理的结果（非阻塞）。

        Returns:
            list[str]: 增量文本列表
        """
        results = []
        while True:
            try:
                item = self._results_queue.get_nowait()
                results.append(item)
            except queue.Empty:
                break
        return results

    def get_model_info(self) -> dict:
        """返回模型信息。"""
        return get_model_info()

    # ================= 内部方法 =================

    def _run_loop(self):
        """
        后台处理循环（运行在独立线程中）。

        流程：
        1. 从麦克风读取一个 chunk (600ms 音频)
        2. 调用 process_chunk() 进行 ASR
        3. 调用 SubtitleCorrector.correct_chunk() 去重
        4. 将增量文本放入队列
        5. 重复直到收到停止信号
        """
        logger.info("转录处理循环已启动。")
        chunk_count = 0

        try:
            while not self._stop_event.is_set():
                if self._stream is None:
                    break

                # 读取音频块（带超时，以便检查停止信号）
                audio_chunk = self._stream.read_chunk_blocking(timeout_ms=500)

                if audio_chunk is None:
                    continue

                # ASR 推理
                result = process_chunk(
                    audio_chunk,
                    self._cache,
                    is_final=False,
                    chunk_size=self.chunk_size,
                    device=self.device,
                )

                raw_text = result.get("text", "")

                # 字幕校正（去重）
                if raw_text:
                    display_text = self._corrector.correct_chunk(raw_text)
                    if display_text:
                        self._results_queue.put(display_text)
                        if self.on_text_callback:
                            try:
                                self.on_text_callback(display_text)
                            except Exception as e:
                                logger.warning(f"回调函数异常: {e}")

                chunk_count += 1

        except Exception as e:
            logger.error(f"转录处理循环异常: {e}", exc_info=True)
        finally:
            logger.info(f"转录处理循环结束，共处理 {chunk_count} 个 chunk。")
