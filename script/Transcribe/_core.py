"""
核心处理逻辑模块
提供独立的 process_chunk() 方法，供外部测试脚本直接调用。

设计原则：
- process_chunk() 是纯函数式接口，状态由外部 cache 参数管理
- 模型以模块级单例形式加载，按需初始化
- 所有 FunASR 模型调用细节封装在此模块内
"""

import logging
import numpy as np
from typing import Optional

from ._config import (
    MODEL_NAME,
    MODEL_REVISION,
    CHUNK_SIZE_CONFIG,
    ENCODER_CHUNK_LOOK_BACK,
    DECODER_CHUNK_LOOK_BACK,
    CHUNK_STRIDE_SAMPLES,
    DEFAULT_DEVICE,
)

logger = logging.getLogger(__name__)

# ================= 模型单例 =================
_model = None
_model_device = None


def _get_model(device: str = DEFAULT_DEVICE):
    """
    获取或初始化 FunASR 流式模型（单例）。
    首次调用时自动下载并加载模型，后续调用复用已加载的模型。
    """
    global _model, _model_device

    if _model is not None and _model_device == device:
        return _model

    # 如果设备变了，释放旧模型
    if _model is not None and _model_device != device:
        logger.info(f"设备从 {_model_device} 切换到 {device}，重新加载模型...")
        _model = None

    if _model is None:
        logger.info(f"正在加载模型 {MODEL_NAME} (device={device})...")
        try:
            from funasr import AutoModel

            _model = AutoModel(
                model=MODEL_NAME,
                model_revision=MODEL_REVISION,
                device=device,
                disable_pbar=True,
                disable_update=True,
            )
            _model_device = device
            logger.info("模型加载完成。")
        except ImportError as e:
            raise ImportError(
                "无法导入 funasr。请确保已安装: pip install funasr"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"模型加载失败: {e}\n"
                f"请检查网络连接（模型需从 ModelScope 下载），"
                f"或手动指定本地模型路径。"
            ) from e

    return _model


def reset_model():
    """
    释放已加载的模型，释放内存。
    主要用于测试场景中重置状态。
    """
    global _model, _model_device
    _model = None
    _model_device = None
    logger.info("模型已重置。")


def process_chunk(
    audio_chunk: np.ndarray,
    cache: dict,
    is_final: bool = False,
    chunk_size: Optional[list] = None,
    encoder_chunk_look_back: int = ENCODER_CHUNK_LOOK_BACK,
    decoder_chunk_look_back: int = DECODER_CHUNK_LOOK_BACK,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """
    处理单个音频块，返回识别文本。

    这是 Transcriber 模块的核心方法，独立于任何外部状态，
    方便 Test.py 等外部脚本直接调用测试。

    Args:
        audio_chunk: float32 numpy 数组，形状 (N,) 或 (1, N)，
                     采样率 16kHz，单声道。
                     推荐长度为 CHUNK_STRIDE_SAMPLES (9600 = 600ms)。
        cache: 持久化缓存字典。调用方负责创建和维护。
               在所有 chunk 之间共享同一个 dict 对象。
               初始值为空字典 {}。
        is_final: 是否为最后一块。设为 True 时强制刷新
                  缓存中的剩余识别结果。
        chunk_size: chunk 配置 [lookback, current, lookahead]。
                    默认使用 [0, 10, 5]（600ms 块 + 300ms 前瞻）。
        encoder_chunk_look_back: 编码器回看帧数，默认 4。
        decoder_chunk_look_back: 解码器回看帧数，默认 1。
        device: 设备名称 "cpu" 或 "cuda"。

    Returns:
        dict: {"text": str, "is_final": bool}
              - text: 该块对应的识别文本（可能为空字符串）
              - is_final: 与输入参数一致
              - error: 如果出错，包含错误信息（正常情况不存在该键）

    Example:
        >>> cache = {}
        >>> result = process_chunk(audio_chunk, cache)
        >>> print(result["text"])
        '你好'
        >>> final = process_chunk(last_chunk, cache, is_final=True)
        >>> print(final["text"])
        '你好世界'
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE_CONFIG

    # ===== 输入验证 =====
    if not isinstance(audio_chunk, np.ndarray):
        return {"text": "", "is_final": is_final, "error": "audio_chunk 必须是 numpy.ndarray"}

    if audio_chunk.dtype != np.float32:
        try:
            audio_chunk = audio_chunk.astype(np.float32)
        except (ValueError, TypeError) as e:
            return {"text": "", "is_final": is_final, "error": f"无法转换为 float32: {e}"}

    # 确保是 1D 数组
    if audio_chunk.ndim > 1:
        audio_chunk = audio_chunk.ravel()

    if len(audio_chunk) == 0:
        return {"text": "", "is_final": is_final}

    if not isinstance(cache, dict):
        return {"text": "", "is_final": is_final, "error": "cache 必须是 dict 类型"}

    # ===== 模型推理 =====
    try:
        model = _get_model(device=device)
        result = model.generate(
            input=audio_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
    except Exception as e:
        logger.error(f"process_chunk 推理失败: {e}")
        return {"text": "", "is_final": is_final, "error": str(e)}

    # ===== 解析结果 =====
    text = ""
    try:
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
                if text is None:
                    text = ""
            elif isinstance(item, str):
                text = item
        elif isinstance(result, dict):
            text = result.get("text", "")
            if text is None:
                text = ""
        elif isinstance(result, str):
            text = result
    except Exception as e:
        logger.warning(f"解析识别结果异常: {e}, raw_result={result}")

    return {"text": str(text), "is_final": is_final}


def get_model_info() -> dict:
    """
    返回当前模型信息，用于日志记录和状态展示。

    Returns:
        dict: 包含 model_name, device, chunk_config 等字段
    """
    info = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": _model_device or "not_loaded",
        "chunk_size": CHUNK_SIZE_CONFIG,
        "chunk_stride_ms": 600,
        "sample_rate": 16000,
        "is_loaded": _model is not None,
    }
    return info
