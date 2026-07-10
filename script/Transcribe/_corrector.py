"""
字幕校正模块
提供实时字幕的去重、合并、标点恢复等功能。

核心问题：流式 ASR 输出的是增量式文本，相邻块之间存在大量重叠。
例如：
  Chunk 1 → "欢迎大"
  Chunk 2 → "欢迎大家"     ← "欢迎大" 是前一块的前缀，需要去重
  Chunk 3 → "欢迎大家来"   ← 同理

校正策略：
  1. 实时去重：跟踪已显示文本，只输出增量部分
  2. 标点恢复：在最终结果上使用 ct-punc 模型（独立于流式模型）
  3. 后处理规范化：移除多余空格、规范标点格式
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ================= 标点模型单例 =================
_punc_model = None
_punc_model_device = None


def _get_punc_model(device: str = "cpu"):
    """获取或初始化 ct-punc 模型（单例，与 ASR 模型缓存策略一致）。"""
    global _punc_model, _punc_model_device

    if _punc_model is not None and _punc_model_device == device:
        return _punc_model

    if _punc_model is not None and _punc_model_device != device:
        logger.info(f"标点模型设备从 {_punc_model_device} 切换到 {device}，重新加载...")
        _punc_model = None

    if _punc_model is None:
        logger.info(f"正在加载标点模型 ct-punc (device={device})...")
        try:
            from funasr import AutoModel
            _punc_model = AutoModel(
                model="ct-punc",
                device=device,
                disable_pbar=True,
                disable_update=True,
            )
            _punc_model_device = device
            logger.info("标点模型加载完成。")
        except Exception as e:
            logger.error(f"标点模型加载失败: {e}")
            raise

    return _punc_model


def reset_punc_model():
    """释放标点模型（用于测试重置）。"""
    global _punc_model, _punc_model_device
    _punc_model = None
    _punc_model_device = None


class SubtitleCorrector:
    """
    实时字幕校正器。

    负责对流式 ASR 输出的增量文本进行去重处理，
    确保 GUI 显示的文本不出现重复内容。

    Usage:
        corrector = SubtitleCorrector()
        for chunk in audio_chunks:
            raw = process_chunk(chunk, cache)
            display_text = corrector.correct_chunk(raw["text"])
            if display_text:
                print(display_text, end="", flush=True)
        final_text = corrector.finalize()  # 带标点的完整文本
    """

    def __init__(self):
        self._full_text = ""           # 累计全部原始文本
        self._displayed_text = ""      # 已通过 correct_chunk 输出的文本
        self._chunk_count = 0

    def correct_chunk(self, raw_text: str) -> str:
        """
        对单块原始文本进行去重校正，返回应显示的增量文本。

        去重逻辑：
        1. 如果 raw_text 以 _displayed_text 的末尾部分开头，
           则只输出 raw_text 中与历史不重叠的新增部分
        2. 如果 raw_text 与历史完全无关（新句子），直接输出
        3. 使用最长公共前缀匹配来找到增量

        Args:
            raw_text: 当前块识别的原始文本

        Returns:
            str: 应该显示的新增文本（可能为空）
        """
        if not raw_text:
            return ""

        self._chunk_count += 1
        self._full_text += raw_text

        # 第一种情况：还没有显示过任何文本，直接显示
        if not self._displayed_text:
            self._displayed_text = raw_text
            return raw_text

        # 第二种情况：新文本完全是旧文本的前缀（模型修正了结果）
        # 例如：旧="欢迎大"，新="欢迎大家"
        # 找到重叠部分
        overlap_len = self._find_overlap(self._displayed_text, raw_text)

        if overlap_len > 0:
            # 有重叠（部分或全部），只输出新增部分
            delta = raw_text[overlap_len:]
        else:
            # 没有重叠，可能是新句子，直接输出
            delta = raw_text

        if delta:
            self._displayed_text = raw_text

        return delta

    def finalize(self, apply_punctuation: bool = True, device: str = "cpu") -> str:
        """
        结束当前会话，返回带标点校正的完整文本。

        在流式会话结束时调用，对累积的全部文本进行最终处理：
        - 标点恢复（使用 ct-punc 模型或规则）
        - 后处理规范化

        Args:
            apply_punctuation: 是否应用标点恢复（默认 True）
            device: 标点模型设备（"cpu" 或 "cuda"）

        Returns:
            str: 校正后的完整文本
        """
        text = self._full_text

        if apply_punctuation:
            text = self._apply_punctuation(text, device=device)

        text = self._post_process(text)
        return text

    def reset(self):
        """重置校正器状态，开始新的识别会话。"""
        self._full_text = ""
        self._displayed_text = ""
        self._chunk_count = 0

    def get_raw_text(self) -> str:
        """返回累积的原始文本（不含标点恢复）。"""
        return self._full_text

    def get_display_text_length(self) -> int:
        """返回已显示文本的字符数。"""
        return len(self._displayed_text)

    # ================= 内部方法 =================

    def _find_overlap(self, old_text: str, new_text: str) -> int:
        """
        找到 old_text 后缀与 new_text 前缀的最长重叠长度。

        使用贪心策略：从最大可能重叠开始向下搜索。

        Args:
            old_text: 已显示的文本
            new_text: 新文本

        Returns:
            int: 重叠字符数
        """
        max_overlap = min(len(old_text), len(new_text))
        # 至少检查 1 个字符，从大到小搜索
        for length in range(max_overlap, 0, -1):
            if old_text[-length:] == new_text[:length]:
                return length
        return 0

    def _apply_punctuation(self, text: str, device: str = "cpu") -> str:
        """
        对文本应用标点恢复。

        优先使用 ct-punc 模型（如果可用且已下载），
        否则使用基于规则的后处理。

        Args:
            text: 原始无标点文本
            device: 计算设备

        Returns:
            str: 带标点的文本
        """
        if not text:
            return text

        try:
            punc_model = _get_punc_model(device=device)

            result = punc_model.generate(input=text)
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    punctuated = item.get("text", text)
                elif isinstance(item, str):
                    punctuated = item
                else:
                    punctuated = text
            elif isinstance(result, str):
                punctuated = result
            else:
                punctuated = text

            return punctuated if punctuated else text

        except ImportError:
            logger.warning("funasr 不可用，跳过标点恢复。")
            return text
        except Exception as e:
            logger.warning(f"标点恢复失败: {e}，使用原始文本。")
            return text

    def _post_process(self, text: str) -> str:
        """
        文本后处理规范化。

        处理内容：
        - 中文标点后添加空格（可选）
        - 移除多余空白字符
        - 规范化中英文混合标点
        """
        if not text:
            return text

        # 移除多余空白
        text = re.sub(r'\s+', '', text)

        # 在中文句号、问号、感叹号后添加换行（便于阅读）
        text = re.sub(r'([。！？；])', r'\1\n', text)

        # 移除末尾多余的换行
        text = text.rstrip('\n')

        return text
