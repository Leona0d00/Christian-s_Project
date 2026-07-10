"""
边缘部署接口模块
提供设备检测、运行时配置、模型导出等边缘部署相关功能。

边缘部署策略：
- 高性能设备：Paraformer-Large + CUDA/GPU
- 中端设备：Paraformer-streaming + CPU
- 低资源设备：Paraformer-Small + ONNX Runtime + INT8 量化
- 超低资源（树莓派等）：ONNX Runtime + 线程优化
"""

import logging
from typing import Dict, Optional

from ._config import EDGE_PROFILES

logger = logging.getLogger(__name__)


def list_available_devices() -> Dict[str, bool]:
    """
    检测当前环境可用的计算设备。

    Returns:
        dict: {"cpu": True, "cuda": bool}
    """
    devices = {"cpu": True, "cuda": False}

    try:
        import torch
        devices["cuda"] = torch.cuda.is_available()
        if devices["cuda"]:
            devices["cuda_device_count"] = torch.cuda.device_count()
            devices["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        logger.info("PyTorch 未安装，仅支持 CPU 模式。")

    # 检查 ONNX Runtime
    try:
        import onnxruntime
        devices["onnx_available"] = True
        providers = onnxruntime.get_available_providers()
        devices["onnx_providers"] = providers
    except ImportError:
        devices["onnx_available"] = False

    return devices


def select_device(device: str) -> str:
    """
    验证并选择计算设备。如果请求的设备不可用，自动降级。

    Args:
        device: 请求的设备名称 ("cpu", "cuda", "auto")

    Returns:
        str: 实际可用设备名称
    """
    available = list_available_devices()

    if device == "auto":
        return "cuda" if available.get("cuda", False) else "cpu"

    if device == "cuda" and not available.get("cuda", False):
        logger.warning("CUDA 不可用，降级为 CPU。")
        return "cpu"

    if device in ("cpu", "cuda"):
        return device

    logger.warning(f"未知设备类型 '{device}'，使用 CPU。")
    return "cpu"


def get_edge_deployment_config(profile: str = "cpu_optimized") -> dict:
    """
    获取预设的边缘部署配置。

    Args:
        profile: 配置名称，可选值:
            - "cpu_optimized" (默认): 平衡模式
            - "low_latency": 低延迟模式
            - "cuda_optimized": GPU 加速模式
            - "onnx_edge": ONNX 边缘部署模式

    Returns:
        dict: 包含 chunk_size, device, latency_ms, description 的配置
    """
    if profile in EDGE_PROFILES:
        return dict(EDGE_PROFILES[profile])

    logger.warning(f"未知配置 '{profile}'，使用默认配置。")
    return dict(EDGE_PROFILES["cpu_optimized"])


def list_edge_profiles() -> Dict[str, dict]:
    """
    列出所有可用的边缘部署配置。

    Returns:
        dict: {profile_name: config_dict}
    """
    return {k: dict(v) for k, v in EDGE_PROFILES.items()}


def export_onnx(
    model_name: str = "paraformer-zh-streaming",
    output_path: Optional[str] = None,
    quantize: bool = False,
    device: str = "cpu",
) -> str:
    """
    导出 FunASR 模型为 ONNX 格式（预留接口）。

    注意：此功能需要 funasr-onnx 包和 onnx 包。
    当前为接口预留，具体实现依赖于 funasr 的 ONNX 导出工具链。

    Args:
        model_name: 模型名称
        output_path: 输出路径，None 则自动生成
        quantize: 是否进行 INT8 量化
        device: 设备类型

    Returns:
        str: 输出文件路径或状态消息
    """
    # 检查依赖
    try:
        import onnx
    except ImportError:
        return (
            "ONNX 导出需要 onnx 包。请执行:\n"
            "  pip install onnx onnxruntime\n"
            "对于 FunASR ONNX 模型，还需要:\n"
            "  pip install funasr-onnx"
        )

    if output_path is None:
        output_path = f"./{model_name}.onnx"
        if quantize:
            output_path = f"./{model_name}_int8.onnx"

    try:
        logger.info(f"正在导出 ONNX 模型: {model_name} → {output_path}")

        # FunASR ONNX 导出的标准路径
        # 注意：FunASR 的 ONNX 导出需要使用 modelscope 的导出工具
        # 此处为接口预留，实际导出需要根据 FunASR 的具体 API 调整
        from modelscope.exporters import Exporter

        # 这个 API 在 funasr>=1.1.0 中可能已变化
        # 用户需要参考最新的 funasr 文档
        logger.warning(
            "ONNX 导出功能为预留接口。请参考 FunASR 最新文档:\n"
            "https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime"
        )

        return (
            f"ONNX 导出接口已调用。\n"
            f"模型: {model_name}\n"
            f"输出: {output_path}\n"
            f"量化: {'INT8' if quantize else 'FP32'}\n"
            f"提示: 请确保已安装 funasr-onnx 并参考 FunASR 文档。"
        )

    except ImportError:
        return (
            "modelscope.exporters 不可用。请确保已安装 modelscope。\n"
            "对于 FunASR >= 1.1.0，ONNX 导出请参考:\n"
            "https://github.com/modelscope/FunASR/tree/main/runtime/python/onnxruntime"
        )
    except Exception as e:
        logger.error(f"ONNX 导出失败: {e}")
        return f"ONNX 导出失败: {e}"


def get_optimization_guide() -> str:
    """
    返回边缘部署优化指南。

    Returns:
        str: Markdown 格式的优化指南
    """
    return """
## 边缘部署优化指南

### 模型选择
| 设备 | 推荐模型 | 参数量 | 内存 |
|------|---------|--------|------|
| 树莓派 4B | paraformer-zh-small | 120MB | <512MB |
| Jetson Nano | paraformer-zh-streaming + TensorRT INT8 | 860MB(INT8) | <1GB |
| Jetson Xavier | paraformer-zh-large + TensorRT | ~2GB | ~2GB |
| PC/服务器 | paraformer-zh-streaming | 220M | ~1GB |

### CPU 优化
- 线程数设为 CPU 核心数的 1.5 倍
- 使用 ONNX Runtime 替代 PyTorch 推理
- 启用 ARM NEON 优化（ARM 设备）
- 导出模型为 INT8 量化减少内存占用

### GPU 优化
- NVIDIA Jetson: 使用 TensorRT + FP16
- 使用 jetson_clocks --fan 提升性能
- INT8 量化将内存从 1.8GB 降至 860MB

### Docker 部署
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
docker run -p 10096:10096 <image>
```
"""
