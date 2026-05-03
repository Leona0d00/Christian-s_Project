"""
环境稳定性测试脚本
检查项：
1. PyTorch 版本及 CUDA 可用性
2. onnxruntime-gpu 版本及可用 Provider
3. openwakeword 模型加载（离线模式）
4. funasr 框架导入及模型加载（离线模式）
5. 音频处理库（pyaudio, soundfile, librosa）
6. 网络库（aiohttp）可选
"""

import sys
import platform

def print_sep(title):
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("=" * 60)

def test_pytorch():
    """测试 PyTorch 版本及 CUDA 环境"""
    print_sep("1. 测试 PyTorch 环境")
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ 警告: CUDA 不可用，将使用 CPU 推理")
        return True
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False

def test_onnxruntime():
    """测试 ONNX Runtime GPU 版本"""
    print_sep("2. 测试 ONNX Runtime")
    try:
        import onnxruntime as ort
        print(f"✓ onnxruntime 版本: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  可用 Provider: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("  ✓ CUDAExecutionProvider 可用，GPU 推理正常")
        else:
            print("  ⚠ 警告: CUDAExecutionProvider 不可用，将回退到 CPU")
        return True
    except ImportError as e:
        print(f"✗ onnxruntime 导入失败: {e}")
        print("  提示: 请手动安装 onnxruntime-gpu==1.21.0")
        return False

def test_openwakeword():
    """测试 OpenWakeWord 模型加载（离线）"""
    print_sep("3. 测试 OpenWakeWord")
    try:
        from openwakeword.model import Model
        # 尝试加载一个最小的内置模型（假设已下载到默认位置）
        # 为避免下载，如果本地无模型则跳过实际加载
        import os
        from pathlib import Path
        model_dir = Path.home() / ".openwakeword"
        if not model_dir.exists():
            print("  ⚠ 未找到本地模型目录 (~/.openwakeword)，跳过模型加载测试")
            print("  可运行以下命令下载模型: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
            return True
        # 尝试加载一个存在的模型文件
        model_files = list(model_dir.glob("*.tflite"))
        if not model_files:
            print("  ⚠ 未找到 .tflite 模型文件，跳过加载测试")
            return True
        test_model = str(model_files[0])
        model = Model(wakeword_models=[test_model])
        print(f"✓ 成功加载模型: {os.path.basename(test_model)}")
        return True
    except Exception as e:
        print(f"✗ OpenWakeWord 测试失败: {e}")
        return False

def test_funasr():
    """测试 FunASR 框架及离线模型加载（SenseVoice）"""
    print_sep("4. 测试 FunASR / SenseVoice")
    try:
        from funasr import AutoModel
        print("✓ FunASR 导入成功")
        # 检查本地是否有 SenseVoice 模型
        import os
        model_path = "./models/SenseVoiceSmall"
        if not os.path.exists(model_path):
            # 尝试用户目录下的模型
            alt_path = os.path.expanduser("~/.cache/modelscope/hub/iic/SenseVoiceSmall")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                print("  ⚠ 未找到本地 SenseVoice 模型，跳过实际加载")
                print("  提示: 可使用 modelscope 下载: snapshot_download('iic/SenseVoiceSmall', local_dir='./models/SenseVoiceSmall')")
                return True
        # 尝试加载模型（仅加载配置，不执行推理）
        model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            device="cuda:0" if __import__('torch').cuda.is_available() else "cpu",
            disable_update=True,
        )
        print(f"✓ 成功加载模型: {model_path}")
        return True
    except Exception as e:
        print(f"✗ FunASR 测试失败: {e}")
        return False

def test_audio_libs():
    """测试音频处理库"""
    print_sep("5. 测试音频库 (pyaudio, soundfile, librosa)")
    success = True
    # pyaudio
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        print(f"✓ PyAudio 版本: {pyaudio.__version__ if hasattr(pyaudio, '__version__') else '未知'}")
        p.terminate()
    except Exception as e:
        print(f"✗ PyAudio 初始化失败: {e}")
        success = False
    # soundfile
    try:
        import soundfile as sf
        print(f"✓ soundfile 版本: {sf.__version__}")
    except ImportError as e:
        print(f"✗ soundfile 导入失败: {e}")
        success = False
    # librosa
    try:
        import librosa
        print(f"✓ librosa 版本: {librosa.__version__}")
    except ImportError as e:
        print(f"✗ librosa 导入失败: {e}")
        success = False
    return success

def test_network():
    """测试网络库 aiohttp（可选）"""
    print_sep("6. 测试网络库 aiohttp")
    try:
        import aiohttp
        print(f"✓ aiohttp 版本: {aiohttp.__version__}")
        return True
    except ImportError:
        print("✗ aiohttp 未安装（离线模式可不安装）")
        return False

def main():
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print("开始环境稳定性测试...")
    
    results = []
    results.append(("PyTorch", test_pytorch()))
    results.append(("ONNX Runtime", test_onnxruntime()))
    results.append(("OpenWakeWord", test_openwakeword()))
    results.append(("FunASR", test_funasr()))
    results.append(("音频库", test_audio_libs()))
    results.append(("网络库", test_network()))
    
    print_sep("测试总结")
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:15} : {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 恭喜！所有核心组件测试通过，环境稳定。")
    else:
        print("\n⚠️ 部分测试未通过，请根据上述错误信息检查环境配置。")

if __name__ == "__main__":
    main()
    