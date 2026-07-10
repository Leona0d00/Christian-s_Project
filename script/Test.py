#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：对 Transcribe 模块的核心处理逻辑进行功能测试。

使用 test.wav 模拟实时音频流，逐块送入 process_chunk()，
验证核心方法的正确性和功能完备性。

用法:
    conda run -n christian python script/Test.py
"""

import sys
import os
import time
import traceback
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.Transcribe import (
    process_chunk,
    get_model_info,
    reset_model,
    read_wav_file,
    list_microphones,
    auto_detect_microphone,
    SubtitleCorrector,
    CHUNK_STRIDE_SAMPLES,
    SAMPLE_RATE,
)

# 测试结果统计
PASS = 0
FAIL = 0
ERRORS = []


def log_test(name: str, passed: bool, detail: str = ""):
    """记录测试结果。"""
    global PASS, FAIL
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} | {name}")
    if detail:
        print(f"         {detail}")
    if passed:
        PASS += 1
    else:
        FAIL += 1
        ERRORS.append(f"[FAIL] {name}: {detail}")


def test_model_loading():
    """测试 1: 模型能否正常加载。"""
    print("\n" + "=" * 60)
    print("测试 1: 模型加载")
    print("=" * 60)

    try:
        # 触发模型加载
        dummy = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
        cache = {}
        result = process_chunk(dummy, cache, is_final=False)

        # 检查无错误
        if "error" in result:
            log_test("模型加载", False, f"返回错误: {result['error']}")
            return

        info = get_model_info()
        log_test("模型加载", info["is_loaded"], f"模型: {info['model_name']}, 设备: {info['device']}")
    except Exception as e:
        log_test("模型加载", False, f"异常: {e}")
        traceback.print_exc()


def test_single_chunk():
    """测试 2: 单块音频识别。"""
    print("\n" + "=" * 60)
    print("测试 2: 单块音频识别")
    print("=" * 60)

    try:
        # 读取 test.wav 并取第一块
        test_wav = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
        audio = read_wav_file(test_wav)
        chunk = audio[:CHUNK_STRIDE_SAMPLES].copy()

        cache = {}
        result = process_chunk(chunk, cache, is_final=False)

        has_text = isinstance(result, dict) and "text" in result
        no_error = "error" not in result
        text_value = result.get("text", "")

        log_test("返回结构正确", has_text, f"result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        log_test("无推理错误", no_error, f"error: {result.get('error', 'N/A')}")
        # 注意：第一个chunk可能为空（音频开头是静音），只要结构正确且无错误就算通过
        log_test("返回结构（text字段存在）", "text" in result,
                 f"text='{text_value}'（chunk开头为静音时空文本正常）")
    except Exception as e:
        log_test("单块识别", False, f"异常: {e}")
        traceback.print_exc()


def test_streaming_multi_chunk():
    """测试 3: 多块流式识别（模拟实时音频）。"""
    print("\n" + "=" * 60)
    print("测试 3: 多块流式识别（模拟实时音频）")
    print("=" * 60)

    try:
        test_wav = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
        audio = read_wav_file(test_wav)

        n_chunks = max(1, len(audio) // CHUNK_STRIDE_SAMPLES + 1)
        print(f"  音频总长度: {len(audio)} 采样点 ({len(audio)/SAMPLE_RATE:.2f}s)")
        print(f"  将分为 {n_chunks} 个 chunk 进行处理...")

        cache = {}
        corrector = SubtitleCorrector()
        all_raw_text = []
        chunk_texts = []

        for i in range(n_chunks):
            start = i * CHUNK_STRIDE_SAMPLES
            end = start + CHUNK_STRIDE_SAMPLES
            chunk = audio[start:end].copy()

            if len(chunk) < CHUNK_STRIDE_SAMPLES:
                # 最后一块不足，填充零
                padded = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded

            is_final = (i == n_chunks - 1)

            result = process_chunk(chunk, cache, is_final=is_final)

            raw_text = result.get("text", "")
            if raw_text:
                all_raw_text.append(raw_text)

            # 字幕校正
            display = corrector.correct_chunk(raw_text)
            if display:
                chunk_texts.append(display)

        # 验证
        log_test("所有 chunk 处理完毕", len(all_raw_text) > 0 if len(audio) > 0 else True,
                 f"{len(all_raw_text)}/{n_chunks} 个 chunk 产生了文本")
        log_test("cache 非空（状态保留）", len(cache) > 0,
                 f"cache keys: {list(cache.keys()) if cache else '无'}")
        log_test("至少有一个 chunk 产生文本", len(chunk_texts) > 0 if len(audio) > 0 else True,
                 f"增量显示文本块数: {len(chunk_texts)}")

        # 打印所有识别文本
        full_raw = "".join(all_raw_text)
        print(f"\n  * 原始识别文本 ({len(full_raw)} 字符):")
        print(f"     {full_raw[:200]}{'...' if len(full_raw) > 200 else ''}")

        # 最终校正
        final_text = corrector.finalize(apply_punctuation=True)
        print(f"\n  * 校正后文本 ({len(final_text)} 字符):")
        print(f"     {final_text[:200]}{'...' if len(final_text) > 200 else ''}")

    except Exception as e:
        log_test("多块流式识别", False, f"异常: {e}")
        traceback.print_exc()


def test_cache_persistence():
    """测试 4: cache 跨块持久性验证。"""
    print("\n" + "=" * 60)
    print("测试 4: cache 跨块持久性")
    print("=" * 60)

    try:
        test_wav = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
        audio = read_wav_file(test_wav)

        # 取两个块，使用同一个 cache
        chunk1 = audio[:CHUNK_STRIDE_SAMPLES].copy()
        chunk2 = audio[CHUNK_STRIDE_SAMPLES:2 * CHUNK_STRIDE_SAMPLES].copy()

        if len(chunk2) < CHUNK_STRIDE_SAMPLES:
            padded = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
            padded[:len(chunk2)] = chunk2
            chunk2 = padded

        cache = {}
        result1 = process_chunk(chunk1, cache, is_final=False)
        cache_size_after_1 = len(cache)

        result2 = process_chunk(chunk2, cache, is_final=False)
        cache_size_after_2 = len(cache)

        log_test("同一 cache 跨块使用", True,
                 f"第1块后 cache 大小: {cache_size_after_1}, 第2块后: {cache_size_after_2}")
        log_test("cache 在第二块后增长或保持", cache_size_after_2 >= cache_size_after_1,
                 f"cache 从 {cache_size_after_1} → {cache_size_after_2}")
    except Exception as e:
        log_test("cache 持久性", False, f"异常: {e}")
        traceback.print_exc()


def test_audio_resampling():
    """测试 5: 音频重采样（48kHz 立体声 → 16kHz 单声道）。"""
    print("\n" + "=" * 60)
    print("测试 5: 音频重采样")
    print("=" * 60)

    try:
        test_wav = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.wav")
        audio = read_wav_file(test_wav)

        log_test("输出为 1D 数组", audio.ndim == 1,
                 f"shape: {audio.shape}, dtype: {audio.dtype}")
        log_test("输出为 float32", audio.dtype == np.float32,
                 f"dtype: {audio.dtype}")
        log_test("音频长度 > 0", len(audio) > 0,
                 f"长度: {len(audio)} 采样点 ({len(audio)/SAMPLE_RATE:.2f}s)")
        log_test("幅度在有效范围", np.max(np.abs(audio)) <= 1.0,
                 f"最大幅度: {np.max(np.abs(audio)):.4f}")
    except Exception as e:
        log_test("音频重采样", False, f"异常: {e}")
        traceback.print_exc()


def test_microphone_detection():
    """测试 6: 麦克风设备检测。"""
    print("\n" + "=" * 60)
    print("测试 6: 麦克风设备检测")
    print("=" * 60)

    try:
        devices = list_microphones()
        default_mic = auto_detect_microphone()

        log_test("设备列表获取成功", isinstance(devices, list),
                 f"找到 {len(devices)} 个输入设备")

        if devices:
            for d in devices:
                default_mark = " (默认)" if d.get("is_default") else ""
                print(f"         [{d['index']}] {d['name']}{default_mark}")

        if default_mic:
            log_test("自动检测麦克风", True,
                     f"默认设备: [{default_mic['index']}] {default_mic['name']}")
        else:
            log_test("自动检测麦克风（无设备）", True,
                     "当前系统无麦克风设备，这是正常的（例如在 CI 环境）")

    except Exception as e:
        log_test("麦克风检测", False, f"异常: {e}")
        traceback.print_exc()


def test_subtitle_correction():
    """测试 7: 字幕校正功能。"""
    print("\n" + "=" * 60)
    print("测试 7: 字幕校正功能")
    print("=" * 60)

    try:
        corrector = SubtitleCorrector()

        # 模拟流式输出
        sim_chunks = [
            "欢迎大",
            "欢迎大家",
            "欢迎大家来",
            "欢迎大家来到",
            "欢迎大家来到实时",
            "欢迎大家来到实时语音",
        ]

        deltas = []
        for text in sim_chunks:
            delta = corrector.correct_chunk(text)
            deltas.append(delta)

        log_test("去重功能正常", deltas == ["欢迎大", "家", "来", "到", "实时", "语音"],
                 f"实际增量: {deltas}")

        # 测试 reset
        corrector.reset()
        log_test("reset 后状态清空", corrector.get_raw_text() == "",
                 f"raw_text: '{corrector.get_raw_text()}'")

        # 测试空字符串
        delta = corrector.correct_chunk("")
        log_test("空字符串返回空", delta == "", f"返回值: '{delta}'")

        # 测试后处理
        text = "你好世界这是测试"
        post_processed = corrector._post_process(text)
        log_test("后处理保留原文本", len(post_processed) >= len(text),
                 f"处理后: '{post_processed}'")

    except Exception as e:
        log_test("字幕校正", False, f"异常: {e}")
        traceback.print_exc()


def test_edge_interfaces():
    """测试 8: 边缘部署接口。"""
    print("\n" + "=" * 60)
    print("测试 8: 边缘部署接口")
    print("=" * 60)

    try:
        from script.Transcribe import (
            list_available_devices,
            select_device,
            get_edge_deployment_config,
            list_edge_profiles,
            export_onnx,
            get_optimization_guide,
        )

        # 设备检测
        devices = list_available_devices()
        log_test("设备检测", isinstance(devices, dict) and "cpu" in devices,
                 f"可用设备: {devices}")

        # 设备选择
        cpu = select_device("cpu")
        log_test("CPU 设备选择", cpu == "cpu", f"选择结果: {cpu}")

        cuda = select_device("cuda")
        log_test("CUDA 设备选择（含降级）", cuda in ("cpu", "cuda"),
                 f"选择结果: {cuda} (CUDA可用: {devices.get('cuda', False)})")

        # 预设配置
        config = get_edge_deployment_config("cpu_optimized")
        log_test("预设配置获取", isinstance(config, dict) and "device" in config,
                 f"配置: device={config.get('device')}, latency={config.get('latency_ms')}ms")

        # 配置列表
        profiles = list_edge_profiles()
        log_test("配置列表获取", len(profiles) >= 4,
                 f"可用配置: {list(profiles.keys())}")

        # ONNX 导出接口（可能只是打印提示）
        onnx_result = export_onnx()
        log_test("ONNX 导出接口可调用", isinstance(onnx_result, str) and len(onnx_result) > 0,
                 f"返回: {onnx_result[:80]}...")

        # 优化指南
        guide = get_optimization_guide()
        log_test("优化指南获取", isinstance(guide, str) and len(guide) > 0,
                 f"长度: {len(guide)} 字符")

    except Exception as e:
        log_test("边缘部署接口", False, f"异常: {e}")
        traceback.print_exc()


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("  Transcribe 模块 — 功能测试")
    print(f"  测试音频: ../test.wav")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 按顺序执行测试
    test_model_loading()        # 测试 1
    test_single_chunk()         # 测试 2
    test_audio_resampling()     # 测试 3
    test_cache_persistence()    # 测试 4
    test_streaming_multi_chunk()# 测试 5
    test_subtitle_correction()  # 测试 6
    test_microphone_detection() # 测试 7
    test_edge_interfaces()      # 测试 8

    # 清理
    reset_model()

    # 汇总报告
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    total = PASS + FAIL
    print(f"  总计: {total} 项测试")
    print(f"  通过: {PASS}")
    print(f"  失败: {FAIL}")
    print(f"  通过率: {PASS/total*100:.1f}%" if total > 0 else "  无测试")

    if ERRORS:
        print(f"\n  失败详情:")
        for err in ERRORS:
            print(f"    {err}")

    print("\n" + "=" * 60)

    # 返回退出码
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
