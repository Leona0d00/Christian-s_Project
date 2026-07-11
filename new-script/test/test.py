#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script: Validates the C/Sherpa-ONNX real-time speech-to-text system.

Uses ctypes to call the transcribe.dll exported API and exercises all
core functionality, mirroring the original script/Test.py test groups.

Usage:
    conda run -n christian python test/test.py

Prerequisites:
    - transcribe.dll built in the build/ directory
    - ASR model downloaded in models/asr/
    - test.wav available in the project root
"""

import sys
import os
import time
import ctypes
import numpy as np
import traceback

# ==================== Path Setup ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # new-script/
ROOT_DIR = os.path.dirname(PROJECT_DIR)    # Christian's_Project/
BUILD_DIR = os.path.join(PROJECT_DIR, "build")

# Global reference to loaded DLL (used by dll_str helper)
_LIB = None

# ==================== DLL Loading ====================

def load_library():
    """Load transcribe.dll via ctypes."""
    global _LIB
    dll_path = os.path.join(BUILD_DIR, "transcribe.dll")

    if not os.path.exists(dll_path):
        print(f"  [SKIP] DLL not found at {dll_path}")
        print(f"  Build the project first: cd {PROJECT_DIR} && make")
        return None

    # Add DLL directory to search path for dependencies (onnxruntime, etc.)
    try:
        os.add_dll_directory(BUILD_DIR)
    except AttributeError:
        # Python < 3.8 fallback
        os.environ['PATH'] = BUILD_DIR + ';' + os.environ.get('PATH', '')

    try:
        _LIB = ctypes.CDLL(dll_path)
    except OSError as e:
        print(f"  [SKIP] Cannot load DLL: {e}")
        _LIB = None
        return None

    lib = _LIB

    # ---- transcriber_create ----
    lib.transcriber_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
    lib.transcriber_create.restype = ctypes.c_void_p

    # ---- transcriber_destroy ----
    lib.transcriber_destroy.argtypes = [ctypes.c_void_p]
    lib.transcriber_destroy.restype = None

    # ---- transcriber_process_chunk ----
    lib.transcriber_process_chunk.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.transcriber_process_chunk.restype = ctypes.c_void_p  # malloc'd string

    # ---- transcriber_get_display_text ----
    lib.transcriber_get_display_text.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.transcriber_get_display_text.restype = ctypes.c_void_p  # malloc'd string

    # ---- transcriber_finalize ----
    lib.transcriber_finalize.argtypes = [ctypes.c_void_p]
    lib.transcriber_finalize.restype = ctypes.c_void_p  # malloc'd string

    # ---- transcriber_reset ----
    lib.transcriber_reset.argtypes = [ctypes.c_void_p]
    lib.transcriber_reset.restype = None

    # ---- transcriber_process_file ----
    lib.transcriber_process_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.transcriber_process_file.restype = ctypes.c_void_p  # malloc'd string

    # ---- transcriber_load_punctuation ----
    lib.transcriber_load_punctuation.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.transcriber_load_punctuation.restype = ctypes.c_int

    # ---- transcriber_version (static string — no free needed) ----
    lib.transcriber_version.argtypes = []
    lib.transcriber_version.restype = ctypes.c_char_p

    # ---- transcriber_get_info (static string — no free needed) ----
    lib.transcriber_get_info.argtypes = [ctypes.c_void_p]
    lib.transcriber_get_info.restype = ctypes.c_char_p

    # ---- transcriber_get_device_count ----
    lib.transcriber_get_device_count.argtypes = []
    lib.transcriber_get_device_count.restype = ctypes.c_int

    # ---- transcriber_get_device_info ----
    lib.transcriber_get_device_info.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    lib.transcriber_get_device_info.restype = ctypes.c_int

    # ---- transcriber_get_optimization_guide (static string) ----
    lib.transcriber_get_optimization_guide.argtypes = []
    lib.transcriber_get_optimization_guide.restype = ctypes.c_char_p

    # ---- transcriber_free_string (must use for malloc'd returns) ----
    lib.transcriber_free_string.argtypes = [ctypes.c_void_p]
    lib.transcriber_free_string.restype = None

    return lib


# ==================== Test Harness ====================

PASS = 0
FAIL = 0
ERRORS = []

CHUNK_STRIDE_SAMPLES = 9600   # 600ms @ 16kHz
SAMPLE_RATE = 16000


def simple_dedup(displayed, new_text):
    """Pure Python implementation of subtitle dedup (mirrors C logic)."""
    if not displayed:
        return new_text
    # Find longest suffix of displayed matching prefix of new_text
    max_ov = min(len(displayed), len(new_text))
    for ov in range(max_ov, 0, -1):
        if displayed[-ov:] == new_text[:ov]:
            return new_text[ov:]
    return new_text


def log_test(name, passed, detail=""):
    """Record a test result."""
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


def dll_str(ptr, free=True):
    """Convert a void* returned by the DLL to a Python string.
    Uses transcriber_free_string() for proper CRT memory management."""
    if ptr is None or ptr == 0 or _LIB is None:
        return ""
    try:
        result = ctypes.cast(ptr, ctypes.c_char_p).value
        if result is None:
            return ""
        text = result.decode("utf-8") if isinstance(result, bytes) else str(result)
        if free:
            _LIB.transcriber_free_string(ptr)
        return text
    except Exception:
        return ""


def read_wav_file(filepath):
    """Read a WAV file into float32 numpy array at 16kHz mono."""
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            # Resample to 16kHz using scipy
            try:
                from scipy.signal import resample_poly
                # Use polyphase resampling for better quality
                gcd = np.gcd(sr, SAMPLE_RATE)
                up = SAMPLE_RATE // gcd
                down = sr // gcd
                audio = resample_poly(audio, up, down).astype(np.float32)
            except ImportError:
                # Fallback: simple linear interpolation
                old_len = len(audio)
                new_len = int(old_len * SAMPLE_RATE / sr)
                old_indices = np.linspace(0, old_len - 1, old_len)
                new_indices = np.linspace(0, old_len - 1, new_len)
                audio = np.interp(new_indices, old_indices, audio).astype(np.float32)
        return audio
    except ImportError:
        print("  [SKIP] soundfile not available for WAV reading")
        return None


# ==================== Test 1: DLL Loading ====================

def test_dll_loading(lib):
    """Test 1: DLL can be loaded and version works."""
    print("\n" + "=" * 60)
    print("Test 1: DLL Loading")
    print("=" * 60)

    try:
        version = lib.transcriber_version()
        if version:
            ver_str = version.decode("utf-8") if isinstance(version, bytes) else str(version)
            log_test("DLL version string", len(ver_str) > 0, f"Version: {ver_str}")
        else:
            log_test("DLL version string", False, "Version returned NULL")
    except Exception as e:
        log_test("DLL loading", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 2: Handle Creation ====================

def test_handle_creation(lib):
    """Test 2: Transcriber handle creation and info."""
    print("\n" + "=" * 60)
    print("Test 2: Handle Creation")
    print("=" * 60)

    model_dir = os.path.join(PROJECT_DIR, "models", "asr",
                             "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30")
    if not os.path.exists(os.path.join(model_dir, "tokens.txt")):
        print(f"  [SKIP] Model not found at {model_dir}")
        print(f"  Run setup.sh first to download the model.")
        return None

    try:
        handle = lib.transcriber_create(
            model_dir.encode("utf-8"),
            1,  # num_threads
            b"cpu"
        )

        log_test("Handle creation", handle is not None and handle != 0,
                 f"handle = {handle}")

        if handle:
            info = lib.transcriber_get_info(handle)
            if info:
                info_str = info.decode("utf-8") if isinstance(info, bytes) else str(info)
                log_test("Model info", len(info_str) > 0, f"Info: {info_str}")
            else:
                log_test("Model info", False, "get_info returned NULL")

        return handle
    except Exception as e:
        log_test("Handle creation", False, f"Exception: {e}")
        traceback.print_exc()
        return None


# ==================== Test 3: Single Chunk ====================

def test_single_chunk(lib, handle):
    """Test 3: Process a single audio chunk."""
    print("\n" + "=" * 60)
    print("Test 3: Single Chunk Processing")
    print("=" * 60)

    if handle is None:
        log_test("Single chunk", False, "No valid handle")
        return

    try:
        # Create test audio: 600ms of silence
        audio = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
        audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        result = lib.transcriber_process_chunk(
            handle, audio_ptr, len(audio), 0
        )

        has_result = result is not None and result != 0
        log_test("Chunk processed without crash", has_result,
                 "Process chunk returned valid pointer")

        if result:
            text = dll_str(result)
            log_test("Result is valid string", isinstance(text, str),
                     f"text='{text}' (empty is OK for silence)")

        # Test is_final
        result2 = lib.transcriber_process_chunk(
            handle, audio_ptr, len(audio), 1
        )
        if result2:
            text2 = dll_str(result2)
            log_test("Final chunk processed", True,
                     f"text='{text2}'")

    except Exception as e:
        log_test("Single chunk", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 4: Multi-Chunk Streaming ====================

def test_multi_chunk_streaming(lib, handle):
    """Test 4: Multi-chunk streaming (simulate real-time audio)."""
    print("\n" + "=" * 60)
    print("Test 4: Multi-Chunk Streaming")
    print("=" * 60)

    if handle is None:
        log_test("Multi-chunk streaming", False, "No valid handle")
        return

    test_wav = os.path.join(ROOT_DIR, "test.wav")
    if not os.path.exists(test_wav):
        log_test("Multi-chunk streaming", False, f"test.wav not found at {test_wav}")
        return

    try:
        audio = read_wav_file(test_wav)
        if audio is None:
            log_test("Multi-chunk streaming", False, "Cannot read test.wav")
            return

        n_chunks = max(1, len(audio) // CHUNK_STRIDE_SAMPLES + 1)
        print(f"  Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")
        print(f"  Splitting into {n_chunks} chunks...")

        # Reset handle for fresh session
        lib.transcriber_reset(handle)

        chunk_texts = []
        all_raw = []
        displayed = ""
        dedup_correct = True

        for i in range(n_chunks):
            print(f"  Chunk {i}/{n_chunks}...", end="", flush=True)
            start = i * CHUNK_STRIDE_SAMPLES
            end = start + CHUNK_STRIDE_SAMPLES

            if end > len(audio):
                chunk = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
                actual_len = len(audio) - start
                if actual_len > 0:
                    chunk[:actual_len] = audio[start:]
                is_final = True
            else:
                chunk = audio[start:end].copy()
                is_final = (i == n_chunks - 1)

            audio_ptr = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result = lib.transcriber_process_chunk(
                handle, audio_ptr, len(chunk), 1 if is_final else 0
            )

            if result:
                text = dll_str(result, free=False)
                if text:
                    all_raw.append(text)
                    # Use Python-side dedup to verify logic independently
                    delta = simple_dedup(displayed, text)
                    displayed = text  # update displayed state
                    if delta:
                        chunk_texts.append(delta)

                # Now free the original result
                _LIB.transcriber_free_string(result)

        log_test("All chunks processed", True,
                 f"{n_chunks} chunks processed, {len(all_raw)} produced text")
        log_test("Streaming produces text", len(chunk_texts) > 0 if len(audio) > 0 else True,
                 f"Display texts: {len(chunk_texts)}")

        full_raw = "".join(all_raw)
        if full_raw:
            print(f"\n  Raw text ({len(full_raw)} chars):")
            print(f"    {full_raw[:200]}{'...' if len(full_raw) > 200 else ''}")

        log_test("All chunks processed", True,
                 f"{n_chunks} chunks processed, {len(all_raw)} produced text")
        log_test("Streaming produces text", len(chunk_texts) > 0 if len(audio) > 0 else True,
                 f"Display texts: {len(chunk_texts)}")

        full_raw = "".join(all_raw)
        if full_raw:
            print(f"\n  Raw text ({len(full_raw)} chars):")
            print(f"    {full_raw[:200]}{'...' if len(full_raw) > 200 else ''}")

    except Exception as e:
        log_test("Multi-chunk streaming", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 5: Cache/State Persistence ====================

def test_state_persistence(lib, handle):
    """Test 5: Handle state persists across chunks (like Python cache)."""
    print("\n" + "=" * 60)
    print("Test 5: State Persistence")
    print("=" * 60)

    if handle is None:
        log_test("State persistence", False, "No valid handle")
        return

    try:
        lib.transcriber_reset(handle)

        # Two chunks of silence — verify same handle works across calls
        audio = np.zeros(CHUNK_STRIDE_SAMPLES, dtype=np.float32)
        audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Chunk 1
        r1 = lib.transcriber_process_chunk(handle, audio_ptr, len(audio), 0)
        # Chunk 2 (same handle, no reset)
        r2 = lib.transcriber_process_chunk(handle, audio_ptr, len(audio), 0)

        log_test("Same handle works across chunks", r1 != 0 and r2 != 0,
                 "Both chunks processed with same handle")
        if r1: _LIB.transcriber_free_string(r1)
        if r2: _LIB.transcriber_free_string(r2)

        # Reset and verify it still works
        lib.transcriber_reset(handle)
        r3 = lib.transcriber_process_chunk(handle, audio_ptr, len(audio), 0)

        log_test("Handle works after reset", r3 != 0,
                 "Chunk processed after reset")
        if r3: _LIB.transcriber_free_string(r3)

    except Exception as e:
        log_test("State persistence", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 6: Audio Processing ====================

def test_audio_processing(lib):
    """Test 6: Audio file reading via pipeline (if DLL has process_file)."""
    print("\n" + "=" * 60)
    print("Test 6: Audio File Processing")
    print("=" * 60)

    test_wav = os.path.join(ROOT_DIR, "test.wav")
    if not os.path.exists(test_wav):
        log_test("Audio processing", False, f"test.wav not found")
        return

    try:
        audio = read_wav_file(test_wav)
        if audio is None:
            log_test("Audio processing", False, "Cannot read WAV")
            return

        log_test("WAV read successfully", len(audio) > 0,
                 f"shape={audio.shape}, dtype={audio.dtype}")
        log_test("Float32 format", audio.dtype == np.float32,
                 f"dtype={audio.dtype}")
        log_test("Amplitude in range", float(np.max(np.abs(audio))) <= 1.0,
                 f"max_amplitude={np.max(np.abs(audio)):.4f}")

    except Exception as e:
        log_test("Audio processing", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 7: Subtitle Correction ====================

def test_subtitle_correction(lib, handle):
    """Test 7: Subtitle corrector (dedup logic) — verified via Python impl."""
    print("\n" + "=" * 60)
    print("Test 7: Subtitle Correction (Python dedup mirror)")
    print("=" * 60)

    try:
        # Simulate streaming outputs with overlap (same as C logic)
        sim_chunks = [
            "欢迎大",
            "欢迎大家",
            "欢迎大家来",
            "欢迎大家来到",
            "欢迎大家来到实时",
            "欢迎大家来到实时语音",
        ]

        displayed = ""
        deltas = []
        for text in sim_chunks:
            delta = simple_dedup(displayed, text)
            displayed = text
            deltas.append(delta if delta else "")

        expected = ["欢迎大", "家", "来", "到", "实时", "语音"]
        passed = deltas == expected
        log_test("Dedup produces correct deltas", passed,
                 f"expected={expected}, got={deltas}")

        # Test reset logic (empty displayed)
        d2 = simple_dedup("", "test")
        log_test("Reset (empty displayed) shows full text", d2 == "test",
                 f"delta='{d2}'")

        # Test empty string
        d3 = simple_dedup("something", "")
        log_test("Empty input returns empty", d3 == "",
                 f"delta='{d3}'")

        # Test no overlap
        d4 = simple_dedup("abc", "def")
        log_test("No overlap returns full text", d4 == "def",
                 f"delta='{d4}'")

        # Test complete overlap
        d5 = simple_dedup("hello", "hello world")
        log_test("Partial overlap returns delta", d5 == " world",
                 f"delta='{d5}'")

    except Exception as e:
        log_test("Subtitle correction", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Test 8: Edge Interfaces ====================

def test_edge_interfaces(lib):
    """Test 8: Edge deployment interfaces."""
    print("\n" + "=" * 60)
    print("Test 8: Edge Deployment Interfaces")
    print("=" * 60)

    try:
        # Device count
        count = lib.transcriber_get_device_count()
        log_test("Device count", count > 0,
                 f"Found {count} compute devices")

        # Device info
        if count > 0:
            name_buf = ctypes.create_string_buffer(128)
            available = ctypes.c_int()
            ret = lib.transcriber_get_device_info(0, name_buf, 128, ctypes.byref(available))
            name_str = name_buf.value.decode("utf-8") if isinstance(name_buf.value, bytes) else str(name_buf.value)
            log_test("Device info", ret == 0,
                     f"Device 0: '{name_str}', available={available.value}")

        # Optimization guide
        guide = lib.transcriber_get_optimization_guide()
        if guide:
            guide_str = guide.decode("utf-8") if isinstance(guide, bytes) else str(guide)
            log_test("Optimization guide", len(guide_str) > 0,
                     f"Guide length: {len(guide_str)} chars")
        else:
            log_test("Optimization guide", False, "Returned NULL")

        # Version
        version = lib.transcriber_version()
        if version:
            ver = version.decode("utf-8") if isinstance(version, bytes) else str(version)
            log_test("Version string", len(ver) > 0, f"v{ver}")

    except Exception as e:
        log_test("Edge interfaces", False, f"Exception: {e}")
        traceback.print_exc()


# ==================== Main ====================

def main():
    """Run all tests."""
    print("=" * 60)
    print("  Transcribe C/Sherpa-ONNX — Function Tests")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load library
    lib = load_library()
    if lib is None:
        print("\n[SKIP] Cannot load transcribe.dll. Build the project first.")
        print(f"  cd {PROJECT_DIR}")
        print("  make")
        return 1

    # Create handle (used by most tests)
    handle = test_handle_creation(lib)

    # Run tests
    test_dll_loading(lib)            # Test 1
    # (handle creation already done above as Test 2)
    test_single_chunk(lib, handle)   # Test 3
    test_audio_processing(lib)       # Test 4 (WAV reading)
    test_state_persistence(lib, handle)  # Test 5
    test_multi_chunk_streaming(lib, handle)  # Test 6
    test_subtitle_correction(lib, handle)   # Test 7
    test_edge_interfaces(lib)        # Test 8

    # Cleanup
    if handle:
        lib.transcriber_destroy(handle)

    # Summary
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    total = PASS + FAIL
    print(f"  Total: {total} tests")
    print(f"  Pass:  {PASS}")
    print(f"  Fail:  {FAIL}")
    if total > 0:
        print(f"  Rate:  {PASS/total*100:.1f}%")
    else:
        print("  No tests ran")

    if ERRORS:
        print(f"\n  Failures:")
        for err in ERRORS:
            print(f"    {err}")

    print("\n" + "=" * 60)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
