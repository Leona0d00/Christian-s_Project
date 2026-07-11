/**
 * config.h — Configuration constants for the Chinese real-time ASR project.
 *
 * Mirrors the original script/Transcribe/_config.py.
 * All audio, model, and performance parameters are centralized here.
 */

#ifndef CONFIG_H
#define CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================= Audio Configuration ================= */
#define SAMPLE_RATE             16000
#define CHUNK_STRIDE_MS         600
#define CHUNK_STRIDE_SAMPLES    (SAMPLE_RATE * CHUNK_STRIDE_MS / 1000)  /* 9600 */
#define CHANNELS                1

/* ================= Streaming Model Configuration ================= */
/* chunk_size is handled internally by sherpa-onnx transducer.
   These values are for documentation and for computing stride. */
#define FEATURE_DIM             80

/* ================= Model Paths (defaults) ================= */
#define DEFAULT_MODEL_DIR       "./models/asr/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
#define DEFAULT_PUNCT_MODEL_DIR "./models/punct/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
#define DEFAULT_PUNCT_MODEL     "./models/punct/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/model.int8.onnx"

/* ================= Runtime Configuration ================= */
#define DEFAULT_NUM_THREADS     4
#define DEFAULT_PROVIDER        "cpu"
#define DECODING_METHOD         "greedy_search"
#define MAX_ACTIVE_PATHS        4

/* ================= Endpoint Detection ================= */
#define ENABLE_ENDPOINT         1
#define RULE1_MIN_TRAILING_SILENCE  2.4f
#define RULE2_MIN_TRAILING_SILENCE  1.2f
#define RULE3_MIN_UTTERANCE_LENGTH  300.0f

/* ================= Edge Deployment Profiles ================= */
#define EDGE_PROFILE_COUNT 3

#ifdef __cplusplus
}
#endif

#endif /* CONFIG_H */
