/**
 * core.h — Core ASR processing module.
 *
 * Provides process_chunk(), the equivalent of Python's
 * script/Transcribe/_core.py process_chunk().
 *
 * This module wraps sherpa-onnx's streaming ASR C API behind a simple
 * stateful interface that mirrors the original FunASR-based API.
 */

#ifndef CORE_H
#define CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque forward declarations (sherpa-onnx types) */
struct SherpaOnnxOnlineRecognizer;
struct SherpaOnnxOnlineStream;

/* ================= Recognizer Singleton ================= */

/**
 * Shared recognizer context. One per process (singleton pattern).
 * Holds the loaded sherpa-onnx model.
 */
typedef struct {
    const struct SherpaOnnxOnlineRecognizer *handle;  /* sherpa-onnx recognizer */
    char model_path[1024];                            /* model directory */
    int  num_threads;
    char provider[32];                                /* "cpu" or "cuda" */
    int  is_loaded;                                   /* 1 = model loaded */
} RecognizerContext;

/* ================= Session State ================= */

/**
 * Per-session streaming state.
 * Replaces the Python `cache` dict.
 *
 * Created once per transcription session and passed to every
 * process_chunk() call for that session.
 */
typedef struct {
    const struct SherpaOnnxOnlineStream *stream;   /* sherpa-onnx stream */
    RecognizerContext *rec_ctx;                     /* shared recognizer */
    int input_finished;                             /* 1 = is_final sent */
    int chunk_count;                                /* number of chunks processed */
} SessionState;

/* ================= Result ================= */

/**
 * Return value from process_chunk().
 * The caller must free `text` (if non-NULL).
 * `error` is a static string — do NOT free.
 */
typedef struct {
    char *text;          /* Recognized text (heap-allocated, caller frees) */
    int   is_final;      /* Echo of the input is_final flag */
    const char *error;   /* Error message (static string) or NULL on success */
} ProcessResult;

/* ================= Core Functions ================= */

/**
 * Load the streaming ASR model (singleton).
 *
 * @param model_dir   Path to directory containing encoder.onnx, decoder.onnx,
 *                    joiner.onnx, and tokens.txt.
 * @param num_threads Number of CPU threads (default 4).
 * @param provider    ONNX execution provider: "cpu" (default).
 * @return RecognizerContext* (never NULL; check is_loaded field).
 */
RecognizerContext* recognizer_get_or_create(
    const char *model_dir,
    int num_threads,
    const char *provider
);

/**
 * Release the shared recognizer and free all associated resources.
 */
void recognizer_destroy(void);

/**
 * Get model information as a human-readable string.
 * Returns a static buffer — do NOT free.
 */
const char* recognizer_get_info(void);

/**
 * Create a new session state for a transcription.
 *
 * @param rec_ctx  The shared recognizer (must be loaded).
 * @return Newly allocated SessionState* (caller must pass to process_chunk
 *         and later free with session_destroy).
 */
SessionState* session_create(RecognizerContext *rec_ctx);

/**
 * Destroy a session state and release the associated sherpa-onnx stream.
 */
void session_destroy(SessionState *state);

/**
 * Reset a session for a new utterance (new stream).
 * Internally destroys the old stream and creates a new one.
 */
void session_reset(SessionState *state);

/**
 * Process a single audio chunk through the ASR model.
 *
 * This is THE core method — equivalent to Python's process_chunk().
 * It is independent of any GUI or pipeline and can be called directly
 * by test scripts.
 *
 * @param audio_chunk  float32 array, 16kHz, mono, shape (num_samples,)
 * @param num_samples  number of samples in audio_chunk
 * @param state        session state (carries stream & recognizer)
 * @param is_final     1 if this is the last chunk (flushes decoder)
 * @return ProcessResult with heap-allocated text (caller frees with free())
 *
 * Usage:
 *   SessionState *state = session_create(recognizer);
 *   ProcessResult r = process_chunk(audio, 9600, state, 0);
 *   printf("%s\n", r.text);
 *   free(r.text);
 */
ProcessResult process_chunk(
    const float *audio_chunk,
    int32_t num_samples,
    SessionState *state,
    int is_final
);

#ifdef __cplusplus
}
#endif

#endif /* CORE_H */
