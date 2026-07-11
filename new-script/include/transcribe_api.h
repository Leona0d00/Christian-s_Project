/**
 * transcribe_api.h — Public C API for the Chinese real-time ASR system.
 *
 * This is the DLL export header. External callers (C/C++ projects,
 * Python ctypes, edge deployment systems) use this header to call
 * the transcription engine.
 *
 * All functions are thread-safe at the handle level (each handle is
 * independent). Strings returned are heap-allocated and must be freed
 * by the caller with free().
 */

#ifndef TRANSCRIBE_API_H
#define TRANSCRIBE_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================= DLL Export Macro ================= */

#ifdef _WIN32
  #ifdef TRANSCRIBE_EXPORTS
    #define TRANSCRIBE_API __declspec(dllexport)
  #else
    #define TRANSCRIBE_API __declspec(dllimport)
  #endif
#else
  #define TRANSCRIBE_API
#endif

/* ================= Opaque Handle ================= */

/**
 * Opaque handle to a transcription session.
 * Created by transcriber_create(), used by all other functions.
 */
typedef struct TranscriberHandle_s TranscriberHandle;

/* ================= Creation / Destruction ================= */

/**
 * Create a new transcriber handle.
 *
 * @param model_dir    Path to directory containing ASR model files
 *                     (encoder.onnx, decoder.onnx, joiner.onnx, tokens.txt).
 * @param num_threads  Number of CPU threads for ONNX inference (default: 4).
 * @param provider     ONNX execution provider: "cpu" (default).
 * @return Handle pointer, or NULL on failure.
 */
TRANSCRIBE_API TranscriberHandle* transcriber_create(
    const char *model_dir,
    int num_threads,
    const char *provider
);

/**
 * Destroy a transcriber handle and free all resources.
 */
TRANSCRIBE_API void transcriber_destroy(TranscriberHandle *handle);

/* ================= Core Processing ================= */

/**
 * Process a single audio chunk.
 *
 * This is THE core method — equivalent to Python's process_chunk().
 * Call it repeatedly with chunks of audio data, sharing the same handle.
 *
 * @param handle       Transcriber handle.
 * @param audio        Float32 array, 16kHz, mono.
 * @param num_samples  Number of samples in audio array.
 * @param is_final     1 if this is the last chunk (flushes decoder buffers).
 * @return Recognized text for this chunk (heap-allocated, caller frees).
 *         Empty string "" if nothing recognized. Never returns NULL.
 */
TRANSCRIBE_API char* transcriber_process_chunk(
    TranscriberHandle *handle,
    const float *audio,
    int num_samples,
    int is_final
);

/* ================= Subtitle Correction ================= */

/**
 * Get the incremental (dedup'd) display text after a chunk.
 *
 * Applies subtitle corrector dedup: if the new text overlaps with
 * previously displayed text, only the non-overlapping portion is returned.
 *
 * @param handle    Transcriber handle.
 * @param raw_text  Raw recognized text from process_chunk.
 * @return Incremental display text (heap-allocated, caller frees),
 *         or NULL if nothing new to display.
 */
TRANSCRIBE_API char* transcriber_get_display_text(
    TranscriberHandle *handle,
    const char *raw_text
);

/* ================= Finalization ================= */

/**
 * Finalize the transcription session.
 *
 * Returns the full accumulated text with punctuation applied
 * (if punctuation model was loaded).
 *
 * @param handle  Transcriber handle.
 * @return Full corrected text (heap-allocated, caller frees).
 */
TRANSCRIBE_API char* transcriber_finalize(TranscriberHandle *handle);

/**
 * Reset the transcriber for a new utterance.
 * Clears the subtitle corrector state and creates a new ASR stream.
 */
TRANSCRIBE_API void transcriber_reset(TranscriberHandle *handle);

/* ================= File Processing ================= */

/**
 * Process an entire WAV file and return the recognized text.
 * Convenience function — internally calls process_chunk in a loop.
 *
 * @param handle    Transcriber handle.
 * @param wav_path  Path to WAV file (16kHz mono preferred, auto-resamples).
 * @param verbose   1 to print incremental text, 0 for silent.
 * @return Full recognized and punctuated text (heap-allocated, caller frees).
 */
TRANSCRIBE_API char* transcriber_process_file(
    TranscriberHandle *handle,
    const char *wav_path,
    int verbose
);

/* ================= Punctuation ================= */

/**
 * Load the punctuation model.
 *
 * @param handle       Transcriber handle.
 * @param punct_model  Path to ct-transformer punctuation model ONNX file.
 * @return 1 on success, 0 on failure (transcription still works without it).
 */
TRANSCRIBE_API int transcriber_load_punctuation(
    TranscriberHandle *handle,
    const char *punct_model
);

/* ================= Version / Info ================= */

/**
 * Return the version string of this library.
 * Static string — do NOT free.
 */
TRANSCRIBE_API const char* transcriber_version(void);

/**
 * Return a human-readable info string about the current recognizer.
 * Static string — do NOT free.
 */
TRANSCRIBE_API const char* transcriber_get_info(TranscriberHandle *handle);

/* ================= Edge Deployment ================= */

/**
 * Get the number of available compute devices.
 */
TRANSCRIBE_API int transcriber_get_device_count(void);

/**
 * Get info for a compute device by index.
 *
 * @param index       Device index (0 to device_count-1).
 * @param name        Output buffer for device name.
 * @param name_size   Size of name buffer.
 * @param available   Output: 1 if device is available.
 * @return 0 on success, -1 if index out of bounds.
 */
TRANSCRIBE_API int transcriber_get_device_info(
    int index,
    char *name,
    int name_size,
    int *available
);

/**
 * Get the optimization guide for edge deployment.
 * Static string — do NOT free.
 */
TRANSCRIBE_API const char* transcriber_get_optimization_guide(void);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_API_H */
