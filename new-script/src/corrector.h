/**
 * corrector.h — Subtitle correction module.
 *
 * Provides SubtitleCorrector: real-time dedup for streaming ASR output.
 * Mirrors script/Transcribe/_corrector.py.
 *
 * Also provides offline punctuation restoration via sherpa-onnx's
 * ct-transformer-zh-en model.
 */

#ifndef CORRECTOR_H
#define CORRECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================= Subtitle Corrector ================= */

/**
 * Subtitle corrector — tracks displayed text and outputs only
 * the incremental (non-overlapping) portion of each new chunk.
 *
 * Algorithm: longest-prefix-overlap dedup.
 *   old = "欢迎大", new = "欢迎大家" → overlap = "欢迎大" (3 chars)
 *   delta = new[3:] = "家"
 */
typedef struct {
    char *full_text;         /* All raw text accumulated (heap) */
    int   full_text_len;
    char *displayed_text;    /* Last text displayed (heap) */
    int   displayed_text_len;
    int   chunk_count;
} SubtitleCorrector;

/**
 * Allocate and initialize a new corrector.
 */
SubtitleCorrector* corrector_create(void);

/**
 * Reset the corrector for a new transcription session.
 */
void corrector_reset(SubtitleCorrector *c);

/**
 * Process a raw text chunk. Returns the incremental (delta) text
 * that should be displayed. Caller must free() the returned string.
 *
 * Returns NULL if there is nothing new to display.
 */
char* corrector_correct_chunk(SubtitleCorrector *c, const char *raw_text);

/**
 * Finalize the session and return the full accumulated raw text.
 * Caller must free() the returned string.
 */
char* corrector_get_full_text(SubtitleCorrector *c);

/**
 * Destroy the corrector and free all memory.
 */
void corrector_destroy(SubtitleCorrector *c);

/* ================= Offline Punctuation (sherpa-onnx) ================= */

/**
 * Initialize the punctuation model (singleton).
 *
 * @param model_path  Path to ct-transformer model ONNX file
 *                    (e.g., model.int8.onnx)
 * @param num_threads Number of CPU threads
 * @param provider    ONNX provider ("cpu")
 * @return 1 on success, 0 on failure
 */
int punct_init(const char *model_path, int num_threads, const char *provider);

/**
 * Apply punctuation to text using the loaded model.
 * Returns newly allocated string with punctuation added.
 * Caller must free() the result.
 *
 * If the model is not loaded, returns a copy of the input.
 */
char* punct_apply(const char *text);

/**
 * Check if the punctuation model is loaded.
 */
int punct_is_loaded(void);

/**
 * Release the punctuation model.
 */
void punct_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* CORRECTOR_H */
