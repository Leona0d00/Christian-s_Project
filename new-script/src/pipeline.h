/**
 * pipeline.h — Real-time transcription pipeline.
 *
 * Ties together audio reading, ASR processing, and subtitle correction
 * into a complete transcription workflow.
 * Mirrors script/Transcribe/_pipeline.py RealTimeTranscriber.
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include "core.h"
#include "corrector.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================= Transcriber Pipeline ================= */

typedef struct {
    RecognizerContext *rec_ctx;      /* Shared recognizer */
    SessionState      *session;      /* Per-session stream state */
    SubtitleCorrector *corrector;    /* Dedup state */
    char              *model_dir;    /* Model directory path */
    int                num_threads;
    char               provider[32]; /* "cpu" */
    int                chunk_stride_samples;
} TranscriberPipeline;

/**
 * Create a new pipeline.
 *
 * @param model_dir    Path to ASR model directory.
 * @param num_threads  Number of CPU threads.
 * @param provider     ONNX provider ("cpu").
 * @return New pipeline (caller must call pipeline_destroy).
 */
TranscriberPipeline* pipeline_create(
    const char *model_dir,
    int num_threads,
    const char *provider
);

/**
 * Initialize punctuation model for this pipeline.
 *
 * @param p            Pipeline.
 * @param punct_model  Path to punctuation model ONNX file.
 * @return 1 on success, 0 if model unavailable (pipeline still works).
 */
int pipeline_init_punctuation(TranscriberPipeline *p, const char *punct_model);

/**
 * Process a WAV file end-to-end, simulating real-time streaming.
 *
 * Chunks the audio and feeds it through the ASR model chunk by chunk.
 * For each chunk, prints incremental text (if verbose) and returns
 * the final corrected text.
 *
 * @param p         Pipeline.
 * @param wav_path  Path to input WAV file.
 * @param verbose   1 to print incremental output, 0 for silent mode.
 * @return Final corrected text (caller frees with free()).
 */
char* pipeline_process_file(TranscriberPipeline *p, const char *wav_path, int verbose);

/**
 * Reset the pipeline for a new file/utterance.
 */
void pipeline_reset(TranscriberPipeline *p);

/**
 * Destroy the pipeline and free all resources.
 */
void pipeline_destroy(TranscriberPipeline *p);

#ifdef __cplusplus
}
#endif

#endif /* PIPELINE_H */
