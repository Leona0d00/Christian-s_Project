/**
 * pipeline.c — Real-time transcription pipeline implementation.
 *
 * Reads WAV files, chunks audio, feeds through ASR, applies subtitle
 * correction, and returns final text with optional punctuation.
 */

#include "pipeline.h"
#include "config.h"
#include "audio.h"
#include "utils.h"

/* Sherpa-ONNX C API — needed for direct stream finalization */
#include "sherpa-onnx/c-api/c-api.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TranscriberPipeline* pipeline_create(
    const char *model_dir,
    int num_threads,
    const char *provider)
{
    TranscriberPipeline *p = (TranscriberPipeline*)calloc(1, sizeof(TranscriberPipeline));
    if (p == NULL) return NULL;

    p->model_dir = util_strdup(model_dir ? model_dir : DEFAULT_MODEL_DIR);
    p->num_threads = num_threads > 0 ? num_threads : DEFAULT_NUM_THREADS;
    if (provider && provider[0]) {
        strncpy(p->provider, provider, sizeof(p->provider) - 1);
    } else {
        strncpy(p->provider, DEFAULT_PROVIDER, sizeof(p->provider) - 1);
    }
    p->provider[sizeof(p->provider) - 1] = '\0';
    p->chunk_stride_samples = CHUNK_STRIDE_SAMPLES;

    /* Load recognizer */
    p->rec_ctx = recognizer_get_or_create(
        p->model_dir, p->num_threads, p->provider
    );

    if (!p->rec_ctx->is_loaded) {
        fprintf(stderr, "[pipeline] ERROR: Failed to load recognizer.\n");
        free(p->model_dir);
        free(p);
        return NULL;
    }

    /* Create session */
    p->session = session_create(p->rec_ctx);
    if (p->session == NULL) {
        fprintf(stderr, "[pipeline] ERROR: Failed to create session.\n");
        free(p->model_dir);
        free(p);
        return NULL;
    }

    /* Create corrector */
    p->corrector = corrector_create();
    if (p->corrector == NULL) {
        fprintf(stderr, "[pipeline] ERROR: Failed to create corrector.\n");
        session_destroy(p->session);
        free(p->model_dir);
        free(p);
        return NULL;
    }

    fprintf(stderr, "[pipeline] Pipeline created. Model: %s, Provider: %s, Threads: %d\n",
            p->model_dir, p->provider, p->num_threads);
    return p;
}

int pipeline_init_punctuation(TranscriberPipeline *p, const char *punct_model) {
    if (p == NULL) return 0;
    return punct_init(punct_model, p->num_threads, p->provider);
}

char* pipeline_process_file(TranscriberPipeline *p, const char *wav_path, int verbose) {
    if (p == NULL || wav_path == NULL) {
        return util_strdup("");
    }

    /* Reset for new file */
    pipeline_reset(p);

    /* Read WAV file */
    AudioBuffer audio;
    memset(&audio, 0, sizeof(audio));

    if (audio_read_wav(wav_path, &audio) != 0) {
        fprintf(stderr, "[pipeline] ERROR: Cannot read WAV file: %s\n", wav_path);
        return util_strdup("");
    }

    float duration_sec = (float)audio.num_samples / audio.sample_rate;
    fprintf(stderr, "[pipeline] Processing %s (%.2f s, %d samples)\n",
            wav_path, duration_sec, audio.num_samples);

    /* Process in chunks, simulating streaming.
     * IMPORTANT: Do NOT set is_final=1 on the last chunk.
     * The sherpa-onnx pattern is:
     *   1) Stream all chunks with is_final=0
     *   2) Add tail padding + call InputFinished once
     *   3) Drain decoder and get final result
     * Calling is_final=1 mid-loop then again with tail padding
     * causes corrupted output (double-finalize bug). */
    int chunk_size = p->chunk_stride_samples;
    int total_chunks = (audio.num_samples + chunk_size - 1) / chunk_size;

    for (int i = 0; i < total_chunks; i++) {
        int start = i * chunk_size;
        int end = start + chunk_size;
        if (end > audio.num_samples) end = audio.num_samples;

        int n = end - start;

        /* Pad final chunk if needed */
        float *chunk_data = NULL;
        int must_free = 0;
        if (n < chunk_size) {
            chunk_data = (float*)calloc(chunk_size, sizeof(float));
            if (chunk_data) {
                memcpy(chunk_data, audio.samples + start, n * sizeof(float));
                must_free = 1;
            } else {
                chunk_data = audio.samples + start;
            }
        } else {
            chunk_data = audio.samples + start;
        }

        /* Process chunk — always is_final=0 (streaming mode) */
        ProcessResult result = process_chunk(chunk_data, chunk_size, p->session, 0);

        if (result.error) {
            fprintf(stderr, "[pipeline] WARNING: process_chunk error: %s\n", result.error);
        }

        /* Apply subtitle correction */
        if (result.text && result.text[0]) {
            char *delta = corrector_correct_chunk(p->corrector, result.text);
            if (delta && delta[0] && verbose) {
                printf("%s", delta);
                fflush(stdout);
            }
            free(delta);
        }

        free(result.text);
        if (must_free) free(chunk_data);
    }

    fprintf(stderr, "[pipeline] Processed %d chunks.\n", total_chunks);

    /* Add tail padding (0.3s silence) and finalize ONCE */
    {
        float tail[4800];
        memset(tail, 0, sizeof(tail));
        SherpaOnnxOnlineStreamAcceptWaveform(p->session->stream, SAMPLE_RATE, tail, 4800);
        SherpaOnnxOnlineStreamInputFinished(p->session->stream);
        p->session->input_finished = 1;

        /* Drain decoder */
        const SherpaOnnxOnlineRecognizer *recognizer = p->rec_ctx->handle;
        while (SherpaOnnxIsOnlineStreamReady(recognizer, p->session->stream)) {
            SherpaOnnxDecodeOnlineStream(recognizer, p->session->stream);
        }

        /* Get final result */
        const SherpaOnnxOnlineRecognizerResult *r =
            SherpaOnnxGetOnlineStreamResult(recognizer, p->session->stream);
        if (r && r->text && r->text[0]) {
            char *delta = corrector_correct_chunk(p->corrector, r->text);
            if (delta && delta[0] && verbose) {
                printf("%s", delta);
                fflush(stdout);
            }
            free(delta);
        }
        if (r) SherpaOnnxDestroyOnlineRecognizerResult(r);
    }

    /* Get full raw text, then apply punctuation if available */
    char *raw_text = corrector_get_full_text(p->corrector);
    char *final_text;

    if (punct_is_loaded()) {
        final_text = punct_apply(raw_text);
        free(raw_text);
    } else {
        final_text = raw_text;
    }

    /* Post-process: strip whitespace */
    util_strip(final_text);

    /* Cleanup */
    audio_free(&audio);

    if (verbose) {
        printf("\n");
    }

    return final_text;
}

void pipeline_reset(TranscriberPipeline *p) {
    if (p == NULL) return;

    if (p->session) {
        session_reset(p->session);
    }
    if (p->corrector) {
        corrector_reset(p->corrector);
    }
}

void pipeline_destroy(TranscriberPipeline *p) {
    if (p == NULL) return;

    if (p->session) {
        session_destroy(p->session);
        p->session = NULL;
    }

    if (p->corrector) {
        corrector_destroy(p->corrector);
        p->corrector = NULL;
    }

    /* Don't destroy recognizer here — it's shared */
    /* recognizer_destroy() should be called separately */

    free(p->model_dir);
    free(p);

    fprintf(stderr, "[pipeline] Pipeline destroyed.\n");
}
