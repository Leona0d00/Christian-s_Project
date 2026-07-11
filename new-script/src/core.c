/**
 * core.c — Core ASR processing using Sherpa-ONNX streaming C API.
 *
 * Implements process_chunk() — the heart of the transcription system.
 * Manages a singleton RecognizerContext (one per process) and per-session
 * SessionState objects that replicate the Python cache dict behavior.
 */

#include "core.h"
#include "config.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Sherpa-ONNX C API header */
#include "sherpa-onnx/c-api/c-api.h"

/* ================= Singleton Recognizer ================= */

static RecognizerContext g_recognizer = {0};

RecognizerContext* recognizer_get_or_create(
    const char *model_dir,
    int num_threads,
    const char *provider)
{
    if (g_recognizer.is_loaded && g_recognizer.handle != NULL) {
        /* Already loaded — check if config matches */
        if (strcmp(g_recognizer.model_path, model_dir) == 0 &&
            strcmp(g_recognizer.provider, provider) == 0) {
            return &g_recognizer;
        }
        /* Config changed — destroy and reload */
        recognizer_destroy();
    }

    /* Build paths to model files.
     * INT8 models use .int8.onnx suffix; standard models use .onnx.
     * Try INT8 variant first, then standard. */
    char tokens_path[2048];
    char encoder_path[2048];
    char decoder_path[2048];
    char joiner_path[2048];

    snprintf(tokens_path,  sizeof(tokens_path),  "%s/tokens.txt",  model_dir);

    /* Try encoder.int8.onnx first, then encoder.onnx */
    snprintf(encoder_path, sizeof(encoder_path), "%s/encoder.int8.onnx", model_dir);
    {
        FILE *f = fopen(encoder_path, "r");
        if (f == NULL) {
            snprintf(encoder_path, sizeof(encoder_path), "%s/encoder.onnx", model_dir);
        } else {
            fclose(f);
        }
    }

    /* decoder.onnx (no INT8 variant typically) */
    snprintf(decoder_path, sizeof(decoder_path), "%s/decoder.onnx", model_dir);
    {
        FILE *f = fopen(decoder_path, "r");
        if (f == NULL) {
            /* Try decoder.int8.onnx */
            snprintf(decoder_path, sizeof(decoder_path), "%s/decoder.int8.onnx", model_dir);
        } else {
            fclose(f);
        }
    }

    /* Try joiner.int8.onnx first, then joiner.onnx */
    snprintf(joiner_path, sizeof(joiner_path), "%s/joiner.int8.onnx", model_dir);
    {
        FILE *f = fopen(joiner_path, "r");
        if (f == NULL) {
            snprintf(joiner_path, sizeof(joiner_path), "%s/joiner.onnx", model_dir);
        } else {
            fclose(f);
        }
    }

    /* Validate at least tokens.txt exists */
    {
        FILE *f = fopen(tokens_path, "r");
        if (f == NULL) {
            fprintf(stderr, "[core] ERROR: tokens.txt not found at %s\n", tokens_path);
            return &g_recognizer;
        }
        fclose(f);
    }

    /* Configure the recognizer */
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.model_config.tokens = tokens_path;
    config.model_config.transducer.encoder = encoder_path;
    config.model_config.transducer.decoder = decoder_path;
    config.model_config.transducer.joiner = joiner_path;
    config.model_config.num_threads = num_threads > 0 ? num_threads : DEFAULT_NUM_THREADS;
    config.model_config.provider = (provider && provider[0]) ? provider : DEFAULT_PROVIDER;
    config.model_config.debug = 0;

    config.decoding_method = DECODING_METHOD;
    config.max_active_paths = MAX_ACTIVE_PATHS;

    config.feat_config.sample_rate = (float)SAMPLE_RATE;
    config.feat_config.feature_dim = FEATURE_DIM;

    config.enable_endpoint = ENABLE_ENDPOINT;
    config.rule1_min_trailing_silence = RULE1_MIN_TRAILING_SILENCE;
    config.rule2_min_trailing_silence = RULE2_MIN_TRAILING_SILENCE;
    config.rule3_min_utterance_length = RULE3_MIN_UTTERANCE_LENGTH;

    fprintf(stderr, "[core] Loading model from: %s\n", model_dir);
    fprintf(stderr, "[core]   tokens:  %s\n", tokens_path);
    fprintf(stderr, "[core]   encoder: %s\n", encoder_path);
    fprintf(stderr, "[core]   decoder: %s\n", decoder_path);
    fprintf(stderr, "[core]   joiner:  %s\n", joiner_path);
    fprintf(stderr, "[core]   provider: %s, threads: %d\n",
            config.model_config.provider, config.model_config.num_threads);

    const SherpaOnnxOnlineRecognizer *rec = SherpaOnnxCreateOnlineRecognizer(&config);

    if (rec == NULL) {
        fprintf(stderr, "[core] ERROR: Failed to create recognizer.\n");
        fprintf(stderr, "[core]   Check that the model files exist and are valid ONNX models.\n");
        return &g_recognizer;
    }

    /* Populate singleton */
    g_recognizer.handle = rec;
    strncpy(g_recognizer.model_path, model_dir, sizeof(g_recognizer.model_path) - 1);
    g_recognizer.model_path[sizeof(g_recognizer.model_path) - 1] = '\0';
    g_recognizer.num_threads = config.model_config.num_threads;
    strncpy(g_recognizer.provider, config.model_config.provider,
            sizeof(g_recognizer.provider) - 1);
    g_recognizer.provider[sizeof(g_recognizer.provider) - 1] = '\0';
    g_recognizer.is_loaded = 1;

    fprintf(stderr, "[core] Model loaded successfully.\n");
    return &g_recognizer;
}

void recognizer_destroy(void) {
    if (g_recognizer.handle != NULL) {
        SherpaOnnxDestroyOnlineRecognizer(g_recognizer.handle);
        g_recognizer.handle = NULL;
    }
    g_recognizer.is_loaded = 0;
    g_recognizer.model_path[0] = '\0';
    fprintf(stderr, "[core] Recognizer destroyed.\n");
}

const char* recognizer_get_info(void) {
    static char info[1024];
    snprintf(info, sizeof(info),
             "model: %s, provider: %s, threads: %d, loaded: %d",
             g_recognizer.model_path[0] ? g_recognizer.model_path : "(none)",
             g_recognizer.provider[0] ? g_recognizer.provider : "cpu",
             g_recognizer.num_threads,
             g_recognizer.is_loaded);
    return info;
}

/* ================= Session State ================= */

SessionState* session_create(RecognizerContext *rec_ctx) {
    if (rec_ctx == NULL || !rec_ctx->is_loaded || rec_ctx->handle == NULL) {
        fprintf(stderr, "[core] ERROR: Cannot create session — recognizer not loaded.\n");
        return NULL;
    }

    SessionState *state = (SessionState*)calloc(1, sizeof(SessionState));
    if (state == NULL) {
        fprintf(stderr, "[core] ERROR: malloc failed for SessionState.\n");
        return NULL;
    }

    state->stream = SherpaOnnxCreateOnlineStream(rec_ctx->handle);
    if (state->stream == NULL) {
        fprintf(stderr, "[core] ERROR: Failed to create online stream.\n");
        free(state);
        return NULL;
    }

    state->rec_ctx = rec_ctx;
    state->input_finished = 0;
    state->chunk_count = 0;

    return state;
}

void session_destroy(SessionState *state) {
    if (state == NULL) return;

    if (state->stream != NULL) {
        SherpaOnnxDestroyOnlineStream(state->stream);
        state->stream = NULL;
    }

    free(state);
}

void session_reset(SessionState *state) {
    if (state == NULL) return;

    if (state->stream != NULL) {
        SherpaOnnxDestroyOnlineStream(state->stream);
    }

    if (state->rec_ctx != NULL && state->rec_ctx->handle != NULL) {
        state->stream = SherpaOnnxCreateOnlineStream(state->rec_ctx->handle);
    } else {
        state->stream = NULL;
    }

    state->input_finished = 0;
    state->chunk_count = 0;
}

/* ================= Core Processing ================= */

ProcessResult process_chunk(
    const float *audio_chunk,
    int32_t num_samples,
    SessionState *state,
    int is_final)
{
    ProcessResult result;
    memset(&result, 0, sizeof(result));
    result.is_final = is_final;

    /* ---- Input validation ---- */
    if (audio_chunk == NULL || num_samples <= 0) {
        if (is_final) {
            /* Empty final chunk: just drain */
            /* fall through */
        } else {
            /* Empty non-final chunk: return empty */
            result.text = util_strdup("");
            return result;
        }
    }

    if (state == NULL) {
        result.error = "SessionState is NULL";
        result.text = util_strdup("");
        return result;
    }

    if (state->rec_ctx == NULL || !state->rec_ctx->is_loaded ||
        state->rec_ctx->handle == NULL) {
        result.error = "Recognizer not loaded. Call recognizer_get_or_create() first.";
        result.text = util_strdup("");
        return result;
    }

    if (state->stream == NULL) {
        result.error = "Stream is NULL. Call session_create() or session_reset() first.";
        result.text = util_strdup("");
        return result;
    }

    /* ---- Accept waveform into stream ---- */
    if (audio_chunk != NULL && num_samples > 0) {
        SherpaOnnxOnlineStreamAcceptWaveform(
            state->stream,
            SAMPLE_RATE,
            audio_chunk,
            num_samples
        );
    }

    /* ---- Mark end of input ---- */
    if (is_final && !state->input_finished) {
        SherpaOnnxOnlineStreamInputFinished(state->stream);
        state->input_finished = 1;
    }

    /* ---- Drain decoder ---- */
    const SherpaOnnxOnlineRecognizer *recognizer = state->rec_ctx->handle;

    while (SherpaOnnxIsOnlineStreamReady(recognizer, state->stream)) {
        SherpaOnnxDecodeOnlineStream(recognizer, state->stream);
    }

    /* ---- Get result ---- */
    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, state->stream);

    if (r != NULL && r->text != NULL && strlen(r->text) > 0) {
        result.text = util_strdup(r->text);
    } else {
        result.text = util_strdup("");
    }

    /* ---- Check endpoint (sentence boundary) ---- */
    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, state->stream)) {
        SherpaOnnxOnlineStreamReset(recognizer, state->stream);
    }

    /* ---- Cleanup ---- */
    if (r != NULL) {
        SherpaOnnxDestroyOnlineRecognizerResult(r);
    }

    state->chunk_count++;
    return result;
}
