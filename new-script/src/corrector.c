/**
 * corrector.c — Subtitle corrector and punctuation restoration.
 *
 * SubtitleCorrector: Pure C dedup using longest-suffix-prefix matching.
 * Punctuation: Wraps sherpa-onnx's OfflinePunctuation C API.
 */

#include "corrector.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

/* Sherpa-ONNX C API */
#include "sherpa-onnx/c-api/c-api.h"

/* ================= Subtitle Corrector ================= */

SubtitleCorrector* corrector_create(void) {
    SubtitleCorrector *c = (SubtitleCorrector*)calloc(1, sizeof(SubtitleCorrector));
    if (c == NULL) return NULL;

    c->full_text = util_strdup("");
    c->displayed_text = util_strdup("");
    c->full_text_len = 0;
    c->displayed_text_len = 0;
    c->chunk_count = 0;

    return c;
}

void corrector_reset(SubtitleCorrector *c) {
    if (c == NULL) return;

    free(c->full_text);
    free(c->displayed_text);

    c->full_text = util_strdup("");
    c->displayed_text = util_strdup("");
    c->full_text_len = 0;
    c->displayed_text_len = 0;
    c->chunk_count = 0;
}

char* corrector_correct_chunk(SubtitleCorrector *c, const char *raw_text) {
    if (c == NULL || raw_text == NULL || raw_text[0] == '\0') {
        return NULL;
    }

    c->chunk_count++;

    /* Append to full accumulated text */
    char *new_full = util_strcat(c->full_text, raw_text);
    if (new_full) {
        free(c->full_text);
        c->full_text = new_full;
        c->full_text_len = (int)strlen(c->full_text);
    }

    /* First time: return the entire raw_text */
    if (c->displayed_text == NULL || c->displayed_text[0] == '\0') {
        free(c->displayed_text);
        c->displayed_text = util_strdup(raw_text);
        c->displayed_text_len = (int)strlen(raw_text);
        return util_strdup(raw_text);
    }

    /* Find overlap between displayed and new text */
    int overlap_len = util_find_overlap(c->displayed_text, raw_text);

    const char *delta;
    if (overlap_len > 0) {
        /* Ensure we don't split a UTF-8 character */
        int safe_pos = util_utf8_safe_cut_pos(raw_text, overlap_len);
        delta = raw_text + safe_pos;
    } else {
        /* No overlap — full text is new */
        delta = raw_text;
    }

    /* Update displayed text */
    free(c->displayed_text);
    c->displayed_text = util_strdup(raw_text);
    c->displayed_text_len = (int)strlen(raw_text);

    /* Return delta if non-empty */
    if (delta == NULL || delta[0] == '\0') {
        return NULL;
    }

    return util_strdup(delta);
}

char* corrector_get_full_text(SubtitleCorrector *c) {
    if (c == NULL || c->full_text == NULL) {
        return util_strdup("");
    }
    return util_strdup(c->full_text);
}

void corrector_destroy(SubtitleCorrector *c) {
    if (c == NULL) return;

    free(c->full_text);
    free(c->displayed_text);
    free(c);
}

/* ================= Offline Punctuation (sherpa-onnx) ================= */

/* Singleton punctuation model */
static struct {
    const SherpaOnnxOfflinePunctuation *handle;
    int is_loaded;
} g_punct = {NULL, 0};

int punct_init(const char *model_path, int num_threads, const char *provider) {
    if (g_punct.is_loaded && g_punct.handle != NULL) {
        return 1; /* Already loaded */
    }

    if (model_path == NULL || model_path[0] == '\0') {
        fprintf(stderr, "[punct] No model path provided, punctuation will be disabled.\n");
        return 0;
    }

    /* Check if file exists */
    FILE *f = fopen(model_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "[punct] WARNING: Model not found at %s. "
                        "Punctuation disabled.\n", model_path);
        return 0;
    }
    fclose(f);

    SherpaOnnxOfflinePunctuationConfig config;
    memset(&config, 0, sizeof(config));

    config.model.ct_transformer = model_path;
    config.model.num_threads = num_threads > 0 ? num_threads : 1;
    config.model.provider = (provider && provider[0]) ? provider : "cpu";
    config.model.debug = 0;

    fprintf(stderr, "[punct] Loading punctuation model: %s\n", model_path);

    g_punct.handle = SherpaOnnxCreateOfflinePunctuation(&config);

    if (g_punct.handle == NULL) {
        fprintf(stderr, "[punct] ERROR: Failed to load punctuation model.\n");
        return 0;
    }

    g_punct.is_loaded = 1;
    fprintf(stderr, "[punct] Punctuation model loaded.\n");
    return 1;
}

char* punct_apply(const char *text) {
    if (text == NULL || text[0] == '\0') {
        return util_strdup("");
    }

    if (!g_punct.is_loaded || g_punct.handle == NULL) {
        /* No punctuation model — return copy of input */
        return util_strdup(text);
    }

    /* Real sherpa-onnx offline punctuation API:
     * SherpaOfflinePunctuationAddPunct() returns const char* (not a struct).
     * The result must be freed with SherpaOfflinePunctuationFreeText(). */
    const char *result =
        SherpaOfflinePunctuationAddPunct(g_punct.handle, text);

    char *output;
    if (result != NULL && result[0] != '\0') {
        output = util_strdup(result);
    } else {
        output = util_strdup(text);
    }

    if (result != NULL) {
        SherpaOfflinePunctuationFreeText(result);
    }
    return output;
}

int punct_is_loaded(void) {
    return g_punct.is_loaded;
}

void punct_destroy(void) {
    if (g_punct.handle != NULL) {
        SherpaOnnxDestroyOfflinePunctuation(g_punct.handle);
        g_punct.handle = NULL;
    }
    g_punct.is_loaded = 0;
    fprintf(stderr, "[punct] Punctuation model destroyed.\n");
}
