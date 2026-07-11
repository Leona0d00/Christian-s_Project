/**
 * audio.c — Audio I/O implementation.
 *
 * Uses SherpaOnnxReadWave for WAV reading. Falls back to manual resampling
 * via linear interpolation when sample rates differ.
 */

#include "audio.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Sherpa-ONNX C API */
#include "sherpa-onnx/c-api/c-api.h"

/* ================= WAV Reading ================= */

int audio_read_wav(const char *filepath, AudioBuffer *out) {
    if (filepath == NULL || out == NULL) {
        return -1;
    }

    memset(out, 0, sizeof(AudioBuffer));

    /* Use sherpa-onnx's built-in WAV reader */
    const SherpaOnnxWave *wave = SherpaOnnxReadWave(filepath);
    if (wave == NULL) {
        fprintf(stderr, "[audio] ERROR: Failed to read WAV file: %s\n", filepath);
        return -1;
    }

    fprintf(stderr, "[audio] Read WAV: %s\n", filepath);
    fprintf(stderr, "[audio]   sample_rate: %d, samples: %d\n",
            wave->sample_rate, wave->num_samples);

    int32_t num_samples = wave->num_samples;
    const float *samples = wave->samples;

    /* Resample if not 16 kHz */
    if (wave->sample_rate != SAMPLE_RATE && wave->sample_rate > 0) {
        float *resampled = NULL;
        int32_t resampled_len = 0;

        int ret = audio_resample(
            samples, num_samples,
            wave->sample_rate, SAMPLE_RATE,
            &resampled, &resampled_len
        );

        if (ret == 0 && resampled != NULL) {
            out->samples = resampled;
            out->num_samples = resampled_len;
        } else {
            /* Resampling failed — use original */
            out->samples = (float*)malloc(num_samples * sizeof(float));
            if (out->samples) {
                memcpy(out->samples, samples, num_samples * sizeof(float));
            }
            out->num_samples = num_samples;
        }
    } else {
        /* Sample rate already 16 kHz — copy directly */
        out->samples = (float*)malloc(num_samples * sizeof(float));
        if (out->samples) {
            memcpy(out->samples, samples, num_samples * sizeof(float));
        }
        out->num_samples = num_samples;
    }

    out->sample_rate = SAMPLE_RATE;
    out->channels = 1;  /* SherpaOnnxReadWave always returns mono */

    SherpaOnnxFreeWave(wave);

    if (out->samples == NULL) {
        fprintf(stderr, "[audio] ERROR: malloc failed.\n");
        return -1;
    }

    fprintf(stderr, "[audio]   output: %d samples @ %d Hz\n",
            out->num_samples, out->sample_rate);
    return 0;
}

void audio_free(AudioBuffer *buf) {
    if (buf == NULL) return;
    free(buf->samples);
    buf->samples = NULL;
    buf->num_samples = 0;
}

/* ================= Resampling ================= */

int audio_resample(
    const float *input, int32_t input_len,
    int32_t orig_sr, int32_t target_sr,
    float **output, int32_t *output_len)
{
    if (input == NULL || input_len <= 0 || output == NULL || output_len == NULL) {
        return -1;
    }

    if (orig_sr == target_sr) {
        /* No resampling needed */
        *output = (float*)malloc(input_len * sizeof(float));
        if (*output == NULL) return -1;
        memcpy(*output, input, input_len * sizeof(float));
        *output_len = input_len;
        return 0;
    }

    /* Linear interpolation resampling */
    double ratio = (double)target_sr / orig_sr;
    int32_t new_len = (int32_t)(input_len * ratio);

    *output = (float*)malloc(new_len * sizeof(float));
    if (*output == NULL) return -1;

    for (int32_t i = 0; i < new_len; i++) {
        double src_pos = i / ratio;
        int32_t src_idx = (int32_t)src_pos;
        double frac = src_pos - src_idx;

        if (src_idx + 1 < input_len) {
            /* Linear interpolation between input[src_idx] and input[src_idx+1] */
            (*output)[i] = (float)(input[src_idx] * (1.0 - frac)
                                 + input[src_idx + 1] * frac);
        } else if (src_idx < input_len) {
            (*output)[i] = input[src_idx];
        } else {
            (*output)[i] = 0.0f;
        }
    }

    *output_len = new_len;
    return 0;
}

/* ================= Stereo to Mono ================= */

void audio_stereo_to_mono(
    const float *input, int32_t num_frames,
    int32_t num_channels, float *output)
{
    if (input == NULL || output == NULL || num_channels <= 1) {
        if (input != NULL && output != NULL) {
            memcpy(output, input, num_frames * sizeof(float));
        }
        return;
    }

    for (int32_t i = 0; i < num_frames; i++) {
        float sum = 0.0f;
        for (int32_t ch = 0; ch < num_channels; ch++) {
            sum += input[i * num_channels + ch];
        }
        output[i] = sum / num_channels;
    }
}
