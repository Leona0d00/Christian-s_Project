/**
 * audio.h — Audio I/O module.
 *
 * WAV file reading via SherpaOnnxReadWave, with optional resampling.
 * Mirrors script/Transcribe/_audio.py.
 */

#ifndef AUDIO_H
#define AUDIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================= Audio Buffer ================= */

typedef struct {
    float   *samples;       /* float32 array, heap-allocated */
    int32_t  num_samples;
    int32_t  sample_rate;
    int      channels;
} AudioBuffer;

/* ================= Functions ================= */

/**
 * Read a WAV file and convert to 16kHz mono float32.
 *
 * Uses SherpaOnnxReadWave internally. Automatically handles:
 *   - Resampling (any sample rate → 16 kHz)
 *   - Multi-channel → mono (average)
 *
 * @param filepath   Path to WAV file.
 * @param out        Output buffer (caller must call audio_free() after use).
 * @return 0 on success, non-zero on failure.
 */
int audio_read_wav(const char *filepath, AudioBuffer *out);

/**
 * Free an AudioBuffer allocated by audio_read_wav().
 */
void audio_free(AudioBuffer *buf);

/**
 * Resample audio from orig_sr to target_sr using linear interpolation.
 *
 * @param input       Input float32 array.
 * @param input_len   Number of input samples.
 * @param orig_sr     Original sample rate.
 * @param target_sr   Target sample rate.
 * @param output      Output float32 array (heap-allocated, caller frees).
 * @param output_len  Number of output samples.
 * @return 0 on success.
 */
int audio_resample(
    const float *input, int32_t input_len,
    int32_t orig_sr, int32_t target_sr,
    float **output, int32_t *output_len
);

/**
 * Convert stereo to mono by averaging channels.
 *
 * @param input         Interleaved multi-channel float32 array.
 * @param num_frames    Number of frames (per channel).
 * @param num_channels  Number of channels.
 * @param output        Mono float32 array (must be pre-allocated,
 *                      size = num_frames).
 */
void audio_stereo_to_mono(
    const float *input, int32_t num_frames,
    int32_t num_channels, float *output
);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_H */
