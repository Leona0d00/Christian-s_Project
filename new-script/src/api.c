/**
 * api.c — Public C API wrapper implementation.
 *
 * This file implements the functions declared in transcribe_api.h.
 * It maps the public DLL interface to the internal core/pipeline/corrector
 * modules behind an opaque TranscriberHandle.
 */

#include "transcribe_api.h"
#include "core.h"
#include "pipeline.h"
#include "corrector.h"
#include "edge.h"
#include "config.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================= Opaque Handle ================= */

struct TranscriberHandle_s {
    TranscriberPipeline *pipeline;
};

/* ================= Creation / Destruction ================= */

TRANSCRIBE_API TranscriberHandle* transcriber_create(
    const char *model_dir,
    int num_threads,
    const char *provider)
{
    TranscriberHandle *h = (TranscriberHandle*)calloc(1, sizeof(TranscriberHandle));
    if (h == NULL) return NULL;

    h->pipeline = pipeline_create(model_dir, num_threads, provider);
    if (h->pipeline == NULL) {
        free(h);
        return NULL;
    }

    return h;
}

TRANSCRIBE_API void transcriber_destroy(TranscriberHandle *handle) {
    if (handle == NULL) return;

    if (handle->pipeline) {
        pipeline_destroy(handle->pipeline);
    }
    free(handle);
}

/* ================= Core Processing ================= */

TRANSCRIBE_API char* transcriber_process_chunk(
    TranscriberHandle *handle,
    const float *audio,
    int num_samples,
    int is_final)
{
    if (handle == NULL || handle->pipeline == NULL || handle->pipeline->session == NULL) {
        return util_strdup("");
    }

    ProcessResult result = process_chunk(
        audio, num_samples, handle->pipeline->session, is_final);

    char *text = result.text;  /* Already heap-allocated by process_chunk */
    if (text == NULL) {
        text = util_strdup("");
    }

    return text;
}

/* ================= Subtitle Correction ================= */

TRANSCRIBE_API char* transcriber_get_display_text(
    TranscriberHandle *handle,
    const char *raw_text)
{
    if (handle == NULL || handle->pipeline == NULL ||
        handle->pipeline->corrector == NULL) {
        return NULL;
    }

    return corrector_correct_chunk(handle->pipeline->corrector, raw_text);
}

/* ================= Finalization ================= */

TRANSCRIBE_API char* transcriber_finalize(TranscriberHandle *handle) {
    if (handle == NULL || handle->pipeline == NULL) {
        return util_strdup("");
    }

    /* Get full raw text */
    char *raw_text = corrector_get_full_text(handle->pipeline->corrector);

    /* Apply punctuation if available */
    char *final_text;
    if (punct_is_loaded()) {
        final_text = punct_apply(raw_text);
        free(raw_text);
    } else {
        final_text = raw_text;
    }

    /* Post-process */
    util_strip(final_text);

    return final_text;
}

TRANSCRIBE_API void transcriber_reset(TranscriberHandle *handle) {
    if (handle == NULL) return;
    pipeline_reset(handle->pipeline);
}

/* ================= File Processing ================= */

TRANSCRIBE_API char* transcriber_process_file(
    TranscriberHandle *handle,
    const char *wav_path,
    int verbose)
{
    if (handle == NULL || handle->pipeline == NULL) {
        return util_strdup("");
    }

    return pipeline_process_file(handle->pipeline, wav_path, verbose);
}

/* ================= Punctuation ================= */

TRANSCRIBE_API int transcriber_load_punctuation(
    TranscriberHandle *handle,
    const char *punct_model)
{
    if (handle == NULL || handle->pipeline == NULL) {
        return 0;
    }
    return pipeline_init_punctuation(handle->pipeline, punct_model);
}

/* ================= Version / Info ================= */

TRANSCRIBE_API const char* transcriber_version(void) {
    return "1.0.0 (sherpa-onnx c-api)";
}

TRANSCRIBE_API const char* transcriber_get_info(TranscriberHandle *handle) {
    static char info_buf[512];
    if (handle == NULL || handle->pipeline == NULL) {
        snprintf(info_buf, sizeof(info_buf), "No model loaded");
    } else {
        snprintf(info_buf, sizeof(info_buf), "%s",
                 recognizer_get_info());
    }
    return info_buf;
}

/* ================= Edge Deployment ================= */

TRANSCRIBE_API int transcriber_get_device_count(void) {
    ComputeDeviceInfo devices[4];
    return edge_list_compute_devices(devices, 4);
}

TRANSCRIBE_API int transcriber_get_device_info(
    int index,
    char *name,
    int name_size,
    int *available)
{
    ComputeDeviceInfo devices[4];
    int count = edge_list_compute_devices(devices, 4);

    if (index < 0 || index >= count) {
        return -1;
    }

    if (name && name_size > 0) {
        strncpy(name, devices[index].name, name_size - 1);
        name[name_size - 1] = '\0';
    }
    if (available) {
        *available = devices[index].available;
    }

    return 0;
}

TRANSCRIBE_API const char* transcriber_get_optimization_guide(void) {
    return edge_get_optimization_guide();
}

/**
 * Free a string returned by any transcriber_* function.
 * Must use this instead of free() to ensure the correct CRT allocator.
 */
TRANSCRIBE_API void transcriber_free_string(char *str) {
    free(str);
}
