/**
 * edge.c — Edge deployment interfaces implementation.
 */

#include "edge.h"
#include "config.h"

#include <string.h>
#include <stdio.h>

/* ================= Device Detection ================= */

int edge_list_compute_devices(ComputeDeviceInfo *devices, int max_devices) {
    if (devices == NULL || max_devices <= 0) return 0;

    int count = 0;

    /* CPU is always available */
    strncpy(devices[count].name, "CPU (ONNX Runtime)", sizeof(devices[count].name) - 1);
    devices[count].available = 1;
    count++;

    /* CUDA: check if available (simplified — real check would probe ONNX provider) */
    if (count < max_devices) {
        strncpy(devices[count].name, "CUDA (requires CUDA-enabled sherpa-onnx build)",
                sizeof(devices[count].name) - 1);
        devices[count].available = 0;  /* Default pre-built has CPU only */
        count++;
    }

    return count;
}

/* ================= Configuration Presets ================= */

static EdgeProfile g_profiles[] = {
    {
        "cpu_optimized",
        CHUNK_STRIDE_SAMPLES,       /* 9600 = 600ms */
        "cpu",
        4,
        "Default configuration balancing latency and accuracy"
    },
    {
        "low_latency",
        7680,                        /* 480ms */
        "cpu",
        4,
        "Low-latency mode for interactive scenarios"
    },
    {
        "cuda_optimized",
        CHUNK_STRIDE_SAMPLES,       /* 9600 = 600ms */
        "cuda",
        1,
        "GPU-accelerated mode (requires CUDA-enabled sherpa-onnx)"
    },
};

static const EdgeProfile* g_profile_ptrs[EDGE_PROFILE_COUNT] = {0};
static int g_profile_ptrs_init = 0;

static void init_profile_ptrs(void) {
    if (g_profile_ptrs_init) return;
    for (int i = 0; i < EDGE_PROFILE_COUNT; i++) {
        g_profile_ptrs[i] = &g_profiles[i];
    }
    g_profile_ptrs_init = 1;
}

const EdgeProfile* edge_get_profile(const char *name) {
    if (name == NULL) return &g_profiles[0];

    for (int i = 0; i < EDGE_PROFILE_COUNT; i++) {
        if (strcmp(g_profiles[i].name, name) == 0) {
            return &g_profiles[i];
        }
    }

    /* Unknown profile → return default */
    return &g_profiles[0];
}

const EdgeProfile** edge_list_profiles(int *count) {
    init_profile_ptrs();
    if (count) *count = EDGE_PROFILE_COUNT;
    return g_profile_ptrs;
}

/* ================= Optimization Guide ================= */

const char* edge_get_optimization_guide(void) {
    return
        "Edge Deployment Optimization Guide\n"
        "==================================\n"
        "\n"
        "Model Selection:\n"
        "  - zipformer-zh-int8:   ~200MB,  suitable for most edge devices\n"
        "  - zipformer-zh-fp32:   ~500MB,  higher accuracy, needs more RAM\n"
        "  - zipformer-zh-14M:     ~50MB,  ultra-lightweight model\n"
        "\n"
        "CPU Optimization:\n"
        "  - Set num_threads to physical core count\n"
        "  - Use INT8 model for 2x smaller memory footprint\n"
        "  - Enable ONNX Runtime optimizations (session options)\n"
        "\n"
        "Memory:\n"
        "  - Minimum:  512MB RAM (INT8 model)\n"
        "  - Recommended: 2GB RAM (FP32 model)\n"
        "  - Streaming uses ~100MB extra for internal buffers\n"
        "\n"
        "Deployment Checklist:\n"
        "  1. Download INT8 model for target device\n"
        "  2. Set num_threads = CPU core count\n"
        "  3. Use 'cpu' provider (included in all builds)\n"
        "  4. Test with representative audio samples\n"
        "  5. Monitor memory usage during streaming\n";
}

/* ================= Select Device ================= */

const char* edge_select_device(const char *preferred) {
    if (preferred == NULL || strcmp(preferred, "auto") == 0) {
        /* Auto-select: prefer CUDA but fall back to CPU */
        ComputeDeviceInfo devices[2];
        int n = edge_list_compute_devices(devices, 2);
        for (int i = 0; i < n; i++) {
            if (strstr(devices[i].name, "CUDA") && devices[i].available) {
                return "cuda";
            }
        }
        return "cpu";
    }

    if (strcmp(preferred, "cuda") == 0) {
        /* Check if CUDA is available */
        ComputeDeviceInfo devices[2];
        int n = edge_list_compute_devices(devices, 2);
        for (int i = 0; i < n; i++) {
            if (strstr(devices[i].name, "CUDA") && devices[i].available) {
                return "cuda";
            }
        }
        fprintf(stderr, "[edge] CUDA not available, falling back to CPU.\n");
        return "cpu";
    }

    return "cpu";
}
