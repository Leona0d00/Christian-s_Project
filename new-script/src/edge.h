/**
 * edge.h — Edge deployment interfaces.
 *
 * Device detection, configuration presets, and optimization guides.
 * Mirrors script/Transcribe/_edge.py.
 */

#ifndef EDGE_H
#define EDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================= Device Detection ================= */

typedef struct {
    char name[128];
    int  available;     /* 1 = available, 0 = not available */
} ComputeDeviceInfo;

/**
 * List available compute devices.
 * Always returns at least CPU (available=1).
 * CUDA availability depends on sherpa-onnx build.
 *
 * @param devices     Output array (caller allocates).
 * @param max_devices Maximum entries in array.
 * @return Number of devices written.
 */
int edge_list_compute_devices(ComputeDeviceInfo *devices, int max_devices);

/* ================= Configuration Presets ================= */

typedef struct {
    const char *name;
    int         chunk_stride_samples;
    const char *provider;
    int         num_threads;
    const char *description;
} EdgeProfile;

/**
 * Get a deployment profile by name.
 *
 * @param name  Profile name: "cpu_optimized", "low_latency", "cuda_optimized"
 * @return Profile struct (static — do NOT free). Returns cpu_optimized if name
 *         is not recognized.
 */
const EdgeProfile* edge_get_profile(const char *name);

/**
 * List all available edge deployment profiles.
 *
 * @param count  Output: number of profiles.
 * @return Array of profile pointers (static — do NOT free).
 */
const EdgeProfile** edge_list_profiles(int *count);

/* ================= Optimization Guide ================= */

/**
 * Return a static string containing edge deployment optimization tips.
 * Do NOT free the returned string.
 */
const char* edge_get_optimization_guide(void);

/* ================= Select Device ================= */

/**
 * Validate and select a compute device.
 * If the preferred device is not available, falls back to CPU.
 *
 * @param preferred  "cpu", "cuda", or "auto"
 * @return Static string: "cpu" or "cuda"
 */
const char* edge_select_device(const char *preferred);

#ifdef __cplusplus
}
#endif

#endif /* EDGE_H */
