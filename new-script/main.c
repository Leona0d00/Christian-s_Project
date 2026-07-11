/**
 * main.c — CLI entry point for the Chinese real-time speech-to-text system.
 *
 * Usage: transcribe_cli [options] <wav_file>
 *
 * This is the edge-deployment interface. No GUI — just file in, text out.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/core.h"
#include "src/pipeline.h"
#include "src/edge.h"
#include "src/config.h"

/* ---- Usage ---- */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Chinese Real-Time Speech-to-Text (Sherpa-ONNX / C)\n"
        "==================================================\n"
        "\n"
        "Usage: %s [options] <wav_file>\n"
        "\n"
        "Options:\n"
        "  -m <dir>      ASR model directory (default: ./models/asr)\n"
        "  -p <str>      ONNX provider: cpu (default)\n"
        "  -t <n>        Number of threads (default: 4)\n"
        "  --punct <f>   Punctuation model ONNX file\n"
        "  -o <file>     Write final text to file\n"
        "  -v            Print incremental text during processing\n"
        "  -q            Suppress all logs (stderr)\n"
        "  -h            Show this help\n"
        "\n"
        "Examples:\n"
        "  %s test.wav\n"
        "  %s -v -m ./models/asr --punct ./models/punct/model.int8.onnx test.wav\n"
        "  %s -o output.txt test.wav\n"
        "\n",
        prog, prog, prog, prog
    );
}

/* ---- Simple arg parser (no getopt dependency) ---- */

static int parse_args(int argc, char *argv[],
                      const char **model_dir,
                      const char **provider,
                      const char **punct_model,
                      const char **output_file,
                      int *num_threads,
                      int *verbose,
                      int *quiet) {
    int wav_idx = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            *verbose = 1;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            *quiet = 1;
        } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            *model_dir = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--provider") == 0) && i + 1 < argc) {
            *provider = argv[++i];
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            *num_threads = atoi(argv[++i]);
            if (*num_threads < 1) *num_threads = 1;
        } else if (strcmp(argv[i], "--punct") == 0 && i + 1 < argc) {
            *punct_model = argv[++i];
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            *output_file = argv[++i];
        } else if (argv[i][0] != '-') {
            wav_idx = i;  /* positional argument: WAV file */
        } else {
            fprintf(stderr, "ERROR: Unknown option: %s\n\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }

    return wav_idx;
}

/* ---- Main ---- */

int main(int argc, char *argv[]) {
    /* Defaults */
    const char *model_dir   = DEFAULT_MODEL_DIR;
    const char *provider    = DEFAULT_PROVIDER;
    const char *punct_model = NULL;
    const char *output_file = NULL;
    int num_threads         = DEFAULT_NUM_THREADS;
    int verbose             = 0;
    int quiet               = 0;

    /* Parse arguments */
    int wav_idx = parse_args(argc, argv, &model_dir, &provider, &punct_model,
                             &output_file, &num_threads, &verbose, &quiet);

    /* Require a WAV file argument */
    if (wav_idx < 0) {
        fprintf(stderr, "ERROR: No WAV file specified.\n\n");
        print_usage(argv[0]);
        return 1;
    }
    const char *wav_path = argv[wav_idx];

    /* Quiet mode: redirect stderr */
    if (quiet) {
        freopen("NUL", "w", stderr);  /* Windows NUL device */
    }

    /* Print banner */
    fprintf(stderr, "============================================\n");
    fprintf(stderr, " Chinese Real-Time Speech-to-Text\n");
    fprintf(stderr, " Engine: Sherpa-ONNX (C)\n");
    fprintf(stderr, "============================================\n");
    fprintf(stderr, "Model:     %s\n", model_dir);
    fprintf(stderr, "Provider:  %s\n", provider);
    fprintf(stderr, "Threads:   %d\n", num_threads);
    fprintf(stderr, "Chunk:     %d ms\n", CHUNK_STRIDE_MS);
    fprintf(stderr, "WAV:       %s\n", wav_path);
    fprintf(stderr, "Verbose:   %s\n", verbose ? "yes" : "no");
    fprintf(stderr, "--------------------------------------------\n");

    /* Create pipeline */
    TranscriberPipeline *p = pipeline_create(model_dir, num_threads, provider);
    if (p == NULL) {
        fprintf(stderr, "FATAL: Failed to create pipeline.\n");
        fprintf(stderr, "Check that the model directory contains:\n");
        fprintf(stderr, "  - encoder.onnx\n");
        fprintf(stderr, "  - decoder.onnx\n");
        fprintf(stderr, "  - joiner.onnx\n");
        fprintf(stderr, "  - tokens.txt\n");
        return 1;
    }

    /* Load punctuation model if specified */
    if (punct_model) {
        pipeline_init_punctuation(p, punct_model);
    }

    /* Process file */
    fprintf(stderr, "Processing...\n");

    if (verbose) {
        fprintf(stderr, "--------------------------------------------\n");
    }

    char *result = pipeline_process_file(p, wav_path, verbose);

    if (verbose) {
        fprintf(stderr, "\n--------------------------------------------\n");
    }

    fprintf(stderr, "Done.\n\n");

    /* Output */
    if (result && result[0]) {
        if (output_file) {
            FILE *f = fopen(output_file, "w");
            if (f) {
                fprintf(f, "%s\n", result);
                fclose(f);
                fprintf(stderr, "Output written to: %s\n", output_file);
            } else {
                fprintf(stderr, "ERROR: Cannot write to %s\n", output_file);
            }
        }

        /* Always print result to stdout */
        printf("%s\n", result);
        free(result);
    } else {
        fprintf(stderr, "(No text recognized)\n");
    }

    /* Cleanup */
    pipeline_destroy(p);
    recognizer_destroy();
    punct_destroy();

    fprintf(stderr, "\nDone.\n");
    return 0;
}
