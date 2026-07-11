/**
 * utils.c — String helpers and utility functions.
 */

#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ---- String helpers ---- */

char* util_strdup(const char *src) {
    if (src == NULL) return NULL;
    size_t len = strlen(src);
    char *dst = (char*)malloc(len + 1);
    if (dst == NULL) return NULL;
    memcpy(dst, src, len + 1);
    return dst;
}

char* util_strcat(char *dst, const char *src) {
    if (src == NULL || src[0] == '\0') return dst;
    if (dst == NULL) return util_strdup(src);

    size_t dst_len = strlen(dst);
    size_t src_len = strlen(src);
    char *new_str = (char*)realloc(dst, dst_len + src_len + 1);
    if (new_str == NULL) {
        free(dst);
        return NULL;
    }
    memcpy(new_str + dst_len, src, src_len + 1);
    return new_str;
}

/* ---- Overlap detection ---- */

int util_find_overlap(const char *old_str, const char *new_str) {
    if (old_str == NULL || new_str == NULL) return 0;

    int old_len = (int)strlen(old_str);
    int new_len = (int)strlen(new_str);
    int max_overlap = old_len < new_len ? old_len : new_len;

    for (int len = max_overlap; len > 0; len--) {
        if (memcmp(old_str + old_len - len, new_str, (size_t)len) == 0) {
            return len;
        }
    }
    return 0;
}

/* ---- String strip ---- */

char* util_strip(char *str) {
    if (str == NULL) return NULL;

    /* Strip leading whitespace */
    char *start = str;
    while (*start && isspace((unsigned char)*start)) {
        start++;
    }

    /* Move to start if needed */
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }

    /* Strip trailing whitespace */
    int len = (int)strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        str[len - 1] = '\0';
        len--;
    }

    return str;
}

/* ---- UTF-8 helpers ---- */

int util_is_utf8_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

int util_utf8_safe_cut_pos(const char *str, int pos) {
    if (str == NULL || pos <= 0) return 0;

    while (pos > 0 && util_is_utf8_continuation((unsigned char)str[pos])) {
        pos--;
    }
    return pos;
}
