/**
 * utils.h — Utility functions: string helpers, memory management.
 *
 * Internal header. Not part of the public API.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Safe string duplication. Returns NULL on allocation failure.
 * The caller must free() the result.
 */
char* util_strdup(const char *src);

/**
 * Safe string concatenation. Frees dst and returns a newly allocated string.
 * If dst is NULL, behaves like util_strdup.
 * The caller must free() the result.
 */
char* util_strcat(char *dst, const char *src);

/**
 * Find the longest overlap between the suffix of `old_str`
 * and the prefix of `new_str`. Returns the overlap length in bytes.
 *
 * This is used by the subtitle corrector for incremental dedup.
 * Works correctly on UTF-8 because we compare complete byte sequences
 * and ASR output is well-formed UTF-8.
 */
int util_find_overlap(const char *old_str, const char *new_str);

/**
 * Strip leading and trailing whitespace in-place.
 * Returns the same pointer (possibly shifted) for convenience.
 * Modifies the string in-place by writing NUL terminators.
 */
char* util_strip(char *str);

/**
 * Check whether a byte is a valid UTF-8 continuation byte.
 * Returns 1 if the byte is 0x80-0xBF (continuation byte).
 */
int util_is_utf8_continuation(unsigned char c);

/**
 * Ensure the cut position does not split a multi-byte UTF-8 character.
 * If `pos` points to a UTF-8 continuation byte, decrement until we
 * find a start byte or reach 0.
 */
int util_utf8_safe_cut_pos(const char *str, int pos);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */
