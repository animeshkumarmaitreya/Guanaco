/*============================================================================
 * Session Persistence — Save / Load KV Cache State to Disk
 *
 * Binary format (.lctx):
 *   [Header: 64 bytes]
 *     magic: "LCTX" (4 bytes), version: uint32(1),
 *     n_layers, n_kv_heads, head_dim, max_seq, current_pos, n_tokens,
 *     reserved (32 bytes)
 *   [Token history: n_tokens * sizeof(int)]
 *   [K cache: n_layers * n_kv_heads * max_seq * head_dim * sizeof(float)]
 *   [V cache: same size]
 *
 * ALL disk I/O happens OUTSIDE the decode loop. Zero hot‐path cost.
 *============================================================================*/

#include "memory.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Accessors needed from kv_cache.c (implemented there) ---- */
extern float* kv_cache_raw_k(KVCache* kv, int layer);
extern float* kv_cache_raw_v(KVCache* kv, int layer);
extern int    kv_cache_head_stride(KVCache* kv);

/* ---- Internal KVCache layout (must match kv_cache.c) ---- */
/* We access these through the raw pointer accessors above, so we only need
 * the config info which we pass explicitly. */

#define LCTX_MAGIC  0x5843544C   /* "LCTX" in little-endian */
#define LCTX_VERSION 1

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    int32_t  n_layers;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  max_seq;
    int32_t  current_pos;
    int32_t  n_tokens;
    uint8_t  reserved[32];
} LctxHeader;

_Static_assert(sizeof(LctxHeader) == 64, "LctxHeader must be exactly 64 bytes");

/* ---------- Save ---------- */

int kv_cache_save(KVCache* kv, const ModelConfig* cfg, int max_seq,
                  int current_pos, const int* tokens, int n_tokens,
                  const char* path) {
    if (!kv || !cfg || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[session] Failed to open '%s' for writing\n", path);
        return -1;
    }

    /* Write header */
    LctxHeader hdr = {0};
    hdr.magic       = LCTX_MAGIC;
    hdr.version     = LCTX_VERSION;
    hdr.n_layers    = cfg->n_layers;
    hdr.n_kv_heads  = cfg->n_kv_heads;
    hdr.head_dim    = cfg->head_dim;
    hdr.max_seq     = max_seq;
    hdr.current_pos = current_pos;
    hdr.n_tokens    = n_tokens;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    /* Write token history */
    if (n_tokens > 0 && tokens) {
        if (fwrite(tokens, sizeof(int), n_tokens, f) != (size_t)n_tokens) goto fail;
    }

    /* Write K and V cache — contiguous per-layer blocks */
    size_t layer_floats = (size_t)cfg->n_kv_heads * max_seq * cfg->head_dim;
    size_t layer_bytes  = layer_floats * sizeof(float);

    for (int l = 0; l < cfg->n_layers; l++) {
        float* k_ptr = kv_cache_raw_k(kv, l);
        if (fwrite(k_ptr, 1, layer_bytes, f) != layer_bytes) goto fail;
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        float* v_ptr = kv_cache_raw_v(kv, l);
        if (fwrite(v_ptr, 1, layer_bytes, f) != layer_bytes) goto fail;
    }

    fclose(f);
    fprintf(stderr, "[session] Saved state to '%s' (pos=%d, %d tokens)\n",
            path, current_pos, n_tokens);
    return 0;

fail:
    fprintf(stderr, "[session] Write error on '%s'\n", path);
    fclose(f);
    return -1;
}

/* ---------- Load ---------- */

int kv_cache_load(KVCache* kv, const ModelConfig* cfg, int max_seq,
                  int* out_pos, int** out_tokens, int* out_n_tokens,
                  const char* path) {
    if (!kv || !cfg || !path) return -1;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;  /* File doesn't exist — not an error, caller skips */

    /* Read and validate header */
    LctxHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    if (hdr.magic != LCTX_MAGIC) {
        fprintf(stderr, "[session] Invalid magic in '%s'\n", path);
        goto fail;
    }
    if (hdr.version != LCTX_VERSION) {
        fprintf(stderr, "[session] Unsupported version %u in '%s'\n", hdr.version, path);
        goto fail;
    }

    /* Validate compatibility with current model */
    if (hdr.n_layers   != cfg->n_layers   ||
        hdr.n_kv_heads != cfg->n_kv_heads ||
        hdr.head_dim   != cfg->head_dim) {
        fprintf(stderr, "[session] Model mismatch: file has layers=%d kv=%d dim=%d, "
                "model has layers=%d kv=%d dim=%d\n",
                hdr.n_layers, hdr.n_kv_heads, hdr.head_dim,
                cfg->n_layers, cfg->n_kv_heads, cfg->head_dim);
        goto fail;
    }
    if (hdr.max_seq != max_seq) {
        fprintf(stderr, "[session] Seq length mismatch: file=%d, current=%d\n",
                hdr.max_seq, max_seq);
        goto fail;
    }

    /* Read token history */
    int* tokens = NULL;
    if (hdr.n_tokens > 0) {
        tokens = (int*)malloc(hdr.n_tokens * sizeof(int));
        if (!tokens) goto fail;
        if (fread(tokens, sizeof(int), hdr.n_tokens, f) != (size_t)hdr.n_tokens) {
            free(tokens);
            goto fail;
        }
    }

    /* Read K and V cache directly into arena memory */
    size_t layer_floats = (size_t)cfg->n_kv_heads * max_seq * cfg->head_dim;
    size_t layer_bytes  = layer_floats * sizeof(float);

    for (int l = 0; l < cfg->n_layers; l++) {
        float* k_ptr = kv_cache_raw_k(kv, l);
        if (fread(k_ptr, 1, layer_bytes, f) != layer_bytes) {
            free(tokens);
            goto fail;
        }
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        float* v_ptr = kv_cache_raw_v(kv, l);
        if (fread(v_ptr, 1, layer_bytes, f) != layer_bytes) {
            free(tokens);
            goto fail;
        }
    }

    fclose(f);

    /* Return results */
    if (out_pos)      *out_pos      = hdr.current_pos;
    if (out_tokens)   *out_tokens   = tokens;
    if (out_n_tokens) *out_n_tokens = hdr.n_tokens;

    fprintf(stderr, "[session] Restored state from '%s' (pos=%d, %d tokens) — skipping prefill\n",
            path, hdr.current_pos, hdr.n_tokens);
    return 0;

fail:
    fclose(f);
    return -1;
}

/* ---------- Prompt Cache (Frozen Prefix) ---------- */

int prompt_cache_save(KVCache* kv, const ModelConfig* cfg, int max_seq,
                      int prefix_len, const char* path) {
    if (!kv || !cfg || !path || prefix_len <= 0) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[pcache] Failed to open '%s' for writing\n", path);
        return -1;
    }

    /* Reuse LctxHeader: current_pos = prefix_len, n_tokens = 0 */
    LctxHeader hdr = {0};
    hdr.magic       = LCTX_MAGIC;
    hdr.version     = LCTX_VERSION;
    hdr.n_layers    = cfg->n_layers;
    hdr.n_kv_heads  = cfg->n_kv_heads;
    hdr.head_dim    = cfg->head_dim;
    hdr.max_seq     = max_seq;
    hdr.current_pos = prefix_len;
    hdr.n_tokens    = 0;  /* No token history for prompt caches */

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    /* Write only the prefix portion of each layer's KV data.
     * We write the FULL layer stride to keep layout simple — the prefix_len
     * tells the loader how many positions are valid. */
    size_t layer_floats = (size_t)cfg->n_kv_heads * max_seq * cfg->head_dim;
    size_t layer_bytes  = layer_floats * sizeof(float);

    for (int l = 0; l < cfg->n_layers; l++) {
        float* k_ptr = kv_cache_raw_k(kv, l);
        if (fwrite(k_ptr, 1, layer_bytes, f) != layer_bytes) goto fail;
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        float* v_ptr = kv_cache_raw_v(kv, l);
        if (fwrite(v_ptr, 1, layer_bytes, f) != layer_bytes) goto fail;
    }

    fclose(f);
    fprintf(stderr, "[pcache] Saved prompt cache to '%s' (prefix_len=%d)\n",
            path, prefix_len);
    return 0;

fail:
    fprintf(stderr, "[pcache] Write error on '%s'\n", path);
    fclose(f);
    return -1;
}

int prompt_cache_load(KVCache* kv, const ModelConfig* cfg, int max_seq,
                      int* out_prefix_len, const char* path) {
    /* Reuse kv_cache_load logic — it's the same format */
    int dummy_n_tokens = 0;
    int rc = kv_cache_load(kv, cfg, max_seq, out_prefix_len, NULL, &dummy_n_tokens, path);
    if (rc == 0) {
        fprintf(stderr, "[pcache] Loaded prompt cache from '%s' (prefix_len=%d)\n",
                path, out_prefix_len ? *out_prefix_len : -1);
    }
    return rc;
}
