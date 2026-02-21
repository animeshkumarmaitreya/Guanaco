/*============================================================================
 * Memory & KV Cache Test Suite — Person C
 *
 * Run:  make test_memory
 *
 * Covers:
 *   Arena   — alignment, overflow, reset, no-overlap, stress
 *   Scratch — reuse, multi-pass simulation, alignment
 *   KVCache — write/read, contiguity, multi-layer isolation, capacity
 *============================================================================*/

#include "memory.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Test framework ---- */

static int failures = 0;
static int passes   = 0;

#define PASS(name) do { printf("  PASS: %s\n", name); passes++; } while(0)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

/* ========================================================================
 *  ARENA TESTS
 * ======================================================================== */

/* A1: Basic create, alloc, destroy. */
static void test_arena_basic(void) {
    Arena* a = arena_create(1024 * 1024);  /* 1 MiB */
    if (!a) { FAIL("arena_basic", "create returned NULL"); return; }

    void* p1 = arena_alloc(a, 100, 64);
    void* p2 = arena_alloc(a, 200, 64);

    if (!p1 || !p2)   { FAIL("arena_basic", "alloc returned NULL"); arena_destroy(a); return; }
    if (p1 == p2)     { FAIL("arena_basic", "same pointer for two allocs"); arena_destroy(a); return; }
    if ((uintptr_t)p1 % 64 != 0 || (uintptr_t)p2 % 64 != 0) {
        FAIL("arena_basic", "alignment violation");
    } else {
        PASS("arena_basic");
    }
    arena_destroy(a);
}

/* A2: 10 000 random-size allocs, every pointer 64-byte aligned. */
static void test_arena_alignment_stress(void) {
    Arena* a = arena_create(64 * 1024 * 1024);  /* 64 MiB */
    if (!a) { FAIL("arena_alignment_stress", "create failed"); return; }

    srand(42);
    for (int i = 0; i < 10000; i++) {
        size_t sz = (rand() % 1000) + 1;       /* 1-1000 bytes */
        void* p = arena_alloc(a, sz, 64);
        if (!p) { FAIL("arena_alignment_stress", "alloc returned NULL too early"); arena_destroy(a); return; }
        if ((uintptr_t)p % 64 != 0) {
            FAIL("arena_alignment_stress", "misaligned pointer");
            arena_destroy(a);
            return;
        }
    }
    PASS("arena_alignment_stress");
    arena_destroy(a);
}

/* A3: Overflow returns NULL, no crash. */
static void test_arena_overflow(void) {
    Arena* a = arena_create(256);  /* tiny */
    if (!a) { FAIL("arena_overflow", "create failed"); return; }

    void* p1 = arena_alloc(a, 200, 64);
    void* p2 = arena_alloc(a, 200, 64);  /* should fail */

    if (p1 && !p2) {
        PASS("arena_overflow");
    } else {
        FAIL("arena_overflow", "overflow not handled correctly");
    }
    arena_destroy(a);
}

/* A4: Reset enables full reuse. */
static void test_arena_reset(void) {
    Arena* a = arena_create(1024);
    if (!a) { FAIL("arena_reset", "create failed"); return; }

    void* p1 = arena_alloc(a, 900, 64);
    if (!p1) { FAIL("arena_reset", "first alloc failed"); arena_destroy(a); return; }

    arena_reset(a);

    void* p2 = arena_alloc(a, 900, 64);
    if (p2) {
        PASS("arena_reset");
    } else {
        FAIL("arena_reset", "alloc after reset failed");
    }
    arena_destroy(a);
}

/* A5: No overlap between adjacent allocations. */
static void test_arena_no_overlap(void) {
    Arena* a = arena_create(1024 * 1024);
    if (!a) { FAIL("arena_no_overlap", "create failed"); return; }

    void* prev = NULL;
    size_t prev_sz = 0;

    size_t sizes[] = {1, 13, 64, 97, 255, 512, 1025, 4096, 7, 33};
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int i = 0; i < n; i++) {
        void* p = arena_alloc(a, sizes[i], 64);
        if (!p) { FAIL("arena_no_overlap", "alloc returned NULL"); arena_destroy(a); return; }

        /* Check new pointer doesn't overlap previous allocation. */
        if (prev) {
            uintptr_t prev_end = (uintptr_t)prev + prev_sz;
            if ((uintptr_t)p < prev_end) {
                FAIL("arena_no_overlap", "allocations overlap");
                arena_destroy(a);
                return;
            }
        }
        prev    = p;
        prev_sz = sizes[i];
    }
    PASS("arena_no_overlap");
    arena_destroy(a);
}

/* A6: Write to allocated memory, then verify after reset+realloc. */
static void test_arena_write_after_reset(void) {
    Arena* a = arena_create(4096);
    if (!a) { FAIL("arena_write_after_reset", "create failed"); return; }

    float* buf = (float*)arena_alloc(a, 256 * sizeof(float), 64);
    if (!buf) { FAIL("arena_write_after_reset", "alloc failed"); arena_destroy(a); return; }

    for (int i = 0; i < 256; i++) buf[i] = (float)i;

    arena_reset(a);

    float* buf2 = (float*)arena_alloc(a, 256 * sizeof(float), 64);
    if (!buf2) { FAIL("arena_write_after_reset", "alloc after reset failed"); arena_destroy(a); return; }

    /* After reset + new alloc, we should be able to write without issues. */
    for (int i = 0; i < 256; i++) buf2[i] = (float)(i * 2);
    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (buf2[i] != (float)(i * 2)) { ok = 0; break; }
    }

    if (ok) PASS("arena_write_after_reset");
    else    FAIL("arena_write_after_reset", "data corruption");
    arena_destroy(a);
}

/* A7: NULL / zero-size edge cases. */
static void test_arena_edge_cases(void) {
    Arena* a = arena_create(4096);
    if (!a) { FAIL("arena_edge_cases", "create failed"); return; }

    void* p1 = arena_alloc(a, 0, 64);     /* zero size */
    void* p2 = arena_alloc(NULL, 100, 64); /* NULL arena */

    if (!p1 && !p2) {
        PASS("arena_edge_cases");
    } else {
        FAIL("arena_edge_cases", "should return NULL for invalid args");
    }
    arena_destroy(a);

    /* Creating with size 0 should also return NULL. */
    Arena* z = arena_create(0);
    if (!z) {
        PASS("arena_create_zero");
    } else {
        FAIL("arena_create_zero", "should return NULL for size 0");
        arena_destroy(z);
    }
}

/* ========================================================================
 *  SCRATCH TESTS
 * ======================================================================== */

/* S1: First alloc returns base; after reset, same pointer again. */
static void test_scratch_reuse(void) {
    Scratch* s = scratch_create(4096);
    if (!s) { FAIL("scratch_reuse", "create failed"); return; }

    void* p1 = scratch_alloc(s, 100, 64);
    scratch_reset(s);
    void* p2 = scratch_alloc(s, 100, 64);

    if (p1 == p2) PASS("scratch_reuse");
    else          FAIL("scratch_reuse", "pointer not reused after reset");
    scratch_destroy(s);
}

/* S2: Simulate 100 forward passes — alloc, write, reset. */
static void test_scratch_multi_pass(void) {
    Scratch* s = scratch_create(1024 * 1024);
    if (!s) { FAIL("scratch_multi_pass", "create failed"); return; }

    for (int pass = 0; pass < 100; pass++) {
        scratch_reset(s);
        float* buf1 = (float*)scratch_alloc(s, 8192, 64);
        float* buf2 = (float*)scratch_alloc(s, 22016, 64);
        if (!buf1 || !buf2) {
            FAIL("scratch_multi_pass", "alloc failed during passes");
            scratch_destroy(s);
            return;
        }
        /* Write pattern to detect stale-data bugs. */
        memset(buf1, pass & 0xFF, 8192);
        memset(buf2, pass & 0xFF, 22016);
    }
    PASS("scratch_multi_pass");
    scratch_destroy(s);
}

/* S3: Alignment stress (same idea as arena). */
static void test_scratch_alignment(void) {
    Scratch* s = scratch_create(10 * 1024 * 1024);
    if (!s) { FAIL("scratch_alignment", "create failed"); return; }

    srand(123);
    for (int i = 0; i < 5000; i++) {
        size_t sz = (rand() % 2000) + 1;
        void* p = scratch_alloc(s, sz, 64);
        if (!p) { FAIL("scratch_alignment", "alloc returned NULL"); scratch_destroy(s); return; }
        if ((uintptr_t)p % 64 != 0) {
            FAIL("scratch_alignment", "misaligned pointer");
            scratch_destroy(s);
            return;
        }
    }
    PASS("scratch_alignment");
    scratch_destroy(s);
}

/* S4: Overflow returns NULL. */
static void test_scratch_overflow(void) {
    Scratch* s = scratch_create(256);
    if (!s) { FAIL("scratch_overflow", "create failed"); return; }

    void* p1 = scratch_alloc(s, 200, 64);
    void* p2 = scratch_alloc(s, 200, 64);

    if (p1 && !p2) {
        PASS("scratch_overflow");
    } else {
        FAIL("scratch_overflow", "overflow not handled");
    }
    scratch_destroy(s);
}

/* ========================================================================
 *  KV CACHE TESTS
 * ======================================================================== */

/* Helper to create a small config for KV tests. */
static ModelConfig test_kv_cfg(int n_layers, int n_kv_heads, int head_dim, int max_seq) {
    ModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers    = n_layers;
    cfg.n_kv_heads  = n_kv_heads;
    cfg.head_dim    = head_dim;
    cfg.n_heads     = n_kv_heads;      /* not used by KV cache but fill in */
    cfg.hidden_dim  = n_kv_heads * head_dim;
    cfg.vocab_size  = 100;
    cfg.ff_dim      = cfg.hidden_dim * 2;
    cfg.max_seq_len = max_seq;
    return cfg;
}

/* Build a simple Tensor wrapper around a float buffer. */
static Tensor make_kv_tensor(float* data, int n_kv_heads, int head_dim) {
    Tensor t;
    memset(&t, 0, sizeof(t));
    t.data      = data;
    t.ndim      = 2;
    t.shape[0]  = n_kv_heads;
    t.shape[1]  = head_dim;
    t.stride[0] = head_dim;
    t.stride[1] = 1;
    t.dtype     = DTYPE_F32;
    return t;
}

/* K1: Write a known pattern at pos 0 layer 0, read it back per-head. */
static void test_kv_write_read(void) {
    ModelConfig cfg = test_kv_cfg(2, 4, 64, 128);
    Arena* arena = arena_create(64 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_write_read", "create failed"); arena_destroy(arena); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;  /* 256 */
    float k_data[256], v_data[256];
    for (int i = 0; i < total; i++) {
        k_data[i] = (float)(i + 1);
        v_data[i] = (float)(i + 1000);
    }

    Tensor kt = make_kv_tensor(k_data, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_data, cfg.n_kv_heads, cfg.head_dim);

    kv_cache_append(kv, 0, &kt, &vt, 0);

    /* Read K for layer 0. */
    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    if (!kr || !kr->data) { FAIL("kv_write_read", "get_k returned NULL"); goto cleanup_wr; }

    /* Check head 0: values should be 1..64. */
    float* kp = (float*)kr->data;
    int ok = 1;
    for (int d = 0; d < 64; d++) {
        if (fabsf(kp[d] - (float)(d + 1)) > 1e-6f) { ok = 0; break; }
    }

    /* Check head 1: values should be 65..128. */
    if (ok) {
        float* kp1 = (float*)kr->data + kr->stride[0]; /* jump to head 1 */
        for (int d = 0; d < 64; d++) {
            if (fabsf(kp1[d] - (float)(64 + d + 1)) > 1e-6f) { ok = 0; break; }
        }
    }

    /* Also check V. */
    Tensor* vr = kv_cache_get_v(kv, 0, 1);
    if (!vr || !vr->data) { FAIL("kv_write_read", "get_v returned NULL"); goto cleanup_wr; }
    float* vp = (float*)vr->data;
    for (int d = 0; d < 64 && ok; d++) {
        if (fabsf(vp[d] - (float)(d + 1000)) > 1e-6f) ok = 0;
    }

    if (ok) PASS("kv_write_read");
    else    FAIL("kv_write_read", "data mismatch");

cleanup_wr:
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K2: Contiguity — write 50 tokens, read back, verify sequential layout. */
static void test_kv_contiguity(void) {
    ModelConfig cfg = test_kv_cfg(1, 2, 32, 256);
    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 256);
    if (!kv) { FAIL("kv_contiguity", "create failed"); arena_destroy(arena); return; }

    int n_kv = cfg.n_kv_heads;
    int d    = cfg.head_dim;
    int num_tokens = 50;

    float k_buf[64]; /* 2 heads x 32 dim */
    float v_buf[64];

    for (int t = 0; t < num_tokens; t++) {
        for (int i = 0; i < n_kv * d; i++) {
            k_buf[i] = (float)(t * 100 + i);
            v_buf[i] = (float)(t * 100 + i + 5000);
        }
        Tensor kt = make_kv_tensor(k_buf, n_kv, d);
        Tensor vt = make_kv_tensor(v_buf, n_kv, d);
        kv_cache_append(kv, 0, &kt, &vt, t);
    }

    Tensor* kr = kv_cache_get_k(kv, 0, num_tokens);
    if (!kr || !kr->data) { FAIL("kv_contiguity", "get_k NULL"); goto cleanup_cont; }

    /* For head 0, tokens should be contiguous: kr->data[pos * d + dim]. */
    int ok = 1;
    float* head0 = (float*)kr->data;
    for (int t = 0; t < num_tokens && ok; t++) {
        for (int dd = 0; dd < d && ok; dd++) {
            float expected = (float)(t * 100 + dd);  /* head 0 portion */
            float got      = head0[t * d + dd];
            if (fabsf(got - expected) > 1e-6f) ok = 0;
        }
    }

    /* For head 1, jump by stride[0]. */
    float* head1 = (float*)kr->data + kr->stride[0];
    for (int t = 0; t < num_tokens && ok; t++) {
        for (int dd = 0; dd < d && ok; dd++) {
            float expected = (float)(t * 100 + d + dd);  /* head 1 portion */
            float got      = head1[t * d + dd];
            if (fabsf(got - expected) > 1e-6f) ok = 0;
        }
    }

    if (ok) PASS("kv_contiguity");
    else    FAIL("kv_contiguity", "non-contiguous or wrong data");

cleanup_cont:
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K3: Cross-layer isolation — writing to layer 1 doesn't corrupt layer 0. */
static void test_kv_cross_layer(void) {
    ModelConfig cfg = test_kv_cfg(2, 2, 64, 64);
    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 64);
    if (!kv) { FAIL("kv_cross_layer", "create failed"); arena_destroy(arena); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float k0[128], k1[128], v_dummy[128];
    for (int i = 0; i < total; i++) {
        k0[i]      = 100.0f;
        k1[i]      = 200.0f;
        v_dummy[i] = 0.0f;
    }

    Tensor kt0 = make_kv_tensor(k0, cfg.n_kv_heads, cfg.head_dim);
    Tensor kt1 = make_kv_tensor(k1, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt  = make_kv_tensor(v_dummy, cfg.n_kv_heads, cfg.head_dim);

    kv_cache_append(kv, 0, &kt0, &vt, 0);
    kv_cache_append(kv, 1, &kt1, &vt, 0);

    /* Layer 0 should still read 100.0. */
    Tensor* r0 = kv_cache_get_k(kv, 0, 1);
    float val0 = ((float*)r0->data)[0];

    /* Layer 1 should read 200.0. */
    Tensor* r1 = kv_cache_get_k(kv, 1, 1);
    float val1 = ((float*)r1->data)[0];

    if (fabsf(val0 - 100.0f) < 1e-6f && fabsf(val1 - 200.0f) < 1e-6f) {
        PASS("kv_cross_layer");
    } else {
        FAIL("kv_cross_layer", "layer data corrupted");
    }

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K4: Fill to max_seq_len, verify last position, next should fail. */
static void test_kv_max_capacity(void) {
    int max_seq = 64;
    ModelConfig cfg = test_kv_cfg(1, 2, 16, max_seq);
    Arena* arena = arena_create(4 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, max_seq);
    if (!kv) { FAIL("kv_max_capacity", "create failed"); arena_destroy(arena); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float k_buf[32], v_buf[32];
    for (int i = 0; i < total; i++) { k_buf[i] = 1.0f; v_buf[i] = 2.0f; }

    Tensor kt = make_kv_tensor(k_buf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_buf, cfg.n_kv_heads, cfg.head_dim);

    /* Fill all positions 0..max_seq-1. */
    for (int t = 0; t < max_seq; t++) {
        int rc = kv_cache_append(kv, 0, &kt, &vt, t);
        if (rc != 0) {
            FAIL("kv_max_capacity", "append failed before max");
            goto cleanup_cap;
        }
    }

    /* pos == max_seq should fail. */
    int rc = kv_cache_append(kv, 0, &kt, &vt, max_seq);
    if (rc != 0) {
        PASS("kv_max_capacity");
    } else {
        FAIL("kv_max_capacity", "append beyond max did not fail");
    }

cleanup_cap:
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K5: Invalid arguments return errors. */
static void test_kv_invalid_args(void) {
    ModelConfig cfg = test_kv_cfg(2, 2, 32, 64);
    Arena* arena = arena_create(4 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 64);
    if (!kv) { FAIL("kv_invalid_args", "create failed"); arena_destroy(arena); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float k_buf[64], v_buf[64];
    for (int i = 0; i < total; i++) { k_buf[i] = 1.0f; v_buf[i] = 1.0f; }
    Tensor kt = make_kv_tensor(k_buf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_buf, cfg.n_kv_heads, cfg.head_dim);

    int ok = 1;

    /* Invalid layer. */
    if (kv_cache_append(kv, -1, &kt, &vt, 0) == 0)  ok = 0;
    if (kv_cache_append(kv, cfg.n_layers, &kt, &vt, 0) == 0) ok = 0;

    /* Negative pos. */
    if (kv_cache_append(kv, 0, &kt, &vt, -1) == 0)  ok = 0;

    /* NULL tensors. */
    if (kv_cache_append(kv, 0, NULL, &vt, 0) == 0)   ok = 0;
    if (kv_cache_append(kv, 0, &kt, NULL, 0) == 0)   ok = 0;

    /* Invalid get_k. */
    if (kv_cache_get_k(kv, -1, 1)  != NULL) ok = 0;
    if (kv_cache_get_k(kv, 0, 0)   != NULL) ok = 0;
    if (kv_cache_get_k(kv, 0, -1)  != NULL) ok = 0;

    if (ok) PASS("kv_invalid_args");
    else    FAIL("kv_invalid_args", "bad arg not rejected");

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K6: Realistic TinyLlama-like sizing to verify memory calculation.
 * TinyLlama: n_layers=22, n_kv_heads=4, head_dim=64, max_seq=2048
 * Expected KV bytes: 2 x 22 x 4 x 2048 x 64 x 4 = 92,274,688 (~88 MiB) */
static void test_kv_tinyllama_sizing(void) {
    ModelConfig cfg = test_kv_cfg(22, 4, 64, 2048);
    /* Need enough arena: struct + 2 x KV blocks ~ 2 x ~44 MiB = ~88 MiB + overhead */
    Arena* arena = arena_create(128 * 1024 * 1024);  /* 128 MiB */
    KVCache* kv  = kv_cache_create(arena, &cfg, 2048);
    if (!kv) { FAIL("kv_tinyllama_sizing", "create failed — arena too small?"); arena_destroy(arena); return; }

    /* Write to first and last layer, first and last pos. */
    int total = cfg.n_kv_heads * cfg.head_dim;  /* 256 */
    float* k_buf = (float*)malloc(total * sizeof(float));
    float* v_buf = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) { k_buf[i] = 42.0f; v_buf[i] = 84.0f; }

    Tensor kt = make_kv_tensor(k_buf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_buf, cfg.n_kv_heads, cfg.head_dim);

    int rc1 = kv_cache_append(kv, 0,  &kt, &vt, 0);
    int rc2 = kv_cache_append(kv, 21, &kt, &vt, 2047);

    Tensor* r = kv_cache_get_k(kv, 21, 2048);

    int ok = (rc1 == 0 && rc2 == 0 && r != NULL);

    /* Verify last layer, head 0, last pos. */
    if (ok) {
        float* ptr = (float*)r->data;
        /* Offset to pos 2047: head 0, token 2047 */
        float val = ptr[2047 * cfg.head_dim];
        if (fabsf(val - 42.0f) > 1e-6f) ok = 0;
    }

    if (ok) PASS("kv_tinyllama_sizing");
    else    FAIL("kv_tinyllama_sizing", "data mismatch at max config");

    free(k_buf);
    free(v_buf);
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K7: Verify Tensor view shapes and strides are correct. */
static void test_kv_view_metadata(void) {
    ModelConfig cfg = test_kv_cfg(2, 4, 64, 128);
    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_view_metadata", "create failed"); arena_destroy(arena); return; }

    /* Append a token so we have pos 0. */
    float k_buf[256], v_buf[256];
    for (int i = 0; i < 256; i++) { k_buf[i] = 1.0f; v_buf[i] = 1.0f; }
    Tensor kt = make_kv_tensor(k_buf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_buf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    if (!kr) { FAIL("kv_view_metadata", "get_k NULL"); goto cleanup_vm; }

    int ok = 1;
    if (kr->ndim != 3)                     ok = 0;
    if (kr->shape[0] != cfg.n_kv_heads)    ok = 0;
    if (kr->shape[1] != 1)                 ok = 0;  /* up_to_pos = 1 */
    if (kr->shape[2] != cfg.head_dim)      ok = 0;
    if (kr->stride[0] != 128 * 64)         ok = 0;  /* max_seq * head_dim */
    if (kr->stride[1] != 64)               ok = 0;  /* head_dim */
    if (kr->stride[2] != 1)                ok = 0;
    if (kr->dtype != DTYPE_F32)            ok = 0;

    if (ok) PASS("kv_view_metadata");
    else    FAIL("kv_view_metadata", "wrong shape/stride/dtype");

cleanup_vm:
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* K8: GQA scenario — n_heads > n_kv_heads. Verify KV cache only stores n_kv_heads. */
static void test_kv_gqa(void) {
    /* Simulate GQA: 8 query heads, 2 KV heads, head_dim=32. */
    ModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_layers    = 1;
    cfg.n_heads     = 8;
    cfg.n_kv_heads  = 2;
    cfg.head_dim    = 32;
    cfg.hidden_dim  = 256;
    cfg.vocab_size  = 100;
    cfg.ff_dim      = 512;
    cfg.max_seq_len = 64;

    Arena* arena = arena_create(4 * 1024 * 1024);
    KVCache* kv  = kv_cache_create(arena, &cfg, 64);
    if (!kv) { FAIL("kv_gqa", "create failed"); arena_destroy(arena); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;  /* 2 x 32 = 64 */
    float k_buf[64], v_buf[64];
    for (int i = 0; i < total; i++) { k_buf[i] = (float)(i + 1); v_buf[i] = (float)(i + 100); }

    Tensor kt = make_kv_tensor(k_buf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_buf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    if (!kr) { FAIL("kv_gqa", "get_k NULL"); goto cleanup_gqa; }

    /* Should have shape (2, 1, 32) — only 2 KV heads, not 8. */
    int ok = 1;
    if (kr->shape[0] != 2)  ok = 0;
    if (kr->shape[1] != 1)  ok = 0;
    if (kr->shape[2] != 32) ok = 0;

    /* Verify head 0 data: 1..32. */
    float* h0 = (float*)kr->data;
    for (int d = 0; d < 32 && ok; d++) {
        if (fabsf(h0[d] - (float)(d + 1)) > 1e-6f) ok = 0;
    }

    /* Verify head 1 data: 33..64. */
    float* h1 = (float*)kr->data + kr->stride[0];
    for (int d = 0; d < 32 && ok; d++) {
        if (fabsf(h1[d] - (float)(32 + d + 1)) > 1e-6f) ok = 0;
    }

    if (ok) PASS("kv_gqa");
    else    FAIL("kv_gqa", "wrong data for GQA KV cache");

cleanup_gqa:
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* ========================================================================
 *  MAIN
 * ======================================================================== */

int main(void) {
    printf("=== Memory & KV Cache Test Suite (Person C) ===\n\n");

    printf("-- Arena Allocator --\n");
    test_arena_basic();
    test_arena_alignment_stress();
    test_arena_overflow();
    test_arena_reset();
    test_arena_no_overlap();
    test_arena_write_after_reset();
    test_arena_edge_cases();

    printf("\n-- Scratch Allocator --\n");
    test_scratch_reuse();
    test_scratch_multi_pass();
    test_scratch_alignment();
    test_scratch_overflow();

    printf("\n-- KV Cache --\n");
    test_kv_write_read();
    test_kv_contiguity();
    test_kv_cross_layer();
    test_kv_max_capacity();
    test_kv_invalid_args();
    test_kv_tinyllama_sizing();
    test_kv_view_metadata();
    test_kv_gqa();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }
    return failures;
}
