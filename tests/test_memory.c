/*============================================================================
 * Memory & KV Cache Test Suite — Person B's test harness
 *
 * Run: make test_memory && ./build/test_memory
 *============================================================================*/

#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

/* ---- Arena tests ---- */

static void test_arena_basic(void) {
    Arena* a = arena_create(1024 * 1024);  /* 1 MiB */
    if (!a) { FAIL("arena_basic", "create failed"); return; }

    void* p1 = arena_alloc(a, 100, 64);
    void* p2 = arena_alloc(a, 200, 64);

    if (!p1 || !p2) { FAIL("arena_basic", "alloc returned NULL"); arena_destroy(a); return; }
    if (p1 == p2)   { FAIL("arena_basic", "same pointer"); arena_destroy(a); return; }

    /* Check 64-byte alignment */
    if ((uintptr_t)p1 % 64 != 0 || (uintptr_t)p2 % 64 != 0) {
        FAIL("arena_basic", "alignment violation");
    } else {
        PASS("arena_basic");
    }
    arena_destroy(a);
}

static void test_arena_alignment_stress(void) {
    Arena* a = arena_create(10 * 1024 * 1024);  /* 10 MiB */
    if (!a) { FAIL("arena_alignment_stress", "create failed"); return; }

    int sizes[] = {13, 97, 1, 255, 4097, 7, 33, 511, 1025, 63};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n; i++) {
        void* p = arena_alloc(a, sizes[i], 64);
        if (!p) { FAIL("arena_alignment_stress", "alloc failed"); arena_destroy(a); return; }
        if ((uintptr_t)p % 64 != 0) {
            FAIL("arena_alignment_stress", "misaligned pointer");
            arena_destroy(a);
            return;
        }
    }
    PASS("arena_alignment_stress");
    arena_destroy(a);
}

static void test_arena_overflow(void) {
    Arena* a = arena_create(256);  /* tiny */
    if (!a) { FAIL("arena_overflow", "create failed"); return; }

    void* p1 = arena_alloc(a, 200, 64);
    void* p2 = arena_alloc(a, 200, 64);  /* should fail — only 256 bytes total */

    if (p1 && !p2) {
        PASS("arena_overflow");
    } else {
        FAIL("arena_overflow", "didn't return NULL on overflow");
    }
    arena_destroy(a);
}

static void test_arena_reset(void) {
    Arena* a = arena_create(1024);
    if (!a) { FAIL("arena_reset", "create failed"); return; }

    arena_alloc(a, 900, 64);
    arena_reset(a);
    void* p = arena_alloc(a, 900, 64);

    if (p) PASS("arena_reset");
    else   FAIL("arena_reset", "alloc after reset failed");
    arena_destroy(a);
}

/* ---- Scratch tests ---- */

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

/* ---- KV Cache tests ---- */

static void test_kv_write_read(void) {
    ModelConfig cfg = {
        .hidden_dim = 256, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 512, .max_seq_len = 128
    };

    Arena* arena = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_write_read", "create failed"); arena_destroy(arena); return; }

    /* Write known data at pos 0, layer 0 */
    float k_data[256], v_data[256];  /* n_kv_heads * head_dim = 4*64 = 256 */
    for (int i = 0; i < 256; i++) {
        k_data[i] = (float)(i + 1);
        v_data[i] = (float)(i + 1000);
    }

    Tensor kt = { .data = k_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    Tensor vt = { .data = v_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    kt.stride[0] = 64; kt.stride[1] = 1;
    vt.stride[0] = 64; vt.stride[1] = 1;

    kv_cache_append(kv, 0, &kt, &vt, 0);

    /* Read back */
    Tensor* k_read = kv_cache_get_k(kv, 0, 1);

    if (!k_read || !k_read->data) {
        FAIL("kv_write_read", "get_k returned NULL");
    } else {
        /* Check head 0, pos 0 values */
        float* kp = (float*)k_read->data;
        int ok = 1;
        for (int d = 0; d < 64; d++) {
            float expected = (float)(d + 1);  /* head 0 */
            if (fabsf(kp[d] - expected) > 1e-6f) { ok = 0; break; }
        }
        if (ok) PASS("kv_write_read");
        else    FAIL("kv_write_read", "data mismatch");
    }

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

static void test_kv_cross_layer(void) {
    ModelConfig cfg = {
        .hidden_dim = 128, .n_heads = 2, .n_kv_heads = 2,
        .head_dim = 64, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 256, .max_seq_len = 64
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 64);
    if (!kv) { FAIL("kv_cross_layer", "create failed"); arena_destroy(arena); return; }

    float k0[128], k1[128], v_dummy[128];
    for (int i = 0; i < 128; i++) {
        k0[i] = 100.0f;
        k1[i] = 200.0f;
        v_dummy[i] = 0.0f;
    }

    Tensor kt0 = { .data = k0, .ndim = 2, .shape = {2, 64}, .dtype = DTYPE_F32 };
    Tensor kt1 = { .data = k1, .ndim = 2, .shape = {2, 64}, .dtype = DTYPE_F32 };
    Tensor vt  = { .data = v_dummy, .ndim = 2, .shape = {2, 64}, .dtype = DTYPE_F32 };
    kt0.stride[0] = 64; kt0.stride[1] = 1;
    kt1.stride[0] = 64; kt1.stride[1] = 1;
    vt.stride[0] = 64; vt.stride[1] = 1;

    kv_cache_append(kv, 0, &kt0, &vt, 0);
    kv_cache_append(kv, 1, &kt1, &vt, 0);

    Tensor* read0 = kv_cache_get_k(kv, 0, 1);
    float val0 = ((float*)read0->data)[0];

    if (fabsf(val0 - 100.0f) > 1e-6f) {
        FAIL("kv_cross_layer", "layer 0 corrupted by layer 1 write");
    } else {
        PASS("kv_cross_layer");
    }

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* ---- Main ---- */

int main(void) {
    printf("=== Memory & KV Cache Test Suite ===\n");

    test_arena_basic();
    test_arena_alignment_stress();
    test_arena_overflow();
    test_arena_reset();
    test_scratch_reuse();
    test_kv_write_read();
    test_kv_cross_layer();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
