/*============================================================================
 * Memory & KV Cache Test Suite — Person B's test harness
 *
 * Run: make test_memory && ./build/test_memory
 *
 * Tests:
 *   Arena:   basic, alignment stress (10k allocs), overflow, reset, no-overlap,
 *            null safety
 *   Scratch: basic, reuse after reset, 100 forward-pass cycles, overflow,
 *            alignment stress, null safety
 *   KV:     write/read, contiguity, multi-layer isolation, max capacity,
 *            TinyLlama sizing, tensor view shapes, V cache
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

/* =========================================================================
 * Arena tests
 * ========================================================================= */

static void test_arena_basic(void) {
    Arena* a = arena_create(1024 * 1024);
    if (!a) { FAIL("arena_basic", "create failed"); return; }

    void* p1 = arena_alloc(a, 100, 64);
    void* p2 = arena_alloc(a, 200, 64);

    if (!p1 || !p2) { FAIL("arena_basic", "alloc returned NULL"); arena_destroy(a); return; }
    if (p1 == p2)   { FAIL("arena_basic", "same pointer"); arena_destroy(a); return; }

    if ((uintptr_t)p1 % 64 != 0 || (uintptr_t)p2 % 64 != 0) {
        FAIL("arena_basic", "alignment violation");
    } else {
        PASS("arena_basic");
    }
    arena_destroy(a);
}

static void test_arena_alignment_stress(void) {
    Arena* a = arena_create(10 * 1024 * 1024);
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

static void test_arena_10k_alignment(void) {
    Arena* a = arena_create(64 * 1024 * 1024);
    if (!a) { FAIL("arena_10k_alignment", "create failed"); return; }

    srand(42);
    for (int i = 0; i < 10000; i++) {
        size_t sz = (size_t)(rand() % 1000) + 1;
        void* p = arena_alloc(a, sz, 64);
        if (!p) { FAIL("arena_10k_alignment", "alloc returned NULL early"); arena_destroy(a); return; }
        if ((uintptr_t)p % 64 != 0) {
            FAIL("arena_10k_alignment", "alignment violation");
            arena_destroy(a);
            return;
        }
    }
    PASS("arena_10k_alignment");
    arena_destroy(a);
}

static void test_arena_overflow(void) {
    Arena* a = arena_create(256);
    if (!a) { FAIL("arena_overflow", "create failed"); return; }

    void* p1 = arena_alloc(a, 200, 64);
    void* p2 = arena_alloc(a, 200, 64);

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

static void test_arena_no_overlap(void) {
    Arena* a = arena_create(1024 * 1024);
    if (!a) { FAIL("arena_no_overlap", "create failed"); return; }

    srand(99);
    uintptr_t prev_end = 0;
    for (int i = 0; i < 500; i++) {
        size_t sz = (size_t)(rand() % 500) + 1;
        void* p = arena_alloc(a, sz, 64);
        if (!p) { FAIL("arena_no_overlap", "alloc failed"); arena_destroy(a); return; }
        uintptr_t start = (uintptr_t)p;
        if (prev_end > 0 && start < prev_end) {
            FAIL("arena_no_overlap", "allocations overlap");
            arena_destroy(a);
            return;
        }
        prev_end = start + sz;
    }
    PASS("arena_no_overlap");
    arena_destroy(a);
}

static void test_arena_null_safety(void) {
    /* These must not crash */
    arena_alloc(NULL, 100, 64);
    arena_reset(NULL);
    arena_destroy(NULL);
    PASS("arena_null_safety");
}

/* =========================================================================
 * Scratch tests
 * ========================================================================= */

static void test_scratch_basic(void) {
    Scratch* s = scratch_create(1024 * 1024);
    if (!s) { FAIL("scratch_basic", "create failed"); return; }

    void* p1 = scratch_alloc(s, 2048 * 4, 64);
    void* p2 = scratch_alloc(s, 5632 * 4, 64);
    if (!p1 || !p2) { FAIL("scratch_basic", "alloc failed"); scratch_destroy(s); return; }
    if ((uintptr_t)p1 % 64 != 0 || (uintptr_t)p2 % 64 != 0) {
        FAIL("scratch_basic", "alignment violation");
    } else {
        PASS("scratch_basic");
    }
    scratch_destroy(s);
}

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

static void test_scratch_100_passes(void) {
    Scratch* s = scratch_create(4 * 1024 * 1024);
    if (!s) { FAIL("scratch_100_passes", "create failed"); return; }

    for (int pass = 0; pass < 100; pass++) {
        scratch_reset(s);
        /* Simulate typical forward-pass allocations */
        float* x    = (float*)scratch_alloc(s, 2048 * sizeof(float), 64);
        float* q    = (float*)scratch_alloc(s, 2048 * sizeof(float), 64);
        float* k    = (float*)scratch_alloc(s, 256 * sizeof(float), 64);
        float* gate = (float*)scratch_alloc(s, 5632 * sizeof(float), 64);
        float* up   = (float*)scratch_alloc(s, 5632 * sizeof(float), 64);
        if (!x || !q || !k || !gate || !up) {
            FAIL("scratch_100_passes", "alloc failed during pass");
            scratch_destroy(s);
            return;
        }
        /* Write patterns (ASAN catches stale-data bugs) */
        memset(x, pass & 0xFF, 2048 * sizeof(float));
        memset(gate, pass & 0xFF, 5632 * sizeof(float));
    }
    PASS("scratch_100_passes");
    scratch_destroy(s);
}

static void test_scratch_overflow(void) {
    Scratch* s = scratch_create(4096);
    if (!s) { FAIL("scratch_overflow", "create failed"); return; }

    void* p1 = scratch_alloc(s, 3000, 64);
    void* p2 = scratch_alloc(s, 3000, 64);

    if (p1 && !p2) PASS("scratch_overflow");
    else           FAIL("scratch_overflow", "didn't return NULL on overflow");
    scratch_destroy(s);
}

static void test_scratch_alignment(void) {
    Scratch* s = scratch_create(8 * 1024 * 1024);
    if (!s) { FAIL("scratch_alignment", "create failed"); return; }

    srand(123);
    for (int i = 0; i < 5000; i++) {
        if (i % 100 == 0) scratch_reset(s);
        size_t sz = (size_t)(rand() % 2000) + 1;
        void* p = scratch_alloc(s, sz, 64);
        if (!p) { FAIL("scratch_alignment", "alloc failed"); scratch_destroy(s); return; }
        if ((uintptr_t)p % 64 != 0) {
            FAIL("scratch_alignment", "alignment violation");
            scratch_destroy(s);
            return;
        }
    }
    PASS("scratch_alignment");
    scratch_destroy(s);
}

static void test_scratch_null_safety(void) {
    scratch_alloc(NULL, 100, 64);
    scratch_reset(NULL);
    scratch_destroy(NULL);
    PASS("scratch_null_safety");
}

/* =========================================================================
 * KV Cache tests
 * ========================================================================= */

static void test_kv_write_read(void) {
    ModelConfig cfg = {
        .hidden_dim = 256, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 512, .max_seq_len = 128
    };

    Arena* arena = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_write_read", "create failed"); arena_destroy(arena); return; }

    float k_data[256], v_data[256];
    for (int i = 0; i < 256; i++) {
        k_data[i] = (float)(i + 1);
        v_data[i] = (float)(i + 1000);
    }

    Tensor kt = { .data = k_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    Tensor vt = { .data = v_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    kt.stride[0] = 64; kt.stride[1] = 1;
    vt.stride[0] = 64; vt.stride[1] = 1;

    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* k_read = kv_cache_get_k(kv, 0, 1);
    if (!k_read || !k_read->data) {
        FAIL("kv_write_read", "get_k returned NULL");
    } else {
        float* kp = (float*)k_read->data;
        int ok = 1;
        for (int d = 0; d < 64; d++) {
            float expected = (float)(d + 1);
            if (fabsf(kp[d] - expected) > 1e-6f) { ok = 0; break; }
        }
        if (ok) PASS("kv_write_read");
        else    FAIL("kv_write_read", "data mismatch");
    }

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

static void test_kv_v_cache(void) {
    ModelConfig cfg = {
        .hidden_dim = 256, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 1, .vocab_size = 100,
        .ff_dim = 512, .max_seq_len = 128
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_v_cache", "create failed"); arena_destroy(arena); return; }

    float k_data[256], v_data[256];
    for (int i = 0; i < 256; i++) {
        k_data[i] = 0.0f;
        v_data[i] = (float)(i + 500);
    }
    Tensor kt = { .data = k_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    Tensor vt = { .data = v_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    kt.stride[0] = 64; kt.stride[1] = 1;
    vt.stride[0] = 64; vt.stride[1] = 1;

    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* v_read = kv_cache_get_v(kv, 0, 1);
    if (!v_read || !v_read->data) {
        FAIL("kv_v_cache", "get_v returned NULL");
    } else {
        float* vp = (float*)v_read->data;
        int ok = 1;
        for (int d = 0; d < 64; d++) {
            if (fabsf(vp[d] - (float)(d + 500)) > 1e-6f) { ok = 0; break; }
        }
        if (ok) PASS("kv_v_cache");
        else    FAIL("kv_v_cache", "V data mismatch");
    }

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

static void test_kv_contiguity(void) {
    ModelConfig cfg = {
        .hidden_dim = 16, .n_heads = 2, .n_kv_heads = 2,
        .head_dim = 8, .n_layers = 1, .vocab_size = 100,
        .ff_dim = 32, .max_seq_len = 64
    };

    Arena* arena = arena_create(16 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 64);
    if (!kv) { FAIL("kv_contiguity", "create failed"); arena_destroy(arena); return; }

    /* Write 10 tokens */
    for (int pos = 0; pos < 10; pos++) {
        float k_data[16], v_data[16];
        for (int i = 0; i < 16; i++) {
            k_data[i] = (float)(pos * 100 + i);
            v_data[i] = 0.0f;
        }
        Tensor kt = { .data = k_data, .ndim = 2, .shape = {2, 8}, .dtype = DTYPE_F32 };
        Tensor vt = { .data = v_data, .ndim = 2, .shape = {2, 8}, .dtype = DTYPE_F32 };
        kt.stride[0] = 8; kt.stride[1] = 1;
        vt.stride[0] = 8; vt.stride[1] = 1;
        kv_cache_append(kv, 0, &kt, &vt, pos);
    }

    /* Verify head 0: k_data[0..7] was written at each pos */
    Tensor* k_read = kv_cache_get_k(kv, 0, 10);
    float* kp = (float*)k_read->data;
    int stride_pos = k_read->stride[1];  /* should be head_dim = 8 */
    int ok = 1;
    for (int pos = 0; pos < 10; pos++) {
        for (int d = 0; d < 8; d++) {
            float expected = (float)(pos * 100 + d);
            float actual   = kp[pos * stride_pos + d];
            if (fabsf(actual - expected) > 1e-6f) { ok = 0; break; }
        }
        if (!ok) break;
    }
    if (ok) PASS("kv_contiguity");
    else    FAIL("kv_contiguity", "tokens not contiguous");

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

static void test_kv_max_capacity(void) {
    ModelConfig cfg = {
        .hidden_dim = 4, .n_heads = 1, .n_kv_heads = 1,
        .head_dim = 4, .n_layers = 1, .vocab_size = 10,
        .ff_dim = 8, .max_seq_len = 16
    };

    Arena* arena = arena_create(1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 16);
    if (!kv) { FAIL("kv_max_capacity", "create failed"); arena_destroy(arena); return; }

    /* Fill to max */
    int ok = 1;
    for (int pos = 0; pos < 16; pos++) {
        float k_data[4], v_data[4];
        for (int i = 0; i < 4; i++) {
            k_data[i] = (float)(pos * 10 + i);
            v_data[i] = (float)(pos * 10 + i);
        }
        Tensor kt = { .data = k_data, .ndim = 2, .shape = {1, 4}, .dtype = DTYPE_F32 };
        Tensor vt = { .data = v_data, .ndim = 2, .shape = {1, 4}, .dtype = DTYPE_F32 };
        kt.stride[0] = 4; kt.stride[1] = 1;
        vt.stride[0] = 4; vt.stride[1] = 1;
        int ret = kv_cache_append(kv, 0, &kt, &vt, pos);
        if (ret != 0) { ok = 0; break; }
    }

    /* Verify all positions */
    if (ok) {
        Tensor* k_read = kv_cache_get_k(kv, 0, 16);
        float* kp = (float*)k_read->data;
        for (int pos = 0; pos < 16 && ok; pos++) {
            for (int d = 0; d < 4; d++) {
                if (fabsf(kp[pos * 4 + d] - (float)(pos * 10 + d)) > 1e-6f) {
                    ok = 0; break;
                }
            }
        }
    }

    if (ok) PASS("kv_max_capacity");
    else    FAIL("kv_max_capacity", "data mismatch at max fill");

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

static void test_kv_tensor_view_shape(void) {
    ModelConfig cfg = {
        .hidden_dim = 256, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 512, .max_seq_len = 128
    };

    Arena* arena = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 128);
    if (!kv) { FAIL("kv_view_shape", "create failed"); arena_destroy(arena); return; }

    /* Write 5 tokens */
    for (int pos = 0; pos < 5; pos++) {
        float k_data[256] = {0}, v_data[256] = {0};
        Tensor kt = { .data = k_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
        Tensor vt = { .data = v_data, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
        kt.stride[0] = 64; kt.stride[1] = 1;
        vt.stride[0] = 64; vt.stride[1] = 1;
        kv_cache_append(kv, 0, &kt, &vt, pos);
    }

    /* Get view for 5 positions */
    Tensor* view = kv_cache_get_k(kv, 0, 5);
    int ok = (view != NULL &&
              view->ndim == 3 &&
              view->shape[0] == 4 &&       /* n_kv_heads */
              view->shape[1] == 5 &&       /* up_to_pos */
              view->shape[2] == 64 &&      /* head_dim */
              view->dtype == DTYPE_F32);

    if (ok) PASS("kv_view_shape");
    else    FAIL("kv_view_shape", "tensor view shape/ndim wrong");

    kv_cache_destroy(kv);
    arena_destroy(arena);
}

static void test_kv_tinyllama_sizing(void) {
    ModelConfig cfg = {
        .hidden_dim = 2048, .n_heads = 32, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 22, .vocab_size = 32000,
        .ff_dim = 5632, .max_seq_len = 2048
    };

    /* Need: 2 × 22 × 4 × 2048 × 64 × 4 = ~88 MiB + overhead */
    size_t arena_size = 100 * 1024 * 1024;  /* 100 MiB */
    Arena* arena = arena_create(arena_size);
    if (!arena) { FAIL("kv_tinyllama", "arena_create failed"); return; }

    KVCache* kv = kv_cache_create(arena, &cfg, 2048);
    if (!kv) { FAIL("kv_tinyllama", "kv_cache_create failed"); arena_destroy(arena); return; }

    /* Write one token at layer 0 and layer 21 */
    int kv_dim = cfg.n_kv_heads * cfg.head_dim;  /* 256 */
    float* k_vec = (float*)calloc((size_t)kv_dim, sizeof(float));
    float* v_vec = (float*)calloc((size_t)kv_dim, sizeof(float));
    for (int i = 0; i < kv_dim; i++) k_vec[i] = (float)i * 0.01f;

    Tensor kt = { .data = k_vec, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    Tensor vt = { .data = v_vec, .ndim = 2, .shape = {4, 64}, .dtype = DTYPE_F32 };
    kt.stride[0] = 64; kt.stride[1] = 1;
    vt.stride[0] = 64; vt.stride[1] = 1;

    kv_cache_append(kv, 0, &kt, &vt, 0);
    kv_cache_append(kv, 21, &kt, &vt, 0);

    Tensor* k_first = kv_cache_get_k(kv, 0, 1);
    Tensor* k_last  = kv_cache_get_k(kv, 21, 1);

    int ok = (k_first && k_first->data && k_last && k_last->data);
    if (ok) PASS("kv_tinyllama");
    else    FAIL("kv_tinyllama", "get_k returned NULL for TinyLlama config");

    free(k_vec);
    free(v_vec);
    kv_cache_destroy(kv);
    arena_destroy(arena);
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void) {
    printf("=== Memory & KV Cache Test Suite (Person B) ===\n\n");

    printf("[Arena]\n");
    test_arena_basic();
    test_arena_alignment_stress();
    test_arena_10k_alignment();
    test_arena_overflow();
    test_arena_reset();
    test_arena_no_overlap();
    test_arena_null_safety();

    printf("\n[Scratch]\n");
    test_scratch_basic();
    test_scratch_reuse();
    test_scratch_100_passes();
    test_scratch_overflow();
    test_scratch_alignment();
    test_scratch_null_safety();

    printf("\n[KV Cache]\n");
    test_kv_write_read();
    test_kv_v_cache();
    test_kv_contiguity();
    test_kv_cross_layer();
    test_kv_max_capacity();
    test_kv_tensor_view_shape();
    test_kv_tinyllama_sizing();

    printf("\n");
    if (failures == 0) {
        printf("All %d tests PASSED ✓\n", 20);
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
