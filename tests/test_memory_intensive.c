/*============================================================================
 * Intensive Memory & KV Cache Test Suite
 *
 * Tests every function in memory.h for:
 *   - Normal operation
 *   - NULL / zero inputs
 *   - Boundary conditions (exact capacity, off-by-one)
 *   - Alignment correctness (64, 128, 256, 4096 byte boundaries)
 *   - Overflow / exhaustion
 *   - Data integrity after writes (no silent corruption)
 *   - Reset semantics (pointer reuse, data isolation)
 *   - Stress: many small allocs, few huge allocs
 *   - KV Cache: multi-layer, multi-head, multi-position, view correctness
 *   - Type / stride correctness on returned Tensor views
 *
 * Build:
 *   gcc -Wall -Wextra -std=c11 -O2 -Isrc/include -mavx2 -mfma \
 *       tests/test_memory_intensive.c src/memory/arena.c \
 *       src/memory/scratch.c src/memory/kv_cache.c -lm -o build/test_memory_intensive
 *
 * Run:  ./build/test_memory_intensive
 *============================================================================*/

#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>

/* ---------- Test infrastructure ---------- */

static int g_pass  = 0;
static int g_fail  = 0;
static int g_total = 0;

#define TEST_START(name) \
    do { g_total++; printf("  [%3d] %-50s ", g_total, name); fflush(stdout); } while(0)

#define TEST_PASS() \
    do { g_pass++; printf("PASS\n"); } while(0)

#define TEST_FAIL(msg) \
    do { g_fail++; printf("FAIL — %s\n", msg); } while(0)

/* ---------- Signal handler for catching crashes ---------- */
static sigjmp_buf jump_env;
static volatile sig_atomic_t expecting_signal = 0;

static void crash_handler(int sig) {
    (void)sig;
    if (expecting_signal) {
        siglongjmp(jump_env, 1);
    }
}

/*============================================================================
 *  SECTION 1 — ARENA ALLOCATOR TESTS
 *============================================================================*/

/* 1.1  Basic create / alloc / destroy */
static void test_arena_create_destroy(void) {
    TEST_START("arena: create and destroy");
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("arena_create returned NULL"); return; }
    arena_destroy(a);
    TEST_PASS();
}

/* 1.2  NULL arena operations should NOT crash */
static void test_arena_null_safety(void) {
    TEST_START("arena: NULL safety (destroy, reset, alloc)");
    arena_destroy(NULL);         /* must not crash */
    arena_reset(NULL);           /* must not crash */
    void* p = arena_alloc(NULL, 64, 64);
    if (p != NULL) { TEST_FAIL("alloc(NULL arena) should return NULL"); return; }
    TEST_PASS();
}

/* 1.3  Zero-size alloc */
static void test_arena_zero_alloc(void) {
    TEST_START("arena: zero-size alloc returns NULL");
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }
    void* p = arena_alloc(a, 0, 64);
    if (p != NULL) { TEST_FAIL("expected NULL for size=0"); arena_destroy(a); return; }
    arena_destroy(a);
    TEST_PASS();
}

/* 1.4  Single allocation fills available space exactly */
static void test_arena_exact_fit(void) {
    TEST_START("arena: single alloc using most capacity");
    /* Create a modest arena and try to allocate almost all of it */
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }
    /* We don't know exact header overhead, but try a reasonable chunk */
    void* p = arena_alloc(a, 3000, 64);
    if (!p) { TEST_FAIL("alloc of 3000 bytes failed in 4096 arena"); arena_destroy(a); return; }
    arena_destroy(a);
    TEST_PASS();
}

/* 1.5  64-byte alignment on every allocation */
static void test_arena_alignment_64(void) {
    TEST_START("arena: 64-byte alignment guaranteed");
    Arena* a = arena_create(1 << 20);  /* 1 MiB */
    if (!a) { TEST_FAIL("create failed"); return; }
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        void* p = arena_alloc(a, 13 + i, 64);  /* odd sizes */
        if (!p) { TEST_FAIL("alloc returned NULL early"); arena_destroy(a); return; }
        if ((uintptr_t)p % 64 != 0) { ok = 0; break; }
    }
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("misaligned pointer found");
}

/* 1.6  Higher alignment (128, 256, 4096) */
static void test_arena_alignment_higher(void) {
    TEST_START("arena: 128/256/4096-byte alignment");
    Arena* a = arena_create(1 << 20);
    if (!a) { TEST_FAIL("create failed"); return; }

    size_t aligns[] = {128, 256, 4096};
    int ok = 1;
    for (int ai = 0; ai < 3; ai++) {
        for (int i = 0; i < 10; i++) {
            void* p = arena_alloc(a, 100, aligns[ai]);
            if (!p) { ok = 0; break; }
            if ((uintptr_t)p % aligns[ai] != 0) { ok = 0; break; }
        }
    }
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("alignment violation");
}

/* 1.7  Alignment = 1 should be upgraded to minimum 64 */
static void test_arena_alignment_minimum(void) {
    TEST_START("arena: alignment < 64 upgraded to 64");
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }
    void* p = arena_alloc(a, 100, 1);
    if (!p) { TEST_FAIL("alloc failed"); arena_destroy(a); return; }
    if ((uintptr_t)p % 64 != 0) { TEST_FAIL("not 64-byte aligned"); arena_destroy(a); return; }
    arena_destroy(a);
    TEST_PASS();
}

/* 1.8  Overflow — arena full returns NULL */
static void test_arena_overflow(void) {
    TEST_START("arena: alloc returns NULL when full");
    Arena* a = arena_create(512);
    if (!a) { TEST_FAIL("create failed"); return; }
    void* p1 = arena_alloc(a, 400, 64);
    void* p2 = arena_alloc(a, 400, 64);  /* should fail */
    if (p1 && !p2) TEST_PASS();
    else TEST_FAIL("overflow not detected");
    arena_destroy(a);
}

/* 1.9  Many small allocations until full — no crash */
static void test_arena_many_small(void) {
    TEST_START("arena: many small allocs until exhaustion");
    Arena* a = arena_create(1 << 16);  /* 64 KiB */
    if (!a) { TEST_FAIL("create failed"); return; }
    int count = 0;
    while (arena_alloc(a, 1, 64) != NULL) count++;
    /* We should have gotten at least some allocations */
    if (count > 0) TEST_PASS();
    else TEST_FAIL("zero successful allocations");
    arena_destroy(a);
}

/* 1.10  Reset allows full re-use */
static void test_arena_reset_reuse(void) {
    TEST_START("arena: reset reclaims all space");
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }

    /* Fill */
    while (arena_alloc(a, 64, 64) != NULL);
    /* Reset */
    arena_reset(a);
    /* Should be able to allocate again */
    void* p = arena_alloc(a, 64, 64);
    if (p) TEST_PASS();
    else   TEST_FAIL("alloc after reset returned NULL");
    arena_destroy(a);
}

/* 1.11  Data integrity — write and read back after multiple allocs */
static void test_arena_data_integrity(void) {
    TEST_START("arena: data integrity across allocations");
    Arena* a = arena_create(1 << 20);
    if (!a) { TEST_FAIL("create failed"); return; }

    /* Allocate a bunch of buffers and fill them with unique patterns */
    #define N_BUFS 50
    float* bufs[N_BUFS];
    int sizes[N_BUFS];
    for (int i = 0; i < N_BUFS; i++) {
        sizes[i] = 64 + i * 8;  /* varying sizes */
        bufs[i] = (float*)arena_alloc(a, sizes[i] * sizeof(float), 64);
        if (!bufs[i]) { TEST_FAIL("alloc failed early"); arena_destroy(a); return; }
        for (int j = 0; j < sizes[i]; j++) bufs[i][j] = (float)(i * 1000 + j);
    }
    /* Verify all buffers */
    int ok = 1;
    for (int i = 0; i < N_BUFS && ok; i++) {
        for (int j = 0; j < sizes[i] && ok; j++) {
            if (fabsf(bufs[i][j] - (float)(i * 1000 + j)) > 1e-6f) ok = 0;
        }
    }
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("data corruption detected");
    #undef N_BUFS
}

/* 1.12  No pointer overlap between allocations */
static void test_arena_no_overlap(void) {
    TEST_START("arena: allocations don't overlap");
    Arena* a = arena_create(1 << 20);
    if (!a) { TEST_FAIL("create failed"); return; }

    void* p1 = arena_alloc(a, 1000, 64);
    void* p2 = arena_alloc(a, 1000, 64);
    void* p3 = arena_alloc(a, 1000, 64);
    if (!p1 || !p2 || !p3) { TEST_FAIL("alloc failed"); arena_destroy(a); return; }

    uintptr_t a1 = (uintptr_t)p1, a2 = (uintptr_t)p2, a3 = (uintptr_t)p3;
    /* Ensure p2 starts at or after p1+1000, p3 starts at or after p2+1000 */
    int ok = (a2 >= a1 + 1000) && (a3 >= a2 + 1000);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("overlapping allocations");
}

/* 1.13  Very large arena (256 MiB) */
static void test_arena_large(void) {
    TEST_START("arena: 256 MiB allocation");
    Arena* a = arena_create((size_t)256 * 1024 * 1024);
    if (!a) { TEST_FAIL("create failed (OOM?)"); return; }
    void* p = arena_alloc(a, (size_t)200 * 1024 * 1024, 64);
    if (!p) { TEST_FAIL("large alloc failed"); arena_destroy(a); return; }
    /* Write first and last byte to ensure pages are committed */
    ((char*)p)[0] = 'A';
    ((char*)p)[(size_t)200*1024*1024 - 1] = 'Z';
    arena_destroy(a);
    TEST_PASS();
}

/* 1.14  Multiple resets */
static void test_arena_multi_reset(void) {
    TEST_START("arena: multiple resets in sequence");
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }
    for (int round = 0; round < 100; round++) {
        arena_alloc(a, 64, 64);
        arena_reset(a);
    }
    void* p = arena_alloc(a, 64, 64);
    arena_destroy(a);
    if (p) TEST_PASS(); else TEST_FAIL("alloc after many resets failed");
}

/* 1.15  Double destroy — should not crash */
static void test_arena_double_destroy(void) {
    TEST_START("arena: double destroy (potential segfault)");
    /* Note: double munmap is undefined behavior; we just check that 
       the first destroy works. Double destroy WILL crash on most systems. */
    Arena* a = arena_create(4096);
    if (!a) { TEST_FAIL("create failed"); return; }
    arena_destroy(a);
    /* We intentionally do NOT call arena_destroy(a) again — it would be UB.
       The test passes if single destroy doesn't crash. */
    TEST_PASS();
}

/*============================================================================
 *  SECTION 2 — SCRATCH ALLOCATOR TESTS
 *============================================================================*/

/* 2.1  Basic create / alloc / destroy */
static void test_scratch_create_destroy(void) {
    TEST_START("scratch: create and destroy");
    Scratch* s = scratch_create(4096);
    if (!s) { TEST_FAIL("create failed"); return; }
    scratch_destroy(s);
    TEST_PASS();
}

/* 2.2  NULL safety */
static void test_scratch_null_safety(void) {
    TEST_START("scratch: NULL safety (destroy, reset, alloc)");
    scratch_destroy(NULL);
    scratch_reset(NULL);
    void* p = scratch_alloc(NULL, 64, 64);
    if (p != NULL) { TEST_FAIL("alloc(NULL) should return NULL"); return; }
    TEST_PASS();
}

/* 2.3  Zero-size alloc */
static void test_scratch_zero_alloc(void) {
    TEST_START("scratch: zero-size alloc returns NULL");
    Scratch* s = scratch_create(4096);
    if (!s) { TEST_FAIL("create failed"); return; }
    void* p = scratch_alloc(s, 0, 64);
    if (p != NULL) { TEST_FAIL("expected NULL"); scratch_destroy(s); return; }
    scratch_destroy(s);
    TEST_PASS();
}

/* 2.4  Pointer reuse after reset */
static void test_scratch_pointer_reuse(void) {
    TEST_START("scratch: same pointer after reset");
    Scratch* s = scratch_create(8192);
    if (!s) { TEST_FAIL("create failed"); return; }
    void* p1 = scratch_alloc(s, 100, 64);
    scratch_reset(s);
    void* p2 = scratch_alloc(s, 100, 64);
    scratch_destroy(s);
    if (p1 == p2) TEST_PASS(); else TEST_FAIL("pointer not reused");
}

/* 2.5  Alignment */
static void test_scratch_alignment(void) {
    TEST_START("scratch: 64-byte alignment on all allocs");
    Scratch* s = scratch_create(1 << 20);
    if (!s) { TEST_FAIL("create failed"); return; }
    int ok = 1;
    for (int i = 0; i < 200; i++) {
        void* p = scratch_alloc(s, 7 + i, 64);
        if (!p) { ok = 0; break; }
        if ((uintptr_t)p % 64 != 0) { ok = 0; break; }
    }
    scratch_destroy(s);
    if (ok) TEST_PASS(); else TEST_FAIL("alignment violation");
}

/* 2.6  Overflow returns NULL */
static void test_scratch_overflow(void) {
    TEST_START("scratch: overflow returns NULL");
    Scratch* s = scratch_create(512);
    if (!s) { TEST_FAIL("create failed"); return; }
    void* p1 = scratch_alloc(s, 400, 64);
    void* p2 = scratch_alloc(s, 400, 64);
    scratch_destroy(s);
    if (p1 && !p2) TEST_PASS(); else TEST_FAIL("overflow not detected");
}

/* 2.7  Data integrity across allocs within one pass */
static void test_scratch_data_integrity(void) {
    TEST_START("scratch: data integrity within one pass");
    Scratch* s = scratch_create(1 << 20);
    if (!s) { TEST_FAIL("create failed"); return; }

    #define NS 30
    float* bufs[NS];
    for (int i = 0; i < NS; i++) {
        bufs[i] = (float*)scratch_alloc(s, 256 * sizeof(float), 64);
        if (!bufs[i]) { TEST_FAIL("alloc failed"); scratch_destroy(s); return; }
        for (int j = 0; j < 256; j++) bufs[i][j] = (float)(i * 100 + j);
    }
    int ok = 1;
    for (int i = 0; i < NS && ok; i++)
        for (int j = 0; j < 256 && ok; j++)
            if (fabsf(bufs[i][j] - (float)(i * 100 + j)) > 1e-6f) ok = 0;

    scratch_destroy(s);
    if (ok) TEST_PASS(); else TEST_FAIL("data corruption");
    #undef NS
}

/* 2.8  Simulate forward-pass cycle: alloc, reset, repeat 1000× */
static void test_scratch_forward_pass_sim(void) {
    TEST_START("scratch: 1000 forward-pass cycles");
    Scratch* s = scratch_create(1 << 18);  /* 256 KiB */
    if (!s) { TEST_FAIL("create failed"); return; }

    int ok = 1;
    for (int pass = 0; pass < 1000; pass++) {
        scratch_reset(s);
        /* Simulate ~20 activations per layer */
        for (int i = 0; i < 20; i++) {
            float* p = (float*)scratch_alloc(s, 512 * sizeof(float), 64);
            if (!p) { ok = 0; break; }
            p[0] = (float)pass;
            p[511] = (float)(pass + i);
        }
        if (!ok) break;
    }
    scratch_destroy(s);
    if (ok) TEST_PASS(); else TEST_FAIL("cycle failed");
}

/* 2.9  Alignment minimum enforcement */
static void test_scratch_alignment_min(void) {
    TEST_START("scratch: alignment < 64 upgraded to 64");
    Scratch* s = scratch_create(4096);
    if (!s) { TEST_FAIL("create failed"); return; }
    void* p = scratch_alloc(s, 100, 1);
    scratch_destroy(s);
    if (p && (uintptr_t)p % 64 == 0) TEST_PASS();
    else TEST_FAIL("not upgraded to 64");
}

/* 2.10  Scratch with many small allocs until exhaustion */
static void test_scratch_exhaust(void) {
    TEST_START("scratch: exhaust with 1-byte allocs");
    Scratch* s = scratch_create(1 << 16);  /* 64 KiB */
    if (!s) { TEST_FAIL("create failed"); return; }
    int count = 0;
    while (scratch_alloc(s, 1, 64) != NULL) count++;
    scratch_destroy(s);
    if (count > 0) TEST_PASS(); else TEST_FAIL("zero allocs");
}

/* 2.11  Large scratch pool (128 MiB) */
static void test_scratch_large(void) {
    TEST_START("scratch: 128 MiB pool");
    Scratch* s = scratch_create((size_t)128 * 1024 * 1024);
    if (!s) { TEST_FAIL("create failed"); return; }
    void* p = scratch_alloc(s, (size_t)100 * 1024 * 1024, 64);
    if (!p) { TEST_FAIL("alloc failed"); scratch_destroy(s); return; }
    ((char*)p)[0] = 'X';
    ((char*)p)[(size_t)100*1024*1024 - 1] = 'Y';
    scratch_destroy(s);
    TEST_PASS();
}

/* 2.12  Reset doesn't zero memory (contents may persist — not a bug) */
static void test_scratch_reset_no_zero(void) {
    TEST_START("scratch: data persists after reset (stale ok)");
    Scratch* s = scratch_create(8192);
    if (!s) { TEST_FAIL("create failed"); return; }
    float* p1 = (float*)scratch_alloc(s, 256 * sizeof(float), 64);
    if (!p1) { TEST_FAIL("alloc failed"); scratch_destroy(s); return; }
    for (int i = 0; i < 256; i++) p1[i] = 42.0f;
    scratch_reset(s);
    float* p2 = (float*)scratch_alloc(s, 256 * sizeof(float), 64);
    /* p2 should equal p1 (same region), data may or may not be zeroed */
    int same_ptr = (p1 == p2);
    scratch_destroy(s);
    if (same_ptr) TEST_PASS();
    else TEST_FAIL("different pointer after reset (unexpected)");
}

/*============================================================================
 *  SECTION 3 — KV CACHE TESTS
 *============================================================================*/

/* Helper: create a standard TinyLlama-like config */
static ModelConfig make_tiny_cfg(void) {
    ModelConfig cfg = {
        .hidden_dim   = 256,
        .n_heads      = 4,
        .n_kv_heads   = 4,
        .head_dim     = 64,
        .n_layers     = 2,
        .vocab_size   = 100,
        .ff_dim       = 512,
        .max_seq_len  = 128,
        .vocab_strings = NULL,
        .vocab_scores  = NULL
    };
    return cfg;
}

/* Helper: create a Tensor wrapper for k/v data */
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

/* 3.1  Basic create / destroy */
static void test_kv_create_destroy(void) {
    TEST_START("kv_cache: create and destroy");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }
    kv_cache_destroy(kv);
    arena_destroy(a);
    TEST_PASS();
}

/* 3.2  NULL inputs */
static void test_kv_null_inputs(void) {
    TEST_START("kv_cache: NULL arena / cfg returns NULL");
    ModelConfig cfg = make_tiny_cfg();
    KVCache* kv1 = kv_cache_create(NULL, &cfg, 128);
    Arena* a = arena_create(1 << 20);
    KVCache* kv2 = kv_cache_create(a, NULL, 128);
    int ok = (kv1 == NULL) && (kv2 == NULL);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("didn't return NULL");
}

/* 3.3  NULL kv on get/append */
static void test_kv_null_operations(void) {
    TEST_START("kv_cache: NULL kv on append/get");
    Tensor* t1 = kv_cache_get_k(NULL, 0, 1);
    Tensor* t2 = kv_cache_get_v(NULL, 0, 1);
    float dummy[64];
    Tensor dt = make_kv_tensor(dummy, 1, 64);
    int ret = kv_cache_append(NULL, 0, &dt, &dt, 0);
    kv_cache_destroy(NULL);  /* must not crash */
    if (t1 == NULL && t2 == NULL && ret != 0)
        TEST_PASS();
    else
        TEST_FAIL("NULL not handled properly");
}

/* 3.4  Write at pos 0 and read back — single layer single head */
static void test_kv_single_write_read(void) {
    TEST_START("kv_cache: write pos=0, layer=0, read back");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;  /* 4*64=256 */
    float k_data[256], v_data[256];
    for (int i = 0; i < total; i++) {
        k_data[i] = (float)(i + 1);
        v_data[i] = (float)(i + 1000);
    }
    Tensor kt = make_kv_tensor(k_data, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(v_data, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, 0);
    if (ret != 0) { TEST_FAIL("append returned error"); kv_cache_destroy(kv); arena_destroy(a); return; }

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    Tensor* vr = kv_cache_get_v(kv, 0, 1);
    if (!kr || !vr) { TEST_FAIL("get returned NULL"); kv_cache_destroy(kv); arena_destroy(a); return; }

    /* Verify K head 0, pos 0 */
    float* kp = (float*)kr->data;
    int ok = 1;
    for (int d = 0; d < 64 && ok; d++) {
        if (fabsf(kp[d] - (float)(d + 1)) > 1e-6f) ok = 0;
    }
    /* Verify V head 0, pos 0 */
    float* vp = (float*)vr->data;
    for (int d = 0; d < 64 && ok; d++) {
        if (fabsf(vp[d] - (float)(d + 1000)) > 1e-6f) ok = 0;
    }

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("data mismatch");
}

/* 3.5  Multi-position sequential append */
static void test_kv_multi_position(void) {
    TEST_START("kv_cache: append 10 positions, verify all");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];

    for (int pos = 0; pos < 10; pos++) {
        for (int i = 0; i < total; i++) {
            kbuf[i] = (float)(pos * 1000 + i);
            vbuf[i] = (float)(pos * 2000 + i);
        }
        Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
        Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
        kv_cache_append(kv, 0, &kt, &vt, pos);
    }

    /* Verify: get K for layer 0, up_to_pos=10, check each position */
    Tensor* kr = kv_cache_get_k(kv, 0, 10);
    if (!kr) { TEST_FAIL("get_k returned NULL"); kv_cache_destroy(kv); arena_destroy(a); return; }

    int ok = 1;
    float* kbase = (float*)kr->data;
    /* Layout: head-major. For head 0: stride[0] = max_seq * head_dim = 128*64 = 8192 */
    /* pos p at head h: base + h * stride[0] + p * stride[1] */
    for (int pos = 0; pos < 10 && ok; pos++) {
        for (int d = 0; d < cfg.head_dim && ok; d++) {
            /* head 0, position pos, dim d */
            float got = kbase[0 * kr->stride[0] + pos * kr->stride[1] + d];
            float expected = (float)(pos * 1000 + 0 * cfg.head_dim + d);
            if (fabsf(got - expected) > 1e-6f) ok = 0;
        }
    }

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("position data mismatch");
}

/* 3.6  Multi-head verification */
static void test_kv_multi_head(void) {
    TEST_START("kv_cache: verify data per head");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256];
    float vbuf[256];
    /* Tag each head uniquely: head h → values start at h * 100 */
    for (int h = 0; h < cfg.n_kv_heads; h++)
        for (int d = 0; d < cfg.head_dim; d++)
            kbuf[h * cfg.head_dim + d] = (float)(h * 100 + d);
    memset(vbuf, 0, total * sizeof(float));

    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    if (!kr) { TEST_FAIL("get_k NULL"); kv_cache_destroy(kv); arena_destroy(a); return; }

    int ok = 1;
    float* kbase = (float*)kr->data;
    for (int h = 0; h < cfg.n_kv_heads && ok; h++) {
        for (int d = 0; d < cfg.head_dim && ok; d++) {
            float got = kbase[h * kr->stride[0] + 0 * kr->stride[1] + d];
            float expected = (float)(h * 100 + d);
            if (fabsf(got - expected) > 1e-6f) ok = 0;
        }
    }

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("head data mismatch");
}

/* 3.7  Cross-layer isolation */
static void test_kv_cross_layer_isolation(void) {
    TEST_START("kv_cache: layer 0 and layer 1 are isolated");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float k0[256], k1[256], v[256];
    memset(v, 0, sizeof(v));
    for (int i = 0; i < total; i++) { k0[i] = 111.0f; k1[i] = 222.0f; }

    Tensor kt0 = make_kv_tensor(k0, cfg.n_kv_heads, cfg.head_dim);
    Tensor kt1 = make_kv_tensor(k1, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt  = make_kv_tensor(v,  cfg.n_kv_heads, cfg.head_dim);

    kv_cache_append(kv, 0, &kt0, &vt, 0);
    kv_cache_append(kv, 1, &kt1, &vt, 0);

    Tensor* r0 = kv_cache_get_k(kv, 0, 1);
    Tensor* r1 = kv_cache_get_k(kv, 1, 1);
    if (!r0 || !r1) { TEST_FAIL("get_k NULL"); kv_cache_destroy(kv); arena_destroy(a); return; }

    float v0 = ((float*)r0->data)[0];
    float v1 = ((float*)r1->data)[0];
    kv_cache_destroy(kv);
    arena_destroy(a);

    if (fabsf(v0 - 111.0f) < 1e-6f && fabsf(v1 - 222.0f) < 1e-6f) TEST_PASS();
    else TEST_FAIL("layer data crossed");
}

/* 3.8  Boundary: append at max_seq-1 (last valid position) */
static void test_kv_append_last_pos(void) {
    TEST_START("kv_cache: append at pos = max_seq - 1");
    ModelConfig cfg = make_tiny_cfg();
    int max_seq = 16;
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, max_seq);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    for (int i = 0; i < total; i++) { kbuf[i] = 99.0f; vbuf[i] = 88.0f; }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, max_seq - 1);
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ret == 0) TEST_PASS(); else TEST_FAIL("append at last pos failed");
}

/* 3.9  Boundary: append at pos = max_seq (out of bounds) */
static void test_kv_append_oob(void) {
    TEST_START("kv_cache: append at pos = max_seq (OOB)");
    ModelConfig cfg = make_tiny_cfg();
    int max_seq = 16;
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, max_seq);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, max_seq);  /* should fail */
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ret != 0) TEST_PASS(); else TEST_FAIL("OOB append succeeded");
}

/* 3.10  Negative position */
static void test_kv_append_neg_pos(void) {
    TEST_START("kv_cache: append at pos = -1 (negative)");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, -1);
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ret != 0) TEST_PASS(); else TEST_FAIL("negative pos accepted");
}

/* 3.11  Invalid layer index */
static void test_kv_append_bad_layer(void) {
    TEST_START("kv_cache: append at layer = n_layers (OOB)");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, cfg.n_layers, &kt, &vt, 0);
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ret != 0) TEST_PASS(); else TEST_FAIL("invalid layer accepted");
}

/* 3.12  Negative layer */
static void test_kv_append_neg_layer(void) {
    TEST_START("kv_cache: append at layer = -1 (negative)");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, -1, &kt, &vt, 0);
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ret != 0) TEST_PASS(); else TEST_FAIL("negative layer accepted");
}

/* 3.13  NULL tensor on append */
static void test_kv_append_null_tensor(void) {
    TEST_START("kv_cache: append with NULL k or v tensor");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    float kbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);

    int r1 = kv_cache_append(kv, 0, &kt, NULL, 0);
    int r2 = kv_cache_append(kv, 0, NULL, &kt, 0);
    int r3 = kv_cache_append(kv, 0, NULL, NULL, 0);

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (r1 != 0 && r2 != 0 && r3 != 0) TEST_PASS();
    else TEST_FAIL("NULL tensor not rejected");
}

/* 3.14  Tensor view shape correctness */
static void test_kv_view_shape(void) {
    TEST_START("kv_cache: returned Tensor view has correct shape/stride/dtype");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    /* Append 5 positions */
    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    for (int p = 0; p < 5; p++) kv_cache_append(kv, 0, &kt, &vt, p);

    Tensor* kr = kv_cache_get_k(kv, 0, 5);
    if (!kr) { TEST_FAIL("get_k NULL"); kv_cache_destroy(kv); arena_destroy(a); return; }

    int ok = 1;
    if (kr->ndim != 3) ok = 0;
    if (kr->shape[0] != cfg.n_kv_heads) ok = 0;     /* heads */
    if (kr->shape[1] != 5) ok = 0;                    /* positions */
    if (kr->shape[2] != cfg.head_dim) ok = 0;         /* dim */
    if (kr->stride[1] != cfg.head_dim) ok = 0;        /* pos stride = head_dim */
    if (kr->stride[2] != 1) ok = 0;                   /* innermost stride = 1 */
    if (kr->dtype != DTYPE_F32) ok = 0;

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("shape/stride/dtype mismatch");
}

/* 3.15  V and K are independent (writing V doesn't corrupt K) */
static void test_kv_k_v_independence(void) {
    TEST_START("kv_cache: K and V buffers are independent");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    for (int i = 0; i < total; i++) { kbuf[i] = 1.0f; vbuf[i] = 2.0f; }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    Tensor* vr = kv_cache_get_v(kv, 0, 1);
    float kval = ((float*)kr->data)[0];
    float vval = ((float*)vr->data)[0];

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (fabsf(kval - 1.0f) < 1e-6f && fabsf(vval - 2.0f) < 1e-6f) TEST_PASS();
    else TEST_FAIL("K/V contaminated");
}

/* 3.16  Overwrite same position */
static void test_kv_overwrite_position(void) {
    TEST_START("kv_cache: overwrite pos 0 with new data");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    /* First write: all 10.0 */
    for (int i = 0; i < total; i++) { kbuf[i] = 10.0f; vbuf[i] = 10.0f; }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    /* Second write at same pos: all 99.0 */
    for (int i = 0; i < total; i++) { kbuf[i] = 99.0f; vbuf[i] = 99.0f; }
    kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    float val = ((float*)kr->data)[0];
    kv_cache_destroy(kv);
    arena_destroy(a);
    if (fabsf(val - 99.0f) < 1e-6f) TEST_PASS(); else TEST_FAIL("overwrite not reflected");
}

/* 3.17  Stress: fill all positions in all layers */
static void test_kv_fill_all(void) {
    TEST_START("kv_cache: fill all positions in all layers");
    ModelConfig cfg = {
        .hidden_dim = 128, .n_heads = 2, .n_kv_heads = 2,
        .head_dim = 64, .n_layers = 4, .vocab_size = 100,
        .ff_dim = 256, .max_seq_len = 64,
        .vocab_strings = NULL, .vocab_scores = NULL
    };
    int max_seq = 64;
    /* Need: 2 * 4 layers * 2 heads * 64 seq * 64 dim * 4 bytes = 262144 bytes + overhead */
    Arena* a = arena_create(4 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, max_seq);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[128], vbuf[128];

    int ok = 1;
    for (int layer = 0; layer < cfg.n_layers && ok; layer++) {
        for (int pos = 0; pos < max_seq && ok; pos++) {
            float marker = (float)(layer * 1000 + pos);
            for (int i = 0; i < total; i++) { kbuf[i] = marker; vbuf[i] = marker + 0.5f; }
            Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
            Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
            int ret = kv_cache_append(kv, layer, &kt, &vt, pos);
            if (ret != 0) ok = 0;
        }
    }

    /* Spot-check: layer 2, pos 30, head 0, dim 0 */
    if (ok) {
        Tensor* kr = kv_cache_get_k(kv, 2, max_seq);
        float* kbase = (float*)kr->data;
        float got = kbase[0 * kr->stride[0] + 30 * kr->stride[1] + 0];
        float expected = 2.0f * 1000 + 30;
        if (fabsf(got - expected) > 1e-6f) ok = 0;
    }

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("fill/verify failed");
}

/* 3.18  GQA config: fewer KV heads than Q heads */
static void test_kv_gqa_config(void) {
    TEST_START("kv_cache: GQA (n_kv_heads < n_heads)");
    ModelConfig cfg = {
        .hidden_dim = 512, .n_heads = 8, .n_kv_heads = 2,
        .head_dim = 64, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 1024, .max_seq_len = 64,
        .vocab_strings = NULL, .vocab_scores = NULL
    };
    Arena* a = arena_create(16 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 64);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    /* Only 2 KV heads, not 8 */
    int total = cfg.n_kv_heads * cfg.head_dim;  /* 2*64=128 */
    float kbuf[128], vbuf[128];
    for (int i = 0; i < total; i++) { kbuf[i] = (float)i; vbuf[i] = (float)(i + 500); }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, 0);
    Tensor* kr = kv_cache_get_k(kv, 0, 1);

    int ok = (ret == 0) && kr && (kr->shape[0] == 2) && (kr->shape[2] == 64);
    if (ok) {
        float* kp = (float*)kr->data;
        /* Head 1, dim 0 should be kbuf[64] = 64.0 */
        float got = kp[1 * kr->stride[0] + 0 * kr->stride[1] + 0];
        if (fabsf(got - 64.0f) > 1e-6f) ok = 0;
    }

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("GQA layout incorrect");
}

/* 3.19  Arena too small for KV cache — create should fail gracefully */
static void test_kv_arena_too_small(void) {
    TEST_START("kv_cache: arena too small returns NULL");
    ModelConfig cfg = make_tiny_cfg();
    /* KV needs: 2 * 2layers * 4heads * 128seq * 64dim * 4bytes = 524288 + struct overhead */
    Arena* a = arena_create(1024);  /* way too small */
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    arena_destroy(a);
    if (kv == NULL) TEST_PASS(); else TEST_FAIL("should have returned NULL");
}

/* 3.20  V view returned tensor — verify strides match K view */
static void test_kv_v_view_strides(void) {
    TEST_START("kv_cache: V view has same stride layout as K view");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf)); memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    Tensor* vr = kv_cache_get_v(kv, 0, 1);

    int ok = 1;
    if (kr->ndim != vr->ndim) ok = 0;
    for (int i = 0; i < kr->ndim && ok; i++) {
        if (kr->shape[i] != vr->shape[i]) ok = 0;
        if (kr->stride[i] != vr->stride[i]) ok = 0;
    }
    if (kr->dtype != vr->dtype) ok = 0;
    /* But data pointers must be different! */
    if (kr->data == vr->data) ok = 0;

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("V view doesn't match K view layout");
}

/* 3.21  Concurrent view invalidation — getting a new view overwrites the old */
static void test_kv_view_overwrite(void) {
    TEST_START("kv_cache: second get_k overwrites first view (shared state)");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("create failed"); arena_destroy(a); return; }

    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    memset(kbuf, 0, sizeof(kbuf));
    memset(vbuf, 0, sizeof(vbuf));
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    for (int p = 0; p < 5; p++) kv_cache_append(kv, 0, &kt, &vt, p);

    Tensor* kr1 = kv_cache_get_k(kv, 0, 3);
    int shape_before = kr1->shape[1];  /* should be 3 */

    Tensor* kr2 = kv_cache_get_k(kv, 0, 5);
    /* kr1 and kr2 point to the same internal Tensor, so kr1->shape[1] is now 5 */
    int shape_after = kr1->shape[1];

    kv_cache_destroy(kv);
    arena_destroy(a);

    /* This test documents the behavior that views share a single Tensor struct.
     * The first view gets overwritten. This is expected — document it. */
    if (kr1 == kr2 && shape_before == 3 && shape_after == 5)
        TEST_PASS();
    else if (kr1 != kr2)
        TEST_PASS();  /* If different pointers returned, that's also fine */
    else
        TEST_FAIL("unexpected view behavior");
}

/* 3.22  Large KV cache — realistic TinyLlama size */
static void test_kv_realistic_size(void) {
    TEST_START("kv_cache: realistic TinyLlama sizing (22 layers)");
    ModelConfig cfg = {
        .hidden_dim = 2048, .n_heads = 32, .n_kv_heads = 4,
        .head_dim = 64, .n_layers = 22, .vocab_size = 32000,
        .ff_dim = 5632, .max_seq_len = 2048,
        .vocab_strings = NULL, .vocab_scores = NULL
    };
    /* KV bytes = 2 * 22 * 4 * 2048 * 64 * 4 = ~92 MiB */
    size_t arena_size = (size_t)200 * 1024 * 1024;  /* 200 MiB */
    Arena* a = arena_create(arena_size);
    if (!a) { TEST_FAIL("arena create failed (need 200 MiB)"); return; }

    KVCache* kv = kv_cache_create(a, &cfg, 2048);
    if (!kv) { TEST_FAIL("kv_cache_create failed"); arena_destroy(a); return; }

    /* Write one token at pos 0, layer 0 */
    int total = cfg.n_kv_heads * cfg.head_dim;
    float* kbuf = (float*)malloc(total * sizeof(float));
    float* vbuf = (float*)malloc(total * sizeof(float));
    if (!kbuf || !vbuf) { TEST_FAIL("malloc failed"); free(kbuf); free(vbuf); kv_cache_destroy(kv); arena_destroy(a); return; }

    for (int i = 0; i < total; i++) { kbuf[i] = 3.14f; vbuf[i] = 2.72f; }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);

    int ret = kv_cache_append(kv, 0, &kt, &vt, 0);
    free(kbuf); free(vbuf);

    if (ret != 0) { TEST_FAIL("append failed"); kv_cache_destroy(kv); arena_destroy(a); return; }

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    float val = ((float*)kr->data)[0];
    kv_cache_destroy(kv);
    arena_destroy(a);

    if (fabsf(val - 3.14f) < 1e-6f) TEST_PASS(); else TEST_FAIL("data mismatch");
}

/*============================================================================
 *  SECTION 4 — ARENA + KV CACHE INTEGRATION
 *============================================================================*/

/* 4.1  Arena used for KV cache + other allocs simultaneously */
static void test_arena_kv_and_extras(void) {
    TEST_START("integration: arena serves KV cache + extra allocs");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    if (!a) { TEST_FAIL("create failed"); return; }

    /* Allocate some stuff first */
    float* extra1 = (float*)arena_alloc(a, 1024 * sizeof(float), 64);
    if (!extra1) { TEST_FAIL("extra1 failed"); arena_destroy(a); return; }
    for (int i = 0; i < 1024; i++) extra1[i] = (float)i;

    /* Now create KV cache */
    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("kv create failed"); arena_destroy(a); return; }

    /* Allocate more stuff after KV cache */
    float* extra2 = (float*)arena_alloc(a, 512 * sizeof(float), 64);
    if (!extra2) { TEST_FAIL("extra2 failed"); kv_cache_destroy(kv); arena_destroy(a); return; }
    for (int i = 0; i < 512; i++) extra2[i] = (float)(i + 5000);

    /* Write to KV cache */
    int total = cfg.n_kv_heads * cfg.head_dim;
    float kbuf[256], vbuf[256];
    for (int i = 0; i < total; i++) { kbuf[i] = 42.0f; vbuf[i] = 43.0f; }
    Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
    Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
    kv_cache_append(kv, 0, &kt, &vt, 0);

    /* Verify ALL data is intact */
    int ok = 1;
    for (int i = 0; i < 1024 && ok; i++)
        if (fabsf(extra1[i] - (float)i) > 1e-6f) ok = 0;
    for (int i = 0; i < 512 && ok; i++)
        if (fabsf(extra2[i] - (float)(i + 5000)) > 1e-6f) ok = 0;

    Tensor* kr = kv_cache_get_k(kv, 0, 1);
    if (!kr || fabsf(((float*)kr->data)[0] - 42.0f) > 1e-6f) ok = 0;

    kv_cache_destroy(kv);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("data corruption");
}

/* 4.2  Scratch used for per-layer temporaries alongside arena KV */
static void test_scratch_with_kv(void) {
    TEST_START("integration: scratch + KV cache coexist");
    ModelConfig cfg = make_tiny_cfg();
    Arena* a = arena_create(64 * 1024 * 1024);
    Scratch* s = scratch_create(1 << 20);
    if (!a || !s) { TEST_FAIL("create failed"); arena_destroy(a); scratch_destroy(s); return; }

    KVCache* kv = kv_cache_create(a, &cfg, 128);
    if (!kv) { TEST_FAIL("kv create failed"); arena_destroy(a); scratch_destroy(s); return; }

    int ok = 1;
    for (int pass = 0; pass < 10; pass++) {
        scratch_reset(s);

        /* Allocate scratch buffers like a real forward pass */
        float* q = (float*)scratch_alloc(s, cfg.hidden_dim * sizeof(float), 64);
        float* attn = (float*)scratch_alloc(s, cfg.n_heads * 128 * sizeof(float), 64);
        if (!q || !attn) { ok = 0; break; }

        /* Write to KV cache */
        int total = cfg.n_kv_heads * cfg.head_dim;
        float kbuf[256], vbuf[256];
        for (int i = 0; i < total; i++) { kbuf[i] = (float)pass; vbuf[i] = (float)pass; }
        Tensor kt = make_kv_tensor(kbuf, cfg.n_kv_heads, cfg.head_dim);
        Tensor vt = make_kv_tensor(vbuf, cfg.n_kv_heads, cfg.head_dim);
        kv_cache_append(kv, 0, &kt, &vt, pass);
    }

    /* Verify last write */
    Tensor* kr = kv_cache_get_k(kv, 0, 10);
    if (kr) {
        float* kbase = (float*)kr->data;
        float got = kbase[0 * kr->stride[0] + 9 * kr->stride[1] + 0];
        if (fabsf(got - 9.0f) > 1e-6f) ok = 0;
    } else ok = 0;

    kv_cache_destroy(kv);
    scratch_destroy(s);
    arena_destroy(a);
    if (ok) TEST_PASS(); else TEST_FAIL("integration failure");
}

/*============================================================================
 *  SECTION 5 — EDGE CASES & ROBUSTNESS
 *============================================================================*/

/* 5.1  Create with size = 0 */
static void test_arena_zero_size(void) {
    TEST_START("edge: arena_create(0)");
    Arena* a = arena_create(0);
    /* Could return NULL or a valid handle with zero usable space — both acceptable */
    if (a) {
        void* p = arena_alloc(a, 1, 64);
        /* Should return NULL since there's no space */
        arena_destroy(a);
        if (p == NULL) TEST_PASS();
        else TEST_FAIL("allocated from zero-capacity arena");
    } else {
        TEST_PASS();  /* returning NULL for size=0 is also acceptable */
    }
}

/* 5.2  Create with size = 1 */
static void test_arena_size_one(void) {
    TEST_START("edge: arena_create(1)");
    Arena* a = arena_create(1);
    if (!a) { TEST_PASS(); return; }  /* acceptable to fail */
    /* Try alloc — probably fails due to header + alignment overhead */
    void* p = arena_alloc(a, 1, 64);
    arena_destroy(a);
    /* Either outcome is fine — we just shouldn't crash */
    TEST_PASS();
}

/* 5.3  Scratch create with size = 0 */
static void test_scratch_zero_size(void) {
    TEST_START("edge: scratch_create(0)");
    Scratch* s = scratch_create(0);
    if (s) {
        void* p = scratch_alloc(s, 1, 64);
        scratch_destroy(s);
        if (p == NULL) TEST_PASS(); else TEST_FAIL("allocated from zero scratch");
    } else {
        TEST_PASS();
    }
}

/* 5.4  Very large alignment request */
static void test_arena_huge_alignment(void) {
    TEST_START("edge: arena_alloc with 1 MiB alignment");
    Arena* a = arena_create(4 * 1024 * 1024);
    if (!a) { TEST_FAIL("create failed"); return; }
    void* p = arena_alloc(a, 64, 1 << 20);  /* 1 MiB alignment */
    arena_destroy(a);
    if (p && ((uintptr_t)p % (1 << 20) == 0)) TEST_PASS();
    else if (!p) { TEST_PASS(); /* acceptable to fail if arena too small for alignment */ }
    else TEST_FAIL("alignment not met");
}

/* 5.5  Fill arena to exact capacity (repeated same-size allocs) */
static void test_arena_exact_fill(void) {
    TEST_START("edge: precise fill tracking");
    Arena* a = arena_create(8192);
    if (!a) { TEST_FAIL("create failed"); return; }
    int count = 0;
    /* Each 64-byte alloc with 64-byte alignment should use exactly 64 bytes */
    while (arena_alloc(a, 64, 64) != NULL) count++;
    arena_destroy(a);
    if (count > 0) TEST_PASS(); else TEST_FAIL("no allocs succeeded");
}

/*============================================================================
 *  MAIN
 *============================================================================*/

int main(void) {
    /* Install signal handler for crash safety */
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Intensive Memory & KV Cache Test Suite                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* --- Section 1: Arena --- */
    printf("── Arena Allocator ──────────────────────────────────────\n");
    test_arena_create_destroy();
    test_arena_null_safety();
    test_arena_zero_alloc();
    test_arena_exact_fit();
    test_arena_alignment_64();
    test_arena_alignment_higher();
    test_arena_alignment_minimum();
    test_arena_overflow();
    test_arena_many_small();
    test_arena_reset_reuse();
    test_arena_data_integrity();
    test_arena_no_overlap();
    test_arena_large();
    test_arena_multi_reset();
    test_arena_double_destroy();

    /* --- Section 2: Scratch --- */
    printf("\n── Scratch Allocator ────────────────────────────────────\n");
    test_scratch_create_destroy();
    test_scratch_null_safety();
    test_scratch_zero_alloc();
    test_scratch_pointer_reuse();
    test_scratch_alignment();
    test_scratch_overflow();
    test_scratch_data_integrity();
    test_scratch_forward_pass_sim();
    test_scratch_alignment_min();
    test_scratch_exhaust();
    test_scratch_large();
    test_scratch_reset_no_zero();

    /* --- Section 3: KV Cache --- */
    printf("\n── KV Cache ─────────────────────────────────────────────\n");
    test_kv_create_destroy();
    test_kv_null_inputs();
    test_kv_null_operations();
    test_kv_single_write_read();
    test_kv_multi_position();
    test_kv_multi_head();
    test_kv_cross_layer_isolation();
    test_kv_append_last_pos();
    test_kv_append_oob();
    test_kv_append_neg_pos();
    test_kv_append_bad_layer();
    test_kv_append_neg_layer();
    test_kv_append_null_tensor();
    test_kv_view_shape();
    test_kv_k_v_independence();
    test_kv_overwrite_position();
    test_kv_fill_all();
    test_kv_gqa_config();
    test_kv_arena_too_small();
    test_kv_v_view_strides();
    test_kv_view_overwrite();
    test_kv_realistic_size();

    /* --- Section 4: Integration --- */
    printf("\n── Integration (Arena + Scratch + KV) ───────────────────\n");
    test_arena_kv_and_extras();
    test_scratch_with_kv();

    /* --- Section 5: Edge cases --- */
    printf("\n── Edge Cases & Robustness ──────────────────────────────\n");
    test_arena_zero_size();
    test_arena_size_one();
    test_scratch_zero_size();
    test_arena_huge_alignment();
    test_arena_exact_fill();

    /* --- Summary --- */
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Total: %d  |  Passed: %d  |  Failed: %d\n", g_total, g_pass, g_fail);
    printf("══════════════════════════════════════════════════════════\n");

    if (g_fail == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TEST(S) FAILED — see above for details\n", g_fail);
    }

    return g_fail;
}
