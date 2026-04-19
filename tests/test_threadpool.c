#include "threadpool.h"

#include <stdio.h>
#include <stdint.h>

#if defined(USE_PTHREAD) && (USE_PTHREAD == 1)
#include <pthread.h>
#endif

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

typedef struct {
    long long sum;
} SumCtx;

static void sum_range(int start, int end, void* ctx) {
    SumCtx* s = (SumCtx*)ctx;
    long long local = 0;
    for (int i = start; i < end; i++) local += i;
    __atomic_fetch_add(&s->sum, local, __ATOMIC_RELAXED);
}

static void test_threadpool_parallel_for_sum(void) {
    ThreadPool* tp = threadpool_create(4);
    if (!tp) {
        FAIL("threadpool_parallel_for_sum", "threadpool_create returned NULL");
        return;
    }

    if (threadpool_num_threads(tp) != 4) {
        FAIL("threadpool_parallel_for_sum", "wrong thread count");
        threadpool_destroy(tp);
        return;
    }

    SumCtx ctx = {0};
    int n = 10000;
    threadpool_parallel_for(tp, 0, n, 128, sum_range, &ctx);

    long long expected = ((long long)n * (long long)(n - 1)) / 2;
    if (ctx.sum != expected) {
        FAIL("threadpool_parallel_for_sum", "sum mismatch");
    } else {
        PASS("threadpool_parallel_for_sum");
    }

    threadpool_destroy(tp);
}

#if defined(USE_PTHREAD) && (USE_PTHREAD == 1)
typedef struct {
    pthread_t main_tid;
    int saw_worker;
} WorkerDetectCtx;

static void detect_worker(int start, int end, void* ctx) {
    (void)start;
    (void)end;
    WorkerDetectCtx* wd = (WorkerDetectCtx*)ctx;

    pthread_t tid = pthread_self();
    if (!pthread_equal(tid, wd->main_tid)) {
        __atomic_store_n(&wd->saw_worker, 1, __ATOMIC_RELAXED);
    }

    /* Add some work so the caller can't finish everything instantly. */
    volatile int sink = 0;
    for (int i = 0; i < 50000; i++) sink += i;
}

static void test_threadpool_uses_multiple_threads(void) {
    ThreadPool* tp = threadpool_create(4);
    if (!tp) {
        FAIL("threadpool_uses_multiple_threads", "threadpool_create returned NULL");
        return;
    }

    WorkerDetectCtx ctx;
    ctx.main_tid = pthread_self();
    ctx.saw_worker = 0;

    threadpool_parallel_for(tp, 0, 256, 1, detect_worker, &ctx);

    if (__atomic_load_n(&ctx.saw_worker, __ATOMIC_RELAXED) == 0) {
        FAIL("threadpool_uses_multiple_threads", "no worker thread observed");
    } else {
        PASS("threadpool_uses_multiple_threads");
    }

    threadpool_destroy(tp);
}
#endif

int main(void) {
    printf("=== ThreadPool Test Suite (scaffolding) ===\n");
    test_threadpool_parallel_for_sum();

#if defined(USE_PTHREAD) && (USE_PTHREAD == 1)
    test_threadpool_uses_multiple_threads();
#else
    printf("  SKIP: threadpool_uses_multiple_threads (USE_PTHREAD=0)\n");
#endif

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
