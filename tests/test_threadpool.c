#include "threadpool.h"

#include <stdio.h>
#include <stdint.h>

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

int main(void) {
    printf("=== ThreadPool Test Suite (scaffolding) ===\n");
    test_threadpool_parallel_for_sum();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
