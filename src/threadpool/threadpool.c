#include "threadpool.h"

#include <stdlib.h>

struct ThreadPool {
    int n_threads;
};

ThreadPool* threadpool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    ThreadPool* tp = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!tp) return NULL;
    tp->n_threads = n_threads;
    return tp;
}

void threadpool_destroy(ThreadPool* tp) {
    free(tp);
}

void threadpool_parallel_for(ThreadPool* tp,
                             int start,
                             int end,
                             int grain,
                             threadpool_for_fn fn,
                             void* ctx) {
    (void)tp;
    (void)grain;
    if (!fn) return;
    if (end <= start) return;

    /* Stub: serial execution. */
    fn(start, end, ctx);
}

int threadpool_num_threads(const ThreadPool* tp) {
    return tp ? tp->n_threads : 1;
}
