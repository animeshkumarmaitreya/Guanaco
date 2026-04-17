#ifndef LLMRT_THREADPOOL_H
#define LLMRT_THREADPOOL_H

/*============================================================================
 * Tiny threadpool (MUST) — scaffolding
 *
 * Current build remains single-threaded; this API will be used to parallelize
 * GEMM tiles and other hot loops once CPU threading lands.
 *============================================================================*/

typedef struct ThreadPool ThreadPool;

typedef void (*threadpool_for_fn)(int start, int end, void* ctx);

ThreadPool* threadpool_create(int n_threads);
void        threadpool_destroy(ThreadPool* tp);

/* Execute fn over [start, end) in chunks. Implementations may run serially. */
void threadpool_parallel_for(ThreadPool* tp,
                             int start,
                             int end,
                             int grain,
                             threadpool_for_fn fn,
                             void* ctx);

int threadpool_num_threads(const ThreadPool* tp);

#endif /* LLMRT_THREADPOOL_H */
