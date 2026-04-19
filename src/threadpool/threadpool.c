#include "threadpool.h"

#include <stdlib.h>

#if defined(USE_PTHREAD) && (USE_PTHREAD == 1)
#include <pthread.h>
#include <string.h>

struct ThreadPool {
    int n_threads; /* total threads including caller */
    int n_workers; /* background worker threads */

    pthread_t* workers;

    pthread_mutex_t mu;
    pthread_cond_t  cv_job;
    pthread_cond_t  cv_done;

    int stop;

    /* Job state (protected by mu unless noted) */
    int job_seq;
    int job_active;

    int start;
    int end;
    int grain;
    threadpool_for_fn fn;
    void* ctx;

    int participants_remaining;

    /* Work distribution for current job (atomic via __atomic builtins) */
    int next;
};

static void threadpool_do_work(ThreadPool* tp,
                              int start,
                              int end,
                              int grain,
                              threadpool_for_fn fn,
                              void* ctx) {
    if (!fn) return;
    if (end <= start) return;
    if (grain < 1) grain = 1;

    for (;;) {
        int s = __atomic_fetch_add(&tp->next, grain, __ATOMIC_RELAXED);
        if (s >= end) break;
        int e = s + grain;
        if (e > end) e = end;
        fn(s, e, ctx);
    }
}

static void* worker_main(void* arg) {
    ThreadPool* tp = (ThreadPool*)arg;
    int last_seq = 0;

    pthread_mutex_lock(&tp->mu);
    for (;;) {
        while (!tp->stop && tp->job_seq == last_seq) {
            pthread_cond_wait(&tp->cv_job, &tp->mu);
        }
        if (tp->stop) {
            pthread_mutex_unlock(&tp->mu);
            return NULL;
        }

        last_seq = tp->job_seq;

        /* Snapshot job fields under lock, then work without holding the mutex. */
        int start = tp->start;
        int end = tp->end;
        int grain = tp->grain;
        threadpool_for_fn fn = tp->fn;
        void* ctx = tp->ctx;

        pthread_mutex_unlock(&tp->mu);
        threadpool_do_work(tp, start, end, grain, fn, ctx);
        pthread_mutex_lock(&tp->mu);

        tp->participants_remaining--;
        if (tp->participants_remaining == 0) {
            tp->job_active = 0;
            pthread_cond_broadcast(&tp->cv_done);
        }
    }
}

ThreadPool* threadpool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;

    ThreadPool* tp = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!tp) return NULL;

    tp->n_threads = n_threads;
    tp->n_workers = (n_threads > 1) ? (n_threads - 1) : 0;
    tp->workers = NULL;
    tp->stop = 0;
    tp->job_seq = 0;
    tp->job_active = 0;
    tp->participants_remaining = 0;
    tp->next = 0;

    if (pthread_mutex_init(&tp->mu, NULL) != 0) {
        free(tp);
        return NULL;
    }
    if (pthread_cond_init(&tp->cv_job, NULL) != 0) {
        pthread_mutex_destroy(&tp->mu);
        free(tp);
        return NULL;
    }
    if (pthread_cond_init(&tp->cv_done, NULL) != 0) {
        pthread_cond_destroy(&tp->cv_job);
        pthread_mutex_destroy(&tp->mu);
        free(tp);
        return NULL;
    }

    if (tp->n_workers > 0) {
        tp->workers = (pthread_t*)calloc((size_t)tp->n_workers, sizeof(pthread_t));
        if (!tp->workers) {
            pthread_cond_destroy(&tp->cv_done);
            pthread_cond_destroy(&tp->cv_job);
            pthread_mutex_destroy(&tp->mu);
            free(tp);
            return NULL;
        }

        for (int i = 0; i < tp->n_workers; i++) {
            if (pthread_create(&tp->workers[i], NULL, worker_main, tp) != 0) {
                /* Best-effort shutdown of already-created workers. */
                pthread_mutex_lock(&tp->mu);
                tp->stop = 1;
                tp->job_seq++;
                pthread_cond_broadcast(&tp->cv_job);
                pthread_mutex_unlock(&tp->mu);

                for (int j = 0; j < i; j++) {
                    pthread_join(tp->workers[j], NULL);
                }
                free(tp->workers);
                pthread_cond_destroy(&tp->cv_done);
                pthread_cond_destroy(&tp->cv_job);
                pthread_mutex_destroy(&tp->mu);
                free(tp);
                return NULL;
            }
        }
    }

    return tp;
}

void threadpool_destroy(ThreadPool* tp) {
    if (!tp) return;

    pthread_mutex_lock(&tp->mu);
    tp->stop = 1;
    tp->job_seq++;
    pthread_cond_broadcast(&tp->cv_job);
    pthread_mutex_unlock(&tp->mu);

    for (int i = 0; i < tp->n_workers; i++) {
        pthread_join(tp->workers[i], NULL);
    }

    free(tp->workers);
    pthread_cond_destroy(&tp->cv_done);
    pthread_cond_destroy(&tp->cv_job);
    pthread_mutex_destroy(&tp->mu);
    free(tp);
}

void threadpool_parallel_for(ThreadPool* tp,
                             int start,
                             int end,
                             int grain,
                             threadpool_for_fn fn,
                             void* ctx) {
    if (!tp || !fn) return;
    if (end <= start) return;
    if (grain < 1) grain = 1;

    if (tp->n_threads <= 1 || tp->n_workers == 0 || (end - start) <= grain) {
        fn(start, end, ctx);
        return;
    }

    pthread_mutex_lock(&tp->mu);
    tp->start = start;
    tp->end = end;
    tp->grain = grain;
    tp->fn = fn;
    tp->ctx = ctx;
    __atomic_store_n(&tp->next, start, __ATOMIC_RELAXED);

    tp->participants_remaining = tp->n_threads;
    tp->job_active = 1;
    tp->job_seq++;
    pthread_cond_broadcast(&tp->cv_job);
    pthread_mutex_unlock(&tp->mu);

    /* Caller participates as one of the threads. */
    threadpool_do_work(tp, start, end, grain, fn, ctx);

    pthread_mutex_lock(&tp->mu);
    tp->participants_remaining--;
    if (tp->participants_remaining == 0) {
        tp->job_active = 0;
        pthread_cond_broadcast(&tp->cv_done);
    }
    while (tp->job_active) {
        pthread_cond_wait(&tp->cv_done, &tp->mu);
    }
    pthread_mutex_unlock(&tp->mu);
}

int threadpool_num_threads(const ThreadPool* tp) {
    return tp ? tp->n_threads : 1;
}

#else

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

#endif
