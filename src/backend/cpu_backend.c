#include "backend_internal.h"

#include "kernels.h"
#include "kernels_ext.h"
#include "quant.h"
#include "threadpool.h"

#include <stdlib.h>

/* Internal hook implemented by src/kernels/cpu/kernels_cpu.c */
extern void kernels_cpu_set_threadpool(ThreadPool* tp);

#include <dirent.h>
#include <string.h>
#include <stdio.h>

int cpu_get_temperature(void) {
    DIR *d = opendir("/sys/class/thermal/");
    int max_temp = 0;
    if (d) {
        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            if (strncmp(dir->d_name, "thermal_zone", 12) == 0) {
                char path[256];
                snprintf(path, sizeof(path), "/sys/class/thermal/%s/temp", dir->d_name);
                FILE *f = fopen(path, "r");
                if (f) {
                    int t = 0;
                    if (fscanf(f, "%d", &t) == 1) {
                        t /= 1000;
                        if (t > max_temp) max_temp = t;
                    }
                    fclose(f);
                }
            }
        }
        closedir(d);
    }
    return max_temp;
}

Backend* backend_cpu_create(const BackendConfig* cfg) {
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    if (!b) return NULL;

    b->cfg = *cfg;
    b->cfg.kind = BACKEND_CPU;

    /* Wire existing CPU kernels directly into the vtable. */
    b->kernels.gemm_f32 = gemm_f32;
    b->kernels.gemm_f32_nn = gemm_f32_nn;
    b->kernels.rmsnorm = cpu_rmsnorm;
    b->kernels.softmax_inplace = softmax_inplace;
    b->kernels.silu_inplace = silu_inplace;
    b->kernels.rope = rope;
    b->kernels.residual_add = residual_add;
    b->kernels.elemwise_mul = elemwise_mul;

    /* Quant decode hooks (Phase B) */
    b->kernels.matvec_q4k_f32 = matvec_q4k_f32;
    b->kernels.matvec_q6k_f32 = matvec_q6k_f32;
    b->kernels.matvec_q8_0_f32 = matvec_q8_0_f32;

    /* CPU threading: create a threadpool and make it available to CPU kernels. */
    ThreadPool* tp = threadpool_create(b->cfg.threads);
    b->impl = tp;
    kernels_cpu_set_threadpool(tp);

    return b;
}

void backend_cpu_destroy(Backend* b) {
    if (!b) return;
    if (b->impl) {
        kernels_cpu_set_threadpool(NULL);
        threadpool_destroy((ThreadPool*)b->impl);
        b->impl = NULL;
    }
    free(b);
}
