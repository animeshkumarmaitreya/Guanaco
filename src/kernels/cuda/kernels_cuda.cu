#include "kernels_cuda.h"

int cuda_kernels_init(int device_id) {
    (void)device_id;
    return -1;
}

void cuda_kernels_shutdown(void) {
}

int cuda_gemm_f32(const Tensor* A, const Tensor* B, Tensor* C) {
    (void)A;
    (void)B;
    (void)C;
    return -1;
}

int cuda_gemm_f32_nn(const Tensor* A, const Tensor* B, Tensor* C) {
    (void)A;
    (void)B;
    (void)C;
    return -1;
}
