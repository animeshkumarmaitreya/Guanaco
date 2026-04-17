#include "quant.h"

#include <stdio.h>

static int test_quant_stub_present(void) {
    /* Stub test: just verify the symbols link and can be called safely. */
    Tensor t = {0};
    float x[1] = {0.0f};
    float y[1] = {0.0f};

    int rc1 = matvec_q8_0_f32(&t, x, y);
    int rc2 = matvec_q4k_f32(&t, x, y);

    /* Until implemented, stubs must return non-zero to avoid accidental use. */
    if (rc1 == 0) return 0;
    if (rc2 == 0) return 0;
    return 1;
}

int main(void) {
    printf("=== Quantization Test Suite (stub) ===\n");

    int ok = 1;
    if (!test_quant_stub_present()) {
        printf("  FAIL: quant_stub_present\n");
        ok = 0;
    } else {
        printf("  PASS: quant_stub_present\n");
    }

    if (!ok) return 1;
    printf("\nAll tests PASSED ✓\n");
    return 0;
}
