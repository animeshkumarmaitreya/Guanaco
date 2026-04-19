#include "quant.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * dump_qblock.c
 * Developer tool to load a raw block from STDIN and compute a test dot product.
 * Usage: cat raw_block.bin | ./dump_qblock
 */
int main(void) {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 256;
  W.dtype = DTYPE_Q4_K;

  // Q4_K block is 144 bytes
  char buf[144];
  if (fread(buf, 1, 144, stdin) != 144) {
    fprintf(stderr, "Failed to read 144 bytes from stdin\n");
    return 1;
  }
  W.data = buf;

  float x[256];
  for (int i = 0; i < 256; i++)
    x[i] = 1.0f;

  float y[1] = {0};
  matvec_q4k_f32(&W, x, y);

  printf("Block compute success. Dot product with 1s: %f\n", y[0]);
  return 0;
}
