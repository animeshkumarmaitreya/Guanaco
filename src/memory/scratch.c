/*============================================================================
 * Scratch Allocator — Person C
 *
 * Per-forward-pass bump allocator. Semantically identical to the Arena but
 * with a different intended lifetime: scratch_reset() is called once per
 * forward pass so that all activation / intermediate buffers are reclaimed
 * without any per-object free.
 *
 * Backed by a single mmap'd region. All returned pointers are 64-byte
 * aligned.
 *============================================================================*/

#include "memory.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* ---------- Internal struct ---------- */

struct Scratch {
    uint8_t* base;       /* start of mmap'd region              */
    size_t   offset;     /* current allocation watermark        */
    size_t   capacity;   /* user-requested capacity             */
    size_t   mmap_size;  /* actual mmap'd size (>= capacity)    */
};

/* ---------- Helpers ---------- */

static inline size_t scratch_align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

/* ---------- Public API ---------- */

Scratch* scratch_create(size_t size) {
    if (size == 0) return NULL;

    Scratch* s = (Scratch*)malloc(sizeof(Scratch));
    if (!s) return NULL;

    size_t page = 4096;
    size_t cap  = scratch_align_up(size, page);

    void* mem = mmap(NULL, cap,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (mem == MAP_FAILED) {
        free(s);
        return NULL;
    }

    s->base      = (uint8_t*)mem;
    s->offset    = 0;
    s->capacity  = size;
    s->mmap_size = cap;
    return s;
}

void* scratch_alloc(Scratch* scr, size_t size, size_t alignment) {
    if (!scr || size == 0 || alignment == 0) return NULL;

    size_t aligned_off = scratch_align_up(scr->offset, alignment);

    if (aligned_off + size > scr->capacity) {
        return NULL;   /* out of space */
    }

    void* ptr = scr->base + aligned_off;
    scr->offset = aligned_off + size;
    return ptr;
}

void scratch_reset(Scratch* scr) {
    if (scr) {
        scr->offset = 0;
    }
}

void scratch_destroy(Scratch* scr) {
    if (scr) {
        if (scr->base) {
            munmap(scr->base, scr->mmap_size);
        }
        free(scr);
    }
}
