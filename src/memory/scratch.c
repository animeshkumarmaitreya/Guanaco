/*============================================================================
 * Scratch Allocator — Person B's real implementation
 *
 * Same bump-allocator mechanics as Arena, but semantically different:
 *   - Arena:   lives for the entire session
 *   - Scratch: reset at the start of every forward() call
 *
 * The engine calls scratch_reset() at the top of each forward pass,
 * then allocates ~20+ buffers per layer (Q, K, V, scores, MLP intermediates).
 *
 * Backed by mmap. Peak-usage tracking for diagnostics.
 *============================================================================*/

#include "memory.h"
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

struct Scratch {
    uint8_t* base;       /* start of mmap'd region */
    size_t   offset;     /* current allocation offset */
    size_t   capacity;   /* usable capacity (header + user-requested size) */
    size_t   mmap_size;  /* actual mmap'd size (page-aligned, >= capacity) */
    size_t   peak;       /* high water mark (debug) */
};

static inline size_t scratch_align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

Scratch* scratch_create(size_t size) {
    size_t header    = scratch_align_up(sizeof(Scratch), 64);
    size_t capacity  = header + size;
    size_t page_size = 4096;
    size_t mmap_size = scratch_align_up(capacity, page_size);

    void* mem = mmap(NULL, mmap_size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (mem == MAP_FAILED) {
        fprintf(stderr, "[scratch] mmap failed for %zu bytes\n", mmap_size);
        return NULL;
    }

    Scratch* s  = (Scratch*)mem;
    s->base     = (uint8_t*)mem;
    s->capacity = capacity;   /* header + user-requested size */
    s->mmap_size = mmap_size; /* actual mmap'd size (page-aligned) */
    s->offset   = header;
    s->peak     = s->offset;

    return s;
}

void* scratch_alloc(Scratch* scr, size_t size, size_t alignment) {
    if (scr == NULL || size == 0) return NULL;
    if (alignment < 64) alignment = 64;

    /* Align the absolute pointer address, not just the offset */
    uintptr_t current_addr = (uintptr_t)scr->base + scr->offset;
    uintptr_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
    size_t aligned_offset  = (size_t)(aligned_addr - (uintptr_t)scr->base);

    if (aligned_offset + size > scr->capacity) return NULL;

    void* ptr = (void*)aligned_addr;
    scr->offset = aligned_offset + size;

    if (scr->offset > scr->peak) scr->peak = scr->offset;

    return ptr;
}

void scratch_reset(Scratch* scr) {
    if (scr == NULL) return;
    scr->offset = scratch_align_up(sizeof(Scratch), 64);
}

void scratch_destroy(Scratch* scr) {
    if (scr == NULL) return;
    size_t sz   = scr->mmap_size;
    void*  base = scr->base;
    munmap(base, sz);
}
