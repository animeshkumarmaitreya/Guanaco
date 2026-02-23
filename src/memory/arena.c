/*============================================================================
 * Arena Allocator — Person B's real implementation
 *
 * Bump allocator backed by mmap. All returned pointers are aligned to the
 * requested alignment (minimum 64 bytes for AVX/cache-line friendliness).
 *
 * Lifetime: session-level (KV cache, weight storage).
 * Never calls malloc/free on the hot path — one mmap at create, one
 * munmap at destroy. arena_alloc is a pointer bump (~1 ns).
 *============================================================================*/

#include "memory.h"
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* macOS uses MAP_ANON; Linux uses MAP_ANONYMOUS */
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

struct Arena {
    uint8_t* base;       /* start of mmap'd region */
    size_t   offset;     /* current allocation offset (bytes from base) */
    size_t   capacity;   /* user-requested capacity (for overflow checks) */
    size_t   mmap_size;  /* actual mmap'd size (page-aligned, >= capacity) */
};

/* Round up x to the next multiple of align (align must be power of 2) */
static inline size_t align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

Arena* arena_create(size_t size) {
    /* The Arena header lives at the start of the mmap'd region.
     * capacity = header + user-requested size so the user gets the full
     * amount they asked for.  mmap_size is page-aligned (>= capacity). */
    size_t header   = align_up(sizeof(Arena), 64);
    size_t capacity = header + size;

    size_t page_size = 4096;
    size_t mmap_size = align_up(capacity, page_size);

    void* mem = mmap(NULL, mmap_size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (mem == MAP_FAILED) {
        fprintf(stderr, "[arena] mmap failed for %zu bytes\n", mmap_size);
        return NULL;
    }

    Arena* a = (Arena*)mem;
    a->base      = (uint8_t*)mem;
    a->capacity  = capacity;   /* header + user-requested size */
    a->mmap_size = mmap_size;  /* actual mmap'd size (page-aligned) */
    a->offset    = header;

    return a;
}

void* arena_alloc(Arena* arena, size_t size, size_t alignment) {
    if (arena == NULL || size == 0) return NULL;
    if (alignment < 64) alignment = 64;  /* minimum 64-byte alignment */

    size_t aligned_offset = align_up(arena->offset, alignment);
    if (aligned_offset + size > arena->capacity) return NULL;

    void* ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

void arena_reset(Arena* arena) {
    if (arena == NULL) return;
    arena->offset = align_up(sizeof(Arena), 64);
}

void arena_destroy(Arena* arena) {
    if (arena == NULL) return;
    size_t mmap_sz = arena->mmap_size;
    void*  base    = arena->base;
    munmap(base, mmap_sz);
}
