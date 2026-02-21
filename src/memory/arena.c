/*============================================================================
 * Arena Allocator — Person C
 *
 * Bump allocator backed by a single mmap'd region.
 * Allocations increment a pointer. Individual frees are impossible.
 * arena_reset() rewinds the pointer to the start (O(1)).
 * arena_destroy() unmaps the backing memory.
 *
 * All returned pointers are 64-byte aligned (AVX-512 / cache-line requirement).
 *============================================================================*/

#include "memory.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* ---------- Internal struct ---------- */

struct Arena {
    uint8_t* base;       /* start of mmap'd region                 */
    size_t   offset;     /* current allocation watermark (bytes)   */
    size_t   capacity;   /* user-requested capacity (logical limit)*/
    size_t   mmap_size;  /* actual mmap'd size (>= capacity)       */
};

/* ---------- Helpers ---------- */

/* Round `n` up to the next multiple of `align`. `align` must be a power of 2. */
static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

/* ---------- Public API ---------- */

Arena* arena_create(size_t size) {
    if (size == 0) return NULL;

    Arena* a = (Arena*)malloc(sizeof(Arena));
    if (!a) return NULL;

    /* Round capacity up to page size to satisfy mmap. */
    size_t page = 4096;
    size_t cap  = align_up(size, page);

    void* mem = mmap(NULL, cap,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    if (mem == MAP_FAILED) {
        free(a);
        return NULL;
    }

    a->base      = (uint8_t*)mem;
    a->offset    = 0;
    a->capacity  = size;   /* logical limit = what the caller asked for */
    a->mmap_size = cap;    /* physical backing may be larger            */
    return a;
}

void* arena_alloc(Arena* arena, size_t size, size_t alignment) {
    if (!arena || size == 0 || alignment == 0) return NULL;

    /* Align current offset upward. */
    size_t aligned_off = align_up(arena->offset, alignment);

    if (aligned_off + size > arena->capacity) {
        return NULL;   /* out of space */
    }

    void* ptr = arena->base + aligned_off;
    arena->offset = aligned_off + size;
    return ptr;
}

void arena_reset(Arena* arena) {
    if (arena) {
        arena->offset = 0;
    }
}

void arena_destroy(Arena* arena) {
    if (arena) {
        if (arena->base) {
            munmap(arena->base, arena->mmap_size);
        }
        free(arena);
    }
}
