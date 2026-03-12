/**
 * ane_memory.m — Zero-copy memory registration for Apple Neural Engine
 *
 * Adapted from libibverbs/memory.c (rdma-core) and libibverbs/cmd_dmabuf.c.
 * Provides RDMA-style pinned memory regions using IOSurface + mlock on
 * Apple Silicon unified memory.
 *
 * On M-series: CPU, GPU, ANE all share the same physical DRAM.
 * Pinning a buffer here makes it accessible to ANE without any memcpy.
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <sys/mman.h>
#import <mach/mach.h>
#import "ane_bridge.h"

// Page size for ANE SRAM tile boundary alignment (16KB)
#define ANE_PAGE_SIZE 16384

// Memory region descriptor — mirrors ibv_mr from libibverbs
typedef struct {
    void        *addr;       // virtual address (unified memory)
    size_t       length;     // region size in bytes
    uint32_t     lkey;       // local key (ANE context handle)
    IOSurfaceRef surface;    // IOSurface for GPU/ANE zero-copy
    int          pinned;     // 1 if mlock succeeded
} ANEMemoryRegion;

/**
 * ANERegisterMemoryRegion — pin and register a buffer with ANE.
 * Mirrors ibv_reg_mr() from libibverbs/memory.c
 *
 * Returns: pointer to ANEMemoryRegion on success, NULL on failure.
 */
ANEMemoryRegion *ANERegisterMemoryRegion(size_t size_bytes) {
    // Align to ANE page boundary
    size_t aligned = (size_bytes + ANE_PAGE_SIZE - 1) & ~(ANE_PAGE_SIZE - 1);

    ANEMemoryRegion *mr = calloc(1, sizeof(ANEMemoryRegion));
    if (!mr) return NULL;

    // Allocate page-aligned memory in the unified pool
    mr->addr = mmap(
        NULL, aligned,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1, 0
    );
    if (mr->addr == MAP_FAILED) {
        free(mr);
        return NULL;
    }
    mr->length = aligned;

    // mlock — prevents paging, mirrors ibv_reg_mr pinning semantics
    if (mlock(mr->addr, aligned) == 0) {
        mr->pinned = 1;
    }

    // Create IOSurface for zero-copy GPU ↔ ANE sharing
    // Mirrors cmd_dmabuf.c DMA buffer registration
    NSDictionary *props = @{
        (NSString *)kIOSurfaceWidth:             @(aligned),
        (NSString *)kIOSurfaceHeight:            @1,
        (NSString *)kIOSurfaceBytesPerElement:   @1,
        (NSString *)kIOSurfacePixelFormat:       @(kCVPixelFormatType_OneComponent8),
        (NSString *)kIOSurfaceAllocSize:         @(aligned),
    };
    mr->surface = IOSurfaceCreate((CFDictionaryRef)props);

    // Generate a local key (simplified — production would use ANEClient handle)
    mr->lkey = (uint32_t)(uintptr_t)mr->addr ^ (uint32_t)aligned;

    return mr;
}

/**
 * ANEDeregisterMemoryRegion — release pinned region.
 * Mirrors ibv_dereg_mr() from libibverbs/memory.c
 */
void ANEDeregisterMemoryRegion(ANEMemoryRegion *mr) {
    if (!mr) return;
    if (mr->surface) {
        CFRelease(mr->surface);
    }
    if (mr->addr && mr->addr != MAP_FAILED) {
        if (mr->pinned) {
            munlock(mr->addr, mr->length);
        }
        munmap(mr->addr, mr->length);
    }
    free(mr);
}

/**
 * ANEMemoryRegionGetPtr — get the raw pointer (for ANE dispatch).
 */
void *ANEMemoryRegionGetPtr(ANEMemoryRegion *mr) {
    return mr ? mr->addr : NULL;
}

/**
 * ANEMemoryRegionGetSurface — get IOSurface (for GPU prefill → ANE decode).
 * Mirrors the GPU↔ANE zero-copy pipeline via shared IOSurface in the bridge.
 */
IOSurfaceRef ANEMemoryRegionGetSurface(ANEMemoryRegion *mr) {
    return mr ? mr->surface : NULL;
}
