#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __APPLE__
#include <IOSurface/IOSurface.h>
#endif

typedef struct ANEMemoryRegion ANEMemoryRegion;

ANEMemoryRegion *ANERegisterMemoryRegion(size_t size_bytes);
void             ANEDeregisterMemoryRegion(ANEMemoryRegion *mr);
void            *ANEMemoryRegionGetPtr(ANEMemoryRegion *mr);
#ifdef __APPLE__
IOSurfaceRef     ANEMemoryRegionGetSurface(ANEMemoryRegion *mr);
#endif
