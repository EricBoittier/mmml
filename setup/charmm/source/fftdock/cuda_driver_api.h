/*
 * cuda_driver_api.h — Runtime loader for the CUDA driver API.
 *
 * Instead of linking directly against libcuda.so (CUDA::cuda_driver),
 * this header loads the driver library via dlopen at first use.  This
 * allows the same binary to run on:
 *
 *   - GPU compute nodes (driver present  → full CUDA acceleration)
 *   - Head / login nodes (driver absent  → graceful error message)
 *
 * Usage:
 *   #include "cuda_driver_api.h"
 *
 *   if (cuda_driver_load() != 0) {
 *       // driver not available — fall back or abort
 *   }
 *   // Now call cuLaunchKernel, cuModuleLoadData, etc. as usual.
 *
 * The header redefines each driver function name as a macro that
 * expands to a call through the loaded function pointer, so existing
 * call sites require no source changes beyond the initial load check.
 *
 * Thread safety: cuda_driver_load() uses a simple flag.  Call it once
 * from the main thread before any parallel work.
 */

#ifndef CUDA_DRIVER_API_H
#define CUDA_DRIVER_API_H

#include <cuda.h>   /* CUresult, CUmodule, CUfunction, etc. */
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Function-pointer types — one per driver API entry point we need.   */
/* ------------------------------------------------------------------ */

typedef CUresult (*pfn_cuCtxGetDevice)(CUdevice *);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*pfn_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuModuleUnload)(CUmodule);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned,
                                       unsigned, CUstream, void **, void **);
typedef CUresult (*pfn_cuGetErrorString)(CUresult, const char **);
typedef CUresult (*pfn_cuInit)(unsigned);
typedef CUresult (*pfn_cuDeviceTotalMem)(size_t *, CUdevice);

/* ------------------------------------------------------------------ */
/* Global function pointers — set by cuda_driver_load().              */
/* ------------------------------------------------------------------ */

static pfn_cuCtxGetDevice        _cu_CtxGetDevice;
static pfn_cuDeviceGet           _cu_DeviceGet;
static pfn_cuDeviceGetAttribute  _cu_DeviceGetAttribute;
static pfn_cuDeviceGetName       _cu_DeviceGetName;
static pfn_cuModuleLoadData      _cu_ModuleLoadData;
static pfn_cuModuleGetFunction   _cu_ModuleGetFunction;
static pfn_cuModuleUnload        _cu_ModuleUnload;
static pfn_cuLaunchKernel        _cu_LaunchKernel;
static pfn_cuGetErrorString      _cu_GetErrorString;
static pfn_cuInit                _cu_Init;
static pfn_cuDeviceTotalMem      _cu_DeviceTotalMem;

/* ------------------------------------------------------------------ */
/* Redirect bare driver calls to our function pointers.               */
/* cuda.h may define versioned macros (e.g. cuDeviceTotalMem ->       */
/* cuDeviceTotalMem_v2).  We undef those first so our redirects win.  */
/* ------------------------------------------------------------------ */

#undef cuCtxGetDevice
#undef cuDeviceGet
#undef cuDeviceGetAttribute
#undef cuDeviceGetName
#undef cuModuleLoadData
#undef cuModuleGetFunction
#undef cuModuleUnload
#undef cuLaunchKernel
#undef cuGetErrorString
#undef cuInit
#undef cuDeviceTotalMem

#define cuCtxGetDevice       _cu_CtxGetDevice
#define cuDeviceGet          _cu_DeviceGet
#define cuDeviceGetAttribute _cu_DeviceGetAttribute
#define cuDeviceGetName      _cu_DeviceGetName
#define cuModuleLoadData     _cu_ModuleLoadData
#define cuModuleGetFunction  _cu_ModuleGetFunction
#define cuModuleUnload       _cu_ModuleUnload
#define cuLaunchKernel       _cu_LaunchKernel
#define cuGetErrorString     _cu_GetErrorString
#define cuInit               _cu_Init
#define cuDeviceTotalMem     _cu_DeviceTotalMem

/* ------------------------------------------------------------------ */
/* Loader implementation (header-only for simplicity).                */
/* ------------------------------------------------------------------ */

#include <dlfcn.h>

static void *_cuda_driver_handle = NULL;
static int   _cuda_driver_loaded = 0;

/*
 * Load a single symbol from the driver library.
 * Returns 0 on success, -1 on failure (with a message to stderr).
 */
static int _cuda_load_sym(void **dest, const char *name) {
    *dest = dlsym(_cuda_driver_handle, name);
    if (!*dest) {
        fprintf(stderr, "FFTDOCK CUDA> Could not find symbol '%s' "
                        "in libcuda.so.1: %s\n", name, dlerror());
        return -1;
    }
    return 0;
}

/*
 * Attempt to dlopen libcuda.so.1 and resolve all needed symbols.
 *
 * Returns  0  on success (all symbols loaded).
 * Returns -1  if the driver library is not available or a symbol
 *             is missing.  A diagnostic is printed to stderr.
 *
 * Safe to call multiple times — subsequent calls are no-ops.
 */
static int cuda_driver_load(void) {
    int rc = 0;

    if (_cuda_driver_loaded)
        return _cuda_driver_handle ? 0 : -1;
    _cuda_driver_loaded = 1;

    _cuda_driver_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!_cuda_driver_handle) {
        fprintf(stderr,
            "\n"
            "FFTDOCK CUDA> Cannot load the NVIDIA GPU driver (libcuda.so.1).\n"
            "FFTDOCK CUDA> GPU acceleration is not available on this machine.\n"
            "FFTDOCK CUDA>\n"
            "FFTDOCK CUDA> If this is a cluster head/login node, this is expected.\n"
            "FFTDOCK CUDA> Submit your job to a GPU compute node, or configure\n"
            "FFTDOCK CUDA> CHARMM with --without-cuda to suppress this message.\n"
            "\n");
        return -1;
    }

    rc |= _cuda_load_sym((void **)&_cu_CtxGetDevice,       "cuCtxGetDevice");
    rc |= _cuda_load_sym((void **)&_cu_DeviceGet,          "cuDeviceGet");
    rc |= _cuda_load_sym((void **)&_cu_DeviceGetAttribute, "cuDeviceGetAttribute");
    rc |= _cuda_load_sym((void **)&_cu_DeviceGetName,      "cuDeviceGetName");
    rc |= _cuda_load_sym((void **)&_cu_ModuleLoadData,     "cuModuleLoadData");
    rc |= _cuda_load_sym((void **)&_cu_ModuleGetFunction,  "cuModuleGetFunction");
    rc |= _cuda_load_sym((void **)&_cu_ModuleUnload,       "cuModuleUnload");
    rc |= _cuda_load_sym((void **)&_cu_LaunchKernel,       "cuLaunchKernel");
    rc |= _cuda_load_sym((void **)&_cu_GetErrorString,    "cuGetErrorString");
    rc |= _cuda_load_sym((void **)&_cu_Init,             "cuInit");
    rc |= _cuda_load_sym((void **)&_cu_DeviceTotalMem,   "cuDeviceTotalMem");

    if (rc != 0) {
        dlclose(_cuda_driver_handle);
        _cuda_driver_handle = NULL;
    }

    return rc;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_DRIVER_API_H */
