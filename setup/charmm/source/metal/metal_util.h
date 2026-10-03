/*
 * metal_util.h
 *
 * Shared Metal infrastructure for CHARMM.  Modeled on
 * source/opencl/ocl_util.h.  Provides:
 *   • A single lazily-initialised MTLDevice / MTLCommandQueue pair
 *   • A compiled-kernel pipeline-state cache
 *   • MTLBuffer allocation / copy / release helpers (MTLStorageModeShared)
 *   • Batched 3-D R2C / C2R FFT plan helpers (MetalFFTPlan struct,
 *     fft3d_r2c_batch / fft3d_c2r_batch) — currently dormant; the
 *     active FFTDOCK pipeline (metal_fftdock_gpu.mm) calls MPSGraph
 *     directly.  Retained for future Metal modules that may need
 *     a self-contained FFT helper.
 *   • C entry points (mtl_device_*, mtl_begin_session, mtl_end_session)
 *     that source/metal/metal_main.F90 binds to.
 *
 * Active consumers (all in source/fftdock/):
 *   metal_fftdock_gpu.mm   — full per-batch docking pipeline
 *   metal_grid_pot.mm      — receptor potential grid generation
 *   metal_grid_lig.mm      — ligand rotamer grid generation
 *
 * Design notes
 * ─────────────
 * • All MTLBuffers used for GPU memory are allocated with
 *   MTLStorageModeShared so the same physical memory is visible to both
 *   CPU and GPU on Apple Silicon (unified memory) — no explicit copy.
 * • The active FFTDOCK pipeline uses MPSGraph FFT (macOS 14+) with the
 *   resultsDictionary: encode form, so MPSGraph writes directly into
 *   our pre-allocated shared buffers (no MPSNDArray.resource extraction
 *   — that property is a private API that doesn't exist on every macOS).
 * • The vDSP CPU fallback (in this file's MetalFFTPlan struct + helpers)
 *   is plumbed for older macOS but is currently dormant — the FFTDOCK
 *   path requires macOS 14+.
 *
 * History: was source/fftdock/metal_common.h, moved here 2026-04.  - YWu
 */

#ifndef METAL_UTIL_H
#define METAL_UTIL_H

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Global Metal state                                                  */
/* ------------------------------------------------------------------ */

/**
 * Initialises (lazily) and returns a pointer to the global Metal state.
 * Call once from the main thread before any parallel GPU work.
 * Returns 0 on success, -1 if no suitable Metal device is available.
 */
int metal_state_init(int device_index);

/** Returns the shared MTLDevice (Objective-C object, opaque to C callers). */
void* metal_get_device(void);

/** Returns the shared MTLCommandQueue. */
void* metal_get_queue(void);

/* ------------------------------------------------------------------ */
/*  Device enumeration / selection — C entry points for metal_main.F90 */
/*                                                                     */
/*  These are the Metal analogues of ocl_device_* in source/opencl/    */
/*  ocl_util.h.  Fortran callers go through module metal_main_mod      */
/*  in source/metal/metal_main.F90.                                    */
/* ------------------------------------------------------------------ */

/**
 * Populate an opaque list of every available Metal device.
 * @param  devices_out  receives a retained NSArray<id<MTLDevice>>*
 *                       cast to void*  (caller owns; release by passing
 *                       to mtl_device_list_release).
 * @return 0 on success, -1 if no devices are present.
 */
int  mtl_device_init(void** devices_out);

/** Release a list returned by mtl_device_init. */
void mtl_device_list_release(void** devices);

/** Print a one-line summary of every device in the list. */
void mtl_device_print(void* devices);

/** Print a one-line summary of a single device (id<MTLDevice> as void*). */
void mtl_device_print_one(void* dev);

/**
 * Fill `c_string` (up to `max_size-1` chars + NUL) with the device
 * identification string:   "<id> <name> <bytes>".
 */
void mtl_device_string(void* dev, char* c_string, int max_size);

/**
 * Retrieve the dev_id-th device from a list produced by mtl_device_init.
 * dev_id is 1-based to match the OpenCL convention on the Fortran side.
 *
 * @return 0 on success, -1 on out-of-range.
 */
int  mtl_device_get(void* devices, int dev_id, void** out_device);

/**
 * Pick the device with the largest recommendedMaxWorkingSetSize from the
 * list, typically the discrete / highest-memory GPU on a multi-GPU Mac.
 *
 * @return 0 on success, -1 if the list is empty.
 */
int  mtl_device_max_mem_get(void* devices, void** out_device);

/**
 * Start a Metal "session": pin the selected device as the module-wide
 * s_device, lazily create s_queue, and load the default kernel library
 * if present beside the executable.
 *
 * No separate context/queue handles are returned — unlike OpenCL, Metal
 * treats the device as the primary state object.  The pair (device,
 * queue) is accessed later via metal_get_device() / metal_get_queue().
 *
 * @return 0 on success, -1 on failure.
 */
int  mtl_begin_session(void* in_dev);

/**
 * End the session: release the command queue and kernel library.  The
 * device object itself is not released (it is an autoreleased singleton
 * owned by the OS).
 *
 * @return 0 always.
 */
int  mtl_end_session(void);

/**
 * Returns a compiled MTLComputePipelineState for the named kernel.
 * The pipeline is cached after the first call.
 * Exits on failure.
 */
void* metal_get_pipeline(const char* kernel_name);

/* ------------------------------------------------------------------ */
/*  Buffer helpers                                                      */
/* ------------------------------------------------------------------ */

/**
 * Allocate a Metal buffer of `bytes` bytes in MTLStorageModeShared.
 * Returns an opaque handle (retained MTLBuffer*).
 * Caller must eventually call metal_buffer_release().
 */
void* metal_buffer_alloc(size_t bytes);

/** Zero-fill a Metal buffer (CPU-side memset on its shared contents). */
void  metal_buffer_zero(void* buf, size_t bytes);

/** Copy `bytes` bytes from CPU pointer `src` into the Metal buffer. */
void  metal_buffer_copy_to(void* buf, const void* src, size_t bytes);

/** Copy `bytes` bytes from the Metal buffer out to CPU pointer `dst`. */
void  metal_buffer_copy_from(void* buf, void* dst, size_t bytes);

/** Return a direct CPU pointer to the Metal buffer's shared contents. */
void* metal_buffer_contents(void* buf);

/** Release a Metal buffer previously returned by metal_buffer_alloc(). */
void  metal_buffer_release(void* buf);

/* ------------------------------------------------------------------ */
/*  Command submission helpers                                          */
/* ------------------------------------------------------------------ */

/**
 * Encode a 1-D compute dispatch into a new command buffer, commit it,
 * and wait for completion.
 *
 *  pipeline   — MTLComputePipelineState* from metal_get_pipeline()
 *  bufs[]     — array of `nbuf` MTLBuffer* handles (in buffer-index order)
 *  offsets[]  — byte offset into each buffer (may be NULL for all-zero)
 *  nbuf       — number of buffers
 *  n_threads  — total number of threads to launch (1-D)
 *  tgroup     — threads per thread-group (e.g. 256 or 512)
 *
 * inline_data / inline_bytes — optional inline constant block appended
 *   after the buffer bindings (use for small scalar parameters if desired;
 *   pass NULL/0 to skip).
 */
void metal_dispatch_1d(void*   pipeline,
                       void**  bufs,
                       size_t* offsets,
                       int     nbuf,
                       size_t  n_threads,
                       size_t  tgroup);

/**
 * Block until all pending work on the shared command queue has completed.
 * Must be called before the CPU reads from a shared Metal buffer, and
 * before a vDSP call that writes into one.
 */
void metal_sync(void);

/* ------------------------------------------------------------------ */
/*  3-D FFT plan (vDSP-based, arbitrary-size)                          */
/* ------------------------------------------------------------------ */

/**
 * Opaque handle for a batched 3-D R2C or C2R FFT plan.
 *
 * Two back-ends are supported:
 *
 *   use_mps == 1  (macOS 14+, preferred)
 *     Uses MPSGraph.fastFourierTransform — a true Metal GPU FFT from Apple's
 *     MetalPerformanceShadersGraph framework.  No size constraint; any grid
 *     dimension works.  mpsg_graph/mpsg_input/mpsg_output hold retained ObjC
 *     objects (MPSGraph*, MPSGraphTensor*, MPSGraphTensor*).
 *
 *   use_mps == 0  (macOS 12/13 fallback)
 *     Uses three vDSP_DFT_Setup objects (one per spatial axis) operating
 *     on the CPU via Apple's Accelerate framework.  Grid dimensions must
 *     satisfy 2^a × 3^b × 5^c with a ≥ 2.
 */
typedef struct MetalFFTPlan MetalFFTPlan;

struct MetalFFTPlan {
    int xdim, ydim, zdim;
    int batch_size;
    int isR2C;      /* 1 = R2C forward, 0 = C2R inverse */
    int use_mps;    /* 1 = MPSGraph GPU FFT  |  0 = vDSP CPU FFT */

    /* MPSGraph GPU FFT (macOS 14+, MetalPerformanceShadersGraph).
     * All four are retained ObjC objects stored as void* for C compatibility.
     *   mpsg_graph  — MPSGraph*
     *   mpsg_input  — MPSGraphTensor* (dynamic-shape real/complex placeholder)
     *   mpsg_output — MPSGraphTensor* (FFT output tensor)
     *   mpsg_exec   — MPSGraphExecutable* (pre-compiled graph for execution
     *                 with caller-supplied output buffers; avoids GPU-private
     *                 allocation and the SIGBUS that follows CPU readback)
     * The graph is compiled once at plan creation and reused for all
     * subsequent calls; the input placeholder shape is resolved at runtime
     * from the MPSGraphTensorData fed at each execution.               */
    void* mpsg_graph;
    void* mpsg_input;
    void* mpsg_output;
    void* mpsg_exec;    /* retained MPSGraphExecutable* */

    /* vDSP fallback (macOS 12/13): 1-D DFT setups for each axis */
    vDSP_DFT_Setup dft_z;   /* size zdim — R2C (forward) or C2R (inverse) */
    vDSP_DFT_Setup dft_y;   /* size ydim — C2C */
    vDSP_DFT_Setup dft_x;   /* size xdim — C2C */
};

/**
 * Execute a batched 3-D R2C FFT.
 *
 *  plan       — MetalFFTPlan* created with isR2C = 1
 *  in_real    — interleaved real input  [batch × Nx × Ny × Nz]
 *               (imaginary parts are zero; only the real array is read)
 *  out_cplx   — interleaved complex output [batch × Nx × Ny × (Nz/2+1)]
 *               layout: [re0, im0, re1, im1, …]
 *  batch      — number of independent transforms
 */
void fft3d_r2c_batch(const MetalFFTPlan* plan,
                     const float*        in_real,
                     float*              out_cplx,
                     int                 batch);

/**
 * Execute a batched 3-D C2R inverse FFT.
 *
 *  plan       — MetalFFTPlan* created with isR2C = 0
 *  in_cplx    — interleaved complex input  [batch × Nx × Ny × (Nz/2+1)]
 *  out_real   — real output                [batch × Nx × Ny × Nz]
 *  batch      — number of independent transforms
 *
 * The result is NOT normalised (consistent with cuFFT behaviour); call
 * the correctEnergy kernel after this to divide by idist.
 */
void fft3d_c2r_batch(const MetalFFTPlan* plan,
                     const float*        in_cplx,
                     float*              out_real,
                     int                 batch);

/**
 * Check whether n factors into 2^a × 3^b × 5^c (a ≥ 2).
 * Prints a diagnostic to stderr and returns 0 if not.
 * Only relevant for the vDSP fallback path (use_mps == 0).
 */
int  vdsp_dft_size_ok(int n, const char* dim_name);

/* ------------------------------------------------------------------ */
/*  MPSGraph GPU FFT interface (macOS 14+, MetalPerformanceShadersGraph) */
/*                                                                       */
/*  The FFT is implemented via MPSGraph.fastFourierTransform with        */
/*  MPSGraphFFTDescriptor.  Data is fed at runtime through               */
/*  MPSGraphTensorData (which can wrap an existing MTLBuffer             */
/*  zero-copy on Apple Silicon unified memory).                          */
/* ------------------------------------------------------------------ */

/**
 * Returns 1 if the MPSGraph GPU FFT back-end is available at runtime
 * (macOS 14+), 0 otherwise.
 */
int  mps_fft_available(void);

/**
 * Build the MPSGraph for a forward (R2C) 3-D batched FFT and store the
 * graph + placeholder + output tensor into plan->mpsg_*.
 * Returns 1 on success, 0 on failure (plan unchanged).
 */
int  mps_create_r2c_plan(MetalFFTPlan* plan);

/**
 * Same as mps_create_r2c_plan but for inverse (C2R).
 */
int  mps_create_c2r_plan(MetalFFTPlan* plan);

/**
 * Release the three retained ObjC objects in plan->mpsg_*
 * (graph, input tensor, output tensor).
 * mpsg_exec is always NULL (compileWithDevice: is intentionally skipped).
 * Safe to call even if the fields are NULL.
 */
void mps_release_plan(MetalFFTPlan* plan);

/**
 * Batched 3-D R2C FFT via MPSGraph GPU.
 *
 *  plan      — MetalFFTPlan* with use_mps == 1 and mpsg_* populated
 *  in_real   — real input  [batch × Nx × Ny × Nz],  row-major
 *  out_cplx  — complex output [batch × Nx × Ny × (Nz/2+1)],
 *              interleaved float pairs (re, im)
 *  batch     — number of independent transforms
 */
void mps_fft3d_r2c_batch(const MetalFFTPlan* plan,
                          const float* in_real,
                          float*       out_cplx,
                          int          batch);

/**
 * Batched 3-D C2R inverse FFT via MPSGraph GPU.
 * Result is NOT normalised (correctEnergy kernel divides by Nx×Ny×Nz).
 */
void mps_fft3d_c2r_batch(const MetalFFTPlan* plan,
                          const float* in_cplx,
                          float*       out_real,
                          int          batch);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* METAL_UTIL_H */
