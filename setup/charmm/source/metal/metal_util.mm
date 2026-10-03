/*
 * metal_util.mm
 *
 * Implementation of the shared Metal device / queue / pipeline state plus
 * the batched 3-D FFT helpers (MPSGraph primary, vDSP fallback) declared
 * in metal_util.h.  Parallel in structure to source/opencl/ocl_util.cpp.
 *
 * The Metal library ("metal_kernels.metallib") is loaded at runtime from
 * either the path in the METAL_KERNELS_PATH environment variable or the
 * executable's directory (CMake's `install` step copies it to install/bin
 * alongside the charmm binary).  Build the .metallib via the
 * fftdock_metal_kernels CMake custom target — see CMakeLists.txt:
 *
 *   xcrun -sdk macosx metal -fno-fast-math \
 *         -c source/fftdock/metal_kernels.metal -o metal_kernels.air
 *   xcrun -sdk macosx metallib metal_kernels.air -o metal_kernels.metallib
 *
 * History: was source/fftdock/metal_common.mm, moved here 2026-04.  - YWu
 */

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include "metal_util.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>

/* -----------------------------------------------------------------------
 * GPU FFT via MetalPerformanceShadersGraph (macOS 14+)
 *
 * NOTE: MPSNDArrayFourierTransform (MetalPerformanceShaders) does NOT exist
 * in the MacOSX26.4 SDK.  The correct GPU FFT API is in a separate framework:
 *
 *   Framework  : MetalPerformanceShadersGraph
 *   Method     : [MPSGraph fastFourierTransformWithTensor:axesTensor:descriptor:name:]
 *                  — one method handles R2C, C2C, and C2R:
 *                    Float32  input + inverse=NO  → R2C (real → complex)
 *                    Float32  input + inverse=YES → C2R (complex → real)
 *                    Complex  input + inverse=NO  → C2C forward
 *   Descriptor : MPSGraphFFTDescriptor  — .inverse and .scalingMode
 *                (.real does NOT exist; R2C vs C2R is implied by data type)
 *   Data input : MPSGraphTensorData initWithMTLBuffer:shape:dataType:
 *                (do NOT use initWithDevice:data: — it expects MPSGraphDevice*,
 *                 not id<MTLDevice>, causing -[MTLDevice metalDevice] crash)
 *
 * Available from macOS 14.0 (Sonoma, WWDC 2023).
 * ----------------------------------------------------------------------- */
#if defined(__MAC_14_0)
#  import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

/* ------------------------------------------------------------------ */
/*  Singleton Metal state                                               */
/* ------------------------------------------------------------------ */

static id<MTLDevice>       s_device  = nil;
static id<MTLCommandQueue> s_queue   = nil;
static id<MTLLibrary>      s_library = nil;

/* Simple pipeline-state cache (up to 16 kernels). */
#define MAX_PIPELINES 16
static struct {
    const char*                 name;
    id<MTLComputePipelineState> pso;
} s_pipeline_cache[MAX_PIPELINES];
static int s_num_pipelines = 0;

/* ------------------------------------------------------------------ */

int metal_state_init(int device_index)
{
    if (s_device != nil)
        return 0;   /* already initialised */

    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (!devices || devices.count == 0) {
            fprintf(stderr, "FFTDOCK Metal> No Metal devices found.\n");
            return -1;
        }

        if (device_index < 0 || device_index >= (int)devices.count) {
            fprintf(stderr, "FFTDOCK Metal> device_index %d out of range "
                            "(0-%lu); using device 0.\n",
                    device_index, (unsigned long)devices.count - 1);
            device_index = 0;
        }

        s_device = devices[device_index];

        fprintf(stdout, "FFTDOCK Metal> Using device: %s\n",
                [s_device.name UTF8String]);

        s_queue = [s_device newCommandQueue];
        if (!s_queue) {
            fprintf(stderr, "FFTDOCK Metal> Failed to create command queue.\n");
            return -1;
        }

        /* Locate the pre-compiled Metal library. */
        NSString* libPath = nil;
        const char* env = getenv("METAL_KERNELS_PATH");
        if (env) {
            libPath = [NSString stringWithUTF8String: env];
        } else {
            /* Search beside the executable. */
            NSString* exeDir =
                [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
            libPath = [exeDir stringByAppendingPathComponent: @"metal_kernels.metallib"];
        }

        NSError* err = nil;
        s_library = [s_device newLibraryWithURL: [NSURL fileURLWithPath: libPath]
                                          error: &err];
        if (!s_library) {
            fprintf(stderr,
                    "FFTDOCK Metal> Cannot load Metal library at '%s'.\n"
                    "  Error: %s\n"
                    "  Build it with:\n"
                    "    xcrun metal   -c metal_kernels.metal -o metal_kernels.air\n"
                    "    xcrun metallib   metal_kernels.air   -o metal_kernels.metallib\n",
                    [libPath UTF8String],
                    err ? [err.localizedDescription UTF8String] : "unknown");
            return -1;
        }
    }
    return 0;
}

void* metal_get_device(void) { return (__bridge void*)s_device; }
void* metal_get_queue (void) { return (__bridge void*)s_queue;  }

void* metal_get_pipeline(const char* kernel_name)
{
    /* Check cache first. */
    for (int i = 0; i < s_num_pipelines; i++)
        if (strcmp(s_pipeline_cache[i].name, kernel_name) == 0)
            return (__bridge void*)s_pipeline_cache[i].pso;

    /* Build a new pipeline state. */
    @autoreleasepool {
        if (!s_library) {
            fprintf(stderr,
                    "FFTDOCK Metal> metal_get_pipeline called before "
                    "metal_state_init.\n");
            exit(1);
        }

        NSString* name = [NSString stringWithUTF8String: kernel_name];
        id<MTLFunction> fn = [s_library newFunctionWithName: name];
        if (!fn) {
            fprintf(stderr,
                    "FFTDOCK Metal> Kernel '%s' not found in Metal library.\n",
                    kernel_name);
            exit(1);
        }

        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [s_device newComputePipelineStateWithFunction: fn error: &err];
        if (!pso) {
            fprintf(stderr,
                    "FFTDOCK Metal> Pipeline creation failed for '%s': %s\n",
                    kernel_name,
                    err ? [err.localizedDescription UTF8String] : "unknown");
            exit(1);
        }

        if (s_num_pipelines >= MAX_PIPELINES) {
            fprintf(stderr, "FFTDOCK Metal> Pipeline cache overflow.\n");
            exit(1);
        }

        s_pipeline_cache[s_num_pipelines].name = strdup(kernel_name);
        s_pipeline_cache[s_num_pipelines].pso  = pso;
        s_num_pipelines++;

        return (__bridge void*)pso;
    }
}

/* ------------------------------------------------------------------ */
/*  Buffer helpers                                                      */
/* ------------------------------------------------------------------ */

void* metal_buffer_alloc(size_t bytes)
{
    id<MTLBuffer> buf =
        [s_device newBufferWithLength: bytes
                              options: MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "FFTDOCK Metal> Failed to allocate %zu byte buffer.\n",
                bytes);
        exit(1);
    }
    return (__bridge_retained void*)buf;
}

void metal_buffer_zero(void* buf, size_t bytes)
{
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    memset(b.contents, 0, bytes);
}

void metal_buffer_copy_to(void* buf, const void* src, size_t bytes)
{
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    memcpy(b.contents, src, bytes);
}

void metal_buffer_copy_from(void* buf, void* dst, size_t bytes)
{
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    memcpy(dst, b.contents, bytes);
}

void* metal_buffer_contents(void* buf)
{
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    return b.contents;
}

void metal_buffer_release(void* buf)
{
    id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf;
    (void)b;   /* ARC releases the object */
}

/* ------------------------------------------------------------------ */
/*  Command submission                                                  */
/* ------------------------------------------------------------------ */

void metal_dispatch_1d(void*   pipeline_opaque,
                       void**  bufs,
                       size_t* offsets,
                       int     nbuf,
                       size_t  n_threads,
                       size_t  tgroup)
{
    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            (__bridge id<MTLComputePipelineState>)pipeline_opaque;

        id<MTLCommandBuffer> cb = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState: pso];

        for (int i = 0; i < nbuf; i++) {
            id<MTLBuffer> b = (__bridge id<MTLBuffer>)bufs[i];
            NSUInteger off  = offsets ? (NSUInteger)offsets[i] : 0;
            [enc setBuffer: b offset: off atIndex: (NSUInteger)i];
        }

        MTLSize tgSize   = MTLSizeMake(tgroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(n_threads, 1, 1);
        /* dispatchThreads:threadsPerThreadgroup: handles non-multiple sizes. */
        [enc dispatchThreads: gridSize threadsPerThreadgroup: tgSize];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void metal_sync(void)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [s_queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

/* ================================================================== */
/*  MPSGraph GPU FFT  (macOS 14+ / MetalPerformanceShadersGraph)       */
/*                                                                      */
/*  Uses MPSGraph.fastFourierTransformWithTensor — one method handles R2C, */
/*  C2C, and C2R based on input data type and descriptor.inverse flag.    */
/*  Framework: MetalPerformanceShadersGraph.  Compared to vDSP:           */
/*                                                                      */
/*   • No size constraint (any grid dimension works).                   */
/*   • Compute happens fully on the GPU, freeing all CPU cores.         */
/*   • Zero-copy data feed via NSData dataWithBytesNoCopy on Apple      */
/*     Silicon unified memory — no host↔device transfer cost.           */
/*   • MPSGraph is compiled once at plan creation; subsequent calls     */
/*     reuse the compiled graph with new input data.                    */
/*                                                                      */
/*  Availability: macOS 14+ (Sonoma, MetalPerformanceShadersGraph).     */
/*  Falls back to vDSP automatically on earlier systems.               */
/* ================================================================== */

int mps_fft_available(void)
{
#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) { return 1; }
#endif
    return 0;
}

/* ------------------------------------------------------------------
 * mps_create_r2c_plan
 *
 * Build an MPSGraph for a batched 3-D R2C (forward) FFT.
 * The graph has a dynamic-shape placeholder so that any batch size
 * can be fed at runtime without rebuilding.
 * ------------------------------------------------------------------ */
int mps_create_r2c_plan(MetalFFTPlan* plan)
{
#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            MPSGraph* graph = [[MPSGraph alloc] init];

            /* Real input placeholder — shape provided at execution time:
             * [batch, Nx, Ny, Nz].  nil = fully dynamic shape.          */
            MPSGraphTensor* inputTensor =
                [graph placeholderWithShape:nil
                                   dataType:MPSDataTypeFloat32
                                       name:@"r2c_input"];

            /* Axes: transform along spatial dimensions 1, 2, 3 (not batch 0). */
            int32_t axes_data[3] = {1, 2, 3};
            NSData* axesNSData = [NSData dataWithBytes:axes_data
                                               length:sizeof(axes_data)];
            MPSGraphTensor* axesTensor =
                [graph constantWithData:axesNSData
                                  shape:@[@3]
                               dataType:MPSDataTypeInt32];

            /* Forward R2C (real → Hermitian-symmetric complex), unnormalized.
             * fastFourierTransformWithTensor determines R2C vs C2C from the
             * input tensor data type: Float32 input → R2C output complex.
             * descriptor.inverse = NO → forward transform.
             * Output shape: [batch, Nx, Ny, Nz/2+1] complex.             */
            MPSGraphFFTDescriptor* desc = [MPSGraphFFTDescriptor descriptor];
            desc.inverse     = NO;
            desc.scalingMode = MPSGraphFFTScalingModeNone;

            MPSGraphTensor* outputTensor =
                [graph fastFourierTransformWithTensor:inputTensor
                                          axesTensor:axesTensor
                                          descriptor:desc
                                                name:@"r2c_fft_out"];

            /* Retain the ObjC objects through void* for C-struct storage.
             * NOTE: we intentionally do NOT call compileWithDevice: here.
             * MPSGraphExecutable.compileWithDevice: fails for FFT graphs
             * with "unable to load function ndArrayFFTRadix4 / unresolved
             * visible function reference: postfixPrimary_cf" because AOT
             * compilation cannot generate the per-size visible-function
             * variants.  The JIT path (runWithMTLCommandQueue:) works fine
             * and handles visible-function stitching internally.           */
            plan->mpsg_graph  = (__bridge_retained void*)graph;
            plan->mpsg_input  = (__bridge_retained void*)inputTensor;
            plan->mpsg_output = (__bridge_retained void*)outputTensor;
            /* plan->mpsg_exec intentionally left NULL */
            return 1;
        }
    }
#endif
    return 0;
}

/* ------------------------------------------------------------------
 * mps_create_c2r_plan
 *
 * Build an MPSGraph for a batched 3-D C2R (inverse) FFT.
 * Input placeholder is MPSDataTypeComplexFloat32 (interleaved pairs),
 * shape [batch, Nx, Ny, Nz/2+1].
 * ------------------------------------------------------------------ */
int mps_create_c2r_plan(MetalFFTPlan* plan)
{
#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            MPSGraph* graph = [[MPSGraph alloc] init];

            /* Complex input placeholder [batch, Nx, Ny, Nz/2+1]. */
            MPSGraphTensor* inputTensor =
                [graph placeholderWithShape:nil
                                   dataType:MPSDataTypeComplexFloat32
                                       name:@"c2r_input"];

            /* Axes: transform along spatial dimensions 1, 2, 3. */
            int32_t axes_data[3] = {1, 2, 3};
            NSData* axesNSData = [NSData dataWithBytes:axes_data
                                               length:sizeof(axes_data)];
            MPSGraphTensor* axesTensor =
                [graph constantWithData:axesNSData
                                  shape:@[@3]
                               dataType:MPSDataTypeInt32];

            /* Inverse C2R (Hermitian-symmetric complex → real), unnormalized.
             * fastFourierTransformWithTensor determines C2R vs C2C from the
             * input tensor data type: ComplexFloat32 input + inverse = YES
             * → real output of shape [batch, Nx, Ny, Nz].
             * Caller's correctEnergy kernel divides by Nx*Ny*Nz.         */
            MPSGraphFFTDescriptor* desc = [MPSGraphFFTDescriptor descriptor];
            desc.inverse     = YES;
            desc.scalingMode = MPSGraphFFTScalingModeNone;

            MPSGraphTensor* outputTensor =
                [graph fastFourierTransformWithTensor:inputTensor
                                          axesTensor:axesTensor
                                          descriptor:desc
                                                name:@"c2r_fft_out"];

            /* See R2C plan comment: compileWithDevice: is intentionally
             * skipped for FFT graphs (visible-function stitching crash).   */
            plan->mpsg_graph  = (__bridge_retained void*)graph;
            plan->mpsg_input  = (__bridge_retained void*)inputTensor;
            plan->mpsg_output = (__bridge_retained void*)outputTensor;
            /* plan->mpsg_exec intentionally left NULL */
            return 1;
        }
    }
#endif
    return 0;
}

/* ------------------------------------------------------------------
 * mps_release_plan — release the three retained ObjC objects
 * (graph, input tensor, output tensor).  mpsg_exec is always NULL
 * because compileWithDevice: is intentionally skipped.
 * ------------------------------------------------------------------ */
void mps_release_plan(MetalFFTPlan* plan)
{
    if (!plan) return;
#if defined(__MAC_14_0)
    @autoreleasepool {
        if (plan->mpsg_graph) {
            MPSGraph* g __unused =
                (__bridge_transfer MPSGraph*)plan->mpsg_graph;
            plan->mpsg_graph = NULL;
        }
        if (plan->mpsg_input) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)plan->mpsg_input;
            plan->mpsg_input = NULL;
        }
        if (plan->mpsg_output) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)plan->mpsg_output;
            plan->mpsg_output = NULL;
        }
        if (plan->mpsg_exec) {
            MPSGraphExecutable* e __unused =
                (__bridge_transfer MPSGraphExecutable*)plan->mpsg_exec;
            plan->mpsg_exec = NULL;
        }
    }
#endif
}

/* ------------------------------------------------------------------
 * mps_fft3d_r2c_batch
 *
 * Execute a batched 3-D R2C FFT via MPSGraph.
 *
 *  in_real   — real input  [batch × Nx × Ny × Nz], row-major floats
 *  out_cplx  — complex output [batch × Nx × Ny × (Nz/2+1)],
 *              MPSDataTypeComplexFloat32 = interleaved (re, im) float pairs
 *
 * Uses graph.runWithMTLCommandQueue: (JIT compilation — compileWithDevice:
 * is intentionally avoided because its AOT path crashes with an unresolved
 * Metal visible-function reference for ndArrayFFTRadix4).
 *
 * MPSGraph writes output into GPU-private memory.  To avoid the SIGBUS
 * that results from calling mpsndarray.readBytes: on private storage, the
 * output MTLBuffer is obtained via mpsndarray.resource (available macOS
 * 12.3+) and blitted into a MTLStorageModeShared staging buffer before
 * the CPU reads it.
 * ------------------------------------------------------------------ */
void mps_fft3d_r2c_batch(const MetalFFTPlan* plan,
                          const float* in_real,
                          float*       out_cplx,
                          int          batch)
{
#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            int Nx  = plan->xdim;
            int Ny  = plan->ydim;
            int Nz  = plan->zdim;
            int Nzc = Nz / 2 + 1;

            MPSGraph*           graph = (__bridge MPSGraph*)plan->mpsg_graph;
            MPSGraphTensor*     inT   = (__bridge MPSGraphTensor*)plan->mpsg_input;
            MPSGraphTensor*     outT  = (__bridge MPSGraphTensor*)plan->mpsg_output;
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_get_queue();

            /* ---- Input buffer (zero-copy if page-aligned, else copy) ---- */
            size_t inBytes = (size_t)batch * Nx * Ny * Nz * sizeof(float);
            id<MTLBuffer> inBuf =
                [s_device newBufferWithBytesNoCopy:(void*)in_real
                                            length:inBytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
            if (!inBuf)
                inBuf = [s_device newBufferWithBytes:in_real
                                              length:inBytes
                                             options:MTLResourceStorageModeShared];

            MPSGraphTensorData* inData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:inBuf
                                shape:@[@(batch), @(Nx), @(Ny), @(Nz)]
                             dataType:MPSDataTypeFloat32];

            /* ---- Run FFT (JIT — output lands in GPU-private memory) ---- */
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                [graph runWithMTLCommandQueue:queue
                                       feeds:@{inT: inData}
                               targetTensors:@[outT]
                            targetOperations:nil];

            /* ---- Copy output to CPU-accessible memory via Metal blit ---- */
            /* MPSNDArray.resource exposes the backing MTLBuffer.  The property
             * was added to the public SDK header in macOS 15; on macOS 14 it
             * exists at runtime as a private API so we access it via KVC +
             * respondsToSelector: to avoid the compile-time "property not
             * found" error on older SDKs.                                    */
            size_t outBytes = (size_t)batch * Nx * Ny * Nzc * 2 * sizeof(float);
            MPSNDArray*   outArr = results[outT].mpsndarray;
            id<MTLBuffer> gpuBuf = nil;
            if ([outArr respondsToSelector:NSSelectorFromString(@"resource")])
                gpuBuf = (id<MTLBuffer>)[outArr valueForKey:@"resource"];

            if (gpuBuf && gpuBuf.storageMode == MTLStorageModeShared) {
                /* Already CPU-accessible (e.g. on unified-memory device that
                 * chose shared storage) — skip the blit.                    */
                memcpy(out_cplx, gpuBuf.contents, outBytes);
            } else if (gpuBuf) {
                /* Private storage — blit to a shared staging buffer. */
                id<MTLBuffer> staging =
                    [s_device newBufferWithLength:outBytes
                                         options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer>     blitCmd = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit   = [blitCmd blitCommandEncoder];
                [blit copyFromBuffer:gpuBuf  sourceOffset:0
                            toBuffer:staging destinationOffset:0
                                size:MIN(outBytes, gpuBuf.length)];
                [blit endEncoding];
                [blitCmd commit];
                [blitCmd waitUntilCompleted];
                memcpy(out_cplx, staging.contents, outBytes);
            } else {
                /* Fallback (resource unavailable): attempt direct readback.
                 * Will work if MPS happened to use shared storage; crashes
                 * (SIGBUS) if private — but we have no alternative here.    */
                [outArr readBytes:out_cplx strideBytes:nil];
            }
        }
        return;
    }
#endif
    (void)plan; (void)in_real; (void)out_cplx; (void)batch;
}

/* ------------------------------------------------------------------
 * mps_fft3d_c2r_batch
 *
 * Execute a batched 3-D C2R inverse FFT via MPSGraph.
 *
 *  in_cplx   — complex input [batch × Nx × Ny × (Nz/2+1)],
 *              interleaved float pairs — same layout as MPSDataTypeComplexFloat32
 *  out_real  — real output [batch × Nx × Ny × Nz]
 *
 * Same JIT + blit strategy as mps_fft3d_r2c_batch (see above).
 * Result is NOT normalised; caller's correctEnergy kernel divides by idist.
 * ------------------------------------------------------------------ */
void mps_fft3d_c2r_batch(const MetalFFTPlan* plan,
                          const float* in_cplx,
                          float*       out_real,
                          int          batch)
{
#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            int Nx  = plan->xdim;
            int Ny  = plan->ydim;
            int Nz  = plan->zdim;
            int Nzc = Nz / 2 + 1;

            MPSGraph*           graph = (__bridge MPSGraph*)plan->mpsg_graph;
            MPSGraphTensor*     inT   = (__bridge MPSGraphTensor*)plan->mpsg_input;
            MPSGraphTensor*     outT  = (__bridge MPSGraphTensor*)plan->mpsg_output;
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_get_queue();

            /* ---- Input buffer ---- */
            size_t inBytes = (size_t)batch * Nx * Ny * Nzc * 2 * sizeof(float);
            id<MTLBuffer> inBuf =
                [s_device newBufferWithBytesNoCopy:(void*)in_cplx
                                            length:inBytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
            if (!inBuf)
                inBuf = [s_device newBufferWithBytes:in_cplx
                                              length:inBytes
                                             options:MTLResourceStorageModeShared];

            MPSGraphTensorData* inData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:inBuf
                                shape:@[@(batch), @(Nx), @(Ny), @(Nzc)]
                             dataType:MPSDataTypeComplexFloat32];

            /* ---- Run inverse FFT (JIT — output in GPU-private memory) ---- */
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                [graph runWithMTLCommandQueue:queue
                                       feeds:@{inT: inData}
                               targetTensors:@[outT]
                            targetOperations:nil];

            /* ---- Copy output to CPU via Metal blit (same as R2C above) ---- */
            size_t outBytes = (size_t)batch * Nx * Ny * Nz * sizeof(float);
            MPSNDArray*   outArr = results[outT].mpsndarray;
            id<MTLBuffer> gpuBuf = nil;
            if ([outArr respondsToSelector:NSSelectorFromString(@"resource")])
                gpuBuf = (id<MTLBuffer>)[outArr valueForKey:@"resource"];

            if (gpuBuf && gpuBuf.storageMode == MTLStorageModeShared) {
                memcpy(out_real, gpuBuf.contents, outBytes);
            } else if (gpuBuf) {
                id<MTLBuffer> staging =
                    [s_device newBufferWithLength:outBytes
                                         options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer>     blitCmd = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit   = [blitCmd blitCommandEncoder];
                [blit copyFromBuffer:gpuBuf  sourceOffset:0
                            toBuffer:staging destinationOffset:0
                                size:MIN(outBytes, gpuBuf.length)];
                [blit endEncoding];
                [blitCmd commit];
                [blitCmd waitUntilCompleted];
                memcpy(out_real, staging.contents, outBytes);
            } else {
                [outArr readBytes:out_real strideBytes:nil];
            }
        }
        return;
    }
#endif
    (void)plan; (void)in_cplx; (void)out_real; (void)batch;
}

/* ================================================================== */
/*  End of MPSGraph GPU FFT section                                    */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/*  vDSP size validation                                                */
/* ------------------------------------------------------------------ */

/* Returns 1 if n is a valid vDSP R2C/C2C size: n = 2^a * 3^b * 5^c, a >= 2.
 * On failure prints the nearest valid size above n and returns 0.           */
int vdsp_dft_size_ok(int n, const char* dim_name)
{
    /* Find the nearest valid size >= n for diagnostic purposes */
    auto is_valid = [](int x) -> bool {
        if (x < 4) return false;
        int t = x;
        /* Must be divisible by 4 (a >= 2) */
        if (t % 4 != 0) return false;
        while (t % 2 == 0) t /= 2;
        while (t % 3 == 0) t /= 3;
        while (t % 5 == 0) t /= 5;
        return (t == 1);
    };

    if (n < 4) {
        fprintf(stderr,
                "FFTDOCK Metal> Grid dimension %s=%d too small "
                "(vDSP requires n >= 4, divisible by 4).\n", dim_name, n);
        return 0;
    }

    /* Count factors of 2 to check a >= 2 */
    int tmp = n, a = 0;
    while (tmp % 2 == 0) { tmp /= 2; a++; }
    while (tmp % 3 == 0)   tmp /= 3;
    while (tmp % 5 == 0)   tmp /= 5;

    if (tmp != 1 || a < 2) {
        /* Find nearest valid size >= n */
        int suggestion = n + (4 - n % 4) % 4;   /* round up to next mult of 4 */
        while (!is_valid(suggestion)) suggestion += 4;

        /* Also explain how to adjust CHARMM grid parameters:
           GridNum = XMAX / DGRI + 2  =>  XMAX = (suggestion - 2) * DGRI    */
        fprintf(stderr,
                "FFTDOCK Metal> ERROR: grid dimension %s=%d is invalid for vDSP FFT.\n"
                "  vDSP requires n = 2^a * 3^b * 5^c  with a >= 2 (divisible by 4).\n"
                "  %d = ", dim_name, n, n);
        if (a > 0) fprintf(stderr, "2^%d * %d", a, tmp * (n / (1 << a)));
        else       fprintf(stderr, "%d (no factor of 2)", n);
        fprintf(stderr,
                "  ->  a=%d < 2, FAILS.\n"
                "  Nearest valid size >= %d :  %d\n"
                "  Adjust CHARMM input: set xmax = %.4g  (with current dgrid)\n"
                "  or use a dgrid that gives GridNum = XMAX/dgrid + 2 = %d.\n"
                "  Common valid sizes: 16, 20, 24, 32, 36, 40, 48, 60, 64, 80,\n"
                "                      96, 100, 120, 128, 160, 192, 200, 240, 256.\n",
                a, n, suggestion,
                (suggestion - 2) * 0.5,   /* illustrative with dgrid=0.5 */
                suggestion);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/*  3-D FFT helpers (vDSP decomposition)                               */
/*                                                                      */
/*  Split-complex layout used internally:                               */
/*    re[] and im[] are separate arrays of length Nx * Ny * Nzc         */
/*    where Nzc = Nz/2 + 1                                              */
/*                                                                      */
/*  The public API uses interleaved complex (float2 / two floats):      */
/*    [re0, im0, re1, im1, ...]                                         */
/* ------------------------------------------------------------------ */

/* Convert interleaved → split for `n` complex values. */
static void interleaved_to_split(const float* il,
                                 float* sr, float* si, int n)
{
    for (int i = 0; i < n; i++) {
        sr[i] = il[2*i    ];
        si[i] = il[2*i + 1];
    }
}

/* Convert split → interleaved for `n` complex values. */
static void split_to_interleaved(const float* sr, const float* si,
                                  float* il, int n)
{
    for (int i = 0; i < n; i++) {
        il[2*i    ] = sr[i];
        il[2*i + 1] = si[i];
    }
}

/* ----
 * Single 3-D R2C FFT on one [Nx][Ny][Nz] real array.
 * Input:   in_r  [Nx * Ny * Nz]   real values, row-major
 * Output:  split complex sr/si  [Nx * Ny * Nzc]  where Nzc = Nz/2+1
 */
static void fft3d_r2c_single(const MetalFFTPlan* plan,
                              const float*        in_r,
                              float* sr, float* si)
{
    int Nx  = plan->xdim;
    int Ny  = plan->ydim;
    int Nz  = plan->zdim;
    int Nzc = Nz / 2 + 1;

    /* Temporaries for row / column passes */
    std::vector<float> tmp_r(Nx * Ny * Nzc, 0.f);
    std::vector<float> tmp_i(Nx * Ny * Nzc, 0.f);
    std::vector<float> zi(Nz, 0.f);   /* imaginary part of Z-row (always 0 for R2C) */
    std::vector<float> zo_r(Nzc), zo_i(Nzc);

    /* === Pass 1: 1-D R2C along Z, for each (ix, iy) === */
    for (int ix = 0; ix < Nx; ix++) {
        for (int iy = 0; iy < Ny; iy++) {
            const float* zrow = in_r + (ix * Ny + iy) * Nz;
            vDSP_DFT_Execute(plan->dft_z,
                             zrow,   zi.data(),
                             zo_r.data(), zo_i.data());
            int base = (ix * Ny + iy) * Nzc;
            for (int iz = 0; iz < Nzc; iz++) {
                tmp_r[base + iz] = zo_r[iz];
                tmp_i[base + iz] = zo_i[iz];
            }
        }
    }

    /* === Pass 2: 1-D C2C along Y, for each (ix, izc) === */
    std::vector<float> yin_r(Ny), yin_i(Ny), yo_r(Ny), yo_i(Ny);
    for (int ix = 0; ix < Nx; ix++) {
        for (int izc = 0; izc < Nzc; izc++) {
            for (int iy = 0; iy < Ny; iy++) {
                yin_r[iy] = tmp_r[(ix * Ny + iy) * Nzc + izc];
                yin_i[iy] = tmp_i[(ix * Ny + iy) * Nzc + izc];
            }
            vDSP_DFT_Execute(plan->dft_y,
                             yin_r.data(), yin_i.data(),
                             yo_r.data(),  yo_i.data());
            for (int iy = 0; iy < Ny; iy++) {
                tmp_r[(ix * Ny + iy) * Nzc + izc] = yo_r[iy];
                tmp_i[(ix * Ny + iy) * Nzc + izc] = yo_i[iy];
            }
        }
    }

    /* === Pass 3: 1-D C2C along X, for each (iy, izc) === */
    std::vector<float> xin_r(Nx), xin_i(Nx), xo_r(Nx), xo_i(Nx);
    for (int iy = 0; iy < Ny; iy++) {
        for (int izc = 0; izc < Nzc; izc++) {
            for (int ix = 0; ix < Nx; ix++) {
                xin_r[ix] = tmp_r[(ix * Ny + iy) * Nzc + izc];
                xin_i[ix] = tmp_i[(ix * Ny + iy) * Nzc + izc];
            }
            vDSP_DFT_Execute(plan->dft_x,
                             xin_r.data(), xin_i.data(),
                             xo_r.data(),  xo_i.data());
            for (int ix = 0; ix < Nx; ix++) {
                tmp_r[(ix * Ny + iy) * Nzc + izc] = xo_r[ix];
                tmp_i[(ix * Ny + iy) * Nzc + izc] = xo_i[ix];
            }
        }
    }

    /* Copy result into caller's split-complex output arrays */
    int total = Nx * Ny * Nzc;
    memcpy(sr, tmp_r.data(), total * sizeof(float));
    memcpy(si, tmp_i.data(), total * sizeof(float));
}

/* ----
 * Single 3-D C2R inverse FFT on one [Nx][Ny][Nzc] complex array.
 * Input:   split complex sr/si  [Nx * Ny * Nzc]
 * Output:  out_r                [Nx * Ny * Nz]  real values
 * Not normalised (consistent with cuFFT).
 */
static void fft3d_c2r_single(const MetalFFTPlan* plan,
                              const float* sr, const float* si,
                              float* out_r)
{
    int Nx  = plan->xdim;
    int Ny  = plan->ydim;
    int Nz  = plan->zdim;
    int Nzc = Nz / 2 + 1;

    std::vector<float> tmp_r(sr, sr + Nx * Ny * Nzc);
    std::vector<float> tmp_i(si, si + Nx * Ny * Nzc);

    /* === Pass 1 (inverse): IFFT along X === */
    std::vector<float> xin_r(Nx), xin_i(Nx), xo_r(Nx), xo_i(Nx);
    for (int iy = 0; iy < Ny; iy++) {
        for (int izc = 0; izc < Nzc; izc++) {
            for (int ix = 0; ix < Nx; ix++) {
                xin_r[ix] = tmp_r[(ix * Ny + iy) * Nzc + izc];
                xin_i[ix] = tmp_i[(ix * Ny + iy) * Nzc + izc];
            }
            vDSP_DFT_Execute(plan->dft_x,
                             xin_r.data(), xin_i.data(),
                             xo_r.data(),  xo_i.data());
            for (int ix = 0; ix < Nx; ix++) {
                tmp_r[(ix * Ny + iy) * Nzc + izc] = xo_r[ix];
                tmp_i[(ix * Ny + iy) * Nzc + izc] = xo_i[ix];
            }
        }
    }

    /* === Pass 2: IFFT along Y === */
    std::vector<float> yin_r(Ny), yin_i(Ny), yo_r(Ny), yo_i(Ny);
    for (int ix = 0; ix < Nx; ix++) {
        for (int izc = 0; izc < Nzc; izc++) {
            for (int iy = 0; iy < Ny; iy++) {
                yin_r[iy] = tmp_r[(ix * Ny + iy) * Nzc + izc];
                yin_i[iy] = tmp_i[(ix * Ny + iy) * Nzc + izc];
            }
            vDSP_DFT_Execute(plan->dft_y,
                             yin_r.data(), yin_i.data(),
                             yo_r.data(),  yo_i.data());
            for (int iy = 0; iy < Ny; iy++) {
                tmp_r[(ix * Ny + iy) * Nzc + izc] = yo_r[iy];
                tmp_i[(ix * Ny + iy) * Nzc + izc] = yo_i[iy];
            }
        }
    }

    /* === Pass 3: C2R inverse FFT along Z === */
    std::vector<float> zo_r(Nz);
    std::vector<float> zo_i_out(Nz);   /* dummy imaginary output */
    for (int ix = 0; ix < Nx; ix++) {
        for (int iy = 0; iy < Ny; iy++) {
            int base = (ix * Ny + iy) * Nzc;
            /* vDSP C2R is done via a forward DFT on conjugated data,
               then taking the real part and scaling — or simply use
               the inverse DFT setup (created with vDSP_DFT_INVERSE). */
            vDSP_DFT_Execute(plan->dft_z,
                             tmp_r.data() + base, tmp_i.data() + base,
                             zo_r.data(),          zo_i_out.data());
            /* Only the real output (first Nz points) is meaningful for C2R.
               Copy just the real part; imaginary is negligible for valid
               Hermitian-symmetric inputs. */
            int out_base = (ix * Ny + iy) * Nz;
            for (int iz = 0; iz < Nz; iz++)
                out_r[out_base + iz] = zo_r[iz];
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Public batched FFT entry points                                     */
/* ------------------------------------------------------------------ */

void fft3d_r2c_batch(const MetalFFTPlan* plan,
                     const float*        in_real,
                     float*              out_cplx,
                     int                 batch)
{
    /* Prefer Metal GPU via MPS (macOS 14+, no size constraint). */
    if (plan->use_mps) {
        mps_fft3d_r2c_batch(plan, in_real, out_cplx, batch);
        return;
    }

    /* vDSP CPU fallback (macOS 12/13). */
    int Nx  = plan->xdim;
    int Ny  = plan->ydim;
    int Nz  = plan->zdim;
    int Nzc = Nz / 2 + 1;
    int idist = Nx * Ny * Nz;
    int odist = Nx * Ny * Nzc;

    std::vector<float> sr(odist), si(odist);

    for (int b = 0; b < batch; b++) {
        fft3d_r2c_single(plan, in_real + b * idist, sr.data(), si.data());
        /* Convert split → interleaved into the output buffer */
        split_to_interleaved(sr.data(), si.data(),
                             out_cplx + b * odist * 2, odist);
    }
}

void fft3d_c2r_batch(const MetalFFTPlan* plan,
                     const float*        in_cplx,
                     float*              out_real,
                     int                 batch)
{
    /* Prefer Metal GPU via MPS (macOS 14+). */
    if (plan->use_mps) {
        mps_fft3d_c2r_batch(plan, in_cplx, out_real, batch);
        return;
    }

    /* vDSP CPU fallback (macOS 12/13). */
    int Nx  = plan->xdim;
    int Ny  = plan->ydim;
    int Nz  = plan->zdim;
    int Nzc = Nz / 2 + 1;
    int idist = Nx * Ny * Nzc;
    int odist = Nx * Ny * Nz;

    std::vector<float> sr(idist), si(idist);

    for (int b = 0; b < batch; b++) {
        /* Convert interleaved input → split format */
        interleaved_to_split(in_cplx + b * idist * 2,
                             sr.data(), si.data(), idist);
        fft3d_c2r_single(plan, sr.data(), si.data(), out_real + b * odist);
    }
}

/* ====================================================================== */
/*                                                                          */
/*  Device enumeration / selection — implementations of the mtl_* API       */
/*                                                                          */
/*  Parallel to ocl_device_* in source/opencl/ocl_util.cpp; consumed by     */
/*  module metal_main_mod in source/metal/metal_main.F90.                   */
/*                                                                          */
/* ====================================================================== */

extern "C" int mtl_device_init(void** devices_out)
{
    if (!devices_out) { return -1; }

    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (!devices || devices.count == 0) {
            fprintf(stderr, "CHARMM Metal> No Metal devices found.\n");
            *devices_out = NULL;
            return -1;
        }
        /* Retain across the C ABI.  Caller releases via
         * mtl_device_list_release. */
        *devices_out = (__bridge_retained void*) devices;
        return 0;
    }
}

extern "C" void mtl_device_list_release(void** devices)
{
    if (!devices || !*devices) { return; }
    @autoreleasepool {
        NSArray<id<MTLDevice>>* arr = (__bridge_transfer NSArray<id<MTLDevice>>*)(*devices);
        (void) arr;  /* arc releases */
        *devices = NULL;
    }
}

/* Build the "<id> <name> <bytes>" line once, used by both
 * mtl_device_print_one and mtl_device_string. */
static NSString*
_mtl_device_line(id<MTLDevice> dev, NSUInteger oneBasedId)
{
    unsigned long long mem = (unsigned long long) [dev recommendedMaxWorkingSetSize];
    return [NSString stringWithFormat:@"    %2lu %@ %llu",
                     (unsigned long) oneBasedId,
                     [dev name],
                     mem];
}

extern "C" void mtl_device_print(void* devices)
{
    if (!devices) { return; }
    @autoreleasepool {
        NSArray<id<MTLDevice>>* arr = (__bridge NSArray<id<MTLDevice>>*) devices;
        printf(" Metal devices: (id #) (name) (bytes of memory)\n");
        NSUInteger i = 1;
        for (id<MTLDevice> dev in arr) {
            NSString* line = _mtl_device_line(dev, i++);
            printf("%s\n", [line UTF8String]);
        }
    }
}

extern "C" void mtl_device_print_one(void* dev_in)
{
    if (!dev_in) { return; }
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) dev_in;
        NSString* line = _mtl_device_line(dev, 0);
        printf("%s\n", [line UTF8String]);
    }
}

extern "C" void mtl_device_string(void* dev_in, char* c_string, int max_size)
{
    if (!c_string || max_size <= 0) { return; }
    c_string[0] = '\0';
    if (!dev_in) { return; }
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) dev_in;
        NSString* line = _mtl_device_line(dev, 0);
        const char* utf = [line UTF8String];
        if (!utf) { return; }
        size_t len = strlen(utf);
        if (len >= (size_t)max_size) { len = (size_t)max_size - 1; }
        memcpy(c_string, utf, len);
        c_string[len] = '\0';
    }
}

extern "C" int mtl_device_get(void* devices, int dev_id, void** out_device)
{
    if (!devices || !out_device) { return -1; }
    @autoreleasepool {
        NSArray<id<MTLDevice>>* arr = (__bridge NSArray<id<MTLDevice>>*) devices;
        /* Fortran side is 1-based for parity with the existing opencl module. */
        int idx = dev_id - 1;
        if (idx < 0 || idx >= (int) arr.count) {
            fprintf(stderr, "CHARMM Metal> device index %d out of range "
                            "(1-%lu)\n", dev_id, (unsigned long) arr.count);
            *out_device = NULL;
            return -1;
        }
        *out_device = (__bridge void*) arr[idx];
        return 0;
    }
}

extern "C" int mtl_device_max_mem_get(void* devices, void** out_device)
{
    if (!devices || !out_device) { return -1; }
    @autoreleasepool {
        NSArray<id<MTLDevice>>* arr = (__bridge NSArray<id<MTLDevice>>*) devices;
        if (arr.count == 0) { return -1; }
        id<MTLDevice> best = nil;
        NSUInteger bestMem = 0;
        for (id<MTLDevice> dev in arr) {
            NSUInteger m = [dev recommendedMaxWorkingSetSize];
            if (best == nil || m > bestMem) { best = dev; bestMem = m; }
        }
        *out_device = (__bridge void*) best;
        return 0;
    }
}

extern "C" int mtl_begin_session(void* in_dev)
{
    if (!in_dev) { return -1; }
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>) in_dev;
        if (s_device != nil && s_device != dev) {
            /* Switching devices mid-run is not supported; mirrors OpenCL
             * "session" semantics which also pin a single context. */
            fprintf(stderr, "CHARMM Metal> session already started on a "
                            "different device; ignoring request.\n");
            return -1;
        }
        s_device = dev;
        if (s_queue == nil) {
            s_queue = [s_device newCommandQueue];
            if (s_queue == nil) {
                fprintf(stderr, "CHARMM Metal> Failed to create MTLCommandQueue.\n");
                return -1;
            }
        }
        /* Library load is best-effort: FFTDOCK-only consumers may not have
         * placed metal_kernels.metallib yet at module-init time. */
        if (s_library == nil) {
            NSString* libPath = nil;
            const char* env = getenv("METAL_KERNELS_PATH");
            if (env) {
                libPath = [NSString stringWithUTF8String: env];
            } else {
                NSString* exeDir =
                    [[[NSBundle mainBundle] executablePath]
                     stringByDeletingLastPathComponent];
                libPath = [exeDir stringByAppendingPathComponent:
                                  @"metal_kernels.metallib"];
            }
            NSError* err = nil;
            s_library = [s_device newLibraryWithURL: [NSURL fileURLWithPath:libPath]
                                              error: &err];
            /* No hard error if missing — metal_fftdock_gpu.mm will surface
             * a message later if it actually needs a kernel. */
            (void) err;
        }
        return 0;
    }
}

extern "C" int mtl_end_session(void)
{
    @autoreleasepool {
        /* Release library and queue; leave s_device alone (autoreleased). */
        if (s_library != nil) { s_library = nil; }
        if (s_queue   != nil) { s_queue   = nil; }
        /* Pipeline cache still points at released functions — clear it. */
        for (int i = 0; i < s_num_pipelines; ++i) {
            if (s_pipeline_cache[i].name) {
                free((void*) s_pipeline_cache[i].name);
                s_pipeline_cache[i].name = NULL;
            }
            s_pipeline_cache[i].pso = nil;
        }
        s_num_pipelines = 0;
    }
    return 0;
}
