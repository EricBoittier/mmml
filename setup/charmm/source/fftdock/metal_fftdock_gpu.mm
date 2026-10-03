/*
 * metal_fftdock_gpu.mm
 *
 * Apple Metal FFT-docking pipeline (KEY_METAL build).  Implements the
 * public API declared in metal_fftdock_gpu.h.
 *
 * Architecture
 * ─────────────
 * Each per-batch docking call encodes the full pipeline into a SINGLE
 * MPSCommandBuffer using MPSGraph.encodeToCommandBuffer: (which queues
 * FFT work into an existing command buffer rather than executing it
 * inline as runWithMTLCommandQueue: does).  Metal's implicit per-
 * command-buffer ordering + resource-hazard tracking guarantees correct
 * sequencing between MPSGraph FFT encoders and our own compute kernels:
 *
 *   [Single MPSCommandBuffer]
 *     encodeToCommandBuffer: (R2C FFT, lig)   → ctx->buf_lig_fft (shared)
 *     computeCommandEncoder   (conjMult)       → buf_lig_fft in-place
 *     computeCommandEncoder   (sumGrids)       → ctx->buf_sum_fft (shared)
 *     encodeToCommandBuffer: (C2R IFFT)        → ctx->buf_energy   (shared)
 *     computeCommandEncoder   (correctEnergy)  → buf_energy in-place
 *   commit + waitUntilCompleted
 *   memcpy buf_energy.contents → host EnergyGrid     (only CPU↔GPU touch)
 *
 * All four buffers live in the MetalDockContext (created in
 * metal_fftdock_setup) and are MTLStorageModeShared so:
 *   (a) MPSGraph can write FFT output directly via resultsDictionary:
 *       — no private MPSNDArray.resource extraction (a private API that
 *         is not reliably present across macOS / SDK combinations)
 *   (b) The CPU can read buf_energy.contents directly after commit, no
 *       blit step needed.
 *
 * Memory layout
 * ──────────────
 *   buf_pot_fft  [num_grid * odist * 2 * float]              shared
 *                  — Receptor R2C FFT, computed once in upload_potential
 *   buf_lig_fft  [num_grid * batch_size * odist * 2 * float] shared
 *                  — Ligand R2C FFT, recomputed per batch
 *   buf_sum_fft  [batch_size * odist * 2 * float]            shared
 *                  — Summed cross-correlation spectrum (per rotamer)
 *   buf_energy   [batch_size * idist * float]                shared
 *                  — Final docking energy at every translation
 *
 * Requires macOS 14+ (MPSGraph FFT).  Hard-aborts in setup() if not
 * available — see comment in metal_fftdock_gpu.h.
 *
 * - YWu
 */

#import  <Metal/Metal.h>
#import  <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import  <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import  <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_util.h"
#include "metal_fftdock_gpu.h"

/* ====================================================================== */
/*  Internal context struct                                                 */
/* ====================================================================== */

typedef struct {
    /* Grid dimensions */
    int xdim, ydim, zdim;
    int batch_size, num_grid;
    int idist;   /* xdim * ydim * zdim          */
    int odist;   /* xdim * ydim * (zdim/2+1)    */
    int Nzc;     /* zdim/2 + 1                  */

    /* MPSGraph — forward R2C FFT
     * Single dynamic-shape graph reused for both potential (num_grid
     * transforms) and ligand (num_grid*batch_size transforms).          */
    void* r2c_graph;     /* retained MPSGraph*           */
    void* r2c_inT;       /* retained MPSGraphTensor* placeholder (Float32)  */
    void* r2c_outT;      /* retained MPSGraphTensor* output                 */

    /* MPSGraph — inverse C2R FFT */
    void* c2r_graph;     /* retained MPSGraph*           */
    void* c2r_inT;       /* retained MPSGraphTensor* placeholder (ComplexFloat32) */
    void* c2r_outT;      /* retained MPSGraphTensor* output                 */

    /* Persistent shared buffers (both CPU+GPU accessible on Apple Silicon
     * unified memory).  Using shared mode lets MPSGraph write FFT output
     * directly into our buffers via resultsDictionary: — no private-memory
     * extraction via MPSNDArray.resource (a private API) needed.           */
    void* buf_pot_fft;   /* retained id<MTLBuffer>
                            [num_grid * odist * 2 * sizeof(float)]
                            Potential R2C FFT output (ComplexFloat32)        */
    void* buf_lig_fft;   /* retained id<MTLBuffer>
                            [num_grid * batch_size * odist * 2 * sizeof(float)]
                            Ligand R2C FFT output (ComplexFloat32)           */
    void* buf_sum_fft;   /* retained id<MTLBuffer>
                            [batch_size * odist * 2 * sizeof(float)]
                            sumGrids output; C2R FFT reads from here        */
    void* buf_energy;    /* retained id<MTLBuffer>
                            [batch_size * idist * sizeof(float)]
                            C2R FFT writes here → correctEnergy in-place
                            → CPU reads after commit                        */

    /* Metal compute pipeline states (from global pipeline cache) */
    void* pipeline_conjMult;
    void* pipeline_sumGrids;
    void* pipeline_correctEnergy;

} MetalDockContext;

/* ====================================================================== */
/*  Helper: dispatch a 1-D compute pass on an existing command buffer      */
/* ====================================================================== */

/* Sets up a compute encoder on cmdBuf, sets scalar integer arguments
 * via setBytes: (matching the Metal kernels' `constant int& [[buffer(n)]]`
 * declarations), and dispatches the specified number of threads.
 *
 * ints[0..n_ints-1]   : values for buffer slots 0 .. n_ints-1
 * bufs[0..n_bufs-1]   : MTLBuffer* handles for slots n_ints .. n_ints+n_bufs-1
 * n_threads            : total 1-D thread count
 */
static void encode_dispatch_1d(id<MTLCommandBuffer>           cmdBuf,
                                id<MTLComputePipelineState>    pipeline,
                                const int*                     ints,
                                int                            n_ints,
                                id<MTLBuffer>* __nullable      bufs,
                                int                            n_bufs,
                                NSUInteger                     n_threads)
{
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    for (int i = 0; i < n_ints; ++i)
        [enc setBytes:&ints[i] length:sizeof(int) atIndex:(NSUInteger)i];

    for (int i = 0; i < n_bufs; ++i)
        [enc setBuffer:bufs[i] offset:0 atIndex:(NSUInteger)(n_ints + i)];

    NSUInteger tgSize = MIN((NSUInteger)256,
                            pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger nGroups = (n_threads + tgSize - 1) / tgSize;
    [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgSize,  1, 1)];
    [enc endEncoding];
}

/* ====================================================================== */
/*  metal_fftdock_setup                                                     */
/* ====================================================================== */

void metal_fftdock_setup(int   gpu_id,
                          int   xdim,
                          int   ydim,
                          int   zdim,
                          int   batch_size,
                          int   num_grid,
                          void** ctx_out)
{
    if (!ctx_out) { return; }
    *ctx_out = NULL;

    /* Initialise (or reuse) the global Metal device + command queue. */
    if (metal_state_init(gpu_id) != 0) {
        fprintf(stderr,
                "FFTDOCK Metal GPU> Cannot initialise Metal device %d.\n",
                gpu_id);
        exit(1);
    }

    /*
     * The single-command-buffer docking pipeline is built on top of
     * MPSGraph FFT, which requires macOS 14+ (Sonoma) at runtime.  When
     * the OS is older we have no safe path forward — the whole docking
     * loop in fftdock.F90 assumes metal_ctx is a valid handle after this
     * call.  Abort loudly instead of silently returning NULL (which used
     * to cause a later crash inside metal_fftdock_run_batch).
     */
#if defined(__MAC_14_0)
    if (!(@available(macOS 14.0, *))) {
        fprintf(stderr,
                "FFTDOCK Metal GPU> MPSGraph FFT requires macOS 14 (Sonoma) "
                "or newer. Current OS is unsupported by this build.\n");
        exit(1);
    }
#else
    fprintf(stderr,
            "FFTDOCK Metal GPU> Built without macOS 14 SDK; MPSGraph FFT "
            "entry points are unavailable. Rebuild against Xcode 15+.\n");
    exit(1);
#endif

    MetalDockContext* ctx =
        (MetalDockContext*)calloc(1, sizeof(MetalDockContext));
    if (!ctx) {
        fprintf(stderr, "FFTDOCK Metal GPU> calloc(MetalDockContext) failed.\n");
        exit(1);
    }

    ctx->xdim       = xdim;
    ctx->ydim       = ydim;
    ctx->zdim       = zdim;
    ctx->batch_size = batch_size;
    ctx->num_grid   = num_grid;
    ctx->idist      = xdim * ydim * zdim;
    ctx->Nzc        = zdim / 2 + 1;
    ctx->odist      = xdim * ydim * ctx->Nzc;

    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_get_device();

#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            /*
             * IMPORTANT: use realToHermiteanFFTWithTensor:axes: (NOT
             * fastFourierTransformWithTensor:axesTensor:) because the two
             * methods differ in WHICH axis is halved for R2C:
             *
             *   fastFourierTransform...  →  halves the FIRST axis in list
             *   realToHermiteanFFT...     →  halves the LAST  axis in list
             *
             * cuFFT always halves the last (innermost, z) dimension:
             *   output [batch, x, y, z/2+1]
             *
             * With axes {1, 2, 3}, realToHermiteanFFT halves axis 3 (z),
             * matching cuFFT.  fastFourierTransform would halve axis 1 (x)
             * and produce [batch, x/2+1, y, z] — corrupting every
             * conjMult / sumGrids index.
             */
            NSArray<NSNumber *>* fftAxes = @[@1, @2, @3];

            /* ---- Build forward R2C graph --------------------------------- */
            MPSGraph* r2cG = [[MPSGraph alloc] init];

            MPSGraphTensor* r2cIn =
                [r2cG placeholderWithShape:nil
                                  dataType:MPSDataTypeFloat32
                                      name:@"r2c_in"];

            MPSGraphFFTDescriptor* fdesc = [MPSGraphFFTDescriptor descriptor];
            fdesc.inverse     = NO;
            fdesc.scalingMode = MPSGraphFFTScalingModeNone;

            MPSGraphTensor* r2cOut =
                [r2cG realToHermiteanFFTWithTensor:r2cIn
                                              axes:fftAxes
                                        descriptor:fdesc
                                              name:@"r2c_out"];

            ctx->r2c_graph = (__bridge_retained void*)r2cG;
            ctx->r2c_inT   = (__bridge_retained void*)r2cIn;
            ctx->r2c_outT  = (__bridge_retained void*)r2cOut;

            /* ---- Build inverse C2R graph --------------------------------- */
            MPSGraph* c2rG = [[MPSGraph alloc] init];

            MPSGraphTensor* c2rIn =
                [c2rG placeholderWithShape:nil
                                  dataType:MPSDataTypeComplexFloat32
                                      name:@"c2r_in"];

            MPSGraphFFTDescriptor* idesc = [MPSGraphFFTDescriptor descriptor];
            idesc.inverse     = YES;
            idesc.scalingMode = MPSGraphFFTScalingModeNone;

            MPSGraphTensor* c2rOut =
                [c2rG HermiteanToRealFFTWithTensor:c2rIn
                                              axes:fftAxes
                                        descriptor:idesc
                                              name:@"c2r_out"];

            ctx->c2r_graph = (__bridge_retained void*)c2rG;
            ctx->c2r_inT   = (__bridge_retained void*)c2rIn;
            ctx->c2r_outT  = (__bridge_retained void*)c2rOut;
        }
    }
#endif /* __MAC_14_0 */

    /* ---- Pre-allocate persistent shared buffers ----------------------- */
    /* All buffers use MTLStorageModeShared so MPSGraph can write FFT
     * output directly via resultsDictionary: and compute kernels can
     * read/write without any blit or copy.  On Apple Silicon the CPU
     * and GPU share the same physical memory.                            */
    size_t potFftBytes = (size_t)num_grid * ctx->odist * 2 * sizeof(float);
    size_t ligFftBytes = (size_t)num_grid * batch_size * ctx->odist * 2 * sizeof(float);
    size_t sumFftBytes = (size_t)batch_size * ctx->odist * 2 * sizeof(float);
    size_t energyBytes = (size_t)batch_size * ctx->idist * sizeof(float);

    id<MTLBuffer> potFftBuf =
        [device newBufferWithLength:potFftBytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> ligFftBuf =
        [device newBufferWithLength:ligFftBytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> sumFftBuf =
        [device newBufferWithLength:sumFftBytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> energyBuf =
        [device newBufferWithLength:energyBytes
                            options:MTLResourceStorageModeShared];
    if (!potFftBuf || !ligFftBuf || !sumFftBuf || !energyBuf) {
        fprintf(stderr,
                "FFTDOCK Metal GPU> newBufferWithLength failed "
                "(pot_fft=%zu, lig_fft=%zu, sum_fft=%zu, energy=%zu bytes). "
                "Check GPU memory and grid dims.\n",
                potFftBytes, ligFftBytes, sumFftBytes, energyBytes);
        free(ctx);
        exit(1);
    }

    ctx->buf_pot_fft = (__bridge_retained void*)potFftBuf;
    ctx->buf_lig_fft = (__bridge_retained void*)ligFftBuf;
    ctx->buf_sum_fft = (__bridge_retained void*)sumFftBuf;
    ctx->buf_energy  = (__bridge_retained void*)energyBuf;

    /* ---- Grab compute pipelines from global cache --------------------- */
    ctx->pipeline_conjMult      = metal_get_pipeline("conjMult");
    ctx->pipeline_sumGrids      = metal_get_pipeline("sumGrids");
    ctx->pipeline_correctEnergy = metal_get_pipeline("correctEnergy");

    fprintf(stdout,
            "FFTDOCK Metal GPU> Context ready: %d×%d×%d, "
            "batch %d, %d grid channels.\n",
            xdim, ydim, zdim, batch_size, num_grid);

    *ctx_out = (void*)ctx;
}

/* ====================================================================== */
/*  metal_fftdock_upload_potential                                          */
/* ====================================================================== */

void metal_fftdock_upload_potential(void*  ctx_ptr,
                                     float* grid_potential)
{
    MetalDockContext* ctx = (MetalDockContext*)ctx_ptr;

#if defined(__MAC_14_0)
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            id<MTLDevice>       device = (__bridge id<MTLDevice>)metal_get_device();
            id<MTLCommandQueue> queue  =
                (__bridge id<MTLCommandQueue>)metal_get_queue();

            MPSGraph*       r2cG    = (__bridge MPSGraph*)ctx->r2c_graph;
            MPSGraphTensor* r2cInT  = (__bridge MPSGraphTensor*)ctx->r2c_inT;
            MPSGraphTensor* r2cOutT = (__bridge MPSGraphTensor*)ctx->r2c_outT;

            int Nx = ctx->xdim, Ny = ctx->ydim, Nz = ctx->zdim;
            int Nzc = ctx->Nzc;
            int NG = ctx->num_grid;

            /* Wrap host potential in a shared MTLBuffer.
             * Zero-copy if the pointer is page-aligned; else copy.     */
            size_t potBytes = (size_t)NG * Nx * Ny * Nz * sizeof(float);
            id<MTLBuffer> potInBuf =
                [device newBufferWithBytesNoCopy:grid_potential
                                          length:potBytes
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
            if (!potInBuf)
                potInBuf = [device newBufferWithBytes:grid_potential
                                               length:potBytes
                                              options:MTLResourceStorageModeShared];

            MPSGraphTensorData* potInData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:potInBuf
                                shape:@[@(NG), @(Nx), @(Ny), @(Nz)]
                             dataType:MPSDataTypeFloat32];

            /* Encode the potential R2C FFT.  Output goes directly into
             * ctx->buf_pot_fft (pre-allocated shared buffer) via
             * resultsDictionary: — no private-memory extraction needed.
             *
             * IMPORTANT: encodeToCommandBuffer: requires MPSCommandBuffer,
             * NOT a raw id<MTLCommandBuffer>.                              */
            id<MTLBuffer> potFftBuf =
                (__bridge id<MTLBuffer>)ctx->buf_pot_fft;

            MPSGraphTensorData* potOutData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:potFftBuf
                                shape:@[@(NG), @(Nx), @(Ny), @(Nzc)]
                             dataType:MPSDataTypeComplexFloat32];

            MPSCommandBuffer *cmdBuf =
                [MPSCommandBuffer commandBufferFromCommandQueue:queue];

            [r2cG encodeToCommandBuffer:cmdBuf
                                  feeds:@{r2cInT: potInData}
                       targetOperations:nil
                      resultsDictionary:@{r2cOutT: potOutData}
                    executionDescriptor:nil];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            fprintf(stdout,
                    "FFTDOCK Metal GPU> Potential R2C FFT complete "
                    "(%d grids, %d×%d×%d).\n", NG, Nx, Ny, Nz);
        }
    }
#else
    (void)ctx_ptr; (void)grid_potential;
#endif
}

/* ====================================================================== */
/*  metal_fftdock_run_batch                                                 */
/* ====================================================================== */

void metal_fftdock_run_batch(void*  ctx_ptr,
                              void*  d_lig_grid_f,
                              float* energy_grid)
{
    /* Diagnostic entry trace.  Useful for verifying that the Fortran
     * caller passes a valid context + ligand buffer pointer.  Re-enable
     * if metal_fftdock_run_batch silently fails to execute.
     *
     * Expected on a healthy run (c46test/fftdock.inp):
     *   FFTDOCK Metal GPU> run_batch ENTRY: ctx=0x... lig=0x... eg=0x...
     *   FFTDOCK Metal GPU> run_batch: __MAC_14_0 defined, entering pipeline
     * - YWu */
    /*
    fprintf(stdout, "FFTDOCK Metal GPU> run_batch ENTRY: ctx=%p lig=%p eg=%p\n",
            ctx_ptr, d_lig_grid_f, energy_grid);
    fflush(stdout);
    */

    if (!ctx_ptr) {
        fprintf(stderr, "FFTDOCK Metal GPU> run_batch: NULL context!\n");
        return;
    }
    if (!d_lig_grid_f) {
        fprintf(stderr, "FFTDOCK Metal GPU> run_batch: NULL d_lig_grid_f!\n");
        return;
    }

    MetalDockContext* ctx = (MetalDockContext*)ctx_ptr;

#if defined(__MAC_14_0)
    /*
    fprintf(stdout, "FFTDOCK Metal GPU> run_batch: __MAC_14_0 defined, entering pipeline\n");
    fflush(stdout);
    */
    if (@available(macOS 14.0, *)) {
        @autoreleasepool {
            id<MTLCommandQueue> queue =
                (__bridge id<MTLCommandQueue>)metal_get_queue();

            int Nx  = ctx->xdim,  Ny = ctx->ydim, Nz = ctx->zdim;
            int Nzc = ctx->Nzc,   BS = ctx->batch_size;
            int NG  = ctx->num_grid;
            int idist = ctx->idist, odist = ctx->odist;

            /* Pipeline states */
            id<MTLComputePipelineState> pConj =
                (__bridge id<MTLComputePipelineState>)ctx->pipeline_conjMult;
            id<MTLComputePipelineState> pSum  =
                (__bridge id<MTLComputePipelineState>)ctx->pipeline_sumGrids;
            id<MTLComputePipelineState> pCorr =
                (__bridge id<MTLComputePipelineState>)ctx->pipeline_correctEnergy;

            /* Persistent buffers */
            id<MTLBuffer> potFftBuf =
                (__bridge id<MTLBuffer>)ctx->buf_pot_fft;
            id<MTLBuffer> sumFftBuf =
                (__bridge id<MTLBuffer>)ctx->buf_sum_fft;
            id<MTLBuffer> energyBuf =
                (__bridge id<MTLBuffer>)ctx->buf_energy;

            /* MPSGraph objects */
            MPSGraph*       r2cG    = (__bridge MPSGraph*)ctx->r2c_graph;
            MPSGraphTensor* r2cInT  = (__bridge MPSGraphTensor*)ctx->r2c_inT;
            MPSGraphTensor* r2cOutT = (__bridge MPSGraphTensor*)ctx->r2c_outT;
            MPSGraph*       c2rG    = (__bridge MPSGraph*)ctx->c2r_graph;
            MPSGraphTensor* c2rInT  = (__bridge MPSGraphTensor*)ctx->c2r_inT;
            MPSGraphTensor* c2rOutT = (__bridge MPSGraphTensor*)ctx->c2r_outT;

            /* Ligand grid buffer produced by calcLigGrid.
             * Layout: [NG * BS, Nx, Ny, Nz] floats (row-major).       */
            id<MTLBuffer> ligRealBuf = (__bridge id<MTLBuffer>)d_lig_grid_f;

            MPSGraphTensorData* ligInData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:ligRealBuf
                                shape:@[@(NG * BS), @(Nx), @(Ny), @(Nz)]
                             dataType:MPSDataTypeFloat32];

            /* ============================================================
             * SINGLE COMMAND BUFFER — all six GPU operations
             *
             * MPSGraph.encodeToCommandBuffer: requires MPSCommandBuffer
             * (wraps a raw MTLCommandBuffer + command queue pair).
             * ============================================================ */
            MPSCommandBuffer *cmdBuf =
                [MPSCommandBuffer commandBufferFromCommandQueue:queue];

            /* ── Step 1: R2C FFT on ligand grids ─────────────────────── */
            /* Output goes directly into ctx->buf_lig_fft (pre-allocated
             * shared buffer) via resultsDictionary:.                      */
            id<MTLBuffer> ligFftBuf =
                (__bridge id<MTLBuffer>)ctx->buf_lig_fft;

            MPSGraphTensorData* ligOutData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:ligFftBuf
                                shape:@[@(NG * BS), @(Nx), @(Ny), @(Nzc)]
                             dataType:MPSDataTypeComplexFloat32];

            [r2cG encodeToCommandBuffer:cmdBuf
                                  feeds:@{r2cInT: ligInData}
                       targetOperations:nil
                      resultsDictionary:@{r2cOutT: ligOutData}
                    executionDescriptor:nil];

            /* Diagnostic: commit R2C FFTs separately and dump the first
             * 3 complex elements of each output.  Useful for verifying
             * the FFT axis ordering and Hermitian halving behave as
             * expected (matching cuFFT R2C output).
             *
             * Expected (c46test/fftdock.inp, 18x18x18, 3 grid channels):
             *   potFFT first 6 floats (3 complex):
             *     2.692738e+04 0.000000e+00  4.963962e+03 2.941474e+03
             *     -7.652308e+02 -7.000751e+01
             *   ligFFT first 6 floats (3 complex):
             *      1.039230e+00 0.000000e+00  -2.000439e-02 -4.568392e-01
             *      3.687641e-01 -2.635092e-02
             * (DC component imaginary part should be ~0 for both)
             *
             * NOTE: enabling this splits the single-command-buffer pipeline
             * into multiple commits, defeating the GPU-resident-everything
             * design.  Use only for debugging.
             * - YWu */
            /*
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            {
                float* potF = (float*)[potFftBuf contents];
                float* ligF = (float*)[ligFftBuf contents];
                fprintf(stdout,
                    "FFTDOCK Metal GPU> STAGE DEBUG after R2C FFTs:\n"
                    "  potFFT first 6 floats (3 complex): %.6e %.6e  %.6e %.6e  %.6e %.6e\n"
                    "  ligFFT first 6 floats (3 complex): %.6e %.6e  %.6e %.6e  %.6e %.6e\n",
                    potF[0], potF[1], potF[2], potF[3], potF[4], potF[5],
                    ligF[0], ligF[1], ligF[2], ligF[3], ligF[4], ligF[5]);
                fflush(stdout);
            }
            cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            */

            /* ── Step 2: conjMult ─────────────────────────────────────── */
            /* In-place: lig_F[b][g][k] *= conj(pot_F[g][k])
             * Kernel signature: buf(0)=N, buf(1)=pot_F, buf(2)=lig_F,
             *                   buf(3)=odist, buf(4)=num_grids           */
            {
                int conj_N = BS * NG * odist;
                int s_odist  = odist;
                int s_ngrids = NG;
                const int ints[3] = { conj_N, s_odist, s_ngrids };
                /* buf(0)=N, buf(3)=odist, buf(4)=num_grids via setBytes
                 * buf(1)=pot_F, buf(2)=lig_F via setBuffer              */
                id<MTLComputeCommandEncoder> enc =
                    [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pConj];
                [enc setBytes:&conj_N   length:sizeof(int) atIndex:0];
                [enc setBuffer:potFftBuf offset:0          atIndex:1];
                [enc setBuffer:ligFftBuf offset:0          atIndex:2];
                [enc setBytes:&s_odist  length:sizeof(int) atIndex:3];
                [enc setBytes:&s_ngrids length:sizeof(int) atIndex:4];
                NSUInteger tg = MIN((NSUInteger)256,
                                    pConj.maxTotalThreadsPerThreadgroup);
                NSUInteger ng = ((NSUInteger)conj_N + tg - 1) / tg;
                [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }

            /* ── Step 3: sumGrids ─────────────────────────────────────── */
            /* sum_F[b][k] = Σ_g lig_F[b][g][k]  → sumFftBuf
             * Kernel signature: buf(0)=N, buf(1)=lig_F, buf(2)=sum_F,
             *                   buf(3)=num_grids, buf(4)=odist           */
            {
                int sum_N    = BS * odist;
                int s_odist  = odist;
                int s_ngrids = NG;
                id<MTLComputeCommandEncoder> enc =
                    [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pSum];
                [enc setBytes:&sum_N    length:sizeof(int) atIndex:0];
                [enc setBuffer:ligFftBuf offset:0          atIndex:1];
                [enc setBuffer:sumFftBuf offset:0          atIndex:2];
                [enc setBytes:&s_ngrids length:sizeof(int) atIndex:3];
                [enc setBytes:&s_odist  length:sizeof(int) atIndex:4];
                NSUInteger tg = MIN((NSUInteger)256,
                                    pSum.maxTotalThreadsPerThreadgroup);
                NSUInteger ng = ((NSUInteger)sum_N + tg - 1) / tg;
                [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }

            /* Diagnostic: commit conjMult + sumGrids and dump sumFFT
             * (combined cross-correlation spectrum, summed across grid
             * types) plus a ligand-grid sanity sample.
             *
             * Expected (c46test/fftdock.inp, 18x18x18, 3 grid channels):
             *   sumFFT first 6 floats (3 complex):
             *      1.069060e+05 0.000000e+00  -9.180345e+03 -9.735963e+03
             *      -3.852557e+02 2.055719e+03
             *   ligGrid (first 1000 of 1749600): sum=5.196152e-01 nonzero=16
             * (only ~16 nonzero values for 1 rotamer's first grid type;
             * the remaining 99 batch slots are zero — cuFFT/MPSGraph both
             * pad the rotamer dimension to batch_size=100)
             * - YWu */
            /*
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            {
                float* sumF = (float*)[sumFftBuf contents];
                fprintf(stdout,
                    "FFTDOCK Metal GPU> STAGE DEBUG after conjMult+sumGrids:\n"
                    "  sumFFT first 6 floats (3 complex): %.6e %.6e  %.6e %.6e  %.6e %.6e\n",
                    sumF[0], sumF[1], sumF[2], sumF[3], sumF[4], sumF[5]);
                float* ligR = (float*)[(__bridge id<MTLBuffer>)(d_lig_grid_f) contents];
                float ligSum = 0;
                int ligNZ = 0;
                for (int ii = 0; ii < NG * idist && ii < 1000; ii++) {
                    ligSum += ligR[ii];
                    if (ligR[ii] != 0.0f) ligNZ++;
                }
                fprintf(stdout,
                    "  ligGrid (first %d of %d): sum=%.6e nonzero=%d\n",
                    NG*idist < 1000 ? NG*idist : 1000, NG*BS*idist, ligSum, ligNZ);
                fflush(stdout);
            }
            cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            */

            /* ── Step 4: C2R inverse FFT ──────────────────────────────── */
            /* Both input (sumFftBuf) and output (energyBuf) are shared-
             * mode.  Output goes directly into energyBuf via
             * resultsDictionary: — no private-memory extraction or blit. */
            MPSGraphTensorData* sumInData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:sumFftBuf
                                shape:@[@(BS), @(Nx), @(Ny), @(Nzc)]
                             dataType:MPSDataTypeComplexFloat32];

            MPSGraphTensorData* c2rOutData =
                [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:energyBuf
                                shape:@[@(BS), @(Nx), @(Ny), @(Nz)]
                             dataType:MPSDataTypeFloat32];

            [c2rG encodeToCommandBuffer:cmdBuf
                                  feeds:@{c2rInT: sumInData}
                       targetOperations:nil
                      resultsDictionary:@{c2rOutT: c2rOutData}
                    executionDescriptor:nil];

            /* ── Step 5: correctEnergy ────────────────────────────────── */
            /* In-place divide by idist on energyBuf (C2R wrote here).
             * Kernel signature: buf(0)=N, buf(1)=idist, buf(2)=data     */
            {
                int ener_N  = BS * idist;
                int s_idist = idist;
                id<MTLComputeCommandEncoder> enc =
                    [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:pCorr];
                [enc setBytes:&ener_N  length:sizeof(int) atIndex:0];
                [enc setBytes:&s_idist length:sizeof(int) atIndex:1];
                [enc setBuffer:energyBuf offset:0         atIndex:2];
                NSUInteger tg = MIN((NSUInteger)256,
                                    pCorr.maxTotalThreadsPerThreadgroup);
                NSUInteger ng = ((NSUInteger)ener_N + tg - 1) / tg;
                [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
            }

            /* No blit needed — energyBuf is shared-mode, CPU can read
             * directly after commit.                                     */

            /* ── Single commit: all GPU operations execute now ────────── */
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            /* ── Step 7: CPU reads final result from shared energyBuf ─── */
            size_t outBytes = (size_t)BS * idist * sizeof(float);
            memcpy(energy_grid, energyBuf.contents, outBytes);

            /* Diagnostic: dump full energy-grid statistics — global &
             * interior minimum, sum, max, and probe energies at corner /
             * centre / interior grid points.  This was the key tool for
             * spotting the conjMult sign-flip bug: the global minimum
             * appearing at the upper grid corner instead of near the
             * binding pocket signaled an FFT spatial reflection.
             *
             * AFTER conjMult sign-flip fix (c46test/fftdock.inp):
             *   The global minimum should fall in the interior (near the
             *   binding pocket centre, NOT at (Nx-1, Ny-1, Nz-1)), and
             *   docking RMSD should be ~0.37.
             *
             * BEFORE the fix the output looked like:
             *   GLOBAL min = -1.7244e+01 at grid (13,14,14)   <- reflected
             *   INTERIOR min (margin=6, range [0,11]): -5.4962e+00 at (11,11,0)
             *   Probe energies at rotamer 0 (all positive ~14-28, no clear
             *   binding pocket) - YWu */
            /*
            {
                float* eg = energy_grid;
                int total = BS * idist;
                float emin = eg[0], emax = eg[0], esum = 0.0f;
                int imin = 0;
                for (int ii = 0; ii < total; ii++) {
                    esum += eg[ii];
                    if (eg[ii] < emin) { emin = eg[ii]; imin = ii; }
                    if (eg[ii] > emax) { emax = eg[ii]; }
                }
                int rotamer_of_min = imin / idist;
                int gridpt_of_min  = imin % idist;
                int gx = gridpt_of_min / (Ny * Nz);
                int gy = (gridpt_of_min % (Ny * Nz)) / Nz;
                int gz = gridpt_of_min % Nz;
                fprintf(stdout,
                    "FFTDOCK Metal GPU> EnergyGrid DEBUG (BS=%d, dims=%dx%dx%d):\n"
                    "  GLOBAL min = %.4e at rotamer %d pt %d = grid (%d,%d,%d)\n"
                    "  sum=%.4e  max=%.4e\n",
                    BS, Nx, Ny, Nz, emin, rotamer_of_min, gridpt_of_min,
                    gx, gy, gz, esum, emax);

                int margin = 6;
                float emin_int = 1e30f;
                int gx_int=0, gy_int=0, gz_int=0;
                for (int ix = 0; ix < Nx - margin; ix++) {
                    for (int iy = 0; iy < Ny - margin; iy++) {
                        for (int iz = 0; iz < Nz - margin; iz++) {
                            float v = eg[(ix*Ny + iy)*Nz + iz];
                            if (v < emin_int) {
                                emin_int = v;
                                gx_int = ix; gy_int = iy; gz_int = iz;
                            }
                        }
                    }
                }
                fprintf(stdout,
                    "  INTERIOR min (margin=%d, range [0,%d]): %.4e at (%d,%d,%d)\n",
                    margin, Nx-margin-1, emin_int, gx_int, gy_int, gz_int);

                int probes[8][3] = {{0,0,0}, {Nx-1,Ny-1,Nz-1}, {Nx/2,Ny/2,Nz/2},
                                    {4,4,4}, {6,6,6}, {8,8,8}, {2,2,2}, {1,1,1}};
                fprintf(stdout, "  Probe energies at rotamer 0:\n");
                for (int p = 0; p < 8; p++) {
                    int ix=probes[p][0], iy=probes[p][1], iz=probes[p][2];
                    float v = eg[(ix*Ny + iy)*Nz + iz];
                    fprintf(stdout, "    (%2d,%2d,%2d) = %12.6e\n", ix, iy, iz, v);
                }
                fflush(stdout);
            }
            */

        } /* @autoreleasepool */
        return;
    }
#endif
    (void)ctx_ptr; (void)d_lig_grid_f; (void)energy_grid;
}

/* ====================================================================== */
/*  metal_fftdock_cleanup                                                   */
/* ====================================================================== */

void metal_fftdock_cleanup(void** ctx_out)
{
    if (!ctx_out || !*ctx_out) return;
    MetalDockContext* ctx = (MetalDockContext*)*ctx_out;

#if defined(__MAC_14_0)
    @autoreleasepool {
        /* Release MPSGraph objects */
        if (ctx->r2c_graph) {
            MPSGraph* g __unused =
                (__bridge_transfer MPSGraph*)ctx->r2c_graph;
            ctx->r2c_graph = NULL;
        }
        if (ctx->r2c_inT) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)ctx->r2c_inT;
            ctx->r2c_inT = NULL;
        }
        if (ctx->r2c_outT) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)ctx->r2c_outT;
            ctx->r2c_outT = NULL;
        }
        if (ctx->c2r_graph) {
            MPSGraph* g __unused =
                (__bridge_transfer MPSGraph*)ctx->c2r_graph;
            ctx->c2r_graph = NULL;
        }
        if (ctx->c2r_inT) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)ctx->c2r_inT;
            ctx->c2r_inT = NULL;
        }
        if (ctx->c2r_outT) {
            MPSGraphTensor* t __unused =
                (__bridge_transfer MPSGraphTensor*)ctx->c2r_outT;
            ctx->c2r_outT = NULL;
        }

        /* Release persistent shared buffers */
        if (ctx->buf_pot_fft) {
            id<MTLBuffer> b __unused =
                (__bridge_transfer id<MTLBuffer>)ctx->buf_pot_fft;
            ctx->buf_pot_fft = NULL;
        }
        if (ctx->buf_lig_fft) {
            id<MTLBuffer> b __unused =
                (__bridge_transfer id<MTLBuffer>)ctx->buf_lig_fft;
            ctx->buf_lig_fft = NULL;
        }
        if (ctx->buf_sum_fft) {
            id<MTLBuffer> b __unused =
                (__bridge_transfer id<MTLBuffer>)ctx->buf_sum_fft;
            ctx->buf_sum_fft = NULL;
        }
        if (ctx->buf_energy) {
            id<MTLBuffer> b __unused =
                (__bridge_transfer id<MTLBuffer>)ctx->buf_energy;
            ctx->buf_energy = NULL;
        }
    }
#endif

    free(ctx);
    *ctx_out = NULL;
    fprintf(stdout, "FFTDOCK Metal GPU> Context freed.\n");
}
