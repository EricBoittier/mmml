/*
 * metal_grid_lig.mm
 *
 * Apple Metal implementation of calcLigGrid — ligand rotamer grid
 * generation.  Exposes the same C entry point as cuda_grid_lig.cu so
 * fftdock.F90 → Generate_Lig_Grid_GPU() can call it unchanged:
 *
 *   void calcLigGrid(BatchIdx, BatchSize, NumRotamers, NumAtoms,
 *                    NumVdwGridUsed, DGrid, XGridNum, YGridNum, ZGridNum,
 *                    SelectAtomsParameters, LigGrid,
 *                    LigRotamerCoors, LigRotamerMinCoors,
 *                    d_LigGrid_F)
 *
 * What it does
 * ─────────────
 * Builds the ligand density grid for every rotamer in the current batch
 * via the generateLigGrid Metal kernel.  The GPU buffer for LigGrid is
 * allocated on the first batch call (BatchIdx == 1) and reused for
 * subsequent batches — same lifecycle as the CUDA version.
 *
 * Persistent GPU memory & d_LigGrid_F write-through
 * ──────────────────────────────────────────────────
 * d_LigGrid_F is declared `void*&` (C++ reference) — exactly like the
 * CUDA version.  Fortran passes c_loc(d_LigGrid), which the C++ ABI
 * binds as a reference to the Fortran module variable d_LigGrid.
 * Assigning `d_LigGrid_F = buf_LigGrid` therefore writes the retained
 * MTLBuffer pointer THROUGH to Fortran, so metal_fftdock_run_batch can
 * later receive the same buffer handle by value.
 *
 * On subsequent batches d_LigGrid_F already holds the previously-
 * retained buffer; we just re-zero it and dispatch the kernel again.
 * The buffer is released in clean_FFTDock_GPU().
 *
 * Mapping to CUDA version (cuda_grid_lig.cu)
 * ───────────────────────────────────────────
 *  cudaMalloc (first batch)   →  metal_buffer_alloc
 *  cudaMemcpy H→D             →  metal_buffer_copy_to
 *  cudaMemset                 →  metal_buffer_zero
 *  cuLaunchKernel             →  metal_dispatch_1d
 *  cudaFree (temp buffers)    →  metal_buffer_release
 *
 * - YWu
 */

#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include "metal_util.h"
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */

extern "C"
void calcLigGrid(const int   BatchIdx,
                 const int   BatchSize,
                 const int   NumRotamers,
                 const int   NumAtoms,
                 const int   NumVdwGridUsed,
                 const float DGrid,
                 const int   XGridNum,
                 const int   YGridNum,
                 const int   ZGridNum,
                 float*      SelectAtomsParameters,  /* [NumAtoms * 4]              */
                 float*      LigGrid,                /* [BatchSize*NumGrids*X*Y*Z]  */
                 float*      LigRotamerCoors,        /* [BatchSize * NumAtoms * 3]  */
                 float*      LigRotamerMinCoors,     /* [BatchSize * 3]             */
                 void*&      d_LigGrid_F)            /* persistent GPU buffer handle*/
{
    /* ---- Initialise Metal (no-op if already done) ---- */
    if (metal_state_init(0) != 0) {
        fprintf(stderr, "FFTDOCK Metal> calcLigGrid: Metal unavailable.\n");
        exit(1);
    }

    const int NumGrids     = NumVdwGridUsed + 1;   /* VdW channels + electrostatics */
    const int NumGridPoints = XGridNum * YGridNum * ZGridNum;

    /* Actual number of rotamers in this batch (last batch may be partial). */
    int RealBatchSize = BatchSize;
    if (BatchIdx * BatchSize > NumRotamers)
        RealBatchSize = NumRotamers % BatchSize;

    size_t sz_rotCoors = (size_t)BatchSize * NumAtoms * 3 * sizeof(float);
    size_t sz_minCoors = (size_t)BatchSize * 3 * sizeof(float);
    size_t sz_params   = (size_t)NumAtoms  * 4 * sizeof(float);
    size_t sz_ligGrid  = (size_t)BatchSize * NumGrids * NumGridPoints * sizeof(float);

    /* ---- Persistent LigGrid buffer: allocate only on first batch ---- */
    /* d_LigGrid_F is a C++ reference (void*&) — same as the CUDA version.
     * Writing d_LigGrid_F = ... writes directly through to the Fortran
     * module variable d_LigGrid.                                            */
    void* buf_LigGrid = NULL;
    if (BatchIdx == 1) {
        buf_LigGrid  = metal_buffer_alloc(sz_ligGrid);
        d_LigGrid_F  = buf_LigGrid;   /* write-through to Fortran's d_LigGrid */
    } else {
        buf_LigGrid  = d_LigGrid_F;   /* reuse existing buffer */
    }
    metal_buffer_zero(buf_LigGrid, sz_ligGrid);

    /* ---- Per-batch temporary buffers ---- */
    void* buf_rotCoors = metal_buffer_alloc(sz_rotCoors);
    void* buf_minCoors = metal_buffer_alloc(sz_minCoors);
    void* buf_params   = metal_buffer_alloc(sz_params);

    metal_buffer_copy_to(buf_rotCoors, LigRotamerCoors,        sz_rotCoors);
    metal_buffer_copy_to(buf_minCoors, LigRotamerMinCoors,     sz_minCoors);
    metal_buffer_copy_to(buf_params,   SelectAtomsParameters,  sz_params);

    /* ---- Scalar constant buffers (buffer indices 0–5 and 6 for dGrid) ---- */
    int   s_BatchSize  = RealBatchSize;
    int   s_NumAtoms   = NumAtoms;
    int   s_NumGrids   = NumGrids;
    int   s_Xn         = XGridNum;
    int   s_Yn         = YGridNum;
    int   s_Zn         = ZGridNum;
    float s_dGrid      = DGrid;

#define SCALAR_BUF(val, type) ({ \
    void* _b = metal_buffer_alloc(sizeof(type)); \
    metal_buffer_copy_to(_b, &(val), sizeof(type)); \
    _b; })

    void* buf_BatchSize = SCALAR_BUF(s_BatchSize, int);
    void* buf_NumAtoms  = SCALAR_BUF(s_NumAtoms,  int);
    void* buf_NumGrids  = SCALAR_BUF(s_NumGrids,  int);
    void* buf_Xn        = SCALAR_BUF(s_Xn,        int);
    void* buf_Yn        = SCALAR_BUF(s_Yn,         int);
    void* buf_Zn        = SCALAR_BUF(s_Zn,         int);
    void* buf_dGrid     = SCALAR_BUF(s_dGrid,      float);

#undef SCALAR_BUF

    /* ---- Buffer array (matches [[buffer(N)]] in metal_kernels.metal):
     *  0: BatchSize  1: NumAtoms  2: NumGrids
     *  3: Xn  4: Yn  5: Zn  6: dGrid
     *  7: rotCoors   8: atomPars  9: minCoors  10: LigGrid  */
    void* bufs[11] = {
        buf_BatchSize, buf_NumAtoms, buf_NumGrids,
        buf_Xn, buf_Yn, buf_Zn, buf_dGrid,
        buf_rotCoors, buf_params, buf_minCoors,
        buf_LigGrid
    };

    /* ---- Dispatch: one thread per rotamer ---- */
    void* pso = metal_get_pipeline("generateLigGrid");

    fprintf(stdout,
            "FFTDOCK Metal> calcLigGrid: batch %d, %d rotamers, "
            "%d atoms, %d grid types\n",
            BatchIdx, RealBatchSize, NumAtoms, NumGrids);

    metal_dispatch_1d(pso,
                      bufs, /*offsets=*/NULL, 11,
                      (size_t)BatchSize, /*tgroup=*/128);

    /* ---- Copy completed LigGrid back to host (for CPU FFT pass) ---- */
    metal_buffer_copy_from(buf_LigGrid, LigGrid, sz_ligGrid);

    /* ---- Release per-batch temporaries (LigGrid buffer is kept) ---- */
    metal_buffer_release(buf_rotCoors);
    metal_buffer_release(buf_minCoors);
    metal_buffer_release(buf_params);
    metal_buffer_release(buf_BatchSize);
    metal_buffer_release(buf_NumAtoms);
    metal_buffer_release(buf_NumGrids);
    metal_buffer_release(buf_Xn);
    metal_buffer_release(buf_Yn);
    metal_buffer_release(buf_Zn);
    metal_buffer_release(buf_dGrid);
}
