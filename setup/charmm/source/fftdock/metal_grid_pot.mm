/*
 * metal_grid_pot.mm
 *
 * Apple Metal implementation of calcPotGrid — receptor potential
 * energy grid generation.  Exposes the same C entry point as the
 * CUDA version (cuda_grid_pot.cu) so fftdock.F90 can call it
 * unchanged:
 *
 *   void calcPotGrid(NumGrids, NumAtoms, DGrid,
 *                    XGridLen, YGridLen, ZGridLen,
 *                    XMin, YMin, ZMin,
 *                    Fa, Fb, Gmax,
 *                    VdwEmax, ElecAttrEmax, ElecReplEmax,
 *                    CCELEC, ElecMode, Dielec,
 *                    SelectAtomsParameters, GridPot, GridRadii)
 *
 * Called from fftdock.F90 → Generate_Potential_Grid_GPU().
 *
 * What it does
 * ─────────────
 * Computes the protein potential energy grid on the GPU via the
 * generateProtGrid Metal kernel (metal_kernels.metal).  The grid has
 * NumGrids channels: (NumGrids-3) VdW probe channels, one H-bond donor,
 * one H-bond acceptor, and one electrostatics channel.  Results are
 * written back to the host-side GridPot array.
 *
 * Mapping to CUDA version (cuda_grid_pot.cu)
 * ───────────────────────────────────────────
 *  cudaMalloc           →  metal_buffer_alloc   (MTLStorageModeShared)
 *  cudaMemcpy H→D       →  metal_buffer_copy_to (memcpy into shared buf)
 *  cudaMemset           →  metal_buffer_zero
 *  cuLaunchKernel       →  metal_dispatch_1d
 *  cudaMemcpy D→H       →  metal_buffer_copy_from
 *  cudaFree             →  metal_buffer_release
 *  nvrtc compile        →  pre-compiled Metal library (metal_state_init)
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
void calcPotGrid(const int   NumGrids,
                 const int   NumAtoms,
                 const float DGrid,
                 const int   XGridLen,
                 const int   YGridLen,
                 const int   ZGridLen,
                 const float XMin,
                 const float YMin,
                 const float ZMin,
                 const float Fa,
                 const float Fb,
                 const float Gmax,
                 const float VdwEmax,
                 const float ElecAttrEmax,
                 const float ElecReplEmax,
                 const float CCELEC,
                 const int   ElecMode,
                 const float Dielec,
                 float*      SelectAtomsParameters,  /* [NumAtoms * 8]          */
                 float*      GridPot,                /* [NumGrids * X * Y * Z]  */
                 float*      GridRadii)              /* [NumGrids - 3]          */
{
    /* ---- Initialise Metal (no-op if already done) ---- */
    if (metal_state_init(0) != 0) {
        fprintf(stderr, "FFTDOCK Metal> calcPotGrid: Metal unavailable.\n");
        exit(1);
    }

    const int NUM_PARAMS_PER_ATOM = 8;
    const int NumVdwProbes = NumGrids - 3;

    int NumGridPoints = XGridLen * YGridLen * ZGridLen;

    /* Byte sizes */
    size_t sz_params  = (size_t)NumAtoms * NUM_PARAMS_PER_ATOM * sizeof(float);
    size_t sz_grid    = (size_t)NumGridPoints * NumGrids * sizeof(float);
    size_t sz_probes  = (size_t)NumVdwProbes * sizeof(float);

    /* ---- Allocate shared Metal buffers ---- */
    void* buf_params = metal_buffer_alloc(sz_params);
    void* buf_probes = metal_buffer_alloc(sz_probes);
    void* buf_grid   = metal_buffer_alloc(sz_grid);

    /* ---- Upload input data ---- */
    metal_buffer_copy_to(buf_params, SelectAtomsParameters, sz_params);
    metal_buffer_copy_to(buf_probes, GridRadii,             sz_probes);
    metal_buffer_zero(buf_grid, sz_grid);

    /* ---- Prepare scalar-constant buffers (one per kernel parameter) ----
     *
     * The generateProtGrid kernel binds each scalar as a separate buffer
     * (constant int& / constant float& at buffer indices 3–20).
     * We allocate one-element shared buffers for each.                   */

    int   s_NumGrids      = NumGrids;
    int   s_NumAtoms      = NumAtoms;
    int   s_Xn            = XGridLen;
    int   s_Yn            = YGridLen;
    int   s_Zn            = ZGridLen;
    float s_XMin          = XMin;
    float s_YMin          = YMin;
    float s_ZMin          = ZMin;
    float s_fa            = Fa;
    float s_fb            = Fb;
    float s_gmax          = Gmax;
    float s_dGrid         = DGrid;
    float s_vdwEmax       = VdwEmax;
    float s_elecReplEmax  = ElecReplEmax;
    float s_elecAttrEmax  = ElecAttrEmax;
    float s_ccelec        = CCELEC;
    int   s_elecMode      = ElecMode;
    float s_dielec        = Dielec;

#define SCALAR_BUF(val, type) ({ \
    void* _b = metal_buffer_alloc(sizeof(type)); \
    metal_buffer_copy_to(_b, &(val), sizeof(type)); \
    _b; })

    void* buf_NumGrids     = SCALAR_BUF(s_NumGrids,     int);
    void* buf_NumAtoms     = SCALAR_BUF(s_NumAtoms,     int);
    void* buf_Xn           = SCALAR_BUF(s_Xn,           int);
    void* buf_Yn           = SCALAR_BUF(s_Yn,           int);
    void* buf_Zn           = SCALAR_BUF(s_Zn,           int);
    void* buf_XMin         = SCALAR_BUF(s_XMin,         float);
    void* buf_YMin         = SCALAR_BUF(s_YMin,         float);
    void* buf_ZMin         = SCALAR_BUF(s_ZMin,         float);
    void* buf_fa           = SCALAR_BUF(s_fa,           float);
    void* buf_fb           = SCALAR_BUF(s_fb,           float);
    void* buf_gmax         = SCALAR_BUF(s_gmax,         float);
    void* buf_dGrid        = SCALAR_BUF(s_dGrid,        float);
    void* buf_vdwEmax      = SCALAR_BUF(s_vdwEmax,      float);
    void* buf_elecReplEmax = SCALAR_BUF(s_elecReplEmax, float);
    void* buf_elecAttrEmax = SCALAR_BUF(s_elecAttrEmax, float);
    void* buf_ccelec       = SCALAR_BUF(s_ccelec,       float);
    void* buf_elecMode     = SCALAR_BUF(s_elecMode,     int);
    void* buf_dielec       = SCALAR_BUF(s_dielec,       float);

#undef SCALAR_BUF

    /* ---- Build buffer array in the order the kernel expects ---- */
    /* Buffer indices match the [[buffer(N)]] annotations in metal_kernels.metal:
     *  0: probes   1: atoms   2: GridPot
     *  3: NumGrids 4: NumAtoms 5: Xn 6: Yn 7: Zn
     *  8: XMin  9: YMin  10: ZMin
     * 11: fa  12: fb  13: gmax  14: dGrid  15: vdwEmax
     * 16: elecReplEmax  17: elecAttrEmax  18: ccelec
     * 19: elecMode  20: dielec                                   */
    void* bufs[21] = {
        buf_probes,
        buf_params,
        buf_grid,
        buf_NumGrids,
        buf_NumAtoms,
        buf_Xn, buf_Yn, buf_Zn,
        buf_XMin, buf_YMin, buf_ZMin,
        buf_fa, buf_fb, buf_gmax, buf_dGrid,
        buf_vdwEmax, buf_elecReplEmax, buf_elecAttrEmax, buf_ccelec,
        buf_elecMode,
        buf_dielec
    };

    /* ---- Dispatch: one thread per grid point ---- */
    void* pso = metal_get_pipeline("generateProtGrid");

    fprintf(stdout,
            "FFTDOCK Metal> calcPotGrid: grid %d×%d×%d = %d points, "
            "%d grid types, %d atoms\n",
            XGridLen, YGridLen, ZGridLen, NumGridPoints, NumGrids, NumAtoms);

    metal_dispatch_1d(pso,
                      bufs, /*offsets=*/NULL, 21,
                      (size_t)NumGridPoints, /*tgroup=*/512);

    /* ---- Copy results back to host ---- */
    metal_buffer_copy_from(buf_grid, GridPot, sz_grid);

    /* ---- Diagnostic: dump first few values of the VdW (grid type 0)
     * and electrostatic (grid type NumGrids-1) channels.  Useful for
     * verifying the kernel formulas match CUDA after modifications.
     *
     * Reference output for the c46test/fftdock.inp test case
     * (18x18x18 grid, 29 grid types, 2603 atoms):
     *   pt 0:  VdW[0]=  -1.763118e+00   Elec[28]=   4.320566e+00
     *   pt 1:  VdW[0]=  -6.814780e-01   Elec[28]=   6.572777e+00
     *   pt 2:  VdW[0]=   3.034595e+00   Elec[28]=   8.415212e+00
     *   pt 3:  VdW[0]=   2.967057e+00   Elec[28]=   1.201593e+01
     *   pt 4:  VdW[0]=   3.361698e+00   Elec[28]=   3.074974e+01
     * (matches CUDA reference within ~0.0003% across the full grid)
     * - YWu */
    /*
    {
        int nPrint = (NumGridPoints < 5) ? NumGridPoints : 5;
        fprintf(stdout, "FFTDOCK Metal> calcPotGrid DEBUG: first %d values "
                "of grid type 0 (VdW) and grid type %d (elec):\n",
                nPrint, NumGrids - 1);
        for (int ip = 0; ip < nPrint; ip++) {
            fprintf(stdout, "  pt %d:  VdW[0]= %14.6e   Elec[%d]= %14.6e\n",
                    ip,
                    GridPot[0 * NumGridPoints + ip],
                    NumGrids - 1,
                    GridPot[(NumGrids - 1) * NumGridPoints + ip]);
        }
    }
    */

    /* ---- Release all temporary buffers ---- */
    metal_buffer_release(buf_params);
    metal_buffer_release(buf_probes);
    metal_buffer_release(buf_grid);
    metal_buffer_release(buf_NumGrids);
    metal_buffer_release(buf_NumAtoms);
    metal_buffer_release(buf_Xn);
    metal_buffer_release(buf_Yn);
    metal_buffer_release(buf_Zn);
    metal_buffer_release(buf_XMin);
    metal_buffer_release(buf_YMin);
    metal_buffer_release(buf_ZMin);
    metal_buffer_release(buf_fa);
    metal_buffer_release(buf_fb);
    metal_buffer_release(buf_gmax);
    metal_buffer_release(buf_dGrid);
    metal_buffer_release(buf_vdwEmax);
    metal_buffer_release(buf_elecReplEmax);
    metal_buffer_release(buf_elecAttrEmax);
    metal_buffer_release(buf_ccelec);
    metal_buffer_release(buf_elecMode);
    metal_buffer_release(buf_dielec);
}
