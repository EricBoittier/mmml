/*
 * metal_kernels.metal
 *
 * Metal Shading Language compute kernels for CHARMM FFT-based docking.
 * GPU-side counterparts of the five CUDA / OpenCL kernels in
 * source/fftdock/kernels/*.cl (which CUDA compiles via nvrtc):
 *
 *   generateProtGrid  — protein potential energy grid (one thread / grid point)
 *   generateLigGrid   — ligand density grid           (one thread / rotamer)
 *   conjMult          — complex conjugate-multiply in frequency domain
 *   sumGrids          — sum frequency-domain grids over grid types
 *   correctEnergy     — post-IFFT normalisation (divide by transform size)
 *
 * Build-time compilation (driven by CMakeLists.txt -> add_custom_command):
 *   xcrun -sdk macosx metal -fno-fast-math \
 *         -c metal_kernels.metal -o metal_kernels.air
 *   xcrun -sdk macosx metallib metal_kernels.air -o metal_kernels.metallib
 *
 * The host code (source/metal/metal_util.mm) loads metal_kernels.metallib
 * at runtime via [device newLibraryWithURL:error:].
 *
 * IMPORTANT: -fno-fast-math is REQUIRED.  Metal's default fast math
 * backend gives pow()/sqrt() implementation-defined accuracy (potentially
 * ~100 ULP), and the VdW soft-core formula chains three pow() calls per
 * atom per grid point.  Without -fno-fast-math the protein potential
 * grid drifts ~5-15% from the cuFFT/CUDA reference.
 *
 * - YWu
 */

#include <metal_stdlib>
using namespace metal;

// ======================================================================
//  generateProtGrid  —  faithful Metal port of kernels/generateProtGrid.cl
//
//  Golden standard: source/fftdock/kernels/generateProtGrid.cl
//  (the .cl kernel that the CUDA build compiles via nvrtc on every
//  CUDA-machine run).  The standalone source/fftdock/cuda_grid_pot.cu
//  is NOT compiled — its `beta` formula differs slightly and was a
//  red herring during early Metal porting.
//
//  Each thread handles one grid point.  For every protein atom the
//  thread accumulates:
//    • Soft-core VdW (12-6 LJ with VdwEmax transition)
//    • Electrostatics (cdie or rdie, with soft-core rc/alpha cap)
//    • H-bond donor/acceptor polynomial well
//
//  Atom parameter layout  (8 floats per atom, same as CL):
//    [0] X   [1] Y   [2] Z   [3] eps   [4] vdwr (Rmin/2)
//    [5] charge   [6] H-donor flag (int as float)
//    [7] H-acceptor flag (int as float)
//
//  Grid layout:  GridPot[ g * NumGridPoints + GridGlobalId ]
//  - YWu
// ======================================================================
kernel void generateProtGrid(
    device const float*  d_probes     [[ buffer(0)  ]],  // VdW probe radii [NGrids-3]
    device const float*  d_parameter  [[ buffer(1)  ]],  // atom params [NAtoms * 8]
    device       float*  d_GridPot    [[ buffer(2)  ]],  // output [NGrids * X*Y*Z]
    constant     int&    NGrids       [[ buffer(3)  ]],
    constant     int&    NAtoms       [[ buffer(4)  ]],
    constant     int&    Xn           [[ buffer(5)  ]],
    constant     int&    Yn           [[ buffer(6)  ]],
    constant     int&    Zn           [[ buffer(7)  ]],
    constant     float&  XMin         [[ buffer(8)  ]],
    constant     float&  YMin         [[ buffer(9)  ]],
    constant     float&  ZMin         [[ buffer(10) ]],
    constant     float&  Fa           [[ buffer(11) ]],
    constant     float&  Fb           [[ buffer(12) ]],
    constant     float&  Gmax         [[ buffer(13) ]],
    constant     float&  DGrid        [[ buffer(14) ]],
    constant     float&  VdwEmax      [[ buffer(15) ]],
    constant     float&  ElecReplEmax [[ buffer(16) ]],
    constant     float&  ElecAttrEmax [[ buffer(17) ]],
    constant     float&  CCELEC_CHARMM [[ buffer(18) ]],
    constant     int&    ElecMode     [[ buffer(19) ]],
    constant     float&  Dielec       [[ buffer(20) ]],
    uint tid [[ thread_position_in_grid ]]
)
{
    int GridGlobalId  = (int)tid;
    int NumGridPoints = Xn * Yn * Zn;
    if (GridGlobalId >= NumGridPoints) return;

    // 3-D index from linear id:  id = (Gridx * Yn + Gridy) * Zn + Gridz
    // Matches the .cl kernel (generateProtGrid.cl) which is the actual
    // runtime golden standard (compiled via nvrtc on the CUDA machine).
    int Gridx =  GridGlobalId / (Yn * Zn);
    int Gridy = (GridGlobalId % (Yn * Zn)) / Zn;
    int Gridz = (GridGlobalId % (Yn * Zn)) % Zn;

    float x = XMin + (float)Gridx * DGrid;
    float y = YMin + (float)Gridy * DGrid;
    float z = ZMin + (float)Gridz * DGrid;

    const int NumFeaturePerAtom = 8;

    for (int n = 0; n < NAtoms; ++n) {
        float atomx = d_parameter[NumFeaturePerAtom * n + 0];
        float atomy = d_parameter[NumFeaturePerAtom * n + 1];
        float atomz = d_parameter[NumFeaturePerAtom * n + 2];
        float eps   = d_parameter[NumFeaturePerAtom * n + 3];
        float vdwr  = d_parameter[NumFeaturePerAtom * n + 4];
        float cg    = d_parameter[NumFeaturePerAtom * n + 5];
        int   hd    = (int)d_parameter[NumFeaturePerAtom * n + 6];
        int   ha    = (int)d_parameter[NumFeaturePerAtom * n + 7];

        float eps_sqrt = sqrt(fabs(eps));
        float r  = sqrt((atomx - x) * (atomx - x)
                      + (atomy - y) * (atomy - y)
                      + (atomz - z) * (atomz - z));
        float rh = r - Fb;

        // ---------- Electrostatics (NGrids - 1) ----------
        float eleconst = CCELEC_CHARMM * cg / Dielec;

        if (ElecMode == 0) {  // cdie
            if (cg > 0.0f) {
                float rc = 2.0f * eleconst / fabs(ElecReplEmax);
                if (r > rc) {
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        eleconst / r;
                } else {
                    float alpha = eleconst / (rc * rc);
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        (ElecReplEmax - alpha * r);
                }
            } else if (cg < 0.0f) {
                float rc = -2.0f * eleconst / fabs(ElecAttrEmax);
                if (r > rc) {
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        eleconst / r;
                } else {
                    float alpha = eleconst / (rc * rc);
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        (ElecAttrEmax - alpha * r);
                }
            }
        } else if (ElecMode == 1) {  // rdie
            if (cg > 0.0f) {
                float rc = sqrt(2.0f * fabs(eleconst / ElecReplEmax));
                if (r > rc) {
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        eleconst / (r * r);
                } else {
                    float alpha = fabs(ElecReplEmax / (2.0f * rc * rc));
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        (ElecReplEmax - alpha * (r * r));
                }
            } else if (cg < 0.0f) {
                float rc = sqrt(2.0f * fabs(eleconst / ElecAttrEmax));
                if (r > rc) {
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        eleconst / (r * r);
                } else {
                    float alpha = fabs(ElecAttrEmax / (2.0f * rc * rc));
                    d_GridPot[NumGridPoints * (NGrids - 1) + GridGlobalId] +=
                        (ElecAttrEmax + alpha * (r * r));
                }
            }
        }

        // ---------- Hydrogen donor grid (NGrids - 3) ----------
        {
            int gridIdx = NGrids - 3;
            float dener = ((rh * rh) * Fa + Gmax) * (float)hd;
            if (dener < 0.0f) {
                d_GridPot[NumGridPoints * gridIdx + GridGlobalId] += dener;
            }
        }

        // ---------- Hydrogen acceptor grid (NGrids - 2) ----------
        {
            int gridIdx = NGrids - 2;
            float aener = ((rh * rh) * Fa + Gmax) * (float)ha;
            if (aener < 0.0f) {
                d_GridPot[NumGridPoints * gridIdx + GridGlobalId] += aener;
            }
        }

        // ---------- Van der Waals (soft-core LJ) ----------
        for (int gridIdx = 0; gridIdx < NGrids - 3; ++gridIdx) {
            float radii   = d_probes[gridIdx];
            float r_min   = vdwr + radii;
            float vdwconst = 1.0f + sqrt(1.0f + 0.5f * fabs(VdwEmax) / eps_sqrt);
            float rc       = r_min * pow(vdwconst, -1.0f / 6.0f);
            // beta uses (vdwconst²-2*vdwconst), per the .cl kernel that
            // CUDA actually compiles via nvrtc.  The standalone
            // cuda_grid_pot.cu has (vdwconst²-vdwconst) — different
            // formula, do NOT use it as the reference.   - YWu
            float beta     = 24.0f * eps_sqrt / VdwEmax
                           * (vdwconst * vdwconst - 2.0f * vdwconst);
            // alpha is computed in CL but not used in the else branch
            // (only beta matters for the soft-core region)
            if (r > rc) {
                // Use explicit multiplication instead of pow() for integer
                // exponents — avoids the exp2(n*log2(x)) path that differs
                // between Metal (≤4 ULP) and CUDA powf (≤2 ULP).
                float s   = r_min / r;
                float s2  = s * s;
                float s6  = s2 * s2 * s2;
                float s12 = s6 * s6;
                d_GridPot[NumGridPoints * gridIdx + GridGlobalId] +=
                    eps_sqrt * (s12 - 2.0f * s6);
            } else {
                d_GridPot[NumGridPoints * gridIdx + GridGlobalId] +=
                    VdwEmax * (1.0f - 0.5f * pow(r / rc, beta));
            }
        }
    }
}


// ======================================================================
//  generateLigGrid  —  faithful Metal port of kernels/generateLigGrid.cl
//
//  One thread per rotamer.  Each thread distributes every ligand atom's
//  contribution via TRILINEAR interpolation across the 8 surrounding
//  grid corners — matching the CL/CUDA kernel exactly.
//
//  Atom parameter layout (4 floats per atom, same as CL):
//    [0] charge   [1] eps   [2] vdwr (unused here)   [3] vdw_grid_idx (as float)
//
//  energyFactor per grid channel j:
//    j == vdw_grid_idx  →  sqrt(fabs(eps))
//    j == numGrids - 1  →  charge
//    else               →  0
//  - YWu
// ======================================================================
kernel void generateLigGrid(
    constant     int&   numRotamers [[ buffer(0)  ]],
    constant     int&   NAtoms      [[ buffer(1)  ]],
    constant     int&   numGrids    [[ buffer(2)  ]],
    constant     int&   Xn          [[ buffer(3)  ]],
    constant     int&   Yn          [[ buffer(4)  ]],
    constant     int&   Zn          [[ buffer(5)  ]],
    constant     float& DGrid       [[ buffer(6)  ]],
    device const float* d_rotamersCoor [[ buffer(7)  ]],  // [BS][NAtoms][3]
    device const float* d_par          [[ buffer(8)  ]],  // [NAtoms][4]
    device const float* d_GridMinCoor  [[ buffer(9)  ]],  // [BS][3]
    device       float* d_LigGrid      [[ buffer(10) ]],  // [BS][numGrids][X*Y*Z]
    uint tid [[ thread_position_in_grid ]]
)
{
    int globalId = (int)tid;
    if (globalId >= numRotamers) return;

    int xlen = Xn;
    int ylen = Yn;
    int zlen = Zn;
    int NumGridPoints = xlen * ylen * zlen;
    int rotamerOffset = globalId * numGrids * NumGridPoints;

    for (int i = 0; i < NAtoms; ++i) {
        float dx = d_rotamersCoor[globalId * 3 * NAtoms + 3 * i + 0]
                 - d_GridMinCoor[globalId * 3 + 0];
        float dy = d_rotamersCoor[globalId * 3 * NAtoms + 3 * i + 1]
                 - d_GridMinCoor[globalId * 3 + 1];
        float dz = d_rotamersCoor[globalId * 3 * NAtoms + 3 * i + 2]
                 - d_GridMinCoor[globalId * 3 + 2];

        int idx_x = (int)floor(dx / DGrid);
        int idx_y = (int)floor(dy / DGrid);
        int idx_z = (int)floor(dz / DGrid);

        float xRatio = (dx - (float)idx_x * DGrid) / DGrid;
        float yRatio = (dy - (float)idx_y * DGrid) / DGrid;
        float zRatio = (dz - (float)idx_z * DGrid) / DGrid;

        float charge       = d_par[i * 4 + 0];
        float eps           = d_par[i * 4 + 1];
        // float vdwr       = d_par[i * 4 + 2];  // unused in CL kernel
        int   vdw_grid_idx = (int)d_par[i * 4 + 3];

        float energyFactor = 0.0f;
        for (int j = 0; j < numGrids; ++j) {
            if (j == vdw_grid_idx) {
                energyFactor = sqrt(fabs(eps));
            } else if (j == numGrids - 1) {
                energyFactor = charge;
            } else {
                energyFactor = 0.0f;
            }

            int gridTypeOffset = j * NumGridPoints;

            // Trilinear interpolation — 8 corners, matching CL exactly
            // (0,0,0)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      (idx_x * ylen + idx_y) * zlen + idx_z] +=
                (1.0f - xRatio) * (1.0f - yRatio) * (1.0f - zRatio) * energyFactor;
            // (0,0,1)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      (idx_x * ylen + idx_y) * zlen + idx_z + 1] +=
                (1.0f - xRatio) * (1.0f - yRatio) * zRatio * energyFactor;
            // (0,1,0)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      (idx_x * ylen + idx_y + 1) * zlen + idx_z] +=
                (1.0f - xRatio) * yRatio * (1.0f - zRatio) * energyFactor;
            // (0,1,1)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      (idx_x * ylen + idx_y + 1) * zlen + idx_z + 1] +=
                (1.0f - xRatio) * yRatio * zRatio * energyFactor;
            // (1,0,0)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      ((idx_x + 1) * ylen + idx_y) * zlen + idx_z] +=
                xRatio * (1.0f - yRatio) * (1.0f - zRatio) * energyFactor;
            // (1,0,1)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      ((idx_x + 1) * ylen + idx_y) * zlen + idx_z + 1] +=
                xRatio * (1.0f - yRatio) * zRatio * energyFactor;
            // (1,1,0)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      ((idx_x + 1) * ylen + idx_y + 1) * zlen + idx_z] +=
                xRatio * yRatio * (1.0f - zRatio) * energyFactor;
            // (1,1,1)
            d_LigGrid[rotamerOffset + gridTypeOffset +
                      ((idx_x + 1) * ylen + idx_y + 1) * zlen + idx_z + 1] +=
                xRatio * yRatio * zRatio * energyFactor;
        }
    }
}


// ======================================================================
//  conjMult  —  faithful Metal port of kernels/conjMult.cl
//
//  In-place frequency-domain operation that produces the cross-correlation
//  spectrum:
//      lig_F[b][g][k]  ←  conj( lig_F[b][g][k] ) × pot_F[g][k]
//
//  After the C2R inverse FFT this gives the docking energy as a function
//  of translation.  The minimum energy at translation t is the best fit.
//
//  CRITICAL — IMAGINARY-PART SIGN
//  ───────────────────────────────
//  The .cl/CUDA reference computes  result = conj(lig) * pot, with
//      result.y = lig.x * pot.y - lig.y * pot.x
//  Computing  lig * conj(pot)  instead (the COMPLEX CONJUGATE of the
//  correct value) flips the sign of the imaginary part:
//      result.y = lig.y * pot.x - lig.x * pot.y    (WRONG)
//  Conjugating the spectrum REFLECTS the spatial IFFT output through
//  the origin (modulo N), placing the energy minimum at (N-t) instead
//  of t.  In the c46test/fftdock.inp test (18×18×18 grid, true binding
//  pose near (5,4,4)) this manifested as the global minimum appearing
//  at (13,14,14) — RMSD 3.6 instead of the expected ~0.37.
//
//  Match the .cl formula EXACTLY.  - YWu
//
//  Complex values are stored interleaved: float2 where .x = real, .y = imag.
//  Grid:  N = batch_size × num_grids × odist   (one thread per complex element)
//  pot_F has only  num_grids × odist  elements (same for every batch item).
// ======================================================================
kernel void conjMult(
    constant int&    N         [[ buffer(0) ]],
    device   float2* pot_F     [[ buffer(1) ]],  // [num_grids * odist]
    device   float2* lig_F     [[ buffer(2) ]],  // [N] — modified in-place
    constant int&    odist     [[ buffer(3) ]],
    constant int&    num_grids [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
)
{
    if ((int)tid >= N) return;

    int k     = (int)tid % odist;
    int g     = ((int)tid / odist) % num_grids;

    float2 p = pot_F[g * odist + k];
    float2 l = lig_F[(int)tid];

    // CUDA/CL formula: result = conj(lig) * pot
    //   conj(l) * p = (l.x - i·l.y)(p.x + i·p.y)
    //               = l.x*p.x + l.y*p.y  +  i*(l.x*p.y - l.y*p.x)
    //
    // NOTE: The imaginary part sign matters!  Computing lig*conj(pot)
    // instead (the complex conjugate) causes the IFFT spatial output
    // to be REFLECTED — minimum at index (N-t) instead of t — which
    // gives wrong docking pose.  Match CUDA/CL exactly:
    //   cuda_batch_fft.cu:36-37  →  result.y = lig.x*pot.y - lig.y*pot.x
    float2 result;
    result.x = l.x * p.x + l.y * p.y;
    result.y = l.x * p.y - l.y * p.x;

    lig_F[(int)tid] = result;
}


// ======================================================================
//  sumGrids  —  faithful Metal port of kernels/sumGrids.cl
//
//  Sums the frequency-domain cross-correlation result across all grid
//  types (VdW probe channels + electrostatics) into a single complex
//  spectrum per batch element:
//
//      sum_F[b][k]  =  Σ_{g=0}^{num_grids-1}  lig_F[b][g][k]
//
//  Grid:  N = batch_size × odist   (one thread per output complex element)
//  - YWu
// ======================================================================
kernel void sumGrids(
    constant int&    N         [[ buffer(0) ]],
    device   float2* lig_F     [[ buffer(1) ]],  // [batch * num_grids * odist]
    device   float2* sum_F     [[ buffer(2) ]],  // [batch * odist] — output
    constant int&    num_grids [[ buffer(3) ]],
    constant int&    odist     [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
)
{
    if ((int)tid >= N) return;

    int b = (int)tid / odist;
    int k = (int)tid % odist;

    float2 s = float2(0.0f, 0.0f);
    for (int g = 0; g < num_grids; g++)
        s += lig_F[b * num_grids * odist + g * odist + k];

    sum_F[(int)tid] = s;
}


// ======================================================================
//  correctEnergy  —  faithful Metal port of kernels/correctEnergy.cl
//
//  Normalises the real-valued energy grid after the inverse FFT.
//  Neither cuFFT nor MPSGraph (with scalingMode = None) normalises
//  automatically; dividing by idist = Nx × Ny × Nz recovers the
//  physical correlation value.
//
//  Grid:  N = batch_size × idist
//  - YWu
// ======================================================================
kernel void correctEnergy(
    constant int&   N     [[ buffer(0) ]],
    constant int&   idist [[ buffer(1) ]],
    device   float* data  [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]]
)
{
    if ((int)tid >= N) return;
    data[(int)tid] /= (float)idist;
}
