/*
 * metal_fftdock_gpu.h
 *
 * Public C interface for the Apple Metal FFT-docking pipeline used in
 * CHARMM's FFTDOCK module (KEY_METAL build path).  All per-batch GPU
 * operations are encoded into a single MTLCommandBuffer and committed
 * once, keeping every intermediate buffer GPU-resident.  Mirrors the
 * concept of cuFFT/CUDA's rigid_FFT_dock but uses MPSGraph FFTs.
 *
 * Pipeline per batch (single command buffer)
 * ──────────────────────────────────────────
 *  1. R2C FFT on ligand grids   (MPSGraph realToHermiteanFFT,
 *                                resultsDictionary: → ctx->buf_lig_fft)
 *  2. conjMult                  (Metal compute kernel, in-place on
 *                                buf_lig_fft;  result = conj(lig)*pot)
 *  3. sumGrids                  (Metal compute kernel,
 *                                buf_lig_fft → buf_sum_fft)
 *  4. C2R inverse FFT           (MPSGraph HermiteanToRealFFT,
 *                                resultsDictionary: → ctx->buf_energy)
 *  5. correctEnergy             (Metal compute kernel, in-place on
 *                                buf_energy: divide by idist=Nx*Ny*Nz)
 *  6. [cmdBuf commit + wait]
 *  7. memcpy buf_energy → host EnergyGrid (only CPU↔GPU touch in hot path)
 *
 * All persistent buffers are MTLStorageModeShared (Apple Silicon unified
 * memory) so MPSGraph can write FFT output directly into them via
 * resultsDictionary: — no MPSNDArray.resource extraction needed (that
 * private API is not available on every macOS / SDK combination).
 *
 * Requires macOS 14+ for MPSGraph FFT support.  Hard-aborts with a clear
 * error message if invoked on older macOS or if built without the macOS
 * 14 SDK headers (this avoids silently returning a NULL context that
 * would crash later inside metal_fftdock_run_batch).
 * - YWu
 */

#ifndef METAL_FFTDOCK_GPU_H
#define METAL_FFTDOCK_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------
 * metal_fftdock_setup
 *
 * Initialises the Metal device + command queue, builds the two
 * MPSGraph FFT graphs (R2C forward and C2R inverse), and allocates
 * the four persistent shared MTLBuffers (potential FFT, ligand FFT,
 * sum FFT, energy output).  Call once before the docking loop.
 *
 *   gpu_id     — Metal device index passed to metal_state_init
 *                (currently selects the default device on Apple Silicon)
 *   xdim/ydim/zdim  — 3-D grid dimensions (Nx, Ny, Nz)
 *   batch_size — rotamers per batch (SIZB parameter from FFTG LCON)
 *   num_grid   — number of grid channels (num_vdw_grid_used + 1)
 *   ctx_out    — receives the opaque MetalDockContext* handle
 *
 * Hard-aborts if MPSGraph FFT is unavailable (macOS < 14 or built
 * without macOS-14 SDK).  This is intentional: silently returning a
 * NULL context would crash later inside metal_fftdock_run_batch with
 * a less obvious message. - YWu
 * ------------------------------------------------------------------ */
void metal_fftdock_setup(int   gpu_id,
                          int   xdim,
                          int   ydim,
                          int   zdim,
                          int   batch_size,
                          int   num_grid,
                          void** ctx_out);

/* ------------------------------------------------------------------
 * metal_fftdock_upload_potential
 *
 * Uploads the receptor potential grid to the GPU and runs its forward
 * R2C FFT.  Output lands directly in ctx->buf_pot_fft (shared buffer)
 * via MPSGraph's resultsDictionary: — no private-memory extraction
 * required.  The transformed potential is reused unchanged for every
 * subsequent batch.
 *
 *   ctx            — handle from metal_fftdock_setup
 *   grid_potential — host array [num_grid * xdim * ydim * zdim]
 *                    (Used_GridPot from the Fortran caller)
 * - YWu
 * ------------------------------------------------------------------ */
void metal_fftdock_upload_potential(void*  ctx,
                                     float* grid_potential);

/* ------------------------------------------------------------------
 * metal_fftdock_run_batch
 *
 * Executes the per-batch docking pipeline (R2C → conjMult → sumGrids
 * → C2R → correctEnergy) inside a single MPSCommandBuffer; commits
 * once at the end; copies the resulting energy grid back to the host
 * EnergyGrid array.  No CPU↔GPU traffic between pipeline stages.
 *
 *   ctx           — handle from metal_fftdock_setup
 *   d_lig_grid_f  — retained id<MTLBuffer> produced by calcLigGrid,
 *                   passed through Fortran's d_LigGrid c_ptr by value
 *                   Layout: [batch_size * num_grid * xdim * ydim * zdim]
 *   energy_grid   — host output [batch_size * xdim * ydim * zdim],
 *                   scanned by Fortran's Translate_Lig_Rotamer to find
 *                   the best translation per rotamer
 * - YWu
 * ------------------------------------------------------------------ */
void metal_fftdock_run_batch(void*  ctx,
                              void*  d_lig_grid_f,
                              float* energy_grid);

/* ------------------------------------------------------------------
 * metal_fftdock_cleanup
 *
 * Releases the two MPSGraph graphs, all four persistent MTLBuffers,
 * and the MetalDockContext struct itself.  Sets *ctx_out to NULL on
 * return so the Fortran-side flag c_associated(metal_ctx) reports
 * the cleanup correctly.
 * - YWu
 * ------------------------------------------------------------------ */
void metal_fftdock_cleanup(void** ctx_out);

#ifdef __cplusplus
}
#endif

#endif /* METAL_FFTDOCK_GPU_H */
