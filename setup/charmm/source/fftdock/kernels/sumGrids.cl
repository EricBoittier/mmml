/* gpu_compat.h is prepended at compile time.
 *
 * Sum ligand grids across grid types for each rotamer.
 *
 * N_sum_F        = total complex elements in output (batch_size * odist)
 * numOfGridsUsed = number of grid types to sum over
 * odist          = complex elements per grid (xdim * ydim * (zdim/2+1))
 *
 * Input  layout: d_ligand_F[rotamer][gridType][odist] as interleaved floats
 * Output layout: d_ligand_sum_F[rotamer][odist] as interleaved floats
 */
KERNEL void sumGrids(int N_sum_F,
                     GLOBAL float * d_ligand_F,
                     GLOBAL float * d_ligand_sum_F,
                     int numOfGridsUsed, int odist, int idist)
{
  for (int i = THREAD_ID; i < N_sum_F; i += GRID_STRIDE)
  {
    int idx_rotamer = i / odist;
    int idx_point   = i % odist;
    int is = 2 * i;
    float sum_re = 0.0f;
    float sum_im = 0.0f;
    for (int g = 0; g < numOfGridsUsed; g++)
    {
      int idx_F = 2 * ((idx_rotamer * numOfGridsUsed + g) * odist + idx_point);
      sum_re += d_ligand_F[idx_F];
      sum_im += d_ligand_F[idx_F + 1];
    }
    d_ligand_sum_F[is]     = sum_re;
    d_ligand_sum_F[is + 1] = sum_im;
  }
}
