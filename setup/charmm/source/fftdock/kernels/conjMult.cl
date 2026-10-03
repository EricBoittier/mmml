/* gpu_compat.h is prepended at compile time.
 *
 * Conjugate multiplication of potential and ligand frequency-domain grids.
 *
 * N     = total number of complex elements (batch_size * num_grid * odist)
 * odist = complex elements per grid (xdim * ydim * (zdim/2+1))
 * dist  = odist * numOfGridsUsed
 *
 * Data is stored as interleaved float pairs: real at [2*i], imag at [2*i+1].
 *
 * For each ligand complex element i, the corresponding potential element
 * is at (i % dist) -- wrapping across grid types within each batch.
 */
KERNEL void conjMult(int N,
                     GLOBAL float * d_potential_F,
                     GLOBAL float * d_ligand_F,
                     int odist, int numOfGridsUsed)
{
  int dist = odist * numOfGridsUsed;
  for (int i = THREAD_ID; i < N; i += GRID_STRIDE)
  {
    int ip = i % dist;
    int il = 2 * i;
    int ip2 = 2 * ip;
    float x = d_ligand_F[il];
    float y = d_ligand_F[il + 1];
    d_ligand_F[il]     = x * d_potential_F[ip2]     + y * d_potential_F[ip2 + 1];
    d_ligand_F[il + 1] = x * d_potential_F[ip2 + 1] - y * d_potential_F[ip2];
  }
}
