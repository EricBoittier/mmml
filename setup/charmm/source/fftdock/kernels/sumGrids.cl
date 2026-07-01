__kernel void sumGrids(int N_sum_F,
                       __global float * d_ligand_F,
                       __global float * d_ligand_sum_F,
                       int numOfGridsUsed, int odist, int idist)
{
  for (int idx_sum_F = 2 * (get_group_id(0) * get_local_size(0) + get_local_id(0));
       idx_sum_F < N_sum_F;
       idx_sum_F += 2 * (get_local_size(0) * get_num_groups(0)))
  {
    int idx_rotamer = 2 * (idx_sum_F / odist);
    int idx_point = 2 * (idx_sum_F % odist);
    d_ligand_sum_F[idx_sum_F] = 0;
    d_ligand_sum_F[idx_sum_F + 1] = 0;
    for (int idx_grid_type = 0; idx_grid_type < numOfGridsUsed; idx_grid_type++)
    {
      int idx_F = (idx_rotamer * numOfGridsUsed + idx_grid_type) * odist + idx_point;
      d_ligand_sum_F[idx_sum_F] += d_ligand_F[idx_F];
      d_ligand_sum_F[idx_sum_F + 1] += d_ligand_F[idx_F + 1];
    }
    d_ligand_sum_F[idx_sum_F] = d_ligand_sum_F[idx_sum_F];
    d_ligand_sum_F[idx_sum_F + 1] = d_ligand_sum_F[idx_sum_F + 1];
  }  
}
