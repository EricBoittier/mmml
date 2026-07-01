__kernel void conjMult(int N,
			 __global float * d_potential_F,
			 __global float * d_ligand_F,
			 int odist, int numOfGridsUsed)
{
  int dist = odist * numOfGridsUsed;
  int idx_p;
  float x, y;
  for (int idx_l = 2 * (get_group_id(0) * get_local_size(0) + get_local_id(0));
       idx_l < N;
       idx_l += 2 * get_local_size(0) * get_num_groups(0))
  {
    idx_p = 2 * (idx_l % dist);
    x = d_ligand_F[idx_l];
    y = d_ligand_F[idx_l + 1];
    d_ligand_F[idx_l] = x * d_potential_F[idx_p] + y * d_potential_F[idx_p + 1];
    d_ligand_F[idx_l + 1] = x * d_potential_F[idx_p + 1] - y * d_potential_F[idx_p];
  }
}
