__kernel void correctEnergy(const int N, const int idist, __global float * d_lig_sum_f) { 
    for (int idx = get_group_id(0)*get_local_size(0) + get_local_id(0);
            idx < N;
            idx += get_local_size(0)*get_num_groups(0) ){
        d_lig_sum_f[idx] = d_lig_sum_f[idx] / idist;
    }
}
