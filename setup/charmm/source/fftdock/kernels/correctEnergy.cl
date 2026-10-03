/* gpu_compat.h is prepended at compile time */

KERNEL void correctEnergy(const int N, const int idist,
                          GLOBAL float * d_lig_sum_f) {
    for (int idx = THREAD_ID; idx < N; idx += GRID_STRIDE) {
        d_lig_sum_f[idx] = d_lig_sum_f[idx] / idist;
    }
}
