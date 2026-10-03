/*
 * GPU compatibility macros for shared CUDA / OpenCL kernel source.
 *
 * OpenCL: the runtime compiler defines __OPENCL_VERSION__.
 * CUDA (nvrtc): __OPENCL_VERSION__ is absent; nvrtc provides
 *   blockIdx, blockDim, threadIdx, gridDim intrinsics.
 *
 * Usage:
 *   KERNEL void myKernel(int N, GLOBAL float * data) {
 *     for (int i = THREAD_ID; i < N; i += GRID_STRIDE) { ... }
 *   }
 */

#ifdef __OPENCL_VERSION__
  #define KERNEL       __kernel
  #define GLOBAL       __global
  #define THREAD_ID    ((int)get_global_id(0))
  #define GRID_STRIDE  ((int)get_global_size(0))
#else /* CUDA via nvrtc */
  #define KERNEL       extern "C" __global__
  #define GLOBAL
  #define THREAD_ID    ((int)(blockIdx.x * blockDim.x + threadIdx.x))
  #define GRID_STRIDE  ((int)(blockDim.x * gridDim.x))
#endif
