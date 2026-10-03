#ifndef CUDADIRECTFORCEPARAMS_H
#define CUDADIRECTFORCEPARAMS_H

// Threads per block for the 1-4 (calc_14_force) kernels.  Single source of
// truth, included by both the launchers (via CudaDirectForceKernels.h) and the
// kernel definitions (via CudaDirectForce14_util.h) so the launch block size
// and the kernels' __launch_bounds__ cannot drift apart.  Binding the launch
// bound to the block size makes ptxas cap registers so the launch cannot fail
// with cudaErrorLaunchOutOfResources ("too many resources requested for
// launch") under a driver/arch JIT of the shipped PTX.  See issue #5.
const int nthread14 = 512;

#endif // CUDADIRECTFORCEPARAMS_H
