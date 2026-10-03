/// \file gpucount.cpp
/// \brief How many GPUs this process can see.
///
/// Exposed to CHARMM scripts as ?NGPU, so a testcase or an input file can
/// decide for itself whether a GPU-backed feature can run here instead of
/// discovering it by aborting.  BLaDE, for instance, needs one GPU per
/// OpenMP thread and dies if it cannot have that.
///
/// The count is what CUDA reports to *this process*, so it honours
/// CUDA_VISIBLE_DEVICES -- which is the number a script actually cares
/// about, not the number of cards physically in the machine.
///
/// The real query is compiled only where the CUDA runtime is already
/// linked (see CMakeLists.txt); everywhere else this reports 0, which is
/// the honest answer for a build that cannot use a GPU at all.

#if CHARMM_HAVE_CUDA == 1
#include <cuda_runtime.h>
#endif

extern "C" int charmm_gpu_count(void)
{
#if CHARMM_HAVE_CUDA == 1
   // Initialising count matters: cudaGetDeviceCount does not write it on
   // the error path, so an uninitialised variable would leak stack garbage
   // on any machine without a working driver.  Treat every non-success
   // code as "no usable GPU" -- a missing driver, an unloaded nvidia_uvm,
   // or no card at all are all the same answer to the caller.
   int count = 0;
   if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
   if (count < 0) return 0;
   return count;
#else
   return 0;
#endif
}
