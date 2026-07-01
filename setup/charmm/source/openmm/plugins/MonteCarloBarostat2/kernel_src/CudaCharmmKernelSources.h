#ifndef CUDA_CHARMM_KERNEL_SOURCES_H_
#define CUDA_CHARMM_KERNEL_SOURCES_H_

#include <string>

namespace OpenMM {

/**
 * This class is a central holding place for the source code of CUDA kernels.
 * The CMake build script inserts declarations into it based on the .cl files in the
 * kernels subfolder.
 */

class CudaCharmmKernelSources {
public:
static const std::string monteCarloBarostat2;

};

}

#endif

