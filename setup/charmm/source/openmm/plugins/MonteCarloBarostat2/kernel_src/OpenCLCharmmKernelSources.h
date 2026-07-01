#ifndef OPENCL_CHARMM_KERNEL_SOURCES_H_
#define OPENCL_CHARMM_KERNEL_SOURCES_H_

#include <string>

namespace OpenMM {

/**
 * This class is a central holding place for the source code of OpenCL kernels.
 * The CMake build script inserts declarations into it based on the .cl files in the
 * kernels subfolder.
 */

class OpenCLCharmmKernelSources {
public:
static const std::string monteCarloBarostat2;

};

}

#endif

