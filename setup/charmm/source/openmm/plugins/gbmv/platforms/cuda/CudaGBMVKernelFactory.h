#ifndef OPENMM_CUDAGBMVKERNELFACTORY_H_
#define OPENMM_CUDAGBMVKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "openmm/KernelFactory.h"
#include "openmm/KernelImpl.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBMV {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the GBMV plugin.
 */

class CudaGBMVKernelFactory : public OpenMM::KernelFactory {
public:
  OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAGBMVKERNELFACTORY_H_*/
