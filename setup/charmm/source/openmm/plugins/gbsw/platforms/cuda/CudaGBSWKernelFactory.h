#ifndef OPENMM_CUDAGBSWKERNELFACTORY_H_
#define OPENMM_CUDAGBSWKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "openmm/KernelFactory.h"
#include "openmm/KernelImpl.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the GBSW plugin.
 */

class CudaGBSWKernelFactory : public OpenMM::KernelFactory {
public:
  OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAGBSWKERNELFACTORY_H_*/
