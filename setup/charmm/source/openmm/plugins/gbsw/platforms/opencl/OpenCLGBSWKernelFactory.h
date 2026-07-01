#ifndef OPENMM_OPENCLGBSWKERNELFACTORY_H_
#define OPENMM_OPENCLGBSWKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"
#include "openmm/Platform.h"
#include "openmm/KernelImpl.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the GBSW plugin.
 */

class OpenCLGBSWKernelFactory : public OpenMM::KernelFactory {
public:
  OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_OPENCLGBSWKERNELFACTORY_H_*/
