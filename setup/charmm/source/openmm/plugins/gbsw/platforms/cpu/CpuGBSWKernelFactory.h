#ifndef OPENMM_CPUGBSWKERNELFACTORY_H_
#define OPENMM_CPUGBSWKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace OpenMMGBSW {

/**
 * This KernelFactory creates all kernels for CpuPlatform.
 */

class CpuGBSWKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_CPUGBSWKERNELFACTORY_H_*/
