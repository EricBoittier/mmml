#ifndef OPENMM_CUDAMSESKERNELFACTORY_H_
#define OPENMM_CUDAMSESKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the MSES plugin.
 */

class CudaMSESKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAMSESKERNELFACTORY_H_*/
