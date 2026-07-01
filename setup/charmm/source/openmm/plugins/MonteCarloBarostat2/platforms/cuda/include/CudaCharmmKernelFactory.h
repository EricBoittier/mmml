#ifndef CUDA_CHARMM_KERNEL_FACTORY_H_
#define CUDA_CHARMM_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class CudaCharmmKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

}
#endif

