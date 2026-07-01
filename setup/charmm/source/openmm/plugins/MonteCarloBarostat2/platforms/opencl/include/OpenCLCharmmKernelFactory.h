#ifndef OPENCL_CHARMM_KERNEL_FACTORY_H_
#define OPENCL_CHARMM_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class OpenCLCharmmKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

}
#endif

