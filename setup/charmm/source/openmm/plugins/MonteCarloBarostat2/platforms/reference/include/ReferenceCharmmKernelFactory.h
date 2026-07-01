#ifndef REFERENCE_CHARMM_KERNEL_FACTORY_H_
#define REFERENCE_CHARMM_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"
#include "openmm/KernelImpl.h"

namespace OpenMM {

class ReferenceCharmmKernelFactory : public KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

}
#endif

