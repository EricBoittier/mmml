#ifndef OPENMM_REFERENCEMSESKERNELFACTORY_H_
#define OPENMM_REFERENCEMSESKERNELFACTORY_H_


#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the 
 * MSES plugin.
 */

class ReferenceMSESKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEMSESKERNELFACTORY_H_*/
