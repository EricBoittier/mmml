#ifndef OPENMM_REFERENCEGBSWKERNELFACTORY_H_
#define OPENMM_REFERENCEGBSWKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM GBSW                              *
 * -------------------------------------------------------------------------- */

#include "GBSWKernels.h"

#include "openmm/Platform.h"
#include "openmm/KernelFactory.h"
#include "openmm/KernelImpl.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {

/**
 * This KernelFactory creates kernels for the reference implementation of the GBSW plugin.
 */

class ReferenceGBSWKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_REFERENCEGBSWKERNELFACTORY_H_*/
