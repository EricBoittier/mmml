#ifndef MSES_KERNELS_H_
#define MSES_KERNELS_H_

#include "MSESForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace OpenMMMSES {

/**
 * This kernel is invoked by MSESForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMSESForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcMSESForce";
    }
    CalcMSESForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MSESForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const MSESForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MSESForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const MSESForce& force) = 0;
};

} // namespace

#endif /*MSES_KERNELS_H_*/
