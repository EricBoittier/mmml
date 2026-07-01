#ifndef GBMV_KERNELS_H_
#define GBMV_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "GBMVForce.h"
#include "openmm/Kernel.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/KernelImpl.h"
#include <string>
#include <vector>

namespace OpenMMGBMV {

/**
 * This kernel is invoked by GBMVForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGBMVForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcGBMVForce";
    }
    CalcGBMVForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GBMVForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const GBMVForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GBMVForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const GBMVForce& force) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context      the context to copy parameters from
     * @param LambdaState  the array of the position, velocity, and force of lambdas
     */
    virtual void getLambdaInfo(OpenMM::ContextImpl& context, std::vector<double>& LambdaState) = 0;
    /**
     * Set lambda information to arrays: lambda positions, velocities, and forces
     *
     * @param context      the context to copy parameters from
     * @param LambdaState  the array of the position, velocity, and force of lambdas
     */
    virtual void setLambdaInfo(OpenMM::ContextImpl& context, std::vector<double>& LambdaState) = 0;
};

} // namespace OpenMMGBMV

#endif /*GBMV_KERNELS_H_*/
