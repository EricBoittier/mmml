#ifndef GBSW_KERNELS_H_
#define GBSW_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWForce.h"
#include "openmm/Kernel.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/KernelImpl.h"
#include <string>
#include <vector>

namespace OpenMMGBSW {

/**
 * This kernel is invoked by GBSWForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcGBSWForceKernel : public OpenMM::KernelImpl {
public:
  bool doingDynamics;

    static std::string Name() {
        return "CalcGBSWForce";
    }

 CalcGBSWForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform),
      doingDynamics(false) {
    }

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GBSWForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const GBSWForce& force) = 0;

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
     * @param force      the GBSWForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const GBSWForce& force) = 0;
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

} // namespace OpenMMGBSW

#endif /*GBSW_KERNELS_H_*/
