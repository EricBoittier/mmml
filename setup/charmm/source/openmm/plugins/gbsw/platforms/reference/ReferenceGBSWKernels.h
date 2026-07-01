#ifndef REFERENCE_GBSW_KERNELS_H_
#define REFERENCE_GBSW_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWKernels.h"
#include "ReferenceGBSW.h"

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {

/**
 * This kernel is invoked by GBSWForce to calculate the forces acting on the system.
 */
class ReferenceCalcGBSWForceKernel : public CalcGBSWForceKernel {
public:
    ReferenceCalcGBSWForceKernel(std::string name, const OpenMM::Platform& platform) : CalcGBSWForceKernel(name, platform) {
    }
    ~ReferenceCalcGBSWForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GBSWForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GBSWForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GBSWForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GBSWForce& force);
    /**
     * Copy lambda information to arrays: lambda positions, velocities, and forces
     *
     * @param context            the context to copy parameters to
     * @param LambdaPosVelForce  the array of the position, velocity, and force of lambdas
     */
    void getLambdaInfo(OpenMM::ContextImpl& context, std::vector<double>& LambdaPosVelForce);
    /**
     * Set lambda information to arrays: lambda positions, velocities, and forces
     *
     * @param context            the context to copy parameters from
     * @param LambdaPosVelForce  the array of the position, velocity, and force of lambdas
     */
    void setLambdaInfo(OpenMM::ContextImpl& context, std::vector<double>& LambdaPosVelForce);
private:
    ReferenceGBSW* gbsw;
    std::vector<RealOpenMM> charges;
    bool isPeriodic;
};

} // namespace OpenMMGBSW

#endif /*REFERENCE_GBSW_KERNELS_H_*/
