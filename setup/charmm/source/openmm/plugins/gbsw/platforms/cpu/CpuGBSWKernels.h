#ifndef OPENMM_CPUGBSWKERNELS_H_
#define OPENMM_CPUGBSWKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWKernels.h"
#include "CpuGBSWForce.h"

#include "openmm/cpu/CpuPlatform.h"
#include "openmm/System.h"

namespace OpenMMGBSW {

/**
 * This kernel is invoked by GBSWForce to calculate the forces acting on the system.
 */
class CpuCalcGBSWForceKernel : public CalcGBSWForceKernel {
public:
    CpuCalcGBSWForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CpuPlatform::PlatformData& data) : CalcGBSWForceKernel(name, platform),
            data(data) {
    }
    ~CpuCalcGBSWForceKernel();
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
    OpenMM::CpuPlatform::PlatformData& data;
    std::vector<std::pair<float, float> > particleParams;
    CpuGBSWForce gbsw;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_CPUGBSWKERNELS_H_*/

