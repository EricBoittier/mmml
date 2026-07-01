#ifndef OPENCL_GBSW_KERNELS_H_
#define OPENCL_GBSW_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWKernels.h"

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/opencl/OpenCLNonbondedUtilities.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {
 
 /**
 * This kernel is invoked by GBSWForce to calculate the forces acting on the system.
 */
class OpenCLCalcGBSWForceKernel : public CalcGBSWForceKernel {
public:
    OpenCLCalcGBSWForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::OpenCLContext& cl) : CalcGBSWForceKernel(name, platform), cl(cl),
            hasCreatedKernels(false), params(NULL), bornSum(NULL), longBornSum(NULL), bornRadii(NULL), bornForce(NULL),
            longBornForce(NULL), gbswChain(NULL) {
    }
    ~OpenCLCalcGBSWForceKernel();
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
    double cutoff, prefactor, surfaceAreaFactor;
    bool hasCreatedKernels;
    int maxTiles;
    OpenMM::OpenCLContext& cl;
    OpenMM::OpenCLArray* params;
    OpenMM::OpenCLArray* bornSum;
    OpenMM::OpenCLArray* longBornSum;
    OpenMM::OpenCLArray* bornRadii;
    OpenMM::OpenCLArray* bornForce;
    OpenMM::OpenCLArray* longBornForce;
    OpenMM::OpenCLArray* gbswChain;
    cl::Kernel computeBornSumKernel;
    cl::Kernel reduceBornSumKernel;
    cl::Kernel force1Kernel;
    cl::Kernel reduceBornForceKernel;
};

} // namespace OpenMMGBSW

#endif /*OPENCL_GBSW_KERNELS_H_*/
