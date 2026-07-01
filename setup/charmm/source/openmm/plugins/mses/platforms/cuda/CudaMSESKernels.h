#ifndef CUDA_MSES_KERNELS_H_
#define CUDA_MSES_KERNELS_H_

#include "MSESKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace OpenMMMSES {

/**
 * This kernel is invoked by MSESForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMSESForceKernel : public CalcMSESForceKernel {
public:
    CudaCalcMSESForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CalcMSESForceKernel(name, platform), hasInitializedKernel(false), cu(cu), system(system), params(NULL) {
    }
    ~CudaCalcMSESForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MSESForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MSESForce& force);
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
     * @param force      the MSESForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const MSESForce& force);
private:
    int numDistPair;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    OpenMM::CudaArray* params;
};

} // namespace

#endif /*CUDA_MSES_KERNELS_H_*/
