#ifndef CUDA_GBSW_KERNELS_H_
#define CUDA_GBSW_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWKernels.h"

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMMGBSW {

/**
 * This kernel is invoked by GBSWForce to calculate the forces acting on the system.
 */
class CudaCalcGBSWForceKernel : public CalcGBSWForceKernel {
public:
  CudaCalcGBSWForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) : 
    CalcGBSWForceKernel(name, platform), cu(cu), 
    hasCreatedKernels(false), params(NULL), bornRadii(NULL), 
    bornForce(NULL), gbswChain(NULL), GridDimXYZ(NULL), 
    lookupTable(NULL), QuadPts(NULL), QuadPtWeights(NULL), nGBSWchainAtoms(NULL),
    cphmdForce(NULL) {
  }
    ~CudaCalcGBSWForceKernel();
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

    bool hasCreatedKernels, usingCPHMD, usingMembrane, usingCutoff, usingCutoffOffset, *writeGroup;
  
    unsigned int lookupMemory, numThreads, maxAtomsPerVoxel, numRadii, numRadii_H,
        numRadii_HEAVY, skipRadii_H, skipRadii_HEAVY, numAngles, 
        maxSfNeighbors, maxTiles, numTitratingGroups;
    unsigned int gridDimlocal[4], timestepCounter, ntimesteps, outputFrequency, nchar;
    double cutoff, prefactor, swLen, PBradiusMod, deltaR, RBuffer, Rmin, AA0, AA1, preintegrate1,
        preintegrate2, preintegrate3, preintegrate4, sgamma, kappa, saltFactorA, saltFactorB,
        randomForceScale, timeFactor, massTimeFactor, cphmdGamma, cphmdGBSWFac, 
        cphmdCoulombicFac, membThickness, membSwLen, cutoffOffset, rfDielectric;
    std::vector<int2> exceptionFixes;
    float4* LambdaXtheta;
    FILE *cphmdFileID;
    char lineBuffer[100];
    
    std::vector<void*> sysExtremaArgs, resetLookupTableArgs, fillLookupTableArgs,
        sortLookupTableArgs, calcBornRArgs, computeGBSWForceArgs, reduceGBSWForceArgs,
        cphmdLambdaDynamicsArgs, cphmdApplyAtomIndexChargesArgs, cphmdApplyAtomIndexForcesArgs;
    
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray* params;
    OpenMM::CudaArray* bornRadii;
    OpenMM::CudaArray* GridDimXYZ;
    OpenMM::CudaArray* lookupTable;
    OpenMM::CudaArray* QuadPts;
    OpenMM::CudaArray* QuadPtWeights;
    OpenMM::CudaArray* nGBSWchainAtoms;
    OpenMM::CudaArray* gbswChain;
    OpenMM::CudaArray* bornForce;
    
    OpenMM::CudaArray* cphmdRandSeed;
    OpenMM::CudaArray* cphmdRandNum;
    OpenMM::CudaArray* cphmdLambdaXtheta;
    OpenMM::CudaArray* cphmdForce;
    OpenMM::CudaArray* cphmdLambdaXvf;
    OpenMM::CudaArray* cphmdLambdaXvOld;
    OpenMM::CudaArray* cphmdAtomRanges;
    OpenMM::CudaArray* cphmdUphUbarr;
    OpenMM::CudaArray* cphmdUmod0123;
    OpenMM::CudaArray* cphmdUmod4567;
    OpenMM::CudaArray* cphmdAtomQfac;
    OpenMM::CudaArray* cphmdChargeStates;
    OpenMM::CudaArray* cphmdExclusions;
    
    CUfunction calcSysExtremaKernel;
    CUfunction resetLookupTableKernel;
    CUfunction fillLookupTableKernel;
    CUfunction sortLookupTableKernel;
    CUfunction calcBornRKernel_hydrogen;
    CUfunction calcBornRKernel_heavy;
    CUfunction computeGBSWForceKernel;
    CUfunction reduceGBSWForceKernel;
    CUfunction cphmdLambdaDynamicsKernel;
    CUfunction cphmdApplyAtomIndexChargesKernel;
    CUfunction cphmdApplyAtomIndexForcesKernel;
};

} // namespace OpenMMGBSW

#endif /*CUDA_GBSW_KERNELS_H_*/
