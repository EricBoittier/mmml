/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "OpenCLGBSWKernels.h"
#include "OpenCLGBSWKernelSources.h"

#include "openmm/opencl/OpenCLForceInfo.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

static void setPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
  if (cl.getUseDoublePrecision())
    kernel.setArg<mm_double4>(index, cl.getPeriodicBoxSizeDouble());
  else
    kernel.setArg<mm_float4>(index, cl.getPeriodicBoxSize());
}

static void setInvPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
  if (cl.getUseDoublePrecision())
    kernel.setArg<mm_double4>(index, cl.getInvPeriodicBoxSizeDouble());
  else
    kernel.setArg<mm_float4>(index, cl.getInvPeriodicBoxSize());
}

static void setPeriodicBoxArgs(OpenCLContext& cl, cl::Kernel& kernel, int index) {
    if (cl.getUseDoublePrecision()) {
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxSizeDouble());
        kernel.setArg<mm_double4>(index++, cl.getInvPeriodicBoxSizeDouble());
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxVecXDouble());
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxVecYDouble());
        kernel.setArg<mm_double4>(index, cl.getPeriodicBoxVecZDouble());
    }
    else {
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxSize());
        kernel.setArg<mm_float4>(index++, cl.getInvPeriodicBoxSize());
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxVecX());
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxVecY());
        kernel.setArg<mm_float4>(index, cl.getPeriodicBoxVecZ());
    }
}

class OpenCLGBSWForceInfo : public OpenCLForceInfo {
public:
    OpenCLGBSWForceInfo(int requiredBuffers, const GBSWForce& force) : OpenCLForceInfo(requiredBuffers), force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, radius1, radius2;
        force.getParticleParameters(particle1, charge1, radius1);
        force.getParticleParameters(particle2, charge2, radius2);
        return (charge1 == charge2 && radius1 == radius2);
    }
private:
    const GBSWForce& force;
};

OpenCLCalcGBSWForceKernel::~OpenCLCalcGBSWForceKernel() {
    if (params != NULL)
        delete params;
    if (bornSum != NULL)
        delete bornSum;
    if (longBornSum != NULL)
        delete longBornSum;
    if (bornRadii != NULL)
        delete bornRadii;
    if (bornForce != NULL)
        delete bornForce;
    if (longBornForce != NULL)
        delete longBornForce;
    if (gbswChain != NULL)
        delete gbswChain;
}

void OpenCLCalcGBSWForceKernel::initialize(const System& system, const GBSWForce& force) {
    throw OpenMMException("GBSWForce is not yet implimented on the OpenCL platform");
    if (cl.getPlatformData().contexts.size() > 1)
        throw OpenMMException("GBSWForce does not support using multiple OpenCL devices");
    OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
    params = OpenCLArray::create<mm_float2>(cl, cl.getPaddedNumAtoms(), "gbswParams");
    int elementSize = (cl.getUseDoublePrecision() ? sizeof(cl_double) : sizeof(cl_float));
    bornRadii = new OpenCLArray(cl, cl.getPaddedNumAtoms(), elementSize, "bornRadii");
    gbswChain = new OpenCLArray(cl, cl.getPaddedNumAtoms(), elementSize, "gbswChain");
    if (cl.getSupports64BitGlobalAtomics()) {
        longBornSum = OpenCLArray::create<cl_long>(cl, cl.getPaddedNumAtoms(), "longBornSum");
        longBornForce = OpenCLArray::create<cl_long>(cl, cl.getPaddedNumAtoms(), "longBornForce");
        bornForce = new OpenCLArray(cl, cl.getPaddedNumAtoms(), elementSize, "bornForce");
        cl.addAutoclearBuffer(*longBornSum);
        cl.addAutoclearBuffer(*longBornForce);
    }
    else {
        bornSum = new OpenCLArray(cl, cl.getPaddedNumAtoms()*nb.getNumForceBuffers(), elementSize, "bornSum");
        bornForce = new OpenCLArray(cl, cl.getPaddedNumAtoms()*nb.getNumForceBuffers(), elementSize, "bornForce");
        cl.addAutoclearBuffer(*bornSum);
        cl.addAutoclearBuffer(*bornForce);
    }
    vector<mm_float4> posqf(cl.getPaddedNumAtoms());
    vector<mm_double4> posqd(cl.getPaddedNumAtoms());
    vector<mm_float2> paramsVector(cl.getPaddedNumAtoms(), mm_float2(1,1));
    const double dielectricOffset = 0.009;
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        radius -= dielectricOffset;
        paramsVector[i] = mm_float2((float) radius, (float) (1.0*radius));
        if (cl.getUseDoublePrecision())
            posqd[i] = mm_double4(0, 0, 0, charge);
        else
            posqf[i] = mm_float4(0, 0, 0, (float) charge);
    }
    if (cl.getUseDoublePrecision())
        cl.getPosq().upload(posqd);
    else
        cl.getPosq().upload(posqf);
    params->upload(paramsVector);
    prefactor = -ONE_4PI_EPS0*((1.0/force.getSoluteDielectric())-(1.0/force.getSolventDielectric()));
    surfaceAreaFactor = -6.0*4*M_PI*force.getSurfaceAreaEnergy();
    cutoff = force.getCutoffDistance();
    bool useCutoff = (force.getNonbondedMethod() != GBSWForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() != GBSWForce::NoCutoff && force.getNonbondedMethod() != GBSWForce::CutoffNonPeriodic);
    string source = OpenCLGBSWKernelSources::gbsw2;
    nb.addInteraction(useCutoff, usePeriodic, false, force.getCutoffDistance(), vector<vector<int> >(), source, force.getForceGroup());
    nb.addParameter(OpenCLNonbondedUtilities::ParameterInfo("gbswParams", "float", 2, sizeof(cl_float2), params->getDeviceBuffer()));;
    nb.addParameter(OpenCLNonbondedUtilities::ParameterInfo("bornForce", "real", 1, elementSize, bornForce->getDeviceBuffer()));;
    cl.addForce(new OpenCLGBSWForceInfo(nb.getNumForceBuffers(), force));
}

double OpenCLCalcGBSWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
  bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
  if (!hasCreatedKernels) {
    // These Kernels cannot be created in initialize(), because the OpenCLNonbondedUtilities has not been initialized yet then.

    hasCreatedKernels = true;
    maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 0);
    map<string, string> defines;
    if (nb.getUseCutoff())
      defines["USE_CUTOFF"] = "1";
    if (nb.getUsePeriodic())
      defines["USE_PERIODIC"] = "1";
    defines["CUTOFF_SQUARED"] = cl.doubleToString(cutoff*cutoff);
    defines["CUTOFF"] = cl.doubleToString(cutoff);
    defines["PREFACTOR"] = cl.doubleToString(prefactor);
    defines["SURFACE_AREA_FACTOR"] = cl.doubleToString(surfaceAreaFactor);
    defines["NUM_ATOMS"] = cl.intToString(cl.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cl.intToString(cl.getNumAtomBlocks());
    defines["FORCE_WORK_GROUP_SIZE"] = cl.intToString(nb.getForceThreadBlockSize());
    defines["TILE_SIZE"] = cl.intToString(OpenCLContext::TileSize);
    int numExclusionTiles = nb.getExclusionTiles().getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cl.intToString(numExclusionTiles);
    int numContexts = cl.getPlatformData().contexts.size();
    int startExclusionIndex = cl.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cl.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cl.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cl.intToString(endExclusionIndex);
    string platformVendor = cl::Platform(cl.getDevice().getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_VENDOR>();
    if (platformVendor == "Apple")
      defines["USE_APPLE_WORKAROUND"] = "1";
    string file;
    if (deviceIsCpu)
      file = OpenCLGBSWKernelSources::gbsw_cpu;
    else
      file = OpenCLGBSWKernelSources::gbsw;
    cl::Program program = cl.createProgram(file, defines);
    bool useLong = cl.getSupports64BitGlobalAtomics();
    int index = 0;
    computeBornSumKernel = cl::Kernel(program, "computeBornSum");
    computeBornSumKernel.setArg<cl::Buffer>(index++, (useLong ? longBornSum->getDeviceBuffer() : bornSum->getDeviceBuffer()));
    computeBornSumKernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer());
    computeBornSumKernel.setArg<cl::Buffer>(index++, params->getDeviceBuffer());
    if (nb.getUseCutoff()) {
      computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getInteractingTiles().getDeviceBuffer());
      computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getInteractionCount().getDeviceBuffer());
      index += 5; // The periodic box size arguments are set when the kernel is executed.
      computeBornSumKernel.setArg<cl_uint>(index++, maxTiles);
      computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getBlockCenters().getDeviceBuffer());
      computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getBlockBoundingBoxes().getDeviceBuffer());
      computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getInteractingAtoms().getDeviceBuffer());
    }
    else
      computeBornSumKernel.setArg<cl_uint>(index++, cl.getNumAtomBlocks()*(cl.getNumAtomBlocks()+1)/2);
    computeBornSumKernel.setArg<cl::Buffer>(index++, nb.getExclusionTiles().getDeviceBuffer());
    force1Kernel = cl::Kernel(program, "computeGBSWForce1");
    index = 0;
    force1Kernel.setArg<cl::Buffer>(index++, (useLong ? cl.getLongForceBuffer().getDeviceBuffer() : cl.getForceBuffers().getDeviceBuffer()));
    force1Kernel.setArg<cl::Buffer>(index++, (useLong ? longBornForce->getDeviceBuffer() : bornForce->getDeviceBuffer()));
    force1Kernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());
    force1Kernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer());
    force1Kernel.setArg<cl::Buffer>(index++, bornRadii->getDeviceBuffer());
    if (nb.getUseCutoff()) {
      force1Kernel.setArg<cl::Buffer>(index++, nb.getInteractingTiles().getDeviceBuffer());
      force1Kernel.setArg<cl::Buffer>(index++, nb.getInteractionCount().getDeviceBuffer());
      index += 5; // The periodic box size arguments are set when the kernel is executed.
      force1Kernel.setArg<cl_uint>(index++, maxTiles);
      force1Kernel.setArg<cl::Buffer>(index++, nb.getBlockCenters().getDeviceBuffer());
      force1Kernel.setArg<cl::Buffer>(index++, nb.getBlockBoundingBoxes().getDeviceBuffer());
      force1Kernel.setArg<cl::Buffer>(index++, nb.getInteractingAtoms().getDeviceBuffer());
    }
    else
      force1Kernel.setArg<cl_uint>(index++, cl.getNumAtomBlocks()*(cl.getNumAtomBlocks()+1)/2);
    force1Kernel.setArg<cl::Buffer>(index++, nb.getExclusionTiles().getDeviceBuffer());
    program = cl.createProgram(OpenCLGBSWKernelSources::gbswReductions, defines);
    reduceBornSumKernel = cl::Kernel(program, "reduceBornSum");
    reduceBornSumKernel.setArg<cl_int>(0, cl.getPaddedNumAtoms());
    reduceBornSumKernel.setArg<cl_int>(1, nb.getNumForceBuffers());
    reduceBornSumKernel.setArg<cl_float>(2, 1.0f);
    reduceBornSumKernel.setArg<cl_float>(3, 0.8f);
    reduceBornSumKernel.setArg<cl_float>(4, 4.85f);
    reduceBornSumKernel.setArg<cl::Buffer>(5, (useLong ? longBornSum->getDeviceBuffer() : bornSum->getDeviceBuffer()));
    reduceBornSumKernel.setArg<cl::Buffer>(6, params->getDeviceBuffer());
    reduceBornSumKernel.setArg<cl::Buffer>(7, bornRadii->getDeviceBuffer());
    reduceBornSumKernel.setArg<cl::Buffer>(8, gbswChain->getDeviceBuffer());
    reduceBornForceKernel = cl::Kernel(program, "reduceBornForce");
    index = 0;
    reduceBornForceKernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
    reduceBornForceKernel.setArg<cl_int>(index++, nb.getNumForceBuffers());
    reduceBornForceKernel.setArg<cl::Buffer>(index++, bornForce->getDeviceBuffer());
    if (useLong)
      reduceBornForceKernel.setArg<cl::Buffer>(index++, longBornForce->getDeviceBuffer());
    reduceBornForceKernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());
    reduceBornForceKernel.setArg<cl::Buffer>(index++, params->getDeviceBuffer());
    reduceBornForceKernel.setArg<cl::Buffer>(index++, bornRadii->getDeviceBuffer());
    reduceBornForceKernel.setArg<cl::Buffer>(index++, gbswChain->getDeviceBuffer());
  }
  if (nb.getUseCutoff()) {
    setPeriodicBoxSizeArg(cl, computeBornSumKernel, 5);
    setInvPeriodicBoxSizeArg(cl, computeBornSumKernel, 6);
    setPeriodicBoxSizeArg(cl, force1Kernel, 7);
    setInvPeriodicBoxSizeArg(cl, force1Kernel, 8);
    if (maxTiles < nb.getInteractingTiles().getSize()) {
      maxTiles = nb.getInteractingTiles().getSize();
      computeBornSumKernel.setArg<cl::Buffer>(3, nb.getInteractingTiles().getDeviceBuffer());
      computeBornSumKernel.setArg<cl_uint>(7, maxTiles);
      computeBornSumKernel.setArg<cl::Buffer>(10, nb.getInteractingAtoms().getDeviceBuffer());
      force1Kernel.setArg<cl::Buffer>(5, nb.getInteractingTiles().getDeviceBuffer());
      force1Kernel.setArg<cl_uint>(9, maxTiles);
      force1Kernel.setArg<cl::Buffer>(12, nb.getInteractingAtoms().getDeviceBuffer());
    }
  }
  cl.executeKernel(computeBornSumKernel, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
  cl.executeKernel(reduceBornSumKernel, cl.getPaddedNumAtoms());
  cl.executeKernel(force1Kernel, nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
  cl.executeKernel(reduceBornForceKernel, cl.getPaddedNumAtoms());
  return 0.0;
}

void OpenCLCalcGBSWForceKernel::copyParametersToContext(ContextImpl& context, const GBSWForce& force) {
    // Make sure the new parameters are acceptable.
    
    int numParticles = force.getNumParticles();
    if (numParticles != cl.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    OpenCLArray& posq = cl.getPosq();
    mm_float4* posqf = (mm_float4*) cl.getPinnedBuffer();
    mm_double4* posqd = (mm_double4*) cl.getPinnedBuffer();
    posq.download(cl.getPinnedBuffer());
    vector<mm_float2> paramsVector(cl.getPaddedNumAtoms(), mm_float2(1,1));
    const double dielectricOffset = 0.009;
    for (int i = 0; i < numParticles; i++) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        radius -= dielectricOffset;
        paramsVector[i] = mm_float2((float) radius, (float) (1.0*radius));
        if (cl.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
    }
    posq.upload(cl.getPinnedBuffer());
    params->upload(paramsVector);
    
    // Mark that the current reordering may be invalid.
    
    cl.invalidateMolecules();
}

void OpenCLCalcGBSWForceKernel::getLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}
void OpenCLCalcGBSWForceKernel::setLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}

