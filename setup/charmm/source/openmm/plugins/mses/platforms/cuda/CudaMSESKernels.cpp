
#include "CudaMSESKernels.h"
#include "CudaMSESKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

using namespace OpenMMMSES;
using namespace OpenMM;
using namespace std;

class CudaMSESForceInfo : public CudaForceInfo {
public:
    CudaMSESForceInfo(const MSESForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumDistPair();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int p1,p2,p3,p4;
        double c1,c2,c3,c4;
        force.getDistPairParameters(index, p1,p2,p3,p4,c1,c2,c3,c4);
        particles.resize(4);
        particles[0] = p1;
        particles[1] = p2;
        particles[2] = p3;
        particles[3] = p4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int p1,p2,p3,p4;
        double c1a,c2a,c3a,c4a, c1b,c2b,c3b,c4b;
        force.getDistPairParameters(group1, p1,p2,p3,p4,c1a,c2a,c3a,c4a);
        force.getDistPairParameters(group2, p1,p2,p3,p4,c1b,c2b,c3b,c4b);
        return (c1a==c1b && c2a==c2b && c3a==c3b && c4a==c4b);
    }
private:
    const MSESForce& force;
};

CudaCalcMSESForceKernel::~CudaCalcMSESForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcMSESForceKernel::initialize(const System& system, const MSESForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumDistPair()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumDistPair()/numContexts;
    numDistPair = endIndex-startIndex;
    if (numDistPair == 0)
        return;
    vector<vector<int> > atoms(numDistPair, vector<int>(4));
    params = CudaArray::create<float4>(cu, numDistPair, "distPairParams");
    vector<float4> paramVector(numDistPair);
    for (int i = 0; i < numDistPair; i++) {
        double c1,c2,c3,c4;
        force.getDistPairParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], c1,c2,c3,c4);
        paramVector[i] = make_float4((float)c1, (float)c2, (float)c3, (float)c4);
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
#if OMM_VER < 82
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float4");
#else
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(*params, "float4");
#endif /* OMM_VER */
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaMSESKernelSources::msesForce, replacements), force.getForceGroup());
    cu.addForce(new CudaMSESForceInfo(force));
}

double CudaCalcMSESForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMSESForceKernel::copyParametersToContext(ContextImpl& context, const MSESForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumDistPair()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumDistPair()/numContexts;
    if (numDistPair != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of MSES Distance-based pair interaction has changed");
    if (numDistPair == 0)
        return;

    // Record the per-bond parameters.

    vector<float4> paramVector(numDistPair);
    for (int i = 0; i < numDistPair; i++) {
        int atom1, atom2, atom3, atom4;
        double c1, c2, c3, c4;
        force.getDistPairParameters(startIndex+i, atom1, atom2, atom3, atom4, c1, c2, c3, c4);
        paramVector[i] = make_float4((float)c1, (float)c2, (float)c3, (float)c4);
    }
    params->upload(paramVector);

    // Mark that the current reordering may be invalid.

    cu.invalidateMolecules();
}
