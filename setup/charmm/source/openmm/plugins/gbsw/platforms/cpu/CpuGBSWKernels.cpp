/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWForce.h"
#include "CpuGBSWKernels.h"

#include "openmm/System.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

CpuCalcGBSWForceKernel::~CpuCalcGBSWForceKernel() {
}

void CpuCalcGBSWForceKernel::initialize(const System& system, const GBSWForce& force) {
    throw OpenMMException("GBSWForce is not yet implimented on the CPU platform");
    int numParticles = system.getNumParticles();
    particleParams.resize(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        data.posq[4*i+3] = (float) charge;
        radius -= 0.009;
        particleParams[i] = make_pair((float) radius, (float) (1.0*radius));
    }
    gbsw.setParticleParameters(particleParams);
    gbsw.setSolventDielectric((float) force.getSolventDielectric());
    gbsw.setSoluteDielectric((float) force.getSoluteDielectric());
    gbsw.setSurfaceAreaEnergy((float) force.getSurfaceAreaEnergy());
    if (force.getNonbondedMethod() != GBSWForce::NoCutoff)
        gbsw.setUseCutoff((float) force.getCutoffDistance());
    data.isPeriodic = (force.getNonbondedMethod() == GBSWForce::CutoffPeriodic);
}

static RealVec& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(RealVec*) data->periodicBoxSize;
}

double CpuCalcGBSWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (data.isPeriodic) {
        RealVec& boxSize = extractBoxSize(context);
        float floatBoxSize[3] = {(float) boxSize[0], (float) boxSize[1], (float) boxSize[2]};
        gbsw.setPeriodic(floatBoxSize);
    }
    double energy = 0.0;
    gbsw.computeForce(data.posq, data.threadForce, includeEnergy ? &energy : NULL, data.threads);
    return energy;
}

void CpuCalcGBSWForceKernel::getLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}
void CpuCalcGBSWForceKernel::setLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}

void CpuCalcGBSWForceKernel::copyParametersToContext(ContextImpl& context, const GBSWForce& force) {
    int numParticles = force.getNumParticles();
    if (numParticles != gbsw.getParticleParameters().size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        data.posq[4*i+3] = (float) charge;
        radius -= 0.009;
        particleParams[i] = make_pair((float) radius, (float) (1.0*radius));
    }
    gbsw.setParticleParameters(particleParams);
}
