/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "ReferenceGBSWKernels.h"
#include "ReferenceGBSW.h"

#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
#include "openmm/reference/ReferenceConstraints.h"
#include "openmm/reference/ReferenceVirtualSites.h"
#include "openmm/internal/ContextImpl.h"

#include <set>

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static RealVec& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(RealVec*) data->periodicBoxSize;
}

ReferenceCalcGBSWForceKernel::~ReferenceCalcGBSWForceKernel() {
    if (gbsw) {
        delete gbsw->getGBSWParameters();
        delete gbsw;
    }
}

void ReferenceCalcGBSWForceKernel::initialize(const System& system, const GBSWForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    vector<RealOpenMM> atomicRadii(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        charges[i] = static_cast<RealOpenMM>(charge);
        atomicRadii[i] = static_cast<RealOpenMM>(radius);
    }
    GBSWParameters* gbswParameters = new GBSWParameters(numParticles, GBSWParameters::GBSWTypeII);
    gbswParameters->setAtomicRadii(atomicRadii);
    gbswParameters->setSolventDielectric(static_cast<RealOpenMM>(force.getSolventDielectric()));
    gbswParameters->setSoluteDielectric(static_cast<RealOpenMM>(force.getSoluteDielectric()));
    if (force.getNonbondedMethod() != GBSWForce::NoCutoff)
        gbswParameters->setUseCutoff(static_cast<RealOpenMM>(force.getCutoffDistance()));
    isPeriodic = (force.getNonbondedMethod() == GBSWForce::CutoffPeriodic);
    gbsw = new ReferenceGBSW(gbswParameters);
    gbsw->setIncludeAceApproximation(true);
}

double ReferenceCalcGBSWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& posData = extractPositions(context);
    vector<RealVec>& forceData = extractForces(context);
    if (isPeriodic)
        gbsw->getGBSWParameters()->setPeriodic(extractBoxSize(context));
    return gbsw->computeBornEnergyForces(posData, charges, forceData);
}

void ReferenceCalcGBSWForceKernel::copyParametersToContext(ContextImpl& context, const GBSWForce& force) {
    int numParticles = force.getNumParticles();
    GBSWParameters* gbswParameters = gbsw->getGBSWParameters();
    if (numParticles != gbswParameters->getAtomicRadii().size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.
    vector<RealOpenMM> atomicRadii(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        charges[i] = (RealOpenMM) charge;
        atomicRadii[i] = (RealOpenMM) radius;
    }
    gbswParameters->setAtomicRadii(atomicRadii);
}

void ReferenceCalcGBSWForceKernel::getLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}
void ReferenceCalcGBSWForceKernel::setLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    // unfinished function
}
