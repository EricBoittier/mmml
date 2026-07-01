/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWForceImpl.h"
#include "GBSWKernels.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <vector>

using namespace OpenMM;
using namespace OpenMMGBSW;
using std::vector;

GBSWForceImpl::GBSWForceImpl(const GBSWForce& owner) : owner(owner) {
}

void GBSWForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcGBSWForceKernel::Name(), context);
    if (owner.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("GBSWForce must have exactly as many particles as the System it belongs to.");
    if (owner.getNonbondedMethod() == GBSWForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        context.getSystem().getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("GBSWForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel.getAs<CalcGBSWForceKernel>().initialize(context.getSystem(), owner);
}

void GBSWForceImpl::setDoingDynamics(bool val) {
  kernel.getAs<CalcGBSWForceKernel>().doingDynamics = val;
}

double GBSWForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcGBSWForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> GBSWForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGBSWForceKernel::Name());
    return names;
}

void GBSWForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGBSWForceKernel>().copyParametersToContext(context, owner);
}

void GBSWForceImpl::getLambdaState(ContextImpl& context, std::vector<double>& LambdaState) {
    kernel.getAs<CalcGBSWForceKernel>().getLambdaInfo(context, LambdaState);
}

void GBSWForceImpl::setLambdaState(ContextImpl& context, std::vector<double>& LambdaState) {
    kernel.getAs<CalcGBSWForceKernel>().setLambdaInfo(context, LambdaState);
}
