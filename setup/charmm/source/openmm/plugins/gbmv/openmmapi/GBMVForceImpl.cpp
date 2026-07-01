/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "GBMVForceImpl.h"
#include "GBMVKernels.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <vector>

using namespace OpenMM;
using namespace OpenMMGBMV;
using std::vector;

GBMVForceImpl::GBMVForceImpl(const GBMVForce& owner) : owner(owner) {
}

void GBMVForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcGBMVForceKernel::Name(), context);
    if (owner.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("GBMVForce must have exactly as many particles as the System it belongs to.");
    if (owner.getNonbondedMethod() == GBMVForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        context.getSystem().getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("GBMVForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel.getAs<CalcGBMVForceKernel>().initialize(context.getSystem(), owner);
}

double GBMVForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcGBMVForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> GBMVForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGBMVForceKernel::Name());
    return names;
}

void GBMVForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGBMVForceKernel>().copyParametersToContext(context, owner);
}

void GBMVForceImpl::getLambdaState(ContextImpl& context, std::vector<double>& LambdaState) {
    kernel.getAs<CalcGBMVForceKernel>().getLambdaInfo(context, LambdaState);
}

void GBMVForceImpl::setLambdaState(ContextImpl& context, std::vector<double>& LambdaState) {
    kernel.getAs<CalcGBMVForceKernel>().setLambdaInfo(context, LambdaState);
}
