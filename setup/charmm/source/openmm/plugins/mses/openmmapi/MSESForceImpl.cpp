
#include "MSESForceImpl.h"
#include "MSESKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <iostream>

using namespace OpenMMMSES;
using namespace OpenMM;
using namespace std;

MSESForceImpl::MSESForceImpl(const MSESForce& owner) : owner(owner) {
}

MSESForceImpl::~MSESForceImpl() {
}

void MSESForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcMSESForceKernel::Name(), context);
    kernel.getAs<CalcMSESForceKernel>().initialize(context.getSystem(), owner);
}

double MSESForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
  if ((groups&(1<<owner.getForceGroup())) != 0)
     return kernel.getAs<CalcMSESForceKernel>().execute(context, includeForces, includeEnergy);
  return 0.0;
}

std::vector<std::string> MSESForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMSESForceKernel::Name());
    return names;
}

void MSESForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMSESForceKernel>().copyParametersToContext(context, owner);
}
