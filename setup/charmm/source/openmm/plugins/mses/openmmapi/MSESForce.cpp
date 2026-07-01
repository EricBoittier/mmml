/* -------------------------------------------------------------------------- *
 *                             OpenMM-MSES                                    *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include "MSESForce.h"
#include "MSESForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace OpenMMMSES;
using namespace OpenMM;
using namespace std;

MSESForce::MSESForce() : usePeriodic(false) { }

int MSESForce::addDistPair(int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp) {
  distPair.push_back(distPairInfo(at1,at2,at3,at4,kc,fmax,dcut,sexp));
  return distPair.size()-1;
}

int MSESForce::getNumDistPair() const {
   return distPair.size();   
}

void MSESForce::getDistPairParameters(int index, int& at1, int& at2, int& at3, int& at4, double& kc, double& fmax, double& dcut, double& sexp) const {
   ASSERT_VALID_INDEX(index, distPair);
   at1 = distPair[index].at1; 
   at2 = distPair[index].at2; 
   at3 = distPair[index].at3; 
   at4 = distPair[index].at4; 
   kc = distPair[index].kc;
   fmax = distPair[index].fmax;
   dcut = distPair[index].dcut;
   sexp = distPair[index].sexp;
}

void MSESForce::setDistPairParameters(int index, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp) {
   ASSERT_VALID_INDEX(index, distPair);
   distPair[index].at1 = at1; 
   distPair[index].at2 = at2; 
   distPair[index].at3 = at3; 
   distPair[index].at4 = at4; 
   distPair[index].kc = kc;
   distPair[index].fmax = fmax;
   distPair[index].dcut = dcut;
   distPair[index].sexp = sexp;
}

ForceImpl* MSESForce::createImpl() const {
    return new MSESForceImpl(*this);
}

void MSESForce::updateParametersInContext(Context& context) {
    dynamic_cast<MSESForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void MSESForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool MSESForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
