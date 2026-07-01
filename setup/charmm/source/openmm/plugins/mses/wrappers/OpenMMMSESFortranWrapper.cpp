#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMMSES.h"
#include "OpenMMMSESCWrapper.h"
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace OpenMM;
using namespace OpenMMMSES;
using namespace std;

extern "C" {
void openmmmses_msesforce_create_(OpenMMMSES_MSESForce*& result) {
    result = OpenMMMSES_MSESForce_create();
}
void OPENMMMSES_MSESFORCE_CREATE(OpenMMMSES_MSESForce*& result) {
    result = OpenMMMSES_MSESForce_create();
}
int openmmmses_msesforce_getnumdistpair_(const OpenMMMSES_MSESForce*& target) {
    return OpenMMMSES_MSESForce_getNumDistPair(target);
}
int OPENMMMSES_MSESFORCE_GETNUMDISTPAIR(const OpenMMMSES_MSESForce*& target) {
    return OpenMMMSES_MSESForce_getNumDistPair(target);
}
int openmmmses_msesforce_adddistpair_(OpenMMMSES_MSESForce*& target,int const& at1, int const& at2, int const& at3, int const& at4, double const& kc, double const& fmax, double const& dcut, double const& sexp) {
    return OpenMMMSES_MSESForce_addDistPair(target, at1, at2, at3, at4, kc, fmax, dcut, sexp);
}
int OPENMMMSES_MSESFORCE_ADDDISTPAIR(OpenMMMSES_MSESForce*& target, int const& at1, int const& at2, int const& at3, int const& at4, double const& kc, double const& fmax, double const& dcut, double const& sexp) {
    return OpenMMMSES_MSESForce_addDistPair(target, at1, at2, at3, at4, kc, fmax, dcut, sexp);
}
void openmmmses_msesforce_setdistpairparameters_(OpenMMMSES_MSESForce*& target, int const& index, int const& at1, int const& at2, int const& at3, int const& at4, double const& kc, double const& fmax, double const& dcut, double const& sexp) {
    OpenMMMSES_MSESForce_setDistPairParameters(target, index, at1, at2, at3, at4, kc, fmax, dcut, sexp);
}
void OPENMMMSES_MSESFORCE_SETDISTPAIRPARAMETERS(OpenMMMSES_MSESForce*& target, int const& index, int const& at1, int const& at2, int const& at3, int const& at4, double const& kc, double const& fmax, double const& dcut, double const& sexp) {
    OpenMMMSES_MSESForce_setDistPairParameters(target, index, at1, at2, at3, at4, kc, fmax, dcut, sexp);
}
void openmmmses_msesforce_updateparametersincontext_(OpenMMMSES_MSESForce*& target, OpenMM_Context*& context) {
    OpenMMMSES_MSESForce_updateParametersInContext(target, context);
}
void OPENMMMSES_MSESFORCE_UPDATEPARAMETERSINCONTEXT(OpenMMMSES_MSESForce*& target, OpenMM_Context*& context) {
    OpenMMMSES_MSESForce_updateParametersInContext(target, context);
}
void openmmmses_msesforce_usesperiodicboundaryconditions_(const OpenMMMSES_MSESForce*& target, OpenMM_Boolean& result) {
    result = OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions(target);
}
void OPENMMMSES_MSESFORCE_USESPERIODICBOUNDARYCONDITIONS(const OpenMMMSES_MSESForce*& target, OpenMM_Boolean& result) {
    result = OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions(target);
}
}
