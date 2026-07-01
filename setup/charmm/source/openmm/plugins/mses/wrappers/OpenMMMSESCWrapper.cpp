#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMMSES.h"
#include "OpenMMMSESCWrapper.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <iostream>

using namespace OpenMM;
using namespace OpenMMMSES;
using namespace std;

extern "C" {
OpenMMMSES_MSESForce* OpenMMMSES_MSESForce_create() {
    return reinterpret_cast<OpenMMMSES_MSESForce*>(new OpenMMMSES::MSESForce());
}
int OpenMMMSES_MSESForce_getNumDistPair(const OpenMMMSES_MSESForce* target) {
    int result = reinterpret_cast<const OpenMMMSES::MSESForce*>(target)->getNumDistPair();
    return result;
}
int OpenMMMSES_MSESForce_addDistPair(OpenMMMSES_MSESForce* target, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp) {
    int result = reinterpret_cast<OpenMMMSES::MSESForce*>(target)->addDistPair(at1,at2,at3,at4,kc,fmax,dcut,sexp);
    return result;
}
void OpenMMMSES_MSESForce_setDistPairParameters(OpenMMMSES_MSESForce* target, int index, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp) {
    reinterpret_cast<OpenMMMSES::MSESForce*>(target)->setDistPairParameters(index,at1,at2,at3,at4,kc,fmax,dcut,sexp);
}
void OpenMMMSES_MSESForce_updateParametersInContext(OpenMMMSES_MSESForce* target, OpenMM_Context* context) {
    reinterpret_cast<OpenMMMSES::MSESForce*>(target)->updateParametersInContext(*reinterpret_cast<OpenMM::Context*>(context));
}
OpenMM_Boolean OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions(const OpenMMMSES_MSESForce* target) {
    bool result = reinterpret_cast<const OpenMMMSES::MSESForce*>(target)->usesPeriodicBoundaryConditions();
    return (result ? OpenMM_True : OpenMM_False);
}
}
