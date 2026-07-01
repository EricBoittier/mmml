#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMGBSW.h"
#include "OpenMMGBSWCWrapper.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

extern "C" {
OpenMMGBSW_GBSWForce* OpenMMGBSW_GBSWForce_create() {
    return reinterpret_cast<OpenMMGBSW_GBSWForce*>(new OpenMMGBSW::GBSWForce());
}
void OpenMMGBSW_GBSWForce_destroy(OpenMMGBSW_GBSWForce* target) {
    delete reinterpret_cast<OpenMMGBSW::GBSWForce*>(target);
}
void OpenMMGBSW_GBSWForce_addCPHMDForce(OpenMMGBSW_GBSWForce* target, double pH, double T_theta, double mTheta, double ts_theta, double beta, int outFreq, char* fileName) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->addCPHMDForce(pH, T_theta, mTheta, ts_theta, beta, outFreq, reinterpret_cast<char *>(fileName));
}
OpenMM_Boolean OpenMMGBSW_GBSWForce_usingCPHMD(const OpenMMGBSW_GBSWForce* target) {
    bool result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->usingCPHMD();
    return (result ? OpenMM_True : OpenMM_False);
}
int OpenMMGBSW_GBSWForce_getNumTitratingGroups(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNumTitratingGroups();
    return result;
}
int OpenMMGBSW_GBSWForce_addTitratingGroupParameters(OpenMMGBSW_GBSWForce* target, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8) {
    int result = reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->addTitratingGroupParameters(resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
    return result;
}
void OpenMMGBSW_GBSWForce_getTitratingGroupParameters(const OpenMMGBSW_GBSWForce* target, int index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getTitratingGroupParameters(index, *reinterpret_cast<double*>(resPKA1), *reinterpret_cast<double*>(resPKA2), *reinterpret_cast<double*>(barrier1), *reinterpret_cast<double*>(barrier2), *reinterpret_cast<double*>(a0), *reinterpret_cast<double*>(a1), *reinterpret_cast<double*>(a2), *reinterpret_cast<double*>(a3), *reinterpret_cast<double*>(a4), *reinterpret_cast<double*>(a5), *reinterpret_cast<double*>(a6), *reinterpret_cast<double*>(a7), *reinterpret_cast<double*>(a8));
}

  void OpenMMGBSW_GBSWForce_setDoingDynamics(OpenMMGBSW_GBSWForce* target, OpenMM_Context * context, int flag) {
    bool val = true;
    if (flag == 0) val = false;

    OpenMMGBSW::GBSWForce * typedTarget = reinterpret_cast<OpenMMGBSW::GBSWForce *>(target);
    OpenMM::Context * typedContext = reinterpret_cast<OpenMM::Context *>(context);

    typedTarget->setDoingDynamics(*typedContext, val);
  }

void OpenMMGBSW_GBSWForce_setTitratingGroupParameters(OpenMMGBSW_GBSWForce* target, int index, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setTitratingGroupParameters(index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int OpenMMGBSW_GBSWForce_addNonbondedException(OpenMMGBSW_GBSWForce* target, int atom1, int atom2) {
    int result = reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->addNonbondedException(atom1, atom2);
    return result;
}
void OpenMMGBSW_GBSWForce_getNonbondedException(const OpenMMGBSW_GBSWForce* target, int index, int* atom1, int* atom2) {
    reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNonbondedException(index, *reinterpret_cast<int*>(atom1), *reinterpret_cast<int*>(atom2));
}
int OpenMMGBSW_GBSWForce_getNumNonbondedExceptions(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNumNonbondedExceptions();
    return result;
}
int OpenMMGBSW_GBSWForce_getNumParticles(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNumParticles();
    return result;
}
int OpenMMGBSW_GBSWForce_addParticle(OpenMMGBSW_GBSWForce* target, double charge, double radius) {
    int result = reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->addParticle(charge, radius);
    return result;
}
void OpenMMGBSW_GBSWForce_getParticleParameters(const OpenMMGBSW_GBSWForce* target, int index, double* charge, double* radius) {
    reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getParticleParameters(index, *reinterpret_cast<double*>(charge), *reinterpret_cast<double*>(radius));
}
void OpenMMGBSW_GBSWForce_setParticleParameters(OpenMMGBSW_GBSWForce* target, int index, double charge, double radius) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setParticleParameters(index, charge, radius);
}
int OpenMMGBSW_GBSWForce_addCphmdParameters(OpenMMGBSW_GBSWForce* target, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2) {
    int result = reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->addCphmdParameters(titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
    return result;
}
void OpenMMGBSW_GBSWForce_getCphmdParameters(const OpenMMGBSW_GBSWForce* target, int index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getCphmdParameters(index, *reinterpret_cast<int*>(titrateResID), *reinterpret_cast<double*>(refChargeState1), *reinterpret_cast<double*>(refChargeState2), *reinterpret_cast<double*>(chargeState1), *reinterpret_cast<double*>(chargeState2));
}
void OpenMMGBSW_GBSWForce_setCphmdParameters(OpenMMGBSW_GBSWForce* target, int index, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setCphmdParameters(index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
char* OpenMMGBSW_GBSWForce_getLambdaOutputFile(const OpenMMGBSW_GBSWForce* target) {
    char * result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getLambdaOutputFile();
    return reinterpret_cast<char*>(result);
}
void OpenMMGBSW_GBSWForce_setLambdaOutputFile(OpenMMGBSW_GBSWForce* target, char* tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setLambdaOutputFile(reinterpret_cast<char *>(tmp));
}
double OpenMMGBSW_GBSWForce_getSystemPH(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getSystemPH();
    return result;
}
void OpenMMGBSW_GBSWForce_setSystemPH(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setSystemPH(tmp);
}
double OpenMMGBSW_GBSWForce_getThetaTemp(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getThetaTemp();
    return result;
}
void OpenMMGBSW_GBSWForce_setThetaTemp(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setThetaTemp(tmp);
}
double OpenMMGBSW_GBSWForce_getThetaMass(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getThetaMass();
    return result;
}
void OpenMMGBSW_GBSWForce_setThetaMass(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setThetaMass(tmp);
}
int OpenMMGBSW_GBSWForce_getLambdaOutputFrequency(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getLambdaOutputFrequency();
    return result;
}
void OpenMMGBSW_GBSWForce_setLambdaOutputFrequency(OpenMMGBSW_GBSWForce* target, int tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setLambdaOutputFrequency(tmp);
}
double OpenMMGBSW_GBSWForce_getPHbeta(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getPHbeta();
    return result;
}
void OpenMMGBSW_GBSWForce_setPHbeta(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setPHbeta(tmp);
}
double OpenMMGBSW_GBSWForce_getSolventDielectric(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getSolventDielectric();
    return result;
}
void OpenMMGBSW_GBSWForce_setSolventDielectric(OpenMMGBSW_GBSWForce* target, double dielectric) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setSolventDielectric(dielectric);
}
double OpenMMGBSW_GBSWForce_getSoluteDielectric(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getSoluteDielectric();
    return result;
}
void OpenMMGBSW_GBSWForce_setSoluteDielectric(OpenMMGBSW_GBSWForce* target, double dielectric) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setSoluteDielectric(dielectric);
}
double OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getSurfaceAreaEnergy();
    return result;
}
void OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(OpenMMGBSW_GBSWForce* target, double energy) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setSurfaceAreaEnergy(energy);
}
double OpenMMGBSW_GBSWForce_getAA0(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getAA0();
    return result;
}
void OpenMMGBSW_GBSWForce_setAA0(OpenMMGBSW_GBSWForce* target, double value) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setAA0(value);
}
double OpenMMGBSW_GBSWForce_getAA1(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getAA1();
    return result;
}
void OpenMMGBSW_GBSWForce_setAA1(OpenMMGBSW_GBSWForce* target, double value) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setAA1(value);
}
int OpenMMGBSW_GBSWForce_getNumGauLegRad(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNumGauLegRad();
    return result;
}
void OpenMMGBSW_GBSWForce_setNumGauLegRad(OpenMMGBSW_GBSWForce* target, int number) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setNumGauLegRad(number);
}
int OpenMMGBSW_GBSWForce_getNumLebAng(const OpenMMGBSW_GBSWForce* target) {
    int result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNumLebAng();
    return result;
}
void OpenMMGBSW_GBSWForce_setNumLebAng(OpenMMGBSW_GBSWForce* target, int number) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setNumLebAng(number);
}
double OpenMMGBSW_GBSWForce_getDebyeHuckelLength(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getDebyeHuckelLength();
    return result;
}
void OpenMMGBSW_GBSWForce_setDebyeHuckelLength(OpenMMGBSW_GBSWForce* target, double length) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setDebyeHuckelLength(length);
}
double OpenMMGBSW_GBSWForce_getSwitchingLength(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getSwitchingLength();
    return result;
}
void OpenMMGBSW_GBSWForce_setSwitchingLength(OpenMMGBSW_GBSWForce* target, double length) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setSwitchingLength(length);
}
double OpenMMGBSW_GBSWForce_getMembraneThickness(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getMembraneThickness();
    return result;
}
void OpenMMGBSW_GBSWForce_setMembraneThickness(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setMembraneThickness(tmp);
}
double OpenMMGBSW_GBSWForce_getMembraneSwLen(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getMembraneSwLen();
    return result;
}
void OpenMMGBSW_GBSWForce_setMembraneSwLen(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setMembraneSwLen(tmp);
}
OpenMMGBSW_GBSWForce_NonbondedMethod OpenMMGBSW_GBSWForce_getNonbondedMethod(const OpenMMGBSW_GBSWForce* target) {
    OpenMMGBSW::GBSWForce::NonbondedMethod result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getNonbondedMethod();
    return static_cast<OpenMMGBSW_GBSWForce_NonbondedMethod>(result);
}
void OpenMMGBSW_GBSWForce_setNonbondedMethod(OpenMMGBSW_GBSWForce* target, OpenMMGBSW_GBSWForce_NonbondedMethod method) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setNonbondedMethod(static_cast<OpenMMGBSW::GBSWForce::NonbondedMethod>(method));
}
double OpenMMGBSW_GBSWForce_getCutoffDistance(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getCutoffDistance();
    return result;
}
void OpenMMGBSW_GBSWForce_setCutoffDistance(OpenMMGBSW_GBSWForce* target, double distance) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setCutoffDistance(distance);
}
void OpenMMGBSW_GBSWForce_updateParametersInContext(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->updateParametersInContext(*reinterpret_cast<OpenMM::Context*>(context));
}
double OpenMMGBSW_GBSWForce_getReactionFieldDielectric(const OpenMMGBSW_GBSWForce* target) {
    double result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->getReactionFieldDielectric();
    return result;
}
void OpenMMGBSW_GBSWForce_setReactionFieldDielectric(OpenMMGBSW_GBSWForce* target, double tmp) {
    reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->setReactionFieldDielectric(tmp);
}
OpenMM_Boolean OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions(const OpenMMGBSW_GBSWForce* target) {
    bool result = reinterpret_cast<const OpenMMGBSW::GBSWForce*>(target)->usesPeriodicBoundaryConditions();
    return (result ? OpenMM_True : OpenMM_False);
}
void OpenMMGBSW_GBSWForce_getLambdaState(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState) {
  reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->getLambdaState(
      *reinterpret_cast<OpenMM::Context*>(context),
      *reinterpret_cast<std::vector<double>*>(lambdaState));
}
void OpenMMGBSW_GBSWForce_setLambdaState(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState) {
  reinterpret_cast<OpenMMGBSW::GBSWForce*>(target)->getLambdaState(
      *reinterpret_cast<OpenMM::Context*>(context),
      *reinterpret_cast<std::vector<double>*>(lambdaState));
}
}
