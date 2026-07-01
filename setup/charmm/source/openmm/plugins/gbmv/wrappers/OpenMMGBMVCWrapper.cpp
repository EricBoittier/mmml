#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMGBMV.h"
#include "OpenMMGBMVCWrapper.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

using namespace OpenMM;
using namespace OpenMMGBMV;
using namespace std;

extern "C" {
OpenMMGBMV_GBMVForce* OpenMMGBMV_GBMVForce_create() {
    return reinterpret_cast<OpenMMGBMV_GBMVForce*>(new OpenMMGBMV::GBMVForce());
}
void OpenMMGBMV_GBMVForce_destroy(OpenMMGBMV_GBMVForce* target) {
    delete reinterpret_cast<OpenMMGBMV::GBMVForce*>(target);
}
void OpenMMGBMV_GBMVForce_addCPHMDForce(OpenMMGBMV_GBMVForce* target, double pH, double T_theta, double mTheta, double ts_theta, double beta, int outFreq, char* fileName) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->addCPHMDForce(pH, T_theta, mTheta, ts_theta, beta, outFreq, reinterpret_cast<char *>(fileName));
}
OpenMM_Boolean OpenMMGBMV_GBMVForce_usingCPHMD(const OpenMMGBMV_GBMVForce* target) {
    bool result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->usingCPHMD();
    return (result ? OpenMM_True : OpenMM_False);
}
int OpenMMGBMV_GBMVForce_getNumTitratingGroups(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNumTitratingGroups();
    return result;
}
int OpenMMGBMV_GBMVForce_addTitratingGroupParameters(OpenMMGBMV_GBMVForce* target, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8) {
    int result = reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->addTitratingGroupParameters(resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
    return result;
}
void OpenMMGBMV_GBMVForce_getTitratingGroupParameters(const OpenMMGBMV_GBMVForce* target, int index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getTitratingGroupParameters(index, *reinterpret_cast<double*>(resPKA1), *reinterpret_cast<double*>(resPKA2), *reinterpret_cast<double*>(barrier1), *reinterpret_cast<double*>(barrier2), *reinterpret_cast<double*>(a0), *reinterpret_cast<double*>(a1), *reinterpret_cast<double*>(a2), *reinterpret_cast<double*>(a3), *reinterpret_cast<double*>(a4), *reinterpret_cast<double*>(a5), *reinterpret_cast<double*>(a6), *reinterpret_cast<double*>(a7), *reinterpret_cast<double*>(a8));
}
void OpenMMGBMV_GBMVForce_setTitratingGroupParameters(OpenMMGBMV_GBMVForce* target, int index, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setTitratingGroupParameters(index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int OpenMMGBMV_GBMVForce_addNonbondedException(OpenMMGBMV_GBMVForce* target, int atom1, int atom2) {
    int result = reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->addNonbondedException(atom1, atom2);
    return result;
}
void OpenMMGBMV_GBMVForce_getNonbondedException(const OpenMMGBMV_GBMVForce* target, int index, int* atom1, int* atom2) {
    reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNonbondedException(index, *reinterpret_cast<int*>(atom1), *reinterpret_cast<int*>(atom2));
}
int OpenMMGBMV_GBMVForce_getNumNonbondedExceptions(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNumNonbondedExceptions();
    return result;
}
int OpenMMGBMV_GBMVForce_getNumParticles(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNumParticles();
    return result;
}
int OpenMMGBMV_GBMVForce_addParticle(OpenMMGBMV_GBMVForce* target, double charge, double radius) {
    int result = reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->addParticle(charge, radius);
    return result;
}
void OpenMMGBMV_GBMVForce_getParticleParameters(const OpenMMGBMV_GBMVForce* target, int index, double* charge, double* radius) {
    reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getParticleParameters(index, *reinterpret_cast<double*>(charge), *reinterpret_cast<double*>(radius));
}
void OpenMMGBMV_GBMVForce_setParticleParameters(OpenMMGBMV_GBMVForce* target, int index, double charge, double radius) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setParticleParameters(index, charge, radius);
}
int OpenMMGBMV_GBMVForce_addCphmdParameters(OpenMMGBMV_GBMVForce* target, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2) {
    int result = reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->addCphmdParameters(titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
    return result;
}
void OpenMMGBMV_GBMVForce_getCphmdParameters(const OpenMMGBMV_GBMVForce* target, int index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getCphmdParameters(index, *reinterpret_cast<int*>(titrateResID), *reinterpret_cast<double*>(refChargeState1), *reinterpret_cast<double*>(refChargeState2), *reinterpret_cast<double*>(chargeState1), *reinterpret_cast<double*>(chargeState2));
}
void OpenMMGBMV_GBMVForce_setCphmdParameters(OpenMMGBMV_GBMVForce* target, int index, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setCphmdParameters(index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
char* OpenMMGBMV_GBMVForce_getLambdaOutputFile(const OpenMMGBMV_GBMVForce* target) {
    char * result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getLambdaOutputFile();
    return reinterpret_cast<char*>(result);
}
void OpenMMGBMV_GBMVForce_setLambdaOutputFile(OpenMMGBMV_GBMVForce* target, char* tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setLambdaOutputFile(reinterpret_cast<char *>(tmp));
}
double OpenMMGBMV_GBMVForce_getSystemPH(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSystemPH();
    return result;
}
void OpenMMGBMV_GBMVForce_setSystemPH(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSystemPH(tmp);
}
double OpenMMGBMV_GBMVForce_getThetaTemp(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getThetaTemp();
    return result;
}
void OpenMMGBMV_GBMVForce_setThetaTemp(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setThetaTemp(tmp);
}
double OpenMMGBMV_GBMVForce_getThetaMass(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getThetaMass();
    return result;
}
void OpenMMGBMV_GBMVForce_setThetaMass(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setThetaMass(tmp);
}
int OpenMMGBMV_GBMVForce_getLambdaOutputFrequency(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getLambdaOutputFrequency();
    return result;
}
void OpenMMGBMV_GBMVForce_setLambdaOutputFrequency(OpenMMGBMV_GBMVForce* target, int tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setLambdaOutputFrequency(tmp);
}
double OpenMMGBMV_GBMVForce_getPHbeta(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getPHbeta();
    return result;
}
void OpenMMGBMV_GBMVForce_setPHbeta(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setPHbeta(tmp);
}
double OpenMMGBMV_GBMVForce_getSolventDielectric(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSolventDielectric();
    return result;
}
void OpenMMGBMV_GBMVForce_setSolventDielectric(OpenMMGBMV_GBMVForce* target, double dielectric) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSolventDielectric(dielectric);
}
double OpenMMGBMV_GBMVForce_getSoluteDielectric(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSoluteDielectric();
    return result;
}
void OpenMMGBMV_GBMVForce_setSoluteDielectric(OpenMMGBMV_GBMVForce* target, double dielectric) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSoluteDielectric(dielectric);
}
double OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSurfaceAreaEnergy();
    return result;
}
void OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(OpenMMGBMV_GBMVForce* target, double energy) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSurfaceAreaEnergy(energy);
}
double OpenMMGBMV_GBMVForce_getAA0(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getAA0();
    return result;
}
void OpenMMGBMV_GBMVForce_setAA0(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setAA0(value);
}
double OpenMMGBMV_GBMVForce_getAA1(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getAA1();
    return result;
}
void OpenMMGBMV_GBMVForce_setAA1(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setAA1(value);
}
int OpenMMGBMV_GBMVForce_getNumGauLegRad(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNumGauLegRad();
    return result;
}
void OpenMMGBMV_GBMVForce_setNumGauLegRad(OpenMMGBMV_GBMVForce* target, int number) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setNumGauLegRad(number);
}
int OpenMMGBMV_GBMVForce_getNumLebAng(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNumLebAng();
    return result;
}
void OpenMMGBMV_GBMVForce_setNumLebAng(OpenMMGBMV_GBMVForce* target, int number) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setNumLebAng(number);
}
double OpenMMGBMV_GBMVForce_getDebyeHuckelLength(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getDebyeHuckelLength();
    return result;
}
void OpenMMGBMV_GBMVForce_setDebyeHuckelLength(OpenMMGBMV_GBMVForce* target, double length) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setDebyeHuckelLength(length);
}
double OpenMMGBMV_GBMVForce_getSwitchingLength(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSwitchingLength();
    return result;
}
void OpenMMGBMV_GBMVForce_setSwitchingLength(OpenMMGBMV_GBMVForce* target, double length) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSwitchingLength(length);
}
// GBMV2: Paramters of GBMV2 model
double OpenMMGBMV_GBMVForce_getBETA_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getBETA_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setBETA_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setBETA_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getLAMBDA_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setLAMBDA_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getALPHA_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getALPHA_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setALPHA_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setALPHA_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getSLOPE_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSLOPE_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSLOPE_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getSHIFT_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getSHIFT_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setSHIFT_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getHSX1_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getHSX1_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setHSX1_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setHSX1_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getHSX2_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getHSX2_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setHSX2_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setHSX2_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getONX_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getONX_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setONX_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setONX_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getOFFX_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getOFFX_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setOFFX_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setOFFX_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getP1_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getP1_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setP1_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setP1_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getP2_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getP2_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setP2_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setP2_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getP3_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getP3_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setP3_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setP3_GBMV2(value);
}
double OpenMMGBMV_GBMVForce_getP6_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getP6_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setP6_GBMV2(OpenMMGBMV_GBMVForce* target, double value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setP6_GBMV2(value);
}
int OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2(const OpenMMGBMV_GBMVForce* target) {
    int result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getCUTNUM_GBMV2();
    return result;
}
void OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(OpenMMGBMV_GBMVForce* target, int value) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setCUTNUM_GBMV2(value);
}
// END GBMV2
double OpenMMGBMV_GBMVForce_getMembraneThickness(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getMembraneThickness();
    return result;
}
void OpenMMGBMV_GBMVForce_setMembraneThickness(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setMembraneThickness(tmp);
}
double OpenMMGBMV_GBMVForce_getMembraneSwLen(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getMembraneSwLen();
    return result;
}
void OpenMMGBMV_GBMVForce_setMembraneSwLen(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setMembraneSwLen(tmp);
}
OpenMMGBMV_GBMVForce_NonbondedMethod OpenMMGBMV_GBMVForce_getNonbondedMethod(const OpenMMGBMV_GBMVForce* target) {
    OpenMMGBMV::GBMVForce::NonbondedMethod result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getNonbondedMethod();
    return static_cast<OpenMMGBMV_GBMVForce_NonbondedMethod>(result);
}
void OpenMMGBMV_GBMVForce_setNonbondedMethod(OpenMMGBMV_GBMVForce* target, OpenMMGBMV_GBMVForce_NonbondedMethod method) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setNonbondedMethod(static_cast<OpenMMGBMV::GBMVForce::NonbondedMethod>(method));
}
double OpenMMGBMV_GBMVForce_getCutoffDistance(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getCutoffDistance();
    return result;
}
void OpenMMGBMV_GBMVForce_setCutoffDistance(OpenMMGBMV_GBMVForce* target, double distance) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setCutoffDistance(distance);
}
double OpenMMGBMV_GBMVForce_getCutonDistance(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getCutonDistance();
    return result;
}
void OpenMMGBMV_GBMVForce_setCutonDistance(OpenMMGBMV_GBMVForce* target, double distance) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setCutonDistance(distance);
}
void OpenMMGBMV_GBMVForce_updateParametersInContext(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->updateParametersInContext(*reinterpret_cast<OpenMM::Context*>(context));
}
double OpenMMGBMV_GBMVForce_getReactionFieldDielectric(const OpenMMGBMV_GBMVForce* target) {
    double result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->getReactionFieldDielectric();
    return result;
}
void OpenMMGBMV_GBMVForce_setReactionFieldDielectric(OpenMMGBMV_GBMVForce* target, double tmp) {
    reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->setReactionFieldDielectric(tmp);
}
OpenMM_Boolean OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions(const OpenMMGBMV_GBMVForce* target) {
    bool result = reinterpret_cast<const OpenMMGBMV::GBMVForce*>(target)->usesPeriodicBoundaryConditions();
    return (result ? OpenMM_True : OpenMM_False);
}
void OpenMMGBMV_GBMVForce_getLambdaState(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState) {
  reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->getLambdaState(
      *reinterpret_cast<OpenMM::Context*>(context),
      *reinterpret_cast<std::vector<double>*>(lambdaState));
}
void OpenMMGBMV_GBMVForce_setLambdaState(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState) {
  reinterpret_cast<OpenMMGBMV::GBMVForce*>(target)->getLambdaState(
      *reinterpret_cast<OpenMM::Context*>(context),
      *reinterpret_cast<std::vector<double>*>(lambdaState));
}
}
