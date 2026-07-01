
#ifndef OPENMM_GBMV_CWRAPPER_H_
#define OPENMM_GBMV_CWRAPPER_H_

#include "OpenMMCWrapper.h"

typedef struct OpenMMGBMV_GBMVForce_struct OpenMMGBMV_GBMVForce;

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  OpenMMGBMV_GBMVForce_NoCutoff = 0, OpenMMGBMV_GBMVForce_CutoffNonPeriodic = 1, OpenMMGBMV_GBMVForce_CutoffPeriodic = 2
} OpenMMGBMV_GBMVForce_NonbondedMethod;

extern OpenMMGBMV_GBMVForce* OpenMMGBMV_GBMVForce_create();
extern void OpenMMGBMV_GBMVForce_destroy(OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_addCPHMDForce(OpenMMGBMV_GBMVForce* target, double pH, double T_theta, double mTheta, double ts_theta, double beta, int outFreq, char* fileName);
extern OpenMM_Boolean OpenMMGBMV_GBMVForce_usingCPHMD(const OpenMMGBMV_GBMVForce* target);
extern int OpenMMGBMV_GBMVForce_getNumTitratingGroups(const OpenMMGBMV_GBMVForce* target);
extern int OpenMMGBMV_GBMVForce_addTitratingGroupParameters(OpenMMGBMV_GBMVForce* target, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8);
extern void OpenMMGBMV_GBMVForce_getTitratingGroupParameters(const OpenMMGBMV_GBMVForce* target, int index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8);
extern void OpenMMGBMV_GBMVForce_setTitratingGroupParameters(OpenMMGBMV_GBMVForce* target, int index, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8);
extern int OpenMMGBMV_GBMVForce_addNonbondedException(OpenMMGBMV_GBMVForce* target, int atom1, int atom2);
extern void OpenMMGBMV_GBMVForce_getNonbondedException(const OpenMMGBMV_GBMVForce* target, int index, int* atom1, int* atom2);
extern int OpenMMGBMV_GBMVForce_getNumNonbondedExceptions(const OpenMMGBMV_GBMVForce* target);
extern int OpenMMGBMV_GBMVForce_getNumParticles(const OpenMMGBMV_GBMVForce* target);
extern int OpenMMGBMV_GBMVForce_addParticle(OpenMMGBMV_GBMVForce* target, double charge, double radius);
extern void OpenMMGBMV_GBMVForce_getParticleParameters(const OpenMMGBMV_GBMVForce* target, int index, double* charge, double* radius);
extern void OpenMMGBMV_GBMVForce_setParticleParameters(OpenMMGBMV_GBMVForce* target, int index, double charge, double radius);
extern int OpenMMGBMV_GBMVForce_addCphmdParameters(OpenMMGBMV_GBMVForce* target, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2);
extern void OpenMMGBMV_GBMVForce_getCphmdParameters(const OpenMMGBMV_GBMVForce* target, int index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2);
extern void OpenMMGBMV_GBMVForce_setCphmdParameters(OpenMMGBMV_GBMVForce* target, int index, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2);
extern char* OpenMMGBMV_GBMVForce_getLambdaOutputFile(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setLambdaOutputFile(OpenMMGBMV_GBMVForce* target, char* tmp);
extern double OpenMMGBMV_GBMVForce_getSystemPH(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSystemPH(OpenMMGBMV_GBMVForce* target, double tmp);
extern double OpenMMGBMV_GBMVForce_getThetaTemp(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setThetaTemp(OpenMMGBMV_GBMVForce* target, double tmp);
extern double OpenMMGBMV_GBMVForce_getThetaMass(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setThetaMass(OpenMMGBMV_GBMVForce* target, double tmp);
extern int OpenMMGBMV_GBMVForce_getLambdaOutputFrequency(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setLambdaOutputFrequency(OpenMMGBMV_GBMVForce* target, int tmp);
extern double OpenMMGBMV_GBMVForce_getPHbeta(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setPHbeta(OpenMMGBMV_GBMVForce* target, double tmp);
extern double OpenMMGBMV_GBMVForce_getSolventDielectric(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSolventDielectric(OpenMMGBMV_GBMVForce* target, double dielectric);
extern double OpenMMGBMV_GBMVForce_getSoluteDielectric(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSoluteDielectric(OpenMMGBMV_GBMVForce* target, double dielectric);
extern double OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(OpenMMGBMV_GBMVForce* target, double energy);
extern double OpenMMGBMV_GBMVForce_getAA0(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setAA0(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getAA1(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setAA1(OpenMMGBMV_GBMVForce* target, double value);
extern int OpenMMGBMV_GBMVForce_getNumGauLegRad(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setNumGauLegRad(OpenMMGBMV_GBMVForce* target, int number);
extern int OpenMMGBMV_GBMVForce_getNumLebAng(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setNumLebAng(OpenMMGBMV_GBMVForce* target, int number);
extern double OpenMMGBMV_GBMVForce_getDebyeHuckelLength(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setDebyeHuckelLength(OpenMMGBMV_GBMVForce* target, double length);
extern double OpenMMGBMV_GBMVForce_getSwitchingLength(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSwitchingLength(OpenMMGBMV_GBMVForce* target, double length);
// GBMV2: Parameters of GBMV2 model
extern double OpenMMGBMV_GBMVForce_getBETA_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setBETA_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getALPHA_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setALPHA_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getSLOPE_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getSHIFT_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getHSX1_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setHSX1_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getHSX2_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setHSX2_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getONX_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setONX_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getOFFX_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setOFFX_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getP1_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setP1_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getP2_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setP2_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getP3_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setP3_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern double OpenMMGBMV_GBMVForce_getP6_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setP6_GBMV2(OpenMMGBMV_GBMVForce* target, double value);
extern int OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(OpenMMGBMV_GBMVForce* target, int value);
// END GBMV2
extern double OpenMMGBMV_GBMVForce_getMembraneThickness(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setMembraneThickness(OpenMMGBMV_GBMVForce* target, double tmp);
extern double OpenMMGBMV_GBMVForce_getMembraneSwLen(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setMembraneSwLen(OpenMMGBMV_GBMVForce* target, double tmp);
extern OpenMMGBMV_GBMVForce_NonbondedMethod OpenMMGBMV_GBMVForce_getNonbondedMethod(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setNonbondedMethod(OpenMMGBMV_GBMVForce* target, OpenMMGBMV_GBMVForce_NonbondedMethod method);
extern double OpenMMGBMV_GBMVForce_getCutoffDistance(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setCutoffDistance(OpenMMGBMV_GBMVForce* target, double distance);
extern double OpenMMGBMV_GBMVForce_getCutonDistance(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setCutonDistance(OpenMMGBMV_GBMVForce* target, double distance);
extern double OpenMMGBMV_GBMVForce_getReactionFieldDielectric(const OpenMMGBMV_GBMVForce* target);
extern void OpenMMGBMV_GBMVForce_setReactionFieldDielectric(OpenMMGBMV_GBMVForce* target, double tmp);
extern void OpenMMGBMV_GBMVForce_updateParametersInContext(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context);
extern OpenMM_Boolean OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions(const OpenMMGBMV_GBMVForce* target);
void OpenMMGBMV_GBMVForce_getLambdaState(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState);
void OpenMMGBMV_GBMVForce_setLambdaState(OpenMMGBMV_GBMVForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState);
#if defined(__cplusplus)
}
#endif
#endif /*OPENMM_GBMV_CWRAPPER_H_*/
