
#ifndef OPENMM_GBSW_CWRAPPER_H_
#define OPENMM_GBSW_CWRAPPER_H_

#include "OpenMMCWrapper.h"

typedef struct OpenMMGBSW_GBSWForce_struct OpenMMGBSW_GBSWForce;

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  OpenMMGBSW_GBSWForce_NoCutoff = 0, OpenMMGBSW_GBSWForce_CutoffNonPeriodic = 1, OpenMMGBSW_GBSWForce_CutoffPeriodic = 2
} OpenMMGBSW_GBSWForce_NonbondedMethod;

extern OpenMMGBSW_GBSWForce* OpenMMGBSW_GBSWForce_create();
extern void OpenMMGBSW_GBSWForce_destroy(OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_addCPHMDForce(OpenMMGBSW_GBSWForce* target, double pH, double T_theta, double mTheta, double ts_theta, double beta, int outFreq, char* fileName);
extern OpenMM_Boolean OpenMMGBSW_GBSWForce_usingCPHMD(const OpenMMGBSW_GBSWForce* target);
extern int OpenMMGBSW_GBSWForce_getNumTitratingGroups(const OpenMMGBSW_GBSWForce* target);
extern int OpenMMGBSW_GBSWForce_addTitratingGroupParameters(OpenMMGBSW_GBSWForce* target, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8);
extern void OpenMMGBSW_GBSWForce_getTitratingGroupParameters(const OpenMMGBSW_GBSWForce* target, int index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8);
extern void OpenMMGBSW_GBSWForce_setTitratingGroupParameters(OpenMMGBSW_GBSWForce* target, int index, double resPKA1, double resPKA2, double barrier1, double barrier2, double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8);
extern int OpenMMGBSW_GBSWForce_addNonbondedException(OpenMMGBSW_GBSWForce* target, int atom1, int atom2);
extern void OpenMMGBSW_GBSWForce_getNonbondedException(const OpenMMGBSW_GBSWForce* target, int index, int* atom1, int* atom2);
extern int OpenMMGBSW_GBSWForce_getNumNonbondedExceptions(const OpenMMGBSW_GBSWForce* target);
extern int OpenMMGBSW_GBSWForce_getNumParticles(const OpenMMGBSW_GBSWForce* target);
extern int OpenMMGBSW_GBSWForce_addParticle(OpenMMGBSW_GBSWForce* target, double charge, double radius);
extern void OpenMMGBSW_GBSWForce_getParticleParameters(const OpenMMGBSW_GBSWForce* target, int index, double* charge, double* radius);

  extern void OpenMMGBSW_GBSWForce_setDoingDynamics(OpenMMGBSW_GBSWForce* target, OpenMM_Context * context, int flag);

extern void OpenMMGBSW_GBSWForce_setParticleParameters(OpenMMGBSW_GBSWForce* target, int index, double charge, double radius);
extern int OpenMMGBSW_GBSWForce_addCphmdParameters(OpenMMGBSW_GBSWForce* target, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2);
extern void OpenMMGBSW_GBSWForce_getCphmdParameters(const OpenMMGBSW_GBSWForce* target, int index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2);
extern void OpenMMGBSW_GBSWForce_setCphmdParameters(OpenMMGBSW_GBSWForce* target, int index, int titrateResID, double refChargeState1, double refChargeState2, double chargeState1, double chargeState2);
extern char* OpenMMGBSW_GBSWForce_getLambdaOutputFile(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setLambdaOutputFile(OpenMMGBSW_GBSWForce* target, char* tmp);
extern double OpenMMGBSW_GBSWForce_getSystemPH(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setSystemPH(OpenMMGBSW_GBSWForce* target, double tmp);
extern double OpenMMGBSW_GBSWForce_getThetaTemp(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setThetaTemp(OpenMMGBSW_GBSWForce* target, double tmp);
extern double OpenMMGBSW_GBSWForce_getThetaMass(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setThetaMass(OpenMMGBSW_GBSWForce* target, double tmp);
extern int OpenMMGBSW_GBSWForce_getLambdaOutputFrequency(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setLambdaOutputFrequency(OpenMMGBSW_GBSWForce* target, int tmp);
extern double OpenMMGBSW_GBSWForce_getPHbeta(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setPHbeta(OpenMMGBSW_GBSWForce* target, double tmp);
extern double OpenMMGBSW_GBSWForce_getSolventDielectric(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setSolventDielectric(OpenMMGBSW_GBSWForce* target, double dielectric);
extern double OpenMMGBSW_GBSWForce_getSoluteDielectric(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setSoluteDielectric(OpenMMGBSW_GBSWForce* target, double dielectric);
extern double OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(OpenMMGBSW_GBSWForce* target, double energy);
extern double OpenMMGBSW_GBSWForce_getAA0(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setAA0(OpenMMGBSW_GBSWForce* target, double value);
extern double OpenMMGBSW_GBSWForce_getAA1(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setAA1(OpenMMGBSW_GBSWForce* target, double value);
extern int OpenMMGBSW_GBSWForce_getNumGauLegRad(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setNumGauLegRad(OpenMMGBSW_GBSWForce* target, int number);
extern int OpenMMGBSW_GBSWForce_getNumLebAng(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setNumLebAng(OpenMMGBSW_GBSWForce* target, int number);
extern double OpenMMGBSW_GBSWForce_getDebyeHuckelLength(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setDebyeHuckelLength(OpenMMGBSW_GBSWForce* target, double length);
extern double OpenMMGBSW_GBSWForce_getSwitchingLength(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setSwitchingLength(OpenMMGBSW_GBSWForce* target, double length);
extern double OpenMMGBSW_GBSWForce_getMembraneThickness(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setMembraneThickness(OpenMMGBSW_GBSWForce* target, double tmp);
extern double OpenMMGBSW_GBSWForce_getMembraneSwLen(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setMembraneSwLen(OpenMMGBSW_GBSWForce* target, double tmp);
extern OpenMMGBSW_GBSWForce_NonbondedMethod OpenMMGBSW_GBSWForce_getNonbondedMethod(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setNonbondedMethod(OpenMMGBSW_GBSWForce* target, OpenMMGBSW_GBSWForce_NonbondedMethod method);
extern double OpenMMGBSW_GBSWForce_getCutoffDistance(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setCutoffDistance(OpenMMGBSW_GBSWForce* target, double distance);
extern double OpenMMGBSW_GBSWForce_getReactionFieldDielectric(const OpenMMGBSW_GBSWForce* target);
extern void OpenMMGBSW_GBSWForce_setReactionFieldDielectric(OpenMMGBSW_GBSWForce* target, double tmp);
extern void OpenMMGBSW_GBSWForce_updateParametersInContext(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context);
extern OpenMM_Boolean OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions(const OpenMMGBSW_GBSWForce* target);
void OpenMMGBSW_GBSWForce_getLambdaState(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState);
void OpenMMGBSW_GBSWForce_setLambdaState(OpenMMGBSW_GBSWForce* target, OpenMM_Context* context, OpenMM_DoubleArray* lambdaState);
#if defined(__cplusplus)
}
#endif
#endif /*OPENMM_GBSW_CWRAPPER_H_*/
