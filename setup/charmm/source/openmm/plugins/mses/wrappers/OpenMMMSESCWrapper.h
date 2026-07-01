
#ifndef OPENMM_MSES_CWRAPPER_H_
#define OPENMM_MSES_CWRAPPER_H_

#include "OpenMMCWrapper.h"

typedef struct OpenMMMSES_MSESForce_struct OpenMMMSES_MSESForce;

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
  OpenMMMSES_MSESForce_NoCutoff = 0, OpenMMMSES_MSESForce_CutoffNonPeriodic = 1, OpenMMMSES_MSESForce_CutoffPeriodic = 2
} OpenMMMSES_MSESForce_NonbondedMethod;

extern OpenMMMSES_MSESForce* OpenMMMSES_MSESForce_create();
extern int OpenMMMSES_MSESForce_getNumDistPair(const OpenMMMSES_MSESForce* target);
extern int OpenMMMSES_MSESForce_addDistPair(OpenMMMSES_MSESForce* target, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp);
extern void OpenMMMSES_MSESForce_setDistPairParameters(OpenMMMSES_MSESForce* target, int index, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp);
extern void OpenMMMSES_MSESForce_updateParametersInContext(OpenMMMSES_MSESForce* target, OpenMM_Context* context);
extern OpenMM_Boolean OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions(const OpenMMMSES_MSESForce* target);
#if defined(__cplusplus)
}
#endif
#endif /*OPENMM_MSES_CWRAPPER_H_*/
