
#ifndef CHARMM_OPENMM_CWRAPPER_H_
#define CHARMM_OPENMM_CWRAPPER_H_

#ifndef OPENMM_EXPORT
#define OPENMM_EXPORT
#endif

/* Global Constants */
 

/* Type Declarations */
 
typedef struct OpenMM_MonteCarloBarostat2_struct OpenMM_MonteCarloBarostat2;

#if defined(__cplusplus)
extern "C" {
#endif

 
 

/* OpenMM::MonteCarloBarostat2*/
extern OPENMM_EXPORT OpenMM_MonteCarloBarostat2* OpenMM_MonteCarloBarostat2_create(double defaultPressure, double temperature, OpenMM_Vec3* pressure3D, double surfaceTension, int frequency);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_destroy(OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT const char* OpenMM_MonteCarloBarostat2_Pressure();
extern OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getDefaultPressure(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT int OpenMM_MonteCarloBarostat2_getFrequency(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setFrequency(OpenMM_MonteCarloBarostat2* target, int freq);
extern OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getTemperature(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setTemperature(OpenMM_MonteCarloBarostat2* target, double temp);
extern OPENMM_EXPORT int OpenMM_MonteCarloBarostat2_getRandomNumberSeed(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setPressureIn3Dimensions(OpenMM_MonteCarloBarostat2* target, OpenMM_Vec3* pressures);
extern OPENMM_EXPORT OpenMM_Vec3* OpenMM_MonteCarloBarostat2_getPressureIn3Dimensions(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setSurfaceTension(OpenMM_MonteCarloBarostat2* target, double tension);
extern OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getSurfaceTension(const OpenMM_MonteCarloBarostat2* target);
extern OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setRandomNumberSeed(OpenMM_MonteCarloBarostat2* target, int seed);

#if defined(__cplusplus)
}
#endif

#endif /*CHARMM_OPENMM_CWRAPPER_H_*/

