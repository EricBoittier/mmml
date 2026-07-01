
#include "OpenMM.h"
#include "OpenMMCharmm.h"
#include "OpenMMCWrapper.h"
#include "CharmmOpenMMCWrapper.h"
#include <cstring>
#include <vector>

using namespace OpenMM;
using namespace std;

extern "C" {

 
 

/* OpenMM::MonteCarloBarostat2*/
OPENMM_EXPORT OpenMM_MonteCarloBarostat2* OpenMM_MonteCarloBarostat2_create(double defaultPressure, double temperature, OpenMM_Vec3* pressure3D, double surfaceTension, int frequency) {
    return reinterpret_cast<OpenMM_MonteCarloBarostat2*>(new MonteCarloBarostat2(defaultPressure, temperature, *reinterpret_cast<Vec3* >(pressure3D), surfaceTension, frequency));
}

OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_destroy(OpenMM_MonteCarloBarostat2* target) {
    delete reinterpret_cast<MonteCarloBarostat2*>(target);
}
 
 

OPENMM_EXPORT const char* OpenMM_MonteCarloBarostat2_Pressure() {
    const string* result = &MonteCarloBarostat2::Pressure();
    return result->c_str();
};


OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getDefaultPressure(const OpenMM_MonteCarloBarostat2* target) {
    double result = reinterpret_cast<const MonteCarloBarostat2*>(target)->getDefaultPressure();
    return result;
};


OPENMM_EXPORT int OpenMM_MonteCarloBarostat2_getFrequency(const OpenMM_MonteCarloBarostat2* target) {
    int result = reinterpret_cast<const MonteCarloBarostat2*>(target)->getFrequency();
    return result;
};


OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setFrequency(OpenMM_MonteCarloBarostat2* target, int freq) {
    reinterpret_cast<MonteCarloBarostat2*>(target)->setFrequency(freq);
};


OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getTemperature(const OpenMM_MonteCarloBarostat2* target) {
    double result = reinterpret_cast<const MonteCarloBarostat2*>(target)->getTemperature();
    return result;
};


OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setTemperature(OpenMM_MonteCarloBarostat2* target, double temp) {
    reinterpret_cast<MonteCarloBarostat2*>(target)->setTemperature(temp);
};


OPENMM_EXPORT int OpenMM_MonteCarloBarostat2_getRandomNumberSeed(const OpenMM_MonteCarloBarostat2* target) {
    int result = reinterpret_cast<const MonteCarloBarostat2*>(target)->getRandomNumberSeed();
    return result;
};


OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setPressureIn3Dimensions(OpenMM_MonteCarloBarostat2* target, OpenMM_Vec3* pressures) {
    reinterpret_cast<MonteCarloBarostat2*>(target)->setPressureIn3Dimensions(*reinterpret_cast<Vec3* >(pressures));
};


OPENMM_EXPORT OpenMM_Vec3* OpenMM_MonteCarloBarostat2_getPressureIn3Dimensions(const OpenMM_MonteCarloBarostat2* target) {
    Vec3* result = &reinterpret_cast<const MonteCarloBarostat2*>(target)->getPressureIn3Dimensions();
    return reinterpret_cast<OpenMM_Vec3*>(result);
};


OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setSurfaceTension(OpenMM_MonteCarloBarostat2* target, double tension) {
    reinterpret_cast<MonteCarloBarostat2*>(target)->setSurfaceTension(tension);
};


OPENMM_EXPORT double OpenMM_MonteCarloBarostat2_getSurfaceTension(const OpenMM_MonteCarloBarostat2* target) {
    double result = reinterpret_cast<const MonteCarloBarostat2*>(target)->getSurfaceTension();
    return result;
};


OPENMM_EXPORT void OpenMM_MonteCarloBarostat2_setRandomNumberSeed(OpenMM_MonteCarloBarostat2* target, int seed) {
    reinterpret_cast<MonteCarloBarostat2*>(target)->setRandomNumberSeed(seed);
};


/* OpenMM::Force*/
OPENMM_EXPORT void OpenMM_Force_destroy(OpenMM_Force* target) {
    delete reinterpret_cast<Force*>(target);
}
 
 

OPENMM_EXPORT int OpenMM_Force_getForceGroup(const OpenMM_Force* target) {
    int result = reinterpret_cast<const Force*>(target)->getForceGroup();
    return result;
};


OPENMM_EXPORT void OpenMM_Force_setForceGroup(OpenMM_Force* target, int group) {
    reinterpret_cast<Force*>(target)->setForceGroup(group);
};

 
}

