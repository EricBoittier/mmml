
#include "OpenMM.h"
#include "OpenMMCharmm.h"
#include "OpenMMCWrapper.h"
#include "CharmmOpenMMCWrapper.h"
#include <cstring>
#include <vector>

using namespace OpenMM;
using namespace std;

/* Utilities for dealing with Fortran's blank-padded strings. */
static void copyAndPadString(char* dest, const char* source, int length) {
    bool reachedEnd = false;
    for (int i = 0; i < length; i++) {
        if (source[i] == 0)
            reachedEnd = true;
        dest[i] = (reachedEnd ? ' ' : source[i]);
    }
}

static string makeString(const char* fsrc, int length) {
    while (length && fsrc[length-1]==' ')
        --length;
    return string(fsrc, length);
}

extern "C" {

 
 

/* OpenMM::MonteCarloBarostat2*/

OPENMM_EXPORT void openmm_montecarlobarostat2_create_(OpenMM_MonteCarloBarostat2*& result, double const& defaultPressure, double const& temperature, OpenMM_Vec3* pressure3D, double const& surfaceTension, int const& frequency) {
    result = OpenMM_MonteCarloBarostat2_create(defaultPressure, temperature, pressure3D, surfaceTension, frequency);
}

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_CREATE(OpenMM_MonteCarloBarostat2*& result, double const& defaultPressure, double const& temperature, OpenMM_Vec3* pressure3D, double const& surfaceTension, int const& frequency) {
    result = OpenMM_MonteCarloBarostat2_create(defaultPressure, temperature, pressure3D, surfaceTension, frequency);
}

OPENMM_EXPORT void openmm_montecarlobarostat2_destroy_(OpenMM_MonteCarloBarostat2*& destroy) {
    OpenMM_MonteCarloBarostat2_destroy(destroy);
    destroy = 0;
}

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_DESTROY(OpenMM_MonteCarloBarostat2*& destroy) {
    OpenMM_MonteCarloBarostat2_destroy(destroy);
    destroy = 0;
}

OPENMM_EXPORT void openmm_montecarlobarostat2_pressure_(char* result, int result_length) {
    const char* result_chars = OpenMM_MonteCarloBarostat2_Pressure();
    copyAndPadString(result, result_chars, result_length);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_PRESSURE(char* result, int result_length) {
    const char* result_chars = OpenMM_MonteCarloBarostat2_Pressure();
    copyAndPadString(result, result_chars, result_length);
};

OPENMM_EXPORT double openmm_montecarlobarostat2_getdefaultpressure_(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getDefaultPressure(target);
};

OPENMM_EXPORT double OPENMM_MONTECARLOBAROSTAT2_GETDEFAULTPRESSURE(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getDefaultPressure(target);
};

OPENMM_EXPORT int openmm_montecarlobarostat2_getfrequency_(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getFrequency(target);
};

OPENMM_EXPORT int OPENMM_MONTECARLOBAROSTAT2_GETFREQUENCY(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getFrequency(target);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_setfrequency_(OpenMM_MonteCarloBarostat2*& target, int const& freq) {
    OpenMM_MonteCarloBarostat2_setFrequency(target, freq);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_SETFREQUENCY(OpenMM_MonteCarloBarostat2*& target, int const& freq) {
    OpenMM_MonteCarloBarostat2_setFrequency(target, freq);
};

OPENMM_EXPORT double openmm_montecarlobarostat2_gettemperature_(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getTemperature(target);
};

OPENMM_EXPORT double OPENMM_MONTECARLOBAROSTAT2_GETTEMPERATURE(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getTemperature(target);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_settemperature_(OpenMM_MonteCarloBarostat2*& target, double const& temp) {
    OpenMM_MonteCarloBarostat2_setTemperature(target, temp);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_SETTEMPERATURE(OpenMM_MonteCarloBarostat2*& target, double const& temp) {
    OpenMM_MonteCarloBarostat2_setTemperature(target, temp);
};

OPENMM_EXPORT int openmm_montecarlobarostat2_getrandomnumberseed_(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getRandomNumberSeed(target);
};

OPENMM_EXPORT int OPENMM_MONTECARLOBAROSTAT2_GETRANDOMNUMBERSEED(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getRandomNumberSeed(target);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_setpressurein3dimensions_(OpenMM_MonteCarloBarostat2*& target, OpenMM_Vec3* pressures) {
    OpenMM_MonteCarloBarostat2_setPressureIn3Dimensions(target, pressures);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_SETPRESSUREIN3DIMENSIONS(OpenMM_MonteCarloBarostat2*& target, OpenMM_Vec3* pressures) {
    OpenMM_MonteCarloBarostat2_setPressureIn3Dimensions(target, pressures);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_getpressurein3dimensions_(const OpenMM_MonteCarloBarostat2*& target, OpenMM_Vec3*& result) {
    result = OpenMM_MonteCarloBarostat2_getPressureIn3Dimensions(target);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_GETPRESSUREIN3DIMENSIONS(const OpenMM_MonteCarloBarostat2*& target, OpenMM_Vec3*& result) {
    result = OpenMM_MonteCarloBarostat2_getPressureIn3Dimensions(target);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_setsurfacetension_(OpenMM_MonteCarloBarostat2*& target, double const& tension) {
    OpenMM_MonteCarloBarostat2_setSurfaceTension(target, tension);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_SETSURFACETENSION(OpenMM_MonteCarloBarostat2*& target, double const& tension) {
    OpenMM_MonteCarloBarostat2_setSurfaceTension(target, tension);
};

OPENMM_EXPORT double openmm_montecarlobarostat2_getsurfacetension_(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getSurfaceTension(target);
};

OPENMM_EXPORT double OPENMM_MONTECARLOBAROSTAT2_GETSURFACETENSION(const OpenMM_MonteCarloBarostat2*& target) {
    return OpenMM_MonteCarloBarostat2_getSurfaceTension(target);
};

OPENMM_EXPORT void openmm_montecarlobarostat2_setrandomnumberseed_(OpenMM_MonteCarloBarostat2*& target, int const& seed) {
    OpenMM_MonteCarloBarostat2_setRandomNumberSeed(target, seed);
};

OPENMM_EXPORT void OPENMM_MONTECARLOBAROSTAT2_SETRANDOMNUMBERSEED(OpenMM_MonteCarloBarostat2*& target, int const& seed) {
    OpenMM_MonteCarloBarostat2_setRandomNumberSeed(target, seed);
};


/* OpenMM::Force*/

OPENMM_EXPORT void openmm_force_destroy_(OpenMM_Force*& destroy) {
    OpenMM_Force_destroy(destroy);
    destroy = 0;
}

OPENMM_EXPORT void OPENMM_FORCE_DESTROY(OpenMM_Force*& destroy) {
    OpenMM_Force_destroy(destroy);
    destroy = 0;
}

OPENMM_EXPORT int openmm_force_getforcegroup_(const OpenMM_Force*& target) {
    return OpenMM_Force_getForceGroup(target);
};

OPENMM_EXPORT int OPENMM_FORCE_GETFORCEGROUP(const OpenMM_Force*& target) {
    return OpenMM_Force_getForceGroup(target);
};

OPENMM_EXPORT void openmm_force_setforcegroup_(OpenMM_Force*& target, int const& group) {
    OpenMM_Force_setForceGroup(target, group);
};

OPENMM_EXPORT void OPENMM_FORCE_SETFORCEGROUP(OpenMM_Force*& target, int const& group) {
    OpenMM_Force_setForceGroup(target, group);
};


}

