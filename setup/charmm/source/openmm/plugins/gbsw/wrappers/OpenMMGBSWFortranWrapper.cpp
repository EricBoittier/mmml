#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMGBSW.h"
#include "OpenMMGBSWCWrapper.h"
#include <cstring>
#include <vector>
#include <cstdlib>

using namespace OpenMM;
using namespace OpenMMGBSW;
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

static void convertStringToChars(char* source, char*& cstr, int& length) {
	length = strlen(source);
	cstr = new char[length+1];
	strcpy(cstr, source);
    free(source);
}

extern "C" {
void openmmgbsw_gbswforce_create_(OpenMMGBSW_GBSWForce*& result) {
    result = OpenMMGBSW_GBSWForce_create();
}
void OPENMMGBSW_GBSWFORCE_CREATE(OpenMMGBSW_GBSWForce*& result) {
    result = OpenMMGBSW_GBSWForce_create();
}
void openmmgbsw_gbswforce_destroy_(OpenMMGBSW_GBSWForce*& destroy) {
    OpenMMGBSW_GBSWForce_destroy(destroy);
    destroy = 0;
}
void OPENMMGBSW_GBSWFORCE_DESTROY(OpenMMGBSW_GBSWForce*& destroy) {
    OpenMMGBSW_GBSWForce_destroy(destroy);
    destroy = 0;
}

  void openmmgbsw_gbswforce_setdoingdynamics_(OpenMMGBSW_GBSWForce*& target, OpenMM_Context*& context, int const& flag) {
    OpenMMGBSW_GBSWForce_setDoingDynamics(target, context, flag);
  }

  void OPENMMGBSW_GBSWFORCE_SETDOINGDYNAMICS(OpenMMGBSW_GBSWForce*& target, OpenMM_Context*& context, int const& flag) {
    OpenMMGBSW_GBSWForce_setDoingDynamics(target, context, flag);
  }

void openmmgbsw_gbswforce_addcphmdforce_(OpenMMGBSW_GBSWForce*& target, double const& pH, double const& T_theta, double const& mTheta, double const& ts_theta, double const& beta, int const& outFreq, char* fileName) {
    OpenMMGBSW_GBSWForce_addCPHMDForce(target, pH, T_theta, mTheta, ts_theta, beta, outFreq, fileName);
}
void OPENMMGBSW_GBSWFORCE_ADDCPHMDFORCE(OpenMMGBSW_GBSWForce*& target, double const& pH, double const& T_theta, double const& mTheta, double const& ts_theta, double const& beta, int const& outFreq, char* fileName) {
    OpenMMGBSW_GBSWForce_addCPHMDForce(target, pH, T_theta, mTheta, ts_theta, beta, outFreq, fileName);
}
void openmmgbsw_gbswforce_usingcphmd_(const OpenMMGBSW_GBSWForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBSW_GBSWForce_usingCPHMD(target);
}
void OPENMMGBSW_GBSWFORCE_USINGCPHMD(const OpenMMGBSW_GBSWForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBSW_GBSWForce_usingCPHMD(target);
}
int openmmgbsw_gbswforce_getnumtitratinggroups_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumTitratingGroups(target);
}
int OPENMMGBSW_GBSWFORCE_GETNUMTITRATINGGROUPS(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumTitratingGroups(target);
}
int openmmgbsw_gbswforce_addtitratinggroupparameters_(OpenMMGBSW_GBSWForce*& target, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    return OpenMMGBSW_GBSWForce_addTitratingGroupParameters(target, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int OPENMMGBSW_GBSWFORCE_ADDTITRATINGGROUPPARAMETERS(OpenMMGBSW_GBSWForce*& target, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    return OpenMMGBSW_GBSWForce_addTitratingGroupParameters(target, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void openmmgbsw_gbswforce_gettitratinggroupparameters_(const OpenMMGBSW_GBSWForce*& target, int const& index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    OpenMMGBSW_GBSWForce_getTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void OPENMMGBSW_GBSWFORCE_GETTITRATINGGROUPPARAMETERS(const OpenMMGBSW_GBSWForce*& target, int const& index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    OpenMMGBSW_GBSWForce_getTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void openmmgbsw_gbswforce_settitratinggroupparameters_(OpenMMGBSW_GBSWForce*& target, int const& index, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    OpenMMGBSW_GBSWForce_setTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void OPENMMGBSW_GBSWFORCE_SETTITRATINGGROUPPARAMETERS(OpenMMGBSW_GBSWForce*& target, int const& index, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    OpenMMGBSW_GBSWForce_setTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int openmmgbsw_gbswforce_addnonbondedexception_(OpenMMGBSW_GBSWForce*& target, int const& atom1, int const& atom2) {
    return OpenMMGBSW_GBSWForce_addNonbondedException(target, atom1, atom2);
}
int OPENMMGBSW_GBSWFORCE_ADDNONBONDEDEXCEPTION(OpenMMGBSW_GBSWForce*& target, int const& atom1, int const& atom2) {
    return OpenMMGBSW_GBSWForce_addNonbondedException(target, atom1, atom2);
}
void openmmgbsw_gbswforce_getnonbondedexception_(const OpenMMGBSW_GBSWForce*& target, int const& index, int* atom1, int* atom2) {
    OpenMMGBSW_GBSWForce_getNonbondedException(target, index, atom1, atom2);
}
void OPENMMGBSW_GBSWFORCE_GETNONBONDEDEXCEPTION(const OpenMMGBSW_GBSWForce*& target, int const& index, int* atom1, int* atom2) {
    OpenMMGBSW_GBSWForce_getNonbondedException(target, index, atom1, atom2);
}
int openmmgbsw_gbswforce_getnumnonbondedexceptions_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumNonbondedExceptions(target);
}
int OPENMMGBSW_GBSWFORCE_GETNUMNONBONDEDEXCEPTIONS(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumNonbondedExceptions(target);
}
int openmmgbsw_gbswforce_getnumparticles_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumParticles(target);
}
int OPENMMGBSW_GBSWFORCE_GETNUMPARTICLES(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumParticles(target);
}
int openmmgbsw_gbswforce_addparticle_(OpenMMGBSW_GBSWForce*& target, double const& charge, double const& radius) {
    return OpenMMGBSW_GBSWForce_addParticle(target, charge, radius);
}
int OPENMMGBSW_GBSWFORCE_ADDPARTICLE(OpenMMGBSW_GBSWForce*& target, double const& charge, double const& radius) {
    return OpenMMGBSW_GBSWForce_addParticle(target, charge, radius);
}
void openmmgbsw_gbswforce_getparticleparameters_(const OpenMMGBSW_GBSWForce*& target, int const& index, double* charge, double* radius) {
    OpenMMGBSW_GBSWForce_getParticleParameters(target, index, charge, radius);
}
void OPENMMGBSW_GBSWFORCE_GETPARTICLEPARAMETERS(const OpenMMGBSW_GBSWForce*& target, int const& index, double* charge, double* radius) {
    OpenMMGBSW_GBSWForce_getParticleParameters(target, index, charge, radius);
}
void openmmgbsw_gbswforce_setparticleparameters_(OpenMMGBSW_GBSWForce*& target, int const& index, double const& charge, double const& radius) {
    OpenMMGBSW_GBSWForce_setParticleParameters(target, index, charge, radius);
}
void OPENMMGBSW_GBSWFORCE_SETPARTICLEPARAMETERS(OpenMMGBSW_GBSWForce*& target, int const& index, double const& charge, double const& radius) {
    OpenMMGBSW_GBSWForce_setParticleParameters(target, index, charge, radius);
}
int openmmgbsw_gbswforce_addcphmdparameters_(OpenMMGBSW_GBSWForce*& target, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    return OpenMMGBSW_GBSWForce_addCphmdParameters(target, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
int OPENMMGBSW_GBSWFORCE_ADDCPHMDPARAMETERS(OpenMMGBSW_GBSWForce*& target, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    return OpenMMGBSW_GBSWForce_addCphmdParameters(target, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbsw_gbswforce_getcphmdparameters_(const OpenMMGBSW_GBSWForce*& target, int const& index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    OpenMMGBSW_GBSWForce_getCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void OPENMMGBSW_GBSWFORCE_GETCPHMDPARAMETERS(const OpenMMGBSW_GBSWForce*& target, int const& index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    OpenMMGBSW_GBSWForce_getCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbsw_gbswforce_setcphmdparameters_(OpenMMGBSW_GBSWForce*& target, int const& index, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    OpenMMGBSW_GBSWForce_setCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void OPENMMGBSW_GBSWFORCE_SETCPHMDPARAMETERS(OpenMMGBSW_GBSWForce*& target, int const& index, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    OpenMMGBSW_GBSWForce_setCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbsw_gbswforce_getlambdaoutputfile_(const OpenMMGBSW_GBSWForce*& target, char*& result) {
    result = OpenMMGBSW_GBSWForce_getLambdaOutputFile(target);
}
void OPENMMGBSW_GBSWFORCE_GETLAMBDAOUTPUTFILE(const OpenMMGBSW_GBSWForce*& target, char*& result) {
    result = OpenMMGBSW_GBSWForce_getLambdaOutputFile(target);
}
void openmmgbsw_gbswforce_setlambdaoutputfile_(OpenMMGBSW_GBSWForce*& target, char* tmp) {
    OpenMMGBSW_GBSWForce_setLambdaOutputFile(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETLAMBDAOUTPUTFILE(OpenMMGBSW_GBSWForce*& target, char* tmp) {
    OpenMMGBSW_GBSWForce_setLambdaOutputFile(target, tmp);
}
double openmmgbsw_gbswforce_getsystemph_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSystemPH(target);
}
double OPENMMGBSW_GBSWFORCE_GETSYSTEMPH(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSystemPH(target);
}
void openmmgbsw_gbswforce_setsystemph_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setSystemPH(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETSYSTEMPH(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setSystemPH(target, tmp);
}
double openmmgbsw_gbswforce_getthetatemp_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getThetaTemp(target);
}
double OPENMMGBSW_GBSWFORCE_GETTHETATEMP(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getThetaTemp(target);
}
void openmmgbsw_gbswforce_setthetatemp_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setThetaTemp(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETTHETATEMP(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setThetaTemp(target, tmp);
}
double openmmgbsw_gbswforce_getthetamass_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getThetaMass(target);
}
double OPENMMGBSW_GBSWFORCE_GETTHETAMASS(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getThetaMass(target);
}
void openmmgbsw_gbswforce_setthetamass_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setThetaMass(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETTHETAMASS(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setThetaMass(target, tmp);
}
int openmmgbsw_gbswforce_getlambdaoutputfrequency_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getLambdaOutputFrequency(target);
}
int OPENMMGBSW_GBSWFORCE_GETLAMBDAOUTPUTFREQUENCY(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getLambdaOutputFrequency(target);
}
void openmmgbsw_gbswforce_setlambdaoutputfrequency_(OpenMMGBSW_GBSWForce*& target, int const& tmp) {
    OpenMMGBSW_GBSWForce_setLambdaOutputFrequency(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETLAMBDAOUTPUTFREQUENCY(OpenMMGBSW_GBSWForce*& target, int const& tmp) {
    OpenMMGBSW_GBSWForce_setLambdaOutputFrequency(target, tmp);
}
double openmmgbsw_gbswforce_getphbeta_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getPHbeta(target);
}
double OPENMMGBSW_GBSWFORCE_GETPHBETA(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getPHbeta(target);
}
void openmmgbsw_gbswforce_setphbeta_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setPHbeta(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETPHBETA(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setPHbeta(target, tmp);
}
double openmmgbsw_gbswforce_getsolventdielectric_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSolventDielectric(target);
}
double OPENMMGBSW_GBSWFORCE_GETSOLVENTDIELECTRIC(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSolventDielectric(target);
}
void openmmgbsw_gbswforce_setsolventdielectric_(OpenMMGBSW_GBSWForce*& target, double const& dielectric) {
    OpenMMGBSW_GBSWForce_setSolventDielectric(target, dielectric);
}
void OPENMMGBSW_GBSWFORCE_SETSOLVENTDIELECTRIC(OpenMMGBSW_GBSWForce*& target, double const& dielectric) {
    OpenMMGBSW_GBSWForce_setSolventDielectric(target, dielectric);
}
double openmmgbsw_gbswforce_getsolutedielectric_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSoluteDielectric(target);
}
double OPENMMGBSW_GBSWFORCE_GETSOLUTEDIELECTRIC(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSoluteDielectric(target);
}
void openmmgbsw_gbswforce_setsolutedielectric_(OpenMMGBSW_GBSWForce*& target, double const& dielectric) {
    OpenMMGBSW_GBSWForce_setSoluteDielectric(target, dielectric);
}
void OPENMMGBSW_GBSWFORCE_SETSOLUTEDIELECTRIC(OpenMMGBSW_GBSWForce*& target, double const& dielectric) {
    OpenMMGBSW_GBSWForce_setSoluteDielectric(target, dielectric);
}
double openmmgbsw_gbswforce_getsurfaceareaenergy_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy(target);
}
double OPENMMGBSW_GBSWFORCE_GETSURFACEAREAENERGY(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy(target);
}
void openmmgbsw_gbswforce_setsurfaceareaenergy_(OpenMMGBSW_GBSWForce*& target, double const& energy) {
    OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(target, energy);
}
void OPENMMGBSW_GBSWFORCE_SETSURFACEAREAENERGY(OpenMMGBSW_GBSWForce*& target, double const& energy) {
    OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(target, energy);
}
double openmmgbsw_gbswforce_getaa0_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getAA0(target);
}
double OPENMMGBSW_GBSWFORCE_GETAA0(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getAA0(target);
}
void openmmgbsw_gbswforce_setaa0_(OpenMMGBSW_GBSWForce*& target, double const& value) {
    OpenMMGBSW_GBSWForce_setAA0(target, value);
}
void OPENMMGBSW_GBSWFORCE_SETAA0(OpenMMGBSW_GBSWForce*& target, double const& value) {
    OpenMMGBSW_GBSWForce_setAA0(target, value);
}
double openmmgbsw_gbswforce_getaa1_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getAA1(target);
}
double OPENMMGBSW_GBSWFORCE_GETAA1(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getAA1(target);
}
void openmmgbsw_gbswforce_setaa1_(OpenMMGBSW_GBSWForce*& target, double const& value) {
    OpenMMGBSW_GBSWForce_setAA1(target, value);
}
void OPENMMGBSW_GBSWFORCE_SETAA1(OpenMMGBSW_GBSWForce*& target, double const& value) {
    OpenMMGBSW_GBSWForce_setAA1(target, value);
}
int openmmgbsw_gbswforce_getnumgaulegrad_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumGauLegRad(target);
}
int OPENMMGBSW_GBSWFORCE_GETNUMGAULEGRAD(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumGauLegRad(target);
}
void openmmgbsw_gbswforce_setnumgaulegrad_(OpenMMGBSW_GBSWForce*& target, int const& number) {
    OpenMMGBSW_GBSWForce_setNumGauLegRad(target, number);
}
void OPENMMGBSW_GBSWFORCE_SETNUMGAULEGRAD(OpenMMGBSW_GBSWForce*& target, int const& number) {
    OpenMMGBSW_GBSWForce_setNumGauLegRad(target, number);
}
int openmmgbsw_gbswforce_getnumlebang_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumLebAng(target);
}
int OPENMMGBSW_GBSWFORCE_GETNUMLEBANG(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getNumLebAng(target);
}
void openmmgbsw_gbswforce_setnumlebang_(OpenMMGBSW_GBSWForce*& target, int const& number) {
    OpenMMGBSW_GBSWForce_setNumLebAng(target, number);
}
void OPENMMGBSW_GBSWFORCE_SETNUMLEBANG(OpenMMGBSW_GBSWForce*& target, int const& number) {
    OpenMMGBSW_GBSWForce_setNumLebAng(target, number);
}
double openmmgbsw_gbswforce_getdebyehuckellength_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getDebyeHuckelLength(target);
}
double OPENMMGBSW_GBSWFORCE_GETDEBYEHUCKELLENGTH(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getDebyeHuckelLength(target);
}
void openmmgbsw_gbswforce_setdebyehuckellength_(OpenMMGBSW_GBSWForce*& target, double const& length) {
    OpenMMGBSW_GBSWForce_setDebyeHuckelLength(target, length);
}
void OPENMMGBSW_GBSWFORCE_SETDEBYEHUCKELLENGTH(OpenMMGBSW_GBSWForce*& target, double const& length) {
    OpenMMGBSW_GBSWForce_setDebyeHuckelLength(target, length);
}
double openmmgbsw_gbswforce_getswitchinglength_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSwitchingLength(target);
}
double OPENMMGBSW_GBSWFORCE_GETSWITCHINGLENGTH(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getSwitchingLength(target);
}
void openmmgbsw_gbswforce_setswitchinglength_(OpenMMGBSW_GBSWForce*& target, double const& length) {
    OpenMMGBSW_GBSWForce_setSwitchingLength(target, length);
}
void OPENMMGBSW_GBSWFORCE_SETSWITCHINGLENGTH(OpenMMGBSW_GBSWForce*& target, double const& length) {
    OpenMMGBSW_GBSWForce_setSwitchingLength(target, length);
}
double openmmgbsw_gbswforce_getmembranethickness_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getMembraneThickness(target);
}
double OPENMMGBSW_GBSWFORCE_GETMEMBRANETHICKNESS(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getMembraneThickness(target);
}
void openmmgbsw_gbswforce_setmembranethickness_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setMembraneThickness(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETMEMBRANETHICKNESS(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setMembraneThickness(target, tmp);
}
double openmmgbsw_gbswforce_getmembraneswlen_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getMembraneSwLen(target);
}
double OPENMMGBSW_GBSWFORCE_GETMEMBRANESWLEN(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getMembraneSwLen(target);
}
void openmmgbsw_gbswforce_setmembraneswlen_(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setMembraneSwLen(target, tmp);
}
void OPENMMGBSW_GBSWFORCE_SETMEMBRANESWLEN(OpenMMGBSW_GBSWForce*& target, double const& tmp) {
    OpenMMGBSW_GBSWForce_setMembraneSwLen(target, tmp);
}
void openmmgbsw_gbswforce_getnonbondedmethod_(const OpenMMGBSW_GBSWForce*& target, OpenMMGBSW_GBSWForce_NonbondedMethod& result) {
    result = OpenMMGBSW_GBSWForce_getNonbondedMethod(target);
}
void OPENMMGBSW_GBSWFORCE_GETNONBONDEDMETHOD(const OpenMMGBSW_GBSWForce*& target, OpenMMGBSW_GBSWForce_NonbondedMethod& result) {
    result = OpenMMGBSW_GBSWForce_getNonbondedMethod(target);
}
void openmmgbsw_gbswforce_setnonbondedmethod_(OpenMMGBSW_GBSWForce*& target, OpenMMGBSW_GBSWForce_NonbondedMethod const& method) {
    OpenMMGBSW_GBSWForce_setNonbondedMethod(target, method);
}
void OPENMMGBSW_GBSWFORCE_SETNONBONDEDMETHOD(OpenMMGBSW_GBSWForce*& target, OpenMMGBSW_GBSWForce_NonbondedMethod const& method) {
    OpenMMGBSW_GBSWForce_setNonbondedMethod(target, method);
}
double openmmgbsw_gbswforce_getcutoffdistance_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getCutoffDistance(target);
}
double OPENMMGBSW_GBSWFORCE_GETCUTOFFDISTANCE(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getCutoffDistance(target);
}
void openmmgbsw_gbswforce_setcutoffdistance_(OpenMMGBSW_GBSWForce*& target, double const& distance) {
    OpenMMGBSW_GBSWForce_setCutoffDistance(target, distance);
}
void OPENMMGBSW_GBSWFORCE_SETCUTOFFDISTANCE(OpenMMGBSW_GBSWForce*& target, double const& distance) {
    OpenMMGBSW_GBSWForce_setCutoffDistance(target, distance);
}
double openmmgbsw_gbswforce_getreactionfielddielectric_(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getReactionFieldDielectric(target);
}
double OPENMMGBSW_GBSWFORCE_GETREACTIONFIELDDIELECTRIC(const OpenMMGBSW_GBSWForce*& target) {
    return OpenMMGBSW_GBSWForce_getReactionFieldDielectric(target);
}
void openmmgbsw_gbswforce_setreactionfielddielectric_(OpenMMGBSW_GBSWForce*& target, double const& distance) {
    OpenMMGBSW_GBSWForce_setReactionFieldDielectric(target, distance);
}
void OPENMMGBSW_GBSWFORCE_SETREACTIONFIELDDIELECTRIC(OpenMMGBSW_GBSWForce*& target, double const& distance) {
    OpenMMGBSW_GBSWForce_setReactionFieldDielectric(target, distance);
}
void openmmgbsw_gbswforce_updateparametersincontext_(OpenMMGBSW_GBSWForce*& target, OpenMM_Context* context) {
    OpenMMGBSW_GBSWForce_updateParametersInContext(target, context);
}
void OPENMMGBSW_GBSWFORCE_UPDATEPARAMETERSINCONTEXT(OpenMMGBSW_GBSWForce*& target, OpenMM_Context* context) {
    OpenMMGBSW_GBSWForce_updateParametersInContext(target, context);
}
void openmmgbsw_gbswforce_usesperiodicboundaryconditions_(const OpenMMGBSW_GBSWForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions(target);
}
void OPENMMGBSW_GBSWFORCE_USESPERIODICBOUNDARYCONDITIONS(const OpenMMGBSW_GBSWForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions(target);
}
void openmmgbsw_gbswforce_getlambdastate_(OpenMMGBSW_GBSWForce*& target, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBSW_GBSWForce_getLambdaState(target, context, lambdaState);
}
void OPENMMGBSW_GBSWFORCE_GETLAMBDASTATE(OpenMMGBSW_GBSWForce*& target, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBSW_GBSWForce_getLambdaState(target, context, lambdaState);
}
void openmmgbsw_gbswforce_setlambdastate_(OpenMMGBSW_GBSWForce*& tarset, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBSW_GBSWForce_setLambdaState(tarset, context, lambdaState);
}
void OPENMMGBSW_GBSWFORCE_SETLAMBDASTATE(OpenMMGBSW_GBSWForce*& tarset, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBSW_GBSWForce_setLambdaState(tarset, context, lambdaState);
}
}
