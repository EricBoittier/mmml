#include "OpenMM.h"
#include "OpenMMCWrapper.h"
#include "OpenMMGBMV.h"
#include "OpenMMGBMVCWrapper.h"
#include <cstring>
#include <vector>
#include <cstdlib>

using namespace OpenMM;
using namespace OpenMMGBMV;
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
void openmmgbmv_gbmvforce_create_(OpenMMGBMV_GBMVForce*& result) {
    result = OpenMMGBMV_GBMVForce_create();
}
void OPENMMGBMV_GBMVFORCE_CREATE(OpenMMGBMV_GBMVForce*& result) {
    result = OpenMMGBMV_GBMVForce_create();
}
void openmmgbmv_gbmvforce_destroy_(OpenMMGBMV_GBMVForce*& destroy) {
    OpenMMGBMV_GBMVForce_destroy(destroy);
    destroy = 0;
}
void OPENMMGBMV_GBMVFORCE_DESTROY(OpenMMGBMV_GBMVForce*& destroy) {
    OpenMMGBMV_GBMVForce_destroy(destroy);
    destroy = 0;
}
void openmmgbmv_gbmvforce_addcphmdforce_(OpenMMGBMV_GBMVForce*& target, double const& pH, double const& T_theta, double const& mTheta, double const& ts_theta, double const& beta, int const& outFreq, char* fileName) {
    OpenMMGBMV_GBMVForce_addCPHMDForce(target, pH, T_theta, mTheta, ts_theta, beta, outFreq, fileName);
}
void OPENMMGBMV_GBMVFORCE_ADDCPHMDFORCE(OpenMMGBMV_GBMVForce*& target, double const& pH, double const& T_theta, double const& mTheta, double const& ts_theta, double const& beta, int const& outFreq, char* fileName) {
    OpenMMGBMV_GBMVForce_addCPHMDForce(target, pH, T_theta, mTheta, ts_theta, beta, outFreq, fileName);
}
void openmmgbmv_gbmvforce_usingcphmd_(const OpenMMGBMV_GBMVForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBMV_GBMVForce_usingCPHMD(target);
}
void OPENMMGBMV_GBMVFORCE_USINGCPHMD(const OpenMMGBMV_GBMVForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBMV_GBMVForce_usingCPHMD(target);
}
int openmmgbmv_gbmvforce_getnumtitratinggroups_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumTitratingGroups(target);
}
int OPENMMGBMV_GBMVFORCE_GETNUMTITRATINGGROUPS(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumTitratingGroups(target);
}
int openmmgbmv_gbmvforce_addtitratinggroupparameters_(OpenMMGBMV_GBMVForce*& target, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    return OpenMMGBMV_GBMVForce_addTitratingGroupParameters(target, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int OPENMMGBMV_GBMVFORCE_ADDTITRATINGGROUPPARAMETERS(OpenMMGBMV_GBMVForce*& target, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    return OpenMMGBMV_GBMVForce_addTitratingGroupParameters(target, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void openmmgbmv_gbmvforce_gettitratinggroupparameters_(const OpenMMGBMV_GBMVForce*& target, int const& index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    OpenMMGBMV_GBMVForce_getTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void OPENMMGBMV_GBMVFORCE_GETTITRATINGGROUPPARAMETERS(const OpenMMGBMV_GBMVForce*& target, int const& index, double* resPKA1, double* resPKA2, double* barrier1, double* barrier2, double* a0, double* a1, double* a2, double* a3, double* a4, double* a5, double* a6, double* a7, double* a8) {
    OpenMMGBMV_GBMVForce_getTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void openmmgbmv_gbmvforce_settitratinggroupparameters_(OpenMMGBMV_GBMVForce*& target, int const& index, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    OpenMMGBMV_GBMVForce_setTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
void OPENMMGBMV_GBMVFORCE_SETTITRATINGGROUPPARAMETERS(OpenMMGBMV_GBMVForce*& target, int const& index, double const& resPKA1, double const& resPKA2, double const& barrier1, double const& barrier2, double const& a0, double const& a1, double const& a2, double const& a3, double const& a4, double const& a5, double const& a6, double const& a7, double const& a8) {
    OpenMMGBMV_GBMVForce_setTitratingGroupParameters(target, index, resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8);
}
int openmmgbmv_gbmvforce_addnonbondedexception_(OpenMMGBMV_GBMVForce*& target, int const& atom1, int const& atom2) {
    return OpenMMGBMV_GBMVForce_addNonbondedException(target, atom1, atom2);
}
int OPENMMGBMV_GBMVFORCE_ADDNONBONDEDEXCEPTION(OpenMMGBMV_GBMVForce*& target, int const& atom1, int const& atom2) {
    return OpenMMGBMV_GBMVForce_addNonbondedException(target, atom1, atom2);
}
void openmmgbmv_gbmvforce_getnonbondedexception_(const OpenMMGBMV_GBMVForce*& target, int const& index, int* atom1, int* atom2) {
    OpenMMGBMV_GBMVForce_getNonbondedException(target, index, atom1, atom2);
}
void OPENMMGBMV_GBMVFORCE_GETNONBONDEDEXCEPTION(const OpenMMGBMV_GBMVForce*& target, int const& index, int* atom1, int* atom2) {
    OpenMMGBMV_GBMVForce_getNonbondedException(target, index, atom1, atom2);
}
int openmmgbmv_gbmvforce_getnumnonbondedexceptions_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumNonbondedExceptions(target);
}
int OPENMMGBMV_GBMVFORCE_GETNUMNONBONDEDEXCEPTIONS(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumNonbondedExceptions(target);
}
int openmmgbmv_gbmvforce_getnumparticles_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumParticles(target);
}
int OPENMMGBMV_GBMVFORCE_GETNUMPARTICLES(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumParticles(target);
}
int openmmgbmv_gbmvforce_addparticle_(OpenMMGBMV_GBMVForce*& target, double const& charge, double const& radius) {
    return OpenMMGBMV_GBMVForce_addParticle(target, charge, radius);
}
int OPENMMGBMV_GBMVFORCE_ADDPARTICLE(OpenMMGBMV_GBMVForce*& target, double const& charge, double const& radius) {
    return OpenMMGBMV_GBMVForce_addParticle(target, charge, radius);
}
void openmmgbmv_gbmvforce_getparticleparameters_(const OpenMMGBMV_GBMVForce*& target, int const& index, double* charge, double* radius) {
    OpenMMGBMV_GBMVForce_getParticleParameters(target, index, charge, radius);
}
void OPENMMGBMV_GBMVFORCE_GETPARTICLEPARAMETERS(const OpenMMGBMV_GBMVForce*& target, int const& index, double* charge, double* radius) {
    OpenMMGBMV_GBMVForce_getParticleParameters(target, index, charge, radius);
}
void openmmgbmv_gbmvforce_setparticleparameters_(OpenMMGBMV_GBMVForce*& target, int const& index, double const& charge, double const& radius) {
    OpenMMGBMV_GBMVForce_setParticleParameters(target, index, charge, radius);
}
void OPENMMGBMV_GBMVFORCE_SETPARTICLEPARAMETERS(OpenMMGBMV_GBMVForce*& target, int const& index, double const& charge, double const& radius) {
    OpenMMGBMV_GBMVForce_setParticleParameters(target, index, charge, radius);
}
int openmmgbmv_gbmvforce_addcphmdparameters_(OpenMMGBMV_GBMVForce*& target, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    return OpenMMGBMV_GBMVForce_addCphmdParameters(target, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
int OPENMMGBMV_GBMVFORCE_ADDCPHMDPARAMETERS(OpenMMGBMV_GBMVForce*& target, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    return OpenMMGBMV_GBMVForce_addCphmdParameters(target, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbmv_gbmvforce_getcphmdparameters_(const OpenMMGBMV_GBMVForce*& target, int const& index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    OpenMMGBMV_GBMVForce_getCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void OPENMMGBMV_GBMVFORCE_GETCPHMDPARAMETERS(const OpenMMGBMV_GBMVForce*& target, int const& index, int* titrateResID, double* refChargeState1, double* refChargeState2, double* chargeState1, double* chargeState2) {
    OpenMMGBMV_GBMVForce_getCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbmv_gbmvforce_setcphmdparameters_(OpenMMGBMV_GBMVForce*& target, int const& index, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    OpenMMGBMV_GBMVForce_setCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void OPENMMGBMV_GBMVFORCE_SETCPHMDPARAMETERS(OpenMMGBMV_GBMVForce*& target, int const& index, int const& titrateResID, double const& refChargeState1, double const& refChargeState2, double const& chargeState1, double const& chargeState2) {
    OpenMMGBMV_GBMVForce_setCphmdParameters(target, index, titrateResID, refChargeState1, refChargeState2, chargeState1, chargeState2);
}
void openmmgbmv_gbmvforce_getlambdaoutputfile_(const OpenMMGBMV_GBMVForce*& target, char*& result) {
    result = OpenMMGBMV_GBMVForce_getLambdaOutputFile(target);
}
void OPENMMGBMV_GBMVFORCE_GETLAMBDAOUTPUTFILE(const OpenMMGBMV_GBMVForce*& target, char*& result) {
    result = OpenMMGBMV_GBMVForce_getLambdaOutputFile(target);
}
void openmmgbmv_gbmvforce_setlambdaoutputfile_(OpenMMGBMV_GBMVForce*& target, char* tmp) {
    OpenMMGBMV_GBMVForce_setLambdaOutputFile(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETLAMBDAOUTPUTFILE(OpenMMGBMV_GBMVForce*& target, char* tmp) {
    OpenMMGBMV_GBMVForce_setLambdaOutputFile(target, tmp);
}
double openmmgbmv_gbmvforce_getsystemph_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSystemPH(target);
}
double OPENMMGBMV_GBMVFORCE_GETSYSTEMPH(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSystemPH(target);
}
void openmmgbmv_gbmvforce_setsystemph_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setSystemPH(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETSYSTEMPH(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setSystemPH(target, tmp);
}
double openmmgbmv_gbmvforce_getthetatemp_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getThetaTemp(target);
}
double OPENMMGBMV_GBMVFORCE_GETTHETATEMP(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getThetaTemp(target);
}
void openmmgbmv_gbmvforce_setthetatemp_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setThetaTemp(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETTHETATEMP(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setThetaTemp(target, tmp);
}
double openmmgbmv_gbmvforce_getthetamass_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getThetaMass(target);
}
double OPENMMGBMV_GBMVFORCE_GETTHETAMASS(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getThetaMass(target);
}
void openmmgbmv_gbmvforce_setthetamass_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setThetaMass(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETTHETAMASS(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setThetaMass(target, tmp);
}
int openmmgbmv_gbmvforce_getlambdaoutputfrequency_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getLambdaOutputFrequency(target);
}
int OPENMMGBMV_GBMVFORCE_GETLAMBDAOUTPUTFREQUENCY(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getLambdaOutputFrequency(target);
}
void openmmgbmv_gbmvforce_setlambdaoutputfrequency_(OpenMMGBMV_GBMVForce*& target, int const& tmp) {
    OpenMMGBMV_GBMVForce_setLambdaOutputFrequency(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETLAMBDAOUTPUTFREQUENCY(OpenMMGBMV_GBMVForce*& target, int const& tmp) {
    OpenMMGBMV_GBMVForce_setLambdaOutputFrequency(target, tmp);
}
double openmmgbmv_gbmvforce_getphbeta_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getPHbeta(target);
}
double OPENMMGBMV_GBMVFORCE_GETPHBETA(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getPHbeta(target);
}
void openmmgbmv_gbmvforce_setphbeta_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setPHbeta(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETPHBETA(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setPHbeta(target, tmp);
}
double openmmgbmv_gbmvforce_getsolventdielectric_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSolventDielectric(target);
}
double OPENMMGBMV_GBMVFORCE_GETSOLVENTDIELECTRIC(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSolventDielectric(target);
}
void openmmgbmv_gbmvforce_setsolventdielectric_(OpenMMGBMV_GBMVForce*& target, double const& dielectric) {
    OpenMMGBMV_GBMVForce_setSolventDielectric(target, dielectric);
}
void OPENMMGBMV_GBMVFORCE_SETSOLVENTDIELECTRIC(OpenMMGBMV_GBMVForce*& target, double const& dielectric) {
    OpenMMGBMV_GBMVForce_setSolventDielectric(target, dielectric);
}
double openmmgbmv_gbmvforce_getsolutedielectric_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSoluteDielectric(target);
}
double OPENMMGBMV_GBMVFORCE_GETSOLUTEDIELECTRIC(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSoluteDielectric(target);
}
void openmmgbmv_gbmvforce_setsolutedielectric_(OpenMMGBMV_GBMVForce*& target, double const& dielectric) {
    OpenMMGBMV_GBMVForce_setSoluteDielectric(target, dielectric);
}
void OPENMMGBMV_GBMVFORCE_SETSOLUTEDIELECTRIC(OpenMMGBMV_GBMVForce*& target, double const& dielectric) {
    OpenMMGBMV_GBMVForce_setSoluteDielectric(target, dielectric);
}
double openmmgbmv_gbmvforce_getsurfaceareaenergy_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy(target);
}
double OPENMMGBMV_GBMVFORCE_GETSURFACEAREAENERGY(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy(target);
}
void openmmgbmv_gbmvforce_setsurfaceareaenergy_(OpenMMGBMV_GBMVForce*& target, double const& energy) {
    OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(target, energy);
}
void OPENMMGBMV_GBMVFORCE_SETSURFACEAREAENERGY(OpenMMGBMV_GBMVForce*& target, double const& energy) {
    OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(target, energy);
}
double openmmgbmv_gbmvforce_getaa0_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getAA0(target);
}
double OPENMMGBMV_GBMVFORCE_GETAA0(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getAA0(target);
}
void openmmgbmv_gbmvforce_setaa0_(OpenMMGBMV_GBMVForce*& target, double const& value) {
    OpenMMGBMV_GBMVForce_setAA0(target, value);
}
void OPENMMGBMV_GBMVFORCE_SETAA0(OpenMMGBMV_GBMVForce*& target, double const& value) {
    OpenMMGBMV_GBMVForce_setAA0(target, value);
}
double openmmgbmv_gbmvforce_getaa1_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getAA1(target);
}
double OPENMMGBMV_GBMVFORCE_GETAA1(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getAA1(target);
}
void openmmgbmv_gbmvforce_setaa1_(OpenMMGBMV_GBMVForce*& target, double const& value) {
    OpenMMGBMV_GBMVForce_setAA1(target, value);
}
void OPENMMGBMV_GBMVFORCE_SETAA1(OpenMMGBMV_GBMVForce*& target, double const& value) {
    OpenMMGBMV_GBMVForce_setAA1(target, value);
}
int openmmgbmv_gbmvforce_getnumgaulegrad_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumGauLegRad(target);
}
int OPENMMGBMV_GBMVFORCE_GETNUMGAULEGRAD(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumGauLegRad(target);
}
void openmmgbmv_gbmvforce_setnumgaulegrad_(OpenMMGBMV_GBMVForce*& target, int const& number) {
    OpenMMGBMV_GBMVForce_setNumGauLegRad(target, number);
}
void OPENMMGBMV_GBMVFORCE_SETNUMGAULEGRAD(OpenMMGBMV_GBMVForce*& target, int const& number) {
    OpenMMGBMV_GBMVForce_setNumGauLegRad(target, number);
}
int openmmgbmv_gbmvforce_getnumlebang_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumLebAng(target);
}
int OPENMMGBMV_GBMVFORCE_GETNUMLEBANG(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getNumLebAng(target);
}
void openmmgbmv_gbmvforce_setnumlebang_(OpenMMGBMV_GBMVForce*& target, int const& number) {
    OpenMMGBMV_GBMVForce_setNumLebAng(target, number);
}
void OPENMMGBMV_GBMVFORCE_SETNUMLEBANG(OpenMMGBMV_GBMVForce*& target, int const& number) {
    OpenMMGBMV_GBMVForce_setNumLebAng(target, number);
}
double openmmgbmv_gbmvforce_getdebyehuckellength_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getDebyeHuckelLength(target);
}
double OPENMMGBMV_GBMVFORCE_GETDEBYEHUCKELLENGTH(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getDebyeHuckelLength(target);
}
void openmmgbmv_gbmvforce_setdebyehuckellength_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setDebyeHuckelLength(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETDEBYEHUCKELLENGTH(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setDebyeHuckelLength(target, length);
}
double openmmgbmv_gbmvforce_getswitchinglength_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSwitchingLength(target);
}
double OPENMMGBMV_GBMVFORCE_GETSWITCHINGLENGTH(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSwitchingLength(target);
}
void openmmgbmv_gbmvforce_setswitchinglength_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSwitchingLength(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETSWITCHINGLENGTH(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSwitchingLength(target, length);
}
// GBMV2: Parameters of GBMV2 model
double openmmgbmv_gbmvforce_getbeta_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getBETA_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETBETA_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getBETA_GBMV2(target);
}
void openmmgbmv_gbmvforce_setbeta_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setBETA_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETBETA_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setBETA_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getlambda_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETLAMBDA_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2(target);
}
void openmmgbmv_gbmvforce_setlambda_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETLAMBDA_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getalpha_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getALPHA_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETALPHA_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getALPHA_GBMV2(target);
}
void openmmgbmv_gbmvforce_setalpha_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setALPHA_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETALPHA_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setALPHA_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getslope_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSLOPE_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETSLOPE_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSLOPE_GBMV2(target);
}
void openmmgbmv_gbmvforce_setslope_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETSLOPE_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getshift_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSHIFT_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETSHIFT_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getSHIFT_GBMV2(target);
}
void openmmgbmv_gbmvforce_setshift_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETSHIFT_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_gethsx1_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getHSX1_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETHSX1_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getHSX1_GBMV2(target);
}
void openmmgbmv_gbmvforce_sethsx1_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setHSX1_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETHSX1_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setHSX1_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_gethsx2_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getHSX2_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETHSX2_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getHSX2_GBMV2(target);
}
void openmmgbmv_gbmvforce_sethsx2_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setHSX2_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETHSX2_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setHSX2_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getonx_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getONX_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETONX_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getONX_GBMV2(target);
}
void openmmgbmv_gbmvforce_setonx_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setONX_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETONX_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setONX_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getoffx_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getOFFX_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETOFFX_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getOFFX_GBMV2(target);
}
void openmmgbmv_gbmvforce_setoffx_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setOFFX_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETOFFX_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setOFFX_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getp1_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP1_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETP1_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP1_GBMV2(target);
}
void openmmgbmv_gbmvforce_setp1_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP1_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETP1_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP1_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getp2_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP2_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETP2_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP2_GBMV2(target);
}
void openmmgbmv_gbmvforce_setp2_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP2_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETP2_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP2_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getp3_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP3_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETP3_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP3_GBMV2(target);
}
void openmmgbmv_gbmvforce_setp3_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP3_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETP3_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP3_GBMV2(target, length);
}
double openmmgbmv_gbmvforce_getp6_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP6_GBMV2(target);
}
double OPENMMGBMV_GBMVFORCE_GETP6_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getP6_GBMV2(target);
}
void openmmgbmv_gbmvforce_setp6_gbmv2_(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP6_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETP6_GBMV2(OpenMMGBMV_GBMVForce*& target, double const& length) {
    OpenMMGBMV_GBMVForce_setP6_GBMV2(target, length);
}
int openmmgbmv_gbmvforce_getcutnum_gbmv2_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2(target);
}
int OPENMMGBMV_GBMVFORCE_GETCUTNUM_GBMV2(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2(target);
}
void openmmgbmv_gbmvforce_setcutnum_gbmv2_(OpenMMGBMV_GBMVForce*& target, int const& length) {
    OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(target, length);
}
void OPENMMGBMV_GBMVFORCE_SETCUTNUM_GBMV2(OpenMMGBMV_GBMVForce*& target, int const& length) {
    OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(target, length);
}
// END GBMV2
double openmmgbmv_gbmvforce_getmembranethickness_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getMembraneThickness(target);
}
double OPENMMGBMV_GBMVFORCE_GETMEMBRANETHICKNESS(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getMembraneThickness(target);
}
void openmmgbmv_gbmvforce_setmembranethickness_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setMembraneThickness(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETMEMBRANETHICKNESS(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setMembraneThickness(target, tmp);
}
double openmmgbmv_gbmvforce_getmembraneswlen_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getMembraneSwLen(target);
}
double OPENMMGBMV_GBMVFORCE_GETMEMBRANESWLEN(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getMembraneSwLen(target);
}
void openmmgbmv_gbmvforce_setmembraneswlen_(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setMembraneSwLen(target, tmp);
}
void OPENMMGBMV_GBMVFORCE_SETMEMBRANESWLEN(OpenMMGBMV_GBMVForce*& target, double const& tmp) {
    OpenMMGBMV_GBMVForce_setMembraneSwLen(target, tmp);
}
void openmmgbmv_gbmvforce_getnonbondedmethod_(const OpenMMGBMV_GBMVForce*& target, OpenMMGBMV_GBMVForce_NonbondedMethod& result) {
    result = OpenMMGBMV_GBMVForce_getNonbondedMethod(target);
}
void OPENMMGBMV_GBMVFORCE_GETNONBONDEDMETHOD(const OpenMMGBMV_GBMVForce*& target, OpenMMGBMV_GBMVForce_NonbondedMethod& result) {
    result = OpenMMGBMV_GBMVForce_getNonbondedMethod(target);
}
void openmmgbmv_gbmvforce_setnonbondedmethod_(OpenMMGBMV_GBMVForce*& target, OpenMMGBMV_GBMVForce_NonbondedMethod const& method) {
    OpenMMGBMV_GBMVForce_setNonbondedMethod(target, method);
}
void OPENMMGBMV_GBMVFORCE_SETNONBONDEDMETHOD(OpenMMGBMV_GBMVForce*& target, OpenMMGBMV_GBMVForce_NonbondedMethod const& method) {
    OpenMMGBMV_GBMVForce_setNonbondedMethod(target, method);
}
double openmmgbmv_gbmvforce_getcutoffdistance_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCutoffDistance(target);
}
double OPENMMGBMV_GBMVFORCE_GETCUTOFFDISTANCE(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCutoffDistance(target);
}
void openmmgbmv_gbmvforce_setcutoffdistance_(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setCutoffDistance(target, distance);
}
void OPENMMGBMV_GBMVFORCE_SETCUTOFFDISTANCE(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setCutoffDistance(target, distance);
}
double openmmgbmv_gbmvforce_getcutondistance_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCutonDistance(target);
}
double OPENMMGBMV_GBMVFORCE_GETCUTONDISTANCE(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getCutonDistance(target);
}
void openmmgbmv_gbmvforce_setcutondistance_(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setCutonDistance(target, distance);
}
void OPENMMGBMV_GBMVFORCE_SETCUTONDISTANCE(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setCutonDistance(target, distance);
}
double openmmgbmv_gbmvforce_getreactionfielddielectric_(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getReactionFieldDielectric(target);
}
double OPENMMGBMV_GBMVFORCE_GETREACTIONFIELDDIELECTRIC(const OpenMMGBMV_GBMVForce*& target) {
    return OpenMMGBMV_GBMVForce_getReactionFieldDielectric(target);
}
void openmmgbmv_gbmvforce_setreactionfielddielectric_(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setReactionFieldDielectric(target, distance);
}
void OPENMMGBMV_GBMVFORCE_SETREACTIONFIELDDIELECTRIC(OpenMMGBMV_GBMVForce*& target, double const& distance) {
    OpenMMGBMV_GBMVForce_setReactionFieldDielectric(target, distance);
}
void openmmgbmv_gbmvforce_updateparametersincontext_(OpenMMGBMV_GBMVForce*& target, OpenMM_Context* context) {
    OpenMMGBMV_GBMVForce_updateParametersInContext(target, context);
}
void OPENMMGBMV_GBMVFORCE_UPDATEPARAMETERSINCONTEXT(OpenMMGBMV_GBMVForce*& target, OpenMM_Context* context) {
    OpenMMGBMV_GBMVForce_updateParametersInContext(target, context);
}
void openmmgbmv_gbmvforce_usesperiodicboundaryconditions_(const OpenMMGBMV_GBMVForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions(target);
}
void OPENMMGBMV_GBMVFORCE_USESPERIODICBOUNDARYCONDITIONS(const OpenMMGBMV_GBMVForce*& target, OpenMM_Boolean& result) {
    result = OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions(target);
}
void openmmgbmv_gbmvforce_getlambdastate_(OpenMMGBMV_GBMVForce*& target, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBMV_GBMVForce_getLambdaState(target, context, lambdaState);
}
void OPENMMGBMV_GBMVFORCE_GETLAMBDASTATE(OpenMMGBMV_GBMVForce*& target, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBMV_GBMVForce_getLambdaState(target, context, lambdaState);
}
void openmmgbmv_gbmvforce_setlambdastate_(OpenMMGBMV_GBMVForce*& tarset, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBMV_GBMVForce_setLambdaState(tarset, context, lambdaState);
}
void OPENMMGBMV_GBMVFORCE_SETLAMBDASTATE(OpenMMGBMV_GBMVForce*& tarset, OpenMM_Context*& context, OpenMM_DoubleArray*& lambdaState) {
    OpenMMGBMV_GBMVForce_setLambdaState(tarset, context, lambdaState);
}
}
