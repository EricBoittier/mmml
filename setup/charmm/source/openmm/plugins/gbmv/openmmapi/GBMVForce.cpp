/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "GBMVForce.h"
#include "GBMVForceImpl.h"

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace OpenMM;
using namespace OpenMMGBMV;

GBMVForce::GBMVForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), 
solventDielectric(80.0), soluteDielectric(1.0), surfaceAreaEnergy(0.0), swLen(0.03), AA0(-0.1801), AA1(1.81745),
    kappa(0.0), numAngles(50), numRadii(24), useCPHMD(false), lambdaOutFile("lambda.lamb"),
    tempTheta(298.0), timestepTheta(0.002), globalPH(7.0), massTheta(10.0), pHbeta(15.0), outputFrequency(1000),
    membraneThickness(0.0), membraneSwLen(0.0), rfDielectric(80.0) {
}

int GBMVForce::addNonbondedException( int atom1, int atom2 ) {
    nonbondedFixes.push_back(NonbondedExceptionFixes( atom1, atom2 ));
    return nonbondedFixes.size()-1;
}

void GBMVForce::getNonbondedException( int index, int& atom1, int& atom2 ) const {
    ASSERT_VALID_INDEX(index, nonbondedFixes);
    atom1 = nonbondedFixes[index].atom1;
    atom2 = nonbondedFixes[index].atom2;
}

int GBMVForce::addParticle(double charge, double radius) {
    particles.push_back(ParticleInfo(charge, radius));
    return particles.size()-1;
}

void GBMVForce::getParticleParameters(int index, double& charge, double& radius) const {
    ASSERT_VALID_INDEX(index, particles);
    charge = particles[index].charge;
    radius = particles[index].radius;
}

void GBMVForce::setParticleParameters(int index, double charge, double radius) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].charge = charge;
    particles[index].radius = radius;
}

int GBMVForce::addTitratingGroupParameters(
    double resPKA1, double resPKA2, double barrier1, double barrier2,
    double a0, double a1, double a2, double a3, double a4, double a5, 
    double a6, double a7, double a8) {
    titratingGroups.push_back(TitratingGroupInfo(
        resPKA1, resPKA2, barrier1, barrier2, 
        a0, a1, a2, a3, a4, a5, a6, a7, a8));
    return titratingGroups.size()-1;
}

void GBMVForce::getTitratingGroupParameters(int index, 
    double& resPKA1, double& resPKA2, double& barrier1, double& barrier2, 
    double& a0, double& a1, double& a2, double& a3, double& a4, double& a5, 
    double& a6, double& a7, double& a8) const {
    ASSERT_VALID_INDEX(index, titratingGroups);
    resPKA1  = titratingGroups[index].resPKA1;
    resPKA2  = titratingGroups[index].resPKA2;
    barrier1 = titratingGroups[index].barrier1;
    barrier2 = titratingGroups[index].barrier2;
    a0       = titratingGroups[index].a0;
    a1       = titratingGroups[index].a1;
    a2       = titratingGroups[index].a2;
    a3       = titratingGroups[index].a3;
    a4       = titratingGroups[index].a4;
    a5       = titratingGroups[index].a5;
    a6       = titratingGroups[index].a6;
    a7       = titratingGroups[index].a7;
    a8       = titratingGroups[index].a8;
}

void GBMVForce::setTitratingGroupParameters(int index, 
    double resPKA1, double resPKA2, double barrier1, double barrier2,
    double a0, double a1, double a2, double a3, double a4, double a5, 
    double a6, double a7, double a8) {
    titratingGroups[index].resPKA1  = resPKA1;
    titratingGroups[index].resPKA2  = resPKA2;
    titratingGroups[index].barrier1 = barrier1;
    titratingGroups[index].barrier2 = barrier2;
    titratingGroups[index].a0       = a0;
    titratingGroups[index].a1       = a1;
    titratingGroups[index].a2       = a2;
    titratingGroups[index].a3       = a3;
    titratingGroups[index].a4       = a4;
    titratingGroups[index].a5       = a5;
    titratingGroups[index].a6       = a6;
    titratingGroups[index].a7       = a7;
    titratingGroups[index].a8       = a8;
}

int GBMVForce::addCphmdParameters(int titrateResID, double refChargeState1, 
    double refChargeState2, double chargeState1, double chargeState2) {
    cphmdInfo.push_back(ParticleCphmdInfo(titrateResID, 
        refChargeState1, refChargeState2, chargeState1, chargeState2));
    return cphmdInfo.size()-1;
}

void GBMVForce::getCphmdParameters(int index, int& titrateResID, double& refChargeState1,
    double& refChargeState2, double& chargeState1, double& chargeState2) const {
    ASSERT_VALID_INDEX(index, cphmdInfo);
    titrateResID = cphmdInfo[index].titrateResID;
    refChargeState1 = cphmdInfo[index].refChargeState1;
    refChargeState2 = cphmdInfo[index].refChargeState2;
    chargeState1 = cphmdInfo[index].chargeState1;
    chargeState2 = cphmdInfo[index].chargeState2;
}

void GBMVForce::setCphmdParameters(int index, int titrateResID, double refChargeState1, 
    double refChargeState2, double chargeState1, double chargeState2) {
    ASSERT_VALID_INDEX(index, cphmdInfo);
    cphmdInfo[index].titrateResID = titrateResID;
    cphmdInfo[index].refChargeState1 = refChargeState1;
    cphmdInfo[index].refChargeState2 = refChargeState2;
    cphmdInfo[index].chargeState1 = chargeState1;
    cphmdInfo[index].chargeState2 = chargeState2;
}

GBMVForce::NonbondedMethod GBMVForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void GBMVForce::setNonbondedMethod(NonbondedMethod method) {
    nonbondedMethod = method;
}

double GBMVForce::getCutoffDistance() const {
    return cutoffDistance;
}

void GBMVForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

double GBMVForce::getCutonDistance() const {
    return cutonDistance;
}

void GBMVForce::setCutonDistance(double distance) {
    cutonDistance = distance;
}

double GBMVForce::getReactionFieldDielectric() const {
    return rfDielectric;
}

void GBMVForce::setReactionFieldDielectric(double tmp) {
    rfDielectric = tmp;
}

ForceImpl* GBMVForce::createImpl() const {
    return new GBMVForceImpl(*this);
}

void GBMVForce::updateParametersInContext(Context& context) {
    dynamic_cast<GBMVForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void GBMVForce::getLambdaState(Context& context, std::vector<double>& LambdaState) {
    dynamic_cast<GBMVForceImpl&>(getImplInContext(context)).getLambdaState(getContextImpl(context), LambdaState);
}

void GBMVForce::setLambdaState(Context& context, std::vector<double>& LambdaState) {
    dynamic_cast<GBMVForceImpl&>(getImplInContext(context)).setLambdaState(getContextImpl(context), LambdaState);
}
