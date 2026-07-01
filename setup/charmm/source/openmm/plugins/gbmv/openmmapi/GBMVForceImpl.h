#ifndef OPENMM_GBMVFORCEFIELDIMPL_H_
#define OPENMM_GBMVFORCEFIELDIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "GBMVForce.h"

#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"

#include <string>

namespace OpenMMGBMV {

/**
 * This is the internal implementation of GBMVForce.
 */

class GBMVForceImpl : public OpenMM::ForceImpl {
public:
    GBMVForceImpl(const GBMVForce& owner);
    void initialize(OpenMM::ContextImpl& context);
    const GBMVForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getLambdaState(OpenMM::ContextImpl& context, std::vector<double>& LambdaState);
    void setLambdaState(OpenMM::ContextImpl& context, std::vector<double>& LambdaState);
private:
    const GBMVForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace OpenMMGBMV

#endif /*OPENMM_GBMVFORCEFIELDIMPL_H_*/
