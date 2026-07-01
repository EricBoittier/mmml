#ifndef OPENMM_GBSWFORCEFIELDIMPL_H_
#define OPENMM_GBSWFORCEFIELDIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "GBSWForce.h"

#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"

#include <string>

namespace OpenMMGBSW {

/**
 * This is the internal implementation of GBSWForce.
 */

class GBSWForceImpl : public OpenMM::ForceImpl {
public:
    GBSWForceImpl(const GBSWForce& owner);
    void initialize(OpenMM::ContextImpl& context);
    const GBSWForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }

  void setDoingDynamics(bool val);

  double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getLambdaState(OpenMM::ContextImpl& context, std::vector<double>& LambdaState);
    void setLambdaState(OpenMM::ContextImpl& context, std::vector<double>& LambdaState);
private:
    const GBSWForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_GBSWFORCEFIELDIMPL_H_*/
