#ifndef OPENMM_MSESFORCEIMPL_H_
#define OPENMM_MSESFORCEIMPL_H_

#include "MSESForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace OpenMMMSES {

class System;

/**
 * This is the internal implementation of MSESForce.
 */

class MSESForceImpl : public OpenMM::ForceImpl 
{
public:
    MSESForceImpl(const MSESForce& owner);
    ~MSESForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const MSESForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context,  bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);

private:
    const MSESForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace

#endif /*OPENMM_MSESFORCEIMPL_H_*/
