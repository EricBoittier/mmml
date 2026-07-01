#ifndef OPENMM_GBSWFORCE_PROXY_H_
#define OPENMM_GBSWFORCE_PROXY_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/serialization/SerializationProxy.h"

namespace OpenMMGBSW {

/**
 * This is a proxy for serializing GBSWForce objects.
 */

class GBSWForceProxy : public OpenMM::SerializationProxy {
public:
    GBSWForceProxy();
    void serialize(const void* object, OpenMM::SerializationNode& node) const;
    void* deserialize(const OpenMM::SerializationNode& node) const;
};

} // namespace OpenMMGBSW

#endif /*OPENMM_GBSWFORCE_PROXY_H_*/
