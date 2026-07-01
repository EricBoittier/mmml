/* -------------------------------------------------------------------------- *
 *                              OpenMMGBSW                                    *
 * -------------------------------------------------------------------------- */

#include "ReferenceGBSWKernelFactory.h"
#include "ReferenceGBSWKernels.h"

#include "openmm/OpenMMException.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;
using namespace OpenMMGBSW;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceGBSWKernelFactory* factory = new ReferenceGBSWKernelFactory();
            platform.registerKernelFactory(CalcGBSWForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerGBSWReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceGBSWKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcGBSWForceKernel::Name())
        return new ReferenceCalcGBSWForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
