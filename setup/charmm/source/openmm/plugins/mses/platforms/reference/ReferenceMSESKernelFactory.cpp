
#include "ReferenceMSESKernelFactory.h"
#include "ReferenceMSESKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMMMSES;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceMSESKernelFactory* factory = new ReferenceMSESKernelFactory();
            platform.registerKernelFactory(CalcMSESForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerMSESReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceMSESKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcMSESForceKernel::Name())
        return new ReferenceCalcMSESForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
