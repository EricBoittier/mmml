#include "CudaCharmmKernelFactory.h"
#include "CudaCharmmKernels.h"
#include <openmm/cuda/CudaContext.h>
#include "openmm/Platform.h"
#include "openmm/PluginInitializer.h"
#include "openmm/internal/ContextImpl.h"

using namespace std;
using namespace OpenMM;

extern "C" void registerPlatforms() {
    // no new platform here
}

/*
 * Called on plugin load if the .so is in the plugins directory.
 */
extern "C" void registerKernelFactories() {
    Platform& platform = Platform::getPlatformByName("CUDA");
    KernelFactory* factory = new CudaCharmmKernelFactory();
    platform.registerKernelFactory(ApplyMonteCarloBarostatKernel2::Name(), factory);
}

KernelImpl* CudaCharmmKernelFactory::createKernelImpl(string name, const Platform& platform,
        ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    CudaContext& cu = *data.contexts[0];

    if (name == ApplyMonteCarloBarostatKernel2::Name())
        return new CudaApplyMonteCarloBarostatKernel2(name, platform, cu);
    throw OpenMMException((string("Unknown kernel name '") + name + "'").c_str());
}

