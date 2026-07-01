#include "OpenCLCharmmKernelFactory.h"
#include "OpenCLCharmmKernels.h"
#include <openmm/opencl/OpenCLContext.h>
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
    Platform& platform = Platform::getPlatformByName("OpenCL");
    KernelFactory* factory = new OpenCLCharmmKernelFactory();
    platform.registerKernelFactory(ApplyMonteCarloBarostatKernel2::Name(), factory);
}

KernelImpl* OpenCLCharmmKernelFactory::createKernelImpl(string name, const Platform& platform, ContextImpl& context) const {
    OpenCLPlatform::PlatformData& data = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
    OpenCLContext& cl = *data.contexts[0];

    if (name == ApplyMonteCarloBarostatKernel2::Name())
        return new OpenCLApplyMonteCarloBarostatKernel2(name, platform, cl);
    throw OpenMMException((string("Unknown kernel name '") + name + "'").c_str());
}

