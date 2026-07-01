/* -------------------------------------------------------------------------- *
 *                              OpenMMGBSW                                    *
 * -------------------------------------------------------------------------- */

#include "OpenCLGBSWKernels.h"
#include "OpenCLGBSWKernelFactory.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <exception>

using namespace OpenMM;
using namespace OpenMMGBSW;

extern "C" void registerPlatforms() {
}

extern "C" void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLGBSWKernelFactory* factory = new OpenCLGBSWKernelFactory();
        platform.registerKernelFactory(CalcGBSWForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" void registerGBSWOpenCLKernelFactories() {
    try {
        Platform::getPlatformByName("OpenCL");
    }
    catch (...) {
        Platform::registerPlatform(new OpenCLPlatform());
    }
    registerKernelFactories();
}

KernelImpl* OpenCLGBSWKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    OpenCLContext& cl = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGBSWForceKernel::Name())
        return new OpenCLCalcGBSWForceKernel(name, platform, cl);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
