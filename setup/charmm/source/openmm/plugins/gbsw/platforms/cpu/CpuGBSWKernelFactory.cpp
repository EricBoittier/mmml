/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "CpuGBSWKernelFactory.h"
#include "CpuGBSWKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace OpenMMGBSW;

extern "C" void registerPlatforms() {
}

extern "C" void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CPU");
        CpuGBSWKernelFactory* factory = new CpuGBSWKernelFactory();
        platform.registerKernelFactory(CalcGBSWForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" void registerGBSWCpuKernelFactories() {
    try {
        Platform::getPlatformByName("CPU");
    }
    catch (...) {
        Platform::registerPlatform(new CpuPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CpuGBSWKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    if (name == CalcGBSWForceKernel::Name())
        return new CpuCalcGBSWForceKernel(name, platform, data);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}
