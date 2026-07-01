/* -------------------------------------------------------------------------- *
 *                              OpenMMGBSW                                    *
 * -------------------------------------------------------------------------- */

#include "CudaGBSWKernelFactory.h"
#include "CudaGBSWKernels.h"

#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/internal/ContextImpl.h"

#include <exception>
#include <cmath>

using namespace OpenMM;
using namespace OpenMMGBSW;

extern "C" void registerPlatforms() {
}

extern "C" void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaGBSWKernelFactory* factory = new CudaGBSWKernelFactory();
        platform.registerKernelFactory(CalcGBSWForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" void registerGBSWCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaGBSWKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGBSWForceKernel::Name())
        return new CudaCalcGBSWForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
