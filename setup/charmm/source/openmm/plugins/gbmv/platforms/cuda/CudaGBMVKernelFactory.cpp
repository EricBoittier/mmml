/* -------------------------------------------------------------------------- *
 *                              OpenMMGBMV                                    *
 * -------------------------------------------------------------------------- */

#include "CudaGBMVKernelFactory.h"
#include "CudaGBMVKernels.h"

#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/internal/ContextImpl.h"

#include <exception>
#include <cmath>

using namespace OpenMM;
using namespace OpenMMGBMV;

extern "C" void registerPlatforms() {
}

extern "C" void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaGBMVKernelFactory* factory = new CudaGBMVKernelFactory();
        platform.registerKernelFactory(CalcGBMVForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" void registerGBMVCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaGBMVKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGBMVForceKernel::Name())
        return new CudaCalcGBMVForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
