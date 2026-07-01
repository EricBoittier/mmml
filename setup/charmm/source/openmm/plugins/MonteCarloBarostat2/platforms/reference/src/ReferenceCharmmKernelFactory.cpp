#include "ReferenceCharmmKernelFactory.h"
#include "ReferenceCharmmKernels.h"
#include "openmm/Platform.h"
#include "openmm/PluginInitializer.h"

using namespace std;
using namespace OpenMM;

/*
 * call on shared library load
 */
#if defined(WIN32)
    #include <windows.h>
    extern "C" void initCharmmReferenceKernels();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            initCharmmReferenceKernels();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) initCharmmReferenceKernels();
#endif

extern "C" void initCharmmReferenceKernels() {
    Platform& platform = Platform::getPlatformByName("Reference");
    KernelFactory* factory = new ReferenceCharmmKernelFactory();
    platform.registerKernelFactory(ApplyMonteCarloBarostatKernel2::Name(), factory);
}

KernelImpl* ReferenceCharmmKernelFactory::createKernelImpl(string name, const Platform& platform, ContextImpl& context) const {
    if (name == ApplyMonteCarloBarostatKernel2::Name())
        return new ReferenceApplyMonteCarloBarostatKernel2(name, platform);
    throw OpenMMException((string("Unknown kernel name '") + name + "'").c_str());
}

