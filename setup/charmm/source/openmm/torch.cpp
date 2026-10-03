#ifdef KEY_OMMTORCH
#include <TorchForce.h>
#include <iostream>

// The force pointer handed to these entry points comes from
// fstore_get(store, i) (see forcesStore.cpp), which returns NULL when the
// caller passes a force index that is out of range.  Dereferencing that
// NULL crashes CHARMM with a hard SIGSEGV instead of a diagnosable error.
// Guard every entry point that dereferences the force the way the parallel
// customForces path does (cf_add_global_param: "if (!f) return -1;"), so a
// bad index becomes a recoverable error rather than a segfault.

extern "C" {
  TorchPlugin::TorchForce * torch_create(const char * filename) {
    TorchPlugin::TorchForce * newForce = new TorchPlugin::TorchForce(filename);
    return newForce;
  }
  void torch_set_uses_pbc(TorchPlugin::TorchForce * force) {
    if (!force) {
      std::cerr << "torch_set_uses_pbc: no torch force at that index"
                << std::endl;
      return;
    }
    force->setUsesPeriodicBoundaryConditions(true);
  }
  void torch_set_outputs_forces(TorchPlugin::TorchForce * force) {
    if (!force) {
      std::cerr << "torch_set_outputs_forces: no torch force at that index"
                << std::endl;
      return;
    }
    force->setOutputsForces(true);
  }
  int torch_add_global_param(TorchPlugin::TorchForce * force,
                  const char * name, double value) {
    if (!force) {
      std::cerr << "torch_add_global_param: no torch force at that index"
                << std::endl;
      return -1;
    }
    return force->addGlobalParameter(name, value);
  }
  void torch_set_global_param(TorchPlugin::TorchForce * force,
			      int param_index, double value) {
    if (!force) {
      std::cerr << "torch_set_global_param: no torch force at that index"
                << std::endl;
      return;
    }
    return force->setGlobalParameterDefaultValue(param_index, value);
  }
}
#endif // KEY_OMMTORCH
