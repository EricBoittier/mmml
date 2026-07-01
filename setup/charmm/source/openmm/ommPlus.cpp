#if KEY_OPENMM==1

#include <OpenMM.h>
#include <OpenMMCWrapper.h>

#include <iostream>

static std::string makeString(const char * fsrc, int length) {
    while (length && fsrc[length-1] == ' ')
        --length;

    return std::string(fsrc, length);
}

double ommGroupPotential(const OpenMM_Context * target, int pbox, int groups) {
    const OpenMM::Context * context =
        reinterpret_cast<const OpenMM::Context *>(target);

    OpenMM::State state = context->getState(OpenMM::State::Energy, pbox,
        groups);
    return state.getPotentialEnergy();
}

int ommGetPlatformByName(const char * name, OpenMM_Platform *& platform) {
  int success = 0;
  OpenMM::Platform * result = NULL;
  std::string platformName(name);

  for (int i = 0; i < OpenMM::Platform::getNumPlatforms(); i++) {
    if (OpenMM::Platform::getPlatform(i).getName() == platformName) {
      result = &OpenMM::Platform::getPlatform(i);
      success = 1;
    }
  }

  if (success == 1) {
    platform = reinterpret_cast<OpenMM_Platform *>(result);
  }

  return success;
}

extern "C" {

double omm_group_potential_(const OpenMM_Context *& context,
		int const& pbox, int const& groups) {
    return ommGroupPotential(context, pbox, groups);
}

double OMM_GROUP_POTENTIAL(const OpenMM_Context *& context,
		int const& pbox, int const& groups) {
    return ommGroupPotential(context, pbox, groups);
}

int omm_platform_getplatformbyname_(const char * name, OpenMM_Platform *& result, int name_length) {
  int success = ommGetPlatformByName(makeString(name, name_length).c_str(), result);
  return success;
}

int OMM_PLATFORM_GETPLATFORMBYNAME(const char * name, OpenMM_Platform *& result, int name_length) {
  int success = ommGetPlatformByName(makeString(name, name_length).c_str(), result);
  return success;
}


} // extern "C"

#endif
