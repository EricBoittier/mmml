#ifdef KEY_OPENMM
#include "forcesStore.h"
#include <stdexcept>
#include <iostream>
#include <typeinfo>

// C++ class methods
OpenMM::Force * ForcesStore::get(int i) {
  OpenMM::Force * outForce;
  try {
    outForce = this->forces.at(i);
  } catch(const std::out_of_range& ex) {
    outForce = NULL;
  }
  return outForce;
}

ForcesStore::ForceType ForcesStore::getType(int i) {
  return this->types.at(i);
}

OpenMM::Force * ForcesStore::copy(int i) {
  OpenMM::Force * outForce = NULL;
  try {
    OpenMM::Force * ogForce = this->get(i);
    switch(this->getType(i)) {
#ifdef KEY_OMMTORCH
    case TORCH_TYPE:
      outForce = new TorchPlugin::TorchForce(
          *static_cast<TorchPlugin::TorchForce *>(ogForce));
      break;
#endif  /* KEY_OMMTORCH */
    case C_ANGLE_TYPE:
      outForce = new OpenMM::CustomAngleForce(
          *static_cast<OpenMM::CustomAngleForce *>(ogForce));
      break;
    case C_BOND_TYPE:
      outForce = new OpenMM::CustomBondForce(
          *static_cast<OpenMM::CustomBondForce *>(ogForce));
      break;
    case C_CV_TYPE: {
      // Deep-copy CustomCVForce.  The default copy constructor only
      // shallow-copies the owned collective-variable Force pointers,
      // which leads to use-after-free when the copy is destroyed
      // (e.g. during an OpenMM context teardown/rebuild).
      auto * og = static_cast<OpenMM::CustomCVForce *>(ogForce);
      auto * cvf = new OpenMM::CustomCVForce(og->getEnergyFunction());
      // Copy global parameters
      for (int p = 0; p < og->getNumGlobalParameters(); ++p)
        cvf->addGlobalParameter(og->getGlobalParameterName(p),
                                og->getGlobalParameterDefaultValue(p));
      // Copy energy parameter derivatives
      for (int p = 0; p < og->getNumEnergyParameterDerivatives(); ++p)
        cvf->addEnergyParameterDerivative(
            og->getEnergyParameterDerivativeName(p));
      // Deep-copy each collective variable force
      for (int v = 0; v < og->getNumCollectiveVariables(); ++v) {
        std::string name = og->getCollectiveVariableName(v);
        const OpenMM::Force & cvOrig = og->getCollectiveVariable(v);
        // getCollectiveVariable returns a reference to a force owned by the
        // original CustomCVForce, so it has to be duplicated.  OpenMM has no
        // generic Force::clone(), so the concrete type is recovered with
        // dynamic_cast and copied with that type's copy constructor.  Keep
        // this list in step with the outer switch; an unrecognised type is
        // handled below rather than dropped.
        //
        // Two known gaps, both pre-existing: a TorchForce or a nested
        // CustomCVForce used as a collective variable is not handled (and now
        // fails closed rather than being silently dropped), and for the types
        // that own TabulatedFunctions these copy constructors copy the
        // function pointer rather than the function -- see the ownership
        // invariant in forcesStore.h for why that is currently safe.
        OpenMM::Force * cvCopy = nullptr;
        if (auto * bf = dynamic_cast<const OpenMM::CustomBondForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomBondForce(*bf);
        else if (auto * af = dynamic_cast<const OpenMM::CustomAngleForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomAngleForce(*af);
        else if (auto * tf = dynamic_cast<const OpenMM::CustomTorsionForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomTorsionForce(*tf);
        else if (auto * ef = dynamic_cast<const OpenMM::CustomExternalForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomExternalForce(*ef);
        else if (auto * cbf = dynamic_cast<const OpenMM::CustomCompoundBondForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomCompoundBondForce(*cbf);
        else if (auto * nbf = dynamic_cast<const OpenMM::CustomNonbondedForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomNonbondedForce(*nbf);
        else if (auto * cbf2 = dynamic_cast<const OpenMM::CustomCentroidBondForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomCentroidBondForce(*cbf2);
        else if (auto * gbf = dynamic_cast<const OpenMM::CustomGBForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomGBForce(*gbf);
        else if (auto * hbf = dynamic_cast<const OpenMM::CustomHbondForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomHbondForce(*hbf);
        else if (auto * mpf = dynamic_cast<const OpenMM::CustomManyParticleForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomManyParticleForce(*mpf);
        else if (auto * rf = dynamic_cast<const OpenMM::RMSDForce *>(&cvOrig))
          cvCopy = new OpenMM::RMSDForce(*rf);
#if OMM_VER >= 84
        else if (auto * vf = dynamic_cast<const OpenMM::CustomVolumeForce *>(&cvOrig))
          cvCopy = new OpenMM::CustomVolumeForce(*vf);
        else if (auto * rgf = dynamic_cast<const OpenMM::RGForce *>(&cvOrig))
          cvCopy = new OpenMM::RGForce(*rgf);
#endif
        if (cvCopy) {
          cvf->addCollectiveVariable(name, cvCopy);
        } else {
          // A collective variable of a type this chain does not know about.
          // Adding the CustomCVForce without it would leave its energy
          // expression referring to a variable that no longer exists: OpenMM
          // then either fails at context creation or, worse, the force
          // evaluates to something wrong.  Fail the whole copy instead --
          // fstore_setup reports the skipped force to the user.
          delete cvf;
          return NULL;
        }
      }
      // Deep-copy tabulated functions
      for (int t = 0; t < og->getNumTabulatedFunctions(); ++t)
        cvf->addTabulatedFunction(og->getTabulatedFunctionName(t),
                                  og->getTabulatedFunction(t).Copy());
      outForce = cvf;
      break;
    }
    case C_CENTROID_BOND_TYPE:
      outForce = new OpenMM::CustomCentroidBondForce(
          *static_cast<OpenMM::CustomCentroidBondForce *>(ogForce));
      break;
    case C_COMPOUND_BOND_TYPE:
      outForce = new OpenMM::CustomCompoundBondForce(
          *static_cast<OpenMM::CustomCompoundBondForce *>(ogForce));
      break;
    case C_EXTERNAL_TYPE:
      outForce = new OpenMM::CustomExternalForce(
          *static_cast<OpenMM::CustomExternalForce *>(ogForce));
      break;
    case C_GB_TYPE:
      outForce = new OpenMM::CustomGBForce(
          *static_cast<OpenMM::CustomGBForce *>(ogForce));
      break;
    case C_H_BOND_TYPE:
      outForce = new OpenMM::CustomHbondForce(
          *static_cast<OpenMM::CustomHbondForce *>(ogForce));
      break;
    case C_MANY_PARTICLE_TYPE:
      outForce = new OpenMM::CustomManyParticleForce(
          *static_cast<OpenMM::CustomManyParticleForce *>(ogForce));
      break;
    case C_NONBONDED_TYPE:
      outForce = new OpenMM::CustomNonbondedForce(
          *static_cast<OpenMM::CustomNonbondedForce *>(ogForce));
      break;
    case C_TORSION_TYPE:
      outForce = new OpenMM::CustomTorsionForce(
          *static_cast<OpenMM::CustomTorsionForce *>(ogForce));
      break;
#if OMM_VER >= 84
    case C_VOLUME_TYPE:
      outForce = new OpenMM::CustomVolumeForce(
          *static_cast<OpenMM::CustomVolumeForce *>(ogForce));
      break;
#endif
    case RMSD_TYPE:
      outForce = new OpenMM::RMSDForce(
          *static_cast<OpenMM::RMSDForce *>(ogForce));
      break;
#if OMM_VER >= 84
    case RG_TYPE:
      outForce = new OpenMM::RGForce(
          *static_cast<OpenMM::RGForce *>(ogForce));
      break;
#endif
    default:
      break;
    }
  } catch(const std::out_of_range& ex) {
    outForce = NULL;
  }
  return outForce;
}

/*
 * The three functions below check the index instead of catching what
 * std::vector::at throws.  A thrown exception cannot be relied on to be
 * caught here: this process holds two C++ runtimes, one bundled inside
 * libchmm and one the system's, so an exception raised by libchmm's copy
 * unwinds looking for a handler that the system terminate handler never
 * finds.  The catch clauses these replace looked like they made an
 * out-of-range index harmless; they did not.
 *
 * Verified: before this change, api_cf_is_enabled(999) -- reachable from
 * pyCHARMM as a mistyped force index -- ended the whole run with
 * "terminating due to uncaught exception of type std::out_of_range"
 * (exit 134), and adding `catch (...)` did not help.
 *
 * Behaviour is otherwise unchanged: an index with no force reports "off",
 * which is what the catch clauses returned.
 */
bool ForcesStore::turnOn(int i) {
  if (i < 0 || i >= static_cast<int>(this->isOn.size())) {
    return false;
  }
  bool old = this->isOn[i];
  this->isOn[i] = true;
  return old;
}

bool ForcesStore::turnOff(int i) {
  if (i < 0 || i >= static_cast<int>(this->isOn.size())) {
    return false;
  }
  bool old = this->isOn[i];
  this->isOn[i] = false;
  return old;
}

#ifdef KEY_OMMTORCH
int ForcesStore::add(TorchPlugin::TorchForce * newForce) {
  int n = this->size();
  this->forces.push_back(newForce);
  this->types.push_back(TORCH_TYPE);
  this->isOn.push_back(true);
  this->inSystem.push_back(false);
  return n;
}
#endif  /* KEY_OMMTORCH */

int ForcesStore::add(OpenMM::Force * newForce, ForcesStore::ForceType kind) {
  int n = this->size();
  this->forces.push_back(newForce);
  this->types.push_back(kind);
  this->isOn.push_back(true);
  this->inSystem.push_back(false);
  return n;
}

int ForcesStore::size() {
  return this->forces.size();
}

/*
 * The three functions below record which stored forces have been copied into
 * the OpenMM System, so that fstore_setup can skip those and a second pass
 * becomes additive rather than duplicating everything.  Index checks rather
 * than exception handling, for the reason given above turnOn.
 */
void ForcesStore::markInSystem(int i) {
  if (i < 0 || i >= static_cast<int>(this->inSystem.size())) {
    return;
  }
  this->inSystem[i] = true;
}

bool ForcesStore::isInSystem(int i) {
  if (i < 0 || i >= static_cast<int>(this->inSystem.size())) {
    return false;
  }
  return this->inSystem[i];
}

void ForcesStore::clearInSystem() {
  for (size_t i = 0; i < this->inSystem.size(); ++i) {
    this->inSystem[i] = false;
  }
}

bool ForcesStore::is_i_on(int i) {
  if (i < 0 || i >= static_cast<int>(this->isOn.size())) {
    return false;
  }
  return this->isOn[i];
}

void ForcesStore::setBucketOverride(int i, int code) {
  if (i < 0 || i >= this->size()) {
    return;                            // ignore out-of-range indices
  }
  if (code < 0) {
    this->bucketOverride.erase(i);     // negative clears any override
  } else {
    this->bucketOverride[i] = code;
  }
}

int ForcesStore::getBucketOverride(int i) {
  std::map<int, int>::const_iterator it = this->bucketOverride.find(i);
  if (it == this->bucketOverride.end()) {
    return -1;
  }
  return it->second;
}


// C interface
ForcesStore * fstore_create() {
  return new ForcesStore();
}

void fstore_del(ForcesStore * store) {
  delete store;
}

OpenMM::Force * fstore_get(ForcesStore * store, int i) {
  return store->get(i);
}

OpenMM::Force * fstore_copy(ForcesStore * store, int i) {
  return store->copy(i);
}

int fstore_turn_on(ForcesStore * store, int i) {
  bool answer = store->turnOn(i);
  int retVal = answer? 1 : 0;
  return retVal;
}

int fstore_turn_off(ForcesStore * store, int i) {
  bool answer = store->turnOff(i);
  int retVal = answer? 1 : 0;
  return retVal;
}

int fstore_add(ForcesStore * store,
               ForcesStore::ForceType kind, char * description, int n) {
  OpenMM::Force * newForce = NULL;
  int newIndex = -1;
  switch(kind) {
#ifdef KEY_OMMTORCH
  case ForcesStore::TORCH_TYPE:
    newIndex = store->add(new TorchPlugin::TorchForce(description));
    break;
#endif  /* KEY_OMMTORCH */
  case ForcesStore::C_ANGLE_TYPE:
    newForce = new OpenMM::CustomAngleForce(description);
    break;
  case ForcesStore::C_BOND_TYPE:
    newForce = new OpenMM::CustomBondForce(description);
    break;
  case ForcesStore::C_CV_TYPE:
    newForce = new OpenMM::CustomCVForce(description);
    break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    newForce = new OpenMM::CustomCentroidBondForce(n, description);
    break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    newForce = new OpenMM::CustomCompoundBondForce(n, description);
    break;
  case ForcesStore::C_EXTERNAL_TYPE:
    newForce = new OpenMM::CustomExternalForce(description);
    break;
  case ForcesStore::C_GB_TYPE:
    newForce = new OpenMM::CustomGBForce();
    break;
  case ForcesStore::C_H_BOND_TYPE:
    newForce = new OpenMM::CustomHbondForce(description);
    break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    newForce = new OpenMM::CustomManyParticleForce(n, description);
    break;
  case ForcesStore::C_NONBONDED_TYPE:
    newForce = new OpenMM::CustomNonbondedForce(description);
    break;
  case ForcesStore::C_TORSION_TYPE:
    newForce = new OpenMM::CustomTorsionForce(description);
    break;
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    newForce = new OpenMM::CustomVolumeForce(description);
    break;
#endif
  default:
    break;
  }
#ifdef KEY_OMMTORCH
  if (kind == ForcesStore::TORCH_TYPE) {
    return newIndex;
  } else
#endif  /* KEY_OMMTORCH */
  if (newForce != NULL) {
    newIndex = store->add(newForce, kind);
  }
  return newIndex;
}

// Return true iff `f` is an OpenMM force of the given ForceType, for a type
// that this build actually supports.  This is the type gate for
// fstore_add_ptr: it rejects kinds not compiled into this build (so we never
// store a type that ForcesStore::copy() has no case for -> NULL -> crash in
// fstore_setup) and a force whose real class does not match the declared
// kind (which would make copy()'s static_cast undefined).  The case list is
// kept in step with the enum in forcesStore.h and the switch in copy(); the
// default branch makes an unknown/unsupported kind fail closed.
//
// PRECONDITION: `f` must already be known to point at an OpenMM::Force from
// THIS process's OpenMM library.  dynamic_cast reads the object's vtable, so
// calling this on an arbitrary non-Force pointer crashes rather than
// returning false -- it narrows a Force to a subclass, it does not
// discriminate Force from not-a-Force.  Establishing that precondition is
// the caller's job: the pyCHARMM layer checks isinstance(force, openmm.Force)
// and that the OpenMM builds match before any pointer reaches this ABI.
static bool fstore_force_is_kind(OpenMM::Force * f, int kind) {
  switch (kind) {
#ifdef KEY_OMMTORCH
  case ForcesStore::TORCH_TYPE:
    return dynamic_cast<TorchPlugin::TorchForce *>(f) != NULL;
#endif
  case ForcesStore::C_ANGLE_TYPE:
    return dynamic_cast<OpenMM::CustomAngleForce *>(f) != NULL;
  case ForcesStore::C_BOND_TYPE:
    return dynamic_cast<OpenMM::CustomBondForce *>(f) != NULL;
  case ForcesStore::C_CV_TYPE:
    return dynamic_cast<OpenMM::CustomCVForce *>(f) != NULL;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return dynamic_cast<OpenMM::CustomCentroidBondForce *>(f) != NULL;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return dynamic_cast<OpenMM::CustomCompoundBondForce *>(f) != NULL;
  case ForcesStore::C_EXTERNAL_TYPE:
    return dynamic_cast<OpenMM::CustomExternalForce *>(f) != NULL;
  case ForcesStore::C_GB_TYPE:
    return dynamic_cast<OpenMM::CustomGBForce *>(f) != NULL;
  case ForcesStore::C_H_BOND_TYPE:
    return dynamic_cast<OpenMM::CustomHbondForce *>(f) != NULL;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return dynamic_cast<OpenMM::CustomManyParticleForce *>(f) != NULL;
  case ForcesStore::C_NONBONDED_TYPE:
    return dynamic_cast<OpenMM::CustomNonbondedForce *>(f) != NULL;
  case ForcesStore::C_TORSION_TYPE:
    return dynamic_cast<OpenMM::CustomTorsionForce *>(f) != NULL;
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    return dynamic_cast<OpenMM::CustomVolumeForce *>(f) != NULL;
#endif
  case ForcesStore::RMSD_TYPE:
    return dynamic_cast<OpenMM::RMSDForce *>(f) != NULL;
#if OMM_VER >= 84
  case ForcesStore::RG_TYPE:
    return dynamic_cast<OpenMM::RGForce *>(f) != NULL;
#endif
  default:
    return false;
  }
}

int fstore_add_ptr(ForcesStore * store, void * forceptr, int kind) {
  if (store == NULL) {
    return FSTORE_ERR_NO_STORE;
  }
  if (forceptr == NULL) {
    return FSTORE_ERR_NULL_FORCE;
  }
  OpenMM::Force * f = reinterpret_cast<OpenMM::Force *>(forceptr);
  if (!fstore_force_is_kind(f, kind)) {
    return FSTORE_ERR_BAD_KIND;
  }
  return store->add(f, static_cast<ForcesStore::ForceType>(kind));
}

int fstore_set_bucket(ForcesStore * store, int i, int code) {
  if (store == NULL) {
    return FSTORE_ERR_NO_STORE;
  }
  // Reject an out-of-range bucket code at the point of the mistake rather
  // than storing it and surprising the caller at system-build time.  A
  // negative code is the documented "clear the override" request.
  if (code > FSTORE_BUCKET_MAX) {
    return FSTORE_ERR_BAD_BUCKET;
  }
  if (i < 0 || i >= store->size()) {
    return FSTORE_ERR_BAD_INDEX;
  }
  store->setBucketOverride(i, code);
  return 0;
}

int fstore_get_bucket(ForcesStore * store, int i) {
  if (store == NULL) {
    return FSTORE_ERR_NO_STORE;
  }
  // Distinguish "no such force" from "this force has no override", so a
  // caller cannot read a bad index as "using the default term".
  if (i < 0 || i >= store->size()) {
    return FSTORE_ERR_BAD_INDEX;
  }
  return store->getBucketOverride(i);
}

#ifdef KEY_OMMTORCH
int fstore_add_torch(ForcesStore * store,
                TorchPlugin::TorchForce * newForce) {
  return store->add(newForce);
}
#endif  /* KEY_OMMTORCH */

int fstore_size(ForcesStore * store) {
  return store->size();
}

// Returns the stored force's ForceType, or -1 if there is no force at `i`.
//
// getType() indexes with std::vector::at, so an out-of-range index throws.
// This function is called across the C interface from Fortran, where a C++
// exception has no handler: it would reach std::terminate and take the whole
// run down instead of reporting a bad index.  ForcesStore::is_i_on already
// catches for the same reason; this keeps the two consistent.
// Returns the stored force's ForceType, or -1 if there is no force at `i`.
//
// The index is checked here rather than left to std::vector::at, because a
// thrown exception cannot be relied on to be caught in this library.  This
// process holds two C++ runtimes -- one bundled inside libchmm, one the
// system's -- so an exception raised by libchmm's copy unwinds looking for a
// handler the system's terminate handler never finds.  Verified: with a
// `try { ... } catch (...)` around this call, an out-of-range index still
// aborted the process (exit 134, "terminating due to uncaught exception of
// type std::out_of_range"), with the stack showing the throw inside libchmm
// and the terminate inside the system libc++abi.
//
// So: no exception is allowed to arise in the first place.  That also keeps
// the promise this makes to Fortran callers, which have no way to handle one.
int fstore_get_type(ForcesStore * store, int i) {
  if (store == NULL || i < 0 || i >= store->size()) {
    return -1;
  }
  return static_cast<int>(store->getType(i));
}

// C interface to the in-System bookkeeping; see forcesStore.h for what each
// one promises.
void fstore_mark_in_system(ForcesStore * store, int i) {
  store->markInSystem(i);
}

int fstore_is_in_system(ForcesStore * store, int i) {
  return store->isInSystem(i) ? 1 : 0;
}

void fstore_clear_in_system(ForcesStore * store) {
  store->clearInSystem();
}

int fstore_is_on(ForcesStore * store, int i) {
  bool answer = store->is_i_on(i);
  int ret_val = 0;
  if (answer) {
    ret_val = 1;
  }
  return ret_val;
}

int fstore_add_rmsd(ForcesStore * store,
                    double * ref_pos, int natom,
                    int * particles, int nparticles) {
  std::vector<OpenMM::Vec3> refPos(natom);
  for (int i = 0; i < natom; ++i)
    refPos[i] = OpenMM::Vec3(ref_pos[3*i], ref_pos[3*i+1], ref_pos[3*i+2]);
  std::vector<int> parts(particles, particles + nparticles);
  auto * f = new OpenMM::RMSDForce(refPos, parts);
  return store->add(f, ForcesStore::RMSD_TYPE);
}

void cf_rmsd_set_reference_positions(ForcesStore * store, int i,
                                      double * ref_pos, int natom) {
  auto * f = static_cast<OpenMM::RMSDForce *>(store->get(i));
  if (!f) return;
  std::vector<OpenMM::Vec3> refPos(natom);
  for (int j = 0; j < natom; ++j)
    refPos[j] = OpenMM::Vec3(ref_pos[3*j], ref_pos[3*j+1], ref_pos[3*j+2]);
  f->setReferencePositions(refPos);
}

void cf_rmsd_set_particles(ForcesStore * store, int i,
                            int * particles, int nparticles) {
  auto * f = static_cast<OpenMM::RMSDForce *>(store->get(i));
  if (!f) return;
  std::vector<int> parts(particles, particles + nparticles);
  f->setParticles(parts);
}

#if OMM_VER >= 84
int fstore_add_rg(ForcesStore * store,
                  int * particles, int nparticles) {
  std::vector<int> parts(particles, particles + nparticles);
  auto * f = new OpenMM::RGForce(parts);
  return store->add(f, ForcesStore::RG_TYPE);
}
#endif
#endif  // KEY_OPENMM
