#ifdef KEY_OPENMM
#include "forcesStore.h"
#include <OpenMM.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <set>

// Helper: get force from store and cast to the expected type.
// Returns nullptr if the cast fails or the index is out of range.
template <typename T>
static T * getForceAs(ForcesStore * store, int i) {
  OpenMM::Force * f = store->get(i);
  if (!f) return nullptr;
  return dynamic_cast<T *>(f);
}

static void copy_string_to_buf(const std::string & s, char * buf, int max_len) {
  int n = std::min((int)s.size(), max_len - 1);
  for (int j = 0; j < n; j++) buf[j] = s[j];
  buf[n] = '\0';
}

extern "C" {

// ============================================================
// Generic functions (dispatched by stored type)
// ============================================================

int cf_add_global_param(ForcesStore * store, int i,
                        const char * name, double value) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    return static_cast<OpenMM::CustomBondForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_ANGLE_TYPE:
    return static_cast<OpenMM::CustomAngleForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_TORSION_TYPE:
    return static_cast<OpenMM::CustomTorsionForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_EXTERNAL_TYPE:
    return static_cast<OpenMM::CustomExternalForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->addGlobalParameter(name, value);
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->addGlobalParameter(name, value);
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    return static_cast<OpenMM::CustomVolumeForce *>(f)
        ->addGlobalParameter(name, value);
#endif
  default:
    return -1;
  }
}

void cf_set_global_param(ForcesStore * store, int i,
                         int param_index, double value) {
  OpenMM::Force * f = store->get(i);
  if (!f) return;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    static_cast<OpenMM::CustomBondForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_ANGLE_TYPE:
    static_cast<OpenMM::CustomAngleForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_TORSION_TYPE:
    static_cast<OpenMM::CustomTorsionForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_EXTERNAL_TYPE:
    static_cast<OpenMM::CustomExternalForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_NONBONDED_TYPE:
    static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_GB_TYPE:
    static_cast<OpenMM::CustomGBForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_H_BOND_TYPE:
    static_cast<OpenMM::CustomHbondForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
  case ForcesStore::C_CV_TYPE:
    static_cast<OpenMM::CustomCVForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    static_cast<OpenMM::CustomVolumeForce *>(f)
        ->setGlobalParameterDefaultValue(param_index, value);
    break;
#endif
  default:
    break;
  }
}

int cf_get_num_global_params(ForcesStore * store, int i) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    return static_cast<OpenMM::CustomBondForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_ANGLE_TYPE:
    return static_cast<OpenMM::CustomAngleForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_TORSION_TYPE:
    return static_cast<OpenMM::CustomTorsionForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_EXTERNAL_TYPE:
    return static_cast<OpenMM::CustomExternalForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->getNumGlobalParameters();
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->getNumGlobalParameters();
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    return static_cast<OpenMM::CustomVolumeForce *>(f)
        ->getNumGlobalParameters();
#endif
  default:
    return -1;
  }
}

void cf_add_energy_param_deriv(ForcesStore * store, int i,
                               const char * name) {
  OpenMM::Force * f = store->get(i);
  if (!f) return;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    static_cast<OpenMM::CustomBondForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_ANGLE_TYPE:
    static_cast<OpenMM::CustomAngleForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_TORSION_TYPE:
    static_cast<OpenMM::CustomTorsionForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_NONBONDED_TYPE:
    static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_GB_TYPE:
    static_cast<OpenMM::CustomGBForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  case ForcesStore::C_CV_TYPE:
    static_cast<OpenMM::CustomCVForce *>(f)
        ->addEnergyParameterDerivative(name);
    break;
  default:
    break;
  }
}

void cf_set_uses_pbc(ForcesStore * store, int i, int periodic) {
  OpenMM::Force * f = store->get(i);
  if (!f) return;
  bool pbc = (periodic != 0);
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    static_cast<OpenMM::CustomBondForce *>(f)
        ->setUsesPeriodicBoundaryConditions(pbc);
    break;
  case ForcesStore::C_ANGLE_TYPE:
    static_cast<OpenMM::CustomAngleForce *>(f)
        ->setUsesPeriodicBoundaryConditions(pbc);
    break;
  case ForcesStore::C_TORSION_TYPE:
    static_cast<OpenMM::CustomTorsionForce *>(f)
        ->setUsesPeriodicBoundaryConditions(pbc);
    break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->setUsesPeriodicBoundaryConditions(pbc);
    break;
  default:
    break;
  }
}

// ============================================================
// CustomBondForce
// ============================================================

int cf_bond_add_per_bond_param(ForcesStore * store, int i,
                               const char * name) {
  auto * f = getForceAs<OpenMM::CustomBondForce>(store, i);
  if (!f) return -1;
  return f->addPerBondParameter(name);
}

int cf_bond_add_bond(ForcesStore * store, int i,
                     int p1, int p2,
                     double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomBondForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addBond(p1, p2, p);
}

// ============================================================
// CustomAngleForce
// ============================================================

int cf_angle_add_per_angle_param(ForcesStore * store, int i,
                                 const char * name) {
  auto * f = getForceAs<OpenMM::CustomAngleForce>(store, i);
  if (!f) return -1;
  return f->addPerAngleParameter(name);
}

int cf_angle_add_angle(ForcesStore * store, int i,
                       int p1, int p2, int p3,
                       double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomAngleForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addAngle(p1, p2, p3, p);
}

// ============================================================
// CustomTorsionForce
// ============================================================

int cf_torsion_add_per_torsion_param(ForcesStore * store, int i,
                                     const char * name) {
  auto * f = getForceAs<OpenMM::CustomTorsionForce>(store, i);
  if (!f) return -1;
  return f->addPerTorsionParameter(name);
}

int cf_torsion_add_torsion(ForcesStore * store, int i,
                           int p1, int p2, int p3, int p4,
                           double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomTorsionForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addTorsion(p1, p2, p3, p4, p);
}

// ============================================================
// CustomExternalForce
// ============================================================

int cf_external_add_per_particle_param(ForcesStore * store, int i,
                                       const char * name) {
  auto * f = getForceAs<OpenMM::CustomExternalForce>(store, i);
  if (!f) return -1;
  return f->addPerParticleParameter(name);
}

int cf_external_add_particle(ForcesStore * store, int i,
                             int particle,
                             double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomExternalForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addParticle(particle, p);
}

// ============================================================
// CustomNonbondedForce
// ============================================================

int cf_nb_add_per_particle_param(ForcesStore * store, int i,
                                 const char * name) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  return f->addPerParticleParameter(name);
}

int cf_nb_add_particle(ForcesStore * store, int i,
                       double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addParticle(p);
}

int cf_nb_add_exclusion(ForcesStore * store, int i,
                        int p1, int p2) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  return f->addExclusion(p1, p2);
}

void cf_nb_set_nonbonded_method(ForcesStore * store, int i, int method) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  f->setNonbondedMethod(
      static_cast<OpenMM::CustomNonbondedForce::NonbondedMethod>(method));
}

void cf_nb_set_cutoff(ForcesStore * store, int i, double cutoff) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  f->setCutoffDistance(cutoff);
}

void cf_nb_set_use_switching_function(ForcesStore * store, int i, int use) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  f->setUseSwitchingFunction(use != 0);
}

void cf_nb_set_switching_distance(ForcesStore * store, int i, double distance) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  f->setSwitchingDistance(distance);
}

int cf_nb_add_interaction_group(ForcesStore * store, int i,
                                int * set1, int n1,
                                int * set2, int n2) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  std::set<int> s1(set1, set1 + n1);
  std::set<int> s2(set2, set2 + n2);
  return f->addInteractionGroup(s1, s2);
}

// ============================================================
// CustomCompoundBondForce
// ============================================================

int cf_compound_add_per_bond_param(ForcesStore * store, int i,
                                   const char * name) {
  auto * f = getForceAs<OpenMM::CustomCompoundBondForce>(store, i);
  if (!f) return -1;
  return f->addPerBondParameter(name);
}

int cf_compound_add_bond(ForcesStore * store, int i,
                         int * particles, int n_particles,
                         double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomCompoundBondForce>(store, i);
  if (!f) return -1;
  std::vector<int> parts(particles, particles + n_particles);
  std::vector<double> p(params, params + n_params);
  return f->addBond(parts, p);
}

// ============================================================
// CustomCentroidBondForce
// ============================================================

int cf_centroid_add_per_bond_param(ForcesStore * store, int i,
                                   const char * name) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  return f->addPerBondParameter(name);
}

int cf_centroid_add_group(ForcesStore * store, int i,
                          int * particles, int n_particles,
                          double * weights, int n_weights) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  std::vector<int> parts(particles, particles + n_particles);
  std::vector<double> w(weights, weights + n_weights);
  return f->addGroup(parts, w);
}

int cf_centroid_add_bond(ForcesStore * store, int i,
                         int * groups, int n_groups,
                         double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  std::vector<int> g(groups, groups + n_groups);
  std::vector<double> p(params, params + n_params);
  return f->addBond(g, p);
}

// ============================================================
// CustomGBForce
// ============================================================

int cf_gb_add_per_particle_param(ForcesStore * store, int i,
                                 const char * name) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return -1;
  return f->addPerParticleParameter(name);
}

int cf_gb_add_particle(ForcesStore * store, int i,
                       double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addParticle(p);
}

int cf_gb_add_computed_value(ForcesStore * store, int i,
                             const char * name,
                             const char * expression,
                             int type) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return -1;
  return f->addComputedValue(
      name, expression,
      static_cast<OpenMM::CustomGBForce::ComputationType>(type));
}

int cf_gb_add_energy_term(ForcesStore * store, int i,
                          const char * expression, int type) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return -1;
  return f->addEnergyTerm(
      expression,
      static_cast<OpenMM::CustomGBForce::ComputationType>(type));
}

void cf_gb_set_nonbonded_method(ForcesStore * store, int i, int method) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return;
  f->setNonbondedMethod(
      static_cast<OpenMM::CustomGBForce::NonbondedMethod>(method));
}

void cf_gb_set_cutoff(ForcesStore * store, int i, double cutoff) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return;
  f->setCutoffDistance(cutoff);
}

// ============================================================
// CustomHbondForce
// ============================================================

int cf_hbond_add_per_donor_param(ForcesStore * store, int i,
                                 const char * name) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->addPerDonorParameter(name);
}

int cf_hbond_add_per_acceptor_param(ForcesStore * store, int i,
                                    const char * name) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->addPerAcceptorParameter(name);
}

int cf_hbond_add_donor(ForcesStore * store, int i,
                       int d1, int d2, int d3,
                       double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addDonor(d1, d2, d3, p);
}

int cf_hbond_add_acceptor(ForcesStore * store, int i,
                          int a1, int a2, int a3,
                          double * params, int n_params) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addAcceptor(a1, a2, a3, p);
}

int cf_hbond_add_exclusion(ForcesStore * store, int i,
                           int donor, int acceptor) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->addExclusion(donor, acceptor);
}

void cf_hbond_set_nonbonded_method(ForcesStore * store, int i, int method) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  f->setNonbondedMethod(
      static_cast<OpenMM::CustomHbondForce::NonbondedMethod>(method));
}

void cf_hbond_set_cutoff(ForcesStore * store, int i, double cutoff) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  f->setCutoffDistance(cutoff);
}

int cf_hbond_get_num_donors(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->getNumDonors();
}

int cf_hbond_get_num_acceptors(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->getNumAcceptors();
}

void cf_hbond_set_donor_parameters(ForcesStore * store, int i,
                                    int idx, int d1, int d2, int d3,
                                    double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setDonorParameters(idx, d1, d2, d3, p);
}

void cf_hbond_get_donor_parameters(ForcesStore * store, int i,
                                    int idx, int * d1, int * d2, int * d3,
                                    double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getDonorParameters(idx, *d1, *d2, *d3, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

void cf_hbond_set_acceptor_parameters(ForcesStore * store, int i,
                                       int idx, int a1, int a2, int a3,
                                       double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setAcceptorParameters(idx, a1, a2, a3, p);
}

void cf_hbond_get_acceptor_parameters(ForcesStore * store, int i,
                                       int idx, int * a1, int * a2, int * a3,
                                       double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getAcceptorParameters(idx, *a1, *a2, *a3, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

int cf_hbond_get_num_per_donor_params(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->getNumPerDonorParameters();
}

int cf_hbond_get_num_per_acceptor_params(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f) return -1;
  return f->getNumPerAcceptorParameters();
}

void cf_hbond_get_per_donor_param_name(ForcesStore * store, int i,
                                        int param_idx, char * buf, int max_len) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  copy_string_to_buf(f->getPerDonorParameterName(param_idx), buf, max_len);
}

void cf_hbond_get_per_acceptor_param_name(ForcesStore * store, int i,
                                           int param_idx, char * buf, int max_len) {
  auto * f = getForceAs<OpenMM::CustomHbondForce>(store, i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  copy_string_to_buf(f->getPerAcceptorParameterName(param_idx), buf, max_len);
}

// ============================================================
// CustomManyParticleForce
// ============================================================

int cf_many_add_per_particle_param(ForcesStore * store, int i,
                                   const char * name) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  return f->addPerParticleParameter(name);
}

int cf_many_add_particle(ForcesStore * store, int i,
                         double * params, int n_params, int type) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  std::vector<double> p(params, params + n_params);
  return f->addParticle(p, type);
}

int cf_many_add_exclusion(ForcesStore * store, int i,
                          int p1, int p2) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  return f->addExclusion(p1, p2);
}

void cf_many_set_nonbonded_method(ForcesStore * store, int i, int method) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  f->setNonbondedMethod(
      static_cast<OpenMM::CustomManyParticleForce::NonbondedMethod>(method));
}

void cf_many_set_cutoff(ForcesStore * store, int i, double cutoff) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  f->setCutoffDistance(cutoff);
}

int cf_many_get_num_particles(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  return f->getNumParticles();
}

void cf_many_set_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params, int n,
                                      int type) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setParticleParameters(idx, p, type);
}

void cf_many_get_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params,
                                      int max_params, int * type) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  int t;
  f->getParticleParameters(idx, p, t);
  *type = t;
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

void cf_many_set_type_filter(ForcesStore * store, int i,
                              int particle_index, int * types, int n) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  std::set<int> s(types, types + n);
  f->setTypeFilter(particle_index, s);
}

void cf_many_get_type_filter(ForcesStore * store, int i,
                              int particle_index, int * types,
                              int max_types, int * num_types) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) { *num_types = 0; return; }
  std::set<int> s;
  f->getTypeFilter(particle_index, s);
  *num_types = (int)s.size();
  int j = 0;
  for (auto it = s.begin(); it != s.end() && j < max_types; ++it, ++j)
    types[j] = *it;
}

int cf_many_get_permutation_mode(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  return (int)f->getPermutationMode();
}

void cf_many_set_permutation_mode(ForcesStore * store, int i, int mode) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return;
  f->setPermutationMode(
      (OpenMM::CustomManyParticleForce::PermutationMode)mode);
}

int cf_many_get_num_per_particle_params(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f) return -1;
  return f->getNumPerParticleParameters();
}

void cf_many_get_per_particle_param_name(ForcesStore * store, int i,
                                          int param_idx, char * buf,
                                          int max_len) {
  auto * f = getForceAs<OpenMM::CustomManyParticleForce>(store, i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  copy_string_to_buf(f->getPerParticleParameterName(param_idx), buf, max_len);
}

// ============================================================
// CustomCVForce
// ============================================================

int cf_cv_add_collective_variable(ForcesStore * store,
                                  int cv_index,
                                  int cv_force_store_index,
                                  const char * name) {
  auto * f = getForceAs<OpenMM::CustomCVForce>(store, cv_index);
  if (!f) return -1;
  // Copy the force from the store to pass to the CV force.
  // CustomCVForce takes ownership of the copy.
  OpenMM::Force * cv_force = store->copy(cv_force_store_index);
  if (!cv_force) return -1;
  return f->addCollectiveVariable(name, cv_force);
}

// ============================================================
// Generic getter
// ============================================================

double cf_get_global_param_default_value(ForcesStore * store, int i,
                                         int param_idx) {
  OpenMM::Force * f = store->get(i);
  if (!f) return 0.0;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    return static_cast<OpenMM::CustomBondForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_ANGLE_TYPE:
    return static_cast<OpenMM::CustomAngleForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_TORSION_TYPE:
    return static_cast<OpenMM::CustomTorsionForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_EXTERNAL_TYPE:
    return static_cast<OpenMM::CustomExternalForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    return static_cast<OpenMM::CustomVolumeForce *>(f)
        ->getGlobalParameterDefaultValue(param_idx);
#endif
  default:
    return 0.0;
  }
}

// ============================================================
// CustomBondForce getters/setters
// ============================================================

int cf_bond_get_num_bonds(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomBondForce>(store, i);
  if (!f) return -1;
  return f->getNumBonds();
}

void cf_bond_set_bond_parameters(ForcesStore * store, int i,
                                  int idx, int p1, int p2,
                                  double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomBondForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setBondParameters(idx, p1, p2, p);
}

void cf_bond_get_bond_parameters(ForcesStore * store, int i,
                                  int idx, int * p1, int * p2,
                                  double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomBondForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getBondParameters(idx, *p1, *p2, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

void cf_angle_get_angle_parameters(ForcesStore * store, int i,
                                    int idx, int * p1, int * p2, int * p3,
                                    double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomAngleForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getAngleParameters(idx, *p1, *p2, *p3, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

// ============================================================
// CustomAngleForce getters/setters
// ============================================================

int cf_angle_get_num_angles(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomAngleForce>(store, i);
  if (!f) return -1;
  return f->getNumAngles();
}

void cf_angle_set_angle_parameters(ForcesStore * store, int i,
                                    int idx, int p1, int p2, int p3,
                                    double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomAngleForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setAngleParameters(idx, p1, p2, p3, p);
}

void cf_torsion_get_torsion_parameters(ForcesStore * store, int i,
                                        int idx, int * p1, int * p2,
                                        int * p3, int * p4,
                                        double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomTorsionForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getTorsionParameters(idx, *p1, *p2, *p3, *p4, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

// ============================================================
// CustomTorsionForce getters/setters
// ============================================================

int cf_torsion_get_num_torsions(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomTorsionForce>(store, i);
  if (!f) return -1;
  return f->getNumTorsions();
}

void cf_torsion_set_torsion_parameters(ForcesStore * store, int i,
                                        int idx, int p1, int p2,
                                        int p3, int p4,
                                        double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomTorsionForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setTorsionParameters(idx, p1, p2, p3, p4, p);
}

void cf_external_get_particle_parameters(ForcesStore * store, int i,
                                          int idx, int * particle,
                                          double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomExternalForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getParticleParameters(idx, *particle, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

// ============================================================
// CustomExternalForce getters/setters
// ============================================================

int cf_external_get_num_particles(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomExternalForce>(store, i);
  if (!f) return -1;
  return f->getNumParticles();
}

void cf_external_set_particle_parameters(ForcesStore * store, int i,
                                          int idx, int particle,
                                          double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomExternalForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setParticleParameters(idx, particle, p);
}

// ============================================================
// CustomNonbondedForce getters/setters
// ============================================================

int cf_nb_get_num_particles(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  return f->getNumParticles();
}

void cf_nb_set_particle_parameters(ForcesStore * store, int i,
                                    int idx, double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setParticleParameters(idx, p);
}

void cf_nb_get_particle_parameters(ForcesStore * store, int i,
                                    int idx, double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getParticleParameters(idx, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

int cf_nb_get_nonbonded_method(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1;
  return static_cast<int>(f->getNonbondedMethod());
}

double cf_nb_get_cutoff(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomNonbondedForce>(store, i);
  if (!f) return -1.0;
  return f->getCutoffDistance();
}

// ============================================================
// CustomCompoundBondForce getters/setters
// ============================================================

int cf_compound_get_num_bonds(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomCompoundBondForce>(store, i);
  if (!f) return -1;
  return f->getNumBonds();
}

void cf_compound_set_bond_parameters(ForcesStore * store, int i,
                                      int idx, int * particles, int np,
                                      double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomCompoundBondForce>(store, i);
  if (!f) return;
  std::vector<int> parts(particles, particles + np);
  std::vector<double> p(params, params + n);
  f->setBondParameters(idx, parts, p);
}

void cf_compound_get_bond_parameters(ForcesStore * store, int i,
                                      int idx, int * particles, int max_p,
                                      double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomCompoundBondForce>(store, i);
  if (!f) return;
  std::vector<int> parts;
  std::vector<double> p;
  f->getBondParameters(idx, parts, p);
  int np = std::min((int)parts.size(), max_p);
  for (int j = 0; j < np; j++) particles[j] = parts[j];
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

// ============================================================
// CustomCentroidBondForce getters
// ============================================================

int cf_centroid_get_num_groups(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  return f->getNumGroups();
}

int cf_centroid_get_num_bonds(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  return f->getNumBonds();
}

void cf_centroid_set_group_parameters(ForcesStore * store, int i,
                                       int idx, int * particles, int np,
                                       double * weights, int nw) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return;
  std::vector<int> parts(particles, particles + np);
  std::vector<double> w(weights, weights + nw);
  f->setGroupParameters(idx, parts, w);
}

void cf_centroid_get_group_parameters(ForcesStore * store, int i,
                                       int idx, int * particles, int max_p,
                                       double * weights, int max_w,
                                       int * num_particles, int * num_weights) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return;
  std::vector<int> parts;
  std::vector<double> w;
  f->getGroupParameters(idx, parts, w);
  *num_particles = (int)parts.size();
  *num_weights = (int)w.size();
  int np = std::min((int)parts.size(), max_p);
  for (int j = 0; j < np; j++) particles[j] = parts[j];
  int nw = std::min((int)w.size(), max_w);
  for (int j = 0; j < nw; j++) weights[j] = w[j];
}

void cf_centroid_set_bond_parameters(ForcesStore * store, int i,
                                      int idx, int * groups, int ng,
                                      double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return;
  std::vector<int> g(groups, groups + ng);
  std::vector<double> p(params, params + n);
  f->setBondParameters(idx, g, p);
}

void cf_centroid_get_bond_parameters(ForcesStore * store, int i,
                                      int idx, int * groups, int max_g,
                                      double * params, int max_params,
                                      int * num_groups, int * num_params) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return;
  std::vector<int> g;
  std::vector<double> p;
  f->getBondParameters(idx, g, p);
  *num_groups = (int)g.size();
  *num_params = (int)p.size();
  int ng = std::min((int)g.size(), max_g);
  for (int j = 0; j < ng; j++) groups[j] = g[j];
  int np = std::min((int)p.size(), max_params);
  for (int j = 0; j < np; j++) params[j] = p[j];
}

int cf_centroid_get_num_per_bond_params(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f) return -1;
  return f->getNumPerBondParameters();
}

void cf_centroid_get_per_bond_param_name(ForcesStore * store, int i,
                                          int param_idx, char * buf,
                                          int max_len) {
  auto * f = getForceAs<OpenMM::CustomCentroidBondForce>(store, i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  copy_string_to_buf(f->getPerBondParameterName(param_idx), buf, max_len);
}

// ============================================================
// CustomGBForce getters/setters
// ============================================================

int cf_gb_get_num_particles(ForcesStore * store, int i) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return -1;
  return f->getNumParticles();
}

void cf_gb_set_particle_parameters(ForcesStore * store, int i,
                                    int idx, double * params, int n) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return;
  std::vector<double> p(params, params + n);
  f->setParticleParameters(idx, p);
}

void cf_gb_get_particle_parameters(ForcesStore * store, int i,
                                    int idx, double * params, int max_params) {
  auto * f = getForceAs<OpenMM::CustomGBForce>(store, i);
  if (!f) return;
  std::vector<double> p;
  f->getParticleParameters(idx, p);
  int n = std::min((int)p.size(), max_params);
  for (int j = 0; j < n; j++) params[j] = p[j];
}

// ============================================================
// Tabulated functions (Continuous1D and Discrete1D)
// ============================================================

int cf_add_tabulated_function_continuous1d(ForcesStore * store, int i,
    const char * name, double * values, int n,
    double min_val, double max_val, int periodic) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + n);
  auto * func = new OpenMM::Continuous1DFunction(vals, min_val, max_val,
                                                  periodic != 0);
  switch (store->getType(i)) {
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->addTabulatedFunction(name, func);
  default:
    delete func;
    return -1;
  }
}

int cf_add_tabulated_function_discrete1d(ForcesStore * store, int i,
    const char * name, double * values, int n) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + n);
  auto * func = new OpenMM::Discrete1DFunction(vals);
  switch (store->getType(i)) {
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->addTabulatedFunction(name, func);
  default:
    delete func;
    return -1;
  }
}

// ============================================================
// updateParametersInContext
// ============================================================

void cf_update_parameters_in_context(ForcesStore * store, int i,
    void * context_ptr) {
  OpenMM::Force * f = store->get(i);
  if (!f || !context_ptr) return;
  OpenMM::Context * ctx = static_cast<OpenMM::Context *>(context_ptr);
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    static_cast<OpenMM::CustomBondForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_ANGLE_TYPE:
    static_cast<OpenMM::CustomAngleForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_TORSION_TYPE:
    static_cast<OpenMM::CustomTorsionForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_EXTERNAL_TYPE:
    static_cast<OpenMM::CustomExternalForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_NONBONDED_TYPE:
    static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_GB_TYPE:
    static_cast<OpenMM::CustomGBForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_H_BOND_TYPE:
    static_cast<OpenMM::CustomHbondForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->updateParametersInContext(*ctx);
    break;
  default:
    break;
  }
}

// ============================================================
// Force groups
// ============================================================

void cf_set_force_group(ForcesStore * store, int i, int group) {
  OpenMM::Force * f = store->get(i);
  if (!f) return;
  f->setForceGroup(group);
}

int cf_get_force_group(ForcesStore * store, int i) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  return f->getForceGroup();
}

// ============================================================
// 2D/3D tabulated functions
// ============================================================

static int add_tabulated_to_force(OpenMM::Force * f, ForcesStore::ForceType t,
                                   const char * name, OpenMM::TabulatedFunction * func) {
  switch (t) {
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_H_BOND_TYPE:
    return static_cast<OpenMM::CustomHbondForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)
        ->addTabulatedFunction(name, func);
  case ForcesStore::C_CV_TYPE:
    return static_cast<OpenMM::CustomCVForce *>(f)
        ->addTabulatedFunction(name, func);
  default:
    delete func;
    return -1;
  }
}

int cf_add_tabulated_function_continuous2d(ForcesStore * store, int i,
    const char * name, double * values, int nx, int ny,
    double xmin, double xmax, double ymin, double ymax, int periodic) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + nx * ny);
  auto * func = new OpenMM::Continuous2DFunction(nx, ny, vals,
      xmin, xmax, ymin, ymax, periodic != 0);
  return add_tabulated_to_force(f, store->getType(i), name, func);
}

int cf_add_tabulated_function_discrete2d(ForcesStore * store, int i,
    const char * name, double * values, int nx, int ny) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + nx * ny);
  auto * func = new OpenMM::Discrete2DFunction(nx, ny, vals);
  return add_tabulated_to_force(f, store->getType(i), name, func);
}

int cf_add_tabulated_function_continuous3d(ForcesStore * store, int i,
    const char * name, double * values, int nx, int ny, int nz,
    double xmin, double xmax, double ymin, double ymax,
    double zmin, double zmax, int periodic) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + nx * ny * nz);
  auto * func = new OpenMM::Continuous3DFunction(nx, ny, nz, vals,
      xmin, xmax, ymin, ymax, zmin, zmax, periodic != 0);
  return add_tabulated_to_force(f, store->getType(i), name, func);
}

int cf_add_tabulated_function_discrete3d(ForcesStore * store, int i,
    const char * name, double * values, int nx, int ny, int nz) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  std::vector<double> vals(values, values + nx * ny * nz);
  auto * func = new OpenMM::Discrete3DFunction(nx, ny, nz, vals);
  return add_tabulated_to_force(f, store->getType(i), name, func);
}

// ============================================================
// Introspection: energy expression, param names, num per-params
// ============================================================

void cf_get_energy_expression(ForcesStore * store, int i,
                               char * buf, int max_len) {
  OpenMM::Force * f = store->get(i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  std::string expr;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    expr = static_cast<OpenMM::CustomBondForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_ANGLE_TYPE:
    expr = static_cast<OpenMM::CustomAngleForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_TORSION_TYPE:
    expr = static_cast<OpenMM::CustomTorsionForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_EXTERNAL_TYPE:
    expr = static_cast<OpenMM::CustomExternalForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_NONBONDED_TYPE:
    expr = static_cast<OpenMM::CustomNonbondedForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    expr = static_cast<OpenMM::CustomCompoundBondForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    expr = static_cast<OpenMM::CustomCentroidBondForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_H_BOND_TYPE:
    expr = static_cast<OpenMM::CustomHbondForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    expr = static_cast<OpenMM::CustomManyParticleForce *>(f)->getEnergyFunction(); break;
  case ForcesStore::C_CV_TYPE:
    expr = static_cast<OpenMM::CustomCVForce *>(f)->getEnergyFunction(); break;
  default:
    buf[0] = '\0'; return;
  }
  copy_string_to_buf(expr, buf, max_len);
}

void cf_get_global_param_name(ForcesStore * store, int i,
                               int param_idx, char * buf, int max_len) {
  OpenMM::Force * f = store->get(i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  std::string name;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    name = static_cast<OpenMM::CustomBondForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_ANGLE_TYPE:
    name = static_cast<OpenMM::CustomAngleForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_TORSION_TYPE:
    name = static_cast<OpenMM::CustomTorsionForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_EXTERNAL_TYPE:
    name = static_cast<OpenMM::CustomExternalForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_NONBONDED_TYPE:
    name = static_cast<OpenMM::CustomNonbondedForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    name = static_cast<OpenMM::CustomCompoundBondForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    name = static_cast<OpenMM::CustomCentroidBondForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_GB_TYPE:
    name = static_cast<OpenMM::CustomGBForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_H_BOND_TYPE:
    name = static_cast<OpenMM::CustomHbondForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    name = static_cast<OpenMM::CustomManyParticleForce *>(f)->getGlobalParameterName(param_idx); break;
  case ForcesStore::C_CV_TYPE:
    name = static_cast<OpenMM::CustomCVForce *>(f)->getGlobalParameterName(param_idx); break;
#if OMM_VER >= 84
  case ForcesStore::C_VOLUME_TYPE:
    name = static_cast<OpenMM::CustomVolumeForce *>(f)->getGlobalParameterName(param_idx); break;
#endif
  default:
    buf[0] = '\0'; return;
  }
  copy_string_to_buf(name, buf, max_len);
}

int cf_get_num_per_params(ForcesStore * store, int i) {
  OpenMM::Force * f = store->get(i);
  if (!f) return -1;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    return static_cast<OpenMM::CustomBondForce *>(f)->getNumPerBondParameters();
  case ForcesStore::C_ANGLE_TYPE:
    return static_cast<OpenMM::CustomAngleForce *>(f)->getNumPerAngleParameters();
  case ForcesStore::C_TORSION_TYPE:
    return static_cast<OpenMM::CustomTorsionForce *>(f)->getNumPerTorsionParameters();
  case ForcesStore::C_EXTERNAL_TYPE:
    return static_cast<OpenMM::CustomExternalForce *>(f)->getNumPerParticleParameters();
  case ForcesStore::C_NONBONDED_TYPE:
    return static_cast<OpenMM::CustomNonbondedForce *>(f)->getNumPerParticleParameters();
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    return static_cast<OpenMM::CustomCompoundBondForce *>(f)->getNumPerBondParameters();
  case ForcesStore::C_CENTROID_BOND_TYPE:
    return static_cast<OpenMM::CustomCentroidBondForce *>(f)->getNumPerBondParameters();
  case ForcesStore::C_GB_TYPE:
    return static_cast<OpenMM::CustomGBForce *>(f)->getNumPerParticleParameters();
  case ForcesStore::C_H_BOND_TYPE:
    // Return donor params count (acceptor available via separate function)
    return static_cast<OpenMM::CustomHbondForce *>(f)->getNumPerDonorParameters();
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    return static_cast<OpenMM::CustomManyParticleForce *>(f)->getNumPerParticleParameters();
  default:
    return -1;
  }
}

void cf_get_per_param_name(ForcesStore * store, int i,
                            int param_idx, char * buf, int max_len) {
  OpenMM::Force * f = store->get(i);
  if (!f || max_len <= 0) { if (max_len > 0) buf[0] = '\0'; return; }
  std::string name;
  switch (store->getType(i)) {
  case ForcesStore::C_BOND_TYPE:
    name = static_cast<OpenMM::CustomBondForce *>(f)->getPerBondParameterName(param_idx); break;
  case ForcesStore::C_ANGLE_TYPE:
    name = static_cast<OpenMM::CustomAngleForce *>(f)->getPerAngleParameterName(param_idx); break;
  case ForcesStore::C_TORSION_TYPE:
    name = static_cast<OpenMM::CustomTorsionForce *>(f)->getPerTorsionParameterName(param_idx); break;
  case ForcesStore::C_EXTERNAL_TYPE:
    name = static_cast<OpenMM::CustomExternalForce *>(f)->getPerParticleParameterName(param_idx); break;
  case ForcesStore::C_NONBONDED_TYPE:
    name = static_cast<OpenMM::CustomNonbondedForce *>(f)->getPerParticleParameterName(param_idx); break;
  case ForcesStore::C_COMPOUND_BOND_TYPE:
    name = static_cast<OpenMM::CustomCompoundBondForce *>(f)->getPerBondParameterName(param_idx); break;
  case ForcesStore::C_CENTROID_BOND_TYPE:
    name = static_cast<OpenMM::CustomCentroidBondForce *>(f)->getPerBondParameterName(param_idx); break;
  case ForcesStore::C_GB_TYPE:
    name = static_cast<OpenMM::CustomGBForce *>(f)->getPerParticleParameterName(param_idx); break;
  case ForcesStore::C_H_BOND_TYPE:
    name = static_cast<OpenMM::CustomHbondForce *>(f)->getPerDonorParameterName(param_idx); break;
  case ForcesStore::C_MANY_PARTICLE_TYPE:
    name = static_cast<OpenMM::CustomManyParticleForce *>(f)->getPerParticleParameterName(param_idx); break;
  default:
    buf[0] = '\0'; return;
  }
  copy_string_to_buf(name, buf, max_len);
}

}  // extern "C"
#endif  // KEY_OPENMM
