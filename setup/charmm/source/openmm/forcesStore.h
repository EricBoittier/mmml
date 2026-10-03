#ifdef KEY_OPENMM
#ifndef FORCES_STORE
#define FORCES_STORE

/**
 * @file forcesStore.h
 * @brief Registry for user-added OpenMM forces and its C ABI.
 *
 * ForcesStore holds OpenMM Force objects that pyCHARMM (or other callers)
 * add to a simulation, tracks each force's ForceType and on/off state, and
 * hands copies to the OpenMM System when it is built (see fstore_setup in
 * fstore.F90).  The extern "C" block below is the ABI that the Fortran
 * api_* wrappers (source/api/api_omm.F90) bind to.
 *
 * @warning Any Force added here (in particular via ::fstore_add_ptr) must
 *          come from the SAME OpenMM library CHARMM is linked against.  A
 *          pointer to a Force created by a different OpenMM build (e.g. a
 *          mismatched conda openmm package) is invalid in this process and
 *          will crash when evaluated.  Callers that obtain forces from
 *          another language binding must verify the OpenMM builds match
 *          before handing a pointer across.
 */

#include <OpenMM.h>
#include <vector>
#include <map>

/** @name fstore error codes
 *  Negative status values returned across the C ABI.  Callers report these
 *  through CHARMM's normal output (see api_omm.F90) rather than the C++ side
 *  writing to stderr, so messages honour prnlev and output redirection.
 *  @{
 */
#define FSTORE_ERR_NO_STORE    -1  //!< the forces store does not exist
#define FSTORE_ERR_NULL_FORCE  -2  //!< the supplied force pointer was NULL
#define FSTORE_ERR_BAD_KIND    -3  //!< not a supported force of that kind
#define FSTORE_ERR_BAD_BUCKET  -4  //!< ETERM bucket code out of range
#define FSTORE_ERR_BAD_INDEX   -5  //!< store index out of range
/** @} */

//! Highest valid ETERM bucket code; must match FB_MAX in fstore.F90.
#define FSTORE_BUCKET_MAX       5

#ifdef KEY_OMMTORCH
#include <TorchForce.h>
#endif

/**
 * @brief Ordered registry of OpenMM forces added at runtime.
 *
 * Each stored force has a stable integer index (its position, returned by
 * the add functions), a ForceType, an on/off flag, and an optional ETERM
 * bucket override.  Indices are not reused within a store's lifetime.
 */
class ForcesStore {
public:
  enum ForceType {
#ifdef KEY_OMMTORCH
    TORCH_TYPE = 0,
#endif  /* KEY_OMMTORCH */
    C_ANGLE_TYPE = 1,
    C_BOND_TYPE = 2,
    C_CV_TYPE = 3,
    C_CENTROID_BOND_TYPE = 4,
    C_COMPOUND_BOND_TYPE = 5,
    C_EXTERNAL_TYPE = 6,
    C_GB_TYPE = 7,
    C_H_BOND_TYPE = 8,
    C_MANY_PARTICLE_TYPE = 9,
    C_NONBONDED_TYPE = 10,
    C_TORSION_TYPE = 11
#if OMM_VER >= 84
    , C_VOLUME_TYPE = 12
#endif
    , RMSD_TYPE = 13
#if OMM_VER >= 84
    , RG_TYPE = 14
#endif
  };

  ForcesStore() {}

  /*
   * Ownership invariant -- deliberately NO destructor freeing `forces`.
   *
   * The store keeps the original force objects and hands copy() results to
   * the OpenMM System, which owns those copies.  The stored originals are
   * intentionally never deleted, so each force is leaked once when the store
   * is destroyed (OMM OFF / OMM CLEAR -> fstore_reset).  That leak is small
   * and bounded, and removing it is NOT as simple as adding a destructor:
   *
   *   CustomCompoundBondForce, CustomCentroidBondForce, CustomGBForce,
   *   CustomHbondForce and CustomManyParticleForce own their
   *   TabulatedFunction objects and delete them in their destructors, but
   *   none of them declares a copy constructor.  The implicit one therefore
   *   copies the TabulatedFunction *pointer*, so a copy() result and the
   *   stored original share it.  Deleting both frees that function twice --
   *   verified: copying such a force and destroying both segfaults.
   *
   * So freeing the originals here would corrupt the heap for any force with
   * a tabulated function.  Doing it properly means giving copy() a deep copy
   * of every owned TabulatedFunction first (the CustomCVForce branch already
   * does this for its own functions via TabulatedFunction::Copy()).  Until
   * then, leaking is the safe behaviour.
   */

  OpenMM::Force * get(int i);
  ForceType getType(int i);
  OpenMM::Force * copy(int i);

#ifdef KEY_OMMTORCH
  int add(TorchPlugin::TorchForce * newForce);
#endif  /* KEY_OMMTORCH */
  
  int add(OpenMM::Force * newForce, ForceType kind);
  
  bool turnOn(int i);
  bool turnOff(int i);
  int size();
  bool is_i_on(int i);

  //! Note that force @p i has been added to the OpenMM System.
  void markInSystem(int i);
  //! Whether force @p i is already in the OpenMM System.
  bool isInSystem(int i);
  //! Forget which forces are in the System, because the System is gone.
  void clearInSystem();

  /**
   * @brief Pin force @p i to a specific ETERM bucket, overriding the default
   *        chosen from its ForceType.
   * @param i    store index of the force; out-of-range indices are ignored.
   * @param code bucket code (see the bucket-code list in fstore_setup);
   *             pass a negative value to clear the override and restore the
   *             ForceType default.
   */
  void setBucketOverride(int i, int code);

  /**
   * @brief Return the ETERM bucket override for force @p i.
   * @param i store index of the force.
   * @return the override bucket code, or -1 if none is set (or @p i is out
   *         of range).
   */
  int getBucketOverride(int i);
private:
  // There is a single store per run (see fstore.F90) and it holds raw force
  // pointers, so copying one is a mistake: the copies would alias the same
  // forces.  Declared and not defined, so any attempt fails to link.
  ForcesStore(const ForcesStore &);
  ForcesStore & operator=(const ForcesStore &);

  std::vector<OpenMM::Force *> forces;
  std::vector<ForceType> types;
  std::vector<bool> isOn;
  /*!
   * Whether this force's copy is currently in the OpenMM System.
   *
   * fstore_setup adds every enabled force it finds, so without this it would
   * add them all again on a second pass and the energy would count each one
   * twice.  Tracking it makes a second pass additive: only forces not yet in
   * the System are added, which is what lets a force be added after the
   * System was built without rebuilding the whole thing.
   *
   * Cleared when the System is destroyed (the copies go with it).
   */
  std::vector<bool> inSystem;
  //! index -> ETERM bucket code; an absent key means "use ForceType default".
  std::map<int, int> bucketOverride;
};

extern "C" {
  ForcesStore * fstore_create();
  void fstore_del(ForcesStore * store);
  OpenMM::Force * fstore_get(ForcesStore * store, int i);
  OpenMM::Force * fstore_copy(ForcesStore * store, int i);
  int fstore_turn_on(ForcesStore * store, int i);
  int fstore_turn_off(ForcesStore * store, int i);
  int fstore_add(ForcesStore * store,
                 ForcesStore::ForceType kind, char * description,
                 int n);

  /**
   * @brief Register a pre-built OpenMM::Force by raw pointer, taking
   *        ownership of it.
   *
   * Adds a force constructed elsewhere (e.g. in Python via the openmm
   * package) without rebuilding it here.  The caller must have relinquished
   * ownership on its side first (Python: `force.thisown = False`).
   *
   * Once @p forceptr is known to be a Force (see the precondition below),
   * its class is checked against @p kind before it is stored, so a
   * mismatched or unsupported kind is rejected instead of causing a bad cast
   * later.  On rejection nothing is stored and the caller still owns the
   * object.
   *
   * @pre @p forceptr MUST already be known to point at an OpenMM::Force
   *      created by THIS process's OpenMM library.  This function cannot
   *      verify that: the type check narrows a Force to a subclass (it reads
   *      the object's vtable), so an arbitrary non-Force pointer, a pointer
   *      from a different OpenMM build, or a freed pointer crashes rather
   *      than being rejected.  Callers crossing a language boundary must
   *      establish this themselves -- the pyCHARMM layer checks
   *      isinstance(force, openmm.Force) and that the Python and CHARMM
   *      OpenMM builds match before calling.
   *
   * @param store    the store (must be non-NULL).
   * @param forceptr the OpenMM::Force* to adopt (must be non-NULL).
   * @param kind     the force's ForcesStore::ForceType as an int.
   * @return the new force's store index (>= 0), or a negative
   *         FSTORE_ERR_* code: #FSTORE_ERR_NO_STORE, #FSTORE_ERR_NULL_FORCE,
   *         or #FSTORE_ERR_BAD_KIND (wrong type for @p kind, or a type this
   *         build does not support).
   */
  int fstore_add_ptr(ForcesStore * store, void * forceptr, int kind);

  /**
   * @brief Pin force @p i to a specific ETERM bucket (ABI wrapper for
   *        ForcesStore::setBucketOverride).
   * @param store the store.
   * @param i     store index of the force.
   * @param code  bucket code in [0, #FSTORE_BUCKET_MAX], or negative to
   *              clear the override.
   * @return 0 on success, or a negative FSTORE_ERR_* code:
   *         #FSTORE_ERR_NO_STORE, #FSTORE_ERR_BAD_BUCKET (code above
   *         #FSTORE_BUCKET_MAX), or #FSTORE_ERR_BAD_INDEX.
   */
  int fstore_set_bucket(ForcesStore * store, int i, int code);

  /**
   * @brief Return the ETERM bucket override for force @p i (ABI wrapper for
   *        ForcesStore::getBucketOverride).
   * @param store the store.
   * @param i     store index of the force.
   * @return the bucket code (>= 0), -1 if this force has no override, or
   *         #FSTORE_ERR_NO_STORE / #FSTORE_ERR_BAD_INDEX.  "No override" is
   *         reported distinctly from "no such force" so a bad index cannot
   *         be mistaken for a force using its default term.
   */
  int fstore_get_bucket(ForcesStore * store, int i);
#ifdef KEY_OMMTORCH
  int fstore_add_torch(ForcesStore * store,
                  TorchPlugin::TorchForce * newForce);
#endif  /* KEY_OMMTORCH */
  int fstore_size(ForcesStore * store);
  int fstore_is_on(ForcesStore * store, int i);
  /*!
   * @brief Note that force @p i's copy is now in the OpenMM System.
   * @param store the forces store
   * @param i     store index of the force; an out-of-range index is ignored
   */
  void fstore_mark_in_system(ForcesStore * store, int i);
  /*!
   * @brief Whether force @p i's copy is already in the OpenMM System.
   *
   * fstore_setup adds every enabled force it finds, so it consults this to
   * skip the ones already there.  Without it a second pass would add them
   * all again and the energy would count each one twice.
   *
   * @param store the forces store
   * @param i     store index of the force
   * @return 1 if the force is in the System, 0 if it is not or there is no
   *         force at @p i
   */
  int fstore_is_in_system(ForcesStore * store, int i);
  /*!
   * @brief Forget which forces are in the System, because the System is gone.
   *
   * Call whenever a System is destroyed or a fresh one is created: its copies
   * of the stored forces go with it, so nothing is "already in the System"
   * any more.  Missing this leaves fstore_setup skipping every force and the
   * energy reported as zero.
   *
   * @param store the forces store
   */
  void fstore_clear_in_system(ForcesStore * store);
  int fstore_get_type(ForcesStore * store, int i);

  // customForces.cpp: generic dispatch
  int cf_add_global_param(ForcesStore * store, int i,
                          const char * name, double value);
  void cf_set_global_param(ForcesStore * store, int i,
                           int param_index, double value);
  int cf_get_num_global_params(ForcesStore * store, int i);
  void cf_add_energy_param_deriv(ForcesStore * store, int i,
                                 const char * name);
  void cf_set_uses_pbc(ForcesStore * store, int i, int periodic);

  // CustomBondForce
  int cf_bond_add_per_bond_param(ForcesStore * store, int i,
                                 const char * name);
  int cf_bond_add_bond(ForcesStore * store, int i,
                       int p1, int p2,
                       double * params, int n_params);

  // CustomAngleForce
  int cf_angle_add_per_angle_param(ForcesStore * store, int i,
                                   const char * name);
  int cf_angle_add_angle(ForcesStore * store, int i,
                         int p1, int p2, int p3,
                         double * params, int n_params);

  // CustomTorsionForce
  int cf_torsion_add_per_torsion_param(ForcesStore * store, int i,
                                       const char * name);
  int cf_torsion_add_torsion(ForcesStore * store, int i,
                             int p1, int p2, int p3, int p4,
                             double * params, int n_params);

  // CustomExternalForce
  int cf_external_add_per_particle_param(ForcesStore * store, int i,
                                         const char * name);
  int cf_external_add_particle(ForcesStore * store, int i,
                               int particle,
                               double * params, int n_params);

  // CustomNonbondedForce
  int cf_nb_add_per_particle_param(ForcesStore * store, int i,
                                   const char * name);
  int cf_nb_add_particle(ForcesStore * store, int i,
                         double * params, int n_params);
  int cf_nb_add_exclusion(ForcesStore * store, int i, int p1, int p2);
  void cf_nb_set_nonbonded_method(ForcesStore * store, int i, int method);
  void cf_nb_set_cutoff(ForcesStore * store, int i, double cutoff);
  void cf_nb_set_use_switching_function(ForcesStore * store, int i, int use);
  void cf_nb_set_switching_distance(ForcesStore * store, int i,
                                    double distance);
  int cf_nb_add_interaction_group(ForcesStore * store, int i,
                                  int * set1, int n1, int * set2, int n2);

  // CustomCompoundBondForce
  int cf_compound_add_per_bond_param(ForcesStore * store, int i,
                                     const char * name);
  int cf_compound_add_bond(ForcesStore * store, int i,
                           int * particles, int n_particles,
                           double * params, int n_params);

  // CustomCentroidBondForce
  int cf_centroid_add_per_bond_param(ForcesStore * store, int i,
                                     const char * name);
  int cf_centroid_add_group(ForcesStore * store, int i,
                            int * particles, int n_particles,
                            double * weights, int n_weights);
  int cf_centroid_add_bond(ForcesStore * store, int i,
                           int * groups, int n_groups,
                           double * params, int n_params);

  // CustomGBForce
  int cf_gb_add_per_particle_param(ForcesStore * store, int i,
                                   const char * name);
  int cf_gb_add_particle(ForcesStore * store, int i,
                         double * params, int n_params);
  int cf_gb_add_computed_value(ForcesStore * store, int i,
                               const char * name, const char * expression,
                               int type);
  int cf_gb_add_energy_term(ForcesStore * store, int i,
                            const char * expression, int type);
  void cf_gb_set_nonbonded_method(ForcesStore * store, int i, int method);
  void cf_gb_set_cutoff(ForcesStore * store, int i, double cutoff);

  // CustomHbondForce
  int cf_hbond_add_per_donor_param(ForcesStore * store, int i,
                                   const char * name);
  int cf_hbond_add_per_acceptor_param(ForcesStore * store, int i,
                                      const char * name);
  int cf_hbond_add_donor(ForcesStore * store, int i,
                         int d1, int d2, int d3,
                         double * params, int n_params);
  int cf_hbond_add_acceptor(ForcesStore * store, int i,
                            int a1, int a2, int a3,
                            double * params, int n_params);
  int cf_hbond_add_exclusion(ForcesStore * store, int i,
                             int donor, int acceptor);
  void cf_hbond_set_nonbonded_method(ForcesStore * store, int i, int method);
  void cf_hbond_set_cutoff(ForcesStore * store, int i, double cutoff);

  // CustomManyParticleForce
  int cf_many_add_per_particle_param(ForcesStore * store, int i,
                                     const char * name);
  int cf_many_add_particle(ForcesStore * store, int i,
                           double * params, int n_params, int type);
  int cf_many_add_exclusion(ForcesStore * store, int i, int p1, int p2);
  void cf_many_set_nonbonded_method(ForcesStore * store, int i, int method);
  void cf_many_set_cutoff(ForcesStore * store, int i, double cutoff);

  // CustomCVForce
  int cf_cv_add_collective_variable(ForcesStore * store, int cv_index,
                                    int cv_force_store_index,
                                    const char * name);

  // RMSDForce
  int fstore_add_rmsd(ForcesStore * store,
                      double * ref_pos, int natom,
                      int * particles, int nparticles);
  void cf_rmsd_set_reference_positions(ForcesStore * store, int i,
                                        double * ref_pos, int natom);
  void cf_rmsd_set_particles(ForcesStore * store, int i,
                              int * particles, int nparticles);

  // RGForce (OpenMM 8.4+)
#if OMM_VER >= 84
  int fstore_add_rg(ForcesStore * store,
                    int * particles, int nparticles);
#endif

  // Generic getter
  double cf_get_global_param_default_value(ForcesStore * store, int i,
                                           int param_idx);

  // CustomBondForce getters/setters
  int cf_bond_get_num_bonds(ForcesStore * store, int i);
  void cf_bond_set_bond_parameters(ForcesStore * store, int i,
                                    int idx, int p1, int p2,
                                    double * params, int n);
  void cf_bond_get_bond_parameters(ForcesStore * store, int i,
                                    int idx, int * p1, int * p2,
                                    double * params, int max_params);

  // CustomAngleForce getters/setters
  int cf_angle_get_num_angles(ForcesStore * store, int i);
  void cf_angle_set_angle_parameters(ForcesStore * store, int i,
                                      int idx, int p1, int p2, int p3,
                                      double * params, int n);
  void cf_angle_get_angle_parameters(ForcesStore * store, int i,
                                      int idx, int * p1, int * p2, int * p3,
                                      double * params, int max_params);

  // CustomTorsionForce getters/setters
  int cf_torsion_get_num_torsions(ForcesStore * store, int i);
  void cf_torsion_set_torsion_parameters(ForcesStore * store, int i,
                                          int idx, int p1, int p2,
                                          int p3, int p4,
                                          double * params, int n);
  void cf_torsion_get_torsion_parameters(ForcesStore * store, int i,
                                          int idx, int * p1, int * p2,
                                          int * p3, int * p4,
                                          double * params, int max_params);

  // CustomExternalForce getters/setters
  int cf_external_get_num_particles(ForcesStore * store, int i);
  void cf_external_set_particle_parameters(ForcesStore * store, int i,
                                            int idx, int particle,
                                            double * params, int n);
  void cf_external_get_particle_parameters(ForcesStore * store, int i,
                                            int idx, int * particle,
                                            double * params, int max_params);

  // CustomNonbondedForce getters/setters
  int cf_nb_get_num_particles(ForcesStore * store, int i);
  void cf_nb_set_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params, int n);
  void cf_nb_get_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params, int max_params);
  int cf_nb_get_nonbonded_method(ForcesStore * store, int i);
  double cf_nb_get_cutoff(ForcesStore * store, int i);

  // CustomCompoundBondForce getters/setters
  int cf_compound_get_num_bonds(ForcesStore * store, int i);
  void cf_compound_set_bond_parameters(ForcesStore * store, int i,
                                        int idx, int * particles, int np,
                                        double * params, int n);
  void cf_compound_get_bond_parameters(ForcesStore * store, int i,
                                        int idx, int * particles, int max_p,
                                        double * params, int max_params);

  // CustomCentroidBondForce getters/setters
  int cf_centroid_get_num_groups(ForcesStore * store, int i);
  int cf_centroid_get_num_bonds(ForcesStore * store, int i);
  void cf_centroid_set_group_parameters(ForcesStore * store, int i,
                                         int idx, int * particles, int np,
                                         double * weights, int nw);
  void cf_centroid_get_group_parameters(ForcesStore * store, int i,
                                         int idx, int * particles, int max_p,
                                         double * weights, int max_w,
                                         int * num_particles, int * num_weights);
  void cf_centroid_set_bond_parameters(ForcesStore * store, int i,
                                        int idx, int * groups, int ng,
                                        double * params, int n);
  void cf_centroid_get_bond_parameters(ForcesStore * store, int i,
                                        int idx, int * groups, int max_g,
                                        double * params, int max_params,
                                        int * num_groups, int * num_params);
  int cf_centroid_get_num_per_bond_params(ForcesStore * store, int i);
  void cf_centroid_get_per_bond_param_name(ForcesStore * store, int i,
                                            int param_idx, char * buf,
                                            int max_len);

  // CustomGBForce getters/setters
  int cf_gb_get_num_particles(ForcesStore * store, int i);
  void cf_gb_set_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params, int n);
  void cf_gb_get_particle_parameters(ForcesStore * store, int i,
                                      int idx, double * params, int max_params);

  // CustomHbondForce getters/setters
  int cf_hbond_get_num_donors(ForcesStore * store, int i);
  int cf_hbond_get_num_acceptors(ForcesStore * store, int i);
  void cf_hbond_set_donor_parameters(ForcesStore * store, int i,
                                      int idx, int d1, int d2, int d3,
                                      double * params, int n);
  void cf_hbond_get_donor_parameters(ForcesStore * store, int i,
                                      int idx, int * d1, int * d2, int * d3,
                                      double * params, int max_params);
  void cf_hbond_set_acceptor_parameters(ForcesStore * store, int i,
                                         int idx, int a1, int a2, int a3,
                                         double * params, int n);
  void cf_hbond_get_acceptor_parameters(ForcesStore * store, int i,
                                         int idx, int * a1, int * a2, int * a3,
                                         double * params, int max_params);
  int cf_hbond_get_num_per_donor_params(ForcesStore * store, int i);
  int cf_hbond_get_num_per_acceptor_params(ForcesStore * store, int i);
  void cf_hbond_get_per_donor_param_name(ForcesStore * store, int i,
                                          int param_idx, char * buf,
                                          int max_len);
  void cf_hbond_get_per_acceptor_param_name(ForcesStore * store, int i,
                                             int param_idx, char * buf,
                                             int max_len);

  // CustomManyParticleForce getters/setters/extras
  int cf_many_get_num_particles(ForcesStore * store, int i);
  void cf_many_set_particle_parameters(ForcesStore * store, int i,
                                        int idx, double * params, int n,
                                        int type);
  void cf_many_get_particle_parameters(ForcesStore * store, int i,
                                        int idx, double * params,
                                        int max_params, int * type);
  void cf_many_set_type_filter(ForcesStore * store, int i,
                                int particle_index, int * types, int n);
  void cf_many_get_type_filter(ForcesStore * store, int i,
                                int particle_index, int * types,
                                int max_types, int * num_types);
  int cf_many_get_permutation_mode(ForcesStore * store, int i);
  void cf_many_set_permutation_mode(ForcesStore * store, int i, int mode);
  int cf_many_get_num_per_particle_params(ForcesStore * store, int i);
  void cf_many_get_per_particle_param_name(ForcesStore * store, int i,
                                            int param_idx, char * buf,
                                            int max_len);

  // Tabulated functions (1D, 2D, 3D)
  int cf_add_tabulated_function_continuous1d(ForcesStore * store, int i,
      const char * name, double * values, int n,
      double min_val, double max_val, int periodic);
  int cf_add_tabulated_function_discrete1d(ForcesStore * store, int i,
      const char * name, double * values, int n);
  int cf_add_tabulated_function_continuous2d(ForcesStore * store, int i,
      const char * name, double * values, int nx, int ny,
      double xmin, double xmax, double ymin, double ymax, int periodic);
  int cf_add_tabulated_function_discrete2d(ForcesStore * store, int i,
      const char * name, double * values, int nx, int ny);
  int cf_add_tabulated_function_continuous3d(ForcesStore * store, int i,
      const char * name, double * values, int nx, int ny, int nz,
      double xmin, double xmax, double ymin, double ymax,
      double zmin, double zmax, int periodic);
  int cf_add_tabulated_function_discrete3d(ForcesStore * store, int i,
      const char * name, double * values, int nx, int ny, int nz);

  // Force groups
  void cf_set_force_group(ForcesStore * store, int i, int group);
  int cf_get_force_group(ForcesStore * store, int i);

  // Introspection
  void cf_get_energy_expression(ForcesStore * store, int i,
                                 char * buf, int max_len);
  void cf_get_global_param_name(ForcesStore * store, int i,
                                 int param_idx, char * buf, int max_len);
  int cf_get_num_per_params(ForcesStore * store, int i);
  void cf_get_per_param_name(ForcesStore * store, int i,
                              int param_idx, char * buf, int max_len);

  // updateParametersInContext
  void cf_update_parameters_in_context(ForcesStore * store, int i,
      void * context_ptr);
}
#endif  // FORCES_STORE
#endif  // KEY_OPENMM
