module fstore
#if KEY_OPENMM == 1
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_associated
  implicit none
  type(c_ptr) :: store = c_null_ptr
  logical :: is_initialized = .false.

  ! Mirror of ForcesStore::ForceType in forcesStore.h.  Kept in sync
  ! manually; the integer values are part of the C ABI of fstore_get_type.
  integer, parameter :: FT_TORCH           = 0
  integer, parameter :: FT_C_ANGLE         = 1
  integer, parameter :: FT_C_BOND          = 2
  integer, parameter :: FT_C_CV            = 3
  integer, parameter :: FT_C_CENTROID_BOND = 4
  integer, parameter :: FT_C_COMPOUND_BOND = 5
  integer, parameter :: FT_C_EXTERNAL      = 6
  integer, parameter :: FT_C_GB            = 7
  integer, parameter :: FT_C_H_BOND        = 8
  integer, parameter :: FT_C_MANY_PARTICLE = 9
  integer, parameter :: FT_C_NONBONDED     = 10
  integer, parameter :: FT_C_TORSION       = 11
#if OMM_VER >= 84
  integer, parameter :: FT_C_VOLUME        = 12
#endif
  integer, parameter :: FT_RMSD            = 13
#if OMM_VER >= 84
  integer, parameter :: FT_RG              = 14
#endif

  ! ETERM bucket codes for a per-force bucket override (fstore_set_bucket).
  ! These are part of the C ABI and MUST match EtermBucket in
  ! tool/pycharmm/pycharmm/omm.py.  A force with no override falls back to
  ! the bucket implied by its ForceType (see fstore_setup).
  integer, parameter :: FB_CFIN = 0   ! internal   (bond/angle/torsion)
  integer, parameter :: FB_CFNB = 1   ! nonbonded  (nonbonded/GB)
  integer, parameter :: FB_CFEX = 2   ! external
  integer, parameter :: FB_CFMB = 3   ! many-body  (compound/centroid/hbond/many)
  integer, parameter :: FB_CFCV = 4   ! collective variable (CV/volume/rmsd/rg)
  integer, parameter :: FB_NNPO = 5   ! neural-network potential (torch)
  integer, parameter :: FB_MAX  = 5   ! highest valid bucket code

  interface
     function fstore_create() bind(c) result(new_store)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr) :: new_store
     end function fstore_create

     function fstore_get(store, i) bind(c) result(force_i)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       type(c_ptr) :: force_i
     end function fstore_get

     function fstore_copy(store, i) bind(c) result(force_i)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       type(c_ptr) :: force_i
     end function fstore_copy

     function fstore_turn_on(store, i) bind(c) result(was_on)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: was_on
     end function fstore_turn_on

     function fstore_turn_off(store, i) bind(c) result(was_on)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: was_on
     end function fstore_turn_off

     function fstore_add(store, kind, description, n) &
          bind(c) result(new_index)
       use, intrinsic :: iso_c_binding, only: c_char, c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: kind, n
       character(kind=c_char, len=1), dimension(*) :: description
       integer(c_int) :: new_index
     end function fstore_add

     !> Register a pre-built OpenMM force by raw pointer (forcesStore.cpp).
     !!
     !! @param[in] store    the forces store
     !! @param[in] forceptr OpenMM::Force* from the same OpenMM library
     !! @param[in] kind     ForcesStore::ForceType value (see FT_* above)
     !! @return    new store index, or -1 if rejected (null pointer, or the
     !!            object is not a supported force of that kind)
     function fstore_add_ptr(store, forceptr, kind) bind(c) &
          result(new_index)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store, forceptr
       integer(c_int), value :: kind
       integer(c_int) :: new_index
     end function fstore_add_ptr

     !> Pin force i to a specific ETERM bucket (forcesStore.cpp).
     !! @param[in] store the forces store
     !! @param[in] i     store index of the force
     !! @param[in] code  bucket code (see FB_* below); negative clears it
     !! @return    0 on success, or a negative FSTORE_ERR_* code
     function fstore_set_bucket(store, i, code) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, code
       integer(c_int) :: status
     end function fstore_set_bucket

     !> Return the ETERM bucket override for force i, or -1 if none.
     !! @param[in] store the forces store
     !! @param[in] i     store index of the force
     !! @return    bucket code (see FB_* below), or -1
     function fstore_get_bucket(store, i) bind(c) result(code)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: code
     end function fstore_get_bucket

     function fstore_add_torch(store, new_force) bind(c) &
          result(new_index)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store, new_force
       integer(c_int) :: new_index
     end function fstore_add_torch

     function fstore_add_rmsd(store, ref_pos, natom, particles, &
          nparticles) bind(c) result(new_index)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: natom, nparticles
       real(c_double) :: ref_pos(3*natom)
       integer(c_int) :: particles(nparticles)
       integer(c_int) :: new_index
     end function fstore_add_rmsd

     subroutine cf_rmsd_set_reference_positions(store, i, ref_pos, &
          natom) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, natom
       real(c_double) :: ref_pos(3*natom)
     end subroutine cf_rmsd_set_reference_positions

     subroutine cf_rmsd_set_particles(store, i, particles, &
          nparticles) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, nparticles
       integer(c_int) :: particles(nparticles)
     end subroutine cf_rmsd_set_particles

#if OMM_VER >= 84
     function fstore_add_rg(store, particles, nparticles) bind(c) &
          result(new_index)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: nparticles
       integer(c_int) :: particles(nparticles)
       integer(c_int) :: new_index
     end function fstore_add_rg
#endif

     function fstore_size(store) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int) :: n
     end function fstore_size

     function fstore_is_on(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function fstore_is_on

     !> Note that force i's copy is now in the OpenMM System.
     subroutine fstore_mark_in_system(store, i) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
     end subroutine fstore_mark_in_system

     !> Whether force i's copy is already in the OpenMM System.
     function fstore_is_in_system(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function fstore_is_in_system

     !> Forget which forces are in the System, because the System is gone.
     subroutine fstore_clear_in_system(store) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: store
     end subroutine fstore_clear_in_system

     function fstore_get_type(store, i) bind(c) result(kind)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: kind
     end function fstore_get_type

     subroutine fstore_del(store) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: store
     end subroutine fstore_del

     ! ---- Generic dispatch functions (customForces.cpp) ----

     function cf_add_global_param(store, i, name, value) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double), value :: value
       integer(c_int) :: idx
     end function cf_add_global_param

     subroutine cf_set_global_param(store, i, param_index, value) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_index
       real(c_double), value :: value
     end subroutine cf_set_global_param

     function cf_get_num_global_params(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_get_num_global_params

     subroutine cf_add_energy_param_deriv(store, i, name) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
     end subroutine cf_add_energy_param_deriv

     subroutine cf_set_uses_pbc(store, i, periodic) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, periodic
     end subroutine cf_set_uses_pbc

     ! ---- CustomBondForce ----

     function cf_bond_add_per_bond_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_bond_add_per_bond_param

     function cf_bond_add_bond(store, i, p1, p2, params, n_params) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, p1, p2, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_bond_add_bond

     ! ---- CustomAngleForce ----

     function cf_angle_add_per_angle_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_angle_add_per_angle_param

     function cf_angle_add_angle(store, i, p1, p2, p3, params, n_params) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, p1, p2, p3, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_angle_add_angle

     ! ---- CustomTorsionForce ----

     function cf_torsion_add_per_torsion_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_torsion_add_per_torsion_param

     function cf_torsion_add_torsion(store, i, p1, p2, p3, p4, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, p1, p2, p3, p4, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_torsion_add_torsion

     ! ---- CustomExternalForce ----

     function cf_external_add_per_particle_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_external_add_per_particle_param

     function cf_external_add_particle(store, i, particle, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, particle, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_external_add_particle

     ! ---- CustomNonbondedForce ----

     function cf_nb_add_per_particle_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_nb_add_per_particle_param

     function cf_nb_add_particle(store, i, params, n_params) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_nb_add_particle

     function cf_nb_add_exclusion(store, i, p1, p2) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, p1, p2
       integer(c_int) :: idx
     end function cf_nb_add_exclusion

     subroutine cf_nb_set_nonbonded_method(store, i, method) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, method
     end subroutine cf_nb_set_nonbonded_method

     subroutine cf_nb_set_cutoff(store, i, cutoff) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double), value :: cutoff
     end subroutine cf_nb_set_cutoff

     subroutine cf_nb_set_use_switching_function(store, i, use_it) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, use_it
     end subroutine cf_nb_set_use_switching_function

     subroutine cf_nb_set_switching_distance(store, i, distance) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double), value :: distance
     end subroutine cf_nb_set_switching_distance

     function cf_nb_add_interaction_group(store, i, &
          set1, n1, set2, n2) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n1, n2
       integer(c_int) :: set1(*), set2(*)
       integer(c_int) :: idx
     end function cf_nb_add_interaction_group

     ! ---- CustomCompoundBondForce ----

     function cf_compound_add_per_bond_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_compound_add_per_bond_param

     function cf_compound_add_bond(store, i, particles, n_particles, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_particles, n_params
       integer(c_int) :: particles(*)
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_compound_add_bond

     ! ---- CustomCentroidBondForce ----

     function cf_centroid_add_per_bond_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_centroid_add_per_bond_param

     function cf_centroid_add_group(store, i, particles, n_particles, &
          weights, n_weights) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_particles, n_weights
       integer(c_int) :: particles(*)
       real(c_double) :: weights(*)
       integer(c_int) :: idx
     end function cf_centroid_add_group

     function cf_centroid_add_bond(store, i, groups, n_groups, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_groups, n_params
       integer(c_int) :: groups(*)
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_centroid_add_bond

     ! ---- CustomGBForce ----

     function cf_gb_add_per_particle_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_gb_add_per_particle_param

     function cf_gb_add_particle(store, i, params, n_params) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_gb_add_particle

     function cf_gb_add_computed_value(store, i, name, expression, &
          comp_type) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, comp_type
       character(kind=c_char, len=1), dimension(*) :: name, expression
       integer(c_int) :: idx
     end function cf_gb_add_computed_value

     function cf_gb_add_energy_term(store, i, expression, comp_type) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, comp_type
       character(kind=c_char, len=1), dimension(*) :: expression
       integer(c_int) :: idx
     end function cf_gb_add_energy_term

     subroutine cf_gb_set_nonbonded_method(store, i, method) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, method
     end subroutine cf_gb_set_nonbonded_method

     subroutine cf_gb_set_cutoff(store, i, cutoff) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double), value :: cutoff
     end subroutine cf_gb_set_cutoff

     ! ---- CustomHbondForce ----

     function cf_hbond_add_per_donor_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_hbond_add_per_donor_param

     function cf_hbond_add_per_acceptor_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_hbond_add_per_acceptor_param

     function cf_hbond_add_donor(store, i, d1, d2, d3, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, d1, d2, d3, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_hbond_add_donor

     function cf_hbond_add_acceptor(store, i, a1, a2, a3, &
          params, n_params) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, a1, a2, a3, n_params
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_hbond_add_acceptor

     function cf_hbond_add_exclusion(store, i, donor, acceptor) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, donor, acceptor
       integer(c_int) :: idx
     end function cf_hbond_add_exclusion

     subroutine cf_hbond_set_nonbonded_method(store, i, method) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, method
     end subroutine cf_hbond_set_nonbonded_method

     subroutine cf_hbond_set_cutoff(store, i, cutoff) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double), value :: cutoff
     end subroutine cf_hbond_set_cutoff

     ! ---- CustomManyParticleForce ----

     function cf_many_add_per_particle_param(store, i, name) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_many_add_per_particle_param

     function cf_many_add_particle(store, i, params, n_params, ptype) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n_params, ptype
       real(c_double) :: params(*)
       integer(c_int) :: idx
     end function cf_many_add_particle

     function cf_many_add_exclusion(store, i, p1, p2) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, p1, p2
       integer(c_int) :: idx
     end function cf_many_add_exclusion

     subroutine cf_many_set_nonbonded_method(store, i, method) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, method
     end subroutine cf_many_set_nonbonded_method

     subroutine cf_many_set_cutoff(store, i, cutoff) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double), value :: cutoff
     end subroutine cf_many_set_cutoff

     ! ---- CustomCVForce ----

     function cf_cv_add_collective_variable(store, cv_index, &
          cv_force_store_index, name) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: cv_index, cv_force_store_index
       character(kind=c_char, len=1), dimension(*) :: name
       integer(c_int) :: idx
     end function cf_cv_add_collective_variable

     ! ---- Generic getter ----

     function cf_get_global_param_default_value(store, i, param_idx) &
          bind(c) result(val)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx
       real(c_double) :: val
     end function cf_get_global_param_default_value

     ! ---- CustomBondForce getters/setters ----

     function cf_bond_get_num_bonds(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_bond_get_num_bonds

     subroutine cf_bond_set_bond_parameters(store, i, idx, &
          p1, p2, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, p1, p2, n
       real(c_double) :: params(*)
     end subroutine cf_bond_set_bond_parameters

     subroutine cf_bond_get_bond_parameters(store, i, idx, &
          p1, p2, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: p1, p2
       real(c_double) :: params(*)
     end subroutine cf_bond_get_bond_parameters

     subroutine cf_angle_get_angle_parameters(store, i, idx, &
          p1, p2, p3, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: p1, p2, p3
       real(c_double) :: params(*)
     end subroutine cf_angle_get_angle_parameters

     ! ---- CustomAngleForce getters/setters ----

     function cf_angle_get_num_angles(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_angle_get_num_angles

     subroutine cf_angle_set_angle_parameters(store, i, idx, &
          p1, p2, p3, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, p1, p2, p3, n
       real(c_double) :: params(*)
     end subroutine cf_angle_set_angle_parameters

     subroutine cf_torsion_get_torsion_parameters(store, i, idx, &
          p1, p2, p3, p4, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: p1, p2, p3, p4
       real(c_double) :: params(*)
     end subroutine cf_torsion_get_torsion_parameters

     ! ---- CustomTorsionForce getters/setters ----

     function cf_torsion_get_num_torsions(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_torsion_get_num_torsions

     subroutine cf_torsion_set_torsion_parameters(store, i, idx, &
          p1, p2, p3, p4, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, p1, p2, p3, p4, n
       real(c_double) :: params(*)
     end subroutine cf_torsion_set_torsion_parameters

     subroutine cf_external_get_particle_parameters(store, i, idx, &
          particle, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: particle
       real(c_double) :: params(*)
     end subroutine cf_external_get_particle_parameters

     ! ---- CustomExternalForce getters/setters ----

     function cf_external_get_num_particles(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_external_get_num_particles

     subroutine cf_external_set_particle_parameters(store, i, idx, &
          particle, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, particle, n
       real(c_double) :: params(*)
     end subroutine cf_external_set_particle_parameters

     ! ---- CustomNonbondedForce getters/setters ----

     function cf_nb_get_num_particles(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_nb_get_num_particles

     subroutine cf_nb_set_particle_parameters(store, i, idx, &
          params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, n
       real(c_double) :: params(*)
     end subroutine cf_nb_set_particle_parameters

     subroutine cf_nb_get_particle_parameters(store, i, idx, &
          params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       real(c_double) :: params(*)
     end subroutine cf_nb_get_particle_parameters

     function cf_nb_get_nonbonded_method(store, i) bind(c) result(m)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: m
     end function cf_nb_get_nonbonded_method

     function cf_nb_get_cutoff(store, i) bind(c) result(cutoff)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       real(c_double) :: cutoff
     end function cf_nb_get_cutoff

     ! ---- CustomCompoundBondForce getters/setters ----

     function cf_compound_get_num_bonds(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_compound_get_num_bonds

     subroutine cf_compound_set_bond_parameters(store, i, idx, &
          particles, np, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, np, n
       integer(c_int) :: particles(*)
       real(c_double) :: params(*)
     end subroutine cf_compound_set_bond_parameters

     subroutine cf_compound_get_bond_parameters(store, i, idx, &
          particles, max_p, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_p, max_params
       integer(c_int) :: particles(*)
       real(c_double) :: params(*)
     end subroutine cf_compound_get_bond_parameters

     ! ---- CustomCentroidBondForce getters/setters ----

     function cf_centroid_get_num_groups(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_centroid_get_num_groups

     function cf_centroid_get_num_bonds(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_centroid_get_num_bonds

     subroutine cf_centroid_set_group_parameters(store, i, idx, &
          particles, np, weights, nw) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, np, nw
       integer(c_int) :: particles(*)
       real(c_double) :: weights(*)
     end subroutine cf_centroid_set_group_parameters

     subroutine cf_centroid_get_group_parameters(store, i, idx, &
          particles, max_p, weights, max_w, &
          num_particles, num_weights) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_p, max_w
       integer(c_int) :: particles(*), num_particles, num_weights
       real(c_double) :: weights(*)
     end subroutine cf_centroid_get_group_parameters

     subroutine cf_centroid_set_bond_parameters(store, i, idx, &
          groups, ng, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, ng, n
       integer(c_int) :: groups(*)
       real(c_double) :: params(*)
     end subroutine cf_centroid_set_bond_parameters

     subroutine cf_centroid_get_bond_parameters(store, i, idx, &
          groups, max_g, params, max_params, &
          num_groups, num_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_g, max_params
       integer(c_int) :: groups(*), num_groups, num_params
       real(c_double) :: params(*)
     end subroutine cf_centroid_get_bond_parameters

     function cf_centroid_get_num_per_bond_params(store, i) &
          bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_centroid_get_num_per_bond_params

     subroutine cf_centroid_get_per_bond_param_name(store, i, &
          param_idx, buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_centroid_get_per_bond_param_name

     ! ---- CustomGBForce getters/setters ----

     function cf_gb_get_num_particles(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_gb_get_num_particles

     subroutine cf_gb_set_particle_parameters(store, i, idx, &
          params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, n
       real(c_double) :: params(*)
     end subroutine cf_gb_set_particle_parameters

     subroutine cf_gb_get_particle_parameters(store, i, idx, &
          params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       real(c_double) :: params(*)
     end subroutine cf_gb_get_particle_parameters

     ! ---- Tabulated functions ----

     function cf_add_tabulated_function_continuous1d(store, i, name, &
          values, n, min_val, max_val, periodic) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n, periodic
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       real(c_double), value :: min_val, max_val
       integer(c_int) :: idx
     end function cf_add_tabulated_function_continuous1d

     function cf_add_tabulated_function_discrete1d(store, i, name, &
          values, n) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, n
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       integer(c_int) :: idx
     end function cf_add_tabulated_function_discrete1d

     ! ---- CustomHbondForce getters/setters ----

     function cf_hbond_get_num_donors(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_hbond_get_num_donors

     function cf_hbond_get_num_acceptors(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_hbond_get_num_acceptors

     subroutine cf_hbond_set_donor_parameters(store, i, idx, &
          d1, d2, d3, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, d1, d2, d3, n
       real(c_double) :: params(*)
     end subroutine cf_hbond_set_donor_parameters

     subroutine cf_hbond_get_donor_parameters(store, i, idx, &
          d1, d2, d3, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: d1, d2, d3
       real(c_double) :: params(*)
     end subroutine cf_hbond_get_donor_parameters

     subroutine cf_hbond_set_acceptor_parameters(store, i, idx, &
          a1, a2, a3, params, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, a1, a2, a3, n
       real(c_double) :: params(*)
     end subroutine cf_hbond_set_acceptor_parameters

     subroutine cf_hbond_get_acceptor_parameters(store, i, idx, &
          a1, a2, a3, params, max_params) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       integer(c_int) :: a1, a2, a3
       real(c_double) :: params(*)
     end subroutine cf_hbond_get_acceptor_parameters

     function cf_hbond_get_num_per_donor_params(store, i) &
          bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_hbond_get_num_per_donor_params

     function cf_hbond_get_num_per_acceptor_params(store, i) &
          bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_hbond_get_num_per_acceptor_params

     subroutine cf_hbond_get_per_donor_param_name(store, i, &
          param_idx, buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_hbond_get_per_donor_param_name

     subroutine cf_hbond_get_per_acceptor_param_name(store, i, &
          param_idx, buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_hbond_get_per_acceptor_param_name

     ! ---- CustomManyParticleForce getters/setters/extras ----

     function cf_many_get_num_particles(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_many_get_num_particles

     subroutine cf_many_set_particle_parameters(store, i, idx, &
          params, n, ptype) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, n, ptype
       real(c_double) :: params(*)
     end subroutine cf_many_set_particle_parameters

     subroutine cf_many_get_particle_parameters(store, i, idx, &
          params, max_params, ptype) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, idx, max_params
       real(c_double) :: params(*)
       integer(c_int) :: ptype
     end subroutine cf_many_get_particle_parameters

     subroutine cf_many_set_type_filter(store, i, particle_index, &
          types, n) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, particle_index, n
       integer(c_int) :: types(*)
     end subroutine cf_many_set_type_filter

     subroutine cf_many_get_type_filter(store, i, particle_index, &
          types, max_types, num_types) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, particle_index, max_types
       integer(c_int) :: types(*), num_types
     end subroutine cf_many_get_type_filter

     function cf_many_get_permutation_mode(store, i) &
          bind(c) result(mode)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: mode
     end function cf_many_get_permutation_mode

     subroutine cf_many_set_permutation_mode(store, i, mode) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, mode
     end subroutine cf_many_set_permutation_mode

     function cf_many_get_num_per_particle_params(store, i) &
          bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_many_get_num_per_particle_params

     subroutine cf_many_get_per_particle_param_name(store, i, &
          param_idx, buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_many_get_per_particle_param_name

     ! ---- Tabulated functions 2D/3D ----

     function cf_add_tabulated_function_continuous2d(store, i, name, &
          values, nx, ny, xmin, xmax, ymin, ymax, periodic) &
          bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, nx, ny, periodic
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       real(c_double), value :: xmin, xmax, ymin, ymax
       integer(c_int) :: idx
     end function cf_add_tabulated_function_continuous2d

     function cf_add_tabulated_function_discrete2d(store, i, name, &
          values, nx, ny) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, nx, ny
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       integer(c_int) :: idx
     end function cf_add_tabulated_function_discrete2d

     function cf_add_tabulated_function_continuous3d(store, i, name, &
          values, nx, ny, nz, xmin, xmax, ymin, ymax, &
          zmin, zmax, periodic) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, nx, ny, nz, periodic
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       real(c_double), value :: xmin, xmax, ymin, ymax, zmin, zmax
       integer(c_int) :: idx
     end function cf_add_tabulated_function_continuous3d

     function cf_add_tabulated_function_discrete3d(store, i, name, &
          values, nx, ny, nz) bind(c) result(idx)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char, c_double
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, nx, ny, nz
       character(kind=c_char, len=1), dimension(*) :: name
       real(c_double) :: values(*)
       integer(c_int) :: idx
     end function cf_add_tabulated_function_discrete3d

     ! ---- Force groups ----

     subroutine cf_set_force_group(store, i, grp) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, grp
     end subroutine cf_set_force_group

     function cf_get_force_group(store, i) bind(c) result(grp)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: grp
     end function cf_get_force_group

     ! ---- Introspection ----

     subroutine cf_get_energy_expression(store, i, buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_get_energy_expression

     subroutine cf_get_global_param_name(store, i, param_idx, &
          buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_get_global_param_name

     function cf_get_num_per_params(store, i) bind(c) result(n)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       integer(c_int) :: n
     end function cf_get_num_per_params

     subroutine cf_get_per_param_name(store, i, param_idx, &
          buf, max_len) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_char
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i, param_idx, max_len
       character(kind=c_char, len=1) :: buf(*)
     end subroutine cf_get_per_param_name

     ! ---- updateParametersInContext ----

     subroutine cf_update_parameters_in_context(store, i, &
          context_ptr) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: store
       integer(c_int), value :: i
       type(c_ptr), value :: context_ptr
     end subroutine cf_update_parameters_in_context

  end interface

contains

  subroutine fstore_init()
    implicit none

    if (is_initialized) return

    store = fstore_create()
    is_initialized = .true.
  end subroutine fstore_init

  subroutine fstore_reset()
    use, intrinsic :: iso_c_binding, only: c_null_ptr
    implicit none

    if (.not. is_initialized) return

    call fstore_del(store)
    is_initialized = .false.
    store = c_null_ptr
    call fstore_init()
  end subroutine fstore_reset

  subroutine check_omm_status(status)
    implicit none
    integer :: status
    if (status .lt. 0) &
       call wrndie(-5, '<fstore>', 'OpenMM error detected')
  end subroutine check_omm_status

  !> Add every enabled stored force to the OpenMM system being built.
  !!
  !! For each active force this assigns an ETERM bucket -- taken from the
  !! per-force override set by fstore_set_bucket if one is present, else from
  !! the force's ForceType -- maps that bucket to an OpenMM force group, and
  !! adds a copy of the force to @p system.
  !!
  !! @param[inout] system the OpenMM system under construction
  subroutine fstore_setup(system)
    use openmm, only: &
         openmm_force, openmm_force_setforcegroup, &
         openmm_system, openmm_system_addforce
    use omm_ecomp, only: omm_incr_eterms
    use stream, only: outu, prnlev
    implicit none
    type(openmm_system) :: system
    type(openmm_force) :: force
    type(c_ptr) :: force_i
    integer :: i, group, n, is_on, status, kind, bkt
    character(len=8) :: bucket_term

    if (.not. is_initialized) return

    n = fstore_size(store)
    do i = 0, n - 1
       is_on = fstore_is_on(store, i)
       ! Already in the System: adding it again would count its energy twice.
       ! Skipping instead is what makes a second pass additive, so a force
       ! added after the System was built costs one addForce rather than a
       ! teardown and rebuild of everything.
       if (is_on .eq. 1 .and. fstore_is_in_system(store, i) .eq. 1) cycle
       if (is_on .eq. 1) then
          force_i = fstore_copy(store, i)
          ! A null copy means the store could not duplicate this force (e.g.
          ! an unsupported/unknown ForceType).  Adding a null force to the
          ! system would crash OpenMM, so skip it and tell the user.
          if (.not. c_associated(force_i)) then
             if (prnlev >= 2) write(outu,'(a,i0,a)') &
                  'CHARMM> fstore_setup: could not copy stored force ', i, &
                  '; skipping it.  Its energy will NOT be included.'
             cycle
          end if
          force = transfer(force_i, openmm_force(0))

          ! A per-force override (fstore_set_bucket) wins over the
          ! ForceType-derived default, so a caller can direct any force's
          ! energy into the bucket of their choice.
          bkt = fstore_get_bucket(store, i)
          bucket_term = ' '
          if (bkt >= 0) then
             select case (bkt)
             case (FB_CFIN); bucket_term = 'cfint'
             case (FB_CFNB); bucket_term = 'cfnon'
             case (FB_CFEX); bucket_term = 'cfext'
             case (FB_CFMB); bucket_term = 'cfmny'
             case (FB_CFCV); bucket_term = 'cfcv'
             case (FB_NNPO)
#if KEY_OMMTORCH == 1
                bucket_term = 'nnpo'
#else
                ! The NNPO energy term only exists in builds with OpenMM-Torch
                ! support.  Without it omm_incr_eterms('nnpo') would fall
                ! through to force group 0 and the energy would be reported
                ! as EVDW, which is worse than refusing the override.
                if (prnlev >= 2) write(outu,'(a,i0,a)') &
                     'CHARMM> fstore_setup: NNPO bucket requested for ' // &
                     'force ', i, ' but this CHARMM was built without ' // &
                     'OpenMM-Torch support.  Using the default bucket ' // &
                     'for this force type instead.  Rebuild with ' // &
                     '--with-torch to use NNPO.'
                bucket_term = ' '
#endif
             case default
                ! Out-of-range override: warn and fall back to the default
                ! bucket for this force's type rather than guessing.
                if (prnlev >= 2) write(outu,'(a,i0,a,i0,a)') &
                     'CHARMM> fstore_setup: bucket override ', bkt, &
                     ' for force ', i, &
                     ' is not a valid bucket code (0-5); using the ' // &
                     'default bucket for this force type instead.'
                bucket_term = ' '
             end select
          end if
          if (bucket_term /= ' ') then
             group = omm_incr_eterms(bucket_term)
             call OpenMM_Force_setForceGroup(force, group)
             status = OpenMM_System_addForce(system, force)
             call check_omm_status(status)
             call fstore_mark_in_system(store, i)
             cycle
          end if

          ! Map ForcesStore::ForceType to a CHARMM ETERM bucket.  Torch
          ! keeps its own dedicated NNPO slot; user-added OpenMM custom
          ! forces are summed into one of five buckets, sharing one
          ! OpenMM force-group bit per active bucket.
          kind = fstore_get_type(store, i)
          select case (kind)
          case (FT_TORCH)
             bucket_term = 'nnpo'
          case (FT_C_BOND, FT_C_ANGLE, FT_C_TORSION)
             bucket_term = 'cfint'
          case (FT_C_NONBONDED, FT_C_GB)
             bucket_term = 'cfnon'
          case (FT_C_EXTERNAL)
             bucket_term = 'cfext'
          case (FT_C_COMPOUND_BOND, FT_C_CENTROID_BOND, &
                FT_C_H_BOND, FT_C_MANY_PARTICLE)
             bucket_term = 'cfmny'
#if OMM_VER >= 84
          case (FT_C_CV, FT_C_VOLUME, FT_RMSD, FT_RG)
#else
          case (FT_C_CV, FT_RMSD)
#endif
             bucket_term = 'cfcv'
          case default
             ! Unknown kind: fall back to NNPO so the energy is still
             ! accounted for somewhere, with a warning.
             if (prnlev >= 2) write(outu,'(a,i0,a)') &
                  'CHARMM> fstore_setup: unknown ForceType ', kind, &
                  ', falling back to NNPO bucket'
             bucket_term = 'nnpo'
          end select
          group = omm_incr_eterms(bucket_term)
          call OpenMM_Force_setForceGroup(force, group)
          status = OpenMM_System_addForce(system, force)
          call check_omm_status(status)
          call fstore_mark_in_system(store, i)
       end if
    end do
  end subroutine fstore_setup
#endif  /* KEY_OPENMM */
end module fstore
