module api_omm
  implicit none
contains

  !> Return 1 if this CHARMM build supports OpenMM-Torch (KEY_OMMTORCH=1),
  !> 0 otherwise.  Python callers must check this before invoking any
  !> api_torch_* function — the no-OMMTORCH stubs call wrndie which
  !> terminates the process via _gfortran_exit, bypassing pytest's
  !> error handling and producing a silent crash.
  function api_has_ommtorch() bind(c) result(supported)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: supported
#if KEY_OMMTORCH == 1
    supported = 1
#else
    supported = 0
#endif
  end function api_has_ommtorch

  !> Return the OpenMM version this CHARMM was built against, encoded
  !> as major*10 + minor (e.g. OpenMM 8.2 -> 82, 8.4 -> 84).  Python
  !> callers use this to gate features that require a specific OpenMM
  !> version, raising NotImplementedError with a clear message instead
  !> of hitting a missing-symbol AttributeError or a generic
  !> "Failed to create" RuntimeError.
  function api_omm_version() bind(c) result(ver)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: ver
#if KEY_OPENMM == 1
    ver = OMM_VER
#else
    ver = 0
#endif
  end function api_omm_version


  !> @brief get the size of the seralized system in characters
  !
  !  This function is called so that storage can be allocated
  !  on the python side before getting the serialized system
  !
  !> @return integer(c_int) number of characters in the serialized system
  function omm_get_system_serial_size() bind(c) result(n)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
#if KEY_OPENMM == 1
    use omm_main, only: serialize_system
#endif /* KEY_OPENMM == 1 */
    implicit none
    integer(c_int) :: n
    character(kind=c_char, len=1), allocatable, dimension(:) :: sys
#if KEY_OPENMM == 1
    call serialize_system(sys)
    n = size(sys)
#else
    n = 0
#endif /* KEY_OPENMM == 1 */
  end function omm_get_system_serial_size

  !> @brief get a serialized system in a format that OpenMM can read
  !
  !  call omm_get_system_serial_size to preallocate storage for out_serial
  !  before calling this subroutine and pass the result as max_size
  !
  !> @param[out] out_serial preallocated char array to hold serialization
  !> @param[in] max_size number of chars that can be safely stored in out_serial
  subroutine omm_get_system_serial(out_serial, max_size) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
#if KEY_OPENMM == 1
    use omm_main, only: serialize_system
#endif /* KEY_OPENMM == 1 */
    implicit none

    character(kind=c_char, len=1) :: out_serial(*)
    integer(kind=c_int), value :: max_size
    integer :: current_size, min_size
    character(kind=c_char, len=1), allocatable, dimension(:) :: sys

    current_size = 0
    min_size = 0

#if KEY_OPENMM == 1
    call serialize_system(sys)
    current_size = size(sys)
    min_size = min(max_size, current_size)
    out_serial(1:min_size) = sys(1:min_size)
#endif /* KEY_OPENMM == 1 */
  end subroutine omm_get_system_serial

  subroutine warn_no_torch()
    call wrndie(-1,'omm_api', &
         'openmm torch integration not present in this charmm build')
  end subroutine warn_no_torch

  subroutine check_torch(status)
    implicit none
    integer :: status

    if (status .ne. 0) then
       call wrndie(-5, '<omm_torch>', &
            'OpenMMTorch threw an error,' // &
            ' check output carefully for OpenMM style errors / warnings')
    end if
  end subroutine check_torch

#if KEY_OMMTORCH == 1

  !> @brief add a torch force to CHARMM's OpenMM system
  !
  ! The module filename should be the full path to
  ! a torch module file on disk.
  !
  !> @param[in] torch_module_fn full path to an on disk torch module file
  !> @param[in] fn_len length in characters of the torch module filename
  function api_torch_add_force(torch_module_fn) &
       bind(c) result(new_index)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_ptr
    use omm_torch, only: torch_create
    use fstore, only: store, fstore_add_torch
    use omm_main, only: omm_invalidate

    implicit none

    character(kind=c_char, len=1) :: torch_module_fn(*)
    type(c_ptr) :: new_force
    integer(c_int) :: new_index

    new_force = torch_create(torch_module_fn)
    new_index = fstore_add_torch(store, new_force)
    ! Adding a new force changes the system; the next `energy omm`
    ! must rebuild the cached context.
    call omm_invalidate()
  end function api_torch_add_force

  subroutine api_torch_outputs_forces(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use omm_torch, only: torch_set_outputs_forces
    use fstore, only: store, fstore_get
    use omm_main, only: omm_invalidate
    implicit none

    integer(c_int), value :: i
    type(c_ptr) :: force

    force = fstore_get(store, i)
    call torch_set_outputs_forces(force)
    call omm_invalidate()
  end subroutine api_torch_outputs_forces

  subroutine api_torch_uses_periodic(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use omm_torch, only: torch_set_uses_pbc
    use fstore, only: store, fstore_get
    use omm_main, only: omm_invalidate
    implicit none

    integer(c_int), value :: i
    type(c_ptr) :: force

    force = fstore_get(store, i)
    call torch_set_uses_pbc(force)
    call omm_invalidate()
  end subroutine api_torch_uses_periodic

  function api_torch_add_global_param(i, name, val) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double, c_ptr
    use fstore, only: store, fstore_get
    use omm_torch, only: torch_add_global_param
    use omm_main, only: omm_invalidate

    implicit none

    integer(kind=c_int), value :: i
    character(kind=c_char, len=1) :: name(*)
    real(c_double), value :: val

    type(c_ptr) :: force_i
    integer(c_int) :: new_index

    force_i = fstore_get(store, i)
    new_index = torch_add_global_param(force_i, name, val)
    ! A new global parameter (or value change) only takes effect on the
    ! next OpenMM context build; mark the system dirty so the running
    ! context is rebuilt before the next `energy omm` evaluation.
    call omm_invalidate()
  end function api_torch_add_global_param

  subroutine api_torch_set_global_param(force_index, param_index, val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    use fstore, only: store, fstore_get
    use omm_torch, only: torch_set_global_param
    use omm_main, only: omm_invalidate

    implicit none

    integer(kind=c_int), value :: force_index, param_index
    real(c_double), value :: val

    type(c_ptr) :: force

    force = fstore_get(store, force_index)
    call torch_set_global_param(force, param_index, val)
    ! See note in api_torch_add_global_param.
    call omm_invalidate()
  end subroutine api_torch_set_global_param

#else /* KEY_OMMTORCH == 1 */

  function api_torch_add_force(torch_module_fn) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    character(kind=c_char, len=1) :: torch_module_fn(*)
    integer(kind=c_int) :: new_index
    new_index = -1
    call warn_no_torch()
  end function api_torch_add_force

  subroutine api_torch_outputs_forces(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    call warn_no_torch()
  end subroutine api_torch_outputs_forces

  subroutine api_torch_uses_periodic(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    call warn_no_torch()
  end subroutine api_torch_uses_periodic

  function api_torch_add_global_param(i, name, val) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double
    implicit none
    integer(kind=c_int), value :: i
    character(kind=c_char, len=1) :: name(*)
    real(c_double), value :: val
    integer(kind=c_int) :: new_index
    new_index = -1
    call warn_no_torch()
  end function api_torch_add_global_param

  subroutine api_torch_set_global_param(force_index, param_index, val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    implicit none
    integer(kind=c_int), value :: force_index, param_index
    real(c_double), value :: val
    call warn_no_torch()
  end subroutine api_torch_set_global_param

#endif /* KEY_OMMTORCH == 1 */

  subroutine warn_no_omm()
    call wrndie(-1,'omm_api', &
         'openmm integration not present in this charmm build')
  end subroutine warn_no_omm

#if KEY_OPENMM == 1
  !> @brief Mark CHARMM's cached OpenMM system as stale.
  !!
  !! CHARMM copies each stored force into the OpenMM system when it builds
  !! that system, then reuses the built system for later energy and dynamics
  !! commands.  A force created or altered *after* that build is therefore
  !! invisible: the energy silently keeps its old value, with no warning and
  !! no error.  Every entry point that creates or alters a stored force calls
  !! this so the system is rebuilt before the next evaluation, which is what
  !! api_custom_force_add_ptr and the OpenMM-Torch entry points already did.
  !!
  !! Read-only entry points do not call it.  Two mutators deliberately do not
  !! either:
  !!   - api_cf_update_parameters_in_context pushes parameters straight into
  !!     the live context on purpose; rebuilding would throw that away and
  !!     defeat the call.
  !!   - api_cf_set_force_group has no lasting effect, because CHARMM assigns
  !!     force groups itself while building the system, so a rebuild would
  !!     change nothing.
  !!
  !! @see omm_main::omm_invalidate
  subroutine cf_system_changed()
    use omm_main, only: omm_invalidate

    implicit none

    call omm_invalidate()
  end subroutine cf_system_changed

  !> @brief Mark the cached OpenMM system as merely having gained forces.
  !!
  !! Weaker than cf_system_changed, and better where it applies: a force that
  !! has only been *added* can be put into the System already built, leaving
  !! the positions, velocities and time alone.  Anything that alters a force
  !! already copied into the System still needs cf_system_changed, because
  !! CHARMM cannot reach into that copy.
  !!
  !! @see omm_main::omm_invalidate_forces_added
  subroutine cf_forces_added()
    use omm_main, only: omm_invalidate_forces_added

    implicit none

    call omm_invalidate_forces_added()
  end subroutine cf_forces_added

  !> @brief Tell CHARMM a stored OpenMM force has been changed from outside.
  !!
  !! The entry points in this file mark the system stale themselves, so a
  !! caller that goes through them never needs this.  It exists for the one
  !! case they cannot cover: a force handed over by pointer
  !! (api_custom_force_add_ptr) and then modified through the caller's own
  !! reference to the object.  CHARMM has no way to notice that, so the
  !! energy would silently keep the value it had before the change.
  !!
  !! Safe to call at any time, and cheap: it sets a flag, and the rebuild it
  !! asks for happens at the next energy or dynamics command.  Calling it
  !! when nothing changed costs one needless rebuild.
  !!
  !! @see omm_main::omm_invalidate
  subroutine api_omm_invalidate() bind(c)
    implicit none

    call cf_system_changed()
  end subroutine api_omm_invalidate

  !> @brief Drive an OpenMM Context created by the caller.
  !!
  !! Lets a script choose the platform and its properties, bring its own
  !! integrator, or run a force that has to be driven from Python.  CHARMM
  !! pushes positions, velocities, time and box vectors in and reads energies
  !! and forces back, as it does with its own Context, and never destroys it.
  !!
  !! @param[in] ctxptr an OpenMM::Context* from the SAME OpenMM library CHARMM
  !!                   is linked against
  !! @return    0 on success; 1 if no System was supplied first; 2 if that
  !!            System has not been filled in yet, so a Context on it would be
  !!            empty; 3 if the Context was built on a different System, in
  !!            which case it is missing every force CHARMM added and its
  !!            energies would be quietly wrong
  function api_omm_set_external_context(ctxptr) result(status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int
    use omm_main, only: omm_adopt_context

    implicit none

    type(c_ptr), value :: ctxptr
    integer(c_int) :: status

    status = omm_adopt_context(ctxptr)
  end function api_omm_set_external_context

  !> @brief Stop driving a Context supplied by the caller.
  subroutine api_omm_release_external_context() bind(c)
    use omm_main, only: omm_release_context

    implicit none

    call omm_release_context()
  end subroutine api_omm_release_external_context

  !> @brief Whether CHARMM is driving a Context it does not own.
  !! @return 1 if so, 0 otherwise.
  function api_omm_external_context_state() result(state) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use omm_main, only: omm_context_is_supplied

    implicit none

    integer(c_int) :: state

    state = omm_context_is_supplied()
  end function api_omm_external_context_state

  !> @brief Fill in an OpenMM System supplied by the caller.
  !!
  !! For a script that wants to create the OpenMM Context itself.  The Context
  !! has to be built on the System CHARMM actually evaluates, and a System
  !! cannot be handed outward from CHARMM afterwards -- a raw
  !! OpenMM::System pointer cannot be turned back into a working object on the
  !! Python side.  So the caller creates the System, keeps its own reference,
  !! and passes the pointer here; CHARMM fills it in, and the caller can then
  !! build a Context on the very object CHARMM will use.
  !!
  !! CHARMM never destroys a System supplied this way.  It is filled in once:
  !! a later rebuild is refused rather than adding every particle and force a
  !! second time.
  !!
  !! @param[in] sysptr an OpenMM::System* from the SAME OpenMM library CHARMM
  !!                   is linked against.  A pointer from a different build,
  !!                   or one that is not a System, cannot be detected here
  !!                   and will crash when the System is used.
  subroutine api_omm_set_external_system(sysptr) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use omm_main, only: omm_adopt_system, omm_invalidate

    implicit none

    type(c_ptr), value :: sysptr

    call omm_adopt_system(sysptr)
    ! Any system already built is CHARMM's own and no longer the one to use.
    call omm_invalidate()
  end subroutine api_omm_set_external_system

  !> @brief Stop using a System supplied by the caller.
  !!
  !! Returns CHARMM to building its own.  The supplied System is not destroyed;
  !! it still belongs to whoever made it.
  subroutine api_omm_release_external_system() bind(c)
    use omm_main, only: omm_release_system, omm_invalidate

    implicit none

    call omm_release_system()
    call omm_invalidate()
  end subroutine api_omm_release_external_system

  !> @brief Report the state of an externally supplied System.
  !!
  !! @return 0 when CHARMM is using its own System, 1 when a supplied System
  !!         is in use and not yet built, 2 when a supplied System has been
  !!         built and so cannot be built again.  Lets the pyCHARMM layer turn
  !!         the refusal into an ordinary Python error instead of leaving
  !!         CHARMM to end the run.
  function api_omm_external_system_state() result(state) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use omm_main, only: omm_system_is_supplied

    implicit none

    integer(c_int) :: state

    state = omm_system_is_supplied()
  end function api_omm_external_system_state

  !> @brief How many forces the store holds.
  !!
  !! Counts every force ever added and not cleared, switched on or off alike,
  !! so it is the right bound for a loop over store indices.  Indices are
  !! never reused within a store's lifetime, so 0 .. n-1 covers all of them.
  !!
  !! @return the number of stored forces; 0 if the store does not exist yet
  function api_cf_get_num_forces() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_size

    implicit none

    integer(c_int) :: n

    n = fstore_size(store)
  end function api_cf_get_num_forces

  !> @brief The ForceType of a stored force.
  !!
  !! @param[in] i store index of the force
  !! @return    the ForcesStore::ForceType value, or -1 if there is no force
  !!            at @p i.  The C side turns an out-of-range index into -1
  !!            rather than letting it throw, so a bad index is reportable
  !!            instead of fatal.
  function api_cf_get_kind(i) result(kind) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_get_type

    implicit none

    integer(c_int), value :: i
    integer(c_int) :: kind

    kind = fstore_get_type(store, i)
  end function api_cf_get_kind

  !> @brief Whether a stored force is switched on.
  !!
  !! Only forces that are on are copied into the OpenMM system, so this is
  !! what decides whether a stored force contributes to the energy.
  !!
  !! @param[in] i store index of the force
  !! @return    1 if the force is on, 0 if it is off.  Also 0 for an index
  !!            that does not exist, so pair it with api_cf_get_num_forces or
  !!            api_cf_get_kind if the difference matters.
  function api_cf_is_enabled(i) result(on) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_is_on

    implicit none

    integer(c_int), value :: i
    integer(c_int) :: on

    on = fstore_is_on(store, i)
  end function api_cf_is_enabled

  !> @brief add a customnb force to CHARMM's OpenMM system
  !
  ! The description should be a valid spec for an OpenMM Customnonbondedforce
  !
  !> @param[in] description a string describing an OpenMM CustomNonbondedforce
  !> @param[in] desc_len length in characters of the customnb description
  function api_custom_force_add(kind, description, n) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, fstore_add

    implicit none

    integer(c_int), value :: kind, n
    character(kind=c_char, len=1), dimension(*) :: description

    integer(c_int) :: new_index

    new_index = fstore_add(store, kind, description, n)
    ! A force was added, which is the one change that does not need the System
    ! rebuilt: it can be added to the System as it stands.  Guarded on success,
    ! since on failure nothing was stored.
    if (new_index >= 0) call cf_forces_added()
  end function api_custom_force_add

  !> Register an OpenMM force that was built outside CHARMM, by raw pointer.
  !!
  !! Lets a caller (pyCHARMM) construct a force with the full OpenMM API on
  !! its own side and hand the finished object to CHARMM, instead of
  !! rebuilding it here type by type.  CHARMM takes ownership of the object,
  !! so the caller must relinquish its own ownership first -- from Python,
  !! set `force.thisown = False` (only after this call succeeds).
  !!
  !! @param[in] forceptr an OpenMM::Force* obtained from the SAME OpenMM
  !!                     library CHARMM is linked against
  !! @param[in] kind     the force's ForcesStore::ForceType value
  !! @return    the new force's store index, or -1 if the force was
  !!            rejected (null pointer, or not a supported force of that
  !!            kind in this build); on -1 nothing was stored and the
  !!            caller still owns the object
  !!
  !! @warning Passing a pointer from a different OpenMM build, a pointer
  !!          that is not an OpenMM force, or a stale/freed pointer cannot
  !!          be fully detected and will crash when the force is evaluated.
  !!          Callers must ensure Python and CHARMM share one OpenMM install.
  function api_custom_force_add_ptr(forceptr, kind) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int
    use fstore, only: store, fstore_add_ptr
    use omm_main, only: omm_invalidate_forces_added
    use stream, only: outu, prnlev

    implicit none

    type(c_ptr), value :: forceptr
    integer(c_int), value :: kind

    integer(c_int) :: new_index

    new_index = fstore_add_ptr(store, forceptr, kind)

    ! A system already built does not contain this force.  Adding a force is
    ! the one change that can be applied to the system as it stands, so ask
    ! for that rather than a rebuild; without either, the force would be
    ! silently ignored when added after an energy or dynamics command.
    if (new_index >= 0) call omm_invalidate_forces_added()

    ! Explain any rejection in CHARMM's own output, with the remedy, rather
    ! than leaving the caller with a bare negative number.
    if (new_index < 0 .and. prnlev >= 2) then
       select case (new_index)
       case (-1)
          write(outu,'(a)') &
               'CHARMM> add OpenMM force: the OpenMM force store does ' // &
               'not exist yet.  Turn OpenMM on (OMM ON, or omm.enable() ' // &
               'in pyCHARMM) before adding forces.'
       case (-2)
          write(outu,'(a)') &
               'CHARMM> add OpenMM force: the force pointer was null.  ' // &
               'The force object was not created, or was already freed.'
       case (-3)
          write(outu,'(a,i0,a)') &
               'CHARMM> add OpenMM force: the object is not a supported ' // &
               'OpenMM force of kind ', kind, &
               '.  Either the kind does not match the force class, or ' // &
               'this CHARMM was built without support for that force ' // &
               'type (e.g. OpenMM-Torch, or a newer OpenMM).'
       case default
          write(outu,'(a,i0)') &
               'CHARMM> add OpenMM force failed with status ', new_index
       end select
       write(outu,'(a)') &
            'CHARMM>   The force was NOT added and its energy will not ' // &
            'be included.'
    end if
  end function api_custom_force_add_ptr

  !> Pin a stored force's energy to a specific CHARMM ETERM bucket.
  !!
  !! Overrides the bucket that would otherwise be chosen from the force's
  !! type, so a caller can decide which energy term a force reports into.
  !! Takes effect the next time the OpenMM system is built.
  !!
  !! @param[in] i    store index of the force
  !! @param[in] code bucket code (see the FB_* parameters in fstore.F90);
  !!                 pass a negative value to clear the override and restore
  !!                 the default for the force type
  !! @return    0 on success, or a negative FSTORE_ERR_* status
  function api_cf_set_bucket(i, code) result(status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_set_bucket, FB_MAX
    use omm_main, only: omm_invalidate
    use stream, only: outu, prnlev

    implicit none

    integer(c_int), value :: i, code
    integer(c_int) :: status

    status = fstore_set_bucket(store, i, code)

    ! The energy term is assigned while the OpenMM system is built, so an
    ! already-built system still has the old term.  Force a rebuild, or the
    ! change would be silently ignored after an energy command has run.
    if (status >= 0) call omm_invalidate()

    if (status < 0 .and. prnlev >= 2) then
       select case (status)
       case (-1)
          write(outu,'(a)') &
               'CHARMM> set energy bucket: the OpenMM force store does ' // &
               'not exist yet.  Turn OpenMM on before adding forces.'
       case (-4)
          write(outu,'(a,i0,a,i0,a)') &
               'CHARMM> set energy bucket: ', code, &
               ' is not a valid bucket code (valid range 0 to ', FB_MAX, ').'
       case (-5)
          write(outu,'(a,i0,a)') &
               'CHARMM> set energy bucket: there is no stored force with ' // &
               'index ', i, '.  Add the force first, then set its bucket.'
       case default
          write(outu,'(a,i0)') &
               'CHARMM> set energy bucket failed with status ', status
       end select
       write(outu,'(a)') &
            'CHARMM>   The bucket was NOT changed; this force keeps the ' // &
            'default energy term for its type.'
    end if
  end function api_cf_set_bucket

  !> Return the ETERM bucket override for a stored force, or -1 if none.
  !! @param[in] i store index of the force
  !! @return    the bucket code set by api_cf_set_bucket, or -1
  function api_cf_get_bucket(i) result(code) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_get_bucket

    implicit none

    integer(c_int), value :: i
    integer(c_int) :: code

    code = fstore_get_bucket(store, i)
  end function api_cf_get_bucket

  function api_fstore_add_rmsd(ref_pos, natom, particles, nparticles) &
       result(new_index) bind(c, name='api_fstore_add_rmsd')
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use fstore, only: store, fstore_add_rmsd
    implicit none
    integer(c_int), value :: natom, nparticles
    real(c_double) :: ref_pos(3*natom)
    integer(c_int) :: particles(nparticles)
    integer(c_int) :: new_index
    new_index = fstore_add_rmsd(store, ref_pos, natom, particles, nparticles)
    ! A force was added, so any system already built lacks it and is stale.
    if (new_index >= 0) call cf_system_changed()
  end function api_fstore_add_rmsd

  subroutine api_cf_rmsd_set_reference_positions(i, ref_pos, natom) &
       bind(c, name='api_cf_rmsd_set_reference_positions')
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use fstore, only: store, cf_rmsd_set_reference_positions
    implicit none
    integer(c_int), value :: i, natom
    real(c_double) :: ref_pos(3*natom)
    call cf_rmsd_set_reference_positions(store, i, ref_pos, natom)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_rmsd_set_reference_positions

  subroutine api_cf_rmsd_set_particles(i, particles, nparticles) &
       bind(c, name='api_cf_rmsd_set_particles')
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_rmsd_set_particles
    implicit none
    integer(c_int), value :: i, nparticles
    integer(c_int) :: particles(nparticles)
    call cf_rmsd_set_particles(store, i, particles, nparticles)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_rmsd_set_particles

  ! Always provide the symbol so Python ctypes lookup succeeds even on
  ! OpenMM < 8.4; the body returns -1 when the underlying
  ! fstore_add_rg helper is unavailable.  Python wrappers call
  ! api_omm_version() first and raise NotImplementedError with a clear
  ! message, so the -1 return is a safety net rather than the primary
  ! error path.
  function api_fstore_add_rg(particles, nparticles) &
       result(new_index) bind(c, name='api_fstore_add_rg')
    use, intrinsic :: iso_c_binding, only: c_int
#if OMM_VER >= 84
    use fstore, only: store, fstore_add_rg
#endif
    implicit none
    integer(c_int), value :: nparticles
    integer(c_int) :: particles(nparticles)
    integer(c_int) :: new_index
#if OMM_VER >= 84
    new_index = fstore_add_rg(store, particles, nparticles)
#else
    new_index = -1
#endif
    ! A force was added, so any system already built lacks it and is stale.
    ! The guard also covers the OpenMM-too-old branch above, which stores
    ! nothing and returns -1.
    if (new_index >= 0) call cf_system_changed()
  end function api_fstore_add_rg

  function api_customnb_add_particle(i, params, n_params) result(new_index) &
                  bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int
    use fstore, only: store, fstore_get
    use openmm, only: openmm_customnonbondedforce_addparticle
    use openmm_types, only: openmm_customnonbondedforce, openmm_doublearray
    use omm_util, only: omm_param_set

    implicit none

    integer(c_int), value :: i, n_params
    real(c_double) :: params(*)
    type(c_ptr) :: force_i
    integer(c_int) :: new_index
    type(openmm_doublearray) :: omm_params

    call omm_param_set(omm_params, params(1:n_params))

    force_i = fstore_get(store, i)
    new_index = openmm_customnonbondedforce_addparticle( &
         transfer(force_i, openmm_customnonbondedforce(0)), &
         omm_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_customnb_add_particle

  function api_customnb_add_particle_param(i, name) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_addPerParticleParameter
    use openmm_types, only: OpenMM_CustomNonbondedForce
    use api_util, only: c2f_string

    implicit none

    character(kind=c_char) :: name(*)
    integer(c_int), value :: i
    type(c_ptr) :: force_i
    integer(c_int) :: new_index
    character(len=256) :: f_name

    f_name = c2f_string(name, 256)
    force_i = fstore_get(store, i)
    new_index = OpenMM_CustomNonbondedForce_addPerParticleParameter( &
         transfer(force_i, OpenMM_CustomNonbondedForce(0)), &
         trim(f_name))
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_customnb_add_particle_param

  function api_customnb_add_exclusion(i, particle1, particle2) &
       result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_addExclusion
    use openmm_types, only: OpenMM_CustomNonbondedForce

    implicit none

    integer(c_int), value :: i, particle1, particle2
    integer(c_int) :: new_index
    type(c_ptr) :: force_i

    force_i = fstore_get(store, i)
    new_index = OpenMM_CustomNonbondedForce_addExclusion( &
         transfer(force_i, OpenMM_CustomNonbondedForce(0)), &
         particle1, particle2)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_customnb_add_exclusion

  subroutine api_customnb_change_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_setNonbondedMethod
    use openmm_types, only: OpenMM_CustomNonbondedForce

    implicit none

    integer(c_int), value :: i, method
    type(c_ptr) :: force_i

    force_i = fstore_get(store, i)
    call OpenMM_CustomNonbondedForce_setNonbondedMethod( &
         transfer(force_i, OpenMM_CustomNonbondedForce(0)), &
         method)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_customnb_change_nonbonded_method

  subroutine api_customnb_change_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_double
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_setCutoffDistance
    use openmm_types, only: OpenMM_CustomNonbondedForce

    implicit none

    integer(c_int), value :: i
    real(c_double), value :: cutoff
    type(c_ptr) :: force_i

    force_i = fstore_get(store, i)
    call OpenMM_CustomNonbondedForce_setCutoffDistance( &
         transfer(force_i, OpenMM_CustomNonbondedForce(0)), &
         cutoff)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_customnb_change_cutoff

  function api_customnb_add_global_param(i, name, val) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double, c_ptr
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_addGlobalParameter
    use openmm_types, only: OpenMM_CustomNonbondedForce
    use api_util, only: c2f_string

    implicit none

    integer(kind=c_int), value :: i
    character(kind=c_char) :: name(*)
    real(c_double), value :: val

    type(c_ptr) :: force_i
    integer(c_int) :: new_index
    character(len=256) :: f_name

    f_name = c2f_string(name, 256)
    force_i = fstore_get(store, i)
    new_index = OpenMM_CustomNonbondedForce_addGlobalParameter( &
         transfer(force_i, OpenMM_CustomNonbondedForce(0)), &
         trim(f_name), val)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_customnb_add_global_param

  subroutine api_customnb_set_global_param(force_index, param_index, val) &
       bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    use fstore, only: store, fstore_get
    use openmm, only: OpenMM_CustomNonbondedForce_getGlobalParameterDefaultValue
    use openmm_types, only: OpenMM_CustomNonbondedForce

    implicit none

    integer(kind=c_int), value :: force_index, param_index
    real(c_double), value :: val

    type(c_ptr) :: force

    force = fstore_get(store, force_index)
    call OpenMM_CustomNonbondedForce_setGlobalParameterDefaultValue( &
         transfer(force, OpenMM_CustomNonbondedForce(0)), &
         param_index, val)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_customnb_set_global_param

  function api_force_turn_on(index) result(old_status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_turn_on

    implicit none

    integer(kind=c_int), value :: index
    integer(kind=c_int) :: old_status

    old_status = fstore_turn_on(store, index)
    ! Enabling a force changes which forces the system contains, so a system
    ! already built is stale.  old_status is the PREVIOUS state, and it also
    ! reads 0 for an index that does not exist, so a bad index asks for a
    ! rebuild it does not need.  That costs time; missing a real rebuild
    ! would silently drop the force from the energy, which is worse.
    if (old_status == 0) call cf_system_changed()
  end function api_force_turn_on

  function api_force_turn_off(index) result(old_status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, fstore_turn_off

    implicit none

    integer(kind=c_int), value :: index

    integer(kind=c_int) :: old_status

    old_status = fstore_turn_off(store, index)
    ! Disabling a force changes which forces the system contains, so a system
    ! already built is stale.  old_status is the PREVIOUS state, so 1 means
    ! the force really was on and has now been switched off -- exactly the
    ! case that needs a rebuild.  Without this, a force the user disabled
    ! keeps contributing to the energy.
    if (old_status == 1) call cf_system_changed()
  end function api_force_turn_off

  ! ---- New cf_* API wrappers ----

  function api_cf_add_global_param(i, name, val) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double
    use fstore, only: store, cf_add_global_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double), value :: val
    integer(c_int) :: idx
    idx = cf_add_global_param(store, i, name, val)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_global_param

  subroutine api_cf_set_global_param(i, param_index, val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_set_global_param
    implicit none
    integer(c_int), value :: i, param_index
    real(c_double), value :: val
    call cf_set_global_param(store, i, param_index, val)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_set_global_param

  function api_cf_get_num_global_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_get_num_global_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_get_num_global_params(store, i)
  end function api_cf_get_num_global_params

  subroutine api_cf_add_energy_param_deriv(i, name) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_add_energy_param_deriv
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    call cf_add_energy_param_deriv(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_add_energy_param_deriv

  subroutine api_cf_set_uses_pbc(i, periodic) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_set_uses_pbc
    implicit none
    integer(c_int), value :: i, periodic
    call cf_set_uses_pbc(store, i, periodic)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_set_uses_pbc

  ! ---- CustomBondForce ----

  function api_cf_bond_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_bond_add_per_bond_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_bond_add_per_bond_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_bond_add_per_bond_param

  function api_cf_bond_add_bond(i, p1, p2, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_bond_add_bond
    implicit none
    integer(c_int), value :: i, p1, p2, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_bond_add_bond(store, i, p1, p2, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_bond_add_bond

  ! ---- CustomAngleForce ----

  function api_cf_angle_add_per_angle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_angle_add_per_angle_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_angle_add_per_angle_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_angle_add_per_angle_param

  function api_cf_angle_add_angle(i, p1, p2, p3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_angle_add_angle
    implicit none
    integer(c_int), value :: i, p1, p2, p3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_angle_add_angle(store, i, p1, p2, p3, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_angle_add_angle

  ! ---- CustomTorsionForce ----

  function api_cf_torsion_add_per_torsion_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_torsion_add_per_torsion_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_torsion_add_per_torsion_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_torsion_add_per_torsion_param

  function api_cf_torsion_add_torsion(i, p1, p2, p3, p4, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_torsion_add_torsion
    implicit none
    integer(c_int), value :: i, p1, p2, p3, p4, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_torsion_add_torsion(store, i, p1, p2, p3, p4, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_torsion_add_torsion

  ! ---- CustomExternalForce ----

  function api_cf_external_add_per_particle_param(i, name) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_external_add_per_particle_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_external_add_per_particle_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_external_add_per_particle_param

  function api_cf_external_add_particle(i, particle, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_external_add_particle
    implicit none
    integer(c_int), value :: i, particle, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_external_add_particle(store, i, particle, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_external_add_particle

  ! ---- CustomNonbondedForce ----

  function api_cf_nb_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_nb_add_per_particle_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_nb_add_per_particle_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_nb_add_per_particle_param

  function api_cf_nb_add_particle(i, params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_add_particle
    implicit none
    integer(c_int), value :: i, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_nb_add_particle(store, i, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_nb_add_particle

  function api_cf_nb_add_exclusion(i, p1, p2) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_add_exclusion
    implicit none
    integer(c_int), value :: i, p1, p2
    integer(c_int) :: idx
    idx = cf_nb_add_exclusion(store, i, p1, p2)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_nb_add_exclusion

  subroutine api_cf_nb_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_set_nonbonded_method
    implicit none
    integer(c_int), value :: i, method
    call cf_nb_set_nonbonded_method(store, i, method)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_nb_set_nonbonded_method

  subroutine api_cf_nb_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_set_cutoff
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call cf_nb_set_cutoff(store, i, cutoff)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_nb_set_cutoff

  subroutine api_cf_nb_set_use_switching_function(i, use_it) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_set_use_switching_function
    implicit none
    integer(c_int), value :: i, use_it
    call cf_nb_set_use_switching_function(store, i, use_it)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_nb_set_use_switching_function

  subroutine api_cf_nb_set_switching_distance(i, distance) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_set_switching_distance
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: distance
    call cf_nb_set_switching_distance(store, i, distance)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_nb_set_switching_distance

  function api_cf_nb_add_interaction_group(i, set1, n1, set2, n2) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_add_interaction_group
    implicit none
    integer(c_int), value :: i, n1, n2
    integer(c_int) :: set1(*), set2(*)
    integer(c_int) :: idx
    idx = cf_nb_add_interaction_group(store, i, set1, n1, set2, n2)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_nb_add_interaction_group

  ! ---- CustomCompoundBondForce ----

  function api_cf_compound_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_compound_add_per_bond_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_compound_add_per_bond_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_compound_add_per_bond_param

  function api_cf_compound_add_bond(i, particles, n_particles, &
       params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_compound_add_bond
    implicit none
    integer(c_int), value :: i, n_particles, n_params
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_compound_add_bond(store, i, particles, n_particles, &
         params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_compound_add_bond

  ! ---- CustomCentroidBondForce ----

  function api_cf_centroid_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_centroid_add_per_bond_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_centroid_add_per_bond_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_centroid_add_per_bond_param

  function api_cf_centroid_add_group(i, particles, n_particles, &
       weights, n_weights) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_add_group
    implicit none
    integer(c_int), value :: i, n_particles, n_weights
    integer(c_int) :: particles(*)
    real(c_double) :: weights(*)
    integer(c_int) :: idx
    idx = cf_centroid_add_group(store, i, particles, n_particles, &
         weights, n_weights)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_centroid_add_group

  function api_cf_centroid_add_bond(i, groups, n_groups, &
       params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_add_bond
    implicit none
    integer(c_int), value :: i, n_groups, n_params
    integer(c_int) :: groups(*)
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_centroid_add_bond(store, i, groups, n_groups, &
         params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_centroid_add_bond

  ! ---- CustomGBForce ----

  function api_cf_gb_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_gb_add_per_particle_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_gb_add_per_particle_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_gb_add_per_particle_param

  function api_cf_gb_add_particle(i, params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_gb_add_particle
    implicit none
    integer(c_int), value :: i, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_gb_add_particle(store, i, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_gb_add_particle

  function api_cf_gb_add_computed_value(i, name, expression, comp_type) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_gb_add_computed_value
    implicit none
    integer(c_int), value :: i, comp_type
    character(kind=c_char, len=1), dimension(*) :: name, expression
    integer(c_int) :: idx
    idx = cf_gb_add_computed_value(store, i, name, expression, comp_type)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_gb_add_computed_value

  function api_cf_gb_add_energy_term(i, expression, comp_type) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_gb_add_energy_term
    implicit none
    integer(c_int), value :: i, comp_type
    character(kind=c_char, len=1), dimension(*) :: expression
    integer(c_int) :: idx
    idx = cf_gb_add_energy_term(store, i, expression, comp_type)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_gb_add_energy_term

  subroutine api_cf_gb_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_gb_set_nonbonded_method
    implicit none
    integer(c_int), value :: i, method
    call cf_gb_set_nonbonded_method(store, i, method)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_gb_set_nonbonded_method

  subroutine api_cf_gb_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_gb_set_cutoff
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call cf_gb_set_cutoff(store, i, cutoff)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_gb_set_cutoff

  ! ---- CustomHbondForce ----

  function api_cf_hbond_add_per_donor_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_hbond_add_per_donor_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_hbond_add_per_donor_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_hbond_add_per_donor_param

  function api_cf_hbond_add_per_acceptor_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_hbond_add_per_acceptor_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_hbond_add_per_acceptor_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_hbond_add_per_acceptor_param

  function api_cf_hbond_add_donor(i, d1, d2, d3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_add_donor
    implicit none
    integer(c_int), value :: i, d1, d2, d3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_hbond_add_donor(store, i, d1, d2, d3, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_hbond_add_donor

  function api_cf_hbond_add_acceptor(i, a1, a2, a3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_add_acceptor
    implicit none
    integer(c_int), value :: i, a1, a2, a3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_hbond_add_acceptor(store, i, a1, a2, a3, params, n_params)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_hbond_add_acceptor

  function api_cf_hbond_add_exclusion(i, donor, acceptor) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_add_exclusion
    implicit none
    integer(c_int), value :: i, donor, acceptor
    integer(c_int) :: idx
    idx = cf_hbond_add_exclusion(store, i, donor, acceptor)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_hbond_add_exclusion

  subroutine api_cf_hbond_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_set_nonbonded_method
    implicit none
    integer(c_int), value :: i, method
    call cf_hbond_set_nonbonded_method(store, i, method)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_hbond_set_nonbonded_method

  subroutine api_cf_hbond_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_set_cutoff
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call cf_hbond_set_cutoff(store, i, cutoff)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_hbond_set_cutoff

  ! ---- CustomManyParticleForce ----

  function api_cf_many_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_many_add_per_particle_param
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_many_add_per_particle_param(store, i, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_many_add_per_particle_param

  function api_cf_many_add_particle(i, params, n_params, ptype) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_many_add_particle
    implicit none
    integer(c_int), value :: i, n_params, ptype
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = cf_many_add_particle(store, i, params, n_params, ptype)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_many_add_particle

  function api_cf_many_add_exclusion(i, p1, p2) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_add_exclusion
    implicit none
    integer(c_int), value :: i, p1, p2
    integer(c_int) :: idx
    idx = cf_many_add_exclusion(store, i, p1, p2)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_many_add_exclusion

  subroutine api_cf_many_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_set_nonbonded_method
    implicit none
    integer(c_int), value :: i, method
    call cf_many_set_nonbonded_method(store, i, method)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_many_set_nonbonded_method

  subroutine api_cf_many_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_many_set_cutoff
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call cf_many_set_cutoff(store, i, cutoff)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_many_set_cutoff

  ! ---- CustomCVForce ----

  function api_cf_cv_add_collective_variable(cv_index, &
       cv_force_store_index, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use fstore, only: store, cf_cv_add_collective_variable
    implicit none
    integer(c_int), value :: cv_index, cv_force_store_index
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = cf_cv_add_collective_variable(store, cv_index, &
         cv_force_store_index, name)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_cv_add_collective_variable

  ! ---- Generic getter ----

  function api_cf_get_global_param_default_value(i, param_idx) &
       result(val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_get_global_param_default_value
    implicit none
    integer(c_int), value :: i, param_idx
    real(c_double) :: val
    val = cf_get_global_param_default_value(store, i, param_idx)
  end function api_cf_get_global_param_default_value

  ! ---- CustomBondForce getters/setters ----

  function api_cf_bond_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_bond_get_num_bonds
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_bond_get_num_bonds(store, i)
  end function api_cf_bond_get_num_bonds

  subroutine api_cf_bond_set_bond_parameters(i, idx, p1, p2, &
       params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_bond_set_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, p1, p2, n
    real(c_double) :: params(*)
    call cf_bond_set_bond_parameters(store, i, idx, p1, p2, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_bond_set_bond_parameters

  subroutine api_cf_bond_get_bond_parameters(i, idx, p1, p2, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_bond_get_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2
    real(c_double) :: params(*)
    call cf_bond_get_bond_parameters(store, i, idx, p1, p2, &
         params, max_params)
  end subroutine api_cf_bond_get_bond_parameters

  ! ---- CustomAngleForce getters/setters ----

  function api_cf_angle_get_num_angles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_angle_get_num_angles
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_angle_get_num_angles(store, i)
  end function api_cf_angle_get_num_angles

  subroutine api_cf_angle_set_angle_parameters(i, idx, p1, p2, p3, &
       params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_angle_set_angle_parameters
    implicit none
    integer(c_int), value :: i, idx, p1, p2, p3, n
    real(c_double) :: params(*)
    call cf_angle_set_angle_parameters(store, i, idx, p1, p2, p3, &
         params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_angle_set_angle_parameters

  ! ---- CustomTorsionForce getters/setters ----

  function api_cf_torsion_get_num_torsions(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_torsion_get_num_torsions
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_torsion_get_num_torsions(store, i)
  end function api_cf_torsion_get_num_torsions

  subroutine api_cf_torsion_set_torsion_parameters(i, idx, &
       p1, p2, p3, p4, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_torsion_set_torsion_parameters
    implicit none
    integer(c_int), value :: i, idx, p1, p2, p3, p4, n
    real(c_double) :: params(*)
    call cf_torsion_set_torsion_parameters(store, i, idx, &
         p1, p2, p3, p4, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_torsion_set_torsion_parameters

  ! ---- CustomExternalForce getters/setters ----

  function api_cf_external_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_external_get_num_particles
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_external_get_num_particles(store, i)
  end function api_cf_external_get_num_particles

  subroutine api_cf_external_set_particle_parameters(i, idx, &
       particle, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_external_set_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, particle, n
    real(c_double) :: params(*)
    call cf_external_set_particle_parameters(store, i, idx, &
         particle, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_external_set_particle_parameters

  ! ---- CustomNonbondedForce getters/setters ----

  function api_cf_nb_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_get_num_particles
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_nb_get_num_particles(store, i)
  end function api_cf_nb_get_num_particles

  subroutine api_cf_nb_set_particle_parameters(i, idx, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_set_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, n
    real(c_double) :: params(*)
    call cf_nb_set_particle_parameters(store, i, idx, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_nb_set_particle_parameters

  function api_cf_nb_get_nonbonded_method(i) result(m) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_nb_get_nonbonded_method
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: m
    m = cf_nb_get_nonbonded_method(store, i)
  end function api_cf_nb_get_nonbonded_method

  function api_cf_nb_get_cutoff(i) result(cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_get_cutoff
    implicit none
    integer(c_int), value :: i
    real(c_double) :: cutoff
    cutoff = cf_nb_get_cutoff(store, i)
  end function api_cf_nb_get_cutoff

  ! ---- CustomCompoundBondForce getters/setters ----

  function api_cf_compound_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_compound_get_num_bonds
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_compound_get_num_bonds(store, i)
  end function api_cf_compound_get_num_bonds

  subroutine api_cf_compound_set_bond_parameters(i, idx, &
       particles, np, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_compound_set_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, np, n
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    call cf_compound_set_bond_parameters(store, i, idx, &
         particles, np, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_compound_set_bond_parameters

  ! ---- CustomCentroidBondForce getters ----

  function api_cf_centroid_get_num_groups(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_centroid_get_num_groups
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_centroid_get_num_groups(store, i)
  end function api_cf_centroid_get_num_groups

  function api_cf_centroid_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_centroid_get_num_bonds
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_centroid_get_num_bonds(store, i)
  end function api_cf_centroid_get_num_bonds

  ! ---- CustomGBForce getters/setters ----

  function api_cf_gb_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_gb_get_num_particles
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_gb_get_num_particles(store, i)
  end function api_cf_gb_get_num_particles

  subroutine api_cf_gb_set_particle_parameters(i, idx, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_gb_set_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, n
    real(c_double) :: params(*)
    call cf_gb_set_particle_parameters(store, i, idx, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_gb_set_particle_parameters

  ! ---- New getters (Item 1) ----

  subroutine api_cf_angle_get_angle_parameters(i, idx, p1, p2, p3, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_angle_get_angle_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2, p3
    real(c_double) :: params(*)
    call cf_angle_get_angle_parameters(store, i, idx, p1, p2, p3, &
         params, max_params)
  end subroutine api_cf_angle_get_angle_parameters

  subroutine api_cf_torsion_get_torsion_parameters(i, idx, &
       p1, p2, p3, p4, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_torsion_get_torsion_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2, p3, p4
    real(c_double) :: params(*)
    call cf_torsion_get_torsion_parameters(store, i, idx, &
         p1, p2, p3, p4, params, max_params)
  end subroutine api_cf_torsion_get_torsion_parameters

  subroutine api_cf_external_get_particle_parameters(i, idx, &
       particle, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_external_get_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: particle
    real(c_double) :: params(*)
    call cf_external_get_particle_parameters(store, i, idx, &
         particle, params, max_params)
  end subroutine api_cf_external_get_particle_parameters

  subroutine api_cf_nb_get_particle_parameters(i, idx, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_nb_get_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    call cf_nb_get_particle_parameters(store, i, idx, params, max_params)
  end subroutine api_cf_nb_get_particle_parameters

  subroutine api_cf_compound_get_bond_parameters(i, idx, &
       particles, max_p, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_compound_get_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, max_p, max_params
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    call cf_compound_get_bond_parameters(store, i, idx, &
         particles, max_p, params, max_params)
  end subroutine api_cf_compound_get_bond_parameters

  subroutine api_cf_gb_get_particle_parameters(i, idx, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_gb_get_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    call cf_gb_get_particle_parameters(store, i, idx, params, max_params)
  end subroutine api_cf_gb_get_particle_parameters

  ! ---- Tabulated functions (Item 2) ----

  function api_cf_add_tabulated_function_continuous1d(i, name, &
       values, n, min_val, max_val, periodic) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_continuous1d
    implicit none
    integer(c_int), value :: i, n, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: min_val, max_val
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_continuous1d(store, i, name, &
         values, n, min_val, max_val, periodic)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_continuous1d

  function api_cf_add_tabulated_function_discrete1d(i, name, &
       values, n) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_discrete1d
    implicit none
    integer(c_int), value :: i, n
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_discrete1d(store, i, name, &
         values, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_discrete1d

  ! ---- CustomHbondForce getters/setters ----

  function api_cf_hbond_get_num_donors(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_get_num_donors
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_hbond_get_num_donors(store, i)
  end function api_cf_hbond_get_num_donors

  function api_cf_hbond_get_num_acceptors(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_get_num_acceptors
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_hbond_get_num_acceptors(store, i)
  end function api_cf_hbond_get_num_acceptors

  subroutine api_cf_hbond_set_donor_parameters(i, idx, &
       d1, d2, d3, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_set_donor_parameters
    implicit none
    integer(c_int), value :: i, idx, d1, d2, d3, n
    real(c_double) :: params(*)
    call cf_hbond_set_donor_parameters(store, i, idx, d1, d2, d3, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_hbond_set_donor_parameters

  subroutine api_cf_hbond_get_donor_parameters(i, idx, &
       d1, d2, d3, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_get_donor_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: d1, d2, d3
    real(c_double) :: params(*)
    call cf_hbond_get_donor_parameters(store, i, idx, d1, d2, d3, &
         params, max_params)
  end subroutine api_cf_hbond_get_donor_parameters

  subroutine api_cf_hbond_set_acceptor_parameters(i, idx, &
       a1, a2, a3, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_set_acceptor_parameters
    implicit none
    integer(c_int), value :: i, idx, a1, a2, a3, n
    real(c_double) :: params(*)
    call cf_hbond_set_acceptor_parameters(store, i, idx, a1, a2, a3, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_hbond_set_acceptor_parameters

  subroutine api_cf_hbond_get_acceptor_parameters(i, idx, &
       a1, a2, a3, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_hbond_get_acceptor_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: a1, a2, a3
    real(c_double) :: params(*)
    call cf_hbond_get_acceptor_parameters(store, i, idx, a1, a2, a3, &
         params, max_params)
  end subroutine api_cf_hbond_get_acceptor_parameters

  function api_cf_hbond_get_num_per_donor_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_get_num_per_donor_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_hbond_get_num_per_donor_params(store, i)
  end function api_cf_hbond_get_num_per_donor_params

  function api_cf_hbond_get_num_per_acceptor_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_hbond_get_num_per_acceptor_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_hbond_get_num_per_acceptor_params(store, i)
  end function api_cf_hbond_get_num_per_acceptor_params

  subroutine api_cf_hbond_get_per_donor_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_hbond_get_per_donor_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_hbond_get_per_donor_param_name(store, i, param_idx, buf, max_len)
  end subroutine api_cf_hbond_get_per_donor_param_name

  subroutine api_cf_hbond_get_per_acceptor_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_hbond_get_per_acceptor_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_hbond_get_per_acceptor_param_name(store, i, param_idx, &
         buf, max_len)
  end subroutine api_cf_hbond_get_per_acceptor_param_name

  ! ---- CustomManyParticleForce getters/setters/extras ----

  function api_cf_many_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_get_num_particles
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_many_get_num_particles(store, i)
  end function api_cf_many_get_num_particles

  subroutine api_cf_many_set_particle_parameters(i, idx, &
       params, n, ptype) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_many_set_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, n, ptype
    real(c_double) :: params(*)
    call cf_many_set_particle_parameters(store, i, idx, params, n, ptype)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_many_set_particle_parameters

  subroutine api_cf_many_get_particle_parameters(i, idx, &
       params, max_params, ptype) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_many_get_particle_parameters
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    integer(c_int) :: ptype
    call cf_many_get_particle_parameters(store, i, idx, params, &
         max_params, ptype)
  end subroutine api_cf_many_get_particle_parameters

  subroutine api_cf_many_set_type_filter(i, particle_index, &
       types, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_set_type_filter
    implicit none
    integer(c_int), value :: i, particle_index, n
    integer(c_int) :: types(*)
    call cf_many_set_type_filter(store, i, particle_index, types, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_many_set_type_filter

  subroutine api_cf_many_get_type_filter(i, particle_index, &
       types, max_types, num_types) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_get_type_filter
    implicit none
    integer(c_int), value :: i, particle_index, max_types
    integer(c_int) :: types(*), num_types
    call cf_many_get_type_filter(store, i, particle_index, types, &
         max_types, num_types)
  end subroutine api_cf_many_get_type_filter

  function api_cf_many_get_permutation_mode(i) result(mode) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_get_permutation_mode
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: mode
    mode = cf_many_get_permutation_mode(store, i)
  end function api_cf_many_get_permutation_mode

  subroutine api_cf_many_set_permutation_mode(i, mode) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_set_permutation_mode
    implicit none
    integer(c_int), value :: i, mode
    call cf_many_set_permutation_mode(store, i, mode)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_many_set_permutation_mode

  function api_cf_many_get_num_per_particle_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_many_get_num_per_particle_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_many_get_num_per_particle_params(store, i)
  end function api_cf_many_get_num_per_particle_params

  subroutine api_cf_many_get_per_particle_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_many_get_per_particle_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_many_get_per_particle_param_name(store, i, param_idx, &
         buf, max_len)
  end subroutine api_cf_many_get_per_particle_param_name

  ! ---- CustomCentroidBondForce setters/getters ----

  subroutine api_cf_centroid_set_group_parameters(i, idx, &
       particles, np, weights, nw) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_set_group_parameters
    implicit none
    integer(c_int), value :: i, idx, np, nw
    integer(c_int) :: particles(*)
    real(c_double) :: weights(*)
    call cf_centroid_set_group_parameters(store, i, idx, particles, np, &
         weights, nw)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_centroid_set_group_parameters

  subroutine api_cf_centroid_get_group_parameters(i, idx, &
       particles, max_p, weights, max_w, &
       num_particles, num_weights) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_get_group_parameters
    implicit none
    integer(c_int), value :: i, idx, max_p, max_w
    integer(c_int) :: particles(*), num_particles, num_weights
    real(c_double) :: weights(*)
    call cf_centroid_get_group_parameters(store, i, idx, particles, max_p, &
         weights, max_w, num_particles, num_weights)
  end subroutine api_cf_centroid_get_group_parameters

  subroutine api_cf_centroid_set_bond_parameters(i, idx, &
       groups, ng, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_set_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, ng, n
    integer(c_int) :: groups(*)
    real(c_double) :: params(*)
    call cf_centroid_set_bond_parameters(store, i, idx, groups, ng, params, n)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end subroutine api_cf_centroid_set_bond_parameters

  subroutine api_cf_centroid_get_bond_parameters(i, idx, &
       groups, max_g, params, max_params, &
       num_groups, num_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    use fstore, only: store, cf_centroid_get_bond_parameters
    implicit none
    integer(c_int), value :: i, idx, max_g, max_params
    integer(c_int) :: groups(*), num_groups, num_params
    real(c_double) :: params(*)
    call cf_centroid_get_bond_parameters(store, i, idx, groups, max_g, &
         params, max_params, num_groups, num_params)
  end subroutine api_cf_centroid_get_bond_parameters

  function api_cf_centroid_get_num_per_bond_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_centroid_get_num_per_bond_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_centroid_get_num_per_bond_params(store, i)
  end function api_cf_centroid_get_num_per_bond_params

  subroutine api_cf_centroid_get_per_bond_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_centroid_get_per_bond_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_centroid_get_per_bond_param_name(store, i, param_idx, &
         buf, max_len)
  end subroutine api_cf_centroid_get_per_bond_param_name

  ! ---- Tabulated functions 2D/3D ----

  function api_cf_add_tabulated_function_continuous2d(i, name, &
       values, nx, ny, xmin, xmax, ymin, ymax, periodic) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_continuous2d
    implicit none
    integer(c_int), value :: i, nx, ny, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: xmin, xmax, ymin, ymax
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_continuous2d(store, i, name, &
         values, nx, ny, xmin, xmax, ymin, ymax, periodic)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_continuous2d

  function api_cf_add_tabulated_function_discrete2d(i, name, &
       values, nx, ny) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_discrete2d
    implicit none
    integer(c_int), value :: i, nx, ny
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_discrete2d(store, i, name, &
         values, nx, ny)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_discrete2d

  function api_cf_add_tabulated_function_continuous3d(i, name, &
       values, nx, ny, nz, xmin, xmax, ymin, ymax, &
       zmin, zmax, periodic) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_continuous3d
    implicit none
    integer(c_int), value :: i, nx, ny, nz, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: xmin, xmax, ymin, ymax, zmin, zmax
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_continuous3d(store, i, name, &
         values, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, periodic)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_continuous3d

  function api_cf_add_tabulated_function_discrete3d(i, name, &
       values, nx, ny, nz) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    use fstore, only: store, cf_add_tabulated_function_discrete3d
    implicit none
    integer(c_int), value :: i, nx, ny, nz
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = cf_add_tabulated_function_discrete3d(store, i, name, &
         values, nx, ny, nz)
    ! The stored force changed, so the built OpenMM system is stale.
    call cf_system_changed()
  end function api_cf_add_tabulated_function_discrete3d

  ! ---- Force groups ----

  subroutine api_cf_set_force_group(i, grp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_set_force_group
    implicit none
    integer(c_int), value :: i, grp
    call cf_set_force_group(store, i, grp)
  end subroutine api_cf_set_force_group

  function api_cf_get_force_group(i) result(grp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_get_force_group
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: grp
    grp = cf_get_force_group(store, i)
  end function api_cf_get_force_group

  ! ---- Introspection ----

  subroutine api_cf_get_energy_expression(i, buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_get_energy_expression
    implicit none
    integer(c_int), value :: i, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_get_energy_expression(store, i, buf, max_len)
  end subroutine api_cf_get_energy_expression

  subroutine api_cf_get_global_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_get_global_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_get_global_param_name(store, i, param_idx, buf, max_len)
  end subroutine api_cf_get_global_param_name

  function api_cf_get_num_per_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use fstore, only: store, cf_get_num_per_params
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = cf_get_num_per_params(store, i)
  end function api_cf_get_num_per_params

  subroutine api_cf_get_per_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    use fstore, only: store, cf_get_per_param_name
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call cf_get_per_param_name(store, i, param_idx, buf, max_len)
  end subroutine api_cf_get_per_param_name

  ! ---- updateParametersInContext (Item 3) ----

  subroutine api_cf_update_parameters_in_context(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use fstore, only: store, cf_update_parameters_in_context
    use omm_main, only: omm_get_context_ptr
    implicit none
    integer(c_int), value :: i
    type(c_ptr) :: ctx_ptr
    call omm_get_context_ptr(ctx_ptr)
    call cf_update_parameters_in_context(store, i, ctx_ptr)
  end subroutine api_cf_update_parameters_in_context

#else /* KEY_OPENMM */

  !> Stub for api_omm_set_external_context in builds without OpenMM.
  !! @param[in] ctxptr ignored; warn_no_omm ends the run first.
  !! @return    nominally 1, but warn_no_omm ends the run first.
  function api_omm_set_external_context(ctxptr) result(status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int
    implicit none
    type(c_ptr), value :: ctxptr
    integer(c_int) :: status
    status = 1; call warn_no_omm()
  end function api_omm_set_external_context

  !> Stub for api_omm_release_external_context in builds without OpenMM.
  subroutine api_omm_release_external_context() bind(c)
    implicit none
    call warn_no_omm()
  end subroutine api_omm_release_external_context

  !> Stub for api_omm_external_context_state in builds without OpenMM.
  !! @return nominally 0, but warn_no_omm ends the run first.
  function api_omm_external_context_state() result(state) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: state
    state = 0; call warn_no_omm()
  end function api_omm_external_context_state

  !> Stub for api_omm_set_external_system in builds without OpenMM.
  !! @param[in] sysptr ignored; warn_no_omm ends the run first.
  subroutine api_omm_set_external_system(sysptr) bind(c)
    use, intrinsic :: iso_c_binding, only: c_ptr
    implicit none
    type(c_ptr), value :: sysptr
    call warn_no_omm()
  end subroutine api_omm_set_external_system

  !> Stub for api_omm_release_external_system in builds without OpenMM.
  subroutine api_omm_release_external_system() bind(c)
    implicit none
    call warn_no_omm()
  end subroutine api_omm_release_external_system

  !> Stub for api_omm_external_system_state in builds without OpenMM.
  !! @return nominally 0, but warn_no_omm ends the run first.
  function api_omm_external_system_state() result(state) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: state
    state = 0; call warn_no_omm()
  end function api_omm_external_system_state

  !> Stub for api_cf_get_num_forces in builds without OpenMM.
  !! @return nominally 0 -- there is no store -- but warn_no_omm ends the run
  !!         first.  The Python layer checks the build before calling.
  function api_cf_get_num_forces() result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: n
    n = 0; call warn_no_omm()
  end function api_cf_get_num_forces

  !> Stub for api_cf_get_kind in builds without OpenMM.
  !! @param[in] i ignored
  !! @return    nominally -1, the same "no such force" code the real one
  !!            uses, but warn_no_omm ends the run first.
  function api_cf_get_kind(i) result(kind) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: kind
    kind = -1; call warn_no_omm()
  end function api_cf_get_kind

  !> Stub for api_cf_is_enabled in builds without OpenMM.
  !! @param[in] i ignored
  !! @return    nominally 0, but warn_no_omm ends the run first.
  function api_cf_is_enabled(i) result(on) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: on
    on = 0; call warn_no_omm()
  end function api_cf_is_enabled

  !> Stub for api_omm_invalidate in builds without OpenMM.
  !!
  !! There is no OpenMM system to mark stale, so there would be nothing to do
  !! even if this returned.  warn_no_omm terminates the run first; the Python
  !! layer checks the build before calling, so a caller sees an exception
  !! rather than reaching this.
  subroutine api_omm_invalidate() bind(c)
    implicit none
    call warn_no_omm()
  end subroutine api_omm_invalidate

  function api_customnb_add_force(description) &
       bind(c) result(new_index)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    character(kind=c_char, len=1) :: description(*)
    integer(c_int) :: new_index
    new_index = -1
    call warn_no_omm()
  end function api_customnb_add_force

  function api_customnb_add_particle(force_index, params, n_params) &
       result(new_index)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none
    integer(c_int), value :: force_index, n_params
    real(c_double) :: params(*)
    integer(c_int) :: new_index
    new_index = -1
    call warn_no_omm()
  end function api_customnb_add_particle

  function api_customnb_add_particle_param(force_index, name) &
       result(new_index)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    character(kind=c_char, len=1) :: name(*)
    integer(c_int), value :: force_index
    integer(c_int) :: new_index
    new_index = -1
    call warn_no_omm()
  end function api_customnb_add_particle_param

  function api_customnb_add_exclusion(force_index, particle1, particle2) &
       result(new_index)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: force_index, particle1, particle2
    integer(c_int) :: new_index
    new_index = -1
    call warn_no_omm()
  end function api_customnb_add_exclusion

  function api_customnb_change_method(force_index, new_method) &
       result(old_method)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: force_index, new_method
    integer(c_int) :: old_method
    old_method = -1
    call warn_no_omm()
  end function api_customnb_change_method

  function api_customnb_change_cutoff(force_index, new_cutoff) &
       result(old_cutoff)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: force_index
    real(c_double), value :: new_cutoff
    real(c_double) :: old_cutoff
    old_cutoff = -1.0
    call warn_no_omm()
  end function api_customnb_change_cutoff

  function api_customnb_add_global_param(i, name, val) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double
    implicit none
    integer(kind=c_int), value :: i
    character(kind=c_char, len=1) :: name(*)
    real(c_double), value :: val
    integer(kind=c_int) :: new_index
    new_index = -1
    call warn_no_omm()
  end function api_customnb_add_global_param

  subroutine api_customnb_set_global_param(force_index, param_index, val) &
       bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    implicit none
    integer(kind=c_int), value :: force_index, param_index
    real(c_double), value :: val
    call warn_no_omm()
  end subroutine api_customnb_set_global_param

  function api_force_turn_on(index) result(old_status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(kind=c_int), value :: index
    logical :: old_status
    old_status = .false.
    call warn_no_omm()
  end function api_force_turn_on

  function api_force_turn_off(index) result(old_status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(kind=c_int), value :: index
    logical :: old_status
    old_status = .false.
    call warn_no_omm()
  end function api_force_turn_off

  ! ---- cf_* stubs (no OpenMM) ----

  function api_cf_add_global_param(i, name, val) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_double
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double), value :: val
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_global_param

  subroutine api_cf_set_global_param(i, param_index, val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, param_index
    real(c_double), value :: val
    call warn_no_omm()
  end subroutine api_cf_set_global_param

  function api_cf_get_num_global_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_get_num_global_params

  subroutine api_cf_add_energy_param_deriv(i, name) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    call warn_no_omm()
  end subroutine api_cf_add_energy_param_deriv

  subroutine api_cf_set_uses_pbc(i, periodic) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, periodic
    call warn_no_omm()
  end subroutine api_cf_set_uses_pbc

  function api_cf_bond_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_bond_add_per_bond_param

  function api_cf_bond_add_bond(i, p1, p2, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, p1, p2, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_bond_add_bond

  function api_cf_angle_add_per_angle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_angle_add_per_angle_param

  function api_cf_angle_add_angle(i, p1, p2, p3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, p1, p2, p3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_angle_add_angle

  function api_cf_torsion_add_per_torsion_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_torsion_add_per_torsion_param

  function api_cf_torsion_add_torsion(i, p1, p2, p3, p4, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, p1, p2, p3, p4, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_torsion_add_torsion

  function api_cf_external_add_per_particle_param(i, name) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_external_add_per_particle_param

  function api_cf_external_add_particle(i, particle, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, particle, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_external_add_particle

  function api_cf_nb_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_nb_add_per_particle_param

  function api_cf_nb_add_particle(i, params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_nb_add_particle

  function api_cf_nb_add_exclusion(i, p1, p2) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, p1, p2
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_nb_add_exclusion

  subroutine api_cf_nb_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, method
    call warn_no_omm()
  end subroutine api_cf_nb_set_nonbonded_method

  subroutine api_cf_nb_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call warn_no_omm()
  end subroutine api_cf_nb_set_cutoff

  subroutine api_cf_nb_set_use_switching_function(i, use_it) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, use_it
    call warn_no_omm()
  end subroutine api_cf_nb_set_use_switching_function

  subroutine api_cf_nb_set_switching_distance(i, distance) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: distance
    call warn_no_omm()
  end subroutine api_cf_nb_set_switching_distance

  function api_cf_nb_add_interaction_group(i, set1, n1, set2, n2) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, n1, n2
    integer(c_int) :: set1(*), set2(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_nb_add_interaction_group

  function api_cf_compound_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_compound_add_per_bond_param

  function api_cf_compound_add_bond(i, particles, n_particles, &
       params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_particles, n_params
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_compound_add_bond

  function api_cf_centroid_add_per_bond_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_centroid_add_per_bond_param

  function api_cf_centroid_add_group(i, particles, n_particles, &
       weights, n_weights) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_particles, n_weights
    integer(c_int) :: particles(*)
    real(c_double) :: weights(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_centroid_add_group

  function api_cf_centroid_add_bond(i, groups, n_groups, &
       params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_groups, n_params
    integer(c_int) :: groups(*)
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_centroid_add_bond

  function api_cf_gb_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_gb_add_per_particle_param

  function api_cf_gb_add_particle(i, params, n_params) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_gb_add_particle

  function api_cf_gb_add_computed_value(i, name, expression, comp_type) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i, comp_type
    character(kind=c_char, len=1), dimension(*) :: name, expression
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_gb_add_computed_value

  function api_cf_gb_add_energy_term(i, expression, comp_type) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i, comp_type
    character(kind=c_char, len=1), dimension(*) :: expression
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_gb_add_energy_term

  subroutine api_cf_gb_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, method
    call warn_no_omm()
  end subroutine api_cf_gb_set_nonbonded_method

  subroutine api_cf_gb_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call warn_no_omm()
  end subroutine api_cf_gb_set_cutoff

  function api_cf_hbond_add_per_donor_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_hbond_add_per_donor_param

  function api_cf_hbond_add_per_acceptor_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_hbond_add_per_acceptor_param

  function api_cf_hbond_add_donor(i, d1, d2, d3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, d1, d2, d3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_hbond_add_donor

  function api_cf_hbond_add_acceptor(i, a1, a2, a3, params, n_params) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, a1, a2, a3, n_params
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_hbond_add_acceptor

  function api_cf_hbond_add_exclusion(i, donor, acceptor) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, donor, acceptor
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_hbond_add_exclusion

  subroutine api_cf_hbond_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, method
    call warn_no_omm()
  end subroutine api_cf_hbond_set_nonbonded_method

  subroutine api_cf_hbond_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call warn_no_omm()
  end subroutine api_cf_hbond_set_cutoff

  function api_cf_many_add_per_particle_param(i, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: i
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_many_add_per_particle_param

  function api_cf_many_add_particle(i, params, n_params, ptype) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, n_params, ptype
    real(c_double) :: params(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_many_add_particle

  function api_cf_many_add_exclusion(i, p1, p2) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, p1, p2
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_many_add_exclusion

  subroutine api_cf_many_set_nonbonded_method(i, method) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, method
    call warn_no_omm()
  end subroutine api_cf_many_set_nonbonded_method

  subroutine api_cf_many_set_cutoff(i, cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double), value :: cutoff
    call warn_no_omm()
  end subroutine api_cf_many_set_cutoff

  function api_cf_cv_add_collective_variable(cv_index, &
       cv_force_store_index, name) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    implicit none
    integer(c_int), value :: cv_index, cv_force_store_index
    character(kind=c_char, len=1), dimension(*) :: name
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_cv_add_collective_variable

  function api_cf_get_global_param_default_value(i, param_idx) &
       result(val) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, param_idx
    real(c_double) :: val
    val = 0.0d0; call warn_no_omm()
  end function api_cf_get_global_param_default_value

  function api_cf_bond_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_bond_get_num_bonds

  subroutine api_cf_bond_set_bond_parameters(i, idx, p1, p2, &
       params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, p1, p2, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_bond_set_bond_parameters

  subroutine api_cf_bond_get_bond_parameters(i, idx, p1, p2, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_bond_get_bond_parameters

  function api_cf_angle_get_num_angles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_angle_get_num_angles

  subroutine api_cf_angle_set_angle_parameters(i, idx, p1, p2, p3, &
       params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, p1, p2, p3, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_angle_set_angle_parameters

  function api_cf_torsion_get_num_torsions(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_torsion_get_num_torsions

  subroutine api_cf_torsion_set_torsion_parameters(i, idx, &
       p1, p2, p3, p4, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, p1, p2, p3, p4, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_torsion_set_torsion_parameters

  function api_cf_external_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_external_get_num_particles

  subroutine api_cf_external_set_particle_parameters(i, idx, &
       particle, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, particle, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_external_set_particle_parameters

  function api_cf_nb_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_nb_get_num_particles

  subroutine api_cf_nb_set_particle_parameters(i, idx, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_nb_set_particle_parameters

  function api_cf_nb_get_nonbonded_method(i) result(m) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: m
    m = -1; call warn_no_omm()
  end function api_cf_nb_get_nonbonded_method

  function api_cf_nb_get_cutoff(i) result(cutoff) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i
    real(c_double) :: cutoff
    cutoff = -1.0d0; call warn_no_omm()
  end function api_cf_nb_get_cutoff

  function api_cf_compound_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_compound_get_num_bonds

  subroutine api_cf_compound_set_bond_parameters(i, idx, &
       particles, np, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, np, n
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_compound_set_bond_parameters

  function api_cf_centroid_get_num_groups(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_centroid_get_num_groups

  function api_cf_centroid_get_num_bonds(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_centroid_get_num_bonds

  function api_cf_gb_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_gb_get_num_particles

  subroutine api_cf_gb_set_particle_parameters(i, idx, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_gb_set_particle_parameters

  subroutine api_cf_angle_get_angle_parameters(i, idx, p1, p2, p3, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2, p3
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_angle_get_angle_parameters

  subroutine api_cf_torsion_get_torsion_parameters(i, idx, &
       p1, p2, p3, p4, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: p1, p2, p3, p4
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_torsion_get_torsion_parameters

  subroutine api_cf_external_get_particle_parameters(i, idx, &
       particle, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: particle
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_external_get_particle_parameters

  subroutine api_cf_nb_get_particle_parameters(i, idx, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_nb_get_particle_parameters

  subroutine api_cf_compound_get_bond_parameters(i, idx, &
       particles, max_p, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_p, max_params
    integer(c_int) :: particles(*)
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_compound_get_bond_parameters

  subroutine api_cf_gb_get_particle_parameters(i, idx, &
       params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_gb_get_particle_parameters

  function api_cf_add_tabulated_function_continuous1d(i, name, &
       values, n, min_val, max_val, periodic) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, n, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: min_val, max_val
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_continuous1d

  function api_cf_add_tabulated_function_discrete1d(i, name, &
       values, n) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, n
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_discrete1d

  ! ---- CustomHbondForce getters/setters stubs ----

  function api_cf_hbond_get_num_donors(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_hbond_get_num_donors

  function api_cf_hbond_get_num_acceptors(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_hbond_get_num_acceptors

  subroutine api_cf_hbond_set_donor_parameters(i, idx, &
       d1, d2, d3, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, d1, d2, d3, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_set_donor_parameters

  subroutine api_cf_hbond_get_donor_parameters(i, idx, &
       d1, d2, d3, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: d1, d2, d3
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_get_donor_parameters

  subroutine api_cf_hbond_set_acceptor_parameters(i, idx, &
       a1, a2, a3, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, a1, a2, a3, n
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_set_acceptor_parameters

  subroutine api_cf_hbond_get_acceptor_parameters(i, idx, &
       a1, a2, a3, params, max_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    integer(c_int) :: a1, a2, a3
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_get_acceptor_parameters

  function api_cf_hbond_get_num_per_donor_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_hbond_get_num_per_donor_params

  function api_cf_hbond_get_num_per_acceptor_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_hbond_get_num_per_acceptor_params

  subroutine api_cf_hbond_get_per_donor_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_get_per_donor_param_name

  subroutine api_cf_hbond_get_per_acceptor_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_hbond_get_per_acceptor_param_name

  ! ---- CustomManyParticleForce getters/setters/extras stubs ----

  function api_cf_many_get_num_particles(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_many_get_num_particles

  subroutine api_cf_many_set_particle_parameters(i, idx, &
       params, n, ptype) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, n, ptype
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_many_set_particle_parameters

  subroutine api_cf_many_get_particle_parameters(i, idx, &
       params, max_params, ptype) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_params
    real(c_double) :: params(*)
    integer(c_int) :: ptype
    call warn_no_omm()
  end subroutine api_cf_many_get_particle_parameters

  subroutine api_cf_many_set_type_filter(i, particle_index, &
       types, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, particle_index, n
    integer(c_int) :: types(*)
    call warn_no_omm()
  end subroutine api_cf_many_set_type_filter

  subroutine api_cf_many_get_type_filter(i, particle_index, &
       types, max_types, num_types) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, particle_index, max_types
    integer(c_int) :: types(*), num_types
    call warn_no_omm()
  end subroutine api_cf_many_get_type_filter

  function api_cf_many_get_permutation_mode(i) result(mode) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: mode
    mode = -1; call warn_no_omm()
  end function api_cf_many_get_permutation_mode

  subroutine api_cf_many_set_permutation_mode(i, mode) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, mode
    call warn_no_omm()
  end subroutine api_cf_many_set_permutation_mode

  function api_cf_many_get_num_per_particle_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_many_get_num_per_particle_params

  subroutine api_cf_many_get_per_particle_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_many_get_per_particle_param_name

  ! ---- CustomCentroidBondForce setters/getters stubs ----

  subroutine api_cf_centroid_set_group_parameters(i, idx, &
       particles, np, weights, nw) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, np, nw
    integer(c_int) :: particles(*)
    real(c_double) :: weights(*)
    call warn_no_omm()
  end subroutine api_cf_centroid_set_group_parameters

  subroutine api_cf_centroid_get_group_parameters(i, idx, &
       particles, max_p, weights, max_w, &
       num_particles, num_weights) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_p, max_w
    integer(c_int) :: particles(*), num_particles, num_weights
    real(c_double) :: weights(*)
    call warn_no_omm()
  end subroutine api_cf_centroid_get_group_parameters

  subroutine api_cf_centroid_set_bond_parameters(i, idx, &
       groups, ng, params, n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, ng, n
    integer(c_int) :: groups(*)
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_centroid_set_bond_parameters

  subroutine api_cf_centroid_get_bond_parameters(i, idx, &
       groups, max_g, params, max_params, &
       num_groups, num_params) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    integer(c_int), value :: i, idx, max_g, max_params
    integer(c_int) :: groups(*), num_groups, num_params
    real(c_double) :: params(*)
    call warn_no_omm()
  end subroutine api_cf_centroid_get_bond_parameters

  function api_cf_centroid_get_num_per_bond_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_centroid_get_num_per_bond_params

  subroutine api_cf_centroid_get_per_bond_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_centroid_get_per_bond_param_name

  ! ---- Tabulated functions 2D/3D stubs ----

  function api_cf_add_tabulated_function_continuous2d(i, name, &
       values, nx, ny, xmin, xmax, ymin, ymax, periodic) &
       result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, nx, ny, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: xmin, xmax, ymin, ymax
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_continuous2d

  function api_cf_add_tabulated_function_discrete2d(i, name, &
       values, nx, ny) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, nx, ny
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_discrete2d

  function api_cf_add_tabulated_function_continuous3d(i, name, &
       values, nx, ny, nz, xmin, xmax, ymin, ymax, &
       zmin, zmax, periodic) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, nx, ny, nz, periodic
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    real(c_double), value :: xmin, xmax, ymin, ymax, zmin, zmax
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_continuous3d

  function api_cf_add_tabulated_function_discrete3d(i, name, &
       values, nx, ny, nz) result(idx) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_double
    implicit none
    integer(c_int), value :: i, nx, ny, nz
    character(kind=c_char, len=1), dimension(*) :: name
    real(c_double) :: values(*)
    integer(c_int) :: idx
    idx = -1; call warn_no_omm()
  end function api_cf_add_tabulated_function_discrete3d

  ! ---- Force groups stubs ----

  subroutine api_cf_set_force_group(i, grp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, grp
    call warn_no_omm()
  end subroutine api_cf_set_force_group

  function api_cf_get_force_group(i) result(grp) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: grp
    grp = -1; call warn_no_omm()
  end function api_cf_get_force_group

  ! ---- Pre-built OpenMM force stubs ----
  !
  ! Like every stub in this branch these report through warn_no_omm, which
  ! calls wrndie and so terminates the run at the default bomb level rather
  ! than returning.  They exist so the symbols resolve; callers are expected
  ! to check for OpenMM support first (pyCHARMM's omm module does, in
  ! _require_openmm_build and _check_openmm_build_matches) and fail with
  ! their own message before reaching one of these.

  !> Stub for api_custom_force_add_ptr in builds without OpenMM.
  !! @param[in] forceptr ignored
  !! @param[in] kind     ignored
  !! @return    nominally -1, but warn_no_omm terminates the run first
  function api_custom_force_add_ptr(forceptr, kind) result(new_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: forceptr
    integer(c_int), value :: kind
    integer(c_int) :: new_index
    new_index = -1; call warn_no_omm()
  end function api_custom_force_add_ptr

  !> Stub for api_cf_set_bucket in builds without OpenMM.
  !! @param[in] i    ignored
  !! @param[in] code ignored
  !! @return    nominally -1, but warn_no_omm terminates the run first
  function api_cf_set_bucket(i, code) result(status) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i, code
    integer(c_int) :: status
    status = -1; call warn_no_omm()
  end function api_cf_set_bucket

  !> Stub for api_cf_get_bucket in builds without OpenMM.
  !! @param[in] i ignored
  !! @return    nominally -4, but warn_no_omm terminates the run first.  A
  !!            failure code is used rather than -1, which callers read as
  !!            "this force has no energy-term override", so that a run with
  !!            the bomb level lowered far enough for the stub to return
  !!            fails rather than silently reporting a default.
  function api_cf_get_bucket(i) result(code) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: code
    code = -4; call warn_no_omm()
  end function api_cf_get_bucket

  ! ---- Introspection stubs ----

  subroutine api_cf_get_energy_expression(i, buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_get_energy_expression

  subroutine api_cf_get_global_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_get_global_param_name

  function api_cf_get_num_per_params(i) result(n) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    integer(c_int) :: n
    n = -1; call warn_no_omm()
  end function api_cf_get_num_per_params

  subroutine api_cf_get_per_param_name(i, param_idx, &
       buf, max_len) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_char
    implicit none
    integer(c_int), value :: i, param_idx, max_len
    character(kind=c_char, len=1) :: buf(*)
    call warn_no_omm()
  end subroutine api_cf_get_per_param_name

  subroutine api_cf_update_parameters_in_context(i) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), value :: i
    call warn_no_omm()
  end subroutine api_cf_update_parameters_in_context

#endif /* KEY_OPENMM */

end module api_omm
