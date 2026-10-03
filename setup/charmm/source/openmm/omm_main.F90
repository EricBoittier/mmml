!> Uses the OpenMM framework to run dynamics on GPU or other hardware.
!> Mike Garrahan and Charlie Brooks, 2012
!
! OpenMM home: https://simtk.org/home/openmm
! See also Eastman and Pande, "OpenMM: A Hardware-Independent Framework
! for Molecular Simulations," CiSE Jul 2010, DOI 10.1109/MCSE.2010.27
!
! openmm_sander.f by Radmer et al. was useful in early development.
! http://simtk.org/home/sander_openmm
!
! OpenMM glossary:
! A System is a set of atoms, bonds, potentials, etc.
! A Context combines a System with an Integrator and a Platform.
! A State is an immutable snapshot of a System, optionally including positions,
!   velocities, forces, or energies.
!
! OpenMM units: nm, ps, kJ/mol
! CHARMM units: Angstrom, AKMA time, kcal/mol
! Common units: atomic mass unit, electron charge
!
module omm_main
   use chm_kinds
   use number
   use psf, only: NATOM
   use stream, only: OUTU, PRNLEV
   use omm_nbopts
   use omm_dynopts
   use OpenMM
   use omm_gbsw, only : qphmd_omm, qphmd_initialized, qomm_cutoff
#if KEY_OPENMM==1
   use omm_gbsw, only : gbswforce
#endif
   use inbnd, only: qgbsw_omm, lommrxn, lommswi

   implicit none

   private
#if KEY_OPENMM==1
   type(OpenMM_System), save :: system
   type(OpenMM_Integrator), save :: integrator
   type(OpenMM_Context), save :: context

   !> An OpenMM System handed in from outside, for a caller that wants to
   !! build the OpenMM Context itself.  CHARMM fills this in but never owns
   !! it: it is not destroyed here, because whoever supplied it is still
   !! holding it.  Zero when nobody has supplied one.  See omm_adopt_system.
   type(OpenMM_System), save :: supplied_system
   logical, save :: system_supplied = .false.

   !> Whether the supplied System has already been filled in.  Filling one
   !! twice would double everything in it -- import_psf adds every particle
   !! and fstore_setup adds every stored force, neither of which checks what
   !! is already there -- so a second attempt is refused rather than silently
   !! counting the whole structure twice.
   logical, save :: supplied_system_populated = .false.

   !> Set when the only change since the System was built is that forces were
   !! added.  That case does not need the System thrown away and rebuilt: the
   !! new forces can be added to the System as it stands and the Context told
   !! to pick them up.  Anything else -- a force mutated, an energy term
   !! changed, a force switched off -- still needs the coarse system_dirty,
   !! because those alter a copy that is already in the System and CHARMM has
   !! no way to reach into it.
   logical, save :: forces_added_only = .false.

   !> A Context handed in from outside, for a caller who wants to choose the
   !! platform, bring their own integrator, or run a force that has to be
   !! driven from Python.  CHARMM drives it but never destroys it.
   type(OpenMM_Context), save :: supplied_context
   logical, save :: context_supplied = .false.

   !> Set at the start of each dynamics command when the Context is the
   !! caller's.  CHARMM's own answer is to rebuild the Context so the
   !! integrator starts from clean state, which measurably matters: with the
   !! reset removed, stochastic restart stages of test/c37test/
   !! omm_dynamics.inp move by 0.8% to 3.6% while deterministic ones do not
   !! shift at all.  A Context CHARMM does not own cannot be rebuilt, so the
   !! equivalent is reinitialize() *discarding* state -- which is the opposite
   !! of what a force change wants, and the two must not be confused.
   logical, save :: context_clean_start_pending = .false.

   ! Previous reported total energy, for the ECHECK energy-change test.
   ! Kept here rather than read back out of EPROP(TOTE): code we call
   ! between two reported states can reset the shared EPROP array, and
   ! replica exchange does exactly that on every exchange attempt.  Reading
   ! EPROP then made the check compare a real total energy against zero.
   real(chm_real), save :: prev_report_tote = ZERO

   type(omm_dynopts_t), save :: dynopts
   type(omm_nbopts_t), save :: nbopts
   integer, save :: cm_stop_freq
   integer, save :: dynamics_mode
   !> Index of the barostat force in the System; -1 when there is none.
   integer, save :: indexBarostat = -1
   !> Index of the Andersen thermostat in the System, so its temperature can
   !! be updated in place; -1 when there is none.  Same purpose as
   !! indexBarostat above.
   integer, save :: indexAndersen = -1
   integer, save :: openmm_version_number
   !> Whether the object currently in `system` / `context` is one CHARMM
   !! created, and therefore CHARMM's to destroy.  Recorded where the object
   !! is created or adopted, because that is the only place that knows.
   !!
   !! Deciding this at teardown by reading system_supplied / context_supplied
   !! instead was a bug: releasing a supplied System clears the flag, so the
   !! next teardown believed the caller's System was its own and freed it,
   !! leaving the caller holding freed memory.
   logical, save :: charmm_owns_system = .false.
   logical, save :: charmm_owns_context = .false.

   logical, save :: openmm_initialized = .false.
   logical, save :: system_dirty = .false.
#endif

   integer, parameter :: DYN_UNINIT = 0, DYN_RESTART = 1, &
         DYN_SETVEL = 2,  DYN_CONTINUE = 3

#if KEY_OPENMM==1
   public :: omm_energy, teardown_openmm, omm_invalidate, &
        omm_pressure_active, check_nbopts
   public :: serialize, serialize_system
   public :: nbopts, get_plugins
   public :: omm_change_lambda
   public :: gbswforce
   public :: omm_minimize
   public :: omm_get_context_ptr
   public :: omm_adopt_system, omm_release_system, omm_system_is_supplied
   public :: omm_invalidate_forces_added
   public :: omm_adopt_context, omm_release_context, omm_context_is_supplied
#endif

   public :: omm_dynamics, omm_eterm_mask

 contains

   !> Performs molecular dynamics using OpenMM.
   subroutine omm_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
         jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
         iprfrq, isvfrq, ntrfrq, openmm_ran, qkuhead)

      use reawri, only: NSTEP

      type(omm_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
      real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
      real(chm_real), intent(inout) :: jhtemp
      real(chm_real), intent(in) :: gamm(:)
      integer, intent(inout) :: npriv
      ! igvopt is intent(inout): like the CPU leapfrog integrator (DYNAMC),
      ! we mark the assigned velocities "consumed" (igvopt=3) after injecting
      ! them, so a later reassignment (ASSVEL/SCAVEL for ihtfrq/ieqfrq, which
      ! reset igvopt=2) is distinguishable from a plain continuation cycle.
      integer, intent(inout) :: igvopt
      integer, intent(in) :: istart, istop, iprfrq, ndegf, isvfrq, ntrfrq
      logical, intent(inout) :: openmm_ran, qkuhead

      openmm_ran=.FALSE.

#if KEY_OPENMM==1
      cm_stop_freq = ntrfrq
      ! Each `dynamics` command needs a fresh OpenMM context so the
      ! integrator's internal state (notably the RNG stream used by
      ! Andersen/Langevin/MC barostat) starts clean.  Before commit
      ! 37ea9baa8 this happened implicitly: nbonds.F90 unconditionally
      ! called teardown_openmm() on every energy/nbond rebuild.  That
      ! commit removed the per-energy-call teardown (the right call for
      ! interactive ENERGY commands) but also lost the per-dynamics-command
      ! reset, causing stochastic _restart_ stages of test/c37test/
      ! omm_dynamics.inp to drift ~1% from their reference energies.
      ! Restore the per-dynamics rebuild here without re-introducing the
      ! per-energy-call cost.
      !
      ! Only invalidate on the FIRST cycle of the command (istart == 1).
      ! DCNTRL calls omm_dynamics once per integration cycle, and a small
      ! isvfrq/nsavc/etc. splits a single `dynamics` command into many
      ! short cycles.  Invalidating on every cycle tore down and rebuilt
      ! the context after each restart-file write (e.g. isvfrq 1 -> every
      ! step), re-seeding velocities mid-run and destroying the integrator
      ! state.  istart == 1 marks the first cycle of each command, which is
      ! exactly the once-per-command reset intended here.
      if (istart == 1) then
         if (omm_context_is_supplied() == 1) then
            ! Same intent as the rebuild below, by the only means available
            ! for a Context CHARMM does not own: discard the integrator's
            ! state.  run_dynamics pushes positions, velocities, time and box
            ! back in on the first cycle of every command, so nothing is lost
            ! that CHARMM does not immediately restore.
            context_clean_start_pending = .true.
         else
            call omm_invalidate()
         end if
      end if
      call check_dynopts(optarg)

      call run_dynamics(vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
            jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
            iprfrq, isvfrq, openmm_ran,qkuhead)
#endif

      if (PRNLEV >= 2) then
         if (openmm_ran .and. istop == nstep) then
            write (OUTU, 1001)
         elseif (istop == nstep) then
            write (OUTU, 1002)
         endif
      endif

      1001 format(/, 1x, 'OpenMM was used to perform energy and force evaluation.', /)
      1002 format(/, 1x, 'OpenMM was NOT used to perform energy and force evaluation.', /)
   end subroutine omm_dynamics

   !> Returns an array indicating whether we can use OpenMM
   !> for each energy term.
   function omm_eterm_mask()
      use energym
      logical :: omm_eterm_mask(LENENT)
      integer, parameter :: my_eterms(22) = &
            [BOND, ANGLE, UREYB, DIHE, IMDIHE, CMAP, &
            VDW, ELEC, IMVDW, IMELEC, EXTNDE, RXNFLD, &
            EWKSUM, EWSELF, EWEXCL, ELRC, CHARM, CDIHE, &
            RESD, GBEnr, GEO, PCHARM]
      integer :: i

      omm_eterm_mask = .false.
#if KEY_OPENMM==1
      do i = 1, size(my_eterms)
         omm_eterm_mask(my_eterms(i)) = .true.
      enddo
#endif
   end function omm_eterm_mask

#if KEY_OPENMM==1 /*openmm*/

   subroutine run_dynamics(vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
         jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
         iprfrq, isvfrq, openmm_ran, qkuhead)
      use averfluc
      use avfl_ucell
      use coord  ! XXX writes
      use deriv  ! XXX writes
      use image, only: XTLTYP
      use reawri, only: NSTEP, JHSTRT, TIMEST, KUNIT
      use contrl, only: NPRINT, IREST
      use consta, only: TIMFAC

      real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
      real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
      real(chm_real), intent(inout) :: jhtemp
      real(chm_real), intent(in) :: gamm(:)
      integer, intent(inout) :: npriv
      integer, intent(inout) :: igvopt   ! set to 3 once velocities are consumed
      integer, intent(in) :: istart, istop, iprfrq, isvfrq, ndegf
      logical, intent(inout) :: openmm_ran, qkuhead

      real*8, parameter :: ChmVelPerOmmVel = TIMFAC * OpenMM_AngstromsPerNm
      real*8 :: pos(3, NATOM), vel_t(3, NATOM), vel_pre(3, NATOM)
      real(chm_real) :: dnum
      integer*4,save :: istep
      integer :: numstp, i
      integer, save :: navestps
      real(chm_real) :: wtf(NATOM)
      logical :: check_tol

      call setup_openmm()
      ! Decide whether to (re)inject state into the OpenMM context this cycle.
      ! DCNTRL calls run_dynamics once per integration cycle; a single
      ! `dynamics` command is split into many short cycles when a reporting
      ! frequency is small (isvfrq/nsavc -> restart/trajectory writes) or when
      ! CHARMM manages the temperature (ihtfrq/ieqfrq -> velocity reassignment).
      !   - istart == 1 is the first cycle of the command: inject the initial
      !     state (restart-file velocities, or freshly assigned velocities).
      !   - igvopt < 3 on a later cycle means CHARMM just reassigned/scaled the
      !     velocities (ASSVEL/SCAVEL set igvopt=2); push them into the context.
      !   - otherwise the context has been integrating continuously (e.g. a
      !     restart-file write mid-run) and must NOT be disturbed -- disturbing
      !     it tore the context down and re-seeded velocities on every write.
      if (istart == 1 .and. IREST > 0) then
         dynamics_mode = DYN_RESTART
      else if (IGVOPT < 3) then
         dynamics_mode = DYN_SETVEL
      else
         dynamics_mode = DYN_CONTINUE
      endif

      ! XXX scale factor for vel_pre from restart file written by DYNAMC,
      ! even if not doing Langevin dynamics
      wtf = TWO * gamm(3*NATOM+1 : 4*NATOM)

      ! Clean integrator state for a Context CHARMM does not own -- the
      ! equivalent of the rebuild CHARMM does for its own.  It belongs here
      ! rather than in check_system, and it forces the state push:
      ! reinitialize with preserveState FALSE throws the positions away, so
      ! discarding state and putting it back must be one decision.  Doing the
      ! discard where the push was optional left OpenMM with no positions at
      ! all ("Particle positions have not been set") on a continuation cycle.
      !
      ! preserveState is FALSE on purpose.  TRUE keeps the integrator's
      ! existing state, which is what picking up a new force wants and the
      ! opposite of what this wants.
      if (context_clean_start_pending) then
         if (PRNLEV >= 2) write (OUTU, '(a)') &
              'OpenMM: clean integrator state for this dynamics command'
         call OpenMM_Context_reinitialize(context, OpenMM_False)
         context_clean_start_pending = .false.
         if (dynamics_mode == DYN_CONTINUE) dynamics_mode = DYN_SETVEL
      end if

      if (dynamics_mode /= DYN_CONTINUE) then
         pos = get_xyz(X, Y, Z) / OpenMM_AngstromsPerNm
         call set_positions(context, pos)
#if KEY_PHMD==1
         if (qphmd_omm .and. qphmd_initialized) call set_lambda_state(context)
#endif
         call OpenMM_Context_setTime(context, npriv * TIMEST)
         if (nbopts%periodic) call import_periodic_box()

         vel_t = get_xyz(vx_t, vy_t, vz_t) / ChmVelPerOmmVel
         if (dynamics_mode == DYN_RESTART) then
            if (PRNLEV >= 6) write (OUTU, '(a)') 'OpenMM: Velocities from restart file'
            vel_pre = get_xyz(vx_pre, vy_pre, vz_pre) / ChmVelPerOmmVel
            do i = 1, NATOM
               vel_pre(:, i) = vel_pre(:, i) * wtf(i)
            enddo
         else if (dynamics_mode == DYN_SETVEL) then
            if (PRNLEV >= 2) write (OUTU, '(a)') 'OpenMM: Velocities scaled or randomized'
            vel_pre = vel_t - half_delta_vel()
            if(dynopts%omm_updateT) call new_Temperature(dynopts)
         else
            if (PRNLEV >= 2) write (OUTU, '(a)') 'OpenMM: Velocities undefined!'
         endif
         call set_velocities(context, vel_pre)
         ! Velocities have been consumed.  Mark igvopt=3 so the next cycle
         ! continues from the context (leapfrog integrators do the same);
         ! ihtfrq/ieqfrq reassignment in DCNTRL will reset igvopt=2 when it
         ! needs the velocities re-injected.
         igvopt = 3

      ! else use velocities already in OpenMM context
      endif

      if (istart <= 1) istep = 0

      if (todo_now(iprfrq, istart-1)) then
         navestps = 0
         jhtemp = ZERO
         call avfl_reset()
         if (dynopts%qPressure) call avfl_ucell_reset()
      endif

      ! Dynamic step 0 typically has very different energy from later steps.
      ! Never check the first reported state of an invocation either: the
      ! previous total it is compared against was produced before this call.
      ! The DYN_CONTINUE branch below arms the check from the second state on.
      check_tol = .false.
      do
         call show_state(pos, vel_t, vel_pre, istep, istart, istop, npriv, ndegf, &
               isvfrq, navestps, jhtemp, qkuhead)
         call check_energy(check_tol)
         if (dynamics_mode == DYN_CONTINUE) then
            check_tol = .true.
            call set_xyz(X, Y, Z, pos * OpenMM_AngstromsPerNm, &
                 nbopts%periodic)
            call set_xyz(vx_t, vy_t, vz_t, vel_t * ChmVelPerOmmVel, &
                 .false.)
            call traj_write(istep, npriv, ndegf, vx_t, vy_t, vz_t)
         endif

         if (istep >= istop) exit
         call run_steps(istep, istop, npriv, isvfrq)
         dynamics_mode = DYN_CONTINUE
      enddo

      do i = 1, NATOM
         ! XXX DYNAMC seems to do this too
         vel_pre(:, i) = vel_pre(:, i) / wtf(i)
      enddo
      call set_xyz(vx_pre, vy_pre, vz_pre, vel_pre * ChmVelPerOmmVel, &
           .false.)
      ! Restart file actually written in dcntrl - not needed here

      ! check whether we need to print average energies
      numstp = ( mod(istart-1,iprfrq) + istep - istart + 1 )
      if(numstp>=iprfrq) then
         dnum = navestps
         call avfl_compute(dnum)
         jhtemp = (jhtemp / dnum) * jhstrt
         if (dynopts%qPressure) call avfl_ucell_compute(dnum)

         ! . Print out the results.
         IF(PRNLEV >= 2) THEN
            call avfl_print_aver(numstp, npriv*TIMEST, tag='DYNAMC>')
            if (dynopts%qPressure) call avfl_ucell_print_aver(numstp)
            call avfl_print_fluc(numstp, npriv*TIMEST, tag='DYNAMC>')
            if (dynopts%qPressure) call avfl_ucell_print_fluc(numstp)
         endif
      endif
      !
      ! End of write averages

      openmm_ran=.TRUE.

      if (kunit > 0) call flush(kunit)

    end subroutine run_dynamics

    subroutine kunit_write(istep, npriv, qkuhead)
      use dynio, only: wreterm
      use reawri, only: timest, kunit
      implicit none
      integer, intent(in) :: istep, npriv
      logical, intent (inout) :: qkuhead
      if (istep > 0 .and. kunit > 0) call wreterm(npriv,npriv*timest,qkuhead)
    end subroutine kunit_write

   subroutine check_energy(check_tol)
      use energym
      use reawri, only: ECHECK, JHSTRT
      logical, intent(in) :: check_tol

      ! NaN /= NaN
      if (EPROP(TOTE) /= EPROP(TOTE) &
            .or. EPROP(TOTKE) /= EPROP(TOTKE)) then
         call WRNDIE(-2, 'omm_main', 'Energy is NaN')
      endif
      if (.not. check_tol) return

      if (JHSTRT > 2 .and. ECHECK > ZERO) then
         ! copied from dynamc
         if (abs(EPROP(TEPR) - EPROP(TOTE)) &
               > max(ECHECK, PTONE * EPROP(TOTKE))) then
            if (WRNLEV >= 2) then
               write (OUTU, '(A/G12.2,A/3(A,G14.4))') &
                     'Total energy change exceeded', &
                     ECHECK, ' kcal and 10% of the total kinetic energy in the last step', &
                     ' Previous E =', EPROP(TEPR), &
                     ' Current E =', EPROP(TOTE), &
                     ' Kinetic =', EPROP(TOTKE)
            endif
            call WRNDIE(-2, 'omm_main', 'Energy change tolerance exceeded')
         endif
      endif
   end subroutine check_energy

   subroutine setup_openmm()
     use omm_glblopts, only : qtor_repex, torsion_lambda
     use omm_mses, only: qmses, qmses_import_omm, update_mses ! AN EXAMPLE OPENMM PLUGIN
     use omm_nbopts, only: current_nbopts
      call get_PlatformDefaults

      call check_system()
      call check_nbopts()
#if KEY_OPENMM==1
      if (openmm_initialized .and. qmses .and. (.not.qmses_import_omm)) call update_mses(context) ! AN EXAMPLE OPENMM PLUGIN
#endif
      if (openmm_initialized) return
      if (prnlev >= 2) write (OUTU, '(A)') 'Setup_OpenMM: Initializing OpenMM context'
      call load_libs()
      call setup_system(.false.)
      call init_context()
      if(qtor_repex) call omm_change_lambda(torsion_lambda)
      dynamics_mode = DYN_UNINIT
      openmm_initialized = .true.
      ! Resnapshot nbopts after full initialization so that the energy
      ! evaluation (which may regenerate nonbond lists) does not cause
      ! a spurious mismatch on the next call.
      nbopts = current_nbopts()
   end subroutine setup_openmm

   subroutine teardown_openmm()
      use new_timer
#if KEY_OPENMM == 1
      use fstore, only: store, fstore_clear_in_system
#endif /* KEY_OPENMM */
      if (.not. openmm_initialized) return
      if (prnlev >= 2) write (OUTU, '(A)') 'Destroying OpenMM context'
      openmm_initialized = .false.
      call timer_start(T_omm_ctx)
      if (.not. charmm_owns_context) then
         ! The caller still holds this Context and the integrator inside it;
         ! destroying either would leave them with freed pointers.
         if (prnlev >= 6) write (OUTU, '(a)') &
              'OpenMM: leaving the supplied Context alone'
      else
         call OpenMM_Context_destroy(context)
         call OpenMM_Integrator_destroy(integrator)
      end if
      charmm_owns_context = .false.
#if KEY_OPENMM == 1
      ! The System's copies of the stored forces are about to stop existing
      ! (or, for a supplied System, to stop being ours to reason about), so
      ! nothing is "already in the System" any more.
      call fstore_clear_in_system(store)
#endif /* KEY_OPENMM */
      if (.not. charmm_owns_system) then
         ! Someone else owns this System and is still holding it; destroying
         ! it here would leave them with a freed pointer.  It stays marked as
         ! populated, so a later rebuild is refused rather than doubling it.
         continue
      else
         call OpenMM_System_destroy(system)
      end if
      charmm_owns_system = .false.
      call timer_stop(T_omm_ctx)
   end subroutine teardown_openmm

   subroutine omm_invalidate()
      if (openmm_initialized) system_dirty = .true.
   end subroutine omm_invalidate

   !> @brief Note that forces were added, without condemning the whole System.
   !!
   !! Use instead of omm_invalidate when the only change is that forces were
   !! added to the store.  check_system then adds them to the System already
   !! built and reinitializes the Context, rather than destroying the System
   !! and putting the whole structure back together.
   !!
   !! Cheaper on the ordinary path, and the only workable answer on a System
   !! CHARMM does not own, which it cannot destroy or empty.
   subroutine omm_invalidate_forces_added()
      if (openmm_initialized) forces_added_only = .true.
   end subroutine omm_invalidate_forces_added

   logical function omm_pressure_active()
      omm_pressure_active = .false.
      if (openmm_initialized) omm_pressure_active = dynopts%qPressure
   end function omm_pressure_active

   subroutine check_system()
#if KEY_OPENMM == 1
      use fstore, only: store, fstore_setup, fstore_clear_in_system
#endif /* KEY_OPENMM */

      if (system_dirty) then
         if (PRNLEV >= 2) write (OUTU, '(a)') 'OpenMM system changed'
         call teardown_openmm()
         system_dirty = .false.
         ! Whatever was in that System went with it.
         forces_added_only = .false.
         return
      end if

#if KEY_OPENMM == 1
      if (forces_added_only .and. openmm_initialized) then
         if (PRNLEV >= 2) write (OUTU, '(a)') &
              'OpenMM system gained forces; adding them without a rebuild'
         ! fstore_setup skips the forces already in the System, so this adds
         ! only the new ones.
         call fstore_setup(system)
         ! The Context caches the System's contents, so it has to be told.
         ! preserveState keeps positions, velocities and time, which is the
         ! whole point of not rebuilding.
         call OpenMM_Context_reinitialize(context, OpenMM_True)
         forces_added_only = .false.
      end if
#endif /* KEY_OPENMM */
   end subroutine check_system

   subroutine check_nbopts()
     use omm_nbopts, only: &
          omm_nbopts_t, &
          same_nbopts, current_nbopts, &
          print_nbopts_diff
     use stream, only: prnlev
     implicit none
     type(omm_nbopts_t) :: nbcurr

      nbcurr = current_nbopts()
      if(prnlev > 5) write(outu,'(a,i4,i4,a,l4,l4)') &
           "nblckcalls=", nbcurr%block_changed, nbopts%block_changed, &
           " qblock=", nbcurr%use_block, nbopts%use_block

      if (openmm_initialized) then
         if (.not. same_nbopts(nbcurr, nbopts)) then
            if (prnlev >= 2) then
               write (outu, '(A)') 'Nonbonded options changed'
               if(prnlev >= 6) &
                    call print_nbopts_diff(nbcurr, nbopts)
            end if
            if (system_supplied) then
               ! Nonbonded options are built into the System, so honouring
               ! them means rebuilding it -- which CHARMM cannot do to a
               ! System it did not create, and filling the same one twice
               ! would count every particle and force again.  Tearing down
               ! here instead reached omm_create_system's refusal, ending the
               ! run while naming neither the nonbonded options nor anything
               ! the user could act on.
               nbopts = nbcurr
               call wrndie(-3, '<CHECK_NBOPTS>', &
                    'Nonbonded options changed, but the OpenMM System was ' // &
                    'supplied from outside and cannot be rebuilt.  Set the ' // &
                    'nonbonded options before handing the System over, or ' // &
                    'clear OpenMM (omm.clear() in pyCHARMM) and supply a ' // &
                    'fresh System and Context.')
               return
            end if
            call teardown_openmm()
         end if
      end if
      nbopts = nbcurr
   end subroutine check_nbopts

   subroutine check_dynopts(optarg)
      type(omm_dynopts_t), intent(inout) :: optarg

      if (openmm_initialized) then

         if ( qphmd_initialized .and. qomm_cutoff .and. .not. lommrxn .and. .not. lommswi ) then
            call wrndie(-2, 'CUDA CPHMD', 'CUDA CPHMD w/ cutoffs &
                reqires OpenMM cutoff schemes. Use nonbonded cutoff options: OMSWitch OMRField OMRX 1')
         endif

        !if(optarg%omm_qrexchg) &
        !     optarg%omm_updateT = .not. &
        !     (optarg%temperatureReference == dynopts%temperatureReference)
        !set whether the temperature is updated, in order to avoid to tear down
        !the openmm context, because creating one is expensive.
        optarg%omm_updateT = .not. (optarg%temperatureReference == dynopts%temperatureReference)
         if (.not. same_dynopts(optarg, dynopts)) then
            if (context_supplied) then
               ! The integrator belongs to whoever supplied the Context, so
               ! these options cannot be applied to it.  Rebuilding to honour
               ! them is not available either -- the Context and the System
               ! are not CHARMM's to replace.  Say so rather than tear down a
               ! Context CHARMM does not own, or pretend the options took.
               if (PRNLEV >= 2) write (OUTU, '(a)') &
                    'CHARMM> OpenMM integrator options changed, but the ' // &
                    'Context and its integrator were supplied from ' // &
                    'outside.  The integrator is unchanged: the timestep, ' // &
                    'temperature and friction in use are the ones it was ' // &
                    'built with, not the ones on this command.'
            else
               if (PRNLEV >= 2) write (OUTU, '(A)') &
                     'OpenMM integrator options changed'
               call teardown_openmm()
            endif
         endif

         ! A thermostat or barostat is a FORCE inside the System, added while
         ! the System is populated -- which for a supplied System happens
         ! before any dynamics options exist.  So the force is simply absent,
         ! and it cannot be added now: the Context is already built on that
         ! System and rebuilding it is not CHARMM's to do.  Without this
         ! check the run went ahead with no thermostat at all, having just
         ! announced "Constant temperature w/ OpenMM using Andersen heatbath
         ! requested" -- the warning above covers the integrator only, and
         ! says nothing about a dropped force.
         if (context_supplied) then
            if (optarg%qAndersen .and. indexAndersen < 0) then
               call wrndie(-2, '<OMM_DYNAMICS>', &
                    'ANDErsen heatbath requested, but the supplied Context ' // &
                    'was built on a System with no thermostat force, which ' // &
                    'CHARMM cannot add now.  Add the thermostat to the ' // &
                    'System before creating the Context, or let CHARMM own ' // &
                    'the Context.')
            endif
            if (optarg%qPressure .and. indexBarostat < 0) then
               call wrndie(-2, '<OMM_DYNAMICS>', &
                    'Constant pressure requested, but the supplied Context ' // &
                    'was built on a System with no barostat force, which ' // &
                    'CHARMM cannot add now.  Add the barostat to the ' // &
                    'System before creating the Context, or let CHARMM own ' // &
                    'the Context.')
            endif
         endif
      endif
      dynopts = optarg
   end subroutine check_dynopts

   !> @brief Drive an OpenMM Context created by the caller.
   !!
   !! Lets a caller choose the platform and its properties, bring their own
   !! integrator, or run a force that has to be driven from Python.  CHARMM
   !! pushes positions, velocities, time and box vectors into it and reads
   !! energies and forces back out, exactly as it does with its own.
   !!
   !! CHARMM never destroys it, and never destroys the integrator inside it.
   !!
   !! The Context must be built on the System CHARMM filled in, which is why
   !! that is checked here rather than trusted: a Context built on some other
   !! System would be missing every force CHARMM added, and the energies would
   !! be quietly wrong rather than obviously broken.
   !!
   !! @param[in] ctxptr an OpenMM::Context* from the SAME OpenMM library
   !!                   CHARMM is linked against.
   !! @return    0 on success; 1 if no System was supplied first; 2 if that
   !!            System has not been filled in yet, so a Context on it would
   !!            be empty; 3 if the Context was built on a different System.
   function omm_adopt_context(ctxptr) result(status)
      use, intrinsic :: iso_c_binding, only: c_ptr

      implicit none

      type(c_ptr), intent(in) :: ctxptr
      integer :: status

      type(OpenMM_Context) :: candidate
      type(OpenMM_System) :: ctx_system
      type(OpenMM_Platform) :: ctx_platform
      character(len=40) :: ctx_platform_name

      status = 0

      if (.not. system_supplied) then
         status = 1
         return
      end if
      if (.not. supplied_system_populated) then
         status = 2
         return
      end if

      candidate = transfer(ctxptr, OpenMM_Context(0))
      call OpenMM_Context_getSystem(candidate, ctx_system)
      if (ctx_system%handle /= supplied_system%handle) then
         status = 3
         return
      end if

      supplied_context = candidate
      context_supplied = .true.

      ! Give back the Context CHARMM built before taking the caller's.
      ! CHARMM owns that one, and once `context` points at the supplied one
      ! teardown_openmm will only ever see the supplied one -- so without this
      ! the Context and integrator CHARMM made are leaked on adoption.
      ! Asking charmm_owns_context rather than context_supplied is what makes
      ! this work: context_supplied was set two lines up, so testing it here
      ! was always false and nothing was ever freed.  On a second adoption
      ! ownership is already false, so the caller's Context is left alone.
      if (openmm_initialized .and. charmm_owns_context) then
         call OpenMM_Context_destroy(context)
         call OpenMM_Integrator_destroy(integrator)
      end if
      charmm_owns_context = .false.

      ! Start driving it now.  Setting only supplied_context would leave
      ! `context` pointing at the one CHARMM built, and since `integrator` is
      ! taken from the supplied Context just below, CHARMM would then push
      ! state into one Context and step the integrator of another.  That pair
      ! fails as "Particle positions have not been set" the first time
      ! dynamics runs, because the stepped Context never received any.
      !
      context = supplied_context

      ! CHARMM steps whatever is in `integrator`, so it has to be the one
      ! inside the Context now being driven.
      call OpenMM_Context_getIntegrator(supplied_context, integrator)

      ! Name the platform.  Choosing it is usually the whole reason for
      ! supplying a Context, and a run that has quietly fallen back to
      ! Reference looks exactly like a correct one in the log otherwise.
      call OpenMM_Context_getPlatform(supplied_context, ctx_platform)
      call OpenMM_Platform_getName(ctx_platform, ctx_platform_name)

      if (prnlev >= 2) write (OUTU, '(a)') &
           'CHARMM> Driving an OpenMM Context supplied from outside on the ' &
           // trim(ctx_platform_name) // ' platform; CHARMM will not ' // &
           'destroy it, and its integrator is yours.'

   end function omm_adopt_context

   !> @brief Stop driving a Context supplied by the caller.
   !!
   !! CHARMM goes back to making its own.  The supplied Context is not
   !! destroyed; it still belongs to whoever created it.
   subroutine omm_release_context()

      implicit none

      if (context_supplied .and. prnlev >= 2) write (OUTU, '(a)') &
           'CHARMM> Releasing the supplied OpenMM Context; CHARMM will ' // &
           'make its own from now on.'
      context_supplied = .false.
      context_clean_start_pending = .false.
      supplied_context = OpenMM_Context(0)

      ! `context` was pointing at the caller's Context.  It is not CHARMM's to
      ! keep and not CHARMM's to destroy, so forget it and mark nothing as set
      ! up -- otherwise the next teardown would free a Context the caller
      ! still holds.
      !
      ! Only when the Context is not CHARMM's.  Called with nothing supplied
      ! this is a no-op, and forgetting a Context CHARMM built would strand it
      ! and the System under it: teardown returns early on
      ! .not. openmm_initialized, so neither would ever be freed.
      if (.not. charmm_owns_context) then
         context = OpenMM_Context(0)
         openmm_initialized = .false.
      end if

      ! The System was adopted alongside it and cannot be filled in a second
      ! time, so it goes back too: releasing the Context returns CHARMM to
      ! making its own of both, which is the only state it can build from.
      call omm_release_system()

   end subroutine omm_release_context

   !> @brief Whether CHARMM is driving a Context it does not own.
   !! @return 1 if so, 0 otherwise.
   function omm_context_is_supplied() result(state)

      implicit none

      integer :: state

      state = 0
      if (context_supplied) state = 1

   end function omm_context_is_supplied

   !> @brief Take an OpenMM System built elsewhere and fill that one instead.
   !!
   !! For a caller that wants to create the OpenMM Context itself.  A Context
   !! has to be built on the System that CHARMM actually evaluates, and a
   !! System cannot be handed out of CHARMM after the fact -- a raw
   !! OpenMM::System pointer cannot be turned back into a usable object on the
   !! Python side.  So the caller makes the System, keeps its own reference to
   !! it, and gives CHARMM the pointer; CHARMM fills it in, and the caller can
   !! then build a Context on the very object CHARMM will evaluate.
   !!
   !! CHARMM does not own the System afterwards and never destroys it.  What
   !! CHARMM does own is what goes *into* it: the PSF, the restraints, the
   !! stored forces and the temperature and pressure control all come from
   !! CHARMM's state, and nothing outside can supply them.
   !!
   !! The System is filled in once.  CHARMM's answer to a later change is
   !! normally to throw its System away and build another, which is not
   !! available for a System it does not own, so a second build is refused
   !! rather than doubling everything already in it.
   !!
   !! @param[in] sysptr an OpenMM::System* from the SAME OpenMM library CHARMM
   !!                   is linked against.  A pointer from another build, or
   !!                   one that is not a System, cannot be detected here and
   !!                   will crash when it is used.
   subroutine omm_adopt_system(sysptr)
      use, intrinsic :: iso_c_binding, only: c_ptr

      implicit none

      type(c_ptr), intent(in) :: sysptr

      supplied_system = transfer(sysptr, OpenMM_System(0))
      system_supplied = .true.
      supplied_system_populated = .false.

      if (prnlev >= 2) write (OUTU, '(a)') &
           'CHARMM> Using an OpenMM System supplied from outside; CHARMM ' // &
           'will fill it in but will not destroy it.'

   end subroutine omm_adopt_system

   !> @brief Forget any System supplied from outside.
   !!
   !! Returns CHARMM to making its own.  Called on OMM CLEAR, so that clearing
   !! OpenMM really does start over rather than leaving a stale arrangement
   !! pointing at an object the caller may already have dropped.
   subroutine omm_release_system()

      implicit none

      if (system_supplied .and. prnlev >= 2) write (OUTU, '(a)') &
           'CHARMM> Releasing the supplied OpenMM System; CHARMM will make ' // &
           'its own from now on.'
      system_supplied = .false.
      supplied_system_populated = .false.
      supplied_system = OpenMM_System(0)

   end subroutine omm_release_system

   !> @brief Whether a System supplied from outside is in use.
   !!
   !! @return 1 when CHARMM is filling in a System it does not own and that
   !!         System has already been built (so it cannot be built again), 0
   !!         otherwise.  Lets the pyCHARMM layer report the refusal above as
   !!         an ordinary Python error, before CHARMM has to end the run.
   function omm_system_is_supplied() result(state)

      implicit none

      integer :: state

      state = 0
      if (system_supplied) state = 1
      if (system_supplied .and. supplied_system_populated) state = 2

   end function omm_system_is_supplied

   !> @brief Build the OpenMM System, and an integrator to go with it.
   !!
   !! Split into omm_create_system and omm_populate_system rather than done in
   !! piece.  The two halves answer to different owners: making the System
   !! object is a step a caller could reasonably do instead (see
   !! omm_create_system), while filling it from the PSF is CHARMM's own work and
   !! stays CHARMM's either way.  Keeping the seam here means the substitution
   !! is a change of one call, not a rewrite of this routine.
   !!
   !! The integrator is deliberately outside both: it belongs to whoever runs
   !! the dynamics, not to the System.
   !!
   !! @param[in] init true while probing platform defaults, when only the
   !!                 particles are needed and the rest of the setup -- forces,
   !!                 restraints, thermostat, barostat, constraints -- is
   !!                 skipped.  It reaches omm_create_system too, because the
   !!                 probe destroys the System it builds: adopting a supplied
   !!                 one there would free memory the caller still holds.
   subroutine setup_system(init)
      use new_timer
      use rndnum, only : rngseeds

      logical :: init

      integer*4 :: ommseed

      ommseed = rngseeds(1)

      if(prnlev>5) write(outu,'(a,i16)') &
           'CHARMM> OpenMM using random seed ',ommseed


      call timer_start(T_omm_sys)

      call omm_create_system(init)
      call omm_populate_system(init, ommseed)

      integrator = new_integrator(dynopts, ommseed)
      if(.not. init) call setup_shake(system, integrator)

      call timer_stop(T_omm_sys)

   end subroutine setup_system

   !> @brief Make the OpenMM System object CHARMM will fill in.
   !!
   !! Named with the module's omm_ prefix rather than plainly, because
   !! openmm_dock already has a module-level Create_System and Fortran does
   !! not distinguish case.  These two are private today, so there is no clash
   !! yet -- but the reason this one exists at all is that a caller may later
   !! be allowed to supply the System, which would mean exporting it.
   !!
   !! On its own so that adopting a System made elsewhere -- by a script that
   !! wants to build the Context itself -- becomes a matter of not calling
   !! this, rather than of unpicking setup_system.
   !!
   !! The System takes ownership of every Force added to it, so those must not
   !! be destroyed separately.
   subroutine omm_create_system(init)
#if KEY_OPENMM == 1
      use fstore, only: store, fstore_clear_in_system
#endif /* KEY_OPENMM */

      implicit none

      logical, intent(in) :: init

      if (system_supplied .and. .not. init) then
         if (supplied_system_populated) then
            ! Filling it again would add every particle and every stored force
            ! a second time, so the energy would silently count the structure
            ! twice.  CHARMM cannot empty someone else's System, and OpenMM
            ! offers no way to, so refuse instead.
            call wrndie(-3, '<OMM_CREATE_SYSTEM>', &
                 'The supplied OpenMM System has already been built and ' // &
                 'cannot be built again.  Supply a fresh System (and a ' // &
                 'Context for it), or clear OpenMM first with OMM CLEAR.')
            return
         end if
         system = supplied_system
         charmm_owns_system = .false.
      else
         call OpenMM_System_create(system)
         charmm_owns_system = .true.
         ! A System just created contains none of the stored forces, whatever
         ! an earlier one contained.  Without this the store still believes
         ! they are in place, fstore_setup skips them all, and the energy
         ! comes back as zero -- which is what happened when the Context was
         ! released and CHARMM built itself a fresh System.
         call fstore_clear_in_system(store)
      end if

   end subroutine omm_create_system

   !> @brief Fill the OpenMM System from CHARMM's structure and options.
   !!
   !! This half is CHARMM's regardless of who created the System: the PSF, the
   !! restraints, the stored forces and the temperature and pressure control
   !! all come from CHARMM's own state, and nothing outside can supply them.
   !!
   !! @param[in] init    true while probing platform defaults; only the
   !!                    particles are imported and everything else is skipped.
   !! @param[in] ommseed random seed for the thermostat and barostat, taken
   !!                    from CHARMM's generator so a run stays reproducible.
   subroutine omm_populate_system(init, ommseed)
      use omm_bonded, only: import_psf
      use omm_restraint, only: setup_restraints
#if KEY_OPENMM == 1
      use fstore, only: fstore_setup
#endif /* KEY_OPENMM */

      implicit none

      logical, intent(in) :: init
      integer*4, intent(in) :: ommseed

      call import_psf(system, nbopts)

      if(.not. init) then
         call setup_restraints(system, nbopts%periodic)
         call setup_cm_freezer(system)
         call setup_thermostat(system, ommseed)
         call setup_barostat(system, ommseed)
#if KEY_OPENMM == 1
         call fstore_setup(system)
#endif /* KEY_OPENMM */
      endif

      ! Only the real build counts.  The probe fills a System of its own and
      ! throws it away, so it must not mark the supplied one as used up.
      if (system_supplied .and. .not. init) supplied_system_populated = .true.

   end subroutine omm_populate_system

   !> Creates and returns an OpenMM integrator with the given options.
   type(OpenMM_Integrator) function new_integrator(opts, ommseed)
      use reawri, only: TIMEST


      type(omm_dynopts_t), intent(in) :: opts
      integer*4, intent(in) :: ommseed
      type(OpenMM_VerletIntegrator) :: verlet
#if OMM_VER < 82
      type(OpenMM_LangevinIntegrator) :: langevin
#else
      type(OpenMM_LangevinMiddleIntegrator) :: langevin
#endif /* OMM_VER */
      type(OpenMM_VariableVerletIntegrator) :: varverlet
      type(OpenMM_VariableLangevinIntegrator) :: varlangevin
      integer*4 :: ijunk

      ! Create particular integrator, and recast to generic one.
      if (opts%qLangevin) then
         if (opts%qVariable) then
            call OpenMM_VariableLangevinIntegrator_create(varlangevin, &
                  opts%temperatureReference, opts%frictionCoefficient, opts%variableTS_tol)
            call OpenMM_VariableLangevinIntegrator_setRandomNumberSeed(varlangevin, &
                 ommseed)
            new_integrator = transfer(varlangevin, OpenMM_Integrator(0))
         else
#if OMM_VER < 82
            call OpenMM_LangevinIntegrator_create(langevin, &
                  opts%temperatureReference, opts%frictionCoefficient, TIMEST)
            call OpenMM_LangevinIntegrator_setRandomNumberSeed(langevin, &
                 ommseed)
#else /* OMM_VER */
            call OpenMM_LangevinMiddleIntegrator_create(langevin, &
                  opts%temperatureReference, opts%frictionCoefficient, TIMEST)
            call OpenMM_LangevinMiddleIntegrator_setRandomNumberSeed(langevin, &
                 ommseed)
#endif /* OMM_VER */
            new_integrator = transfer(langevin, OpenMM_Integrator(0))
         endif
      else
         if (opts%qVariable) then
            call OpenMM_VariableVerletIntegrator_create(varverlet, opts%variableTS_tol)
            new_integrator = transfer(varverlet, OpenMM_Integrator(0))
         else
            call OpenMM_VerletIntegrator_create(verlet, TIMEST)
            new_integrator = transfer(verlet, OpenMM_Integrator(0))
         endif
      endif
   end function new_integrator

   !> Resets the temperature on an OpenMM Integrator.
   subroutine new_Temperature(opts)

     type(omm_dynopts_t), intent(in) :: opts
     type (OpenMM_Force) force
#if OMM_VER < 82
     type(OpenMM_LangevinIntegrator) :: langevin
#else
     type(OpenMM_LangevinMiddleIntegrator) :: langevin
#endif /* OMM_VER */
     type(OpenMM_VariableLangevinIntegrator) :: varlangevin
     type(OpenMM_MonteCarloBarostat) :: barostat
     type(OpenMM_MonteCarloAnisotropicBarostat) :: anisobarostat
     type(OpenMM_MonteCarloMembraneBarostat) :: membarostat

     real :: temperature
     type(OpenMM_AndersenThermostat) :: andersen
     character(len=64) :: paramName

     ! Create particular integrator, and recast to generic one.
     if (opts%qLangevin) then
        if (opts%qVariable) then
           varlangevin = transfer(integrator, OpenMM_VariableLangevinIntegrator(0))
           call OpenMM_VariableLangevinIntegrator_setTemperature( &
                varlangevin, opts%temperatureReference)
           temperature = OpenMM_VariableLangevinIntegrator_getTemperature( &
                varlangevin)
        else
#if OMM_VER < 82
           langevin = transfer(integrator, OpenMM_LangevinIntegrator(0))
           call OpenMM_LangevinIntegrator_setTemperature( &
                langevin, opts%temperatureReference)
           temperature = OpenMM_LangevinIntegrator_getTemperature( &
                langevin)
#else /* OMM_VER */
           langevin = transfer(integrator, OpenMM_LangevinMiddleIntegrator(0))
           call OpenMM_LangevinMiddleIntegrator_setTemperature( &
                langevin, opts%temperatureReference)
           temperature = OpenMM_LangevinMiddleIntegrator_getTemperature( &
                langevin)
#endif /* OMM_VER */
        endif
        if(prnlev>0) write(outu,*) "Langevin> updating the temperature into ",temperature,' K'
     endif
     if(opts%qPressure) then
        call OpenMM_System_getForce(system, indexBarostat, force)
        select case (dynopts%barostatType)
        case (BARO_ANISOTROPIC)
           anisobarostat = transfer(force, OpenMM_MonteCarloAnisotropicBarostat(0))
           call OpenMM_MonteCarloAnisotropicBarostat_setDefaultTemperature( &
                anisobarostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloAnisotropicBarostat_getDefaultTemperature( &
                anisobarostat)
        case (BARO_MEMBRANE)
           membarostat = transfer(force, OpenMM_MonteCarloMembraneBarostat(0))
           call OpenMM_MonteCarloMembraneBarostat_setDefaultTemperature( &
                membarostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloMembraneBarostat_getDefaultTemperature( &
                membarostat)
        case default
           barostat = transfer(force,OpenMM_MonteCarloBarostat(0))
#ifdef OPENMM_API_UPDATE
           call OpenMM_MonteCarloBarostat_setDefaultTemperature(barostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloBarostat_getDefaultTemperature(barostat)
#else
           call OpenMM_MonteCarloBarostat_setTemperature(barostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloBarostat_getTemperature(barostat)
#endif
        end select
        ! The calls above set the FORCE's default, which OpenMM reads only
        ! when a Context is constructed.  A Context that already exists keeps
        ! using the value it was built with unless the corresponding context
        ! parameter is set as well -- so without this the barostat goes on
        ! sampling at the old temperature.  Verified: with the per-command
        ! rebuild suppressed, the force default read back 450 K while the
        ! live context still reported 298 K.
        select case (dynopts%barostatType)
        case (BARO_ANISOTROPIC)
           call OpenMM_MonteCarloAnisotropicBarostat_Temperature(paramName)
        case (BARO_MEMBRANE)
           call OpenMM_MonteCarloMembraneBarostat_Temperature(paramName)
        case default
           call OpenMM_MonteCarloBarostat_Temperature(paramName)
        end select
        call OpenMM_Context_setParameter(context, trim(paramName), &
             dble(opts%temperatureReference))
        if(prnlev>0) write(outu,*) "Pressure> updating the temperature into ",temperature,' K'
     endif
     ! The Andersen thermostat is a force too, and was not updated here at
     ! all -- so an Andersen run whose temperature changed without a rebuild
     ! kept thermostatting at the original temperature.  Same treatment:
     ! the force default for any later rebuild, and the context parameter
     ! for the Context in use now.
     if (opts%qAndersen .and. indexAndersen >= 0) then
        call OpenMM_System_getForce(system, indexAndersen, force)
        andersen = transfer(force, OpenMM_AndersenThermostat(0))
        call OpenMM_AndersenThermostat_setDefaultTemperature( &
             andersen, opts%temperatureReference)
        call OpenMM_AndersenThermostat_Temperature(paramName)
        call OpenMM_Context_setParameter(context, trim(paramName), &
             dble(opts%temperatureReference))
        if (prnlev > 0) write (outu, *) &
             "Andersen> updating the temperature into ", &
             OpenMM_AndersenThermostat_getDefaultTemperature(andersen), ' K'
     endif
   end subroutine new_Temperature

   subroutine get_plugins(plugin_dir)
     use stream, only: prnlev, outu
     character(len=*), intent(in) :: plugin_dir
     type(OpenMM_StringArray) :: plugin_list
     character(len=200) :: plugin_name
     integer*4 :: ii, n

     if (PRNLEV >= 6) write (OUTU, "(1x, 2a)") &
          'In OpenMM plugin directory ', trim(plugin_dir)
     call OpenMM_Platform_loadPluginsFromDirectory(plugin_dir, plugin_list)
     n = OpenMM_StringArray_getSize(plugin_list)
     if (n>0) then
        if(prnlev>=6) write(OUTU, "(1x, 'Using OpenMM plugins:')")
        do ii=1,n
           call OpenMM_StringArray_get(plugin_list, ii, plugin_name)
           if(prnlev>=6) write(OUTU, "(1x, i4, 1x, A)") ii, TRIM(plugin_name)
        enddo
     else
        if(prnlev>=6) write(OUTU, "(1x, A, 1x, A)") &
             'Found no OpenMM plugins in', trim(plugin_dir)
     endif
     if(prnlev>=6) write(OUTU, "(/)")
     call OpenMM_StringArray_destroy(plugin_list)
   end subroutine get_plugins

   subroutine load_env_var_plugins(env_var, prev_dir)
     implicit none
     character(len=*), intent(in) :: env_var, prev_dir
     character(len=1024) :: plugin_dir = ""

     call getenv(env_var, plugin_dir)
     if ( (trim(plugin_dir) /= "") .and. &
          (trim(plugin_dir) /= trim(prev_dir))) then
        call get_plugins(plugin_dir)
     end if
   end subroutine load_env_var_plugins

   !> Loads any shared libraries containing GPU implementations.
   subroutine load_libs()
     use new_timer
     use stream, only: prnlev, outu

     implicit none

     character(len=1024) :: dir_name, plugin_dir
     character(len=10) :: omm_version
     integer :: ii
     logical, save :: loaded = .false.

     ! include plugin locations set at compile time
     include "plugin_locs.f90"

     if (loaded)  return
     call timer_start(T_omm_load)
     call OpenMM_Platform_getOpenMMVersion(omm_version)
     ! convert omm_version into integer
     ii = index(omm_version,".") - 1
     read(omm_version(ii:ii),*) openmm_version_number
     if (PRNLEV >= 2) write (OUTU, "(1x, 2a)") &
          'OpenMM version ', trim(omm_version)

     ! directories set in plugin_locs.f90 at compile time
     call get_plugins(OPENMM_PLUGIN_DIR)
     call get_plugins(CHARMM_PLUGIN_DIR)
#if KEY_OMMTORCH == 1
     ! Only load TORCH_PLUGIN_DIR if it's different from already loaded dirs
     if (trim(TORCH_PLUGIN_DIR) /= trim(OPENMM_PLUGIN_DIR) .and. &
         trim(TORCH_PLUGIN_DIR) /= trim(CHARMM_PLUGIN_DIR) .and. &
         len_trim(TORCH_PLUGIN_DIR) > 0) then
        call get_plugins(TORCH_PLUGIN_DIR)
     endif
#endif

     ! get plugins from OpenMMs default locations
     ! and environment variable CHARMM_PLUGIN_DIR
     call load_env_var_plugins("OPENMM_PLUGIN_DIR", OPENMM_PLUGIN_DIR)
     call load_env_var_plugins("CHARMM_PLUGIN_DIR", CHARMM_PLUGIN_DIR)

     call load_device_info()
     call timer_stop(T_omm_load)

     loaded = .true.

   end subroutine load_libs

   subroutine load_device_info()
     use stream, only: prnlev, outu
     character(len=80) :: cuda_compiler
     character(len=20) :: device_prop, omm_precision
     character(len=10) :: platform_name, device_id, opencl_impl
     type(OpenMM_Platform) :: platform
     integer*4 :: ii, n

     device_id = ''
     omm_precision = ''
     call getenv('OPENMM_DEVICE', device_id)
     call getenv('OPENCL_PLATFORM', opencl_impl)
     call getenv('CUDA_COMPILER', cuda_compiler)
     call getenv('OPENMM_PRECISION', omm_precision)
     if(trim(nbopts%omm_deviceid) /= '' .and. &
          trim(nbopts%omm_deviceid)/=trim(device_id)) then
        device_id = nbopts%omm_deviceid
     else
        nbopts%omm_deviceid = device_id
     endif
     if(trim(nbopts%omm_precision) /= '' .and. &
          trim(nbopts%omm_precision)/=trim(omm_precision)) then
        omm_precision = nbopts%omm_precision
     else
        nbopts%omm_precision = omm_precision
     endif

     n = OpenMM_Platform_getNumPlatforms()
     do ii = 0, n-1
        call OpenMM_Platform_getPlatform(ii, platform)
        call OpenMM_Platform_getName(platform, platform_name)
        if (PRNLEV >= 2) write (OUTU, '(x,a,i0,2a)') &
             'Available OpenMM platform ', ii, ': ', platform_name
        device_prop = device_prop_name(platform_name)
        call set_defaultprop(platform, device_prop, device_id)
        if (trim(platform_name) == 'OpenCL') then
           call set_defaultprop(platform, 'OpenCLPlatformIndex', opencl_impl)
           call set_defaultprop(platform, 'OpenCLPrecision', omm_precision)
        endif
        if (trim(platform_name) == 'CUDA') then
           call set_defaultprop(platform, 'CudaCompiler', cuda_compiler)
           call set_defaultprop(platform, 'CudaPrecision', omm_precision)
        endif
     enddo
    end subroutine load_device_info

   !> Disables translation and rotation of the system's center of mass.
   subroutine setup_cm_freezer(system)
      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_CMMotionRemover) :: cm_freezer
      integer :: ijunk

      if (cm_stop_freq > 0) then
         call OpenMM_CMMotionRemover_create(cm_freezer, cm_stop_freq)
         ijunk = OpenMM_System_addForce(system, &
               transfer(cm_freezer, OpenMM_Force(0)))
      endif
   end subroutine setup_cm_freezer

   !> Adds a thermostat to the system if constant temperature is requested.
   subroutine setup_thermostat(system, ommseed)
      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: ommseed
      type(OpenMM_AndersenThermostat) :: andersen

      indexAndersen = -1
      if (dynopts%qAndersen) then
         call OpenMM_AndersenThermostat_create(andersen, &
               dynopts%temperatureReference, dynopts%collisionFrequency)
         call OpenMM_AndersenThermostat_setRandomNumberSeed(andersen, &
              ommseed)
         ! Keep the index: new_Temperature needs to find this force again to
         ! update its temperature in place, exactly as it does the barostat.
         indexAndersen = OpenMM_System_addForce(system, &
               transfer(andersen, OpenMM_Force(0)))
      endif
   end subroutine setup_thermostat

   !> Adds a barostat to the system if constant pressure is requested.
   subroutine setup_barostat(system, ommseed)
      use number
      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: ommseed
      type(OpenMM_MonteCarloBarostat) :: barostat
      type(OpenMM_MonteCarloAnisotropicBarostat) :: anisobarostat
      type(OpenMM_MonteCarloMembraneBarostat) :: membarostat
      integer*4 :: ijunk
      real*8 :: pressure3(3)
      integer*4 :: xymode, zmode
      ! Convert surface tension from dyne/cm to bar*nm
      real(chm_real) :: SFACT = 98.6923
#if OMM_VER >= 86
      ! OpenMM 8.6 added a scaleMoleculesAsRigid argument to all three Monte
      ! Carlo barostat constructors, choosing whether a volume move scales the
      ! centroid of each molecule (keeping the molecule rigid) or every atom
      ! independently. Every earlier OpenMM scaled molecules rigidly with no
      ! choice in the matter, and that is also the C++ API default, so passing
      ! OpenMM_True keeps the pressure coupling identical to what CHARMM has
      ! always done here. Scaling atoms independently would stretch bonds and
      ! is not what a rigid-water constant-pressure run expects.
      integer*4, parameter :: SCALE_MOLECULES_AS_RIGID = OpenMM_True
#endif /* OMM_VER >= 86 */

      indexBarostat = -1
      if (.not. dynopts%qPressure) return

      select case (dynopts%barostatType)
      case (BARO_ANISOTROPIC)
         ! Modes 1 & 2: per-axis pressure control, no surface tension
         ! Mode 1 (PRZZ only): scaleX=0, scaleY=0, scaleZ=1
         ! Mode 2 (PRXX+PRYY): scaleX=1, scaleY=1, scaleZ=0
         pressure3 = dynopts%pressureIn3Dimensions
         call OpenMM_MonteCarloAnisotropicBarostat_create(anisobarostat, &
               pressure3, dynopts%temperatureReference, &
               merge(1, 0, pressure3(1) /= ZERO), &
               merge(1, 0, pressure3(2) /= ZERO), &
               merge(1, 0, pressure3(3) /= ZERO), &
#if OMM_VER >= 86
               dynopts%pressureFrequency, SCALE_MOLECULES_AS_RIGID)
#else /**/
               dynopts%pressureFrequency)
#endif
         call OpenMM_MonteCarloAnisotropicBarostat_setRandomNumberSeed( &
              anisobarostat, ommseed)
         indexBarostat = OpenMM_System_addForce(system, &
               transfer(anisobarostat, OpenMM_Force(0)))

      case (BARO_MEMBRANE)
         ! Modes 3 & 4: surface tension active, XY coupled
         ! Mode 3 (TENS only):  XYIsotropic + ConstantVolume, pressure=0
         ! Mode 4 (TENS+PRZZ): XYIsotropic + ZFree, pressure=Pz
         xymode = OpenMM_MonteCarloMembraneBarostat_XYIsotropic
         if (dynopts%pressureIn3Dimensions(3) /= ZERO) then
            zmode = OpenMM_MonteCarloMembraneBarostat_ZFree
         else
            zmode = OpenMM_MonteCarloMembraneBarostat_ConstantVolume
         endif
         call OpenMM_MonteCarloMembraneBarostat_create(membarostat, &
               dynopts%pressureIn3Dimensions(3), &
               dynopts%surfaceTension * SFACT / OpenMM_AngstromsPerNm, &
               dynopts%temperatureReference, xymode, zmode, &
#if OMM_VER >= 86
               dynopts%pressureFrequency, SCALE_MOLECULES_AS_RIGID)
#else /**/
               dynopts%pressureFrequency)
#endif
         call OpenMM_MonteCarloMembraneBarostat_setRandomNumberSeed( &
              membarostat, ommseed)
         indexBarostat = OpenMM_System_addForce(system, &
               transfer(membarostat, OpenMM_Force(0)))

      case default
         ! Isotropic MC barostat
         call OpenMM_MonteCarloBarostat_create(barostat, &
              dynopts%pressureReference, dynopts%temperatureReference, &
#if OMM_VER >= 86
              dynopts%pressureFrequency, SCALE_MOLECULES_AS_RIGID)
#else /**/
              dynopts%pressureFrequency)
#endif
         call OpenMM_MonteCarloBarostat_setRandomNumberSeed(barostat, ommseed)
         indexBarostat = OpenMM_System_addForce(system, &
               transfer(barostat, OpenMM_Force(0)))
      end select
   end subroutine setup_barostat

   !> Enables constraints on atom pair lengths.
   ! Actual algorithm is not SHAKE but CCMA, described in
   ! Eastman and Pande, JCTC Feb 2010, DOI 10.1021/ct900463w
   subroutine setup_shake(system, integrator)
      use shake
      use psf, only: pair_fixed_atoms

      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_Integrator), intent(inout) :: integrator
      integer :: iatom, jatom, icnst, iadded
      real(chm_real) :: r

      if (QSHAKE) then
         do icnst = 1, NCONST
            iatom = SHKAPR(1, icnst) - 1
            jatom = SHKAPR(2, icnst) - 1
            if(.not.pair_fixed_atoms(iatom+1,jatom+1)) then
               r = sqrt(CONSTR(icnst)) / OpenMM_AngstromsPerNm
               iadded = OpenMM_System_addConstraint(system, &
                    iatom, jatom, r)
            endif
         enddo
         call OpenMM_Integrator_setConstraintTolerance(integrator, SHKTOL)
      endif
   end subroutine setup_shake

   logical function omm_platform_ok(platform_name)
     use omm_glblopts, only: qnocpu
     implicit none
     character(len=*), intent(in) :: platform_name

     omm_platform_ok = .true.

     if (qnocpu .and. &
          platform_name(1:4) .ne. 'CUDA' .and. &
          platform_name(1:6) .ne. 'OpenCL') then
        ! BIOVIA Code Start : Bug fix
        call wrndie(-3, '<omm_main%omm_platform_ok>', 'No GPU platform can be initialized')
        ! BIOVIA Code End
        omm_platform_ok = .false.
     end if
   end function omm_platform_ok

   subroutine init_context()
     use new_timer, only: timer_start, timer_stop, t_omm_ctx
     use OpenMM
     use omm_util, only: omm_platform_getplatformbyname

     implicit none

     type(OpenMM_Platform) :: platform
     character(len=10) :: platformName, deviceId, implId
     character(len=20) :: deviceProp, implProp
     character(len=1024) :: env_string

     integer :: omm_success = 1
     logical :: err = .false.

      ! A Context supplied by the caller is already built, on the System
      ! CHARMM filled in.  Building another would leave CHARMM driving one
      ! the caller cannot see, so there is nothing to do here.
      if (context_supplied) then
         context = supplied_context
         call OpenMM_Context_getIntegrator(context, integrator)
         return
      end if
      call timer_start(T_omm_ctx)

      env_string = ''
      call getenv('OPENMM_PLATFORM', env_string)

      if(trim(nbopts%omm_platform) /= '' .and. &
           trim(nbopts%omm_platform) /= trim(env_string)) then
         env_string = nbopts%omm_platform
      else
         nbopts%omm_platform = trim(env_string)
      endif

      if (trim(env_string) /= '') then
         ! Try to use user specific platform
         ! call OpenMM_Platform_getPlatformByName(env_string, platform)
         omm_success = omm_platform_getplatformbyname(trim(env_string), platform)

         if (omm_success .eq. 0) then
            call wrndie(-3, '<omm_main%init_context>', &
                 'platform ' // trim(env_string) // ' unavailable')
            return
         end if

         call OpenMM_Platform_getName(platform, platformName)
         call load_device_info()

         call OpenMM_Context_create_2(context, &
               system, integrator, platform)
      else
         ! Let OpenMM Context choose best platform.
         call OpenMM_Context_create(context, &
               system, integrator)
         call OpenMM_Context_getPlatform(context, platform)
      endif
      charmm_owns_context = .true.

      call OpenMM_Platform_getName(platform, platformName)
      err = omm_platform_ok(platformName)

      if (PRNLEV >= 2) then
         write (OUTU, '(2a)') &
              'Init_context: Using OpenMM platform ', platformName
         if (trim(platformName) == 'OpenCL') then
            call show_prop(platform, 'OpenCLPlatformIndex')
            call show_prop(platform, 'OpenCLPrecision')
         endif
         deviceProp = device_prop_name(platformName)
         call show_prop(platform, deviceProp)
         if (trim(platformName) == 'CUDA') then
            ! cuda2, unreleased as of Sep 2012
            call show_prop(platform, 'CudaCompiler')
            call show_prop(platform, 'CudaPrecision')
         endif
      endif
      call timer_stop(T_omm_ctx)
    end subroutine init_context

   subroutine get_PlatformDefaults
     use omm_glblopts
     use omm_util, only: omm_platform_getplatformbyname

     implicit none

     type(OpenMM_Platform) :: platform
     integer :: prnlev_current, omm_success = 1
     character(len=10) :: env_string = '', platformName = '', deviceId = '', &
             devicePrecision = ''
     character(len=20) :: deviceProp
     logical, save :: setdefaults = .false.

     logical :: err = .false.

     if(setdefaults) return
     if(prnlev>=2) &
          write(OUTU,'(a)') &
          'get_PlatformDefaults> Finding default platform values'

     prnlev_current = prnlev
     prnlev = 0

     call check_system()
     call check_nbopts()
     call load_libs()
     call setup_system(.true.)

     env_string = ''
     call getenv('OPENMM_PLATFORM', env_string)

     if (trim(env_string) /= '') then ! Try to use user specific platform
        ! call OpenMM_Platform_getPlatformByName(env_string, platform)
        omm_success = omm_platform_getplatformbyname(trim(env_string), platform)

        if (omm_success .eq. 0) then
           call wrndie(-3, '<omm_main%get_platformdefaults>', &
                'platform ' // trim(env_string) // ' unavailable')
           return
        end if

        call OpenMM_Platform_getName(platform, platformName)
        call OpenMM_Context_create_2(context, &
               system, integrator, platform)
     else ! Let OpenMM Context choose best platform.
        call OpenMM_Context_create(context, &
               system, integrator)
        call OpenMM_Context_getPlatform(context, platform)
     endif

     call OpenMM_Platform_getName(platform, platformName)
     err = omm_platform_ok(platformName)

     if (trim(platformName) /= 'Reference' .and. &
         trim(platformName) /= 'CPU') then
        deviceProp = device_prop_name(platformName)
        call OpenMM_Platform_getPropertyValue(platform, &
             context, trim(deviceProp), deviceId)
     endif

     if (trim(platformName) == 'OpenCL') then
        call OpenMM_Platform_getPropertyValue(platform, &
             context, 'OpenCLPrecision', devicePrecision)
     elseif (trim(platformName) == 'CUDA') then
        call OpenMM_Platform_getPropertyValue(platform, &
             context, 'CudaPrecision', devicePrecision)
     endif

     omm_default_platform = platformName
     omm_default_precision = devicePrecision
     omm_default_deviceid = deviceId

     prnlev = prnlev_current

     if(prnlev>=2) &
          write(OUTU,'(6a)') &
          'get_PlatformDefaults> Default values found: platform=', &
          trim(platformName), ' precision=',trim(devicePrecision), &
          ' deviceid=', trim(deviceId)

     call OpenMM_Context_destroy(context)
     call OpenMM_Integrator_destroy(integrator)
     call OpenMM_System_destroy(system)

     setdefaults = .true.
   end subroutine get_PlatformDefaults

   subroutine set_defaultprop(platform, propName, propValue)
      type(OpenMM_Platform), intent(in) :: platform
      character(len=*), intent(in) :: propName, propValue

      if (trim(propName) /= '' .and. trim(propValue) /= '') then
         call OpenMM_Platform_setPropertyDefaultValue(platform, &
               trim(propName), trim(propValue))
      ! else use default value
      endif
    end subroutine set_defaultprop

    subroutine show_prop(platform, propName)
      type(OpenMM_Platform), intent(in) :: platform
      character(len=*), intent(in) :: propName
      character(len=80) :: propValue

      if (trim(propName) /= '') then
         call OpenMM_Platform_getPropertyValue(platform, &
               context, trim(propName), propValue)
         if(prnlev>=2) write (OUTU, '(7x,3a)') &
               trim(propName), ' = ', trim(propValue)
      endif
   end subroutine show_prop

   character(len=20) function device_prop_name(platform_name)
      character(len=*), intent(in) :: platform_name

      if (trim(platform_name) == 'Cuda') then ! Leave for back compatability?
         device_prop_name = 'CudaDevice'
      else if (trim(platform_name) == 'CUDA') then
         device_prop_name = 'CudaDeviceIndex'
      else if (trim(platform_name) == 'OpenCL') then
         device_prop_name = 'OpenCLDeviceIndex'
      else  ! 'Reference' etc.
         device_prop_name = ''
      endif
   end function device_prop_name

   ! Output current state information.
   subroutine show_state(pos, vel_t, vel_pre, &
         istep, istart, istop, npriv, ndegf, isvfrq, navestps, jhtemp, qkuhead)
      use consta, only: KBOLTZ
      use contrl, only: NPRINT
      use coord  ! XXX writes
      use deriv  ! XXX writes
      use energym
      use averfluc
      use avfl_ucell
      use image, only: xtltyp, xucell
      use psf
      use reawri, only: NSTEP, TIMEST
      use new_timer
      use omm_ecomp, only : omm_assign_eterms

      real*8, intent(inout) :: pos(3, NATOM), vel_t(3, NATOM), vel_pre(3, NATOM)
      integer*4, intent(in) :: istep, istart, istop, npriv, ndegf, isvfrq
      integer*4, intent(inout) :: navestps
      logical, intent(inout) :: qkuhead
      real(chm_real), intent(inout) :: jhtemp
      type(OpenMM_State) :: state
      real*8 :: vel_post(3, NATOM)
      real*8 :: box_a(3), box_b(3), box_c(3)
      logical :: lhdr
      integer*4 :: data_wanted
      integer*4 :: enforce_periodic
      real*8 :: timeInPs, eP, eK
      real*8 :: temperature

      call timer_start(T_energy)
      data_wanted = ior( &
            ior(OpenMM_State_Positions, OpenMM_State_Velocities), &
            ior(OpenMM_State_Forces, OpenMM_State_Energy))
      enforce_periodic = OpenMM_False
      if (nbopts%periodic) enforce_periodic = OpenMM_True
      call OpenMM_Context_getState(context, &
            data_wanted, enforce_periodic, state)

      if (dynamics_mode == DYN_CONTINUE) then
         pos = get_positions(state)
         call fetch_velocities(state, vel_pre, vel_post)
#if KEY_PHMD==1
         if (qphmd_omm .and. qphmd_initialized) call get_lambda_state(context)
#endif
         vel_t = (vel_pre + vel_post) / TWO
      ! else use velocities given by CHARMM
      endif

      if(istep == 0 .or. (istep>=istart)) then
         ! In dynamics we need to zero these energy terms between calls.
         ! The previous total comes from our own copy, not from EPROP(TOTE):
         ! see prev_report_tote above.
         EPROP(TEPR) = prev_report_tote
         EPROP(EPOT) = ZERO
         EPROP(TOTKE) = ZERO
         EPROP(TOTE) = ZERO
         call export_energy(state, vel_t)
         prev_report_tote = EPROP(TOTE)
         eP = EPROP(EPOT)
         eK = EPROP(TOTKE)
         temperature = 2 * eK / (NDEGF * KBOLTZ)
         EPROP(TEMPS) = temperature
         jhtemp = jhtemp + temperature
         if (dynopts%qPressure) then     ! Periodic box lengths needed for CPT
            call export_periodic_box(state)
            call avfl_ucell_update()
         endif
         call omm_assign_eterms(context, enforce_periodic)
         navestps = navestps + 1
         call avfl_update(eprop, eterm, epress)
         if (todo_now(NPRINT, istep) .or. istep == istop) then
            if (PRNLEV >= 6) then
               write (OUTU, '(/,x,a,i9,3x,a,f12.3,2x,a,f9.2)') &
                     'ISTEP =', istep, 'TIME(PS) =', npriv*TIMEST, &
                     'TEMP(K) =', temperature
               write (OUTU, '(x,a,f15.5,2(2x,a,f15.5))') &
                     'Etot  =', eK+eP, 'EKtot  =', eK, 'EPtot  =', eP
            endif
            call kunit_write(istep,npriv,qkuhead)
            lhdr = istep == 0
            if(prnlev>0) then
               call printe(outu,eprop,eterm,'DYNA','DYN',lhdr, &
                 istep,npriv*timest,zero,.true.)
            end if

            if (dynopts%qPressure) call prnxtld(outu,'DYNA',xtltyp,xucell,.true.,zero, &
                 .true.,epress)
         endif
      endif
      call set_forces(state,dx,dy,dz)
      call OpenMM_State_destroy(state)
      call timer_stop(T_energy)
   end subroutine show_state

   logical function todo_now(freq, istep)
      integer, intent(in) :: freq, istep
      if (freq > 0) then
         todo_now = mod(istep, freq) == 0
      else
         todo_now = .false.
      endif
   end function todo_now

   !> Sets CHARMM energies and forces for the given coordinates.
   subroutine omm_energy(x, y, z)
     use omm_ecomp, only : omm_assign_eterms
     use omm_nbopts, only: current_nbopts
     use deriv  ! XXX writes
     use energym  ! XXX writes
     use new_timer
     real(chm_real), intent(in) :: x(:), y(:), z(:)
     real(chm_real) :: Epterm
     type(OpenMM_State) :: state
     real*8 :: pos(3, NATOM)
     integer*4 :: data_wanted
     integer*4 :: enforce_periodic
     integer*4 :: itype, group

     call setup_openmm()
     pos = get_xyz(X, Y, Z) / OpenMM_AngstromsPerNm
     call set_positions(context, pos)
#if KEY_PHMD==1
     if (qphmd_omm .and. qphmd_initialized) call set_lambda_state(context)
#endif
     if (nbopts%periodic) call import_periodic_box()

     data_wanted = ior(OpenMM_State_Energy, OpenMM_State_Forces)
     enforce_periodic = OpenMM_False
     if (nbopts%periodic) enforce_periodic = OpenMM_True
     call timer_start(T_energy)
     call OpenMM_Context_getState(context, data_wanted, enforce_periodic, state)
     call export_energy(state)
     call export_forces(state, DX, DY, DZ)
     call OpenMM_State_destroy(state)
     call omm_assign_eterms(context, enforce_periodic)

      call timer_stop(T_energy)
      ! Re-snapshot nbopts after the energy evaluation, which may
      ! regenerate nonbond lists and change the nonbonded state.
      nbopts = current_nbopts()
   end subroutine omm_energy

   !> Sets CHARMM energies and forces for the given coordinates.
   subroutine omm_minimize(x, y, z, nsteps, tolerance)
     use, intrinsic::iso_c_binding, only: c_null_ptr
     use omm_ecomp, only : omm_assign_eterms
     use energym  ! XXX writes
     use new_timer

     implicit none

     real(chm_real), intent(inout) :: x(:), y(:), z(:)
     real(chm_real), intent(in) :: tolerance
     integer*4, intent(in) :: nsteps
     real(chm_real) :: Epterm
     type(OpenMM_State) :: state
     real*8 :: pos(3, NATOM)
     integer*4 :: data_wanted
     integer*4 :: enforce_periodic
     integer*4 :: itype, group

     call setup_openmm()
     pos = get_xyz(X, Y, Z) / OpenMM_AngstromsPerNm
     call set_positions(context, pos)
     if (nbopts%periodic) call import_periodic_box()

     data_wanted = ior(OpenMM_State_Positions, OpenMM_State_Energy)
     enforce_periodic = OpenMM_False
     if (nbopts%periodic) enforce_periodic = OpenMM_True
     call timer_start(T_omm)
     call OpenMM_LocalEnergyMinimizer_Minimize(context, &
          tolerance * OpenMM_AngstromsPerNm / OpenMM_KcalPerKJ, &
          nsteps &
#if OMM_VER > 80
          , transfer(c_null_ptr, OpenMM_MinimizationReporter(0)) &
#endif /* OMM_VER > 80 */
     )
     call OpenMM_Context_getState(context, data_wanted, enforce_periodic, state)
     call export_energy(state)
     pos = get_positions(state)
     call set_xyz(X, Y, Z, pos * OpenMM_AngstromsPerNm, &
          nbopts%periodic)

     call OpenMM_State_destroy(state)
     call omm_assign_eterms(context, enforce_periodic)
      call timer_stop(T_omm)

   end subroutine omm_minimize

   !> Sets CHARMM energy variables according to the given OpenMM state.
   ! Note, no need to zero the masked terms since this gives incorrect
   ! results for energy calls and is irrelevant for dynamics
   subroutine export_energy(state, vel)
      use energym  ! XXX writes
      type(OpenMM_State), intent(in) :: state
      real*8, intent(in), optional :: vel(:,:)
      real*8 :: box_a(3), box_b(3), box_c(3)
      real(chm_real) :: eP, eK

      eP = OpenMM_State_getPotentialEnergy(state) / OpenMM_KJPerKcal
      if (present(vel)) then
         eK = kinetic_energy(vel) / OpenMM_KJPerKcal
      else
         eK = OpenMM_State_getKineticEnergy(state) / OpenMM_KJPerKcal
      endif
      EPROP(TOTKE) = EPROP(TOTKE) + eK
      EPROP(EPOT) = EPROP(EPOT) + eP
      EPROP(TOTE) = EPROP(TOTE) + eK + eP
      if (nbopts%periodic) then
         call OpenMM_State_getPeriodicBoxVectors(state, &
               box_a, box_b, box_c)
         EPROP(VOLUME) = OpenMM_AngstromsPerNm**3 * (box_a(1) * box_b(2) * box_c(3))
      endif
   end subroutine export_energy

   !> Computes kinetic energy from a set of velocities.
   function kinetic_energy(vel)
      use psf, only: AMASS
      real(chm_real) :: kinetic_energy  ! kJ
      real*8, intent(in) :: vel(:,:)
      real(chm_real) :: vsq(NATOM)

      vsq = sum(vel**2, dim=1)
      kinetic_energy = dot_product(AMASS(1:NATOM), vsq) / TWO
   end function kinetic_energy

   !> Retrieves velocities at (t - dt/2) and (t + dt/2) by
   !> integrating forward one step and restoring the original state.
   subroutine fetch_velocities(state0, vel_pre, vel_post)
      type(OpenMM_State), intent(in) :: state0
      real*8, intent(out) :: vel_pre(3, NATOM), vel_post(3, NATOM)
      type(OpenMM_Vec3Array) :: pos0, vel0
      real*8 :: time0

      time0 = OpenMM_State_getTime(state0)
      call OpenMM_State_getPositions(state0, pos0)
      call OpenMM_State_getVelocities(state0, vel0)
      vel_pre = get_v3a(vel0)

      call OpenMM_Integrator_step(integrator, 1)
      vel_post = new_velocities(context)

      call OpenMM_Context_setTime(context, time0)
      call OpenMM_Context_setPositions(context, pos0)
      call OpenMM_Context_setVelocities(context, vel0)
   end subroutine fetch_velocities

   !> Converts three parallel arrays into a single 3xN array.
   function get_xyz(x, y, z) result(dest)
      real*8 :: dest(3, NATOM)
      real(chm_real), intent(in) :: x(:), y(:), z(:)
      integer :: n

      n = size(dest, dim=2)
      dest(1, :) = x(1:n)
      dest(2, :) = y(1:n)
      dest(3, :) = z(1:n)
   end function get_xyz

   !> Copies a 3xN array into three parallel arrays.
   subroutine set_xyz(x, y, z, src, periodic)
     use image, only : imxcen, imycen, imzcen, &
          ntrans, imtrns, imname
     use bases_fcm, only: bimag
      real(chm_real), intent(out) :: x(:), y(:), z(:)
      real*8, intent(in) :: src(3, NATOM)
      logical :: periodic

      x(1:NATOM) = src(1, :)
      y(1:NATOM) = src(2, :)
      z(1:NATOM) = src(3, :)
!      if(periodic)  call imcent(imxcen, imycen, imzcen, bimag%imcenf, &
!           ntrans, imtrns, imname, x, y, z, 0, zero, zero, zero, &
!           zero, zero, zero, .false.)
   end subroutine set_xyz

   !> Converts a Vec3Array into a 3xN array.
   function get_v3a(v3a) result(dest)
      real*8 :: dest(3, NATOM)
      type(OpenMM_Vec3Array), intent(in) :: v3a
      real*8 :: vec_i(3)
      integer*4 :: i

      do i = 1, NATOM
         call OpenMM_Vec3Array_get(v3a, i, vec_i)
         dest(:, i) = vec_i
      enddo
   end function get_v3a

   !> Copies a 3xN array into a previously created Vec3Array.
   subroutine set_v3a(v3a, src)
      type(OpenMM_Vec3Array), intent(inout) :: v3a
      real*8, intent(in) :: src(3, NATOM)
      real*8 :: vec_i(3)
      integer*4 :: i

      do i = 1, NATOM
         vec_i = src(:, i)
         call OpenMM_Vec3Array_set(v3a, i, vec_i)
      enddo
   end subroutine set_v3a

   !> Adds forces from OpenMM to forces computed elswhere in CHARMM.
   subroutine export_forces(state, dx, dy, dz)
      type(OpenMM_State), intent(in) :: state
      real(chm_real), intent(inout) :: dx(:), dy(:), dz(:)
      type(OpenMM_Vec3Array) :: forces
      real*8 :: force_i(3)
      integer*4 :: i

      call OpenMM_State_getForces(state, forces)
      do i = 1, NATOM
         call OpenMM_Vec3Array_get(forces, i, force_i)
         force_i = -force_i / (OpenMM_KJPerKcal * OpenMM_AngstromsPerNm)
         dx(i) = dx(i) + force_i(1)
         dy(i) = dy(i) + force_i(2)
         dz(i) = dz(i) + force_i(3)
      enddo
   end subroutine export_forces

   !> Sets forces from OpenMM to forces computed elswhere in CHARMM.
   subroutine set_forces(state, dx, dy, dz)
      type(OpenMM_State), intent(in) :: state
      real(chm_real), intent(inout) :: dx(:), dy(:), dz(:)
      type(OpenMM_Vec3Array) :: forces
      real*8 :: force_i(3)
      integer*4 :: i

      call OpenMM_State_getForces(state, forces)
      do i = 1, NATOM
         call OpenMM_Vec3Array_get(forces, i, force_i)
         force_i = -force_i / (OpenMM_KJPerKcal * OpenMM_AngstromsPerNm)
         dx(i) = force_i(1)
         dy(i) = force_i(2)
         dz(i) = force_i(3)
!         write(*,*)'Forces on atom',i,'=',dx(i),dy(i),dz(i)
      enddo
      end subroutine set_forces

   !> Returns velocities from a new State of an OpenMM context.
   function new_velocities(context) result(vel)
      real*8 :: vel(3, NATOM)
      type(OpenMM_Context), intent(in) :: context
      type(OpenMM_State) :: state
      type(OpenMM_Vec3Array) :: velocities

      call OpenMM_Context_getState(context, &
            OpenMM_State_Velocities, OpenMM_False, state)
      call OpenMM_State_getVelocities(state, velocities)
      vel = get_v3a(velocities)
      call OpenMM_State_destroy(state)
   end function new_velocities

   subroutine set_velocities(context, vel)
      type(OpenMM_Context), intent(inout) :: context
      real*8, intent(in) :: vel(3, NATOM)  ! (t - dt/2)
      type(OpenMM_Vec3Array) :: velocities

      call OpenMM_Vec3Array_create(velocities, NATOM)
      call set_v3a(velocities, vel)
      call OpenMM_Context_setVelocities(context, velocities)
      call OpenMM_Vec3Array_destroy(velocities)
   end subroutine set_velocities

   !> Returns the change in velocity for half a timestep,
   !> assuming that the context's positions are current.
   function half_delta_vel()
      use psf, only: AMASS
      real(chm_real) :: half_delta_vel(3, NATOM)  ! nm/ps
      type(OpenMM_State) :: state
      type(OpenMM_Vec3Array) :: forces
      real(chm_real) :: accel(3, NATOM)
      real(chm_real) :: force_i(3), delta_t
      integer :: i

      call OpenMM_Context_getState(context, &
            OpenMM_State_Forces, OpenMM_False, state)
      call OpenMM_State_getForces(state, forces)
      do i = 1, NATOM
         if (AMASS(i) /= 0) then
            call OpenMM_Vec3Array_get(forces, i, force_i)
            accel(:, i) = force_i / AMASS(i)
         else
            accel(:, i) = ZERO
         endif
      enddo
      call OpenMM_State_destroy(state)
      delta_t = OpenMM_Integrator_getStepSize(integrator)
      half_delta_vel = accel * (delta_t / TWO)
   end function half_delta_vel

#if KEY_PHMD==1
   ! copy lambda parameters (pos, vel, force) from OpenMM to CHARMM
   subroutine get_lambda_state(context)
      use phmd, only : ntitr, ph_theta, thetaold, vph_theta, vphold, dphold
      implicit none
      type(OpenMM_Context), intent(inout) :: context
      type(openmm_doublearray) :: lambdatmp
      real*8 :: tmp, lambdastate(ntitr*5)
      integer :: i
      call OpenMM_doublearray_create(lambdatmp, ntitr*5)
      call OpenMMGBSW_GBSWForce_getLambdaState(gbswforce, context, lambdatmp)
      do i = 1, ntitr*5
         call OpenMM_doublearray_get(lambdatmp, i, tmp)
         lambdastate(i) = tmp
      enddo
      do i = 1, ntitr
         ph_theta(i)  = lambdastate(i)
         thetaold(i)  = lambdastate(i + ntitr)
         vph_theta(i) = lambdastate(i + ntitr*2)
         vphold(i)    = lambdastate(i + ntitr*3)
         dphold(i)    = lambdastate(i + ntitr*4)
      enddo
      call OpenMM_doublearray_destroy(lambdatmp)
   end subroutine get_lambda_state

   ! send lambda parameters (pos, vel, force) from CHARMM to OpenMM
   subroutine set_lambda_state(context)
      use phmd, only : ntitr, ph_theta, thetaold, vph_theta, vphold, dphold
      implicit none
      type(OpenMM_Context), intent(inout) :: context
      type(openmm_doublearray) :: lambdatmp
      real*8 :: tmp, lambdastate(ntitr*5)
      integer :: i
      call OpenMM_doublearray_create(lambdatmp, ntitr*5)
      do i = 1, ntitr
         lambdastate(i)           = ph_theta(i)
         lambdastate(i + ntitr)   = thetaold(i)
         lambdastate(i + ntitr*2) = vph_theta(i)
         lambdastate(i + ntitr*3) = vphold(i)
         lambdastate(i + ntitr*4) = dphold(i)
      enddo
      do i = 1, ntitr*5
         tmp = lambdastate(i)
         call OpenMM_doublearray_set(lambdatmp, i, tmp)
      enddo
      call OpenMMGBSW_GBSWForce_setLambdaState(gbswforce, context, lambdatmp)
      call OpenMM_doublearray_destroy(lambdatmp)
   end subroutine set_lambda_state
#endif /* KEY_PHMD */

   function get_positions(state) result(pos)
      real*8 :: pos(3, NATOM)
      type(OpenMM_State), intent(in) :: state
      type(OpenMM_Vec3Array) :: positions

      call OpenMM_State_getPositions(state, positions)
      pos = get_v3a(positions)
   end function get_positions

   subroutine set_positions(context, pos)
      type(OpenMM_Context), intent(inout) :: context
      real*8, intent(in) :: pos(3, NATOM)
      type(OpenMM_Vec3Array) :: positions

      call OpenMM_Vec3Array_create(positions, NATOM)
      call set_v3a(positions, pos)
      call OpenMM_Context_setPositions(context, positions)
      call OpenMM_Vec3Array_destroy(positions)
   end subroutine set_positions

   subroutine export_periodic_box(state)
      use omm_restraint, only: update_restraint_box
      use image, only: XUCELL, XTLABC  ! XXX writes
      type(OpenMM_State), intent(in) :: state
      real*8 :: box(3), box_a(3), box_b(3), box_c(3)
      integer :: i

      call OpenMM_State_getPeriodicBoxVectors(state, &
            box_a, box_b, box_c)
      box = [box_a(1), box_b(2), box_c(3)]
      XUCELL(1:3) = box * OpenMM_AngstromsPerNm
      XUCELL(4:6) = NINETY
      XTLABC = ZERO
      XTLABC(1) = XUCELL(1)
      XTLABC(3) = XUCELL(2)
      XTLABC(6) = XUCELL(3)
      call xtlmsr(XUCELL)
      call update_restraint_box(context, box)
   end subroutine export_periodic_box

   subroutine import_periodic_box()
      use image, only: XUCELL
      real*8 :: box(3), box_a(3), box_b(3), box_c(3)

      if (any(XUCELL(4:6) /= NINETY)) call wrndie(-1, 'OpenMM', &
            'Currently supports only orthorhombic lattices')
      box = XUCELL(1:3) / OpenMM_AngstromsPerNm
      box_a = ZERO
      box_b = ZERO
      box_c = ZERO
      box_a(1) = box(1)
      box_b(2) = box(2)
      box_c(3) = box(3)
      call OpenMM_Context_setPeriodicBoxVectors(context, &
            box_a, box_b, box_c)
   end subroutine import_periodic_box

   subroutine run_steps(istep, istop, npriv, isvfrq)
      use contrl, only: NPRINT, MDSTEP
      use reawri, only: JHSTRT, NSAVC, NSAVV, IUNCRD, IUNVEL, IUNWRI, TIMEST
      use new_timer
      type(OpenMM_VariableVerletIntegrator) :: varverlet
      type(OpenMM_VariableLangevinIntegrator) :: varlangevin
      integer*4, intent(inout) :: istep, npriv
      integer*4, intent(in) :: isvfrq, istop
      integer*4 :: nsteps_per_report
      real*8 :: endtime

      integer :: clock_start, clock_stop, clock_rate, clock_max
      real(chm_real) :: clock_diff, wall_s, sim_ps, ns_day

      call timer_start(T_dynamc)

      nsteps_per_report = istop
      if (NPRINT > 0) then
         nsteps_per_report = min(nsteps_per_report, NPRINT * (istep/NPRINT + 1))
      endif
      if (nsavc > 0 .and. iuncrd > 0) then
         nsteps_per_report = min(nsteps_per_report, nsavc*((istep/nsavc)+1))
      endif
      if (nsavv > 0 .and. iunvel > 0) then
         nsteps_per_report = min(nsteps_per_report, nsavv*((istep/nsavv)+1))
      endif
      if (isvfrq > 0 .and. iunwri > 0) then
         nsteps_per_report = min(nsteps_per_report, isvfrq*((istep/isvfrq)+1))
      endif
      nsteps_per_report = max(1, nsteps_per_report-istep)
      if (prnlev >= 6) write (OUTU, "(1x, 'Number of steps to integrate before next write:', i9,/)") &
            nsteps_per_report

      if (PRNLEV >= 6) call system_clock(clock_start)

#if KEY_PHMD==1
         if (qphmd_omm .and. qphmd_initialized) then
            call openmmgbsw_gbswforce_setdoingdynamics(gbswforce, context, 1)
         end if
#endif

      if (dynopts%qVariable) then
         endtime = (npriv + nsteps_per_report) * TIMEST
         ! TODO hide integrator subtype
         if (dynopts%qLangevin) then
            varlangevin = transfer(integrator, OpenMM_VariableLangevinIntegrator(0))
            call OpenMM_VariableLangevinIntegrator_stepTo(varlangevin, endtime)
         else
            varverlet = transfer(integrator, OpenMM_VariableVerletIntegrator(0))
            call OpenMM_VariableVerletIntegrator_stepTo(varverlet, endtime)
         endif
      else
         call OpenMM_Integrator_step(integrator, nsteps_per_report)
      endif

#if KEY_PHMD==1
         if (qphmd_omm .and. qphmd_initialized) then
            call openmmgbsw_gbswforce_setdoingdynamics(gbswforce, context, 0)
         end if
#endif

      if (PRNLEV >= 6) then
         call system_clock(clock_stop, clock_rate, clock_max)
         clock_diff = clock_stop - clock_start
         if (clock_diff < ZERO) clock_diff = clock_diff + TWO * clock_max
         wall_s = clock_diff / clock_rate
         sim_ps = nsteps_per_report * TIMEST
         ns_day = 86.4_chm_real * sim_ps / wall_s
         write (OUTU, '(A,F0.2,A,F0.2,A)') &
               'elapsed = ', wall_s, ' s, rate = ', ns_day, ' ns/day'
      endif

      istep = istep + nsteps_per_report
      npriv = npriv + nsteps_per_report
      jhstrt = jhstrt + nsteps_per_report
      mdstep = mdstep + nsteps_per_report

      call timer_stop(T_dynamc)
   end subroutine run_steps

   subroutine traj_write(istep, npriv, ndegf, vx, vy, vz)
      use coord
      use cvio, only: writcv
      use ctitla, only: NTITLA, TITLEA
      use psf, only: CG, IMOVE
      use reawri, only: NSTEP, DELTA, NSAVC, NSAVV, IUNCRD, IUNVEL
      integer, intent(in) :: istep, npriv, ndegf
      real(chm_real), intent(in) :: vx(:), vy(:), vz(:)

      if (IUNCRD > 0 .and. todo_now(NSAVC, istep)) then
         call writcv(X, Y, Z,  &
#if KEY_CHEQ==1
               CG, .false.,  &
#endif
               NATOM, IMOVE, NATOM, npriv, istep, ndegf, DELTA,  &
               NSAVC, NSTEP, TITLEA, NTITLA, IUNCRD, .false.,  &
               .false., [0], .false., [ZERO])
      endif
      if (IUNVEL > 0 .and. todo_now(NSAVV, istep)) then
         call writcv(vx, vy, vz,  &
#if KEY_CHEQ==1
               CG, .false.,  &
#endif
               NATOM, IMOVE, NATOM, npriv, istep, ndegf, DELTA,  &
               NSAVV, NSTEP, TITLEA, NTITLA, IUNVEL, .true.,  &
               .false., [0], .false., [ZERO])
      endif
   end subroutine traj_write

   subroutine serialize_system(out_str)
     use, intrinsic :: iso_c_binding, only: c_char
     use openmm, only: OpenMM_XmlSerializer_serializeSystem

     implicit none

     character(kind=c_char, len=1), allocatable, dimension(:) :: out_str

     call OpenMM_XmlSerializer_serializeSystem(system, out_str)
   end subroutine serialize_system

   subroutine serialize(to_seri, seri_unit)
     character(len=4), intent(in) :: to_seri ! OpenMM object to serialize
     integer, intent(in) :: seri_unit
     character(len=1), allocatable, dimension(:) :: arr

     ! variable for OpenMM_Context_getState
     type(OpenMM_State) :: state
     integer*4 data_wanted
     integer*4 :: enforce_periodic

     ! same parameters as show_state subroutine
     data_wanted = ior( &
       ior(OpenMM_State_Positions, OpenMM_State_Velocities), &
       ior(OpenMM_State_Forces, OpenMM_State_Energy))
     enforce_periodic = OpenMM_False

      if (nbopts%periodic) enforce_periodic = OpenMM_True
     ! refuse to output to bad unit number
     if((seri_unit < 0) .or. (seri_unit == 5)) then
          call wrndie(-4, '<OMM>', &
               'bad unit number for serialization (< 0 or 5)')
          return
     endif

     if(system_dirty) then
          call wrndie(-4, '<OMM>', 'OpenMM System dirty')
          return
     endif

     select case(to_seri)
     case ('SYST')
       call OpenMM_XmlSerializer_serializeSystem(system, arr)
     case ('STAT')
       call OpenMM_Context_getState(context, data_wanted, enforce_periodic, &
                                    state)
       call OpenMM_XmlSerializer_serializeState(state, arr)
       call OpenMM_State_destroy(state)
     case ('INTE')
       call OpenMM_XmlSerializer_serializeIntegrator(integrator, arr)
     case default
       call wrndie(-4, '<OMM>', &
         'bad OpenMM object for serialization (SYST, STAT, INTE)')
       return
     end select

     write(seri_unit, "(a)") array_to_string(arr, size(arr))
     deallocate(arr)
   end subroutine serialize

   function array_to_string(arr, arr_len)
     character(len=1), intent(in), allocatable, dimension(:) :: arr
     integer, intent(in) :: arr_len

     character(len=arr_len) :: array_to_string
     integer :: i

     do i = 1, arr_len
       array_to_string(i:i) = arr(i)
     end do
   end function array_to_string

   subroutine omm_change_lambda(lambda)
     use omm_bonded, only : reparameterize_torsions

     real(chm_real), intent(in) :: lambda

     call reparameterize_torsions(system, context, lambda)

   end subroutine omm_change_lambda

   subroutine omm_get_context_ptr(ctx_ptr)
     use, intrinsic :: iso_c_binding, only: c_ptr
     use openmm, only: openmm_context
     implicit none
     type(c_ptr), intent(out) :: ctx_ptr
     ctx_ptr = transfer(context, ctx_ptr)
   end subroutine omm_get_context_ptr

#endif /* (openmm)*/

end module omm_main
