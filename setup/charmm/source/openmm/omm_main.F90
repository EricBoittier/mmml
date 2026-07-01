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

   type(omm_dynopts_t), save :: dynopts
   type(omm_nbopts_t), save :: nbopts
   integer, save :: cm_stop_freq
   integer, save :: dynamics_mode
   integer, save :: indexBarostat
   integer, save :: openmm_version_number
   logical, save :: openmm_initialized = .false.
   logical, save :: system_dirty = .false.
#endif

   integer, parameter :: DYN_UNINIT = 0, DYN_RESTART = 1, &
         DYN_SETVEL = 2,  DYN_CONTINUE = 3

#if KEY_OPENMM==1
   public :: omm_energy, teardown_openmm, omm_invalidate, check_nbopts
   public :: serialize, serialize_system
   public :: nbopts, get_plugins
   public :: omm_change_lambda
   public :: gbswforce
   public :: omm_minimize
#endif

   public :: omm_dynamics, omm_eterm_mask

 contains

   !> Performs molecular dynamics using OpenMM.
   subroutine omm_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
         jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
         iprfrq, isvfrq, ntrfrq, openmm_ran)

      use reawri, only: NSTEP

      type(omm_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
      real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
      real(chm_real), intent(inout) :: jhtemp
      real(chm_real), intent(in) :: gamm(:)
      integer, intent(inout) :: npriv
      integer, intent(in) :: istart, istop, iprfrq, ndegf, igvopt, isvfrq, ntrfrq
      logical, intent(inout) :: openmm_ran

      openmm_ran=.FALSE.

#if KEY_OPENMM==1
      cm_stop_freq = ntrfrq
      call check_dynopts(optarg)

      call run_dynamics(vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
            jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
            iprfrq, isvfrq, openmm_ran)
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
         iprfrq, isvfrq, openmm_ran)
      use averfluc
      use avfl_ucell
      use coord  ! XXX writes
      use image, only: XTLTYP
      use reawri, only: NSTEP, JHSTRT, TIMEST
      use contrl, only: NPRINT, IREST
      use consta, only: TIMFAC

      real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
      real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
      real(chm_real), intent(inout) :: jhtemp
      real(chm_real), intent(in) :: gamm(:)
      integer, intent(inout) :: npriv
      integer, intent(in) :: istart, istop, iprfrq, isvfrq, ndegf, igvopt
      logical, intent(inout) :: openmm_ran

      real*8, parameter :: ChmVelPerOmmVel = TIMFAC * OpenMM_AngstromsPerNm
      real*8 :: pos(3, NATOM), vel_t(3, NATOM), vel_pre(3, NATOM)
      real(chm_real) :: dnum
      integer*4,save :: istep
      integer :: numstp, i
      integer, save :: navestps
      real(chm_real) :: wtf(NATOM)
      logical :: check_tol

      call setup_openmm()
      if (IREST > 0) then
         dynamics_mode = DYN_RESTART
      else if (IGVOPT < 3) then
         dynamics_mode = DYN_SETVEL
      endif

      ! XXX scale factor for vel_pre from restart file written by DYNAMC,
      ! even if not doing Langevin dynamics
      wtf = TWO * gamm(3*NATOM+1 : 4*NATOM)

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

      ! else use velocities already in OpenMM context
      endif

      if (istart <= 1) istep = 0

      if (todo_now(iprfrq, istart-1)) then
         navestps = 0
         jhtemp = ZERO
         call avfl_reset()
         if (dynopts%qPressure) call avfl_ucell_reset()
      endif

      ! Dynamic step 0 typically has very different energy from later steps
      check_tol = (dynamics_mode == DYN_CONTINUE)
      do
         call show_state(pos, vel_t, vel_pre, istep, istart, istop, npriv, ndegf, &
               isvfrq, navestps, jhtemp)
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

   end subroutine run_dynamics

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
   end subroutine setup_openmm

   subroutine teardown_openmm()
      use new_timer
      if (.not. openmm_initialized) return
      if (prnlev >= 2) write (OUTU, '(A)') 'Destroying OpenMM context'
      openmm_initialized = .false.
      call timer_start(T_omm_ctx)
      call OpenMM_Context_destroy(context)
      call OpenMM_Integrator_destroy(integrator)
      call OpenMM_System_destroy(system)
      call timer_stop(T_omm_ctx)
   end subroutine teardown_openmm

   subroutine omm_invalidate()
      if (openmm_initialized) system_dirty = .true.
   end subroutine omm_invalidate

   subroutine check_system()
      if (.not. system_dirty) return
      if (PRNLEV >= 2) write (OUTU, '(a)') 'OpenMM system changed'
      call teardown_openmm()
      system_dirty = .false.
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
            if (PRNLEV >= 2) write (OUTU, '(A)') &
                  'OpenMM integrator options changed'
            call teardown_openmm()
         endif
      endif
      dynopts = optarg
   end subroutine check_dynopts

   subroutine setup_system(init)
      use omm_bonded, only: import_psf
      use omm_restraint, only: setup_restraints
      use new_timer
      use rndnum, only : rngseeds

      logical :: init

      integer*4 :: ommseed

      ommseed = rngseeds(1)

      if(prnlev>5) write(outu,'(a,i16)') &
           'CHARMM> OpenMM using random seed ',ommseed


      call timer_start(T_omm_sys)

      ! System takes ownership of Forces; don't destroy them yourself.
      call OpenMM_System_create(system)
      call import_psf(system, nbopts)

      if(.not. init) then
         call setup_restraints(system, nbopts%periodic)
         call setup_cm_freezer(system)
         call setup_thermostat(system, ommseed)
         call setup_barostat(system, ommseed)
      endif
      integrator = new_integrator(dynopts, ommseed)
      if(.not. init) call setup_shake(system, integrator)

      call timer_stop(T_omm_sys)

   end subroutine setup_system

   !> Creates and returns an OpenMM integrator with the given options.
   type(OpenMM_Integrator) function new_integrator(opts, ommseed)
      use reawri, only: TIMEST


      type(omm_dynopts_t), intent(in) :: opts
      integer*4, intent(in) :: ommseed
      type(OpenMM_VerletIntegrator) :: verlet
      type(OpenMM_LangevinIntegrator) :: langevin
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
            call OpenMM_LangevinIntegrator_create(langevin, &
                  opts%temperatureReference, opts%frictionCoefficient, TIMEST)
            call OpenMM_LangevinIntegrator_setRandomNumberSeed(langevin, &
                 ommseed)
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

#if KEY_OPENMM==1
     use OpenMM_Charmm
#endif

     type(omm_dynopts_t), intent(in) :: opts
     type (OpenMM_Force) force
     type(OpenMM_LangevinIntegrator) :: langevin
     type(OpenMM_VariableLangevinIntegrator) :: varlangevin
     type(OpenMM_MonteCarloBarostat) :: barostat
#if KEY_OPENMM==1
     type(OpenMM_MonteCarloBarostat2) :: barostat2
#endif

     real :: temperature

     ! Create particular integrator, and recast to generic one.
     if (opts%qLangevin) then
        if (opts%qVariable) then
           varlangevin = transfer(integrator, OpenMM_VariableLangevinIntegrator(0))
           call OpenMM_VariableLangevinIntegrator_setTemperature( &
                varlangevin, opts%temperatureReference)
           temperature = OpenMM_VariableLangevinIntegrator_getTemperature( &
                varlangevin)
        else
           langevin = transfer(integrator, OpenMM_LangevinIntegrator(0))
           call OpenMM_LangevinIntegrator_setTemperature( &
                langevin, opts%temperatureReference)
           temperature = OpenMM_LangevinIntegrator_getTemperature( &
                langevin)
        endif
        if(prnlev>0) write(outu,*) "Langevin> updating the temperature into ",temperature,' K'
     endif
     if(opts%qPressure) then
        call OpenMM_System_getForce(system, indexBarostat, force)
        if (dynopts%qPressure2) then
           barostat2 = transfer(force,OpenMM_MonteCarloBarostat2(0))
           call OpenMM_MonteCarloBarostat2_setTemperature(barostat2, opts%temperatureReference)
           temperature = OpenMM_MonteCarloBarostat2_getTemperature(barostat2)
        else
           barostat = transfer(force,OpenMM_MonteCarloBarostat(0))
#ifdef OPENMM_API_UPDATE
           call OpenMM_MonteCarloBarostat_setDefaultTemperature(barostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloBarostat_getDefaultTemperature(barostat)
#else
           call OpenMM_MonteCarloBarostat_setTemperature(barostat, opts%temperatureReference)
           temperature = OpenMM_MonteCarloBarostat_getTemperature(barostat)
#endif
        endif
        if(prnlev>0) write(outu,*) "Pressure> updating the temperature into ",temperature,' K'
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
      integer*4 :: ijunk

      if (dynopts%qAndersen) then
         call OpenMM_AndersenThermostat_create(andersen, &
               dynopts%temperatureReference, dynopts%collisionFrequency)
         call OpenMM_AndersenThermostat_setRandomNumberSeed(andersen, &
              ommseed)
         ijunk = OpenMM_System_addForce(system, &
               transfer(andersen, OpenMM_Force(0)))
      endif
   end subroutine setup_thermostat

   !> Adds a barostat to the system if constant pressure is requested.
   subroutine setup_barostat(system, ommseed)
#if KEY_OPENMM==1
      use OpenMM_Charmm
#endif
      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: ommseed
      type(OpenMM_MonteCarloBarostat) :: barostat
#if KEY_OPENMM==1
      type(OpenMM_MonteCarloBarostat2) :: barostat2
#endif
      integer*4 :: ijunk
      ! Convert surface tension from dyne/cm to Atm*Angstrom
      real(chm_real) :: SFACT = 98.6923

      if (dynopts%qPressure) then
#if KEY_OPENMM==1
         !> Choose a barostat2 to the system if constant pressure in 3D is requested.
         if (dynopts%qPressure2) then
            call OpenMM_MonteCarloBarostat2_create(barostat2, &
                  dynopts%pressureReference, dynopts%temperatureReference, &
                  dynopts%pressureIn3Dimensions,  &
                  dynopts%surfaceTension * SFACT / OpenMM_AngstromsPerNm, &
                  dynopts%pressureFrequency)
            call OpenMM_MonteCarloBarostat2_setRandomNumberSeed(barostat2, &
                 ommseed)
            indexBarostat = OpenMM_System_addForce(system, &
                  transfer(barostat2, OpenMM_Force(0)))
         else
#endif
            call OpenMM_MonteCarloBarostat_create(barostat, &
                 dynopts%pressureReference, dynopts%temperatureReference, dynopts%pressureFrequency)
            call OpenMM_MonteCarloBarostat_setRandomNumberSeed(barostat, &
                 ommseed)
            indexBarostat = OpenMM_System_addForce(system, &
                  transfer(barostat, OpenMM_Force(0)))
#if KEY_OPENMM==1
         endif
#endif
      endif
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
         istep, istart, istop, npriv, ndegf, isvfrq, navestps, jhtemp)
      use consta, only: KBOLTZ
      use contrl, only: NPRINT
      use coord  ! XXX writes
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
         ! In dynamics we need to zero these energy terms between calls
         EPROP(TEPR) = EPROP(TOTE)
         EPROP(EPOT) = ZERO
         EPROP(TOTKE) = ZERO
         EPROP(TOTE) = ZERO
         call export_energy(state, vel_t)
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
            lhdr = istep == 0
            if(prnlev>0) call printe(outu,eprop,eterm,'DYNA','DYN',lhdr, &
                 istep,npriv*timest,zero,.true.)
            if (dynopts%qPressure) call prnxtld(outu,'DYNA',xtltyp,xucell,.true.,zero, &
                 .true.,epress)
         endif
      endif
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

#endif /* (openmm)*/

end module omm_main
