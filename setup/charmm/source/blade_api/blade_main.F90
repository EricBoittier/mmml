module blade_main
  use chm_kinds
  use blade_module, only: system
  use stream

  implicit none

  integer, parameter :: DYN_UNINIT = 0, DYN_RESTART = 1, &
       DYN_SETVEL = 2,  DYN_CONTINUE = 3

  integer :: dynamics_mode, ngpus = 0
  integer, dimension(:), allocatable :: gpus
  logical, save :: blade_initialized = .false.
  logical, save :: system_dirty = .false.

  integer, save :: navestps

   ! type blade_dynopts_t
   !    real(chm_real) :: temperatureReference, pressureReference
   !    real(chm_real) :: volumeFluctuation
   !    integer :: pressureFrequency
   !    logical :: blade_qrexchg
   ! end type blade_dynopts_t
#if KEY_BLADE == 1
   interface
      subroutine blade_set_step(system, istep) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr, c_int
        implicit none
        type(c_ptr), value :: system
        integer(c_int), value :: istep
      end subroutine blade_set_step

      subroutine blade_update_domdec(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_update_domdec

      subroutine blade_rectify_holonomic(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_rectify_holonomic

      subroutine blade_get_force(system, report_energy) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr, c_int
        implicit none
        type(c_ptr), value :: system
        integer(c_int), value :: report_energy
      end subroutine blade_get_force

      subroutine blade_update(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_update

      subroutine blade_check_gpu(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_check_gpu

      subroutine blade_run_energy(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_run_energy

      subroutine blade_dynamics_initialize(system) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr
        implicit none
        type(c_ptr), value :: system
      end subroutine blade_dynamics_initialize

      subroutine blade_range_begin(message) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr, c_char
        implicit none
        character(len=1, kind=c_char) :: message(*)
      end subroutine blade_range_begin

      subroutine blade_range_end() bind(c)
        implicit none
      end subroutine blade_range_end
   end interface

 contains

!    subroutine omm_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
!          jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
!          iprfrq, isvfrq, ntrfrq, openmm_ran)
   subroutine blade_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
        jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
        iprfrq, isvfrq)
     use blade_dynopts, only: blade_dynopts_t
     ! use, intrinsic :: iso_c_binding, only: c_char, c_null_char
     ! use blade_module, only: blade_range_begin, blade_range_end

     implicit none

     type(blade_dynopts_t), intent(inout) :: optarg
     real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
     real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
     real(chm_real), intent(inout) :: jhtemp
     real(chm_real), intent(in) :: gamm(:)
     integer, intent(in) :: ndegf, igvopt, istart, istop, iprfrq, isvfrq
     integer, intent(inout) :: npriv

     ! call blade_range_begin('BLaDE omp parallel' // c_null_char)
     !$omp parallel
! ! DEBUG
!               if(prnlev >= 2) write(outu,'(A,i10,A,i10)') &
!                    'DYNAMC: blade_dynamics from step ', &
!                    istart, ' to step ', istop

     call setup_blade()
     call dynamics_setup(optarg)
     call dynamics_initial_conditions(vx_t,vy_t,vz_t,vx_pre,vy_pre,vz_pre,gamm,igvopt)
     call dynamics(optarg, jhtemp, ndegf, npriv, istart, istop, iprfrq, isvfrq)
     call dynamics_final_conditions(vx_t,vy_t,vz_t,vx_pre,vy_pre,vz_pre,gamm)
     !$omp end parallel
     ! call blade_range_end()
   end subroutine blade_dynamics

   subroutine dynamics_setup(optarg)
     use reawri, only: delta
     use consta, only: atmosp
     use blade_dynopts, only: blade_dynopts_t
     use blade_options_module, only: blade_add_run_dynopts

     implicit none

     type(blade_dynopts_t), intent(inout) :: optarg

     if (dynamics_mode == DYN_UNINIT) then
        call blade_add_run_dynopts(system, &
             0, 0, 0, &
             delta, &
             optarg%temperatureReference, &
             optarg%pressureFrequency, &
             optarg%volumeFluctuation, &
             optarg%pressureReference*atmosp)
     ! use delta, not timest, because timest is in ps, we need AKMA
     ! arguments 2-4 were istart, istart, istop - istart, but they are ignored
        call blade_dynamics_initialize(system)
     !    blade_initialized = .true.
     end if
   end subroutine dynamics_setup

   subroutine dynamics_initial_conditions(vx_t,vy_t,vz_t,vx_pre,vy_pre,vz_pre,gamm,igvopt)
     use contrl, only: irest
     use blade_coords_module, only: copy_state_c2b
     use psf, only: natom

     implicit none

     real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
     real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
     real(chm_real), intent(in) :: gamm(:)
     integer, intent(in) :: igvopt

     real(chm_real) :: wtf(NATOM)
     integer :: i

     ! write (outu, '(a,i5,a,i5,a,i5)') 'irest = ', irest, ' igvopt = ', igvopt, 'dynamics_mode = ', dynamics_mode
     if (IREST > 0) then
        !$omp barrier
        !$omp master
        dynamics_mode = DYN_RESTART
        !$omp end master
        !$omp barrier
     else if (IGVOPT < 3) then
        !$omp barrier
        !$omp master
        dynamics_mode = DYN_SETVEL
        !$omp end master
        !$omp barrier
     end if

     if (dynamics_mode /= DYN_CONTINUE) then
        !$omp barrier
        !$omp master
        if (dynamics_mode == DYN_RESTART) then
           if (PRNLEV >= 2) write (OUTU, '(a)') 'BLaDE: Velocities from restart file'
           ! Waste of effort. Just don't unscale it later
           ! Might as well make it compatible with other garbagey restart files
           wtf = 2.0 * gamm(3*NATOM+1 : 4*NATOM)
           do i = 1, NATOM
              vx_pre(i)=vx_pre(i)*wtf(i)
              vy_pre(i)=vy_pre(i)*wtf(i)
              vz_pre(i)=vz_pre(i)*wtf(i)
           enddo
        else if (dynamics_mode == DYN_SETVEL) then
           if (PRNLEV >= 2) write (OUTU, '(a)') 'BLaDE: Velocities scaled or randomized'
           ! Open MM backs it up by half a time step. Blade doesn't easily have that capacity
           vx_pre(1:natom)=vx_t(1:natom)
           vy_pre(1:natom)=vy_t(1:natom)
           vz_pre(1:natom)=vz_t(1:natom)
        else
           if (PRNLEV >= 2) write (OUTU, '(a)') 'OpenMM: Velocities undefined!'
        endif
        !$omp end master
        !$omp barrier
        call copy_state_c2b(system,vx_pre,vy_pre,vz_pre)
        call blade_rectify_holonomic(system)
        call blade_set_step(system, 0)
        call blade_update_domdec(system)
     endif
   end subroutine dynamics_initial_conditions

   subroutine energy_initial_conditions(qrectifyshake)
     use blade_coords_module, only: copy_state_c2b
     use psf, only: natom

     implicit none

     logical, intent(in) :: qrectifyshake
     real(chm_real) :: wtf(NATOM)

     !$omp barrier
     !$omp master
     wtf = 0.0
     !$omp end master
     !$omp barrier
     call copy_state_c2b(system,wtf,wtf,wtf)
     if (qrectifyshake) call blade_rectify_holonomic(system)
     call blade_set_step(system, 0)
     call blade_update_domdec(system)
   end subroutine energy_initial_conditions

   subroutine dynamics_final_conditions(vx_t,vy_t,vz_t,vx_pre,vy_pre,vz_pre,gamm)
     use contrl, only: irest
     use blade_coords_module, only: copy_state_b2c
     use psf, only: natom

     implicit none

     real(chm_real), intent(inout) :: vx_t(:), vy_t(:), vz_t(:)
     real(chm_real), intent(inout) :: vx_pre(:), vy_pre(:), vz_pre(:)
     real(chm_real), intent(in) :: gamm(:)

     real(chm_real) :: wtf(NATOM)
     integer :: i

     !$omp barrier
     !$omp master
      call copy_state_b2c(system,vx_pre,vy_pre,vz_pre)
      vx_t(1:natom)=vx_pre(1:natom)
      vy_t(1:natom)=vy_pre(1:natom)
      vz_t(1:natom)=vz_pre(1:natom)
      ! Superfluous unscaling
      ! Do it anyways to make restarts interoperable
      wtf = 2.0 * gamm(3*NATOM+1 : 4*NATOM)
      do i = 1, NATOM
         vx_pre(i)=vx_pre(i)/wtf(i)
         vy_pre(i)=vy_pre(i)/wtf(i)
         vz_pre(i)=vz_pre(i)/wtf(i)
      enddo
      !$omp end master
      !$omp barrier
      call blade_check_gpu(system)
   end subroutine dynamics_final_conditions

   subroutine dynamics(optarg, jhtemp, ndegf, npriv, istart, istop, iprfrq, isvfrq)
     use blade_dynopts, only: blade_dynopts_t

      implicit none

      type(blade_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: jhtemp
      integer, intent(in) :: ndegf, istart, istop, iprfrq, isvfrq
      integer, intent(inout) :: npriv
      integer,save :: istep

      !$omp barrier
      !$omp master
      if (istart <= 1) istep = 0

      call initialize_output(optarg, jhtemp, istart, iprfrq)
      !$omp end master
      !$omp barrier

      do
         call blade_set_step(system, istep)
         call blade_update_domdec(system)
         call blade_get_force(system,merge(1,0,report_energy(istep,istop,isvfrq)))

         call print_output(optarg,jhtemp,ndegf,istep,istart,istop,npriv,isvfrq)
         if (istep >= istop) exit

         call blade_update(system)
         !$omp barrier
         !$omp master
         istep = istep + 1
         npriv = npriv + 1
         dynamics_mode = DYN_CONTINUE
         !$omp end master
         !$omp barrier
      enddo
   end subroutine dynamics

   subroutine initialize_output(optarg, jhtemp, istart, iprfrq)
      use averfluc
      use avfl_ucell
      use blade_dynopts, only: blade_dynopts_t
      implicit none
      type(blade_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: jhtemp
      integer, intent(in) :: istart, iprfrq

      if (todo_now(iprfrq, istart-1)) then
         navestps = 0
         jhtemp = zero
         call avfl_reset()
         ! if (dynopts%qPressure) call avfl_ucell_reset()
         if (optarg%pressureFrequency > 0) call avfl_ucell_reset()
      endif
   end subroutine initialize_output

   subroutine print_output(optarg,jhtemp,ndegf,istep,istart,istop,npriv,isvfrq)
      use consta, only: KBOLTZ
      use contrl, only: NPRINT
      use energym
      use averfluc
      use avfl_ucell
      use image, only: xtltyp, xucell
      use coord
      use cvio, only: writcv
      use ctitla, only: NTITLA, TITLEA
      use psf, only: CG, IMOVE
      use reawri, only: NSTEP, DELTA, JHSTRT, NSAVC, NSAVV, IUNCRD, IUNVEL, IUNWRI, TIMEST
      use number, only: zero
      use psf, only: NATOM
      use blade_coords_module, only: copy_spatial_b2c, copy_alchemical_b2c, &
            copy_energy_b2c, copy_box_b2c, blade_recv_energy
      use blade_dynopts, only: blade_dynopts_t

#if KEY_BLOCK == 1
      use block_ltm, only: nblock
      use lambdam, only: nsavl, iunldm, msld_writld
#if KEY_LIBRARY == 1
      use api_msldata, only: fill_msldata, msldata_init, &
           msldata_set_names, msldata_add_rows
#endif /* KEY_LIBRARY */
#endif /* KEY_BLOCK */

      implicit none

      type(blade_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: jhtemp
      integer, intent(in) :: ndegf, istep, istart, istop, isvfrq
      integer, intent(inout) :: npriv

      real(chm_real) :: eP, eK, temperature
      logical :: lhdr

      if (report_energy(istep,istop,isvfrq)) then
         if(istep == 0 .or. (istep>=istart)) then
            !$omp barrier
            !$omp master
            ! In dynamics we need to zero these energy terms between calls
            eprop(tepr) = eprop(tote)
            eprop(epot) = zero
            eprop(totke) = zero
            eprop(tote) = zero
            call blade_recv_energy(system)
            call copy_energy_b2c(system)
            eP = eprop(epot)
            eK = eprop(totke)
            temperature = 2 * eK / (ndegf * kboltz)
            eprop(temps) = temperature
            jhtemp = jhtemp + temperature
            ! if (dynopts%qPressure) then
            if (optarg%pressureFrequency > 0) then ! Period box len needed for CPT
               call copy_box_b2c(system)
               call avfl_ucell_update()
            endif
            navestps = navestps + 1
            call avfl_update(eprop, eterm, epress)
            if (todo_now(nprint, istep) .or. istep == istop) then
               lhdr = istep == 0
               if(prnlev>0) call printe(outu,eprop,eterm,'DYNA','DYN',lhdr, &
                    istep,npriv*timest,zero,.true.)
               ! if (dynopts%qPressure) ...
               if (optarg%pressureFrequency > 0) call prnxtld(outu,'DYNA',xtltyp,xucell,.true.,zero, &
                    .true.,epress)
            endif
            !$omp end master
            !$omp barrier
         endif
      endif

      if (dynamics_mode == DYN_CONTINUE) then
         if (iuncrd > 0 .and. todo_now(nsavc,istep)) then
            !$omp barrier
            !$omp master
            call copy_spatial_b2c(system)

            call writcv(X, Y, Z,  &
#if KEY_CHEQ==1
                  CG, .false.,  &
#endif
                  NATOM, IMOVE, NATOM, npriv, istep, ndegf, DELTA, &
                  NSAVC, NSTEP, TITLEA, NTITLA, IUNCRD, .false.,  &
                  .false., [0], .false., [ZERO])
            !$omp end master
            !$omp barrier
         endif

         if (iunvel > 0 .and. todo_now(nsavv,istep)) then
            call wrndie(-5,'<blade_main>', 'nsavv greater than zero is not supported with blade')
         endif

#if KEY_BLOCK == 1
         if (iunldm > 0 .and. todo_now(nsavl,istep)) then
            !$omp barrier
            !$omp master
            call copy_alchemical_b2c(system)

            call msld_writld(nblock,npriv, &
                 istep,nstep, &
                 delta )
            !$omp end master
            !$omp barrier
         endif

#if KEY_LIBRARY == 1
     if (fill_msldata .and. (istep .eq. istop)) then
        call msldata_init(1)
        call msldata_set_names()
        call msldata_add_rows(istep, npriv * timest)
     end if
#endif /* KEY_LIBRARY */
#endif /* KEY_BLOCK */
      endif
   end subroutine print_output

   logical function report_energy(istep,istop,isvfrq)
      use contrl, only: NPRINT
      use reawri, only: NSAVC, NSAVV, IUNCRD, IUNVEL, IUNWRI
      implicit none
      integer, intent(in) :: istep, istop, isvfrq

      report_energy = .false.
      if (istep == 0) report_energy = .true.
      if (istep == istop) report_energy = .true.
      if (todo_now(nprint,istep)) report_energy = .true.
      if (todo_now(nsavc,istep) .and. iuncrd > 0) report_energy = .true.
      if (todo_now(nsavv,istep) .and. iunvel > 0) report_energy = .true.
      if (todo_now(isvfrq,istep) .and. iunwri > 0) report_energy = .true.
   end function report_energy

   subroutine setup_system(init)
     use, intrinsic :: iso_c_binding, only: &
          c_associated, &
          c_null_char
     use blade_module, only: system, &
          blade_init_system, &
          blade_set_device, &
          blade_set_verbose, &
          export_psf_to_blade, &
          export_param_to_blade, &
          export_coords_to_blade, &
#if KEY_BLOCK == 1
          export_block_to_blade, &
#endif
          export_options_to_blade
     use blade_module, only: &
          blade_interpretter, blade_fn_use, blade_fn_len, blade_fname
     use new_timer, only: T_blade, timer_start, timer_stop
     ! use omm_restraint, only: setup_restraints
     ! use rndnum, only : rngseeds

      implicit none

      logical :: init
      ! integer*4 :: ommseed

      ! ommseed = rngseeds(1)

      ! if(prnlev>5) write(outu,'(a,i16)') &
      !      'CHARMM> OpenMM using random seed ',ommseed

      call timer_start(T_blade)

      if (system_dirty) then
         call teardown_system(system)
      endif

      if (.not. blade_initialized) then
         if (.not. c_associated(system)) then
            !$omp barrier
            !$omp master
            system = blade_init_system(ngpus, gpus)
            !$omp end master
            !$omp barrier
         end if
      end if

      call blade_set_device(system)
      call blade_set_verbose(system,0)

      if (.not. blade_initialized) then
         call export_psf_to_blade()
         call export_param_to_blade()
         call export_coords_to_blade()
         call export_block_to_blade()
         call export_options_to_blade()

         if (blade_fn_use) then
            !$omp master
            if(prnlev >= 2) write(outu,'(A,A)') &
                 'BLADE_MAIN: streaming in file ', &
                 blade_fname(1:blade_fn_len)
            !$omp end master
            call blade_interpretter(blade_fname(1:blade_fn_len) // c_null_char, system)
         endif

         !$omp barrier
         !$omp master
         blade_initialized = .true.
         !$omp end master
         !$omp barrier
      endif

      ! if(.not. init) then
      !    call setup_restraints(system, nbopts%periodic)
      !    call setup_cm_freezer(system)
      !    call setup_thermostat(system, ommseed)
      !    call setup_barostat(system, ommseed)
      ! endif
      ! integrator = new_integrator(dynopts, ommseed)
      ! if(.not. init) call setup_shake(system, integrator)

      call timer_stop(T_blade)
    end subroutine setup_system

   subroutine teardown_system(system)
     use, intrinsic :: iso_c_binding, only: c_ptr, c_associated, c_null_ptr
     use blade_module, only: blade_dest_system
     use new_timer, only: T_blade, timer_start, timer_stop
     implicit none
     type(c_ptr) :: system

     call timer_start(T_blade)

     !$omp barrier
     !$omp master
     system_dirty = .false.
     blade_initialized = .false.
     dynamics_mode = DYN_UNINIT
     !$omp end master
     !$omp barrier

     if (.not. c_associated(system)) then
        return
     end if

     !$omp barrier
     !$omp master
     call blade_dest_system(system)
     system = c_null_ptr
     !$omp end master
     !$omp barrier

     call timer_stop(T_blade)
   end subroutine teardown_system

    subroutine setup_blade()
     ! use omm_glblopts, only : qtor_repex, torsion_lambda
     ! call get_PlatformDefaults

     ! call check_system()
     ! call check_nbopts()
     ! if (openmm_initialized) return
     ! if (blade_initialized) return
     ! if (prnlev >= 2) write (OUTU, '(A)') 'Setup_OpenMM: Initializing OpenMM context'
     ! call load_libs()
     call setup_system(.false.)
     ! call init_context()
     ! if(qtor_repex) call omm_change_lambda(torsion_lambda)
     ! dynamics_mode = DYN_UNINIT
     ! blade_initialized = .true.
   end subroutine setup_blade

   subroutine blade_repd_energy(x, y, z)
     use new_timer, only: T_energy, timer_start, timer_stop
     use blade_coords_module, only: copy_energy_b2c, copy_force_b2c, &
           blade_recv_force, blade_recv_energy

     implicit none

     real(chm_real), intent(in) :: x(:), y(:), z(:)

     !$omp parallel
     call timer_start(T_energy) ! OMPWARNING
     call energy_initial_conditions(.false.) ! don't rectify shake
     call blade_get_force(system,1)
     !$omp barrier
     !$omp master
     call blade_recv_energy(system) ! this call also covered by blade_run_energy
     call copy_energy_b2c(system)
     ! call blade_recv_force(system)
     ! call copy_force_b2c(system)
     !$omp end master
     !$omp barrier
     call timer_stop(T_energy) ! OMPWARNING
     !$omp end parallel
   end subroutine blade_repd_energy

   !> Sets CHARMM energies and forces for the given coordinates.
   subroutine blade_energy(x, y, z)
     ! use omm_ecomp, only : omm_assign_eterms
     use deriv  ! XXX writes
     use energym  ! XXX writes
     use new_timer, only: T_energy, timer_start, timer_stop
     use blade_coords_module, only: copy_energy_b2c, copy_force_b2c, &
           blade_recv_force, blade_recv_energy

     implicit none

     real(chm_real), intent(in) :: x(:), y(:), z(:)
     ! real(chm_real) :: Epterm
     ! type(OpenMM_State) :: state
     ! real*8 :: pos(3, NATOM)
     ! integer*4 :: data_wanted
     ! integer*4 :: enforce_periodic
     ! integer*4 :: itype, group

     !$omp parallel

     call setup_blade()
!      pos = get_xyz(X, Y, Z) / OpenMM_AngstromsPerNm
!      call set_positions(context, pos)
! #if KEY_PHMD==1
!      if (qphmd_omm .and. qphmd_initialized) call set_lambda_state(context)
! #endif
!      if (nbopts%periodic) call import_periodic_box()

!      data_wanted = ior(OpenMM_State_Energy, OpenMM_State_Forces)
!      enforce_periodic = OpenMM_False
!      if (nbopts%periodic) enforce_periodic = OpenMM_True
     call timer_start(T_energy) ! OMPWARNING
     ! call OpenMM_Context_getState(context, data_wanted, enforce_periodic, state)

     ! Calling this function creates superfluous files.
     ! call blade_run_energy(system)
     ! Call it the slow way instead:
     if (dynamics_mode == DYN_UNINIT) & ! False if BladeIsNotDirty was called
        call blade_dynamics_initialize(system) ! only relevant piece from dynamics_setup
     call energy_initial_conditions(.true.) ! rectify shake constraints
     call blade_get_force(system,1)
     ! end slow way

     !$omp barrier
     !$omp master
     call blade_recv_energy(system) ! this call also covered by blade_run_energy
     call copy_energy_b2c(system)
     call blade_recv_force(system)
     call copy_force_b2c(system)
     !$omp end master
     !$omp barrier

     ! call export_forces(state, DX, DY, DZ)
     ! call OpenMM_State_destroy(state)
     ! call omm_assign_eterms(context, enforce_periodic)

     ! call teardown_system(system) ! Teardown called on subsequent runs if not needed
     call timer_stop(T_energy) ! OMPWARNING

     !$omp end parallel

   end subroutine blade_energy

    !> Returns an array indicating whether we can use Blade
    !> for each energy term.
    function blade_eterm_mask()
      use energym, only: lenent, &
           bond, angle, dihe, imdihe, cmap, elec, vdw, imelec, imvdw, &
           ewksum, ewself, ewexcl, epot, totke, tote
      implicit none

      logical :: blade_eterm_mask(lenent)
      integer, parameter :: my_eterms(15) = [ &
              bond, angle, dihe, imdihe, cmap, elec, vdw, imelec, imvdw, &
              ewksum, ewself, ewexcl, epot, totke, tote &
              ]
      integer :: i

      blade_eterm_mask = .false.
#if KEY_BLADE==1
      do i = 1, size(my_eterms)
         blade_eterm_mask(my_eterms(i)) = .true.
      enddo
#endif /* KEY_BLADE */
    end function blade_eterm_mask

   logical function todo_now(freq, istep)
      integer, intent(in) :: freq, istep
      if (freq > 0) then
         todo_now = mod(istep, freq) == 0
      else
         todo_now = .false.
      endif
   end function todo_now
#endif /* KEY_BLADE */
  end module blade_main
