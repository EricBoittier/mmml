module blade_main
  use, intrinsic :: iso_c_binding, only: c_associated, c_double, c_int
  use chm_kinds
  use blade_module, only: system
  use stream

  implicit none

  integer, parameter :: DYN_UNINIT = 0, DYN_RESTART = 1, &
       DYN_SETVEL = 2,  DYN_CONTINUE = 3

  integer :: dynamics_mode = DYN_UNINIT, ngpus = 0
  integer, dimension(:), allocatable :: gpus
  logical, save :: blade_initialized = .false.
  logical, save :: system_dirty = .false.
  integer, save :: blade_verbose_level = 0

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

      subroutine blade_get_force(system, report_energy, refill_random) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr, c_int
        implicit none
        type(c_ptr), value :: system
        integer(c_int), value :: report_energy
        integer(c_int), value :: refill_random
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

      integer(c_int) function blade_minimizer(system,nsteps,mintype,steplen) bind(c)
        use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int
        implicit none
        type(c_ptr), value :: system
        integer(c_int), value :: nsteps, mintype
        real(c_double), value :: steplen
      end function blade_minimizer

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

#if KEY_BLADE == 1
   !> Push a new pH-dependent BIELAM vector into a clean live BLaDE MSLD
   !> system. A system that has not been initialized yet is a successful
   !> no-op: its normal setup path will read the new CHARMM bias values.
   integer(c_int) function blade_sync_ph_bias_from_values(biases, count)
     use blade_block_module, only: blade_sync_msld_bias
     use lambdam, only: qmld, nblock

     implicit none
     real(c_double), dimension(*), intent(in) :: biases
     integer(c_int), intent(in) :: count
     integer(c_int) :: sync_status

     blade_sync_ph_bias_from_values = 1_c_int
     if (.not. blade_initialized .or. .not. c_associated(system)) return
     if (system_dirty .or. .not. qmld .or. count /= nblock .or. count < 1) then
       blade_sync_ph_bias_from_values = 0_c_int
       return
     endif

     sync_status = blade_sync_msld_bias(system, biases, count)
     if (sync_status == 0_c_int) then
       blade_sync_ph_bias_from_values = 0_c_int
       return
     endif
   end function blade_sync_ph_bias_from_values
#endif

   !> Set the BLaDE verbose level
   subroutine blade_set_verbose_level(level)
     implicit none
     integer, intent(in) :: level
     blade_verbose_level = level
   end subroutine blade_set_verbose_level

! UNUSED
!    !> Get the BLaDE verbose level
!    integer function blade_get_verbose_level()
!      implicit none
!      blade_get_verbose_level = blade_verbose_level
!    end function blade_get_verbose_level

   !> blade_log - main logging function called by BLaDE core
   !> Converts C string to Fortran and writes to CHARMM output stream
   subroutine blade_log(message) bind(c, name='blade_log')
     use, intrinsic :: iso_c_binding, only: c_char, c_null_char
     use stream, only: OUTU
     implicit none
     character(kind=c_char), intent(in) :: message(*)

     character(len=1024) :: fortran_string
     integer :: i, string_length

     ! Convert C string to Fortran string
     string_length = 0
     do i = 1, 1024  ! Use fixed upper bound instead of size()
        if (message(i) == c_null_char) exit
        string_length = string_length + 1
        fortran_string(i:i) = message(i)
     end do

     ! Write to CHARMM output
     if (string_length > 0) then
        write(OUTU, '(a)', advance='no') fortran_string(1:string_length)
     end if
   end subroutine blade_log

!    subroutine omm_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
!          jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
!          iprfrq, isvfrq, ntrfrq, openmm_ran)
   subroutine blade_dynamics(optarg, vx_t, vy_t, vz_t, vx_pre, vy_pre, vz_pre, &
        jhtemp, gamm, ndegf, igvopt, npriv, istart, istop, &
        iprfrq, isvfrq)
     use blade_dynopts, only: blade_dynopts_t
     use contrl, only: irest
#if KEY_BLOCK == 1
     use lambdam, only: qmld, nsitemld, nsubmld, thetamld, thetavmld
#endif
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
#if KEY_BLOCK == 1
     integer :: i
#endif

#if KEY_BLOCK == 1
     if (irest > 0 .and. qmld) then
        do i = 2, nsitemld
           if (any(.not. (abs(thetamld(i,1:nsubmld(i))) <= huge(thetamld(i,1)))) .or. &
                any(.not. (abs(thetavmld(i,1:nsubmld(i))) <= huge(thetavmld(i,1))))) &
                call wrndie(-5, 'BLaDE restart', &
                     'non-finite MSLD theta or theta velocity')
        enddo
     endif
#endif

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
     use coord, only: x, y, z
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
     else if (IGVOPT < 3 .and. dynamics_mode /= DYN_CONTINUE) then
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
           if (any(.not. (abs(x(1:natom)) <= huge(x(1)))) .or. &
                any(.not. (abs(y(1:natom)) <= huge(y(1)))) .or. &
                any(.not. (abs(z(1:natom)) <= huge(z(1)))) .or. &
                any(.not. (abs(vx_pre(1:natom)) <= huge(vx_pre(1)))) .or. &
                any(.not. (abs(vy_pre(1:natom)) <= huge(vy_pre(1)))) .or. &
                any(.not. (abs(vz_pre(1:natom)) <= huge(vz_pre(1))))) then
              call wrndie(-5, 'BLaDE restart', &
                   'non-finite restart coordinates or velocities')
           endif
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
     use blade_module, only: blade_check_interrupt

      implicit none

      type(blade_dynopts_t), intent(inout) :: optarg
      real(chm_real), intent(inout) :: jhtemp
      integer, intent(in) :: ndegf, istart, istop, iprfrq, isvfrq
      integer, intent(inout) :: npriv
      integer, save :: istep = 0, blade_step = 0, continuation_npriv = 0
      logical :: interrupted
      logical :: reuse_pending_random

      !$omp barrier
      !$omp master
      if (istart <= 1) then
         istep = 0
         if (dynamics_mode == DYN_CONTINUE) then
            npriv = continuation_npriv
         else
            blade_step = 0
            continuation_npriv = npriv
         endif
      endif

      call initialize_output(optarg, jhtemp, istart, iprfrq)
      !$omp end master
      !$omp barrier

      reuse_pending_random = dynamics_mode == DYN_CONTINUE
      do
         call blade_set_step(system, blade_step)
         call blade_update_domdec(system)
         call blade_get_force(system, &
              merge(1,0,report_energy(istep,istop,isvfrq)), &
              merge(0,1,reuse_pending_random))
         reuse_pending_random = .false.

         call print_output(optarg,jhtemp,ndegf,istep,istart,istop,npriv,isvfrq)
         !$omp single
         interrupted = blade_check_interrupt() /= 0
         !$omp end single copyprivate(interrupted)
         if (interrupted .or. istep >= istop) exit

         call blade_update(system)
         !$omp barrier
         !$omp master
         istep = istep + 1
         blade_step = blade_step + 1
         npriv = npriv + 1
         continuation_npriv = npriv
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
      use api_msldata, only: fill_msldata, msldata_init, &
           msldata_set_names, msldata_add_rows
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
         if (istep >= istart .and. iuncrd > 0 .and. todo_now(nsavc,istep)) then
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

         if (istep >= istart .and. iunvel > 0 .and. todo_now(nsavv,istep)) then
            call wrndie(-5,'<blade_main>', 'nsavv greater than zero is not supported with blade')
         endif

#if KEY_BLOCK == 1
         if (istep >= istart .and. iunldm > 0 .and. todo_now(nsavl,istep)) then
            !$omp barrier
            !$omp master
            call copy_alchemical_b2c(system)

            call msld_writld(nblock,npriv, &
                 istep,nstep, &
                 delta )
            !$omp end master
            !$omp barrier
         endif

     if (fill_msldata .and. (istep .eq. istop)) then
        call msldata_init(1)
        call msldata_set_names()
        call msldata_add_rows(istep, npriv * timest)
     end if
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
          blade_set_seed, &
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
     use parallel, only: mynodg
     use rndnum, only: rngseeds
     ! use omm_restraint, only: setup_restraints

      implicit none

      logical :: init

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
      call blade_set_verbose(system,blade_verbose_level)

      if (.not. blade_initialized) then
         call blade_set_seed(system,rngseeds(1),mynodg)
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

   subroutine blade_repd_energy(x, y, z, vx, vy, vz)
     use new_timer, only: T_energy, timer_start, timer_stop
     use blade_coords_module, only: copy_box_c2b, copy_coords_c2b, &
           copy_theta_c2b, copy_state_c2b, blade_send_coordinates, &
           blade_init_lambda_from_theta, copy_energy_b2c, blade_recv_energy
     use lambdam, only: qmld

     implicit none

     real(chm_real), intent(in) :: x(:), y(:), z(:)
     real(chm_real), intent(inout), optional :: vx(*), vy(*), vz(*)

     !$omp parallel
     call timer_start(T_energy) ! OMPWARNING
     if (present(vx)) then
        call copy_state_c2b(system, vx, vy, vz)
     else
        call copy_box_c2b(system)
        call copy_coords_c2b(system)
        if (qmld) call copy_theta_c2b(system)
        call blade_send_coordinates(system)
        if (qmld) call blade_init_lambda_from_theta(system)
     endif
     call blade_set_step(system, 0)
     call blade_update_domdec(system)
     call blade_get_force(system,1,0)
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
     ! use deriv  ! XXX writes
     ! use energym  ! XXX writes
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
     if (dynamics_mode /= DYN_CONTINUE) &
        call energy_initial_conditions(.true.) ! rectify shake constraints
     call blade_get_force(system,1,merge(0,1,dynamics_mode == DYN_CONTINUE))
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

   subroutine blade_minimize(x, y, z, nsteps, mintype, steplen, status)
     ! use, intrinsic::iso_c_binding, only: c_null_ptr
     use, intrinsic :: iso_c_binding, only: c_int
     use blade_coords_module, only: copy_energy_b2c, copy_force_b2c, &
           blade_recv_force, blade_recv_energy, &
           copy_state_b2c
     use psf, only: natom
     ! use omm_ecomp, only : omm_assign_eterms
     ! use energym  ! XXX writes
     ! use new_timer

     implicit none

     real(chm_real), intent(inout) :: x(:), y(:), z(:)
     real(chm_real), intent(in) :: steplen
     integer*4, intent(in) :: nsteps
     integer*4, intent(in) :: mintype
     integer*4, intent(out), optional :: status
     real(chm_real) :: vx(natom), vy(natom), vz(natom)
     integer(c_int) :: status_local, status_thread
     ! real(chm_real) :: Epterm
     ! type(OpenMM_State) :: state
     ! real*8 :: pos(3, NATOM)
     ! integer*4 :: data_wanted
     ! integer*4 :: enforce_periodic
     ! integer*4 :: itype, group

     status_local = 0

     !$omp parallel private(status_thread)

     call setup_blade()

     call blade_dynamics_initialize(system)
     call energy_initial_conditions(.true.) ! rectify shake constraints

     status_thread = blade_minimizer(system,nsteps,mintype,steplen)
     !$omp critical(blade_min_status)
     status_local = max(status_local, status_thread)
     !$omp end critical(blade_min_status)

     !$omp barrier
     !$omp master
     if (status_local == 0) then
        call blade_recv_energy(system)
        call copy_energy_b2c(system)
        call blade_recv_force(system)
        call copy_force_b2c(system)
     endif
     call copy_state_b2c(system,vx,vy,vz)
     !$omp end master
     !$omp barrier

     !$omp end parallel

     if (present(status)) status = status_local

   end subroutine blade_minimize

    !> Returns an array indicating whether we can use Blade
    !> for each energy term.
    function blade_eterm_mask()
      use energym, only: lenent, &
           bond, angle, dihe, imdihe, cmap, elec, vdw, imelec, imvdw, &
           ewksum, ewself, ewexcl, epot, totke, tote
#if KEY_MLMM==1 && KEY_MLPTORCH==1
      use energym, only: mlps ! eemlp
#endif
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
#if KEY_MLMM==1 && KEY_MLPTORCH==1
      blade_eterm_mask(mlps) = .true. ! eemlp
#endif
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
