module blade_ctrl_module
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV

   implicit none

   logical, save, private :: blade_active = .false.

   !> Flag to skip CPU neighbor list building when BLaDE is active.
   !> When true, nbonds() will skip the expensive NBNDCCF() call
   !> since BLaDE builds its own GPU-based neighbor lists.
   logical, save :: blade_skip_cpu_nblist = .false.

   !> Flag to force CPU neighbor list building even when BLaDE is active.
   !> Set by CPUNB keyword in NBONDS command. Resets to false after each call.
   logical, save :: blade_force_cpu_nblist = .false.

contains

! UNUSED
!    !> Set flag to skip CPU neighbor list building (C-callable)
!    subroutine blade_set_skip_cpu_nblist(skip) bind(c)
!       use, intrinsic :: iso_c_binding, only: c_int
!       integer(c_int), intent(in), value :: skip
!       blade_skip_cpu_nblist = (skip /= 0)
!    end subroutine blade_set_skip_cpu_nblist

   !> True if the present command includes the blade option
   !> or was preceded by an BLADE ON command, otherwise false.
   logical function blade_requested(comlyn, comlen, caller, &
        restart_mode, start_requested)
     use, intrinsic :: iso_c_binding, only: c_associated
     use string, only: indxa, nexta4
     use dimens_fcm, only: mxcmsz
     use blade_module, only: system
     use blade_main, only: &
          blade_initialized, system_dirty, dynamics_mode, &
          DYN_UNINIT, DYN_CONTINUE

      implicit none

      character(len=*), intent(in) :: comlyn
      integer, intent(in) :: comlen
      character(len=*), intent(in) :: caller
      integer, intent(in), optional :: restart_mode
      logical, intent(in), optional :: start_requested

      character(len=mxcmsz) :: gpu_com
      character(len=4) :: wrd
      integer :: gpu_com_index, gpu_com_len
      logical :: abic_requested, start_wins

      ! if (PRNLEV >= 2) write (OUTU, '(a)') 'BLaDE: Assuming system dirty until proven otherwise'
      system_dirty = .true.
      dynamics_mode = DYN_UNINIT

      blade_requested = blade_active
      if (INDXA(comlyn, comlen, 'BLADE') <= 0) return
#if KEY_BLADE == 1
      abic_requested = INDXA(comlyn, comlen, 'ABIC') > 0
      start_wins = .false.
      if (present(start_requested)) start_wins = start_requested

      if (abic_requested .and. present(restart_mode)) then
         if (restart_mode == 1) then
            call wrndie(-3, caller, &
               'BLaDE ABIC and RESTart are contradictory.')
         end if
      end if
      if (abic_requested .and. .not. start_wins) then
         if (.not. blade_initialized .or. .not. c_associated(system)) then
            call wrndie(-3, caller, &
               'BLaDE ABIC requires an initialized BLaDE state.')
         end if
      end if
      if (abic_requested .and. .not. start_wins) then
         system_dirty = .false.
         dynamics_mode = DYN_CONTINUE
         write (OUTU, '(a)') &
            'ABIC Keyword found: Assume Blade Is Current: Will not reinitialize BLaDE'
      end if

      gpu_com_index = indxa(comlyn, comlen, 'GPUI')
      if (gpu_com_index > 0) then

         gpu_com_len = comlen - gpu_com_index + 1
         gpu_com(1:gpu_com_len) = comlyn(gpu_com_index:comlen)
         call blade_parse_gpu(gpu_com, gpu_com_len)
      end if

      call blade_compatible(caller)
      blade_requested = .true.
#else /* KEY_BLADE */
      ! Silence the "unused dummy argument" warning when BLaDE is off.
      ! if (.false.) gpu_com_index = len_trim(gpu_com) + len(wrd) + gpu_com_len
      ! Above line was commented out because it will never be called (if false is always false)
      call wrndie(-3, caller, 'BLaDE code is not compiled.')
#endif /* KEY_BLADE */
   end function blade_requested

   !> interprets blade subcommand GPUIDS
   !> to specifiy specific compute devices via cuda gpu ids
   !> this should be a space delimited list of integers
   subroutine blade_parse_gpu(comlyn, comlen)
     use blade_main, only: ngpus, gpus
     use memory, only: chmalloc, chmrealloc, chmdealloc
     use string, only: nexti
     implicit none
     character(len=*) :: comlyn
     integer :: comlen
     integer :: n_new_gpus, max_new_gpus, next_gpu
     integer, dimension(:), allocatable :: new_gpus
     logical :: update_gpus

     max_new_gpus = 5
     n_new_gpus = 0
     call chmalloc(__FILE__, 'blade_parse_gpu', 'new_gpus', &
          n_new_gpus, intg=new_gpus)
     do while (comlen > 0) ! check for nexti
        next_gpu = nexti(comlyn, comlen)
        n_new_gpus = n_new_gpus + 1
        if (n_new_gpus > max_new_gpus) then
           max_new_gpus = max_new_gpus * 2
           call chmrealloc(__FILE__, 'blade_parse_gpu', 'new_gpus', &
                max_new_gpus, intg=new_gpus)
        end if
        new_gpus(n_new_gpus) = next_gpu
     end do

     update_gpus = .false.
     if (n_new_gpus > 0) then
        if (n_new_gpus .ne. ngpus) then
           update_gpus = .true.
        else if (.not. allocated(gpus)) then
           update_gpus = .true.
        else if (any(new_gpus .ne. gpus)) then
           update_gpus = .true.
        end if
     end if

     if (update_gpus) then
        if (.not. allocated(gpus)) then
           call chmalloc(__FILE__, 'blade_parse_gpu', 'gpus', &
                n_new_gpus, intg=gpus)
        end if
        if (n_new_gpus .ne. ngpus) then
           ngpus = n_new_gpus
           call chmrealloc(__FILE__, 'blade_parse_gpu', 'gpus', &
                ngpus, intg=gpus)
        end if
        gpus(1:ngpus) = new_gpus(1:ngpus)
     endif

     call chmdealloc(__FILE__, 'blade_parse_gpu', 'new_gpus', &
          n_new_gpus, intg=new_gpus)

     if (update_gpus) then  ! gpu ids changed, warning: need to turn blade off
        call wrndie(2, &
             '<blade_ctrl.F90>', &
             'GPUIds setting change only has effect after ' // &
             'GPU OFF/GPU ON cycle')
     end if
   end subroutine blade_parse_gpu

#if KEY_BLADE == 1
   !> Parse the verbose level for BLaDE debug messages
   subroutine blade_parse_verbose(comlyn, comlen)
     use string, only: nexti
     use blade_main, only: blade_set_verbose_level
     implicit none
     character(len=*) :: comlyn
     integer :: comlen
     integer :: verbose_level
     
     verbose_level = nexti(comlyn, comlen)
     if (verbose_level < 0) verbose_level = 0
     call blade_set_verbose_level(verbose_level)
     if (PRNLEV >= 2) then
        write (OUTU, '(a,i0)') 'BLaDE verbose level set to: ', verbose_level
     end if
   end subroutine blade_parse_verbose

! UNUSED
!    !> Get the current BLaDE verbose level
!    integer function get_blade_verbose_level()
!       use blade_main, only: blade_get_verbose_level
!       get_blade_verbose_level = blade_get_verbose_level()
!    end function get_blade_verbose_level
#endif /* KEY_BLADE */

   !> Interprets a top-level command to enable or disable BLaDE
   !> BLADE ON - Sets blade_active. System may be created later as needed.
   !> BLADE OFF - Clears blade_active but retains system.

   subroutine blade_command(comlyn, comlen)
     ! use, intrinsic :: iso_c_binding, only: c_char, c_null_char
     use stream, only: outu
     use string, only: nexta4, nextwd

#if KEY_BLADE == 1
     use blade_module, only: system, blade_fn_use, blade_fn_len, blade_fn_max, blade_fname
     use blade_main, only: setup_blade
#endif /* KEY_BLADE */

      implicit none

      character(len=*), intent(inout) :: comlyn
      integer, intent(inout) :: comlen

      logical :: blade_state_cmd
      character(len=20) :: blank
      character(len=4) :: wrd, wrd1

      integer :: i, ishift

      blade_state_cmd = .false.

#if KEY_BLADE == 1
      do while (comlen > 0)
         wrd = nexta4(comlyn,comlen)

         cmds: select case(wrd)
         case('ON  ') cmds
            blade_state_cmd = .true.
            blade_active = .true.
            blade_skip_cpu_nblist = .true.  ! Skip CPU neighbor list building
            ! call setup_blade() ! Setup when dynamics or energy is called
         case('OFF ') cmds
            blade_state_cmd = .true.
            blade_active = .false.
            blade_skip_cpu_nblist = .false.  ! Resume CPU neighbor list building
            ! Clean up GPU resources to make device available
            call teardown_blade_system()
         case('FILE') cmds
            call nextwd(comlyn, comlen, blade_fname, blade_fn_max, blade_fn_len)
            ishift = 0
            do i = 1, blade_fn_len
               if (blade_fname(i:i) .eq. '"') then
                  ishift = ishift + 1
               else if (ishift .ne. 0) then
                  blade_fname(i-ishift:i-ishift)=blade_fname(i:i)
               end if
            end do
            blade_fn_len = blade_fn_len - ishift
            blade_fn_use = .true.
            ! Moved to blade_main.F90 setup_system
            ! call blade_interpretter(blade_fname(1:blade_fn_len) // c_null_char, system)
         case('GPUI') cmds
            call blade_parse_gpu(comlyn, comlen)
         case('VERB') cmds
            call blade_parse_verbose(comlyn, comlen)
         end select cmds
      end do

      if (blade_state_cmd .and. PRNLEV >= 2) then
         if (blade_active) then
            call blade_compatible('<CHARMM>')
            write (OUTU, '(a)') &
                 'Energy and dynamics calculations will use BLaDE.'
         else
            write (OUTU, '(a)') &
                 'Energy and dynamics calculations will not use BLaDE unless requested.'
         end if
      end if
#else /* KEY_BLADE */
      call wrndie(-1, '<CHARMM>', 'BLaDE code is not compiled.')
#endif /* KEY_BLADE */
    end subroutine blade_command

   !> Check whether unsupported features are turned on
   subroutine blade_compatible(caller)
      ! use replica_ltm, only: qRep
      character(len=*), intent(in) :: caller

#if KEY_BLADE == 1
      ! if (qRep) then
      !    call wrndie(-1, caller, 'Replicas not supported with BLaDE')
      ! endif
#else /* KEY_BLADE */
      call wrndie(-1, '<CHARMM>', 'BLaDE code is not compiled.')
#endif /* KEY_BLADE */
    end subroutine blade_compatible

   !> Clean up BLaDE system and GPU resources
   subroutine teardown_blade_system()
#if KEY_BLADE == 1
      use blade_module, only: system
      use blade_main, only: teardown_system
      use, intrinsic :: iso_c_binding, only: c_associated
      
      ! Force synchronization before cleanup
      if (c_associated(system)) then
         ! Add verbose output for debugging
         if (PRNLEV >= 2) then
            write (OUTU, '(a)') 'BLaDE: Starting GPU cleanup...'
         end if
         
         call teardown_system(system)
         
         if (PRNLEV >= 2) then
            write (OUTU, '(a)') 'BLaDE: GPU cleanup completed.'
         end if
      else
         if (PRNLEV >= 2) then
            write (OUTU, '(a)') 'BLaDE: No active system to clean up.'
         end if
      end if
#endif /* KEY_BLADE */
   end subroutine teardown_blade_system
end module blade_ctrl_module
