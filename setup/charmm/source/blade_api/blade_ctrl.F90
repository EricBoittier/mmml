module blade_ctrl_module
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV

   implicit none

   logical, save, private :: blade_active = .false.

contains

   !> True if the present command includes the blade option
   !> or was preceded by an BLADE ON command, otherwise false.
   logical function blade_requested(comlyn, comlen, caller)
     use string, only: indxa, nexta4
     use dimens_fcm, only: mxcmsz
     use blade_main, only: &
          system_dirty, dynamics_mode, &
          DYN_UNINIT, DYN_CONTINUE

      implicit none

      character(len=*), intent(in) :: comlyn
      integer, intent(in) :: comlen
      character(len=*), intent(in) :: caller

      character(len=mxcmsz) :: gpu_com
      character(len=4) :: wrd
      integer :: gpu_com_index, gpu_com_len

      ! if (PRNLEV >= 2) write (OUTU, '(a)') 'BLaDE: Assuming system dirty until proven otherwise'
      system_dirty = .true.
      dynamics_mode = DYN_UNINIT

      blade_requested = blade_active
      if (INDXA(comlyn, comlen, 'BLADE') <= 0) return

      if (INDXA(comlyn, comlen, 'ABIC') > 0) then
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
            ! call setup_blade() ! Setup when dynamics or energy is called
         case('OFF ') cmds
            blade_state_cmd = .true.
            blade_active = .false.
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
end module blade_ctrl_module
