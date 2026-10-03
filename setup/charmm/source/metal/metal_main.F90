! ===========================================================================
! metal_main.F90  —  Fortran device-management module for the Metal backend
! ===========================================================================
!
! Parallel in structure to source/opencl/opencl_main.F90 (module
! opencl_main_mod).  Provides device enumeration, selection, and simple
! session begin / end so CHARMM code paths that only need "give me a
! Metal device" can do so without pulling in source/fftdock/.
!
! C entry points live in source/metal/metal_util.mm (mtl_* prefix).
!
! CONSUMERS
! ---------
! - source/fftdock/fftdock.F90 (KEY_METAL branch) currently forwards its
!   GPUI <n> argument directly to metal_fftdock_setup; this module is
!   optional for FFTDOCK but provides a stable public API for future
!   METAL DEVI SELE <n> style commands and for other Metal consumers.
! - Future Metal-backed modules (e.g. a Metal DOMDEC port) would use this
!   module as their device-picker without knowing FFTDOCK internals.
!
! COMPILE FLAG
!   KEY_METAL == 1   activates this module
!
! - YWu
! ===========================================================================

module metal_main_mod

#if KEY_METAL == 1

  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_char, c_int, c_associated

  implicit none

  !
  ! Module-wide state (mirrors opencl_main_mod):
  !
  !   metal_devices    — opaque NSArray<id<MTLDevice>>*  (from mtl_device_init)
  !   selected_device  — opaque id<MTLDevice>*           (one element of above)
  !   metal_is_initialized — .true. after mtl_device_init succeeds
  !
  ! Both pointers are owned by the C side (retained); use
  ! metal_device_list_release()/mtl_end_session() on shutdown.
  !
  type(c_ptr), save :: metal_devices    = c_null_ptr
  type(c_ptr), save :: selected_device  = c_null_ptr
  logical,     save :: metal_is_initialized = .false.

  !
  ! ----------------------------------------------------------------------- !
  !  C interface — implementations in source/metal/metal_util.mm            !
  ! ----------------------------------------------------------------------- !
  !
  interface
     function mtl_device_init(devices_out) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr)    :: devices_out           ! OUT
       integer(c_int) :: status
     end function mtl_device_init

     subroutine mtl_device_list_release(devices) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr) :: devices                  ! INOUT (zeroed on return)
     end subroutine mtl_device_list_release

     subroutine mtl_device_print(devices) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: devices
     end subroutine mtl_device_print

     subroutine mtl_device_print_one(dev) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: dev
     end subroutine mtl_device_print_one

     subroutine mtl_device_string(dev, c_string, max_size) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int
       implicit none
       type(c_ptr), value            :: dev
       character(len=1, kind=c_char) :: c_string(*)
       integer(c_int), value         :: max_size
     end subroutine mtl_device_string

     function mtl_device_get(devices, dev_id, out_device) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value    :: devices
       integer(c_int), value :: dev_id          ! 1-based
       type(c_ptr)           :: out_device      ! OUT
       integer(c_int)        :: status
     end function mtl_device_get

     function mtl_device_max_mem_get(devices, out_device) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: devices
       type(c_ptr)        :: out_device         ! OUT
       integer(c_int)     :: status
     end function mtl_device_max_mem_get

     function mtl_begin_session(in_dev) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: in_dev
       integer(c_int)     :: status
     end function mtl_begin_session

     function mtl_end_session() bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
       integer(c_int) :: status
     end function mtl_end_session
  end interface

contains

  ! ------------------------------------------------------------------------
  ! metal_init
  !   Populate `metal_devices` from mtl_device_init.  Safe to call
  !   repeatedly — later calls are a no-op after the first success.
  ! ------------------------------------------------------------------------
  subroutine metal_init()
    implicit none
    integer(c_int) :: status
    if (metal_is_initialized) return
    metal_devices    = c_null_ptr
    selected_device  = c_null_ptr
    status = mtl_device_init(metal_devices)
    if (status /= 0) then
       metal_is_initialized = .false.
       return
    end if
    metal_is_initialized = .true.
  end subroutine metal_init

  ! ------------------------------------------------------------------------
  ! metal_device_show
  !   Print all available Metal devices (equivalent of OCL DEVI SHOW).
  ! ------------------------------------------------------------------------
  subroutine metal_device_show()
    implicit none
    if (.not. metal_is_initialized) call metal_init()
    if (c_associated(metal_devices)) call mtl_device_print(metal_devices)
  end subroutine metal_device_show

  ! ------------------------------------------------------------------------
  ! metal_device_show_current
  !   Print the currently selected device (equivalent of OCL DEVI SHOW CURR).
  ! ------------------------------------------------------------------------
  subroutine metal_device_show_current()
    use, intrinsic :: iso_c_binding, only: c_associated
    use stream,   only: outu
    implicit none
    if (.not. c_associated(selected_device)) then
       write(outu, '(a)') 'No Metal device is selected'
    else
       write(outu, '(a)') ' Metal device: (id #) (name) (bytes of memory)'
       call mtl_device_print_one(selected_device)
    end if
  end subroutine metal_device_show_current

  ! ------------------------------------------------------------------------
  ! metal_device_select
  !   Select the dev_i-th device (1-based) from the enumeration list.
  !   Does NOT start a session — call metal_begin_session when ready.
  ! ------------------------------------------------------------------------
  subroutine metal_device_select(dev_i)
    implicit none
    integer(c_int), intent(in) :: dev_i
    integer(c_int)             :: status
    if (.not. metal_is_initialized) call metal_init()
    if (.not. c_associated(metal_devices)) return
    status = mtl_device_get(metal_devices, dev_i, selected_device)
    if (status /= 0) selected_device = c_null_ptr
  end subroutine metal_device_select

  ! ------------------------------------------------------------------------
  ! metal_device_select_default
  !   Pick the device with the largest recommendedMaxWorkingSetSize.
  ! ------------------------------------------------------------------------
  subroutine metal_device_select_default()
    implicit none
    integer(c_int) :: status
    if (.not. metal_is_initialized) call metal_init()
    if (.not. c_associated(metal_devices)) return
    status = mtl_device_max_mem_get(metal_devices, selected_device)
    if (status /= 0) selected_device = c_null_ptr
  end subroutine metal_device_select_default

  ! ------------------------------------------------------------------------
  ! metal_begin_session
  !   Start a Metal session bound to `selected_device`.  Equivalent of
  !   ocl_begin_session, but returns no context/queue (Metal uses the
  !   device as the primary state object).  Sub-modules retrieve the
  !   queue later via metal_get_queue() (C-level) if they need it.
  ! ------------------------------------------------------------------------
  function metal_begin_session() result(status)
    implicit none
    integer(c_int) :: status
    if (.not. c_associated(selected_device)) then
       status = -1
       return
    end if
    status = mtl_begin_session(selected_device)
  end function metal_begin_session

  ! ------------------------------------------------------------------------
  ! metal_end_session
  !   Tear down the session; releases the command queue and library but
  !   leaves the device enumeration list intact.
  ! ------------------------------------------------------------------------
  function metal_end_session() result(status)
    implicit none
    integer(c_int) :: status
    status = mtl_end_session()
  end function metal_end_session

#endif /* KEY_METAL */

end module metal_main_mod
