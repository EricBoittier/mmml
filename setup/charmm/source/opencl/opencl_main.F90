module opencl_main_mod
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
  implicit none

  ! public :: ocl_init, ocl_device_show, ocl_device_select

  type(c_ptr) :: &
       ocl_devices = c_null_ptr, &
       selected_device = c_null_ptr

  logical :: ocl_is_initialized = .false.

  interface
     function ocl_device_init(devices) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr) :: devices
       integer(c_int) :: status
     end function ocl_device_init

     subroutine ocl_device_print(devices) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: devices
     end subroutine ocl_device_print

     subroutine ocl_device_print_one(dev) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: dev
     end subroutine ocl_device_print_one

     subroutine ocl_device_string(dev, c_string, max_size) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int
       implicit none
       type(c_ptr), value :: dev
       character(len=1, kind=c_char) :: c_string(*)
       integer(c_int), value :: max_size
     end subroutine ocl_device_string

     function ocl_device_get(devices, dev_id, out_device) bind(c) &
          result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       ! input arguments
       type(c_ptr), value :: devices
       integer(c_int), value :: dev_id

       ! output argument
       type(c_ptr) :: out_device

       ! result returned
       integer(c_int) :: status
     end function ocl_device_get

     function ocl_device_max_mem_get(devices, out_device) bind(c) &
          result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int

       implicit none

       ! input arguments
       type(c_ptr), value :: devices

       ! output argument
       type(c_ptr) :: out_device

       ! result returned
       integer(c_int) :: status
     end function ocl_device_max_mem_get

     function ocl_begin_session(in_dev, out_ctx, out_q) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       ! input argument
       type(c_ptr), value :: in_dev
       ! output arguments
       type(c_ptr) :: out_ctx, out_q
       ! result returned
       integer(c_int) :: status
     end function ocl_begin_session
     
     function ocl_end_session(ctx, q) bind(c) result(status)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       ! arguments
       type(c_ptr) :: ctx, q
       ! result returned
       integer(c_int) :: status
     end function ocl_end_session
  end interface
contains
  subroutine ocl_init()
    use, intrinsic :: iso_c_binding, only: c_null_ptr, c_int
    implicit none
    integer(c_int) :: status
    ocl_devices = c_null_ptr
    selected_device = c_null_ptr
    status = ocl_device_init(ocl_devices)
    ocl_is_initialized = .true.
  end subroutine ocl_init

  subroutine ocl_device_show()
    implicit none
    if (.not. ocl_is_initialized) then
       call ocl_init()
    end if
    call ocl_device_print(ocl_devices)
  end subroutine ocl_device_show

  subroutine ocl_device_show_current()
    use, intrinsic :: iso_c_binding, only: c_associated, c_char
    use stream, only: outu, prnlev
    use api_util, only: c2f_string
    implicit none

    integer, parameter :: max_size = 80
    character(len=1, kind=c_char) :: c_string(max_size)
    character(len=max_size) :: dev_string

    if (.not. c_associated(selected_device)) then
       write(outu, '(a)') 'No OpenCL device is selected'
    else
       call ocl_device_string(selected_device, c_string, max_size)
       dev_string = c2f_string(c_string, max_size)
       write(outu, '(a, /a)') &
            ' OpenCL device: (id #) (name) (bytes of memory)', &
            trim(dev_string)
    end if
  end subroutine ocl_device_show_current

  subroutine ocl_device_select(dev_i)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int), intent(in) :: dev_i
    integer(c_int) :: status
    if (.not. ocl_is_initialized) then
       call ocl_init()
    end if
    status = ocl_device_get(ocl_devices, dev_i, selected_device)
  end subroutine ocl_device_select
end module opencl_main_mod
