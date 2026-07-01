module opencl_parse_mod
  implicit none
  private :: ocl_parse_bad_command
  public :: ocl_parse
contains
  subroutine ocl_parse(comlyn, comlen)
    use string, only: nexta4
    use opencl_main_mod, only: ocl_device_show, ocl_device_select
    implicit none

    ! args
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    !local vars
    character(len=4) :: ocl_subcommand, device_subcommand
    integer :: device_selected

    ocl_subcommand = nexta4(comlyn, comlen)
    commands: select case(ocl_subcommand)
    case('DEVI') commands
       call ocl_parse_device(comlyn, comlen)
    case default
       call ocl_parse_bad_command(ocl_subcommand)
    end select commands
  end subroutine ocl_parse

  subroutine ocl_parse_bad_command(command)
    implicit none
    character(len=4), intent(in) :: command
    call wrndie(-3, '<opencl>', 'Unrecognized command: ' // command)
  end subroutine ocl_parse_bad_command

  subroutine ocl_parse_device(comlyn, comlen)
    use string, only: nexta4, nexti
    use opencl_main_mod, only: ocl_device_show, ocl_device_select
    implicit none

    ! args
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    !local vars
    character(len=4) :: device_subcommand
    integer :: device_selected

    device_subcommand = nexta4(comlyn, comlen)
    device_commands: select case(device_subcommand)
    case('SHOW') device_commands
       call ocl_parse_show(comlyn, comlen)
    case('SELE') device_commands
       device_selected = nexti(comlyn, comlen)
       call ocl_device_select(device_selected)
    case default
       call ocl_parse_bad_command(device_subcommand)
    end select device_commands
  end subroutine ocl_parse_device

  subroutine ocl_parse_show(comlyn, comlen)
    use string, only: nexta4, nexti
    use opencl_main_mod, only: ocl_device_show, ocl_device_show_current
    implicit none

    ! args
    character(len=*), intent(inout) :: comlyn
    integer, intent(inout) :: comlen

    !local vars
    character(len=4) :: show_subcommand

    show_subcommand = nexta4(comlyn, comlen)
    show_commands: select case(show_subcommand)
    case('   ') show_commands
       call ocl_device_show()
    case('ALL ')
       call ocl_device_show()
    case('CURR') show_commands
       call ocl_device_show_current()
    case default
       call ocl_parse_bad_command(show_subcommand)
    end select show_commands
  end subroutine ocl_parse_show
end module opencl_parse_mod
