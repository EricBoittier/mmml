!> routines to configure and run molecular dynamics
module api_minimize
  implicit none
contains

#if KEY_LIBRARY == 1
  !> @brief run steepest descent minimization
  !
  !> param[in] min_opts type(min_settings) options for the minimization algorithm
  !> param[in] sd_opts type(min_sd_settings) options specific to the steepest descent algorithm
  !> @return integer(c_int) 1 if successful, otherwise there was an error
  integer(c_int) function minimize_run_sd(min_opts, sd_opts) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_types, only: min_settings, min_sd_settings
    use minmiz_module, only: minmiz
    implicit none

    type(min_settings) :: min_opts
    type(min_sd_settings) :: sd_opts

    character(len=128) :: command_line = ' '
    integer :: command_line_len = 0

    minimize_run_sd = 0
    call minmiz(command_line, command_line_len, min_opts, sd_opts=sd_opts)
    minimize_run_sd = 1    
  end function minimize_run_sd

  !> @brief run the minimization
  !
  !> param[in] min_opts type(min_settings) options for the minimization algorithm
  !> param[in] abnr_opts type(min_abnr_settings) options specific to the ABNR algorithm
  !> @return integer(c_int) 1 if successful, otherwise there was an error
  integer(c_int) function minimize_run_abner(min_opts, abnr_opts) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_types, only: min_settings, min_abnr_settings
    use minmiz_module, only: minmiz
    implicit none

    type(min_settings) :: min_opts
    type(min_abnr_settings) :: abnr_opts
    
    character(len=128) :: command_line = ' '
    integer :: command_line_len = 0

    minimize_run_abner = 0
    call minmiz(command_line, command_line_len, min_opts, abnr_opts=abnr_opts)
    minimize_run_abner = 1    
  end function minimize_run_abner

  !> @brief ABNR with optional CRYSTAL LATTice / NOCOords (KEY_LIBRARY dynopt path).
  integer(c_int) function minimize_run_abnr_lattice(min_opts, abnr_opts, &
       lattice_flag, nocoords_flag) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_types, only: min_settings, min_abnr_settings
    use minmiz_module, only: minmiz
    implicit none

    type(min_settings) :: min_opts
    type(min_abnr_settings) :: abnr_opts
    integer(c_int), value :: lattice_flag, nocoords_flag

    character(len=128) :: command_line = ' '
    integer :: command_line_len = 0

    minimize_run_abnr_lattice = 0
    if (lattice_flag /= 0) then
       command_line = 'LATT'
       command_line_len = 4
       if (nocoords_flag /= 0) then
          command_line(5:9) = ' NOCO'
          command_line_len = 9
       end if
    end if
    call minmiz(command_line, command_line_len, min_opts, abnr_opts=abnr_opts)
    minimize_run_abnr_lattice = 1
  end function minimize_run_abnr_lattice
#endif /* KEY_LIBRARY */

end module api_minimize
