!> process, store, and export velocity xyz for each dyn step
module api_velos
  use api_dataframe, only: t_dataframe
  implicit none

#if KEY_LIBRARY == 1
  logical :: fill_velos = .false.

  type(t_dataframe) :: velos

contains

  !> @brief Has the velos table been initialized and allocated?
  !
  !> @return 1 <==> velos is initialized and allocated, 0 otherwise
  function velos_is_active() bind(c) result(is_active)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: is_active

    is_active = 0
    if (velos%is_active) is_active = 1
  end function velos_is_active

  !> @brief deallocate velos storage
  subroutine velos_del() bind(c)
    use api_dataframe, only: dataframe_del
    implicit none
    call dataframe_del(velos)
  end subroutine velos_del

  !> @brief get the max number of rows allocated for the velos
  !
  !> @return the max number of rows allocated for the velos
  function velos_get_nrows() bind(c) result(nrows)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: nrows
    nrows = dataframe_get_nrows(velos)
  end function velos_get_nrows

  !> @brief get the max number of columns allocated for the velos
  !
  !> @return the max number of columns allocated for the velos
  function velos_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_ncols
    implicit none
    integer(c_int) :: ncols
    ncols = dataframe_get_ncols(velos)
  end function velos_get_ncols

  !> @brief get a copy of the velos
  !
  !> @param[out] out_names a string array for the name of each column
  !> @param[out] out_rows a flattened matrix of data points
  subroutine velos_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(velos, out_names, out_rows)
  end subroutine velos_get

  !> @brief fill the velos next time dynamics is run
  !
  !> @return Was 'fill velos' already on? 1<==>yes, 0 otherwise
  integer(c_int) function velos_on() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    velos_on = 0
    if (fill_velos) velos_on = 1

    fill_velos = .true.
  end function velos_on

  !> @brief do not fill the velos next time dynamics is run
  !
  !> @return Was 'fill velos' on? 1<==>yes, 0 otherwise
  integer(c_int) function velos_off() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    velos_off = 1
    if (fill_velos) velos_off = 0

    fill_velos = .false.
  end function velos_off

  !> @brief allocate storage for velos and initialize velos attributes
  !
  ! If fill_velos is true, on the next dynamics run, dynamics will run
  ! from istart step to istop step, adding natom rows to velos every nsavv-th
  ! step. So velos needs (((istop - istart) / nsavv) + 1) * natom rows.
  ! The number of columns is fixed at 5: step, time, vx, vy, vz.
  !
  !> @param[in] istart the upcoming dynamics run starts on this step number
  !> @param[in] istop the upcoming dynamics run stops on this step number
  !> @param[in] nsavv traditional charmm output written every nsavv steps
  subroutine velos_init()
    use api_dataframe, only: dataframe_init
    use psf, only: natom
    implicit none

    integer :: nrows, ncols

    nrows = natom
    ncols = 5  ! step, time, vx, vy, vz

    call dataframe_init(nrows, ncols, velos)
  end subroutine velos_init

  !> @brief set the names of the columns for velos
  !
  ! The names are fixed. This routine should be called in the dynamics routine
  ! after velos_init
  subroutine velos_set_names()
    use api_dataframe, only: dataframe_name_size, dataframe_set_names
    implicit none

    character(len=dataframe_name_size), dimension(1:5) :: new_names

    new_names = (/ 'STEP', 'TIME', 'VX  ', 'VY  ', 'VZ  ' /)
    call dataframe_set_names(velos, new_names)
  end subroutine velos_set_names

  !> @brief add a set of velocities to velos
  !
  ! This routine should be called in the dynamics routine.
  !
  !> @param[in] step the current step number of dynamics
  !> @param[in] time the current simulation time for the dynamics step
  !> @param[in] vx x coord of the velocities
  !> @param[in] vy y coord of the velocities
  !> @param[in] vz z coord of the velocities
  subroutine velos_add_rows(step, time, vx, vy, vz)
    use chm_kinds, only: chm_real
    use api_dataframe, only: dataframe_add_row
    use psf, only: natom

    implicit none

    integer, intent(in) :: step
    real(chm_real), intent(in) :: time
    real(chm_real), dimension(1:natom), intent(in) :: vx, vy, vz

    real(chm_real), dimension(1:5) :: new_row
    real(chm_real) :: new_step
    integer :: i

    new_step = real(step, chm_real)
    do i = 1, natom
       new_row = (/ new_step, time, vx(i), vy(i), vz(i) /)
       call dataframe_add_row(velos, new_row)
    end do
  end subroutine velos_add_rows
#endif /* KEY_LIBRARY */
end module api_velos
