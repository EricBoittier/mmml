!> process, store, and export multi site lambda data from last dynamics run
module api_msldata
  use api_dataframe, only: t_dataframe
  implicit none

#if KEY_LIBRARY == 1 && KEY_BLOCK == 1
  logical :: &
       fill_msldata = .false., &  ! requesting dynamics to collect msld data?
       is_active = .false.  ! have the msld data structures been initialized?

  integer, parameter :: &
       step_cols = 7, &
       bias_cols = 7, &
       block_cols = 3

  type(t_dataframe) :: &
       block_data, &
       bias_data, &
       step_data, & ! step #, time, nblocks, nbiasv, nsitemld
       theta_data, &
       nsubs_data

contains

  !> @brief Have dataframes been initialized and allocated?
  !
  !> @return 1 <==> dataframes initialized and allocated, 0 otherwise
  function msldata_is_active() bind(c) result(out_is_active)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: out_is_active

    out_is_active = 0
    if (is_active) out_is_active = 1
  end function msldata_is_active

  !> @brief get the max number of rows allocated for step data
  !
  !> @return the max number of rows allocated for step data
  function msldata_get_nsteps() bind(c) result(nsteps)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: nsteps
    nsteps = dataframe_get_nrows(step_data)
  end function msldata_get_nsteps

  !> @brief get the max number of cols allocated for step data
  !
  !> @return the max number of cols
  function msldata_step_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: ncols
    ncols = step_cols
  end function msldata_step_get_ncols

  !> @brief get the max number of cols allocated for bias data
  !
  !> @return the max number of cols
  function msldata_bias_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: ncols
    ncols = bias_cols
  end function msldata_bias_get_ncols

  !> @brief get the max number of cols allocated for block data
  !
  !> @return the max number of cols
  function msldata_block_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: ncols
    ncols = block_cols
  end function msldata_block_get_ncols

  !> @brief get the max number of cols allocated for theta data
  !
  !> @return the max number of cols
  function msldata_theta_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: ncols
    ncols = theta_data%ncols
  end function msldata_theta_get_ncols

  !> @brief get a copy of the step data
  !
  !> @param[out] out_names a string array for each column name
  !> @param[out] out_rows a flattened matrix of data points
  subroutine msldata_step_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(step_data, out_names, out_rows)
  end subroutine msldata_step_get

  !> @brief get a copy of the bias data
  !
  !> @param[out] out_names a string array for each column name
  !> @param[out] out_rows a flattened matrix of data points
  subroutine msldata_bias_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(bias_data, out_names, out_rows)
  end subroutine msldata_bias_get

  !> @brief get a copy of the block data
  !
  !> @param[out] out_names a string array for each column name
  !> @param[out] out_rows a flattened matrix of data points
  subroutine msldata_block_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(block_data, out_names, out_rows)
  end subroutine msldata_block_get

  !> @brief get a copy of the theta data
  !
  !> @param[out] out_rows a flattened matrix of data points
  subroutine msldata_theta_get(out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use api_dataframe, only: dataframe_get_values
    implicit none
    real(c_double), dimension(*) :: out_rows
    call dataframe_get_values(theta_data, out_rows)
  end subroutine msldata_theta_get

  !> @brief get a copy of the nsubs data
  !
  !> @param[out] out_rows a flattened matrix of data points
  subroutine msldata_nsubs_get(out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double
    use api_dataframe, only: dataframe_get_values
    implicit none
    real(c_double), dimension(*) :: out_rows
    call dataframe_get_values(nsubs_data, out_rows)
  end subroutine msldata_nsubs_get

  !> @brief fill the dataframes the next dynamics run
  !
  !> @return Was fill already on? 1 <==> yes, 0 otherwise
  integer(c_int) function msldata_on() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none

    msldata_on = 0
    if (fill_msldata) msldata_on = 1

    fill_msldata = .true.
  end function msldata_on

  !> @brief do not fill dataframes the next dynamics run
  !
  !> @return Was fill on? 1 <==> yes, 0 otherwise
  integer(c_int) function msldata_off() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    msldata_off = 1
    if (fill_msldata) msldata_off = 0

    fill_msldata = .false.
  end function msldata_off

  !> @brief allocate storage and initialize attributes
  !
  ! If fill_msldata is true, the next dynamics run,
  ! the final state of the lambda dynamics data will be
  ! saved in this module.
  !
  !> @param[in] nsteps total number of expected dynamics steps
  subroutine msldata_init(nsteps)
    use api_dataframe, only: dataframe_init
    use lambdam, only: &
         nbiasv, nblock, &
         fcnal_form, nsitemld, nsubmld
    implicit none
    integer, intent(in) :: nsteps

    integer :: max_nsubs

    max_nsubs = 0

    call dataframe_init(nsteps, step_cols, step_data)
    call dataframe_init(nbiasv * nsteps, bias_cols, bias_data)
    call dataframe_init(nblock * nsteps, block_cols, block_data)
    if (fcnal_form == '2sin' .or. fcnal_form == '2exp') then
       call dataframe_init(nsteps, nsitemld - 1, theta_data)
    else
       max_nsubs = maxval(nsubmld(2:nsitemld))
       call dataframe_init(nsteps * (nsitemld - 1), max_nsubs, theta_data)
       call dataframe_init(nsteps, nsitemld - 1, nsubs_data)
    end if
    is_active = .true.
  end subroutine msldata_init

  !> @brief free storage and zero out parameters
  subroutine msldata_del() bind(c)
    use api_dataframe, only: dataframe_del
    implicit none
    call dataframe_del(step_data)
    call dataframe_del(bias_data)
    call dataframe_del(block_data)
    is_active = .false.
  end subroutine msldata_del

  !> @brief set the column names
  !
  ! The names are fixed. Call this routine in the dynamics routine
  ! only after msldata_init
  subroutine msldata_set_names()
    use api_dataframe, only: dataframe_name_size, dataframe_set_names
    implicit none

    character(len=dataframe_name_size), dimension(:), allocatable :: new_names

    new_names = (/ 'STEP   ', 'TIME   ', 'NBLOCKS', 'NBIASV ', 'NSITES ', &
         'TBLD   ', 'FCNFORM' /)
    call dataframe_set_names(step_data, new_names)

    new_names = (/ 'IBVIDI', 'IBVIDJ', 'IBCLAS', 'IRREUP', 'IRRLOW', &
         'IKBIAS', 'IPBIAS' /)
    call dataframe_set_names(bias_data, new_names)

    new_names = (/ 'ISITE ', 'BIELAM', 'BIXLAM' /)
    call dataframe_set_names(block_data, new_names)
  end subroutine msldata_set_names

  !> @brief convert fncal_form to a real so we can put it into a dataframe
  !
  !> @param[in] string representing the fncal_form; won't be modified
  !> @return time the current simulation time for the dynamics step
  function convert_form(old_form) result(new_form)
    use chm_kinds, only: chm_real
    implicit none
    character(len=*) :: old_form
    real(chm_real) :: new_form

    new_form = 0.0
    if (old_form == '2sin') then
       new_form = 1.0
    else if (old_form == 'nsin') then
       new_form = 2.0
    else if (old_form == '2exp') then
       new_form = 3.0
    else if (old_form == 'nexp') then
       new_form = 4.0
    else if (old_form == 'norm') then
       new_form = 5.0
    else if (old_form == 'fixd') then
       new_form = 6.0
    end if
  end function convert_form

  !> @brief add a set of lambda dynamics data
  !
  ! Call this routine in the dynamics routine only after msldata_init.
  !
  !> @param[in] step the current step number of dynamics
  !> @param[in] time the current simulation time for the dynamics step
  subroutine msldata_add_rows(step, time)
    use lambdam,only: &
         nblock, nbiasv, nsitemld, tbld, fcnal_form, &
         ibvidi, ibvidj, ibclas, irreup, irrlow, ikbias, ipbias, &
         isitemld, bielam, bixlam, &
         nsubmld, thetamld
    use chm_kinds, only: chm_real
    use api_dataframe, only: dataframe_add_row

    implicit none

    integer, intent(in) :: step
    real(chm_real), intent(in) :: time

    real(chm_real), dimension(:), allocatable :: new_row
    real(chm_real) :: new_step, new_form
    integer :: i, j

    new_step = real(step, chm_real)
    new_form = convert_form(fcnal_form)

    do i = 1, msldata_get_nsteps()
       new_row = (/ new_step, time, &
            real(nblock, chm_real), &
            real(nbiasv, chm_real), &
            real(nsitemld, chm_real), &
            tbld, new_form /)
       call dataframe_add_row(step_data, new_row)
    end do

    do i = 1, nbiasv
       new_row = (/ real(ibvidi(i), chm_real), &
            real(ibvidj(i), chm_real), &
            real(ibclas(i), chm_real),  &
            irreup(i), irrlow(i), &
            ikbias(i), real(ipbias(i), chm_real) /)
       call dataframe_add_row(bias_data, new_row)
    end do

    do i = 1, nblock
       new_row = (/ real(isitemld(i), chm_real), &
            bielam(i), &
            bixlam(i) /)
       call dataframe_add_row(block_data, new_row)
    end do

    if (fcnal_form == '2sin' .or. fcnal_form == '2exp') then
       new_row = (/(thetamld(i, 1), i = 2, nsitemld)/)
       call dataframe_add_row(theta_data, new_row)
    else
       do i = 2, nsitemld
          new_row = (/(thetamld(i, j), j = 1, nsubmld(i))/)
          call dataframe_add_row(theta_data, new_row)
       end do
       new_row = (/(real(nsubmld(i), chm_real), i = 2, nsitemld)/)
       call dataframe_add_row(nsubs_data, new_row)
    end if
  end subroutine msldata_add_rows
#endif /* KEY_LIBRARY */
end module api_msldata
