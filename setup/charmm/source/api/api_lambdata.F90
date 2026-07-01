!> process, store, and export lambda data from the last dynamics run
module api_lambdata
  use api_dataframe, only: t_dataframe
  implicit none

#if KEY_LIBRARY == 1 && KEY_BLOCK == 1
  logical :: fill_lambdata = .false.
  type(t_dataframe) :: bias_data, bixlamsq_data

contains

  !> @brief Have dataframes been initialized and allocated?
  !
  !> @return 1 <==> dataframes initialized and allocated, 0 otherwise
  function lambdata_is_active() bind(c) result(is_active)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: is_active

    is_active = 0
    if (bias_data%is_active .and. bixlamsq_data%is_active) is_active = 1
  end function lambdata_is_active

  !> @brief get the max number of rows allocated for bias
  !
  !> @return the max number of rows allocated for bias
  function lambdata_bias_get_nrows() bind(c) result(out_nrows)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: out_nrows
    out_nrows = dataframe_get_nrows(bias_data)
  end function lambdata_bias_get_nrows

  !> @brief get the last nbiasv
  !
  !> @return the last nbiasv
  function lambdata_get_nbiasv() bind(c) result(out_nbiasv)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: out_nbiasv
    out_nbiasv = lambdata_bias_get_nrows()
  end function lambdata_get_nbiasv

  !> @brief get the max number of rows allocated for bixlamsq
  !
  !> @return the max number of rows allocated for bixlamsq
  function lambdata_bixlamsq_get_nrows() bind(c) result(out_nrows)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: out_nrows
    out_nrows = dataframe_get_nrows(bixlamsq_data)
  end function lambdata_bixlamsq_get_nrows

  !> @brief get the last nblock
  !
  !> @return the last nblock
  function lambdata_get_nblock() bind(c) result(out_nblock)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: out_nblock
    out_nblock = lambdata_bixlamsq_get_nrows()
  end function lambdata_get_nblock

  !> @brief get the bias table max number of columns
  !
  !> @return the bias table max number of columns
  function lambdata_bias_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_ncols
    implicit none
    integer(c_int) :: ncols
    ncols = dataframe_get_ncols(bias_data)
  end function lambdata_bias_get_ncols

  !> @brief get the bixlamsq table max number of columns
  !
  !> @return the bixlamsq table max number of columns
  function lambdata_bixlamsq_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_ncols
    implicit none
    integer(c_int) :: ncols
    ncols = dataframe_get_ncols(bixlamsq_data)
  end function lambdata_bixlamsq_get_ncols

  !> @brief get a copy of the bias table
  !
  !> @param[out] out_names a string array for each column name
  !> @param[out] out_rows a flattened matrix of data points
  subroutine lambdata_bias_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(bias_data, out_names, out_rows)
  end subroutine lambdata_bias_get

  !> @brief get a copy of the bixlamsq table
  !
  !> @param[out] out_names a string array for each column name
  !> @param[out] out_rows a flattened matrix of data points
  subroutine lambdata_bixlamsq_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(bixlamsq_data, out_names, out_rows)
  end subroutine lambdata_bixlamsq_get

  !> @brief fill the dataframes the next dynamics run
  !
  !> @return Was fill already on? 1 <==> yes, 0 otherwise
  integer(c_int) function lambdata_on() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    lambdata_on = 0
    if (fill_lambdata) lambdata_on = 1

    fill_lambdata = .true.
  end function lambdata_on

  !> @brief do not fill dataframes the next dynamics run
  !
  !> @return Was fill on? 1 <==> yes, 0 otherwise
  integer(c_int) function lambdata_off() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    lambdata_off = 1
    if (fill_lambdata) lambdata_off = 0

    fill_lambdata = .false.
  end function lambdata_off

  !> @brief allocate storage and initialize attributes
  !
  ! If fill_lambdata is true, the next dynamics run,
  ! the final state of the lambda dynamics data will be
  ! saved in this module.
  !
  !> @param[in] nbiasv total number of biasing potential
  !> @param[in] nbock total number of blocks
  subroutine lambdata_init(in_nbiasv, in_nblock)
    use api_dataframe, only: dataframe_init
    implicit none
    integer, intent(in) :: in_nbiasv, in_nblock

    call dataframe_init(in_nbiasv, 9, bias_data)
    call dataframe_init(in_nblock, 3, bixlamsq_data)
  end subroutine lambdata_init

  !> @brief free storage and zero out parameters
  subroutine lambdata_del() bind(c)
    use api_dataframe, only: dataframe_del
    implicit none
    call dataframe_del(bias_data)
    call dataframe_del(bixlamsq_data)
  end subroutine lambdata_del

  !> @brief set the column names
  !
  ! The names are fixed. Call this routine in the dynamics routine
  ! only after lambdata_init
  subroutine lambdata_set_names()
    use api_dataframe, only: dataframe_name_size, dataframe_set_names
    implicit none

    character(len=dataframe_name_size), dimension(:), allocatable :: new_names

    new_names = (/ 'STEP', 'TIME', &
         'VIDI', 'VIDJ', 'CLAS', &
         'REUP', 'RLOW', &
         'KBS ', 'PBS ' /)
    call dataframe_set_names(bias_data, new_names)

    new_names = (/ 'STEP', 'TIME', 'XLM2' /)
    call dataframe_set_names(bixlamsq_data, new_names)
  end subroutine lambdata_set_names

  !> @brief add a set of lambda dynamics data
  !
  ! Call this routine in the dynamics routine only after lambdata_init.
  !
  !> @param[in] step the current step number of dynamics
  !> @param[in] time the current simulation time for the dynamics step
  subroutine lambdata_add_rows(step, time)
    use lambdam,only: &
         bixlam, ibvidi, ibvidj, ibclas, &
         irreup, irrlow, ikbias, ipbias

    use chm_kinds, only: chm_real
    use api_dataframe, only: dataframe_add_row

    implicit none

    integer, intent(in) :: step
    real(chm_real), intent(in) :: time

    real(chm_real), dimension(:), allocatable :: new_row
    real(chm_real) :: new_step
    integer :: i

    new_step = real(step, chm_real)

    do i = 1, lambdata_get_nbiasv()
       new_row = (/ new_step, time, &
            real(ibvidi(i), chm_real), &
            real(ibvidj(i), chm_real), &
            real(ibclas(i), chm_real), &
            irreup(i), irrlow(i), ikbias(i), &
            real(ipbias(i), chm_real) /)
       call dataframe_add_row(bias_data, new_row)
    end do

    do i = 1, lambdata_get_nblock()
       new_row = (/ new_step, time, bixlam(i) * bixlam(i) /)
       call dataframe_add_row(bixlamsq_data, new_row)
    end do
  end subroutine lambdata_add_rows
#endif /* KEY_LIBRARY */
end module api_lambdata
