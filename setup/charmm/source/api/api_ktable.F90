!> routines to store dynamics energy data
module api_ktable
  use api_dataframe, only: t_dataframe
  implicit none

#if KEY_LIBRARY == 1
  logical :: fill_ktable = .false.
  type(t_dataframe) :: ktable

contains

  !> @brief Has the ktable been initialized and allocated?
  !
  !> @return 1 <==> ktable is initialized and allocated, 0 otherwise
  function ktable_is_active() bind(c) result(is_active)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: is_active

    is_active = 0
    if (ktable%is_active) is_active = 1
  end function ktable_is_active

  !> @brief deallocate ktable storage
  subroutine ktable_del() bind(c)
    use api_dataframe, only: dataframe_del
    implicit none
    call dataframe_del(ktable)
  end subroutine ktable_del

  !> @brief get the max number of rows allocated for the ktable
  !
  !> @return the max number of rows allocated for the ktable
  function ktable_get_nrows() bind(c) result(nrows)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_nrows
    implicit none
    integer(c_int) :: nrows
    nrows = dataframe_get_nrows(ktable)
  end function ktable_get_nrows

  !> @brief get the max number of columns allocated for the ktable
  !
  !> @return the max number of columns allocated for the ktable
  function ktable_get_ncols() bind(c) result(ncols)
    use, intrinsic :: iso_c_binding, only: c_int
    use api_dataframe, only: dataframe_get_ncols
    implicit none
    integer(c_int) :: ncols
    ncols = dataframe_get_ncols(ktable)
  end function ktable_get_ncols

  !> @brief get a copy of the ktable
  !
  !> @param[out] out_names a string array for the name of each column
  !> @param[out] out_rows a flattened matrix of data points
  subroutine ktable_get(out_names, out_rows) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    use api_dataframe, only: dataframe_get
    implicit none

    ! args
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    call dataframe_get(ktable, out_names, out_rows)
  end subroutine ktable_get

  !> @brief fill the ktable next time dynamics is run
  !
  !> @return Was 'fill ktable' already on? 1<==>yes, 0 otherwise
  integer(c_int) function ktable_on() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    ktable_on = 0
    if (fill_ktable) ktable_on = 1

    fill_ktable = .true.
  end function ktable_on

  !> @brief do not fill the ktable next time dynamics is run
  !
  !> @return Was 'fill ktable' on? 1<==>yes, 0 otherwise
  integer(c_int) function ktable_off() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_double
    implicit none
    ktable_off = 1
    if (fill_ktable) ktable_off = 0

    fill_ktable = .false.
  end function ktable_off

  !> @brief allocate storage for ktable and initialize ktable attributes
  !
  ! If fill_ktable is true, on the next dynamics run, dynamics will run
  ! from istart step to istop step, adding a row to ktable every nprint-nth
  ! step. So ktable needs ((istop - istart) / nprint) + 1 rows. The number
  ! of columns is mostly fixed. Additional columns about pressure will be
  ! collected if constant pressure is on.
  !
  !> @param[in] istart the upcoming dynamics run starts on this step number
  !> @param[in] istop the upcoming dynamics run stops on this step number
  !> @param[in] nprint traditional charmm output printed every nprint steps
  subroutine ktable_init()
    use api_dataframe, only: dataframe_init
    use reawri, only: qcnstp
    implicit none

    integer :: nrows, ncols

    nrows = 1
    ncols = 17

    if (qcnstp) ncols = ncols + 19

    call dataframe_init(nrows, ncols, ktable)
  end subroutine ktable_init

  !> @brief set the names of the columns for ktable
  !
  ! The names are fixed. This routine should be called in the dynamics routine.
  subroutine ktable_set_names()
    use api_dataframe, only: dataframe_name_size, dataframe_set_names
    use energym, only: &
         ceprop, &
         tote, totke, epot, temps, &
         ceterm, &
         bond, angle, dihe, imdihe, vdw, elec, hbond, charm, dmc, rgy

    use reawri, only: qcnstp
    implicit none

    character(len=dataframe_name_size), dimension(:), allocatable :: &
         new_names

    new_names = (/ 'STEP', 'TIME', &
         ceprop(tote), ceprop(totke), ceprop(epot), 'EP-K', &
         ceprop(temps), ceterm(bond), ceterm(angle), &
         ceterm(dihe), ceterm(imdihe), &
         ceterm(vdw), ceterm(elec), ceterm(hbond), &
         ceterm(charm), ceterm(dmc),ceterm(rgy) /)

    if (qcnstp) then
       new_names = [new_names, (/ &
            'REFP', &
            'PEXX', 'PEXY', 'PEXZ', &
            'PEYX', 'PEYY', 'PEYZ', &
            'PEZX', 'PEZY', 'PEZZ', &
            'PIXX', 'PIXY', 'PIXZ', &
            'PIYX', 'PIYY', 'PIYZ', &
            'PIZX', 'PIZY', 'PIZZ' /)]

    end if

    call dataframe_set_names(ktable, new_names)
  end subroutine ktable_set_names

  !> @brief add a row of data to ktable
  !
  ! This routine should be called in the dynamics routine.
  !
  !> @param[in] step the current step number of dynamics
  !> @param[in] time the current simulation time for the dynamics step
  subroutine ktable_add_row(step, time)
    use chm_kinds, only: chm_real
    use api_dataframe, only: dataframe_add_row
    use energym, only: &
         eprop, &
         tote, totke, epot, temps, &
         eterm, &
         bond, angle, dihe, &
#if KEY_CMAP == 1
         cmap, &
#endif
         imdihe, vdw, elec, hbond, charm, dmc, rgy, &
         epress, &
         pexx, pexy, pexz, &
         peyx, peyy, peyz, &
         pezx, pezy, pezz, &
         pixx, pixy, pixz, &
         piyx, piyy, piyz, &
         pizx, pizy, pizz

    use reawri, only: qcnstp, refp
#if KEY_MULTICOM==1 /*  VO string */
    use multicom_aux, only: me_local
#endif

    implicit none

    integer, intent(in) :: step
    real(chm_real), intent(in) :: time

    real(chm_real), dimension(:), allocatable :: new_row

#if KEY_MULTICOM == 1
    if (me_local .ne. 0) return
#endif

    new_row = (/ real(step, chm_real), time, &
         eprop(tote), eprop(totke), &
         eprop(epot), eprop(epot) - eprop(totke), &
         eprop(temps), eterm(bond), eterm(angle), &
#if KEY_CMAP == 1
         eterm(dihe) + eterm(cmap), &
#else
         eterm(dihe), &
#endif
         eterm(imdihe), eterm(vdw), &
         eterm(elec), eterm(hbond), eterm(charm), &
         eterm(dmc), eterm(rgy) /)

    if (qcnstp) then
       new_row = [new_row, (/ &
            refp, &
            epress(pexx), epress(pexy), epress(pexz), &
            epress(peyx), epress(peyy), epress(peyz), &
            epress(pezx), epress(pezy), epress(pezz), &
            epress(pixx), epress(pixy), epress(pixz), &
            epress(piyx), epress(piyy), epress(piyz), &
            epress(pizx), epress(pizy), epress(pizz) /)]
    end if

    call dataframe_add_row(ktable, new_row)
  end subroutine ktable_add_row
#endif /* KEY_LIBRARY */
end module api_ktable
