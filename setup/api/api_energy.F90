!> eval and manip of the potential energy of a macromolecular system
module api_energy
  use chm_kinds, only: chm_real
  implicit none

#if KEY_LIBRARY == 1
  integer, parameter :: &
       eprop_name_size = 4, &
       eterm_name_size = 4

contains

  !> @brief print the energy table
  !
  !> @return integer error code: 1 == success
  integer(c_int) function print_energy() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int

    use eutil, only: gete0

    implicit none

    print_energy = -1
    call gete0('ENER', '', 0)
    print_energy = 1
  end function print_energy

  !> @brief get previous total potential energy
  !
  !> @return real previous total potential energy
  real(c_double) function get_old_energy() bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use energym, only: eold
    implicit none
    get_old_energy = eold
  end function get_old_energy

  !> @brief get array of eprop statuses (is eprop(i) on/off?)
  !
  ! copies the energym%qeprop array into C integer array
  !
  !> @param[in] out_qeprops c int array to hold eprop(i) on/off
  !> @return integer status: 1 success, 0 fail
  integer(c_int) function get_eprop_statuses(out_qeprops) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use api_util, only: f2c_logical
    use energym, only: qeprop, lenenp

    implicit none

    integer(c_int), dimension(*) :: out_qeprops
    integer :: i

    get_eprop_statuses = 0
    do i = 1, lenenp
       out_qeprops(i) = f2c_logical(qeprop(i))
    end do
    get_eprop_statuses = 1
  end function get_eprop_statuses

  !> @brief get the energy properties array
  !
  ! copies the energym%eprop array into the argument array
  !
  !> @param[in] out_eprops c array of doubles to hold all the eprops
  !> @return integer status: 1 success, 0 fail
  integer(c_int) function get_energy_properties(out_eprops) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int

    use energym, only: eprop, lenenp

    implicit none

    real(c_double), dimension(*) :: out_eprops

    get_energy_properties = 0
    out_eprops(1:lenenp) = eprop(1:lenenp)
    get_energy_properties = 1
  end function get_energy_properties

  !> @brief get an energy property from the eprop array by its index
  !
  ! calls wrndie if index is out of bounds and returns 0.0
  !
  !> @param[in] property_index index of property in the eprop array
  !> @return real property value at index of eprop array or 0.0
  real(c_double) function get_energy_property(property_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int

    use energym, only: eprop, lenenp

    implicit none

    integer(c_int) :: property_index

    get_energy_property = 0.0
    if (property_index < 1 .or. property_index > lenenp) then
       call wrndie(-5, '<api_energy%get_energy_property>', &
            'property_index out of bounds')
    else
       get_energy_property = eprop(property_index)
    end if
  end function get_energy_property

  !> @brief get array of eterm statuses (is eterm(i) on/off?)
  !
  ! copies the energym%qeterm array into C integer array
  !
  !> @param[in] out_qeterms c int array to hold eterm(i) on/off
  !> @return integer status: 1 success, 0 fail
  integer(c_int) function get_eterm_statuses(out_qeterms) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use api_util, only: f2c_logical
    use energym, only: qeterm, lenenp

    implicit none

    integer(c_int), dimension(*) :: out_qeterms
    integer :: i

    get_eterm_statuses = 0
    do i = 1, lenenp
       out_qeterms(i) = f2c_logical(qeterm(i))
    end do
    get_eterm_statuses = 1
  end function get_eterm_statuses

  !> @brief get the energy terms array
  !
  ! copies the energym%eterm array into the argument array
  !
  !> @param[in] out_eterms c array of doubles to hold all the eterms
  !> @return integer status: 1 success, 0 fail
  integer(c_int) function get_energy_terms(out_eterms) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int

    use energym, only: eterm, lenent

    implicit none

    real(c_double), dimension(*) :: out_eterms

    get_energy_terms = 0
    out_eterms(1:lenent) = eterm(1:lenent)
    get_energy_terms = 1
  end function get_energy_terms

  !> @brief get an energy term from the eterm array by its index
  !
  ! calls wrndie if index is out of bounds and returns 0.0
  !
  !> @param[in] term_index index of term in the eterm array
  !> @return real term value at index of eterm array or 0.0
  real(c_double) function get_energy_term(term_index) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int

    use energym, only: eterm, lenent

    implicit none

    integer(c_int) :: term_index

    get_energy_term = 0.0
    if (term_index < 1 .or. term_index > lenent) then
       call wrndie(-5, '<api_energy%get_energy_term>', &
            'term_index out of bounds')
    else
       get_energy_term = eterm(term_index)
    end if
  end function get_energy_term

  !> @brief run main energy routine, output parallel arrays of names/values
  !
  ! currently limited to 28 name/value pairs
  !
  !> @param[out] out_names pointer to  arrays of C chars naming each ener value
  !> @param[out] out_vals real array of ener values, limited to 28
  !> @return integer number of name/value pairs actually returned <= 28
  integer(c_int) function get_energy(out_names, out_vals) bind(c)
    use, intrinsic :: iso_c_binding, only: &
         c_int, c_ptr, c_double
    use api_util, only: f2c_string
    use eutil, only: gete0

    implicit none

    ! output arguments
    type(c_ptr), target, dimension(*) :: out_names
    real(c_double), dimension(*) :: out_vals

    ! locals
    integer :: i, nchars
    integer(c_int) :: nvals = 0
    character(len=64), dimension(28) :: names

    get_energy = 0

    call gete0('ENER', '', 0)

    do i = 1, nvals
       nchars = len_trim(names(i))
       call f2c_string(names(i), out_names(i), nchars)
    end do

    get_energy = nvals ! success
  end function get_energy

  !> @brief get the number of valid energy properties
  !
  !> @return integer the number of valid energy props
  integer(c_int) function get_num_eprops() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use energym, only: lenenp
    implicit none
    get_num_eprops = lenenp
  end function get_num_eprops

  !> @brief get the number of valid energy terms
  !
  !> @return integer the number of valid energy terms
  integer(c_int) function get_num_eterms() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    use energym, only: lenent
    implicit none
    get_num_eterms = lenent
  end function get_num_eterms

  !> @brief get the number of characters in each energy property name
  !
  !> @return integer the number of characters in each energy property name
  integer(c_int) function get_eprop_name_size() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    get_eprop_name_size = eprop_name_size
  end function get_eprop_name_size

  !> @brief get the name for an energy property from the ceprop array by index
  !
  ! calls wrndie if index is out of bounds and returns 0
  !
  !> @param[in] property_index index of property in the eprop array
  !> @return real status flag 1 if successful, 0 if not
  integer(c_int) function get_eprop_name(property_index, out_name) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use energym, only: ceprop, lenenp
    implicit none
    integer(c_int) :: property_index
    type(c_ptr) :: out_name

    get_eprop_name = 0
    if (property_index < 1 .or. property_index > lenenp) then
       call wrndie(-5, '<api_energy%get_eprop_name>', &
            'property_index out of bounds')
    else
       call f2c_string(ceprop(property_index), out_name, eprop_name_size)
    end if
    get_eprop_name = 1
  end function get_eprop_name

  !> @brief get the names for all energy properties from the ceprop array
  !
  !> @param[out] out_names array of c strings to hold copy of ceprop
  !> @return integer status flag 1 if successful, 0 if not
  integer(c_int) function get_eprop_names(out_names) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use energym, only: ceprop, lenenp
    implicit none
    type(c_ptr), dimension(*) :: out_names
    integer :: i

    get_eprop_names = 0
    do i = 1, lenenp
       call f2c_string(ceprop(i), out_names(i), eprop_name_size)
    end do
    get_eprop_names = 1
  end function get_eprop_names

  !> @brief get an energy property from the eprop array by name
  !
  ! calls wrndie if name is not in ceprop
  !
  !> @param[in] in_name name of property in the ceprop array
  !> @return real property value at index of eprop array or 0.0
  real(c_double) function get_eprop_by_name(in_name)
    use, intrinsic :: iso_c_binding, only: c_double, c_char
    use api_util, only: c2f_string
    use energym, only: ceprop, eprop, lenenp, qeprop
    implicit none

    ! args
    character(kind=c_char, len=1), dimension(*) :: in_name

    ! local vars
    character(len=eprop_name_size) :: eprop_name
    integer :: i
    logical :: found

    get_eprop_by_name = 0.0
    found = .false.
    eprop_name = c2f_string(in_name, eprop_name_size)
    do i = 1, lenenp
       if (qeprop(i) .and. (ceprop(i) .eq. eprop_name)) then
          get_eprop_by_name = eprop(i)
          found = .true.
          exit
       end if
    end do

    if (.not. found) then
       call wrndie(-5, '<api_energy%get_eprop_by_name>', &
            'property name ' // eprop_name // ' not found')
    end if
  end function get_eprop_by_name

  !> @brief get the number of characters in each energy term name
  !
  !> @return integer the number of characters in each energy term name
  integer(c_int) function get_eterm_name_size() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    get_eterm_name_size = eterm_name_size
  end function get_eterm_name_size

  !> @brief get the name for an energy term from the ceterm array by index
  !
  ! calls wrndie if index is out of bounds and returns 0
  !
  !> @param[in] term_index index of term in the eterm array
  !> @return real status flag 1 if successful, 0 if not
  integer(c_int) function get_eterm_name(term_index, out_name) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use energym, only: ceterm, lenent
    implicit none
    integer(c_int) :: term_index
    type(c_ptr) :: out_name

    get_eterm_name = 0
    if (term_index < 1 .or. term_index > lenent) then
       call wrndie(-5, '<api_energy%get_eterm_name>', &
            'term_index out of bounds')
    else
       call f2c_string(ceterm(term_index), out_name, eterm_name_size)
    end if
    get_eterm_name = 1
  end function get_eterm_name

  !> @brief get the names for all energy terms from the ceterm array
  !
  !> @param[out] out_names array of c strings to hold copy of ceterm
  !> @return integer status flag 1 if successful, 0 if not
  integer(c_int) function get_eterm_names(out_names) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    use api_util, only: f2c_string
    use energym, only: ceterm, lenent
    implicit none
    type(c_ptr), dimension(*) :: out_names
    integer :: i

    get_eterm_names = 0
    do i = 1, lenent
       call f2c_string(ceterm(i), out_names(i), eterm_name_size)
    end do
    get_eterm_names = 1
  end function get_eterm_names


  !> @brief get an energy term from the eterm array by name
  !
  ! calls wrndie if name is not in ceterm
  !
  !> @param[in] in_name name of term in the ceterm array
  !> @return real term value at index of eterm array or 0.0
  real(c_double) function get_eterm_by_name(in_name)
    use, intrinsic :: iso_c_binding, only: c_double, c_char
    use api_util, only: c2f_string
    use energym, only: ceterm, eterm, lenenp, qeterm
    implicit none

    ! args
    character(kind=c_char, len=1), dimension(*) :: in_name

    ! local vars
    character(len=eterm_name_size) :: eterm_name
    integer :: i
    logical :: found

    get_eterm_by_name = 0.0
    found = .false.
    eterm_name = c2f_string(in_name, eterm_name_size)
    do i = 1, lenenp
       if (qeterm(i) .and. (ceterm(i) .eq. eterm_name)) then
          get_eterm_by_name = eterm(i)
          found = .true.
          exit
       end if
    end do

    if (.not. found) then
       call wrndie(-5, '<api_energy%get_eterm_by_name>', &
            'term name ' // eterm_name // ' not found')
    end if
  end function get_eterm_by_name
#endif /* KEY_LIBRARY */
end module api_energy
