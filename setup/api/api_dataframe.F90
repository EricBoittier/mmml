!> routines to store named columns of data one row at a time
module api_dataframe
  use chm_kinds, only: chm_real

  implicit none

  ! TODO: make name sizes dynamic
  integer, parameter :: dataframe_name_size = 16

  type t_dataframe
     logical :: is_active = .false.

     integer :: &
          nrows = 0, &
          ncols = 0, &
          current_row = 0

     character(len=dataframe_name_size), dimension(:), allocatable :: &
          names

     real(chm_real), dimension(:), allocatable  :: &
          values
  end type t_dataframe

#if KEY_LIBRARY == 1

contains

  !> @brief allocate storage for a table and initialize table attributes
  !
  ! Space for nrows * ncols real(chm_real) elements will be allocated.
  ! Space for ncols strings each of length dataframe_name_size will also
  ! be allocated. Then the attributes will be initialized.
  !
  !> @param[in] nrows max number of data rows
  !> @param[in] ncols length of each data row, also the number of names
  !> @param[out] new_table the table that will be initialized
  subroutine dataframe_init(nrows, ncols, new_table)
    use memory, only: chmalloc, chmrealloc
    implicit none
    integer, intent(in) :: nrows, ncols
    type(t_dataframe) :: new_table

    if (.not. allocated(new_table%names)) then
       call chmalloc(__FILE__, 'dataframe_init', 'out_names', &
            ncols, ch16=new_table%names)
    else
       call chmrealloc(__FILE__, 'dataframe_init', 'out_names', &
            ncols, ch16=new_table%names)
    end if  ! allocate names

    if (.not. allocated(new_table%values)) then
       call chmalloc(__FILE__, 'dataframe_init', 'out_values', &
            nrows * ncols, crl=new_table%values)
    else
       call chmrealloc(__FILE__, 'dataframe_init', 'out_values', &
            nrows * ncols, crl=new_table%values)
    end if

    new_table%nrows = nrows
    new_table%ncols = ncols
    new_table%current_row = 0
    new_table%is_active = .true.
  end subroutine dataframe_init

  !> @brief free up storage from a table
  !
  ! the table must be re-initialized to be used again
  ! after calling this routine
  !
  !> @param[out] table free up the storage from this particular table
  subroutine dataframe_del(table)
    use memory, only: chmdealloc
    implicit none

    type(t_dataframe) :: table

    if (table%is_active) then
       call chmdealloc(__FILE__, 'dataframe_init', 'table%names', &
            table%ncols, ch16=table%names)
       call chmdealloc(__FILE__, 'dataframe_init', 'table%values', &
            table%nrows * table%ncols, crl=table%values)
       table%current_row = 0
       table%nrows = 0
       table%ncols = 0
       table%is_active = .false.
    end if
  end subroutine dataframe_del

  !> @brief set the column names for a table
  !
  ! each name must be of length dataframe_name_size
  !
  !> @param[out] table this tables columns names will be set to
  !>             a copy of in_names
  !> @param[in] in_names copy these column names into the table
  subroutine dataframe_set_names(table, in_names)
    implicit none
    type(t_dataframe) :: table
    character(len=dataframe_name_size), dimension(:), intent(in) :: in_names

    integer :: i

    do i = 1, table%ncols
       table%names(i)(1:dataframe_name_size) = &
            in_names(i)(1:dataframe_name_size)
    end do
  end subroutine dataframe_set_names

  !> @brief append a new row to the table
  !
  ! new_row must have length at least table%ncols
  !
  !> @param[out] table append a copy of new_row to this
  !> @param[in] new_row a copy of this array will be appended
  !>            to the rows of table
  subroutine dataframe_add_row(table, new_row)
    use chm_kinds, only: chm_real
    implicit none
    type(t_dataframe) :: table
    real(chm_real), dimension(:), intent(in) :: new_row

    integer :: start, end

    start = (table%current_row * table%ncols) + 1
    end = start + table%ncols
    table%values(start:end) = new_row(1:table%ncols)
    table%current_row = table%current_row + 1
  end subroutine dataframe_add_row

  !> @brief get max number of rows allocated for table
  !
  !> @param[in] table the max number of rows for this table
  !> @return integer(c_int) max number of rows allocated for table
  integer(c_int) function dataframe_get_nrows(table)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    type(t_dataframe), intent(in) :: table
    dataframe_get_nrows = table%nrows
  end function dataframe_get_nrows

  !> @brief get the number of columns of each row of table
  !
  ! this is also the number of column names
  !
  !> @param[in] table the number of columns for this table
  !> @return integer(c_int) the number of columns of table
  integer(c_int) function dataframe_get_ncols(table)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    type(t_dataframe), intent(in) :: table
    dataframe_get_ncols = table%ncols
  end function dataframe_get_ncols

  !> @brief get the size each column name should be
  !
  ! in characters
  !
  !> @return integer(c_int) the size of each column name in characters
  integer(c_int) function dataframe_get_name_size() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    dataframe_get_name_size = dataframe_name_size
  end function dataframe_get_name_size

  !> @brief get a copy of the column names for this table
  !
  ! out_names should be pre-allocated to hold table%ncols strings
  ! each string should be dataframe_name_size characters long
  !
  !> @param[in] table copy the column names from this table
  !> @param[out] out_names a 2D C char array of size
  !>             ncols by dataframe_name_size
  subroutine dataframe_get_names(table, out_names)
    use, intrinsic :: iso_c_binding, only: c_ptr
    use api_util, only: f2c_string
    implicit none

    ! args
    type(t_dataframe), intent(in) :: table
    type(c_ptr), dimension(*) :: out_names

    ! locals
    integer :: i

    do i = 1, table%ncols
       call f2c_string(table%names(i), out_names(i), dataframe_name_size)
    end do
  end subroutine dataframe_get_names

  !> @brief get a copy of the i-th row of table
  !
  ! out_row should have size table%ncols of real(c_double) elts
  !
  !> @param[in] table copy the index-th row of this table
  !> @param[in] index the index of the desired row starting from 1
  !> @param[out] out_row a c_double array of length table%ncols
  subroutine dataframe_get_row(table, index, out_row)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none

    ! args
    type(t_dataframe), intent(in) :: table
    integer(c_int) :: index
    real(c_double), dimension(*) :: out_row

    ! locals
    integer :: start, end

    if ((index .gt. 0) .and. (index .le. table%current_row)) then
       start = (table%ncols * (index - 1)) + 1
       end = start + table%ncols
       out_row(1:table%ncols) = table%values(start:end)
    else
       call wrndie(-5, '<api_dataframe%dataframe_get_row>', &
            'index out of bounds')
    end if
  end subroutine dataframe_get_row

  !> @brief get a copy of all the values in the table
  !
  ! out_values should have size table%ncols * table%current_row
  ! of real(c_double) elts
  !
  !> @param[in] table copy the values of this table
  !> @param[out] out_values a c_double array of length
  !>             table%ncols * table%current_row
  subroutine dataframe_get_values(table, out_values)
    use, intrinsic :: iso_c_binding, only: c_double
    implicit none

    ! args
    type(t_dataframe), intent(in) :: table
    real(c_double), dimension(*) :: out_values

    integer :: end

    if (table%current_row .gt. 0) then
       end = table%ncols * table%current_row
       out_values(1:end) = table%values(1:end)
    else
       call wrndie(-5, '<api_dataframe%dataframe_get_values>', &
            'no values in table')
    end if
  end subroutine dataframe_get_values

  !> @brief get a copy of the column names and data rows of this table
  !
  ! out_names should have size table%ncols of strings each of length
  ! dataframe_name_size c_char's
  !
  ! out_rows should have size table%nrows by table%ncols of c_double elts
  !
  !> @param[in] table copy this table
  !> @param[in] out_names a copy of the table's column names
  !> @param[out] out_rows a row-major flat c_double array of the table's data
  subroutine dataframe_get(table, out_names, out_rows)
    use, intrinsic :: iso_c_binding, only: c_double, c_ptr
    implicit none

    ! args
    type(t_dataframe) :: table
    type(c_ptr), dimension(*) :: out_names
    real(c_double), dimension(*) :: out_rows

    ! locals
    integer :: dataframe_size

    dataframe_size = table%nrows * table%ncols

    call dataframe_get_names(table, out_names)
    out_rows(1:dataframe_size) = table%values(1:dataframe_size)
  end subroutine dataframe_get
#endif /* KEY_LIBRARY */
end module api_dataframe
