module domdec_bonded_types

#if KEY_DOMDEC==1
  use iso_c_binding
  implicit none
  public

  ! 1-7  : bonds, angles, dihedrals, etc.
  ! 8    : Thole 1-4 exclusions
  ! 9    : Drude hyper polarizability
  ! 10   : anisotropy
  integer, parameter :: ngrouptype = 10
  integer, parameter :: TYPE_BOND=1, TYPE_ANGLE=2, TYPE_DIHE=3, TYPE_IMDIHE=4
  integer, parameter :: TYPE_IN14=5, TYPE_EX14=6, TYPE_CMAP=7
  integer, parameter :: TYPE_EX14THOLE=8, TYPE_HYPER=9
  integer, parameter :: TYPE_ANISO=10
  integer, parameter :: grouptype_storage_size(ngrouptype) = (/ &
       2, 3, 4, 4, 2, 2, 8, &                ! bonds, angles, dihedrals, etc.
       2, 1, &                               ! Thole 1-4 exclusions, Drude hyper polarizability
       5/)                                   ! Anisotropy

  ! linked list type definition
  type ll_type
     sequence
     integer ind
     integer next
  end type ll_type

  ! Data structures for bonds, angles, dihedrals, and cmap, including size in bytes
  type, bind(C) :: bondlist_t
     integer(c_int) i, j, itype, ishift
  end type bondlist_t
  integer, parameter :: sizeof_bondlist_t = 4*4

  type, bind(C) :: anglelist_t
     integer(c_int) i, j, k, itype, ishift1, ishift2
  end type anglelist_t
  integer, parameter :: sizeof_anglelist_t = 6*4

  type, bind(C) :: dihelist_t
     integer(c_int) i, j, k, l, itype, ishift1, ishift2, ishift3
  end type dihelist_t
  integer, parameter :: sizeof_dihelist_t = 8*4

  type, bind(C) :: list14_t
     integer(c_int) i, j, ishift
  end type list14_t
  integer, parameter :: sizeof_list14_t = 3*4

#if KEY_CMAP==1
  type, bind(C) :: cmaplist_t
     integer(c_int) i1, j1, k1, l1, i2, j2, k2, l2, itype, ishift1, ishift2, ishift3
  end type cmaplist_t
#endif

  type, bind(C) :: list14thole_t
     integer(c_int) i, j, ith, jth, ishift
     real(c_double) aa
  end type list14thole_t

  type, bind(C) :: hyperlist_t
     integer(c_int) i, j, ishift
  end type hyperlist_t

  type anisolist_t
     sequence
     integer ind
  end type anisolist_t

  public :: add_to_list

#endif

contains

#if KEY_DOMDEC==1
  ! *
  ! * type_ind = type index
  ! * i        = bond index
  ! * t        = bond assigned atom index
  ! *
  subroutine add_to_list(type_ind, i, t, ll_ind, n_tmp, size_counter, ll_head, ll_data)
    implicit none
    ! Input / Output
    integer, intent(in) :: type_ind, i, t
    integer, intent(inout) :: ll_ind
    integer, intent(inout) :: ll_head(:,:), n_tmp(:,:), size_counter
    type(ll_type), intent(inout) :: ll_data(:)

    n_tmp(t,type_ind) = n_tmp(t,type_ind) + 1
    ll_ind = ll_ind + 1
    ll_data(ll_ind)%ind = i
    ll_data(ll_ind)%next = ll_head(t,type_ind)
    ll_head(t,type_ind) = ll_ind

    if ((type_ind == TYPE_BOND .or. type_ind == TYPE_ANGLE .or. type_ind == TYPE_EX14) .and. &
         q_solvent(t)) then
       ! Solvent: no atom indices needed
       size_counter = size_counter + 2
    else
       size_counter = size_counter + grouptype_storage_size(type_ind) + 2
    endif

    return
  end subroutine add_to_list

  ! Returns true if atom i is a solvent
  logical function q_solvent(i)
    use groupxfast,only:invgroup, group, group_out
    implicit none
    ! Input
    integer, intent(in) :: i
    ! Variables
    integer ig, itype
    
    ig = invgroup(i)
    call group_out(group(ig), itype)

    q_solvent = (itype == 1)

    return
  end function q_solvent
#endif

end module domdec_bonded_types
