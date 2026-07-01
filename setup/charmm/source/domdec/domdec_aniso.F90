module domdec_aniso

#if KEY_DOMDEC==1
  use chm_kinds
  use domdec_bonded_types,only:anisolist_t
  implicit none
  private

  logical q_aniso

  ! anisotropy list
  integer nanisolist
  type(anisolist_t), allocatable, dimension(:) :: anisolist

  logical :: q_anisolist_current = .false.

  ! Tables
  integer, pointer, dimension(:) :: anisotbl
  ! Number
  integer nanisotbl

  ! Public subroutines
  public init_aniso, uninit_aniso
  public check_aniso_total, check_aniso
  public aniso_associate_tbl, set_aniso_alloc_len, get_aniso_num, set_aniso_ntbl
  public build_aniso_ll, get_aniso_ind
  public make_anisolist_current

  ! Public variables
  public q_aniso
  public nanisolist, anisolist

#endif

contains

#if KEY_DOMDEC==1
  ! *
  ! * Initializes anisotrpy
  ! *
  subroutine init_aniso()
    use aniso_fcm,only:naniso, lstani1, lstani2, lstani3, lstani4
    implicit none
    integer i, j

    if (naniso == 0) then
       q_aniso = .false.
       return
    else
       q_aniso = .true.
    endif

    ! Allocate anisolist
    if (allocated(anisolist)) then
       if (size(anisolist) < naniso) then
          deallocate(anisolist)
       endif
    endif
    if (.not.allocated(anisolist)) then
       allocate(anisolist(naniso))
    endif

    return
  end subroutine init_aniso

  ! *
  ! * Uninitializes lone pairs
  ! *
  subroutine uninit_aniso()
    use memory,only:chmdealloc
    implicit none
    integer i, j

    q_aniso = .false.

    ! anisolist
    if (allocated(anisolist)) then
       deallocate(anisolist)
    endif

    ! anisotbl
    nullify(anisotbl)

    return
  end subroutine uninit_aniso

  ! *
  ! * Checks that there are correct number of lone pairs
  ! *
  subroutine check_aniso()
    implicit none
    ! Variables
    integer naniso_sum
    integer tbl(1)

    if (.not.q_aniso) return

    tbl(1) = nanisotbl
    call igcomb(tbl, 1)
    naniso_sum = tbl(1)

    call check_aniso_total(naniso_sum)

    return
  end subroutine check_aniso

  ! *
  ! * Checks the lone pair total number
  ! *
  subroutine check_aniso_total(naniso_sum)
    use stream,only:outu,prnlev
    use aniso_fcm,only:naniso
    implicit none
    ! Input
    integer, intent(in) :: naniso_sum

    if (.not.q_aniso) return

    ! Check the numbers:
    if (naniso_sum /= naniso) then
       if (prnlev > 2) then
          write (outu,'(a,i8)') 'correct: naniso',naniso
          write (outu,'(a,i8)') 'actual:  naniso',naniso_sum
          call wrndie(-5,'<domdec_aniso>','Incorrect number of anisotropics')
       endif
    endif
    
    return
  end subroutine check_aniso_total

  ! *
  ! * Builds aniso structure for each atom
  ! * Only the atom with the lowest index has the list
  ! * NOTE: only called in initialization
  ! *
  subroutine build_aniso_ll(ntmp, storage_size, ll_ind, ll_head, ll_data)
    use aniso_fcm,only:naniso, lstani1, lstani2, lstani3, lstani4
    use domdec_bonded_types,only:ll_type, add_to_list, TYPE_ANISO
    implicit none
    ! Input / Output
    integer, intent(inout) :: ntmp(:,:), storage_size
    integer, intent(inout) :: ll_ind, ll_head(:,:)
    type(ll_type), intent(inout) :: ll_data(:)
    ! Variables
    integer ii, t

    do ii=1,naniso
       t = min(lstani1(ii), lstani2(ii), lstani3(ii), lstani4(ii))
       call add_to_list(TYPE_ANISO, ii, t, ll_ind, ntmp, storage_size, ll_head, ll_data)
    enddo

    return
  end subroutine build_aniso_ll

  ! *
  ! * Returns anisotropy atom indices
  ! *
  subroutine get_aniso_ind(i, aniso_type, ind, nind)
    use aniso_fcm,only:lstani1, lstani2, lstani3, lstani4
    implicit none
    ! Input / Output
    integer, intent(in) :: i, aniso_type
    integer, intent(out) :: ind(:), nind

    nind = 5
    ind(1:nind) = (/ lstani1(i), lstani1(i)+1, lstani2(i), lstani3(i), lstani4(i) /)

    return
  end subroutine get_aniso_ind

  ! *
  ! * Associate anisotbl
  ! *
  subroutine aniso_associate_tbl(tbl)
    use nblist_types,only:intarray_t
    use domdec_bonded_types,only:TYPE_ANISO
    implicit none
    ! Input
    type(intarray_t), intent(in), target, dimension(:) :: tbl

    nullify(anisotbl)
    anisotbl => tbl(TYPE_ANISO)%array

    return
  end subroutine aniso_associate_tbl

  ! *
  ! * Set allocation lengths for aniso interactions
  ! *
  subroutine set_aniso_alloc_len(alloc_len)
    use aniso_fcm,only:naniso
    use domdec_bonded_types,only:TYPE_ANISO
    implicit none
    ! Input / Output
    integer, intent(out) :: alloc_len(:)

    alloc_len(TYPE_ANISO) = naniso

    return
  end subroutine set_aniso_alloc_len

  ! *
  ! * Returns the total number of aniso interactions
  ! *
  integer function get_aniso_num()
    use aniso_fcm,only:naniso
    implicit none

    get_aniso_num = naniso

    return
  end function get_aniso_num

  ! *
  ! * Sets nbondtbl, nangletbl, etc..
  ! *
  subroutine set_aniso_ntbl(ntbl)
    use domdec_bonded_types,only:TYPE_ANISO
    implicit none
    ! Input / Output
    integer, intent(in) :: ntbl(:)

    nanisotbl = ntbl(TYPE_ANISO)
    q_anisolist_current = .false.
    
    return
  end subroutine set_aniso_ntbl

  ! *
  ! * Build anisolist
  ! * NOTE: right now anisolist = anisotbl. However, that might change in the future, when
  ! *       we change anisolist to contain the atom indices instead.
  ! *
  subroutine build_anisolist(nanisotbl, anisotbl, anisolist)
    implicit none
    ! Input / Output
    integer, intent(in) :: nanisotbl
    integer, intent(in) :: anisotbl(:)
    type(anisolist_t), intent(out) :: anisolist(:)
    ! Variables
    integer i

!$omp parallel do schedule(static) private(i)
    do i=1,nanisotbl
       anisolist(i)%ind = anisotbl(i)
    enddo
!$omp end parallel do

    nanisolist = nanisotbl

    return
  end subroutine build_anisolist

  ! *
  ! * Makes anisolist current
  ! *
  subroutine make_anisolist_current()
    implicit none

    if (.not.q_anisolist_current) then
       call build_anisolist(nanisotbl, anisotbl, anisolist)
       q_anisolist_current = .true.
    endif

    return
  end subroutine make_anisolist_current

#endif

end module domdec_aniso
