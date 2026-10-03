module selctam
  use chm_kinds
  use chm_types
  use dimens_fcm
  !CHARMM Element source/fcm/selcta.fcm 1.1
  !
  !   Selection definitions common file.
  !
  !     MAXSKY - Initial capacity of the definition table (it now grows
  !              on demand; see select_ensure_cap)
  !     MNAMSK - Maximum number of characters per definition
  !     CAPSKY - Currently allocated capacity of the table
  !
  !     NUMSKY - Number of active definitions
  !     NAMSKY - Name of definitions
  !     LNAMSK - Length of each definition name
  !     PTRSKY - Heap pointer for selection array
  !     LENSKY - Number of atoms in selection array (for checking)
  !
  !
  INTEGER, PARAMETER :: MAXSKY=200,MNAMSK=20
  INTEGER :: NUMSKY
  INTEGER :: CAPSKY = 0
  INTEGER,allocatable,dimension(:) ::  LENSKY,LNAMSK
  CHARACTER(len=MNAMSK),allocatable,dimension(:) :: NAMSKY
  type(chm_iptr),allocatable,dimension(:),save :: PTRSKY

contains

  subroutine select_init()
    numsky=0
    call select_ensure_cap(maxsky)
    return
  end subroutine select_init

  subroutine select_ensure_cap(nmin)
    !  Ensure the stored-selection table can hold at least NMIN entries,
    !  growing on demand (doubling) rather than dying at a fixed cap.
    !  Existing entries 1..NUMSKY -- names, name-lengths, atom-counts, and
    !  the PTRSKY pointer components -- are preserved.
    !
    !  The intrinsic-typed arrays go through the CHARMM memory wrappers
    !  (chmalloc for the first allocation, chmrealloc to grow).  PTRSKY is
    !  an array of the derived type chm_iptr, for which no chmrealloc
    !  overload exists, so it is grown by hand via move_alloc with explicit
    !  error checking.  Intrinsic assignment of chm_iptr pointer-associates
    !  the %a component (no deep copy, no double free), so live selection
    !  arrays survive the move untouched.
    !
    !  NOTE: NAMSKY is character(len=MNAMSK); the chm(re)alloc ch20 overload
    !  below is only valid while MNAMSK == 20 (asserted at run time).
    use memory, only: chmalloc, chmrealloc
    integer, intent(in) :: nmin
    integer :: newcap, ierr
    type(chm_iptr), allocatable :: tmpsky(:)

    if (nmin <= capsky) return

    ! The NAMSKY (re)allocation uses the ch20 wrapper overload; keep the
    ! character length and that overload in sync.  Fail loudly and locally
    ! if MNAMSK is ever changed without updating the calls below.
    if (mnamsk /= 20) &
         call wrndie(-5, '<select_ensure_cap>', &
         'MNAMSK changed from 20: update the ch20 chm(re)alloc calls for NAMSKY.')

    if (capsky <= 0) then
       newcap = maxsky
    else
       newcap = capsky
    end if
    do while (newcap < nmin)
       newcap = newcap * 2
    end do

    ! Grow atomically w.r.t. the by-hand PTRSKY allocation: build the new
    ! PTRSKY into a temporary and validate it BEFORE touching any of the
    ! persistent (wrapper-managed) arrays, so a PTRSKY allocation failure
    ! leaves the whole table unchanged rather than half-grown.
    allocate(tmpsky(newcap), stat=ierr)
    if (ierr /= 0) then
       call wrndie(-3, '<select_ensure_cap>', &
            'allocation of stored-selection table (PTRSKY) failed')
       return
    end if
    if (numsky > 0) tmpsky(1:numsky) = PTRSKY(1:numsky)

    if (capsky <= 0) then
       ! first-time allocation of the intrinsic-typed arrays
       call chmalloc('selcta_ltm.F90', 'select_ensure_cap', 'LENSKY', &
            newcap, intg=LENSKY)
       call chmalloc('selcta_ltm.F90', 'select_ensure_cap', 'LNAMSK', &
            newcap, intg=LNAMSK)
       call chmalloc('selcta_ltm.F90', 'select_ensure_cap', 'NAMSKY', &
            newcap, ch20=NAMSKY)
    else
       ! grow, preserving the existing entries
       call chmrealloc('selcta_ltm.F90', 'select_ensure_cap', 'LENSKY', &
            newcap, intg=LENSKY)
       call chmrealloc('selcta_ltm.F90', 'select_ensure_cap', 'LNAMSK', &
            newcap, intg=LNAMSK)
       call chmrealloc('selcta_ltm.F90', 'select_ensure_cap', 'NAMSKY', &
            newcap, ch20=NAMSKY)
    end if

    call move_alloc(tmpsky, PTRSKY)
    capsky = newcap
  end subroutine select_ensure_cap

end module selctam

