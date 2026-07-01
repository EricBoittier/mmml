module mccent
  use chm_kinds
  implicit none

  !
  !    XCENT        Pointer to list of group center x positions
  !    YCENT        Pointer to list of group center y positions
  !    ZCENT        Pointer to list of group center z positions
  !    QCENT        Pointer to list of group center q positions
  !
  !    LCENTR       Flag to suppress center update in ENBOND
  !                      0:  Normal allocation and calculation of centers
  !                      1:  No     allocation and calculation of centers
  !                     -1:  No     allocation but calculation of centers
  !
  integer :: lcentr
  real(chm_real),dimension(:),allocatable :: XCENT, YCENT, ZCENT, QCENT

contains

  subroutine mc_dealloc_centers(filename, procname, req_size)
    use memory
    character(len=*),intent(in) :: filename, procname
    integer,intent(in) :: req_size

    if (allocated(xcent)) then
      call chmdealloc(filename, procname, 'XCENT', req_size, crl=xcent)
      call chmdealloc(filename, procname, 'YCENT', req_size, crl=ycent)
      call chmdealloc(filename, procname, 'ZCENT', req_size, crl=zcent)
      call chmdealloc(filename, procname, 'QCENT', req_size, crl=qcent)
    endif
  end subroutine mc_dealloc_centers

  subroutine mc_alloc_centers(filename, procname, req_size)
    use memory
    character(len=*),intent(in) :: filename, procname
    integer,intent(in) :: req_size

    if (.not. allocated(xcent)) then
      call chmalloc(filename, procname, 'XCENT', req_size, crl=xcent)
      call chmalloc(filename, procname, 'YCENT', req_size, crl=ycent)
      call chmalloc(filename, procname, 'ZCENT', req_size, crl=zcent)
      call chmalloc(filename, procname, 'QCENT', req_size, crl=qcent)
    else if (size(xcent) < req_size) then
      call chmrealloc(filename, procname, 'XCENT', req_size, crl=xcent)
      call chmrealloc(filename, procname, 'YCENT', req_size, crl=ycent)
      call chmrealloc(filename, procname, 'ZCENT', req_size, crl=zcent)
      call chmrealloc(filename, procname, 'QCENT', req_size, crl=qcent)
    endif
  end subroutine mc_alloc_centers

end module mccent

