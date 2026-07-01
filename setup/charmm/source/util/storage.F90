module storage
  use chm_types
  use memory
  implicit none

  character(len=*), parameter, private :: fname = 'storage.src'
  integer, parameter :: MAXSTO=9
  ! PTRSTO - Pointers for NATOM length storage arrays
  type(chm_array), allocatable, dimension(:) :: ptrsto

contains
  ! TODO subroutines to store and retrieve

  subroutine storage_allocate()
    call chmalloc_chm_array(fname,'storage_allocate','ptrsto', &
         MAXSTO,ptrsto)
    ptrsto(1:MAXSTO)%len=0
    return
  end subroutine storage_allocate

  subroutine resize_array(fil,routine,var,arr,n)
    type(chm_array) :: arr
    character(len=*) :: fil,routine,var
    integer,intent(in) :: n
    integer :: n0
    if (allocated(arr%a)) then
       n0 = size(arr%a)
       call chmdealloc(fil,routine,var,n0,crl=arr%a)
    endif
    call chmalloc(fil,routine,var,n,crl=arr%a)
    arr%len=n
    return
  end subroutine resize_array

  subroutine storage_deallocate
    character(len=*), parameter :: proc = 'storage_deallocate'
    integer :: i
    do i = 1, MAXSTO
       IF (allocated(ptrsto(i)%a)) then
          call chmdealloc(fname,proc,'PTRSTO(I)%a', &
               ptrsto(i)%len,crl=ptrsto(i)%a)
          ptrsto(i)%len=0
       endif
    ENDDO
    call chmdealloc_chm_array(fname,proc,'PTRSTO(I)%a',MAXSTO,ptrsto)
  end subroutine storage_deallocate

end module storage

