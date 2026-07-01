module deriv
  use chm_kinds
  use dimens_fcm
  character(len=*),private,parameter :: file_name   ="deriv_ltm.src"
!
!  deriv.fcm -  Forces common block.
!  DX,DY,DZ - Force components
!
      real(chm_real),allocatable,dimension(:),save :: DX,DY,DZ

contains
  subroutine allocate_deriv()
    use memory
    character(len=*),parameter :: routine_name="allocate_deriv"
#ifdef KEY_RESIZE
    integer:: i0
    i0=1
    call chmalloc(file_name,routine_name,'dx ',2*i0,crl=dx)
    call chmalloc(file_name,routine_name,'dy ',2*i0,crl=dy)
    call chmalloc(file_name,routine_name,'dz ',2*i0,crl=dz)
#else
    call chmalloc(file_name,routine_name,'dx ',maxaim,crl=dx)
    call chmalloc(file_name,routine_name,'dy ',maxaim,crl=dy)
    call chmalloc(file_name,routine_name,'dz ',maxaim,crl=dz)
#endif    
    return
  end subroutine allocate_deriv
end module deriv

