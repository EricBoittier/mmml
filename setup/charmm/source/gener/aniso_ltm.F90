module aniso_fcm

  use dimens_fcm
  use chm_kinds
  implicit none
  character(len=*),private,parameter :: file_name   ="aniso_ltm.src"
  !     NANISO  - Maximum number of anisotropic terms in the list
  !     LSTANI1 - 1st component in the list
  !     LSTANI2 - 2nd component in the list
  !     LSTANI3 - 3rd component in the list
  !     LSTANI4 - 4th component in the list
  !     K11    - force constant for parallel component
  !     K22    - force constant for perpendicular component
  !     K33    - force constant orthogonal to 11 & 22 
  !
  INTEGER,save :: NANISO
  integer,allocatable,dimension(:) :: LSTANI1, LSTANI2, LSTANI3, LSTANI4
  
  real(chm_real),save,allocatable,dimension(:) :: K11, K22, K33

contains
  subroutine allocate_aniso()
    use memory
    character(len=*),parameter :: routine_name="allocate_aniso"
!MFC-- maxa needs to be changed. This should be a reallocate-type routine
    call chmalloc(file_name,routine_name,'lstani1 ',maxa,intg=lstani1)
    call chmalloc(file_name,routine_name,'lstani2 ',maxa,intg=lstani2)
    call chmalloc(file_name,routine_name,'lstani3 ',maxa,intg=lstani3)
    call chmalloc(file_name,routine_name,'lstani4 ',maxa,intg=lstani4)
    call chmalloc(file_name,routine_name,'k11 ',maxa,crl=k11)
    call chmalloc(file_name,routine_name,'k22 ',maxa,crl=k22)
    call chmalloc(file_name,routine_name,'k33 ',maxa,crl=k33)
    naniso = 0
    return
  end subroutine allocate_aniso


  
end module aniso_fcm

