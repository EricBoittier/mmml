module dhdgbc
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_DHDGB==1
!AP/MF
  character(len=*),private,parameter :: file_name   ="dhdgbc_ltm.src"
  real(chm_real),dimension(:),allocatable,save :: scomp
contains
   subroutine allocate_dhdgbc()
      use memory
      character(len=*),parameter :: routine_name="allocate_dhdgbc"
      call chmalloc(file_name,routine_name,'scomp ',maxa,crl=scomp)
   end subroutine allocate_dhdgbc
!  REAL(chm_real) SCOMP(10)
#endif
END MODULE DHDGBC
