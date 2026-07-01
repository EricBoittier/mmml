module derivdhdgb
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_DHDGB==1
!AP/MF
  character(len=*),private,parameter :: file_name   ="derivdhdgb_ltm.src"
  INTEGER MAXNUMS2,TOTALS2
  PARAMETER (MAXNUMS2=5)
  PARAMETER (TOTALS2=10)
  real(chm_real),allocatable,dimension(:),save :: DS_DHDGB
!  REAL(chm_real) DS_DHDGB(TOTALS2)
!  COMMON/DHDGB4/DS_DHDGB

contains
 subroutine allocate_derivdhdgb()
    use memory
    character(len=*),parameter :: routine_name="allocate_derivdhdgb"
    call chmalloc(file_name,routine_name,'ds_dhdgb ',maxaim,crl=DS_DHDGB)
    return
 end subroutine allocate_derivdhdgb
! SUBROUTINE DERIVDHDGB_INIT() 
!   INTEGER I
!   DO I=1,TOTALS2
!      DS_DHDGB(I)=0.D0
!   ENDDO
! END SUBROUTINE DERIVDHDGB_INIT
#endif
END MODULE DERIVDHDGB
