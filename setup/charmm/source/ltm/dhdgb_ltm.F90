module dhdgb
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_DHDGB==1
!AP/MF
  character(len=*),private,parameter :: file_name="dhdgb_ltm.src"
  LOGICAL QFHDGB
  INTEGER MAXNUMS,TOTALS 
  PARAMETER (MAXNUMS=5)
  PARAMETER (TOTALS=10)
  REAL(chm_real) MITHICK_FHDGB,MATHICK_FHDGB
  REAL(chm_real) SAMASS
  REAL(chm_real) FRIC_DHDGB
  PARAMETER (MITHICK_FHDGB=0.D0)
  PARAMETER (MATHICK_FHDGB=25.D0)
! REAL(chm_real) SDEF(TOTALS)
! REAL(chm_real) DS_DHDGB(TOTALS)
! REAL(chm_real) VS_DHDGB(TOTALS)
! REAL(chm_real) SCOMP(TOTALS)
!  COMMON/DHDGB1/QFHDGB
!  COMMON /DHDGB2/SDEF
!  COMMON /DHDGB3/ SAMASS,FRIC_DHDGB
!  SAVE /DHDGB1/
!  SAVE /DHDGB3/
   real(chm_real),save,allocatable,dimension(:) :: sdef
contains
  subroutine allocate_dhdgb()
     use memory 
     character(len=*),parameter :: routine_name="allocate_dhdgb"
     call chmalloc(file_name,routine_name,'sdef ',maxaim,crl=sdef)
     return
  end subroutine allocate_dhdgb

! SUBROUTINE DHDGB_INIT()
!   INTEGER I
!   QFHDGB=.FALSE.
!   DO I=1,TOTALS
!      SDEF(I)=MATHICK_FHDGB
!   ENDDO
! END SUBROUTINE DHDGB_INIT
#endif
END MODULE DHDGB
