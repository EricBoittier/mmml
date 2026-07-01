module sccgsbp
  use chm_kinds
  use dimens_fcm
#if KEY_SCCDFTB==1
  use sccdftb, only: maxcen   
#endif
  implicit none
! QC, Nov. 20. 2003
#if KEY_PBEQ==1
#if KEY_GSBP==1
#if KEY_SCCDFTB==1
!
!     Define the data necessary for a SCC-DFTB-MM-GSBP calculation

!     QGSBPSCC        ------> FLAG GSBP contributions to SCC 
!     LSCCMM          ------> IF UPDATE MM LST FOR SCC
!     GAMAGS (NSCCTC^2/2)--> Lower triangle for QM-QM rxn field
!     OMEGAGS(NSCCTC) -----> Outer rxn field, MM rxn field
!
      INTEGER,PARAMETER :: MMDIM=MAXCEN*(MAXCEN+1)/2
      real(chm_real) GAMAGS(MMDIM)
      real(chm_real) OMEGAGS(MAXCEN)
      LOGICAL QGSBPSCC,LSCCMM
      
#endif /*  SCCDFTB*/
#endif /*  GSBP*/
#endif /*  PBEQ*/
!
end module sccgsbp

