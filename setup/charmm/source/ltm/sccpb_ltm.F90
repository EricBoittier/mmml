!CHARMM Element source/fcm/sccpb.fcm $Revision: 1.2 $
module sccpb

  use chm_kinds

!    XIAO_QC_UW0609 Added 
#if KEY_PBEQ==1
#if KEY_SCCDFTB==1

!
!     Define the data necessary for a SCC-DFTB-MM-PB calculation

!     QGSBPSCC        ------> FLAG PB contributions to SCC 
!     QCHDRAD         ------> FLAG to switch on charge dependant radii
!     RF_QM(NSCCTC)   ------> QM rxn field
!     RF_MM(NSCCTC)   ------> MM rxn field
      INTEGER,parameter :: MAXCEN2 = 650
      REAL(chm_real) RF_QM(MAXCEN2)
      REAL(chm_real) RF_MM(MAXCEN2)
      REAL(chm_real) ESCCTBPBOLD,ESCCTBPBGAS,ESCCTBPB
      LOGICAL QPBSCC

      LOGICAL QCHDRAD
      INTEGER FMODE
      INTEGER,PARAMETER :: NCOEMAX=6
      REAL(chm_real) COEFA(NCOEMAX),COEFB(NCOEMAX)
      REAL(chm_real) COEFC(NCOEMAX),COEFD(NCOEMAX)

#endif /*  SCCDFTB*/
#endif /*  PBEQ*/

end module sccpb

