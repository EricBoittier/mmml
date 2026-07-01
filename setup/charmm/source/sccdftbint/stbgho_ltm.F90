module stbgho
  use chm_kinds
  use dimens_fcm
#if KEY_SCCDFTB==1
!
!     Defines the data necessary for a QM calculation on a system.
!         
!  
!     Variable  Index    Purpose 
! 
!     QLINK              True - containing QM link atoms
!     BT                 C' = BT.C
!     BTM                C  = BTM.C'
!     NATQM              Number of QM atom 
!     NQMLNK             Number of QM link atoms
!     IGHOSL             Flag of QM atoms being a GHO atom
!     IQLINK             ATOM NUMBER OF QM LINK ATOM
!     JQLINK             MM ATOMS TO WHICH QM LINK ATOM CONNECTS
!     KQLINK             QM ATOM TO WHICH QM LINK ATOM CONNECTS
!     MAXQMLINK          MAX. No. of QM LINK ATOMS ALLOWED
!
      LOGICAL QLINK
      INTEGER NATQM,NQMLNK,IQLINK,JQLINK,KQLINK,MAXQMLINK,MXQM16
      real(chm_real)  BT,BTM,QMATMQ,DBTMMM,PHO,PBHO,FAO,FAOB,SAO, &
              WHO,WBHO

      PARAMETER (MAXQMLINK=5,MXQM16=16*MAXQMLINK)
      COMMON/QLINKI/NATQM,NQMLNK,IQLINK(MAXQMLINK),JQLINK(3,MAXQMLINK),   &
                    KQLINK(MAXQMLINK)
      COMMON/QLINKL/QLINK
      COMMON/QLINKF/BT(MXQM16),BTM(MXQM16),QMATMQ(MAXQMLINK), &
                    DBTMMM(3,3,MXQM16)

      INTEGER  NORBG
      COMMON/QLINKN/NORBG
!
! label of GHO atoms and QM atom directly linked to GHO atoms in
! the QM domain, similiar to IQLINK and KQLINK
!
      INTEGER MXNN
      PARAMETER (MXNN=650)

      INTEGER IGHOSL
      COMMON /GHOSEL/ IGHOSL(MXNN)

      INTEGER IGLNK,KGLNK
      COMMON/QLINKQ/ IGLNK(MAXQMLINK),KGLNK(MAXQMLINK)
!
! empirical repulsion term between A-B 
!
      real(chm_real) :: CTWO(5)  = [ -55.715D0, -45.902D0, -55.496D0, 0.0D0, 0.0D0 ]
      real(chm_real) :: CFOUR(5) = [  7.040D0,   1.721D0,   5.006D0, 0.0D0, 0.0D0 ]
      real(chm_real),dimension(5) :: CONE  = [ 6.422D0,  16.767D0,   9.262D0, 0.0D0, 0.0D0 ]
!      COMMON/QECOR/ CONE(5), CTWO(5), CFOUR(5)
!--mfc--       DATA CTWO  /-55.715D0, -45.902D0, -55.496D0, 0.0D0, 0.0D0/
!--mfc--       DATA CFOUR /  7.040D0,   1.721D0,   5.006D0, 0.0D0, 0.0D0/
!--mfc--       DATA CONE  /  6.422D0,  16.767D0,   9.262D0, 0.0D0, 0.0D0/
!--mfc--       DATA CTWO  /-55.715D0, -45.902D0, -55.496D0, 0.0D0, 0.0D0/
!--mfc--       DATA CFOUR /  7.040D0,   1.721D0,   5.006D0, 0.0D0, 0.0D0/
#endif 
end module stbgho

