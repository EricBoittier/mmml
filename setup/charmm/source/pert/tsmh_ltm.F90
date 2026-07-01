module tsmh
  use chm_kinds
  
#if KEY_TSM==1 /*tsmh_fcm*/
  !  TSMH.FCM
  !
  !  CONTAINS DATA STRUCTURE FOR SELECTION LISTS USED IN THERMODYNAMIC
  !  PERTURBATION METHODS.  BASE ARRAY: BPERT; LENGTH ARRAY: LPERT
  !  DEFINES STRUCTURE ON THE HEAP.
  !
  !  AUTHOR: Stephen H. Fleischman  (12/86)
  !
  !  REACLS:           REACTANT SELECTION LIST  INTEGER*2  NATOM
  !  PRODLS:           PRODUCT SELECTION LIST   INTEGER*2  NATOM
  !  BSKIPR:           BOND SKIP LIST FOR REACTANT INTEGER*2 NBONDS
  !  BSKIPP:           BOND SKIP LIST FOR PRODUCT INTEGER*2 NBONDS
  !  ASKIPR:           ANGLE SKIP LIST FOR REACTANT INTEGER*2 NTHETA
  !  ASKIPP:           ANGLE SKIP LIST FOR PRODUCT INTEGER*2 NTHETA
  !*** clbiii mod for ub
  !  UBSKIPR:          Urey-Bradley angle skip list for product
  !  UBSKIPP:          Urey-Bradley angle skip list for product
  !***end of clbiii mod for ub
  !  ASKIPP:           ANGLE SKIP LIST FOR PRODUCT INTEGER*2 NTHETA
  !  PSKIPR:           DIHED SKIP LIST FOR REACTANT INTEGER*2 NPHI
  !  PSKIPP:           DIHED SKIP LIST FOR PRODUCT INTEGER*2 NPHI
  !  ISKIPR:           IMPROPER SKIP LIST FOR REACTANT INTEGER*2 NIMPHI
  !  ISKIPP:           IMPROPER SKIP LIST FOR PRODUCT INTEGER*2 NIMPHI
#if KEY_CMAP==1
  !  CTSKIPR:          Dihedral Cross-term skip list for reactant
  !  CTSKIPP:          Dihedral Cross-term skip list for product
#endif 
  !  PIGGLS:           "PIGGY" ATOM LIST
  !  BACKLS:           "BACK" ATOM LIST
  INTEGER SNPRTT
  !*** clbiii mod for ub
#if KEY_IF==1 || KEY_CMAP==1
  !
#endif
  !!      PARAMETER (SNPRTT=16)
#if KEY_ELSE==1
  !
#endif
  !!      PARAMETER (SNPRTT=14)
#if KEY_ENDIF==1
  !
#endif
!!!      PARAMETER (SNPRTT=12)
!!!***end of clbiii mod for ub
  !! 
  !!      INTEGER SNPERT,REACLS,PRODLS,BSKIPR,BSKIPP,ASKIPR,ASKIPP
  !!      INTEGER PSKIPR,PSKIPP,ISKIPR,ISKIPP,PIGGLS,BACKLS
!!!*** clbiii mod for ub
  !!      INTEGER UBSKIPR, UBSKIPP
  !!
#if KEY_IF==1 || KEY_CMAP==1
  !
#endif
  !!      INTEGER CTSKIPR,CTSKIPP
#if KEY_ENDIF==1
  !
#endif
  !!      COMMON /NPERT/ SNPERT,REACLS,PRODLS,BSKIPR,BSKIPP,ASKIPR, &
  !!                     ASKIPP,UBSKIPR, UBSKIPP, &
#if KEY_IF==1 || KEY_CMAP==1
  !
#endif
  !!                     CTSKIPR,CTSKIPP, &
#if KEY_ENDIF==1
  !
#endif
  !!                     PSKIPR,PSKIPP,ISKIPR,ISKIPP,PIGGLS,BACKLS
!!!      COMMON /NPERT/ SNPERT,REACLS,PRODLS,BSKIPR,BSKIPP,ASKIPR,
!!!     1               ASKIPP,PSKIPR,PSKIPP,ISKIPR,ISKIPP,PIGGLS,BACKLS
!!!***end of clbiii mod for ub


  integer,allocatable,dimension(:)        :: REACLS     !(NATOM)
  integer,allocatable,dimension(:)        :: PRODLS     !(NATOM)
  integer,allocatable,dimension(:)        :: BSKIPR     !(NBOND)
  integer,allocatable,dimension(:)        :: BSKIPP     !(NBOND)
  integer,allocatable,dimension(:)        :: ASKIPR     !(NTHETA)
  integer,allocatable,dimension(:)        :: ASKIPP     !(NTHETA)
  integer,allocatable,dimension(:)        :: UBSKIPR    !(NTHETA)
  integer,allocatable,dimension(:)        :: UBSKIPP    !(NTHETA)
#if KEY_CMAP==1
  integer,allocatable,dimension(:)        :: CTSKIPR    !(NCRTERM)     
#endif
#if KEY_CMAP==1
  integer,allocatable,dimension(:)        :: CTSKIPP    !(NCRTERM)     
#endif
  integer,allocatable,dimension(:)        :: PSKIPR     !(NPHI)
  integer,allocatable,dimension(:)        :: PSKIPP     !(NPHI)
  integer,allocatable,dimension(:)        :: ISKIPR     !(NIMPHI)
  integer,allocatable,dimension(:)        :: ISKIPP     !(NIMPHI)
  integer,allocatable,dimension(:)        :: PIGGLS     !(NATOM)
  integer,allocatable,dimension(:)        :: BACKLS     !(NATOM)

#endif /* (tsmh_fcm)*/

end module tsmh

