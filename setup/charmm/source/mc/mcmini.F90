module mcmini
  implicit none
#if KEY_MC==1
contains

SUBROUTINE MINSUB(NSTEP,MCMTYP,STEP,TOLFUN,TOLGRD,TOLSTP, &
     BNBND,BIMAG,X,Y,Z)
  !
  !       Performs a minimization without recourse to the command line.
  !       Meant to be called from MC prior to applying the acceptance
  !       criterion.
  !
  !       Aaron R. Dinner
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use block_ltm
  use contrl
  use deriv
  use egrad
  use energym
  use fourdm
  use memory
  use image
  use psf
  use stream

  use steepd_module, only: steep2

#if KEY_DHDGB == 1
  use dhdgb, only: qfhdgb, sdef, totals
#endif  
  implicit none
  !
  INTEGER NSTEP, MCMTYP
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  real(chm_real) :: X(:), Y(:), Z(:)
  !
  INTEGER  CONVRG, I, NVAR, NCALLS
  real(chm_real)   STEP, TOLGRD, TOLFUN, TOLSTP
  !
  !       Pointers.
  !
  real(chm_real),allocatable,dimension(:) :: GRAD, VREF, GBEST
  real(chm_real),allocatable,target,dimension(:) :: VARB, VBEST
  real(chm_real),pointer,dimension(:) :: VARFIN
  LOGICAL QNOENER

  !
  !       Turn on forces again
  !
#if KEY_BLOCK==1
  NOFORC = .FALSE.  
#endif
  QNOENER = .FALSE.
  !
  !       Do some initialization and parse the command line.
  !
  CONVRG = 0
  !
  IF (NPRINT .LT. 0 .OR. NPRINT .GT. NSTEP) NPRINT = 0
  !
  !       Determine the number of variables (from IMOVE in psf.f90)
  !
  CALL CALCNVAR(.FALSE.,(/0/),NVAR)
  !
  !       Allocate storage.
  !
  call chmalloc('mcmini.src','MINSUB','VARB',NVAR,crl=VARB)
  call chmalloc('mcmini.src','MINSUB','GRAD',NVAR,crl=GRAD)
  call chmalloc('mcmini.src','MINSUB','VREF',NVAR,crl=VREF)
  call chmalloc('mcmini.src','MINSUB','VBEST',NVAR,crl=VBEST)
  call chmalloc('mcmini.src','MINSUB','GBEST',NVAR,crl=GBEST)

  !
  !       Fill the variable array with the coordinates to be optimized.
  !
  CALL GETVR1(.TRUE.,NATOM, VARB, IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/), &       
#endif
       XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
       .FALSE.,(/ZERO/),(/0/), & 
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB == 1
       , qfhdgb, sdef, totals &
#endif
       )
  !
  IF (MCMTYP .EQ. 0) THEN
     CALL STEEP2(0, NVAR, VARB, GRAD, VREF, &
          NSTEP,STEP,TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS, &
          VBEST, GBEST, NPRINT, QNOENER &
#if KEY_DHDGB == 1
          ,qfhdgb &
#endif
          )
     VARFIN => VBEST
  ELSE IF (MCMTYP .EQ. 1) THEN
     !         Use VREF  in place of NEWV (CONJUG in conjug.src)
     !         Use VBEST in place of P    (CONJUG in conjug.src)
     CALL CONJG2(NVAR, VARB, VREF, GRAD, &
#if KEY_CHEQ==1
          NATOM,.FALSE.,EPROP(SDMIN),              & 
#endif
          0,NSTEP,100,PT9999,1,NPRINT,STEP, VBEST, &
          OUTU,TOLFUN,TOLGRD,100,TOLSTP,CONVRG,NCALLS)
     VARFIN => VARB

  ELSE
     CALL WRNDIE (-5, '<MINSUB>', &
          'INTERNAL ERROR---UNKNOWN ALGORITHM')
  ENDIF
  !
  !       Fill the coordinate arrays with the optimized variables.
  !
  CALL PUTVR1(.TRUE.,NATOM, VARFIN, IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       .FALSE.,(/ZERO/),(/0/), &             
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &              
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &                    
#endif
#if KEY_DHDGB == 1
       , qfhdgb, sdef, totals &
#endif
       )
  !
  !       Clear up any temporary storage space.
  !
  call chmdealloc('mcmini.src','MINSUB','GRAD',NVAR,crl=GRAD)
  call chmdealloc('mcmini.src','MINSUB','VARB',NVAR,crl=VARB)
  call chmdealloc('mcmini.src','MINSUB','VREF',NVAR,crl=VREF)
  call chmdealloc('mcmini.src','MINSUB','VBEST',NVAR,crl=VBEST)
  call chmdealloc('mcmini.src','MINSUB','GBEST',NVAR,crl=GBEST)

  !
  !       Turn off forces again
  !
#if KEY_BLOCK==1
  NOFORC = .TRUE.  
#endif
  !
  !       Minimization does not update the final energy
  !
  CALL ENERGY(X, Y, Z, DX, DY, DZ, BNBND, BIMAG, 0)
  !
  RETURN
END SUBROUTINE MINSUB
#endif 

end module mcmini

