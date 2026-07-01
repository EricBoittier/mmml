module egrad

contains

SUBROUTINE EGRAD1(NVAR,VARB,VREF,FUNC,GRAD,ISTEP,ICALL,ERSTAT)
  !-----------------------------------------------------------------------
  !     The energy and forces are calculated and returned by this routine.
#if KEY_CHEQ==1
  use cheq,only:cgfix,cgtmp,cgmodel,   &                  
#endif
#if KEY_CHEQ==1
     DCH,SUMDCH,DDCH                   
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln,fqcfor      
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use vector
  use bases_fcm
  use coord
  use contrl
  use energym
  use fourdm
  use deriv
  use holonom,only:holonoma,holonomf
  use image
  use parallel,only: gcomb
  use psf
  use shake
  use stream
  use coordc
#if KEY_RXNCOR == 1
  use lupcns_module, only: lupcns
#endif
  use lupcom
  use icfix
#if KEY_FLUCQ==1
  use flucq
#endif 
  use traj_mod,only: wrttrj
  use memory
  use heurist,only:updeci
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
  use derivdhdgb
#endif
  implicit none

  !     Passed variables.
  real(chm_real),allocatable,dimension(:) :: DDF
  real(chm_real),allocatable,dimension(:) :: LUPCX
  real(chm_real),allocatable,dimension(:) :: LUPCY
  real(chm_real),allocatable,dimension(:) :: LUPCZ
  real(chm_real),allocatable,dimension(:) :: X1
  real(chm_real),allocatable,dimension(:) :: Y1
  real(chm_real),allocatable,dimension(:) :: Z1
  real(chm_real),allocatable,dimension(:) :: DTX

  INTEGER ISTEP, ICALL, NVAR, ERSTAT
  real(chm_real)  FUNC, GRAD(*), VARB(*), VREF(*)

  !     Local variables.
  INTEGER  I, IVAR, NTOT, NAT3
  LOGICAL  QOK
  real(chm_real)   EDIF, ETWO
#if KEY_DHDGB==1
!AP/MF
  real(chm_real) zer(totals)
  real(chm_real) dumm(totals)
  real(chm_real),allocatable,dimension(:) :: sdef1
#endif


  !     Pointers.
  INTEGER  LUPDIM

  !     Allocate space for reference coordinate sets if holonomic cons.
  IF (QHOLO) THEN
     call chmalloc('egrad1.src','EGRAD1','X1',NATOM,crl=X1)
     call chmalloc('egrad1.src','EGRAD1','Y1',NATOM,crl=Y1)
     call chmalloc('egrad1.src','EGRAD1','Z1',NATOM,crl=Z1)

     x1(1:NATOM) = x(1:NATOM)
     y1(1:NATOM) = y(1:NATOM)
     z1(1:NATOM) = z(1:NATOM)
#if KEY_DHDGB==1
!AP/MF
  DO I=1,TOTALS
     ZER(I)=0.D0
  ENDDO
     call chmalloc('egrad1.src','EGRAD1','sdef1',totals,crl=sdef1)
     sdef1(1:totals)=sdef(1:totals)
#endif

     CALL PUTVR1(MINXYZ,NATOM,VREF,IMOVE,X1,Y1,Z1, &
#if KEY_CHEQ==1
          .FALSE.,CG,CGFIX,CGTMP, & 
#endif
          .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
          .FALSE.,(/ZERO/),(/0/), & 
#endif
          .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
          ,QFHDGB,SDEF1,TOTALS &
#endif
         )
  ENDIF

  !     Fill the coordinate arrays with the variables.

  CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       QCGMIN,CG,CGFIX,CGTMP, &  
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
      ,QFHDGB,SDEF,TOTALS &
#endif

       )

  IF(QHOLO) THEN
     CALL HOLONOMA(X,Y,Z,X1,Y1,Z1, &
          .FALSE.,.FALSE.,QOK)

     !       refill the variable arrays.
     CALL GETVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z,.FALSE., &
#if KEY_CHEQ==1
          .FALSE.,CG, &             
#endif
          'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
          QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
          .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
          DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
!AP/MF
          ,QFHDGB,SDEF,TOTALS &
#endif

          )
#if KEY_DHDGB==1
!AP/MF
      DO I =1,TOTALS
           ZER(I)=0.D0
      ENDDO
#endif
  ENDIF
  !
  !     Do list updates if appropriate
  !yw   Always update even in the process of line search.
  !yw   This is the decision of the CHARMM meeting on November 16, 1991
#if KEY_NIH==1
  ! This was a poor decision.  The NIH version will do it differently. - BRB
  ! Perhaps this decision should be reconsidered...
  IF(ICALL > 0) THEN
     CALL UPDECI(ISTEP,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
  ENDIF
#else /**/
  CALL UPDECI(ISTEP,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
#endif 
  !
  !     Calculate the energy and forces.
  !

  IF (QSADLE) THEN
#if KEY_FOURD==1
     IF(DIM4) CALL WRNDIE(-4,'<EGRAD1>','No Hessian with 4D') 
#endif
     !     use gradient squared as the energy
     NAT3=NATOM*3
     call chmalloc('egrad1.src','EGRAD1','DTX',NAT3,crl=DTX)
     NTOT=(NAT3*(NAT3+1))/2
     call chmalloc('egrad1.src','EGRAD1','DDF',NTOT,crl=DDF)
     DDF(1:NTOT)=zero
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,ICALL,NAT3,DDF)
     IF(QNUMER) CALL DNUMER(DX,DY,DZ)
#if KEY_PARALLEL==1
     CALL VDGBR(DX,DY,DZ,1)
     CALL GCOMB(DDF,NTOT)
#endif 
     CALL GETVR1(MINXYZ,NATOM,DTX,IMOVE,DX,DY,DZ,.FALSE., &
#if KEY_CHEQ==1
          QCGMIN,CG,                         & 
#endif
          'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,FQSELN,               & 
#endif
          .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
        ,.FALSE.,ZER,totals &
#endif
     )
     CALL DOTPR(DTX,DTX,NAT3,FUNC)
     FUNC=FUNC/NAT3
     ETWO=TWO/NAT3
     EPROP(EPOT)=FUNC
     CALL RALEG2(DTX,GRAD,NATOM,DDF)

     call chmdealloc('egrad1.src','EGRAD1','DDF',NTOT,crl=DDF)
     CALL SCALR8(GRAD,NAT3,ETWO)
     CALL PUTVR1(MINXYZ,NATOM,GRAD,IMOVE,DX,DY,DZ, &
#if KEY_CHEQ==1
          .FALSE.,DCH,CGFIX,CGTMP,  & 
#endif
          .FALSE.,'NONE',(/ZERO/),(/ZERO/),.FALSE., &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,FQSELN,      & 
#endif
          .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE.,ZER,totals &
#endif
       )
  ELSE
     !     normal energy call
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,ICALL &
#if KEY_DHDGB==1
!AP/MF
         ,SDEFIN=SDEF,DS_DHDGBOUT=DS_DHDGB &
#endif

)
     ! SAPATEL
#if KEY_CHEQ==1
     !  What does this mean???
     IF (QCGMIN.AND.(CGMODEL == 0)) THEN
        DDCH(1:natom-1)=DDCH(1:natom-1)-DDCH(NATOM)
     ENDIF
#endif 
     ! SAPATEL
     IF(QNUMER) CALL DNUMER(DX,DY,DZ)
#if KEY_PARALLEL==1
     CALL VDGBR(DX,DY,DZ,1)
#endif 
     !
     ! Remove force components along any holonomic constraints
     IF(QHOLO) THEN
        CALL HOLONOMF(DX,DY,DZ,X,Y,Z,.FALSE.,.FALSE.,QOK)
        IF (.NOT.QOK) ERSTAT = 2
     ENDIF
     !
#if KEY_RXNCOR==1
     ! ... remove force component along LUP constraints, K.Kuczera 14-Mar-97
     IF(QLUPCS) THEN
        LUPDIM = NLUPC*NATOM
        call chmalloc('egrad1.src','EGRAD1','LUPCX',LUPDIM,crl=LUPCX)
        call chmalloc('egrad1.src','EGRAD1','LUPCY',LUPDIM,crl=LUPCY)
        call chmalloc('egrad1.src','EGRAD1','LUPCZ',LUPDIM,crl=LUPCZ)
        CALL LUPCNS(X,Y,Z,XCOMP,YCOMP,ZCOMP,DX,DY,DZ,AMASS,NATOM, &
             LUPCX,LUPCY,LUPCZ)
        call chmdealloc('egrad1.src','EGRAD1','LUPCX',LUPDIM,crl=LUPCX)
        call chmdealloc('egrad1.src','EGRAD1','LUPCY',LUPDIM,crl=LUPCY)
        call chmdealloc('egrad1.src','EGRAD1','LUPCZ',LUPDIM,crl=LUPCZ)
     ENDIF
#endif 
     !
     ! SAPTEL
#if KEY_CHEQ==1
     IF(QCGMIN) THEN
        IF (CGMODEL == 0) THEN
           FUNC = EPROP(EPOT)+SUMDCH
        ELSE IF (CGMODEL == 1) THEN
           FUNC = EPROP(EPOT)
        ELSE IF (CGMODEL == 2) THEN
           FUNC = EPROP(EPOT)+EPROP(CGPOT)
        ENDIF
     ELSE
#endif 
        ! SAPATEL
        FUNC = EPROP(EPOT)
#if KEY_CHEQ==1
     ENDIF
#endif 

     CALL GETVR1(MINXYZ,NATOM,GRAD,IMOVE,DX,DY,DZ,.FALSE., &
#if KEY_CHEQ==1
          QCGMIN,DCH, &             
#endif
          'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,FQSELN, &    
#endif
#if KEY_FOURD==0
          .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
          DIM4,DFDIM,IMOVE4 &       
#endif
#if KEY_DHDGB==1
!AP/MF
         ,QFHDGB,DS_DHDGB,TOTALS &
#endif

          )

  ENDIF

#if KEY_DEBUG==1
  ! create a trajectory file with alternating minimization coordinates and forces.
  CALL WRTTRJ(X,Y,Z)
  CALL WRTTRJ(DX,DY,DZ)
#endif 

  IF (LMINUC) THEN
     IVAR = NVAR - XDIM
     DO I = 1,XDIM
        IVAR = IVAR + 1
        GRAD(IVAR) = DXTL(I)
     ENDDO
     IF (PRNLEV >= 7) THEN
        WRITE(OUTU,'(A,I6)') ' Total number of variables   = ', NVAR
        WRITE(OUTU,'(A,I6)') ' Number of crystal variables = ', XDIM
        WRITE(OUTU,'(A)') ' Crystal variable derivatives :'
        WRITE(OUTU,'(3F16.5)') (GRAD(NVAR-XDIM+I),I=1,XDIM)
     ENDIF
  ENDIF

  !     Free storage space.
  if(allocated(x1))call chmdealloc('egrad1.src','EGRAD1','X1',NATOM,crl=X1)
  if(allocated(y1))call chmdealloc('egrad1.src','EGRAD1','Y1',NATOM,crl=Y1)
  if(allocated(z1))call chmdealloc('egrad1.src','EGRAD1','Z1',NATOM,crl=Z1)

  RETURN
END SUBROUTINE EGRAD1

SUBROUTINE EGRADNEB(NVAR,VARB,VREF,GRAD,PGRAD,PSGRAD,ERSTAT, &
     PTHTAN)
  !-----------------------------------------------------------------------
  !     The energy and forces are calculated and returned by this routine.
  !
#if KEY_FLUCQ==1
  use flucqm, only: fqseln,fqcfor       
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use holonom,only:holonomf
  !
  use bases_fcm
  use coord
  use contrl
  use energym
  use fourdm
  use deriv
  !
#if KEY_RPATH==1
  use epathmod,only: PJDX,PJDY,PJDZ,PSDX,PSDY,PSDZ,PTANX,PTANY,PTANZ 
#endif
  !
  use image
  use psf
  use shake
  use stream
  use coordc
#if KEY_RXNCOR==1
  use lupcns_module, only: lupcns
#endif
  use lupcom
  use icfix
#if KEY_FLUCQ==1
  use flucq
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals
#endif
 
  use memory
  implicit none
  !
  !     Passed variables.
  !
  real(chm_real),allocatable,dimension(:) :: LUPCX
  real(chm_real),allocatable,dimension(:) :: LUPCY
  real(chm_real),allocatable,dimension(:) :: LUPCZ
  real(chm_real),allocatable,dimension(:) :: X1
  real(chm_real),allocatable,dimension(:) :: Y1
  real(chm_real),allocatable,dimension(:) :: Z1

  INTEGER ISTEP, ICALL, NVAR, ERSTAT

  real(chm_real) FUNC,GRAD(*), VARB(*), VREF(*), PGRAD(*), PSGRAD(*)
  real(chm_real) PTHTAN(*)
  !
  !     Local variables.
  !
  INTEGER  I, IVAR, DDF, DTX, NTOT, NAT3
  LOGICAL  QOK
  real(chm_real)   EDIF, ETWO
  !
  !     Pointers.
  !
  INTEGER  LUPDIM
  !
#if KEY_RPATH==1
  !
  !     Do some initialisation.
  !
  !     Allocate space for reference coordinates if holonomic constraints.
  !

  IF (QHOLO) THEN
     call chmalloc('egrad1.src','EGRADNEB','X1',NATOM,crl=X1)
     call chmalloc('egrad1.src','EGRADNEB','Y1',NATOM,crl=Y1)
     call chmalloc('egrad1.src','EGRADNEB','Z1',NATOM,crl=Z1)
     X1(1:natom) = X(1:NATOM)
     Y1(1:natom) = Y(1:NATOM)
     Z1(1:natom) = Z(1:NATOM)
  ENDIF
#if KEY_PARALLEL==1
  CALL VDGBR(PJDX,PJDY,PJDZ,1)
  CALL VDGBR(PSDX,PSDY,PSDZ,1)
#endif 
  !
  ! Remove force component along any holonomic constraints.
  IF(QHOLO) THEN
     CALL HOLONOMF(PJDX,PJDY,PJDZ,X,Y,Z,.FALSE.,.FALSE.,QOK)
     IF (.NOT.QOK) ERSTAT = 2
     CALL HOLONOMF(PSDX,PSDY,PSDZ,X,Y,Z,.FALSE.,.FALSE.,QOK)
     IF (.NOT.QOK) ERSTAT = 2
  ENDIF
  !
#if KEY_RXNCOR==1
  ! ... remove force component along LUP constraints, K.Kuczera 14-Mar-97
  IF(QLUPCS) THEN
     LUPDIM = NLUPC*NATOM
     call chmalloc('egrad1.src','EGRADNEB','LUPCX',LUPDIM,crl=LUPCX)
     call chmalloc('egrad1.src','EGRADNEB','LUPCY',LUPDIM,crl=LUPCY)
     call chmalloc('egrad1.src','EGRADNEB','LUPCZ',LUPDIM,crl=LUPCZ)
     CALL LUPCNS(X,Y,Z,XCOMP,YCOMP,ZCOMP, &
          PJDX,PJDY,PJDZ,AMASS,NATOM, &
          LUPCX,LUPCY,LUPCZ)
     CALL LUPCNS(X,Y,Z,XCOMP,YCOMP,ZCOMP, &
          PSDX,PSDY,PSDZ,AMASS,NATOM, &
          LUPCX,LUPCY,LUPCZ)
     call chmdealloc('egrad1.src','EGRADNEB','LUPCX',LUPDIM,crl=LUPCX)
     call chmdealloc('egrad1.src','EGRADNEB','LUPCY',LUPDIM,crl=LUPCY)
     call chmdealloc('egrad1.src','EGRADNEB','LUPCZ',LUPDIM,crl=LUPCZ)
  ENDIF
#endif 
  !
  pgrad(1:nvar) = GRAD(1:NVAR)

  CALL GETVR1(MINXYZ,NATOM,PGRAD,IMOVE,PJDX,PJDY,PJDZ,.FALSE., &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/), &       
#endif
       'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,FQSELN, &     
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) & 
#endif
#if KEY_FOURD==1
       DIM4,DFDIM,IMOVE4 &      
#endif
#if KEY_DHDGB==1
      ,.FALSE., (/ZERO/),TOTALS &
#endif
       )
  PSGRAD(1:NVAR)=zero
  CALL GETVR1(MINXYZ,NATOM,PSGRAD,IMOVE,PSDX,PSDY,PSDZ,.FALSE., &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/), &      
#endif
       'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
       .FALSE.,FQCFOR,FQSELN, & 
#endif
       .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
!AP/MF
      ,.FALSE., (/ZERO/),TOTALS &
#endif
      )
  PTHTAN(1:NVAR)=zero
  CALL GETVR1(MINXYZ,NATOM,PTHTAN,IMOVE,PTANX,PTANY,PTANZ,.FALSE., &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/), &      
#endif
       'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
       .FALSE.,FQCFOR,FQSELN, & 
#endif
       .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
      ,.FALSE., (/ZERO/),TOTALS &
#endif
     )

  !br...B980902.br fix FDIM to DFDIM in the FOURD code above (Simon Berneche)
  !
  !
  !     Free storage space.
  !
  if (QHOLO) then
     call chmdealloc('egrad1.src','EGRADNEB','X1',NATOM,crl=X1)
     call chmdealloc('egrad1.src','EGRADNEB','Y1',NATOM,crl=Y1)
     call chmdealloc('egrad1.src','EGRADNEB','Z1',NATOM,crl=Z1)
  endif
  !
#endif 
  RETURN
END SUBROUTINE EGRADNEB

SUBROUTINE GETVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
     QCGMIN,cheq_CG,       & 
#endif
     XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
     QFLUC,CG,FQSELN,      & 
#endif
     QDIM4,FDIM,IMOVE4 &
#if KEY_DHDGB==1
!AP/MF
      ,QFHDGB,SDEF,TOTALS &
#endif

)
  !-----------------------------------------------------------------------
  !     Put the coordinates and crystal variables into the variable array.
  !

#if KEY_CHEQ==1
  use cheq,only:GETVR1Q         
#endif

  use chm_kinds
  use dimens_fcm
  use tsms_mod
  use tsmh
#if KEY_DHDGB==1
!AP/MF
  use stream
#endif
  implicit none
  !
#if KEY_DHDGB==1
!AP/MF
  LOGICAL,intent(in),optional :: QFHDGB
  real(chm_real),intent(in),optional :: SDEF(:)
  INTEGER,intent(in),optional :: TOTALS
#endif
  CHARACTER(len=4) XTLTYP
  INTEGER     IMOVE(*), NATOM
  LOGICAL     MINXYZ, LMINUC
  real(chm_real)      VARB(*), X(*), Y(*), Z(*)
  real(chm_real) :: XTLABC(:) ,XTLREF(:)
  ! SAPATEL
#if KEY_CHEQ==1
  real(chm_real)       CHEQ_CG(*)
  LOGICAL      QCGMIN
#endif 
  ! SAPATEL
#if KEY_FLUCQ==1
  real(chm_real)       CG(*)
  LOGICAL     QFLUC
  INTEGER     FQSELN(*)
#endif 
  LOGICAL     QDIM4
  real(chm_real)      FDIM(*)
  INTEGER     IMOVE4(*)
  !
  !
  INTEGER     I, IP
  !
#if KEY_TSM==1
  IF(QTSM .AND. MINXYZ) THEN
     CALL GETVR1A(NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
          QCGMIN,CHEQ_CG,                      & 
#endif
          XTLTYP,XTLABC,XTLREF,BACKLS)
     RETURN
  ENDIF
#endif 
  !
  IP = 0
  IF (MINXYZ) THEN
     DO I = 1,NATOM
        IF (IMOVE(I)  ==  0) THEN
           VARB(IP + 1) = X(I)
           VARB(IP + 2) = Y(I)
           VARB(IP + 3) = Z(I)
           IP = IP + 3
        ENDIF
     ENDDO
#if KEY_DHDGB==1
!AP/MF
    IF (QFHDGB) THEN
      DO I=1,TOTALS
       IP=IP+1
       VARB(IP)=SDEF(I)
      ENDDO
    ENDIF
#endif
#if KEY_FOURD==1
     IF(QDIM4) THEN
        DO I = 1,NATOM
           IF (IMOVE4(I)  ==  0) THEN
              IP = IP + 1
              VARB(IP) = FDIM(I)
           ENDIF
        ENDDO
     ENDIF
#endif 
#if KEY_FLUCQ==1
     IF(QFLUC) THEN
        DO I = 1,NATOM
           IF (FQSELN(I)  >  0) THEN
              IP = IP + 1
              VARB(IP) = CG(I)
           ENDIF
        ENDDO
     ENDIF
#endif 
  ENDIF
  !
  IF (LMINUC) CALL GETXTL(VARB(IP+1),XTLABC,XTLTYP,XTLREF)
  ! SAPATEL
#if KEY_CHEQ==1
  IF (QCGMIN) CALL GETVR1Q(VARB(IP+1),CHEQ_CG)
#endif 
  ! SAPATEL
  RETURN
END SUBROUTINE GETVR1

#if KEY_TSM==1
SUBROUTINE GETVR1A(NATOM,VARB,IMOVE,X,Y,Z,LMINUC, &
#if KEY_CHEQ==1
     QCGMIN,CG,                & 
#endif
     XTLTYP,XTLABC,XTLREF,BACKLS)
  !
#if KEY_CHEQ==1
  use cheq,only:GETVR1Q         
#endif

  use chm_kinds
  use dimens_fcm
  use tsms_mod
  implicit none
  !
  CHARACTER(len=4) XTLTYP
  INTEGER     IMOVE(*),BACKLS(*),NATOM
  LOGICAL     LMINUC
  real(chm_real)      VARB(*), X(*), Y(*), Z(*)
  real(chm_real)      XTLABC(6), XTLREF(6)
  !
  ! SAPATEL
#if KEY_CHEQ==1
  LOGICAL      QCGMIN
  real(chm_real)       CG(*)
#endif 
  ! SAPATEL
  INTEGER     I, IP
  !
  IP = 0
  DO I = 1,NATOM
     IF (IMOVE(I)  ==  0.AND.BACKLS(I) == 0) THEN
        VARB(IP + 1) = X(I)
        VARB(IP + 2) = Y(I)
        VARB(IP + 3) = Z(I)
        IP = IP + 3
     ENDIF
  ENDDO
  !
  IF (LMINUC) CALL GETXTL(VARB(IP+1),XTLABC,XTLTYP,XTLREF)
  ! SAPATEL
#if KEY_CHEQ==1
  IF (QCGMIN) CALL GETVR1Q(VARB(IP+1),CG)
#endif 
  ! SAPATEL
  RETURN
END SUBROUTINE GETVR1A
#endif 

SUBROUTINE PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
     QCGMIN,CHEQ_CG,CG_FIX,CGT,           & 
#endif
     LMINUC,XTLTYP,XTLABC,XTLREF,LCOORD, &
#if KEY_FLUCQ==1
     QFLUC,CG,FQSELN,                     & 
#endif
     QDIM4,FDIM,IMOVE4 &
#if KEY_DHDGB==1
!AP/MF
     ,QFHDGB,SDEF,TOTALS &
#endif
      )
  !-----------------------------------------------------------------------
  !     Fill the coordinate and crystal arrays with the variables.
  !
#if KEY_CHEQ==1
  use cheq,only: putVR1Q         
#endif

  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use tsms_mod
  use tsmh
  !
  implicit none
  !
#if KEY_DHDGB==1
!AP/MF
  LOGICAL,intent(in),optional :: QFHDGB
  real(chm_real),intent(out),optional :: SDEF(:)
  INTEGER,intent(in),optional :: TOTALS
#endif
  CHARACTER(len=4) XTLTYP
  INTEGER     IMOVE(*),NATOM
  !     lcoord:  Whether this is a coordinate or force copy. Only matters for
  !     TSM piggyback.
  LOGICAL     MINXYZ, LMINUC, LCOORD
  real(chm_real)      VARB(*), X(*), Y(*), Z(*)
  real(chm_real) :: XTLABC(:), XTLREF(:)
  ! SAPATEL
#if KEY_CHEQ==1
  LOGICAL QCGMIN
  real(chm_real) CHEQ_CG(*), CG_FIX(*), CGT(*)
#endif 
  ! SAPATEL
#if KEY_FLUCQ==1
  LOGICAL     QFLUC
  real(chm_real)      CG(*)
  INTEGER     FQSELN(*)
#endif 
  LOGICAL     QDIM4
  real(chm_real)      FDIM(*)
  INTEGER     IMOVE4(*)
  !
  INTEGER     I, IP
  LOGICAL     QFORCE
  !
  QFORCE=.NOT.LCOORD
  !
#if KEY_TSM==1
  IF(QTSM .AND. MINXYZ) THEN
     CALL PUTVR1A(NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
          QCGMIN,CHEQ_CG,CG_FIX,CGT,        & 
#endif
          LMINUC,XTLTYP,XTLABC,XTLREF, &
          BACKLS,LCOORD)
     RETURN
  ENDIF
#endif 
  !
  IP = 0
  IF (MINXYZ) THEN
     DO I = 1,NATOM
        IF(IMOVE(I)  ==  0) THEN
           X(I) = VARB(IP + 1)
           Y(I) = VARB(IP + 2)
           Z(I) = VARB(IP + 3)
           IP = IP + 3
        ELSE IF(QFORCE) THEN
           X(I) = ZERO
           Y(I) = ZERO
           Z(I) = ZERO
        ENDIF
     ENDDO
#if KEY_DHDGB==1
!AP/MF
  IF (QFHDGB) THEN
     DO I=1,TOTALS
        IP=IP+1
        SDEF(I)=VARB(IP)
     ENDDO
  ELSE
     DO I=1,TOTALS
        SDEF(I)=0.0D0
     ENDDO   
  ENDIF
#endif
#if KEY_FOURD==1
     IF(QDIM4) THEN
        DO I = 1,NATOM
           IF (IMOVE4(I)  ==  0) THEN
              IP = IP + 1
              FDIM(I) = VARB(IP)
           ELSE IF(QFORCE) THEN
              FDIM(I) = ZERO
           ENDIF
        ENDDO
     ENDIF
#endif 
#if KEY_FLUCQ==1
     IF(QFLUC) THEN
        DO I = 1,NATOM
           IF (FQSELN(I)  >  0) THEN
              IP = IP + 1
              CG(I) = VARB(IP)
           ELSE IF(QFORCE) THEN
              CG(I) = ZERO
           ENDIF
        ENDDO
     ENDIF
#endif 
  ENDIF
  !
  IF (LMINUC) CALL PUTXTL(VARB(IP+1),XTLABC,XTLTYP,XTLREF)
  ! SAPATEL
#if KEY_CHEQ==1
  IF (QCGMIN) CALL PUTVR1Q(VARB(IP+1),CHEQ_CG,CG_FIX,CGT)
#endif 
  ! SAPATEL
  RETURN
END SUBROUTINE PUTVR1

#if KEY_TSM==1
SUBROUTINE PUTVR1A(NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
     QCGMIN,CG,CG_FIX,CGT,        & 
#endif
     LMINUC,XTLTYP,XTLABC,XTLREF,BACKLS,LCOORD)
  !-----------------------------------------------------------------------
  !     Fill the coordinate and crystal arrays with the variables.
  !
#if KEY_CHEQ==1
  use cheq,only: putVR1Q         
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use tsms_mod
  implicit none
  !
  CHARACTER(len=4) XTLTYP
  INTEGER     IMOVE(*),BACKLS(*),NATOM
  !     lcoord:  Whether this is a coordinate or force copy. Only matters for
  !     TSM piggyback.
  LOGICAL     LMINUC, LCOORD
  real(chm_real)      VARB(*), X(*), Y(*), Z(*)
  real(chm_real)      XTLABC(6), XTLREF(6)
  ! SAPATEL
#if KEY_CHEQ==1
  LOGICAL QCGMIN
  real(chm_real) CG(*), CG_FIX(*), CGT(*)
#endif 
  ! SAPATEL
  !
  INTEGER     I, IP
  LOGICAL     QFORCE
  !
  QFORCE=.NOT.LCOORD
  !
  IP = 0
  DO I = 1,NATOM
     IF (IMOVE(I) == 0 .AND. BACKLS(I) == 0) THEN
        X(I) = VARB(IP + 1)
        Y(I) = VARB(IP + 2)
        Z(I) = VARB(IP + 3)
        IP = IP + 3
     ELSE IF(QFORCE) THEN
        X(I) = ZERO
        Y(I) = ZERO
        Z(I) = ZERO
     ENDIF
  ENDDO
  IF(PIGSET) THEN
     IF(LCOORD) THEN
        CALL PIGCVSET(X,Y,Z)
     ELSE
        CALL BACK0(X,Y,Z)
     ENDIF
  ENDIF
  !
  IF (LMINUC) CALL PUTXTL(VARB(IP+1),XTLABC,XTLTYP,XTLREF)
  ! SAPATEL
#if KEY_CHEQ==1
  IF (QCGMIN) CALL PUTVR1Q(VARB(IP+1),CG,CG_FIX,CGT)
#endif 
  ! SAPATEL
  RETURN
END SUBROUTINE PUTVR1A
#endif 

SUBROUTINE CHECKP(NVAR,P,VARB)
  !-----------------------------------------------------------------------
  !     ---- to impose any shake constraints on the vector of search
  !     ---- direction involved in conj. and powell minimizations
  !
#if KEY_CHEQ==1
  use cheq, only: cgtmp, cgfix  
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln             
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use contrl
  use coord
  use deriv
  use image
  use shake
  use psf
  use holonom,only:holonomf
  use icfix
  use energym
  use fourdm
#if KEY_FLUCQ==1
  use flucq
#endif 
  use memory
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals
#endif
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: PX
  real(chm_real),allocatable,dimension(:) :: PY
  real(chm_real),allocatable,dimension(:) :: PZ
  INTEGER NVAR
  real(chm_real) P(*), VARB(*)
#if KEY_DHDGB==1
!AP/MF
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
  !
  !
  LOGICAL QOK
  !
  IF (.NOT.QHOLO) RETURN
  !
  call chmalloc('egrad1.src','CHECKP','PX',NATOM,crl=PX)
  call chmalloc('egrad1.src','CHECKP','PY',NATOM,crl=PY)
  call chmalloc('egrad1.src','CHECKP','PZ',NATOM,crl=PZ)
  !
  CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       .FALSE.,CG,CGFIX,CGTMP, & 
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,FDIM,IMOVE4 &        
#endif
#if KEY_DHDGB==1
      ,.FALSE.,DUM_FHDGB,TOTALS &
#endif
       )
  CALL PUTVR1(MINXYZ,NATOM,P,IMOVE,PX,PY,PZ, &
#if KEY_CHEQ==1
       .FALSE.,CG,CGFIX,CGTMP, & 
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.FALSE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,DFDIM,IMOVE4 &       
#endif
#if KEY_DHDGB==1
      ,.FALSE.,DUM_FHDGB,TOTALS &
#endif
       )
  CALL HOLONOMF(PX,PY,PZ,X,Y,Z, &
       .FALSE.,.FALSE.,QOK)
  CALL GETVR1(MINXYZ,NATOM,P,IMOVE,PX,PY,PZ, &
       LMINUC, &
#if KEY_CHEQ==1
       .FALSE.,CG, &             
#endif
       XTLTYP,XTLABC,XTLREF, &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &        
#endif
#if KEY_FOURD==0
       .FALSE.,(/ZERO/),(/0/) &  
#endif
#if KEY_FOURD==1
       DIM4,DFDIM,IMOVE4 &       
#endif
#if KEY_DHDGB==1
      ,.FALSE., (/ZERO/),TOTALS &
#endif
       )
  call chmdealloc('egrad1.src','CHECKP','PX',NATOM,crl=PX)
  call chmdealloc('egrad1.src','CHECKP','PY',NATOM,crl=PY)
  call chmdealloc('egrad1.src','CHECKP','PZ',NATOM,crl=PZ)
  !
  RETURN
END SUBROUTINE CHECKP

SUBROUTINE CALCNVAR(QTSM,BACKLS,NVAR)
  !
  ! Calculate NVAR with possible TSM PiggyBack atoms.
  ! Stephen Fleischman 1/92
  !
#if KEY_FLUCQ==1
  use flucqm, only: fqseln             
#endif
  use chm_kinds
  use dimens_fcm
  use exfunc
  use contrl
  use psf
  use fourdm
  use image
#if KEY_FLUCQ==1
  use flucq
#endif 
  implicit none
  !
  LOGICAL QTSM
  INTEGER BACKLS(*),NVAR
  !
  !
  INTEGER I
  !
  NVAR = 0
  IF(MINXYZ) THEN
     IF(QTSM) THEN
        DO I = 1,NATOM
           IF(IMOVE(I) == 0 .AND. BACKLS(I) == 0) NVAR = NVAR + 3
        ENDDO
     ELSE
        DO I = 1,NATOM
           IF(IMOVE(I)  ==  0) NVAR = NVAR + 3
        ENDDO
     ENDIF
#if KEY_FOURD==1
     IF(DIM4) THEN
        IF(QTSM) CALL WRNDIE(-4,'<CALCNVAR>', &
             'TSM and FOURD are incompatible')
        DO I = 1,NATOM
           IF (IMOVE4(I)  ==  0) NVAR = NVAR + 1
        ENDDO
     ENDIF
#endif 
#if KEY_FLUCQ==1
     IF(QFLUC) THEN
        DO I = 1,NATOM
           IF (FQSELN(I)  >  0) NVAR = NVAR + 1
        ENDDO
     ENDIF
#endif 
  ENDIF
  !
  IF (LMINUC) THEN
     CALL XTLSYM(XTLABC,XUCELL,XTLTYP,XDIM,XTLREF)
     NVAR = NVAR + XDIM
  ENDIF
  !
  IF(NVAR  <=  0) THEN
     CALL WRNDIE(0,'<CALCNVAR>','No variables to be optimized.')
     RETURN
  ENDIF
  !
  RETURN
END SUBROUTINE CALCNVAR

SUBROUTINE DNUMER(DX,DY,DZ)
  !-----------------------------------------------------------------------
  !     The forces are calculated numerically by calling testfd
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use coord
  use number
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: IDXFD
  real(chm_real),allocatable,dimension(:) :: IDXAD
  real(chm_real),allocatable,dimension(:) :: IXNEW
  real(chm_real),allocatable,dimension(:) :: IYNEW
  real(chm_real),allocatable,dimension(:) :: IZNEW
  integer,allocatable,dimension(:) :: ISLCT
  real(chm_real) DX(*),DY(*),DZ(*)
  !
  INTEGER LENV
  INTEGER I,J,LENC
  CHARACTER(len=24) COMM
  !
  DX(1:NATOM) = ZERO
  DY(1:NATOM) = ZERO
  DZ(1:NATOM) = ZERO
  !
  LENC=22
  COMM='STEP 0.001 TOL 99999.9'
  LENV=3*NATOM
  call chmalloc('egrad1.src','DNUMER','IDXFD',LENV,crl=IDXFD)
  call chmalloc('egrad1.src','DNUMER','IDXAD',LENV,crl=IDXAD)
  call chmalloc('egrad1.src','DNUMER','IXNEW',NATOM,crl=IXNEW)
  call chmalloc('egrad1.src','DNUMER','IYNEW',NATOM,crl=IYNEW)
  call chmalloc('egrad1.src','DNUMER','IZNEW',NATOM,crl=IZNEW)
  call chmalloc('egrad1.src','DNUMER','ISLCT',NATOM,intg=ISLCT)
  CALL TESTFD(COMM,LENC, &
       NATOM,X,Y,Z,WMAIN, &
       IXNEW,IYNEW,IZNEW, &
       .FALSE.,[ZERO],[ZERO],[ZERO], &
       IDXFD,IDXAD, &
       ISLCT,IMOVE,.FALSE.,.FALSE.)

  CALL R3COPY(NATOM,DX,DY,DZ,IDXAD)
  !
  call chmdealloc('egrad1.src','DNUMER','IDXFD',LENV,crl=IDXFD)
  call chmdealloc('egrad1.src','DNUMER','IDXAD',LENV,crl=IDXAD)
  call chmdealloc('egrad1.src','DNUMER','IXNEW',NATOM,crl=IXNEW)
  call chmdealloc('egrad1.src','DNUMER','IYNEW',NATOM,crl=IYNEW)
  call chmdealloc('egrad1.src','DNUMER','IZNEW',NATOM,crl=IZNEW)
  call chmdealloc('egrad1.src','DNUMER','ISLCT',NATOM,intg=ISLCT)
  !
  RETURN
END SUBROUTINE DNUMER

SUBROUTINE TESTFD(COMLYN,COMLEN,NATOM,X,Y,Z,WMAIN, &
     XNEW,YNEW,ZNEW, &
     LDIM4,FDIM,FDNEW,DFDIM, &
     DXAD,DXFD,ISLCT,IMOVE,LCRYS,LHOMO)
  !
  !     Routine to test analytical first derivatives in CHARMM by
  !     finite differences along all cartesian dimensions of all
  !     selected atoms.
  !     ENERgy should be called before invoking this routine.
  !
  !     TEST FIRSt [TOL <real>] [STEP <real>] [UNIT <int>] [MASS <int>]
  !                <selection-spec.>
  !
  !     Bernard R. Brooks and A. Brunger, 22-APR-83
  !
  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use energym
  use eutil
  use image
  use eimg, only: lattrn
  use select
  use stream
  use string
  use chutil, only: atomid
  use param_store, only: set_param

  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,NATOM
  real(chm_real) :: X(:), Y(:), Z(:), WMAIN(*)
  real(chm_real) :: XNEW(:), YNEW(:), ZNEW(:)
  real(chm_real) FDIM(*),FDNEW(*),DFDIM(*)
  real(chm_real) DXAD(*), DXFD(*), SF(6)
  INTEGER ISLCT(*),IMOVE(*)
  LOGICAL LCRYS, LHOMO, LDIM4
  !
  INTEGER   IUNIT, IAT, IAT3, I, NOK, MASSW, NDIM
  real(chm_real) STEP,TOL1
  real(chm_real) CR(4),DEL,VALMIN,DIF,VAL
  LOGICAL QPRINT, OK
  CHARACTER(len=8) SID,RID,RESN,AC
  real(chm_real) XTLINV(6),WRK1(3,3),XTLSAV(6)
  !
  CHARACTER(len=2) :: SNAME(4)=(/' X',' Y',' Z','FD'/)
  !
  STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
  TOL1=GTRMF(COMLYN,COMLEN,'TOL',PT0001)
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  MASSW=GTRMI(COMLYN,COMLEN,'MASS',0)
  !
  QPRINT=(IUNIT.EQ.OUTU .AND. PRNLEV.GE.2) .OR. &
       (IUNIT.NE.OUTU .AND. IOLEV.GT.0)
  !
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  CALL GETE(X,Y,Z,X,Y,Z,MASSW)
  VALMIN=1.0
  DEL=STEP
  NDIM=3
  IF(LDIM4) NDIM=4
  !
  IAT3=0
  DO IAT=1,NATOM
     XNEW(IAT)=X(IAT)
     YNEW(IAT)=Y(IAT)
     ZNEW(IAT)=Z(IAT)
     IF(LDIM4) FDNEW(IAT)=FDIM(IAT)
     IF (ISLCT(IAT).EQ.1 .AND. IMOVE(IAT).EQ.0) THEN
        DXAD(IAT3+1)=DX(IAT)
        DXAD(IAT3+2)=DY(IAT)
        DXAD(IAT3+3)=DZ(IAT)
        IF(LDIM4) THEN
           DXAD(IAT3+4)=DFDIM(IAT)
           IAT3=IAT3+1
        ENDIF
        IAT3=IAT3+3
     ENDIF
  ENDDO
  IF(LCRYS) THEN
     IF(LHOMO) THEN
        CALL INVT33S(XTLINV,XTLABC,OK)
        CALL MULNXNFL(WRK1,EPRESS(VIXX:VIZZ),XTLINV,3)
        CALL LATTRN(XTLTYP,WRK1,XTLSAV,XTLREF)
        DO IAT=1,XDIM
           DXAD(IAT3+1)=-XTLSAV(IAT)
           IAT3=IAT3+1
        ENDDO
     ELSE
        DO IAT=1,XDIM
           DXAD(IAT3+1)=DXTL(IAT)
           IAT3=IAT3+1
        ENDDO
     ENDIF
  ENDIF
  !
  IAT3=0
  DO IAT=1,NATOM
     IF (ISLCT(IAT).EQ.1 .AND. IMOVE(IAT).EQ.0) THEN
        CR(1)=XNEW(IAT)
        CR(2)=YNEW(IAT)
        CR(3)=ZNEW(IAT)
        IF(LDIM4) CR(4)=FDNEW(IAT)
        DO I=1,NDIM
           IAT3=IAT3+1
           !
           ! make finite diff approx in (IAT,I) direction (I: X<>1, Y<>2, Z<>3)
           !
           CR(I)=CR(I)+DEL
           X(IAT)=CR(1)
           Y(IAT)=CR(2)
           Z(IAT)=CR(3)
           IF(LDIM4) FDIM(IAT)=CR(4)
           !
           CALL GETE(X,Y,Z,XNEW,YNEW,ZNEW,MASSW)
           DXFD(IAT3)=EPROP(EPOT)
           !
           CR(I)=CR(I)-2.0*DEL
           X(IAT)=CR(1)
           Y(IAT)=CR(2)
           Z(IAT)=CR(3)
           IF(LDIM4) FDIM(IAT)=CR(4)
           !
           CALL GETE(X,Y,Z,XNEW,YNEW,ZNEW,MASSW)
           !
           DXFD(IAT3)=(DXFD(IAT3)-EPROP(EPOT))/(DEL+DEL)
           CR(I)=CR(I)+DEL
           X(IAT)=XNEW(IAT)
           Y(IAT)=YNEW(IAT)
           Z(IAT)=ZNEW(IAT)
           IF(LDIM4) FDIM(IAT)=FDNEW(IAT)
           !
        ENDDO
     ENDIF
  ENDDO

  IF(LCRYS) THEN
     IF(LHOMO) CALL INVT33S(XTLINV,XTLABC,OK)
     DO IAT=1,XDIM
        xtlsav(1:6) = XTLABC(1:6)
        CALL GETXTL(SF,XTLABC,XTLTYP,XTLREF)
        IAT3=IAT3+1
        !
        ! make finite diff approx in crystal size
        !
        SF(IAT)=SF(IAT)+DEL
        CALL PUTXTL(SF,XTLABC,XTLTYP,XTLREF)
        !
        IF(PRNLEV.GE.7) CALL PRNXTLD(OUTU,'TFD+',XTLTYP,XUCELL, &
             .FALSE.,ZERO,.FALSE.,[ZERO])
        !
        IF(LHOMO) THEN
           CALL MULNXNLL(WRK1,XTLABC,XTLINV,3)
           DO I=1,NATOM
              X(I) = WRK1(1,1)*XNEW(I) + &
                   WRK1(1,2)*YNEW(I) + WRK1(1,3)*ZNEW(I)
              Y(I) = WRK1(2,1)*XNEW(I) + &
                   WRK1(2,2)*YNEW(I) + WRK1(2,3)*ZNEW(I)
              Z(I) = WRK1(3,1)*XNEW(I) + &
                   WRK1(3,2)*YNEW(I) + WRK1(3,3)*ZNEW(I)
           ENDDO
        ENDIF
        !
        CALL GETE(X,Y,Z,XNEW,YNEW,ZNEW,MASSW)
        DXFD(IAT3)=EPROP(EPOT)
        !
        SF(IAT)=SF(IAT)-TWO*DEL
        CALL PUTXTL(SF,XTLABC,XTLTYP,XTLREF)
        !
        IF(PRNLEV.GE.7) CALL PRNXTLD(OUTU,'TFD-',XTLTYP,XUCELL, &
             .FALSE.,ZERO,.FALSE.,[ZERO])
        !
        IF(LHOMO) THEN
           CALL MULNXNLL(WRK1,XTLABC,XTLINV,3)
           DO I=1,NATOM
              X(I) = WRK1(1,1)*XNEW(I) + &
                   WRK1(1,2)*YNEW(I) + WRK1(1,3)*ZNEW(I)
              Y(I) = WRK1(2,1)*XNEW(I) + &
                   WRK1(2,2)*YNEW(I) + WRK1(2,3)*ZNEW(I)
              Z(I) = WRK1(3,1)*XNEW(I) + &
                   WRK1(3,2)*YNEW(I) + WRK1(3,3)*ZNEW(I)
           ENDDO
        ENDIF
        !
        CALL GETE(X,Y,Z,XNEW,YNEW,ZNEW,MASSW)
        !
        DXFD(IAT3)=(DXFD(IAT3)-EPROP(EPOT))/(DEL+DEL)
        SF(IAT)=SF(IAT)+DEL
        CALL PUTXTL(SF,XTLABC,XTLTYP,XTLREF)
        !
        IF(LHOMO) THEN
           DO I=1,NATOM
              X(I) = XNEW(I)
              Y(I) = YNEW(I)
              Z(I) = ZNEW(I)
           ENDDO
        ENDIF
        !
        xtlabc(1:6) = XTLSAV(1:6)
     ENDDO
  ENDIF
  !
  !
  IF(QPRINT) WRITE(IUNIT,23) STEP, MASSW, TOL1
23 FORMAT(' TESTFD: Parameters: STEP=',F10.5,'  MASSweighting=',I4, &
       /,' TESTFD: The following first derivatives', &
       ' differ by more than TOL=',F12.6,//, &
       '  DIM.',9X,'ATOM',17X,'ANALYTIC',8X,'FINITE-DIFF',5X, &
       'DEVIATION')
  NOK=0
  IAT3=0
  DO IAT=1,NATOM
     IF (ISLCT(IAT).EQ.1 .AND. IMOVE(IAT).EQ.0) THEN
        CALL ATOMID(IAT,SID,RID,RESN,AC)
        DO I=1,NDIM
           IAT3=IAT3+1
           DIF=DXAD(IAT3)-DXFD(IAT3)
           VAL=ABS(DXAD(IAT3))
           IF(VAL.LT.VALMIN) VAL=VALMIN
           IF(ABS(DIF/VAL).GE.TOL1) THEN
              IF(QPRINT) WRITE(IUNIT,28) IAT,SNAME(I), &
                   SID(1:idleng),RID(1:idleng),RESN(1:idleng), &
                   AC(1:idleng),DXAD(IAT3),DXFD(IAT3),DIF
28            FORMAT(1X,I4,A2,' (',4(1X,A),')',3F16.8)
           ELSE
              NOK=NOK+1
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  IF(LCRYS) THEN
     DO IAT=1,XDIM
        IAT3=IAT3+1
        DIF=DXAD(IAT3)-DXFD(IAT3)
        VAL=ABS(DXAD(IAT3))
        IF(VAL.LT.VALMIN) VAL=VALMIN
        IF(ABS(DIF/VAL).GE.TOL1) THEN
           IF(QPRINT) WRITE(IUNIT,29) IAT,XTLTYP, &
                DXAD(IAT3),DXFD(IAT3),DIF
29         FORMAT(1X,I4,2X,A4,' crystal dimension ',3F16.8)
        ELSE
           NOK=NOK+1
        ENDIF
     ENDDO
  ENDIF
  CALL set_param('NOK', NOK)
  IF(QPRINT) WRITE(IUNIT,36) NOK
36 FORMAT(/ &
       ' TESTFD: A total of',I6,' elements were within the tolerance')
  RETURN
END SUBROUTINE TESTFD

SUBROUTINE R3COPY(N,X,Y,Z,SRC)
  use chm_kinds
  implicit none
  real(chm_real) X(*),Y(*),Z(*),SRC(*)
  INTEGER N,I,J
  J=1
  DO I=1,N
     X(I)=SRC(J)
     Y(I)=SRC(J+1)
     Z(I)=SRC(J+2)
     J=J+3
  ENDDO
  RETURN
END SUBROUTINE R3COPY



SUBROUTINE ENEOUT(OUTUX,NCYCLE,STEP,DUMMY)
  !-----------------------------------------------------------------------
  !     ---- output routine called from minimizers
  !
  use chm_kinds
  use dimens_fcm
  use number
  use vector
  use image
  use energym
  use contrl
  use stream
  use parallel
  use repdstr
  implicit none
  !
  INTEGER OUTUX,NCYCLE
  real(chm_real) STEP,DUMMY
  !
  !
  real(chm_real) EPREV,XGNORM
  LOGICAL QHDR
  SAVE EPREV
  DATA EPREV /ZERO/
  !
  IF (NPRINT  >  0 .AND. NCYCLE  >=  0) THEN
     IF (MOD(NCYCLE,NPRINT)  ==  0) THEN
        QHDR = NCYCLE == 0
#if KEY_REPDSTR==1
        IF(QREPDSTR)THEN
           CALL PRINTE(OUTUX, EPROP, ETERM, 'MINI', 'MIN', QHDR, &
                NCYCLE, ZERO, STEP, .FALSE.)
        ELSE
#endif 
           CALL PRINTE(OUTUX, EPROP, ETERM, 'MINI', 'MIN', QHDR, &
                NCYCLE, ZERO, STEP, .TRUE.)
#if KEY_REPDSTR==1
        ENDIF                               
#endif
        IF (LMINUC .AND. PRNLEV >= 2) THEN
           CALL XTLLAT(XUCELL,XTLABC)
           XGNORM=SQRT(DOTVEC(DXTL,DXTL,XDIM) / XDIM)
           CALL PRNXTLD(OUTUX,'MINI',XTLTYP,XUCELL,.TRUE.,XGNORM, &
                .FALSE.,[ZERO])
        ENDIF
        !MM       EPROP(PJNK1)=NCYCLE
        !MM       EPROP(PJNK2)=STEP
        !MM       EPROP(PJNK3)=EPROP(EPOT)-EPREV
        EPREV=EPROP(EPOT)
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE ENEOUT

#if KEY_TNPACK==1 /*egrad2*/
SUBROUTINE EGRAD2(NVAR,VARB,FUNC,GRAD,QLOCAL,ISTEP,ICALL,ERSTAT)
  !-----------------------------------------------------------------------
  !     The energy and forces and Hessian are calculated and returned.
  !     This routine is re-introduced for TNPACK.
  !     Youngdo Won, July 28, 1995
  !
#if KEY_CHEQ==1
  use cheq,only:cgtmp,cgfix,   &                  
#endif
#if KEY_CHEQ==1
     DCH,SUMDCH,DDCH                        
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqseln,fqcfor        
#endif
  use chm_kinds
  use dimens_fcm
  use exfunc
     !
  use bases_fcm
  use coord
  use contrl
  use energym
  use deriv
  use image
  use parallel,only: gcomb
  use psf
  use shake
  use euler
#if KEY_FLUCQ==1
  use flucq     
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:totals
#endif
  use memory
  use number
  use heurist,only:updeci
#if KEY_DHDGB==1
  use dhdgb,only:totals
#endif
  implicit none
  !
  !     Passed variables.
  !
  real(chm_real),allocatable,dimension(:) :: DTX
  INTEGER ISTEP, ICALL, NVAR, ERSTAT
  real(chm_real)  FUNC, GRAD(*), VARB(*)
  LOGICAL QLOCAL
  !
  !     Local variables.
  !
  INTEGER  I, IVAR, NTOT, NAT3
  integer,parameter :: IMV40(1) = (/ 0 /)
  !
  real(chm_real)  ETWO
  real(chm_real),parameter :: FDIM0(1) = (/ ZERO /)
  real(chm_real),parameter :: XTL0(6) = (/ ZERO, ZERO, ZERO, ZERO, ZERO, ZERO /)
  LOGICAL QESAVE(LENENT)
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
  !
  !     Pointers.
  !
  INTEGER  X1, Y1, Z1
  !
  IF(QHOLO) THEN
     CALL WRNDIE(-2,'<EGRAD2>', &
          'SHAKE or constraints not compatible with Hessian.')
     ERSTAT=10
     RETURN
  ENDIF
  !
  IF(LMINUC) THEN
     CALL WRNDIE(-2,'<EGRAD2>', &
          'Hessian code not available for unit cell changes.')
     ERSTAT=11
     RETURN
  ENDIF
  !
  IF(QSADLE) THEN
     CALL WRNDIE(-2,'<EGRAD2>', &
          'Hessian code not available for SADDle option.')
     ERSTAT=12
     RETURN
  ENDIF
  !
  !     Do some initialisation.
  !
  !     Fill the coordinate arrays with the variables.
  !
  CALL PUTVR1(MINXYZ,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       .FALSE.,CG,CGFIX,CGTMP,    & 
#endif
       LMINUC,XTLTYP,XTLABC,XTLREF,.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN,                      & 
#endif
       .FALSE.,FDIM0,IMV40  &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
  )
  !
  !     Do list updates
  CALL UPDECI(ISTEP,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
  !
  IF(QLOCAL) THEN
     DO I=1,LENENT
        QESAVE(I)=QETERM(I)
        QETERM(I)=.FALSE.
     ENDDO
     IF(QEULER) THEN
        QETERM(BOND)=.TRUE.
        QETERM(ANGLE)=.TRUE.
        QETERM(DIHE)=.FALSE.
        QETERM(IMDIHE)=.FALSE.
#if KEY_CMAP==1
        QETERM(CMAP)=.FALSE.
#endif 
        QETERM(ELEC)=.FALSE.
        QETERM(VDW)=.FALSE.
        QETERM(EHARM)=.TRUE.    !! EULER M.D.
     ELSE
        QETERM(BOND)=.TRUE.
        QETERM(ANGLE)=.TRUE.
        QETERM(DIHE)=.TRUE.
        QETERM(IMDIHE)=.TRUE.
#if KEY_CMAP==1
        QETERM(CMAP)=.TRUE.
#endif 
        QETERM(ELEC)=.TRUE.
        QETERM(VDW)=.TRUE.
     ENDIF
  ENDIF
  !
  !     Calculate the energy, forces and Hessian.
  NAT3=NATOM*3
  call chmalloc('egrad1.src','EGRAD2','DTX',NAT3,crl=DTX)
  NTOT=(NAT3*(NAT3+1))/2
  if(.not. allocated(iddf)) call chmalloc('egrad1.src','EGRAD2','IDDF',NTOT,crl=IDDF)
  iddf = zero
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,ICALL,NAT3,IDDF)
#if KEY_PARALLEL==1
  CALL VDGBR(DX,DY,DZ,1)
  CALL GCOMB(IDDF,NTOT)
#endif 
  CALL GETVR1(MINXYZ,NATOM,DTX,IMOVE,DX,DY,DZ,.FALSE., &
#if KEY_CHEQ==1
       QCGMIN,CG,                           & 
#endif
       'NONE',XTL0,XTL0, &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,FQSELN,                 & 
#endif
       .FALSE.,FDIM0,IMV40                  &
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE.,(/ZERO/),TOTALS &
#endif
        )
  !
  FUNC = EPROP(EPOT)
  CALL GETVR1(MINXYZ,NATOM,GRAD,IMOVE,DX,DY,DZ,.FALSE., &
#if KEY_CHEQ==1
       QCGMIN,CG,                            & 
#endif
       'NONE',XTL0,XTL0, &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,FQSELN,                 & 
#endif
       .FALSE.,FDIM0,IMV40                  &
#if KEY_DHDGB==1
!AP/MF
        ,.FALSE.,(/ZERO/),TOTALS &
#endif
        )
  !
  IF(QLOCAL) THEN
        QETERM(1:lenent)=QESAVE(1:lenent)
  ENDIF
  !
  !     Free storage space.
  !
  call chmdealloc('egrad1.src','EGRAD2','DTX',NAT3,crl=DTX)
  !
  RETURN
END SUBROUTINE EGRAD2
#endif /* (egrad2)*/

end module egrad

