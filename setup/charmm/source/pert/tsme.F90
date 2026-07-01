#if KEY_TSM==1
!
! Energy calculating parts of perturbation module
!
SUBROUTINE TSME(X,Y,Z,DX,DY,DZ,QFIRST)

  !     Author: Stephen Fleischman
  !
  use ewald,only: lewald
  use flucqm,only:fqcfor

  ! START OF DECLARATIONS USING INCLUDE STATEMENTS
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use bases_fcm
  use energym
  use inbnd
  use image
  use fast
  use memory
  use psf
  use tsmh
  use tsms_mod
  use timerm
  use stream
#if KEY_FLUCQ==1
  use flucq     
#endif
  use datstr,only:dupldt_nbond,freedt_nbond
  implicit none
  ! END OF DECLARATIONS USING INCLUDE STATEMENTS
  real(chm_real),allocatable,dimension(:) :: colost
  real(chm_real),allocatable,dimension(:) :: olddx, olddy, olddz
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  !     Qfirst is true for the primary atom calculation and false for the
  !     image atom calculation. Since initialization is done when it is true
  !     this must go first. Even if NTRANS is zero, the routine must be called
  !     a second time to free memory.
  LOGICAL QFIRST
  INTEGER NNB14X
  INTEGER,PARAMETER  :: COLO_FREE=1,COLO_FILL=2
  type(chm_iptr),save :: ICOLO
  !
  integer, dimension(1), target :: pzer1 = [0]
  integer, dimension(:), pointer :: &
          TCINBLO,  TCJNB, &
          TCINBLOG, TRINBLO, TRJNB, &
          TPINBLO,  TPJNB,   TINBLO, &
          TJNB,     TRINBLOG,TPINBLOG, &
          TIBLO14,  TINB14, &
          TRIBLO14, TRINB14, &
          TPIBLO14, TPINB14
  !
  !     This is for a possibly weird use of the SKIPE command.
  !     During a normal TSM run the QTSM flag would be tested in the calling
  !     routine.
  IF(.NOT.QETERM(TSM)) RETURN
  !     protect the original value.
  NNB14X=NNB14
  !
  !     Error trapping - Ewald cannot be used with certain TSM options.
  IF(QETERM(ELEC).AND.LEWALD) THEN
     IF(NCOLO > 0) THEN
        CALL WRNDIE(-5,'<TSME>', &
             'The TSM COLOcate option cannot be used with Ewald Summation.')
     ELSEIF(LPOWER /= 1) THEN
        CALL WRNDIE(-5,'<TSME>', &
             'Only Linear lambda scaling is allowed when using TSM with '// &
             'Ewald summation.')
     ENDIF
  ENDIF

  IF (QFIRST) THEN
     !        using data structure routines for BNBNDC, ATB, 16-APR-85 :
     IF(UPPERT) THEN
        CALL MAKPDT(BNBNDR,BNBND,NRCTAT)
        CALL MAKPDT(BNBNDP,BNBND,NPRDAT)
     END IF
     !
     IF (NCOLO > 0) THEN
        CALL FREEDT_nbond(BNBNDC)
        CALL DUPLDT_nbond(BNBNDC,BNBND)
        call chmalloc('tsme.src','TSME','COLOST',NCOLO,crl=COLOST)
     ELSE
        call chmalloc('tsme.src','TSME','COLOST',1,crl=COLOST)
        CALL MKDUMC(BNBNDC)
     END IF
     CALL COLOEXT(ACOLO,NCOLO,ICOLO,COLO_FILL)
  ENDIF
  !      !get memory to save forces.
  IF(QFIRST) THEN
     call chmalloc('tsme.src','TSME','OLDDX',NATOM,crl=OLDDX)
     call chmalloc('tsme.src','TSME','OLDDY',NATOM,crl=OLDDY)
     call chmalloc('tsme.src','TSME','OLDDZ',NATOM,crl=OLDDZ)
  ELSE IF(NTRANS > 0.AND.NATIM > 0) THEN
     call chmalloc('tsme.src','TSME','OLDDX',NATIM,crl=OLDDX)
     call chmalloc('tsme.src','TSME','OLDDY',NATIM,crl=OLDDY)
     call chmalloc('tsme.src','TSME','OLDDZ',NATIM,crl=OLDDZ)
  ENDIF

  IF(QFIRST) THEN
     !        primary atom forces.
     !
     ! should be separated but maybe OK all at once ???
     if(associated(bnbnd%jnb)) then
        TCINBLO =>   BNBNDC%INBLO
        TCJNB =>     BNBNDC%JNB
        TCINBLOG =>  BNBNDC%INBLOG
        TRINBLO =>   BNBNDR%INBLO
        TRJNB =>     BNBNDR%JNB
        TPINBLO =>   BNBNDP%INBLO
        TPJNB =>     BNBNDP%JNB
        TINBLO =>    BNBND%INBLO
        TJNB =>      BNBND%JNB
        TRINBLOG =>  BNBNDR%INBLOG
        TPINBLOG =>  BNBNDP%INBLOG 
        TIBLO14 =>   BNBND%IBLO14
        TINB14 =>    BNBND%INB14
        TRIBLO14 =>  BNBNDR%IBLO14
        TRINB14 =>   BNBNDR%INB14
        TPIBLO14 =>  BNBNDP%IBLO14
        TPINB14 =>   BNBNDP%INB14
     else
        TCINBLO =>   pzer1
        TCJNB =>     pzer1
        TCINBLOG =>  pzer1
        TRINBLO =>   pzer1
        TRJNB =>     pzer1
        TPINBLO =>   pzer1
        TPJNB =>     pzer1
        TINBLO =>    pzer1
        TJNB =>      pzer1
        TRINBLOG =>  pzer1
        TPINBLOG =>  pzer1 
        TIBLO14 =>   pzer1
        TINB14 =>    pzer1
        TRIBLO14 =>  pzer1
        TRINB14 =>   pzer1
        TPIBLO14 =>  pzer1
        TPINB14 =>   pzer1
     endif
     
     CALL PENER2A(OLDDX,OLDDY,OLDDZ,DX,DY,DZ, &
          X,Y,Z,TCINBLO,TCJNB, &
          TCINBLOG, TRINBLO,TRJNB, &
          TPINBLO,TPJNB,TINBLO, &
          TJNB,TRINBLOG,TPINBLOG, &
          NNB14X, &
          TIBLO14,TINB14, &
          TRIBLO14,TRINB14, &
          TPIBLO14,TPINB14, &
          REACLS,PRODLS,BSKIPR, &
          BSKIPP,ASKIPR,ASKIPP, &
          UBSKIPR,UBSKIPP, &
          PSKIPR,PSKIPP,ISKIPR, &
          ISKIPP, &
#if KEY_CMAP==1
          CTSKIPR,CTSKIPP, & 
#endif
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,              & 
#endif
          COLOST,ICOLO%A)
     !***end of clbiii mod for ub inclusion
  ELSE
     !        image forces.
     CALL PENER2B(OLDDX,OLDDY,OLDDZ, &
          DX,DY,DZ,X,Y,Z, &
          REACLS,PRODLS, &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,              & 
#endif
          COLOST,ICOLO%A)
  ENDIF
  !     free up some memory.
  IF(.NOT.QFIRST) THEN
     CALL COLOEXT(ACOLO,NCOLO,ICOLO,COLO_FREE)
     CALL FREEDT_nbond(BNBNDC)
     !         IF (NCOLO > 0) THEN
     !             call chmdealloc('tsme.src','TSME','COLOST',NCOLO,crl=COLOST)
     !         ELSE
     !             call chmdealloc('tsme.src','TSME','COLOST',1,crl=COLOST)
     !         END IF
     IF(UPPERT) UPPERT=.FALSE.
  ELSE
     IF (NCOLO > 0) THEN
        call chmdealloc('tsme.src', 'TSME', 'COLOST', ncolo, crl=colost)
     ELSE
        call chmdealloc('tsme.src', 'TSME', 'COLOST', 1, crl=COLOST)
     ENDIF
  ENDIF
  IF(QFIRST) THEN
     call chmdealloc('tsme.src','TSME','OLDDX',NATOM,crl=OLDDX)
     call chmdealloc('tsme.src','TSME','OLDDY',NATOM,crl=OLDDY)
     call chmdealloc('tsme.src','TSME','OLDDZ',NATOM,crl=OLDDZ)
  ELSE IF(NTRANS > 0.AND.NATIM > 0) THEN
     call chmdealloc('tsme.src','TSME','OLDDX',NATIM,crl=OLDDX)
     call chmdealloc('tsme.src','TSME','OLDDY',NATIM,crl=OLDDY)
     call chmdealloc('tsme.src','TSME','OLDDZ',NATIM,crl=OLDDZ)
  ENDIF
  RETURN
END SUBROUTINE TSME

SUBROUTINE PENER2A(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,X,Y,Z, &
     INBLO,JNB,INBLOG, &
     INBLOR,JNBR,INBLOP,JNBP, &
     INBLOX,JNBX,INBLGR,INBLGP,NNB14X, &
     IBLO14X,INB14X, &
     IBLO14R,INB14R, &
     IBLO14P,INB14P, &
     REACLS,PRODLS,BSKIPR,BSKIPP,ASKIPR,ASKIPP, &
                                !*** clbiii mod for ub
     UBSKIPR, UBSKIPP, &
                                !***end of clbiii mod for ub
     PSKIPR,PSKIPP,ISKIPR,ISKIPP, &
#if KEY_CMAP==1
     CTSKIPR,CTSKIPP, &   
#endif
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,   & 
#endif
     COLOST,ICOLO)

  ! START OF DECLARATIONS USING INCLUDE STATEMENTS
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
  use code
  use param
  use eintern
  use enbond_mod
  use energym
  use hbondm
  use image
  use number
  use psf
  use fast
  use memory
  use stream
  use tsms_mod
  use timerm
  use cmapm
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif 
  implicit none
  ! END OF DECLARATIONS USING INCLUDE STATEMENTS
  !
  integer,allocatable,dimension(:) :: hskip
  INTEGER INBLO(*),INBLOX(*),INBLOR(*),INBLOP(*)
  INTEGER INBLOG(*),INBLGR(*),INBLGP(*)
  INTEGER JNB(*),JNBX(*),JNBR(*),JNBP(*)
  INTEGER I,ICOLO(*)
  real(chm_real) COLOST(*)
  INTEGER REACLS(*),PRODLS(*),BSKIPR(*),BSKIPP(*),ASKIPR(*), &
       ASKIPP(*),PSKIPR(*),PSKIPP(*),ISKIPR(*),ISKIPP(*)
#if KEY_CMAP==1
  INTEGER CTSKIPR(*),CTSKIPP(*)
#endif 
  !*** clbiii mod for ub
  INTEGER UBSKIPR(*), UBSKIPP(*)
  !*** clbiii mod for ub
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 

  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*)
  !     Ewald exclusion
  INTEGER IBLO14X(*),INB14X(*)
  INTEGER IBLO14R(*),INB14R(*)
  INTEGER IBLO14P(*),INB14P(*)
  real(chm_real) RLAMBDA,PLAMBDA
  real(chm_real) EST2
  INTEGER OLDFAST,OLDLFAST
  !     Ewald exclusion counters
  INTEGER NNB14X,NNB14R,NNB14P
  !
  INTEGER,PARAMETER :: COLO_FREE=1,COLO_FILL=2

#if KEY_DOMDEC==1
  if (q_domdec) then
     call wrndie(-5,'<tsme>','PENER2A not ready for DOMDEC')
  endif
#endif 

  !
  ! calculate product and reactant lambda factors for tsm energy terms
  !
  IF (LPOWER == 1) THEN
     RLAMBDA = LAMBDA
     PLAMBDA = ONE-LAMBDA
  ELSE
     RLAMBDA = ONE-(ONE-LAMBDA)**LPOWER
     PLAMBDA = ONE - LAMBDA**LPOWER
  END IF
  EST2 = ZERO
  VPRTTR = ZERO
  VPRTTP = ZERO
  VPRTNR = ZERO
  VPRTNP = ZERO
  VPRTVR = ZERO
  VPRTVP = ZERO
  VPRTER = ZERO
  VPRTEP = ZERO
  DO I = 1,LENENT
     TSMTRM(I) = ZERO
  ENDDO
  !     leave this in in case some part of the rest of the program uses it.
  ETERM(TSM) = ZERO
  !--   make copies of the forces
  OLDDX(1:natom) = DX(1:natom)
  OLDDY(1:natom) = DY(1:natom)
  OLDDZ(1:natom) = DZ(1:natom)
  !  bond terms
  IF(NBOND > 0.AND.QETERM(BOND)) THEN
     ! reactant
     IF ( (.NOT.DNTRB).OR.(DNTRB.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(BOND),DX,DY,DZ,NATOM)
        CALL EBOND(TSMTMP(BOND),IB,JB,ICB,NBOND,CBC,CBB, &
             DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),1,BSKIPR,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,REACLS, &
                NATOM,TSMTRM(BOND),TSMTMP(BOND))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(BOND)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(BOND),TSMTMP(BOND))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPB).OR.(DNTPB.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(BOND),DX,DY,DZ,NATOM)
        CALL EBOND(TSMTMP(BOND),IB,JB,ICB,NBOND,CBC,CBB, &
             DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),1,BSKIPP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(BOND),TSMTMP(BOND))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(BOND)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(BOND),TSMTMP(BOND))
        END IF
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(BOND) = -TSMTRM(BOND)
  END IF
  IF(NTHETA > 0.AND.QETERM(ANGLE)) THEN
     ! reactant
     IF ( (.NOT.DNTRA).OR.(DNTRA.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(ANGLE),DX,DY,DZ,NATOM)
        CALL EANGLE(TSMTMP(ANGLE),IT,JT,KT,ICT,NTHETA,CTC,CTB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),1,ASKIPR,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ, &
                REACLS,NATOM,TSMTRM(ANGLE),TSMTMP(ANGLE))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(ANGLE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(ANGLE),TSMTMP(ANGLE))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPA).OR.(DNTPA.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(ANGLE),DX,DY,DZ,NATOM)
        CALL EANGLE(TSMTMP(ANGLE),IT,JT,KT,ICT,NTHETA,CTC,CTB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),1,ASKIPP,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(ANGLE),TSMTMP(ANGLE))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(ANGLE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(ANGLE),TSMTMP(ANGLE))
        END IF
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(ANGLE) = -TSMTRM(ANGLE)
  END IF
  !*** clbiii mod for addition of ureyb term
  IF(NTHETA > 0.AND.QETERM(UREYB)) THEN
     ! reactant
     IF ( (.NOT.DNTRA).OR.(DNTRA.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(UREYB),DX,DY,DZ,NATOM)
        CALL EBOND(TSMTMP(UREYB),IT,KT,ICT,NTHETA,CTUC,CTUB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),1,UBSKIPR,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ, &
                REACLS,NATOM,TSMTRM(UREYB),TSMTMP(UREYB))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(UREYB)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(UREYB),TSMTMP(UREYB))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPA).OR.(DNTPA.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(UREYB),DX,DY,DZ,NATOM)
        CALL EBOND(TSMTMP(UREYB),IT,KT,ICT,NTHETA,CTUC,CTUB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),1,UBSKIPP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(UREYB),TSMTMP(UREYB))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(UREYB)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(UREYB),TSMTMP(UREYB))
        END IF
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(UREYB) = -TSMTRM(UREYB)
  END IF
  !*** clbiii mod for addition of ureyb term
  IF(NPHI > 0.AND.QETERM(DIHE)) THEN
     ! reactant
     IF ( (.NOT.DNTRP).OR.(DNTRP.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(DIHE),DX,DY,DZ,NATOM)
        CALL EPHI(TSMTMP(DIHE),IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB, &
             CPCOS,CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE., &
             (/ZERO/),1,PSKIPR,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,REACLS, &
                NATOM,TSMTRM(DIHE),TSMTMP(DIHE))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(DIHE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(DIHE),TSMTMP(DIHE))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPP).OR.(DNTPP.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(DIHE),DX,DY,DZ,NATOM)
        CALL EPHI(TSMTMP(DIHE),IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB, &
             CPCOS,CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE., &
             (/ZERO/),1,PSKIPP,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(DIHE),TSMTMP(DIHE))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(DIHE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(DIHE),TSMTMP(DIHE))
        END IF
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(DIHE) = -TSMTRM(DIHE)
  END IF
  IF(NIMPHI > 0.AND.QETERM(IMDIHE)) THEN
     ! reactant
     IF ( (.NOT.DNTRI).OR.(DNTRI.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(IMDIHE),DX,DY,DZ,NATOM)
        CALL EPHI(TSMTMP(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI, &
             CIC,CID,CIB,CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),1,ISKIPR,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,REACLS, &
                NATOM,TSMTRM(IMDIHE),TSMTMP(IMDIHE))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(IMDIHE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(IMDIHE),TSMTMP(IMDIHE))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPI).OR.(DNTPI.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(IMDIHE),DX,DY,DZ,NATOM)
        CALL EPHI(TSMTMP(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI, &
             CIC,CID,CIB,CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),1,ISKIPP,(/ZERO/),(/0/),.FALSE. &
             )

        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(IMDIHE),TSMTMP(IMDIHE))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(IMDIHE)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(IMDIHE),TSMTMP(IMDIHE))
        END IF
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(IMDIHE) = -TSMTRM(IMDIHE)
  END IF

#if KEY_CMAP==1
  IF(NCRTERM > 0.AND.QETERM(CMAP)) THEN
     ! reactant
     IF ( (.NOT.DNTRP).OR.(DNTRP.AND.SUBRCT) ) THEN
        CALL INISTF(TSMTMP(CMAP),DX,DY,DZ,NATOM)
        CALL ECMAP(TSMTMP(CMAP),I1CT,J1CT,K1CT,L1CT, &
             I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
             DX,DY,DZ,X,Y,Z, &
             .FALSE., (/ZERO/), 1, CTSKIPR, (/ZERO/), (/0/), .FALSE.)
        IF (SUBRCT) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,REACLS, &
                NATOM,TSMTRM(CMAP),TSMTMP(CMAP))
        ELSE
           VPRTTR = VPRTTR + TSMTMP(CMAP)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA, &
                NATOM,TSMTRM(CMAP),TSMTMP(CMAP))
        END IF
     END IF
     ! product
     IF ( (.NOT.DNTPP).OR.(DNTPP.AND.SUBPRD) ) THEN
        CALL INISTF(TSMTMP(CMAP),DX,DY,DZ,NATOM)
        CALL ECMAP(TSMTMP(CMAP),I1CT,J1CT,K1CT,L1CT, &
             I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
             DX,DY,DZ,X,Y,Z, &
             .FALSE., (/ZERO/), 1, CTSKIPP, (/ZERO/), (/0/), .FALSE.)
        IF (SUBPRD) THEN
           CALL DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PRODLS, &
                NATOM,TSMTRM(CMAP),TSMTMP(CMAP))
        ELSE
           VPRTTP = VPRTTP + TSMTMP(CMAP)
           CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA, &
                NATOM,TSMTRM(CMAP),TSMTMP(CMAP))
        END IF
     END IF
     !---   We are calculating how much to subtract out from the total potential
     !---   energy to get the correct E(lambda)
     TSMTRM(CMAP) = -TSMTRM(CMAP)
  END IF
#endif 

  !
  IF(NHB > 0.AND.QETERM(HBOND)) THEN
     call chmalloc('tsme.src','PENER2A','HSKIP',NHB,intg=HSKIP)
     CALL HBSEL(REACLS,IHB,JHB,HSKIP,1,NHB)
     CALL INISTF(TSMTMP(HBOND),DX,DY,DZ,NATOM)
     CALL EHBOND(TSMTMP(HBOND),IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,HSKIP,CTONHB, &
          CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
     CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA,NATOM, &
          TSMTRM(HBOND),TSMTMP(HBOND))
     VPRTTR = VPRTTR + TSMTMP(HBOND)

     CALL HBSEL(PRODLS,IHB,JHB,HSKIP,1,NHB)
     CALL INISTF(TSMTMP(HBOND),DX,DY,DZ,NATOM)
     CALL EHBOND(TSMTMP(HBOND),IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,HSKIP,CTONHB, &
          CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
     CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA,NATOM, &
          TSMTRM(HBOND),TSMTMP(HBOND))
     VPRTTP = VPRTTP+ TSMTMP(HBOND)
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(HBOND) = -TSMTRM(HBOND)
     call chmdealloc('tsme.src','PENER2A','HSKIP',NHB,intg=HSKIP)
  END IF
  IF(NATOM > 0) THEN
     ! make new nonbond lists
     IF(UPPERT) THEN
        CALL NNLST2(NATOM,NGRP,INBLOX,JNBX,INBLOR,JNBR,REACLS, &
             INBLGR)
        CALL EWEXLST(NATOM,REACLS,IBLO14X,INB14X,NNB14X, &
             IBLO14R,INB14R,NNB14R)
        CALL SETEWEX(NNB14X,NNB14R,NNB14P)
     END IF
     CALL INISTF2(TSMTMP(VDW),TSMTMP(ELEC),DX,DY,DZ,NATOM)
     CALL ENBOND(TSMTMP(VDW),TSMTMP(ELEC),BNBNDR, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(ELEC),(/ZERO/),(/0/),.FALSE., &
          QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,    & 
#endif
          .FALSE.,NST2,EST2,.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

     VPRTTR = VPRTTR + TSMTMP(VDW) + TSMTMP(ELEC)
     VPRTNR = VPRTNR + TSMTMP(VDW) + TSMTMP(ELEC)
     VPRTVR = VPRTVR + TSMTMP(VDW)
     VPRTER = VPRTER + TSMTMP(ELEC)
     CALL PERDRV2(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA,NATOM, &
          TSMTRM(VDW),TSMTRM(ELEC),TSMTMP(VDW),TSMTMP(ELEC))
     IF(UPPERT) THEN
        CALL NNLST2(NATOM,NGRP,INBLOX,JNBX,INBLOP,JNBP,PRODLS, &
             INBLGP)
        CALL EWEXLST(NATOM,PRODLS,IBLO14X,INB14X, &
             NNB14X,IBLO14P,INB14P,NNB14P)
        CALL SETEWEX(NNB14X,NNB14R,NNB14P)
     END IF
     CALL INISTF2(TSMTMP(VDW),TSMTMP(ELEC),DX,DY,DZ,NATOM)
     CALL ENBOND(TSMTMP(VDW),TSMTMP(ELEC),BNBNDP, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(ELEC),(/ZERO/),(/0/),.FALSE., &
          QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,    & 
#endif
          .FALSE.,NST2,EST2,.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

     VPRTTP = VPRTTP + TSMTMP(VDW) + TSMTMP(ELEC)
     VPRTNP = VPRTNP + TSMTMP(VDW) + TSMTMP(ELEC)
     VPRTVP = VPRTVP + TSMTMP(VDW)
     VPRTEP = VPRTEP + TSMTMP(ELEC)
     CALL PERDRV2(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA,NATOM, &
          TSMTRM(VDW),TSMTRM(ELEC),TSMTMP(VDW),TSMTMP(ELEC))
     !
     ! colocated atoms
     IF (NCOLO > 0) THEN
        OLDFAST = FASTER
        OLDLFAST = LFAST
        FASTER = -1
        LFAST = -1
        !
        ! color(r)/<env+color>
        !
        CALL NNLST3(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ACOLO, &
             NCOLO,REACLS,PRODLS,INBLOG,ICOLO,1)
        CALL INISTF2(TSMTMP(VDW),TSMTMP(ELEC),DX,DY,DZ,NATOM)
        CALL ENBOND(TSMTMP(VDW),TSMTMP(ELEC),BNBNDC, &
             1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(ELEC), &
             (/ZERO/),(/0/),.FALSE.,.FALSE.,QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,    & 
#endif
             .FALSE.,NST2,EST2,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA      & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

        CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA,NATOM, &
             TSMTRM(ELEC),TSMTMP(ELEC))
        VPRTTR = VPRTTR + TSMTMP(ELEC)
        VPRTNR = VPRTNR + TSMTMP(ELEC)
        VPRTER = VPRTER + TSMTMP(ELEC)
        !
        ! calculate colo(r)/Prod atom interaction
        !
        CALL NNLST3(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ACOLO, &
             NCOLO,REACLS,PRODLS,INBLOG,ICOLO,6)
        CALL INISTF2(TSMTMP(VDW),TSMTMP(ELEC),DX,DY,DZ,NATOM)
        CALL ENBOND(TSMTMP(VDW),TSMTMP(ELEC),BNBNDC, &
             1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(ELEC), &
             (/ZERO/),(/0/),.FALSE.,.FALSE.,QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,    & 
#endif
             .FALSE.,NST2,EST2,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA      & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

        CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,ONE-PLAMBDA,NATOM, &
             TSMTRM(ELEC),TSMTMP(ELEC))

        VPRTTP = VPRTTP - TSMTMP(ELEC)
        VPRTNP = VPRTNP - TSMTMP(ELEC)
        VPRTEP = VPRTEP - TSMTMP(ELEC)
        !
        ! colo(p)/<prod + env>
        !

        DO I = 1,NCOLO
           COLOST(I) = CG(ACOLO(I))
           CG(ACOLO(I)) = CCOLO(I)
        enddo
        CALL NNLST3(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ACOLO, &
             NCOLO,REACLS,PRODLS,INBLOG,ICOLO,2)
        CALL INISTF2(TSMTMP(VDW),TSMTMP(ELEC),DX,DY,DZ,NATOM)
        CALL ENBOND(TSMTMP(VDW),TSMTMP(ELEC),BNBNDC, &
             1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(ELEC), &
             (/ZERO/),(/0/),.FALSE.,.FALSE.,QETERM(ELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,    & 
#endif
             .FALSE.,NST2,EST2,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA      & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

        ! with the other terms we are calculating that which should be
        ! subtracted out of the total hybrid molecule energy (and forces).
        ! in the regular energy calculation this term is not calculated
        ! hence the -lambda^N instead of plambda.
        CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA-ONE,NATOM, &
             TSMTRM(ELEC),TSMTMP(ELEC))
        VPRTTP = VPRTTP + TSMTMP(ELEC)
        VPRTNP = VPRTNP + TSMTMP(ELEC)
        VPRTEP = VPRTEP + TSMTMP(ELEC)
        ! restore charges
        DO I = 1,NCOLO
           CG(ACOLO(I)) = COLOST(I)
        enddo

        LFAST = OLDLFAST
        FASTER = OLDFAST
     END IF
     !---      We are calculating how much to subtract out from the total potential
     !---      energy to get the correct E(lambda)
     TSMTRM(VDW) = -TSMTRM(VDW)
     TSMTRM(ELEC) = -TSMTRM(ELEC)
  END IF

  DO I = 1,NATOM
     DX(I)= OLDDX(I)
     DY(I)= OLDDY(I)
     DZ(I)= OLDDZ(I)
  enddo
  RETURN
END SUBROUTINE PENER2A

SUBROUTINE PENER2B(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,X,Y,Z, &
     REACLS,PRODLS, &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,   & 
#endif
     COLOST,ICOLO)

  ! START OF DECLARATIONS USING INCLUDE STATEMENTS
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
  use code
  use param
  use energym
  use fast
  use hbondm
  use image
  use number
  use psf
  use memory
  use stream
  use tsms_mod
  use timerm
  use datstr,only:dupldt_image,freedt_image
  implicit none
  ! END OF DECLARATIONS USING INCLUDE STATEMENTS
  !
  integer,allocatable,dimension(:) :: rimgls, pimgls, iskip2
  INTEGER I,ICOLO(*)
  real(chm_real) COLOST(*)
  INTEGER REACLS(*),PRODLS(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*)
  real(chm_real) RLAMBDA,PLAMBDA
  real(chm_real) EST2,EIMG
  INTEGER OLDFAST,OLDLFAST
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  !
  ! calculate product and reactant lambda factors for image tsm energy terms
  !
  IF (NTRANS <= 0) RETURN
  IF (LPOWER == 1) THEN
     RLAMBDA = LAMBDA
     PLAMBDA = ONE-LAMBDA
  ELSE
     RLAMBDA = ONE-(ONE-LAMBDA)**LPOWER
     PLAMBDA = ONE - LAMBDA**LPOWER
  END IF
  EST2 = ZERO
  !--   make copies of the forces
  OLDDX(1:natim) = DX(1:natim)
  OLDDY(1:natim) = DY(1:natim)
  OLDDZ(1:natim) = DZ(1:natim)

  call chmalloc('tsme.src','PENER2B','RIMGLS',NATIM,intg=RIMGLS)
  CALL EXTSL2(NATOM,REACLS,NATIM,RIMGLS, &
       BIMAG%IMATTR)
  call chmalloc('tsme.src','PENER2B','PIMGLS',NATIM,intg=PIMGLS)
  CALL EXTSL2(NATOM,PRODLS,NATIM,PIMGLS, &
       BIMAG%IMATTR)
  call chmalloc('tsme.src','PENER2B','ISKIP2',NIMHB,intg=ISKIP2)
  !
  !     duplicate the bimag structure so that a new non-bonded list
  !     can be generated
  !
  IF(UPPERT) THEN
     CALL MKIPDT(BIMAGR,BIMAG,NRCTAT)
     CALL MKIPDT(BIMAGP,BIMAG,NPRDAT)
  END IF
  CALL INISTF4(EIMG,TSMTMP(IMVDW),TSMTMP(IMELEC), &
       TSMTMP(IMHBND),DX,DY,DZ,NATIM)
  CALL IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGR, &
       RIMGLS,PIMGLS,ACOLO,NCOLO, &
       ISKIP2,ICOLO,1,UPPERT)
  VPRTTR = VPRTTR + EIMG
  VPRTNR = VPRTNR + EIMG
  VPRTVR = VPRTVR + TSMTMP(IMVDW)
  VPRTER = VPRTER + TSMTMP(IMELEC)
  CALL PERDRV3(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA,NATIM, &
       TSMTRM(IMVDW),TSMTRM(IMELEC),TSMTRM(IMHBND), &
       TSMTMP(IMVDW),TSMTMP(IMELEC),TSMTMP(IMHBND))

  CALL INISTF4(EIMG,TSMTMP(IMVDW),TSMTMP(IMELEC), &
       TSMTMP(IMHBND),DX,DY,DZ,NATIM)
  CALL IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGP, &
       RIMGLS,PIMGLS,ACOLO,NCOLO, &
       ISKIP2,ICOLO,2,UPPERT)
  VPRTTP = VPRTTP + EIMG
  VPRTNP = VPRTNP + EIMG
  VPRTVP = VPRTVP + TSMTMP(IMVDW)
  VPRTEP = VPRTEP + TSMTMP(IMELEC)
  CALL PERDRV3(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA,NATIM, &
       TSMTRM(IMVDW),TSMTRM(IMELEC),TSMTRM(IMHBND), &
       TSMTMP(IMVDW),TSMTMP(IMELEC),TSMTMP(IMHBND))
  IF(NCOLO > 0) THEN
     OLDFAST = FASTER
     OLDLFAST = LFAST
     FASTER = -1
     LFAST = -1
     !!          CALL INITDT(BIMAGC,SIMAG)
     CALL DUPLDT_image(BIMAGC,BIMAG)
     CALL INISTF4(EIMG,TSMTMP(IMVDW),TSMTMP(IMELEC), &
          TSMTMP(IMHBND),DX,DY,DZ,NATIM)
     CALL IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGC, &
          RIMGLS,PIMGLS,ACOLO,NCOLO, &
          ISKIP2,ICOLO,3,UPPERT)
     CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,RLAMBDA,NATIM, &
          TSMTRM(IMELEC),TSMTMP(IMELEC))
     VPRTTR = VPRTTR + TSMTMP(IMELEC)
     VPRTNR = VPRTNR + TSMTMP(IMELEC)
     VPRTER = VPRTER + TSMTMP(IMELEC)
     !
     CALL INISTF4(EIMG,TSMTMP(IMVDW),TSMTMP(IMELEC), &
          TSMTMP(IMHBND),DX,DY,DZ,NATIM)
     CALL IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGC, &
          RIMGLS,PIMGLS,ACOLO,NCOLO, &
          ISKIP2,ICOLO,4,UPPERT)
     CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,ONE-PLAMBDA,NATIM, &
          TSMTRM(IMELEC),TSMTMP(IMELEC))
     VPRTTP = VPRTTP - TSMTMP(IMELEC)
     VPRTNP = VPRTNP - TSMTMP(IMELEC)
     VPRTEP = VPRTEP - TSMTMP(IMELEC)

     !         restore charges.
     DO I = 1,NCOLO
        COLOST(I) = CG(ACOLO(I))
        CG(ACOLO(I)) = CCOLO(I)
     enddo
     CALL EXTCCG(NATOM,NATIM,CG,BIMAG%IMATTR)
     CALL INISTF4(EIMG,TSMTMP(IMVDW),TSMTMP(IMELEC), &
          TSMTMP(IMHBND),DX,DY,DZ,NATIM)
     CALL IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGC, &
          RIMGLS,PIMGLS,ACOLO,NCOLO, &
          ISKIP2,ICOLO,5,UPPERT)

     CALL PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,PLAMBDA-ONE,NATIM, &
          TSMTRM(IMELEC),TSMTMP(IMELEC))
     VPRTTP = VPRTTP + TSMTMP(IMELEC)
     VPRTNP = VPRTNP + TSMTMP(IMELEC)
     VPRTEP = VPRTEP + TSMTMP(IMELEC)
     DO I = 1,NCOLO
        CG(ACOLO(I)) = COLOST(I)
     enddo
     CALL EXTCCG(NATOM,NATIM,CG,BIMAG%IMATTR)
     FASTER = OLDFAST
     LFAST = OLDLFAST
     CALL FREEDT_image(BIMAGC)
  END IF
  !---  We are calculating how much to subtract out from the total potential
  !---  energy to get the correct E(lambda)
  TSMTRM(IMVDW) = -TSMTRM(IMVDW)
  TSMTRM(IMELEC) = -TSMTRM(IMELEC)
  TSMTRM(IMHBND) = -TSMTRM(IMHBND)

  call chmdealloc('tsme.src','PENER2B','RIMGLS',NATIM,intg=RIMGLS)
  call chmdealloc('tsme.src','PENER2B','PIMGLS',NATIM,intg=PIMGLS)
  call chmdealloc('tsme.src','PENER2B','ISKIP2',NIMHB,intg=ISKIP2)

  DX(1:natim)= OLDDX(1:natim)
  DY(1:natim)= OLDDY(1:natim)
  DZ(1:natim)= OLDDZ(1:natim)

  RETURN
END SUBROUTINE PENER2B

SUBROUTINE EXTSL2(NATOM,ISLCT1,NATIM,ISLCT2,IMATTR)
  !
  ! This routine extends the selection list to include pairs of image
  !  atoms
  !
  !  By: Stephen Fleischman  11/86
  !
  use chm_kinds
  implicit none
  INTEGER ISLCT1(*),ISLCT2(*),IMATTR(*)
  INTEGER NATOM,NATIM,I
  
  ISLCT2(1:natom) = ISLCT1(1:natom)
  DO I = NATOM+1,NATIM
     ISLCT2(I) = ISLCT1(IMATTR(I))
  enddo
  RETURN
END SUBROUTINE EXTSL2

SUBROUTINE EXTCCG(NATOM,NATIM,CG,IMATTR)
  !
  ! This routine extends the selection list to include pairs of image
  !  atoms
  !
  !  By: Stephen Fleischman  11/86
  !
  use chm_kinds
  implicit none
  INTEGER IMATTR(*)
  INTEGER NATOM,NATIM,I
  real(chm_real) CG(*)
  
  DO I = NATOM+1,NATIM
     CG(I) = CG(IMATTR(I))
  enddo
  RETURN
END SUBROUTINE EXTCCG

SUBROUTINE IMINT4(X,Y,Z,DX,DY,DZ,EIMG,BNBND,BIMAG,BIMAGC, &
     RIMGLS,PIMGLS,ACOLO,NCOLO,ISKIP,ICOLO,OPT,UPPERT)
  !
  !  this routine handles the setup and calling of energy routines
  !  to determine the image energy contribution.
  !
  !    Taken for routine by Bernard R. Brooks    9/83
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor                 
#endif

  ! START OF DECLARATIONS USING INCLUDE STATEMENTS
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use enbond_mod
  use energym
  use inbnd
  use image
  use hbondm
  use param
  use code
  use psf
  use fast
  use stream
#if KEY_FLUCQ==1
  use flucq      
#endif
  use datstr
  use machutil,only:die
  implicit none
  ! END OF DECLARATIONS USING INCLUDE STATEMENTS

  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),EIMG
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  type(imageDataStructure) BIMAGC
  INTEGER ICOLO(*)
  LOGICAL DOVDW
  real(chm_real) ENBX,EELX,EST2X,EIMST2
  INTEGER I,IMX,OPT,N3OPT(3),ACOLO(*),NCOLO

  INTEGER RIMGLS(*),PIMGLS(*),ISKIP(*)
  type(nonbondDataStructure) BDUMMY
  INTEGER NNB14X
  LOGICAL UPPERT
  DATA N3OPT/1,6,2/
  !     set up a dummy data structure for enbond.
  !
  IF(NTRANS == 0) RETURN

  CALL ALIASDT_nbond(BDUMMY, BNBND)
  !     Make sure we have the current counters, flags and cutoffs.
  CALL GETBND(BNBND,.TRUE.)
  !        We will restore this later. Otherwise the NNB14 variable in
  !         NBINTS will be lost.
  NNB14X = NNB14
  !
  !     construct coordinates for all image atoms.
  !
  ENBX=ZERO
  EELX=ZERO
  EST2X=ZERO
  EIMST2=ZERO

  CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAG)
  CALL NBSET_G14_FROM_IMG(BDUMMY, BIMAG)
  NNG14=BIMAG%NIMING

  !
  !  check if any self-energy terms are present
  !
  IF (bimagc%NIMNBS > 0 .OR. bimagc%NIMNBX > 0) THEN
     IMX=NATIM
     DX(1:imx)=DX(I)*TWO
     DY(1:imx)=DY(I)*TWO
     DZ(1:imx)=DZ(I)*TWO
     IF (OPT == 1) THEN
        IF(UPPERT) THEN
           CALL NNLST2(NATIM,NIMGRP,bimag%IMBLOS, &
                bimag%IMJNBS,bimagc%IMBLOS, &
                bimagc%IMJNBS,RIMGLS,bimagc%IMBLOX)
           !                If there are non-self interactions we will handle
           !                the exclusions there.
           IF (bimagc%NIMNB > 0) THEN
              CALL EWEXLST(NATOM,RIMGLS, &
                   bimag%IMIBLO, &
                   bimag%IMINB,bimag%NIMINB, &
                   bimagc%IMIBLO, &
                   bimagc%IMINB,bimagc%NIMINB)
           ENDIF
        ENDIF
        DOVDW = QETERM(IMVDW)
     ELSE IF (OPT == 2) THEN
        IF(UPPERT) THEN
           CALL NNLST2(NATIM,NIMGRP,bimag%IMBLOS, &
                bimag%IMJNBS,bimagc%IMBLOS, &
                bimagc%IMJNBS,PIMGLS,bimagc%IMBLOX)
           IF (bimagc%NIMNB > 0) THEN
              CALL EWEXLST(NATOM,PIMGLS, &
                   bimag%IMIBLO, &
                   bimag%IMINB,bimag%NIMINB, &
                   bimagc%IMIBLO, &
                   bimagc%IMINB,bimagc%NIMINB)
           ENDIF
        ENDIF
        DOVDW = QETERM(IMVDW)
     ELSE IF (OPT > 2.AND.OPT <= 5) THEN
        CALL NNLST3(NATIM,NIMGRP,bimag%IMBLOS, &
             bimag%IMJNBS,bimagc%IMBLOS, &
             bimagc%IMJNBS,ACOLO,NCOLO,RIMGLS,PIMGLS, &
             bimagc%IMBLOX,ICOLO,N3OPT(OPT-2))
        !             get the number of self-image pairs
        DOVDW = .FALSE.
     ELSE
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'Programmer error in imint4, illegal opt'
        CALL DIE
     END IF
     CALL UPDNIM(NATIM,bimagc%IMBLOS,bimagc%NIMNBS, &
          bimagc%IMBLOX,bimagc%NIMNBX)
     IF (bimagc%NIMNBS > 0 .OR. &
          bimagc%NIMNBX > 0) THEN


        CALL NBSET_FROM_IMG_SX(BDUMMY, BIMAGC)
        NNNB =bimagc%NIMNBS
        NNNBG=bimagc%NIMNBX
        !
        !       We will not do the Ewald image exclusions here.
        !       Note: we need to create a special image self exclusion list to
        !       do this properly.  For now, EWALD must not be used if atoms
        !       are excluded from their own images!
        NNB14= 0

        CALL SETBND(BDUMMY)
        CALL ENBOND(ENBX,EELX,BDUMMY, &
             1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.FALSE.,ZERO,(/ZERO/),(/0/),.FALSE.,DOVDW, &
             QETERM(IMELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,      & 
#endif
             .FALSE.,NST2,EST2X,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

        ENBX = ENBX*HALF
        EELX = EELX*HALF
        EST2X = EST2X*HALF
        DX(1:imx)=DX(1:imx)*HALF
        DY(1:imx)=DY(1:imx)*HALF
        DZ(1:imx)=DZ(1:imx)*HALF
     ENDIF
  END IF
  !
  !      compute image nonbonded energies
  IF (bimag%NIMNB > 0 .OR. bimag%NIMNBG > 0) THEN

     IF (OPT == 1) THEN
        CALL NNLST2(NATIM,NIMGRP,bimag%IMBLO, &
             bimag%IMJNB,bimagc%IMBLO, &
             bimagc%IMJNB,RIMGLS,bimagc%IMBLOG)
        CALL EWEXLST(NATOM,RIMGLS, &
             bimag%IMIBLO, &
             bimag%IMINB,bimag%NIMINB, &
             bimagc%IMIBLO, &
             bimagc%IMINB,bimagc%NIMINB)
        DOVDW = QETERM(IMVDW)
     ELSE IF (OPT == 2) THEN
        CALL NNLST2(NATIM,NIMGRP,bimag%IMBLO, &
             bimag%IMJNB,bimagc%IMBLO, &
             bimagc%IMJNB,PIMGLS,bimagc%IMBLOG)
        CALL EWEXLST(NATOM,PIMGLS, &
             bimag%IMIBLO, &
             bimag%IMINB,bimag%NIMINB, &
             bimagc%IMIBLO, &
             bimagc%IMINB,bimagc%NIMINB)
        DOVDW = QETERM(IMVDW)
     ELSE IF (OPT > 2.AND.OPT <= 5) THEN
        CALL NNLST3(NATIM,NIMGRP,bimag%IMBLO, &
             bimag%IMJNB,bimagc%IMBLO, &
             bimagc%IMJNB,ACOLO,NCOLO,RIMGLS,PIMGLS, &
             bimagc%IMBLOG, &
             ICOLO,N3OPT(OPT-2))
        DOVDW = .FALSE.
     ELSE
        IF(WRNLEV >= 2) WRITE(OUTU,*) &
             'Programmer error in imint4, illegal opt'
        CALL DIE
     END IF

     CALL UPDNIM(NATIM,bimagc%IMBLO,bimagc%NIMNB, &
          bimagc%IMBLOG,bimagc%NIMNBG)
     IF (bimagc%NIMNB > 0 .OR. bimagc%NIMNBG > 0) THEN
        !
        !       We do the Ewald image exclusions here.
        !       Note: we need to create a special image self exclusion list to
        !       do this properly.  Here we assume that none of the image exclusions
        !       are of the self type.
        NNB14=bimagc%NIMINB

        CALL NBSET_FROM_IMG_G(BDUMMY, BIMAGC)
        NNNB =bimagc%NIMNB
        NNNBG=bimagc%NIMNBG

        CALL SETBND(BDUMMY)
        CALL ENBOND(TSMTMP(IMVDW),TSMTMP(IMELEC),BDUMMY, &
             1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,TSMTMP(IMELEC), &
             (/ZERO/),(/0/),.FALSE.,DOVDW,QETERM(IMELEC), &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,           & 
#endif
             .FALSE.,NST2,EIMST2,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
             ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
             )

     END IF
  END IF

  call FREEDT_nbond(BDUMMY)
  !
  !     compute image hbond energies
  !
  IF(NIMHB > 0.AND.QETERM(IMHBND).AND.(OPT == 1.OR.OPT == 2)) THEN
     IF (OPT == 1) THEN
        CALL HBSEL(RIMGLS,IHB,JHB,ISKIP,NHB+1,NIMHB)
     ELSE
        CALL HBSEL(PIMGLS,IHB,JHB,ISKIP,NHB+1,NIMHB)
     END IF
     CALL EHBOND(TSMTMP(IMHBND),IHB(NHB+1),JHB(NHB+1),KHB(NHB+1), &
          LHB(NHB+1),ICH(NHB+1),NIMHB,CHBA,CHBB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,ISKIP,CTONHB, &
          CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
  END IF
  TSMTMP(IMVDW)=TSMTMP(IMVDW)+ENBX
  TSMTMP(IMELEC)=TSMTMP(IMELEC)+EELX
  EIMG = TSMTMP(IMVDW)+TSMTMP(IMELEC)+TSMTMP(IMHBND)
  !
  !  Restore NNB14 to BNBND, First make sure we have all of
  !  the BNBND values in the scalar common block (Except for NNB14).
  CALL GETBND(BNBND,.TRUE.)
  !        Now replace NNB14.
  NNB14 = NNB14X
  !        Stick this back in BNBND.
  CALL SETBND(BNBND)


  RETURN
END SUBROUTINE IMINT4

SUBROUTINE HBSEL(ISLCT,IHB,JHB,ISKIP,START,FINISH)
  use chm_kinds
  implicit none
  INTEGER ISLCT(*),ISKIP(*),IHB(*),JHB(*)
  INTEGER I,START,FINISH

  DO I=START,FINISH
     IF (ISLCT(IHB(I)) == 1 .OR. ISLCT(JHB(I)) == 1) THEN
        ISKIP(I)=0
     ELSE
        ISKIP(I)=1
     END IF
  enddo
  RETURN
END SUBROUTINE HBSEL

SUBROUTINE NNLST2(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ISLCT, &
    INBLOG)
  !
  !     make new nonbond lists
  !
  use chm_kinds
 implicit none
 INTEGER JNBX(*),JNB(*),ISLCT(*)
 INTEGER INBLOG(*),INBLO(*),INBLOX(*)
 INTEGER NATOM,NGRP

 INTEGER IFIRST,ILAST,I,J,N,K

 IFIRST=1
 N=0
 DO I=1,NATOM
    ILAST=INBLOX(I)
    IF(ISLCT(I) == 1) THEN
       DO J=IFIRST,ILAST
          N=N+1
          JNB(N)=JNBX(J)
       enddo
    ELSE
       DO J = IFIRST,ILAST
          K = ABS(JNBX(J))
          IF(ISLCT(K) == 1) THEN
             N=N+1
             JNB(N) = JNBX(J)
          END IF
       enddo
    END IF
    INBLO(I)=N
    IFIRST=ILAST+1
 enddo
 INBLOG(1:ngrp)=0
 RETURN
END SUBROUTINE NNLST2

SUBROUTINE NNLST3(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ACOLO, &
    NCOLO,REACLS,PRODLS,INBLOG,ICOLO,OPT)
  !
  use chm_kinds
  use stream
  use machutil,only:die
 implicit none
 !
 !     make new nonbond lists
 !
 INTEGER JNBX(*),JNB(*),REACLS(*),PRODLS(*)
 INTEGER INBLOG(*),INBLO(*),INBLOX(*),ACOLO(*),NCOLO
 INTEGER NATOM,NGRP,OPT,ICOLO(*)
 !
 INTEGER IFIRST,ILAST,I,J,N,JX,TEST

 IF (OPT >= 1.AND.OPT <= 3) THEN
    TEST = 0
 ELSE IF(OPT >= 4.AND.OPT <= 6) THEN
    TEST = 1
 ELSE
    IF(WRNLEV >= 2) WRITE(OUTU,*) &
         'illegal opt value in function lstchk'
    IF(WRNLEV >= 2) WRITE(OUTU,*) 'this is a programming error'
    CALL DIE
 END IF
 IFIRST=1
 N=0

 IF (OPT == 1.OR.OPT == 4) THEN
    DO I=1,NATOM
       ILAST=INBLOX(I)
       IF (ICOLO(I) == 1) THEN
          DO J=IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (REACLS(JX) == TEST.AND.PRODLS(JX) == TEST) THEN
                N=N+1
                JNB(N)=JNBX(J)
             END IF
          enddo
       ELSE IF (REACLS(I) == TEST.AND.PRODLS(I) == TEST) THEN
          DO J = IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (ICOLO(JX) == 1) THEN
                N=N+1
                JNB(N) = JNBX(J)
             END IF
          enddo
       END IF
       INBLO(I)=N
       IFIRST=ILAST+1
    enddo
 ELSE IF (OPT == 2.OR.OPT == 5) THEN
    DO I=1,NATOM
       ILAST=INBLOX(I)
       IF (ICOLO(I) == 1) THEN
          DO J=IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (REACLS(JX) == TEST) THEN
                N=N+1
                JNB(N)=JNBX(J)
             END IF
          enddo
       ELSE IF (REACLS(I) == TEST) THEN
          DO J = IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (ICOLO(JX) == 1) THEN
                N=N+1
                JNB(N) = JNBX(J)
             END IF
          enddo
       END IF
       INBLO(I)=N
       IFIRST=ILAST+1
    enddo
 ELSE IF (OPT == 3.OR.OPT == 6) THEN
    DO I=1,NATOM
       ILAST=INBLOX(I)
       IF (ICOLO(I) == 1) THEN
          DO J=IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (PRODLS(JX) == TEST) THEN
                N=N+1
                JNB(N)=JNBX(J)
             END IF
          enddo
       ELSE IF (PRODLS(I) == TEST) THEN
          DO J = IFIRST,ILAST
             JX = ABS(JNBX(J))
             IF (ICOLO(JX) == 1) THEN
                N=N+1
                JNB(N) = JNBX(J)
             END IF
          enddo
       END IF
       INBLO(I)=N
       IFIRST=ILAST+1
    enddo
 END IF
 INBLOG(1:ngrp)=0
 RETURN
END SUBROUTINE NNLST3

SUBROUTINE EWEXLST(NATOM,ISLCT,IBLO14X,INB14X,NNB14X, &
    IBLO14,INB14,NNB14)
  !
  !     make new ewald exclusion lists.
  !
  use chm_kinds
 implicit none
 INTEGER ISLCT(*)
 !     New
 INTEGER IBLO14(*),INB14(*),NNB14
 !     Old
 INTEGER IBLO14X(*),INB14X(*),NNB14X
 INTEGER NATOM

 INTEGER IFIRST,ILAST,I,J,N,K

 IF(NNB14X <= 0) THEN
    NNB14 =0
    RETURN
 ENDIF
 IFIRST=1
 N=0
 DO I=1,NATOM
    ILAST=IBLO14X(I)
    IF(ISLCT(I) == 1) THEN
       DO J=IFIRST,ILAST
          N=N+1
          INB14(N)=INB14X(J)
       enddo
    ELSE
       DO J = IFIRST,ILAST
          K = abs(INB14X(J))
          IF(ISLCT(K) == 1) THEN
             N=N+1
             INB14(N) = INB14X(J)
          END IF
       enddo
    END IF
    IBLO14(I)=N
    IFIRST=ILAST+1
 enddo
 NNB14 = N
 RETURN
END SUBROUTINE EWEXLST

SUBROUTINE COLOEXT(ACOLO,NCOLO,ICOLO,OPT)

  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
  use memory
  use image
  use psf
  use stream
 implicit none
 type(chm_iptr) :: icolo
 INTEGER,PARAMETER :: COLO_FREE=1,COLO_FILL=2
 INTEGER ACOLO(*),NCOLO,OPT
 INTEGER NATOMX

 IF (NTRANS > 0) THEN
    NATOMX = NATIM
 ELSE
    NATOMX = NATOM
 ENDIF
 IF (OPT == COLO_FREE) THEN
    call chmdealloc('tsme.src','COLOEXT','ICOLO',NATOMX,intgp=ICOLO%A)
 ELSE IF (OPT == COLO_FILL) THEN
    call chmalloc('tsme.src','COLOEXT','ICOLO',NATOMX,intgp=ICOLO%A)
    CALL COLOEXT2(NATOM,NATIM,NATOMX,NTRANS,NCOLO,ACOLO, &
         ICOLO%A,bimag%IMATTR)
 ELSE
    CALL WRNDIE(-4,'<coloext>','Unknown option.')
 ENDIF
 RETURN
END SUBROUTINE COLOEXT

SUBROUTINE COLOEXT2(NATOM,NATIM,NATOMX,NTRANS,NCOLO,ACOLO,ICOLO, &
    IMATTR)
  !
  use chm_kinds
 implicit none
 INTEGER NATOM,NATIM,NTRANS,NATOMX
 INTEGER NCOLO,ACOLO(*),ICOLO(*)
 INTEGER IMATTR(*)
 !
 INTEGER I

 ICOLO(1:natomx) = 0
 ICOLO(ACOLO(1:ncolo)) = 1

 IF (NTRANS > 0.AND.NATIM > NATOM) THEN
    DO I = NATOM+1,NATIM
       ICOLO(I) = ICOLO(IMATTR(I))
    enddo
 ENDIF
 RETURN
END SUBROUTINE COLOEXT2

SUBROUTINE INISTF(ENERGY,DX,DY,DZ,NATOM)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*),ENERGY
  INTEGER I,NATOM

  ENERGY = ZERO
  DX(1:natom)=ZERO
  DY(1:natom)=ZERO
  DZ(1:natom)=ZERO
  RETURN
END SUBROUTINE INISTF

SUBROUTINE INISTF2(ENERGY1,ENERGY2,DX,DY,DZ,NATOM)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*),ENERGY1,ENERGY2
  INTEGER I,NATOM

  ENERGY1 = ZERO
  ENERGY2 = ZERO
  DX(1:natom)=ZERO
  DY(1:natom)=ZERO
  DZ(1:natom)=ZERO
  RETURN
END SUBROUTINE INISTF2

#if KEY_UNUSED==1 /*inistf3_unused*/
SUBROUTINE INISTF3(ENERGY1,ENERGY2,ENERGY3,DX,DY,DZ,NATOM)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*),ENERGY1,ENERGY2,ENERGY3
  INTEGER I,NATOM

  ENERGY1 = ZERO
  ENERGY2 = ZERO
  ENERGY3 = ZERO
  DX(1:natom)=ZERO
  DY(1:natom)=ZERO
  DZ(1:natom)=ZERO
  RETURN
END SUBROUTINE INISTF3
#endif /* (inistf3_unused)*/

SUBROUTINE INISTF4(ENERGY1,ENERGY2,ENERGY3,ENERGY4,DX,DY,DZ,NATOM)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*),ENERGY1,ENERGY2,ENERGY3,ENERGY4
  INTEGER I,NATOM

  ENERGY1 = ZERO
  ENERGY2 = ZERO
  ENERGY3 = ZERO
  ENERGY4 = ZERO
  DX(1:natom)=ZERO
  DY(1:natom)=ZERO
  DZ(1:natom)=ZERO
  RETURN
END SUBROUTINE INISTF4

SUBROUTINE PERDRV(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,LAMBDA, &
     NATOM,EU,EN)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*),DX(*),DY(*),DZ(*), &
       EU,EN,LAMBDA
  INTEGER NATOM,I

  EU = EU+ LAMBDA*EN
  OLDDX(1:natom) = OLDDX(1:natom) - LAMBDA*DX(1:natom)
  OLDDY(1:natom) = OLDDY(1:natom) - LAMBDA*DY(1:natom)
  OLDDZ(1:natom) = OLDDZ(1:natom) - LAMBDA*DZ(1:natom)
  RETURN
END SUBROUTINE PERDRV

SUBROUTINE PERDRV2(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,LAMBDA, &
     NATOM,EU1,EU2,EN1,EN2)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*),DX(*),DY(*),DZ(*),LAMBDA
  real(chm_real) EU1,EU2,EN1,EN2
  INTEGER NATOM,I

  EU1 = EU1 + LAMBDA*EN1
  EU2 = EU2 + LAMBDA*EN2
  OLDDX(1:natom) = OLDDX(1:natom) - LAMBDA*DX(1:natom)
  OLDDY(1:natom) = OLDDY(1:natom) - LAMBDA*DY(1:natom)
  OLDDZ(1:natom) = OLDDZ(1:natom) - LAMBDA*DZ(1:natom)
  RETURN
END SUBROUTINE PERDRV2

SUBROUTINE PERDRV3(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,LAMBDA, &
     NATOM,EU1,EU2,EU3,EN1,EN2,EN3)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*),DX(*),DY(*),DZ(*),LAMBDA
  real(chm_real) EU1,EU2,EU3,EN1,EN2,EN3
  INTEGER NATOM,I

  EU1 = EU1 + LAMBDA*EN1
  EU2 = EU2 + LAMBDA*EN2
  EU3 = EU3 + LAMBDA*EN3
  OLDDX(1:natom) = OLDDX(1:natom) - LAMBDA*DX(1:natom)
  OLDDY(1:natom) = OLDDY(1:natom) - LAMBDA*DY(1:natom)
  OLDDZ(1:natom) = OLDDZ(1:natom) - LAMBDA*DZ(1:natom)
  RETURN
END SUBROUTINE PERDRV3

#if KEY_UNUSED==1 /*perdrv4_unused*/
SUBROUTINE PERDRV4(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,LAMBDA, &
     NATOM,EU1,EU2,EU3,EU4,EN1,EN2,EN3,EN4)
  !
  use chm_kinds
  use number
  implicit none
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*),DX(*),DY(*),DZ(*),LAMBDA
  real(chm_real) EU1,EU2,EU3,EU4,EN1,EN2,EN3,EN4
  INTEGER NATOM,I

  EU1 = EU1 + LAMBDA*EN1
  EU2 = EU2 + LAMBDA*EN2
  EU3 = EU3 + LAMBDA*EN3
  EU4 = EU4 + LAMBDA*EN4
  OLDDX(1:natom) = OLDDX(1:natom) - LAMBDA*DX(1:natom)
  OLDDY(1:natom) = OLDDY(1:natom) - LAMBDA*DY(1:natom)
  OLDDZ(1:natom) = OLDDZ(1:natom) - LAMBDA*DZ(1:natom)
  RETURN
END SUBROUTINE PERDRV4
#endif /* (perdrv4_unused)*/

SUBROUTINE DNTSUB(OLDDX,OLDDY,OLDDZ,DX,DY,DZ,SLCT,NATOM,EU,EN)
  !
  use chm_kinds
  use number
  implicit none
  INTEGER SLCT(*)
  INTEGER NATOM,I
  real(chm_real) OLDDX(*),OLDDY(*),OLDDZ(*),DX(*),DY(*),DZ(*),EU,EN

  DO I = 1,NATOM
     IF(SLCT(I) == 0) THEN
        OLDDX(I) = OLDDX(I) - DX(I)
        OLDDY(I) = OLDDY(I) - DY(I)
        OLDDZ(I) = OLDDZ(I) - DZ(I)
     END IF
  enddo
  EU = EU +EN
  RETURN
END SUBROUTINE DNTSUB

SUBROUTINE MAKPDT(BNBNDX,BNBND,NPATOM)
  !
  !     Sets up a data structure for the perturbation interaction
  !     energies.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use inbnd
  use psf
  use datstr,only:freedt_nbond
  implicit none
  !
  type(nonbondDataStructure) BNBNDX
  type(nonbondDataStructure) BNBND
  INTEGER NPATOM
  !
  INTEGER I
  !
  !
  CALL FREEDT_nbond(BNBNDX)

  allocate(bnbndx%JNB(NPATOM*NATOM))
  allocate(bnbndx%INBLO(NATOM))
  allocate(bnbndx%JNBG(NPATOM*NGRP))
  allocate(bnbndx%INBLOG(NGRP))
  if (associated(bnbnd%IBLO14)) allocate(bnbndx%IBLO14(size(bnbnd%IBLO14)))
  if (associated(bnbnd%INB14)) allocate(bnbndx%INB14(size(bnbnd%INB14)))

  !     Copy some flag and cutoff arrays.

  BNBNDX%LNBOPT = BNBND%LNBOPT
  BNBNDX%NBDIST = BNBND%NBDIST
  BNBNDX%NBINTS = BNBND%NBINTS

  RETURN
END SUBROUTINE MAKPDT

SUBROUTINE MKIPDT(BIMAGX,BIMAG,NPATOM)
  !
  !     Sets up an image data structure for the perturbation interaction
  !     energies.
  !
  use chm_kinds
  use chm_types
  use chm_types
  use dimens_fcm
  use image
  use datstr,only:freedt_image
  implicit none
  !
  type(imageDataStructure) BIMAGX
  type(imageDataStructure) BIMAG

  INTEGER NPATOM
  !
  CALL FREEDT_image(BIMAGX)
  allocate(bimagx%IMJNB(NPATOM*NATIM))
  allocate(bimagx%IMBLO(NATIM))
  allocate(bimagx%IMJNBS(NPATOM*NATIM))
  allocate(bimagx%IMBLOS(NATIM))
  allocate(bimagx%IMJNBG(NPATOM*NIMGRP))
  allocate(bimagx%IMBLOG(NIMGRP))
  allocate(bimagx%IMJNBX(NPATOM*NIMGRP))
  allocate(bimagx%IMBLOX(NIMGRP))
  if (allocated(bimag%IMATPT)) then
     allocate(bimagx%IMATPT(size(bimag%IMATPT)))
     BIMAGX%IMATPT = BIMAG%IMATPT
  endif
  if (allocated(bimag%IMATTR)) then
     allocate(bimagx%IMATTR(size(bimag%IMATTR)))
     BIMAGX%IMATTR = BIMAG%IMATTR
  endif
  if (allocated(bimag%IMCENF)) then
     allocate(bimagx%IMCENF(size(bimag%IMCENF)))
     BIMAGX%IMCENF = BIMAG%IMCENF
  endif
  !     Ewald exclusion lists.
  if (allocated(bimag%IMIBLO)) allocate(bimagx%IMIBLO(size(bimag%IMIBLO)))
  if (allocated(bimag%IMINB)) allocate(bimagx%IMINB(size(bimag%IMINB)))

  RETURN
END SUBROUTINE MKIPDT

SUBROUTINE MKDUMC(BNBNDX)
  !
  !     Sets up a dummy data structure.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use inbnd
  use datstr,only:freedt_nbond
  implicit none
  !
  !!      INTEGER BNBNDX(*),LNBNDX(*)
  type(nonbondDataStructure) BNBNDX

  !
  CALL FREEDT_nbond(BNBNDX)

  RETURN
END SUBROUTINE MKDUMC

SUBROUTINE PIGCVSET(X,Y,Z)
  use chm_kinds
  use dimens_fcm
  use tsms_mod
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER IPIGG
  ! sets back atom coordinates or velocities to piggy etc.
  !
  DO IPIGG =1,NPIGG
     X(BACK(IPIGG)) = X(PIGGY(IPIGG))
     Y(BACK(IPIGG)) = Y(PIGGY(IPIGG))
     Z(BACK(IPIGG)) = Z(PIGGY(IPIGG))
  enddo
  RETURN
END SUBROUTINE PIGCVSET

SUBROUTINE BACK0(X,Y,Z)
  use chm_kinds
  use dimens_fcm
  use number
  use tsms_mod
  implicit none
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER IPIGG
  ! zero coordinates, velocities, forces etc for back atoms.
  !
  X(BACK(1:nPIGG)) = ZERO
  Y(BACK(1:nPIGG)) = ZERO
  Z(BACK(1:nPIGG)) = ZERO
  RETURN
END SUBROUTINE BACK0

SUBROUTINE PIGFSET(DX,DY,DZ)
  use chm_kinds
  use dimens_fcm
  use number
  use tsms_mod
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*)
  INTEGER IPIGG
  ! sums piggy and back atom derivatives
  DO IPIGG=1,NPIGG
     DX(PIGGY(IPIGG)) = DX(PIGGY(IPIGG)) + DX(BACK(IPIGG))
     DY(PIGGY(IPIGG)) = DY(PIGGY(IPIGG)) + DY(BACK(IPIGG))
     DZ(PIGGY(IPIGG)) = DZ(PIGGY(IPIGG)) + DZ(BACK(IPIGG))
     !
     !        Back atoms are just place holders.
     !        This way the back atom forces won't contribute to virials
     !        and gradients.
     DX(BACK(IPIGG)) =  ZERO
     DY(BACK(IPIGG)) =  ZERO
     DZ(BACK(IPIGG)) =  ZERO
  enddo
  RETURN
end SUBROUTINE PIGFSET

SUBROUTINE PIGMSET(AMASS,QSTORE)
  use chm_kinds
  use dimens_fcm
  use number
  use tsms_mod
  implicit none
  real(chm_real) AMASS(*)
  INTEGER IPIGG
  LOGICAL QSTORE
  ! scales piggy mass
  ! Qstore = True then store amass(piggy) in apigg (done once and not
  !          during slow growth updates).
  IF(QSTORE) THEN
     DO IPIGG=1,NPIGG
        APIGG(IPIGG) = AMASS(PIGGY(IPIGG))
        AMASS(PIGGY(IPIGG)) = (ONE-LAMBDA)*APIGG(IPIGG) + &
             LAMBDA*AMASS(BACK(IPIGG))
     enddo
  ELSE
     DO IPIGG=1,NPIGG
        AMASS(PIGGY(IPIGG)) = (ONE-LAMBDA)*APIGG(IPIGG) + &
             LAMBDA*AMASS(BACK(IPIGG))
     enddo
  ENDIF
  RETURN
END SUBROUTINE PIGMSET

SUBROUTINE PIGMRSET(AMASS)
  use chm_kinds
  use dimens_fcm
  use number
  use tsms_mod
  implicit none
  real(chm_real) AMASS(*)
  INTEGER IPIGG
  ! restores piggy mass.
  DO IPIGG=1,NPIGG
     AMASS(PIGGY(IPIGG)) =  APIGG(IPIGG)
  enddo
  RETURN
END SUBROUTINE PIGMRSET

SUBROUTINE PUMBPER(NPUMB,PUMBDH,VPUMB,VPUMBS,ERR)
  !
  ! START OF DECLARATIONS SPECIFIED BY INCLUDE FILES
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use code
  use param
  use stream
  implicit none
  ! end of declarations specified by include files
  real(chm_real) VPUMB(*),VPUMBS(*)
  INTEGER PUMBDH(*),NPUMB,IPER,IC,I
  LOGICAL ERR,GO,PER

  ERR = .FALSE.
  DO I = 1,NPUMB
     GO = .TRUE.
     IC = ICP(PUMBDH(I))
     !        DO WHILE(GO)
     do while(GO) 
        IPER = CPD(IC)
        IF (ABS(IPER) == 3) THEN
           PER = .TRUE.
           GO = .FALSE.
        ELSEIF (IPER >= 0) THEN
           PER = .FALSE.
           GO = .FALSE.
        ELSE
           IC = IC + 1
        ENDIF
     enddo
     !        ENDDO
     IF (PER) THEN
        VPUMBS(I) = CPC(IC) - VPUMB(I)
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,3(I4,''-''),I4)') &
             'Umbrella sampling for the dihedral angle ', &
             IP(PUMBDH(I)),JP(PUMBDH(I)),KP(PUMBDH(I)),LP(PUMBDH(I))
        IF(PRNLEV >= 2) WRITE(OUTU,'(1X,A,1X,3(F10.4,1X))') &
             'Used Vsurrogate,Vactual,Vsurrogate -Vactual:', &
             CPC(IC),VPUMB(I),VPUMBS(I)
     ELSE
        IF(WRNLEV >= 2) &
             WRITE(OUTU,'(A,A,/,3(I4,''-''),I4)') '<PUMBR>', &
             'Could not find V3 term for dihedral angle consisting of atoms ', &
             IP(PUMBDH(I)),JP(PUMBDH(I)),KP(PUMBDH(I)),LP(PUMBDH(I))
        CALL DIEWRN(0)
        ERR = .TRUE.
        RETURN
     ENDIF
  enddo
  RETURN
END SUBROUTINE PUMBPER

SUBROUTINE PUMEPHI(EP,X,Y,Z)
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use tsms_mod
  use psf
  implicit none
  !
  !     Calculates the torsion angle energy for umbrella sampling.
  !     The code is essentialy a gutted version of the subroutine
  !     EPHI by Bernie Brooks.  A periodicity of 3 is assumed with
  !     a 0 or 180 deg phase shift.  This should have been checked for
  !     in the setup.  The force constants should be the difference
  !     between V(surrogate) and V(actual) to use Jorgensen's
  !     terminology.
  !
  !     THE FUNCTIONAL FORMS ARE:
  !             EPROP(EPOT) = K* (1.0 + PHASE* F(PERIODICITY,COS(PHI)) )
  !     WHERE
  !         F(3,C) = 4C**3 - 3C
  !     AND  PHASE = 1.0 or -1.0
  !
  !     Gutting and modification by Stephen Fleischman 6-87
  !
  !     Since this routine is independent from the rest of the
  !      dihedral routines it is not changed for the time being.
  !      If you want to enlarge the functionality, you can
  !      contact me at blondel@tammy.harvard.edu for incorporation
  !      of the new formulae. A.Blondel 1994.
  !
  !
  real(chm_real) EP
  real(chm_real) X(*),Y(*),Z(*),VCONST
  !
  !
  real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
  real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA,RB,RAR,RBR
  real(chm_real) AXR,AYR,AZR,BXR,BYR,BZR,CT
  real(chm_real) RLAMBDA,PLAMBDA
  LOGICAL PHASE
  INTEGER IPR,IPHI,I,J,K,L
  !
  RLAMBDA = (ONE-LAMBDA)**LPOWER
  PLAMBDA = LAMBDA**LPOWER
  !
  IF(NPUMB == 0) RETURN
  EP=ZERO
  !
  DO IPHI = 1,NPUMB
     IPR = ABS(PUMBDH(IPHI))
     PHASE = .FALSE.
     IF(PUMBDH(IPHI) < 0) PHASE = .TRUE.
     I=IP(IPR)
     J=JP(IPR)
     K=KP(IPR)
     L=LP(IPR)
     FX=X(I)-X(J)
     FY=Y(I)-Y(J)
     FZ=Z(I)-Z(J)
     GX=X(J)-X(K)
     GY=Y(J)-Y(K)
     GZ=Z(J)-Z(K)
     HX=X(L)-X(K)
     HY=Y(L)-Y(K)
     HZ=Z(L)-Z(K)
     AX=FY*GZ-FZ*GY
     AY=FZ*GX-FX*GZ
     AZ=FX*GY-FY*GX
     BX=HY*GZ-HZ*GY
     BY=HZ*GX-HX*GZ
     BZ=HX*GY-HY*GX
     RA2=AX*AX+AY*AY+AZ*AZ
     RB2=BX*BX+BY*BY+BZ*BZ
     RA=SQRT(RA2)
     RB=SQRT(RB2)
     IF(RA <= 0.1)  RA=PTONE
     RAR=ONE/RA
     IF(RB <= 0.1) RB=PTONE
     RBR=ONE/RB
     AXR=AX*RAR
     AYR=AY*RAR
     AZR=AZ*RAR
     BXR=BX*RBR
     BYR=BY*RBR
     BZR=BZ*RBR
     CT=AXR*BXR+AYR*BYR+AZR*BZR
     IF (CT >   ONE ) CT= ONE
     IF (CT < (-ONE)) CT=-ONE
     IF (PUMTYP(IPHI) == 1.AND. (.NOT.DNTRP) ) THEN
        VCONST = RLAMBDA*VPUMBS(IPHI)
     ELSEIF (PUMTYP(IPHI) == 2.AND. (.NOT.DNTPP) ) THEN
        VCONST = PLAMBDA*VPUMBS(IPHI)
     ELSE
        VCONST =VPUMBS(IPHI)
     ENDIF
     IF(PHASE) THEN
        EP = EP + VCONST*(ONE-CT*(FOUR*(CT*CT)-THREE))
     ELSE
        EP = EP + VCONST*(ONE+CT*(FOUR*(CT*CT)-THREE))
     ENDIF
  enddo
  RETURN
END SUBROUTINE PUMEPHI

SUBROUTINE SETEWEX(NNB14X,NNB14R,NNB14P)
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use bases_fcm
  use inbnd
  implicit none
  !
  !     Just sets the counters in the nonbond data structures.
  INTEGER NNB14X,NNB14R,NNB14P
  !     Stick the Ewald exclusion list counters into the data structures.
  !     Make sure we have the current counters in the common block.
  CALL GETBND(BNBND,.TRUE.)
  NNB14 = NNB14R
  CALL SETBND(BNBNDR)
  NNB14 = NNB14P
  CALL SETBND(BNBNDP)
  NNB14 = NNB14X
  CALL SETBND(BNBND)
  RETURN
END SUBROUTINE SETEWEX

SUBROUTINE UPDNIM(NATOM,INBLO1,NIM1,INBLO2,NIM2)
  !
  use chm_kinds
  implicit none
  INTEGER NATOM
  INTEGER INBLO1(NATOM),NIM1
  INTEGER INBLO2(NATOM),NIM2
  !     Just gets where last element of inblo type arrays point to
  !     which is the size of the used part of the equivalent jnb type
  !     array.  This is used because the image data structure still has
  !     NIMNBX, NIMNBS etc in the data structure (bimag) itself while
  !     the primary atom non-bond structure doesnt (bnbnd).
  !     Needed by NNLST2 and NNLST3.
  NIM1 = INBLO1(NATOM)
  NIM2 = INBLO2(NATOM)
  return
end SUBROUTINE UPDNIM

#endif 
SUBROUTINE NULL_TE
  RETURN
END SUBROUTINE NULL_TE

