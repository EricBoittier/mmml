#if KEY_TSM==1
SUBROUTINE ICFCNF(DX,DY,DZ,X,Y,Z,AMASS)
  !-----------------------------------------------------------------------
  !     This routine iteratively projects out forces in the directions
  !     of the constrained coordinates in ICFIX.
  !     Based on ICFCNS by D. Tobias.
  !     N.B. 1) If SHAKE constraints are also present, this is overkill,
  !     since the routine is called at every SHAKE iteration.
  !     2) Convergence criterion based on TOLGRD minimization parameter,
  !     like in SHAKEF.
  !
  !     K. Kuczera, Lawrence, KS 13-Jul-1993
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use stream
  use number
  use icfix
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  real(chm_real) AMASS(*)
  real(chm_real) XTMP(4),YTMP(4),ZTMP(4)
  real(chm_real) SX,SY,SZ,S2,SP,SF2,PX,PY,PZ,SS
  real(chm_real) AMASST(4),EPSS,FACTF
  INTEGER I,IIC,ICTYPE,ICNT,II
  INTEGER NUSED,NITER
  LOGICAL DONE
  DATA EPSS,FACTF/1.0D-12,1.0D-3/
  !
  IF(NICF == 0) RETURN
  DO IIC=1,NICF
     ICFADJ(IIC)=.TRUE.
  END DO
  NITER=0

  !
  ! ... Loop over iterations
  !
  done = .false.
  loop20: do while(.NOT.DONE)
     NUSED=0
     NITER=NITER+1
     IF(NITER > MAXI) THEN
        CALL WRNDIE(-4,'<ICFCNF>',' Iteration limit exceeded')
     END IF
     !
     ! ... Loop over ICFIX constraints: IIC=1,NICF
     !
     done = .true.
     DO IIC=1,NICF
        !
        ! ... Define coordinate type and atoms involved
        !
        !       WRITE(OUTU,800) NITER,IIC
        ! 800   FORMAT(1X,' Iteration and coordinate = ',2I4)
        !
        ICTYPE=ICFTYP(IIC)
        ICNT=ICTYPE+1
        !
        !JG052203   Reaction Coordinate specification
        !           RC = R1-R2
        IF(ICTYPE == 4) ICNT = 3
        !
        NUSED=NUSED+ICNT
        DO I=1,ICNT
           II=ICFATN(I,IIC)
           XTMP(I)=X(II)
           YTMP(I)=Y(II)
           ZTMP(I)=Z(II)
           AMASST(I)=AMASS(II)
        END DO
        !
        !     Get the s vectors.
        !
        CALL WILSON(XTMP,YTMP,ZTMP,SMF,ICTYPE,NUSED)
        !
        ! ... Calculate projection of force on s vectors
        !
        SP = ZERO
        S2 = ZERO
        SF2= ZERO
        NUSED=NUSED-ICNT
        DO I=1,ICNT
           NUSED=NUSED+1
           II=ICFATN(I,IIC)
           SX=SMF(1,NUSED)
           SY=SMF(2,NUSED)
           SZ=SMF(3,NUSED)
           PX = DX(II)*SX
           PY = DY(II)*SY
           PZ = DZ(II)*SZ
           SP=SP + PX + PY + PZ
           S2=S2 + SX**2 + SY**2 + SZ**2
           SF2=SF2 + DX(II)**2 + DY(II)**2 + DZ(II)**2
        END DO
        !
        !       WRITE(OUTU,990) NITER,IIC,SP,S2,Sf2
        ! 990   FORMAT(1X,' ---F-test: NITER, IIC, SP, S2, SP2 =',2I4,3E12.4)
        !
        IF(S2 < EPSS) THEN
           CALL WRNDIE(-4,'<ICFCNF>',' S-vector norm too low')
        END IF
        !
        SS = SQRT(S2*SF2)*TOLI(IIC)*FACTF
        IF(ABS(SP) < SS) ICFADJ(IIC) = .FALSE.
        DONE=(DONE.AND.(.NOT.ICFADJ(IIC)))
        SP=SP/S2
        !
        !
        ! ... Subtract parallel contribution
        !
        NUSED=NUSED-ICNT
        DO I=1,ICNT
           NUSED=NUSED+1
           II=ICFATN(I,IIC)
           SX=SMF(1,NUSED)
           SY=SMF(2,NUSED)
           SZ=SMF(3,NUSED)
           DX(II)=DX(II)-SP*SX
           DY(II)=DY(II)-SP*SY
           DZ(II)=DZ(II)-SP*SZ
        END DO
     END DO ! IIC=1,NICF
     !
     ! ... End of loop over constraints
     ! ... Check convergence
     !
     !     DONE = .TRUE.
     !     DO IIC=1,NICF
     !        DONE=(DONE.AND.(.NOT.ICFADJ(IIC)))
     !        ! print *,iic,done,.not.icfadj(iic)
     !     END DO
     !     print *,niter,done
     !
     !
  enddo loop20
  !
  ! ... Iteration finished, communicate result to common block
  !
  ANYADJ = .NOT. DONE
  !
  IF(PRNLEV >= 6) WRITE(OUTU,900) NITER
900 FORMAT(' ICFCNF: Internal coordinate forces reset in', I4, &
       ' iterations')
  !
  !
  RETURN
END SUBROUTINE ICFCNF

SUBROUTINE DYNICT(X,Y,Z,NATOM,NPRIV,AKMATI,TOTE,TOTKE, &
     DX0,DY0,DZ0,DXP,DYP,DZP,DXF,DYF,DZF,XSAVE,YSAVE,ZSAVE)
  !
  !     This is an analogue of icpert:DYNICP
  !     For thermodynamic integration conformational free energy method
  !
  !     Author: Krzysztof Kuczera  Lawrence, KS 02-Nov-1993
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  use stream
  use deriv
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  !...these arrays are stored on 
  real(chm_real) DX0(*),DY0(*),DZ0(*)
  real(chm_real) DXP(*),DYP(*),DZP(*)
  real(chm_real) DXF(*),DYF(*),DZF(*)
  real(chm_real) XSAVE(*),YSAVE(*),ZSAVE(*)
  !
  real(chm_real) TOTE,TOTKE,AKMATI
  real(chm_real) ESBNP,ESBFP,ESBRP
  real(chm_real) SCALE,DSCALE
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) DUDI0,DUDIP,DUDIR
  real(chm_real) ICVAL(3,MXICP)
  CHARACTER(LEN=80) ERRLIN
  INTEGER NATOM,NPRIV,ICP
  INTEGER I,J,K,L,IC,ITYPE,INC,ICOL,NUSED
  LOGICAL LPRINT
  !
  LPRINT=(IUNICP > 0.AND.MOD(NPRIV,ISVICP) == 0)
  !
  IF(LPRINT) THEN
     !
     !     Get the interaction energy for the unperturbed ic's.
     !
     CALL EIICT(ESBNP,X,Y,Z,DX0,DY0,DZ0)
     !
     !     Get unperturbed derivative using total unperturbed force
     !
     CALL ECNFTI(X,Y,Z,DX,DY,DZ,DUDI0,NATOM)
     !
     ICOL=1
     !
     ! ... Get values of the unperturbed coordinates
     !
     DO IC=1,NICP
        I=ICPATN(1,IC)
        J=ICPATN(2,IC)
        K=ICPATN(3,IC)
        L=ICPATN(4,IC)
        CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
        IF(ICPTYP(IC) == 1) ICVAL(ICOL,IC)=RIJ
        IF(ICPTYP(IC) == 2) ICVAL(ICOL,IC)=TIJK
        IF(ICPTYP(IC) == 3) ICVAL(ICOL,IC)=PIJKL
        IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
           IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
                'Unknown ic type for pert. no. ',IC,'. Programmer error?'
           CALL WRNDIE(-4,'<DYNICT>',ERRLIN)
        END IF
     END DO
     !
     IF(PRNLEV >= 2) THEN
        WRITE(IUNICP,100) NPRIV,AKMATI,TOTE,TOTKE,ESBNP,DUDI0
     END IF
100  FORMAT(I7,F10.4,4D16.8)
     !
     !     Save the coordinates of the atoms which move with the perturbation
     !
     DO IC=1,NICP
        CALL SAVICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
        IF(LMOV2(IC)) THEN
           CALL SAVICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
        END IF
     END DO
     !
     !     Get the perturbation data for each increment.
     !     Loop over perturbations in + and - directions
     !
     SCALE=ZERO
     DSCALE=ONE/ICPINC
     DO INC=1,ICPINC
        SCALE=SCALE+DSCALE
        !
        !     Move the atoms for the forward (+dzeta) perturbation.
        !
        CALL MVICP(X,Y,Z,ONE,SCALE)
        !
        !     Get the interaction energy for the forward perturbation.
        !
        CALL EIICT(ESBFP,X,Y,Z,DXP,DYP,DZP)
        !
        !    Get derivative using total force in perturbed configuration
        !
        DO I=1,NATOM
           DXF(I)=DX(I)-DX0(I)+DXP(I)
           DYF(I)=DY(I)-DY0(I)+DYP(I)
           DZF(I)=DZ(I)-DZ0(I)+DZP(I)
        END DO
        CALL ECNFTI(X,Y,Z,DXF,DYF,DZF,DUDIP,NATOM)
        !
        ICOL=2
        !
        ! ... Get values of the perturbed coordinates
        !
        DO IC=1,NICP
           I=ICPATN(1,IC)
           J=ICPATN(2,IC)
           K=ICPATN(3,IC)
           L=ICPATN(4,IC)
           CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
           IF(ICPTYP(IC) == 1) ICVAL(ICOL,IC)=RIJ
           IF(ICPTYP(IC) == 2) ICVAL(ICOL,IC)=TIJK
           IF(ICPTYP(IC) == 3) ICVAL(ICOL,IC)=PIJKL
           IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
              IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
                   'Unknown ic type for pert. no. ',IC,'. Programmer error?'
              CALL WRNDIE(-4,'<DYNICT>',ERRLIN)
           END IF
        END DO
        !
        !
        !     Restore the unperturbed coordinates.
        !
        DO IC=1,NICP
           CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
           IF(LMOV2(IC)) THEN
              CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
           END IF
        END DO
        !
        !     Move the atoms for the reverse (-dzeta) perturbation.
        !
        CALL MVICP(X,Y,Z,MINONE,SCALE)
        !
        !     Get the interaction energy for the reverse perturbation and write
        !     out the energies.
        !
        CALL EIICT(ESBRP,X,Y,Z,DXP,DYP,DZP)
        !
        !    Get derivative using total force in perturbed configuration
        !
        DO I=1,NATOM
           DXF(I)=DX(I)-DX0(I)+DXP(I)
           DYF(I)=DY(I)-DY0(I)+DYP(I)
           DZF(I)=DZ(I)-DZ0(I)+DZP(I)
        END DO
        CALL ECNFTI(X,Y,Z,DXF,DYF,DZF,DUDIR,NATOM)
        !
        IF(PRNLEV >= 2) THEN
           WRITE(IUNICP,101) SCALE,ESBFP,ESBRP,DUDIP,DUDIR
        END IF
101     FORMAT(7X,F10.4,4D16.8)
        !
        ICOL=3
        !
        ! ... Get values of the perturbed coordinates
        !
        DO IC=1,NICP
           I=ICPATN(1,IC)
           J=ICPATN(2,IC)
           K=ICPATN(3,IC)
           L=ICPATN(4,IC)
           CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
           IF(ICPTYP(IC) == 1) ICVAL(ICOL,IC)=RIJ
           IF(ICPTYP(IC) == 2) ICVAL(ICOL,IC)=TIJK
           IF(ICPTYP(IC) == 3) ICVAL(ICOL,IC)=PIJKL
           IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
              IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
                   'Unknown ic type for pert. no. ',IC,'. Programmer error?'
              CALL WRNDIE(-4,'<DYNICT>',ERRLIN)
           END IF
           !
           IF(PRNLEV >= 2) WRITE(IUNICP,102) IC,ICPTYP(IC),ICVAL(1,IC), &
                ICVAL(2,IC),ICVAL(3,IC)
        END DO
        !
102     FORMAT(9X,2I4,3D16.8)
        !
        !     Restore the unperturbed coordinates.
        !
        DO IC=1,NICP
           CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
           IF(LMOV2(IC)) THEN
              CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
           END IF
        END DO
        !
        ! ... End of loop
        !
     END DO ! (INC)
  END IF ! IF(LPRINT)
  !
  RETURN
END SUBROUTINE DYNICT

SUBROUTINE DYNICM(X,Y,Z,NATOM,NPRIV,AKMATI,TOTE,TOTKE)
  !
  !     This is an analogue of icfcnf:DYNICT
  !     For thermodynamic integration conformational free energy method
  !     Evaluation of energy derivatives for a set of elementary internal
  !     coordinates (vs. one arbitrarily complicated coordinate in DYNICT)
  !     This routine does only TI, TP part thrown out for simplicity.
  !
  !     If gradient code is modified, make DYNICB consistent
  !
  !     Author: Krzysztof Kuczera  Lawrence, KS 24-Mar-1995
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  use stream
  use deriv
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  !
  real(chm_real) TOTE,TOTKE,AKMATI
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) ICVAL(MXICP),DUDI(MXICP)
  CHARACTER(LEN=80) ERRLIN
  INTEGER NATOM,NPRIV,ICP
  INTEGER I,J,K,L,IC,ITYPE,INC,ICOL
  LOGICAL LPRINT
  real(chm_real) DZET,DUDIX
  !
  !
  ! ... If output unit is open and MD step is multiple of save frequency
  !
  LPRINT=(IUNICP > 0.AND.MOD(NPRIV,ISVICP) == 0)
  !
  IF(LPRINT) THEN
     !
     ! ...  Get gradient of potential with respect to the set of NICP
     ! ... "perturbation coordinates". These coordinates must also be
     ! ...  fixed using ICFIX. They must be "elementary" -- individual
     ! ...  bonds, angles, dihedrals (at this stage)
     !
     ! ... Loop over set of internal coordinates
     ! ... DUDI(ICP) contains the derivative of the potential energy
     ! ... wrt the internal coordinate no. ICP
     !
     DO ICP=1,NICP
        !
        DUDIX=ZERO
        !
        IF(LMOV2(ICP)) THEN
           DZET=ONE/TWO
           CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
                ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
           !
           DZET=-DZET
           CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV2, &
                ICPMV2(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
           DUDI(ICP)=DUDIX
           !
        ELSE
           DZET=ONE
           CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
                ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
           DUDI(ICP)=DUDIX
        END IF
        !
     END DO
     !
     !
     IF(PRNLEV >= 2 .AND. .NOT. QICWR) THEN
        !
        ! ... Get values of the "perturbation" coordinates
        !
        DO IC=1,NICP
           I=ICPATN(1,IC)
           J=ICPATN(2,IC)
           K=ICPATN(3,IC)
           L=ICPATN(4,IC)
           CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
           IF(ICPTYP(IC) == 1) ICVAL(IC)=RIJ
           IF(ICPTYP(IC) == 2) ICVAL(IC)=TIJK
           IF(ICPTYP(IC) == 3) ICVAL(IC)=PIJKL
           IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
              IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
                   'Unknown ic type for pert. no. ',IC,'. Programmer error?'
              CALL WRNDIE(-4,'<DYNICM>',ERRLIN)
           END IF
        END DO
        !
        ! ... Output section
        !
        ! ... #1. Write coordinate values once to save space
        !
        WRITE(IUNICP,100) (ICVAL(IC), IC=1,NICP)
        QICWR=.TRUE.
        !
     ENDIF ! (PRNLEV >= 1 .AND..NOT.QICWR)
     !
     ! ... #2. Write step number, time, total and kinetic energy
     !
     IF(PRNLEV >= 2) WRITE(IUNICP,101) NPRIV,AKMATI,TOTE,TOTKE
     !
     ! ... #3. Write potential energy gradient
     !
     IF(PRNLEV >= 2) WRITE(IUNICP,100) (DUDI(IC),IC=1,NICP)
     !
100  FORMAT(5D16.8)
101  FORMAT(I7,F10.4,2D16.8)
     !
  END IF ! IF(LPRINT)
  !
  !
  RETURN
END SUBROUTINE DYNICM

SUBROUTINE GTICVL(I,J,K,L,ICOL,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z, &
     ICVAL)
  !
  ! ... Get values of the coordinates  undergoing perturbation
  ! ... K. Kuczera, Lawrence, KS 02-Nov-1993
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  use stream
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  INTEGER I,J,K,L,IC,ICOL
  INTEGER ICVAL(3,MXICP)
  CHARACTER(LEN=80) ERRLIN
  !
  ! ... Get values of the coordinates  undergoing perturbation
  !
  DO IC=1,NICP
     I=ICPATN(1,IC)
     J=ICPATN(2,IC)
     K=ICPATN(3,IC)
     L=ICPATN(4,IC)
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     IF(ICPTYP(IC) == 1) ICVAL(ICOL,IC)=RIJ
     IF(ICPTYP(IC) == 2) ICVAL(ICOL,IC)=TIJK
     IF(ICPTYP(IC) == 3) ICVAL(ICOL,IC)=PIJKL
     IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
        IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
             'Unknown ic type for pert. no. ',IC,'. Programmer error?'
        CALL WRNDIE(-4,'<GTICVL>',ERRLIN)
     END IF
  END DO
  !
  RETURN
END SUBROUTINE GTICVL

SUBROUTINE ECNFTI(X,Y,Z,DX,DY,DZ,DUDIX,NATOM)
  !
  !     This is a dummy routine to call MVICP0
  !
  !     Author: K. Kuczera Lawrence, KS 03-Nov-1993
  !             based on MOVICP
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) DZET,DUDIX
  INTEGER ICP, NATOM
  !
  DUDIX=ZERO
  !
  !     Loop over perturbations.
  !
  DO ICP=1,NICP
     !
     IF(LMOV2(ICP)) THEN
        DZET=ONE/TWO
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
        !
        DZET=-DZET
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV2, &
             ICPMV2(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
        !
     ELSE
        DZET=ONE
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
     END IF
     !
  END DO
  !
  RETURN
END SUBROUTINE ECNFTI

SUBROUTINE MVICP0(ICP,ICPTYP,ICPATN,NMVICP, &
     ICPMVA,DZETA,DUDIX,X,Y,Z,DX,DY,DZ)
  !
  !     This routine calculates (dx_i/dr) - the derivatives of
  !     the cartesian coordinates wrt to the perturbation
  !     and evaluates the scalar product with the forces
  !
  !     Author: K.Kuczera, Lawrence, KS 03-Nov-1993
  !             Based on MVICP2
  !
  use chm_kinds
  use number
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) DZETA,DUDIX
  real(chm_real) XIJ,YIJ,ZIJ,XJK,YJK,ZJK,XJL,YJL,ZJL,XJN,YJN,ZJN
  real(chm_real) DRX,DRY,DRZ,NX,NY,NZ,NORM
  real(chm_real) RADI
  real(chm_real) A(3,3),DPHI
  INTEGER ICPTYP(*),ICPATN(4,*),NMVICP(*),ICPMVA(*)
  INTEGER I,J,K,L,M,N,ICP,ISTOP
  CHARACTER(LEN=80) ERRLIN
  !
  RADI=ACOS(MINONE)/ONE8TY
  ISTOP=NMVICP(ICP)
  !
  IF (ICPTYP(ICP) == 1) THEN
     I=ICPATN(1,ICP)
     J=ICPATN(2,ICP)
     XIJ=X(J)-X(I)
     YIJ=Y(J)-Y(I)
     ZIJ=Z(J)-Z(I)
     NORM=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
     NORM=DZETA/NORM
     DRX=NORM*XIJ
     DRY=NORM*YIJ
     DRZ=NORM*ZIJ
     !
     DO M=1,ISTOP
        N=ICPMVA(M)
        DUDIX=DUDIX+DRX*DX(N)
        DUDIX=DUDIX+DRY*DY(N)
        DUDIX=DUDIX+DRZ*DZ(N)
     END DO
     !
  END IF
  !
  IF(ICPTYP(ICP) == 2) THEN
     I=ICPATN(1,ICP)
     J=ICPATN(2,ICP)
     K=ICPATN(3,ICP)
     XIJ=X(I)-X(J)
     YIJ=Y(I)-Y(J)
     ZIJ=Z(I)-Z(J)
     XJK=X(K)-X(J)
     YJK=Y(K)-Y(J)
     ZJK=Z(K)-Z(J)
     NX=YIJ*ZJK-YJK*ZIJ
     NY=XJK*ZIJ-XIJ*ZJK
     NZ=XIJ*YJK-XJK*YIJ
     NORM=SQRT(NX*NX+NY*NY+NZ*NZ)
     NX=NX/NORM
     NY=NY/NORM
     NZ=NZ/NORM
     !
     CALL AROT0(NX,NY,NZ,DZETA,A)
     DO M=1,ISTOP
        N=ICPMVA(M)
        XJN=X(N)-X(J)
        YJN=Y(N)-Y(J)
        ZJN=Z(N)-Z(J)
        DUDIX=DUDIX+(A(1,1)*XJN+A(1,2)*YJN+A(1,3)*ZJN)*DX(N)
        DUDIX=DUDIX+(A(2,1)*XJN+A(2,2)*YJN+A(2,3)*ZJN)*DY(N)
        DUDIX=DUDIX+(A(3,1)*XJN+A(3,2)*YJN+A(3,3)*ZJN)*DZ(N)
     END DO
     !
  END IF
  !
  IF(ICPTYP(ICP) == 3) THEN
     J=ICPATN(2,ICP)
     K=ICPATN(3,ICP)
     L=ICPATN(4,ICP)
     XJK=X(K)-X(J)
     YJK=Y(K)-Y(J)
     ZJK=Z(K)-Z(J)
     NORM=SQRT(XJK*XJK+YJK*YJK+ZJK*ZJK)
     NX=XJK/NORM
     NY=YJK/NORM
     NZ=ZJK/NORM
     CALL AROT0(NX,NY,NZ,DZETA,A)
     DO M=1,ISTOP
        N=ICPMVA(M)
        XJN=X(N)-X(J)
        YJN=Y(N)-Y(J)
        ZJN=Z(N)-Z(J)
        DUDIX=DUDIX+(A(1,1)*XJN+A(1,2)*YJN+A(1,3)*ZJN)*DX(N)
        DUDIX=DUDIX+(A(2,1)*XJN+A(2,2)*YJN+A(2,3)*ZJN)*DY(N)
        DUDIX=DUDIX+(A(3,1)*XJN+A(3,2)*YJN+A(3,3)*ZJN)*DZ(N)
     END DO
     !
  END IF
  !
  !
  RETURN
END SUBROUTINE MVICP0


SUBROUTINE AROT0(NX,NY,NZ,DZETA,A)
  !
  !     This routine finds a matrix generating an infinitesimal
  !     rotation around axis n=[nx,ny,nz], scaled by DZETA
  !
  !     Author: Krzysztof Kuczera, Lawrence, KS 03-Nov-1993
  !
  use chm_kinds
  use number
  implicit none
  !
  real(chm_real) NX,NY,NZ,DZETA
  real(chm_real) A(3,3)
  !
  A(1,1)= ZERO
  A(1,2)=-NZ*DZETA
  A(1,3)= NY*DZETA
  A(2,1)= NZ*DZETA
  A(2,2)= ZERO
  A(2,3)=-NX*DZETA
  A(3,1)=-NY*DZETA
  A(3,2)= NX*DZETA
  A(3,3)= ZERO
  !
  !
  RETURN
END SUBROUTINE AROT0


SUBROUTINE BLCFTI(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Use trajectory files generated by CNFTI (Thermodynamic
  !     Integration of conformational free energy) to calculate
  !     some decompositions.
  !     In the simplest case I would envision fixing the solvent
  !     and evaluating contributions from potential energy terms
  !     and/or regions of system using "SKIPE" or "BLOCK" or both.
  !
  !     An additional option is the possibility of decomposing
  !     all contributions into dE -TdS.
  !     Based on M. Mezei & D.L.Beveridge, Ann.N.Y.Acad.Sci 482
  !    (1986), 1-23,
  !    -T*(dS(x)/dx) = beta*[<E(x)*(dE(x)/dx)> - <E(x)>*<(dE(x)/dx)>]
  !     where x=conformational parameter and averages are over E(x).
  !     This makes sense only for the total energy, strictly speaking.
  !
  !     A: Free energy dA/dx = < dE/dx >
  !     B: Entropy: -T*(dS/dx)
  !                = beta*[< E*(dE/dx)> - <E>*<(dE/dx) >]
  !                = beta* < (E - <E>) (dE/dx - <dE/dx>) >
  !     C: Internal energy dU/dx
  !                = dA/dx - ( -T * (dS/dx) )
  !                = < dE/dx - beta * (E - <E>) (dE/dx - <dE/dx>) >
  !
  !     Error estimates:
  !     1. Standard error propagation formula, for time series with
  !        no autocorrelation and cross-correlation limited to
  !        covariance <(Xi*Yi)>-<X><Y>.
  !        N.B. the covariance of the mean is used as estimate of
  !        error, this is 1/N times the covariance of the time series.
  !     2. For NCONT /= 0 the statistical errors are calculated from
  !        the variance of sub-averages, found from trajectory
  !        fragments of NCONT data points.
  !
  !     K.Kuczera, Lawrence, KS 09-Nov-1993
  !
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                     
#endif

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use memory
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use cvio
  use deriv
  use energym
  use psf
  use icpert

  implicit none
  real(chm_real),allocatable,dimension(:) :: ipxsave, ipysave, ipzsave
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
  INTEGER ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
  INTEGER IHBFRX,INBFRX,ILBFRX,IMGFRX
  INTEGER I,BCNT
  real(chm_real) TEMP,BETA,BLNUM
  !
  real(chm_real) YTOTE,ESUM,BESUM,EFLU,TEMPY,DEDI
  real(chm_real) SSUM,BSSUM,SFLU
  real(chm_real) USUM,UFLU,BUFLU
  real(chm_real) DE2E,DEE2,BSFLU
  real(chm_real) DADI,AFLU,BDADI,BAFLU
  real(chm_real) BBS, BBU
  !
  real(chm_real) SUM,E2E,EE2,DELTA,S0,S1
  !
  real(chm_real) BDATA
  real(chm_real) VALIC0
  CHARACTER(LEN=4) :: HDRC='CORD', HDRD='VELD'
  !
  call chmalloc('icfcnf.src','BLCFTI','itemp',natom,cr4=itemp)
  call chmalloc('icfcnf.src','BLCFTI','ifreat',natom,intg=ifreat)
  call chmalloc('icfcnf.src','BLCFTI','ipxsave',natom,crl=ipxsave)
  call chmalloc('icfcnf.src','BLCFTI','ipysave',natom,crl=ipysave)
  call chmalloc('icfcnf.src','BLCFTI','ipzsave',natom,crl=ipzsave)

  !
  ! Finish processing command line
  !
  NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
  CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
  TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
  !
  ! Initialize variables
  !
  BETA=ONE/(KBOLTZ*TEMP)
  BLNUM=ZERO
  BCNT=0
  BDATA=ZERO
  ESUM =ZERO
  EFLU =ZERO
  BESUM=ZERO
  BBU = ZERO
  BBS = ZERO
  !
  DADI=ZERO
  AFLU=ZERO
  BDADI=ZERO
  BAFLU=ZERO
  SUM =ZERO
  SFLU =ZERO
  SSUM=ZERO
  SFLU=ZERO
  BSSUM=ZERO
  BSFLU=ZERO
  BUFLU=ZERO
  E2E =ZERO
  EE2 =ZERO
  DE2E =ZERO
  DEE2 =ZERO
  !
  ISTATS=1
  NAT=NATOM
  IUNIT=NFU
  NFREAT=0
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  Analysis of CFTI conformational free energy'
  WRITE(OUTU,*) '    Thermodynamic integration formula '
  WRITE(OUTU,*)
  !
  !  Loop:
  !  Read in coordinates from a simulation at x_i=const
  !  Update energy lists as needed
  !  Calculate energy and forces
  !  Accumulate averages and fluctuations of them
  !
  !
570 CONTINUE
  CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
       CG,QCG,                                    & 
#endif
       ITEMP,NAT,IFREAT,NFREAT,NFU, &
       NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
       DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
  !
  ! Update energy lists as needed (LCODES=.FALSE.)
  !
  CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE., &
       .FALSE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])
  !
  ! Call ENERGY: call gives total potential energy in EPROP(EPOT)
  !              and gradient DX,DY,DZ
  !
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
  !
  ! Pass forces to DYNICA, which returns the gradient dE(x)/dx_i
  ! in [DEDI] and values of coordinates x in VALIC0
  !
  !kk...10-FEB-98      CALL DYNICB(X,Y,Z,DX,DY,DZ,VALIC0,DEDI,NATOM)
  CALL DYNICA(X,Y,Z,AMASS,DX,DY,DZ,VALIC0,DEDI,NATOM, &
       IPXSAVE,IPYSAVE,IPZSAVE)
  !
  YTOTE = EPROP(EPOT)
  !
  ! Accumulate values and squares; store data for statistics
  !
  BLNUM=BLNUM+ONE
  !
  ESUM = ESUM + YTOTE
  EFLU = EFLU + YTOTE*YTOTE
  DADI=DADI+DEDI
  AFLU=AFLU+DEDI**2
  TEMPY  =YTOTE*DEDI
  SSUM=SSUM+TEMPY
  SFLU=SFLU+TEMPY*TEMPY
  DE2E=DE2E+TEMPY*DEDI
  DEE2=DEE2+TEMPY*YTOTE
  !
  !
  IF(PRNLEV >= 7 .AND. NCONT /= 0 .AND.INT(BLNUM) == 1) THEN
     WRITE(OUTU,550)
  END IF
  IF(PRNLEV >= 7) THEN
     WRITE(OUTU,551) INT(BLNUM),BCNT+1,ISTEP,VALIC0,YTOTE,DEDI
  END IF
  !
550 FORMAT(5X,'Frame',2X,'Bin',3X,'Step',8X,' x ',11X, &
       'U',16X,' dU/dx ')
551 FORMAT(1X,I7,I6,I7,4X,F12.6,2F16.8)
  !
  ! Block averages:
  ! Calculate average values of dU,dA,and TdS over blocks of
  ! NCONT data points. Use the variance of the averages to estimate
  ! statistical errors.
  !
  BDATA=BDATA+ONE
  BESUM=BESUM+YTOTE
  BDADI=BDADI+DEDI
  BSSUM=BSSUM+YTOTE*DEDI
  !
  IF(MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
     IF(INT(BDATA) /= NCONT) THEN
        WRITE(OUTU,*) ' BLCFTI> Warning: BDATA =',INT(BDATA), &
             '  /=  NCONT =',NCONT,' for block #',BCNT+1
     END IF
     BCNT=BCNT+1
     BESUM=BESUM/BDATA
     BDADI=BDADI/BDATA
     BAFLU=BAFLU+BDADI**2
     BSSUM=BETA*(BSSUM/BDATA-BESUM*BDADI)
     BBS = BBS + BSSUM
     BSFLU=BSFLU + BSSUM**2
     TEMPY = BDADI-BSSUM
     BBU = BBU + TEMPY
     BUFLU=BUFLU + TEMPY*TEMPY
     BSSUM=ZERO
     BDADI=ZERO
     BESUM=ZERO
     BDATA=ZERO
     !
  ENDIF
  !
  IF (ISTATS  >=  0) GOTO 570

  call chmdealloc('icfcnf.src','BLCFTI','itemp',natom,cr4=itemp)
  call chmdealloc('icfcnf.src','BLCFTI','ifreat',natom,intg=ifreat)

  call chmdealloc('icfcnf.src','BLCFTI','ipxsave',natom,crl=ipxsave)
  call chmdealloc('icfcnf.src','BLCFTI','ipysave',natom,crl=ipysave)
  call chmdealloc('icfcnf.src','BLCFTI','ipzsave',natom,crl=ipzsave)

  !
  IF (INT(BLNUM) <= 0) CALL WRNDIE(-3,'<BLCFTI>','No data.')
  !
  ! End of reading loop; calculate overall averages and fluctuations.
  ! Standard errors, assuming no correlations
  !
  IF (NCONT  >=  0) THEN
     S0=SQRT(BLNUM)
     S1=SQRT(DBLE(BCNT))
     !
     !     Block sub-average error estimates
     !     DADI -> x;        ESUM -> y;        SSUM -> x*y;
     !     AFLU -> x^2;        EFLU -> y^2;        SFLU -> (x*y)^2;
     !        DE2E -> x^2 * y;        DEE2 -> x * y^2
     !                    dA/di = < x >;
     !             -TdS/di = beta * ( <x*y> - <x> <y> )
     !                      = beta * < (x-<x>)*(y-<y>) >
     !                dU/di = < x > - beta * ( <x*y> - <x> <y> )
     !                      = < x -  beta*(x-<x>)*(y-<y>)>
     !
     !  A = < (x-<x>)^2 * (y-<y>)^2 >
     !    = <(x*y)^2> - 2<y*x^2><y> - 2<x*y^2><x> - 3<x>^2<y>^2
     !      + <x^2><y>^2 + <x>^2<y^2> + 4<x*y><x><y>
     !    = <(x*y)^2> + 2 * ( <x>*C - D ) * <y> - F
     !  F = 2<x*y^2><x> - <y^2><x>^2 + B * <y>^2
     !  B = <x^2> - <x>^2
     !  C = <x*Y> - <x><y>
     !  D = <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y>
     !    = <y*x^2> - <x^2><y> - <x> * C
     !  E = < (x - a(x-<x>)(y-<y>))^2 >
     !    = <x^2> - 2a ( <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y> ) + A*a^2
     !    = <x^2> - 2a * D + A * a^2
     !
     !  TEMPY  -> <x^2>
     !  AFLU   -> B
     !  SSUM   -> C
     !  DE2E   -> D
     !  DEE2   -> F
     !  SFLU   -> A
     !
     ESUM = ESUM/BLNUM
     EFLU = EFLU/BLNUM
     DADI = DADI/BLNUM
     TEMPY   = AFLU/BLNUM
     AFLU = TEMPY - DADI**2
     SSUM = SSUM/BLNUM - DADI*ESUM
     DE2E = DE2E/BLNUM - DADI*SSUM - TEMPY*ESUM
     DEE2 = 2 * DADI * DEE2/BLNUM
     DEE2 = DEE2 - EFLU * DADI**2 + AFLU * ESUM**2
     SFLU = SFLU/BLNUM - DEE2
     SFLU = SFLU + 2 * (DADI*SSUM - DE2E) * ESUM
     !
     !  UFLU   -> SQRT(E - USUM^2) / S0
     !  AFLU   -> SQRT(AFLU) / S0
     !  SFLU   -> beta * SQRT ( A - C^2 ) /S0
     !
     UFLU = SQRT(TEMPY-2*BETA*DE2E+SFLU*BETA*BETA)/S0
     AFLU = SQRT(AFLU) / S0
     SFLU = BETA * SQRT (SFLU - SSUM**2) / S0
     SSUM = BETA * SSUM
     USUM = DADI - SSUM
     !
     !  Block error:
     !
     BBS = BBS/BCNT
     BBU = BBU/BCNT
     BAFLU=SQRT(BAFLU/BCNT - DADI**2) / S1
     BSFLU=SQRT(BSFLU/BCNT - BBS**2) / S1
     BUFLU=SQRT(BUFLU/BCNT - BBU**2) / S1
     !
     WRITE(OUTU,*) ' Total number of data points was',INT(BLNUM)
     WRITE(OUTU,*) ' There were',BCNT,' data sub-averages.'
     WRITE(OUTU,*)
     IF(BCNT <= 0) CALL WRNDIE(-3,'<BLCFTI>', &
          'Zero data sub-averages found.')
     IF(BCNT < 6) WRITE(OUTU,*) ' BLCFTI> Warning: too few', &
          ' sub-averages, error estimates may be distorted'
     !
     !
     IF(PRNLEV >= 0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,580) TEMP
        WRITE(OUTU,*)
        WRITE(OUTU,590)
        WRITE(OUTU,*)
        WRITE(OUTU,600) VALIC0,DADI,AFLU,BAFLU
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Internal energy gradient at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,591)
        WRITE(OUTU,*)
        WRITE(OUTU,601) VALIC0,USUM,UFLU,BUFLU,BBU
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Gradient of -TS at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,592)
        WRITE(OUTU,*)
        WRITE(OUTU,602) VALIC0,SSUM,SFLU,BSFLU,BBS
     ENDIF
  ENDIF
580 FORMAT('  Free energy gradient at T = ',F10.2,' K, [kcal/mol]')
590 FORMAT(5X,5X,'x',7X,'dA/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)')
600 FORMAT(1X, F10.4,2X,3(E12.6,1X),'+++++dA+++')
591 FORMAT(5X,5X,'x',7X,'dU/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3x, 'dU/dx(block)')
592 FORMAT(5X,5X,'x',7X,'-TdS/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3x, '-TdS/dx(block)')
601 FORMAT(1X, F10.4,2X,4(E12.6,1X),'+++++dU+++')
602 FORMAT(1X, F10.4,2X,4(E12.6,1X),'+++-TdS+++')
  !
  RETURN
END SUBROUTINE BLCFTI

SUBROUTINE BLCFTJ(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Use ICP files generated by CNFTI (Thermodynamic
  !     Integration of conformational free energy) to calculate
  !     some decompositions.
  !     In the simplest case I would envision fixing the solvent
  !     and evaluating contributions from potential energy terms
  !     and/or regions of system using "SKIPE" or "BLOCK" or both.
  !
  !     An additional option is the possibility of decomposing
  !     all contributions into dE -TdS.
  !     Based on M. Mezei & D.L.Beveridge, Ann.N.Y.Acad.Sci 482
  !    (1986), 1-23,
  !    -T*(dS(x)/dx) = beta*[<E(x)*(dE(x)/dx)> - <E(x)>*<(dE(x)/dx)>]
  !     where x=conformational parameter and averages are over E(x).
  !     This makes sense only for the total energy, strictly speaking.
  !
  !     A: Free energy dA/dx = < dE/dx >
  !     B: Entropy: -T*(dS/dx)
  !                 = beta*[< E*(dE/dx)> - <E>*<(dE/dx) >]
  !                 = beta* < (E - <E>) (dE/dx - <dE/dx>) >
  !     C: Internal energy dU/dx
  !                 = dA/dx - ( -T * (dS/dx) )
  !                 = < dE/dx - beta * (E - <E>) (dE/dx - <dE/dx>) >
  !
  !     Error estimates:
  !     1. Standard error propagation formula, for time series with
  !        no autocorrelation and cross-correlation limited to
  !        covariance <(Xi*Yi)>-<X><Y>.
  !        N.B. the covariance of the mean is used as estimate of
  !        error, this is 1/N times the covariance of the time series.
  !     2. For NCONT /= 0 the statistical errors are calculated from
  !        the variance of sub-averages, found from trajectory
  !        fragments of NCONT data points.
  !
  !     K.Kuczera, Lawrence, KS 09-Nov-1993
  !
  use bases_fcm
  use number
  use consta
  use coord
  use ctitla
  use deriv
  use energym
  use psf
  use icpert
  use stream
  use string

  implicit none
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NCONT,ICPU
  INTEGER I,J,K,BCNT,NPERT
  real(chm_real) TEMP,BETA,BLNUM
  !
  real(chm_real) YTOTE,ESUM,BESUM,EFLU,TEMPY,DEDI
  real(chm_real) SSUM,BSSUM,SFLU
  real(chm_real) USUM,UFLU,BUFLU
  real(chm_real) DE2E,DEE2,BSFLU
  real(chm_real) DADI,AFLU,BDADI,BAFLU
  real(chm_real) BBS, BBU
  real(chm_real) JUNK,TOTEY,TOTKY,YTEMP,ETEMP
  !
  INTEGER NDEGF,ISTEP
  real(chm_real) SUM,E2E,EE2,DELTA,S0,S1
  !
  real(chm_real) BDATA
  real(chm_real) VALIC0
  !MFC ERROR ISTEP used before defined.
  !    very minor since it is only used in a print statement
  !    setting to -1 to avoid compiler warnings
  istep=-1
  !MFC
  !
  ! Finish processing command line
  !
  NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
  TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
  ICPU =GTRMI(COMLYN, COMLEN, 'UICP',     -1)
  !      NPERT=GTRMI(COMLYN, COMLEN, 'NPER',     1)
  WRITE(OUTU,*)
  WRITE(OUTU,*) ' The data are read from unit: ',ICPU
  !
  ! Initialize variables
  !
  BETA=ONE/(KBOLTZ*TEMP)
  BLNUM=ZERO
  BCNT=0
  BDATA=ZERO
  ESUM =ZERO
  EFLU =ZERO
  BESUM=ZERO
  BBU = ZERO
  BBS = ZERO
  !
  DADI=ZERO
  AFLU=ZERO
  BDADI=ZERO
  BAFLU=ZERO
  SUM =ZERO
  SFLU =ZERO
  SSUM=ZERO
  BSSUM=ZERO
  BSFLU=ZERO
  BUFLU=ZERO
  E2E =ZERO
  EE2 =ZERO
  DE2E =ZERO
  DEE2 =ZERO
  YTEMP=ZERO
  ETEMP=ZERO
  !
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  Analysis of CFTI conformational free energy'
  WRITE(OUTU,*) '    Thermodynamic integration formula '
  WRITE(OUTU,*)
  !
  ! ... Read first line
  READ(ICPU,*) I,J,NDEGF,JUNK
  NPERT=J
  !
  !
  !  Loop:
  !  Read in data
  !  Accumulate averages and fluctuations of them
  !
  !
570 CONTINUE
  !
  ! ... Read line with energies
  READ(ICPU,'(I7,F10.4,4D16.8)',END=571) I,JUNK,TOTEY,TOTKY, &
       TEMPY,DEDI
  !
  DO I=1,NPERT
     READ(ICPU,*)
     READ(ICPU,'(9X,2I4,3D16.8)') J,K,VALIC0,JUNK,TEMPY
  END DO
  YTOTE=TOTEY-TOTKY
  YTEMP = YTEMP + TOTKY
  ETEMP = ETEMP + TOTKY * TOTKY
  !
  ! Accumulate values and squares; store data for statistics
  !
  BLNUM=BLNUM+ONE
  !
  ESUM = ESUM + YTOTE
  EFLU = EFLU + YTOTE*YTOTE
  DADI=DADI+DEDI
  AFLU=AFLU+DEDI**2
  TEMPY  =YTOTE*DEDI
  SSUM=SSUM+TEMPY
  SFLU=SFLU+TEMPY*TEMPY
  DE2E=DE2E+TEMPY*DEDI
  DEE2=DEE2+TEMPY*YTOTE
  !
  !
  IF(PRNLEV >= 7 .AND. NCONT /= 0 .AND.INT(BLNUM) == 1) THEN
     WRITE(OUTU,550)
  END IF
  IF(PRNLEV >= 7) THEN
     WRITE(OUTU,551) INT(BLNUM),BCNT+1,ISTEP,VALIC0, &
          YTOTE,DEDI
  END IF
  !
550 FORMAT(5X,'Frame',2X,'Bin',3X,'Step',8X,' x ',11X, &
       'U',16X,' dU/dx')
551 FORMAT(1X,I7,I6,I7,4X,F12.6,2F16.8)
  !
  ! Block averages:
  ! Calculate average values of dU,dA,and TdS over blocks of
  ! NCONT data points. Use the variance of the averages to estimate
  ! statistical errors.
  !
  BDATA=BDATA+ONE
  BESUM=BESUM+YTOTE
  BDADI=BDADI+DEDI
  BSSUM=BSSUM+YTOTE*DEDI
  !
  IF(MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
     IF(INT(BDATA) /= NCONT) THEN
        WRITE(OUTU,*) ' BLCFTJ> Warning: BDATA =',INT(BDATA), &
             '  /=  NCONT =',NCONT,' for block #',BCNT+1
     END IF
     BCNT=BCNT+1
     BESUM=BESUM/BDATA
     BDADI=BDADI/BDATA
     BAFLU=BAFLU+BDADI**2
     BSSUM=BETA*(BSSUM/BDATA-BESUM*BDADI)
     BBS = BBS + BSSUM
     BSFLU=BSFLU + BSSUM**2
     TEMPY = BDADI-BSSUM
     BBU = BBU + TEMPY
     BUFLU=BUFLU + TEMPY*TEMPY
     BSSUM=ZERO
     BDADI=ZERO
     BESUM=ZERO
     BDATA=ZERO
     !
  END IF
  !
  GOTO 570
  !
571 CONTINUE
  IF (INT(BLNUM) <= 0) CALL WRNDIE(-3,'<BLCFTJ>','No data.')
  !
  ! End of reading loop; calculate overall averages and fluctuations.
  ! Standard errors, assuming no correlations
  !
  IF (NCONT  >=  0) THEN
     S0=SQRT(BLNUM)
     S1=SQRT(DBLE(BCNT))
     YTEMP = YTEMP/BLNUM
     ETEMP = SQRT(ETEMP/BLNUM - YTEMP*YTEMP)/S0
     TEMPY = 2/(KBOLTZ*NDEGF)
     YTEMP = YTEMP * TEMPY
     ETEMP = ETEMP * TEMPY
     !
     !        Block sub-average error estimates
     !     DADI -> x;        ESUM -> y;        SSUM -> x*y;
     !     AFLU -> x^2;        EFLU -> y^2;        SFLU -> (x*y)^2;
     !        DE2E -> x^2 * y;        DEE2 -> x * y^2
     !                    dA/di = < x >;
     !             -TdS/di = beta * ( <x*y> - <x> <y> )
     !                      = beta * < (x-<x>)*(y-<y>) >
     !                dU/di = < x > - beta * ( <x*y> - <x> <y> )
     !                      = < x -  beta*(x-<x>)*(y-<y>)>
     !
     !  A = < (x-<x>)^2 * (y-<y>)^2 >
     !    = <(x*y)^2> - 2<y*x^2><y> - 2<x*y^2><x> - 3<x>^2<y>^2
     !      + <x^2><y>^2 + <x>^2<y^2> + 4<x*y><x><y>
     !    = <(x*y)^2> + 2 * ( <x>*C - D ) * <y> - F
     !  F = 2<x*y^2><x> - <y^2><x>^2 + B * <y>^2
     !  B = <x^2> - <x>^2
     !  C = <x*Y> - <x><y>
     !  D = <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y>
     !    = <y*x^2> - <x^2><y> - <x> * C
     !  E = < (x - a(x-<x>)(y-<y>))^2 >
     !    = <x^2> - 2a ( <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y> ) + A*a^2
     !    = <x^2> - 2a * D + A * a^2
     !
     !  TEMPY  -> <x^2>
     !  AFLU   -> B
     !  SSUM   -> C
     !  DE2E   -> D
     !  DEE2   -> F
     !  SFLU   -> A
     !
     ESUM = ESUM/BLNUM
     EFLU = EFLU/BLNUM
     DADI = DADI/BLNUM
     TEMPY   = AFLU/BLNUM
     AFLU = TEMPY - DADI**2
     SSUM = SSUM/BLNUM - DADI*ESUM
     DE2E = DE2E/BLNUM - DADI*SSUM - TEMPY*ESUM
     DEE2 = 2 * DADI * DEE2/BLNUM
     DEE2 = DEE2 - EFLU * DADI**2 + AFLU * ESUM**2
     SFLU = SFLU/BLNUM - DEE2
     SFLU = SFLU + 2 * (DADI*SSUM - DE2E) * ESUM
     !
     !  UFLU   -> SQRT(E - USUM^2) / S0
     !  AFLU   -> SQRT(AFLU) / S0
     !  SFLU   -> beta * SQRT ( A - C^2 ) /S0
     !
     UFLU = SQRT(TEMPY-2*BETA*DE2E+SFLU*BETA*BETA)/S0
     AFLU = SQRT(AFLU) / S0
     SFLU = BETA * SQRT (SFLU - SSUM**2) / S0
     SSUM = BETA * SSUM
     USUM = DADI - SSUM
     !
     !  Block error:
     !
     BBS = BBS/BCNT
     BBU = BBU/BCNT
     BAFLU=SQRT(BAFLU/BCNT - DADI**2) / S1
     BSFLU=SQRT(BSFLU/BCNT - BBS**2) / S1
     BUFLU=SQRT(BUFLU/BCNT - BBU**2) / S1
     !
     WRITE(OUTU,*) ' Total number of data points was',INT(BLNUM)
     WRITE(OUTU,*) ' There were',BCNT,' data sub-averages.'
     WRITE(OUTU,*)
     IF(BCNT <= 0) CALL WRNDIE(-3,'<BLCFTJ>', &
          'Zero data sub-averages found.')
     IF(BCNT < 6) WRITE(OUTU,*) ' BLCFTJ> Warning: too few', &
          ' sub-averages, error estimates may be distorted'
     !
     !
     IF(PRNLEV >= 0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,581) YTEMP, ETEMP
        WRITE(OUTU,*)
        WRITE(OUTU,580) TEMP
        WRITE(OUTU,*)
        WRITE(OUTU,590)
        WRITE(OUTU,*)
        WRITE(OUTU,600) VALIC0,DADI,AFLU,BAFLU
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Internal energy gradient at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,591)
        WRITE(OUTU,*)
        WRITE(OUTU,601) VALIC0,USUM,UFLU,BUFLU,BBU
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Gradient of -TS at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,592)
        WRITE(OUTU,*)
        WRITE(OUTU,602) VALIC0,SSUM,SFLU,BSFLU,BBS
     ENDIF
  ENDIF
580 FORMAT('  Free energy gradient at T = ',F10.2,' K, [kcal/mol]')
581 FORMAT('  The average temperature is: ',F9.2,' +- ',F5.2,' K')
590 FORMAT(5X,5X,'x',7X,'dA/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)')
600 FORMAT(1X, F10.4,2X,3(E12.6,1X),'+++++dA+++')
591 FORMAT(5X,5X,'x',7X,'dU/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3x, 'dU/dx(block)')
592 FORMAT(5X,5X,'x',7X,'-TdS/dx',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3x, '-TdS/dx(block)')
601 FORMAT(1X, F10.4,2X,4(E12.6,1X),'+++++dU+++')
602 FORMAT(1X, F10.4,2X,4(E12.6,1X),'+++-TdS+++')
  !
  RETURN
END SUBROUTINE BLCFTJ

SUBROUTINE DYNICA(X,Y,Z,AMASS,DX,DY,DZ,VALIC0,DUDI0,NATOM, &
     XSAVE,YSAVE,ZSAVE)
  !
  !     This is an analogue of icpert:DYNICT
  !     For thermodynamic integration conformational free energy method
  !
  !     Author: Krzysztof Kuczera  Lawrence, KS 09-Nov-1993
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  use stream
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),AMASS(*)
  !
  real(chm_real) XSAVE(*),YSAVE(*),ZSAVE(*)
  real(chm_real) TOTE,TOTKE,AKMATI
  real(chm_real) ESBNP,ESBFP,ESBRP
  real(chm_real) SCALE,DSCALE
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) DUDI0,DUDIP,DUDIR
  real(chm_real) ICVAL(3,MXICP)
  !kk...10-FEB-98      real(chm_real) VALIC0(MXICP)
  real(chm_real) VALIC0
  CHARACTER(LEN=80) ERRLIN
  INTEGER NATOM,NPRIV,ICP
  INTEGER I,J,K,L,IC,ITYPE,INC,ICOL,NUSED
  LOGICAL LPRINT
  !
  !     Get the interaction energy for the unperturbed ic's.
  !
  !     CALL EIICT(ESBNP,X,Y,Z,DX,DY,DZ)
  !
  !     Get the derivative of the energy wrt the conformational coordinate
  !
  CALL ECNFTI(X,Y,Z,DX,DY,DZ,DUDI0,NATOM)
  !
  ICOL=1
  !
  ! ... Get values of the unperturbed coordinates
  !
  DO IC=1,NICP
     I=ICPATN(1,IC)
     J=ICPATN(2,IC)
     K=ICPATN(3,IC)
     L=ICPATN(4,IC)
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     IF(ICPTYP(IC) == 1) ICVAL(ICOL,IC)=RIJ
     IF(ICPTYP(IC) == 2) ICVAL(ICOL,IC)=TIJK
     IF(ICPTYP(IC) == 3) ICVAL(ICOL,IC)=PIJKL
     IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
        IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
             'Unknown ic type for pert. no. ',IC,'. Programmer error?'
        CALL WRNDIE(-4,'<DYNICA>',ERRLIN)
     END IF
     !
     !kk...10-FEB-98        VALIC0(IC)=ICVAL(1,IC)
     !
  END DO
  !kk...10-FEB-98
  VALIC0=ICVAL(1,1)
  !
  RETURN
END SUBROUTINE DYNICA

SUBROUTINE BLCFTM(COMLYN,COMLEN,DUDI,VALIC0, &
     DEDI,DADI,AFLU,BDADI,BAFLU)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Use trajectory files generated during CFTM ( Multidimensional
  !     Conformational Thermodynamic Integration) to calculate
  !     free energy gradient wrt set of conformational coordinates.
  !     This routine would usually read in a trajectory generated with
  !     fixed coordinates and calculate free energy gradient components.
  !     E.g. use 'update EXSG  WAT*' to leave only solute terms,
  !     fix solvent to get only solute-solute and solute-solvent,
  !     or SKIPE to get parts of energy.
  !
  !     1. Standard formula from fluctuations (std. dev. of mean used)
  !     2. For NCONT /= 0 the statistical errors are calculated from
  !        the variance of sub-averages, found from trajectory
  !        fragments of NCONT data points.
  !     3. Coordinate group contribution analysis
  !        If a nonzero value of 'NGRUp' is specified on the command
  !        line, the derivatives of defined groups of coordinates
  !        will be accumulated (with statistical errors). This section
  !        takes additional input:
  !        NGRUP = number of coordinate groups
  !        if NGRUP > 0 the following lines must contain:
  !          (LGRUP(I),I=1,NIC)    20I4 -- the group number of each coor
  !        followed by strings giving group symbols
  !          (GSYMB(J),J=1,NGRUP)  20A4
  !
  !     4: Array DEDI[2*MXICP+1] stores DUDI[NICP]. The gradient along
  !        the path is stored in the (1+NICP)th element of the array.
  !          The gradients of the groups are stored in DEDI[2+NICP]...
  !        DEDI[1+NICP+NGRUP]
  !
  !
  !     Y.Wang, Lawrence, KS 23-Mar-1996
  !
  !     K.Kuczera, Lawrence, KS 27-Apr-1995
  !                             02-May-1995
  !
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                       
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use memory
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use cvio
  use deriv
  use energym
  use psf
  use icpert

  implicit none
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
  INTEGER ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
  INTEGER IHBFRX,INBFRX,ILBFRX,IMGFRX
  INTEGER I,BCNT,NUMBERL
  real(chm_real) TEMP,BETA,BLNUM
  real(chm_real) BDATA
  real(chm_real) DEDI(*),DADI(*),AFLU(*),BDADI(*),BAFLU(*)
  real(chm_real) DUDI(*),VALIC0(*)
  !
  INTEGER J,K
  real(chm_real) DELTA, S0, S1
  CHARACTER(LEN=4) :: HDRC='CORD', HDRD='VELD'

  call chmalloc('icfcnf.src','BLCFTM','itemp',natom,cr4=itemp)
  call chmalloc('icfcnf.src','BLCFTM','ifreat',natom,intg=ifreat)

  !
  ! Finish processing command line
  !
  NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
  CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
  TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
  !
  ! Initialize variables
  !
  BETA=ONE/(KBOLTZ*TEMP)
  BLNUM=ZERO
  BCNT=0
  BDATA=ZERO
  NUMBERL= NICP + NGRUP + 1
  !
  DO I=1,NUMBERL
     DADI(I)=ZERO
     AFLU(I)=ZERO
     BDADI(I)=ZERO
     BAFLU(I)=ZERO
  END DO
  !
  ISTATS=1
  NAT=NATOM
  IUNIT=NFU
  NFREAT=0
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  Analysis of CFTM conformational free energy'
  WRITE(OUTU,*) '    Thermodynamic integration formula '
  WRITE(OUTU,*)
  !
  !  Loop:
  !  Read in coordinates from a simulation at x_i=const
  !  Update energy lists as needed
  !  Calculate energy and forces
  !  Accumulate averages and fluctuations
  !
  !
570 CONTINUE
  CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
       CG,QCG,                                     & 
#endif
       ITEMP,NAT,IFREAT,NFREAT,NFU, &
       NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
       DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
  !
  ! Update energy lists as needed (LCODES=.FALSE.)
  !
  CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE., &
       .FALSE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])
  !
  ! Call ENERGY: call gives total potential energy in EPROP(EPOT)
  !              and gradient DX,DY,DZ
  !
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
  !
  ! Pass forces to DYNICB, which returns the gradient dE(x)/dx_i
  ! in [DUDI] and values of coordinates x_i in VALIC0
  !
  CALL DYNICB(X,Y,Z,DX,DY,DZ,VALIC0,DUDI,NATOM)
  !
  ! Accumulate values and squares; store data for statistics
  !
  BLNUM=BLNUM+ONE
  !
  !  store the gradient dE(x)/dx_i to DEDI(1)...DEDI(NICP), calculate
  !  SUM(DUDI(I)*DIRV(I))(I=1,NICP) and store it to DEDI(1+NICP),
  !  accumulate group components and put then to DEDI(2+NICP) to
  !  DEDI(1+NICP+NGRUP)
  !
  DO I=NICP+1,NUMBERL
     DEDI(I) = ZERO
  END DO
  !
  DO I=1,NICP
     DEDI(I) = DUDI(I)
     IF (YLAMBD  >  -999) THEN
        DEDI(1+NICP) = DEDI(1+NICP) + DUDI(I)*DIRV(I)
     END IF
     IF(NGRUP > 0) THEN
        J = NICP + 1 + LGRUP(I)
        DEDI(J) = DEDI(J) + DUDI(I)
     END IF
  END DO
  !
  DO I=1,NUMBERL
     DADI(I)=DADI(I)+DEDI(I)
     AFLU(I)=AFLU(I)+DEDI(I)**2
  END DO
  !
  !
  IF(PRNLEV >= 7 .AND. NCONT /= 0 .AND.INT(BLNUM) == 1) THEN
     WRITE(OUTU,550)
  END IF
  IF(PRNLEV >= 7) THEN
     DO I=1,NICP
        WRITE(OUTU,551) INT(BLNUM),BCNT+1,ISTEP,I,VALIC0(I),DUDI(I)
     END DO
  END IF
  !
550 FORMAT(5X,'Frame',2X,'Bin',6X,'Step',3X,'i',9X,'x_i',9X,'dU/dx_i')
551 FORMAT(1X,I7,I6,I7,I4,F12.6,F16.8)
  !
  ! Block averages:
  ! Calculate average values of dA/dx_i over blocks of
  ! NCONT data points. Use the variance of the averages to estimate
  ! statistical errors.
  !
  BDATA=BDATA+ONE
  DO I=1,NUMBERL
     BDADI(I)=BDADI(I)+DEDI(I)
  END DO
  !
  IF(MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
     IF(INT(BDATA) /= NCONT) THEN
        WRITE(OUTU,*) ' BLCFTM> Warning: BDATA =',INT(BDATA), &
             '  /=  NCONT =',NCONT,' for block #',BCNT+1
     END IF
     BCNT=BCNT+1
     DO I=1,NUMBERL
        BDADI(I)=BDADI(I)/BDATA
        BAFLU(I)=BAFLU(I)+BDADI(I)**2
        BDADI(I)=ZERO
     END DO
     BDATA=ZERO
  END IF
  !
  IF (ISTATS  >=  0) GOTO 570

  call chmdealloc('icfcnf.src','BLCFTM','itemp',natom,cr4=itemp)
  call chmdealloc('icfcnf.src','BLCFTM','ifreat',natom,intg=ifreat)

  IF (INT(BLNUM) <= 0) CALL WRNDIE(-3,'<BLCFTM>','No data.')
  !
  ! End of reading loop; calculate overall averages and fluctuations.
  ! Standard errors, assuming no correlations
  !
  IF (NCONT  >=  0) THEN
     S0=SQRT(BLNUM)
     S1=SQRT(DBLE(BCNT))
     !
     DO I=1,NUMBERL
        DADI(I)=DADI(I)/BLNUM
        AFLU(I)=SQRT(AFLU(I)/BLNUM - DADI(I)**2)/S0
        BAFLU(I)=SQRT(BAFLU(I)/BCNT - DADI(I)**2)/S1
     END DO
     !
     !
     WRITE(OUTU,*) ' Total number of data points was',INT(BLNUM)
     WRITE(OUTU,*) ' There were',BCNT,' data sub-averages.'
     WRITE(OUTU,*)
     IF(BCNT <= 0) CALL WRNDIE(-3,'<BLCFTM>', &
          'Zero data sub-averages found.')
     IF(BCNT < 6) WRITE(OUTU,*) ' BLCFTM> Warning: too few', &
          ' sub-averages, error estimates may be distorted'
     !
     !
     IF(PRNLEV >= 0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,580) TEMP
        WRITE(OUTU,*)
        WRITE(OUTU,590)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,600) I,VALIC0(I),DADI(I),AFLU(I),BAFLU(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Free energy gradient along reaction path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,610) YLAMBD,DADI(I),AFLU(I),BAFLU(I)
        ENDIF
     ENDIF
  ENDIF
580 FORMAT('  Free energy gradient at T = ',F10.2,' K,  [kcal/mol]')
590 FORMAT(5X,'i',5X,'x_i',7X,'dA/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)')
600 FORMAT(1X,I4, F10.4,2X,3(E14.6,1X),'+++++dA+++')
610 FORMAT(1X,I4, 2X,3(E14.6,1X),'+++dA/dL++')
  !
  ! ... Group analysis
  !
  IF(NGRUP > 0) THEN
     !
     DO J=1,NGRUP
        GAFLU(J)=ZERO
     END DO
     !
     ! ...Accumulate derivatives and errors in first member
     DO I=1,NICP
        J=LGRUP(I)
        GAFLU(J)=GAFLU(J)+AFLU(I)**2
     END DO
     !  ... Output : print out values with statistical errors
     !
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Free energy gradient accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)     dA/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
575     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,700) J,GSYM(J), &
                VALIC0(I),DADI(K),AFLU(K),BAFLU(K)
        ELSE
           I = I + 1
           GOTO 575
        ENDIF
     END DO
     !
700  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++dA_g++')
     !
     ! ... Analyze group contributions
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '  --Analysis of group gradient contributions--'
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [dA/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !
     !
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0=DADI(K)
     !           S1=GAFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !
     !  710  FORMAT(1X,I4,4X,A4,2X,2(F10.5,2X))
     !
  END IF ! IF(NGRUP > 0)
  !
  !
  RETURN
END SUBROUTINE BLCFTM

SUBROUTINE BLCFTS(COMLYN,COMLEN,DUDI,VALIC0, &
     GUFLU,GSFLU,DEDI,DADI,AFLU,BDADI,BAFLU,SSUM, &
     BSSUM,SFLU,USUM,UFLU,BUFLU,BSFLU,DE2E,DEE2,BBU,BBS)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Use trajectory files generated during CFTM ( Multidimensional
  !     Conformational Thermodynamic Integration) to calculate
  !     free energy, TdS and "total potential energy" gradients both along
  !     the coordinates and along the direction of the conformational
  !     change, wrt set of conformational coordinates.
  !     This routine would usually read in a trajectory generated with
  !     fixed coordinates and calculate free energy gradient components.
  !     E.g. use 'update EXSG  WAT*' to leave only solute terms,
  !     fix solvent to get only solute-solute and solute-solvent,
  !     or SKIPE to get parts of energy.
  !
  !     1. Standard formula from fluctuations (std. dev. of mean used)
  !     2. For NCONT /= 0 the statistical errors are calculated from
  !        the variance of sub-averages, found from trajectory
  !        fragments of NCONT data points.
  !     3. Coordinate group contribution analysis
  !        If a nonzero value of 'NGRUp' is specified on the command
  !        line, the derivatives of defined groups of coordinates
  !        will be accumulated (with statistical errors). This section
  !        takes additional input:
  !        NGRUP = number of coordinate groups
  !        if NGRUP > 0 the following lines must contain:
  !          (LGRUP(I),I=1,NIC)    20I4 -- the group number of each coor
  !        followed by strings giving group symbols
  !          (GSYMB(J),J=1,NGRUP)  20A4
  !
  !     A: Free energy dA/dx = < dE/dx >
  !     B: Entropy: -T*(dS/dx)
  !                 = beta*[< E*(dE/dx)> - <E>*<(dE/dx) >]
  !                 = beta* < (E - <E>) (dE/dx - <dE/dx>) >
  !     C: Internal energy dU/dx
  !                 = dA/dx - ( -T * (dS/dx) )
  !                 = < dE/dx - beta * (E - <E>) (dE/dx - <dE/dx>) >
  !     D: Array DEDI[2*MXICP+1] stores DUDI[NICP]. The gradient along
  !        the path is stored in the (1+NICP)th element of the array. The
  !        gradients of the groups are stored in DEDI[2+NICP]...
  !        DEDI[1+NICP+NGRUP]
  !
  !
  !     Y.Wang, Lawrence, KS 23-Mar-1996
  !     K.Kuczera, Lawrence, KS 27-Apr-1995
  !                             02-May-1995
  !
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only: qcg                   
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use memory
  use stream
  use string
  use bases_fcm
  use consta
  use coord
  use ctitla
  use cvio
  use deriv
  use energym
  use psf
  use icpert

  implicit none
  real(chm_real4),allocatable,dimension(:) :: itemp
  integer,allocatable,dimension(:) :: ifreat
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NCONT,NFU,NUNIT,NBEGN,NSTOP,NSKIP,NFILE
  INTEGER ISTATS,IUNIT,NAT,NFREAT,ISTEP,NDEGF,NSAVV
  INTEGER IHBFRX,INBFRX,ILBFRX,IMGFRX
  INTEGER I,BCNT
  real(chm_real) TEMP,BETA,BLNUM
  !
  INTEGER NDU,NUMBERL
  real(chm_real) YTOTE,ESUM,BESUM,EFLU,TEMPY
  real(chm_real) DUDI(*),VALIC0(*),GUFLU(*),GSFLU(*)
  real(chm_real) DEDI(*),DADI(*),AFLU(*),BDADI(*),BAFLU(*)
  real(chm_real) SSUM(*),BSSUM(*),SFLU(*),USUM(*),UFLU(*),BUFLU(*)
  real(chm_real) DE2E(*),DEE2(*),BSFLU(*),BBU(*),BBS(*)
  !
  INTEGER J,K
  real(chm_real) DELTA, S0, S1
  !
  real(chm_real) BDATA
  CHARACTER(LEN=4) :: HDRC='CORD', HDRD='VELD'

  call chmalloc('icfcnf.src','BLCFTM','itemp',natom,cr4=itemp)
  call chmalloc('icfcnf.src','BLCFTM','ifreat',natom,intg=ifreat)
  !
  ! Finish processing command line
  !
  NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
  CALL TRJSPC(COMLYN,COMLEN,NUNIT,NFU,NBEGN,NSKIP,NSTOP)
  TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
  NDU =GTRMI(COMLYN, COMLEN, 'DUNI',     -1)
  !
  ! Initialize variables
  !
  BETA=ONE/(KBOLTZ*TEMP)
  BLNUM=ZERO
  BCNT=0
  BDATA=ZERO
  ESUM =ZERO
  EFLU =ZERO
  BESUM=ZERO
  NUMBERL= NICP + NGRUP + 1
  !
  DO I=1,NUMBERL
     DADI(I)=ZERO
     AFLU(I)=ZERO
     BDADI(I)=ZERO
     BAFLU(I)=ZERO
     SSUM(I) =ZERO
     SFLU(I) =ZERO
     BBS(I) = ZERO
     BSSUM(I)=ZERO
     BSFLU(I)=ZERO
     BBU(I) = ZERO
     BUFLU(I)=ZERO
     DE2E(I) =ZERO
     DEE2(I) =ZERO
  END DO
  !
  ISTATS=1
  NAT=NATOM
  IUNIT=NFU
  NFREAT=0
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  Analysis of CFTS conformational free energy'
  WRITE(OUTU,*) '    Thermodynamic integration formula '
  WRITE(OUTU,*)
  !
  !  Loop:
  !  Read in coordinates from a simulation at x_i=const
  !  Update energy lists as needed
  !  Calculate energy and forces
  !  Accumulate averages and fluctuations of them
  !
  !
570 CONTINUE
  CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
       CG,QCG,                                      & 
#endif
       ITEMP,NAT,IFREAT,NFREAT,NFU, &
       NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF, &
       DELTA,NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
  !
  ! Update energy lists as needed (LCODES=.FALSE.)
  !
  CALL UPDATE(COMLYN, COMLEN, X,Y,Z,WMAIN,.TRUE., &
       .FALSE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])
  !
  ! Call ENERGY: call gives total potential energy in EPROP(EPOT)
  !              and gradient DX,DY,DZ
  !
  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
  !
  ! Pass forces to DYNICB, which returns the gradient dE(x)/dx_i
  ! in [DUDI] and values of coordinates x_i in VALIC0
  !
  CALL DYNICB(X,Y,Z,DX,DY,DZ,VALIC0,DUDI,NATOM)
  !
  !    Save total potential energy U, and dU/dx_i to  data-unit NDU
  !
  YTOTE = EPROP(EPOT)
  IF(NDU  >  0) THEN
     WRITE(NDU,571) YTOTE, (DUDI(I), I=1,NICP)
  END IF
571 FORMAT(1X,5(E14.8,1X))
  !
  ! Accumulate values and squares; store data for statistics
  !
  BLNUM=BLNUM+ONE
  !
  !  store the gradient dE(x)/dx_i to DEDI(1)...DEDI(NICP), calculate
  !  SUM(DUDI(I)*DIRV(I))(I=1,NICP) and store it to DEDI(1+NICP),
  !  accumulate group components and put then to DEDI(2+NICP) to
  !  DEDI(1+NICP+NGRUP)
  !
  DO I=NICP+1,NUMBERL
     DEDI(I) = ZERO
  END DO
  !
  DO I=1,NICP
     DEDI(I) = DUDI(I)
     IF (YLAMBD  >  -999) THEN
        DEDI(1+NICP) = DEDI(1+NICP) + DUDI(I)*DIRV(I)
     ENDIF
     IF(NGRUP > 0) THEN
        J = NICP + 1 + LGRUP(I)
        DEDI(J) = DEDI(J) + DUDI(I)
     END IF
  END DO
  !
  ESUM = ESUM + YTOTE
  EFLU = EFLU + YTOTE*YTOTE
  DO I=1,NUMBERL
     DADI(I)=DADI(I)+DEDI(I)
     AFLU(I)=AFLU(I)+DEDI(I)**2
     TEMPY  =YTOTE*DEDI(I)
     SSUM(I)=SSUM(I)+TEMPY
     SFLU(I)=SFLU(I)+TEMPY*TEMPY
     DE2E(I)=DE2E(I)+TEMPY*DEDI(I)
     DEE2(I)=DEE2(I)+TEMPY*YTOTE
  END DO
  !
  !
  IF(PRNLEV >= 7 .AND. NCONT /= 0 .AND.INT(BLNUM) == 1) THEN
     WRITE(OUTU,550)
  END IF
  IF(PRNLEV >= 7) THEN
     DO I=1,NICP
        WRITE(OUTU,551) INT(BLNUM),BCNT+1,ISTEP,I,VALIC0(I), &
             YTOTE,DUDI(I)
     END DO
  END IF
  !
550 FORMAT(5X,'Frame',2X,'Bin',6X,'Step',3X,'i',9X,'x_i',9X, &
       'U','dU/dx_i')
551 FORMAT(1X,I7,I6,I7,I4,F12.6,2F16.8)
  !
  ! Block averages:
  ! Calculate average values of dU,dA,and TdS over blocks of
  ! NCONT data points. Use the variance of the averages to estimate
  ! statistical errors.
  !
  BDATA=BDATA+ONE
  BESUM=BESUM+YTOTE
  DO I=1,NUMBERL
     BDADI(I)=BDADI(I)+DEDI(I)
     BSSUM(I)=BSSUM(I)+YTOTE*DEDI(I)
  END DO
  !
  IF(MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
     IF(INT(BDATA) /= NCONT) THEN
        WRITE(OUTU,*) ' BLCFTS> Warning: BDATA =',INT(BDATA), &
             '  /=  NCONT =',NCONT,' for block #',BCNT+1
     END IF
     BCNT=BCNT+1
     BESUM=BESUM/BDATA
     DO I=1,NUMBERL
        BDADI(I)=BDADI(I)/BDATA
        BAFLU(I)=BAFLU(I)+BDADI(I)**2
        BSSUM(I)=BETA*(BSSUM(I)/BDATA-BESUM*BDADI(I))
        BBS(I) = BBS(I) + BSSUM(I)
        BSFLU(I)=BSFLU(I) + BSSUM(I)**2
        TEMPY = BDADI(I)-BSSUM(I)
        BBU(I) = BBU(I) + TEMPY
        BUFLU(I)=BUFLU(I) + TEMPY * TEMPY
        BSSUM(I)=ZERO
        BDADI(I)=ZERO
     END DO
     BESUM=ZERO
     BDATA=ZERO
     !
  END IF
  !
  IF (ISTATS  >=  0) GOTO 570

  call chmdealloc('icfcnf.src','BLCFTM','itemp',natom,cr4=itemp)
  call chmdealloc('icfcnf.src','BLCFTM','ifreat',natom,intg=ifreat)

  IF (INT(BLNUM) <= 0) CALL WRNDIE(-3,'<BLCFTS>','No data.')
  !
  ! End of reading loop; calculate overall averages and fluctuations.
  ! Standard errors, assuming no correlations
  !
  IF (NCONT  >=  0) THEN
     S0=SQRT(BLNUM)
     S1=SQRT(DBLE(BCNT))
     !
     !        Block sub-average error estimates
     !    DADI(I) -> x;        ESUM -> y;        SSUM(I) -> x*y;
     !    AFLU(I) -> x^2;        EFLU -> y^2;        SFLU(I) -> (x*y)^2;
     !        DE2E(I) -> x^2 * y;        DEE2(I) -> x * y^2
     !                    dA/di = < x >;
     !             -TdS/di = beta * ( <x*y> - <x> <y> )
     !                      = beta * < (x-<x>)*(y-<y>) >
     !                dU/di = < x > - beta * ( <x*y> - <x> <y> )
     !                      = < x -  beta*(x-<x>)*(y-<y>)>
     !
     !  A = < (x-<x>)^2 * (y-<y>)^2 >
     !    = <(x*y)^2> - 2<y*x^2><y> - 2<x*y^2><x> - 3<x>^2<y>^2
     !      + <x^2><y>^2 + <x>^2<y^2> + 4<x*y><x><y>
     !    = <(x*y)^2> + 2 * ( <x>*C - D ) * <y> - F
     !  F = 2<x*y^2><x> - <y^2><x>^2 + B * <y>^2
     !  B = <x^2> - <x>^2
     !  C = <x*Y> - <x><y>
     !  D = <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y>
     !    = <y*x^2> - <x^2><y> - <x> * C
     !  E = < (x - a(x-<x>)(y-<y>))^2 >
     !    = <x^2> - 2a ( <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y> ) + A*a^2
     !    = <x^2> - 2a * D + A * a^2
     !
     !  TEMPY     -> <x^2>
     !  AFLU(I)   -> B
     !  SSUM(I)   -> C
     !  DE2E(I)   -> D
     !  DEE2(I)   -> F
     !  SFLU(I)   -> A
     !
     ESUM = ESUM/BLNUM
     EFLU = EFLU/BLNUM
     DO I=1,NUMBERL
        DADI(I) = DADI(I)/BLNUM
        TEMPY   = AFLU(I)/BLNUM
        AFLU(I) = TEMPY - DADI(I)**2
        SSUM(I) = SSUM(I)/BLNUM - DADI(I)*ESUM
        DE2E(I) = DE2E(I)/BLNUM - DADI(I)*SSUM(I) - TEMPY*ESUM
        DEE2(I) = 2 * DADI(I) * DEE2(I)/BLNUM
        DEE2(I) = DEE2(I) - EFLU * DADI(I)**2 + AFLU(I) * ESUM**2
        SFLU(I) = SFLU(I)/BLNUM - DEE2(I)
        SFLU(I) = SFLU(I) + 2 * (DADI(I)*SSUM(I) - DE2E(I)) * ESUM
        !
        !  UFLU(I)   -> SQRT(E - USUM(I)^2) / S0
        !  AFLU(I)   -> SQRT(AFLU(I)) / S0
        !  SFLU(I)   -> beta * SQRT ( A - C^2 ) /S0
        !
        UFLU(I) = SQRT(TEMPY-2*BETA*DE2E(I)+SFLU(I)*BETA*BETA)/S0
        AFLU(I) = SQRT(AFLU(I)) / S0
        SFLU(I) = BETA * SQRT (SFLU(I) - SSUM(I)**2) / S0
        SSUM(I) = BETA * SSUM(I)
        USUM(I) = DADI(I) - SSUM(I)
        !
        !  Block error:
        !
        BBS(I) = BBS(I) /BCNT
        BBU(I) = BBU(I) /BCNT
        BAFLU(I)=SQRT(BAFLU(I)/BCNT - DADI(I)**2) / S1
        BSFLU(I)=SQRT(BSFLU(I)/BCNT - BBS(I)**2) / S1
        BUFLU(I)=SQRT(BUFLU(I)/BCNT - BBU(I)**2) / S1
     END DO
     !
     WRITE(OUTU,*) ' Total number of data points was',INT(BLNUM)
     WRITE(OUTU,*) ' There were',BCNT,' data sub-averages.'
     WRITE(OUTU,*)
     IF(BCNT <= 0) CALL WRNDIE(-3,'<BLCFTS>', &
          'Zero data sub-averages found.')
     IF(BCNT < 6) WRITE(OUTU,*) ' BLCFTS> Warning: too few', &
          ' sub-averages, error estimates may be distorted'
     !
     !
     IF(PRNLEV >= 0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,580) TEMP
        WRITE(OUTU,*)
        WRITE(OUTU,590)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,600) I,VALIC0(I),DADI(I),AFLU(I),BAFLU(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Free energy gradient along reaction path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,610) YLAMBD,DADI(I),AFLU(I),BAFLU(I)
        ENDIF
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Internal energy gradient at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,591)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,601) I,VALIC0(I),USUM(I),UFLU(I),BUFLU(I),BBU(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Internal energy gradient along reaction', &
                ' path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,611) YLAMBD,USUM(I),UFLU(I),BUFLU(I)
        ENDIF
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Gradient of -TS at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,592)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,602) I,VALIC0(I),SSUM(I),SFLU(I),BSFLU(I),BBS(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Entropy gradient along reaction path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,612) YLAMBD,SSUM(I),SFLU(I),BSFLU(I)
        ENDIF
     ENDIF
  ENDIF
580 FORMAT('  Free energy gradient at T = ',F10.2,' K, [kcal/mol]')
590 FORMAT(5X,'i',5X,'x_i',7X,'dA/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)')
600 FORMAT(1X,I4, F10.4,2X,3(E12.6,1X),'+++++dA+++')
610 FORMAT(1X,I4, 2X,3(E12.6,1X),'+++dA/dL++')
591 FORMAT(5X,'i',5X,'x_i',7X,'dU/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3X,'dU/dx_i(block)')
592 FORMAT(5X,'i',5X,'x_i',7X,'-TdS/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3X,'_TdS/dx_i(block)')
601 FORMAT(1X,I4, F10.4,2X,4(E12.6,1X),'+++++dU+++')
611 FORMAT(1X,I4, 2X,3(E12.6,1X),'+++dU/dL++')
602 FORMAT(1X,I4, F10.4,2X,4(E12.6,1X),'+++-TdS+++')
612 FORMAT(1X,I4, 2X,3(E12.6,1X),'++-TdS/dL+')
  !
  ! ... Group analysis
  !
  IF(NGRUP > 0) THEN
     !
     !  ... Output : print out values with statistical errors
     !
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Free energy gradient accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)     dA/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
572     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,700) J,GSYM(J), &
                VALIC0(I),DADI(K),AFLU(K),BAFLU(K)
        ELSE
           I = I + 1
           GOTO 572
        ENDIF
     END DO
     !
700  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++dA_g++')
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Gradient of U accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)     dU/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
573     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,701) J,GSYM(J), &
                VALIC0(I),USUM(K),UFLU(K),BUFLU(K)
        ELSE
           I = I + 1
           GOTO 573
        ENDIF
     END DO
     !
701  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++dU_g++')
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Gradient of -TS accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)   -TdS/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
574     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,702) J,GSYM(J), &
                VALIC0(I),SSUM(K),SFLU(K),BSFLU(K)
        ELSE
           I = I + 1
           GOTO 574
        ENDIF
     END DO
     !
702  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++-TdS_g++')
     !
     ! ... Analyze group contributions
     !
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '  --Analysis of group gradient contributions--'
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [dA/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !C
     !         DO J=1,NGRUP
     !           GAFLU(J)=ZERO
     !           GSFLU(J)=ZERO
     !           GUFLU(J)=ZERO
     !         END DO
     !C
     !         DO I=1,NICP
     !           J=LGRUP(I)
     !           GAFLU(J)=GAFLU(J)+DADI(I)**2
     !           GSFLU(J)=GSFLU(J)+SSUM(I)**2
     !           GUFLU(J)=GUFLU(J)+USUM(I)**2
     !         END DO
     !C
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0 = DADI(K)
     !           S1=GAFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [dU/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0=USUM(K)
     !           S1=GUFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [-TdS/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0=SSUM(K)
     !           S1=GSFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !  710  FORMAT(1X,I4,4X,A4,2X,2(F10.5,2X))
     !C
  END IF ! IF(NGRUP > 0)
  !
  !
  RETURN
END SUBROUTINE BLCFTS

SUBROUTINE BLCFTC(COMLYN,COMLEN,VALIC0, &
     GUFLU,GSFLU,DEDI,DADI,AFLU,BDADI,BAFLU,SSUM, &
     BSSUM,SFLU,USUM,UFLU,BUFLU,BSFLU,DE2E,DEE2,BBU,BBS)
  !-----------------------------------------------------------------------
  !     Basic function:
  !     Use ICP  files generated during CFTM ( Multidimensional
  !     Conformational Thermodynamic Integration) to calculate
  !     free energy, TdS and "total potential energy" gradients both along
  !     the coordinates and along the direction of the conformational
  !     change, wrt set of conformational coordinates.
  !
  !     1. Standard formula from fluctuations (std. dev. of mean used)
  !     2. For NCONT /= 0 the statistical errors are calculated from
  !        the variance of sub-averages, found from trajectory
  !        fragments of NCONT data points. NT is the total number of
  !        frames in the file
  !     3. Coordinate group contribution analysis
  !        If a nonzero value of 'NGRUp' is specified on the command
  !        line, the derivatives of defined groups of coordinates
  !        will be accumulated (with statistical errors). This section
  !        takes additional input:
  !        NGRUP = number of coordinate groups
  !        if NGRUP > 0 the following lines must contain:
  !          (LGRUP(I),I=1,NIC)    20I4 -- the group number of each coor
  !        followed by strings giving group symbols
  !          (GSYMB(J),J=1,NGRUP)  20A4
  !
  !     A: Free energy dA/dx = < dE/dx >
  !     B: Entropy: -T*(dS/dx)
  !                 = beta*[< E*(dE/dx)> - <E>*<(dE/dx) >]
  !                 = beta* < (E - <E>) (dE/dx - <dE/dx>) >
  !     C: Internal energy dU/dx
  !                 = dA/dx - ( -T * (dS/dx) )
  !                 = < dE/dx - beta * (E - <E>) (dE/dx - <dE/dx>) >
  !     D: Array DEDI[2*MXICP+1] stores derivtives. The gradient along
  !        the path is stored in the (1+NICP)th element of the array. The
  !        gradients of the groups are stored in DEDI[2+NICP]...
  !        DEDI[1+NICP+NGRUP]
  !
  !
  !     Y.Wang, Lawrence, KS 23-Mar-1996
  !     K.Kuczera, Lawrence, KS 27-Apr-1995
  !                             02-May-1995
  !
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use string
  use consta
  use ctitla
  use psf
  use icpert

  implicit none
  CHARACTER(LEN=*) COMLYN
  INTEGER COMLEN
  !
  INTEGER NCONT,ICPU,LIC
  INTEGER I,BCNT,NDEGF,JUNK
  real(chm_real) TEMP,BETA,BLNUM,RJUNK,YTEMP,ETEMP,TOTEY,TOTKY
  !
  INTEGER NUMBERL
  real(chm_real) YTOTE,ESUM,BESUM,EFLU,TEMPY
  real(chm_real) VALIC0(*),GUFLU(*),GSFLU(*)
  real(chm_real) DEDI(*),DADI(*),AFLU(*),BDADI(*),BAFLU(*)
  real(chm_real) SSUM(*),BSSUM(*),SFLU(*),USUM(*),UFLU(*),BUFLU(*)
  real(chm_real) DE2E(*),DEE2(*),BSFLU(*),BBU(*),BBS(*)
  !
  INTEGER NIC,J,K,ISTEP
  real(chm_real) S0, S1
  !
  real(chm_real) BDATA
  !MFC ERROR ISTEP used before defined.
  !    very minor since it is only used in a print statement
  !    setting to -1 to avoid compiler warnings
  istep=-1
  !MFC
  !
  ! Finish processing command line
  !
  NCONT =GTRMI(COMLYN, COMLEN, 'CONT',      0)
  TEMP  =GTRMF(COMLYN, COMLEN, 'TEMP', THRHUN)
  ICPU =GTRMI(COMLYN, COMLEN, 'UICP',     -1)
  WRITE(OUTU,*)
  WRITE(OUTU,*) ' The data are read from unit: ',ICPU
  !
  ! Initialize variables
  !
  BETA=ONE/(KBOLTZ*TEMP)
  BLNUM=ZERO
  BCNT=0
  BDATA=ZERO
  YTEMP=ZERO
  ETEMP=ZERO
  ESUM =ZERO
  EFLU =ZERO
  BESUM=ZERO
  NUMBERL= NICP + NGRUP + 1
  !
  DO I=1,NUMBERL
     DADI(I)=ZERO
     AFLU(I)=ZERO
     BDADI(I)=ZERO
     BAFLU(I)=ZERO
     SSUM(I) =ZERO
     SFLU(I) =ZERO
     BSSUM(I)=ZERO
     BBS(I) = ZERO
     BSFLU(I)=ZERO
     BBU(I) = ZERO
     BUFLU(I)=ZERO
     DE2E(I) =ZERO
     DEE2(I) =ZERO
  END DO
  !
  WRITE(OUTU,*)
  WRITE(OUTU,*) '  Analysis of CFTC conformational free energy'
  WRITE(OUTU,*) '    Thermodynamic integration formula '
  WRITE(OUTU,*)
  !
  !
  ! ... Read first line
  READ(ICPU,*) LIC,JUNK,NDEGF,TEMPY
  IF(NICP /= LIC) THEN
     WRITE(6,*) ' ****Error: NICP, LIC =',NICP,LIC
     STOP
  END IF
  !
  ! ... Read values of fixed internal coordinates
  READ(ICPU,'(5D16.8)') (VALIC0(I),I=1,NICP)
  !
  !
  !  Loop:
  !  Read in data
  !  Accumulate averages and fluctuations of them
  !
  !
570 CONTINUE
  !
  ! ... Read line with energies
  READ(ICPU,'(I7,F10.4,2D16.8)',END=571) JUNK,RJUNK,TOTEY,TOTKY
  YTOTE=TOTEY-TOTKY
  YTEMP = YTEMP + TOTKY
  ETEMP = ETEMP + TOTKY * TOTKY
  !
  ! ... Read in potential energy gradient
  READ(ICPU,'(5D16.8)') (DEDI(I),I=1,NICP)
  !
  ! ... Accumulate averages
  !_______________________________________________________________

  BLNUM=BLNUM+ONE
  !
  !  calculate SUM(DEDI(I)*DIRV(I))(I=1,NICP) and store it to DEDI(1+NICP),
  !  accumulate group components and put then to DEDI(2+NICP) to
  !  DEDI(1+NICP+NGRUP)
  !
  DO I=NICP+1,NUMBERL
     DEDI(I) = ZERO
  END DO
  !
  DO I=1,NICP
     IF (YLAMBD  >  -999) THEN
        DEDI(1+NICP) = DEDI(1+NICP) + DEDI(I)*DIRV(I)
     ENDIF
     IF(NGRUP > 0) THEN
        J = NICP + 1 + LGRUP(I)
        DEDI(J) = DEDI(J) + DEDI(I)
     END IF
  END DO
  !
  ESUM = ESUM + YTOTE
  EFLU = EFLU + YTOTE*YTOTE
  DO I=1,NUMBERL
     DADI(I)=DADI(I)+DEDI(I)
     AFLU(I)=AFLU(I)+DEDI(I)**2
     TEMPY  =YTOTE*DEDI(I)
     SSUM(I)=SSUM(I)+TEMPY
     SFLU(I)=SFLU(I)+TEMPY*TEMPY
     DE2E(I)=DE2E(I)+TEMPY*DEDI(I)
     DEE2(I)=DEE2(I)+TEMPY*YTOTE
  END DO
  !
  !
  IF(PRNLEV >= 7 .AND. NCONT /= 0 .AND.INT(BLNUM) == 1) THEN
     WRITE(OUTU,550)
  END IF
  IF(PRNLEV >= 7) THEN
     DO I=1,NICP
        WRITE(OUTU,551) INT(BLNUM),BCNT+1,ISTEP,I,VALIC0(I), &
             YTOTE,DEDI(I)
     END DO
  END IF
  !
550 FORMAT(5X,'Frame',2X,'Bin',6X,'Step',3X,'i',9X,'x_i',9X, &
       'U','dU/dx_i')
551 FORMAT(1X,I7,I6,I7,I4,F12.6,2F16.8)
  !
  ! Block averages:
  ! Calculate average values of dU,dA,and TdS over blocks of
  ! NCONT data points. Use the variance of the averages to estimate
  ! statistical errors.
  !
  BDATA=BDATA+ONE
  BESUM=BESUM+YTOTE
  DO I=1,NUMBERL
     BDADI(I)=BDADI(I)+DEDI(I)
     BSSUM(I)=BSSUM(I)+YTOTE*DEDI(I)
  END DO
  !
  IF(MOD(INT(BLNUM),ABS(NCONT)) == 0) THEN
     IF(INT(BDATA) /= NCONT) THEN
        WRITE(OUTU,*) ' BLCFTC> Warning: BDATA =',INT(BDATA), &
             '  /=  NCONT =',NCONT,' for block #',BCNT+1
     END IF
     BCNT=BCNT+1
     BESUM=BESUM/BDATA
     DO I=1,NUMBERL
        BDADI(I)=BDADI(I)/BDATA
        BAFLU(I)=BAFLU(I)+BDADI(I)**2
        BSSUM(I)=BETA*(BSSUM(I)/BDATA-BESUM*BDADI(I))
        BBS(I) = BBS(I) + BSSUM(I)
        BSFLU(I)=BSFLU(I) + BSSUM(I)**2
        TEMPY = BDADI(I)-BSSUM(I)
        BBU(I) = BBU(I) + TEMPY
        BUFLU(I)=BUFLU(I) + TEMPY * TEMPY
        BSSUM(I)=ZERO
        BDADI(I)=ZERO
     END DO
     BESUM=ZERO
     BDATA=ZERO
     !
  END IF
  GOTO 570
  !
571 CONTINUE
  IF (INT(BLNUM) <= 0) CALL WRNDIE(-3,'<BLCFTC>','No data.')
  !
  ! End of reading loop; calculate overall averages and fluctuations.
  ! Standard errors, assuming no correlations
  !
  IF (NCONT  >=  0) THEN
     S0=SQRT(BLNUM)
     S1=SQRT(DBLE(BCNT))
     YTEMP = YTEMP/BLNUM
     ETEMP = SQRT(ETEMP/BLNUM - YTEMP*YTEMP)/S0
     TEMPY = 2/(KBOLTZ*NDEGF)
     YTEMP = YTEMP * TEMPY
     ETEMP = ETEMP * TEMPY
     !
     !        Block sub-average error estimates
     !    DADI(I) -> x;        ESUM -> y;        SSUM(I) -> x*y;
     !    AFLU(I) -> x^2;        EFLU -> y^2;        SFLU(I) -> (x*y)^2;
     !        DE2E(I) -> x^2 * y;        DEE2(I) -> x * y^2
     !                    dA/di = < x >;
     !             -TdS/di = beta * ( <x*y> - <x> <y> )
     !                      = beta * < (x-<x>)*(y-<y>) >
     !                dU/di = < x > - beta * ( <x*y> - <x> <y> )
     !                      = < x -  beta*(x-<x>)*(y-<y>)>
     !
     !  A = < (x-<x>)^2 * (y-<y>)^2 >
     !    = <(x*y)^2> - 2<y*x^2><y> - 2<x*y^2><x> - 3<x>^2<y>^2
     !      + <x^2><y>^2 + <x>^2<y^2> + 4<x*y><x><y>
     !    = <(x*y)^2> + 2 * ( <x>*C - D ) * <y> - F
     !  F = 2<x*y^2><x> - <y^2><x>^2 + B * <y>^2
     !  B = <x^2> - <x>^2
     !  C = <x*Y> - <x><y>
     !  D = <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y>
     !    = <y*x^2> - <x^2><y> - <x> * C
     !  E = < (x - a(x-<x>)(y-<y>))^2 >
     !    = <x^2> - 2a ( <y*x^2> - <x^2><y> - <x*y><x> + <x>^2<y> ) + A*a^2
     !    = <x^2> - 2a * D + A * a^2
     !
     !  TEMPY     -> <x^2>
     !  AFLU(I)   -> B
     !  SSUM(I)   -> C
     !  DE2E(I)   -> D
     !  DEE2(I)   -> F
     !  SFLU(I)   -> A
     !
     ESUM = ESUM/BLNUM
     EFLU = EFLU/BLNUM
     DO I=1,NUMBERL
        DADI(I) = DADI(I)/BLNUM
        TEMPY   = AFLU(I)/BLNUM
        AFLU(I) = TEMPY - DADI(I)**2
        SSUM(I) = SSUM(I)/BLNUM - DADI(I)*ESUM
        DE2E(I) = DE2E(I)/BLNUM - DADI(I)*SSUM(I) - TEMPY*ESUM
        DEE2(I) = 2 * DADI(I) * DEE2(I)/BLNUM
        DEE2(I) = DEE2(I) - EFLU * DADI(I)**2 + AFLU(I) * ESUM**2
        SFLU(I) = SFLU(I)/BLNUM - DEE2(I)
        SFLU(I) = SFLU(I) + 2 * (DADI(I)*SSUM(I) - DE2E(I)) * ESUM
        !
        !  UFLU(I)   -> SQRT(E - USUM(I)^2) / S0
        !  AFLU(I)   -> SQRT(AFLU(I)) / S0
        !  SFLU(I)   -> beta * SQRT ( A - C^2 ) /S0
        !
        UFLU(I) = SQRT(TEMPY-2*BETA*DE2E(I)+SFLU(I)*BETA*BETA)/S0
        AFLU(I) = SQRT(AFLU(I)) / S0
        SFLU(I) = BETA * SQRT (SFLU(I) - SSUM(I)**2) / S0
        SSUM(I) = BETA * SSUM(I)
        USUM(I) = DADI(I) - SSUM(I)
        !
        !  Block error:
        !
        BBS(I) = BBS(I) /BCNT
        BBU(I) = BBU(I) /BCNT
        BAFLU(I)=SQRT(BAFLU(I)/BCNT - DADI(I)**2) / S1
        BSFLU(I)=SQRT(BSFLU(I)/BCNT - BBS(I)**2) / S1
        BUFLU(I)=SQRT(BUFLU(I)/BCNT - BBU(I)**2) / S1
     END DO
     !
     WRITE(OUTU,*) ' Total number of data points was',INT(BLNUM)
     WRITE(OUTU,*) ' There were',BCNT,' data sub-averages.'
     WRITE(OUTU,*)
     IF(BCNT <= 0) CALL WRNDIE(-3,'<BLCFTC>', &
          'Zero data sub-averages found.')
     IF(BCNT < 6) WRITE(OUTU,*) ' BLCFTC> Warning: too few', &
          ' sub-averages, error estimates may be distorted'
     !
     !
     IF(PRNLEV >= 0) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,581) YTEMP, ETEMP
        WRITE(OUTU,*)
        WRITE(OUTU,580) TEMP
        WRITE(OUTU,*)
        WRITE(OUTU,590)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,600) I,VALIC0(I),DADI(I),AFLU(I),BAFLU(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Free energy gradient along reaction path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,610) YLAMBD,DADI(I),AFLU(I),BAFLU(I)
        ENDIF
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Internal energy gradient at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,591)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,601) I,VALIC0(I),USUM(I),UFLU(I),BUFLU(I),BBU(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Internal energy gradient along reaction', &
                ' path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,611) YLAMBD,USUM(I),UFLU(I),BUFLU(I)
        ENDIF
        WRITE(OUTU,*)
        WRITE(OUTU,*) '  Gradient of -TS at the same T'
        WRITE(OUTU,*)
        WRITE(OUTU,592)
        WRITE(OUTU,*)
        DO I=1,NICP
           WRITE(OUTU,602) I,VALIC0(I),SSUM(I),SFLU(I),BSFLU(I),BBS(I)
        END DO
        IF (YLAMBD  >  -999) THEN
           WRITE(OUTU,*)
           WRITE(OUTU,*) ' Entropy gradient along reaction path:'
           WRITE(OUTU,*)
           I = 1 + NICP
           WRITE(OUTU,612) YLAMBD,SSUM(I),SFLU(I),BSFLU(I)
        ENDIF
     ENDIF
  ENDIF
581 FORMAT('  The average temperature is: ',F9.2,' +- ',F5.2,' K')
580 FORMAT('  Free energy gradient at T = ',F10.2,' K, [kcal/mol]')
590 FORMAT(5X,'i',5X,'x_i',7X,'dA/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)')
600 FORMAT(1X,I4, F10.4,2X,3(E12.6,1X),'+++++dA+++')
610 FORMAT(1X,I4, 2X,3(E14.6,1X),'+++dA/dL++')
591 FORMAT(5X,'i',5X,'x_i',7X,'dU/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3X,'dU/dx_i(block)')
592 FORMAT(5X,'i',5X,'x_i',7X,'-TdS/dx_i',3X,'S.D.(fluct)',4X, &
       'S.D.(block)',3X,'-TdS/dx_i(block)')
601 FORMAT(1X,I4, F10.4,2X,4(E12.6,1X),'+++++dU+++')
611 FORMAT(1X,I4, 2X,3(E14.6,1X),'+++dU/dL++')
602 FORMAT(1X,I4, F10.4,2X,4(E12.6,1X),'+++-TdS+++')
612 FORMAT(1X,I4, 2X,3(E14.6,1X),'++-TdS/dL+')
  !
  ! ... Group analysis
  !
  IF(NGRUP > 0) THEN
     !
     !  ... Output : print out values with statistical errors
     !
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Free energy gradient accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)     dA/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
572     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,700) J,GSYM(J), &
                VALIC0(I),DADI(K),AFLU(K),BAFLU(K)
        ELSE
           I = I + 1
           GOTO 572
        ENDIF
     END DO
     !
700  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++dA_g++')
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Gradient of U accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)     dU/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
573     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,701) J,GSYM(J), &
                VALIC0(I),USUM(K),UFLU(K),BUFLU(K)
        ELSE
           I = I + 1
           GOTO 573
        ENDIF
     END DO
     !
701  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++dU_g++')
     WRITE(OUTU,*)
     WRITE(OUTU,*)
     WRITE(OUTU,*) ' --Gradient of -TS accumulated over groups--'
     WRITE(OUTU,*)
     WRITE(OUTU,*) '   j  Name   x(j)   -TdS/dx(j)   S.D.(fluct) ', &
          '  S.D.(block)'
     WRITE(OUTU,*)
     !
     DO J=1,NGRUP
        K = NICP + 1 + J
        I = 1
574     IF (LGRUP(I)  ==  J) THEN
           WRITE(OUTU,702) J,GSYM(J), &
                VALIC0(I),SSUM(K),SFLU(K),BSFLU(K)
        ELSE
           I = I + 1
           GOTO 574
        ENDIF
     END DO
     !
702  FORMAT(1X,I4,2X,A4,F8.2,1X,3(F11.5,1X),3X,'++-TdS_g++')
     !
     ! ... Analyze group contributions
     !
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '  --Analysis of group gradient contributions--'
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [dA/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !C
     !         DO J=1,NGRUP
     !           GAFLU(J)=ZERO
     !           GSFLU(J)=ZERO
     !           GUFLU(J)=ZERO
     !         END DO
     !C
     !         DO I=1,NICP
     !           J=LGRUP(I)
     !           GAFLU(J)=GAFLU(J)+DADI(I)**2
     !           GSFLU(J)=GSFLU(J)+SSUM(I)**2
     !           GUFLU(J)=GUFLU(J)+USUM(I)**2
     !         END DO
     !C
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0 = DADI(K)
     !           S1=GAFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [dU/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0=USUM(K)
     !           S1=GUFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !       WRITE(OUTU,*)
     !       WRITE(OUTU,*) '   j    Name     [-TdS/dx]_av     S.D. '
     !       WRITE(OUTU,*)
     !         DO J=1,NGRUP
     !           IF(NGRU(J) <= 0) STOP ' Too few group members!!'
     !           K = NICP + 1 + J
     !           S0=SSUM(K)
     !           S1=GSFLU(J)
     !           IF (NGRU(J)  ==  1) THEN
     !             WRITE(OUTU,*) " Warning: Only one member in the group"
     !           ELSE
     !             S1=S1/NGRU(J) - S0**2
     !           END IF
     !           WRITE(OUTU,710) J,GSYM(J),S0,SQRT(S1/NGRU(J))
     !         END DO
     !  710  FORMAT(1X,I4,4X,A4,2X,2(F10.5,2X))
     !
  END IF ! IF(NGRUP > 0)
  !
  !
  RETURN
END SUBROUTINE BLCFTC

SUBROUTINE DYNICB(X,Y,Z,DX,DY,DZ,VALIC0,DUDI0,NATOM)
  !
  !     This is an analogue of DYNICA, for CFTM calculations
  !     Called from BLCFTM to get free energy gradient from trajectory
  !     The free energy gradient calculation part should be indentical
  !     to that in  DYNICM, which does the same thing on the fly.
  !
  !     Author: Krzysztof Kuczera  Lawrence, KS 27-Apr-1995
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  use stream
  use intcor2,only:geticv
  implicit none
  !
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  !
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) DUDI0(MXICP)
  real(chm_real) VALIC0(MXICP)
  real(chm_real) DZET,DUDIX
  CHARACTER(LEN=80) ERRLIN
  INTEGER NATOM,ICP
  INTEGER I,J,K,L,IC,ITYPE,INC
  LOGICAL LPRINT
  !
  !
  !     Get the energy gradient wrt the conformational coordinates
  !
  ! ... Loop over set of internal coordinates
  ! ... DUDI(ICP) contains the derivative of the potential energy
  ! ... wrt the internal coordinate no. ICP
  !
  DO ICP=1,NICP
     !
     DUDIX=ZERO
     !
     IF(LMOV2(ICP)) THEN
        DZET=ONE/TWO
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
        !
        DZET=-DZET
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV2, &
             ICPMV2(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
        DUDI0(ICP)=DUDIX
        !
     ELSE
        DZET=ONE
        CALL MVICP0(ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZET,DUDIX,X,Y,Z,DX,DY,DZ)
        DUDI0(ICP)=DUDIX
     END IF
     !
  END DO
  !
  !
  ! ... Get values of the "perturbation coordinates"
  !
  DO IC=1,NICP
     I=ICPATN(1,IC)
     J=ICPATN(2,IC)
     K=ICPATN(3,IC)
     L=ICPATN(4,IC)
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     IF(ICPTYP(IC) == 1) VALIC0(IC)=RIJ
     IF(ICPTYP(IC) == 2) VALIC0(IC)=TIJK
     IF(ICPTYP(IC) == 3) VALIC0(IC)=PIJKL
     IF(ICPTYP(IC) < 1.OR.ICPTYP(IC) > 3) THEN
        IF(WRNLEV >= 2) WRITE(ERRLIN,'(A,I4,A)') &
             'Unknown ic type for pert. no. ',IC,'. Programmer error?'
        CALL WRNDIE(-4,'<DYNICB>',ERRLIN)
     END IF
     !
  END DO
  !
  !
  RETURN
END SUBROUTINE DYNICB

SUBROUTINE EIICT(EINICP,X,Y,Z,DX,DY,DZ)
  !
  !     This is a FORTRAN copy of EIICP :
  !     difference:  DX,DY,DZ are passed up to DYNICT for TI
  !     calculations
  !
  !     K. Kuczera, Lawrence, KS 05-Nov-1993
  !
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
  use cnst_fcm
  use fast
  use hbondm
  use image
  use inbnd
  use psf
  use memory
  use icpert
  use datstr,only:dupldt_nbond,freedt_nbond
  implicit none
  !
  integer,allocatable,dimension(:) :: iskip
  real(chm_real),allocatable,dimension(:) :: rtemp
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real) EINICP
  INTEGER ICP,LEN,OLDFSTR,OLDLFST
  !
  !
  CALL FREEDT_nbond(BNBNDC)
  CALL DUPLDT_nbond(BNBNDC,BNBND)
  !
  !     Have to bypass fast routines since they don't take
  !     skip lists.
  !
  OLDFSTR=FASTER
  OLDLFST=LFAST
  FASTER=-1
  LFAST=-1
  LEN=MAX(NBOND,NTHETA,NPHI,NIMPHI,NHB,NCSPHI &
#if KEY_CMAP==1
       ,NCRTERM &     
#endif
  )
  call chmalloc('icfcnf.src','EIICT','iskip',len,intg=iskip)
  call chmalloc('icfcnf.src','EIICT','rtemp',natom,crl=rtemp)

  CALL EIICT2(EINICP,X,Y,Z,DX,DY,DZ,BNBNDC, &
       ISLICP,JSLICP,ISKIP,RTEMP, &
       BNBNDC%INBLO,BNBNDC%JNB, &
       BNBND%INBLO,BNBND%JNB, &
       BNBNDC%INBLOG)
  CALL FREEDT_nbond(BNBNDC)
  call chmdealloc('icfcnf.src','EIICT','iskip',len,intg=iskip)
  call chmdealloc('icfcnf.src','EIICT','rtemp',natom,crl=rtemp)
  FASTER=OLDFSTR
  LFAST=OLDLFST
  RETURN
END SUBROUTINE EIICT

SUBROUTINE EIICT2(EINICP,X,Y,Z,DX,DY,DZ,BNBND,ISLCT,JSLCT,ISKIP, &
     RTEMP,INBLO,JNB,INBLOX,JNBX,INBLOG)
  !
  !     Based on pert/icpert.src(EIICP2). This version passes out forces
  !     in DX,DY,DZ. Also,   Urey-Bradley terms added
  !     The lists of internal energy terms to be skipped and
  !     nonbonded terms to be included are based on ISLCT and JSLCT
  !     arrays which are set up in ICPSET.
  !
  !     ISLCT(I)=1 for atoms included in the "INTE" selections
  !                (logical sum of the possible two selections)
  !     JSLCT(I)=1 for all atoms in system
  !
  !     K. Kuczera, Mar-97
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor           
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use cnst_fcm
  use code
  use eintern
  use enbond_mod
  use energym
  use hbondm
  use image
  use number
  use psf
  use fast
  use param
  use stream
  use usermod,only: usere
  use cmapm
#if KEY_FLUCQ==1
  use flucq
#endif 
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif 
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  real(chm_real) EINICP
  real(chm_real) RTEMP(*)
  real(chm_real),parameter :: CCSW0(1) = (/ ZERO /)
  type(nonbondDataStructure) BNBND

  INTEGER INBLO(*),INBLOX(*),INBLOG(*)
  INTEGER I,J,K,N,IFIRST,ILAST
  INTEGER JNB(*),JNBX(*)
  INTEGER ISLCT(*),JSLCT(*),ISKIP(*)
  !
  ! ... initialization
  !

#if KEY_DOMDEC==1
  if (q_domdec) then
     call wrndie(-5,'<icfcnf>','EIICT2 not ready for DOMDEC')
  endif
#endif 

  EINICP=ZERO
  DO I=1,LENENT
     ETERM(I) = ZERO
  END DO
  DO I=1,NATOMT
     DX(I) = ZERO
     DY(I) = ZERO
     DZ(I) = ZERO
  END DO
  !
  !     Get the various energy terms.
  ! ...user
  IF(QETERM(USER)) THEN
     CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ,.FALSE.,[zero],NATOM)
  END IF
  !
  ! ...bonds
  IF(NBOND > 0.AND.QETERM(BOND)) THEN
     DO I=1,NBOND
        ISKIP(I)=1
        IF(ISLCT(IB(I)) == 1 .AND. JSLCT(JB(I)) == 1) ISKIP(I)=0
        IF(ISLCT(JB(I)) == 1 .AND. JSLCT(IB(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL EBOND(ETERM(BOND),IB,JB,ICB,NBOND,CBC,CBB,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

     !
  END IF  ! BOND
  !
  ! ...angles
  IF(NTHETA > 0.AND.QETERM(ANGLE)) THEN
     DO I=1,NTHETA
        ISKIP(I)=1
        IF(ISLCT(JT(I)) == 1 .AND. JSLCT(JT(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL EANGLE(ETERM(ANGLE),IT,JT,KT,ICT,NTHETA,CTC,CTB,DX,DY,DZ, &
          X,Y,Z,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

     !
  END IF ! ANGLE/THETA
  !
  ! ...Urey-Bradley terms
  ! ... same ISKIP as for THETA
  !
  IF(NTHETA > 0.AND.QETERM(UREYB)) THEN
     CALL EBOND(ETERM(UREYB),IT,KT,ICT,NTHETA,CTUC,CTUB,DX,DY,DZ, &
          X,Y,Z,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

  END IF ! UREYB
  !
  ! ...dihedrals
  IF(NPHI > 0.AND.QETERM(DIHE)) THEN
     DO I=1,NPHI
        ISKIP(I)=1
        IF(ISLCT(JP(I)) == 1 .AND. JSLCT(KP(I)) == 1) ISKIP(I)=0
        IF(ISLCT(KP(I)) == 1 .AND. JSLCT(JP(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL EPHI(ETERM(DIHE),IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB, &
          CPCOS,CPSIN,DX,DY,DZ,X,Y,Z, &
          .FALSE.,CCSW0,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

     !
  END IF ! PHI/DIHE
  !
  ! ...impropers
  IF(NIMPHI > 0.AND.QETERM(IMDIHE)) THEN
     DO I=1,NIMPHI
        ISKIP(I)=1
        IF(ISLCT(IM(I)) == 1 .AND. JSLCT(IM(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL EPHI(ETERM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI,CIC,CID,CIB, &
          CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
          .FALSE.,CCSW0,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

     !
  END IF ! IMPHI

#if KEY_CMAP==1
  ! ...dihedral cross-terms
  IF(NCRTERM > 0.AND.QETERM(CMAP)) THEN
     DO I=1,NCRTERM
        ISKIP(I)=1
        IF(ISLCT(I1CT(I)) == 1 .AND. JSLCT(I1CT(I)) == 1.AND. &
             ISLCT(I2CT(I)) == 1 .AND. JSLCT(I2CT(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL ECMAP(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
          I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
          DX,DY,DZ,X,Y,Z, &
          .FALSE., (/ZERO/), 1, ISKIP, (/ZERO/), (/0/), .FALSE.)
     !
  END IF ! CMAP
#endif 
  !
  ! ...nonbonded
  IF(NATOM > 0) THEN
     IFIRST=1
     N=0
     DO I=1,NATOM
        ILAST=INBLOX(I)
        IF(ISLCT(I) == 1) THEN
           DO J=IFIRST,ILAST
              K=JNBX(J)
              IF(K < 0) K=-K
              IF(JSLCT(K) == 1) THEN
                 N=N+1
                 JNB(N)=JNBX(J)
              END IF
           END DO ! J
        END IF ! ISLCT(I) == 1
        !
        IF(JSLCT(I) == 1 .AND. ISLCT(I) /= 1) THEN
           DO J=IFIRST,ILAST
              K=JNBX(J)
              IF(K < 0) K=-K
              IF(ISLCT(K) == 1) THEN
                 N=N+1
                 JNB(N)=JNBX(J)
              END IF
           END DO ! J
        END IF ! JSLCT(I) == 1
        !
        INBLO(I)=N
        IFIRST=ILAST+1
     END DO ! I
     !
     DO I=1,NGRP
        INBLOG(I)=0
     END DO
     !
     CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBND, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),.FALSE.,ZERO,(/ZERO/),(/0/),.FALSE.,QETERM(VDW), &
          QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,          & 
#endif
          .FALSE.,NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

     !
  END IF ! NATOM > 0
  !
  ! ...H-bonds
  IF(NHB > 0.AND.QETERM(HBOND)) THEN
     DO I=1,NHB
        ISKIP(I)=1
        IF(ISLCT(IHB(I)) == 1 .AND. JSLCT(JHB(I)) == 1) ISKIP(I)=0
        IF(ISLCT(JHB(I)) == 1 .AND. JSLCT(IHB(I)) == 1) ISKIP(I)=0
     END DO
     !
     CALL EHBOND(ETERM(HBOND),IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,ISKIP,CTONHB, &
          CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
     !
  END IF ! HB
  !
  ! ...constraints: harmonic on atoms
  IF(QCNSTR.AND.QETERM(CHARM)) THEN
     DO I=1,NATOM
        RTEMP(I)=0.0
        IF(ISLCT(I) == 1 .AND. JSLCT(I) == 1) RTEMP(I)=KCNSTR(I)
     END DO
     !
     CALL ECNSTR(ETERM(CHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          .FALSE., (/ ZERO /), (/ ZERO /), (/ 0 /), .FALSE. &
          ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
          )

     !
  END IF ! CHARM
  !
  !
  ! ...constraints: pca harmonic on atoms
  IF(QCNSTR.AND.QETERM(PCHARM).and.numpca>0) THEN
     DO I=1,NATOM
        RTEMP(I)=0.0
        IF(ISLCT(I) == 1 .AND. JSLCT(I) == 1) RTEMP(I)=KCNSTR(I)
     END DO
     !
     CALL ECNSTR(ETERM(PCHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          .FALSE., (/ ZERO /), (/ ZERO /), (/ 0 /), .FALSE. &
          ,numpca,pcax,pcay,pcaz &
          )

     !
  END IF ! CHARM
  !
  ! ...constraints: harmonic on dihedrals
  IF((NCSPHI > 0) .AND. QETERM(CDIHE)) THEN
     DO I=1,NCSPHI
        ISKIP(I)=1
        IF(ISLCT(JCS(I)) == 1 .AND. JSLCT(KCS(I)) == 1) ISKIP(I)=0
        IF(ISLCT(KCS(I)) == 1 .AND. JSLCT(JCS(I)) == 1) ISKIP(I)=0
     END DO
     !
#if KEY_DOMDEC==1
     if (q_domdec) then
        CALL WRNDIE(-5,'<EIICT2>',&
             'HARMONIC RESTRAINTS NOT READY FOR DOMDEC')
     endif
#endif 
     CALL EPHI(ETERM(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
          CCSC,CCSD,CCSB,CCSCOS,CCSSIN,DX,DY,DZ,X,Y,Z, &
          .TRUE.,CCSW,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

     !
  END IF ! CDIHE
  !
  ! ...image nonbond
  IF(NTRANS > 0) THEN
     CALL IMINTR(X,Y,Z,DX,DY,DZ,ISLCT,JSLCT)
  END IF
  !
  ! ...added Urey-Bradley term
  EINICP=ETERM(TSM)+ETERM(BOND)+ETERM(ANGLE)+ETERM(DIHE)+ &
       ETERM(UREYB)+ &
       ETERM(IMDIHE)+ETERM(VDW)+ETERM(ELEC)+ETERM(HBOND)+ &
       ETERM(CHARM)+ETERM(CDIHE)+ETERM(IMVDW)+ETERM(IMHBND)+ &
       ETERM(IMELEC)+ETERM(IMST2)
  !
  !
  return
end SUBROUTINE EIICT2


#endif 
SUBROUTINE NULL_ICF
  RETURN
END SUBROUTINE NULL_ICF

