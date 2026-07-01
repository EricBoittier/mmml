#if KEY_TSM==1
SUBROUTINE ICFSET
  !
  !     Called from TSMS, this routine sets up ic constraints.
  !
  !     Author: Doug Tobias
  !
  !     The command syntax for ic constraints is as follows:
  !
  !     FIX {ic-spec} [TOLI <real>]
  !
  !     where ic-spec :==
  !
  !      [DIST or BOND] 2x{atom-spec}
  !      [ANGL or THET] 3x{atom-spec}
  !      [DIHE or PHI]  4x{atom-spec}
  !
  !      [RDIF or RXNC] 3x{atom-spec}  => see definition below
  !
  !     and
  !
  !     TOLI is the tolerance (in Angstroms or degrees, default RSMALL);
  !     atom-spec :== segid resid iupac;
  !
  use chm_kinds
  use dimens_fcm
  use comand
  use coord
  use exfunc
  use icfix
  use intcor2,only:geticv
  use number
  use psf
  use shake
  use stream
  use string
  use chutil,only:getatn

  implicit none
  !
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  INTEGER I,J,K,L
  CHARACTER(LEN=4) WRD
  character(len=8) ASEG,ARES,AATOM
  CHARACTER(LEN=80) ERRLIN
  !
  ! Reset QHOLO based on other holonomic constraints
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  NICF=NICF+1
  IF(NICF > MAXICF) THEN
     WRITE(ERRLIN,'(A,I4,A)') &
          'Exceeded maximum number of constraints (',MAXICF,').'
     CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
  ENDIF
  !
  IF(WRD == 'DIST'.OR.WRD == 'BOND') THEN
     !
     !     distance constraint
     !
     ICFTYP(NICF)=1
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     I=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     J=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     IF(IMOVE(I) > 0.OR.IMOVE(J) > 0) THEN
        NICF=NICF-1
        WRITE(ERRLIN,'(A)') &
             'Fixed atoms currently not allowed in ic constraints.'
        CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
     ENDIF
     K=-99
     L=-99
     ICFATN(1,NICF)=I
     ICFATN(2,NICF)=J
     ICFATN(3,NICF)=K
     ICFATN(4,NICF)=L
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     SVAL(NICF)=RIJ
     TOLI(NICF)=GTRMF(COMLYN,COMLEN,'TOLI',RSMALL)
     !
  ELSE IF(WRD == 'ANGL'.OR.WRD == 'THET') THEN
     !
     !     bond angle constraint
     !
     ICFTYP(NICF)=2
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     I=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     J=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     K=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     IF(IMOVE(I) > 0.OR.IMOVE(J) > 0.OR.IMOVE(K).GT.0) THEN
        NICF=NICF-1
        WRITE(ERRLIN,'(A)') &
             'Fixed atoms currently not allowed in ic constraints.'
        CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
     ENDIF
     L=-99
     ICFATN(1,NICF)=I
     ICFATN(2,NICF)=J
     ICFATN(3,NICF)=K
     ICFATN(4,NICF)=L
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     SVAL(NICF)=TIJK
     TOLI(NICF)=GTRMF(COMLYN,COMLEN,'TOLI',RSMALL)
     !
  ELSE IF(WRD == 'DIHE'.OR.WRD == 'PHI ') THEN
     !
     !     dihedral angle constraint
     !
     ICFTYP(NICF)=3
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     I=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     J=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     K=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     L=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     IF(IMOVE(I) > 0.OR.IMOVE(J) > 0.OR.IMOVE(K).GT.0.OR. &
          IMOVE(L) > 0) THEN
        NICF=NICF-1
        WRITE(ERRLIN,'(A)') &
             'Fixed atoms currently not allowed in ic constraints.'
        CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
     ENDIF
     ICFATN(1,NICF)=I
     ICFATN(2,NICF)=J
     ICFATN(3,NICF)=K
     ICFATN(4,NICF)=L
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     SVAL(NICF)=PIJKL
     TOLI(NICF)=GTRMF(COMLYN,COMLEN,'TOLI',RSMALL)
     !
  ELSE IF(WRD == 'RDIF'.OR.WRD == 'RXNC') THEN
     !
     !JG052203   Reaction Coordinate constraint
     !           RC = [R(IJ) - R(KJ)]
     !
     ICFTYP(NICF)=4
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     I=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     J=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     K=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     IF(IMOVE(I) > 0.OR.IMOVE(J) > 0.OR.IMOVE(K).GT.0) THEN
        NICF=NICF-1
        WRITE(ERRLIN,'(A)') &
             'Fixed atoms currently not allowed in ic constraints.'
        CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
     ENDIF
     L=-99
     ICFATN(1,NICF)=I
     ICFATN(2,NICF)=J
     ICFATN(3,NICF)=K
     ICFATN(4,NICF)=L
     TIJK = -99.99
     TIJK = GTRMF(COMLYN,COMLEN,'RCTS',TIJK)
     IF(TIJK ==  -99.99) THEN
        WRITE(ERRLIN,'(A,F15.5)') &
             'Reaction Coordinate ill-defined',TIJK
        CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
     ENDIF
     SVAL(NICF)=TIJK
     TOLI(NICF)=GTRMF(COMLYN,COMLEN,'TOLI',0.01*RSMALL)
     !
  ELSE
     NICF=NICF-1
     WRITE(ERRLIN,'(A,A4,A)') &
          'Unrecognized command "',WRD,'".'
     CALL WRNDIE(-4,'<ICFSET>',ERRLIN)
  ENDIF
  !
  CALL XTRANE(COMLYN,COMLEN,'ICFSET')
  !
  IF(NICF > 0)  QHOLO=.TRUE.
  !
  RETURN
END SUBROUTINE ICFSET

SUBROUTINE PRNICF
  !
  !     Print list of ic constraints.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use dimens_fcm
  use icfix
  use stream
  use chutil,only:atomid
  implicit none
  !
  CHARACTER(LEN=8) SIDI,RIDI,RENI,ACI,SIDJ,RIDJ,RENJ,ACJ
  CHARACTER(LEN=8) SIDK,RIDK,RENK,ACK,SIDL,RIDL,RENL,ACL
  INTEGER IC,I
  !
  IF(PRNLEV >= 2) WRITE(OUTU,200)
  loop1000: DO IC=1,NICF
     IF(ICFTYP(IC) == 1) THEN
        I=ICFATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        I=ICFATN(2,IC)
        CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
        IF(PRNLEV >= 2) WRITE(OUTU,300) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),SVAL(IC),TOLI(IC)
     ELSE IF(ICFTYP(IC) == 2) THEN
        I=ICFATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        I=ICFATN(2,IC)
        CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
        I=ICFATN(3,IC)
        CALL ATOMID(I,SIDK,RIDK,RENK,ACK)
        IF(PRNLEV >= 2) WRITE(OUTU,310) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),RIDK(1:idleng),ACK(1:idleng), &
             SVAL(IC),TOLI(IC)
     ELSE IF(ICFTYP(IC) == 3) THEN
        I=ICFATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        I=ICFATN(2,IC)
        CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
        I=ICFATN(3,IC)
        CALL ATOMID(I,SIDK,RIDK,RENK,ACK)
        I=ICFATN(4,IC)
        CALL ATOMID(I,SIDL,RIDL,RENL,ACL)
        IF(PRNLEV >= 2) WRITE(OUTU,320) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),RIDK(1:idleng),ACK(1:idleng), &
             RIDL(1:idleng),ACL(1:idleng),SVAL(IC),TOLI(IC)
     ELSE IF(ICFTYP(IC) == 4) THEN
        I=ICFATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        I=ICFATN(2,IC)
        CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
        I=ICFATN(3,IC)
        CALL ATOMID(I,SIDK,RIDK,RENK,ACK)
        IF(PRNLEV >= 2) WRITE(OUTU,330) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),RIDK(1:idleng),ACK(1:idleng), &
             SVAL(IC),TOLI(IC)
     ENDIF
  enddo loop1000
  IF(PRNLEV >= 2) WRITE(OUTU,400) MAXI
  !
200 FORMAT(/,' List of internal coordinate constraints:',/)
300 FORMAT(1X,'DIST',1X,2(A,1X,A,1X),21X,F9.4,1X,'TOLI =',E13.6)
310 FORMAT(1X,'ANGL',1X,3(A,1X,A,1X),11X,F9.4,1X,'TOLI =',E13.6)
320 FORMAT(1X,'DIHE',1X,4(A,1X,A,1X),1X,F9.4,1X,'TOLI =',E13.6)
330 FORMAT(1X,'RXNC',1X,3(A,1X,A,1X),11X,F9.4,1X,'TOLI =',E13.6)
400 FORMAT(/,' MAXimum Iterations for ic resetting: ',I7)
  RETURN
END SUBROUTINE PRNICF

SUBROUTINE ICFCNS(XREF,YREF,ZREF,X,Y,Z,AMASS,NATOM)
  !
  !     This routine enforces internal coordinate constraints
  !     during dynamics (Tobias and Brooks, JCP 89, 5115 (1988)).
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use icfix
  use intcor2,only:geticv
  use number
  use stream
  implicit none
  !
  INTEGER NATOM
  real(chm_real) XREF(natom),YREF(natom),ZREF(natom),X(natom),Y(natom),Z(natom)
  real(chm_real) XTMP(4),YTMP(4),ZTMP(4)
  real(chm_real) SX,SY,SZ,S2,LAMBDA,LAMBDM
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) AMASST(4)
  real(chm_real) AMASS(natom)
  INTEGER IIC,ICTYPE,ICNT,I,J,K,L,II,NITER
  INTEGER NUSED
  CHARACTER(LEN=80) ERRLIN
  LOGICAL DONE
  ! namkh 11/02/04
  real(chm_real) RKJ,RXNCOOR
  !
  IF(NICF == 0) RETURN
  NITER=0
  DONE=.FALSE.
  ANYADJ=.FALSE.
  !
  !     Iteratively solve the constraint equations and reset
  !     the coordinates.
  !
  do while(.NOT.DONE)
     !
     NITER=NITER+1
     IF(NITER > MAXI) THEN
        WRITE(ERRLIN,'(A,I6,A)') &
             'ic constraint resetting not accomplished in ', &
             MAXI,' iterations.'
        CALL WRNDIE(-4,'<ICFCNS>',ERRLIN)
     ENDIF
     NUSED=0
     !
     loop1000: DO IIC=1,NICF
        ICTYPE=ICFTYP(IIC)
        I=ICFATN(1,IIC)
        J=ICFATN(2,IIC)
        K=ICFATN(3,IIC)
        L=ICFATN(4,IIC)
        ICTYPE=ICFTYP(IIC)
        !JG052203
        IF(ICTYPE <= 3) THEN
           !
           CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
        ELSE
           RIJ = SQRT((X(I)-X(J))**2+(Y(I)-Y(J))**2+(Z(I)-Z(J))**2)
           RKJ = SQRT((X(K)-X(J))**2+(Y(K)-Y(J))**2+(Z(K)-Z(J))**2)
           RXNCOOR = RIJ-RKJ
           !CCC        write(6,*)'RXNC_Value=',RXNCOOR,' RIJ=',RIJ,' RKJ=',RKJ
           !
        ENDIF
        !
        IF(ICTYPE == 1) THEN
           DS(IIC)=SVAL(IIC)-RIJ
        ELSE IF(ICTYPE == 2) THEN
           DS(IIC)=SVAL(IIC)-TIJK
        ELSE IF(ICTYPE == 3) THEN
           DS(IIC)=MOD(SVAL(IIC)-PIJKL,THR6TY)
           DS(IIC)=DS(IIC)-360.0*MOD(INT(DS(IIC))/180,2)
           !JG052203
        ELSE IF(ICTYPE == 4) THEN
           DS(IIC)=SVAL(IIC)-RXNCOOR
           !
        ENDIF
        ICFADJ(IIC)=(ABS(DS(IIC)) > TOLI(IIC))
        ICNT=ICTYPE+1
        !JG052203
        IF(ICTYPE == 4) ICNT = 3
        !
        NUSED=NUSED+ICNT
        IF(ICFADJ(IIC)) THEN
           ANYADJ=.TRUE.
           DO I=1,ICNT
              II=ICFATN(I,IIC)
              XTMP(I)=X(II)
              YTMP(I)=Y(II)
              ZTMP(I)=Z(II)
              AMASST(I)=AMASS(II)
           enddo
           !
           !     Get the s vectors.
           !
           CALL WILSON(XTMP,YTMP,ZTMP,SMF,ICTYPE,NUSED)
           !
           !     Form the G coefficients.
           !
           NUSED=NUSED-ICNT
           GCOEF(IIC)=ZERO
           DO I=1,ICNT
              NUSED=NUSED+1
              II=ICFATN(I,IIC)
              SX=SMF(1,NUSED)
              SY=SMF(2,NUSED)
              SZ=SMF(3,NUSED)
              S2=SX*SX+SY*SY+SZ*SZ
              GCOEF(IIC)=GCOEF(IIC)+S2/AMASS(II)
           enddo
           !
           !     Reset the coordinates.
           !
           LAMBDA=DS(IIC)/GCOEF(IIC)
           IF(ICTYPE == 2.OR.ICTYPE == 3) LAMBDA=DEGRAD*LAMBDA
           NUSED=NUSED-ICNT
           DO I=1,ICNT
              NUSED=NUSED+1
              II=ICFATN(I,IIC)
              LAMBDM=LAMBDA/AMASS(II)
              SX=SMF(1,NUSED)
              SY=SMF(2,NUSED)
              SZ=SMF(3,NUSED)
              X(II)=X(II)+LAMBDM*SX
              Y(II)=Y(II)+LAMBDM*SY
              Z(II)=Z(II)+LAMBDM*SZ
           enddo
        ENDIF
     enddo loop1000
     !
     !     Check for convergence.
     !
     DONE=.TRUE.
     DO IIC=1,NICF
        ! APH: Note: loop vectorization caused problems in the original code
        !            this is why I changed it to use IF -statement
        !          DONE=(DONE.AND..NOT.ICFADJ(IIC))
        IF (ICFADJ(IIC)) DONE = .FALSE.
     enddo
     !
  enddo
  !
  !      IF(PRNLEV >= 2) WRITE(OUTU,100) NITER
  !100   FORMAT(' ICFCNS: internal coordinates reset in',I6,
  !     & ' iterations')
  RETURN
END SUBROUTINE ICFCNS

SUBROUTINE WILSON(X,Y,Z,S,ICTYPE,NUSED)
  !
  !     Gets the Wilson s vectors for bonds, angles, and dihedral angles.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use number
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),S(3,*)
  real(chm_real) XTMP,YTMP,ZTMP,NORM,SINTJ
  real(chm_real) NIJX,NIJY,NIJZ,NJKX,NJKY,NJKZ,NKLX,NKLY,NKLZ
  real(chm_real) RIJ,RJK,RKL,COSTJ,SIN2TJ,COSTK,SIN2TK
  real(chm_real) RIJCJ,RIJSJ2,RJKSJ2,R2SJ2,RJKSK2,RKLSK2,RKLCK,R2SK2
  INTEGER ICTYPE,NUSED
  INTEGER I,J,K,L
  !
  IF(ICTYPE == 1) THEN
     !
     !     Bonds
     !
     I=NUSED-1
     J=NUSED
     XTMP=X(2)-X(1)
     YTMP=Y(2)-Y(1)
     ZTMP=Z(2)-Z(1)
     NORM=SQRT(XTMP*XTMP+YTMP*YTMP+ZTMP*ZTMP)
     NORM=ONE/NORM
     S(1,J)=XTMP*NORM
     S(2,J)=YTMP*NORM
     S(3,J)=ZTMP*NORM
     S(1,I)=-S(1,J)
     S(2,I)=-S(2,J)
     S(3,I)=-S(3,J)
     !
  ELSE IF(ICTYPE == 2) THEN
     !
     !     Angles
     !
     I=NUSED-2
     J=NUSED-1
     K=NUSED
     NIJX=X(1)-X(2)
     NIJY=Y(1)-Y(2)
     NIJZ=Z(1)-Z(2)
     RIJ=SQRT(NIJX*NIJX+NIJY*NIJY+NIJZ*NIJZ)
     NIJX=NIJX/RIJ
     NIJY=NIJY/RIJ
     NIJZ=NIJZ/RIJ
     NJKX=X(3)-X(2)
     NJKY=Y(3)-Y(2)
     NJKZ=Z(3)-Z(2)
     RJK=SQRT(NJKX*NJKX+NJKY*NJKY+NJKZ*NJKZ)
     NJKX=NJKX/RJK
     NJKY=NJKY/RJK
     NJKZ=NJKZ/RJK
     COSTJ=NIJX*NJKX+NIJY*NJKY+NIJZ*NJKZ
     XTMP=NIJY*NJKZ-NIJZ*NJKY
     YTMP=NIJZ*NJKX-NIJX*NJKZ
     ZTMP=NIJX*NJKY-NIJY*NJKX
     SINTJ=SQRT(XTMP*XTMP+YTMP*YTMP+ZTMP*ZTMP)
     NORM=RIJ*SINTJ
     NORM=ONE/NORM
     S(1,I)=(COSTJ*NIJX-NJKX)*NORM
     S(2,I)=(COSTJ*NIJY-NJKY)*NORM
     S(3,I)=(COSTJ*NIJZ-NJKZ)*NORM
     NORM=RJK*SINTJ
     NORM=ONE/NORM
     S(1,K)=(COSTJ*NJKX-NIJX)*NORM
     S(2,K)=(COSTJ*NJKY-NIJY)*NORM
     S(3,K)=(COSTJ*NJKZ-NIJZ)*NORM
     S(1,J)=-S(1,I)-S(1,K)
     S(2,J)=-S(2,I)-S(2,K)
     S(3,J)=-S(3,I)-S(3,K)
     !
  ELSE IF(ICTYPE == 3) THEN
     !
     !     Dihedrals
     !
     I=NUSED-3
     J=NUSED-2
     K=NUSED-1
     L=NUSED
     NIJX=X(2)-X(1)
     NIJY=Y(2)-Y(1)
     NIJZ=Z(2)-Z(1)
     RIJ=SQRT(NIJX*NIJX+NIJY*NIJY+NIJZ*NIJZ)
     NIJX=NIJX/RIJ
     NIJY=NIJY/RIJ
     NIJZ=NIJZ/RIJ
     NJKX=X(3)-X(2)
     NJKY=Y(3)-Y(2)
     NJKZ=Z(3)-Z(2)
     RJK=SQRT(NJKX*NJKX+NJKY*NJKY+NJKZ*NJKZ)
     NJKX=NJKX/RJK
     NJKY=NJKY/RJK
     NJKZ=NJKZ/RJK
     NKLX=X(4)-X(3)
     NKLY=Y(4)-Y(3)
     NKLZ=Z(4)-Z(3)
     RKL=SQRT(NKLX*NKLX+NKLY*NKLY+NKLZ*NKLZ)
     NKLX=NKLX/RKL
     NKLY=NKLY/RKL
     NKLZ=NKLZ/RKL
     COSTJ=NIJX*NJKX+NIJY*NJKY+NIJZ*NJKZ
     XTMP=NIJY*NJKZ-NIJZ*NJKY
     YTMP=NIJZ*NJKX-NIJX*NJKZ
     ZTMP=NIJX*NJKY-NIJY*NJKX
     SIN2TJ=XTMP*XTMP+YTMP*YTMP+ZTMP*ZTMP
     RIJCJ=RJK+RIJ*COSTJ
     RIJSJ2=RIJ*SIN2TJ
     RJKSJ2=COSTJ/(RJK*SIN2TJ)
     R2SJ2=RIJ*RJK*SIN2TJ
     S(1,I)=-XTMP/RIJSJ2
     S(2,I)=-YTMP/RIJSJ2
     S(3,I)=-ZTMP/RIJSJ2
     S(1,J)=RIJCJ*XTMP/R2SJ2
     S(2,J)=RIJCJ*YTMP/R2SJ2
     S(3,J)=RIJCJ*ZTMP/R2SJ2
     S(1,K)=-XTMP*RJKSJ2
     S(2,K)=-YTMP*RJKSJ2
     S(3,K)=-ZTMP*RJKSJ2
     COSTK=NJKX*NKLX+NJKY*NKLY+NJKZ*NKLZ
     XTMP=NJKY*NKLZ-NJKZ*NKLY
     YTMP=NJKZ*NKLX-NJKX*NKLZ
     ZTMP=NJKX*NKLY-NJKY*NKLX
     SIN2TK=XTMP*XTMP+YTMP*YTMP+ZTMP*ZTMP
     RKLCK=RJK+RKL*COSTK
     RJKSK2=COSTK/(RJK*SIN2TK)
     RKLSK2=RKL*SIN2TK
     R2SK2=RJK*RKL*SIN2TK
     S(1,J)=S(1,J)+XTMP*RJKSK2
     S(2,J)=S(2,J)+YTMP*RJKSK2
     S(3,J)=S(3,J)+ZTMP*RJKSK2
     S(1,K)=S(1,K)-RKLCK*XTMP/R2SK2
     S(2,K)=S(2,K)-RKLCK*YTMP/R2SK2
     S(3,K)=S(3,K)-RKLCK*ZTMP/R2SK2
     S(1,L)=XTMP/RKLSK2
     S(2,L)=YTMP/RKLSK2
     S(3,L)=ZTMP/RKLSK2
     !
  ELSE IF(ICTYPE == 4) THEN
     !
     !JG052203    Reaction Coordinate Definition:
     !            Rc = {RIJ-RKJ}
     !
     I=NUSED-2
     J=NUSED-1
     K=NUSED
     NIJX=X(1)-X(2)
     NIJY=Y(1)-Y(2)
     NIJZ=Z(1)-Z(2)
     RIJ=SQRT(NIJX*NIJX+NIJY*NIJY+NIJZ*NIJZ)
     NIJX=NIJX/RIJ
     NIJY=NIJY/RIJ
     NIJZ=NIJZ/RIJ
     NJKX=X(3)-X(2)
     NJKY=Y(3)-Y(2)
     NJKZ=Z(3)-Z(2)
     RJK=SQRT(NJKX*NJKX+NJKY*NJKY+NJKZ*NJKZ)
     NJKX=NJKX/RJK
     NJKY=NJKY/RJK
     NJKZ=NJKZ/RJK
     S(1,I)= NIJX
     S(2,I)= NIJY
     S(3,I)= NIJZ
     S(1,K)=-NJKX
     S(2,K)=-NJKY
     S(3,K)=-NJKZ
     S(1,J)=-S(1,I)-S(1,K)
     S(2,J)=-S(2,I)-S(2,K)
     S(3,J)=-S(3,I)-S(3,K)
     !
  ENDIF
  !
  return
end SUBROUTINE WILSON

#endif 

SUBROUTINE NULL_IF
  RETURN
END SUBROUTINE NULL_IF

