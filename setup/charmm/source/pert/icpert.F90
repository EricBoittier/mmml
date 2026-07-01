#if KEY_TSM==1
SUBROUTINE ICPSET
  !
  !     Set up ic perturbations.
  !
  !     Author: Doug Tobias
  !
  !     The command syntax for ic perturbations (invoked from TSMS)
  !     is as follows:
  !
  !      MOVE {ic-spec} BY <real> INTE {sele-spec}
  !
  !     where {ic-spec} is one of the following
  !
  !      [DIST] or [BOND] 2x{atom-spec}
  !      [ANGL] or [THET] 3x{atom-spec}
  !      [DIHE] or [PHI]  4x{atom-spec}
  !
  !     and {sele-spec} is
  !
  !      [SELE {atom-selection} END [SELE {atom selection} END]]
  !
  !     If a double selection is indicated, each selected group will
  !     be moved half the perturbation.
  !
  use chm_kinds
  use dimens_fcm
  use comand
  use icpert
  use psf
  use number
  use selctam
  use memory
  use select
  use stream
  use string
  use chutil,only:getatn

  implicit none
  !
  integer,allocatable,dimension(:) :: islct, jslct
  real(chm_real) DZ
  INTEGER I,J,K,L,ICP
  INTEGER IMODE, NSLCT
  CHARACTER(LEN=4) WRD
  character(len=8) ASEG,ARES,AATOM
  CHARACTER(LEN=80) ERRLIN
  !
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  NICP=NICP+1
  IF(NICP > MXICP) THEN
     WRITE(ERRLIN,'(A,I4,A)') &
          'Exceeded maximum number of ic perturbations (',MXICP,').'
     CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
  ENDIF

  !
  IF(WRD == 'DIST'.OR.WRD == 'BOND') THEN
     !
     !     distance perturbation
     !
     ICPTYP(NICP)=1
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     I=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ASEG=NEXTA8(COMLYN,COMLEN)
     ARES=NEXTA8(COMLYN,COMLEN)
     AATOM=NEXTA8(COMLYN,COMLEN)
     J=GETATN(ASEG,ARES,AATOM,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
     ICPATN(1,NICP)=I
     ICPATN(2,NICP)=J
     ICPATN(3,NICP)=-99
     ICPATN(4,NICP)=-99
     !
  ELSE IF(WRD == 'ANGL'.OR.WRD == 'THET') THEN
     !
     !     bond angle perturbation
     !
     ICPTYP(NICP)=2
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
     ICPATN(1,NICP)=I
     ICPATN(2,NICP)=J
     ICPATN(3,NICP)=K
     ICPATN(4,NICP)=-99
     !
  ELSE IF(WRD == 'DIHE'.OR.WRD == 'PHI ') THEN
     !
     !     dihedral angle perturbation
     !
     ICPTYP(NICP)=3
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
     ICPATN(1,NICP)=I
     ICPATN(2,NICP)=J
     ICPATN(3,NICP)=K
     ICPATN(4,NICP)=L
     !
  ELSE
     NICP=NICP-1
     WRITE(ERRLIN,'(A,A4,A)') &
          'Unrecognized ic type ',WRD,'".'
     CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
  ENDIF
  !
  !     Process BY command.
  !
  IF(INDX(COMLYN,COMLEN,'BY',2) > 0) THEN
     DZ=GTRMF(COMLYN,COMLEN,'BY',ZERO)
     IF(DZ == ZERO) THEN
        WRITE(ERRLIN,'(A)') 'Zero perturbation specified.'
        CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
     ENDIF
     DZETA(NICP)=DZ
  ELSE
     WRITE(ERRLIN,'(A)') 'BY command not found.'
     CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
  ENDIF
  !
  !     Process INTE selection.
  !
  IF(INDXA(COMLYN,COMLEN,'INTE') > 0) THEN
     call chmalloc('icpert.src','ICPSET','islct',natom,intg=islct)
     call chmalloc('icpert.src','ICPSET','jslct',natom,intg=jslct)
     IMODE=0
     CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
          .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG,.TRUE., &
          (/ZERO/),(/ZERO/),(/ZERO/),.TRUE.,1,(/ZERO/))
     IF(IMODE /= 0) THEN
        WRITE(ERRLIN,'(A)') 'Atom selection error.'
        CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
     ELSE
        CALL ADDMVA(ISLCT,NATOM,NSLCT)
        IF(NSLCT > 0) THEN
           NMOV1(NICP)=NSLCT
           call chmalloc('icpert.src','ICPSET','ICPMV1(NICP)',NSLCT,intgp=ICPMV1(NICP)%a)
           CALL GETMVA(ISLCT,ICPMV1(NICP)%a,NATOM)
        ELSE
           WRITE(ERRLIN,'(A)') 'Zero atoms selected.'
           CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
        ENDIF
     ENDIF
     LMOV2(NICP)=.FALSE.
     CALL TRIME(COMLYN,COMLEN)
     IF(COMLEN > 0) THEN
        !
        !     Double selection.
        !
        IMODE=0
        CALL SELRPN(COMLYN,COMLEN,JSLCT,NATOM,1,IMODE, &
             .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG,.TRUE., &
             (/ZERO/),(/ZERO/),(/ZERO/),.TRUE.,1,(/ZERO/))
        IF(IMODE /= 0) THEN
           WRITE(ERRLIN,'(A)') 'Atom selection error.'
           CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
        ELSE
           CALL ADDMVA(JSLCT,NATOM,NSLCT)
           IF(NSLCT > 0) THEN
              LMOV2(NICP)=.TRUE.
              NMOV2(NICP)=NSLCT
              call chmalloc('icpert.src','ICPSET','ICPMV2(NICP)',NSLCT,intgp=ICPMV2(NICP)%a)
              CALL GETMVA(JSLCT,ICPMV2(NICP)%a,NATOM)
           ELSE
              WRITE(ERRLIN,'(A)') 'Zero atoms selected.'
              CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
           ENDIF
        ENDIF
     ENDIF
     call chmdealloc('icpert.src','ICPSET','islct',natom,intg=islct)
     call chmdealloc('icpert.src','ICPSET','jslct',natom,intg=jslct)
  ELSE
     WRITE(ERRLIN,'(A)') 'INTE command not found.'
     CALL WRNDIE(-4,'<ICPSET>',ERRLIN)
  ENDIF
  !
  CALL TRIME(COMLYN,COMLEN)
  IF(COMLEN /= 0) THEN
     CALL XTRANE(COMLYN,COMLEN,'ICPSET')
     CALL DIEWRN(0)
  ENDIF
  RETURN
END SUBROUTINE ICPSET

SUBROUTINE PRNICP
  !
  !     Print list of ic perturbations.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use dimens_fcm
  use icpert
  use stream
  use chutil,only:atomid
  implicit none
  !
  CHARACTER(LEN=8) SIDI,RIDI,RENI,ACI,SIDJ,RIDJ,RENJ,ACJ
  CHARACTER(LEN=8) SIDK,RIDK,RENK,ACK,SIDL,RIDL,RENL,ACL
  INTEGER IC,I,J,K,L
  !
  IF(PRNLEV >= 2) WRITE(OUTU,200)
  DO IC=1,NICP
     IF(ICPTYP(IC) == 1) THEN
        I=ICPATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        J=ICPATN(2,IC)
        CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
        IF(PRNLEV >= 2) WRITE(OUTU,300) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),DZETA(IC)
     ELSE IF(ICPTYP(IC) == 2) THEN
        I=ICPATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        J=ICPATN(2,IC)
        CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
        K=ICPATN(3,IC)
        CALL ATOMID(K,SIDK,RIDK,RENK,ACK)
        IF(PRNLEV >= 2) WRITE(OUTU,310) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),RIDK(1:idleng),ACK(1:idleng),DZETA(IC)
     ELSE IF(ICPTYP(IC) == 3) THEN
        I=ICPATN(1,IC)
        CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
        J=ICPATN(2,IC)
        CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
        K=ICPATN(3,IC)
        CALL ATOMID(K,SIDK,RIDK,RENK,ACK)
        L=ICPATN(4,IC)
        CALL ATOMID(L,SIDL,RIDL,RENL,ACL)
        IF(PRNLEV >= 2) WRITE(OUTU,320) &
             RIDI(1:idleng),ACI(1:idleng),RIDJ(1:idleng), &
             ACJ(1:idleng),RIDK(1:idleng),ACK(1:idleng), &
             RIDL(1:idleng),ACL(1:idleng),DZETA(IC)
     ENDIF
  enddo
  !
200 FORMAT(/,' The following ic perturbation(s) will be done:',/)
300 FORMAT(1X,'DIST',1X,2(A,1X,A,1X),21X,'BY',F14.8)
310 FORMAT(1X,'ANGL',1X,3(A,1X,A,1X),11X,'BY',F14.8)
320 FORMAT(1X,'DIHE',1X,4(A,1X,A,1X),1X,'BY',F14.8)
  RETURN
END SUBROUTINE PRNICP

SUBROUTINE ADDMVA(ISLCT,NATOM,NSLCT)
  !
  !     This routine adds selected atoms.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none
  !
  INTEGER NATOM,NSLCT
  INTEGER I
  INTEGER ISLCT(*)
  !
  NSLCT=0
  DO I=1,NATOM
     IF(ISLCT(I) == 1) NSLCT=NSLCT+1
  enddo
  RETURN
END SUBROUTINE ADDMVA

SUBROUTINE GETMVA(ISLCT,ICPMVA,NATOM)
  !
  !     This routine fills ICPMVA with atom numbers of the atoms
  !     selected in ISLCT.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none
  !
  INTEGER ICPMVA(*)
  INTEGER NATOM
  INTEGER I,J
  INTEGER ISLCT(*)
  !
  J=0
  DO I=1,NATOM
     IF(ISLCT(I) == 1) THEN
        J=J+1
        ICPMVA(J)=I
     ENDIF
  enddo
  RETURN
END SUBROUTINE GETMVA

SUBROUTINE INITSL(ISLCT,JSLCT,NATOM)
  !
  !     Initializes ISLCT and JSLCT.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none

  INTEGER NATOM,I
  INTEGER ISLCT(*),JSLCT(*)

  ISLCT(1:natom)=0
  JSLCT(1:natom)=1

  RETURN
END SUBROUTINE INITSL

SUBROUTINE GETISL(ICPMVA,ISLCT,NMOVE)
  !
  !     Fills the ISLCT array.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none
  !
  INTEGER ICPMVA(*)
  INTEGER NMOVE
  INTEGER I,J
  INTEGER ISLCT(*)
  !
  DO I=1,NMOVE
     J=ICPMVA(I)
     ISLCT(J)=1
  enddo
  RETURN
END SUBROUTINE GETISL

SUBROUTINE DYNICP(X,Y,Z,NATOM,NPRIV,AKMATI,TOTE,TOTKE, &
     XSAVE,YSAVE,ZSAVE)
  !
  !     This is the control routine for ic perturbations during dynamics.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use dimens_fcm
  use icpert
  use number
  use stream
  use intcor2,only:geticv
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XSAVE(*),YSAVE(*),ZSAVE(*)
  real(chm_real) TOTE,TOTKE,AKMATI
  real(chm_real) ESBNP,ESBFP,ESBRP
  real(chm_real) SCALE,DSCALE
  real(chm_real) RIJ,TIJK,PIJKL,TJKL,RKL
  real(chm_real) ICVAL(3,MXICP)
  CHARACTER(LEN=80) ERRLIN
  INTEGER NATOM,NPRIV
  INTEGER I,J,K,L,IC,INC,ICOL
  LOGICAL LPRINT
  ! for running averages --RJP
  real(chm_real) DELTAE,DELTAA,STEPSZ,ENEDIF,FRACTI
  real(chm_real) RTSMST,POSITI
  INTEGER,save :: STPCNT=0,KK,PRTCNT=0
  LOGICAL QPRINT
  !
  !     Find out if ic pert data should be written.
  !
  LPRINT=(IUNICP > 0.AND.MOD(NPRIV,ISVICP) == 0)
  !
  !     If not, return.
  !
  IF(.NOT.LPRINT) RETURN
  !
  !     Otherwise...
  !     Increment counters if averaging -RJP
  IF (LRUNNA) THEN
     RTSMST = TSMSTP
     FRACTI = RTSMST/(RTSMST + 1)
     TSMSTP = TSMSTP + 1  !counter for perturbation calcs
     STPCNT = STPCNT + 1  !counter for printing out averages
     QPRINT = .FALSE.
     PRTCNT = PRTCNT + 1 !counter for writing out raw data
     ! set print flag if right step
     IF (PRTCNT == PEVERY) THEN
        QPRINT = .TRUE.
        PRTCNT = 0
     ENDIF
  ENDIF
  !
  !     Get the interaction energy for the unperturbed ic's and
  !     write out some data.
  CALL GETICV(7,10,14,11,.FALSE.,RIJ,TIJK,PIJKL, &
       TJKL,RKL,X,Y,Z)
  !
  CALL EIICP(ESBNP,X,Y,Z)
  ICOL=1
  DO IC=1,NICP
     I=ICPATN(1,IC)
     J=ICPATN(2,IC)
     K=ICPATN(3,IC)
     L=ICPATN(4,IC)
     CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
     IF(ICPTYP(IC) == 1) THEN
        ICVAL(ICOL,IC)=RIJ
     ELSE IF(ICPTYP(IC) == 2) THEN
        ICVAL(ICOL,IC)=TIJK
     ELSE IF(ICPTYP(IC) == 3) THEN
        ICVAL(ICOL,IC)=PIJKL
     ENDIF
  enddo
  IF ((.NOT.LRUNNA).OR.(QPRINT))   & !RJP
       WRITE(IUNICP,100) NPRIV,AKMATI,TOTE,TOTKE,ESBNP
  !
  !     Save the coordinates of the atoms which move with the perturbation.
  !
  DO IC=1,NICP
     CALL SAVICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
     IF(LMOV2(IC)) &
          CALL SAVICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
  enddo

  !     Get the perturbation data for each increment.
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
     CALL EIICP(ESBFP,X,Y,Z)
     !
     ICOL=2
     DO IC=1,NICP
        I=ICPATN(1,IC)
        J=ICPATN(2,IC)
        K=ICPATN(3,IC)
        L=ICPATN(4,IC)
        CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
        IF(ICPTYP(IC) == 1) THEN
           ICVAL(ICOL,IC)=RIJ
        ELSE IF(ICPTYP(IC) == 2) THEN
           ICVAL(ICOL,IC)=TIJK
        ELSE IF(ICPTYP(IC) == 3) THEN
           ICVAL(ICOL,IC)=PIJKL
        ENDIF
     enddo
     !
     !     Restore the unperturbed coordinates.
     !
     DO IC=1,NICP
        CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
        IF(LMOV2(IC)) &
             CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
     enddo
     !
     !     Move the atoms for the reverse (-dzeta) perturbation.
     !
     CALL MVICP(X,Y,Z,MINONE,SCALE)
     !
     !     Get the interaction energy for the reverse perturbation and write
     !     out the energies and ic values.
     !
     CALL EIICP(ESBRP,X,Y,Z)
     !
     IF ((.NOT.LRUNNA).OR.(QPRINT))   & !RJP
          WRITE(IUNICP,101) SCALE,ESBFP,ESBRP
     ICOL=3
     DO IC=1,NICP
        I=ICPATN(1,IC)
        J=ICPATN(2,IC)
        K=ICPATN(3,IC)
        L=ICPATN(4,IC)
        CALL GETICV(I,J,K,L,.FALSE.,RIJ,TIJK,PIJKL,TJKL,RKL,X,Y,Z)
        IF(ICPTYP(IC) == 1) THEN
           ICVAL(ICOL,IC)=RIJ
        ELSE IF(ICPTYP(IC) == 2) THEN
           ICVAL(ICOL,IC)=TIJK
        ELSE IF(ICPTYP(IC) == 3) THEN
           ICVAL(ICOL,IC)=PIJKL
        ENDIF
        IF (.NOT.LNPRIC) THEN
           IF ((.NOT.LRUNNA).OR.(QPRINT))    & !RJP
                WRITE(IUNICP,102) IC,ICPTYP(IC),ICVAL(1,IC), &
                ICVAL(2,IC),ICVAL(3,IC)
        ENDIF
     enddo
     !
     !     Restore the unperturbed coordinates.
     !
     DO IC=1,NICP
        CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV1(IC),ICPMV1(IC)%a)
        IF(LMOV2(IC)) &
             CALL RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOV2(IC),ICPMV2(IC)%a)
     enddo
     ! *******running averages ****** RJP
     IF (LRUNNA) THEN
        ! first forward
        ENEDIF = (ESBFP - ESBNP)/KTEMPE
        DELTAE = ENEDIF - MINFOR(INC)
        IF (DELTAE < 0) THEN
           MINFOR(INC) = ENEDIF
           SUMFOR(INC) = (SUMFOR(INC)*EXP(DELTAE)) + 1
        ELSE
           SUMFOR(INC) = SUMFOR(INC) + EXP(-DELTAE)
        ENDIF
        EAVFOR(INC) = (EAVFOR(INC)*FRACTI) + (ENEDIF/TSMSTP)
        ! backward
        ENEDIF = (ESBRP - ESBNP)/KTEMPE
        DELTAE = ENEDIF - MINBAC(INC)
        IF (DELTAE < 0) THEN
           MINBAC(INC) = ENEDIF
           SUMBAC(INC) = (SUMBAC(INC)*EXP(DELTAE)) + 1
        ELSE
           SUMBAC(INC) = SUMBAC(INC) + EXP(-DELTAE)
        ENDIF
        EAVBAC(INC) = (EAVBAC(INC)*FRACTI) + (ENEDIF/TSMSTP)
        ! average values for internal coordinates (unperturbed)
        DO IC=1,NICP
           ICAVER(IC) = ICAVER(IC)*FRACTI + (ICVAL(1,IC)/TSMSTP)
        ENDDO
     ENDIF  !if running averages
     !  end of running averages ******************************
  enddo
  !  print running averages if appropriate -RJP
  IF (LRUNNA) THEN
     IF (STPCNT == RPRCYC) THEN
        STPCNT = 0
        STEPSZ = ONE/ICPINC
        ! start from the perturbations furthest back and work forward
        ! do back perturbations in reverse order
        DO KK = 1,ICPINC
           INC = ICPINC - KK + 1
           DELTAA = MINBAC(INC) - LOG(SUMBAC(INC)/TSMSTP)
           DELTAA = DELTAA*KTEMPE
           POSITI = -INC*STEPSZ*DZETA(1) + ICAVER(1)
           WRITE(RUNITN,'(A6,1X,I14,1X,F9.4,1X,F14.8,1X,F14.8)') &
                'RUNAV>',TSMSTP,POSITI,DELTAA,EAVBAC(INC)*KTEMPE
        ENDDO
        ! do forward perturbations
        DO INC = 1,ICPINC
           DELTAA = MINFOR(INC) -LOG(SUMFOR(INC)/TSMSTP)
           DELTAA = DELTAA*KTEMPE
           POSITI = INC*STEPSZ*DZETA(1) + ICAVER(1)
           WRITE(RUNITN,'(A6,1X,I14,1X,F9.4,1X,F14.8,1X,F14.8)') &
                'RUNAV>',TSMSTP,POSITI,DELTAA,EAVFOR(INC)*KTEMPE
        ENDDO
        ! do internal coordinates (unperturbed only)
        DO IC=1,NICP
           WRITE(RUNITN,'(A7,I4,F14.8)') &
                'RUNAVI>',IC,ICAVER(IC)
        ENDDO  !loop over IC's
     ENDIF !if STPCNT == RPRCYC
  ENDIF !if running average
  !
100 FORMAT(I7,F10.4,3D16.8)
101 FORMAT(7X,F10.4,2D16.8)
102 FORMAT(9X,2I4,3D16.8)
  RETURN
END SUBROUTINE DYNICP

SUBROUTINE SAVICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOVE,ICPMVA)
  !
  !     This routine saves the coordinates of the move atoms.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),XSAVE(*),YSAVE(*),ZSAVE(*)
  INTEGER ICPMVA(*)
  INTEGER NMOVE,I
  !
  DO I=1,NMOVE
     XSAVE(ICPMVA(I))=X(ICPMVA(I))
     YSAVE(ICPMVA(I))=Y(ICPMVA(I))
     ZSAVE(ICPMVA(I))=Z(ICPMVA(I))
  enddo
  RETURN
END SUBROUTINE SAVICP

SUBROUTINE RSTICP(X,Y,Z,XSAVE,YSAVE,ZSAVE,NMOVE,ICPMVA)
  !
  !     This routine restores the coordinates of the move atoms.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),XSAVE(*),YSAVE(*),ZSAVE(*)
  INTEGER ICPMVA(*)
  INTEGER NMOVE,I
  !
  DO I=1,NMOVE
     X(ICPMVA(I))=XSAVE(ICPMVA(I))
     Y(ICPMVA(I))=YSAVE(ICPMVA(I))
     Z(ICPMVA(I))=ZSAVE(ICPMVA(I))
  enddo
  RETURN
END SUBROUTINE RSTICP

SUBROUTINE MVICP(X,Y,Z,FAC,SCALE)
  !
  !     This is a dummy routine to call MVICP2.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use dimens_fcm
  use number
  use icpert
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),FAC,SCALE,DZ
  INTEGER ICP
  !
  !     Loop over perturbations.
  !
  DO ICP=1,NICP
     IF(LMOV2(ICP)) THEN
        DZ=SCALE*DZETA(ICP)/TWO
        CALL MVICP2(X,Y,Z,FAC, ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZ)
        DZ=-DZ
        CALL MVICP2(X,Y,Z,FAC, ICP,ICPTYP,ICPATN,NMOV2, &
             ICPMV2(ICP)%a,DZ)
     ELSE
        DZ=SCALE*DZETA(ICP)
        CALL MVICP2(X,Y,Z,FAC, ICP,ICPTYP,ICPATN,NMOV1, &
             ICPMV1(ICP)%a,DZ)
     ENDIF
  enddo
  RETURN
END SUBROUTINE MVICP

SUBROUTINE MVICP2(X,Y,Z,FAC, ICP,ICPTYP,ICPATN,NMVICP, &
     ICPMVA,DZETA)
  !
  !     This routine moves atoms involved in IC perturbations.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use consta
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),FAC,DZETA
  real(chm_real) XIJ,YIJ,ZIJ,XJK,YJK,ZJK,XJN,YJN,ZJN
  real(chm_real) DRX,DRY,DRZ,NX,NY,NZ,NORM
  real(chm_real) A(3,3),DPHI
  INTEGER ICPTYP(*),ICPATN(4,*),NMVICP(*),ICPMVA(*)
  INTEGER I,J,K,L,M,N,ICP,ISTOP
  !
  ISTOP=NMVICP(ICP)
  !
  IF(ICPTYP(ICP) == 1) THEN
     !
     !     distance perturbation
     !
     I=ICPATN(1,ICP)
     J=ICPATN(2,ICP)
     XIJ=X(J)-X(I)
     YIJ=Y(J)-Y(I)
     ZIJ=Z(J)-Z(I)
     NORM=SQRT(XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ)
     NORM=FAC*DZETA/NORM
     DRX=NORM*XIJ
     DRY=NORM*YIJ
     DRZ=NORM*ZIJ
     DO M=1,ISTOP
        N=ICPMVA(M)
        X(N)=X(N)+DRX
        Y(N)=Y(N)+DRY
        Z(N)=Z(N)+DRZ
     enddo
     !
  ELSE IF(ICPTYP(ICP) == 2) THEN
     !
     !     bond angle perturbation
     !
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
     DPHI=DZETA*DEGRAD
     CALL AROT(NX,NY,NZ,FAC,DPHI,A)
     DO M=1,ISTOP
        N=ICPMVA(M)
        XJN=X(N)-X(J)
        YJN=Y(N)-Y(J)
        ZJN=Z(N)-Z(J)
        X(N)=X(J)+A(1,1)*XJN+A(1,2)*YJN+A(1,3)*ZJN
        Y(N)=Y(J)+A(2,1)*XJN+A(2,2)*YJN+A(2,3)*ZJN
        Z(N)=Z(J)+A(3,1)*XJN+A(3,2)*YJN+A(3,3)*ZJN
     enddo
     !
  ELSE IF(ICPTYP(ICP) == 3) THEN
     !
     !     dihedral angle perturbation
     !
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
     DPHI=DZETA*DEGRAD
     CALL AROT(NX,NY,NZ,FAC,DPHI,A)
     DO M=1,ISTOP
        N=ICPMVA(M)
        XJN=X(N)-X(J)
        YJN=Y(N)-Y(J)
        ZJN=Z(N)-Z(J)
        X(N)=X(J)+A(1,1)*XJN+A(1,2)*YJN+A(1,3)*ZJN
        Y(N)=Y(J)+A(2,1)*XJN+A(2,2)*YJN+A(2,3)*ZJN
        Z(N)=Z(J)+A(3,1)*XJN+A(3,2)*YJN+A(3,3)*ZJN

        !
     enddo
  ENDIF
  RETURN
END SUBROUTINE MVICP2

SUBROUTINE AROT(NX,NY,NZ,FAC,DPHI,A)
  !
  !     This routine finds a matrix to rotate a vector DPHI degrees
  !     about the vector N.
  !
  !     Author: Doug Tobias
  !
  use chm_kinds
  use number
  implicit none
  !
  real(chm_real) NX,NY,NZ,FAC,DPHI
  real(chm_real) A(3,3)
  real(chm_real) DP2,SINDP2,E0,E1,E2,E3
  real(chm_real) E0S,E1S,E2S,E3S,E1E2,E2E3,E1E3
  real(chm_real) FACE0,FACE01,FACE02,FACE03
  !
  DP2=HALF*DPHI
  E0=COS(DP2)
  SINDP2=SIN(DP2)
  E1=NX*SINDP2
  E2=NY*SINDP2
  E3=NZ*SINDP2
  E0S=E0*E0
  E1S=E1*E1
  E2S=E2*E2
  E3S=E3*E3
  E1E2=E1*E2
  E2E3=E2*E3
  E1E3=E1*E3
  FACE0=FAC*E0
  FACE01=FACE0*E1
  FACE02=FACE0*E2
  FACE03=FACE0*E3
  A(1,1)=E0S+E1S-E2S-E3S
  A(1,2)=TWO*(E1E2-FACE03)
  A(1,3)=TWO*(E1E3+FACE02)
  A(2,1)=TWO*(E1E2+FACE03)
  A(2,2)=E0S-E1S+E2S-E3S
  A(2,3)=TWO*(E2E3-FACE01)
  A(3,1)=TWO*(E1E3-FACE02)
  A(3,2)=TWO*(E2E3+FACE01)
  A(3,3)=E0S-E1S-E2S+E3S
  RETURN
END SUBROUTINE AROT

SUBROUTINE EIICP(EINICP,X,Y,Z)
  !
  !     This routine calculates the interaction energy of the atoms in
  !     the perturbed ic's with the rest of the system.  This routine
  !     and the associated routines below are tailored versions of the
  !     original charmm interaction energy routines.
  !
  !     Author: Doug Tobias
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
  use datstr
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: ihdxp, ihdyp, ihdzp
  integer,allocatable,dimension(:) :: iskip
  real(chm_real),allocatable,dimension(:) :: rtemp
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) EINICP
  INTEGER LEN,OLDFSTR,OLDLFST
  !
  !
  CALL FREEDT_nbond(BNBNDC)
  CALL DUPLDT_nbond(BNBNDC,BNBND)

  call chmalloc('icpert.src','EIICP','ihdxp',maxa,crl=ihdxp)
  call chmalloc('icpert.src','EIICP','ihdyp',maxa,crl=ihdyp)
  call chmalloc('icpert.src','EIICP','ihdzp',maxa,crl=ihdzp)

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
  call chmalloc('icpert.src','EIICP','iskip',len,intg=iskip)
  call chmalloc('icpert.src','EIICP','rtemp',natom,crl=rtemp)

  CALL EIICP2(EINICP,X,Y,Z,BNBNDC, &
       ISLICP,JSLICP,ISKIP,RTEMP, &
       BNBNDC%INBLO,BNBNDC%JNB, &
       BNBND%INBLO,BNBND%JNB,BNBNDC%INBLOG, &
       IHDXP,IHDYP,IHDZP)
  CALL FREEDT_nbond(BNBNDC)
  call chmdealloc('icpert.src','EIICP','iskip',len,intg=iskip)
  call chmdealloc('icpert.src','EIICP','rtemp',natom,crl=rtemp)
  FASTER=OLDFSTR
  LFAST=OLDLFST
  !
  call chmdealloc('icpert.src','EIICP','ihdxp',maxa,crl=ihdxp)
  call chmdealloc('icpert.src','EIICP','ihdyp',maxa,crl=ihdyp)
  call chmdealloc('icpert.src','EIICP','ihdzp',maxa,crl=ihdzp)
  !
  RETURN
END SUBROUTINE EIICP

SUBROUTINE EIICP2(EINICP,X,Y,Z,BNBND,ISLCT,JSLCT,ISKIP,RTEMP, &
     INBLO,JNB,INBLOX,JNBX,INBLOG,DX,DY,DZ)
  !
  !     This routine does the actual work of the interaction energies.
  !
#if KEY_RMD==1
  use cross, only: ecross,NCRUN                       
#endif
#if KEY_MRMD==1
  use mrmd_fcm,only: emrmd,mrmd_active
#endif
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
  use parallel
  use cmapm
#if KEY_FLUCQ==1
  use flucq                        
#endif
  use usermod,only: usere
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif 
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) EINICP
  real(chm_real) RTEMP(*)
  type(nonbondDataStructure) BNBND

  INTEGER INBLO(*),INBLOX(*),INBLOG(*)
  INTEGER I,J,K,N,IFIRST,ILAST
  INTEGER JNB(*),JNBX(*)
  INTEGER ISLCT(*),JSLCT(*),ISKIP(*)
  !

#if KEY_DOMDEC==1
  if (q_domdec) then
     call wrndie(-5,'<icpert>','EIICP2 not ready for DOMDEC')
  endif
#endif 

  EINICP=ZERO
  DO I=1,LENENT
     ETERM(I) = ZERO
  ENDDO
  DO I=1,NATOM
     DX(I)=ZERO
     DY(I)=ZERO
     DZ(I)=ZERO
  ENDDO
  !
  !     Get the various energy terms.
  !
#if KEY_PARALLEL==1
  IF(MYNOD == 0) THEN   
#endif
     IF(QETERM(USER)) &
          CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ,.FALSE.,[zero],NATOM)
#if KEY_RMD==1
     IF((NCRUN > 0).AND.QETERM(CROS)) THEN
        CALL ECROSS(ETERM(CROS),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#endif 
#if KEY_MRMD==1
     IF(MRMD_ACTIVE.AND.QETERM(MRMD)) THEN
        CALL EMRMD(ETERM(MRMD),X,Y,Z,DX,DY,DZ,NATOM)
     ENDIF
#endif 
#if KEY_PARALLEL==1
  ENDIF                 
#endif
  !
  IF(NBOND > 0.AND.QETERM(BOND)) THEN
     DO I=1,NBOND
        IF((ISLCT(IB(I)) == 1.AND.JSLCT(JB(I)) == 1).OR. &
             (ISLCT(JB(I)) == 1.AND.JSLCT(IB(I)) == 1)) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL EBOND(ETERM(BOND),IB,JB,ICB,NBOND,CBC,CBB,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

  ENDIF
  !
  IF(NTHETA > 0.AND.QETERM(ANGLE)) THEN
     DO I=1,NTHETA
        IF(ISLCT(JT(I)) == 1.AND.JSLCT(JT(I)) == 1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL EANGLE(ETERM(ANGLE),IT,JT,KT,ICT,NTHETA,CTC,CTB,DX,DY,DZ, &
          X,Y,Z,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

  ENDIF
  !
  IF(NPHI > 0.AND.QETERM(DIHE)) THEN
     DO I=1,NPHI
        IF((ISLCT(JP(I)) == 1.AND.JSLCT(KP(I)) == 1).OR. &
             (ISLCT(KP(I)) == 1.AND.JSLCT(JP(I)) == 1)) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL EPHI(ETERM(DIHE),IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB, &
          CPCOS,CPSIN,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

  ENDIF
  !
  IF(NIMPHI > 0.AND.QETERM(IMDIHE)) THEN
     DO I=1,NIMPHI
        IF(ISLCT(IM(I)) == 1.AND.JSLCT(IM(I)) == 1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL EPHI(ETERM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI,CIC,CID,CIB, &
          CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

  ENDIF

#if KEY_CMAP==1
  IF(NCRTERM > 0.AND.QETERM(CMAP)) THEN
     DO I=1,NCRTERM
        IF(ISLCT(I1CT(I)) == 1.AND.JSLCT(I1CT(I)) == 1.AND. &
             ISLCT(I2CT(I)) == 1.AND.JSLCT(I2CT(I)) == 1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL ECMAP(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
          I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
          DX,DY,DZ,X,Y,Z, &
          .FALSE., (/ZERO/), 1, ISKIP, (/ZERO/), (/0/), .FALSE.)
  ENDIF
#endif 

  !
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
              ENDIF
           enddo
        ELSE IF(JSLCT(I) == 1) THEN
           DO J=IFIRST,ILAST
              K=JNBX(J)
              IF(K < 0) K=-K
              IF(ISLCT(K) == 1) THEN
                 N=N+1
                 JNB(N)=JNBX(J)
              ENDIF
           enddo
        ENDIF
        INBLO(I)=N
        IFIRST=ILAST+1
     enddo

     INBLOG(1:ngrp)=0

     CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBND, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ, &
          X,Y,Z,.FALSE.,(/ZERO/),.TRUE.,ZERO,(/ZERO/),(/0/),.FALSE.,QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                & 
#endif
          .FALSE.,NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA             & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

  ENDIF
  !
#if KEY_PARALLEL==1
  IF(MYNOD == 0)THEN                   
#endif
     IF(NHB > 0.AND.QETERM(HBOND)) THEN
        DO I=1,NHB
           IF((ISLCT(IHB(I)) == 1.AND.JSLCT(JHB(I)) == 1).OR. &
                (ISLCT(JHB(I)) == 1.AND.JSLCT(IHB(I)) == 1)) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        enddo
        CALL EHBOND(ETERM(HBOND),IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
             DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,ISKIP,CTONHB, &
             CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
     ENDIF
#if KEY_PARALLEL==1
  ENDIF                                
#endif
  !
  IF(QCNSTR.AND.QETERM(CHARM)) THEN
     DO I=1,NATOM
        IF(ISLCT(I) == 1.AND.JSLCT(I) == 1) THEN
           RTEMP(I)=KCNSTR(I)
        ELSE
           RTEMP(I)=ZERO
        ENDIF
     enddo
     CALL ECNSTR(ETERM(CHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          .FALSE., (/ ZERO /), (/ ZERO /), (/ 0 /), .FALSE. &
          ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
          )

  ENDIF
  !
  !
  IF(QCNSTR.AND.QETERM(PCHARM).and.numpca>0) THEN
     DO I=1,NATOM
        IF(ISLCT(I) == 1.AND.JSLCT(I) == 1) THEN
           RTEMP(I)=KCNSTR(I)
        ELSE
           RTEMP(I)=ZERO
        ENDIF
     enddo
     CALL ECNSTR(ETERM(PCHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          .FALSE., (/ ZERO /), (/ ZERO /), (/ 0 /), .FALSE. &
          ,numpca,pcax,pcay,pcaz &
          )

  ENDIF
  !
  IF((NCSPHI > 0).AND.QETERM(CDIHE)) THEN
     DO I=1,NCSPHI
        IF((ISLCT(JCS(I)) == 1.AND.JSLCT(KCS(I)) == 1).OR. &
             (ISLCT(KCS(I)) == 1.AND.JSLCT(JCS(I)) == 1)) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     !  ***** ATTENTION *****  I changed 0 -> CCSD  AB.94.
#if KEY_DOMDEC==1
     if (q_domdec) then
        CALL WRNDIE(-5,'<EIICP2>',&
             'HARMONIC RESTRAINTS NOT READY FOR DOMDEC')
     endif
#endif 
     CALL EPHI(ETERM(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
          CCSC,CCSD,CCSB,CCSCOS,CCSSIN,DX,DY,DZ,X,Y,Z, &
          .TRUE.,CCSW,.FALSE.,(/ZERO/),1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

     ! AB.
  ENDIF
  !
  IF(NTRANS > 0) CALL IMINTR(X,Y,Z,DX,DY,DZ,ISLCT,JSLCT)
  !
  EINICP=ETERM(TSM)+ETERM(BOND)+ETERM(ANGLE)+ETERM(DIHE)+ &
       ETERM(IMDIHE)+ETERM(VDW)+ETERM(ELEC)+ETERM(HBOND)+ &
       ETERM(CHARM)+ETERM(CDIHE)+ETERM(IMVDW)+ETERM(IMHBND)+ &
       ETERM(IMELEC)+ETERM(IMST2)
  !
#if KEY_PARALLEL==1
  CALL GCOMB(EINICP,1)                
#endif
  !
  !     Currently the ic pert forces are not used for anything, so
  !     the following code is not needed.
  !      DO 5000 I=1,NATOM
  !        IF(IMOVE(I) > 0) THEN
  !          DX(I)=ZERO
  !          DY(I)=ZERO
  !          DZ(I)=ZERO
  !        ENDIF
  !5000  CONTINUE
  RETURN
END SUBROUTINE EIICP2

SUBROUTINE IMINTR(X,Y,Z,DX,DY,DZ,ISLCT1,JSLCT1)
  !
  !     Computes image non-bonded interaction energies between selected
  !     sets of atoms
  !
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use psf
  use hbondm
  use image
  use memory
  use inbnd
  use icpert
  use datstr,only:dupldt_image,freedt_image
  implicit none
  !
  integer,allocatable,dimension(:) :: iskip2, islct2, jslct2
  real(chm_real),allocatable,dimension(:) :: rtemp
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  !
  !
  INTEGER ISLCT1(*),JSLCT1(*)
  !
  ! DUPLICATE THE BIMAG STRUCTURE SO THAT A NEW NON-BONDED LIST
  ! CAN BE GENERATED
  !
  CALL DUPLDT_image(BIMAGC,BIMAG)
  !
  ! EXTEND THE ATOM SELECTION LIST TO INCLUDE IMAGE ATOMS
  !
  call chmalloc('icpert.src','IMINTR','iskip2',nimhb,intg=iskip2)
  call chmalloc('icpert.src','IMINTR','islct2',natim,intg=islct2)
  call chmalloc('icpert.src','IMINTR','jslct2',natim,intg=jslct2)
  call chmalloc('icpert.src','IMINTR','rtemp',natim,crl=rtemp)

  CALL EXTSEL(NATOM,ISLCT1,JSLCT1,NATIM,ISLCT2,JSLCT2,BIMAG%IMATTR)
  !
  ! NOW CALL IMINT2 WITH ARRAYS FOR IMAGE NON-BONDED LISTS ETC.
  !
  CALL IMINT2(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,BIMAGC,ISLCT2,JSLCT2,ISKIP2)
  CALL FREEDT_image(BIMAGC)
  call chmdealloc('icpert.src','IMINTR','iskip2',nimhb,intg=iskip2)
  call chmdealloc('icpert.src','IMINTR','islct2',natim,intg=islct2)
  call chmdealloc('icpert.src','IMINTR','jslct2',natim,intg=jslct2)
  call chmdealloc('icpert.src','IMINTR','rtemp',natim,crl=rtemp)
  RETURN
END SUBROUTINE IMINTR

SUBROUTINE EXTSEL(NATOM,ISLCT1,JSLCT1,NATIM,ISLCT2,JSLCT2,IMATTR)
  !
  ! THIS ROUTINE EXTENDS THE SELECTION LIST TO INCLUDE PAIRS OF IMAGE ATOMS.
  !
  use chm_kinds
  implicit none
  INTEGER ISLCT1(*),JSLCT1(*),ISLCT2(*),JSLCT2(*),IMATTR(*)
  INTEGER NATOM,NATIM,I
  !
  ! FIRST COPY PRIMARY ATOM LIST
  !
  ISLCT2(1:natom)=ISLCT1(1:natom)
  JSLCT2(1:natom)=JSLCT1(1:natom)
  !
  ! GET IMAGE ATOMS
  !
  DO I=NATOM+1,NATIM
     ISLCT2(I)=ISLCT1(IMATTR(I))
     JSLCT2(I)=JSLCT1(IMATTR(I))
  enddo
  RETURN
END SUBROUTINE EXTSEL

SUBROUTINE IMINT2(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,BIMAGC, &
     ISLCT,JSLCT,ISKIP)
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor           
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use code
  use enbond_mod
  use energym
  use hbondm
  use image
  use image_routines_module, only: transo
  use inbnd
  use number
  use param
  use psf
  use fast
#if KEY_FLUCQ==1
  use flucq     
#endif
  use datstr
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  type(imageDataStructure) BIMAGC
  !
  real(chm_real) ENBX,EELX,EST2X
  INTEGER I,IMX
  INTEGER ISLCT(*),JSLCT(*),ISKIP(*)
  !      INTEGER BDUMMY(SNBNDT+1)
  type(nonbondDataStructure) BDUMMY
  !
  IF(NTRANS == 0) RETURN
  !
  ! SET UP A DUMMY DATA STRUCTURE FOR ENBOND.
  !
  !!      DO 1000 I=1,SNBND
  !!        BDUMMY(I)=BNBND(I)
  !!1000  CONTINUE
  CALL ALIASDT_nbond(BDUMMY,BNBND)

  !rn...Removed per Doug Tobias 4/12/95
  !      IF(.NOT.(LHEAP(BNBND(LNBOPT)+11))) THEN
  !        CALL WRNDIE(1,'<IMINT2>',
  !     & 'ST2 INTERACTIONS MUST BE COMPUTED FROM LISTS. FLAG CHANGED')
  !        LHEAP(BNBND(LNBOPT)+11)=.TRUE.
  !      ENDIF
  !rn...
  !
  ! CONSTRUCT COORDINATES FOR ALL IMAGE ATOMS.
  !
  CALL TRANSO(X,Y,Z,DX,DY,DZ,.TRUE.,.FALSE., &
       (/ zero /), &
       NATOM,NTRANS, &
       IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
       NOROT,NATIM &
#if KEY_FLUCQ==1
       ,QFLUC,CG,FQCFOR      & 
#endif
       )
  !
  ETERM(IMVDW)=ZERO
  ETERM(IMELEC)=ZERO
  ETERM(IMST2)=ZERO
  ENBX=ZERO
  EELX=ZERO
  EST2X=ZERO
  !
  ! CHECK IF ANY SELF-ENERGY TERMS ARE PRESENT
  !
  CALL UPDNIM(NATIM,BIMAGC%IMJNBS,BIMAGC%NIMNBS, &
       BIMAGC%IMBLOX,BIMAGC%NIMNBX)
  IF(BIMAG%NIMNBS > 0.OR.BIMAG%NIMNBX > 0) THEN
     !
     ! SELF TERMS ARE PRESENT
     !
     IMX=NATIM
     !     Currently the ic pert forces are not used for anything, so
     !     the following code is not needed.
     !        DO 2000 I=1,IMX
     !          DX(I)=DX(I)*TWO
     !          DY(I)=DY(I)*TWO
     !          DZ(I)=DZ(I)*TWO
     !2000    CONTINUE
     CALL NONLST(NATIM,NIMGRP,BIMAG%IMBLOS, &
          BIMAG%IMJNBS,BIMAGC%IMBLOS, &
          BIMAGC%IMJNBS,ISLCT,JSLCT,BIMAGC%IMBLOX)
     CALL NBSET_FROM_IMG_SX(BDUMMY, BIMAGC)
     CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAGC)
     CALL ENBOND(ENBX,EELX,BDUMMY, &
          1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),QETERM(EWEXCL),ETERM(EWEXCL), &
          (/ZERO/),(/0/),.FALSE.,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                & 
#endif
          QETERM(IMST2),NST2,EST2X,.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA             & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

     !
     ENBX=ENBX*HALF
     EELX=EELX*HALF
     EST2X=EST2X*HALF
     !     Currently the ic pert forces are not used for anything, so
     !     the following code is not needed.
     !        DO 2100 I=1,IMX
     !          DX(I)=DX(I)*HALF
     !          DY(I)=DY(I)*HALF
     !          DZ(I)=DZ(I)*HALF
     !2100    CONTINUE
  ENDIF
  !
  ! COMPUTE IMAGE NONBONDED ENERGIES
  !
  CALL UPDNIM(NATIM,BIMAGC%IMJNB,BIMAGC%NIMNB, &
       BIMAGC%IMBLOG,BIMAGC%NIMNBG)
  IF(BIMAGC%NIMNB > 0.OR.BIMAGC%NIMNBG > 0) THEN
     CALL NONLST(NATIM,NIMGRP,BIMAG%IMBLO, &
          BIMAG%IMJNB,BIMAGC%IMBLO, &
          BIMAGC%IMJNB,ISLCT,JSLCT,BIMAGC%IMBLOG)
     CALL NBSET_FROM_IMG_G(BDUMMY, BIMAGC)
     CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAGC)
     CALL ENBOND(ETERM(IMVDW),ETERM(IMELEC),BDUMMY, &
          1,NATIM,CG,RSCLF,NIMGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/),QETERM(EWEXCL),ETERM(EWEXCL), &
          (/ZERO/),(/0/),.FALSE.,QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                & 
#endif
          .FALSE.,NST2,EST2X,.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA             & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )

  ENDIF
  !
  ! COMPUTE IMAGE HBOND ENERGIES
  !
  ETERM(IMHBND)=ZERO
  IF(NIMHB > 0.AND.QETERM(IMHBND)) THEN
     DO I=1,NHB+1,NIMHB
        IF((ISLCT(IHB(I)) == 1.AND.JSLCT(JHB(I)) == 1).OR. &
             (ISLCT(JHB(I)) == 1.AND.JSLCT(IHB(I)) == 1)) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     enddo
     CALL EHBOND(ETERM(IMHBND),IHB(NHB+1),JHB(NHB+1),KHB(NHB+1), &
          LHB(NHB+1),ICH(NHB+1),NIMHB,CHBA,CHBB, &
          DX,DY,DZ,X,Y,Z,.FALSE.,0,0,0,1,ISKIP,CTONHB, &
          CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
  ENDIF
  !
  ETERM(IMVDW)=ETERM(IMVDW)+ENBX
  ETERM(IMELEC)=ETERM(IMELEC)+EELX
  ETERM(IMST2)=ETERM(IMST2)+EST2X
  !
  !     Currently we don't use the forces for anything, so we don't
  !     need the call to transi.
  !      CALL TRANSI(X,Y,Z,DX,DY,DZ,.FALSE.,0,NATOM,NTRANS,
  !     & IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR,NOROT,NATIM,
  !     & IMINV,IMFORC,IMTORQ)
  !
  RETURN
END SUBROUTINE IMINT2

SUBROUTINE NONLST(NATOM,NGRP,INBLOX,JNBX,INBLO,JNB,ISLCT,JSLCT, &
     INBLOG)
  !
  !     MAKE NEW NONBOND LISTS
  !
  use chm_kinds
  implicit none
  INTEGER JNBX(*),JNB(*),ISLCT(*),JSLCT(*)
  INTEGER INBLOX(*),INBLO(*),INBLOG(*)
  INTEGER NATOM,NGRP
  !
  INTEGER IFIRST,ILAST,I,J,K,N
  !
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
           ENDIF
        enddo
     ELSE IF(JSLCT(I) == 1) THEN
        DO J=IFIRST,ILAST
           K=JNBX(J)
           IF(K < 0) K=-K
           IF(ISLCT(K) == 1) THEN
              N=N+1
              JNB(N)=JNBX(J)
           ENDIF
        enddo
     ENDIF
     INBLO(I)=N
     IFIRST=ILAST+1
  enddo

  INBLOG(1:ngrp)=0

  return
end SUBROUTINE NONLST

#endif 

SUBROUTINE NULL_IP
  RETURN
END SUBROUTINE NULL_IP

