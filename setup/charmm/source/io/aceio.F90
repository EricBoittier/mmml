#if KEY_ACE==1
!======================================================================
!
SUBROUTINE ACEIO(IUNIT,NTITL,TITLE,ICARD,JUNIT)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE CALLS ACERDR to read the atom-type arrays
  !       EFVOL, CHRAD, ACEHYD, ACEA0, ACEA1, ACEA2 (param.f90).
  !     Distribution to all nodes in parallel mode.
  !
  !     Author: Michael Schaefer, ULP Strasbourg
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use param
  use parallel,only: psnd8
#if KEY_STRINGM==1
  use machio, only: ifreeu
  use multicom_aux
  use stream, only: iolev
#endif
  implicit none
  !
  INTEGER IUNIT,NTITL,ICARD,JUNIT
  CHARACTER(len=*) TITLE(*)
  !
#if KEY_STRINGM==1 /*  VO stringm v */
  integer :: oldiol
  logical :: qstr
  common /replicaio/ qstr ! need global variable
  qstr=iunit.le.size(ifreeu) ! protect against OOB on ifreeu
  if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
  if (qstr) then
   oldiol=iolev
   iolev=1
  endif
#endif
  !-----------------------------------------------------------------------
  IF(ICARD.NE.0) THEN
     IF(NATC.LE.0) THEN
        CALL WRNDIE(-2,'<ACEIO>', &
             'Cannot read a card ACEparam file prior to rtf/param files')
        RETURN
     ENDIF
  ENDIF
  !
  CALL ACERDR(IUNIT,NTITL,TITLE,ICARD,JUNIT,NATC,ITC,ATC, &
       COMLYN,COMLEN,MXCMSZ, &
       EFVOL,CHRAD,ACEHYD,ACEA0,ACEA1,ACEA2)
  !
#if KEY_STRINGM==1 /*  VO : restore iolev */
  if (qstr) iolev=oldiol 
#endif
  !
#if KEY_PARALLEL==1
  CALL PSND8(EFVOL,MAXATC)
  CALL PSND8(CHRAD,MAXATC)
  CALL PSND8(ACEHYD,MAXATC)
  CALL PSND8(ACEA0,MAXATC)
  CALL PSND8(ACEA1,MAXATC)
  CALL PSND8(ACEA2,MAXATC)
#endif 
  RETURN
END SUBROUTINE ACEIO
!
!
!======================================================================
!
SUBROUTINE ACERDR(IUNIT,NTITL,TITLE,ICARD,JUNIT,NATC,ITC,ATC, &
     COMLYN,COMLEN,MXCMSZ, &
     EFVOL,CHRAD,ACEHYD,ACEA0,ACEA1,ACEA2)
  !
  !     This routine reads ACE parameters from a file or stndin (only
  !     possible if ascii file read, CARD=1).
  !     It can read card format input `free field' with a title.
  !
  !     ICARD controls the input mode:
  !      0=binary
  !      1=card/ascii
  !
  !     SYNTAX:  After the title, card file data is divided into sections
  !      beginning with a keyword line and followed by data lines read
  !      free field:
  !     VOLU
  !         effective atom volumes [A^3] (atom type array)
  !     SIGM
  !         hydrophobic parmeter [cal/mol/A^2] (atom type array)
  !     EQUI
  !         specification of new atom types (for volume assignment to
  !         atom types that do not exist in RTF)
  !         -- not yet implemented (Dec 2001)
  !     RMIN
  !         specific minimum charge radius [A] (atom type array)
  !         -- not yet implemented (Dec 2001)
  !     A012
  !         parameters of the ACE3 born solvation radius correction,
  !         a0, a1, a2 (atom type array)
  !         -- not yet implemented in the parameter files (Dec 2001)
  !
  use chm_kinds
  use number
  use exfunc
  use stream
  use string, only:trime,nexta4,nextf,eqwdwc
  !
  implicit none
  !
  INTEGER IUNIT,NTITL,ICARD,JUNIT,NATC,ITC(*)
  CHARACTER(len=*) TITLE(*),ATC(*),COMLYN
  INTEGER COMLEN,MXCMSZ
  real(chm_real)  EFVOL(*),CHRAD(*),ACEHYD(*), &
       ACEA0(*),ACEA1(*),ACEA2(*)
  !
  !     local:
  INTEGER I,J
  real(chm_real)    R1,R2,R3
  LOGICAL :: EOF,QPRINT=.FALSE. ,DATLIN
  CHARACTER(len=4) :: WRD,CURRNT,VOLU='VOLU',RMIN='RMIN',EQUI='EQUI',A012='A012',SIGM='SIGM'
  CHARACTER(len=4) AI
  !
  !     Keywords for command line reading:
  !-----------------------------------------------------------------------
  IF(IOLEV.GT.0) THEN
     IF (ICARD.EQ.0) THEN
        !         Begin Procedure READ-BINARY-FILE
        CALL WRNDIE(-2,'<ACERDR>','BINARY READ NOT IMPLEMENTED')
     ELSE
        !         read ACE parameters from cards
        CURRNT='JUNK'
        CALL TRYORO(IUNIT,'FORMATTED')
        CALL RDTITL(TITLE,NTITL,IUNIT,0)
        EOF=.FALSE.
        !         get input line:
100     CONTINUE
        CALL XTRANE(COMLYN,COMLEN,'ACERDR')
110     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.FALSE., &
             QPRINT,'ACERDR> ')
        CALL TRIME(COMLYN,COMLEN)
        IF ((.NOT.EOF).AND.(COMLEN.LE.0)) GOTO 110
        DATLIN=.FALSE.
        WRD=NEXTA4(COMLYN,COMLEN)
        IF (WRD.EQ.'VOLU') THEN
           CURRNT=VOLU
           IF(PRNLEV.GE.1) WRITE(JUNIT,210)
210        FORMAT(/9X,'EFFECTIVE VOLUME PARAMETERS',/, &
                8X,'ITC   TYPE   CODE       VOLUME')
        ELSEIF (WRD.EQ.'SIGM') THEN
           CURRNT=SIGM
           IF(PRNLEV.GE.1) WRITE(JUNIT,220)
220        FORMAT(/9X,'HYDROPHOBIC PARAMETERS     ',/, &
                8X,'ITC   TYPE   CODE        SIGMA')
        ELSEIF (WRD.EQ.'RMIN') THEN
           CURRNT=RMIN
           IF(PRNLEV.GE.1) WRITE(JUNIT,230)
230        FORMAT(/9X,'MINIMUM CHARGE RADII not yet implemented...',/, &
                10X,'#  ATOM    CODE         RMIN')
        ELSEIF (WRD.EQ.'EQUI') THEN
           CURRNT=EQUI
           IF(PRNLEV.GE.1) WRITE(JUNIT,240)
240        FORMAT(/9X,'SPECIAL ATOM TYPES not yet implemented...',/, &
                10X,'#  ATOM    CODE')
        ELSEIF (WRD.EQ.'A012') THEN
           CURRNT=A012
           IF(PRNLEV.GE.1) WRITE(JUNIT,250)
250        FORMAT(/9X,'SELF ENERGY CORRECTION (ACE2) PARAMETERS',/, &
                8X,'ITC   TYPE   CODE           A0           A1', &
                '           A2')
        ELSEIF (WRD.EQ.'END ') THEN
           EOF=.TRUE.
        ELSE
           DATLIN=.TRUE.
        ENDIF
        IF (.NOT.(DATLIN.OR.EOF)) GOTO 100
        IF(CURRNT.EQ.'JUNK') &
             CALL WRNDIE(-1,'<ACERDR>','HEADER CARD NOT FOUND')
        IF(.NOT.EOF) THEN
           IF (CURRNT.EQ.VOLU) THEN
              !             This section reads effective volumes:
              AI=WRD
              R1=NEXTF(COMLYN,COMLEN)
              I=0
              DO J=1,NATC
                 IF(EQWDWC(ATC(J),AI)) THEN
                    I=J
                    EFVOL(I)=R1
                    IF(PRNLEV.GE.1) THEN
                       WRITE(JUNIT,310) ITC(I),ATC(I),I,EFVOL(I)
310                    FORMAT(6X,I5,3X,A4,I7,F13.3)
                    ENDIF
                 ENDIF
              ENDDO
              IF(I.EQ.0) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,315) AI
315              FORMAT(' ACERDR> WARNING: ATOM TYPE FOR VOLUME ', &
                      A4,' DOESNT EXIST')
              ENDIF
           ELSEIF (CURRNT.EQ.SIGM) THEN
              !             This section reads hydrophobic parameters:
              AI=WRD
              R1=NEXTF(COMLYN,COMLEN)
              I=0
              DO J=1,NATC
                 IF(EQWDWC(ATC(J),AI)) THEN
                    I=J
                    ACEHYD(I)=0.001*R1
                    IF(PRNLEV.GE.1) THEN
                       WRITE(JUNIT,320) ITC(I),ATC(I),I,1000.0*ACEHYD(I)
320                    FORMAT(6X,I5,3X,A4,I7,F13.3)
                    ENDIF
                 ENDIF
              ENDDO
              IF(I.EQ.0) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,325) AI
325              FORMAT(' ACERDR> WARNING: ATOM TYPE FOR ACEHYD ', &
                      A4,' DOESNT EXIST')
              ENDIF
           ELSEIF (CURRNT.EQ.RMIN) THEN
              !             This section reads minimum charge radii:
              AI=WRD
              R1=NEXTF(COMLYN,COMLEN)
              I=0
              DO J=1,NATC
                 IF(EQWDWC(ATC(J),AI)) THEN
                    I=J
                    CHRAD(I)=R1
                    IF(PRNLEV.GE.1) THEN
                       WRITE(JUNIT,330) ITC(I),ATC(I),I,CHRAD(I)
330                    FORMAT(6X,I5,3X,A4,I7,F13.3)
                    ENDIF
                 ENDIF
              ENDDO
              IF(I.EQ.0) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,335) AI
335              FORMAT(' ACERDR> WARNING: ATOM TYPE FOR CHRAD ', &
                      A4,' DOESNT EXIST')
              ENDIF
           ELSEIF (CURRNT.EQ.EQUI) THEN
              GOTO 100
           ELSEIF (CURRNT.EQ.A012) THEN
              !             This section reads self energy (ACE2) parameters:
              AI=WRD
              R1=NEXTF(COMLYN,COMLEN)
              R2=NEXTF(COMLYN,COMLEN)
              R3=NEXTF(COMLYN,COMLEN)
              I=0
              DO J=1,NATC
                 IF(EQWDWC(ATC(J),AI)) THEN
                    I=J
                    ACEA0(I)=R1
                    ACEA1(I)=R2
                    ACEA2(I)=R3
                    IF(PRNLEV.GE.1) THEN
                       WRITE(JUNIT,350) ITC(I),ATC(I),I, &
                            ACEA0(I),ACEA1(I),ACEA2(I)
350                    FORMAT(6X,I5,3X,A4,I7,3E13.4)
                    ENDIF
                    !                 need (a0+1) in energy routines, modify here:
                    ACEA0(I)=ACEA0(I)+ONE
                 ENDIF
              ENDDO
              IF(I.EQ.0) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,355) AI
355              FORMAT(' ACERDR> WARNING: ATOM TYPE FOR A012 ', &
                      A4,' DOESNT EXIST')
              ENDIF
           ENDIF
        ENDIF
        IF (.NOT.EOF) GOTO 100
     ENDIF
  ENDIF ! (IOLEV.GT.0)
  RETURN
END SUBROUTINE ACERDR
#else /**/
SUBROUTINE NULL_ACEIO
  RETURN
END SUBROUTINE NULL_ACEIO
#endif /*  ACE*/

