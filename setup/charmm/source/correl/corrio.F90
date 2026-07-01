SUBROUTINE READTS(ISERIE,NSER,LNEW, &
     MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
     VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
     SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
  !
  !     This routine reads time series
  !
  !  ENTER     {   READ  unit-number [CARD] [edit-spec]          }
  !
  !  READ { time-series-name } unit-spec edit-spec  [ FILE              ]
  !       { CORRelation-funct}                      [ CARD              ]
  !                                                 [ DUMB [COLUmn int] ]
  !
  !      By  Bernard R. Brooks    18-Oct-1984
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use comand
  use ctitla
  use stream
  use string
  use parallel,only:psnd4,psnd8
  implicit none
  INTEGER ISERIE,NSER,MAXSER,MAXTOT
  character(len=*) ICLASS(*)
  INTEGER VELCOD(*),GECOD(*),VECCOD(*)
  real(chm_real) SERVAL(*)
  LOGICAL LNEW,LMASS(*)
  INTEGER STOT(*),SSKIP(*)
  real(chm_real) SDELTA(*),SOFFST(*)
  real(chm_real) SAVEG(*),SFLCT(*)
  INTEGER SERPT(*),SERNQ(*)
  character(len=*) SNAME(*)
  real(chm_real) TQ(MAXTOT,MAXSER)
  INTEGER QSIZE
  INTEGER QAT(*)
  !
  INTEGER ISTOP,IUNIT,VECCD,I,J,NTOT,ICOL,N
  LOGICAL EOF
  character(len=4) CH
  !
  ISTOP=ISERIE+NSER-1
  IF(ISTOP.GT.MAXSER) ISTOP=MAXSER
  IF(ISTOP.LT.ISERIE) ISTOP=ISERIE
  IF(LNEW) ISTOP=MAXSER
  !
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  IF(IUNIT.LT.0) CALL WRNDIE(0,'<READTS>','NO UNIT SPECIFIED')
  !
  IF(IOLEV.GT.0) THEN
     IF(INDXA(COMLYN,COMLEN,'CARD').GT.0) THEN
        ! process-card-file

        CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
        CALL WRTITL(TITLEB,NTITLB,OUTU,1)
        !
        READ(IUNIT,25) VECCD
25      FORMAT(6X,I4)
        ! check-istop-for-overflow

        IF(LNEW) ISTOP=ISERIE+VECCD-1
        IF(VECCD.LT.ISTOP-ISERIE+1) THEN
           CALL WRNDIE(1,'<READTS>', &
                'Fewer series in file than requested.')
           IF(PRNLEV.GE.2) WRITE(OUTU,33) VECCD,ISTOP-ISERIE+1
           ISTOP=ISERIE+VECCD-1
        ENDIF
        IF(VECCD.GT.ISTOP-ISERIE+1) THEN
           CALL WRNDIE(1,'<READTS>', &
                'More series in file than requested.')
           IF(PRNLEV.GE.2) WRITE(OUTU,34) VECCD,ISTOP-ISERIE+1
        ENDIF
        IF(ISTOP.GT.MAXSER) THEN
           CALL WRNDIE(-1,'<READTS>', &
                'Maximum number of time series exceeded')
           ISTOP=MAXSER
        ENDIF
        !
        READ(IUNIT,35) (SNAME(I)(1:idleng),I=ISERIE,ISTOP)
35      FORMAT(8X,10(4X,A,4X))
        READ(IUNIT,45) (STOT(I),I=ISERIE,ISTOP)
45      FORMAT(8X,10(I12))
        READ(IUNIT,45) (VECCOD(I),I=ISERIE,ISTOP)
        READ(IUNIT,35) (ICLASS(I),I=ISERIE,ISTOP)
        READ(IUNIT,45) (VELCOD(I),I=ISERIE,ISTOP)
        READ(IUNIT,45) (SSKIP(I),I=ISERIE,ISTOP)
        READ(IUNIT,85) (SDELTA(I),I=ISERIE,ISTOP)
        READ(IUNIT,85) (SOFFST(I),I=ISERIE,ISTOP)
85      FORMAT(8X,10(F12.6))
        READ(IUNIT,45) (GECOD(I),I=ISERIE,ISTOP)
        READ(IUNIT,85) (SERVAL(I),I=ISERIE,ISTOP)
        NTOT=0
        DO I=ISERIE,ISTOP
           IF(STOT(I).GT.NTOT) NTOT=STOT(I)
        ENDDO
        DO I=1,NTOT
           READ(IUNIT,115) (TQ(I,J),J=ISERIE,ISTOP)
115        FORMAT(8X,10(F12.6))
        ENDDO
        !
     ELSE IF(INDXA(COMLYN,COMLEN,'DUMB').GT.0) THEN
        ! process-dumb-file

        ICOL=GTRMI(COMLYN,COMLEN,'COLU',1)
        CALL XTRANE(COMLYN,COMLEN,'CORREL')
        EOF=.FALSE.
        DO N=1,MAXTOT
           CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.FALSE., &
                .FALSE.,'> ')
           IF(EOF) GOTO 500
           DO I=1,ICOL-1
              CH=NEXTA4(COMLYN,COMLEN)
           ENDDO
           DO I=ISERIE,ISTOP
              TQ(N,I)=NEXTF(COMLYN,COMLEN)
           ENDDO
        ENDDO
        N=MAXTOT+1
500     CONTINUE
        N=N-1
        DO I=ISERIE,ISTOP
           STOT(I)=N
        ENDDO
     ELSE
        ! process-binary-file
        !     IF(INDXA(COMLYN,COMLEN,'FILE').GT.0)
        CALL RDTITL(TITLEB,NTITLB,IUNIT,-1)
        CALL WRTITL(TITLEB,NTITLB,OUTU,1)
        READ(IUNIT) VECCD
        ! check-istop-for-overflow

        IF(LNEW) ISTOP=ISERIE+VECCD-1
        IF(VECCD.LT.ISTOP-ISERIE+1) THEN
           CALL WRNDIE(1,'<READTS>', &
                'Fewer series in file than requested.')
           IF(PRNLEV.GE.2) WRITE(OUTU,33) VECCD,ISTOP-ISERIE+1
           ISTOP=ISERIE+VECCD-1
        ENDIF
        IF(VECCD.GT.ISTOP-ISERIE+1) THEN
           CALL WRNDIE(1,'<READTS>', &
                'More series in file than requested.')
           IF(PRNLEV.GE.2) WRITE(OUTU,34) VECCD,ISTOP-ISERIE+1
        ENDIF
        IF(ISTOP.GT.MAXSER) THEN
           CALL WRNDIE(-1,'<READTS>', &
                'Maximum number of time series exceeded')
           ISTOP=MAXSER
        ENDIF
        !
        READ(IUNIT) (SNAME(I),I=ISERIE,ISTOP)
        READ(IUNIT) (STOT(I),I=ISERIE,ISTOP)
        READ(IUNIT) (VECCOD(I),I=ISERIE,ISTOP)
        READ(IUNIT) (ICLASS(I),I=ISERIE,ISTOP)
        READ(IUNIT) (VELCOD(I),I=ISERIE,ISTOP)
        READ(IUNIT) (SSKIP(I),I=ISERIE,ISTOP)
        READ(IUNIT) (SDELTA(I),I=ISERIE,ISTOP)
        READ(IUNIT) (SOFFST(I),I=ISERIE,ISTOP)
        READ(IUNIT) (GECOD(I),I=ISERIE,ISTOP)
        READ(IUNIT) (SERVAL(I),I=ISERIE,ISTOP)
        DO I=ISERIE,ISTOP
           READ(IUNIT) (TQ(J,I),J=1,STOT(I))
        ENDDO
     ENDIF
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(VECCD, 1)
  N=ISTOP-ISERIE+1
  CALL PSNDC(SNAME(ISERIE:istop),N)
  CALL PSND4(STOT(ISERIE:istop),N)
  CALL PSND4(VECCOD(ISERIE:istop),N)
  CALL PSNDC(ICLASS(ISERIE:istop),N)
  CALL PSND4(VELCOD(ISERIE:istop),N)
  CALL PSND4(SSKIP(ISERIE:istop),N)
  CALL PSND8(SDELTA(ISERIE:istop),N)
  CALL PSND8(SOFFST(ISERIE:istop),N)
  CALL PSND4(GECOD(ISERIE:istop),N)
  CALL PSND8(SERVAL(ISERIE:istop),N)
  DO I=ISERIE,ISTOP
     CALL PSND8(TQ(1:stot(i),I),STOT(I))
  ENDDO
#endif 
  !
  RETURN
  !
33 FORMAT(' Found=',I4,'  Expected=',I4,'  Excess vectors untouched')
34 FORMAT(' Found=',I4,'  Expected=',I4,'  Excess data ingnored.')
  !
END SUBROUTINE READTS

SUBROUTINE WRITTS(ISERIE,NSER, &
     MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
     VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
     SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
  !
  !
  !     This routine writes time series or time series plots.
  !
  !       { ALL                }             [ FILE          ]
  ! WRITe { time-series-name   } unit-spec   [ CARD          ]
  !       {CORRelation-function}             [ PLOT          ]
  !                                          [ DUMB [ TIME ] ]
  !
  !      By  Bernard R. Brooks    18-Oct-1984
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use comand
  use consta
  use ctitla
  use stream
  use string
  implicit none
  INTEGER ISERIE,NSER,MAXSER,MAXTOT
  character(len=*) ICLASS(*)
  INTEGER VELCOD(*),GECOD(*),VECCOD(*)
  real(chm_real) SERVAL(*)
  LOGICAL LMASS(*)
  INTEGER STOT(*),SSKIP(*)
  real(chm_real) SDELTA(*),SOFFST(*)
  real(chm_real) SAVEG(*),SFLCT(*)
  INTEGER SERPT(*),SERNQ(*)
  character(len=*) SNAME(*)
  real(chm_real) TQ(MAXTOT,MAXSER+1)
  INTEGER QSIZE
  INTEGER QAT(*)
  !
  INTEGER ISTOP,VECCD,IUNIT,I,J,NTOT
  real(chm_real) DEL
  LOGICAL LTIME, ERROR
  INTEGER MAXOFMT,OFMTLN
  PARAMETER (MAXOFMT=20)
  character(len=MAXOFMT) OFMT,OFMT1 
  !
  ISTOP=ISERIE+NSER-1
  IF(ISTOP.GT.MAXSER) ISTOP=MAXSER
  IF(ISTOP.LT.ISERIE) ISTOP=ISERIE
  VECCD=ISTOP-ISERIE+1
  DEL=SSKIP(ISERIE)*SDELTA(ISERIE)
  !
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
  IF(IUNIT.LT.0) THEN
     CALL WRNDIE(0,'<WRITTS>','NO UNIT SPECIFIED')
     RETURN
  ENDIF
  !
  CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
  IF(OUTU.EQ.IUNIT .AND. PRNLEV.LT.2) RETURN
  IF(OUTU.NE.IUNIT .AND.  IOLEV.LT.0) RETURN
  !
  IF(INDXA(COMLYN,COMLEN,'CARD').GT.0) THEN
     ! process-card-write
     !
     CALL WRTITL(TITLEA,NTITLA,IUNIT,0)
     WRITE(IUNIT,25) VECCD
25   FORMAT(' NSER:',I4)
     WRITE(IUNIT,35) (SNAME(I)(1:idleng),I=ISERIE,ISTOP)
35   FORMAT(' NAMES: ',10(4X,A,4X))
     WRITE(IUNIT,45) (STOT(I),I=ISERIE,ISTOP)
45   FORMAT(' TOTALS:',10(I12))
     WRITE(IUNIT,50) (VECCOD(I),I=ISERIE,ISTOP)
50   FORMAT(' VECCOD:',10(I12))
     WRITE(IUNIT,55) (ICLASS(I)(1:idleng),I=ISERIE,ISTOP)
55   FORMAT(' CLASS: ',10(4X,A,4X))
     WRITE(IUNIT,65) (VELCOD(I),I=ISERIE,ISTOP)
65   FORMAT(' VELCOD:',10(I12))
     WRITE(IUNIT,75) (SSKIP(I),I=ISERIE,ISTOP)
75   FORMAT(' SKIP:  ',10(I12))
     WRITE(IUNIT,85) (SDELTA(I),I=ISERIE,ISTOP)
85   FORMAT(' DELTA: ',10(F12.6))
     WRITE(IUNIT,90) (SOFFST(I),I=ISERIE,ISTOP)
90   FORMAT(' OFFST: ',10(F12.6))
     WRITE(IUNIT,95) (GECOD(I),I=ISERIE,ISTOP)
95   FORMAT(' GECOD: ',10(I12))
     WRITE(IUNIT,105) (SERVAL(I),I=ISERIE,ISTOP)
105  FORMAT(' VALUE: ',10(F12.6))
     NTOT=0
     DO I=ISERIE,ISTOP
        IF(STOT(I).GT.NTOT) NTOT=STOT(I)
     ENDDO
     DO I=1,NTOT
        WRITE(IUNIT,115) I,(TQ(I,J),J=ISERIE,ISTOP)
115     FORMAT(I5,3X,10(F12.6))
     ENDDO
     !
  ELSE IF(INDXA(COMLYN,COMLEN,'DUMB').GT.0) THEN
     ! process-dumb-write
     !
     LTIME=(INDXA(COMLYN,COMLEN,'TIME').GT.0)
     NTOT=0
     DO I=ISERIE,ISTOP
        IF(NTOT.LT.STOT(I)) NTOT=STOT(I)
     ENDDO
     IF(QLONGL)THEN
        CALL ENCODI(ISTOP-ISERIE+2,OFMT1,MAXOFMT,OFMTLN)
        OFMT='('//OFMT1(1:OFMTLN)//'F15.6)'
     ELSE
        OFMT='(8(F15.6))'
     ENDIF
     DO I=1,NTOT
        IF(LTIME) THEN
           WRITE(IUNIT,OFMT) (I-1)*DEL+SOFFST(ISERIE), &
                (TQ(I,J),J=ISERIE,ISTOP)
        ELSE
           WRITE(IUNIT,OFMT) (TQ(I,J),J=ISERIE,ISTOP)
        ENDIF
        ! 145        FORMAT(8(F15.6))
     ENDDO
     !
  ELSE IF(INDXA(COMLYN,COMLEN,'PLOT').GT.0) THEN
     ! process-plot-write
     !
     NTOT=STOT(ISERIE)
     WRITE(IUNIT) TITLEA(1)(1:80)
     WRITE(IUNIT) NTOT
     WRITE(IUNIT) ((I-1)*DEL+SOFFST(ISERIE),I=1,NTOT)
     WRITE(IUNIT) (TQ(I,ISERIE),I=1,NTOT)
     !
  ELSE
     ! process-binary-write
     !     IF(INDXA(COMLYN,COMLEN,'FILE').GT.0)
     CALL WRTITL(TITLEA,NTITLA,IUNIT,-1)
     WRITE(IUNIT) VECCD
     WRITE(IUNIT) (SNAME(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (STOT(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (VECCOD(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (ICLASS(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (VELCOD(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (SSKIP(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (SDELTA(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (SOFFST(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (GECOD(I),I=ISERIE,ISTOP)
     WRITE(IUNIT) (SERVAL(I),I=ISERIE,ISTOP)
     DO I=ISERIE,ISTOP
        WRITE(IUNIT) (TQ(J,I),J=1,STOT(I))
     ENDDO
     !
  ENDIF
  IF(IUNIT.NE.OUTU) CALL VCLOSE(IUNIT,'KEEP',ERROR)
  RETURN
  !
END SUBROUTINE WRITTS

SUBROUTINE SHOWTS(ISERIE,NSER, &
     MAXSER,MAXTOT,ICLASS,VELCOD,GECOD, &
     VECCOD,SERVAL,LMASS,STOT,SSKIP,SDELTA,SOFFST, &
     SAVEG,SFLCT,SERPT,SERNQ,SNAME,TQ,QSIZE,QAT)
  !
  !
  !     This routine shows data for specific time series.
  !
  !       { ALL                }
  !  SHOW { time-series-name   } unit-spec
  !       {CORRelation-function}
  !
  !      By  Bernard R. Brooks    18-Oct-1984
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use param_store, only: set_param
  !
  use comand
  use consta
  use ctitla
  use stream
  use string
  !
  implicit none
  INTEGER ISERIE,NSER,MAXSER,MAXTOT
  character(len=*) ICLASS(*)
  INTEGER VELCOD(*),GECOD(*),VECCOD(*)
  real(chm_real) SERVAL(*)
  LOGICAL LMASS(*)
  INTEGER STOT(*),SSKIP(*)
  real(chm_real) SDELTA(*),SOFFST(*)
  real(chm_real) SAVEG(*),SFLCT(*)
  INTEGER SERPT(*),SERNQ(*)
  character(len=*) SNAME(*)
  real(chm_real) TQ(MAXTOT,MAXSER+1)
  INTEGER QSIZE
  INTEGER QAT(*)
  !
  INTEGER ISTOP,VECCD,IUNIT,I,J,NTOT,K
  real(chm_real) U(3,3),UR(3,3),UT,UTR
  real(chm_real) AVE,FLUCT,RVEC(3),AVER
  !
  ISTOP=ISERIE+NSER-1
  IF(ISTOP.GT.MAXSER) ISTOP=MAXSER
  IF(ISTOP.LT.ISERIE) ISTOP=ISERIE
  VECCD=ISTOP-ISERIE+1
  !
  DO I=ISERIE,ISTOP
     AVE=0.0
     NTOT=STOT(I)
     DO J=1,NTOT
        AVE=AVE+TQ(J,I)
     ENDDO
     IF(NTOT.GT.0) AVE=AVE/NTOT
     SAVEG(I)=AVE
     FLUCT=0.0
     DO J=1,NTOT
        FLUCT=FLUCT+(TQ(J,I)-AVE)**2
     ENDDO
     SFLCT(I)=0.0
     IF(FLUCT.GT.0) SFLCT(I)=SQRT(FLUCT/NTOT)
  ENDDO
  !
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  IF(PRNLEV.LT.2 .AND. IUNIT.EQ.OUTU) RETURN
  !
  !-----------------------------------------------------------------------
  ! compute p2 averages for normalized vectors
  IF(ISTOP-ISERIE.EQ.2) THEN
     !
     IF(INDXA(COMLYN,COMLEN,'P2').GT.0) THEN
        NTOT=STOT(ISERIE)
        !
        IF(NTOT.EQ.0) THEN
           CALL WRNDIE(-1,'<SHOWTS>', &
                'Zero time steps found for P2 average')
           RETURN
        ENDIF
        DO I=1,3
           DO J=1,3
              U(I,J)=ZERO
              UR(I,J)=ZERO
           ENDDO
        ENDDO
        !
        AVE=ZERO
        DO K=1,NTOT
           DO I=1,3
              RVEC(I)=TQ(K,ISERIE+I-1)
           ENDDO
           UT=ZERO
           DO I=1,3
              UT=UT+RVEC(I)**2
           ENDDO
           IF(UT.LT.RSMALL) THEN
              CALL WRNDIE(-1,'<SHOWTS>', &
                   'Zero vector found for P2 average')
              RETURN
           ENDIF
           UT=ONE/SQRT(UT)
           DO I=1,3
              RVEC(I)=RVEC(I)*UT
           ENDDO
           UT=UT**3
           AVE=AVE+UT
           DO I=1,3
              DO J=1,3
                 U(I,J)=U(I,J)+RVEC(I)*RVEC(J)
                 UR(I,J)=UR(I,J)+RVEC(I)*RVEC(J)*UT
              ENDDO
           ENDDO
        ENDDO
        !
        AVE=AVE/NTOT
        DO I=1,3
           DO J=1,3
              U(I,J)=U(I,J)/NTOT
              UR(I,J)=UR(I,J)/NTOT
           ENDDO
        ENDDO
        !
        UT=0.0
        UTR=0.0
        DO I=1,3
           DO K=1,3
              UT=UT+U(I,K)*U(K,I)
              UTR=UTR+UR(I,K)*UR(K,I)
           ENDDO
        ENDDO
        UT=1.5*UT-0.5
        AVER=AVE**2
        UTR=1.5*UTR-0.5*AVER
        AVE=AVE**(-THIRD)
        !
        IF(PRNLEV.GE.2) WRITE(IUNIT,15) UT,UTR,AVER,UT*AVER/UTR,AVE
15      FORMAT(' P2   AVERAGE (square of order parameter) =',F15.8,/ &
             ' P2R3 AVERAGE (NMR cross-relaxation coef) =',F15.8,/ &
             ' R3S  AVERAGE (    <Rij**-3>**2         ) =',F15.8,/ &
             ' P2RA  RATIO  (    P2 * R3S / P2R3      ) =',F15.8,/ &
             ' R3R  AVERAGE (    <Rij**-3>**(-1/3)    ) =',F15.8,/ &
             )
        !
        call set_param('P2',UT)
        call set_param('P2R3',UTR)
        call set_param('R3S',AVER)
        call set_param('P2RA',UT*AVER/UTR)
        call set_param('R3R',AVE)
        !
     ENDIF
  ENDIF
  !-----------------------------------------------------------------------
  !
  call set_param('AVER',SAVEG(ISERIE))
  call set_param('FLUC',SFLCT(ISERIE))
  !
  IF(PRNLEV.GE.2) THEN
     WRITE(IUNIT,25) VECCD
25   FORMAT(' NSER:',I4)
     WRITE(IUNIT,35) (SNAME(I)(1:idleng),I=ISERIE,ISTOP)
35   FORMAT(' NAMES:  ',10(4X,A,4X))
     WRITE(IUNIT,45) (STOT(I),I=ISERIE,ISTOP)
45   FORMAT(' TOTALS: ',10(I12))
     WRITE(IUNIT,47) (SAVEG(I),I=ISERIE,ISTOP)
47   FORMAT(' AVERAGE:',10(F12.6))
     WRITE(IUNIT,48) (SFLCT(I),I=ISERIE,ISTOP)
48   FORMAT(' FLUCT:  ',10(F12.6))
     WRITE(IUNIT,50) (VECCOD(I),I=ISERIE,ISTOP)
50   FORMAT(' VECCOD: ',10(I12))
     WRITE(IUNIT,55) (ICLASS(I)(1:idleng),I=ISERIE,ISTOP)
55   FORMAT(' CLASS:  ',10(4X,A,4X))
     WRITE(IUNIT,65) (VELCOD(I),I=ISERIE,ISTOP)
65   FORMAT(' VELCOD: ',10(I12))
     WRITE(IUNIT,75) (SSKIP(I),I=ISERIE,ISTOP)
75   FORMAT(' SKIP:   ',10(I12))
     WRITE(IUNIT,85) (SDELTA(I),I=ISERIE,ISTOP)
85   FORMAT(' DELTA:  ',10(F12.6))
     WRITE(IUNIT,90) (SOFFST(I),I=ISERIE,ISTOP)
90   FORMAT(' OFFST:  ',10(F12.6))
     WRITE(IUNIT,95) (GECOD(I),I=ISERIE,ISTOP)
95   FORMAT(' GECOD:  ',10(I12))
     WRITE(IUNIT,105) (SERVAL(I),I=ISERIE,ISTOP)
105  FORMAT(' VALUE:  ',10(F12.6))
  ENDIF
  !
  return
end SUBROUTINE SHOWTS

