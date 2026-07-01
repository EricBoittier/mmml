SUBROUTINE MANTIM(MAXTOT,MANCOD,NUTIL,FACTOR,FACTR2,ISER,JSER, &
     SNAME,STOT,SSKIP,SDELTA,SOFFST,VECCOD,TQ,TQ2,FAH)
  !
  !     THIS ROUTINE MANIPULATES THE TIME SERIES Q(T) (TQ ARRAY)
  !    MANCOD - DAVE      Q(T) = Q(T) - <Q>
  !           - NORM      Q(T) = Q(T)/<Q(T)>
  !           - SQUA      Q(T) = Q(T)**2
  !           - COS       Q(T) = COS(Q(T))
  !           - ACOS      Q(T) = ACOS(Q(T))
  !           - COS2      Q(T) = 3*COS(Q(T))**2 - 1
  !           - AVER      Q(T) = < Q(T) >(T=T-NUTIL+1,T)
  !                    i.e......( PARTIAL AVERAGE OF TIME SERIES )
  !           - SQRT      Q(T) = SQRT(T)
  !           - FLUC      Q(T)    UNCHANGED
  !                           THE ZERO TIME FLUCTUATIONS ARE
  !                            COMPUTED AND PRINTED OUT.
  !                            THE FOLLOWING VARIABLES ARE COMPUTED
  !                           A = <QA(0).QB(0)>
  !                           B = SQRT(<QA(0)**2>)
  !                           C = SQRT(<QB(0)**2>)
  !                           D = A/(B*C)
  !           - DINI      Q(T) = Q(T) - Q(1)
  !           - DELN      Q(T) = Q(T) - <Q(T)>(T=T-NUTIL+1,T)
  !           - OSC       Q(T) UNCHANGED
  !                           THE AVERAGE NUMBER OF OSCILLATIONS PER
  !                           UNIT STEP TIME IS COMPUTED AND PRINTED
  !                           OUT.
  !***        - COPY      Q(T) = Q2(T1); T1=FIRST,...,LAST (defaults to whole series) 
  !           - ADD       Q(T) = Q(T) + Q2(T)
  !                           Q2(T) IS READ ON FILE NUTIL
  !           - KMUL      Q(T) = Q(T) * Q2(T)
  !           - RATI      Q(T) = Q(T) / Q2(T)
  !           - CONT      Q(T) = Q(T)+360.0*N(T), N(T) AN INTEGER
  !                           CHOSEN TO MAKE THE SERIES CONTINUOUS.
  !           - MAP real1 real2  Q(T) is mapped to [real1,real2] (default[0,360])
  !           - PROB      Q(T) = PROB(Q(T))
  !                           THE INTEGER VALUE READ REPRESENTS THE
  !                           NUMBER OF EQUAL SUBDIVISIONS TAKEN FROM
  !                           MINIMUM TO THE MAXIMUM
  !           - HIST      Q(T) = PROB(Q(T))
  !                           THE INTEGER VALUE READ REPRESENTS THE
  !                           NUMBER OF EQUAL SUBDIVISIONS TAKEN FROM
  !                           THE SPECIFIED MINIMUM TO THE MAXIMUM
  !           - STATe     Q(T) = 1.0 , minimum < Q(T) < maximum
  !                              0.0 ,  Q(T) < mimimum
  !                              0.0 ,  Q(T) > maximum
  !sb+ln added following two options operation on two vector series
  !           - CROS      Q(T) = Q(T) x Q2(T)  (3D vector cross product)
  !           - DOTP      Q(T) =  x-comp=Q(T) . Q2(T); y-comp=angle between
  !                               Q(T) and Q2(T) in degrees; z-comp untouched
  !           - LOG       Q(T) = LOG(Q(T))
  !           - EXP       Q(T) = EXP(Q(T))
  !           - IPOWer    Q(T) = Q(T) ** integer
  !           - MULT      Q(T) = real * Q(T)
  !           - DMIN      Q(T) = Q(T) - QMIN
  !           - ABS       Q(T) = ABS(Q(T))
  !           - SHIF      Q(T) = Q(T) + real
  !           - ZERO      Q(T) = 0.0
  !           - DIVI      Q(T) = Q(T) / real
  !           - DIVF      Q(T) = Q(T) / Q(1)
  !           - DIVM      Q(T) = Q(T) / ABS(Q(MAX))
  !           - INTE      Q(T) = Integral(0 to T) (Q(T)dT)
  !           - MOVI      Q(T) = MovingAverage( Q(T2) )(T2=T-NUTIL+1,T)
  !           - SPHE      Q(T) = vector series convrted to spherical:r,phi,theta
  !           - TEST      Q(T) = COS(2*PI*T*real/TTOT)
  !
  !     Author: S. Swaminathan
  !     Rewritten: Bernard R. Brooks  9/84
  !
  use chm_kinds
  use exfunc
  use number
  use dimens_fcm
  !
  use stream
  use string, only:indxa,gtrma,gtrmi,nexti
  use comand
  use consta
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: RDUM
  integer,allocatable,dimension(:) :: IDTP
  real(chm_real),allocatable,dimension(:) :: RESULT
  real(chm_real),allocatable,dimension(:) :: SCRTCH
  integer,allocatable,dimension(:) :: INDPT
  real(chm_real),allocatable,dimension(:) :: WDUM
  real(chm_real),allocatable,dimension(:) :: NINSUB

  INTEGER MAXTOT,NUTIL
  character(len=4) MANCOD
  real(chm_real) FACTOR,FACTR2
  INTEGER ISER,JSER
  character(len=*) SNAME(*)
  INTEGER STOT(*),SSKIP(*)
  real(chm_real) SDELTA(*),SOFFST(*)
  INTEGER VECCOD(*)
  real(chm_real) TQ(MAXTOT,*),TQ2(MAXTOT,*)
  real(chm_real) FAH(*)
  !
  real(chm_real) CONV,QAVE,QMAX,QDIM,SAB,SA,SB,FA,FB,SABNRM
  real(chm_real) QTEMP,DENOM,NBOSC,STEPT,CON,FREQ,MINTQ,MAXTQ
  INTEGER NTOT,NSKIP,VECCD
  real(chm_real) DELTA,D,A,B
  INTEGER I,K,J,NEWT,L,JSTRT,JFIN,IS
  LOGICAL LWARN
  !
  INTEGER IORDER
  INTEGER IFIRST,ILAST,IF1,NTSTOT,NIN,NOUT
  LOGICAL QREPL,QWEIG
  character(len=8) WNAME
  !
  CONV=180.0/PI
  NTOT=STOT(ISER)
  NSKIP=SSKIP(ISER)
  VECCD=VECCOD(ISER)
  DELTA=SDELTA(ISER)
  !
  ! check-second-time-series
  IF(MANCOD.EQ.'FLUC'.OR.MANCOD.EQ.'COPY'.OR.MANCOD.EQ.'ADD'.OR. &
       MANCOD.EQ.'RATI'.OR.MANCOD.EQ.'KMUL'.OR. &
#if KEY_ADUMB==1
       MANCOD.EQ.'WHIS'.OR. &      
#endif
       MANCOD.EQ.'CROS'.OR.MANCOD.EQ.'DOTP') THEN
     LWARN=.FALSE.
     IF(JSER.LE.0) THEN
        CALL WRNDIE(0,'<MANTIM>','Second time series required.')
        RETURN
     ENDIF
     !
     IF(VECCOD(ISER).NE.VECCOD(JSER)) THEN
        CALL WRNDIE(-1,'<MANTIM>','Vector codes dont match')
        IF(WRNLEV.GE.2) WRITE(OUTU,902) &
             SNAME(ISER)(1:idleng),VECCOD(ISER), &
             SNAME(JSER)(1:idleng),VECCOD(JSER)
        LWARN=.TRUE.
     ENDIF
     !
     IF(MANCOD.EQ.'CROS' .AND. (VECCOD(ISER).NE.3))THEN
        CALL WRNDIE(-1,'<MANTIM>','CROSsproduct requires 3D-vect')
        IF(WRNLEV.GE.2) WRITE(OUTU,902) &
             SNAME(ISER)(1:idleng),VECCOD(ISER), &
             SNAME(JSER)(1:idleng),VECCOD(JSER)
        LWARN=.TRUE.
     ENDIF
     !
     !        dont make further tests for the copy option.
     IF(MANCOD.NE.'COPY') THEN
        IF(STOT(ISER).NE.STOT(JSER)) THEN
           CALL WRNDIE(1,'<MANTIM>', &
                'Lengths of time series dont match')
           IF(WRNLEV.GE.2) WRITE(OUTU,905) &
                SNAME(ISER)(1:idleng),STOT(ISER), &
                SNAME(JSER)(1:idleng),STOT(JSER)
           LWARN=.TRUE.
        ENDIF
        QTEMP=SSKIP(ISER)*SDELTA(ISER)
        QTEMP=QTEMP/(SSKIP(JSER)*SDELTA(JSER))-1.0
        IF(ABS(QTEMP).GT.0.001) THEN
           CALL WRNDIE(1,'<MANTIM>','Time intervals do not match')
           IF(WRNLEV.GE.2) THEN
              WRITE(OUTU,915) SNAME(ISER)(1:idleng), &
                   SSKIP(ISER),SDELTA(ISER)
              WRITE(OUTU,915) SNAME(JSER)(1:idleng), &
                   SSKIP(JSER),SDELTA(JSER)
           ENDIF
           LWARN=.TRUE.
        ENDIF
     ENDIF
     IF(LWARN.AND.PRNLEV.GE.2) WRITE(OUTU,925)
  ENDIF
  !
  !=======================================================================
  ! to subtract-average-from-time-series
  IF(MANCOD.EQ.'DAVE') THEN
     DO I=1,VECCD
        QAVE=0.0
        DO K=1,NTOT
           QAVE=QAVE+TQ(K,I)
        ENDDO
        QAVE=QAVE/NTOT
        DO K=1,NTOT
           TQ(K,I)=TQ(K,I)-QAVE
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to divide-time-series-by-norm-at-each-step
  ELSE IF(MANCOD.EQ.'NORM') THEN
     DO I=1,NTOT
        QAVE=0.0
        DO J=1,VECCD
           QAVE=QAVE+TQ(I,J)**2
        ENDDO
        IF(QAVE.GT.0.0) THEN
           QAVE=SQRT(QAVE)
           DO J=1,VECCD
              TQ(I,J)=TQ(I,J)/QAVE
           ENDDO
        ENDIF
     ENDDO
     !
     !=======================================================================
     ! to square-the-time-series
  ELSE IF(MANCOD.EQ.'SQUA') THEN
     IF(VECCD.EQ.1) THEN
        DO I=1,NTOT
           TQ(I,1)=TQ(I,1)**2
        ENDDO
     ELSE
        DO I=1,NTOT
           QTEMP=0.0
           DO K=1,VECCD
              QTEMP=QTEMP+TQ(I,K)**2
              TQ(I,K)=0.0
           ENDDO
           TQ(I,1)=QTEMP
        ENDDO
     ENDIF
     !
     !=======================================================================
     ! to cosine-the-time-series
  ELSE IF(MANCOD.EQ.'COS ') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=COS(TQ(I,K)/CONV)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to arc-cosine-the-time-series
     !  Answer is in degrees.
  ELSE IF(MANCOD.EQ.'ACOS') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           QTEMP=TQ(I,K)
           IF(QTEMP.GT.ONE) QTEMP=ONE
           IF(QTEMP.LT.MINONE) QTEMP=MINONE
           TQ(I,K)=CONV*ACOS(QTEMP)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to cosine-square-minus-one
  ELSE IF(MANCOD.EQ.'COS2') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=3.0*COS(TQ(I,K)/CONV)**2-1.0
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to compress-time-series
  ELSE IF(MANCOD.EQ.'AVER') THEN
     NEWT=NTOT/NUTIL
     IF((NEWT*NUTIL).LT.NTOT) NEWT=NEWT+1
     DO L=1,VECCD
        DO I=1,NEWT
           JSTRT=(I-1)*NUTIL+1
           JFIN=I*NUTIL
           IF(JFIN.GT.NTOT) JFIN=NTOT
           QDIM=0.0
           DO J=JSTRT,JFIN
              QDIM=QDIM+TQ(J,L)
           ENDDO
           TQ(I,L)=QDIM/(JFIN-JSTRT+1)
        ENDDO
        STOT(L+ISER-1)=NEWT
        SSKIP(L+ISER-1)=NSKIP*NUTIL
     ENDDO
     !
     !=======================================================================
     ! to square-root-time-series
  ELSE IF(MANCOD.EQ.'SQRT') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           IF(TQ(I,K).GT.ZERO) THEN
              TQ(I,K)=SQRT(TQ(I,K))
           ELSE
              TQ(I,K)=ZERO
           ENDIF
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to print-statistics-from-time-series
  ELSE IF(MANCOD.EQ.'FLUC') THEN
     !        check-second-time-series
     SAB=0.0
     SA=0.0
     SB=0.0
     DO K=1,VECCD
        DO J=1,NTOT
           FA=TQ(J,K)
           FB=TQ2(J,K)
           SA=SA+FA*FA
           SB=SB+FB*FB
           SAB=SAB+FA*FB
        ENDDO
     ENDDO
     !
     IF(NTOT.LE.0) THEN
        CALL WRNDIE(0,'<MANTIM>','NO POINTS TO AVERAGE')
     ELSE
        SAB=SAB/NTOT
        SA=SQRT(SA/NTOT)
        SB=SQRT(SB/NTOT)
        SABNRM=0.0
        IF(SAB.NE.0.0) SABNRM=SAB/(SA*SB)
        IF(PRNLEV.GE.2) WRITE(OUTU,881)
881     FORMAT(/,'****** ZERO TIME FLUCTUATIONS ******',/)
        IF(PRNLEV.GE.2) WRITE(OUTU,880)SAB,SA,SB,SABNRM
880     FORMAT(2X, &
             'AB=<QA(0)*QB(0)> = ',1PG14.6,10X, &
             'A =SQRT(<QA(0)**2>) = ',1PG14.6,/1X, &
             'B =SQRT(<QB(0)**2>) = ',1PG14.6,10X, &
             'ABN = AB/(A*B) = ',1PG14.6)
     ENDIF
     !
     !=======================================================================
     ! to fit-time-series-to-polynomial
  ELSE IF(MANCOD.EQ.'POLY') THEN
     IF(VECCD.GT.1) THEN
        CALL WRNDIE(0,'<MANTIM>', &
             'Can only analyze simple time series.')
        RETURN
     ENDIF
     !
     QREPL=(INDXA(COMLYN,COMLEN,'REPL').GT.0)
     WNAME=GTRMA(COMLYN,COMLEN,'WEIG')
     IORDER=NEXTI(COMLYN,COMLEN)+1
     !
     IF(WNAME.EQ.' ') THEN
        I=0
     ELSE IF(WNAME(1:4).EQ.'CORR') THEN
        I=JSER
     ELSE
        I=0
        DO J=1,JSER
           IF(WNAME.EQ.SNAME(J)) I=J
        ENDDO
        !
        IF(PRNLEV.GE.2) WRITE(OUTU,'(3A)') '"',WNAME(1:IDLENG), &
             '" doesnt exist. Weighting ignored.'
     ENDIF
     QWEIG=(I.GT.0)
     !
     call chmalloc('mantim.src','MANTIM','RDUM',NTOT,crl=RDUM)
     call chmalloc('mantim.src','MANTIM','IDTP',NTOT,intg=IDTP)
     call chmalloc('mantim.src','MANTIM','RESULT',IORDER,crl=RESULT)
     call chmalloc('mantim.src','MANTIM','SCRTCH',IORDER*(IORDER+1),crl=SCRTCH)
     call chmalloc('mantim.src','MANTIM','INDPT',2*IORDER,intg=INDPT)
     !
     IF(QWEIG) THEN
        CALL POLYFIT(NTOT,RDUM,DELTA,IDTP,TQ, &
             QWEIG,TQ2(1,I),RESULT,SCRTCH, &
             INDPT,IORDER,QREPL)
     ELSE
        call chmalloc('mantim.src','MANTIM','WDUM',NTOT,crl=WDUM)
        CALL POLYFIT(NTOT,RDUM,DELTA,IDTP,TQ, &
             QWEIG,WDUM,RESULT,SCRTCH, &
             INDPT,IORDER,QREPL)
        call chmdealloc('mantim.src','MANTIM','WDUM',NTOT,crl=WDUM)
     ENDIF

     call chmdealloc('mantim.src','MANTIM','RDUM',NTOT,crl=RDUM)
     call chmdealloc('mantim.src','MANTIM','IDTP',NTOT,intg=IDTP)
     call chmdealloc('mantim.src','MANTIM','RESULT',IORDER,crl=RESULT)
     call chmdealloc('mantim.src','MANTIM','SCRTCH',IORDER*(IORDER+1),crl=SCRTCH)
     call chmdealloc('mantim.src','MANTIM','INDPT',2*IORDER,intg=INDPT)

     !
     !=======================================================================
     ! to subtract-first-element-from-series
  ELSE IF(MANCOD.EQ.'DINI') THEN
     DO K=1,VECCD
        QAVE=TQ(1,K)
        DO J=1,NTOT
           TQ(J,K)=TQ(J,K)-QAVE
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to subtract-time-weighted-average
  ELSE IF(MANCOD.EQ.'DELN') THEN
     DO K=1,VECCD
        QDIM=0.0
        IS=NTOT-NUTIL+2
        IF(IS.LT.1) IS=1
        DO J=IS,NTOT
           QDIM=QDIM+TQ(J,K)
        ENDDO
        DO J=NTOT,1,-1
           QTEMP=TQ(J,K)
           IF(J.GE.NUTIL) THEN
              DENOM=NUTIL
              QDIM=QDIM+TQ(J-NUTIL+1,K)
           ELSE
              DENOM=J
           ENDIF
           TQ(J,K)=TQ(J,K)-QDIM/DENOM
           QDIM=QDIM-QTEMP
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to print-number-of-oscillations
  ELSE IF(MANCOD.EQ.'OSC ') THEN
     DO J=1,VECCD
        CALL OSC(TQ(1,J),NTOT,NBOSC)
        STEPT=NSKIP*DELTA
        CON=STEPT*SPEEDL
        FREQ=NBOSC/CON
        IF(PRNLEV.GE.2) WRITE(OUTU,1105)STEPT,J,NBOSC,FREQ
1105    FORMAT(/, &
             ' Number of oscillations per calculation step time (', &
             E11.5,' psec)',/ &
             '  Vector =',I4,'  Number of oscillations =',E11.5, &
             '  Frequency in cm-1 is:',F8.2)
     ENDDO
     !
     !=======================================================================
     ! to copy-time-series
  ELSE IF(MANCOD.EQ.'COPY') THEN
     !        check-second-time-series
     IFIRST=GTRMI(COMLYN,COMLEN,'FIRS',1)
     ILAST=GTRMI(COMLYN,COMLEN,'LAST',STOT(JSER))
     IF(IFIRST.LE.0) IFIRST=1
     IF(ILAST.GT. STOT(JSER) .OR. ILAST.LE.0) ILAST=STOT(JSER)
     IF(ILAST.LE.IFIRST) THEN
        CALL WRNDIE(-1,'<MANTIM>','FIRST>LAST??') 
     ENDIF
     NTSTOT=0
     IF1=0
     IF(ILAST.NE.STOT(JSER) .OR. IFIRST.NE.1)THEN
        NTSTOT=ILAST-IFIRST+1
        IF1=IFIRST-1
        IF(PRNLEV .GE.2) WRITE(OUTU,'(/A,I12,A,I12,A/)')  &
             'Copying subset from',IFIRST, &
             ' to',ILAST,' (incl) of second series'
     ENDIF
     DO K=1,VECCD
        NTOT=STOT(JSER+K-1)
        IF(NTSTOT.NE.0)NTOT=NTSTOT
        DO I=1,NTOT
           TQ(I,K)=TQ2(I+IF1,K)
        ENDDO
        STOT(ISER+K-1)=NTOT
        SDELTA(ISER+K-1)=SDELTA(JSER+K-1)
        SOFFST(ISER+K-1)=SOFFST(JSER+K-1)
        SSKIP(ISER+K-1)=SSKIP(JSER+K-1)
     ENDDO
     !
     !======================================================================
     !sb+ln added option:
     ! to dotproduct-two-time-series
  ELSE IF(MANCOD.EQ.'DOTP') THEN
     DO I=1,NTOT
        D=TQ(I,1)*TQ2(I,1) + &
             TQ(I,2)*TQ2(I,2) + &
             TQ(I,3)*TQ2(I,3)
        ! rudimentary error trapping
        A=MAX(SQRT(TQ(I,1)**2+TQ(I,2)**2+TQ(I,3)**2),RSMALL)
        B=MAX(SQRT(TQ2(I,1)**2+TQ2(I,2)**2+TQ2(I,3)**2),RSMALL)
        TQ(I,1)=D
        TQ2(I,1)=RADDEG*ACOS(D/(A*B))
     ENDDO
     !
     !======================================================================
     !sb+ln added option:
     ! to CROSSproduct-two-time-series
  ELSE IF(MANCOD.EQ.'CROS') THEN
     DO I=1,NTOT
        A=TQ(I,2)*TQ2(I,3)-TQ(I,3)*TQ2(I,2)
        B=TQ(I,3)*TQ2(I,1)-TQ(I,1)*TQ2(I,3)
        TQ(I,3)=TQ(I,1)*TQ2(I,2)-TQ(I,2)*TQ2(I,1)
        TQ(I,1)=A
        TQ(I,2)=B
     ENDDO
     !
     !=======================================================================
     ! to add-time-series
  ELSE IF(MANCOD.EQ.'ADD ') THEN
     !        check-second-time-series
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)+TQ2(I,K)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to mulitply time series, element by element (Kronekar) <*> RMV
  ELSE IF(MANCOD.EQ.'KMUL') THEN
     !        check-second-time-series
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)*TQ2(I,K)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to ratio-time-series <*> RMV
  ELSE IF(MANCOD.EQ.'RATI') THEN
     !        check-second-time-series
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)/TQ2(I,K)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to continuous-dihedral-time-series <*> BRB
  ELSE IF(MANCOD.EQ.'CONT') THEN
     QTEMP=180.0
     IF(FACTOR.NE.0.0) QTEMP=FACTOR
     DO K=1,VECCD
        SA=0.0
        QAVE=0.0
        DO I=1,NTOT
           SB=TQ(I,K)-SA
440        CONTINUE
           IF(SB.GT. QTEMP) THEN
              SB=SB-QTEMP-QTEMP
              GOTO 440
           ENDIF
           IF(SB.LT.-QTEMP) THEN
              SB=SB+QTEMP+QTEMP
              GOTO 440
           ENDIF
           SA=SA+SB
           QAVE=QAVE+SA
           TQ(I,K)=SA
        ENDDO
        !           allow for one level adjustment when average is out of range.
        QAVE=QAVE/NTOT
        SA=0.0
        IF(QAVE.GT.180.0) SA=360.0
        IF(QAVE.LT.-180.0)SA=-360.0
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)-SA
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to map (angle) timeseries to [FACTOR,FACTR2]; L Nilsson
  ELSE IF(MANCOD.EQ.'MAP ') THEN
     QTEMP=FACTR2-FACTOR
     DO J=1,VECCD
        DO I=1,NTOT
           DO WHILE(TQ(I,J) < FACTOR )
              TQ(I,J) = TQ(I,J) + QTEMP
           ENDDO
           DO WHILE(TQ(I,J) > FACTR2 )
              TQ(I,J) = TQ(I,J) - QTEMP
           ENDDO
        ENDDO
     ENDDO 
     !
     !=======================================================================
     ! to automatically  map (angle) timeseries to cont. range; L Nilsson
  ELSE IF(MANCOD.EQ.'AMAP') THEN
     QTEMP=FACTR2-FACTOR
     DO J=1,VECCD
     ! find if series values are almost all inside or outside range [FACTOR, FACTR2]
        NIN=0
        NOUT=0
        DO I=1,NTOT
          IF(TQ(I,J) > FACTOR .AND. TQ(I,J) <  FACTR2) THEN
            NIN=NIN+1
          ELSE
            NOUT=NOUT+1
          ENDIF
        ENDDO
        ! if either inside or outside is > 95% (arbitrary, user could choose)
        QTEMP=QTEMP
        IF(NIN > 0.95 * NTOT) THEN
           FACTOR = QTEMP/2.0  - 180.0
           FACTR2 = QTEMP/2.0  + 180.0
        ELSE IF (NOUT > 0.95 * NTOT ) THEN
           FACTOR = QTEMP/2.0
           FACTR2 = QTEMP/2.0 + 360
        ELSE
           CALL WRNDIE(-1,'<MANTIM>','<95% of angles in either range. Auto map NOT done.')
           NIN=0
           NOUT=0
        ENDIF
        IF(PRNLEV .GE. 2)THEN
           WRITE(OUTU,'(A,I2,A,3I8)') & 
                'INITIAL ANGLE DISTR. FOR VECCOD ',J,'; NIN,NOUT,NTOT:',NIN,NOUT,NTOT
        ENDIF
        IF(NIN /= 0 .OR. NOUT /= 0) THEN
          IF(PRNLEV .GE.2 )THEN
            ! recompute numbers in or out with the interval to be used
            NIN=0
            NOUT=0
            DO I=1,NTOT
              IF(TQ(I,J) > FACTOR .AND. TQ(I,J) <  FACTR2) THEN
                NIN=NIN+1
              ELSE
                NOUT=NOUT+1
              ENDIF
            ENDDO
            WRITE(OUTU,'(A,2F6.1)') 'AUTMOAP TO RANGE: ',FACTOR, FACTR2
            WRITE(OUTU,'(A,I2,A,3I8)') &
                 'ANGLE DISTR. FOR VECCOD ',J,'; NIN,NOUT,NTOT:',NIN,NOUT,NTOT
          ENDIF
          DO I=1,NTOT
             DO WHILE(TQ(I,J) < FACTOR )
                TQ(I,J) = TQ(I,J) + QTEMP
             ENDDO
             DO WHILE(TQ(I,J) > FACTR2 )
                TQ(I,J) = TQ(I,J) - QTEMP
             ENDDO
          ENDDO
        ENDIF
     ENDDO 
     !
     !=======================================================================
     ! to fill-time-series-with-probabilities
  ELSE IF(MANCOD.EQ.'PROB') THEN
     call chmalloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
     DO J=1,VECCD
        MINTQ=TQ(1,J)
        MAXTQ=TQ(1,J)
        DO I=1,NTOT
           IF (TQ(I,J).LT.MINTQ) MINTQ=TQ(I,J)
           IF (TQ(I,J).GT.MAXTQ) MAXTQ=TQ(I,J)
        ENDDO
        CALL SUBDIV(TQ(1,J),NTOT,MINTQ,MAXTQ,NUTIL,NINSUB)
        STOT(ISER+J-1)=NUTIL
        SDELTA(ISER+J-1)=(MAXTQ-MINTQ)/NUTIL
        SOFFST(ISER+J-1)=MINTQ+SDELTA(ISER+J-1)*HALF
        SSKIP(ISER+J-1)=1
     ENDDO
     call chmdealloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
     !
     !=======================================================================
     ! to generate-a-histogram
  ELSE IF(MANCOD.EQ.'HIST') THEN
     call chmalloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
     MINTQ=FACTOR
     MAXTQ=FACTR2
     DO J=1,VECCD
        CALL SUBDIV(TQ(1,J),NTOT,MINTQ,MAXTQ,NUTIL,NINSUB)
        STOT(ISER+J-1)=NUTIL
        SDELTA(ISER+J-1)=(MAXTQ-MINTQ)/NUTIL
        SOFFST(ISER+J-1)=MINTQ+SDELTA(ISER+J-1)*HALF
        SSKIP(ISER+J-1)=1
     ENDDO
     call chmdealloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
     !
     !=======================================================================
     ! to fill-time-series-with-state
  ELSE IF(MANCOD.EQ.'STAT') THEN
     !        check-second-time-series
     MINTQ=FACTOR
     MAXTQ=FACTR2
     DO K=1,VECCD
        DO I=1,NTOT
           IF(TQ(I,K).LT.MINTQ) THEN
              TQ(I,K)=ZERO
           ELSE IF(TQ(I,K).GT.MAXTQ) THEN
              TQ(I,K)=ZERO
           ELSE
              TQ(I,K)=ONE
           ENDIF
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to generate-a-weighted-histogram, C.Bartels, 1998
#if KEY_ADUMB==1
  ELSE IF(MANCOD.EQ.'WHIS') THEN
     call chmalloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
     MINTQ=FACTOR
     MAXTQ=FACTR2
     DO J=1,VECCD
        CALL WHIST(TQ(1,J),NTOT,MINTQ,MAXTQ,NUTIL,NINSUB,TQ2(1,J))
        STOT(ISER+J-1)=NUTIL
        SDELTA(ISER+J-1)=(MAXTQ-MINTQ)/NUTIL
        SOFFST(ISER+J-1)=MINTQ+SDELTA(ISER+J-1)*HALF
        SSKIP(ISER+J-1)=1
     ENDDO
     call chmdealloc('mantim.src','MANTIM','NINSUB',NUTIL,crl=NINSUB)
#endif 
     !
     !=======================================================================
     ! to heaviside-the-time-series
  ELSE IF(MANCOD.EQ.'HEAV') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           IF(TQ(I,K).LT.ZERO) THEN
              TQ(I,K)=ZERO
           ELSE
              TQ(I,K)=ONE
           ENDIF
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to log-the-time-series
  ELSE IF(MANCOD.EQ.'LOG ') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           QTEMP=TQ(I,K)
           IF(QTEMP.LT.1.E-12) QTEMP=1.E-12
           TQ(I,K)=LOG(QTEMP)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to exponentiate-the-time-series
  ELSE IF(MANCOD.EQ.'EXP ') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           QTEMP=TQ(I,K)
           IF(QTEMP.GT.82.0) QTEMP=82.0
           TQ(I,K)=EXP(QTEMP)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to power-the-time-series
  ELSE IF(MANCOD.EQ.'IPOW') THEN
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)**NUTIL
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to multiply-time-series-by-a-constant
  ELSE IF(MANCOD.EQ.'MULT') THEN
     QTEMP=FACTOR
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=QTEMP*TQ(I,K)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to offset-time-series-from-minimum
  ELSE IF(MANCOD.EQ.'DMIN') THEN
     DO J=1,VECCD
        QTEMP=TQ(1,J)
        DO K=1,NTOT
           IF(TQ(K,J).LE.QTEMP)QTEMP=TQ(K,J)
        ENDDO
        DO K=1,NTOT
           TQ(K,J)=TQ(K,J)-QTEMP
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to absolute-value-of-time-series
  ELSE IF(MANCOD.EQ.'ABS ') THEN
     IF(VECCD.EQ.1) THEN
        DO I=1,NTOT
           TQ(I,1)=ABS(TQ(I,1))
        ENDDO
     ELSE
        DO I=1,NTOT
           QTEMP=ZERO
           DO K=1,VECCD
              QTEMP=QTEMP+TQ(I,K)**2
              TQ(I,K)=ZERO
           ENDDO
           IF(QTEMP.GT.ZERO) TQ(I,1)=SQRT(QTEMP)
        ENDDO
     ENDIF
     !
     !=======================================================================
     ! to add-a-constant-to-time-series
  ELSE IF(MANCOD.EQ.'SHIF') THEN
     QTEMP=FACTOR
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=QTEMP+TQ(I,K)
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to zero-time-series
  ELSE IF(MANCOD.EQ.'ZERO') THEN
     DO K=1,VECCD
        DO J=1,NTOT
           TQ(J,K)=0.0
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to divide-time-series-by-a-constant
  ELSE IF(MANCOD.EQ.'DIVI') THEN
     QTEMP=ONE/FACTOR
     DO K=1,VECCD
        DO I=1,NTOT
           TQ(I,K)=TQ(I,K)*QTEMP
        ENDDO
     ENDDO
     !
     !=======================================================================
     ! to divide-series-by-first-value
  ELSE IF(MANCOD.EQ.'DIVF') THEN
     QAVE=0.0
     DO K=1,VECCD
        QAVE=QAVE+TQ(1,K)**2
     ENDDO
     IF(QAVE.GT.ZERO) THEN
        QAVE=SQRT(QAVE)
        DO K=1,VECCD
           DO J=1,NTOT
              TQ(J,K)=TQ(J,K)/QAVE
           ENDDO
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<MANTIM>','First element has a zero norm')
     ENDIF
     !
     !=======================================================================
     ! to divide-series-by-maximum-value
  ELSE IF(MANCOD.EQ.'DIVM') THEN
     QMAX=0.0
     DO J=1,NTOT
        QAVE=0.0
        DO K=1,VECCD
           QAVE=QAVE+TQ(1,K)**2
        ENDDO
        IF(QAVE.GT.QMAX) QMAX=QAVE
     ENDDO
     IF(QMAX.GT.ZERO) THEN
        QAVE=SQRT(QMAX)
        DO K=1,VECCD
           DO J=1,NTOT
              TQ(J,K)=TQ(J,K)/QAVE
           ENDDO
        ENDDO
     ELSE
        CALL WRNDIE(-1,'<MANTIM>','Maximum value has a zero norm')
     ENDIF
     !
     !=======================================================================
     ! to differentiate-the-series
  ELSE IF(MANCOD.EQ.'DERI') THEN
     IF(VECCD.NE.1) THEN
        CALL WRNDIE(0,'<MANTIM>', &
             'VECCOD should be 1 for DERIvative.')
     ENDIF
     STEPT=NSKIP*DELTA
     DO K=1,VECCD
        DO I=1,NTOT-1
           TQ(I,K)=(TQ(I+1,K)-TQ(I,K))/STEPT
        ENDDO
        TQ(NTOT,K)=TQ(NTOT-1,K)
     ENDDO

     !=======================================================================
     ! to integrate-the-series
  ELSE IF(MANCOD.EQ.'INTE') THEN
     IF(VECCD.NE.1) THEN
        CALL WRNDIE(0,'<MANTIM>', &
             'VECCOD should be 1 for INTEgrate.')
     ENDIF
     DO K=1,VECCD
        DO J=1,NTOT
           FAH(J)=TQ(J,K)
        ENDDO
        STEPT=NSKIP*DELTA
! write result back to TQ; avoids copy back operation and solves
! problem that occurs when input and output are the same array
        CALL QSF(STEPT,FAH,TQ(:,K),NTOT)
     ENDDO
     !=======================================================================
     ! BEGIN Moving average (G. Lamoureux)
  ELSE IF(MANCOD.EQ.'MOVI') THEN
     !        IF(VECCD.NE.1) THEN
     !           CALL WRNDIE(0,'<MANTIM>',
     !    &                  'VECCOD should be 1 for MOVIng.')
     !        ENDIF
     DO K=1,VECCD
        !           FAH contains the portion of the timeseries to average
        !           QAVE is the moving sum
        !           NEWT is the number of data to average
        DO J=1,NUTIL
           FAH(J)=0.0
        ENDDO
        QAVE=0.0
        NEWT=0
        DO I=1,NTOT
           J=MOD(I,NUTIL)+1
           QAVE=QAVE-FAH(J)
           FAH(J)=TQ(I,K)
           QAVE=QAVE+FAH(J)
           IF(NEWT.LT.NUTIL) NEWT=NEWT+1
           TQ(I,K)=QAVE/NEWT
        ENDDO
     ENDDO
     ! END Moving average (G. Lamoureux)
     !=======================================================================
     ! to convert cartesian to spherical coordinates. L. Nilsson
  ELSE IF(MANCOD.EQ.'SPHE') THEN
     IF(VECCD.NE.3)THEN
        CALL WRNDIE(-2,'<MANTIM>', &
             '3-component series needed for conversion to spherical coor')
     ELSE
        DO J=1,NTOT
           QTEMP=TQ(J,1)**2+TQ(J,2)**2+TQ(J,3)**2
           IF(QTEMP.LT.RSMALL)THEN
              SB=0.0
           ELSE
              QTEMP=SQRT(QTEMP)
              SB=ACOS(TQ(J,3)/QTEMP) * RADDEG
           ENDIF
           IF(ABS(TQ(J,1)).LT.RSMALL.AND.ABS(TQ(J,2)).LT.RSMALL)THEN
              SA=0.0
           ELSE
              SA=ATAN2(TQ(J,2),TQ(J,1)) * RADDEG
           ENDIF
           TQ(J,1)=QTEMP
           TQ(J,2)=SA
           TQ(J,3)=SB
        ENDDO
     ENDIF
     !
     !=======================================================================
     ! to generate-test-series
  ELSE IF(MANCOD.EQ.'TEST') THEN
     QTEMP=FACTOR
     DO K=1,VECCD
        DO J=1,NTOT
           TQ(J,K)=COS(((J-1)*TWO*PI*QTEMP)/NTOT)
        ENDDO
     ENDDO
     !
     !=======================================================================
  ELSE
     CALL WRNDIE(0,'<MANTIM>', &
          'Unrecognized manipulation operation.')
     IF(WRNLEV.GE.2) WRITE(OUTU,222) MANCOD
222  FORMAT(' Nothing done for "',A,'"')
     !
     !=======================================================================
     !
  ENDIF
  RETURN
  !yw
902 FORMAT(' Series "',A,'"  VECCOD=',I2, &
       ', Series "',A,'"  VECCOD=',I2)
905 FORMAT(' Series "',A,'"  TOTAL=',I6, &
       ', Series "',A,'"  TOTAL=',I6)
915 FORMAT(' Series "',A,'"  SKIP=',I4,' DELTA=',F12.6)
925 FORMAT(/' PROCEEDING ANYWAY. GOOD LUCK.')
  !
END SUBROUTINE MANTIM

SUBROUTINE OSC(TQ,NTOT,NBOSC)
  !
  ! THIS ROUTINE COMPUTES THE AVERAGE NUMBER OF OSCILLATIONS
  ! PER UNIT STEP TIME CONTAINED WITHIN TQ. THIS NUMBER IS
  ! RETURNED BY NBOSC
  !
  !     Author: B. Brooks
  !
  use chm_kinds
  use stream
  implicit none
  real(chm_real) TQ(*)
  INTEGER NTOT
  real(chm_real) NBOSC
  !
  INTEGER ASC,NOSC,I,TMOSCI,TMOSCF
  !
  ASC=0
  NOSC=0
  DO I=1,NTOT
     IF(ASC.EQ.1) THEN
        !           ascending portion
        IF(TQ(I).LT.TQ(I-1)) THEN
           ASC=-1
           IF(NOSC.EQ.0) TMOSCI=I
           NOSC=NOSC + 1
           TMOSCF=I
        ENDIF
     ELSE IF(ASC.EQ.-1) THEN
        !           descending portion
        IF(TQ(I).GT.TQ(I-1)) THEN
           ASC=1
           IF(NOSC.EQ.0) TMOSCI=I
           NOSC=NOSC + 1
           TMOSCF=I
        ENDIF
     ELSE IF(ASC.EQ.0) THEN
        !           initial part
        IF(TQ(I).LT.TQ(I-1)) ASC=-1
        IF(TQ(I).GT.TQ(I-1)) ASC=1
     ENDIF
  ENDDO
  !
  !
  !  number of oscillations per unit step time
  !
  NOSC=NOSC-1
  IF(NOSC.GT.0) THEN
     NBOSC=2*(TMOSCF-TMOSCI)
     NBOSC=NOSC/NBOSC
  ELSE
     NBOSC=0.0
  ENDIF
  !
  RETURN
END SUBROUTINE OSC

SUBROUTINE SUBDIV(TQ,NTOT,MINTQ,MAXTQ,NINT,NINSUB)
  !
  ! THIS ROUTINE FINDS THE INTERVAL BETWEEN TWO SUBDIVISIONS
  ! (INTTQ) OF THE TQ TIME SERIES GIVEN THE NUMBER OF INTERVALS
  ! (NINT) BETWEEN THE MINIMUM AND THE MAXIMUM.
  !
  !     Author: David Perahia
  !
  use chm_kinds
  use number
  use stream
  implicit none
  real(chm_real) TQ(*),MINTQ,MAXTQ
  INTEGER NTOT,NINT
  real(chm_real) NINSUB(*)
  !
  real(chm_real) INTTQ,RDIV
  INTEGER I,II,IDIV,JDIV
  !
  INTTQ=(MAXTQ-MINTQ)/NINT
  !
  RDIV=MINTQ
  IF(PRNLEV.GE.6) WRITE(OUTU,1417)
1417 FORMAT(3X,'THE SELECTED SUBDIVISIONS ARE'/)
  DO II=1,NINT+1
     IF(PRNLEV.GE.6) WRITE(OUTU,1416) II,RDIV
     RDIV=RDIV+INTTQ
1416 FORMAT(5X,I4,2X,F8.3)
  ENDDO
  !
  DO I=1,NINT
     NINSUB(I)=0
  ENDDO
  !
  DO II=1,NTOT
     RDIV=(TQ(II)-MINTQ)/INTTQ+ONE
     IDIV=RDIV
     JDIV=(RDIV-RSMALL)
     IF(IDIV.NE.JDIV .AND. JDIV.EQ.NINT) IDIV=NINT
     IF(IDIV.GT.0 .AND. IDIV.LE.NINT) NINSUB(IDIV)=NINSUB(IDIV)+1
  ENDDO
  !
  DO I=1,NINT
     TQ(I)=NINSUB(I)
     TQ(I)=TQ(I)/NTOT
  ENDDO
  !
  RETURN
END SUBROUTINE SUBDIV

! weighted histogram, C.Bartels
#if KEY_ADUMB==1

SUBROUTINE WHIST(TQ,NTOT,MINTQ,MAXTQ,NINT,NINSUB,WE)
  !
  ! histogram from weighted time series
  !
  !
  use chm_kinds
  use number
  use stream
  implicit none
  real(chm_real) TQ(*),MINTQ,MAXTQ,NINSUB(*),WE(*)
  INTEGER NTOT,NINT
  !
  real(chm_real) INTTQ,RDIV,TOTWE,TOT,TOTS
  INTEGER I,II,IDIV,JDIV
  !
  INTTQ=(MAXTQ-MINTQ)/NINT
  !
  RDIV=MINTQ
  IF(PRNLEV.GE.6) WRITE(OUTU,1417)
1417 FORMAT(3X,'THE SELECTED SUBDIVISIONS ARE'/)
  DO II=1,NINT+1
     IF(PRNLEV.GE.6) WRITE(OUTU,1416) II,RDIV
     RDIV=RDIV+INTTQ
1416 FORMAT(5X,I4,2X,F8.3)
  ENDDO
  !
  DO I=1,NINT
     NINSUB(I)=0
  ENDDO
  !
  TOTWE=0.0
  TOT=0.0
  TOTS=0.0
  DO II=1,NTOT
     TOT=TOT+TQ(II)*WE(II)
     TOTS=TOTS+TQ(II)*TQ(II)*WE(II)
     TOTWE=TOTWE+WE(II)
     RDIV=(TQ(II)-MINTQ)/INTTQ+ONE
     IDIV=RDIV
     JDIV=(RDIV-RSMALL)
     IF(IDIV.NE.JDIV .AND. JDIV.EQ.NINT) IDIV=NINT
     IF(IDIV.GT.0 .AND. IDIV.LE.NINT) THEN
        !           WRITE(OUTU,*) II,WE(II)
        NINSUB(IDIV)=NINSUB(IDIV)+WE(II)
     END IF
  ENDDO
  !
  DO I=1,NINT
     TQ(I)=NINSUB(I)
     TQ(I)=TQ(I)/TOTWE
  ENDDO
  !
  IF(PRNLEV.GE.0.AND.TOTWE.GT.0) &
       WRITE(OUTU,400) TOT/TOTWE,SQRT(TOTS/TOTWE-(TOT/TOTWE)**2)
400 FORMAT('   Average: ',F15.8,'   Standard Dev.: ',F15.8)
  !
  RETURN
END SUBROUTINE WHIST

! end weighted histogram
#endif 
!

SUBROUTINE POLYFIT(NTOT,RDUM,DELTA,IDTP,TQ,QWEIG,WEIG, &
     RESULT,SCRTCH,INDPT,IORDER,QREPL)
  !
  use chm_kinds
  use stream
  use param_store, only: set_param

  implicit none
  !
  INTEGER NTOT,IDTP(*),INDPT(*),IORDER
  real(chm_real)  RDUM(*),TQ(*),WEIG(*),RESULT(*),SCRTCH(*),DELTA
  !
  LOGICAL QWEIG,QREPL
  !
  INTEGER I,J
  real(chm_real) RVAL,SSQ,WTOT
  !
  DO J=1,NTOT
     RDUM(J)=J*DELTA
     IDTP(J)=0
  ENDDO
  !
  IF(.NOT.QWEIG) THEN
     DO J=1,NTOT
        WEIG(J)=1.0
     ENDDO
  ENDIF
  !
  CALL LSSOLV(RDUM,IDTP,TQ,WEIG,NTOT,RESULT,SCRTCH, &
       INDPT,IORDER)
  !
  SSQ=0.0
  WTOT=0.0
  DO J=1,NTOT
     RVAL=0.0
     DO I=1,IORDER
        RVAL=RVAL+RDUM(J)**(IORDER-I)*RESULT(I)
     ENDDO
     SSQ=SSQ+WEIG(J)*(RVAL-TQ(J))**2
     WTOT=WTOT+WEIG(J)
     IF(QREPL) TQ(J)=RVAL
  ENDDO
  IF(WTOT.EQ.0.0) RETURN
  SSQ=SSQ/WTOT
  !
  IF(PRNLEV.GE.3) THEN
     WRITE(OUTU,42)
42   FORMAT(' POLYFIT>  Solution to fit:')
     DO J=1,IORDER
        WRITE(OUTU,43) J-1,RESULT(IORDER-J+1)
43      FORMAT('        For exponent:',I3,'  coefficient=',F16.6)
     ENDDO
     WRITE(OUTU,44) SSQ,WTOT
44   FORMAT(' POLYFIT>  Average S**2 =',F14.4, &
          '  Total weight =',F14.4)
  ENDIF
  !
  !
  IF(IORDER.GE.1) call set_param('P0',RESULT(IORDER))
  IF(IORDER.GE.2) call set_param('P1',RESULT(IORDER-1))
  IF(IORDER.GE.3) call set_param('P2',RESULT(IORDER-2))
  IF(IORDER.GE.4) call set_param('P3',RESULT(IORDER-3))
  IF(IORDER.GE.5) call set_param('P4',RESULT(IORDER-4))
  IF(IORDER.GE.6) call set_param('P5',RESULT(IORDER-5))
  IF(IORDER.GE.7) call set_param('P6',RESULT(IORDER-6))
  IF(IORDER.GE.8) call set_param('P7',RESULT(IORDER-7))
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,66) (RESULT(J),J=1,IORDER)
66 FORMAT(/7F12.6)
  !
  return
end SUBROUTINE POLYFIT

