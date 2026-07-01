#if KEY_ADUMB==1 /*adumb*/
SUBROUTINE UMBAN(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Process the UMBRELLA command.
  !
  use bases_fcm
  use dimens_fcm
  use chm_kinds
  use code
  use coord
  use coordc
  use noem
  use number
  use psf
  use stream
  use umb
  use memory
  implicit none
  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN

  CALL UMBAN2(COMLYN, COMLEN, X, Y, Z, WMAIN, &
       ICB, ICT, ICP, ICI, MUSTUP)

  !
  RETURN
END SUBROUTINE UMBAN
!
SUBROUTINE UMBAN2(COMLYN,COMLEN,X,Y,Z,WT,ICB,ICT,ICP,ICI, &
     MUSTUP)
  use bases_fcm
  use dimens_fcm
  use econtmod
  use chm_kinds
  use intcor_module
  use number
  use noem
  use psf
  use umb
  use consta
  use stream
  use replica_mod
  use umbcor
  use memory
  use chutil
  use select
  use string
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,domdec_system_changed
#endif
#if KEY_ADUMBRXNCOR==1
  use rxncom
  use rxdefs, only: indgrf,setstt
#endif 
  !
  implicit none
  INTEGER ICB(*),ICT(*),ICP(*),ICI(*)
  INTEGER COMLEN
  character(len=*) COMLYN
  LOGICAL QWEIG,MUSTUP
  real(chm_real) X(*),Y(*),Z(*),WT(*),TOLD
  !
  INTEGER QAT(4),NQAT,I,STCNT
  character(len=4) WRD
  character(len=8) SIDI,RIDI,RENI,ACI,SIDJ,RIDJ,RENJ,ACJ
  character(len=8) SIDK,RIDK,RENK,ACK,SIDL,RIDL,RENL,ACL
  INTEGER IS,J,RMSMEM,RMSSTR
  LOGICAL ERR
  INTEGER I1,I2,SYMM,NRMSAT,NORIAT,NSYMA1
  LOGICAL ORIENT
  character(len=4) WRDD
  INTEGER,allocatable,dimension(:) :: FLAGS, ISLCT, JSLCT
#if KEY_ADUMBRXNCOR==1
  character(len=4) RXNNAM
  integer ir
#endif 

  !
  !
  ! Adaptive umbrella uses econt arrays. Check to see if already 
  ! allocated. If not allocate here. cb3
  if(.not. allocated(econt)) call allocate_econt
  !
  !
  call chmalloc("cmds.src","umban2","flags",natom,intg=flags)
  !
  !     THE FIRST WORD IN THE COMMAND LINE DETERMINES WHAT IS BEING DONE.
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  !
  !     BRANCH TO THE APPROPRIATE SECTION.
  !
  !-----------------------------------------------------------------------
  !
  IF(WRD == 'INIT') THEN
     IF(NSIUM == ZERO) THEN
        FUPUM=10000
        NEQUM=1000
        WUNUM=0
        WCUNUM=0
        TUM=300
        MAXDEV=10
        AGEUM=1.0
        STONUM=.TRUE.
        NEXUM=2
     END IF
     FUPUM=GTRMI(COMLYN,COMLEN,'UPDA',FUPUM)
     NEQUM=GTRMI(COMLYN,COMLEN,'EQUI',NEQUM)
     WCUNUM=GTRMI(COMLYN,COMLEN,'UCUN',WCUNUM)
     WUNUM=GTRMI(COMLYN,COMLEN,'WUNI',WUNUM)
     RUNUM=GTRMI(COMLYN,COMLEN,'RUNI',0)
     NEXUM=GTRMI(COMLYN,COMLEN,'NEXT',NEXUM)
     ADFREQ=GTRMI(COMLYN,COMLEN,'FREQ',1)
     IF(PRNLEV > 2) WRITE(OUTU,'(A28,1x,I6,1x,A21)') &
          'sampling will occur at every', &
          ADFREQ,'steps in the dynamics'
     TOLD=TUM
     TUM=GTRMF(COMLYN,COMLEN,'TEMP',TUM)
     MAXDEV=GTRMF(COMLYN,COMLEN,'THRE',MAXDEV)
     AGEUM=GTRMF(COMLYN,COMLEN,'AGIN',AGEUM)
#if KEY_ADUMBRXNCOR==1
     maxbias=GTRMF(COMLYN,COMLEN,'MAXB',RBIG) 
#endif
     IF ((NSIUM == ZERO).OR.(LCORRV)) THEN
        TOLD=TUM
        NSIUM=GTRMI(COMLYN,COMLEN,'NSIM',0)
        CALL UMINIT()
        IF(PRNLEV > 2) WRITE(OUTU,*) &
             'Adaptive umbrella sampling initialized:'
     ELSE IF(PRNLEV >= 2) THEN
        WRITE(OUTU,380)
380     FORMAT('Adaptive umbrella sampling, changed parameters:')
     END IF
#if KEY_ADUMBRXNCOR==1
!need to initialize rxncor's data structures
     if (numbrxn > 0) then
          call chmalloc('cmds.src','umban2','NMLSTT',numbrxn,intg=NMLSTT)
          call chmalloc('cmds.src','umban2','NRXSTT',numbrxn,intg=NRXSTT)
          call chmalloc('cmds.src','umban2','LODEL',numbrxn,crl=LODEL)
          call chmalloc('cmds.src','umban2','HIDEL',numbrxn,crl=HIDEL)
          call chmalloc('cmds.src','umban2','DELDEL',numbrxn,crl=DELDEL)
          do i=1,numbrxn
             lodel(rxncorindex(i))=umbmin(i)
             hidel(rxncorindex(i))=umbmax(i)
          end do
!       CALL SETSTT(COMLYN,COMLEN,IR,LODEL,HIDEL, &
!            DELDEL,NRXSTT,NMLSTT, &
!            TREELO,numbrxn,NODNAM)
       ir = nmlstt(1)*nrxstt(1)
!       call chmalloc('rxndef.src','rxpars','DELSTP',IR,intg=DELSTP)

     endif
#endif 
     IF (LCORRV) THEN
        PCORMD = GTRMI(COMLYN,COMLEN,'CPFR',1)
        IF(PRNLEV > 2) WRITE(OUTU,'(A30,1x,I6,1x,A23)') &
             'correlatns will be writtn evry', &
             PCORMD,'update(s) of umbr poten'
        WCRUNI = GTRMI(COMLYN,COMLEN,'WCUN',-1)
        IF ((WCRUNI > 0).AND.(PRNLEV.GT.2))WRITE(OUTU,'(A45,1X,I8)') &
             'will write corr variable accumulators to unit', &
             WCRUNI
        RCRUNI = GTRMI(COMLYN,COMLEN,'RCUN',-1)
        IF((RCRUNI > 0).AND.(PRNLEV.GT.2))WRITE(OUTU,'(A46,1X,I8)') &
             'will read corr variable accumulators from unit', &
             RCRUNI
        IF (LCNEED) THEN

           call chmalloc('cmds.src','UMBAN2','HPCRRD',INSIUM,NDISTA,crl=HPCRRD)

           call chmalloc('cmds.src','UMBAN2','HPTOAV',INSIUM,NDISTA,crl=HPTOAV)

           call chmalloc('cmds.src','UMBAN2','HPCRMS1',INSIUM,NRMSDI,crl=HPCRMS1)

           call chmalloc('cmds.src','UMBAN2','HPTORM1',INSIUM,NRMSDI,crl=HPTORM1)

           call chmalloc('cmds.src','UMBAN2','HPCRMS2',INSIUM,NRMSDI,crl=HPCRMS2)

           call chmalloc('cmds.src','UMBAN2','HPTORM2',INSIUM,NRMSDI,crl=HPTORM2)

           call chmalloc('cmds.src','UMBAN2','HPCPYX',NATOM,crl=HPCPYX)

           call chmalloc('cmds.src','UMBAN2','HPCPYY',NATOM,crl=HPCPYY)

           call chmalloc('cmds.src','UMBAN2','HPCPYZ',NATOM,crl=HPCPYZ)

           call chmalloc('cmds.src','UMBAN2','HPCP2X',NATOM,crl=HPCP2X)

           call chmalloc('cmds.src','UMBAN2','HPCP2Y',NATOM,crl=HPCP2Y)

           call chmalloc('cmds.src','UMBAN2','HPCP2Z',NATOM,crl=HPCP2Z)

           call chmalloc('cmds.src','UMBAN2','HPATMN',2,NATOM,intg=HPATMN)

           call chmalloc('cmds.src','UMBAN2','HPNDIS',NDISTA,crl=HPNDIS)

           call chmalloc('cmds.src','UMBAN2','HPCOR1',NRMSDI,crl=HPCOR1)

           call chmalloc('cmds.src','UMBAN2','HPCOR2',NRMSDI,crl=HPCOR2)

           call chmalloc('cmds.src','UMBAN2','HPXCPS',MXRMSA,crl=HPXCPS)

           call chmalloc('cmds.src','UMBAN2','HPYCPS',MXRMSA,crl=HPYCPS)

           call chmalloc('cmds.src','UMBAN2','HPZCPS',MXRMSA,crl=HPZCPS)

           call chmalloc('cmds.src','UMBAN2','HPTEMX',MXRMSA,crl=HPTEMX)

           call chmalloc('cmds.src','UMBAN2','HPTEMY',MXRMSA,crl=HPTEMY)

           call chmalloc('cmds.src','UMBAN2','HPTEMZ',MXRMSA,crl=HPTEMZ)

           call chmalloc('cmds.src','UMBAN2','HPTE2X',MXRMSA,crl=HPTE2X)

           call chmalloc('cmds.src','UMBAN2','HPTE2Y',MXRMSA,crl=HPTE2Y)

           call chmalloc('cmds.src','UMBAN2','HPTE2Z',MXRMSA,crl=HPTE2Z)

           LCNEED = .FALSE.
        ENDIF
        call INITCORR(HPCRRD,HPTOAV,HPCRMS1,HPCRMS2,HPTORM1, &
             HPTORM2)
     ENDIF
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,390) TUM,MAXDEV,AGEUM,NEXUM, &
             NSIUM,FUPUM,NEQUM,WCUNUM, &
             WUNUM,RUNUM
390     FORMAT('   Temperature                  :',G10.3,/, &
             '   Threshold factor             :',G10.3,/, &
             '   Aging factor                 :',G10.3,/, &
             '   Number of extrapolated bins  :',I10,/, &
             '   Number of simulations        :',I10,/, &
             '   Update frequency             :',I10,/, &
             '   Number of equilibration steps:',I10,/, &
             '   Write umbrella coord. to unit:',I10,/, &
             '   Write data to unit           :',I10,/, &
             '   Read histograms from unit    :',I10)
     ENDIF
     IF(TUM /= TOLD) THEN
        IF(IEUM > 0) THEN

           CALL UMCHT(TOLD,TUM,IEUM,STSFUM)

           call UMSTA3(STHIUM,STMFUM,STEMUM,STFPUM,STDAUM, &                               
                STNSUM,STUPUM,STERUM,STFLUM,STSFUM,STAGUM)     

           call UMWRT(STHIUM,STMFUM,STEMUM,STFPUM,STUPUM, &
                STSFUM)

        ELSE IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) 'WARNING: no energy umbrella'
           WRITE(OUTU,*) '         -> temperature not adapted'
        END IF
     END IF
     IF(RUNUM > ZERO) THEN

        call UMRFI(STHIUM,STEMUM,STFPUM,STNSUM,STUPUM, &
             STSFUM)

     END IF
     IF(RCRUNI > ZERO) THEN

        call UMRDCOR(HPTOAV,HPTORM1,HPTORM2)

     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'PROB') THEN
     WPUNUM=GTRMI(COMLYN,COMLEN,'PUNI',0)
     WTUNUM=GTRMI(COMLYN,COMLEN,'TUNI',0)
     RCUNUM=GTRMI(COMLYN,COMLEN,'UCUN',0)
     TEUM=GTRMF(COMLYN,COMLEN,'TEMP',TUM)
     IF(TEUM /= TUM .AND. IEUM == 0) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,393)
393     FORMAT('*****  WARNING  ***** ',/, &
             ' Energy umbrella needed to change temperature.')
     ELSE IF(RCUNUM == ZERO) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,394)
394     FORMAT('*****  WARNING  ***** ',/, &
             ' Unit to read umbrella coordinates (UCUN) is not defined.')
     ELSE IF(.not. allocated(STHIUM)) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,396)
396     FORMAT('*****  WARNING  ***** ',/, &
             ' No potential of mean force available.')
     ELSE
        call UMGETP(STMFUM,STEMUM,STCAUM,STFLUM)

     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'STON') THEN
     STONUM=.TRUE.
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'STOF') THEN
     STONUM=.FALSE.
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'WRIT') THEN
     IF(allocated(STHIUM)) THEN
        call UMWRT(STHIUM,STMFUM,STEMUM,STFPUM,STUPUM, &
             STSFUM)

     ELSE
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,397)
397     FORMAT('*****  WARNING  ***** ',/, &
             ' No statistic available.')
     ENDIF
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'ENER') THEN
     ! set-up-energy umbrella
     !
     ! The energy umbrella is processed here
     !
#if KEY_DOMDEC==1
     if (q_domdec) call wrndie(-2,'<UMBAN2>',&
        'ADUMB ENER may not work with domain decomposition. Use at your own risk.')
#endif 
     IF(allocated(STHIUM)) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,400)
400     FORMAT('*****  WARNING  ***** ',/, &
             ' Satistic already initialized.')
        GOTO 700
     ENDIF
     IF(IEUM /= ZERO) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,405)
405     FORMAT('*****  WARNING  ***** ',/, &
             ' Energy umbrella already defined.')
        GOTO 700
     ENDIF
     NUMBR=NUMBR+1
     IEUM=NUMBR
     TYPUM(NUMBR)=2
     IF(NUMBR > MAXUMP) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,410)
410     FORMAT('*****  WARNING  ***** ',/, &
             ' Too many umbrellas.')
        GOTO 700
     ENDIF
     NBUM(NUMBR)=GTRMI(COMLYN,COMLEN,'NRES',0)
     IF(NBUM(NUMBR) <= 0) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,415)
415     FORMAT(' *****  WARNING  ***** ',/, &
             ' Resolution for statistics is not', &
             ' specified.')
        GOTO 700
     ENDIF
     TFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'TRIG',0)
     PFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'POLY',0)
     IF(NBUM(NUMBR) < (PFUM(NUMBR)+2*TFUM(NUMBR))) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,420)
420     FORMAT(' *****  WARNING  ***** ',/, &
             ' More fit functions than data points.')
        GOTO 700
     ENDIF
     EMAXUM=GTRMF(COMLYN,COMLEN,'MAXE',ZERO)
     EMINUM=GTRMF(COMLYN,COMLEN,'MINE',ZERO)
     TMAXUM=1000.0
     TMINUM=273.0
     TMAXUM=GTRMF(COMLYN,COMLEN,'MAXT',TMAXUM)
     TMINUM=GTRMF(COMLYN,COMLEN,'MINT',TMINUM)
     !
     IF(PRNLEV >= 2) WRITE(OUTU,425) NUMBR,TFUM(NUMBR), &
          PFUM(NUMBR),NBUM(NUMBR),EMINUM,EMAXUM, &
          TMINUM,TMAXUM
425  FORMAT(' ENERGY UMBRELLA ADDED'/I5,':',4X, &
          'TRIG=',I4,'  POLY=',I4,',  NRES=',I4/, &
          10X,'MINE=',G10.3,'  MAXE=',G10.3/, &
          10X,'MINT=',G10.3,'  MAXT=',G10.3)
     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'NOE ') THEN
     ! set-up-NOE umbrella
     ! New type of umbrella coordinate on which we have currently a project running.
     ! Thus, please do not use. C. Bartels, August 1998
     !
     ! The NOE umbrella is processed here
#if KEY_DOMDEC==1
     !
     !if (q_domdec) 
     call wrndie(-10,'<UMBAN2>','ADUMB NOE not supported with domain decomposition.')
#endif 
     IF(NNUM >= MAXNUM) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,475)
475     FORMAT('*****  WARNING  ***** ',/, &
             ' Too many NOE data sets.')
        call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
        RETURN
     ENDIF
     NUMBR=NUMBR+1
     IF(NNUM == ZERO) THEN
        INUM=NUMBR
        TYPUM(NUMBR)=5
        IF(NUMBR > MAXUMP) THEN
           IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,410)
           GOTO 750
        ENDIF
        NBUM(NUMBR)=GTRMI(COMLYN,COMLEN,'NRES',0)
        IF(NBUM(NUMBR) <= 0) THEN
           IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,415)
           GOTO 750
        ENDIF
        TFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'TRIG',0)
        PFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'POLY',0)
        IF(NBUM(NUMBR) < (PFUM(NUMBR)+2*TFUM(NUMBR))) THEN
           IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,420)
           GOTO 750
        ENDIF
        EMAXNU=1.0
        EMINNU=0.0
        EMAXNU=GTRMF(COMLYN,COMLEN,'MAXE',EMAXNU)
        EMINNU=GTRMF(COMLYN,COMLEN,'MINE',EMINNU)
        LUPPNU=EMAXNU
        LLOWNU=EMINNU
        LUPPNU=GTRMF(COMLYN,COMLEN,'UPPE',LUPPNU)
        LLOWNU=GTRMF(COMLYN,COMLEN,'LOWE',LLOWNU)
     ENDIF
     !
     NNUM=NNUM+1
     ! get limit
     ELIMNU(NNUM) = 10.0
     ELIMNU(NNUM) = GTRMF(COMLYN,COMLEN,'LIMI',ELIMNU(NNUM))
     ! allocate data structures for forces

     call chmalloc('cmds.src','UMBAN2','STFNUM(NNUM)',NATOM,crl=STFNUM(NNUM)%x)
     call chmalloc('cmds.src','UMBAN2','STFNUM(NNUM)',NATOM,crl=STFNUM(NNUM)%y)
     call chmalloc('cmds.src','UMBAN2','STFNUM(NNUM)',NATOM,crl=STFNUM(NNUM)%z)

     ! allocate real data structures for NOE data set

     call chmalloc('cmds.src','UMBAN2','RMNNUM(NNUM)',NOEMAX,crl=RMNNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','KMNNUM(NNUM)',NOEMAX,crl=KMNNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','RMXNUM(NNUM)',NOEMAX,crl=RMXNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','KMXNUM(NNUM)',NOEMAX,crl=KMXNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','FMXNUM(NNUM)',NOEMAX,crl=FMXNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','TCNNUM(NNUM)',NOEMAX,crl=TCNNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','AVENUM(NNUM)',NOEMAX,crl=AVENUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','MINNUM(NNUM)',NOEMAX,log=MINNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','EXPNUM(NNUM)',NOEMAX,crl=EXPNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','RSWNUM(NNUM)',NOEMAX,crl=RSWNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','SEXNUM(NNUM)',NOEMAX,crl=SEXNUM(NNUM)%a)

     ! allocate integer data structures for NOE data set

     call chmalloc('cmds.src','UMBAN2','IPTNUM(NNUM)',NOEMAX,intg=IPTNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','JPTNUM(NNUM)',NOEMAX,intg=JPTNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','INMNUM(NNUM)',NOEMAX,intg=INMNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','JNMNUM(NNUM)',NOEMAX,intg=JNMNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','LISNUM(NNUM)',NOEMAX,intg=LISNUM(NNUM)%a)

     call chmalloc('cmds.src','UMBAN2','RAMNUM(NNUM)',NOEMAX,intg=RAMNUM(NNUM)%a)

     ! fill in data structures for noe data set
     CALL NUMFIL( &
          IPTNUM(NNUM)%a,JPTNUM(NNUM)%a, &
          INMNUM(NNUM)%a,JNMNUM(NNUM)%a, &
          LISNUM(NNUM)%a,RMNNUM(NNUM)%a, &
          KMNNUM(NNUM)%a,RMXNUM(NNUM)%a, &
          KMXNUM(NNUM)%a,FMXNUM(NNUM)%a, &
          TCNNUM(NNUM)%a,AVENUM(NNUM)%a, &
          MINNUM(NNUM)%a,EXPNUM(NNUM)%a &
          !JH (soft asymptote)
          ,RSWNUM(NNUM)%a,SEXNUM(NNUM)%a, &
          RAMNUM(NNUM)%a &
          )
     NUMNUM(NNUM)=NOENUM
     NM2NUM(NNUM)=NOENM2
     SCANUM(NNUM)=NOESCA
     !
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,476) TFUM(NUMBR), &
             PFUM(NUMBR),NBUM(NUMBR),EMINNU,EMAXNU,NNUM
        !
        DO I=1,NNUM
           WRITE(OUTU,477) I,NUMNUM(I),NORMNU(I),ELIMNU(I)
        ENDDO
     ENDIF
476  FORMAT(' HARMONIC UMBRELLA ADDED / MODIFIED'/4X, &
          'TRIG=',I4,'  POLY=',I4,',  NRES=',I4/, &
          4X,'MINE=',G10.3,'  MAXE=',G10.3,'NNUM=',I4)
477  FORMAT('    DATA SET ',I2,':',4X, &
          'NUM =',I4,'  NORM=',G10.3,',  LIMI=',G10.3)

     !-----------------------------------------------------------------------
  ELSE IF(WRD == 'DIHE') THEN
     ! set-up-dihedral-constraints
     !
     ! The dihedral umbrellas are processed here
     !
     IF(allocated(STHIUM)) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,450)
450     FORMAT('*****  WARNING  ***** ',/, &
             ' Satistic already initialized.')
        GOTO 800
     ENDIF
     NUMPHI=NUMPHI+1
     NUMBR=NUMBR+1
     TYPUM(NUMBR)=1
     IPHIUM(NUMPHI)=NUMBR
     IF(NUMBR > MAXUMP) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,460)
460     FORMAT('*****  WARNING  ***** ',/, &
             ' Too many umbrellas.')
        GOTO 800
     ENDIF
     NBUM(NUMBR)=GTRMI(COMLYN,COMLEN,'NRES',0)
     IF(NBUM(NUMBR) <= 0) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,470)
470     FORMAT(' *****  WARNING  ***** ',/, &
             ' Resolution for statistics is not', &
             ' specified.')
        GOTO 800
     ENDIF
     TFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'TRIG',0)
     PFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'POLY',0)
     IF(NBUM(NUMBR) < (PFUM(NUMBR)+2*TFUM(NUMBR))) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,480)
480     FORMAT(' *****  WARNING  ***** ',/, &
             ' More fit functions than data points.')
        GOTO 800
     ENDIF

     !C AB.
     CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,FLAGS, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     IF(NQAT /= 4) GOTO 800
     IUM(NUMPHI)=QAT(1)
     I=IUM(NUMPHI)
     IF(I <= 0.OR.I > NATOMT) GOTO 800
     CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
     JUM(NUMPHI)=QAT(2)
     I=JUM(NUMPHI)
     IF(I <= 0.OR.I > NATOMT) GOTO 800
     CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
     KUM(NUMPHI)=QAT(3)
     I=KUM(NUMPHI)
     IF(I <= 0.OR.I > NATOMT) GOTO 800
     CALL ATOMID(I,SIDK,RIDK,RENK,ACK)
     LUM(NUMPHI)=QAT(4)
     I=LUM(NUMPHI)
     IF(I <= 0.OR.I > NATOMT) GOTO 800
     CALL ATOMID(I,SIDL,RIDL,RENL,ACL)
     !
     ! APH 3/24/2014: DOMDEC initialization moved to domdec/domdec_cons.src
     !
#if KEY_DOMDEC==1
     call domdec_system_changed()
#endif
     IF(PRNLEV >= 2) WRITE(OUTU,490) NUMBR, &
          RIDI(1:idleng),SIDI(1:idleng),ACI(1:idleng), &
          RIDJ(1:idleng),SIDJ(1:idleng),ACJ(1:idleng), &
          RIDK(1:idleng),SIDK(1:idleng),ACK(1:idleng), &
          RIDL(1:idleng),SIDL(1:idleng),ACL(1:idleng), &
          TFUM(NUMBR),PFUM(NUMBR),NBUM(NUMBR)
490  FORMAT(' NEW DIHEDRAL UMBRELLA ADDED'/I5,':',4X, &
          3(A,1X,A,1X,A,'/  '),(A,1X,A,1X,A),/ &
          ,6X,'TRIG=',I4,'  POLY=',I4,',  NRES=',I4)
#if KEY_ADUMBRXNCOR==1
!-----------------------------------------------------------------------
  ELSE IF(WRD.EQ.'RXNC') THEN
     IF(allocated(STHIUM)) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,451)
451     FORMAT('*****  WARNING  ***** ',/, &
             ' Satistic already initialized.')
        GOTO 800
     ENDIF
     numbrxn=numbrxn+1
     NUMBR=NUMBR+1
     TYPUM(NUMBR)=6
     umbindex(numbrxn)=NUMBR
     IF(NUMBR > MAXUMP) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,461)
461     FORMAT('*****  WARNING  ***** ',/, &
             ' Too many umbrellas.')
        GOTO 800
     ENDIF
     NBUM(NUMBR)=GTRMI(COMLYN,COMLEN,'NRES',0)
     IF(NBUM(NUMBR) <= 0) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,471)
471     FORMAT(' *****  WARNING  ***** ',/, &
             ' Resolution for statistics is not', &
             ' specified.')
        GOTO 800
     ENDIF
     TFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'TRIG',0)
     PFUM(NUMBR)=GTRMI(COMLYN,COMLEN,'POLY',0)
     IF(NBUM(NUMBR) < (PFUM(NUMBR)+2*TFUM(NUMBR))) THEN
        IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,481)
481     FORMAT(' *****  WARNING  ***** ',/, &
             ' More fit functions than data points.')
        GOTO 800
     ENDIF
! we need a minimum and maximum
        umbmin(numbrxn)=gtrmf(comlyn,comlen,'MIN',zero)
        umbmax(numbrxn)=gtrmf(comlyn,comlen,'MAX',zero)
! the following is taken from the source for the RXNCOR UMBRELLA command (source/rxncor/cmds.src)
! The rxncor system stores a separate list of numbrxn reactin coordinates, each identified as nodes TREELO(I) to TREEHI(I) in the tree.
! This finds which tree, not the actual node number.
       CALL GTRMWD(COMLYN,COMLEN,'NAME',4,RXNNAM,80,IR)
       IF (IR  ==  0) THEN
          CALL WRNDIE (2, '<umban2>', &
               'NO NAME SPECIFIED:  DEFAULTING TO FIRST COORDINATE')
          IR = 1
       ELSE
          IR = INDGRF(RXNNAM,1,numbrxn,NODNAM,1,TREELO)
          !         Error handling taken care of in INDGRF.
          IF (IR  ==  0) IR = 1
       ENDIF

        rxncorindex(numbrxn)=ir

        if (prnlev.ge.2) write(outu,'("new reaction coordinate umbrella specified: name ",A,&
           " index ",I4," trig ",I4," poly ",I4," nres ",I4," min ",F8.3," max ",F8.3)') &
           rxnnam, ir, tfum(numbr), pfum(numbr), nbum(numbr),umbmin(numbrxn),umbmax(numbrxn)
#endif 
     !C-----------------------------------------------------------------------
     ! set up variables to be correlated with umbrella coordinates
     !      RJPetrella,  Jan 2002
     !C-----------------------------------------------------------------------
  ELSE IF(WRD == 'CORR') THEN   !RJP
     LCORRV = .TRUE.
     LCNEED = .TRUE.  !memory flag
     ! ******************************************************************
     ! Correlated Distances
     ! ******************************************************************
     IF(INDXA(COMLYN,COMLEN,'DIST') > 0) THEN
        NDISTA = NDISTA + 1
        I = GTRMI(COMLYN,COMLEN,'UNIT',-1)
        IF (NDISTA > 1) THEN
           IF((I /= -1).AND.(.NOT.LPRCRU)) THEN
              CALL WRNDIE(-4,'<UMBAN2>', &
                   'SPCIFY UNIT #s FOR ALL RMSD DIST OR NONE')
           ELSE IF ((I == -1).AND.(LPRCRU)) THEN
              CALL WRNDIE(-4,'<UMBAN2>', &
                   'UNIT NUMBER MISSING FOR DISTANCE OUTPUT')
           ELSE IF (I /= -1) THEN
              IF((I <= 0).OR.(I == 6).OR.(I.EQ.5)) &
                   CALL WRNDIE(-4,'<UMBAN2>', &
                   'BAD FORTRAN UNIT NUMBER (5 or 6)')
              LPRCRU = .TRUE.
              CDISUN(NDISTA) = I
           ENDIF
        ELSE !NDISTA = 1
           IF (I /= -1) THEN
              IF((I <= 0).OR.(I == 6).OR.(I.EQ.5)) &
                   CALL WRNDIE(-4,'<UMBAN2>', &
                   'BAD FORTRAN UNIT NUMBER (5 or 6)')
              LPRCRU = .TRUE.
              CDISUN(NDISTA) = I
           ENDIF
        ENDIF

        !     Parse the double atom selection
        call chmalloc("cmds.src","umban2","islct",natom,intg=islct)
        call chmalloc("cmds.src","umban2","jslct",natom,intg=jslct)

        CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT, &
             X,Y,Z,WT,.TRUE.,ERR)
        !     create the i and j atom lists for distance calculations
        
        CALL CORDISLST(ISLCT,JSLCT)
        
        call chmdealloc("cmds.src","umban2","islct",natom,intg=islct)
        call chmdealloc("cmds.src","umban2","jslct",natom,intg=jslct)

        IF(PRNLEV > 2)THEN
           WRITE(OUTU,'(2x,A32,1x,I4,1x,A5,1x,I8,1x,A3,1x,I8)') &
                'Write corr dist stats for dist#',NDISTA,'atoms', &
                DISIAT(NDISTA),'and',DISJAT(NDISTA)
           IF (LPRCRU) THEN
              WRITE(OUTU,'(4x,A14,1x,I6)') &
                   'on unit number',CDISUN(NDISTA)
           ENDIF
        ENDIF
        ! ******************************************************************
        ! Correlated RMSD's
        ! ******************************************************************
     ELSE IF(INDXA(COMLYN,COMLEN,'RMSD') > 0) THEN
        SYMM = 1
        IF (INDXA(COMLYN,COMLEN,'COR1') > 0) THEN
           IF(LCCOR1) THEN
              CALL WRNDIE(-5,'<UMBAN2>', &
                   'WARNING: RMSD REF SET 1 COORDS OVERWRITTEN')
           ENDIF

           call chmalloc('cmds.src','UMBAN2','HPCO1X',NATOM,crl=HPCO1X)

           call chmalloc('cmds.src','UMBAN2','HPCO1Y',NATOM,crl=HPCO1Y)

           call chmalloc('cmds.src','UMBAN2','HPCO1Z',NATOM,crl=HPCO1Z)

           call MKCOOR1(HPCO1X,HPCO1Y,HPCO1Z)

           IF(PRNLEV > 2) WRITE(OUTU,'(A42)') &
                'coordnates copied to rmsd reference set 1'
           LCCOR1 = .TRUE.
        ELSE IF (INDXA(COMLYN,COMLEN,'COR2') > 0) THEN
           IF (LCCOR2) THEN
              CALL WRNDIE(-5,'<UMBAN2>', &
                   'WARNING: RMSD REF SET 1 COORDS OVERWRITTEN')
           ENDIF

           call chmalloc('cmds.src','UMBAN2','HPCO2X',NATOM,crl=HPCO2X)

           call chmalloc('cmds.src','UMBAN2','HPCO2Y',NATOM,crl=HPCO2Y)

           call chmalloc('cmds.src','UMBAN2','HPCO2Z',NATOM,crl=HPCO2Z)

           call MKCOOR2(HPCO2X,HPCO2Y,HPCO2Z)

           IF(PRNLEV > 2) WRITE(OUTU,'(A42)') &
                'coordnates copied to rmsd reference set 2'
           LCCOR2 = .TRUE.
        ELSE IF(INDXA(COMLYN,COMLEN,'SETU') > 0) THEN
           RMSMEM = GTRMI(COMLYN,COMLEN,'NATM',-1)
           IF ((RMSMEM /= -1).AND.(RMSMMS.NE.-1)) THEN
              CALL WRNDIE(-5,'<UMBAN2>', &
                   'NATM DEFINED MORE THAN ONCE')
           ELSE
              IF (RMSMEM > RMSMMS) THEN
                 RMSMMS = RMSMEM + 1000
              ENDIF

              call chmalloc('cmds.src','UMBAN2','HPRMLS',RMSMMS,intg=HPRMLS)

              call chmalloc('cmds.src','UMBAN2','HPORLS',RMSMMS,intg=HPORLS)

              call chmalloc('cmds.src','UMBAN2','HPSYLS',RMSMMS,intg=HPSYLS)

              IF(PRNLEV > 2)WRITE(OUTU,'(A17,1x,I6,1x,A16)') &
                   'allocating memory for',RMSMMS-1000,'rmsd atoms'
           ENDIF
           RMSSTR = GTRMI(COMLYN,COMLEN,'NSTR',-1)
           IF ((RMSTRS /= -1).AND.(RMSSTR.NE.-1)) THEN
              CALL WRNDIE(-5,'<UMBAN2>', &
                   'NSTR DEFINED MORE THAN ONCE')
              ! note: initial val of RMSMMS should be set to -1
           ELSE
              IF (RMSSTR > RMSTRS) THEN
                 RMSTRS = RMSSTR + 10
              ENDIF

              call chmalloc('cmds.src','UMBAN2','HPRMEN',RMSTRS,intg=HPRMEN)

              call chmalloc('cmds.src','UMBAN2','HPOREN',RMSTRS,intg=HPOREN)

              call chmalloc('cmds.src','UMBAN2','HPSYEN',RMSTRS,intg=HPSYEN)

              IF(PRNLEV > 2)WRITE(OUTU,'(A21,1x,I6,1x,A16)') &
                   'allocating memory for',RMSTRS-10,'rmsd structures'

              call chmalloc('cmds.src','UMBAN2','HPSYDL',RMSTRS,intg=HPSYDL)

              call chmalloc('cmds.src','UMBAN2','HPSYMN',RMSTRS,intg=HPSYMN)

              call chmalloc('cmds.src','UMBAN2','HPLORI',RMSTRS,log=HPLORI)

              call chmalloc('cmds.src','UMBAN2','HPMXIC',RMSTRS,intg=HPMXIC)

              call chmalloc('cmds.src','UMBAN2','HPMNIC',RMSTRS,intg=HPMNIC)

              ! initialize symmetry dihedral atom arrays
           ENDIF
        ELSE IF(INDXA(COMLYN,COMLEN,'SUBS') > 0) THEN !rmsd substructure entry
           IF (.NOT.LCCOR1) &
                CALL WRNDIE(-5,'<UMBAN2>', &
                'CORRELATED COOR SET 1 NOT SPECIFIED')
           IF (.NOT.LCCOR2) &
                CALL WRNDIE(-5,'<UMBAN2>', &
                'CORRELATED COOR SET 2 NOT SPECIFIED')
           IF (RMSMMS == -1) &
                CALL WRNDIE(-5,'<UMBAN2>', &
                'NATM NOT SPECFD --APPROX # ATM IN RMSD CALC')
           NRMSDI = NRMSDI + 1
           IF (NRMSDI > RMSTRS) &
                CALL WRNDIE(-5,'<UMBAN2>', &
                'NUMBR OF RMSD SUBSTRCTRES > MEM ALLOCATION')
           I1 = GTRMI(COMLYN,COMLEN,'UNT1',-1)
           I2 = GTRMI(COMLYN,COMLEN,'UNT2',-1)
           IF (NRMSDI > 1) THEN
              IF(((I1 /= -1).OR.(I2.NE.-1)).AND.(.NOT.LPRRMU)) &
                   THEN
                 CALL WRNDIE(-4,'<UMBAN2>', &
                      'SPCIFY UNIT #s FOR ALL RMSD CORR OR NONE')
              ELSE IF (((I1 == -1).OR.(I2.EQ.-1)).AND. &
                   (LPRRMU)) THEN
                 CALL WRNDIE(-4,'<UMBAN2>', &
                      'UNIT NUMBER MISSING FOR DISTANCE OUTPUT')
              ELSE IF ((I1 /= -1).AND.(I2.NE.-1)) THEN
                 IF((I1 <= 0).OR.(I1 == 6).OR.(I1.EQ.5) &
                      .OR.(I2 <= 0).OR.(I2 == 6).OR.(I2.EQ.5)) &
                      CALL WRNDIE(-4,'<UMBAN2>', &
                      'BAD FORTRAN UNIT NUMBER (5 or 6)')
                 LPRRMU = .TRUE.
                 CRMSU1(NRMSDI) = I1
                 CRMSU2(NRMSDI) = I2
              ENDIF
           ELSE !NRSMDI = 1
              IF ((I1 /= -1).OR.(I2.NE.-1)) THEN
                 IF((I1 <= 0).OR.(I1 == 6).OR.(I1.EQ.5).OR. &
                      (I2 <= 0).OR.(I2 == 6).OR.(I2.EQ.5)) &
                      CALL WRNDIE(-4,'<UMBAN2>', &
                      'BAD FORTRAN UNIT NUMBER (5 or 6)')
                 LPRRMU = .TRUE.
                 CRMSU1(NRMSDI) = I1
                 CRMSU2(NRMSDI) = I2
              ENDIF
           ENDIF
           !  parse orientation option
           ORIENT = .FALSE.
           IF(INDXA(COMLYN,COMLEN,'ORIE') > 0) &
                ORIENT = .TRUE.
           !     Parse the double atom selection

           call chmalloc("cmds.src","umban2","islct",natom,intg=islct)
           call chmalloc("cmds.src","umban2","jslct",natom,intg=jslct)
           !     create the i and j atom lists for rmsd calculations

           CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT, &
                X,Y,Z,WT,.TRUE.,ERR)
           
           call CORRMSDLST(ISLCT,JSLCT,NRMSAT,NORIAT,HPRMLS,HPRMEN, &
                HPLORI,ORIENT,HPORLS,HPOREN)

           call chmdealloc("cmds.src","umban2","islct",natom,intg=islct)
           call chmdealloc("cmds.src","umban2","jslct",natom,intg=jslct)

           IF(PRNLEV > 2)THEN
              WRITE(OUTU,'(2x,A38,1x,I8)') &
                   'Will write 1st correlated rmsd for set',NRMSDI
              IF (LPRRMU) THEN
                 WRITE(OUTU,'(4x,A14,1x,I6)') &
                      'on unit number',CRMSU1(NRMSDI)
              ENDIF
              WRITE(OUTU,'(2x,A38,1x,I8)') &
                   'Will write 2nd correlated rmsd for set',NRMSDI
              IF (LPRRMU) THEN
                 WRITE(OUTU,'(4x,A14,1x,I6)') &
                      'on unit number',CRMSU2(NRMSDI)
              ENDIF
           ENDIF
           ! parse the atom selection for symmetries
           call chmalloc("cmds.src","umban2","islct",natom,intg=islct)
           ! note that mode=0 and dflt=0 below
           CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,0, &
                0,.FALSE.,1,' ',0, RESID,RES,IBASE,SEGID,NICTOT, &
                NSEG,.TRUE.,X,Y,Z,.TRUE.,1,WT)

           call SYMATMLST(ISLCT,NRMSDI,HPSYLS,HPSYEN,NSYMA1,NATOM, &
                HPMNIC,HPMXIC)

           call chmdealloc("cmds.src","umban2","islct",natom,intg=islct)

           CALL SYMDIHE(COMLYN,COMLEN,NRMSDI,HPSYDL, &
                icr_struct%LENIC, &
                icr_struct%PIC, icr_struct%IAR, &
                icr_struct%JAR, icr_struct%KAR, &
                icr_struct%LAR, icr_struct%TAR, &
                HPSYMN,SYMM)
           IF((SYMM > 1).AND.(NRMSAT.GT.(NATOM-3))) &
                CALL WRNDIE(-4,'<UMBAN2>', &
                'RMSD ATOM SELE TOO LARGE FOR SYMMETRY OP')
           IF(ORIENT) THEN
              WRDD = ' '
           ELSE
              WRDD = 'OUT'
           ENDIF
           IF(PRNLEV > 2)THEN
              WRITE(OUTU,'(A6,1X,I4,1X,A16,1X,I6,1X,A5)') &
                   'RMSD #',NRMSDI,'WILL BE DONE FOR',NRMSAT, &
                   'ATOMS'
              WRITE(OUTU,'(A4,A3,1X,A26,1X,I3)') &
                   'WITH',WRDD,'REORIENTATION.  SYMMETRY =',SYMM
              IF(SYMM >= 2) WRITE(OUTU,'(I6,A30)') &
                   NSYMA1,'ATOMS INVOLVED IN SYMMETRY OP'
              IF (ORIENT) THEN
                 WRITE(OUTU,'(A38,1X,I6,1X,A6)') &
                      'REORIENTATION WILL BE DONE RELATIVE TO',NORIAT, &
                      'ATOMS'
              ENDIF
           ENDIF
        ENDIF !if memory or substructure
     ENDIF  !if 'DIST' or 'RMSD'
     !
     !-----------------------------------------------------------------------
  ELSE
     IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,500)
500  FORMAT(' *****  ERROR  ***** ', &
          ' Unknown umbrella specified.')
     CALL DIEWRN(0)
     call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
     RETURN
     !-----------------------------------------------------------------------
  ENDIF
  call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
  RETURN
  !
  !----------------------
  ! to throw out energy
700 CONTINUE
  IEUM=0
  NUMBR=NUMBR-1
  IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,430)
430 FORMAT(' ***** ERROR in UMBAN ***** ',/, &
       ' Some component of the energy umbrella', &
       ' were invalid.  It was ignored.')
  CALL DIEWRN(0)
  call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
  RETURN
  !
  !----------------------
  ! to throw out NOE umbrella
750 CONTINUE
  INUM=0
  NUMBR=NUMBR-1
  IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,435)
435 FORMAT(' ***** ERROR in UMBAN ***** ',/, &
       ' Some component of the NOE umbrella', &
       ' were invalid.  It was ignored.')
  CALL DIEWRN(0)
  call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
  RETURN
  !
  !----------------------
  ! to throw-out-dihedral
800 CONTINUE
  NUMPHI=NUMPHI-1
  NUMBR=NUMBR-1
  IF((WRNLEV >= 2).AND.(PRNLEV > 2)) WRITE(OUTU,440)
440 FORMAT(' ***** ERROR in UMBAN ***** ',/, &
       ' Some component of a dihedral umbrella', &
       ' was invalid.  It was ignored.')
  CALL DIEWRN(0)
  call chmdealloc("cmds.src","umban2","flags",natom,intg=flags)
  RETURN
  !
END SUBROUTINE UMBAN2
!
!
!-------------------------------------------------------------
SUBROUTINE SYMATMLST(ISLCT,NTHRMS,SYMLST, &
     SYMEND,NSYMA1,NATOMX,MINIC,MAXIC)
  !   fills the symmetry-atom arrays
  use bases_fcm
  use chm_kinds
  use intcor_module
  use intcor2,only:findic
  use umbcor
  use stream
  !
  implicit none
  INTEGER ISLCT(*),NTHRMS,SYMLST(*),SYMEND(*)
  INTEGER NSYMA1,NATOMX,MINIC(*),MAXIC(*)
  INTEGER I,FIRSIC,LASTIC
  !
  NSYMA1 = 0
  MINIC(NTHRMS) = 999999999
  MAXIC(NTHRMS) = -1
  DO I = 1,NATOMX
     IF (ISLCT(I) > 0) THEN
        NSYMAT = NSYMAT + 1
        NSYMA1 = NSYMA1 + 1
        SYMLST(NSYMAT) = I
        IF(NSYMAT > RMSMMS) THEN
           CALL WRNDIE(-4,'<SYMATMLST>', &
                'TOO MANY SYMMETRY ATOMS--INCREASE MEMORY')
        ENDIF
        CALL FINDIC(1,icr_struct%LENIC,icr_struct%IAR, &
             icr_struct%JAR,icr_struct%KAR, &
             icr_struct%LAR,I,FIRSIC,LASTIC)
        IF (FIRSIC < MINIC(NTHRMS)) &
             MINIC(NTHRMS) = FIRSIC
        IF (LASTIC > MAXIC(NTHRMS)) &
             MAXIC(NTHRMS) = LASTIC
     ENDIF
  ENDDO
  IF (NSYMAT <= 0) THEN
     IF(PRNLEV > 2)THEN
        WRITE(OUTU,'(A40)') &
             'NO ATOMS SELECTED FOR SYMMETRY OP OF RMSD'
        WRITE(OUTU,'(A12,I8)') &
             'SUBSTRUCTURE',NTHRMS
     ENDIF
     CALL WRNDIE(-5,'<SYMATMLST>', &
          'NO ATOMS SELECTED FOR SYMMETRY OP')
  ENDIF
  SYMEND(NTHRMS) = NSYMAT
  RETURN
END SUBROUTINE SYMATMLST
!
! -------------------------------------------------------------
SUBROUTINE CORDISLST(ISLCT,JSLCT)
  !
  !   create the atom lists necessary for correlated distance
  !   calculations  -RJP
  !
  !
  use dimens_fcm
  use chm_kinds
  use umbcor
  use psf
  implicit none
  !
  ! passed variables
  INTEGER ISLCT(*),JSLCT(*)
  !
  ! local variables
  INTEGER COUNTI,COUNTJ,I,J

  COUNTI = 0
  COUNTJ = 0
  DO I = 1,NATOM
     IF (ISLCT(I) > 0) THEN
        IF (COUNTI == 0) THEN
           DISIAT(NDISTA) = I
           COUNTI = COUNTI + 1
        ELSE
           CALL WRNDIE(-4,'<UMBAN2>', &
                'MORE THAN ONE ATOM CHOSEN IN 1ST SELE')
        ENDIF
     ENDIF
     IF (JSLCT(I) > 0) THEN
        IF (COUNTJ == 0) THEN
           DISJAT(NDISTA) = I
           COUNTJ = COUNTJ + 1
        ELSE
           CALL WRNDIE(-4,'<UMBAN2>', &
                'MORE THAN ONE ATOM CHOSEN IN 2ND SELE')
        ENDIF
     ENDIF
  ENDDO

  IF(COUNTI == 0) THEN
     CALL WRNDIE(-4,'<UMBAN2>', &
          'NO ATOMS CHOSEN IN 1ST SELECTION')
  ENDIF
  IF(COUNTJ == 0) THEN
     CALL WRNDIE(-4,'<UMBAN2>', &
          'NO ATOMS CHOSEN IN 2ND SELECTION')
  ENDIF

  RETURN
END SUBROUTINE CORDISLST
!-----------------------------------------------------------------------
!
SUBROUTINE MKCOOR1(CCOR1X,CCOR1Y,CCOR1Z)
  !     copy coordinates from main set to CCOOR1 set
  !
  use dimens_fcm
  use chm_kinds

  use psf
  use coord
  !
  implicit none
  real(chm_real) CCOR1X(*),CCOR1Y(*),CCOR1Z(*)
  INTEGER I
  DO I = 1,NATOM
     CCOR1X(I) = X(I)
     CCOR1Y(I) = Y(I)
     CCOR1Z(I) = Z(I)
  ENDDO
  RETURN
END SUBROUTINE MKCOOR1
!-----------------------------------------------------------------------
!
SUBROUTINE MKCOOR2(CCOR2X,CCOR2Y,CCOR2Z)
  !     copy coordinates from main set to CCOOR2 set
  !
  use dimens_fcm
  use chm_kinds
  use psf
  use coord
  !
  implicit none
  real(chm_real) CCOR2X(*),CCOR2Y(*),CCOR2Z(*)
  INTEGER I
  DO I = 1,NATOM
     CCOR2X(I) = X(I)
     CCOR2Y(I) = Y(I)
     CCOR2Z(I) = Z(I)
  ENDDO
  RETURN
END SUBROUTINE MKCOOR2
!
! ----------------------------------------------------------------------
SUBROUTINE SYMDIHE(COMLYN,COMLEN,NTHDIH, &
     SYDLIN,LENIC,PIC,IAR,JAR,KAR,LAR,TAR, &
     DISYMN,SYMM)
  use dimens_fcm
  use chm_kinds
  !  adds a dihedral angle to the symmetry dihedral angle arrays
  use psf
  use stream
  use string
  use select
  !
  implicit none
  !      INTEGER SYDIA1(*),SYDIA2(*),SYDIA3(*),SYDIA4(*)
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  INTEGER SYDLIN(*)
  INTEGER NTHDIH,LENIC,DISYMN(*)
  real(chm_real) PIC(*)
  INTEGER IAR(*),JAR(*),KAR(*),LAR(*),SYMM
  LOGICAL TAR(*)
  !
  INTEGER ISLCT(1),QAT(4),NQAT
  INTEGER I,J,K,L,IIC,STORIC
  LOGICAL FOUND,LPOS,LNEG,T
  LOGICAL DONE,EOF
  character(len=4) WRD
  !
  SYMM = GTRMI(COMLYN,COMLEN,'FOLD',1)
  DISYMN(NTHDIH) = SYMM
  DO I=1,4
     QAT(I)=0
  ENDDO
  WRD=NEXTA4(COMLYN,COMLEN)
  IF(WRD == 'SYMM' .OR. WRD.EQ.'DIHE') THEN
     CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,ISLCT, &
          SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
     I=QAT(1)
     J=QAT(2)
     K=QAT(3)
     T=(K < 0)
     IF(T) CALL WRNDIE(-5,'<SYMDIHE>', &
          'SYMMETRY-DEFINING DIHEDRALS MUST BE PROPER')
     L=QAT(4)
     !
     IF(I <= 0.OR.J.LE.0.OR.K.LE.0.OR.L.LE.0) THEN
        ! ATOM-DOESNT-EXIST
        CALL WRNDIE(-5,'<SYMDIHE>', &
             'ATOM IN DIHEDRAL ANGLE DOESNT EXIST')
     ELSE !if atoms exist
        !
        STORIC = 0
        FOUND=.FALSE.
        DO IIC=1,LENIC
           IF(I == IAR(IIC).AND.L.EQ.LAR(IIC)) THEN
              LPOS=(J == JAR(IIC).AND.K.EQ.KAR(IIC))
              LNEG=(J == KAR(IIC).AND.K.EQ.JAR(IIC))
           ELSE
              IF(I == LAR(IIC).AND.L.EQ.IAR(IIC)) THEN
                 LNEG=(J == JAR(IIC).AND.K.EQ.KAR(IIC))
                 LPOS=(J == KAR(IIC).AND.K.EQ.JAR(IIC))
              ELSE
                 LNEG=.FALSE.
                 LPOS=.FALSE.
              ENDIF
           ENDIF
           !
           IF(LNEG) THEN
              IF(PRNLEV >= 2) WRITE(OUTU,10) IIC
10            FORMAT(15X,'FOUND DIHEDRAL IN IC',I5,' AS OPPOSITE')
              IF(T.NEQV.TAR(IIC).AND.WRNLEV >= 2) WRITE(OUTU,20)
20            FORMAT(20X,'BUT TYPE VALUES DONT MATCH')
              IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
              IF (STORIC /= 0) THEN
                 CALL WRNDIE(-5,'<SYMDIHE>', &
                      'DIHEDRAL ANGLE APPEARS > ONCE IN IC TABLE')
              ELSE
                 STORIC = IIC
              ENDIF
           ENDIF
           IF(LPOS) THEN
              IF(PRNLEV >= 2) WRITE(OUTU,30) IIC
30            FORMAT(15X,'FOUND DIHEDRAL IN IC',I5,' AS POSITIVE')
              IF(T.NEQV.TAR(IIC) .AND. WRNLEV >= 2.AND.PRNLEV.GE.2) &
                   WRITE(OUTU,20)
              IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
              IF (STORIC /= 0) THEN
                 CALL WRNDIE(-5,'<SYMDIHE>', &
                      'DIHEDRAL ANGLE APPEARS > ONCE IN IC TABLE')
              ELSE
                 STORIC = IIC
              ENDIF
           ENDIF
        ENDDO
        IF(.NOT.FOUND) THEN
           IF(PRNLEV >= 2) THEN
              WRITE(OUTU,'(A10,1x,I6,1x,I6,1x,I6,1x,I6,A10)') &
                   'DIHEDRAL ',I,J,K,L,'NOT FOUND'
           ENDIF
           CALL WRNDIE(-5,'<SYMDIHE>', &
                'SYMMETRY-DEFINING DIHEDRAL NOT FOUND')
        ENDIF
        SYDLIN(NTHDIH) = STORIC
        DONE =.TRUE.
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE SYMDIHE
!-----------------------------------------------------------------------
SUBROUTINE CORRMSDLST(ISLCT,JSLCT,NRMSAT,NORIAT, &
     RMSLST,RMAEND,LORIEN, &
     ORIENT,ORILST,ORIEND)
  !
  !    Store the atoms for rmsd calculations in a common
  !    array; fill orientation (T/F) array
  !
  use bases_fcm
  use dimens_fcm
  use chm_kinds
  use psf
  use umbcor
  !
  implicit none
  !
  INTEGER ISLCT(*),JSLCT(*),RMSLST(*),RMAEND(*)   !selection and rmsd atom list
  INTEGER ORILST(*),ORIEND(*)
  LOGICAL LORIEN(*),ORIENT
  INTEGER NRMSAT,NORIAT
  !
  INTEGER I,LOCCNT
  !
  NRMSAT = 0
  NORIAT = 0
  DO I = 1,NATOM
     IF (ISLCT(I) > 0) THEN
        NRMSAT = NRMSAT + 1  !count for this substructure
        RMACNT = RMACNT + 1  !count over all substructures (in common)
        IF(RMACNT > RMSMMS) &
             CALL WRNDIE(-4,'<CORRMSDLST>', &
             'RMSD SELECTIONS EXCEED ALLOCATED MEMORY')
        RMSLST(RMACNT) = I
     ENDIF
     ! process orientation selection
     IF(ORIENT) THEN
        IF(JSLCT(I) > 0) THEN
           NORIAT = NORIAT + 1  !count over this substructure
           ORATCT = ORATCT + 1  !count over all substructures
           ORILST(ORATCT) = I
        ENDIF
     ENDIF
  ENDDO
  RMAEND(NRMSDI) = RMACNT
  LORIEN(NRMSDI) = ORIENT
  ORIEND(NRMSDI) = ORATCT
  !  record whether # atoms exceeds maximum
  LOCCNT = NORIAT + NRMSAT
  IF(LOCCNT > MXRMSA) THEN
     MXRMSA = LOCCNT
  ENDIF
  RETURN
END SUBROUTINE CORRMSDLST

SUBROUTINE NUMFIL(IPT,JPT,INM,JNM,LIS, &
     RMN,KMN,RMX,KMX,FMX,TCN,AVE,MINF,EXP &
     ,RSW,SEX,RAM)
  !JH (soft asymptote)
  !
  !     Copy NOE data structures to NOE-umbrella data structures
  !
  use dimens_fcm
  use chm_kinds
  use noem
  use stream
  use umb
  implicit none
  ! . Passed variables.
  INTEGER IPT(*),JPT(*),INM(*),JNM(*),LIS(*)
  real(chm_real)  RMN(*),KMN(*),RMX(*),KMX(*),FMX(*),TCN(*),AVE(*), &
       EXP(*)
  LOGICAL MINF(*)
  !JH
  real(chm_real)  RSW(*),SEX(*)
  INTEGER RAM(*)
  ! . Local variables
  INTEGER I
  !
  NORMNU(NNUM)=0.0
  DO I=1,NOENUM
     IPT(I)=NOEIPT(I)
     JPT(I)=NOEJPT(I)
     JNM(I)=NOEJNM(I)
     INM(I)=NOEINM(I)
     RMN(I)=NOERMN(I)
     KMN(I)=NOEKMN(I)
     RMX(I)=NOERMX(I)
     KMX(I)=NOEKMX(I)
     FMX(I)=NOEFMX(I)
     TCN(I)=NOETCN(I)
     AVE(I)=NOEAVE(I)
     EXP(I)=NOEEXP(I)
     MINF(I)=NOEMIN(I)
     !JH (soft asymptote)
     RSW(I)=NOERSW(I)
     SEX(I)=NOESEX(I)
     RAM(I)=NOERAM(I)
     IF(PRNLEV > 2) WRITE(OUTU,*) &
          I,IPT(I),JPT(I),KMN(I),RMN(I),KMX(I),RMX(I)
     !
     NORMNU(NNUM)=NORMNU(NNUM)+KMN(I)+KMX(I)
  ENDDO
  DO I=1,NOEMAX
     LIS(I)=NOELIS(I)
  ENDDO
  !
  return
end SUBROUTINE NUMFIL

#endif /* (adumb)*/

SUBROUTINE NULL_CMDS
  RETURN
END SUBROUTINE NULL_CMDS

