#if KEY_EMAP==1 /*emap*/
SUBROUTINE EMAPDOCK(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This routine parses docking options
  !
  !     By XIONGWU WU  JULY 17, 2001
  !          Enhanced      Dec, 2004
  !
  use chm_kinds
  !
#if KEY_EMAP==1
  use dimens_fcm
  use number
  !
  !
  use stream
  use string
  use energym
  use emapmod
#endif 
  implicit none
  !
  INTEGER COMLEN
  CHARACTER(LEN=*) COMLYN
  CHARACTER(LEN=80) NAME
  INTEGER LENGTH
  INTEGER IDEMP,NRIG,IDRIGS(MXNRIG),IDEMP1,IDRIG
  INTEGER I,IRIG,ntgrid,nrgrid,NLOOP
  INTEGER NCYC,NSTEP,NDATA,NDATAS,IDEMPS
  REAL(CHM_REAL) DELTT,DELTR,RATIOT,RATIOR,TEMP,TEMPF,CFIX,DTCO
  REAL(CHM_REAL) CORR0,CORRO,CORRMAX,CORRT,CORROIJ,CORRTIJ,COTOENG
  INTEGER IT,JT,KT
  LOGICAL NEWEMP,NEWRIG,LMANY,LGTMC,LCORE,LDDR,LFFT,LFMAP
  LOGICAL LFIXED(MXNRIG),LGTVOL
  REAL(CHM_REAL) SITE1(3),SANGLE1,SITE2(3),SANGLE2
  REAL(CHM_REAL) REX1,REX2,REY1,REY2,REZ1,REZ2
  REAL(CHM_REAL) LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2
  !  Select a docking method
  LMANY=INDXA(COMLYN,COMLEN,'MANY')  >  0
  LPBC=INDXA(COMLYN,COMLEN,'PBC')  >  0
  LFFT=INDXA(COMLYN,COMLEN,' FFT')  >  0
  LFMAP=INDXA(COMLYN,COMLEN,'FMAP')  >  0
  LGTVOL=INDXA(COMLYN,COMLEN,'SPACE')  >  0
  LGTMC=INDXA(COMLYN,COMLEN,'GTMC')  >  0
  LDDR=INDXA(COMLYN,COMLEN,'DDR')  >  0
  LCORE=INDXA(COMLYN,COMLEN,'CORE')  >  0
  !  Define target map object
  CALL GTRMWA(COMLYN,COMLEN,'MAPI',4,NAME,80,LENGTH)
  IF(LENGTH <= 0.AND..NOT.LFMAP)THEN
     CALL WRNDIE(0,'<EMAPOPT>','NO EMAP NAME SPECIFIED!')
     RETURN
  ENDIF
  NEWEMP=.FALSE.
  IDEMP=-1
  IF(LENGTH > 0)NEWEMP=CHKEMPNM(NAME,LENGTH,IDEMP)
  IF(NEWEMP)THEN
     IF(LFMAP.AND.LMANY)THEN
        IDEMP=-1
     ELSE
        CALL WRNDIE(0,'<EMAP>','NO EMAP:'//NAME(1:LENGTH))
     ENDIF
  ENDIF
  !  Read in rigid domains to be fitted or take from COMPLex definition
  NRIG=0

  CALL GTRMWA(COMLYN,COMLEN,'RIGI',4,NAME,80,LENGTH)
  do while(LENGTH > 0)
     NRIG=NRIG+1
     NEWRIG=CHKRIGNM(NAME,LENGTH,IDRIG)
     IF(NEWRIG)THEN
        CALL WRNDIE(0,'<EMAP>','NO rigid ID:'//NAME(1:LENGTH))
     ENDIF
     IDRIGS(NRIG)=IDRIG
     LFIXED(NRIG)=.FALSE.
     !GOTO 300
     CALL GTRMWA(COMLYN,COMLEN,'RIGI',4,NAME,80,LENGTH)
  enddo

  IF(NRIG == 0)THEN
     NRIG=NEMCOMP
     DO I=1,NEMCOMP
        IDRIGS(I)=IDCOMPS(I)
        LFIXED(I)=LCOMPFIX(I)
     ENDDO
  ENDIF
  !  GRID parameters
  NTGRID=FOUR
  NRGRID=FOUR
  NTGRID=GTRMI(COMLYN,COMLEN,'NTRA',NTGRID)
  NRGRID=GTRMI(COMLYN,COMLEN,'NROT',NRGRID)
  SITE1(1)=GTRMF(COMLYN,COMLEN,'SX1',ZERO)
  SITE1(2)=GTRMF(COMLYN,COMLEN,'SY1',ZERO)
  SITE1(3)=GTRMF(COMLYN,COMLEN,'SZ1',ONE)
  SANGLE1=GTRMF(COMLYN,COMLEN,'ANG1',ONE8TY)
  SITE2(1)=GTRMF(COMLYN,COMLEN,'SX2',ZERO)
  SITE2(2)=GTRMF(COMLYN,COMLEN,'SY2',ZERO)
  SITE2(3)=GTRMF(COMLYN,COMLEN,'SZ2',ONE)
  SANGLE2=GTRMF(COMLYN,COMLEN,'ANG2',ONE8TY)
  REX1=GTRMF(COMLYN,COMLEN,'REX1',-RBIG)
  REX2=GTRMF(COMLYN,COMLEN,'REX2',RBIG)
  REY1=GTRMF(COMLYN,COMLEN,'REY1',-RBIG)
  REY2=GTRMF(COMLYN,COMLEN,'REY2',RBIG)
  REZ1=GTRMF(COMLYN,COMLEN,'REZ1',-RBIG)
  REZ2=GTRMF(COMLYN,COMLEN,'REZ2',RBIG)
  LGX1=GTRMF(COMLYN,COMLEN,'LGX1',-RBIG)
  LGX2=GTRMF(COMLYN,COMLEN,'LGX2',RBIG)
  LGY1=GTRMF(COMLYN,COMLEN,'LGY1',-RBIG)
  LGY2=GTRMF(COMLYN,COMLEN,'LGY2',RBIG)
  LGZ1=GTRMF(COMLYN,COMLEN,'LGZ1',-RBIG)
  LGZ2=GTRMF(COMLYN,COMLEN,'LGZ2',RBIG)
  PBCX=GTRMF(COMLYN,COMLEN,'PBCX',RBIG)
  PBCY=GTRMF(COMLYN,COMLEN,'PBCY',RBIG)
  PBCZ=GTRMF(COMLYN,COMLEN,'PBCZ',RBIG)
  !   MONTE CARLO parameters
  NLOOP=1
  NCYC=10
  NSTEP=1000
  IF(IDEMP > 0)THEN
     DELTT=(DEMAPX(IDEMP)+DEMAPY(IDEMP)+DEMAPZ(IDEMP))/THREE
  ELSE
     IDEMP1=IDEMRIG(IDRIGS(1))
     DELTT=(DEMAPX(IDEMP1)+DEMAPY(IDEMP1)+DEMAPZ(IDEMP1))/THREE
  ENDIF
  DELTR=10.0
  TEMP=0.01
  TEMPF=300.0D0
  RATIOT=HALF
  RATIOR=HALF
  CFIX=ONE
  DTCO=0.02
  NLOOP=GTRMI(COMLYN,COMLEN,'LOOP',NLOOP)
  NCYC=GTRMI(COMLYN,COMLEN,'NCYC',NCYC)
  NSTEP=GTRMI(COMLYN,COMLEN,'NSTE',NSTEP)
  IF(LFMAP)THEN
     TEMP=0.002D0*GTRMF(COMLYN,COMLEN,'TEMP',TEMPF)
  ELSE
     TEMP=GTRMF(COMLYN,COMLEN,'TEMP',TEMP)
  ENDIF
  DELTT=GTRMF(COMLYN,COMLEN,'TRAN',DELTT)
  DELTR=GTRMF(COMLYN,COMLEN,'ROTA',DELTR)
  RATIOT=GTRMF(COMLYN,COMLEN,'TRAT',RATIOT)
  RATIOR=GTRMF(COMLYN,COMLEN,'RRAT',RATIOR)
  BCORE=GTRMF(COMLYN,COMLEN,'BCOR',BCORE)
  CFIX=GTRMF(COMLYN,COMLEN,'CFIX',CFIX)
  DTCO=GTRMF(COMLYN,COMLEN,'DTCO',DTCO)
  COTOENG=TEMP/DTCO
  COTOENG=GTRMF(COMLYN,COMLEN,'CTOE',COTOENG)
  IEMPTRJ=GTRMI(COMLYN,COMLEN,'TRAJ',-1)
  IF (IEMPTRJ > 0) THEN
     IF(UEMPTRJ <= 0)THEN
        CALL WRNDIE(-5,'<EMAPOPT>', &
             'NO TRAJECTORY UNIT SPECIFIED!')
        RETURN
     ENDIF
  ENDIF
  ! print dock options
  IF(LPBC)THEN
    WRITE(OUTU,1101)PBCX,PBCY,PBCZ
  ENDIF
1101 FORMAT(" PERIODIC BOUNDARY X, Y, Z: ", 3F10.2)
  !  Perform docking
  IF(LMANY)THEN
     IF(LFMAP)THEN
        IF(LGTMC)THEN
           CALL FMAPGTMCN(IDEMP,NRIG,IDRIGS,LFIXED,NLOOP, &
                NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR)
        ELSE
           CALL FMAPMCN(IDEMP,NRIG,IDRIGS,LFIXED,NLOOP, &
                NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
                LDDR,LCORE,COTOENG)
        ENDIF
     ELSE
        CALL EMAPGTMCN(IDEMP,NRIG,IDRIGS,LFIXED,NLOOP, &
             NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
             LDDR,LCORE)
     ENDIF
  ELSE 
     IF(LGTMC)THEN
        IF(LFMAP)THEN
           CALL FMAPGTMC1(IDEMP,NRIG,IDRIGS,LFIXED,NTGRID,NRGRID, &
                NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
                CFIX,DTCO,LGTVOL,SITE1,SANGLE1,SITE2,SANGLE2, &
                REX1,REX2,REY1,REY2,REZ1,REZ2, &
                LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2)
        ELSE
           CALL EMAPGTMC1(IDEMP,NRIG,IDRIGS,LFIXED,NTGRID,NRGRID, &
                NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
                LDDR,LCORE,CFIX,DTCO)
        ENDIF
     ELSE IF(LFFT)THEN
        !          CALL EMAPFFT(IDEMP,NRIG,IDRIGS,NTGRID,NRGRID,
        !     &             NCYC,NSTEP,TEMP,RATIOT,RATIOR,LDDR,LCORE,CFIX)
        CALL WRNDIE(0,'<EMAP>', &
             'FFT DOCKING OPTION IS NOT IMPLEMENTED YET')
     ELSE
        CALL WRNDIE(0,'<EMAP>', &
             'UNRECONGNIZED DOCKING OPTION')
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE EMAPDOCK


SUBROUTINE EMAPGTMCN(IDEMP0,NRIG,IDRIGS,LFIXED,NLOOP, &
     NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior, &
     LDDR,LCORE)
  !-----------------------------------------------------------------------
  !   This routine perform a many-body GTMC docking to an emap
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER NRIG,IDRIGS(*),IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,idrig1,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS,NLOOP,ILOOP
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORR0,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,ICOPY1,IOVER,NCOPY(MXNRIG),IDRIGCP
  real(chm_real) COCOPY(MXNRIG),CTCOPY(MXNRIG),CORROCP,CORRTCP
  INTEGER IT,JT,KT,IR,JR,KR,NCORR
  LOGICAL OVERLAP,NEWEMP,NEWRIG,LDDR,LCORE,LFIXED(MXNRIG)
  !
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIG),LFIXED(IRIG)
  ENDDO
  WRITE(OUTU,1035)NLOOP,NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
  WRITE(OUTU,1037)LDDR
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("DOMAIN ",I2,2X,A20,"FIXED:",L4)
1035 FORMAT("NLOOP=",I8," NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
1037 FORMAT(" LDDR=",L4)
  !   Create a scratch map  IDEMPS for docking
  NDATA=LEMAPX(IDEMP0)*LEMAPY(IDEMP0)*LEMAPZ(IDEMP0)
  NEWEMP=CHKEMPNM('SCRATCH',7,IDEMPS)
  IF(NEWEMP)THEN
     CALL EMAPDUP(IDEMP0,IDEMPS)
  ELSE
     CALL WRNDIE(0,'<EMAPGTMC N>','CANNOT CREATE EMAP: SCRATCH')
  ENDIF
  CORRMAX = ZERO
  !  Perform NLOOP times of Many-body docking 
  loop200: DO ILOOP=1,NLOOP
     !   At each Loop, perform a GTMC for Each  RIGID domain 
     loop300: DO IRIG=1,NRIG
        IDRIG=IDRIGS(IRIG)
        !   Skip fixed rigid domains
        IF(LFIXED(IRIG))cycle loop300
        IF(PRNLEV > 2)WRITE(OUTU,'("Fiting rigid: ",I3)')IRIG
        !   Create a map from all rigid domains except IRIG
        !    build up scratch emap from the rest of rigid domains
        CALL EMAPINIT(IDEMPS,ZERO)
        DO JRIG=1,NRIG
           IDRIGJ=IDRIGS(JRIG)
           IF(IRIG /= JRIG)CALL EMAPADD(IDEMPS,IDRIGJ)
        ENDDO
        CALL EMAPSTAT(IDEMPS)
        !  Perform GTMC dock for IRIG with IDEMPS and IDEMP0 to find the best fits
        CALL EMAPGRIDN(IDEMP0,IDEMPS,IDRIG,NTGRID,NRGRID, &
             NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior,LDDR,LCORE,CORRMAX)
        NCORR=0
        CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIG,LDDR,LCORE,CORRO,NCORR)
        CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
        CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
        IF(PRNLEV > 2)WRITE(OUTU,1050)ILOOP,IDRIG,CORRO,CORRT
     enddo loop300
  enddo loop200
  IF(PRNLEV > 2) &
       WRITE(OUTU,'("DOWN WITH FITTING")')
  CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
  CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
  IF(PRNLEV > 4)WRITE(OUTU,1003)CORRMAX
  CALL RMEMAP(IDEMPS)
1003 FORMAT("AFTER GTMC FITTING, CORRMAX=",F10.6)
1050 FORMAT(" At ILOOP=",I4," RIGID ",I3,"  CORRT=",2F10.6)
  RETURN
END SUBROUTINE EMAPGTMCN


SUBROUTINE EMAPGTMC1(IDEMP0,NRIG,IDRIGS,LFIXED, &
     NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior, &
     LDDR,LCORE,CFIX,DELTCO)
  !-----------------------------------------------------------------------
  !   This routine perform a series single-body GTMC fitting to a emap
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use string
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER NRIG,IDRIGS(*),IDRIGSS,NFIXED
  INTEGER IDRIGN,IDLIST(MXNRIG),NRIGFIT,IDRIGSFIT(MXNRIG)
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS,LENGTH
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORR0,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,ICOPY1,IOVER,NCOPY(MXNRIG),IDRIGCP
  real(chm_real) CRIGSFIT(MXNRIG),CRIGS(MXNRIG),deltco
  INTEGER IT,JT,KT,IR,JR,KR
  LOGICAL OVERLAP,NEWEMP,NEWRIG,LDDR,LCORE,LFIXED(MXNRIG)
  character(len=40) NAMERIG
  !
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIG),LFIXED(IRIG)
  ENDDO
  WRITE(OUTU,1035)NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
  WRITE(OUTU,1037)LDDR,LCORE
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("RIGID DOMAIN ",I4,2X,A20,"FIXED:",L4)
1035 FORMAT(" NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
1037 FORMAT(" LDDR=",L4,"  LCORE=",L4)
  !   Copy the map to IDEMPS to maintain IDEMP intact
  !      Create scratch emaps
  NDATA=LEMAPX(IDEMP0)*LEMAPY(IDEMP0)*LEMAPZ(IDEMP0)
  NEWEMP=CHKEMPNM('SCRATCH',7,IDEMPS)
  IF(NEWEMP)THEN
     CALL EMAPDUP(IDEMP0,IDEMPS)
  ELSE
     CALL WRNDIE(0,'<EMAPGTMC1>','CANNOT CREATE EMAP: SCRATCH')
  ENDIF
  !  Define copy numbers to avoid repeated search for identical copies
  !  NCOPY(I) is the index of the other identical RIGID or 
  !      negative of total number of the copies
  loop100: Do IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     IDEMP=IDEMRIG(IDRIG)
     NCOPY(IRIG)=-1
     IF(LFIXED(IRIG))cycle loop100
     loop110: DO JRIG=IRIG+1,NRIG
        IF(LFIXED(JRIG))cycle loop110
        IDRIGJ=IDRIGS(JRIG)
        IF(IDEMP == IDEMRIG(IDRIGJ))THEN
           NCOPY(IRIG)=JRIG
           cycle loop100
        ENDIF
     enddo loop110
  enddo loop100
  !  Perform single-body docking for each unfitted RIGID
  loop200: DO IRIG=1,NRIG
     !  SKIP FIXED RIGID DOAMINS
     IF(LFIXED(IRIG))cycle loop200
     IF(PRNLEV > 2)WRITE(OUTU,'("Fiting rigid: ",I3)')IRIG
     !  build up scratch emap from the fixed rigid domains
     CALL EMAPINIT(IDEMPS,ZERO)
     NFIXED=0
     DO JRIG=1,NRIG
        IDRIGJ=IDRIGS(JRIG)
        IF(LFIXED(JRIG))THEN
           NFIXED=NFIXED+1
           CALL EMAPADD(IDEMPS,IDRIGJ)
        ENDIF
     ENDDO
     IF(NFIXED >= 1)THEN
        RHOCUT=EMRCUT
        CALL EMAPCORE(IDEMPS,RHOCUT,EMICORE)
        ! Derive the docking emap from original and the docked part emap
        !  IDEMPS will be the emap without docked domains
        CALL EMAPREDUCE(IDEMP0,IDEMPS,IDEMPS)
     ELSE
        CALL EMAPCOPY(IDEMP0,IDEMPS)
     ENDIF
     !   Create new rigid domains for docking operation
     JRIG=IRIG
     NRIGFIT=0
210  NRIGFIT=NRIGFIT+1
     IDRIGJ=IDRIGS(JRIG)
     LENGTH=LEN(EMRIGID(IDRIGJ))
     CALL TRIME(EMRIGID(IDRIGJ),LENGTH)
     NAMERIG=EMRIGID(IDRIGJ)(1:LENGTH)//'FIT'
     IF(.NOT.CHKRIGNM(NAMERIG,LENGTH,IDRIGN))CALL WRNDIE(0, &
          '<EMAPDOCK1>','CANNOT CREATE RIGID:'//NAMERIG(1:LENGTH))
     CALL EMAPRIGDUP(IDRIGJ,IDRIGN)
     IDRIGSFIT(NRIGFIT)=IDRIGN
     IDLIST(NRIGFIT)=JRIG
     JRIG=NCOPY(JRIG)
     IF(JRIG > 0)GOTO 210
     !  Perform GTMC dock to find n copies of best fits
     CALL EMAPGRID1(IDEMPS,NRIGFIT,IDRIGSFIT,CRIGSFIT,NTGRID,NRGRID, &
          NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior,LDDR,LCORE)
     !  Determine the fitting status of the fitted domains
     CORRMAX=CRIGSFIT(1)-DELTCO
     DO JRIG=1,NRIGFIT
        ICOPY=IDLIST(JRIG)
        IF(CRIGSFIT(JRIG) > CORRMAX.OR. &
             CRIGSFIT(JRIG) > CFIX)THEN
           LFIXED(ICOPY)=.TRUE.
           IF(PRNLEV > 2)WRITE(OUTU,1050)ICOPY,CRIGSFIT(JRIG)
        ENDIF
        CALL EMAPRIGCPY(IDRIGSFIT(JRIG),IDRIGS(ICOPY))
        CRIGS(ICOPY)=CRIGSFIT(JRIG)
     ENDDO
     !  Delete scratch rigid domains
     DO JRIG=NRIGFIT,1,-1
        CALL RMEMRIG(IDRIGSFIT(JRIG))
     ENDDO
  enddo loop200
  IF(PRNLEV > 2) &
       WRITE(OUTU,'("DOWN WITH FITTING")')
  CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
  CALL EMAPSTAT(IDEMPS)
  CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
  IF(PRNLEV > 4)WRITE(OUTU,1003)CORRT
  CALL RMEMAP(IDEMPS)
1003 FORMAT("AFTER GTMC FITTING, Final CORRMAX=",F10.6,F10.6)
1050 FORMAT("RIGID ",I3," was fitted with CORRT=",F10.6)
  RETURN
END SUBROUTINE EMAPGTMC1

SUBROUTINE EMAPGRID1(IDEMP0,NRIG,IDRIGS,CTCOPY,NTGRID,NRGRID, &
     NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior,LDDR,LCORE)
  !-----------------------------------------------------------------------
  !   This routine perform grid fit a rigid fragment to a emap
  !   and nrig solutions will be returned
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP
  INTEGER NRIG,IDRIGS(*),IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,idrig1,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORR0,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,IRANK,IOVER
  real(chm_real) CTCOPY(MXNRIG)
  INTEGER IT,JT,KT,IR,JR,KR
  LOGICAL NEWRIG,LDDR,LCORE
  !
  CORR0=0.2
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     !        WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIGS(IRIG))
  ENDDO
  WRITE(OUTU,1035)NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
  WRITE(OUTU,1037)LDDR,LCORE
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("RIGID DOMAIN ",I4,2X,A40)
1035 FORMAT(" NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
1037 FORMAT("LDDR=",L4," LCORE=",L4)
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('GRMCRIGS',8,IDRIGSS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGSS))
  ! Calculate initial correlation
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
     CTCOPY(IRIG)=CORRT
  ENDDO
  ! Sort the NRIG copies in decent correlations
  DO IRIG=1,NRIG-1
     CORRO=CTCOPY(IRIG)
     IDRIG=IDRIGS(IRIG)
     DO JRIG=IRIG+1,NRIG
        IF(CORRO < CTCOPY(JRIG))THEN
           CALL EMAPRIGCPY(IDRIG,IDRIGSS)
           CALL EMAPRIGCPY(IDRIGS(JRIG),IDRIG)
           CALL EMAPRIGCPY(IDRIGSS,IDRIGS(JRIG))
           CTCOPY(IRIG)=CTCOPY(JRIG)
           CTCOPY(JRIG)=CORRO
        ENDIF
     ENDDO
  ENDDO
  !  Use the first RIGID domain to define the grid
  IDRIG=IDRIGS(1)
  CALL EMAPRIGDUP(IDRIG,IDRIGSS)
  CALL EMAPSAVE(IDRIGSS)
  mtxs=memapx(idemp0)
  mtys=memapy(idemp0)
  mtzs=memapz(idemp0)
  dtx=demapx(idemp0)
  dty=demapy(idemp0)
  dtz=demapz(idemp0)
  ltx=lemapx(idemp0)
  lty=lemapy(idemp0)
  ltz=lemapz(idemp0)
  dgx=dtx*lemapx(idemp0)/(ntgrid+1)
  dgy=dty*lemapy(idemp0)/(ntgrid+1)
  dgz=dtz*lemapz(idemp0)/(ntgrid+1)
  xg0=mtxs*dtx
  yg0=mtys*dty
  zg0=mtzs*dtz
  Rd=360.0/(nrgrid)
  IDEMP=IDEMRIG(IDRIG)
1000 IDEMP=IDEMRIG(IDRIG)
  xp=cemapx(idemp)+temrig(1,idrigss)
  yp=cemapy(idemp)+temrig(2,idrigss)
  zp=cemapz(idemp)+temrig(3,idrigss)
  dxp=xp-xg0-ANINT((xp-xg0)/dgx)*dgx
  dyp=yp-yg0-ANINT((yp-yg0)/dgy)*dgy
  dzp=zp-zg0-ANINT((zp-zg0)/dgz)*dgz
  do it=1,ntgrid
     xi=xg0+dxp+it*dgx
     do jt=1,ntgrid
        Yi=yg0+dyp+jt*dgy
        loop300: do kt=1,ntgrid
           Zi=zg0+dzp+kt*dgz
           do ir=1,nrgrid
              ri=ir*rd 
              do jr=1,nrgrid
                 rj=jr*rd
                 do kr=1,nrgrid
                    rk=kr*rd
                    CALL EMAPRESTORE(IDRIGSS)
                    CALL EMAPtrn(IDRIGSS,XI-xp,YI-yp,ZI-zp)
                    CALL EMAPROT(IDRIGSS,ONE,ZERO,ZERO,ri)
                    CALL EMAPROT(IDRIGSS,ZERO,ONE,ZERO,rj)
                    CALL EMAPROT(IDRIGSS,ZERO,ZERO,ONE,rk)
                    CALL EMAPMC1(IDEMP0,IDRIGSS,NCYC,NSTEP, &
                         DELTT,DELTR,TEMP,RATIOT,RATIOR,LDDR,LCORE,CORRT)
                    !  Update the best fittings
                    !  Determine CORRMAX RANK
                    loop450: DO IRANK=1,NRIG
                       IF(CORRT > CTCOPY(IRANK))exit loop450
                    ENDDO loop450

                    IF (IRANK <= NRIG) then
                       !  Determine overlaping with own copies
                       loop400: DO IOVER=1,NRIG
                          CALL EMAPCRR(IDRIGSS,IDRIGS(IOVER),LDDR,LCORE,CORRO)
                          IF(CORRO > CORR0)exit loop400
                       ENDDO loop400
400                    CONTINUE     
                       IF (IRANK <= IOVER) then
                          !  Update rigid copies
                          IF(IOVER > NRIG)IOVER=NRIG
                          !  The scratch is better than its overlapping copy
                          !   replace iover and rerank icopy to iover 
                          DO ICOPY=IOVER,IRANK+1,-1
                             CALL EMAPRIGCPY(IDRIGS(ICOPY-1),IDRIGS(ICOPY))
                             CTCOPY(ICOPY)=CTCOPY(ICOPY-1)
                          ENDDO
                          CALL EMAPRIGCPY(IDRIGSS,IDRIGS(IRANK))
                          CTCOPY(IRANK)=CORRT
                       endif
                    endif
                    !500                 CONTINUE
                    IF(PRNLEV > 3) &
                         WRITE(OUTU,1005)it,jt,kt,ir,jr,kr, &
                         xi,yi,zi,ri,rj,rk,CORRT,CTCOPY(1)
                 enddo
              enddo
           end do
        enddo loop300
     enddo
  end do
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('GRMCRIGS',8,IDRIGSS)
  CALL RMEMRIG(IDRIGSS)
1002 FORMAT("RIGID ",A10," MOVED to ",3F10.5," CORRMAX=",F10.5)
1005 FORMAT(" GRID ",6I3,6F7.1,2F7.4)
1006 FORMAT("New max. correlation:  ",F10.6)
1050 FORMAT("RIGID ",I3," was fitted with CORRT,CORRO=",F10.6,F10.6)
1060 FORMAT(I6,I6,F10.6)
1070 FORMAT(I6,I6,2F10.6)
  return
end SUBROUTINE EMAPGRID1


SUBROUTINE EMAPGRIDN(IDEMP,IDEMPS,IDRIG,NTGRID,NRGRID, &
     NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior, &
     LDDR,LCORE,CORRMAX)
  !-----------------------------------------------------------------------
  !   This routine perform grid fit of a rigid fragment+map to an emap
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,idrig1,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORR0,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,NCORR,IOVER,NCOPY(MXNRIG),IDRIGCP
  real(chm_real) COCOPY(MXNRIG),CTCOPY(MXNRIG),CORROCP,CORRTCP
  INTEGER IT,JT,KT,IR,JR,KR
  LOGICAL OVERLAP,NEWRIG,LDDR,LCORE
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('GRMCRIGS',8,IDRIGSS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGSS))
  CALL EMAPRIGDUP(IDRIG,IDRIGSS)
  CALL EMAPSAVE(IDRIGSS)
  !
  mtxs=memapx(idemps)
  mtys=memapy(idemps)
  mtzs=memapz(idemps)
  dtx=demapx(idemps)
  dty=demapy(idemps)
  dtz=demapz(idemps)
  ltx=lemapx(idemps)
  lty=lemapy(idemps)
  ltz=lemapz(idemps)
  dgx=dtx*lemapx(idemps)/(ntgrid+1)
  dgy=dty*lemapy(idemps)/(ntgrid+1)
  dgz=dtz*lemapz(idemps)/(ntgrid+1)
  xg0=mtxs*dtx
  yg0=mtys*dty
  zg0=mtzs*dtz
  Rd=360.0/(nrgrid)
  ! Calculate initial correlation
  NCORR=0
  CALL EMAPCMMR(IDEMP,IDEMPS,IDRIGSS,LDDR,LCORE,CORRT,NCORR)
  IDEMPJ=IDEMRIG(IDRIGss)
  xp=cemapx(IDEMPJ)+temrig(1,idrigss)
  yp=cemapy(IDEMPJ)+temrig(2,idrigss)
  zp=cemapz(IDEMPJ)+temrig(3,idrigss)
  IF(PRNLEV > 3) &
       WRITE(OUTU,1005)0,0,0,0,0,0, &
       xp,yp,zp,0.0,0.0,0.0,CORRT,CORRMAX
  CORRMAX=CORRT
  dxp=xp-xg0-ANINT((xp-xg0)/dgx)*dgx
  dyp=yp-yg0-ANINT((yp-yg0)/dgy)*dgy
  dzp=zp-zg0-ANINT((zp-zg0)/dgz)*dgz
  do it=1,ntgrid
     xi=xg0+dxp+it*dgx
     do jt=1,ntgrid
        Yi=yg0+dyp+jt*dgy
        loop300: do kt=1,ntgrid
           Zi=zg0+dzp+kt*dgz
           do ir=1,nrgrid
              ri=ir*rd 
              do jr=1,nrgrid
                 rj=jr*rd
                 do kr=1,nrgrid
                    rk=kr*rd
                    CALL EMAPRESTORE(IDRIGSS)
                    CALL EMAPtrn(IDRIGSS,XI-xp,YI-yp,ZI-zp)
                    CALL EMAPROT(IDRIGSS,ONE,ZERO,ZERO,ri)
                    CALL EMAPROT(IDRIGSS,ZERO,ONE,ZERO,rj)
                    CALL EMAPROT(IDRIGSS,ZERO,ZERO,ONE,rk)
                    CALL EMAPMCN(IDEMP,IDEMPS,IDRIGSS,NCYC,NSTEP, &
                         DELTT,DELTR,TEMP,RATIOT,RATIOR,LDDR,LCORE,CORRT)
                    !  Update the best fittings
                    !  Determine CORRMAX RANK
                    IF(CORRT > CORRMAX)THEN
                       CALL EMAPRIGCPY(IDRIGSS,IDRIG)
                       CORRMAX=CORRT
                    ENDIF
                    IF(PRNLEV > 3) &
                         WRITE(OUTU,1005)it,jt,kt,ir,jr,kr, &
                         xi,yi,zi,ri,rj,rk,CORRT,CORRMAX
                 enddo
              enddo
           end do
        enddo loop300
     enddo
  end do
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('GRMCRIGS',8,IDRIGSS)
  CALL RMEMRIG(IDRIGSS)
1002 FORMAT("RIGID ",A10," MOVED to ",3F10.5," CORRMAX=",F10.5)
1005 FORMAT(" GRID ",6I3,6F7.1,2F7.4)
1006 FORMAT("New max. correlation:  ",F10.6)
1050 FORMAT("RIGID ",I3," was fitted with CORRT,CORRO=",F10.6,F10.6)
1060 FORMAT(I6,I6,F10.6)
1070 FORMAT(I6,I6,2F10.6)
  return
end SUBROUTINE EMAPGRIDN


SUBROUTINE EMAPMC1(IDEMP0,IDRIG,NCYC,MCSTEP, &
     DELTT0,DELTR0,TEMP,ratiot,ratior,LDDR,LCORE,CORRMAX)
  !-----------------------------------------------------------------------
  !   This routine fit a rigid domain to a destinate emap with MC method
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use reawri
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,NCYC,MCSTEP
  INTEGER IDRIG,IDRIGS
  INTEGER I,ICYC,ISTEP
  real(chm_real) TX,TY,TZ,CORRO,CORRT,FACT,DECORR
  real(chm_real) DELTR0,DELTT0,DELTR,DELTT,TEMP, &
       RATIOT,RATIOR,RATT,RATR
  real(chm_real) CORRMAX,RAND01
  INTEGER NACCT,NACCR,NDATA,NXYZS
  LOGICAL NEWEMP,NEWRIG,LCORE,LDDR
  !
  DELTT=DELTT0
  DELTR=DELTR0
  CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRMAX)
  CORRP=CORRMAX
  IF(PRNLEV > 4)WRITE(OUTU,1021)
1021 FORMAT(" MC_1    ICYC    CORRMAX     CORRT  ", &
       "   RATT     RATR     DELTT     DELTR")
  IF(PRNLEV > 4)WRITE(OUTU,1002) &
       0,CORRMAX,CORRMAX,RATIOT,RATIOR,DELTT,DELTR
  IF(NCYC*MCSTEP <= 0)RETURN
  IF(CORRMAX < -ONE)RETURN
  !
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<GTMC1>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGS))
  CALL EMAPRIGDUP(IDRIG,IDRIGS)
  CALL EMAPSAVE(IDRIGS)
  IDEMP=IDEMRIG(IDRIGS)
  DO ICYC=1,NCYC
     NACCT=0
     NACCR=0
     DO ISTEP=1,MCSTEP
        ! TRANSLATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTT
        TY=(ONE-TWO*RANDOM(ISEED))*DELTT
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTT
        CALL EMAPTRN(IDRIGS,TX,TY,TZ)
        CALL EMAPCRM(IDEMP0,IDRIGS,LDDR,LCORE,CORRT)
        DECORR=CORRT-CORRP
        IF(DECORR < ZERO)THEN
           FACT=EXP(DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              GOTO 100
           ENDIF
        ENDIF
        IF(CORRMAX < CORRT)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           CORRMAX=CORRT
           IF(PRNLEV > 5)WRITE(OUTU,1010)EMRIGID(IDRIG),CORRMAX
        ENDIF
        NACCT=NACCT+1
        CORRP=CORRT
        CALL EMAPSAVE(IDRIGS)
100     CONTINUE
        ! ROTATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTR
        TY=(ONE-TWO*RANDOM(ISEED))*DELTR
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTR
        CALL EMAPROT(IDRIGS,ONE,ZERO,ZERO,TX)
        CALL EMAPROT(IDRIGS,ZERO,ONE,ZERO,TY)
        CALL EMAPROT(IDRIGS,ZERO,ZERO,ONE,TZ)
        CALL EMAPCRM(IDEMP0,IDRIGS,LDDR,LCORE,CORRT)
        DECORR=CORRT-CORRP
        IF(DECORR < ZERO)THEN
           FACT=EXP(DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              GOTO 200
           ENDIF
        ENDIF
        IF(CORRMAX < CORRT)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           CORRMAX=CORRT
           IF(PRNLEV > 5)WRITE(OUTU,1010)EMRIGID(IDRIG),CORRMAX
        ENDIF
        NACCR=NACCR+1
        CORRP=CORRT
        CALL EMAPSAVE(IDRIGS)
200     CONTINUE
     ENDDO
     !   Update moving size
     RATT=ONE*NACCT/MCSTEP
     RATR=ONE*NACCR/MCSTEP
     IF(PRNLEV > 4)WRITE(OUTU,1002) &
          ICYC,CORRMAX,CORRP,RATT,RATR,DELTT,DELTR
     IF(RATT > RATIOT)THEN
        DELTT=1.05*DELTT
     ELSE
        DELTT=0.95*DELTT
     ENDIF
     IF(RATR > RATIOR)THEN
        DELTR=1.05*DELTR
     ELSE
        DELTR=0.95*DELTR
     ENDIF
     !   Using the best conformation as start
     !          CALL EMAPRIGCPY(IDRIG,IDRIGS)
     CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRP)
     IF(ABS(CORRP-CORRMAX) > RSMALL) &
          WRITE(OUTU,1011)IDRIG,CORRP,CORRMAX
     IF(PRNLEV > 5)WRITE(OUTU,1001)IDRIG,CORRT,CORRMAX
  ENDDO
600 CONTINUE
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  CALL RMEMRIG(IDRIGS)
1001 FORMAT(10X,I4,5F10.6)
1011 FORMAT("RIGID ",I5," has problem with CORRMAX and CORRT:",5F10.6)
1002 FORMAT(I10,4X,6F10.6)
1010 FORMAT("New Lowest State for ",A10," with EMIN=",F10.5)
  return
end SUBROUTINE EMAPMC1


SUBROUTINE EMAPMCN(IDEMP0,IDEMPS,IDRIG,NCYC,MCSTEP, &
     DELTT0,DELTR0,TEMP,ratiot,ratior,LDDR,LCORE,CORRMAX)
  !-----------------------------------------------------------------------
  !   This routine fit a rigid domain+ map to a destinate map with MC method
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use reawri
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,NCYC,MCSTEP
  INTEGER IDRIGS,NCORR
  INTEGER I,ICYC,ISTEP,IRIG,JRIG,idrig,IDRIGJ,IDEMPJ,IDEMPS
  real(chm_real) TX,TY,TZ,CORRO,CORRT,FACT,DECORR
  real(chm_real) DELTR0,DELTT0,DELTR,DELTT,TEMP, &
       RATIOT,RATIOR,RATT,RATR
  real(chm_real) CORRMAX,RAND01
  INTEGER NACCT,NACCR,NDATA,NXYZS
  LOGICAL NEWEMP,NEWRIG,LDDR,LCORE
  !
  DELTT=DELTT0
  DELTR=DELTR0
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGS))
  CALL EMAPRIGDUP(IDRIG,IDRIGS)
  CALL EMAPSAVE(IDRIGS)
  NCORR=1
  CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIG,LDDR,LCORE,CORRT,NCORR)
  CORRMAX=CORRT
  IF(PRNLEV > 2)WRITE(OUTU,1021)
1021 FORMAT(" MC_N    ICYC    CORRMAX     CORRT  ", &
       "   RATT     RATR     DELTT     DELTR")
  IF(PRNLEV > 4)WRITE(OUTU,1002) &
       0,CORRMAX,CORRMAX,RATIOT,RATIOR,DELTT,DELTR
  IF(CORRT < -ONE)GOTO 600
  DO ICYC=1,NCYC
     NACCT=0
     NACCR=0
     CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIGS,LDDR,LCORE,CORRT,NCORR)
     IF(CORRT < -ONE)GOTO 500
     CORRP=CORRT
     loop200: DO ISTEP=1,MCSTEP
        ! TRANSLATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTT
        TY=(ONE-TWO*RANDOM(ISEED))*DELTT
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTT
        CALL EMAPTRNIN(IDRIGS,TX,TY,TZ)
        CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIGS,LDDR,LCORE,CORRT,NCORR)
        DECORR=CORRT-CORRP
        IF(DECORR < ZERO)THEN
           FACT=EXP(DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              GOTO 100
           ENDIF
        ENDIF
        IF(CORRMAX < CORRT)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           IF(PRNLEV > 5)WRITE(OUTU,1010)CORRT,CORRMAX
           CORRMAX=CORRT
        ENDIF
        NACCT=NACCT+1
        CORRP=CORRT
        CALL EMAPSAVE(IDRIGS)
100     CONTINUE
        ! ROTATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTR
        TY=(ONE-TWO*RANDOM(ISEED))*DELTR
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTR
        CALL EMAPROT(IDRIGS,ONE,ZERO,ZERO,TX)
        CALL EMAPROT(IDRIGS,ZERO,ONE,ZERO,TY)
        CALL EMAPROT(IDRIGS,ZERO,ZERO,ONE,TZ)
        CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIGS,LDDR,LCORE,CORRT,NCORR)
        DECORR=CORRT-CORRP
        IF(DECORR < ZERO)THEN
           FACT=EXP(DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              cycle loop200
           ENDIF
        ENDIF
        IF(CORRMAX < CORRT)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           IF(PRNLEV > 5)WRITE(OUTU,1010)CORRT,CORRMAX
           CORRMAX=CORRT
        ENDIF
        NACCR=NACCR+1
        CORRP=CORRT
        CALL EMAPSAVE(IDRIGS)
     ENDDO loop200

     CALL EMAPRIGCPY(IDRIG,IDRIGS)
     CALL EMAPSAVE(IDRIGS)
500  CONTINUE
     RATT=ONE*NACCT/MCSTEP
     RATR=ONE*NACCR/MCSTEP
     IF(RATT > RATIOT)THEN
        DELTT=1.05*DELTT
     ELSE
        DELTT=0.95*DELTT
     ENDIF
     IF(RATR > RATIOR)THEN
        DELTR=1.05*DELTR
     ELSE
        DELTR=0.95*DELTR
     ENDIF
     CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIG,LDDR,LCORE,CORRT,NCORR)
     IF(PRNLEV > 4)WRITE(OUTU,1002)ICYC,CORRT,CORRMAX, &
          RATT,RATR,DELTT,DELTR
  ENDDO
600 CONTINUE
  !  REMOVE scratch rigid fragments 
  !      NEWRIG=RMRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  CALL RMEMRIG(IDRIGS)
1001 FORMAT(10X,I4,5F10.6)
1002 FORMAT(I10,4X,6F10.6)
1010 FORMAT("New Lowest State for ",A10," with EMIN=",F10.5)
  return
end SUBROUTINE EMAPMCN

SUBROUTINE FMAPGTMC1(IDEMP0,NRIG,IDRIGS,LFIXED, &
     NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior, &
     CFIX,DTENG,LGTVOL,SITE1,SANGLE1,SITE2,SANGLE2,  &
     REX1,REX2,REY1,REY2,REZ1,REZ2,  &
     LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2)
  !-----------------------------------------------------------------------
  !   This routine perform a series single-body GTMC fitting to a emap
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use string
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER NRIG,IDRIGS(*),IDRIGSS,NFIXED
  INTEGER IDRIGN,IDLIST(MXNRIG),NRIGFIT,IDRIGSFIT(MXNRIG)
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS,LENGTH
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  REAL(CHM_REAL) SITE1(3),SANGLE1,SITE2(3),SANGLE2
  REAL(CHM_REAL) REX1,REX2,REY1,REY2,REZ1,REZ2
  REAL(CHM_REAL) LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2
  REAL(CHM_REAL) XP,YP,ZP,XI,YI,ZI,DTX,DTY,DTZ,RD
  REAL(CHM_REAL) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  REAL(CHM_REAL) CORR0,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  REAL(CHM_REAL) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  REAL(CHM_REAL) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,ICOPY1,IOVER,NCOPY(MXNRIG),IDRIGCP
  REAL(CHM_REAL) CRIGSFIT(MXNRIG),CRIGS(MXNRIG),DTENG
  INTEGER IT,JT,KT,IR,JR,KR
  LOGICAL OVERLAP,NEWEMP,NEWRIG,LFIXED(MXNRIG),LGTVOL
  CHARACTER(LEN=40) NAMERIG
  !
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIG),LFIXED(IRIG)
  ENDDO
  WRITE(OUTU,1035)NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("RIGID DOMAIN ",I4,2X,A20,"FIXED:",L4)
1035 FORMAT(" NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
  !   Copy the map to IDEMPS to maintain IDEMP intact
  !      Create scratch emaps
  NDATA=LEMAPX(IDEMP0)*LEMAPY(IDEMP0)*LEMAPZ(IDEMP0)
  NEWEMP=CHKEMPNM('SCRATCH',7,IDEMPS)
  IF(NEWEMP)THEN
     CALL EMAPDUP(IDEMP0,IDEMPS)
  ELSE
     CALL WRNDIE(0,'<EMAPGTMC1>','CANNOT CREATE EMAP: SCRATCH')
  ENDIF
  !  Define copy numbers to avoid repeated search for identical copies
  !  NCOPY(I) is the index of the other identical RIGID or 
  !      negative of total number of the copies
  loop100: Do IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     IDEMP=IDEMRIG(IDRIG)
     NCOPY(IRIG)=-1
     IF(LFIXED(IRIG))cycle loop100
     loop110: DO JRIG=IRIG+1,NRIG
        IF(LFIXED(JRIG))cycle loop110
        IDRIGJ=IDRIGS(JRIG)
        IF(IDEMP == IDEMRIG(IDRIGJ))THEN
           NCOPY(IRIG)=JRIG
           cycle loop100
        ENDIF
     enddo loop110
  enddo loop100
  !  Perform single-body docking for each unfitted RIGID
  loop200: DO IRIG=1,NRIG
     !  SKIP FIXED RIGID DOAMINS
     IF(LFIXED(IRIG))cycle loop200
     IF(PRNLEV > 2)WRITE(OUTU,'("Fiting rigid: ",I3)')IRIG
     !   Create new rigid domains for docking operation
     JRIG=IRIG
     NRIGFIT=0
     !  Build up temporary array of rigid domains with the same map objects
210  NRIGFIT=NRIGFIT+1
     IDRIGJ=IDRIGS(JRIG)
     LENGTH=LEN(EMRIGID(IDRIGJ))
     CALL TRIME(EMRIGID(IDRIGJ),LENGTH)
     NAMERIG=EMRIGID(IDRIGJ)(1:LENGTH)//'FIT'
     IF(.NOT.CHKRIGNM(NAMERIG,LENGTH,IDRIGN))CALL WRNDIE(0, &
          '<EMAPDOCK1>','CANNOT CREATE RIGID:'//NAMERIG(1:LENGTH))
     CALL EMAPRIGDUP(IDRIGJ,IDRIGN)
     IDRIGSFIT(NRIGFIT)=IDRIGN
     IDLIST(NRIGFIT)=JRIG
     JRIG=NCOPY(JRIG)
     IF(JRIG > 0)GOTO 210
     IF(LGTVOL)THEN
        !  Perform GTMC dock to find n copies of best fits
        CALL FMAPGRID1(IDEMPS,NRIGFIT,IDRIGSFIT,CRIGSFIT, &
             NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior)
     ELSE
        !  Perform surface grid dock to find n copies of best fits
        CALL FMAPSURF1(IDEMPS,NRIGFIT,IDRIGSFIT,CRIGSFIT, &
         NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
         SITE1,SANGLE1,SITE2,SANGLE2,REX1,REX2,REY1,REY2,REZ1,REZ2, &
         LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2)
     ENDIF
     !  Determine the fitting status of the fitted domains
     CORRMAX=CRIGSFIT(1)+DTENG
     DO JRIG=1,NRIGFIT
        ICOPY=IDLIST(JRIG)
        IF(CRIGSFIT(JRIG) < CORRMAX)THEN
           LFIXED(ICOPY)=.TRUE.
           IF(PRNLEV > 2)WRITE(OUTU,1050)ICOPY,CRIGSFIT(JRIG)
        ENDIF
        CALL EMAPRIGCPY(IDRIGSFIT(JRIG),IDRIGS(ICOPY))
        CRIGS(ICOPY)=CRIGSFIT(JRIG)
     ENDDO
     !  Delete scratch rigid domains
     DO JRIG=NRIGFIT,1,-1
        CALL RMEMRIG(IDRIGSFIT(JRIG))
     ENDDO
  enddo loop200
  IF(PRNLEV > 2) &
       WRITE(OUTU,'("DOWN WITH FITTING")')
  CALL RMEMAP(IDEMPS)
1050 FORMAT("RIGID ",I3," was fitted with EFMAPMIN=",E14.6)
  RETURN
END SUBROUTINE FMAPGTMC1


SUBROUTINE FMAPGRID1(IDEMP0,NRIG,IDRIGS,CTCOPY,NTGRID,NRGRID, &
     NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior)
  !-----------------------------------------------------------------------
  !   This routine perform grid fit a rigid fragment to a emap
  !   and nrig solutions will be returned
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP1,IDEMP
  INTEGER NRIG,IDRIGS(*),IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) EOVER,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  real(chm_real) EFMAP,EFMAPMIN,ECORE,EELE,ESOLV,ECONS,RAND01
  INTEGER ICOPY,IRANK,IOVER
  real(chm_real) CTCOPY(MXNRIG)
  real(chm_real) xs,ys,zs,xe,ye,ze
  real(chm_real) amx,amy,amz,arx,ary,arz,dmx,dmy,dmz,drx,dry,drz
  INTEGER IT,JT,KT,IR,JR,KR
  LOGICAL NEWRIG
  !
  EOVER=ZERO
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     !        WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIGS(IRIG))
  ENDDO
  WRITE(OUTU,1035)NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("RIGID DOMAIN ",I4,2X,A40)
1035 FORMAT(" NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('GRMCRIGS',8,IDRIGSS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGSS))
  ! Calculate initial correlation
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     ECORE=ZERO
     EELE=ZERO
     ESOLV=ZERO
     ECONS=ZERO
     CALL EMAPERM(IDEMP0,IDRIG,ECORE,EELE,ESOLV,ECONS)
     EFMAP=ECORE+EELE+ESOLV+ECONS
     CTCOPY(IRIG)=EFMAP
  ENDDO
  ! Sort the NRIG copies in acende energy
  DO IRIG=1,NRIG-1
     CORRO=CTCOPY(IRIG)
     IDRIG=IDRIGS(IRIG)
     DO JRIG=IRIG+1,NRIG
        IF(CORRO > CTCOPY(JRIG))THEN
           CALL EMAPRIGCPY(IDRIG,IDRIGSS)
           CALL EMAPRIGCPY(IDRIGS(JRIG),IDRIG)
           CALL EMAPRIGCPY(IDRIGSS,IDRIGS(JRIG))
           CTCOPY(IRIG)=CTCOPY(JRIG)
           CTCOPY(JRIG)=CORRO
        ENDIF
     ENDDO
  ENDDO
  !  Use the first RIGID domain to define the grid
  IDRIG=IDRIGS(1)
  IDEMP1=IDEMRIG(IDRIG)
  CALL EMAPRIGDUP(IDRIG,IDRIGSS)
  CALL EMAPSAVE(IDRIGSS)
  DTX=demapx(idemp1)*lemapx(idemp1)
  DTY=demapY(idemp1)*lemapY(idemp1)
  DTZ=demapZ(idemp1)*lemapZ(idemp1)
  RD=DTX
  IF(RD < DTY)RD=DTY
  IF(RD < DTZ)RD=DTZ
  xs=demapx(idemp0)*memapx(idemp0)-RD
  ys=demapy(idemp0)*memapy(idemp0)-RD
  zs=demapz(idemp0)*memapz(idemp0)-RD
  dgx=(demapx(idemp0)*lemapx(idemp0)+TWO*RD)/(ntgrid)
  dgy=(demapy(idemp0)*lemapy(idemp0)+TWO*RD)/(ntgrid)
  dgz=(demapz(idemp0)*lemapz(idemp0)+TWO*RD)/(ntgrid)
  Rd=360.0/(nrgrid)
  IDEMP=IDEMRIG(IDRIG)
1000 IDEMP=IDEMRIG(IDRIG)
  xp=temrig(1,idrigss)+cemapx(idemp1)
  yp=temrig(2,idrigss)+cemapy(idemp1)
  zp=temrig(3,idrigss)+cemapz(idemp1)
  dxp=xp-(DINT((xp-xs)/dgx)+ONE)*dgx
  dyp=yp-(DINT((yp-ys)/dgy)+ONE)*dgy
  dzp=zp-(DINT((zp-zs)/dgz)+ONE)*dgz
  do it=1,ntgrid
     xi=dxp+it*dgx
     do jt=1,ntgrid
        Yi=dyp+jt*dgy
        loop300: do kt=1,ntgrid
           Zi=dzp+kt*dgz
           do ir=1,nrgrid
              ri=ir*rd 
              do jr=1,nrgrid
                 rj=jr*rd
                 do kr=1,nrgrid
                    rk=kr*rd
                    CALL EMAPRESTORE(IDRIGSS)
                    CALL EMAPtrn(IDRIGSS,XI-xp,YI-yp,ZI-zp)
                    CALL EMAPROT(IDRIGSS,ONE,ZERO,ZERO,ri)
                    CALL EMAPROT(IDRIGSS,ZERO,ONE,ZERO,rj)
                    CALL EMAPROT(IDRIGSS,ZERO,ZERO,ONE,rk)
                    CALL FMAPMC1(IDEMP0,IDRIGSS,NCYC,NSTEP, &
                         DELTT,DELTR,TEMP,RATIOT,RATIOR,CORRT)
                    !  Update the best fittings
                    !  Determine CORRMAX RANK
                    loopa: DO IRANK=1,NRIG
                       IF(CORRT < CTCOPY(IRANK))exit loopa
                    ENDDO loopa

                    ! IF(IRANK > NRIG)GOTO 500
                    IF(IRANK <= NRIG)then
                       !  Determine overlaping with own copies
                       loop400: DO IOVER=1,NRIG
                          IDRIGJ=IDRIGS(IOVER)
                          CALL EMAPERR(IDRIGJ,IDRIGSS,ECORE,EELE,ESOLV,ECONS)
                          EFMAP=ECORE+EELE+ESOLV+ECONS
                          IF(EFMAP > EOVER)exit loop400
                       ENDDO loop400

                       ! IF(IRANK > IOVER)GOTO 500
                       IF(IRANK <= IOVER)then
                          !  Update rigid copies
                          IF(IOVER > NRIG)IOVER=NRIG
                          !  The scratch is better than its overlapping copy
                          !   replace iover and rerank icopy to iover 
                          DO ICOPY=IOVER,IRANK+1,-1
                             CALL EMAPRIGCPY(IDRIGS(ICOPY-1),IDRIGS(ICOPY))
                             CTCOPY(ICOPY)=CTCOPY(ICOPY-1)
                          ENDDO
                          CALL EMAPRIGCPY(IDRIGSS,IDRIGS(IRANK))
                          CTCOPY(IRANK)=CORRT
                       endif
                    endif
                    ! 500                 CONTINUE
                    IF(PRNLEV > 3) &
                         WRITE(OUTU,1005)it,jt,kt,ir,jr,kr, &
                         xi-xp,yi-yp,zi-zp,ri,rj,rk,CORRT,CTCOPY(1)
                 enddo
              enddo
           end do
        enddo loop300
     enddo
  end do
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('GRMCRIGS',8,IDRIGSS)
  CALL RMEMRIG(IDRIGSS)
1002 FORMAT("RIGID ",A10," MOVED to ",3F10.2," EFMAPMIN=",E14.6)
1005 FORMAT("GRID",6I3,6F6.0,2F9.2)
1006 FORMAT("New min. energy:  ",E14.6)
  return
end SUBROUTINE FMAPGRID1


SUBROUTINE FMAPSURF1(IDEMP0,NRIG,IDRIGS,CTCOPY,NTGRID,NRGRID, &
         NCYC,NSTEP,TEMP,DELTT,DELTR,RATIOT,RATIOR, &
         SITE1,SANGLE1,SITE2,SANGLE2,  &
         REX1,REX2,REY1,REY2,REZ1,REZ2,  &
         LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2)  
  !-----------------------------------------------------------------------
  !   This routine perform grid fit a rigid fragment to a emap
  !   and nrig solutions will be returned
  !
  use chm_kinds
  use consta
  use dimens_fcm
  use number
  use vector
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP1,IDEMP
  INTEGER NRIG,IDRIGS(*),IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  REAL(CHM_REAL) XP,YP,ZP,XI,YI,ZI,DTX,DTY,DTZ,RD
  REAL(CHM_REAL) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  REAL(CHM_REAL) EOVER,CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  REAL(CHM_REAL) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  REAL(CHM_REAL) RI,RJ,RK,RRXYZS,RHOCUT
  REAL(CHM_REAL) EFMAP,EFMAPMIN,ECORE,EELE,ESOLV,ECONS,RAND01
  INTEGER ICOPY,IRANK,IOVER
  REAL(CHM_REAL) CTCOPY(MXNRIG)
  REAL(CHM_REAL) XS,YS,ZS,XE,YE,ZE
  REAL(CHM_REAL) ST(3,1000),SR(3,1000),SRT(3),SRR(3),PHI, &
       DST,DSR,DSTR,COSTR
  REAL(CHM_REAL) SITE1(3),SANGLE1,SITE2(3),SANGLE2
  REAL(CHM_REAL) REX1,REX2,REY1,REY2,REZ1,REZ2
  REAL(CHM_REAL) LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2
  INTEGER IT,JT,KT,IR,JR,KR
  INTEGER NTST,NTSR,NST,NSR
  LOGICAL NEWRIG
  !
  EOVER=1000.0D0
  !    Printe out input information
  WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     !        WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIGS(IRIG))
  ENDDO
  WRITE(OUTU,1035)NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("RIGID DOMAIN ",I4,2X,A40)
1035 FORMAT(" NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
  !  Idensity the surface grid points of the working map
  CALL EMAPSGRID(IDEMP0,ntgrid,NTST,ST, &
        SITE1,SANGLE1,REX1,REX2,REY1,REY2,REZ1,REZ2)
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('GRMCRIGS',8,IDRIGSS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<MC>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGSS))
  ! Calculate initial correlation
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     CALL EMAPERM(IDEMP0,IDRIG,ECORE,EELE,ESOLV,ECONS)
     EFMAP=ECORE+EELE+ESOLV+ECONS
     CTCOPY(IRIG)=EFMAP
     CTCOPY(IRIG)=RBIG
  ENDDO
  ! Sort the NRIG copies in acende energy
  DO IRIG=1,NRIG-1
     CORRO=CTCOPY(IRIG)
     IDRIG=IDRIGS(IRIG)
     DO JRIG=IRIG+1,NRIG
        IF(CORRO > CTCOPY(JRIG))THEN
           CALL EMAPRIGCPY(IDRIG,IDRIGSS)
           CALL EMAPRIGCPY(IDRIGS(JRIG),IDRIG)
           CALL EMAPRIGCPY(IDRIGSS,IDRIGS(JRIG))
           CTCOPY(IRIG)=CTCOPY(JRIG)
           CTCOPY(JRIG)=CORRO
        ENDIF
     ENDDO
  ENDDO
  !  Use the first RIGID domain to define the grid
  IDRIG=IDRIGS(1)
  IDEMP1=IDEMRIG(IDRIG)
  CALL EMAPRIGDUP(IDRIG,IDRIGSS)
  CALL EMAPSAVE(IDRIGSS)
  IDEMP=IDEMRIG(IDRIG)
1000 IDEMP=IDEMRIG(IDRIG)
  !  Idensity the surface grid points of the working map
  CALL EMAPSGRID(IDEMP,nrgrid,NTSR,SR,  &
            SITE2,SANGLE2,LGX1,LGX2,LGY1,LGY2,LGZ1,LGZ2)
  do NST=1,NTST
     DST=SQRT(DOTVEC(ST(1,NST),ST(1,NST),3))
     DO NSR=1,NTSR
        DSR=SQRT(DOTVEC(SR(1,NSR),SR(1,NSR),3))
        DSTR=DOTVEC(ST(1,NST),SR(1,NSR),3)
        COSTR=DSTR/(DST*DSR)
        IF(COSTR > ONE)THEN
           PHI=ONE8TY
           SRT(1)=ONE
           SRT(2)=ONE
           SRT(3)=-(ST(1,NST)+ST(2,NST))*ST(3,NST) &
                /(ST(3,NST)*ST(3,NST)+RSMALL)
        ELSE IF(COSTR < -ONE)THEN
           PHI=ZERO
        ELSE
           PHI=ONE8TY-RADDEG*ACOS(COSTR)
           CALL CROSS3(ST(1,NST),SR(1,NSR),SRT)
        ENDIF
        !            CALL EMAPRESTORE(IDRIGSS)
        CALL EMAPRIGINI(IDRIGSS)
        TEMRIG(1,IDRIGSS)=ST(1,NST)*(DSR+DST)/DST &
             +CEMAPX(IDEMP0)-CEMAPX(IDEMP)
        TEMRIG(2,IDRIGSS)=ST(2,NST)*(DSR+DST)/DST &
             +CEMAPY(IDEMP0)-CEMAPY(IDEMP)
        TEMRIG(3,IDRIGSS)=ST(3,NST)*(DSR+DST)/DST &
             +CEMAPZ(IDEMP0)-CEMAPZ(IDEMP)
        DSTR=DOTVEC(SRT,SRT,3)
        IF(DSTR > PT01) &
             CALL EMAPROT(IDRIGSS,SRT(1),SRT(2),SRT(3),PHI)
        !  debug
        !            CALL MULMXN(SRR,REMRIG(1,IDRIGSS),3,3,SR(1,NSR),3,1)
        !            CALL CROSS3(ST(1,NST),SRR,SRT)
        !            CALL MULMXN(SRR,SR(1,NSR),1,3,REMRIG(1,IDRIGSS),3,3)
        !            CALL CROSS3(ST(1,NST),SRR,SRT)
        !   Perform MC search
        CALL FMAPMC1(IDEMP0,IDRIGSS,NCYC,NSTEP, &
             DELTT,DELTR,TEMP,RATIOT,RATIOR,CORRT)
        !  Update the best fittings
        !  Determine CORRMAX RANK
        loop450: DO IRANK=1,NRIG
           IF(CORRT < CTCOPY(IRANK))exit loop450
        ENDDO loop450

        IF(IRANK > NRIG)GOTO 500
        !  Determine overlaping with own copies
        DO IOVER=1,NRIG
           IDRIGJ=IDRIGS(IOVER)
           CALL EMAPERR(IDRIGJ,IDRIGSS,ECORE,EELE,ESOLV,ECONS)
           EFMAP=ECORE+EELE+ESOLV+ECONS
           IF(EFMAP > EOVER)GOTO 400
        ENDDO
400     CONTINUE     
        IF(IRANK > IOVER)GOTO 500
        !  Update rigid copies
        IF(IOVER > NRIG)IOVER=NRIG
        !  The scratch is better than its overlapping copy
        !   replace iover and rerank icopy to iover 
        DO ICOPY=IOVER,IRANK+1,-1
           CALL EMAPRIGCPY(IDRIGS(ICOPY-1),IDRIGS(ICOPY))
           CTCOPY(ICOPY)=CTCOPY(ICOPY-1)
        ENDDO
        CALL EMAPRIGCPY(IDRIGSS,IDRIGS(IRANK))
        CTCOPY(IRANK)=CORRT
500     CONTINUE
        IF(PRNLEV > 3)THEN
           WRITE(OUTU,1005)NST,NSR,ST(1,NST),ST(2,NST),ST(3,NST), &
                SR(1,NSR),SR(2,NSR),SR(3,NSR),CORRT,CTCOPY(1)
           IF(IEMPTRJ > 0) &
                CALL EMAPTRAJ(IDRIGSS,(NST-1)*NTSR+NSR,1,CORRT,CTCOPY(1),0)
        ENDIF
     ENDDO
  ENDDO
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('GRMCRIGS',8,IDRIGSS)
  CALL RMEMRIG(IDRIGSS)
1002 FORMAT("RIGID ",A10," MOVED to ",3F10.2," EFMAPMIN=",E14.6)
1005 FORMAT(2I6,6F8.2,2F10.2)
1006 FORMAT("New min. energy:  ",E14.6)
  return
end SUBROUTINE FMAPSURF1


SUBROUTINE FMAPMC1(IDEMP0,IDRIG,NCYC,MCSTEP, &
     DELTT0,DELTR0,TEMP,ratiot,ratior,EFMAPMIN)
  !-----------------------------------------------------------------------
  !   This routine fit a rigid domain to a destinate emap with MC method
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use reawri
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,NCYC,MCSTEP
  INTEGER IDRIG,IDRIGS
  INTEGER I,ICYC,ISTEP
  real(chm_real) TX,TY,TZ,FACT,DECORR
  real(chm_real) DELTR0,DELTT0,DELTR,DELTT,TEMP, &
       RATIOT,RATIOR,RATT,RATR
  real(chm_real) DELTTX,DELTTY,DELTTZ,DELTRX,DELTRY,DELTRZ
  real(chm_real) EFMAP,EFMAPP,EFMAPMIN,ECORE,EELE,ESOLV,ECONS,RAND01
  INTEGER NACCT,NACCR,NDATA,NXYZS
  LOGICAL NEWEMP,NEWRIG
  !
  DELTT=DELTT0
  DELTR=DELTR0
  CALL EMAPERM(IDEMP0,IDRIG,ECORE,EELE,ESOLV,ECONS)
  EFMAPP=ECORE+EELE+ESOLV+ECONS
  EFMAPMIN=EFMAPP
  IF(PRNLEV > 4)WRITE(OUTU,1021)
1021 FORMAT(" MC_1    ICYC    EFMAPMIN     EFMAP  ", &
       "   RATT     RATR     DELTT     DELTR")
  IF(PRNLEV > 4)WRITE(OUTU,1002) &
       0,EFMAPMIN,EFMAPP,RATIOT,RATIOR,DELTT,DELTR
  IF(NCYC*MCSTEP <= 0)RETURN
  !
  !  Create scratch rigid fragments for the rigid fragment
  NEWRIG=CHKRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  IF(.NOT.NEWRIG)CALL WRNDIE(0,'<GTMC1>', &
       'CANNOT CREAT SCRATCH rigid ID:'//EMRIGID(IDRIGS))
  CALL EMAPRIGDUP(IDRIG,IDRIGS)
  CALL EMAPSAVE(IDRIGS)
  IDEMP=IDEMRIG(IDRIGS)
  DO ICYC=1,NCYC
     NACCT=0
     NACCR=0
     DELTTX=DELTT*TRNEMAP(1,IDEMP)
     DELTTY=DELTT*TRNEMAP(2,IDEMP)
     DELTTZ=DELTT*TRNEMAP(3,IDEMP)
     DELTRX=DELTR*ROTEMAP(1,IDEMP)
     DELTRY=DELTR*ROTEMAP(2,IDEMP)
     DELTRZ=DELTR*ROTEMAP(3,IDEMP)
     DO ISTEP=1,MCSTEP
        ! TRANSLATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTTX
        TY=(ONE-TWO*RANDOM(ISEED))*DELTTY
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTTZ
        CALL EMAPTRN(IDRIGS,TX,TY,TZ)
        CALL EMAPERM(IDEMP0,IDRIGS,ECORE,EELE,ESOLV,ECONS)
        EFMAP=ECORE+EELE+ESOLV+ECONS
        DECORR=EFMAP-EFMAPP
        IF(DECORR > ZERO)THEN
           FACT=EXP(-DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              GOTO 100
           ENDIF
        ENDIF
        IF(EFMAPMIN > EFMAP)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           EFMAPMIN=EFMAP
           IF(PRNLEV > 5)WRITE(OUTU,1010)EMRIGID(IDRIG),EFMAPMIN
        ENDIF
        NACCT=NACCT+1
        EFMAPP=EFMAP
        CALL EMAPSAVE(IDRIGS)
100     CONTINUE
        ! ROTATION
        TX=(ONE-TWO*RANDOM(ISEED))*DELTRX
        TY=(ONE-TWO*RANDOM(ISEED))*DELTRY
        TZ=(ONE-TWO*RANDOM(ISEED))*DELTRZ
        CALL EMAPROT(IDRIGS,ONE,ZERO,ZERO,TX)
        CALL EMAPROT(IDRIGS,ZERO,ONE,ZERO,TY)
        CALL EMAPROT(IDRIGS,ZERO,ZERO,ONE,TZ)
        CALL EMAPERM(IDEMP0,IDRIGS,ECORE,EELE,ESOLV,ECONS)
        EFMAP=ECORE+EELE+ESOLV+ECONS
        DECORR=EFMAP-EFMAPP
        IF(DECORR > ZERO)THEN
           FACT=EXP(-DECORR/TEMP)
           RAND01=RANDOM(ISEED)
           IF(FACT < RAND01)THEN
              CALL EMAPRESTORE(IDRIGS)
              GOTO 200
           ENDIF
        ENDIF
        IF(EFMAPMIN >= EFMAP)THEN
           CALL EMAPRIGCPY(IDRIGS,IDRIG)
           EFMAPMIN=EFMAP
           IF(PRNLEV > 5)WRITE(OUTU,1010)EMRIGID(IDRIG),EFMAPMIN
        ENDIF
        NACCR=NACCR+1
        EFMAPP=EFMAP
        CALL EMAPSAVE(IDRIGS)
200     CONTINUE
     ENDDO
     !   Update moving sizes
     RATT=ONE*NACCT/MCSTEP
     RATR=ONE*NACCR/MCSTEP
     IF(PRNLEV > 4)WRITE(OUTU,1002) &
          ICYC,EFMAPMIN,EFMAPP,RATT,RATR,DELTT,DELTR
     IF(RATT > RATIOT)THEN
        DELTT=1.05*DELTT
     ELSE
        DELTT=0.95*DELTT
     ENDIF
     IF(RATR > RATIOR)THEN
        DELTR=1.05*DELTR
     ELSE
        DELTR=0.95*DELTR
     ENDIF
     !  Using the minimum configuration as  start
     !        CALL EMAPRIGCPY(IDRIG,IDRIGS)
     CALL EMAPERM(IDEMP0,IDRIG,ECORE,EELE,ESOLV,ECONS)
     EFMAP=ECORE+EELE+ESOLV+ECONS
     IF(ABS(EFMAP-EFMAPMIN) > RSMALL) &
          WRITE(OUTU,1011)IDRIG,EFMAP,EFMAPMIN
     IF(PRNLEV > 5)WRITE(OUTU,1001)IDRIG,EFMAPP,EFMAPMIN
  ENDDO
  !  REMOVE the scratch rigid fragment 
  !      NEWRIG=RMRIGNM('MC'//EMRIGID(idrig),20,IDRIGS)
  CALL RMEMRIG(IDRIGS)
1001 FORMAT(10X,I4,5F10.2)
1011 FORMAT("RIGID ",I5," has problem with EFMAPMIN and EFMAP:",5F10.2)
1002 FORMAT(I10,F14.2,F14.2,6F8.2)
1010 FORMAT("New Lowest State for ",A10," with EMIN=",F10.2)
  return
end SUBROUTINE FMAPMC1



SUBROUTINE FMAPGTMCN(IDEMP0,NRIG,IDRIGS,LFIXED,NLOOP, &
     NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior)
  !-----------------------------------------------------------------------
  !   This routine perform a many-body GTMC docking among many rigid fragments
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER NRIG,IDRIGS(*),IDRIGSS
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,idrig1,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS,NLOOP,ILOOP
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  INTEGER ICOPY,ICOPY1,IOVER,NCOPY(MXNRIG),IDRIGCP
  real(chm_real) COCOPY(MXNRIG),CTCOPY(MXNRIG),CORROCP,CORRTCP
  INTEGER IT,JT,KT,IR,JR,KR,NCORR
  LOGICAL OVERLAP,NEWEMP,NEWRIG,LFIXED(MXNRIG)
  !
  !    Printe out input information
  IF(IDEMP0>0)THEN
    WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  ELSE
    WRITE(OUTU,1029)
  ENDIF
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIG),LFIXED(IRIG)
  ENDDO
  WRITE(OUTU,1035)NLOOP,NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
1029 FORMAT("Dock without TARGET EMAP ")
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("DOMAIN ",I2,2X,A20,"FIXED:",L4)
1035 FORMAT("NLOOP=",I8," NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
  !   Create a scratch map  IDEMPS for docking
  NDATA=LEMAPX(IDEMP0)*LEMAPY(IDEMP0)*LEMAPZ(IDEMP0)
  NEWEMP=CHKEMPNM('SCRATCH',7,IDEMPS)
  IF(NEWEMP)THEN
     CALL EMAPDUP(IDEMP0,IDEMPS)
  ELSE
     CALL WRNDIE(0,'<EMAPGTMC N>','CANNOT CREATE EMAP: SCRATCH')
  ENDIF
  !  Perform NLOOP times of Many-body docking 
  DO ILOOP=1,NLOOP
     !   At each Loop, perform a GTMC for Each  RIGID domain 
     loop300: DO IRIG=1,NRIG
        IDRIG=IDRIGS(IRIG)
        !   Skip fixed rigid domains
        IF(LFIXED(IRIG))cycle loop300
        IF(PRNLEV > 2)WRITE(OUTU,'("Fiting rigid: ",I3)')IRIG
        !   Create a map from all rigid domains except IRIG
        !    build up scratch emap from the rest of rigid domains
        CALL EMAPINIT(IDEMPS,ZERO)
        DO JRIG=1,NRIG
           IDRIGJ=IDRIGS(JRIG)
           IF(IRIG /= JRIG)CALL EMAPADD(IDEMPS,IDRIGJ)
        ENDDO
        CALL EMAPSTAT(IDEMPS)
        !  Perform GTMC dock for IRIG with IDEMPS and IDEMP0 to find the best fits
        !          CALL FMAPGRIDN(IDEMP0,IDEMPS,IDRIG,NTGRID,NRGRID,
        !     &   NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior,CORRMAX)
        NCORR=0
        !          CALL EMAPCMMR(IDEMP0,IDEMPS,IDRIG,LDDR,LCORE,CORRO,NCORR)
        CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
        !          CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
        IF(PRNLEV > 2)WRITE(OUTU,1050)ILOOP,IDRIG,CORRO,CORRT
     enddo loop300
  enddo
  IF(PRNLEV > 2) &
       WRITE(OUTU,'("DOWN WITH FITTING")')
  CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
  !      CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
  IF(PRNLEV > 4)WRITE(OUTU,1003)CORRMAX
  CALL RMEMAP(IDEMPS)
1003 FORMAT("AFTER GTMC FITTING, CORRMAX=",F10.6)
1050 FORMAT(" At ILOOP=",I4," RIGID ",I3,"  CORRT=",2F10.6)
  RETURN
END SUBROUTINE FMAPGTMCN


SUBROUTINE FMAPMCN(IDEMP0,NRIG,IDRIGS,LFIXED,NLOOP, &
     NTGRID,NRGRID,NCYC,NSTEP,TEMP,DELTT,DELTR,ratiot,ratior, &
     LDDR,LCORE,COTOENG)
  !-----------------------------------------------------------------------
  !   This routine perform a many-body GTMC docking to an emap
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  !
  use stream
  use energym
  use emapmod
  implicit none
  INTEGER IDEMP0,IDEMP,IDEMPJ,IDEMPS,IDEMP1,IDEMP2,NDATA
  INTEGER NRIG,IDRIGS(*),IDRIGSS,ISEED
  INTEGER I,J,IRIG,JRIG,idrig,idrigj,idrig1,ntgrid,nrgrid
  INTEGER NCYC,NSTEP,NXYZS,NLOOP,ILOOP,ISTEP,ICYC
  INTEGER MTXS,MTYS,MTZS,LTX,LTY,LTZ
  real(chm_real) xP,yP,zP,xi,yi,zi,DTX,DTY,DTZ,RD
  real(chm_real) DGX,DGY,DGZ,XG0,YG0,ZG0,DXP,DYP,DZP
  real(chm_real) CORRO,CORROI,CORRMAX,CORRT,CORROIJ,CORRTIJ
  real(chm_real) RATIOT,RATIOR,DELTT,DELTR,TEMP,CFIX,RATT,RATR
  real(chm_real) DELTTX,DELTTY,DELTTZ,DELTRX,DELTRY,DELTRZ
  real(chm_real) RI,RJ,RK,RRXYZS,RHOCUT
  real(chm_real) RAND01,DECORR,FACT,TX,TY,TZ
  real(chm_real) ECORRT,ECOLD,ECNEW,ERIGT,ERIGOLD,ERIGNEW
  real(chm_real) EFMAP,ECORE,EELE,ESOLV,ECONS
  INTEGER ICOPY,ICOPY1,IOVER,NCOPY(MXNRIG),IDRIGCP
  real(chm_real) COCOPY(MXNRIG),CTCOPY(MXNRIG),COTOENG
  INTEGER IT,JT,KT,IR,JR,KR,NACCT,NACCR
  LOGICAL OVERLAP,NEWEMP,NEWRIG,LDDR,LCORE,LFIXED(MXNRIG)
  !
  !    Printe out input information
  IF(IDEMP0>0)THEN
    WRITE(OUTU,1030)IDEMP0,EMAPID(IDEMP0)
  ELSE
    WRITE(OUTU,1029)
  ENDIF
  WRITE(OUTU,1031)NRIG
  DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     IDEMP=IDEMRIG(IDRIG)
     WRITE(OUTU,1032)IDRIGS(IRIG),EMRIGID(IDRIG),LFIXED(IRIG),(TRNEMAP(I,IDEMP),I=1,3),(ROTEMAP(I,IDEMP),I=1,3)
  ENDDO
  WRITE(OUTU,1035)NLOOP,NTGRID,NRGRID, &
       NCYC,NSTEP,TEMP,RATIOT,RATIOR
  WRITE(OUTU,1037)LDDR
1029 FORMAT("Dock without TARGET EMAP ")
1030 FORMAT("TARGET EMAP:",I2,2X,A40)
1031 FORMAT(I3," DOCK DOMAINS")
1032 FORMAT("DOMAIN ",I2,2X,A20,"FIXED:",L4, "  Trans. x, y, z: ",3F4.1, "  Rotat. x, y, z: ",3F4.1)
1035 FORMAT("NLOOP=",I8," NTGRID=",I3," NRGRID=",I3," NCYC=",I4,/ &
       " NSTEP=",I4," TEMP=",F10.4," RATIOT=",F6.2," RATIOR=",F6.2)
1037 FORMAT(" LDDR=",L4)
  !  Calculate Initial energies
  ERIGT=ZERO
  ECORRT=ZERO
  loop100: DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     IF(LFIXED(IRIG))cycle loop100
     ERIGOLD=ZERO
     ECOLD=ZERO
     !  Calculate map matching energy before trial moving
     IF(IDEMP0 > 0)THEN
        CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
        IF(PRNLEV > 7)WRITE(OUTU,1050)0,IDRIG,CORRT
        ECOLD=COTOENG*CORRT
     ENDIF
     !  Calculate interaction energy with other rigid domains
     loop120: DO JRIG=1,NRIG
        IF(JRIG == IRIG)cycle loop120
        IDRIGJ=IDRIGS(JRIG)
        CALL EMAPERR(IDRIG,IDRIGJ,ECORE,EELE,ESOLV,ECONS)
        EFMAP=ECORE+EELE+ESOLV+ECONS
        ERIGOLD=ERIGOLD+EFMAP           
     enddo loop120
     IF(PRNLEV > 4)WRITE(OUTU,1010)IRIG,IDRIG,ERIGOLD,ECOLD
     ERIGT=ERIGT+ERIGOLD
     ECORRT=ECORRT+ECOLD
  enddo loop100
  IF(PRNLEV > 4)WRITE(OUTU,1021)
1021 FORMAT(" MC_N    ICYC    ECORRT     ERIGT  ", &
       "   RATT     RATR     DELTT     DELTR")
  IF(PRNLEV > 4)WRITE(OUTU,1002) &
       0,ECORRT,ERIGT,RATIOT,RATIOR,DELTT,DELTR
  IF(NLOOP <= 0)RETURN
  !  Perform NLOOP times of Many-body docking 
  loop200: do ICYC=1,NCYC
     NACCT=0
     NACCR=0
     loop220: DO ISTEP=1,NSTEP
        !   At each Loop, perform a GTMC for Each  RIGID domain 
        ERIGT=ZERO
        ECORRT=ZERO
        loop300: DO IRIG=1,NRIG
           IDRIG=IDRIGS(IRIG)
           IDEMPS=IDEMRIG(IDRIG)
           !   Skip fixed rigid domains
           IF(LFIXED(IRIG))cycle loop300
           !   Save current transform
           CALL EMAPSAVE(IDRIG)
           ERIGOLD=ZERO
           ECOLD=ZERO
           !  Calculate map matching energy before trial moving
           IF(IDEMP0 > 0)THEN
              CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
              IF(PRNLEV > 7)WRITE(OUTU,1050)ILOOP,IDRIG,CORRT
              ECOLD=COTOENG*CORRT
           ENDIF
           !  Calculate interaction energy with other rigid domains
           loop400: DO JRIG=1,NRIG
              IF(JRIG == IRIG)cycle loop400
              IDRIGJ=IDRIGS(JRIG)
              CALL EMAPERR(IDRIG,IDRIGJ,ECORE,EELE,ESOLV,ECONS)
              EFMAP=ECORE+EELE+ESOLV+ECONS
              ERIGOLD=ERIGOLD+EFMAP           
           enddo loop400
           DELTTX=DELTT*TRNEMAP(1,IDEMPS)
           DELTTY=DELTT*TRNEMAP(2,IDEMPS)
           DELTTZ=DELTT*TRNEMAP(3,IDEMPS)
           DELTRX=DELTR*ROTEMAP(1,IDEMPS)
           DELTRY=DELTR*ROTEMAP(2,IDEMPS)
           DELTRZ=DELTR*ROTEMAP(3,IDEMPS)
           DO ILOOP=1,NLOOP
              ! TRANSLATION
              TX=(ONE-TWO*RANDOM(ISEED))*DELTTX
              TY=(ONE-TWO*RANDOM(ISEED))*DELTTY
              TZ=(ONE-TWO*RANDOM(ISEED))*DELTTZ
              CALL EMAPTRNIN(IDRIG,TX,TY,TZ)
              ERIGNEW=ZERO
              ECNEW=ZERO
              !  Calculate NEW map matching energy before trial moving
              IF(IDEMP0 > 0)THEN
                 CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
                 IF(PRNLEV > 7)WRITE(OUTU,1050)ILOOP,IDRIG,CORRT
                 ECNEW=COTOENG*CORRT
              ENDIF
              !  Calculate NEW interaction energy with other rigid domains
              loop420: DO JRIG=1,NRIG
                 IF(JRIG == IRIG)cycle loop420
                 IDRIGJ=IDRIGS(JRIG)
                 CALL EMAPERR(IDRIG,IDRIGJ,ECORE,EELE,ESOLV,ECONS)
                 EFMAP=ECORE+EELE+ESOLV+ECONS
                 ERIGNEW=ERIGNEW+EFMAP           
              enddo loop420
              DECORR=ECNEW+ERIGNEW-ECOLD-ERIGOLD
              IF(DECORR > ZERO)THEN
                 FACT=EXP(-DECORR/TEMP)
                 RAND01=RANDOM(ISEED)
                 IF(FACT < RAND01)THEN
                    CALL EMAPRESTORE(IDRIG)
                    GOTO 440
                 ENDIF
              ENDIF
              NACCT=NACCT+1
              ERIGOLD=ERIGNEW
              ECOLD=ECNEW
              CALL EMAPSAVE(IDRIG)
440           CONTINUE
              ! ROTATION
              TX=(ONE-TWO*RANDOM(ISEED))*DELTRX
              TY=(ONE-TWO*RANDOM(ISEED))*DELTRY
              TZ=(ONE-TWO*RANDOM(ISEED))*DELTRZ
              CALL EMAPROT(IDRIG,ONE,ZERO,ZERO,TX)
              CALL EMAPROT(IDRIG,ZERO,ONE,ZERO,TY)
              CALL EMAPROT(IDRIG,ZERO,ZERO,ONE,TZ)
              ERIGNEW=ZERO
              ECNEW=ZERO
              !  Calculate NEW map matching energy before trial moving
              IF(IDEMP0 > 0)THEN
                 CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
                 IF(PRNLEV > 7)WRITE(OUTU,1050)ILOOP,IDRIG,CORRT
                 ECNEW=COTOENG*CORRT
              ENDIF
              !  Calculate NEW interaction energy with other rigid domains
              loop460: DO JRIG=1,NRIG
                 IF(JRIG == IRIG)cycle loop460
                 IDRIGJ=IDRIGS(JRIG)
                 CALL EMAPERR(IDRIG,IDRIGJ,ECORE,EELE,ESOLV,ECONS)
                 EFMAP=ECORE+EELE+ESOLV+ECONS
                 ERIGNEW=ERIGNEW+EFMAP           
              enddo loop460
              DECORR=ECNEW+ERIGNEW-ECOLD-ERIGOLD
              IF(DECORR > ZERO)THEN
                 FACT=EXP(-DECORR/TEMP)
                 RAND01=RANDOM(ISEED)
                 IF(FACT < RAND01)THEN
                    CALL EMAPRESTORE(IDRIG)
                    GOTO 480
                 ENDIF
              ENDIF
              NACCR=NACCR+1
              ERIGOLD=ERIGNEW
              ECOLD=ECNEW
              CALL EMAPSAVE(IDRIG)
480           CONTINUE
              !  End of NLOOP for this rigid domain
           ENDDO
           ERIGT=ERIGT+ERIGOLD
           ECORRT=ECORRT+ECOLD
        enddo loop300
        !  End of NSTEP
     enddo loop220
     RATT=ONE*NACCT/(NSTEP*NRIG*NLOOP)
     RATR=ONE*NACCR/(NSTEP*NRIG*NLOOP)
     IF(PRNLEV > 4)WRITE(OUTU,1002) &
          ICYC,ECORRT,ERIGT,RATT,RATR,DELTT,DELTR
     IF(RATT > RATIOT)THEN
        DELTT=1.05*DELTT
     ELSE
        DELTT=0.95*DELTT
     ENDIF
     IF(RATR > RATIOR)THEN
        DELTR=1.05*DELTR
     ELSE
        DELTR=0.95*DELTR
     ENDIF
     !  End of NCYC
  enddo loop200
  IF(PRNLEV > 2) &
       WRITE(OUTU,'("DOWN WITH FITTING")')
  ERIGT=ZERO
  ECORRT=ZERO
  loop800: DO IRIG=1,NRIG
     IDRIG=IDRIGS(IRIG)
     IF(LFIXED(IRIG))cycle loop800
     ERIGOLD=ZERO
     ECOLD=ZERO
     !  Calculate map matching energy before trial moving
     IF(IDEMP0 > 0)THEN
        CALL EMAPCRM(IDEMP0,IDRIG,LDDR,LCORE,CORRT)
        IF(PRNLEV > 7)WRITE(OUTU,1050)ILOOP,IDRIG,CORRT
        ECOLD=COTOENG*CORRT
     ENDIF
     !  Calculate interaction energy with other rigid domains
     loop820: DO JRIG=1,NRIG
        IF(JRIG == IRIG)cycle loop820
        IDRIGJ=IDRIGS(JRIG)
        CALL EMAPERR(IDRIG,IDRIGJ,ECORE,EELE,ESOLV,ECONS)
        EFMAP=ECORE+EELE+ESOLV+ECONS
        ERIGOLD=ERIGOLD+EFMAP           
     enddo loop820
     IF(PRNLEV > 4)WRITE(OUTU,1010)IRIG,IDRIG,ERIGOLD,ECOLD
     ERIGT=ERIGT+ERIGOLD
     ECORRT=ECORRT+ECOLD
  enddo loop800
  IF(PRNLEV > 4)WRITE(OUTU,1020)ERIGT,ECORRT
  IF(IDEMP0 > 0)THEN
     !   Create a scratch map  IDEMPS for correlation calculation
     NDATA=LEMAPX(IDEMP0)*LEMAPY(IDEMP0)*LEMAPZ(IDEMP0)
     NEWEMP=CHKEMPNM('SCRATCH',7,IDEMPS)
     IF(NEWEMP)THEN
        CALL EMAPDUP(IDEMP0,IDEMPS)
     ELSE
        CALL WRNDIE(0,'<EMAPGTMC N>','CANNOT CREATE EMAP: SCRATCH')
     ENDIF
     CALL EMAPSUM(IDEMPS,NRIG,IDRIGS)
     CALL EMAPCMM(IDEMP0,IDEMPS,LDDR,LCORE,CORRT)
     IF(PRNLEV > 4)WRITE(OUTU,1003)CORRT
     CALL RMEMAP(IDEMPS)
  ENDIF
1002 FORMAT(I10,F14.2,F14.2,6F8.2)
1003 FORMAT("AFTER FMCN FITTING, CORRMAX=",E14.6)
1005 FORMAT("AFTER FMCN FITTING, EFMAPT=",E14.6)
1010 FORMAT(" IRIG: ",I4," RIGID: ",I3,"  ERIG=",E14.6," ECORR=",E14.6)
1020 FORMAT(" TOTAL:  ERIGT=",E14.6," ECORRT=",E14.6)
1050 FORMAT(" At ILOOP=",I4," RIGID ",I3,"  CORRT=",2E14.6)
  RETURN
END SUBROUTINE FMAPMCN


SUBROUTINE EMAPSGRID(IDEMP,ngrid,NST,ST,  &
            SITE,SANGLE,REX1,REX2,REY1,REY2,REZ1,REZ2)
  !-----------------------------------------------------------------------
  !   This routine find the vectors from center of the map to the core surface
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  use stream
  use emapmod
  implicit none
  INTEGER IDEMP,NGRID,NST
  REAL(CHM_REAL) ST(3,NGRID*NGRID*NGRID)
  REAL(CHM_REAL) SITE(3),SANGLE
  REAL(CHM_REAL) REX1,REX2,REY1,REY2,REZ1,REZ2
  CALL EMAPSGRID1(MEMAPX(IDEMP),MEMAPY(IDEMP),MEMAPZ(IDEMP), &
       LEMAPX(IDEMP),LEMAPY(IDEMP),LEMAPZ(IDEMP), &
       DEMAPX(IDEMP),DEMAPY(IDEMP),DEMAPZ(IDEMP), &
       CEMAPX(IDEMP),CEMAPY(IDEMP),CEMAPZ(IDEMP), &
       empgrd(IDEMP)%core,ngrid,NST,ST, &
       SITE,SANGLE,REX1,REX2,REY1,REY2,REZ1,REZ2)
  RETURN
END SUBROUTINE EMAPSGRID

SUBROUTINE EMAPSGRID1(MX,MY,MZ,NX,NY,NZ,DX,DY,DZ,CX,CY,CZ, &
     CORE, ngrid,NST,ST,SITE,SANGLE,REX1,REX2,REY1,REY2,REZ1,REZ2)
  !-----------------------------------------------------------------------
  !   This routine find the vectors from center of the map to the core surface
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  implicit none
  !
  INTEGER NGRID,NST,CORE(*)
  INTEGER MX,MY,MZ,NX,NY,NZ
  REAL(CHM_REAL) DX,DY,DZ,CX,CY,CZ
  REAL(CHM_REAL) ST(3,NGRID*NGRID*NGRID)
  REAL(CHM_REAL) SITE(3),SANGLE,ANG
  REAL(CHM_REAL) REX1,REX2,REY1,REY2,REZ1,REZ2
  INTEGER I,J,K,IX,IY,IZ,IXYZ,NXYZMAP
  REAL(CHM_REAL) D,R,R0,R1,RR,SS,PROD
  REAL(CHM_REAL) XI,YI,ZI,XJ,YJ,ZJ,XR,YR,ZR
      XJ=SITE(1)-CX
  YJ=SITE(2)-CY
  ZJ=SITE(3)-CZ
  SS=SQRT(XJ*XJ+YJ*YJ+ZJ*ZJ)
  XJ=XJ/SS
  YJ=YJ/SS
  ZJ=ZJ/SS
  IF(NGRID.EQ.1)THEN
    D=ONE
  ELSE
    D=ONE/(NGRID-1)
  ENDIF
  R=(DX+DY+DZ)/THREE
  NST=0
  DO I=1,NGRID
    XI=(I-1)*D-HALF
    DO J=1,NGRID
      YI=(J-1)*D-HALF
      loop900: DO K=1,NGRID
        ZI=(K-1)*D-HALF
     !Get rid of non-surface grid points
        IF((I-1)*(I-NGRID)*(J-1)*(J-NGRID) &
            *(K-1)*(K-NGRID).NE.0)cycle loop900
     ! get rid of out of angle range grid points
        PROD=XI*XJ+YI*YJ+ZI*ZJ
        RR=SQRT(XI*XI+YI*YI+ZI*ZI)
        ANG=ACOS(PROD/RR)*180.0/PI
        IF(ABS(ANG).GT.SANGLE)cycle loop900
        R0=SQRT(XI*XI+YI*YI+ZI*ZI)
        R1=(NX*DX+NY*DY+NZ*DZ)/THREE
100         IX=NINT((CX+XI*R1/R0)/DX)-MX
        IF(IX*(NX-IX-1).LT.0)GOTO 500
        IY=NINT((CY+YI*R1/R0)/DY)-MY
        IF(IY*(NY-IY-1).LT.0)GOTO 500
        IZ=NINT((CZ+ZI*R1/R0)/DZ)-MZ
        IF(IZ*(NZ-IZ-1).LT.0)GOTO 500
        IXYZ=NXYZMAP(IX,IY,IZ,NX,NY,NZ)
        IF(CORE(IXYZ).GT.0)GOTO 600
500     R1=R1-R
        IF(R1.GT.R)GOTO 100
600     CONTINUE
    ! get rid of out of subset grid points
        XR=XI*R1/R0
        YR=YI*R1/R0
        ZR=ZI*R1/R0
        IF ((XR+CX) .LT. REX1)cycle loop900
        IF ((XR+CX) .GT. REX2)cycle loop900
        IF ((YR+CY) .LT. REY1)cycle loop900
        IF ((YR+CY) .GT. REY2)cycle loop900
        IF ((ZR+CZ) .LT. REZ1)cycle loop900
        IF ((ZR+CZ) .GT. REZ2)cycle loop900
        NST=NST+1
        ST(1,NST)=XR
        ST(2,NST)=YR
        ST(3,NST)=ZR
      enddo loop900
    ENDDO
  ENDDO
  RETURN
END SUBROUTINE EMAPSGRID1
#endif /* (emap)*/

subroutine emapdockdummy()
  return
end subroutine emapdockdummy

