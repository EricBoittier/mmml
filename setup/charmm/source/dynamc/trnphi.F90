SUBROUTINE MONDIH(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE MONITORS DIHEDRAL TRANSITIONS FROM A
  !     DYNAMICS RUN.
  !
  !      CAROL POST   10/7/83
  !
  use chm_kinds
  use dimens_fcm
  use param
  use psf
  use stream
  use string
  use memory
  use number
  use select
  implicit none
  !
  integer,allocatable,dimension(:) :: ISLCT,IPS,JPS,KPS,LPS,ICPS, &
       NTRNS,INDEX,IFREAT
  real(chm_real),allocatable,dimension(:) :: GDAT,GOLD,PNOW,POLD,PCUM
  real(chm_real4),allocatable,dimension(:) :: ITEMP
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  INTEGER NFU,NUNIT
  INTEGER NBEGN,NSTOP,NSKIP,IOUNIT,ISHOW
  !
  call chmalloc('trnphi.src','MONDIH','ISLCT',NATOM,intg=ISLCT)
  call chmalloc('trnphi.src','MONDIH','IPS',NPHI,intg=IPS)
  call chmalloc('trnphi.src','MONDIH','JPS',NPHI,intg=JPS)
  call chmalloc('trnphi.src','MONDIH','KPS',NPHI,intg=KPS)
  call chmalloc('trnphi.src','MONDIH','LPS',NPHI,intg=LPS)
  call chmalloc('trnphi.src','MONDIH','ICPS',NPHI,intg=ICPS)
  call chmalloc('trnphi.src','MONDIH','GDAT',NPHI,crl=GDAT)
  call chmalloc('trnphi.src','MONDIH','GOLD',NPHI,crl=GOLD)
  call chmalloc('trnphi.src','MONDIH','PNOW',NPHI,crl=PNOW)
  call chmalloc('trnphi.src','MONDIH','POLD',NPHI,crl=POLD)
  call chmalloc('trnphi.src','MONDIH','PCUM',NPHI,crl=PCUM)
  call chmalloc('trnphi.src','MONDIH','NTRNS',NPHI,intg=NTRNS)
  call chmalloc('trnphi.src','MONDIH','INDEX',NPHI,intg=INDEX)
  call chmalloc('trnphi.src','MONDIH','IFREAT',NATOM,intg=IFREAT)
  call chmalloc('trnphi.src','MONDIH','ITEMP',NATOM,cr4=ITEMP)

  CALL SELCTA(COMLYN,COMLEN,ISLCT,(/ZERO/),(/ZERO/),(/ZERO/),(/ZERO/),.FALSE.)
  !
  NFU=GTRMI(COMLYN,COMLEN,'FIRS',-1)
  NUNIT=GTRMI(COMLYN,COMLEN,'NUNI',1)
  NBEGN=GTRMI(COMLYN,COMLEN,'BEGI',0)
  NSTOP=GTRMI(COMLYN,COMLEN,'STOP',0)
  NSKIP=GTRMI(COMLYN,COMLEN,'SKIP',1)
  IOUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  ISHOW=0
  IF (INDXA(COMLYN,COMLEN,'SHOW') > 0) THEN
     ISHOW=1
     IF (INDXA(COMLYN,COMLEN,'ALL') > 0) ISHOW=2
  ENDIF
  IF(PRNLEV < 2) ISHOW=0
  !
  CALL MONDH2(ISLCT,IPS,JPS,KPS, &
       LPS,ICPS,GDAT,GOLD, &
       PNOW,POLD,PCUM, &
       NTRNS,INDEX,IFREAT, ITEMP, &
       IOUNIT,ISHOW,NFU,NUNIT,NBEGN,NSTOP,NSKIP)
  call chmdealloc('trnphi.src','MONDIH','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('trnphi.src','MONDIH','IPS',NPHI,intg=IPS)
  call chmdealloc('trnphi.src','MONDIH','JPS',NPHI,intg=JPS)
  call chmdealloc('trnphi.src','MONDIH','KPS',NPHI,intg=KPS)
  call chmdealloc('trnphi.src','MONDIH','LPS',NPHI,intg=LPS)
  call chmdealloc('trnphi.src','MONDIH','ICPS',NPHI,intg=ICPS)
  call chmdealloc('trnphi.src','MONDIH','GDAT',NPHI,crl=GDAT)
  call chmdealloc('trnphi.src','MONDIH','GOLD',NPHI,crl=GOLD)
  call chmdealloc('trnphi.src','MONDIH','PNOW',NPHI,crl=PNOW)
  call chmdealloc('trnphi.src','MONDIH','POLD',NPHI,crl=POLD)
  call chmdealloc('trnphi.src','MONDIH','PCUM',NPHI,crl=PCUM)
  call chmdealloc('trnphi.src','MONDIH','NTRNS',NPHI,intg=NTRNS)
  call chmdealloc('trnphi.src','MONDIH','INDEX',NPHI,intg=INDEX)
  call chmdealloc('trnphi.src','MONDIH','IFREAT',NATOM,intg=IFREAT)
  call chmdealloc('trnphi.src','MONDIH','ITEMP',NATOM,cr4=ITEMP)
  RETURN
END SUBROUTINE MONDIH

SUBROUTINE MONDH2(ISLCT,IPS,JPS,KPS,LPS,ICPS,GDAT,GOLD, &
     PNOW,POLD,PCUM,NTRNS,INDEX,IFREAT,ITEMP, &
     IOUNIT,ISHOW,NFU,NUNIT,NBEGN,NSTOP,NSKIP)
  !-----------------------------------------------------------------------

#if KEY_CHEQ==1
  use cheq,only:qcg     
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use cvio
  use param
  use ctitla
  use psf
  use code
  use coord
  use comand
  use intcor2,only:geticv
  implicit none
  real(chm_real) GDAT(*),GOLD(*),PNOW(*),POLD(*),PCUM(*)
  real(chm_real) DELTA
  REAL(chm_real4)   ITEMP(*)
  INTEGER NTRNS(*),INDEX(*),IFREAT(*)
  INTEGER IPS(*),JPS(*),KPS(*),LPS(*),ICPS(*)
  INTEGER ISLCT(*)
  !
  INTEGER IOUNIT,ISHOW,NFU,NUNIT,NBEGN,NSTOP,NSKIP
  !
  character(len=4) HDRC,HDRD
  LOGICAL LOK
  real(chm_real) RJUN
  INTEGER NAT,NFREAT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF
  INTEGER NSAVV,IPHI,NPHIS,NINIT
  INTEGER JJ,IA,JA,KA,LA
  DATA HDRC,HDRD/'CORD','VELD'/
  !
  JJ=0
  DO  IPHI=1,NPHI
     IA=IP(IPHI)
     JA=JP(IPHI)
     KA=KP(IPHI)
     LA=LP(IPHI)
     LOK=(ISLCT(IA) == 1 .AND. ISLCT(JA).EQ.1 .AND. ISLCT(KA).EQ.1 &
          .AND. ISLCT(LA) == 1)
     IF (LOK) THEN
        JJ=JJ+1
        INDEX(JJ)=IPHI
     ENDIF
  enddo
  NPHIS=JJ
  !
  CALL SHFTIC(NPHIS,4,INDEX,ICPS,IPS,JPS,KPS,LPS,ICP,IP,JP,KP,LP)
  !
  CALL CODES([0],[0],ICPS,[0], &
       NATOM,IMOVE,IAC,0,[0],[0], &
       0,[0],[0],[0],NPHIS,IPS,JPS,KPS,LPS,0,[0],[0],[0],[0], &
       .FALSE.,0,                                     & ! DRUDE
#if KEY_CMAP==1
       [0],0,[0],[0],[0],[0],[0],[0],[0],[0], & 
#endif
       .TRUE.,.FALSE.)
  NAT=NATOM
  IUNIT=NFU
  ISTATS=1
  NINIT=1
200 CONTINUE
  !
  CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
       CG,QCG,                                    & 
#endif
       ITEMP,NAT,IFREAT,NFREAT,NFU,NUNIT,IUNIT,NFILE, &
       ISTEP,ISTATS,NDEGF,DELTA, &
       NBEGN,NSTOP,NSKIP,NSAVV,HDRC,HDRD, &
       TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
  IF(NPHIS == 0) RETURN
  !
  DO IPHI=1,NPHIS
     CALL GETICV(IPS(IPHI),JPS(IPHI),KPS(IPHI),LPS(IPHI),.FALSE., &
          RJUN,RJUN,GDAT(IPHI),RJUN,RJUN,X,Y,Z)
  enddo
  !
  !     MONITOR DIHEDRAL TRANSITIONS
  !
  CALL TRNPHI(GDAT,GOLD,PCUM,POLD,PNOW,CPB,CPD,ATYPE,RES, &
       IBASE,IPS,JPS,KPS,LPS,ICPS,NRES,NTRNS,NPHIS,ISTEP, &
       NINIT,IOUNIT,ISTATS,ISHOW)
  !
  NINIT=0
  IF (ISTATS >= 0) GOTO 200
  !
  RETURN
END SUBROUTINE MONDH2

SUBROUTINE TRNPHI(GDAT,GOLD,PCUM,POLD,PNOW,CPB,CPD,ATYPE,RES, &
     IBASE,IP,JP,KP,LP,ICP,NRES,NTRNS,NPHI,ISTEP, &
     NINIT,IOUNIT,ISTATS,ISHOW)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE MONITORS THE TRANSITIONS IN THE DIHEDRALS
  !     A transition is counted if the torsion angle GDAT(NPHI) goes
  !     from one well to another well between calls to TRNPHI.  A well
  !     is centered at a minimum in the torsion potential, with
  !     the upper limit 15 degrees before the adjacent maximum.   That is,
  !     there is a 30 degree region (maximum +- 15 degrees) which must be
  !     crossed before a transition is counted.
  !     The net rotation is determined and stored in PCUM(IPHI).  The
  !     algorithm assumes the dihedral angle changes by less than 150
  !     degrees between calls to TRNPHI.  NOTE: determining the net
  !     rotation is based on the sign as well as the absolute value
  !     assigned to the wells.  If these are changed, the net rotation
  !     may not be accurate.
  !     
  !      C. B. Post, Sept. 1983
  !
  use chm_kinds
  use number
  use consta
  use stream
  use chutil,only:getres
  !
  implicit none
  INTEGER IP(*),JP(*),KP(*),LP(*),ICP(*)
  INTEGER IBASE(*)
  INTEGER NTRNS(*)
  real(chm_real) GDAT(*),GOLD(*),PCUM(*),PNOW(*),POLD(*),CPB(*)
  character(len=*) ATYPE(*),RES(*)
  INTEGER CPD(*)
  INTEGER NRES,NPHI,ISTEP,NINIT,IOUNIT,ISTATS,ISHOW
  !
  real(chm_real) PINC,PNET,RSIGN,FLAG
  INTEGER IRES,NTRES,NTRAN,I,J,IC,IPER,IPR,IPHI,ICMX
  INTEGER IA,JA,KA,LA
  real(chm_real) TWOPID,PID,WID12,PHAS,PHID,PHIOLD,PHIDIF
  real(chm_real) A3L,A3H,A2LH,B2LH,B3LH,C2L,C2H,C3L,C3H,D2L,D2H
  DATA FLAG,WID12/999.D0,30.D0/
  !
  TWOPID=360.0
  PID=180.0
  !
  !     Limits of the wells for a torsion potential with a periodicity of 3.
  !
  A3L=WID12
  A3H=120.0-WID12
  B3LH=120.0+WID12
  C3L=-(120.0-WID12)
  C3H=-WID12
  !
  !     Limits of wells for a torsion potential with a periodicity of 2.
  !     Minima=0,180.0 degrees.  Maxima=+-90.0 degrees.
  !     Well A: (-90.0 to +90.0)-WID12  Well B: (+90.0 to -90.0)+WID12
  !     OR
  !     Minima=+-90.0 degrees.  Maxima=0,180.0 degrees.
  !     Well C: WID12 to (180.0-WID12)  Well D: -(180.0-WID12) to -WID12
  !
  A2LH=90.0-WID12
  B2LH=90.0+WID12
  C2L=WID12
  C2H=180.0-WID12
  D2L=-(180.0-WID12)
  D2H=-WID12
  !
  !     INITIALIZE ARRAYS IF FIRST TIME THROUGH
  IF(NINIT == 1) THEN
     DO J=1,NPHI
        NTRNS(J)=0
        PCUM(J)=0.0
        POLD(J)=FLAG
        GOLD(J)=GDAT(J)
     ENDDO
  ENDIF
  !
  !
15 FORMAT(' UNKNOWN FIRST ATOM OF DIHEDRAL',I5,'.'/ &
       ' ITS TRANSITIONS WILL NOT BE COUNTED')
25 FORMAT(' ONLY DIHEDRALS WITH 2 OR 3 PERIODICITY WILL BE COUNTED.'/ &
       '  DIHEDRAL',I4,1X,A4,3X,3(A4,'-'),A4,1X, &
       ' PERIODICITY',I3,' WILL NOT BE TABULATED.')
35 FORMAT(I7,2X,I5,6X,I4,2X,I3,1X,A4,2X,3(A4,'-'),A4,2X,3(F8.2,2X))
  !
  DO IPHI=1,NPHI
     IF (IP(IPHI) == 0) THEN
        IF (NINIT == 1) THEN
           IF(PRNLEV >= 2) WRITE(IOUNIT,15) IPHI
        ENDIF
        cycle
     ENDIF
     IC=ICP(IPHI)
     IF(IC == 0) cycle
     !brb..07-FEB-99 to handle multiple term dihedrals
     !brb        PHAS=CPB(IC)
     !brb        IPER=CPD(IC)
     !
     IPER=0
     ICMX=0
30   CONTINUE
     IPR=CPD(IC)
     IF (IPR < 0) THEN
        IF(IPER < -IPR) THEN
           IPER=-IPR
           ICMX=IC
        ENDIF
        IC=IC+1
        GOTO 30
     ENDIF
     IF(IPER <= IPR) THEN
        IPER=IPR
        ICMX=IC
     ENDIF
     IC=ICMX
     PHAS=CPB(IC)
     !brb..07-FEB-99
     PHID=GDAT(IPHI)
     IF((ABS(PHID)-PID) > 0.001) THEN
        IF(PRNLEV >= 2) THEN
           WRITE(IOUNIT,'(A)')  &
                ' TRNPHI> Dihedral angle is out of range.'
           WRITE(IOUNIT,'(A,2I10)')      ' IPHI,IC   = ',IPHI,IC
           WRITE(IOUNIT,'(A,F10.3,I10)') ' PHAS,IPER = ',PHAS,IPER
           WRITE(IOUNIT,'(A,2F10.3)')    ' PHID,PID  = ',PHID,PID
        ENDIF
        CALL WRNDIE(-3,'<TRNPHI>', &
             'PROBLEM WITH PHASE ARGUMENT OR DIHEDRAL ANGLE')
        RETURN
     ENDIF
     !
     !     Determine which well, if any, for dihedral IPHI of specified
     !     periodicity.  GDAT() contains angles in degrees, -180.<=phi<=+180.
     !     NOTE: deter. of net rotation depends on sign of well values. See
     !     above comments.
     !
     IF (IPER == 2) THEN
        !     minima at 0,180 degrees, or at +-90 degrees.
        !
        IF (ABS(PHAS-PI) > 0.001) THEN
           IF (PHID < C2H.AND.PHID > C2L) THEN
              PNOW(IPHI)=90.0
              IF (PHID < 90.0) THEN
                 RSIGN=-1.0
              ELSE
                 RSIGN=1.0
              ENDIF
           ELSE IF (PHID < D2H.AND.PHID > D2L) THEN
              PNOW(IPHI)=-90.0
              IF (PHID > -90.0) THEN
                 RSIGN=-1.0
              ELSE
                 RSIGN=1.0
              ENDIF
           ELSE
              PNOW(IPHI)=POLD(IPHI)
           ENDIF
        ELSE
           IF (ABS(PHID) < A2LH) THEN
              PNOW(IPHI)=0.0
              IF (PHID > 0.0) THEN
                 RSIGN=-1.0
              ELSE
                 RSIGN=1.0
              ENDIF
           ELSE IF (ABS(PHID) > B2LH) THEN
              PNOW(IPHI)=180.0
              RSIGN=1.0
           ELSE
              PNOW(IPHI)=POLD(IPHI)
           ENDIF
        ENDIF
        !
     ELSE IF (IPER == 3) THEN
        !     minima at +-60,180 degrees
        !RCZ  10/24/91 IF (ABS(PHAS-0.0) > 0.001)
        IF (ABS(PHAS) > 0.001 .AND. ABS(PHAS-PI).GT.0.001) THEN
           IF(PRNLEV >= 2) THEN
              WRITE(IOUNIT,'(A)')  &
                   ' TRNPHI> Dihedral phase is out of range.'
              WRITE(IOUNIT,'(A,2I10)')      ' IPHI,IC   = ',IPHI,IC
              WRITE(IOUNIT,'(A,F10.3,I10)') ' PHAS,IPER = ',PHAS,IPER
              WRITE(IOUNIT,'(A,2F10.3)')    ' PHID,PID  = ',PHID,PID
           ENDIF
           CALL WRNDIE(-3,'<TRNPHI>', &
                'PROBLEM WITH PHASE ARGUMENT OR DIHEDRAL ANGLE')
           RETURN
        ENDIF
        !
        IF (PHID < A3H .AND. PHID > A3L) THEN
           PNOW(IPHI)=60.0
           RSIGN=-1.0
        ELSE IF (ABS(PHID) > B3LH) THEN
           PNOW(IPHI)=180.0
           RSIGN=1.0
        ELSE IF (PHID < C3H.AND.PHID > C3L) THEN
           PNOW(IPHI)=-60.0
           IF (ABS(POLD(IPHI)) < 120.0) THEN
              RSIGN=-1.0
           ELSE
              RSIGN=1.0
           ENDIF
        ELSE
           PNOW(IPHI)=POLD(IPHI)
        ENDIF
        !
     ELSE 
        IF (NINIT == 1) THEN
           IA=IP(IPHI)
           JA=JP(IPHI)
           KA=KP(IPHI)
           LA=LP(IPHI)
           IRES=GETRES(JA,IBASE,NRES)
           IF(PRNLEV >= 2) WRITE(IOUNIT,25) IRES,RES(IRES), &
                ATYPE(IA),ATYPE(JA),ATYPE(KA),ATYPE(LA),IPER
        ENDIF
        GOTO 200
     ENDIF
     !
     !     except for first time through,
     !     see if current dihedral of IPHI is different from last one
     !
     IF (NINIT == 1 .OR. POLD(IPHI).EQ.FLAG) POLD(IPHI)=PNOW(IPHI)
     PHIOLD=POLD(IPHI)
     !
     IF (PNOW(IPHI) /= PHIOLD .AND. PHIOLD.NE.FLAG) THEN
        NTRNS(IPHI)=NTRNS(IPHI)+1
        PINC=SIGN(ONE,PHIOLD)
        PINC=RSIGN*PINC/IPER
        PCUM(IPHI)=PCUM(IPHI)+PINC
        IF (ISHOW >= 1) THEN
           IA=IP(IPHI)
           JA=JP(IPHI)
           KA=KP(IPHI)
           LA=LP(IPHI)
           IRES=GETRES(JA,IBASE,NRES)
           IF(PRNLEV >= 2) WRITE(IOUNIT,35) ISTEP,NTRNS(IPHI),IPHI, &
                IRES,RES(IRES),ATYPE(IA),ATYPE(JA),ATYPE(KA),ATYPE(LA), &
                PHID,PNOW(IPHI),PHIOLD
        ENDIF
        POLD(IPHI)=PNOW(IPHI)
     ENDIF
     !
     !     now correct GDAT for accumulation in the case of passing thru +-180.0
     !
200  CONTINUE
     PHIDIF=PHID-GOLD(IPHI)
     IF (.NOT.(PHIDIF <= PID)) THEN
210     CONTINUE
        PHIDIF=PHIDIF-TWOPID
        IF (.NOT.(PHIDIF <= PID)) GOTO 210
     ENDIF
     IF (.NOT.(PHIDIF >= -PID)) THEN
230     CONTINUE
        PHIDIF=PHIDIF+TWOPID
        IF (.NOT.(PHIDIF >= -PID)) GOTO 230
     ENDIF
     GDAT(IPHI)=PHIDIF+GOLD(IPHI)
     GOLD(IPHI)=GDAT(IPHI)
     !
300  CONTINUE
  enddo
  !
  !
  IF (NINIT == 1 .AND. ISHOW >= 1) WRITE(IOUNIT,45)
45 FORMAT(/,1X,' STEP  ',' TRNS. NO. ',' DIHE ', &
       ' RES:ATM2  ATM1-ATM2-ATM3-ATM4 ',' DIHE ANG ', &
       ' NEW MIN. ',' OLD MIN. ')
  !
  !
  IF(ISHOW == 2) THEN
     WRITE(IOUNIT,51) ISTEP
     DO IPHI=1,NPHI
        IA=IP(IPHI)
        JA=JP(IPHI)
        KA=KP(IPHI)
        LA=LP(IPHI)
        IRES=GETRES(JA,IBASE,NRES)
        WRITE(IOUNIT,52) NTRNS(IPHI),IPHI,IRES,RES(IRES), &
             ATYPE(IA),ATYPE(JA),ATYPE(KA),ATYPE(LA), &
             GDAT(IPHI),PNOW(IPHI)
        !
     ENDDO
  ENDIF
51 FORMAT(/' DIHEDRAL VALUES FOR STEP',I7)
52 FORMAT(1X,I5,6X,I4,2X,I3,1X,A4,2X,3(A4,'-'),A4,2X,2(F8.2,2X))
  !
  !
  IF (ISTATS < 0) THEN
     NTRES=0
     NTRAN=0
     IF(PRNLEV >= 2) WRITE(IOUNIT,55) ISTEP
     DO I=1,NPHI
        IF (NTRNS(I) > 0) THEN
           IA=IP(I)
           JA=JP(I)
           KA=KP(I)
           LA=LP(I)
           IRES=GETRES(JA,IBASE,NRES)
           PNET=TWOPID*PCUM(I)
           IF(PRNLEV >= 2) WRITE(IOUNIT,65)I,NTRNS(I),IRES,RES(IRES), &
                ATYPE(IA),ATYPE(JA),ATYPE(KA),ATYPE(LA),PNET,PNOW(I)
           NTRES=NTRES+1
           NTRAN=NTRAN+NTRNS(I)
        ENDIF
     ENDDO
     IF(PRNLEV >= 2) WRITE(IOUNIT,75) NTRES,NTRAN
  ENDIF
55 FORMAT(//,' TOTAL TRANSITIONS AFTER STEP=',I7,5X,' DIHEDRALS ', &
       'WITH NO TRANSITIONS WILL NOT BE LISTED.'/1X,'DIHE', &
       ' NO.TRNS.  RES:ATM2  ATM1-ATM2-ATM3-ATM4', &
       ' NET ROTATION ',' LAST TRANSITION TO:')
65 FORMAT(I5,I6,5X,I3,1X,A4,2X,3(A4,'-'),A4,F12.1,5X,2F8.1)
75 FORMAT(' NO. DIHEDRALS WITH TRANSITIONS: ',I4, &
       ' TOTAL NO. TRANSITIONS: ',I8/)
  !
  RETURN
END SUBROUTINE TRNPHI

