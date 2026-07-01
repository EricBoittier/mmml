module tps
  use chm_types
  use memory
  implicit none

  type(chm_xyz),pointer,dimension(:) :: trjcd1,trjcd2,trjvl1,trjvl2
  type(chm_xyz),save :: revvel
  integer,allocatable,dimension(:) :: trjidx
  integer,pointer,    dimension(:) :: itpsd1,itpsd2,htxold,htxnew,htxavg
  integer :: htxdir,htxcnt

  integer,allocatable,dimension(:) :: ATSHPR,ATSFPR,ATHFPR,ACSHPR,ACSFPR,ACHFPR


  integer :: ntpath,nsavp,npracc,mxshfn,ntfrac
  integer :: itprst,shotlo,shotln
  real(chm_real) :: tpsavfrac,pshoot,tfract,phalfp
  integer :: nunit,firstu,nbegin,nskip,nstop,vfirst,ifshot
  integer :: itpsun,itpspr,ihunit,ihfreq,nhsavf,ihprnf,nhstrt,accuni
  integer :: tpsdou,tpsdiu,pnsav

  integer nshftt, nshott, nshfta, nshota
  integer ishoot,nhalft,nhalfa
  integer shfnum, nxtsht
  integer :: pdir,tpsstp,nstsav,itpsst
  integer :: ntsavc,ntsavv
  integer nregsv,nhsavt, itpssv,itpssd
  integer IBASIN

  logical qtpusr,qhsamp
  logical qbasa1, qbasa2, qbasb1, qbasb2 ,shfton
  logical qshtmd , qhalfp, qhflag

contains
  
#if KEY_TPS==1 /*tps_main*/
  subroutine tps_parse(comlyn,comlen)
  use exfunc
  use number
  use stream
  use cvio,only:trjspc
  use string
    character(len=*) :: comlyn
    integer :: comlen
    !        The number of transition paths to calculate.
    NTPATH = GTRMI(COMLYN, COMLEN,'NTPA',0)
    !        The frequency of saving paths to the trajectory and velocity files
    NSAVP  = GTRMI(COMLYN, COMLEN,'NSAVP',0)
    !        The frequency of printing acceptance statistics
    NPRACC = GTRMI(COMLYN, COMLEN,'NPRA',0)
    !        The fraction by which to perturb the velocities during shooting.
    TPSaVFRac = GTRMF(COMLYN, COMLEN,'VFRA',ZERO)
    !        The fraction of time to shoot (as opposed to shift)
    PSHOOT = GTRMF(COMLYN, COMLEN,'PSHO',ONE)
    !        The maximum number of steps to change in shift
    MXSHFN = GTRMI(COMLYN, COMLEN,'IMXS',1)
    !        The amount to change TEMP each shoot move
    TFRACT = GTRMF(COMLYN, COMLEN,'TFRA',ONE)
    !        The number of times to change TEMP by TFRACT
    NTFRAC = GTRMI(COMLYN, COMLEN,'NTFR',NTPATH)
    !        The position of the restart file
    ITPRST = GTRMI(COMLYN, COMLEN,'IRST',0)
    !        Window to shoot from.  If user doesn't enter a value, we'll
    !        set the window to be the entire path in TPINIT
    SHOTLO = GTRMI(COMLYN, COMLEN,'ISLO',0)
    !        Half the length of the window
    SHOTLN = GTRMI(COMLYN, COMLEN,'ISLN',0)
    !        Information to read trj and vel file
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)
!LNI March 2011, c36a6. Changed keyword IFIRst to FIRStu to be consistent with all other
! trajectory specifications. Be nice and pick up the old syntax as well ... for now
    IF(FIRSTU < 0)THEN
       IF(PRNLEV >= 2) WRITE( OUTU,*) 'Keyword IFIRst is obsolete - use FIRStu'
       FIRSTU = GTRMI(COMLYN,COMLEN,'IFIR', -1)
    ENDIF
    !        The first velocity unit (all other variables taken from above call
    !        to TRJSPC)
    VFIRST = FIRSTU + NUNIT
    VFIRST = GTRMI(COMLYN,COMLEN,'VFIR',VFIRST)
    !        The first shooting point (-1 means choose randomly for trjs,
    !        NREGSV/2 for starting from structure)
    IFSHOT = GTRMI(COMLYN,COMLEN,'IFSH',-1)
    !        Fraction of shoots which are half paths
    PHALFP = GTRMF(COMLYN,COMLEN,'PHAL',ZERO)
    !        Do we use usersb information for coordinates?
    !        Note that this requires code in the USERSB routine
    QTPUSR = (INDXA(COMLYN,COMLEN,'USER')  >  0)
    !        Allows printout of the values of the order parameters each step
    ITPSUN = GTRMI(COMLYN,COMLEN,'ITPU',OUTU)
    ITPSPR = GTRMI(COMLYN,COMLEN,'ITPR',0)
    !        Sampling the H ensemble (path accepted if it EVER enters B)
    !        Since this is connected to determination of <h[x(t)]>,
    !        call it HSAMP
    QHSAMP = (INDXA(COMLYN,COMLEN,'HSAM')  >  0)
    !        Output file for h[x(t)] info
    IHUNIT = GTRMI(COMLYN,COMLEN,'IHUN',OUTU)
    IHFREQ = GTRMI(COMLYN,COMLEN,'IHFR',0)
    !        How often is h[x(t)] evaluated in each trj
    !        (-1 means same as NSAVC)
    NHSAVF = GTRMI(COMLYN,COMLEN,'NHSV',-1)
    !        Timestep for printing h[x(t)] (in terms of NHSAVF)
    IHPRNF = GTRMI(COMLYN,COMLEN,'IHPR',1)
    !        How soon do we start updating and printing HTXAVG
    NHSTRT = GTRMI(COMLYN,COMLEN,'NHST',0)
    !        Print out acceptance arrays
    ACCUNI = GTRMI(COMLYN,COMLEN,'ACCU',OUTU)
    !        Output for saved seeds for use with langevin dynamics.
    !        Zero means don't output
    TPSDOU = GTRMI(COMLYN,COMLEN,'SDUN',0)
    !        Read in set of seeds.  If zero, no seeds read, assign randomly
    TPSDIU = GTRMI(COMLYN,COMLEN,'SDIN',0)
    !        The number of saved coordinates in read-in dcd
    PNSAV = GTRMI(COMLYN,COMLEN,'PNSA',-1)

    return
  end subroutine tps_parse

  subroutine allocate_tps_trj(trj,nat,n)
    type(chm_xyz),pointer,dimension(:) :: trj
    integer i,nat,n

    allocate(trj(n))
    do i=1,n
       call chmalloc("tps.src","allocate_tps_trj","trj(i)%x", &
            nat,crl=trj(i)%x)
       call chmalloc("tps.src","allocate_tps_trj","trj(i)%y", &
            nat,crl=trj(i)%y)
       call chmalloc("tps.src","allocate_tps_trj","trj(i)%z", &
            nat,crl=trj(i)%z)
    enddo
    return
  end subroutine allocate_tps_trj

  subroutine deallocate_tps_trj(trj,nat,n)
    type(chm_xyz),pointer,dimension(:) :: trj
    integer i,nat,n

    do i=1,n
       call chmdealloc("tps.src","allocate_tps_trj","trj(i)%x", &
            nat,crl=trj(i)%x)
       call chmdealloc("tps.src","allocate_tps_trj","trj(i)%y", &
            nat,crl=trj(i)%y)
       call chmdealloc("tps.src","allocate_tps_trj","trj(i)%z", &
            nat,crl=trj(i)%z)
    enddo
    deallocate(trj)
    return
  end subroutine deallocate_tps_trj

  SUBROUTINE TPMAIN(NSTEP,JHSTRT,IGVOPT,XN,YN,ZN,VX, &
       VY,VZ,NATOM,ISEED, &
       TEMP,QALWRT, &
#if KEY_TSM==1
       BACKLS,                                & 
#endif
       DELTA,GAMMA, &
       NDEGF,ILANG)
    !
    !       This decides what kind of TPS move to make.
    !
    !       A. R. Dinner and M. F. Hagan
    !
  use clcg_mod,only:random
  use number
#if KEY_RXNCOR==1
  use rxncom                     
#endif
#if KEY_SMD==1
  use smd,only: smdini           
#endif
#if KEY_SMD==1
  use rxenemod,only: ascend      
#endif

    implicit none
    !
    INTEGER ILANG
    INTEGER NATOM, ISEED, NSTEP, JHSTRT, IGVOPT
#if KEY_TSM==1
    INTEGER BACKLS(*) 
#endif
    INTEGER NDEGF
    real(chm_real)  GAMMA(*),DELTA
    real(chm_real) :: XN(:), YN(:), ZN(:), VX(*), VY(*), VZ(*), TEMP
    LOGICAL QALWRT(6)
    !
    !       The initial path has to be generated before we can shift
    SHFTON = .FALSE.
    IF(PDIR  == 1) THEN
       QHFLAG = .FALSE.

       IF (TPSSTP  >  0 .AND. .NOT. QSHTMD) THEN
          SHFTON = RANDOM(ISEED)  >=  PSHOOT
       ENDIF
    ENDIF
    IF(SHFTON) THEN
       CALL TPSHFT(NSTEP,XN,YN, &
            ZN,VX,VY,VZ,NATOM,ISEED)
    ELSE
       CALL TPSHOT(NSTEP,IGVOPT,XN,YN,ZN, &
            VX,VY,VZ,NATOM,ISEED, &
            TEMP,QALWRT, &
#if KEY_TSM==1
            BACKLS,  & 
#endif
            DELTA,GAMMA, &
            NDEGF,ILANG)
    ENDIF

    !       Do not use delta x (u) vector for the first step.
    IGVOPT = 2

    !       Set JHSTRT to zero to avoid the initial energy check.
    JHSTRT = 0

    !       Save the initial starting position
    ITPSSV = ITPSST

#if KEY_SMD==1
    !       Initialize variables for SMD if necessary.
    CALL ASCEND
    CALL SMDINI(PDIR,NRXNCR,TREELO,DELVAL,DL0PTR, &
         SMDDEL,BASALO,BASAHI,BASBLO, &
         BASBHI,QFLIP,PUMBPR)
#endif 
    RETURN
  END SUBROUTINE TPMAIN

  SUBROUTINE TPSGET_p(NATOM,X,Y,Z,SAVARR,FACTOR)
    !
    !       Get a set of coordinates from the TPS format.
    !
    INTEGER NATOM
    type(chm_xyz) :: savarr
    real(chm_real)  X(*), Y(*), Z(*), FACTOR
    !
    INTEGER I
    
    X(1:natom) = SAVARR%x(1:natom)*FACTOR
    Y(1:natom) = SAVARR%y(1:natom)*FACTOR
    Z(1:natom) = SAVARR%z(1:natom)*FACTOR
    
    RETURN
  END SUBROUTINE TPSGET_P

  SUBROUTINE TPSSAV_p(NATOM,X,Y,Z,SAVARR,FACTOR)
    !
    !       Save a set of coordinates in the TPS format.
    !
    type(chm_xyz) :: savarr
    INTEGER NATOM
    real(chm_real)  X(*), Y(*), Z(*), FACTOR
    !
    INTEGER I

    SAVARR%x(1:natom) = X(1:natom)*FACTOR
    SAVARR%y(1:natom) = Y(1:natom)*FACTOR
    SAVARR%z(1:natom) = Z(1:natom)*FACTOR

    RETURN
  END SUBROUTINE TPSSAV_P

  SUBROUTINE TPRDCV_p(TRJ,NATOM,firstu, &
       X,Y,Z,VELCOD,DELTA, &
       TIMEST,MULT,ISTART,IEND)
    !
    !       Read from coordinate or velocity file to TRJCD1 or
    !       TRJVL1, respectively.
    !
  use number
  use ctitla
  use cvio
  use stream

    type(chm_xyz),pointer,dimension(:) :: trj
    INTEGER NATOM,firstu
    real(chm_real)  X(*),Y(*),Z(*)
    INTEGER VELCOD,ISEED
    INTEGER MULT, ISTART, IEND
    real(chm_real)  DELTA,TIMEST
    !
    INTEGER,allocatable,dimension(:) :: FREEAT
    real(chm_real4),allocatable,dimension(:) :: TEMP
    integer NAT,IUNIT,ISTATS
    INTEGER NFREAT,NFILE,ISTEP,NDEGF,NTOT
    INTEGER I,J,NSTEP,NSTOP2,SEEDTMP
    real(chm_real)  DELTA2
    character(len=4)  HDR1,HDR2
    !
    call chmalloc('tps.src','TPRDCV','FREEAT',NATOM,intg=FREEAT)
    call chmalloc('tps.src','TPRDCV','TEMP',NATOM,cr4=TEMP)

    IF (VELCOD == 1) THEN
       HDR1 = 'VELD'
       HDR2 = 'VELD'
    ELSE
       HDR1 = 'COOR'
       HDR2 = 'CORD'
    ENDIF

    NTOT = 0
    IUNIT = FIRSTU
    ISTATS = 1

    loop200: do while(ISTATS  >=  0)       ! 200 CONTINUE
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            (/ ZERO /), .FALSE., &  
#endif
            TEMP,NATOM,FREEAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,NFILE,ISTEP,ISTATS,NDEGF,DELTA2, &
            NBEGIN,NSTOP,NSKIP,ntsavv,HDR1,HDR2, &
            TITLEA,NTITLA,.FALSE., (/ ZERO /), .true.)

       !       Make some checks the first time through.
       IF (NTOT == 0) THEN
          !         Not going to allow more than one unit at this point
          IF (NUNIT /= 1) CALL WRNDIE(-5,'<TPRDCV>', &
               'This version does not accept NUNIT above 1')
          IF (IOLEV  >  0) WRITE (OUTU,818)  &
               'TPRDCV> NSKIP,NFILE,NREGSV,PNSAV',NSKIP,NFILE,NREGSV,PNSAV
818       FORMAT(1X,A31,1X,I6,1X,I6,1X,I6,1X,I6)
          !         5-13-09 M Hagan to avoid including non-existent frame in the trj
          IF (PNSAV.GT.NFILE) CALL WRNDIE(-5,'<TPRDCV>', &
               'Decrease PNSAV to match NFILE')
          !         the following line would also avoid problems
          !          IF (PNSAV.GT.NFILE) PNSAV = NFILE
       ENDIF

       !       Read such that the latest trajectory overwrites earlier ones in the file.  
       I = MOD(NTOT,PNSAV) + 1 
       !       Center shorter (old) trajectories in longer (new) arrays.
       IF (NREGSV  >  PNSAV) I = I + INT((NREGSV - PNSAV)*0.5)
       !       Save coordinates to proper step
       CALL TPSSAV_p(NATOM,X,Y,Z,TRJ(i),ONE)
       !       Fill in empty space at the ends if necessary.
       IF (NREGSV  >  PNSAV) THEN
          IF (I  ==  1 + INT((NREGSV - PNSAV)*0.5)) THEN
             DO J = 1, I-1
                CALL TPSSAV_p(NATOM,X,Y,Z,TRJ(j),ONE)
             ENDDO
          ELSEIF (I  ==  PNSAV + INT((NREGSV - PNSAV)*0.5)) THEN
             DO J = I+1, NREGSV
                CALL TPSSAV_p(NATOM,X,Y,Z,TRJ(j),ONE)
             ENDDO
          ENDIF
       ENDIF

       NTOT = NTOT + 1

    enddo loop200      ! IF (ISTATS  >=  0) GOTO 200

    IF (VELCOD  ==  1) THEN
       MULT = NSKIP/NTSAVV
       ISTART = NBEGIN/NTSAVV
       IEND   = NSTOP/NTSAVV
    ENDIF

    call chmdealloc('tps.src','TPRDCV','FREEAT',NATOM,intg=FREEAT)
    call chmdealloc('tps.src','TPRDCV','TEMP',NATOM,cr4=TEMP)

    RETURN
  END SUBROUTINE TPRDCV_p

  SUBROUTINE TPSCHK(QBASA1,QBASB1,QCALL,ITPSUN,ITPSPR,TPSSTP, &
       ITPSST,STRNG,PDIR)
    !
    !       Calculate the order parameters and check if they are in the
    !       basins.
    !
    !       This routine requires that the coordinates of interest be
    !       in the main X, Y, and Z arrays already.
    !
  use number
  use rxncom
#if KEY_RXNCOR==1
  use rxenemod,only: ascend 
#endif
  use stream

    INTEGER ITPSUN,ITPSPR,TPSSTP,ITPSST,PDIR
    LOGICAL QBASA1,QBASB1, QCALL
    CHARACTER(len=5) STRNG
    !
    INTEGER RA,RB,R
    real(chm_real)  DA, DB, PERIOD, TMPAHI, TMPALO, TMPBHI, TMPBLO
    !
    QBASA1= .TRUE.
    QBASB1= .TRUE.

    IF (QCALL) CALL ASCEND

    RA=1
    loop10: do while((RA <= NRXNCR).AND.QBASA1)
       DA = DELVAL(1,TREELO(RA))
       !         ARD and Jie HU 06-06-30 for periodicity and TPS
       PERIOD = PUMBPR(RA)
       TMPAHI = BASAHI(RA)
       TMPALO = BASALO(RA)
       QBASA1 = (DDPERI(DA     - TMPALO, PERIOD) * &
            DDPERI(TMPAHI - TMPALO, PERIOD)  >  ZERO) .AND. &
            (DDPERI(DA     - TMPAHI, PERIOD) * &
            DDPERI(TMPAHI - TMPALO, PERIOD)  <  ZERO)
       RA = RA + 1
    enddo loop10

    RB=1
    loop20: do while((RB <= NRXNCR).AND.QBASB1)
       DB = DELVAL(1,TREELO(RB))
       !         ARD and Jie HU 06-06-30 for periodicity and TPS
       PERIOD = PUMBPR(RB)
       TMPBHI = BASBHI(RB)
       TMPBLO = BASBLO(RB)
       QBASB1 = (DDPERI(DB     - TMPBLO, PERIOD) * &
            DDPERI(TMPBHI - TMPBLO, PERIOD)  >  ZERO) .AND. &
            (DDPERI(DB     - TMPBHI, PERIOD) * &
            DDPERI(TMPBHI - TMPBLO, PERIOD)  <  ZERO)
       RB = RB + 1
    enddo loop20

    !       Printout values of the order parameters
    IF (IOLEV > 0 .AND. ITPSUN.GT.0 .AND. ITPSPR.GT.0) THEN
       IF (MOD(TPSSTP,ITPSPR)  ==  0) THEN
          WRITE(ITPSUN,*)
          WRITE(ITPSUN,30) TPSSTP,ITPSST,STRNG,PDIR
          IF (NRXNCR  >  0) WRITE(ITPSUN,40) &
               (DELVAL(1,TREELO(R)),R=1,NRXNCR)
       ENDIF
    ENDIF
30  FORMAT('TPSCHK> ',I6,1X,I8,1X,A6,1X,I2)
40  FORMAT('RXNCOR> ',999F12.5)

    CALL TPUSER(QBASA1,QBASB1,QCALL)

    IF (IOLEV > 0 .AND. ITPSPR.GT.0) THEN
       IF (MOD(TPSSTP,ITPSPR)  ==  0) CALL GFLUSH(ITPSUN)
    ENDIF

    RETURN
  END SUBROUTINE TPSCHK

  SUBROUTINE TPCMMT()
    !
    !       Check whether the dynamics has reached a basin and add
    !       to IBASIN to count the time in a basin.
    !      
    !       IBASIN += -1 for A, IBASIN += 1 for B, IBASIN += 0 for neither.
    !
    !       Overlapping basins add 0.
    !
    !       IBASIN is initialized in the TPCINI call from DCNTRL.  Thus, all 
    !       points in a basin contribute to IBASIN regardless of how they are 
    !       spaced in time.
    !
    !       ARD and Ao Ma 05-06-30
    !

    !
    LOGICAL QBASB1,QBASA1

    CALL TPSCHK(QBASA1,QBASB1,.FALSE.,0,0,0,0,'     ',1)
    IF (QBASA1) IBASIN = IBASIN - 1
    IF (QBASB1) IBASIN = IBASIN + 1

    RETURN
  END SUBROUTINE TPCMMT

  SUBROUTINE HTXINI(HTX,X,Y,Z, &
       NATOM,ISEED)
    !
    !       Initialize  the H array.
    !
  use number
  use stream

    INTEGER HTX(*),NATOM
    INTEGER ISEED
    real(chm_real)  X(*),Y(*),Z(*)
    !
    INTEGER I,J,K
    INTEGER STEP
    LOGICAL FLAGT,QBASA1,QBASB1,QBASA2
    !
    HTXAVG(1:nhsavt) = 0
    !
    FLAGT = .FALSE.
    DO I = 1, NREGSV
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(i),ONE)
       STEP=(I-1)*NTSAVC
       J = STEP/NHSAVF+1
       !         Calculate the OP
       CALL TPSCHK(QBASA1,QBASB1,.TRUE.,ITPSUN,ITPSPR,0,STEP, &
            'HTXIN',1)
       !
       IF (QBASB1) THEN
          FLAGT=.TRUE.
          K=1
       ELSEIF(QBASA1) THEN
          K=-1
       ELSE
          K=0
       ENDIF
       !
       HTX(J) = K
    ENDDO
    QBASA1 = HTX(1)       ==  -1
    QBASA2 = HTX(NREGSV)  ==  -1
    CALL TPHDIR(QBASA1,QBASA2,ISEED)
    !
    IF (.NOT. FLAGT) THEN
       CALL WRNDIE(0,'<HTXINI>', &
            'Initial trajectory does not enter B')
       HTXDIR = 0
    ENDIF

    RETURN
  END SUBROUTINE HTXINI

  SUBROUTINE TPHCLC(TIMEST)
    !
    !       Updates and prints the HTXAVG array.
    !
  use number
  use stream

    real(chm_real)  TIMEST
    !
    INTEGER I1,I2,I3,I
    IF(TPSSTP  >=  NHSTRT) THEN
       IF(HTXDIR  ==  0) THEN
          !         Not ready to calculate the average HTX
          CALL WRNDIE(1,'<TPHCLC>','HTXDIR = 0 after HSTART')
       ELSE
          IF(HTXDIR  ==  1) THEN
             I1 = 1
             I2 = NHSAVT
          ELSE IF(HTXDIR  ==  -1) THEN
             I1 = NHSAVT
             I2 = 1
          ELSE
             WRITE(OUTU,*) 'TPHCLC>  HTXDIR ',HTXDIR
             CALL WRNDIE(-5,'<TPHCLC>','Bad HTXDIR value')
          ENDIF

          I3 = 0
          DO I=I1,I2,HTXDIR
             I3 = I3 + 1
             IF(HTXOLD(I)  ==  1) THEN
                HTXAVG(I3) = HTXAVG(I3) + 1
             ENDIF
          ENDDO
          HTXCNT = HTXCNT + 1
       ENDIF
       !         Print out
       IF(IOLEV > 0 .AND. IHFREQ .GT. 0 .AND. &
            MOD(TPSSTP,IHFREQ)  ==  0 .AND. HTXCNT  >  0) THEN
          WRITE(IHUNIT,*)
          WRITE(IHUNIT,1) 'STEP', TPSSTP, 'UPDATES', HTXCNT,'DIR ', &
               HTXDIR
          IF(IHPRNF  >  0) THEN
             DO I =1,NHSAVT,IHPRNF
                WRITE(IHUNIT,2) REAL((I-1)*NHSAVF)*TIMEST, &
                     HTXOLD(I*HTXDIR+(1-HTXDIR)/2*(NHSAVT + 1)), &
                     DBLE(HTXAVG(I))/DBLE(HTXCNT),I*HTXDIR+(1-HTXDIR)/ &
                     2*(NHSAVT + 1)
             ENDDO
1            FORMAT(1X,A4,1X,I6,1X,A9,1X,I6,1X,A4,1X,I2)
2            FORMAT(1X,F12.3,1X,I4,1X,F12.5,1X,I8)
          ENDIF
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE TPHCLC


  SUBROUTINE TPSRST(DOIT,X,Y,Z,VX,VY,VZ,NATOM,DONE, &
       ISVFRQ)
    !
    !       Determine whether to write a restart file and setup the
    !       appropriate data structures if so.
    !
  use number

    INTEGER NATOM, ISVFRQ
    real(chm_real)  X(*), Y(*), Z(*), VX(*), VY(*), VZ(*)
    LOGICAL DONE, DOIT
    !
    INTEGER N2

    DOIT = .FALSE.

    IF (.NOT. DONE)   RETURN
    IF (PDIR  ==  1 .AND. .NOT. SHFTON ) RETURN

    !       Make sure it is the ending phase of a move
    !       Write the file if we are done.
    DOIT = TPSSTP  ==  NTPATH
    !       Co-opt ISVFRQ to count paths instead of MD steps.
    IF (ISVFRQ  >  0) THEN
       IF (MOD(TPSSTP,ISVFRQ)  ==  0) DOIT = .TRUE.
    ENDIF
    IF (TPSSTP  ==  0) DOIT = .FALSE.
    IF (DOIT) THEN
       !         Setup the data structure so the right things are saved.
       N2 = TRJIDX(ITPRST)
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n2),ONE)
       CALL TPSGET_p(NATOM,VX,VY,VZ,TRJVL1(n2),ONE)
    ENDIF

    RETURN
  END SUBROUTINE TPSRST

  SUBROUTINE TPSWRI(tpsstp, &
       NATOM,FREEAT,NFREAT,NDEGF,DELTA, &
       TITLEA,NTITLA,IUNCRD,IUNVEL, &
       X,Y,Z)
    !
    !       Write to the trajectory and velocity files.
    !
  use number
  use stream
  use cvio
  use rxncom
#if KEY_RXNCOR==1
  use rxenemod, only: ascend   
#endif

    INTEGER NATOM,tpsstp
    INTEGER NFREAT, NDEGF, FREEAT(*), NTITLA, IUNCRD, IUNVEL
    INTEGER RA,TPOUTU
    real(chm_real)  DELTA,X(*),Y(*),Z(*)
    character(len=*) TITLEA(*)
    !
    INTEGER I, J, K, M, N, N2
    real(chm_real),allocatable,dimension(:) :: XS, YS, ZS
    real(chm_real)  VAL
    !
    IF (PDIR  ==  -1 .AND. .NOT. SHFTON .AND. .NOT. QHALFP) RETURN

    IF (NSAVP  >  0) THEN
       IF ((TPSSTP  >  0) .AND. (MOD(TPSSTP,NSAVP)  ==  0)) THEN
          !           Save original XYZ just in case
          call chmalloc('tps.src','TPSWRI','XS',NATOM,crl=XS)
          call chmalloc('tps.src','TPSWRI','YS',NATOM,crl=YS)
          call chmalloc('tps.src','TPSWRI','ZS',NATOM,crl=ZS)

          xs(1:natom)=x(1:natom)
          ys(1:natom)=y(1:natom)
          zs(1:natom)=z(1:natom)

          IF (NTSAVC  >  0) THEN
             M = NSTSAV + NTSAVC
             K = TPSSTP/NSAVP
             K = (K - 1)*M
             N = NSTSAV/NTSAVC + 1
             DO I = 1, N
                !               Number steps such that first frame is 1

                !               Coordinate write out
                N2 = TRJIDX(I)
                CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n2),ONE)

                !               Write values of the order parameters
                IF (allocated(TPOPUN)) THEN
                   CALL ASCEND
                   DO RA = 1,NRXNCR
                      VAL = DELVAL(1,TREELO(RA))
                      TPOUTU = TPOPUN(RA)
                      IF (IOLEV > 0 .AND. TPOUTU .GT. 0) &
                           WRITE(TPOUTU,25) TPSSTP,I,RA,VAL
25                    FORMAT('TPSWRI>  ',I6,1X,I6,1X,I4,1X,F18.9)
                   ENDDO
                ENDIF

                CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                     (/ ZERO /), .FALSE., &  
#endif
                     NATOM,FREEAT,NFREAT,NTSAVC,K+I*NTSAVC, &
                     NDEGF,DELTA,NTSAVC,(NTPATH/NSAVP)*M,TITLEA, &
                     NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))

             ENDDO

             !             Why doesn't this do anything!!!
             CALL GFLUSH(IUNCRD)
             IF (allocated(TPOPUN)) THEN
                DO RA = 1,NRXNCR
                   TPOUTU = TPOPUN(RA)
                   CALL GFLUSH(TPOUTU)
                ENDDO
             ENDIF

             !             If desired, write out seeds for Langevin
             IF (TPSDOU > 0 .AND. IOLEV.GT.0) THEN
                WRITE(TPSDOU,50) (J,ITPSD1(J),J=1,N)
50              FORMAT (I6,1X,I12)
             ENDIF

          ENDIF
          IF (NTSAVV  >  0) THEN
             M = NSTSAV + NTSAVV
             K = TPSSTP/NSAVP
             K = (K - 1)*M
             N = NSTSAV/NTSAVV + 1
             DO I = 1, N

                !               Velocity write out
                N2 = TRJIDX(I)
                CALL TPSGET_p(NATOM,X,Y,Z,TRJVL1(n2),ONE)
                CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                     (/ ZERO /), .FALSE., &  
#endif
                     NATOM,FREEAT,NFREAT,NTSAVV,K+I*NTSAVV, &
                     NDEGF,DELTA,NTSAVV,(NTPATH/NSAVP)*M,TITLEA, &
                     NTITLA,IUNVEL,.TRUE.,.FALSE., (/ 0 /), .FALSE., (/ ZERO /))
             ENDDO
             CALL GFLUSH(IUNVEL)
          ENDIF

          !           Return original XYZ
          X(1:natom)=XS(1:natom)
          Y(1:natom)=YS(1:natom)
          Z(1:natom)=ZS(1:natom)
          call chmdealloc('tps.src','TPSWRI','XS',NATOM,crl=XS)
          call chmdealloc('tps.src','TPSWRI','YS',NATOM,crl=YS)
          call chmdealloc('tps.src','TPSWRI','ZS',NATOM,crl=ZS)

       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE TPSWRI

  SUBROUTINE TPRDSD(MULT,ISTART,IEND)
    !
    !       Read in seeds to use for DLNGV during 2-step start.  Doing so
    !       is necessary because the 2-step start will not return the right
    !       displacement vectors unless the seed is the same as that used
    !       during the original run (since the forces, including the random
    !       components, are used to back calculate the displacements).
    !
    !       If no seed file is supplied, the seeds are assigned randomly.
    !
  use clcg_mod,only:random
  use reawri
    implicit none
    !
    INTEGER MULT,ISTART,IEND
    !
    INTEGER I,J,K,N
    !
    IF (TPSDIU > 0) THEN
       I = 0
       N = 0
10     READ(TPSDIU,*,END=20) J, K
       I = I + 1
       IF (I  >=  ISTART .AND. I  <=  IEND) THEN
          IF (MOD(I,MULT)  ==  0) THEN
             ITPSD1(MOD(N,NREGSV)+1) = K
             N = N + 1
          ENDIF
       ENDIF
       GOTO 10
20     RETURN
    ENDIF

    DO I=1,NREGSV
       ITPSD1(I) = ISEED
       J = RANDOM(ISEED)
    ENDDO
    RETURN

  END SUBROUTINE TPRDSD

  SUBROUTINE TPSUPD(NATOM,X,Y,Z,VX,VY,VZ)
    !
    !       Update the stored coordinates for shooting.
    !
  use number
  use parallel

    INTEGER NATOM
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  VX(*), VY(*), VZ(*)
    !
    INTEGER I
    !
    !       See if this is a regular saving interval
    IF (NTSAVC  >  0) THEN
       IF (MOD(ITPSST,NTSAVC) == 0) THEN
          I = (ITPSST/NTSAVC) + 1
#if KEY_PARALLEL==1
          CALL VDGBR(X,Y,Z,1)                        
#endif
          CALL TPSSAV_p(NATOM,X,Y,Z,TRJCD2(i),ONE)
          !           Also save the seed
          ITPSD2(I) = ITPSSD
       ENDIF
    ENDIF
    IF (NTSAVV  >  0) THEN
       IF (MOD(ITPSST,NTSAVV) == 0) THEN
          I = (ITPSST/NTSAVV) + 1
#if KEY_PARALLEL==1
          CALL VDGBR(VX,VY,VZ,1)                     
#endif
          CALL TPSSAV_p(NATOM,VX,VY,VZ,TRJVL2(i),-DBLE(PDIR))
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE TPSUPD

  SUBROUTINE TPHUPD()
    !
    !       Update the HTXNEW array
    !
  use number

    INTEGER I,J
    LOGICAL QBASB1,QBASA1
    !
    IF(.NOT. QHSAMP) THEN
       RETURN
    ELSE IF(MOD(ITPSST,NHSAVF) == 0) THEN
       !         If it is a regular saving interval, calculate order parameter.
       CALL TPSCHK(QBASA1,QBASB1,.FALSE.,0,0,0,0,'     ',1)
       I = ITPSST / NHSAVF + 1
       CALL TPHASN(QBASA1,QBASB1,I)
    ENDIF

    RETURN
  END SUBROUTINE TPHUPD

  SUBROUTINE TPHASN(QBASA1,QBASB1,INDEX)
    !
    !       Save the value of H[x(t)].
    !

    INTEGER INDEX
    LOGICAL QBASA1,QBASB1
    !
    IF (QBASB1) THEN
       QHFLAG = .TRUE.
       HTXNEW(INDEX) = 1
    ELSEIF (QBASA1) THEN
       HTXNEW(INDEX) = -1
    ELSE
       HTXNEW(INDEX) = 0
    ENDIF

    RETURN
  END SUBROUTINE TPHASN

  SUBROUTINE TPHFLG(HARRAY,I,I2)
    !
    !       Checks the designated range of HTXOLD for a 1
    !

    INTEGER HARRAY(*),I,I2
    !
    LOGICAL QTEMP
    !
    QTEMP = .FALSE.
10  IF (I <= I2 .AND. .NOT. QTEMP) THEN
       QTEMP = (HARRAY(I)  ==  1)
       I = I + 1
       GOTO 10
    ENDIF

    IF(QTEMP) QHFLAG = .TRUE.

    RETURN
  END SUBROUTINE TPHFLG

  SUBROUTINE TPCINI()
    !
    !       Initializations for commitor probability statistics.
    ! 
  use rxncom

    !
    IBASIN = 0
    RXNIND = 1
    !
    return
  end SUBROUTINE TPCINI

  SUBROUTINE TPSHOT(NSTEP,IGVOPT,X,Y,Z, &
       VX,VY,VZ,NATOM,ISEED,TEMP,QALWRT, &
#if KEY_TSM==1
       BACKLS,  & 
#endif
       DELTA,GAMMA,NDEGF,ILANG)
    !
    !       This is the shooting routine for TPS sampling
    !
    !       It determines the direction of move and then sets up the data
    !       structures for the call to DYNAMC.
    !
    !       Aaron R. Dinner and M. F. Hagan
    !
  use clcg_mod,only:random
  use number
  use stream
  use parallel

    INTEGER ILANG
    INTEGER NATOM, ISEED, NSTEP, IGVOPT
    INTEGER NDEGF
#if KEY_TSM==1
    INTEGER BACKLS(*) 
#endif
    real(chm_real) :: X(:), Y(:), Z(:), VX(*), VY(*), VZ(*), TEMP
    real(chm_real)  GAMMA(*), DELTA
    LOGICAL QALWRT(6)
    !
    INTEGER N2
    real(chm_real)  F, VFCTMP
    LOGICAL QSCALE
    !
    !CC        write(*,*)'TPSHOT>init,me,pdir,qshtmd,qhalfp=',
    !CC     $       mynod,pdir,qshtmd,qhalfp
    !
    !     PDIR is random dependent!
    !     broadcast PDIR, so everybody has the same as process zero
    !     NOTE: this might not be the best place to do this broadcast!!!
    !
#if KEY_PARALLEL==1
    CALL PSND4(PDIR,1)                      
#endif
    !
    IF (PDIR  ==  1) THEN
       !         Initial shoot

       !         Get a random starting point
       !         It may be more optimal to use a Gaussian distribution
       !         It is possible to pick the same point twice in a row, but
       !         the velocity perturbations are also random
       IF (QSHTMD) THEN
          ISHOOT = IFSHOT
       ELSE
          ISHOOT = NXTSHT
       ENDIF

       NXTSHT = INT(RANDOM(ISEED)*(SHOTLN)) + SHOTLO
       ITPSST = ISHOOT*NTSAVC

       !         What kind of shoot?
       IF(QSHTMD .OR. RANDOM(ISEED)  >=  PHALFP) THEN
          QHALFP = .FALSE.
          !           Flip the direction flag
          PDIR = -1
          NSTEP = NSTSAV - ITPSST
       ELSE
          QHALFP = .TRUE.
          IF (RANDOM(ISEED)  <  DBLE(ISHOOT)/DBLE(NREGSV)) THEN
             PDIR = -1
             NSTEP = NSTSAV - ITPSST
          ELSE
             PDIR = 1
             NSTEP = ITPSST
          ENDIF
       ENDIF

       IF (PRNLEV >= 0) THEN
          WRITE(OUTU,*)
          WRITE(OUTU,2000) ISHOOT,TPSSTP,QHALFP, MINONE*PDIR
       ENDIF
2000   FORMAT('TPSHOT>  Shooting from segment ',I4, ' at step', I6, &
            ' QHALFP ',L1, ' dir ', F3.0)

       !         Load in the data to shoot
       N2 = TRJIDX(ISHOOT+1)
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n2),ONE)
       CALL TPSGET_p(NATOM,VX,VY,VZ,TRJVL1(n2),MINONE*PDIR)
       ITPSSD = ITPSD1(N2)

       !         Only PRTURB velocities after initial.
       !         Note that this would creates a slight discrepancy between the first
       !         path and subsequent ones even if FRAC is zero due to the fact
       !         that the first path is not SHAKEd until DYNAMC, so send to PRTURB with
       !         FRAC = 0 for first step, unless we are reading a trj.
       IF (TPSSTP  >  0) THEN
          VFCTMP = tpsavFRAC
          QSCALE = .TRUE.
       ELSE
          VFCTMP = ZERO
          QSCALE = .FALSE.
       ENDIF

       !         Don't PRTURB if only doing half shoots
       !         Also don't PRTURB if Langevin dynamics and FRAC = 0
       !         There may be problems if FRAC  /=  0 if Langevin and SHAKE is on,
       !         because the kinetic energy can drop
       IF((.NOT. QHALFP) .AND. (ILANG == 0.OR.tpsavFRAC /= 0)) THEN
          CALL PRTURB(X,Y,Z,VX,VY,VZ,ISEED,VFCTMP,TEMP, &
               IGVOPT,DELTA,GAMMA,QALWRT &
#if KEY_TSM==1
               ,BACKLS  & 
#endif
               ,QSCALE,NDEGF,ILANG)
       ENDIF

       !         Save velocities for the reverse
       IF (.NOT. QHALFP) &
            CALL TPSSAV_p(NATOM,VX,VY,VZ,REVVEL,ONE)

       !         Check to see if we need to store the initial point in various arrays
       CALL TPSUPD(NATOM,X,Y,Z,VX,VY,VZ)

       !         This point will never get checked in DYNAMC for HSAM, so copy from old
       IF (QHSAMP) HTXNEW(ISHOOT+1) = HTXOLD(ISHOOT+1)

    ELSE
       !         2nd half of shoot

       !         Flip the direction flag
       PDIR =  1

       !         Restore the starting step to its original value
       ITPSST  = NSTSAV - NSTEP

       !         Integrate to zero, which would represent initial conditions
       NSTEP = ITPSST

       !         Load in the data to shoot backward
       N2 = TRJIDX(ISHOOT+1)
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n2),ONE)
       CALL TPSGET_p(NATOM,VX,VY,VZ,REVVEL,-ONE)
       ITPSSD = ITPSD1(N2)
    ENDIF

    RETURN
  END SUBROUTINE TPSHOT

  SUBROUTINE TPSHFT(NSTEP,XN,YN, &
       ZN,VX,VY,VZ,NATOM,ISEED)
    !
    !       This is the shifting routine for TPS sampling
    !
    !       It determines the direction of shift and the shifting point
    !       and then sets up data structures for call to DYNAMC
    !
    !       M. Hagan and A. R. Dinner
    !
  use clcg_mod,only:random
  use number
  use coord
  use stream

    INTEGER NSTEP
    INTEGER NATOM,ISEED
    real(chm_real)  XN(*),YN(*),ZN(*),VX(*),VY(*),VZ(*)
    !
    INTEGER SHFTLN, N1, N2, NCALL

    !       Pick length to shift.
    !       SHFNUM is the number of regularly saved points from either edge.
    SHFNUM = INT(RANDOM(ISEED)*MXSHFN) + 1
    SHFTLN = (SHFNUM)*NTSAVV

    !       To ensure detailed balance, the forward and reverse directions
    !       should always be equally probable.
    IF(RANDOM(ISEED)  <  HALF) THEN
       !         Shift backwards.  The first SHFTLN steps are discarded and an
       !         additional SHFTLN steps are integrated starting with the final step.
       PDIR = -1
       !         The starting point for integration is the translated last step.
       ITPSST = NSTSAV - SHFTLN
       N1 = TRJIDX(SHFNUM+1)
       N2 = TRJIDX(NREGSV)
       NCALL = 0
    ELSE
       !         Shift forwards.   The last  SHFTLN steps are discarded and an
       !         additional SHFTLN steps are integrated starting with the first step.
       PDIR = 1
       !         The starting point is the translated starting structure (zeroth step).
       ITPSST = SHFTLN
       N1 = TRJIDX(NREGSV-SHFNUM)
       N2 = TRJIDX(1)
       NCALL = NSTSAV
    ENDIF

    IF (PRNLEV >= 0) THEN
       WRITE(OUTU,*)
       WRITE(OUTU,2000) ITPSST/NTSAVV+1,TPSSTP,MINONE*PDIR
    ENDIF
2000 FORMAT('TPSHFT>  Shifting from segment ',I4, ' at step', I6, &
         ' dir ', F3.0)
    !       Immediately check to see if the initial point is still in a basin
    CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n1),ONE)

    CALL TPSCHK(QBASA1,QBASB1,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
         NCALL,'SHFTI',PDIR)

    !       If not in either, reject the step now.
    IF (.NOT.QBASA1.AND..NOT.QBASB1.AND..NOT.QHSAMP) THEN
       NSTEP = 0
       RETURN
    ENDIF

    !       We're here so the move is possible.
    NSTEP = SHFTLN
    !       Load in the right starting structure.
    CALL TPSGET_p(NATOM,XN,YN,ZN,TRJCD1(n2),ONE)
    CALL TPSGET_p(NATOM,VX,VY,VZ,TRJVL1(n2),-PDIR*ONE)
    !       Load seed in case we are doing Langevin dynamics
    ITPSSD = ITPSD1(n2)

    RETURN
  END SUBROUTINE TPSHFT

  SUBROUTINE PRTURB(X,Y,Z,VX,VY,VZ,ISEED,frac,TEMP, &
       IGVOPT,DELTA,GAMMA,QALWRT &
#if KEY_TSM==1
       ,BACKLS  & 
#endif
       ,QSCALE,NDEGF,ILANG)
    !
    !       Perturb the velocities for a shooting move.
    !
  use bases_fcm
  use consta
  use coordc
  use deriv
  use number
  use psf
  use shake

    INTEGER ISEED, IGVOPT,NDEGF,ILANG
#if KEY_TSM==1
    INTEGER BACKLS(*) 
#endif
    real(chm_real) :: X(:), Y(:), Z(:), VX(*),VY(*),VZ(*),frac, TEMP
    real(chm_real)  GAMMA(*), DELTA
    LOGICAL QALWRT(6), QSCALE
    !
    integer i
    real(chm_real) TK1, TK2, VNEW, S, SD, BOLTZ
    real(chm_real),allocatable,dimension(:) :: XOP,YOP,ZOP,XNP,YNP,ZNP
    integer,allocatable,dimension(:) :: ISKP


    !       Get k_B*T
    BOLTZ = TEMP*KBOLTZ

    !       Determine old kinetic energy (factor of 2 will divide so ignore).
    TK1 = ZERO
    DO I = 1, NATOM
       TK1 = TK1 + AMASS(I)*(VX(I)*VX(I)+VY(I)*VY(I)+VZ(I)*VZ(I))
    ENDDO

    !       Add the perturbative velocity vectors
    DO I = 1, NATOM
       IF(IMOVE(I)  ==  0) THEN
          SD=BOLTZ/AMASS(I)
          SD=SQRT(SD)
          CALL GAUSSI(ZERO,SD,VNEW,ISEED,1)
          VX(I) = VX(I) + VNEW*FRAC
          CALL GAUSSI(ZERO,SD,VNEW,ISEED,1)
          VY(I) = VY(I) + VNEW*FRAC
          CALL GAUSSI(ZERO,SD,VNEW,ISEED,1)
          VZ(I) = VZ(I) + VNEW*FRAC
       ENDIF
    ENDDO

    !       Remove global translation and rotation
    !       Be careful about doing this if Langevin dynamics on
    CALL STOPRT(X,Y,Z,VX,VY,VZ,AMASS,IGVOPT,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
         ,BACKLS  & 
#endif
         )

    !       If QSHTMD, must rescale BEFORE SHAKE
    IF(QSHTMD .AND. QSCALE) THEN
       !         Determine new kinetic energy.
       TK2 = ZERO
       DO I = 1, NATOM
          TK2 = TK2 + AMASS(I)*(VX(I)*VX(I)+VY(I)*VY(I)+VZ(I)*VZ(I))
       ENDDO

       S = SQRT(TK1/TK2)

       DO I = 1, NATOM
          VX(I) = S*VX(I)
          VY(I) = S*VY(I)
          VZ(I) = S*VZ(I)
       ENDDO
    ENDIF

    !       SHAKE here before rescaling to keep total energy constant.

    IF (QHOLO) THEN
       call chmalloc('tps.src','PRTURB','XOP',NATOM,crl=XOP)
       call chmalloc('tps.src','PRTURB','YOP',NATOM,crl=YOP)
       call chmalloc('tps.src','PRTURB','ZOP',NATOM,crl=ZOP)
       call chmalloc('tps.src','PRTURB','XNP',NATOM,crl=XNP)
       call chmalloc('tps.src','PRTURB','YNP',NATOM,crl=YNP)
       call chmalloc('tps.src','PRTURB','ZNP',NATOM,crl=ZNP)
       call chmalloc('tps.src','PRTURB','ISKP',NATOM,intg=ISKP)
       CALL TPSSHK(VX,VY,VZ,X,Y,Z,XCOMP,YCOMP,ZCOMP,DX,DY,DZ, &
            XOP,YOP,ZOP, &
            XNP,YNP,ZNP,IMOVE,GAMMA, &
            AMASS,DELTA,NATOM,BNBND,BIMAG,ISKP, &
            ILANG)
       call chmdealloc('tps.src','PRTURB','XOP',NATOM,crl=XOP)
       call chmdealloc('tps.src','PRTURB','YOP',NATOM,crl=YOP)
       call chmdealloc('tps.src','PRTURB','ZOP',NATOM,crl=ZOP)
       call chmdealloc('tps.src','PRTURB','XNP',NATOM,crl=XNP)
       call chmdealloc('tps.src','PRTURB','YNP',NATOM,crl=YNP)
       call chmdealloc('tps.src','PRTURB','ZNP',NATOM,crl=ZNP)
       call chmdealloc('tps.src','PRTURB','ISKP',NATOM,intg=ISKP)
    ENDIF

    IF(QSCALE .AND. .NOT. QSHTMD) THEN
       !         Determine new kinetic energy.
       TK2 = ZERO
       DO I = 1, NATOM
          TK2 = TK2 + AMASS(I)*(VX(I)*VX(I)+VY(I)*VY(I)+VZ(I)*VZ(I))
       ENDDO

       S = SQRT(TK1/TK2*TFRACT)

       DO I = 1, NATOM
          VX(I) = S*VX(I)
          VY(I) = S*VY(I)
          VZ(I) = S*VZ(I)
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE PRTURB

  SUBROUTINE TPSSHK(VX,VY,VZ,X,Y,Z,XCOMP,YCOMP,ZCOMP,DX,DY,DZ, &
       XOLD,YOLD,ZOLD,XNEW,YNEW,ZNEW,IMOVE,GAMMA, &
       AMASS,DELTA,NATOM,BNBND,BIMAG,ISKP, &
       ILANG)
    !
    !       Make the perturbed velocities consistent with SHAKE.
    !
    use number
    use energym
    use holonom, only: holonoma

    INTEGER IMOVE(*), NATOM, ISKP(*)  !!, BNBND(*), BIMAG(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    INTEGER ILANG
    real(chm_real) :: VX(*), VY(*), VZ(*), DX(:), DY(:), DZ(:)
    real(chm_real) :: X(:), Y(:), Z(:), XCOMP(*), YCOMP(*), ZCOMP(*)
    real(chm_real) XOLD(*), YOLD(*), ZOLD(*),  &
         XNEW(*), YNEW(*), ZNEW(*)
    real(chm_real) DELTA, AMASS(*), GAMMA(*)
    !
    INTEGER I, NATOM2, NATOM3, STEMP
    real(chm_real) DELTA2, DELTAS, FACT, ALPHA
    LOGICAL QOK
    !
    DELTA2 = DELTA*DELTA
    DELTAS = HALF*DELTA2
    NATOM2 = NATOM + NATOM
    NATOM3 = NATOM + NATOM2

    DO I = 1, NATOM
       XCOMP(I) = X(I)
       YCOMP(I) = Y(I)
       ZCOMP(I) = Z(I)
    ENDDO

    !       Do shake just in case coordinates don't fit constraints
    CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)

    !       Save SHAKEd coordinates
    DO I = 1, NATOM
       XCOMP(I)=X(I)
       YCOMP(I)=Y(I)
       ZCOMP(I)=Z(I)
    ENDDO

    !       Get the forces after SHAKE
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    !
    !       If Langevin Dynamics are on, also get the random forces.
    IF(ILANG > 0) THEN
       STEMP=ITPSSD
       CALL DLNGV(GAMMA,STEMP)
    ENDIF

    DO I = 1, NATOM

       IF (IMOVE(I) == 0) THEN

          FACT=DELTAS/AMASS(I)
          IF(PDIR*ITPSSD > 0) THEN
             ALPHA=TWO*GAMMA(I+NATOM2)*GAMMA(I+NATOM3)*DELTA2
          ELSE
             ALPHA = TWO*GAMMA(I+NATOM3)*DELTA2
          ENDIF
          !           Get forward displacement
          XOLD(I)=VX(I)*ALPHA-DX(I)*FACT
          YOLD(I)=VY(I)*ALPHA-DY(I)*FACT
          ZOLD(I)=VZ(I)*ALPHA-DZ(I)*FACT

          !           Get backward displacement
          IF(PDIR*ITPSSD > 0) THEN
             ALPHA = ALPHA/GAMMA(I+NATOM2)
          ELSE
             ALPHA = TWO*GAMMA(I+NATOM3)*DELTA2
          ENDIF
          XNEW(I)=VX(I)*ALPHA+DX(I)*FACT
          YNEW(I)=VY(I)*ALPHA+DY(I)*FACT
          ZNEW(I)=VZ(I)*ALPHA+DZ(I)*FACT

       ELSE
          XOLD(I)=ZERO
          YOLD(I)=ZERO
          ZOLD(I)=ZERO
          XNEW(I)=ZERO
          YNEW(I)=ZERO
          ZNEW(I)=ZERO

       ENDIF
    ENDDO

    !       Get the previous position and SHAKE it
    DO I = 1, NATOM
       X(I) = XCOMP(I) - XNEW(I)
       Y(I) = YCOMP(I) - YNEW(I)
       Z(I) = ZCOMP(I) - ZNEW(I)
    ENDDO
    CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)

    DO I = 1, NATOM

       !         Get the SHAKEd backward displacement
       XNEW(I)=XCOMP(I)-X(I)
       YNEW(I)=YCOMP(I)-Y(I)
       ZNEW(I)=ZCOMP(I)-Z(I)

       !         Now get the next position (same as previous loop but in other
       !         direction; do it here to save a loop).  As before, SHAKE the
       !         new coordinates.
       X(I)=XCOMP(I)+XOLD(I)
       Y(I)=YCOMP(I)+YOLD(I)
       Z(I)=ZCOMP(I)+ZOLD(I)

    ENDDO

    CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)

    DO I = 1, NATOM

       !         Get the SHAKEd forward dispacement
       XOLD(I)=X(I)-XCOMP(I)
       YOLD(I)=Y(I)-YCOMP(I)
       ZOLD(I)=Z(I)-ZCOMP(I)

       !         Get what the velocities should have been to give SHAKEd results
       FACT=GAMMA(I+NATOM3)
       VX(I)=(XNEW(I)+XOLD(I))*FACT
       VY(I)=(YNEW(I)+YOLD(I))*FACT
       VZ(I)=(ZNEW(I)+ZOLD(I))*FACT

       X(I)=XCOMP(I)
       Y(I)=YCOMP(I)
       Z(I)=ZCOMP(I)

    ENDDO

    RETURN
  END SUBROUTINE TPSSHK

  SUBROUTINE TPHDIR(QBASA1,QBASA2,ISEED)
    !
    !       Determine dir of path for calculation of <h[x(t)]>
    !
  use clcg_mod,only:random
  use number

    INTEGER ISEED
    LOGICAL QBASA1,QBASA2

    IF(QBASA1 .AND. .NOT. QBASA2) THEN
       HTXDIR = 1
    ELSE IF(QBASA2 .AND. .NOT. QBASA1) THEN
       HTXDIR = -1
    ELSE IF(QBASA1 .AND. QBASA2) THEN
       IF(RANDOM(ISEED)  <  HALF) THEN
          HTXDIR = 1
       ELSE
          HTXDIR = -1
       ENDIF
    ELSE
       HTXDIR = 0
    ENDIF

    RETURN
  END SUBROUTINE TPHDIR

  SUBROUTINE TPUSER(QBASA1,QBASB1,QCALL)
    !
    !       This allows additional basins based on USERSB routines.
    !
    !       QBASA1 --- Whether the structure is in basin A
    !       QBASB1 --- Whether the structure is in basin B
    !       QTPUSR --- Whether USER coordinates are defined
    !       QCALL  --- If it is necessary to evaluate coordinates first
    !       ITPSUN --- Unit number for writing coordinates
    !       ITPSPR --- Frequency of writing coordinates
    !       TPSSTP --- TPS path number
    !

    LOGICAL QBASA1,QBASB1,QCALL
    !
    IF (.NOT. QTPUSR) RETURN
    !       Insert code here (see TPSCHK GOTO loops for structure).

    RETURN
  END SUBROUTINE TPUSER


  SUBROUTINE TPINIT(ISEED,NATOM,X,Y,Z,VX,VY,VZ, &
       NSTEP,IREST,NSAVC,IUNCRD,NSAVV,IUNVEL,DELTA,TIMEST)
    !
    !       This is the initialization routine for transition path sampling.
    !
  use clcg_mod,only:random
  use number
  use stream
  use rxncom

    INTEGER ISEED, NATOM, NSTEP
    INTEGER IREST
    INTEGER NSAVV, IUNVEL, NSAVC, IUNCRD
    real(chm_real)  DELTA,TIMEST
    real(chm_real)  X(*), Y(*), Z(*), VX(*), VY(*), VZ(*)
    !
    INTEGER I, J, N, N2, FACTOR, MULT, ISTART, IEND

    !       Suppress write out during the dynamics so the frames can be
    !       written in physical order.  Mandate that NTSAVC and NTSAVV
    !       be factors of NSTEP so that the endpoints are saved for the
    !       acceptance criterion.

    !       Just to avoid having this be zero
    ITPSSD = ISEED

    NTSAVC = NSAVC
    NTSAVV = NSAVV
    IF (NTSAVC  <=  0) NTSAVC = NSTEP
    IF (NTSAVV  <=  0) NTSAVV = NSTEP
    IF (MOD(NSTEP,NTSAVC) /= 0) CALL WRNDIE(-1,'<TPINIT>', &
         'NSTEP is not a multiple of NSAVC')
    IF (MOD(NSTEP,NTSAVV) /= 0) CALL WRNDIE(-1,'<TPINIT>', &
         'NSTEP is not a multiple of NSAVV')
    NSAVC  = 0
    NSAVV  = 0
    !       Allocate space for saving the structures and velocities
    !       at regular intervals (NTSAVC and NTSAVV).
    NREGSV = (NSTEP/NTSAVV) + 1
    call allocate_tps_trj(trjcd1,natom,nregsv)
    call allocate_tps_trj(trjcd2,natom,nregsv)
    call allocate_tps_trj(trjvl1,natom,nregsv)
    call allocate_tps_trj(trjvl2,natom,nregsv)


    !       Allocate space for saving the point we are shooting from
    call chmalloc('tps.src','TPINIT','REVVEL%x',NATOM,crl=REVVEL%x)
    call chmalloc('tps.src','TPINIT','REVVEL%y',NATOM,crl=REVVEL%y)
    call chmalloc('tps.src','TPINIT','REVVEL%z',NATOM,crl=REVVEL%z)

    !       Allocate space for the array that will give the correct index
    !       for the TRJ index array.  TRJIDX(I) returns the index in TRJCD1 or
    !       TRJVL1 of the Ith point
    call chmalloc('tps.src','TPINIT','TRJIDX',NREGSV,intg=TRJIDX)

    !       Arrays for acceptance statistics from each point

    call chmalloc('tps.src','TPINIT','ATSHPR',NREGSV,intg=ATSHPR)
    call chmalloc('tps.src','TPINIT','ATSFPR',NREGSV,intg=ATSFPR)
    call chmalloc('tps.src','TPINIT','ATHFPR',NREGSV,intg=ATHFPR)
    call chmalloc('tps.src','TPINIT','ACSHPR',NREGSV,intg=ACSHPR)
    call chmalloc('tps.src','TPINIT','ACSFPR',NREGSV,intg=ACSFPR)
    call chmalloc('tps.src','TPINIT','ACHFPR',NREGSV,intg=ACHFPR)
    ATSHPR(1:nregsv) = 0
    ATSFPR(1:nregsv) = 0
    ATHFPR(1:nregsv) = 0
    ACSHPR(1:nregsv) = 0
    ACSFPR(1:nregsv) = 0
    ACHFPR(1:nregsv) = 0

    !       Arrays to hold seeds (used for Langevin dynamics)
    call chmalloc('tps.src','TPINIT','ITPSD1',NREGSV,intgp=ITPSD1)
    call chmalloc('tps.src','TPINIT','ITPSD2',NREGSV,intgp=ITPSD2)

    !       Determine the window to shoot from
    IF((SHOTLO  <  0) .OR. (SHOTLN  <=  0) .OR. &
         (SHOTLO + SHOTLN - 1  >  NREGSV)) THEN
       !         No value or bad value entered
       CALL WRNDIE(1,'<TPINIT>', &
            'Shooting window will be the entire trajectory')
       SHOTLO = 0
       SHOTLN = NREGSV
    ENDIF

    !       Initialize TPSSTP to 0 so that the first shot generates the
    !       starting trajectory and we have NTPATH paths sampled after that.
    TPSSTP = 0
    !       If we are reading trj, then the first path is already generated
    IF (IREST  ==  2) TPSSTP = 1
    !
    NSTSAV = NSTEP

    PDIR = 1

    !       Set up the data structures so that the first step
    !       can be treated like the rest in TPMAIN.

    IF (ITPRST  >  NREGSV .OR. ITPRST  <=  0) THEN
       CALL WRNDIE(1,'<TPINIT>','Changing IRST to middle point')
       ITPRST = NREGSV/2
    ENDIF

    IF (IREST  ==  1) NXTSHT = ITPRST

    !       If reading a trajectory, the default IFSHOT is NREGSV/2,
    !       else pick randomly
    IF (IFSHOT > NREGSV .OR. IFSHOT < 0 .AND. IREST == 0) THEN
       CALL WRNDIE(5,'<TPINIT>', &
            'Changing first shoot to NREGSV/2')
       IFSHOT = NREGSV/2
       NXTSHT = IFSHOT
    ELSE
       NXTSHT = IFSHOT
    ENDIF

    !       A flag to keep shooting from the middle point if
    !       path not accepted
    IF (IREST  ==  0) THEN
       QSHTMD = .TRUE.
    ELSE
       QSHTMD = .FALSE.
    ENDIF
    !       Note that TRJCD1 starts at zero, so the index is shifted
    IF(IREST  /=  2) THEN
       CALL TPSSAV_p(NATOM,X,Y,Z,TRJCD1(nxtsht+1),ONE)
       CALL TPSSAV_p(NATOM,VX,VY,VZ,TRJVL1(nxtsht+1),ONE)
    ELSE
       CALL WRNDIE(1,'<TPINIT>', &
            'Reading trajectory to initiate Transition Path Sampling')

       !         Read trj file
       IF (PNSAV  <=  0) PNSAV = NREGSV
       CALL TPRDCV_p(TRJCD1,NATOM,FIRSTU, &
            X, Y, Z,0,DELTA, &
            TIMEST,MULT,ISTART,IEND)
       !         Read vel file
       CALL TPRDCV_p(TRJVL1,NATOM,VFIRST, &
            VX,VY,VZ,1,DELTA, &
            TIMEST,MULT,ISTART,IEND)

       !         Get seeds saved for Langevin dynamics
       CALL TPRDSD(MULT,ISTART,IEND)

       !         Randomly select first shooting point
       IF (IFSHOT  <  0) THEN
          NXTSHT = INT(RANDOM(ISEED)*(SHOTLN)) + SHOTLO
       ELSE
          NXTSHT = IFSHOT
       ENDIF
    ENDIF

    !       Initialize the TRJ index array.  We start with the first element
    !       referencing the 0th step and the last element referencing the last
    !       step.
    DO I = 1, NREGSV
       TRJIDX(I) = i
    ENDDO

    !       Make sure that shifting stays within limits
    IF (MXSHFN  >  NREGSV/2) THEN
       CALL WRNDIE(-1,'<TPINIT>', &
            'Decreasing IMXS to half the path length')
       MXSHFN = NREGSV/2
    ENDIF

    IF(NTFRAC  ==  0) TFRACT = ONE

    IF (QHSAMP) THEN
       !         Make sure that RXNENE is called even if trace has not been
       !         set.  This is needed because HSAM checks the order parameters
       !         during dynamics
       RXNIND = 1

       HTXDIR = 0
       IF(PRNLEV > 0) WRITE(OUTU,*) &
            'TPINIT>  HSAMPLING has been requested'
       !         Set up variables for H sampling
       !         Shifting doesn't work if NHSAVF is not a factor of NTSAVC
       IF(NHSAVF  <=  0 .OR. NHSAVF  >  NTSAVC) NHSAVF = NTSAVC

       IF(MOD(NTSAVC,NHSAVF)  /=  0) THEN
          CALL WRNDIE(2,'<TPINIT>', &
               'NHSAVF must be factor of NTSAVC, modifying NHSAVF')
          !           Try to do what the user wanted
          FACTOR = MAX(1,NTSAVC/NHSAVF)
          NHSAVF = NTSAVC / FACTOR
       ENDIF
       NHSAVT = NSTEP/NHSAVF + 1
       !         Allocate space for the HTX arrays
       call chmalloc('tps.src','TPINIT','HTXNEW',NHSAVT,intgp=HTXNEW)
       call chmalloc('tps.src','TPINIT','HTXOLD',NHSAVT,intgp=HTXOLD)
       call chmalloc('tps.src','TPINIT','HTXAVG',NHSAVT,intgp=HTXAVG)

       !         The number of updates for HTXAVG
       HTXCNT = 0

       !         Initialize the H array for the trj we just read in
       IF (IREST  ==  2) THEN
          CALL HTXINI(HTXOLD,X,Y,Z,NATOM,ISEED)
       ENDIF
    ENDIF

    QBASA1 = .FALSE.
    QBASA2 = .FALSE.
    QBASB1 = .FALSE.
    QBASB2 = .FALSE.
    SHFTON = .FALSE.

    !       Initialize the acceptance ratios.
    !       Set variables to -1 for shoots so first not counted.
    NSHOTT = -1
    NSHOTA = -1
    IF (IREST  ==  2) THEN
       NSHOTT = 0
       NSHOTA = 0
    ENDIF
    NSHFTT =  0
    NSHFTA =  0
    NHALFT = 0
    NHALFA = 0

    RETURN
  END SUBROUTINE TPINIT

  SUBROUTINE TPSACC(ISEED,NATOM,X,Y,Z,VX,VY,VZ, &
       NSTEP,IREST,TIMEST)
    !
    !       Call the right acceptance criterion routine.
    !       Also, increment the step counter.
    !
  use number
  use stream
  use param_store, only: set_param

  implicit none

    INTEGER ISEED, NATOM
    INTEGER NSTEP
    INTEGER IREST
    real(chm_real)  TIMEST
    real(chm_real)  X(*), Y(*), Z(*)
    real(chm_real)  VX(*), VY(*), VZ(*)
    !
    INTEGER N1,N2,N3,N4,N5,N6,I
    INTEGER PDIRSV
    real(chm_real)  R1, R2, R3
    character(len=5) STRNG

    PDIRSV = PDIR
    IF (SHFTON) THEN
       NSHFTT = NSHFTT + 1
       ATSFPR(ITPSSV/NTSAVV+1) = ATSFPR(ITPSSV/NTSAVV+1)+1
       IF(NSTEP  >  0) THEN
          CALL SHFACC(ISEED,NATOM,X,Y,Z,VX,VY,VZ,NSHFTA)
          STRNG='SHIFT'

       ENDIF
    ELSE IF (QHALFP) THEN
       CALL HAFACC(NATOM,X,Y,Z,VX,VY,VZ,ISEED)
       !         Note that if the shot half path does not go in either basin,
       !         all entries come up false
       STRNG='HALFP'
    ELSE
       CALL SHTACC(ISEED,NATOM, &
            ((TPSSTP == 0).AND.(IREST /= 2)), &
            NSHOTT,NSHOTA,VX,VY,VZ)
       STRNG='SHOOT'
    ENDIF

    !       Write the acceptance ratios
    IF (SHFTON .OR. QHALFP .OR. PDIR  ==  1) THEN
       IF (QHSAMP) CALL TPHCLC(TIMEST)

       IF(PRNLEV  >=  0 .AND. NPRACC  >  0 .AND. TPSSTP .GT. 0) THEN
          IF (MOD(TPSSTP,NPRACC)  ==  0 .OR. TPSSTP .EQ. NTPATH) THEN
             R1 = ZERO
             R2 = ZERO
             R3 = ZERO
             IF (NSHOTT  >  0) R1 = DBLE(NSHOTA)/DBLE(NSHOTT)
             IF (NSHFTT  >  0) R2 = DBLE(NSHFTA)/DBLE(NSHFTT)
             IF (NHALFT  >  0) R3 = DBLE(NHALFA)/DBLE(NHALFT)

             call set_param('RSHO',R1)
             call set_param('RHAL',R2)
             call set_param('RSHF',R3)

             WRITE (OUTU,*)
             WRITE (OUTU,888) &
                  'TPSACC:','STEP','FULLS','ACCEPT','RATIO', &
                  'HALFS','ACCEPT','RATIO', &
                  'SHIFT','ACCEPT','RATIO'
             WRITE (OUTU,888) &
                  '-------','------','------', '------', &
                  '-----','------','------','------', &
                  '------','------','-----'
             WRITE (OUTU,999) &
                  'TPSACC>',TPSSTP,NSHOTT,NSHOTA,R1, &
                  NHALFT,NHALFA,R3,NSHFTT,NSHFTA,R2
             WRITE (OUTU,888) &
                  '-------','------','------', '------', &
                  '-----','------','------','------', &
                  '------','------','-----'
             WRITE (OUTU,*)
888          FORMAT(A7,1X,A9,3(1X,A6,1X,A6,1X,A6))
999          FORMAT(A7,1X,I9,3(1X,I6,1X,I6,1X,F6.3))
             !
             !             Also write out the acceptance arrays
             DO I=1,NREGSV
                R1 = ZERO
                R2 = ZERO
                R3 = ZERO
                N1 = ATSHPR(I)
                N2 = ACSHPR(I)
                IF(N1 > 0) R1 = DBLE(N2)/DBLE(N1)
                N3 = ATSFPR(I)
                N4 = ACSFPR(I)
                IF(N3 > 0) R2 = DBLE(N4)/DBLE(N3)
                N5 = ATHFPR(I)
                N6 = ACHFPR(I)
                IF(N5 > 0) R3 = DBLE(N6)/DBLE(N5)
                WRITE(ACCUNI,899) I,TPSSTP,N1,N2,R1,N5,N6,R3,N3,N4,R2
             ENDDO
             WRITE(ACCUNI,*)
             CALL GFLUSH(ACCUNI)
899          FORMAT(I7,1X,I9,3(1X,I6,1X,I6,1X,F6.3))
          ENDIF
       ENDIF

       !         Check to see if T scaling is done
       IF((NSHOTA + NHALFA)  ==  NTFRAC) TFRACT = ONE

       QBASA1 = .FALSE.
       QBASA2 = .FALSE.
       QBASB1 = .FALSE.
       QBASB2 = .FALSE.

       TPSSTP = TPSSTP + 1

       !         This flag is necessary for TPMAIN
       PDIR = 1

    ENDIF
    RETURN
  END SUBROUTINE TPSACC

  SUBROUTINE SHTACC(ISEED,NATOM,LACC,NSHOTT,NSHOTA,VX,VY,VZ)
    !
    !       Apply the acceptance criterion and update the data structure
    !       accordingly for a full shooting attempt.
    !
  use coord
  use number

    INTEGER nshott, NSHOTA
    INTEGER ISEED, NATOM
    real(chm_real)  VX(*),VY(*),VZ(*)
    LOGICAL LACC
    !
    INTEGER I, J, TPSSWP
    type(chm_xyz),pointer,dimension(:) :: tpsswpp
    integer,pointer,dimension(:) :: tpsswp_ip
    !

    IF (PDIR  ==  -1) THEN

       NSHOTT = NSHOTT + 1
       ATSHPR(ITPSSV/NTSAVV+1) = ATSHPR(ITPSSV/NTSAVV+1)+1

       !         Get the structure (it could be different than in XYZ arrays due to
       !         restart check which precedes this call.
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD2(nregsv),ONE)

       !         Calculate the order parameter value at the current endpoint.
       !         It will be stored in the common variable DELVAL(1,NGRAPH+1).
       CALL TPSCHK(QBASA1,QBASB1,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
            (NREGSV-1)*NTSAVV,'SHOOT',PDIR)

       IF(QHSAMP.AND.(MOD(ITPSST,NHSAVF) == 0)) THEN
          J = ITPSST / NHSAVF + 1
          CALL TPHASN(QBASA1,QBASB1,J)
       ENDIF

       !         If not in either, reject the step now.
       !         Unfortunately, I don't think we can reject immediately if
       !         we are H sampling, as it is possible one side of a path to hit
       !         A and B
       IF (.NOT.LACC.AND..NOT.QBASA1.AND..NOT.QBASB1 .AND. .NOT. &
            QHSAMP) THEN
          PDIR = 1
          RETURN
       ENDIF

    ELSE
       !         Both sides of path have been completed
       !         Get the structure (it could be different than in XYZ arrays due to
       !         restart check which precedes this call.
       CALL TPSGET_p(NATOM,X,Y,Z,TRJCD2(1),ONE)

       !         Calculate the order parameter.
       CALL TPSCHK(QBASA2,QBASB2,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
            0,'SHOOT',PDIR)

       IF(QHSAMP.AND.(MOD(ITPSST,NHSAVF) == 0)) THEN
          J = ITPSST / NHSAVF + 1
          CALL TPHASN(QBASA2,QBASB2,J)
       ENDIF

       IF(QHSAMP) THEN
          QBASB1 = QHFLAG
          QBASB2 = QHFLAG
       ENDIF
       !         Stop automatically shooting from first point?
       IF ((QBASA1.AND.QBASB2).OR.(QBASB1.AND.QBASA2)) &
            QSHTMD = .FALSE.

       IF (LACC.OR.((QBASA1.AND.QBASB2).OR.(QBASB1.AND.QBASA2))) THEN
          !           Accept
          NSHOTA = NSHOTA + 1
          ACSHPR(ITPSSV/NTSAVV+1) = ACSHPR(ITPSSV/NTSAVV+1)+1


          !           Swap TPSCD1 and 2, TPSVL1 and 2, NPPNT1 and 2
          !           This way the trial trajectory becomes the saved one without
          !           copying the coordinates and velocities
          !        TPSSWP = TRJCD2
          !        TRJCD2 = TRJCD1
          !        TRJCD1 = TPSSWP

          TPSSWPp => TRJCD2
          TRJCD2 => TRJCD1
          TRJCD1 => TPSSWPp

          TPSSWPp => TRJVL2
          TRJVL2  => TRJVL1
          TRJVL1  => TPSSWPp
          !           Also the seeds, in case Langevin in use
          TPSSWP_ip => ITPSD2
          ITPSD2    => ITPSD1
          ITPSD1    => TPSSWP_ip

          !           The new points have been saved in the correct order
          DO I = 1, NREGSV
             TRJIDX(I) = i
          ENDDO

          IF(QHSAMP) THEN
             !             Update the HTX arrays
             TPSSWP_ip => HTXNEW
             HTXNEW    => HTXOLD
             HTXOLD    => TPSSWP_ip
             !             Update the direction
             !             note that QBASA1 corresponds to the end of the path here
             CALL TPHDIR(QBASA2,QBASA1,ISEED)
          ENDIF
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE SHTACC

  SUBROUTINE HAFACC(NATOM,X,Y,Z,VX,VY,VZ,ISEED)
    !
    !       Apply the acceptance criterion and update the data structure
    !       accordingly for a half shooting attempt.
    !
  use number

    INTEGER NATOM
    INTEGER ISEED
    real(chm_real)  X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
    !
    INTEGER I, J, N1, N2, I1, I2,I3, ISAV1,ISAV2,NSHOOT,N11
    !
    NHALFT = NHALFT + 1
    ATHFPR(ITPSSV/NTSAVV+1) = ATHFPR(ITPSSV/NTSAVV+1)+1 
    IF (PDIR  ==  -1) THEN
       !         Forward shoot
       N2 = NREGSV
       N1 = TRJIDX(1)
       N11 = 1
    ELSE
       !         Backwards shoot
       N2 = 1
       N1 = TRJIDX(NREGSV)
       N11 = NREGSV
    ENDIF

    !       Get the structure which was shot
    !       (it could be different than in XYZ arrays due to
    !       restart check which precedes this call).
    CALL TPSGET_p(NATOM,X,Y,Z,TRJCD2(n2),ONE)

    !       Calculate the order parameter value at the current endpoint.
    !       It will be stored in the common variable DELVAL(1,NGRAPH+1).
    CALL TPSCHK(QBASA1,QBASB1,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
         (N2-1)*NTSAVV,'HALFP',PDIR)
    !
    !       Put this point in the HTXNEW array
    IF(QHSAMP.AND.(MOD(ITPSST,NHSAVF) == 0)) THEN
       J = ITPSST / NHSAVF + 1
       CALL TPHASN(QBASA1,QBASB1,J)
    ENDIF

    !       If not in either, reject the step now.
    IF (.NOT.QBASA1.AND..NOT.QBASB1.AND..NOT.QHSAMP) THEN
       PDIR = 1
       RETURN
    ENDIF

    !       Now the other side (already existed)
    CALL TPSGET_p(NATOM,X,Y,Z,TRJCD1(n1),ONE)

    !       Calculate the order parameter.
    !       must change the itppst going to tpschk here
    CALL TPSCHK(QBASA2,QBASB2,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
         (N11-1)*NTSAVV,'HALFP',PDIR)

    IF(QHSAMP) THEN
       NSHOOT = ITPSSV/NHSAVF + 1
       IF(PDIR  ==  -1) THEN
          !           NSHOOT was the first integration point, which does not get checked
          ISAV1 = 1
          ISAV2 = NSHOOT
       ELSE
          !           Again, NSHOOT point will not have been checked
          ISAV1 = NSHOOT
          ISAV2 = NHSAVT
       ENDIF

       IF (.NOT. QHFLAG) CALL TPHFLG(HTXOLD,ISAV1,ISAV2)
       QBASB1 = QHFLAG
       QBASB2 = QHFLAG
    ENDIF

    IF ((QBASA1.AND.QBASB2).OR.(QBASB1.AND.QBASA2)) THEN
       !         Accept
       NHALFA = NHALFA + 1
       ACHFPR(ITPSSV/NTSAVV+1) = ACHFPR(ITPSSV/NTSAVV+1)+1


       IF(PDIR  ==  -1) THEN
          !           Note that ISHOOT+1 corresponds to the shooting point in TRJCD2
          I1 = ISHOOT+1
          I2 = NREGSV
       ELSE
          I1 = 1
          I2 = ISHOOT
       ENDIF

       DO I=I1,I2
          CALL TPSGET_p(NATOM,X,Y,Z,TRJCD2(i),ONE)
          CALL TPSGET_p(NATOM,VX,VY,VZ,TRJVL2(i),ONE)
          J = TRJIDX(I)
          CALL TPSSAV_p(NATOM,X,Y,Z,TRJCD1(j),ONE)
          CALL TPSSAV_p(NATOM,VX,VY,VZ,TRJVL1(j),ONE)

          !           Exchange saved seeds, in case Langevin on
          ITPSD1(J) = ITPSD2(I)
       ENDDO

       IF(QHSAMP) THEN
          !           We are going to copy from HTXNEW to HTXOLD (since typically that
          !           would be fewer points.  So, we have to switch the ISAV's
          IF(PDIR  ==  1) THEN
             !             NSHOOT was the first integration point, which does not get checked
             ISAV1 = 1
             ISAV2 = NSHOOT - 1
          ELSE
             !             Again, NSHOOT point will not have been checked
             ISAV1 = NSHOOT + 1
             ISAV2 = NHSAVT
          ENDIF

          DO I=ISAV1,ISAV2
             J=HTXNEW(I)
             HTXOLD(I) = j
          ENDDO

          IF(PDIR  ==  -1) THEN
             CALL TPHDIR(QBASA2,QBASA1,ISEED)
          ELSE
             CALL TPHDIR(QBASA1,QBASA2,ISEED)
          ENDIF
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE HAFACC

  SUBROUTINE SHFACC(ISEED,NATOM,XACC,YACC,ZACC, &
       VXACC,VYACC,VZACC,NSHFTA)
    !
    !       Apply the acceptance criterion and update the data structure
    !       accordingly for a shift move.  The initial point has already
    !       been checked in TPSHFT.
    !
  use coord
  use number

    INTEGER ISEED, NATOM
    INTEGER NSHFTA
    real(chm_real)  XACC(*), YACC(*), ZACC(*),  &
         VXACC(*), VYACC(*), VZACC(*)
    LOGICAL LACC
    !
    INTEGER I, J, N, TPSSWP, N2, CSTART, NSHIFT, I1, I2,N1
    integer,pointer,dimension(:) :: itemp

    !       Get the structure (it could be different than in XYZ arrays due to
    !       restart check which precedes this call.  X,Y,Z are common variables
    IF (PDIR  ==  -1) THEN
       !         Reverse shift
       N1 = NREGSV
    ELSE
       N1 = 1
    ENDIF
    CALL TPSGET_p(NATOM,X,Y,Z,TRJCD2(n1),ONE)

    !       Calculate the order parameter and check if in basins.
    CALL TPSCHK(QBASA2,QBASB2,.TRUE.,ITPSUN,ITPSPR,TPSSTP, &
         (N1-1)*NTSAVV,'SHIFT',PDIR)

    IF(QHSAMP) THEN
       IF(MOD(ITPSST,NHSAVF) == 0) THEN
          J = ITPSST / NHSAVF + 1
          CALL TPHASN(QBASA2,QBASB2,J)
       ENDIF

       NSHIFT = SHFNUM*NTSAVV/NHSAVF + 1
       !         Determine whether remaining section of path visits B

       IF(PDIR  ==  -1) THEN
          !           NHSAVT was the first integration point, and will not have already
          !           been checked
          I1 = NSHIFT
          I2 = NHSAVT
       ELSE
          !           Again, 1st point will not have been checked
          I1 = 1
          I2 = NHSAVT - NSHIFT
       ENDIF

       IF(.NOT. QHFLAG) CALL TPHFLG(HTXOLD,I1,I2)

       QBASB1 = QHFLAG
       QBASB2 = QHFLAG

    ENDIF

    IF ((QBASA1.AND.QBASB2).OR.(QBASB1.AND.QBASA2)) THEN
       !         Accept
       NSHFTA = NSHFTA + 1
       ACSFPR(ITPSSV/NTSAVV+1) = ACSFPR(ITPSSV/NTSAVV+1)+1

       IF (PDIR  ==  -1) THEN

          !           reverse shift.
          !           Set up new TRJIDX
          DO I = 1, NREGSV
             J = TRJIDX(I) + SHFNUM
             IF (J  >  NREGSV) J = J - NREGSV
             TRJIDX(I) = j
          ENDDO

          !           Copy all points greater than shift point from 2 to 1

          DO I = NREGSV-SHFNUM+1, NREGSV
             CALL TPSGET_p(NATOM,XACC,YACC,ZACC,TRJCD2(i),ONE)
             CALL TPSGET_p(NATOM,VXACC,VYACC,VZACC,TRJVL2(i),ONE)
             J = TRJIDX(I)
             CALL TPSSAV_p(NATOM,XACC,YACC,ZACC,TRJCD1(j),ONE)
             CALL TPSSAV_p(NATOM,VXACC,VYACC,VZACC,TRJVL1(j),ONE)
             !             Exchange saved seeds, in case Langevin on
             ITPSD1(J) = ITPSD2(I)
          ENDDO

          IF(QHSAMP) THEN
             !             Copy the HTX arrays
             !             This time it's easier to copy from OLD to NEW
             !             As above, NHSAVT will not already be in HTXNEW
             DO J=1,NHSAVT - NSHIFT + 1
                I = HTXOLD(J+NSHIFT-1)
                HTXNEW(J) = i
             ENDDO
          ENDIF

       ELSE

          !           Forward shift
          !           Set up new TRJIDX
          DO I = 1, NREGSV
             J = TRJIDX(I) - SHFNUM
             IF (J  <=  0) J = J + NREGSV
             TRJIDX(I) = j
          ENDDO

          !           Copy all points less than
          !           shift point from 2 to 1

          DO I = 1, SHFNUM
             CALL TPSGET_p(NATOM,XACC,YACC,ZACC,TRJCD2(i),ONE)
             CALL TPSGET_p(NATOM,VXACC,VYACC,VZACC,TRJVL2(i),ONE)
             J = TRJIDX(I)
             CALL TPSSAV_p(NATOM,XACC,YACC,ZACC,TRJCD1(j),ONE)
             CALL TPSSAV_p(NATOM,VXACC,VYACC,VZACC,TRJVL1(j),ONE)
             !             Exchange saved seeds, in case Langevin on
             ITPSD1(J) = ITPSD2(I)
          ENDDO

          IF(QHSAMP) THEN
             !             Copy the HTX arrays
             !             As above, 1 will not be in HTXNEW
             DO J=1,NHSAVT - NSHIFT + 1
                I = HTXOLD(J)
                HTXNEW(J+NSHIFT-1) = i
             ENDDO
          ENDIF
       ENDIF

       IF (QHSAMP) THEN
          ITEMP  => HTXNEW
          HTXNEW => HTXOLD
          HTXOLD => ITEMP

          IF(PDIR  ==  -1) THEN
             CALL TPHDIR(QBASA1,QBASA2,ISEED)
          ELSE
             CALL TPHDIR(QBASA2,QBASA1,ISEED)
          ENDIF
       ENDIF

    ENDIF

    RETURN
  END SUBROUTINE SHFACC

  SUBROUTINE TPFREE(NATOM,NSTEP, &
       IUNCRD,IUNVEL)
    !
    !       Free dynamically allocated arrays for path sampling.
    !
    INTEGER NPPONT, NPPNT1, NPPNT2, NATOM
    INTEGER TPSCD1, TPSVL1, TPSCD2, TPSVL2
    INTEGER NSTEP, IUNCRD, IUNVEL
    !
    INTEGER I, J, N

    IF (NTSAVC   >  0) THEN
       N = (NSTEP/NTSAVC) + 1
       IF (IUNCRD  >  0) call deallocate_tps_trj(trjcd1,natom,n)
       IF (IUNCRD  >  0) call deallocate_tps_trj(trjcd2,natom,n)

       call chmdealloc('tps.src','TPFREE','ITPSD1',N,intgp=ITPSD1)
       call chmdealloc('tps.src','TPFREE','ITPSD2',N,intgp=ITPSD2)

       call chmdealloc('tps.src','TPFREE','ATSHPR',N,intg=ATSHPR)
       call chmdealloc('tps.src','TPFREE','ATSFPR',N,intg=ATSFPR)
       call chmdealloc('tps.src','TPFREE','ATHFPR',N,intg=ATHFPR)
       call chmdealloc('tps.src','TPFREE','ACSHPR',N,intg=ACSHPR)
       call chmdealloc('tps.src','TPFREE','ACSFPR',N,intg=ACSFPR)
       call chmdealloc('tps.src','TPFREE','ACHFPR',N,intg=ACHFPR)
    ENDIF

    IF (NTSAVV   >  0) THEN
       N = (NSTEP/NTSAVV) + 1
       IF (IUNvel  >  0) call deallocate_tps_trj(trjvl1,natom,n)
       IF (IUNvel  >  0) call deallocate_tps_trj(trjvl2,natom,n)
    ENDIF

    call chmdealloc('tps.src','TPINIT','REVVEL%x',NATOM,crl=REVVEL%x)
    call chmdealloc('tps.src','TPINIT','REVVEL%y',NATOM,crl=REVVEL%y)
    call chmdealloc('tps.src','TPINIT','REVVEL%z',NATOM,crl=REVVEL%z)

    call chmdealloc('tps.src','TPFREE','TRJIDX',N,intg=TRJIDX)

    IF(QHSAMP) THEN
       call chmdealloc('tps.src','TPFREE','HTXNEW',NHSAVT,intgp=HTXNEW)
       call chmdealloc('tps.src','TPFREE','HTXOLD',NHSAVT,intgp=HTXOLD)
       call chmdealloc('tps.src','TPFREE','HTXAVG',NHSAVT,intgp=HTXAVG)
    ENDIF
    RETURN
  END SUBROUTINE TPFREE

#else /* (tps_main)*/

SUBROUTINE NULL_TPS
  RETURN
END SUBROUTINE NULL_TPS
#endif /* (tps_main)*/

end module tps

