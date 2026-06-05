#if KEY_IMCUBES==1 /*cubes*/
SUBROUTINE NBNDGCM(NATIM,NIMGRP,NNNB,JNB,MAXJNB, &
     INBLO,X,Y,Z,NIMNB,IMJNB,MAXJMB, &
     IMBLO,IMINB,IMIBLO, &
     NIMNBS,IMJNBS,IMBLOS, &
     NNNBG,MXJnBG,JNBG,INBLOG, &
     INB14,IBLO14,ING14,IGLO14, &
     liminv,iminv,ntrans,imatpt,IMATTR, &
     CUTNB,CTEXNB,LGROUP,LEXTND, &
     LQUAD,LGRAD,LVATOM,WRNMIN,CMPLTD,EPS, &
#if KEY_TSM==1
     LTSM,REACLS,PRODLS, &      
#endif
     RSCMX,RSCMY,RSCMZ,RSQ, &
     RSDX,RSDY,RSDZ,RSQXX,RSQYY,RSQZZ, &
     RSQXY,RSQYZ,RSQZX,RSXMAX,RSYMAX,RSZMAX,RSDISP, &
     RSPOT,RSFX,RSFY,RSFZ, &
     RSGXX,RSGYY,RSGZZ,RSGXY,RSGYZ,RSGZX, &
     ATSX,ATSY,ATSZ,ATPOT,ATFX,ATFY,ATFZ, &
     ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX, &
     resize)
  !
  !-----------------------------------------------------------------------
  !     Made from NBNDGC in Jan. 1999 by Mike Crowley
  !        Added capability to do images by the cubes method.
  !        No group lists are done at all, though this can
  !          be added with little effort.
  !        Not working yet: extended electrostatics
  !                            looks like this needs group list, so this
  !                             will never work with this list builder
  !                         TSM
  !                         REPLICA
  !
  !                 BIG changes for parallel,
  !                    memory savings....Michael Crowley/Scripps
  !                    6/1999
  !
  !     Line numbers in NBNDGC:
  !       666 Error encountered during atom pairlist generation
  !       999 Done with atom pairlist generation
  !
  !     Line number in XDIST:
  !      9999 Exclude this particle pair
  !
  !
  !
#if KEY_LOOKUP==1
  use LOOKUP,only:qvv,ctddw,nnbbycb,nnbibycb,iwwflg                              
#endif
  use chm_kinds
  use number
  use exfunc
  use dimens_fcm
  use psf
  use stream
  use timerm
  use consta
  use memory
#if KEY_PARALLEL==1
  use parallel             
#endif
#if KEY_REPLICA==1
  use replica_mod          
#endif
  use machutil,only:timrb,timre,die
  implicit none
  !
  INTEGER NNNB,MAXJNB
  INTEGER JNB(*)
  INTEGER INBLO(*)
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER NNNBG,MXJNBG
  INTEGER JNBG(*)
  INTEGER INBLOG(*)
  INTEGER   IBLO14(*),IGLO14(*)
  INTEGER INB14(*),ING14(*)
  real(chm_real) CUTNB,CTEXNB
  LOGICAL LGROUP,LEXTND,LQUAD,LGRAD,LVATOM
  real(chm_real) WRNMIN
  LOGICAL CMPLTD
  real(chm_real) EPS
  real(chm_real) RSCMX(*),RSCMY(*),RSCMZ(*),RSQ(*)
  real(chm_real) RSDX(*),RSDY(*),RSDZ(*),RSQXX(*),RSQYY(*),RSQZZ(*)
  real(chm_real) RSQXY(*),RSQYZ(*),RSQZX(*),RSXMAX(*), &
       RSYMAX(*),RSZMAX(*)
  INTEGER RSDISP(*)
  real(chm_real) RSPOT(*),RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real) RSGXX(*),RSGYY(*),RSGZZ(*),RSGXY(*), &
       RSGYZ(*),RSGZX(*)
  real(chm_real) ATSX(*),ATSY(*),ATSZ(*),ATPOT(*), &
       ATFX(*),ATFY(*),ATFZ(*)
  real(chm_real) ATGXX(*),ATGYY(*),ATGZZ(*), &
       ATGXY(*),ATGYZ(*),ATGZX(*)
#if KEY_TSM==1
  LOGICAL LTSM
  INTEGER REACLS(*),PRODLS(*)
#endif 
  !

#if KEY_REPLICA==1
  INTEGER iRepNo, iRepID
#else /**/
  INTEGER repNoA,repNoG,repID,nRepXG,nRepXA
  LOGICAL qRep
#endif /*  REPLICA*/
  !
  INTEGER IGPBD(4)
  INTEGER I,J,IS,IQ,NAT,NGST2,NGPE,NGPX,IRS
  INTEGER JRS,JS,JQ,IRST,JRST
  real(chm_real) CTNBSQ,CTEXSQ
  real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
  real(chm_real) R2,R,XI,YI,ZI
  real(chm_real) QSUM,DXT,DYT,DZT
  real(chm_real) QXXT,QYYT,QZZT,QXYT,QZXT,QYZT,CGT
  real(chm_real) XX,YY,ZZ,XY,YZ,ZX
  real(chm_real) R1,R3,R5,R2X3,R2X5,R2X7,DOT,QXT,QYT,QZT,RQR,CR3
  real(chm_real) TEMP,TEMP2
  logical resize,trial,doimages

  !     RHI                      Radius for distance search
  !     ncubx,ncuby,ncubz        Cube grid dimensions
  !     HPNBRL,HPNBRH            Heap indices for NBRLO, NBRHI arrays
  !     HPNBR                    Heap index for NBR array (XDISTM result)
  !     HPM0,HPM1                Heap indices for XDISTM work arrays
  !     HPN0,HPN1,HPN2           Heap indices for XDISTM work arrays
  !     HPSUM3                   Heap index for XDISTM's SUM3 array
  !     MCL                      Counter for CL arrays
  !     NNNBGI                   Counters like NNNBG (see code)
  !     HPPTR,HPPTR2,HPSIZ,PROBE For heapsort
  !     LCHILD,RCHILD            For heapsort too
  !     X0,Y0,Z0                 Lower left proximal corner of cube grid
  !     NX14,NX14MX              Counters for IGLO14/ING14
  !     NBR                      Counter 1..27 for neighbors
  !     NBRLO, NBRHI             From corresponding XDISTM arrays
  !     XDIST                    Function internal to this file
  !     OK                       .TRUE. iff XDISTM succeeded
  !     EFFIC                    Ratio of XDIST outputs to actual successes
  !     NEAR                     .FALSE. iff need extended electrostatics
  !     JRSA                     ABS(JRS)
  !     HPMBR                    Heap index for MBR, the list array from XDISTM
  !     HPMBRL,HPMBRH            Heap indices for MBRLO, MBRHI image arrays
  !     HPN0M,HPN1M,HPN2M        Heap indices for XDIST image work arrays
  !     HPM0M,HPM1M              Heap indices for XDIST work arrays
  !     HPSUM3M                  Heap index for XDISTM's SUM3 array for images
  !     MAXJMB                   Max size if image list
  !     ...........images...............
  INTEGER HPMBR,HPMBRL,HPMBRH, &
       MAXJMB,HPNBRL0,HPNBRH0,HPMBRL0,HPMBRH0

  integer,allocatable,dimension(:) :: HPN0,HPN1,HPN2,HPSUM3
  integer,allocatable,dimension(:) :: HPM0,HPM1,hpm2
  integer,allocatable,dimension(:) :: HPN0M,HPN1M,HPN2M,HPSUM3M
  integer,allocatable,dimension(:) :: hpn0s,hpn1s,hpn2s
  integer,allocatable,dimension(:) :: HPM0M,HPM1M,hpm2m
  integer,allocatable,dimension(:) :: HPM0s,HPM1s,hpm2s
  integer NATIM,NIMGRP
  integer NIMNB
  integer IMJNB(*),IMBLO(*),IMINB(*),IMIBLO(*),IMATTR(*)
  integer NIMNBS,IMJNBS(*),IMBLOS(*)
  integer MNNBG
  integer itrans,iminv(*),ntrans,imatpt(*)
  logical liminv
  integer istart,ii
  !
  real(chm_real) RHI
  real(chm_real) RHIINV
  !
  integer,parameter :: MAXCUBES=100000

  INTEGER mycube(MAXCUBES),ncubx,ncuby,ncubz,ncube
  !
  integer :: local_prnlev
  INTEGER HPNBRL,HPNBRH
  save  HPNBRL0,HPNBRH0
  save  HPMBRL0,HPMBRH0
  !**************************************************************************
  !MFC****NOTE  HPNBRL,HPNBRH, HPMBRL,HPMBRH
  !              need to be checked and reallocated for changes in
  !              NATOM and NATIM, This is a bug that has not been
  !              corrected for yet.
  !**************************************************************************
  !
  INTEGER HPNBR
  real(chm_real) X0,Y0,Z0
  INTEGER NX14,NX14MX
  LOGICAL XDISTM
  INTEGER NBR
  INTEGER NBRLO, NBRHI
  INTEGER MBRLO, MBRHI
  INTEGER NBRX,MBRX,mbrxs
  LOGICAL OK
  INTEGER EFFIC
  real(chm_real) MARGIN
  LOGICAL NEAR
  INTEGER JRSA
  integer nnnbi,nnnbj

  integer NNNNB,NNIMNB,NTIM,ntima
  real(chm_real) rNNNB,rNIMNB
  integer iok,jok
  real(chm_real) RUOK

  save mycube

  data  HPNBRL0,HPNBRH0/-1,-1/
  data  HPMBRL0,HPMBRH0/-1,-1/

  DATA    EFFIC/1/
  DATA MARGIN/0.01/

  NIMNBS=0
  trial=.false.
  if(resize)trial=.true.
  resize=.false.
  doimages=.true.
  if(natim.eq.natom)doimages=.false.
  ntim=natim-natom
  !
  !---- Initial housekeeping --------------------------------------------
  !
  IF(PRNLEV.GT.2) WRITE(OUTU,'(A)') 'Using Image CUBE search'
  !
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
  CALL WRNDIE(-2,'<NBNDGCM>', &
       'Cannot use image cubes with PARASCAL')
#endif 
#endif 
  IF (TIMER.GT.0) CALL TIMRB
  !
#if KEY_REPLICA==1
  IF (qRep) THEN
     nRepXG = 0
     nRepXA = 0
  ENDIF ! qRep
#endif /*  REPLICA*/
  !
  CMPLTD=.FALSE.            ! Will remain .FALSE. until very end
  CTNBSQ=CUTNB*CUTNB
  !
  ! Store the current atom configuration.
  DO I=1,NATOM
     ATSX(I)=X(I)
     ATSY(I)=Y(I)
     ATSZ(I)=Z(I)
  ENDDO
  !
  IF (LEXTND) THEN
     CALL WRNDIE(-3,'<NBNDGCM>', &
          'BYCB not implemented for EXTENDED electrostatics.')
  END IF
  !
  !==== Start of code specific to NBNDGC =================================
  !
  ! LIMITATIONS/ASSUMPTIONS:  In the interest of efficiency, it is
  ! assumed that the user has set CUTNB sufficiently large as to include
  ! all pairs of groups that are on the 1-4 contact list.  Also, it is
  ! assumed that no ST2 waters are present.
  !
  !---- Zero out result counters ----------------------------------------
  !
  NNNB = 0
  NNNBG = 0
  NIMNB = 0
  MNNBG = 0
  NBRLO=0
  NBRHI=0
  MBRLO=0
  MBRHI=0
  !
  !---- Ascertain that no ST2 waters are present ------------------------
  !
  !
  !---- Find bounding box around molecule -------------------------------
  !
  !     The loop on I is vectorizable.  Again, as above, it was
  !     necessary to use the intrinsic function MIN for the Convex to
  !     vectorize this code.
  !
  XMIN = X(1)
  XMAX = XMIN
  YMIN = Y(1)
  YMAX = YMIN
  ZMIN = Z(1)
  ZMAX = ZMIN
  DO I = 2, NATIM
     XMIN = MIN(XMIN,X(I))
     YMIN = MIN(YMIN,Y(I))
     ZMIN = MIN(ZMIN,Z(I))
     XMAX = MAX(XMAX,X(I))
     YMAX = MAX(YMAX,Y(I))
     ZMAX = MAX(ZMAX,Z(I))
  END DO
  XD = XMAX - XMIN
  YD = YMAX - YMIN
  ZD = ZMAX - ZMIN
  !
  !==== Build atom list if appropriate ==================================
  !
  !     If an error is encountered in this phase, GOTO 666.
  !
  IF (.NOT.LVATOM) GOTO 999
  !
  !---- Establish cube parameters ---------------------------------------
  RHI = CUTNB
  !     Solvent-solvent lookup code needs all atoms within one water
  !     molecule to be in same cube when run in parallel, thus the cube
  !     sides have to be increased by 2xr(O-H). LNI
#if KEY_LOOKUP==1
#if KEY_PARALLEL==1
  IF(QVV.AND.(NUMNOD.GT.1)) RHI=RHI+CTDDW
#endif 
#endif 
  RHIINV = ONE/RHI
  ncubx = INT((XD/RHI) + MARGIN + 1)
  ncuby = INT((YD/RHI) + MARGIN + 1)
  ncubz = INT((ZD/RHI) + MARGIN + 1)
  ncube = ncubx*ncuby*ncubz
#if KEY_PARALLEL==1
  if(ncube.gt.MAXCUBES)then
     call wrndie(-4,'<NBNDGCM>','Maximum cubes limit exceeded.')
  endif
  if(trial)then
     mycube=0
     do i=ncube+mynod+1,MAXCUBES,numnod
        mycube(i)=1
     enddo
  endif
#endif 
  X0 = XMIN - 0.5 * (ncubx * RHI - XD)
  Y0 = YMIN - 0.5 * (ncuby * RHI - YD)
  Z0 = ZMIN - 0.5 * (ncubz * RHI - ZD)
  !
  !---- Allocate work areas for XDIST -----------------------------------
  !      Cannot see what the "effic" parameter is for, set to 1 for now
  !
  call chmalloc('nbndgcm.src','NBNDGCM','HPN0',NATOM,intg=HPN0)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN1',NATOM,intg=HPN1)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN2',NATOM,intg=HPN2)
  call chmalloc('nbndgcm.src','NBNDGCM','HPSUM3',NATOM,intg=HPSUM3)

  call chmalloc('nbndgcm.src','NBNDGCM','HPM0',ncube,intg=HPM0)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM1',ncube,intg=HPM1)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM2',ncube,intg=HPM2)

  !     ........images........
  ntima=ntim+1
  call chmalloc('nbndgcm.src','NBNDGCM','HPN0M',NTIMa,intg=HPN0M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN1M',NTIMa,intg=HPN1M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN2M',NTIMa,intg=HPN2M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPSUM3M',NTIMa,intg=HPSUM3M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN0S',NTIMa,intg=HPN0S)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN1S',NTIMa,intg=HPN1S)
  call chmalloc('nbndgcm.src','NBNDGCM','HPN2S',NTIMa,intg=HPN2S)

  call chmalloc('nbndgcm.src','NBNDGCM','HPM0M',ncube,intg=HPM0M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM1M',ncube,intg=HPM1M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM2M',ncube,intg=HPM2M)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM0S',ncube,intg=HPM0S)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM1S',ncube,intg=HPM1S)
  call chmalloc('nbndgcm.src','NBNDGCM','HPM2S',ncube,intg=HPM2S)

  hpsum3 = zero
  hpsum3m(1:NATIM-NATOM) = zero
#if KEY_TSM==1
  IF (LTSM) THEN
     DO I = 1, NATOM
        IF (REACLS(I).EQ.1) HPSUM3(I) = 1
        IF (PRODLS(I).EQ.1) HPSUM3(I) = 2
     END DO
     DO I = NATOM+1, NATIM
        IF (REACLS(IMATTR(I)).EQ.1) HPSUM3(I-NATOM) = 1
        IF (PRODLS(IMATTR(I)).EQ.1) HPSUM3(I-NATOM) = 2
     END DO
  END IF
#endif 
  !
  !---- Call XDIST -------------------------------------------------------
  !
  !     XDIST solves the basic distance-search problem, with a few
  !     modifications:  it does 1-2 and 1-3 exclusions and 1-4
  !     interactions properly, and it does TSM exclusions properly.
  !     Here we have it NOT return self-self interactions.
  !
  OK = XDISTM(NATOM,NATIM,nimgrp, &
       X,Y,Z, &
       IMOVE,RHI,RHIINV,cutnb, &
       X0,Y0,Z0, ncubx,ncuby,ncubz,ncube,mycube, &
       INBLO(natom+1),inblo(1),JNB,MAXJNB, &
       HPM0,HPM1, &
       HPN0,HPN1,HPN2, &
       HPSUM3, &
       repNoA,repID,nRepXA,qRep,INB14,IBLO14, &
       imblo(natim+natom+1),imblo(natom+1),IMJNB,MAXJMB, &
       IMBLOS(natim+natom+1),imblos(natom+1),IMJNBS, &
       HPM0M,HPM1M, &
       HPM0s,HPM1s, &
       HPN0M,HPN1M,HPN2M, &
       HPN0s,HPN1s,HPN2s, &
       HPSUM3M, &
       IMINB,IMIBLO, &
       NBRX,MBRX,mbrxs, &
       HPM2,HPM2M,HPM2s, &
       .FALSE., &
       liminv,iminv,ntrans,imatpt,IMATTR,ngrp,igpbs, trial)
  if(trial)then
     nbrx=nbrx*1.3
     nnnb=max(200,nbrx)
     mbrx=mbrx*1.3
     NIMNB=max(200,MBRX)
     if(.not.doimages)nimnb=0
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN0M',NTIMa,intg=HPN0M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN1M',NTIMa,intg=HPN1M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN2M',NTIMa,intg=HPN2M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPSUM3M',NTIMa,intg=HPSUM3M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN0S',NTIMa,intg=HPN0S)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN1S',NTIMa,intg=HPN1S)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN2S',NTIMa,intg=HPN2S)

     call chmdealloc('nbndgcm.src','NBNDGCM','HPM0M',ncube,intg=HPM0M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM1M',ncube,intg=HPM1M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM2M',ncube,intg=HPM2M)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM0S',ncube,intg=HPM0S)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM1S',ncube,intg=HPM1S)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM2S',ncube,intg=HPM2S)

     call chmdealloc('nbndgcm.src','NBNDGCM','HPM0',ncube,intg=HPM0)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM1',ncube,intg=HPM1)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPM2',ncube,intg=HPM2)

     call chmdealloc('nbndgcm.src','NBNDGCM','HPN0',NATOM,intg=HPN0)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN1',NATOM,intg=HPN1)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPN2',NATOM,intg=HPN2)
     call chmdealloc('nbndgcm.src','NBNDGCM','HPSUM3',NATOM,intg=HPSUM3)

     resize = .not. OK
     return
  endif
  ruok=zero
  IF (.NOT.OK) THEN
     ruok=one
  ENDIF
#if KEY_PARALLEL==1
  call gcomb(ruok,1)      
#endif
  IF (ruok.gt.HALF) THEN
     OK = .FALSE.
     NNNB=NBRX
     NIMNB=MBRX
     NIMNBs=MBRXs
     GOTO 666
  ENDIF
  !
  OK = .FALSE.              ! For later jumps to 666
  !
  !---- Convert the result to a CHARMM pairlist  -------------------------
  !
  !     The NBR array effectively contains a pairlist, but the runs are
  !     out of order.  The outer loop on I is not vectorizable, but the
  !     inner loop on NBR is.
  !
  NNNB=NBRX
  NIMNB=mbrx
  NIMNBs=mbrxs
  !
  !     Fill in null values for the "real" atom interactions.
  if(doimages)then ! MFC Fix 10-Aug-99
     DO I=1,NATOM
        IMBLO(I)=0
        IMBLOs(I)=0
        IMBLO(NATIM+I)=1
        IMBLOs(NATIM+I)=1
     ENDDO
  endif
  !
  !
  !---- Clean up after building of atom list -----------------------------
  !
  !     Free heap allocations, and return with bad NNNB if error occurred
  !
  OK = .TRUE.
  !
  !     Previous code might have jumped here on error, with .NOT.OK
666 continue
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN0M',NTIMa,intg=HPN0M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN1M',NTIMa,intg=HPN1M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN2M',NTIMa,intg=HPN2M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPSUM3M',NTIMa,intg=HPSUM3M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN0S',NTIMa,intg=HPN0S)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN1S',NTIMa,intg=HPN1S)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN2S',NTIMa,intg=HPN2S)

  call chmdealloc('nbndgcm.src','NBNDGCM','HPM0M',ncube,intg=HPM0M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM1M',ncube,intg=HPM1M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM2M',ncube,intg=HPM2M)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM0S',ncube,intg=HPM0S)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM1S',ncube,intg=HPM1S)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM2S',ncube,intg=HPM2S)

  call chmdealloc('nbndgcm.src','NBNDGCM','HPM0',ncube,intg=HPM0)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM1',ncube,intg=HPM1)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPM2',ncube,intg=HPM2)

  call chmdealloc('nbndgcm.src','NBNDGCM','HPN0',NATOM,intg=HPN0)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN1',NATOM,intg=HPN1)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPN2',NATOM,intg=HPN2)
  call chmdealloc('nbndgcm.src','NBNDGCM','HPSUM3',NATOM,intg=HPSUM3)
  !
  IF (.NOT.OK) THEN
     resize=.true.
     RETURN
  END IF
  !
  !     Line 999 might have been "gone to" if .NOT.LVATOM
  !
999 CONTINUE
  !
  !==== Build group list if appropriate ==================================
  !
  !     No group list implemented by this method at this time.
  !
  !
  !
  !---- Finish extended electrostatics ----------------------------------
  !
  !     The outer loop on IRST is not vectorizable, but the inner loop
  !     on I is.
  !
  !---- Flag success and print statistics --- ---------------------------
  !
  CMPLTD=.TRUE.
  !
#if KEY_LOOKUP==1
  NNBBYCB=NNNB         
#endif
#if KEY_LOOKUP==1
  NNBIBYCB=NIMNB       
#endif

#if KEY_PARALLEL==1
  local_prnlev=plnod0
#else /**/
  local_prnlev=prnlev
#endif 
  IF (local_prnlev.GE.5) THEN
     if(prnlev.ge.5)then
        WRITE(OUTU,*) ' NBNDGCM found:'
        !
        IF (LVATOM) WRITE(OUTU,*) NNNB, ' atom pairs'
        IF (LVATOM) WRITE(OUTU,*) NIMNB, ' image atom pairs'
        IF (LVATOM) WRITE(OUTU,*) NIMNBs, ' self atom pairs'
     endif
#if KEY_PARALLEL==1
     rNNNB=nnnb
     rnimnb=nimnb
     call gcomb(rNNNB,1)
     call gcomb(rNIMNB,1)
     nNNNB=rnnnb
     nnimnb=rnimnb
     IF (PRNLEV.GT.3) WRITE(OUTU,*) NNNNB,  'total atom  pairs'
     IF (PRNLEV.GT.3) WRITE(OUTU,*) NNIMNB, 'total image pairs'
#endif 
#if KEY_REPLICA==1
     IF (qRep) WRITE(OUTU,'(I9,A/)') &
          nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
#endif /*  REPLICA*/
     !
     IF (LGROUP .OR. NST2.GT.0) THEN
        CALL WRNDIE(-2,'<NBNDGCM>', &
             'Cannot use image cubes group list or st2')

     END IF                 ! (LGROUP .OR. NST2.GT.0)
     !
  END IF                    ! (Plnod0.GE.5)
  !
  IF (TIMER.GE.1) THEN
     IF (PRNLEV.GE.2) WRITE(OUTU,*) 'TOTAL TIME IN NBNDGCM: '
     CALL TIMRE
     CALL TIMRB
  END IF                    ! (TIMER.EQ.1)
  IF(PRNLEV.GT.8) THEN
     WRITE(OUTU,889) 'NIMNB ',NIMNB
     WRITE(OUTU,888) 'IMBLO '
     WRITE(OUTU,898) (IMBLO(I),I=1,NATOMT)
     WRITE(OUTU,888) 'IMJNB '
     WRITE(OUTU,898) (IMJNB(I),I=1,NIMNB)
  END IF
888 FORMAT(2X,A6,': '/,(20I5))
898 FORMAT(10I7)
889 FORMAT(2X,A6,': '/,(5I10))
  RETURN
END SUBROUTINE NBNDGCM

!======================================================================
! Main subroutine
!======================================================================
!
LOGICAL FUNCTION XDISTM(NAT,NATIM,nimgrp, &
    X,Y,Z,IMOVE,RHI,RHIINV,cutnb, &
    X0,Y0,Z0,ncubx,ncuby,ncubz,ncube,mycube, &
    NBRLO,NBRHI,NBR,LIMIT, &
    LSTM0,LSTM1, &
    LSTN0,LSTN1,LSTN2, &
    SUM3, &
    repNoGA,repID,nRepX,qRep,INB14,IBLO14, &
    MBRLO,MBRHI,MBR,MLIMIT, &
    MBRLOs,MBRHIs,MBRs, &
    LSTM0M,LSTM1M,LSTM0s,LSTM1s, &
    LSTN0M,LSTN1M,LSTN2M, &
    lstn0s,LSTN1s,LSTN2s, &
    SUM3M, &
    IMB14,IMBLO14, &
    NBRX,MBRX,mbrxs, &
    LSTM2,LSTM2M,LSTM2s, &
    QSELF , &
    liminv,iminv,ntrans,imatpt,IMATTR,ngrp,igpbs, &
    trial)
  !-----------------------------------------------------------------------

  use new_timer,only:timer_start,timer_stop,         & 
     T_setgrd,T_bldlst,T_grdim,T_grduc            
#if KEY_LOOKUP==1
  use LOOKUP                                        
#endif

  use chm_kinds
  use exfunc
  use stream
#if KEY_PARALLEL==1
  use parallel        
#endif
 implicit none
#if KEY_PARALLEL==0
 integer mynod,numnod
 parameter (mynod=0,numnod=1)
#endif 
 !
 !     Parameters
 !
 !     NAT                          Number of particles
 !     X(NAT),Y(NAT),Z(NAT)         Coordinates
 !     IMOVE(NATim)                   IMOVE Fixed(1), LP(-1), 0 otherwise
 !     RHI                          Maximum desired distance
 !     X0,Y0,Z0                     Corner of cube [1,1,1]
 !     LIMIT                        Max # of pairs expected
 !     ncubx,ncuby,ncubz            Number of cubes in each direction
 !
 !     NBRLO(NAT)                   Neighbors of atom N are listed in
 !     NBRHI(NAT)                   NBR(NBRLO(N)..NBRHI(N))
 !     NBR(LIMIT)
 !
 !     LSTM0(ncube),LSTM1(ncube)      Work arrays--see explanation
 !     LSTN0(NAT),LSTN1(NAT),LSTN2(NAT) More work arrays
 !
 !     SUM3(NAT)                    For each atom, a number in 0..2.
 !
 !     QSELF                         .TRUE. iff want self-self int'ns
 !
 INTEGER NAT,NATIM,NTIM,nimgrp
 real(chm_real) X(NATim),Y(NATim),Z(NATim)
 INTEGER IMOVE(NATim)
 real(chm_real) RHI,RHIINV,cutnb
 real(chm_real) X0,Y0,Z0
 INTEGER LIMIT
 INTEGER ncubx,ncuby,ncubz,ncube,mycube(*)
 INTEGER NBRLO(NAT+1)
 INTEGER NBRHI(NAT)
 INTEGER NBR(LIMIT)
 INTEGER LSTM0(ncube),LSTM1(ncube)
 INTEGER LSTN0(NAT),LSTN1(NAT),LSTN2(NAT)
 INTEGER INB14(*),IBLO14(*)
 INTEGER SUM3(NAT)
 LOGICAL QSELF
 !.........IMAGES...............
 INTEGER MLIMIT,MLIMITs
 INTEGER MBRLO(NATIM+1),MBRLOs(NATIM+1)
 INTEGER MBRHI(NATIM),MBRHIs(NATIM)
 INTEGER MBR(MLIMIT),MBRs(MLIMIT)
 INTEGER LSTM0M(ncube),LSTM1M(ncube),LSTM0s(ncube),LSTM1s(ncube)
 INTEGER LSTM2(ncube),LSTM2M(ncube),LSTM2s(ncube)
 INTEGER LSTN0M(NATIM-NAT),LSTN1M(NATIM-NAT),LSTN2M(NATIM-NAT)
 INTEGER lstn0s(natim-nat), LSTN1s(NATIM-NAT),LSTN2s(NATIM-NAT)
 INTEGER IMB14(*),IMBLO14(*)
 INTEGER SUM3M(NATim)
 integer iminv(*),ntrans,imatpt(*),IMATTR(*),itrans,ngrp
 logical liminv
 integer ns
 integer igpbs(*)
 ! passed
 INTEGER repNoGA(*), repID(*), nRepX
 LOGICAL qRep
#if KEY_REPLICA==1
 ! local
 INTEGER iRepNo, iRepID
#endif /*  REPLICA*/
 !
 CHARACTER CHAR(80) ! DEBUG
 !
 !     Explanation of parameters that are conceptually essential:
 !
 !     NAT,X,Y,Z   number of atoms; their coordinates
 !     IMOVE       array specifying whether given particle moves (0),
 !                  fixed(1), lone pair(-1)
 !     0..RHI2     range of interparticle distances of interest.
 !
 !     Explanation of parameters interesting only to implementors:
 !
 !     The parameters X0..Z0, M[XYZ]MAX define the cube mapping thus:
 !     3D space is divided into cubes of side RHI.  These cubes are
 !     named by three integer indices: INT( (X-X0)/RHI ), and so forth.
 !     The indices must lie in the range 1..ncubx, and so forth.
 !
 !     The arrays NBRLO, NBRHI and NBR hold the results thus:
 !     For given particle number N, the particles N2 >= N that lie in
 !     the specified range are found in ascending order in the subarray
 !     NBR(NBRLO(N)..NBRHI(N)).  IMPORTANT:  Because the final loop is
 !     over cubes, the subsequences NBR(NBRLO(N)..NBRHI(N)) are not in
 !     order of N.
 !
 !     The self-self interaction N2 == N is included iff QSELF.
 !     1-2 and 1-3 interactions are left
 !     off NBR() and 1-4 interactions are included as negative numbers.
 !
 !     The work arrays LSTM[01] and LSTN[012] are used to hold
 !     intermediate results.  See the comments in the code for a broad
 !     overview of how the results are stored.
 !
 !----------------------------------------------------------------------
 !
 !     Local data
 !     ----------
 !
 !     Various distances
 !     RHI2               Range of desired squared distances
 !     DDX,DDY,DDZ        Work variables to compute interparticle
 !
 !     Sundry indices in cube space
 !     M,M2,MX,MY,MZ      Work indices for cubes
 !     ncube              Number of cubes calc'ed from ncubx etc
 !     ncubxy             ncubx * ncuby
 !     MADD(27)           Offsets to use for neighbors, excl self
 !
 !     Some particle indices
 !     N,N2               Particle indices
 !
 !     Subarray limits and indices
 !     NLO,NHI,NLO2,NHI2  Limits for vectorized cube search
 !     I                  Index for LSTN0 in final loop
 !     NIX,NIX2           Indices for vectorized cube search
 !
 !     NBR27              1..27 (which neighbor)
 !     NBRN               How many particles in neighbors 1..27
 !     NBRNR              How many close particles in neighbors 1..27
 !
 !     Miscellaneous items
 !     NBRX               Index of NBR array
 !     XX,YY,ZZ           Coordinates of atom in outer loop
 !     FIXED              IMOVE(N).gt.0
 !     SUM3N              SUM3(N)
 !     TMP                Miscellaneous integer
 !     EXCL14             (+1,-1,0) if (exclude, 1-4, neither)
 !     DONE               Controls while loops
 !
 real(chm_real) RHI2,cutnbsq
 real(chm_real) DDX,DDY,DDZ
 INTEGER M,M2,MX,MY,MZ
 INTEGER ncubxy
 INTEGER MADD(27)
 INTEGER N,N2
 INTEGER NLO,NHI,NLO2,NHI2
 INTEGER I,ILO
 INTEGER NIX,NIX2
 INTEGER NBR27
 INTEGER NBRN,NBRNR
 !
 INTEGER NBRX,NBRTOT
 real(chm_real) XX,YY,ZZ
 LOGICAL FIXED
 INTEGER SUM3N
 INTEGER TMP
 INTEGER EXCL14
 LOGICAL DONE
 INTEGER INBX, NXIMAX, NXI
 !
 INTEGER IM,MLO,MHI,mslo,mshi
 INTEGER MBRX,NXIM,NXIMMAX
 INTEGER MBRXs,NXIMs,NXIMMAXs
 integer mm,lim
 integer mmint,mmaxt,mdel
 !============================================================================
 integer krs,irs,is,jtrans,ktrans
 integer itemp,istrt,iend,j
 integer nself,nreg
 !============================================================================
 !
 logical trial,doimages
 !
 !       balancing variables
 integer nbrt,nbrshare,nbrstart,npprs
 integer mmtop,mmbot,itag

 ! number of elements in lstn2m vector
 ! an upper bound on nbrnr
 integer :: lstn2m_num_elts

 !----------------------------------------------------------------------
 !     Executable code begins here

 lstn2m_num_elts = natim - nat

 call timer_start(T_setgrd)
 XDISTM = .FALSE.
 !----------------------------------------------------------------------
 !
 NTIM=NATIM-NAT
 doimages=.true.
 if(ntim.eq.0)doimages=.false.
 !
 !---- Compute frequently used intermediate scalar results -------------
 !
 RHI2 = RHI * RHI
 cutnbsq=cutnb*cutnb
 ncubxy = ncubx * ncuby
 !------------------------------------------------------------------------
 !---- Check if the total number of cubes has increased and assign them
 !        round_robin for parallel or at least set mycybe() to one for
 !        one processor running
 !------------------------------------------------------------------------

 !
 IF (PRNLEV.GE.6) THEN
    WRITE (OUTU,*) &
         'NBONDGM Building particle interaction list using grid'
    WRITE (OUTU,*) 'Number of primary particles    =', NAT
    WRITE (OUTU,*) 'Number of image particles      =', NTIM
    WRITE (OUTU,*) 'Number of cells in X dimension =', ncubx
    WRITE (OUTU,*) 'Number of cells in Y dimension =', ncuby
    WRITE (OUTU,*) 'Number of cells in Z dimension =', ncubz
    WRITE (OUTU,*) 'Number of cells, total         =', ncube
    WRITE (OUTU,*) 'Cell size                      =', RHI
 END IF
 !
 !---- Prepare MADD array-----------------------------------------------
 !
 !     The MADD array is defined so that for a given cube M,
 !     the 27-cube neighborhood consists of the cubes M+MADD(1..27).
 !     The inner loop is vectorizable.
 !
 DO MX = -1, 1
    DO MY = -1, 1
       DO MZ = -1, 1
          NBR27 = (MX+1)*9 + (MY+1)*3 + (MZ+1) + 1
          MADD(NBR27) = MX + ncubx * (MY + ncuby * MZ)
       END DO
    END DO
 END DO
 !
 !     Remove duplicate offsets from MADD, or we will get duplicate
 !     atoms and therefore array-bound problems down the line.  Any
 !     element that is set to NCUBE is effectively deleted since later
 !     M+NCUBE is out of the range 1..NCUBE.
 !
 !     Crude method OK since max number of operations is 13 * 27 = 351
 !
 !     Not vectorizable.
 !
 DO NBR27 = 1, 27
    DO TMP = 1, NBR27-1
       IF (MADD(TMP).EQ.MADD(NBR27)) MADD(NBR27) = NCUBE + 1
    END DO
 END DO
 !
 !---- Prepare a list of particles for each cube -----------------------
 !
 !     The result will, after several steps, be in a CHARMM pairlist
 !     such that for a given cube M, the indices of the particles in M
 !     are found in the contiguous elements LSTN1(LSTM1(M-1)+1..LSTM1(M)).
 !
 !     First, temporarily set up LSTN1 such that for a given atom N,
 !     LSTN1(N) is the index of the cube it's in.  This will be
 !     discarded after the next step.  Vectorizable.
 !
 DO N = 1, NAT
    LSTN1(N) = ( INT((Z(N)-Z0)*RHIINV)  * ncuby &
         + INT((Y(N)-Y0)*RHIINV)) * ncubx &
         + INT((X(N)-X0)*RHIINV) + 1
 END DO
#if KEY_LOOKUP==1
#if KEY_PARALLEL==1
 !     Put water hydrogens in same box as the oxygen of the same molecule
 !     IWWFLG(N) is non-zero only for solvent atoms involved in lookups,
 !     and then it is equal to the atom number of the oxygen.
 !     Oxygens are supposed to come first so we do not check for this...
 !
 IF(QVV.AND.(NUMNOD.GT.1))THEN
    DO N = 1, NAT
       IF(IWWFLG(N).GT.0 .AND. N.NE.IWWFLG(N))THEN
          LSTN1(N)=LSTN1(IWWFLG(N))
       ENDIF
    ENDDO
 ENDIF
#endif 
#endif 
 !============================================================================
 !============================================================================
 !     NTRANS    INTEGER*4             Number of transformations
 !     NIMGRP    INTEGER               Number of total groups
 !     IMINV     INTEGER  (NTRANS)     Inverse transformation number
 !     IMATPT    INTEGER   (NTRANS)    Pointer to last atom of
 !                                     transformation
 !     IMATTR    INTEGER   (NIMGRP)    Transformation no. for each group
 !     IGPBS     Groups   Base pointer to first atom in each group.
 !
 do i=1,ntim
    LSTN1S(i)=0
    LSTN1m(i)=0
 enddo
 nself=0
 nreg=0
 itemp=nat+1
 DO ITRANS=1,NTRANS
    ISTRT=ITEMP
    IEND=IMATPT(ITRANS)
    ITEMP=IEND+1
    if(itrans.eq. iminv(itrans))then
       DO I=ISTRT,IEND
          nself=nself+1
          LSTN1S(i-NAT) = ( INT((Z(i)-Z0)*RHIINV)  * ncuby &
               + INT((Y(i)-Y0)*RHIINV)) * ncubx &
               + INT((X(i)-X0)*RHIINV) + 1
       ENDDO
#if KEY_LOOKUP==1
#if KEY_PARALLEL==1
       !     Put water hydrogens in same box as the oxygen of the same molecule
       !     IWWFLG(N) is non-zero only for solvent atoms involved in lookups,
       !     and then it is equal to the atom number of the oxygen.
       !     Oxygens are supposed to come first so we do not check for this...
       !
       IF(QVV)THEN
          DO I = ISTRT,IEND
             IF(IWWFLG(I).GT.0 .AND. I.NE.IWWFLG(I))THEN
                LSTN1S(I-NAT)=LSTN1S(IWWFLG(I)-NAT)
             ENDIF
          ENDDO
       ENDIF
#endif 
#endif /*   ywfix Jul-6-2007 of ln070628*/

    else
       if(itrans.lt.iminv(itrans) )then
          DO I=ISTRT,IEND
             nreg=nreg+1
             LSTN1M(i-NAT) = ( INT((Z(i)-Z0)*RHIINV)  * ncuby &
                  + INT((Y(i)-Y0)*RHIINV)) * ncubx &
                  + INT((X(i)-X0)*RHIINV) + 1
          ENDDO
#if KEY_LOOKUP==1
#if KEY_PARALLEL==1
          IF(QVV.AND.(NUMNOD.GT.1))THEN
             DO I = ISTRT,IEND
                IF(IWWFLG(I).GT.0 .AND. I.NE.IWWFLG(I))THEN
                   LSTN1M(I-NAT)=LSTN1M(IWWFLG(I)-NAT)
                ENDIF
             ENDDO
          ENDIF
#endif 
#endif 
       endif
    endif
 enddo
#if KEY_OLD_IMCUBES==1
 DO N = NAT+1, NATIM
    LSTN1M(N-NAT) = ( INT((Z(N)-Z0)*RHIINV)  * ncuby &
         + INT((Y(N)-Y0)*RHIINV)) * ncubx &
         + INT((X(N)-X0)*RHIINV) + 1
 END DO
#if KEY_LOOKUP==1
#if KEY_PARALLEL==1
 IF(QVV.AND.(NUMNOD.GT.1))THEN
    DO I = NAT+1,NATIM
       IF(IWWFLG(I).GT.0 .AND. I.NE.IWWFLG(I))THEN
          LSTN1M(I-NAT)=LSTN1M(IWWFLG(I)-NAT)
       ENDIF
    ENDDO
 ENDIF
#endif 
#endif 

 !============================================================================
 !============================================================================
#endif 

 !
 !     Invert LSTN1:  Set up LSTM0/LSTN0 as intertwined linked lists
 !     such that for a given cube M, you can recursively read off the
 !     atoms that it contains.  This will be discarded after the next
 !     step.
 !
 !     Vectorizable.
 DO M = 1, NCUBE
    LSTM0(M) = 0
    LSTM0M(M) = 0
    LSTM0s(M) = 0
    LSTM2(M)=0
    LSTM2M(M)=0
    LSTM2s(M)=0
 END DO
 !
 !     This loop cannot be vectorized because it contains a recurrence.
 DO N = 1, NAT
    M = LSTN1(N)
    LSTN0(N) = LSTM0(M)
    LSTM0(M) = N
 END DO
 DO N =  1, NTIM
    if(lstn1m(n).gt.0)then
       M = LSTN1M(N)
       LSTN0M(N) = LSTM0M(M)
       LSTM0M(M) = N
    endif
 END DO
 DO N =  1, NTIM
    if(lstn1s(n).gt.0)then
       M = LSTN1s(N)
       LSTN0s(N) = LSTM0s(M)
       LSTM0s(M) = N
    endif
 END DO
 !
 !     Convert to CHARMM pairlist:  make LSTM1/LSTN1 a CHARMM pairlist
 !     equal to LSTM0/LSTN0.  Incidentally, we know that LSTN1 will
 !     contain lots of little runs of descending indices because each
 !     linked list in LSTM0/LSTN0 is traversed in descending order of
 !     particle index.  We will capitalize on this in a later sort
 !     operation.
 !
 !     This loop cannot be vectorized because it contains a recurrence.
 !     I is LSTN1 location most recently filled
 !
 I = 0
 IM = 0
 Is = 0
 DO M = 1, NCUBE
    !
    !        .... primary .....
    N = LSTM0(M)
    DO WHILE (N.NE.0)
       I = I + 1
       LSTN1(I) = N
       N = LSTN0(N)
    END DO
    LSTM1(M) = I
    !
    !        .... images .....
    N = LSTM0M(M)
    DO WHILE (N.NE.0)
       IM = IM + 1
       LSTN1M(IM) = N
       N = LSTN0M(N)
    END DO
    LSTM1M(M) = IM
    !
    !        .... self images .....
    N = LSTM0s(M)
    DO WHILE (N.NE.0)
       Is = Is + 1
       LSTN1s(Is) = N
       N = LSTN0s(N)
    END DO
    LSTM1s(M) = Is

 END DO
 !
 !---- Create the final result:  NBR/NBRLO/NBRHI -----------------------
 !
 !     This is the CPU-intensive part of the code.  It is written
 !     for efficiency, not clarity--so unfortunately it is a rather
 !     tall nested loop.
 !
 !     One major design choice was to make M the outer loop.  This
 !     permits one to cache results for which a particular cube M is at
 !     the center of the 27-cube neighborhood.
 !
 !     For each cube M, first store indices of all atoms from 27-cube
 !     region in LSTN0.  Sort LSTN0 in ascending order--once per cube
 !     instead of once per particle.  Then, require N2 >= N during
 !     vectorized distance comparisons. If (.NOT.QSELF), disallow N2 ==
 !     N.
 !
 !     Don't attempt to avoid the wraparound effect:  for a given M,
 !     not all of the cubes M+MADD(1..27) are actually neighbors... but
 !     on a vector machine this doesn't matter.
 !
 !     There is a major loop over cube index M, in which I will
 !     intersperse comment dividers.
 !
 !     First, empty out the return array:
 !
 NBRX = 1
 MBRX = 1
 MBRXs = 1
 NHI = 0
 MHI = 0
 MSHI = 0
 do m=1,nat
    nbrlo(m)=0
    nbrhi(m)=0
 enddo
 do m=1,natim-nat
    mbrlo(m)=0
    mbrhi(m)=0
    mbrlos(m)=0
    mbrhis(m)=0
 enddo
 !
 !---- Start of major loop over cubes M --------------------------------
 !
 !     This outer loop cannot be vectorized.
 !
#if KEY_PARALLEL==1
 if(trial) then
    MDEL=(NCUBE-1)/numnod + 1
    MMINT=mynod*mdel+1
    mmaxt=min(MMINT+mdel-1,ncube)
 else
#endif 
    mmint=1
    mmaxt=ncube
#if KEY_PARALLEL==1
 endif
#endif 
 call timer_stop(T_setgrd)
 call timer_start(T_bldlst)
 NBRTOT=0
 !  --------------------------------------------------------------------
 DO M = MMINT, MMAXT
#if KEY_PARALLEL==1
    if(.not.trial .and. mycube(m).ne.1 )goto 666
#endif 
    !
    !------- Determine range of atoms LSTN1(NLO..NHI) in center cube M ----
    !
    if(m.eq.1)then
       NLO =  1
       MLO =  1
       MSLO = 1
    else
       NLO=LSTM1(M-1)+1
       MLO=LSTM1M(M-1)+1
       MSLO=LSTM1s(M-1)+1
    ENDIF
    NHI = LSTM1(M)
    MHI = LSTM1M(M)
    MSHI = LSTM1s(M)
    !
    !------- Set LSTN0 equal to an array of neighbor atoms ----------------
    !
    !        LSTN0 will be an array of the indices of atoms which are in
    !        the cube M and its neighbors.  In the outer loop, M2 takes on
    !        up to 27 indices of cubes adjacent to M.  For each M2,
    !        LSTN1(NLO2..NHI2) is the set of atoms in M2.  We traverse
    !        that list backwards so we end up with little runs of
    !        ascending order, which makes the sort in the next step
    !        easier.
    !
    !        This outer loop cannot be vectorized.
    !
    NBRN = 0
    DO NBR27 = 1, 27
       !
       !           Propose an M2, and if it's in range, then...
       M2 = M + MADD(NBR27)
       IF (M2.GE.1 .AND. M2.LE.NCUBE) THEN ! ----------
          !
          !              Set NLO2..NHI2 equal to range of indices into LSTN1
          IF (M2.EQ.1) THEN
             NLO2 = 1
          ELSE
             NLO2 = LSTM1(M2-1) + 1
          END IF
          NHI2 = LSTM1(M2)
          !
          !              Loop over those indices, filling LSTN0 with atom numbers
          !              Vectorizable.
          DO NIX2 = NHI2, NLO2, -1
             NBRN = NBRN + 1
             LSTN0(NBRN) = LSTN1(NIX2)
          END DO
          !
       END IF                             ! ----------
    END DO
    !
    !      call timer_start(T_grdim)

    call qsort_ints(LSTN0, NBRN)

    !      call timer_stop(T_grdim)
    !
    !------- For every atom N in M, build the corresponding NBR subarray ..
    !
    !        The outer loops NIX and I generate atom indices N and N2,
    !        respectively.  The result will end up in NBR(NBRLO(N)..NBRHI(N)).
    !        Since LSTN0(1..NBRN) is already sorted, we know that our
    !        result will be sorted too.
    !
    !        This outer loop (NIX) is not vectorizable.  The inner loop
    !        (I) is fully vectorizable.
    !
    DO NIX = NLO, NHI
       !
       !           Record often-used parameters for atom N
       N = LSTN1(NIX)
       XX = X(N)
       YY = Y(N)
       ZZ = Z(N)
       FIXED = IMOVE(N).GT.0
       SUM3N = SUM3(N)
       !
       IF(N.GT.1) THEN
          NXI=IBLO14(N-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(N)
       !
#if KEY_REPLICA==1
       IF (qRep) THEN
          iRepNo = RepNoGA(N)
          iRepID = RepID(iRepNo)
       ENDIF ! qRep
#endif /*  REPLICA*/
       !
       !           Start this run of N2's
       LIM=LIMIT
       NBRLO(N) = NBRX-1
       !
       !           Quickly find first index ILO for which LSTN0(ILO) >= N
       !           Expect to throw out 50% of N2's this way, on average
       ILO = NBRN + 1
       DO I = 1, NBRN
          IF( LSTN0(I).GE.N ) ILO = MIN(ILO,I)
       END DO
       !
       !           Handle self-self interactions
       IF( ILO.LE.NBRN .AND. LSTN0(ILO).EQ.N ) THEN
          ILO = ILO + 1
          IF (QSELF) THEN
             NBR(NBRX) = -N
             NBRX = NBRX + 1
          END IF
       END IF
       !
       !           Construct, in LSTN2, an array of N2's within distance cutoff
       !           Expect to throw out 85% of N2's, for large systems
       NBRNR = 0
       !$$$            TZERO = CPUTIME(0.0)
       !      call timer_start(T_grduc)
       DO I = ILO, NBRN
          N2 = LSTN0(I)
          DDX = X(N2) - XX
          DDY = Y(N2) - YY
          DDZ = Z(N2) - ZZ
          IF (DDX*DDX+DDY*DDY+DDZ*DDZ .LE. cutnbsq) THEN
             NBRNR = NBRNR + 1
             LSTN2(NBRNR) = N2
          END IF
       END DO
       !      call timer_stop(T_grduc)
       !
       !           Enforce remaining conditions, placing results in NBR
       !           Fraction of pairs thrown out depends on many factors
       !$$$            TZERO = CPUTIME(0.0)
       DO I = 1, NBRNR
          !
          N2 = LSTN2(I)
          !
          !              Determine exclusion status from exclusion array
          !              then disqualify if
          !              (1) both fixed
          !              (2) one reactant (SUM3==1) and one product (SUM3==2)
          !              (3) if replica. exclude as explained below.

          !C               is there an exclusion between atoms N and N2?
          !C                 EXCL14 = -1,0,1??
          !
          ! Search through the exclusion lists for this atom pair
          !  (NOTE: This will be only slightly slower than the old ZTBL code,
          !   but it works and doesn't require the memory for ZTBL.
          !   This change was expedient to allow for very large systems,
          !   or for when REPLICA is heavily used.  - BRB  11/7/96
          !
          INBX=IABS(INB14(NXI))
          DO WHILE(NXI.LE.NXIMAX .AND. N2.GT.INBX)
             NXI=NXI+1
             INBX=IABS(INB14(NXI))
          ENDDO
          IF(NXI.GT.NXIMAX) THEN
             EXCL14=1
          ELSE IF(N2.EQ.INB14(NXI)) THEN  ! exclusion found
             EXCL14=0
          ELSE
             IF(N2.EQ.INBX) THEN
                EXCL14=-1    ! it's a 1-4 pair
             ELSE
                EXCL14=1
             ENDIF
          ENDIF
          !
          IF (FIXED .AND. IMOVE(N2).gt.0) EXCL14 = 0
          ! ==================================================================
#if KEY_REPLICA==1
          !# <caves>-Aug-4-1993 (Leo Caves)
          ! Replica Exclusions (Atom/Group Exclusion)
          ! Rationale: If groups belong to same subsystem (repID) but are not in
          ! same replica unit (repNoGA) - then EXCLUDE the atom/group pair
          IF (qRep) THEN
             IF ( iRepID .EQ. repID(repNoGA(N2)) .AND. &
                  iRepNo .NE. repNoGA(N2) )   THEN
                nRepX = nRepX + 1
                EXCL14 = 0
             ENDIF
          ENDIF
#endif /*  REPLICA*/
          ! ==================================================================
#if KEY_TSM==1
          IF ((SUM3N+SUM3(N2)).EQ.3) EXCL14 = 0
#endif 
          !
          !              0: Don't add; 1: add positive; -1: add negative
          IF( EXCL14.NE.0 ) THEN
             if(.not.trial)then
                NBR(NBRX) = SIGN(N2,EXCL14)
                NBRX = NBRX + 1
                IF (NBRX .GT. LIM) THEN
                   call timer_stop(T_bldlst)
                   RETURN
                endif
             else
                NBRX = NBRX + 1
                LSTM2(M)=LSTM2(M)+1
             endif
          END IF
          !
       END DO
       !                 .....end of I=1,NBRNR loop
       NBRHI(N) = NBRX - 1
    END DO
    !               .....end of NIX=MLO,MHI loop
    !----------------End primary ----------------------------------------
    !
    !----------------Start images ----------------------------------------
    if(doimages)then
       DO NIX = MLO, MHI
          !
          !           Record often-used parameters for atom N
          N = LSTN1M(NIX)+NAT
          XX = X(N)
          YY = Y(N)
          ZZ = Z(N)
          FIXED = IMOVE(N).gt.0
          SUM3N = SUM3M(N-NAT)
          !
          IF(N.GT.1) THEN
             NXIM=IMBLO14(N-1)+1
          ELSE
             NXIM=1
          ENDIF
          NXIMMAX=IMBLO14(N)
          !
#if KEY_REPLICA==1
          IF (qRep) THEN
             iRepNo = RepNoGA(N)
             iRepID = RepID(iRepNo)
          ENDIF ! qRep
#endif /*  REPLICA*/
          !
          !           Start this run of N2's
          MBRLO(N-NAT) = MBRX-1
          NBRNR=0
          ILO=1
          DO I = ILO, NBRN
             N2 = LSTN0(I)
             DDX = X(N2) - XX
             DDY = Y(N2) - YY
             DDZ = Z(N2) - ZZ
             IF (DDX*DDX+DDY*DDY+DDZ*DDZ .LE. cutnbsq) THEN
                NBRNR = NBRNR + 1
                LSTN2M(NBRNR) = N2
             END IF
             if (nbrnr .eq. lstn2m_num_elts) exit
          END DO
          !
          !           Enforce remaining conditions, placing results in NBR
          !           Fraction of pairs thrown out depends on many factors
          DO I = 1, NBRNR
             !
             N2 = LSTN2M(I)
             !
             !              Determine exclusion status from exclusion array
             !              then disqualify if
             !              (1) both fixed
             !              (2) one reactant (SUM3==1) and one product (SUM3==2)
             !              (3) if replica. exclude as explained below.
             !               is there an exclusion between atoms N and N2?
             !                 EXCL14 = -1,0,1??
             !
             ! Search through the exclusion lists for this atom pair
             !  (NOTE: This will be only slightly slower than the old ZTBL code,
             !   but it works and doesn't require the memory for ZTBL.
             !   This change was expedient to allow for very large systems,
             !   or for when REPLICA is heavily used.  - BRB  11/7/96
             !
             INBX=IABS(IMB14(NXIM))
             DO WHILE(NXIM.LE.NXIMMAX .AND. N2.GT.INBX)
                NXIM=NXIM+1
                INBX=IABS(IMB14(NXIM))
             ENDDO
             IF(NXIM.GT.NXIMMAX) THEN
                EXCL14=1
             ELSE IF(N2.EQ.IMB14(NXIM)) THEN  ! exclusion found
                EXCL14=0
             ELSE
                IF(N2.EQ.INBX) THEN
                   EXCL14=-1    ! it's a 1-4 pair
                ELSE
                   EXCL14=1
                ENDIF
             ENDIF
             !
             IF (FIXED .AND. IMOVE(N2).gt.0) EXCL14 = 0
             ! ==================================================================
#if KEY_REPLICA==1
             !# <caves>-Aug-4-1993 (Leo Caves)
             ! Replica Exclusions (Atom/Group Exclusion)
             ! Rationale: If groups belong to same subsystem (repID) but are not in
             ! same replica unit (repNoGA) - then EXCLUDE the atom/group pair ELSE
             IF (qRep) THEN
                IF ( iRepID .EQ. repID(repNoGA(N2)) .AND. &
                     iRepNo .NE. repNoGA(N2) )   THEN
                   nRepX = nRepX + 1
                   EXCL14 = 0
                ENDIF
             ENDIF
#endif /*  REPLICA*/
             ! ==================================================================
#if KEY_TSM==1
             IF ((SUM3N+SUM3(N2)).EQ.3) EXCL14 = 0
#endif 
             !
             !              0: Don't add; 1: add positive; -1: add negative
             IF( EXCL14.NE.0 ) THEN
                if(.not.trial)then
                   MBR(MBRX) = SIGN(N2,EXCL14)
                   MBRX = MBRX + 1
                   IF (MBRX .GT. MLIMIT)then
                      call timer_stop(T_bldlst)
                      RETURN
                   END IF
                else
                   MBRX = MBRX + 1
                   LSTM2M(M)=LSTM2M(M)+1
                endif
             END IF
             !
          END DO
          MBRHI(N-NAT) = MBRX - 1
       END DO
    endif
    !
    !
    !----------------End images ----------------------------------------
    !
    !----------------Start Self images ----------------------------------------
    if(doimages .and. (is .gt. 0) )then
       mlimits=natim
       DO NIX = MsLO, MsHI
          !
          !           Record often-used parameters for atom N
          N = LSTN1s(NIX)+NAT
          ns = imattr(n)
          XX = X(N)
          YY = Y(N)
          ZZ = Z(N)
          FIXED = IMOVE(N).gt.0
          SUM3N = SUM3M(N-NAT)
          !
          IF(N.GT.1) THEN
             NXIM=IMBLO14(N-1)+1
          ELSE
             NXIM=1
          ENDIF
          NXIMMAX=IMBLO14(N)
          !
#if KEY_REPLICA==1
          IF (qRep) THEN
             iRepNo = RepNoGA(N)
             iRepID = RepID(iRepNo)
          ENDIF ! qRep
#endif /*  REPLICA*/
          !
          !           Start this run of N2's
          MBRLOs(N-NAT) = MBRXs-1
          MBRLO(N-NAT) = MBRX-1
          NBRNR=0
          ILO=0
          n2=0

          do while(n2.lt.ns .and. ilo.le.nbrn)
             ilo=ilo+1
             N2 = LSTN0(ILO)
          enddo
          INBX=IABS(IMB14(NXIM))
          DO WHILE(NXIM.LE.NXIMMAX .AND. N2.GT.INBX)
             NXIM=NXIM+1
             INBX=IABS(IMB14(NXIM))
          ENDDO
          IF(NXIM.GT.NXIMMAX) THEN
             EXCL14=1
          ELSE IF(N2.EQ.IMB14(NXIM)) THEN  ! exclusion found
             EXCL14=0
          ELSE
             IF(N2.EQ.INBX) THEN
                EXCL14=-1    ! it's a 1-4 pair
             ELSE
                EXCL14=1
             ENDIF
          ENDIF
          !
          IF (FIXED .AND. IMOVE(N2).gt.0) EXCL14 = 0
          if(n2.eq.ns)then
             if(.not.trial) then
                MBRs(MBRXs) = SIGN(N2,EXCL14)
             endif
             ilo=ilo+1
             IF (MBRXs .GT. MLIMITs)then
                call timer_stop(T_bldlst)
                write(outu,*) "Self List overflow",mbrxs-1,mlimits
                RETURN
             END IF
             MBRXs = MBRXs + 1
          endif
          MBRHIs(N-NAT) = MBRXs - 1
          IF(N.GT.1) THEN
             NXIM=IMBLO14(N-1)+1
          ELSE
             NXIM=1
          ENDIF
          NXIMMAX=IMBLO14(N)

          DO I = ILO, NBRN
             N2 = LSTN0(I)
             DDX = X(N2) - XX
             DDY = Y(N2) - YY
             DDZ = Z(N2) - ZZ
             IF (DDX*DDX+DDY*DDY+DDZ*DDZ .LE. cutnbsq) THEN
                NBRNR = NBRNR + 1
                LSTN2s(NBRNR) = N2
             END IF
          END DO
          !
          !           Enforce remaining conditions, placing results in NBR
          !           Fraction of pairs thrown out depends on many factors
          DO I = 1, NBRNR
             !
             N2 = LSTN2s(I)
             !
             !              Determine exclusion status from exclusion array
             !              then disqualify if
             !              (1) both fixed
             !              (2) one reactant (SUM3==1) and one product (SUM3==2)
             !              (3) if replica. exclude as explained below.
             !               is there an exclusion between atoms N and N2?
             !                 EXCL14 = -1,0,1??
             !
             ! Search through the exclusion lists for this atom pair
             !  (NOTE: This will be only slightly slower than the old ZTBL code,
             !   but it works and doesn't require the memory for ZTBL.
             !   This change was expedient to allow for very large systems,
             !   or for when REPLICA is heavily used.  - BRB  11/7/96
             !
             INBX=IABS(IMB14(NXIM))
             DO WHILE(NXIM.LE.NXIMMAX .AND. N2.GT.INBX)
                NXIM=NXIM+1
                INBX=IABS(IMB14(NXIM))
             ENDDO
             IF(NXIM.GT.NXIMMAX) THEN
                EXCL14=1
             ELSE IF(N2.EQ.IMB14(NXIM)) THEN  ! exclusion found
                EXCL14=0
             ELSE
                IF(N2.EQ.INBX) THEN
                   EXCL14=-1    ! it's a 1-4 pair
                ELSE
                   EXCL14=1
                ENDIF
             ENDIF
             !
             IF (FIXED .AND. IMOVE(N2).gt.0) EXCL14 = 0
             ! ==================================================================
#if KEY_REPLICA==1
             !# <caves>-Aug-4-1993 (Leo Caves)
             ! Replica Exclusions (Atom/Group Exclusion)
             ! Rationale: If groups belong to same subsystem (repID) but are not in
             ! same replica unit (repNoGA) - then EXCLUDE the atom/group pair ELSE
             IF (qRep) THEN
                IF ( iRepID .EQ. repID(repNoGA(N2)) .AND. &
                     iRepNo .NE. repNoGA(N2) )   THEN
                   nRepX = nRepX + 1
                   EXCL14 = 0
                ENDIF
             ENDIF
#endif /*  REPLICA*/
             ! ==================================================================
#if KEY_TSM==1
             IF ((SUM3N+SUM3(N2)).EQ.3) EXCL14 = 0
#endif 
             !
             !              0: Don't add; 1: add positive; -1: add negative
             IF( EXCL14.NE.0 ) THEN
                if(.not.trial)then
                   MBR(MBRX) = SIGN(N2,EXCL14)
                   MBRX = MBRX + 1

                   IF (MBRX .GT. MLIMIT)then
                      call timer_stop(T_bldlst)
                      write(outu,*) &
                           "Self List overflow",mbrxs-1,mlimits
                      RETURN
                   END IF
                else
                   MBRX = MBRX + 1
                   LSTM2M(M)=LSTM2M(M)+1
                endif
             END IF
             !
          END DO
          MBRHI(N-NAT) = MBRX - 1
       END DO
    endif
    !
    !
    !----------------End Self images ----------------------------------------
    !
    !
    !---- End of major loop over cubes M ----------------------------------
    !
666 continue
 END DO
 !
 !---  Figure out a load-balanced distribution of cubes after -------
 !---    collecting the info on # pairs in each cube ----------------
 !
 if(trial)then
    if(numnod.gt.1)then
#if KEY_PARALLEL==1
       call gbor(lstm2,ncube)
       call gbor(lstm2m,ncube)
#endif 
       do m=1,ncube
          NBRTOT=NBRTOT+LSTM2M(M)+LSTM2(M)
       enddo
       nbrshare=nbrtot/numnod + 1
       nbrstart=mynod*nbrshare
       nbrt=0
       !
       !           First sort the cells by number of pairs that each cell
       !            generates.
       !
       !           lstm2(M) contains num of pairs from cell M
       !           lstm2m(M) pointer to cell with next lower num of pairs
       !
       !           Initialize lstm2 and lstm2m
       !
       do m=1,ncube
          nbrt=nbrt+LSTM2M(M)+LSTM2(M)
          LSTM1(M)=LSTM2M(M)+LSTM2(M)
          LSTM1M(M)=M-1
          mycube(m)=0
       enddo
       mmTOP=1
       mmbot=1
       do m=2,ncube
          if(lstm1(m).ge.lstm1(mmtop))then
             lstm1m(m)=mmtop
             mmtop=m
          else
             mm=mmtop
100          if(lstm1m(mm).eq.0)then
                lstm1m(m)=0
                lstm1m(mm)=m
                mmbot=m
             elseif(lstm1(lstm1m(mm)).gt.lstm1(m))then
                mm=lstm1m(mm)
                goto 100
             else
                lstm1m(m)=lstm1m(mm)
                lstm1m(mm)=m
             endif
          endif
       enddo
       mm=mmtop
       !
       !    Put list into LSTM0 in ascending order of num pairs per cell
       !      LSTM0(1) has least pairs, LSTM0(ncube) generates most pairs
       !
       LSTM0(ncube)=mmtop
       do m=ncube-1,1,-1
          LSTM0(M)=LSTM1M(Mm)
          mm=LSTM1M(Mm)
       enddo
       !
       !   Give out cells to nodes so that each node gets about
       !    the same number of pairs. Start by giving the cells
       !    with the most pairs (that have not already been assigned)
       !    till too many are assigned, back off one and fill in
       !    with the cells with the least pairs.
       !      itag=0 for not this node's cell
       !          =1 for this cell assigned to this node
       !
       mlo=1
       mhi=ncube
       itag=0
       npprs=0
       do i=ncube-mynod,1,-numnod*2
          mycube(lstm0(i))=1
       enddo
       do i=ncube-2*numnod+mynod+1,1,-numnod*2
          mycube(lstm0(i))=1
       enddo
       !
       !    for the trial, nbrx and mbrx are incorrect since they were
       !     generated from a bogus set of cells.
       !    Redetermine nbrx, mbrx from lstm2 and lstm2m using the mycube()
       !      distribution.
       !
       nbrx=0
       mbrx=0
       do m=1,ncube
          if(mycube(m).eq.1)then
             nbrx= nbrx+lstm2(m)
             mbrx=mbrx+lstm2m(m)
          endif
       enddo
       !
       !    For non-parallel and one processor parallel, only need to fix
       !        counters and set mycube to be all ones so all cubes will be
       !        serviced in next pass (non-trial).
       !
    else  !------------------------------if(numnode.gt.1)
       do m=1,ncube
          mycube(m)=1
       enddo
       mbrx=mbrx-1
       mbrxs=mbrxs-1
       nbrx=nbrx-1
    endif
 else    !-------------------------------if(trial)
    mbrx=mbrx-1
    mbrxs=mbrxs-1
    !         mbrxs=0
    nbrx=nbrx-1
 endif
 if(trial.and. (prnlev.ge.6)) &
      write(OUTU,'(I8,A,2I8)') mynod, " node has pairs mlo, mhi ", mlo, mhi
 XDISTM = .TRUE.

 call timer_stop(T_bldlst)
 RETURN
END FUNCTION XDISTM

#endif /*  (cubes)*/

SUBROUTINE NULL_NBNDGCM
 RETURN
END SUBROUTINE NULL_NBNDGCM

