SUBROUTINE NBNDGC(NNNB,JNB,MAXJNB,INBLO,X,Y,Z, &
     NNNBG,MXJNBG,JNBG,INBLOG,INB14,IBLO14,ING14,IGLO14, &
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
     ATGXX,ATGYY,ATGZZ,ATGXY,ATGYZ,ATGZX)
#if KEY_NO_BYCU==0
  !
  !-----------------------------------------------------------------------
  !     Made from NBONDG in June 1990 by J. Thomas Ngo.  This routine
  !     should behave identically to NBONDG as viewed from the outside.
  !     It differs in its implementation:  NBNDGC does the search on a
  !     cubical grid.  It completes the problem in linear time, as
  !     opposed to quadratic.  On the Convex C220, which is a vector
  !     machine, it is faster than NBONDG for any system larger than a
  !     few hundred atoms.  It is extremely memory-intensive.
  !
  !     There is a difference:  NBONDG warns about too-close atoms.
  !     This implementation does not.
  !
  !     Important note to developers:  See the comment labeled
  !     "IMPORTANT NOTE" below, regarding the superfluousness of the
  !     sorting step in XDIST.
  !
  !     Line numbers in NBNDGC:
  !       666 Error encountered during atom pairlist generation
  !       999 Done with atom pairlist generation
  !      1666 Error encountered during group pairlist generation
  !      1999 Done with group pairlist generation
  !
  !     Line number in XDIST:
  !      9999 Exclude this particle pair
  !
  !     Old comments:
  !
  !     THIS ROUTINE CONSTRUCTS THE NONBONDED LISTS AND ACCUMULATES
  !     A SET OF ATOM POTENTIALS AND GRADIENTS FOR ELECTROSTATIC
  !     INTERACTIONS OUTSIDE OF THE CUTOFF IF REQUESTED.
  !
  !     ATPOT AND ATF_ HOLD THE POTENTIAL AND ITS FIRST DERIVATIVES
  !     FOR ELECTROSTATIC INTERACTIONS OUTSIDE OF THE CUTOFF (IN UNITS OF
  !     KCAL/MOLE AND ANGSTROMS).
  !
  !     LGROUP IS A FLAG TO SPECIFY THAT A GROUP LIST WILL BE MADE
  !     LEXTND IS A FLAG SPECIFYING EXTENDED ELECTROSTATICS
  !     LQUAD IS A FLAG THAT IS SET TO INCLUDE RESIDUE QUADRUPOLE MOMENTS.
  !     GRADIENTS IS A FLAG THAT IS SET TO CALCULATE THE FIELD GRADIENTS.
  !
  !     22-AUG-1981  By Bernard R. Brooks
  !     EXTENDED ELECTROSTATICS BY DAVID STATES
  !
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use psf
  use stream
  use timerm
  use consta
#if KEY_REPLICA==1
  use replica_mod       
#endif
  use machutil,only:die,timrb,timre
  use memory

  implicit none

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
  real(chm_real) RSQXY(*),RSQYZ(*),RSQZX(*), &
       RSXMAX(*),RSYMAX(*),RSZMAX(*)
  INTEGER RSDISP(*)
  real(chm_real) RSPOT(*),RSFX(*),RSFY(*),RSFZ(*)
  real(chm_real) RSGXX(*),RSGYY(*),RSGZZ(*), &
       RSGXY(*),RSGYZ(*),RSGZX(*)
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
  !
  !     Variables for O(N) search
  !     =========================
  !
  !     RHI                      Radius for distance search
  !     MXMAX,MYMAX,MZMAX        Cube grid dimensions
  !     NBRLo,NBRHi              NBRLO, NBRHI arrays
  !     NBR                      NBR array (XDIST result)
  !     lstM0,lstM1              XDIST work arrays
  !     lstN0,lstN1,lstN2        XDIST work arrays
  !     HPSUM3                   XDIST's SUM3 array
  !     MCL                      Counter for CL arrays
  !     NNNBGI                   Counters like NNNBG (see code)
  !     HPPTR,HPPTR2,HPSIZ,PROBE For heapsort
  !     LCHILD,RCHILD            For heapsort too
  !     X0,Y0,Z0                 Lower left proximal corner of cube grid
  !     NX14,NX14MX              Counters for IGLO14/ING14
  !     NBRcnt                   Counter 1..27 for neighbors
  !     NBRLO, NBRHI             From corresponding XDIST arrays
  !     XDIST                    Function internal to this file
  !     OK                       .TRUE. iff XDIST succeeded
  !     EFFIC                    Ratio of XDIST outputs to actual successes
  !     NEAR                     .FALSE. iff need extended electrostatics
  !     JRSA                     ABS(JRS)
  !
  real(chm_real) RHI
  INTEGER MXMAX,MYMAX,MZMAX
  real(chm_real) X0,Y0,Z0
  INTEGER NX14,NX14MX
  LOGICAL XDIST
  INTEGER NBRcnt
  INTEGER NBRL, NBRH
  LOGICAL OK
  INTEGER EFFIC
  REAL(chm_real) MARGIN
  LOGICAL NEAR
  INTEGER JRSA
  integer,allocatable,dimension(:) :: nbrlo,nbrhi,lstn0,lstn1,lstn2,sum3,nbr, &
       lstm0,lstm1

  

  DATA    EFFIC/6/
  DATA MARGIN/0.01/
  !
  !     End of variables for O(N) search
  !
  !---- Initial housekeeping --------------------------------------------
  !
  IF(PRNLEV > 2) WRITE(OUTU,'(A)') 'Using CUBE search'
  !
  IF (TIMER > 0) CALL TIMRB
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
  CTEXSQ=CTEXNB*CTEXNB
  !
  !     Zero out counters for groups of various charge types
  DO I = 1,4
     IGPBD(I)=0
  END DO
  !
  !
  !---- Initialize extended electrostatics ------------------------------
  !
  !     Store the starting atom positions and initialize atom potentials
  !
  !     The loop on I is vectorizable.
  !
  IF (LEXTND) THEN
     DO I = 1,NATOM
        ATSX(I)=X(I)
        ATSY(I)=Y(I)
        ATSZ(I)=Z(I)
        ATPOT(I)=0.0
        ATFX(I)=0.0
        ATFY(I)=0.0
        ATFZ(I)=0.0
        IF (LGRAD) THEN
           ATGXX(I)=0.0
           ATGYY(I)=0.0
           ATGZZ(I)=0.0
           ATGXY(I)=0.0
           ATGYZ(I)=0.0
           ATGZX(I)=0.0
        END IF
     END DO
  END IF
  !
  !---- Initialize extended electrostatics ------------------------------
  !
  !     FIND GEOMETRIC CENTER FOR EACH RESIDUE AND THE MULTIPOLE MOMENTS
  !     FOR THE CHARGE DISTRIBUTIONS
  !
  !.... Loop over groups I begins here ..................................
  !
  !     The outer loop on I is not vectorizable, but the two inner loops
  !     on J are.  The first inner loop on J would not have been, at
  !     least on the Convex, had statements like XMIN = MIN(XMIN,X(J))
  !     been left as IF (X(J) < XMIN) XMIN=X(J).
  !
  DO I = 1,NGRP
     !
     !....... Increment counter for group charge type ......................
     !
     J=IGPTYP(I)+1
     IGPBD(J)=IGPBD(J)+1
     !
     !....... Find geometric center and bounding box .......................
     !
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF (NAT <= 0) CALL DIE
     XMIN=X(IS)
     XMAX=XMIN
     YMIN=Y(IS)
     YMAX=YMIN
     ZMIN=Z(IS)
     ZMAX=ZMIN
     DXT=0.0
     DYT=0.0
     DZT=0.0
     DO J = IS,IQ
        DXT=DXT+X(J)
        DYT=DYT+Y(J)
        DZT=DZT+Z(J)
        XMIN = MIN(XMIN,X(J))
        YMIN = MIN(YMIN,Y(J))
        ZMIN = MIN(ZMIN,Z(J))
        XMAX = MAX(XMAX,X(J))
        YMAX = MAX(YMAX,Y(J))
        ZMAX = MAX(ZMAX,Z(J))
     END DO
     XD=DXT/NAT
     YD=DYT/NAT
     ZD=DZT/NAT
     RSCMX(I)=XD
     RSCMY(I)=YD
     RSCMZ(I)=ZD
     !
     !....... Find smallest bounding box centered on geometric center ......
     !
     RSXMAX(I)=MAX(XMAX-XD,XD-XMIN)
     RSYMAX(I)=MAX(YMAX-YD,YD-YMIN)
     RSZMAX(I)=MAX(ZMAX-ZD,ZD-ZMIN)
     !
     !        This comment was here in file, but I don't know what it means.
     !        Saving it for posterity.
     !        SIZE OF LARGEST VDW RADIUS OF ATOMS IN GROUP TIMES CUTOFF FACTOR
     !
     !....... Compute multipole moments for this group .....................
     !
     IF (LEXTND) THEN
        RSPOT(I)=0.0
        RSFX(I)=0.0
        RSFY(I)=0.0
        RSFZ(I)=0.0
        RSGXX(I)=0.0
        RSGYY(I)=0.0
        RSGZZ(I)=0.0
        RSGXY(I)=0.0
        RSGYZ(I)=0.0
        RSGZX(I)=0.0
        QSUM=0.0
        DXT=0.0
        DYT=0.0
        DZT=0.0
        QXXT=0.0
        QYYT=0.0
        QZZT=0.0
        QXYT=0.0
        QZXT=0.0
        QYZT=0.0
        DO J = IS,IQ
           CGT=CCELEC*CG(J)/EPS
           QSUM=QSUM+CGT
           DXT=DXT+CGT*X(J)
           DYT=DYT+CGT*Y(J)
           DZT=DZT+CGT*Z(J)
           QXXT=QXXT+CGT*X(J)*X(J)
           QYYT=QYYT+CGT*Y(J)*Y(J)
           QZZT=QZZT+CGT*Z(J)*Z(J)
           QXYT=QXYT+CGT*X(J)*Y(J)
           QZXT=QZXT+CGT*X(J)*Z(J)
           QYZT=QYZT+CGT*Y(J)*Z(J)
        END DO
        RSQ(I)=QSUM
        RSDX(I)=DXT-QSUM*XD
        RSDY(I)=DYT-QSUM*YD
        RSDZ(I)=DZT-QSUM*ZD
        IF (LQUAD) THEN
           QXXT=QXXT+QSUM*XD*XD-2.0*DXT*XD
           QYYT=QYYT+QSUM*YD*YD-2.0*DYT*YD
           QZZT=QZZT+QSUM*ZD*ZD-2.0*DZT*ZD
           RSQXX(I)=2.0*QXXT-QYYT-QZZT
           RSQYY(I)=2.0*QYYT-QXXT-QZZT
           RSQZZ(I)=2.0*QZZT-QXXT-QYYT
           RSQXY(I)=3.0*(QXYT+QSUM*XD*YD-DXT*YD-DYT*XD)
           RSQYZ(I)=3.0*(QYZT+QSUM*YD*ZD-DYT*ZD-DZT*YD)
           RSQZX(I)=3.0*(QZXT+QSUM*XD*ZD-DXT*ZD-DZT*XD)
        END IF
     END IF
  END DO
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
  !
  !---- Ascertain that no ST2 waters are present ------------------------
  !
  IF (IGPBD(4) /= 0) THEN
     IF (WRNLEV >= 2) THEN
        WRITE (OUTU,*) 'The O(N) version of NBONDG does not give'
        WRITE (OUTU,*) 'the same answers as the traditional version if'
        WRITE (OUTU,*) 'there are any ST2 waters.  Goodbye!'
     END IF
     CALL DIE
  END IF
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
  DO I = 2, NATOM
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
  !
  !     I don't know why I added 2 to the number of cubes along each
  !     dimension.  Perhaps I was just being safe?  Anyway, on a scalar
  !     machine the extra cubes do not produce a performance hit.  (This
  !     code obsolete.)
  !
  !$$$      RHI = CUTNB
  !$$$      X0 = XMIN - RHI
  !$$$      Y0 = YMIN - RHI
  !$$$      Z0 = ZMIN - RHI
  !$$$      MXMAX = (XMAX-X0)/RHI + 2
  !$$$      MYMAX = (YMAX-Y0)/RHI + 2
  !$$$      MZMAX = (ZMAX-Z0)/RHI + 2
  !
  !     In this new trial code, I make the cube system large enough to
  !     hold all of the atoms, but with a margin at least (RHI * MARGIN).
  !     Then I center the atoms in the cube system.
  !
  RHI = CUTNB
  MXMAX = INT((XD/RHI) + MARGIN + 1)
  MYMAX = INT((YD/RHI) + MARGIN + 1)
  MZMAX = INT((ZD/RHI) + MARGIN + 1)
  X0 = XMIN - 0.5 * (MXMAX * RHI - XD)
  Y0 = YMIN - 0.5 * (MYMAX * RHI - YD)
  Z0 = ZMIN - 0.5 * (MZMAX * RHI - ZD)
  !
  !---- Allocate work areas for XDIST -----------------------------------
  !
  call chmalloc('nbndgc.src','NBNDGC','NBR',MAXJNB*EFFIC,intg=NBR)
  call chmalloc('nbndgc.src','NBNDGC','NBRLo',NATOM,intg=NBRLo)
  call chmalloc('nbndgc.src','NBNDGC','NBRHi',NATOM,intg=NBRHi)
  call chmalloc('nbndgc.src','NBNDGC','lstN0  ',NATOM,intg=lstN0  )
  call chmalloc('nbndgc.src','NBNDGC','lstN1  ',NATOM,intg=lstN1  )
  call chmalloc('nbndgc.src','NBNDGC','lstN2  ',NATOM,intg=lstN2  )
  call chmalloc('nbndgc.src','NBNDGC','SUM3',NATOM,intg=SUM3)
  call chmalloc('nbndgc.src','NBNDGC','lstM0',MXMAX*MYMAX*MZMAX,intg=lstM0)
  call chmalloc('nbndgc.src','NBNDGC','lstM1',MXMAX*MYMAX*MZMAX,intg=lstM1)
  !
  !---- Set up SUM3 array for XDIST --------------------------------------
  !
  !     SUM3(I) is
  !       1 if REACLS(I) == 1
  !       2 if PRODLS(I) == 1
  !       0 otherwise
  !     Assumption:  an atom is never a product and a reactant
  !     simultaneously.
  !
  !     Both loops on I are vectorizable.
  !
  DO I = 1, NATOM
     SUM3(I) = 0
  END DO
#if KEY_TSM==1
  IF (LTSM) THEN
     DO I = 1, NATOM
        IF (REACLS(I) == 1) SUM3(I) = 1
        IF (PRODLS(I) == 1) SUM3(I) = 2
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
  OK = XDIST(NATOM,X,Y,Z,IMOVE,RHI, &
       X0,Y0,Z0, MXMAX,MYMAX,MZMAX, &
       NBRLo,NBRHi,NBR,MAXJNB*EFFIC, &
       lstM0,lstM1, &
       lstN0,lstN1,lstN2, &
       SUM3, &
       repNoA,repID,nRepXA,qRep,INB14,IBLO14, &
       .FALSE.)
  IF (OK) then
     
     !---- Convert the result to a CHARMM pairlist  -------------------------
     !
     !     The NBR array effectively contains a pairlist, but the runs are
     !     out of order.  The outer loop on I is not vectorizable, but the
     !     inner loop on NBR is.
     !
     NNNB  = 0
     DO I = 1,NATOM
        NBRL = NBRLo(I)
        NBRH = NBRHi(I)
        IF (NNNB + NBRH-NBRL+1  >=  MAXJNB) then
           ok=.false.
           exit
        endif
        DO NBRcnt = NBRL, NBRH
           NNNB = NNNB + 1
           JNB(NNNB) = NBR(NBRcnt)
        END DO
        INBLO(I) = NNNB
     END DO
  endif
  !
  !---- Clean up after building of atom list -----------------------------
  !
  !     Free allocations, and return with bad NNNB if error occurred
  !
  !
  !     Previous code might have jumped here on error, with .NOT.OK

  call chmdealloc('nbndgc.src','NBNDGC','lstM0',MXMAX*MYMAX*MZMAX,intg=lstM0)
  call chmdealloc('nbndgc.src','NBNDGC','lstM1',MXMAX*MYMAX*MZMAX,intg=lstM1)
  call chmdealloc('nbndgc.src','NBNDGC','NBR',MAXJNB*EFFIC,intg=NBR)
  call chmdealloc('nbndgc.src','NBNDGC','NBRLo',NATOM,intg=NBRLo)
  call chmdealloc('nbndgc.src','NBNDGC','NBRHi',NATOM,intg=NBRHi)
  call chmdealloc('nbndgc.src','NBNDGC','lstN0  ',NATOM,intg=lstN0  )
  call chmdealloc('nbndgc.src','NBNDGC','lstN1  ',NATOM,intg=lstN1  )
  call chmdealloc('nbndgc.src','NBNDGC','lstN2  ',NATOM,intg=lstN2  )
  call chmdealloc('nbndgc.src','NBNDGC','SUM3',NATOM,intg=SUM3)

  IF (.NOT.OK) THEN
     NNNB = MAXJNB + 1
     RETURN
  END IF
  !
  !     Line 999 might have been "gone to" if .NOT.LVATOM
  !
999 CONTINUE
  !
  !==== Build group list if appropriate ==================================
  !
  !     If an error is encountered in this phase, GOTO 1666.
  !
  IF (.NOT.LGROUP) GOTO 1999
  !
  !---- Establish cube parameters ---------------------------------------
  !
  !     I don't know why I added 2 to the number of cubes along each
  !     dimension.  Perhaps I was just being safe?  Anyway, on a serial
  !     machine the excess cubes do not produce a performance hit.
  !     (This code obsolete.)
  !
  !$$$      IF (LEXTND) THEN
  !$$$         RHI = CTEXNB
  !$$$      ELSE
  !$$$         RHI = CUTNB
  !$$$      END IF
  !$$$      X0 = XMIN - RHI
  !$$$      Y0 = YMIN - RHI
  !$$$      Z0 = ZMIN - RHI
  !$$$      MXMAX = (XMAX-X0)/RHI + 2
  !$$$      MYMAX = (YMAX-Y0)/RHI + 2
  !$$$      MZMAX = (ZMAX-Z0)/RHI + 2
  !
  !     In this new trial code, I make the cube system large enough to
  !     hold all of the atoms, but with a margin at least (RHI * MARGIN).
  !     Then I center the atoms in the cube system.
  !
  IF (LEXTND) THEN
     RHI = CTEXNB
  ELSE
     RHI = CUTNB
  END IF
  MXMAX = INT((XD/RHI) + MARGIN + 1)
  MYMAX = INT((YD/RHI) + MARGIN + 1)
  MZMAX = INT((ZD/RHI) + MARGIN + 1)
  X0 = XMIN - 0.5 * (MXMAX * RHI - XD)
  Y0 = YMIN - 0.5 * (MYMAX * RHI - YD)
  Z0 = ZMIN - 0.5 * (MZMAX * RHI - ZD)
  !
  !---- Allocate work areas for XDIST -----------------------------------
  !
  call chmalloc('nbndgc.src','NBNDGC','NBR',MXJNBG*EFFIC,intg=NBR)
  call chmalloc('nbndgc.src','NBNDGC','NBRLo',ngrp,intg=NBRLo)
  call chmalloc('nbndgc.src','NBNDGC','NBRHi',ngrp,intg=NBRHi)
  call chmalloc('nbndgc.src','NBNDGC','lstN0  ',ngrp,intg=lstN0  )
  call chmalloc('nbndgc.src','NBNDGC','lstN1  ',ngrp,intg=lstN1  )
  call chmalloc('nbndgc.src','NBNDGC','lstN2  ',ngrp,intg=lstN2  )
  call chmalloc('nbndgc.src','NBNDGC','SUM3',ngrp,intg=SUM3)
  call chmalloc('nbndgc.src','NBNDGC','lstM0',MXMAX*MYMAX*MZMAX,intg=lstM0)
  call chmalloc('nbndgc.src','NBNDGC','lstM1',MXMAX*MYMAX*MZMAX,intg=lstM1)
  !
  !---- Set up SUM3 array for XDIST --------------------------------------
  !
  !     Since product-reactant distinctions are not used with the group
  !     list, we just set all elements of SUM3 to zero.  See comments to
  !     corresponding code in atom-list section for details.
  !
  !     This loop is vectorizable.
  !
  DO IRS = 1, NGRP
     SUM3(IRS) = 0
  END DO
  !
  !---- Call XDIST -------------------------------------------------------
  !
  !     XDIST solves the basic distance-search problem, without
  !     modification.  Here we DO have it return self-self interactions.
  !
  !
  OK = XDIST(NGRP,RSCMX,RSCMY,RSCMZ,IMOVEG,RHI, &
       X0,Y0,Z0, MXMAX,MYMAX,MZMAX, &
       NBRLo,NBRHi,NBR,MXJNBG*EFFIC, &
       lstM0,lstM1, &
       lstN0,lstN1,lstN2, &
       SUM3, &
       repNoA,repID,nRepXG,qRep,INB14,IBLO14, &
       .TRUE. )
  IF (OK) then

     !---- Convert the result to a CHARMM pairlist  -------------------------
     !
     !     As we do it, we must filter out excluded interactions.  Whenever
     !     two groups are further away than CTNBSQ we must also compute
     !     multipole interaction coefficients.  According to the existing
     !     comment:  "If the residues are far enough apart, do only a
     !     single evaluation and use second order polynomial interpolation."
     !
     !     NGST2  Number of qualifying ST2-ST2 pairs
     !     NGPE   Number of pairs done by extended elec
     !     NGPX   Number of pairs beyond cutoffs
     !
     !     This is a rather tall piece of code because of the extended
     !     electrostatics.  None of the loops (IRS, NBR, J) are
     !     vectorizable.
     !
     NGST2 = 0
     NGPE = 0
     NGPX = 0
     !
     !     NNNBG    Last JNBG element filled
     !
     NNNBG = 0
     !
     DO IRS = 1, NGRP
        !
        NBRL = NBRLo(IRS)
        NBRH = NBRHi(IRS)
        !
        DO NBRcnt = NBRL, NBRH
           JRS = NBR(NBRcnt)
           JRSA = ABS(JRS)
           XD = RSCMX(IRS) - RSCMX(JRSA)
           YD = RSCMY(IRS) - RSCMY(JRSA)
           ZD = RSCMZ(IRS) - RSCMZ(JRSA)
           XX=XD*XD
           YY=YD*YD
           ZZ=ZD*ZD
           R2=XX+YY+ZZ
           !
           IF (R2 < CTNBSQ) THEN
              !
              !.......... Addition to group list starts here ........................
              !
              NNNBG = NNNBG + 1
              IF (NNNBG  >  MXJNBG) then
                 ok = .false.
                 exit
              endif
              JNBG(NNNBG) = JRS
              !
              !.......... Addition to group list ends here ..........................
              !
           ELSE IF (LEXTND .AND. R2 >= CTEXSQ) THEN ! (R2 < CTNBSQ)
              !
              !.......... Extended electrostatics starts here .......................
              !
              !           If LEXTND and the two groups are further apart than
              !           CUTNB but closer than CTEXSQ, then do extended
              !           electrostatics INSTEAD of adding to the group pairlist.
              !
              JS=IGPBS(JRSA)+1
              JQ=IGPBS(JRSA+1)
              XY=XD*YD
              YZ=YD*ZD
              ZX=ZD*XD
              R=SQRT(R2)
              R1=1.0/R
              R3=R1/R2
              R5=R3/R2
              R2X3=3.0/R2
              R2X5=5.0/R2
              R2X7=7.0/R2
              !
              DO J = 1,2
                 !
                 IF (J == 1) THEN
                    IRST=IRS
                    JRST=JRSA
                 ELSE
                    IRST=JRSA
                    JRST=IRS
                    XD=-XD
                    YD=-YD
                    ZD=-ZD
                 END IF
                 DXT=RSDX(JRST)*R3
                 DYT=RSDY(JRST)*R3
                 DZT=RSDZ(JRST)*R3
                 DOT=(DXT*XD+DYT*YD+DZT*ZD)
                 !
                 IF (LQUAD) THEN
                    !
                    QXXT=RSQXX(JRST)*R5
                    QYYT=RSQYY(JRST)*R5
                    QZZT=RSQZZ(JRST)*R5
                    QXYT=RSQXY(JRST)*R5
                    QYZT=RSQYZ(JRST)*R5
                    QZXT=RSQZX(JRST)*R5
                    QXT=QXXT*XD+QXYT*YD+QZXT*ZD
                    QYT=QYYT*YD+QYZT*ZD+QXYT*XD  ! B990105.rjp
                    QZT=QZZT*ZD+QZXT*XD+QYZT*YD
                    RQR=(QXT*XD+QYT*YD+QZT*ZD)/2.0
                    !
                    CGT=RSQ(JRST)*R1
                    !
                    RSPOT(IRST)=RSPOT(IRST)+CGT+DOT+RQR
                    !
                    CR3=CGT/R2
                    DOT=DOT*R2X3
                    RQR=RQR*R2X5
                    TEMP=CR3+DOT+RQR
                    !
                    RSFX(IRST)=RSFX(IRST)+DXT+QXT-TEMP*XD
                    RSFY(IRST)=RSFY(IRST)+DYT+QYT-TEMP*YD
                    RSFZ(IRST)=RSFZ(IRST)+DZT+QZT-TEMP*ZD
                    !
                    TEMP2=CR3*R2X3+DOT*R2X5+RQR*R2X7
                    DXT=DXT*R2X3+QXT*R2X5
                    DYT=DYT*R2X3+QYT*R2X5
                    DZT=DZT*R2X3+QZT*R2X5
                    !
                    RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD-TEMP+QXXT
                    RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD-TEMP+QYYT
                    RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD-TEMP+QZZT
                    RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD+QXYT
                    RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD+QYZT
                    RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD+QZXT
                    !
                 ELSE             ! (LQUAD)
                    !
                    CGT=RSQ(JRST)*R1
                    !
                    RSPOT(IRST)=RSPOT(IRST)+CGT+DOT
                    !
                    CR3=CGT/R2
                    DOT=DOT*R2X3
                    TEMP=CR3+DOT
                    !
                    RSFX(IRST)=RSFX(IRST)+DXT-TEMP*XD
                    RSFY(IRST)=RSFY(IRST)+DYT-TEMP*YD
                    RSFZ(IRST)=RSFZ(IRST)+DZT-TEMP*ZD
                    !
                    TEMP2=CR3*R2X3+DOT*R2X5
                    DXT=DXT*R2X3
                    DYT=DYT*R2X3
                    DZT=DZT*R2X3
                    !
                    RSGXX(IRST)=RSGXX(IRST)+TEMP2*XX-2.0*DXT*XD
                    RSGYY(IRST)=RSGYY(IRST)+TEMP2*YY-2.0*DYT*YD
                    RSGZZ(IRST)=RSGZZ(IRST)+TEMP2*ZZ-2.0*DZT*ZD
                    RSGXY(IRST)=RSGXY(IRST)+TEMP2*XY-DXT*YD-DYT*XD
                    RSGYZ(IRST)=RSGYZ(IRST)+TEMP2*YZ-DYT*ZD-DZT*YD
                    RSGZX(IRST)=RSGZX(IRST)+TEMP2*ZX-DZT*XD-DXT*ZD
                    !
                 END IF           ! (LQUAD)
                 !
              END DO              ! J = 1, 2
              !
              NGPE=NGPE+1
              !
              !.......... Extended electrostatics ends here .........................
              !
           END IF                 ! (R2)
           !
        END DO                    ! NBRcnt = NBRL, NBRH
        !
        !     Record the last index in this run of pairs
        INBLOG(IRS) = NNNBG
        !
     END DO                    ! IRS = 1, NGRP
     !
  endif
  !---- Clean up after building of group list ---------------------------
  !
  !     Free allocations, and return with bad NNNBG if error occurred
  !
  call chmdealloc('nbndgc.src','NBNDGC','lstM0',MXMAX*MYMAX*MZMAX,intg=lstM0)
  call chmdealloc('nbndgc.src','NBNDGC','lstM1',MXMAX*MYMAX*MZMAX,intg=lstM1)
  call chmdealloc('nbndgc.src','NBNDGC','NBRLo',ngrp,intg=NBRLo)
  call chmdealloc('nbndgc.src','NBNDGC','NBRHi',ngrp,intg=NBRHi)
  call chmdealloc('nbndgc.src','NBNDGC','lstN0  ',ngrp,intg=lstN0  )
  call chmdealloc('nbndgc.src','NBNDGC','lstN1  ',ngrp,intg=lstN1  )
  call chmdealloc('nbndgc.src','NBNDGC','lstN2  ',ngrp,intg=lstN2  )
  call chmdealloc('nbndgc.src','NBNDGC','SUM3',ngrp,intg=SUM3)
  call chmalloc('nbndgc.src','NBNDGC','NBR',MXJNBG*EFFIC,intg=NBR)
  !
  IF (.NOT.OK) THEN
     NNNBG = MXJNBG + 1
     RETURN
  END IF
  !
  !     Line 1999 might have been "gone to" if (.NOT.LGROUP)
  !
1999 CONTINUE
  !
  !==== Done with code specific to NBNDGC ================================
  !
  !---- Finish extended electrostatics ----------------------------------
  !
  !     The outer loop on IRST is not vectorizable, but the inner loop
  !     on I is.
  !
  IF (LEXTND) THEN
     !
     DO IRST = 1,NGRP
        IS=IGPBS(IRST)+1
        IQ=IGPBS(IRST+1)
        DO I = IS,IQ
           XI=X(I)-RSCMX(IRST)
           YI=Y(I)-RSCMY(IRST)
           ZI=Z(I)-RSCMZ(IRST)
           QXT=RSGXX(IRST)*XI+RSGXY(IRST)*YI+RSGZX(IRST)*ZI
           QYT=RSGYY(IRST)*YI+RSGXY(IRST)*XI+RSGYZ(IRST)*ZI
           QZT=RSGZZ(IRST)*ZI+RSGZX(IRST)*XI+RSGYZ(IRST)*YI
           ATPOT(I)=ATPOT(I)+RSPOT(IRST)+((2.0*RSFX(IRST)+QXT)*XI+ &
                (2.0*RSFY(IRST)+QYT)*YI+(2.0*RSFZ(IRST)+QZT)*ZI)/2.0
           ATFX(I)=ATFX(I)+RSFX(IRST)+QXT
           ATFY(I)=ATFY(I)+RSFY(IRST)+QYT
           ATFZ(I)=ATFZ(I)+RSFZ(IRST)+QZT
           IF (LGRAD) THEN
              ATGXX(I)=ATGXX(I)+RSGXX(IRST)
              ATGYY(I)=ATGYY(I)+RSGYY(IRST)
              ATGZZ(I)=ATGZZ(I)+RSGZZ(IRST)
              ATGXY(I)=ATGXY(I)+RSGXY(IRST)
              ATGYZ(I)=ATGYZ(I)+RSGYZ(IRST)
              ATGZX(I)=ATGZX(I)+RSGZX(IRST)
           END IF           ! (LGRAD)
        END DO              ! I = IS,IQ
     END DO                 ! IRST = 1, NGRP
     !
     !     TERMINATION OF THE ROUTINE  (comment added later:  not really!)
     !
     !     This loop is vectorizable.
     !
     DO I = 1,NATOM
        CGT=CG(I)
        ATPOT(I)=ATPOT(I)*CGT/2.0
        ATFX(I)=ATFX(I)*CGT
        ATFY(I)=ATFY(I)*CGT
        ATFZ(I)=ATFZ(I)*CGT
        IF (LGRAD) THEN
           ATGXX(I)=ATGXX(I)*CGT
           ATGYY(I)=ATGYY(I)*CGT
           ATGZZ(I)=ATGZZ(I)*CGT
           ATGXY(I)=ATGXY(I)*CGT
           ATGYZ(I)=ATGYZ(I)*CGT
           ATGZX(I)=ATGZX(I)*CGT
        END IF              ! (LGRAD)
     END DO                 ! (I = 1, NATOM)
  END IF                    ! (LEXTND)
  !
  !---- Flag success and print statistics --- ---------------------------
  !
  CMPLTD=.TRUE.
  !
  IF (PRNLEV >= 5) THEN
     !
     WRITE(OUTU,*) ' NBNDGC found:'
     !
     IF (LVATOM) WRITE(OUTU,*) NNNB, ' atom pairs'
#if KEY_REPLICA==1
     IF (qRep) WRITE(OUTU,'(I9,A/)') &
          nRepXA,' REPLICA ATOM  PAIRS EXCLUDED'
#endif /*  REPLICA*/
     !
     IF (LGROUP .OR. NST2 > 0) THEN
        WRITE(OUTU,*) NNNBG, ' group pairs'
        IF (NST2 > 0) WRITE(OUTU,*) &
             NGST2, ' of which were ST2-ST2 interactions'
        IF (LEXTND) WRITE(OUTU,*) &
             NGPE, ' pairs used extended electrostatics'
        WRITE(OUTU,*) &
             NGPX, ' group pairs were beyond cutoffs'
        WRITE(OUTU,*) &
             NGRP, ' groups:', &
             IGPBD(1), '--no charges ', &
             IGPBD(2), '--neutral ', &
             IGPBD(3), '--charged ', &
             IGPBD(4), '--ST2 '
#if KEY_REPLICA==1
        IF (qRep) WRITE(OUTU,'(I9,A/)') &
             nRepXG,' REPLICA GROUP PAIRS EXCLUDED'
#endif /*  REPLICA*/
     END IF                 ! (LGROUP .OR. NST2 > 0)
     !
  END IF                    ! (PRNLEV >= 5)
  !
  IF (TIMER >= 1) THEN
     IF (PRNLEV >= 2) WRITE(OUTU,*) 'TOTAL TIME IN NBNDGC: '
     CALL TIMRE
     CALL TIMRB
  END IF                    ! (TIMER == 1)
  RETURN
END SUBROUTINE NBNDGC

!=======================================================================
! Here is XDIST, derived from ZDIST 1.20 and adapted to interface
! optimally with CHARMM.  Made vectorizable too.
!=======================================================================
!
!                 XDIST by J. Thomas Ngo, 2/19/1990
!                 Generic distance-searching module
!                         Overhauled 8/92
!
! DESCRIPTION
!
! XDIST accomplishes linear time (O(N)) construction of a list of
! particles lying within a distance range 0 <= dist <= RHI, given a
! list of Cartesian particle coordinates.  This version has been
! adapted to integrate optimally with the CHARMM subroutine NBONDG,
! and has been modified so that it will vectorize with Convex FORTRAN
! 5.1.1.  It should compile and run on any machine.
!
! Here are the modifications that have been made for optimal
! integration with CHARMM:
!   * Self-self interactions can be included or excluded
!
! ALGORITHM
!
! Space is divided into cubes of side RHI, where RHI is the desired
! upper bound on interparticle distance.  If the particle density
! is bounded, then the number of particles per cube is bounded.
! This underlies the O(N) behavior of the search.
!
! First, an O(number-of-atoms) operation:
! Decide which cube each atom is in.
!
! Next, an O(number-of-cubes) operation:
! For each cube,
!    Find all qualifying interactions within the cube
!    Find all qualifying interactions between particles in the cube
!        and particles in immediately neighboring cubes
!
! The actual implementation contains several conversions of internal
! format:  to linked lists, then to contiguous arrays, etc.  This
! permits vectorization.  See the code for definitive comments.
!
! WISH LIST
!
!     In the future, might need to filter undefined coordinates.
!
!======================================================================
! Main subroutine
!======================================================================
!
LOGICAL FUNCTION XDIST(NAT,X,Y,Z,IMOVEGA,RHI, &
     X0,Y0,Z0,MXMAX,MYMAX,MZMAX, &
     NBRLO,NBRHI,NBR,LIMIT, &
     LSTM0,LSTM1, &
     LSTN0,LSTN1,LSTN2, &
     SUM3, &
     repNoGA,repID,nRepX,qRep,INB14,IBLO14, &
     QSELF )
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  use stream
  implicit none
  !
  !     Parameters
  !
  !     NAT                          Number of particles
  !     X(NAT),Y(NAT),Z(NAT)         Coordinates
  !     IMOVEGA(NAT)                 IMOVE or IMOVEG, depending
  !     RHI                          Maximum desired distance
  !     X0,Y0,Z0                     Corner of cube [1,1,1]
  !     LIMIT                        Max # of pairs expected
  !     MXMAX,MYMAX,MZMAX            Number of cubes in each direction
  !
  !     NBRLO(NAT)                   Neighbors of atom N are listed in
  !     NBRHI(NAT)                   NBR(NBRLO(N)..NBRHI(N))
  !     NBR(LIMIT)
  !
  !     LSTM0(MMAX),LSTM1(MMAX)      Work arrays--see explanation
  !     LSTN0(NAT),LSTN1(NAT),LSTN2(NAT) More work arrays
  !
  !     SUM3(NAT)                    For each atom, a number in 0..2.
  !
  !     QSELF                         .TRUE. iff want self-self int'ns
  !
  INTEGER NAT
  real(chm_real) X(NAT),Y(NAT),Z(NAT)
  INTEGER IMOVEGA(NAT)
  real(chm_real) RHI
  real(chm_real) X0,Y0,Z0
  INTEGER LIMIT
  INTEGER MXMAX,MYMAX,MZMAX
  INTEGER NBRLO(NAT)
  INTEGER NBRHI(NAT)
  INTEGER NBR(LIMIT)
  INTEGER LSTM0(MXMAX*MYMAX*MZMAX),LSTM1(MXMAX*MYMAX*MZMAX)
  INTEGER LSTN0(NAT),LSTN1(NAT),LSTN2(NAT)
  INTEGER INB14(*),IBLO14(*)
  INTEGER SUM3(NAT)
  LOGICAL QSELF
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
  !     IMOVEGA     array specifying whether given particle moves
  !     0..RHI2     range of interparticle distances of interest.
  !
  !     Explanation of parameters interesting only to implementors:
  !
  !     The parameters X0..Z0, M[XYZ]MAX define the cube mapping thus:
  !     3D space is divided into cubes of side RHI.  These cubes are
  !     named by three integer indices: INT( (X-X0)/RHI ), and so forth.
  !     The indices must lie in the range 1..MXMAX, and so forth.
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
  !     MMAX               Number of cubes calc'ed from MXMAX etc
  !     MXYMAX             MXMAX * MYMAX
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
  !     FIXED              IMOVEGA(N) > 0
  !     SUM3N              SUM3(N)
  !     TMP                Miscellaneous integer
  !     EXCL14             (+1,-1,0) if (exclude, 1-4, neither)
  !     DONE               Controls while loops
  !
  real(chm_real) RHI2
  REAL(chm_real) DDX,DDY,DDZ
  INTEGER M,M2,MX,MY,MZ
  INTEGER MMAX
  INTEGER MXYMAX
  INTEGER MADD(27)
  INTEGER N,N2
  INTEGER NLO,NHI,NLO2,NHI2
  INTEGER I,ILO
  INTEGER NIX,NIX2
  INTEGER NBR27
  INTEGER NBRN,NBRNR
  !
  INTEGER NBRX
  REAL(chm_real) XX,YY,ZZ
  LOGICAL FIXED
  INTEGER SUM3N
  INTEGER TMP
  INTEGER EXCL14
  LOGICAL DONE
  INTEGER INBX, NXIMAX, NXI
  !
  !----------------------------------------------------------------------
  !     Executable code begins here
  !----------------------------------------------------------------------
  !
  !---- Compute frequently used intermediate scalar results -------------
  !
  RHI2 = RHI * RHI
  MXYMAX = MXMAX * MYMAX
  MMAX = MXYMAX * MZMAX
  !
  IF (PRNLEV >= 2) THEN
     WRITE (OUTU,*) &
          'NBONDG Building particle interaction list using grid'
     WRITE (OUTU,*) 'Number of particles            =', NAT
     WRITE (OUTU,*) 'Number of cells in X dimension =', MXMAX
     WRITE (OUTU,*) 'Number of cells in Y dimension =', MYMAX
     WRITE (OUTU,*) 'Number of cells in Z dimension =', MZMAX
     WRITE (OUTU,*) 'Number of cells, total         =', MMAX
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
           MADD(NBR27) = MX + MXMAX * (MY + MYMAX * MZ)
        END DO
     END DO
  END DO
  !
  !     Remove duplicate offsets from MADD, or we will get duplicate
  !     atoms and therefore array-bound problems down the line.  Any
  !     element that is set to MMAX is effectively deleted since later
  !     M+MMAX is out of the range 1..MMAX.
  !
  !     Crude method OK since max number of operations is 13 * 27 = 351
  !
  !     Not vectorizable.
  !
  DO NBR27 = 1, 27
     DO TMP = 1, NBR27-1
        IF (MADD(TMP) == MADD(NBR27)) MADD(NBR27) = MMAX + 1
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
     LSTN1(N) = ( INT((Z(N)-Z0)/RHI)  * MYMAX &
          + INT((Y(N)-Y0)/RHI)) * MXMAX &
          + INT((X(N)-X0)/RHI) + 1
  END DO
  !
  !     Invert LSTN1:  Set up LSTM0/LSTN0 as intertwined linked lists
  !     such that for a given cube M, you can recursively read off the
  !     atoms that it contains.  This will be discarded after the next
  !     step.
  !
  !     Vectorizable.
  DO M = 1, MMAX
     LSTM0(M) = 0
  END DO
  !
  !     This loop cannot be vectorized because it contains a recurrence.
  DO N = 1, NAT
     M = LSTN1(N)
     LSTN0(N) = LSTM0(M)
     LSTM0(M) = N
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
  DO M = 1, MMAX
     N = LSTM0(M)
     DO WHILE (N /= 0)
        I = I + 1
        LSTN1(I) = N
        N = LSTN0(N)
     END DO
     LSTM1(M) = I
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
  NHI = 0
  !
  !---- Start of major loop over cubes M --------------------------------
  !
  !     This outer loop cannot be vectorized.
  !
  DO M = 1, MMAX
     !
     !------- Determine range of atoms LSTN1(NLO..NHI) in center cube M ----
     !
     NLO = NHI + 1
     NHI = LSTM1(M)
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
        IF (M2 >= 1 .AND. M2 <= MMAX) THEN ! ----------
           !
           !              Set NLO2..NHI2 equal to range of indices into LSTN1
           IF (M2 == 1) THEN
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
     !        IMPORTANT NOTE:  In earlier versions of NBNDGC.SRC, the
     !        purpose of this sort was to prepare for navigation around the
     !        CHARMM pairlist that represents the 1-2, 1-3, and 1-4 atom or
     !        group pairs.  Now that this code uses the compressed-table
     !        (ZTBL) representation instead, this sort is nearly
     !        superfluous.  Unfortunately, the rest of CHARMM---in
     !        particular, the energy code---still expects to see sorted
     !        INBLO/JNB and INBLOG/JNBG arrays.  When that situation is
     !        rectified, it should be possible to remove this code,
     !        doubling the speed of nonbonded list generation without
     !        affecting the time required to compute energies.
     !
     !        It makes sense to sort now rather than after the distance
     !        check because this way is tantamount to sorting the union of
     !        the lists of the neighbors of particle N, where N is a
     !        particle in cube M.
     !
     call qsort_ints(LSTN0, NBRN)

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
        FIXED = IMOVEGA(N) > 0
        SUM3N = SUM3(N)
        !
        IF(N > 1) THEN
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
        NBRLO(N) = NBRX
        !
        !           Quickly find first index ILO for which LSTN0(ILO) >= N
        !           Expect to throw out 50% of N2's this way, on average
        ILO = NBRN + 1
        DO I = 1, NBRN
           IF( LSTN0(I) >= N ) ILO = MIN(ILO,I)
        END DO
        !
        !           Handle self-self interactions
        IF( ILO <= NBRN .AND. LSTN0(ILO) == N ) THEN
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
        DO I = ILO, NBRN
           N2 = LSTN0(I)
           DDX = X(N2) - XX
           DDY = Y(N2) - YY
           DDZ = Z(N2) - ZZ
           IF (DDX*DDX+DDY*DDY+DDZ*DDZ  <=  RHI2) THEN
              NBRNR = NBRNR + 1
              LSTN2(NBRNR) = N2
           END IF
        END DO
        !$$$            TIME = CPUTIME(TZERO)
        !$$$            TIMDST = TIMDST + TIME
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
           DO WHILE(NXI <= NXIMAX .AND. N2 > INBX)
              NXI=NXI+1
              INBX=IABS(INB14(NXI))
           ENDDO
           IF(NXI > NXIMAX) THEN
              EXCL14=1
           ELSE IF(N2 == INB14(NXI)) THEN  ! exclusion found
              EXCL14=0
           ELSE
              IF(N2 == INBX) THEN
                 EXCL14=-1    ! it's a 1-4 pair
              ELSE
                 EXCL14=1
              ENDIF
           ENDIF
           !
           IF (FIXED .AND. IMOVEGA(N2) > 0) EXCL14 = 0
           ! ==================================================================
#if KEY_REPLICA==1
           !# <caves>-Aug-4-1993 (Leo Caves)
           ! Replica Exclusions (Atom/Group Exclusion)
           ! Rationale: If groups belong to same subsystem (repID) but are not in same
           ! replica unit (repNoGA) - then EXCLUDE the atom/group pair
           IF (qRep) THEN
              IF ( iRepID  ==  repID(repNoGA(N2)) .AND. &
                   iRepNo  /=  repNoGA(N2) )   THEN
                 nRepX = nRepX + 1
                 EXCL14 = 0
              ENDIF
           ENDIF
#endif /*  REPLICA*/
           ! ==================================================================
#if KEY_TSM==1
           IF ((SUM3N+SUM3(N2)) == 3) EXCL14 = 0
#endif 
           !
           !              0: Don't add; 1: add positive; -1: add negative
           IF( EXCL14 /= 0 ) THEN
              NBR(NBRX) = SIGN(N2,EXCL14)
              NBRX = NBRX + 1
           END IF
           !
        END DO
        !$$$            TIME = CPUTIME(TZERO)
        !$$$            TIMOTH = TIMOTH + TIME
        NBRHI(N) = NBRX - 1
        IF (NBRX+NAT  >=  LIMIT) THEN
           XDIST = .FALSE.
           RETURN
        ENDIF
     END DO
     !
     !---- End of major loop over cubes M ----------------------------------
     !
  END DO
  !$$$      WRITE (*,*) ' Times: ', TIMSRT, TIMDST, TIMOTH
  !
  XDIST = .TRUE.
#endif 
  RETURN
END FUNCTION XDIST

