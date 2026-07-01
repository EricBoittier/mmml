module mcmamod
#if KEY_MCMA==1 /*mcma_main*/
  use chm_kinds
  use dimens_fcm

  implicit none

  public :: MCMA, allocate_mcma
  private

  !
  !   Arrays used in PJS's Monte Carlo moves.   PJS.
  !   2D Gaussian Biasing:
  !   MAXZON = Max NZONES (2D or 1D Guassians) for each torsion or torsion pair
  !   MAX2D  = Max number of torsion pairs supported (20 phi/psi + 13 chi1/2).
  !   MXMCIC = Max length of MC IC table
  !   1D Gaussian Biasing:
  !   MAX1D  = Max number of independently biased torsions supported
  !           (chi1 for Cys,Ser,Thr,Val, omega preceding prolines).
  !   ICPSI(I)  is the # of IC entry that contains the psi torsion of res # I.
  !   ICSIDE(I) is the # of IC entry that is the first IC of side-chain move #I,
  !   where 0 <= I <= NSIDE.
  !
  !   WARNING: Currently, the arrays X0,Y0,Z0 have two conflicting uses.
  !            1) Docking: ligand coords relative to ligand center of mass
  !            2) Folding: Coords corresponding to low-energy side-chain packing.
  !        So, for docking with MAIN-chain moves, more arrays need to be added.

  integer,PARAMETER :: MAXZON=8, MAX2D=33, MXMCIC=5000, MAX1D=5, &
       MAXTOR=MAX2D+MAX1D

  real(chm_real) P(MAXZON,MAXTOR),SUMP(MAXTOR),ZONBIN(MAXZON,MAXTOR), &
       MEAN1(MAXZON,MAXTOR),SIGMA1(MAXZON,MAXTOR), &
       MEAN2(MAXZON,MAX2D), SIGMA2(MAXZON,MAX2D)
  INTEGER NZONES(MAXTOR),TORTYP(MXMCIC),SEGNUM(MXMCIC), &
       MOV2D(MXMCIC),MVINDX(MXMCIC),MVATM0(MXMCIC), &
       MVATMF(MXMCIC),ICSIDE(MXMCIC),NMOVES, &
       NSIDE
  integer, allocatable :: ICPSI(:)
  real(chm_real), allocatable :: X0(:), Y0(:), Z0(:)
  real(chm_real) DCOSMX,DPHIMX,DPSIMX,DXYZMX,SIGSC, &
       XCM,YCM,ZCM,OLDXCM,OLDYCM,OLDZCM,OLDCOS, &
       OLDPHI,OLDPSI,NEWCOS,NEWPHI,NEWPSI,MTOT,DX,DY,DZ,BIN2D
  INTEGER MVUNIT,ISEED,IFIRST,ILAST
  CHARACTER(len=4) :: ACTION,SEGMOV,RESMOV

contains

  subroutine allocate_mcma()
    use dimens_fcm
    use memory
    character(len=*), parameter :: fname = 'mcma.src', pname = 'allocate_mcma'

    call chmalloc(fname,pname,'ICPSI',MAXRES,intg=ICPSI)
    call chmalloc(fname,pname,'X0',MAXA,crl=X0)
    call chmalloc(fname,pname,'Y0',MAXA,crl=Y0)
    call chmalloc(fname,pname,'Z0',MAXA,crl=Z0)
  end subroutine allocate_mcma

SUBROUTINE MCMA
  !
  !  Author: Peter J. Steinbach, 2003-2004
  !
  !  Monte Carlo-minimization/annealing (MCMA) routines for
  !  biased sampling of polypeptide conformations and for
  !  docking simulations.
  !  Accepted coordinates are stored in COMP set.
  !
  !  1. Moves for peptide/protein structure prediction:
  !  bias, back, main, side, all
  !  Limitation: Biased moves are only implemented for peptides.
  !              Nonbiased moves can be performed by scripting.
  !
  !  2. Moves for docking one molecule (segid SEGMOV, resid RESMOV):
  !  rota, tran, rotr
  !  Acceptance of docking moves: accm, accr, acct, accb
  !  or rotated (or both) each step.
  !  Random changes are made to PHI, PSI, COS(THETA) and
  !  X, Y, Z.  PHI, PSI, and THETA are the Euler angles
  !  specifying the current orientation of the ligand relative
  !  to the previously accepted orientation.
  !  Limitation: Implementation only moves one ligand molecule
  !              relative to the rest of the system.
  !
  !  References: PJ Steinbach, Proteins 2004, in press.
  !              Abagyan & Totrov, JMB 1994.
  !              Li and Scheraga, PNAS 1987.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  !  Note fstshk.f90 is not present here.
  use bases_fcm
  use comand
  use contrl
  use coord
  use coordc
  use energym
  use hbondm
  use image
  use inbnd
  use param
  use pert
  use psf
  use shake
  use stream
  use string
  use timerm
  use galgor
  use clcg_mod, only: random

  implicit none
  !
  INTEGER I,J,K,L,N,NJUMPS,IZONE,IRES,LRES,ISEG,NANNL, &
       IWRIGL,IPRINT
  real(chm_real) TMAX,UNIFRM,PHI1,PHI2,COSPHI,SINPHI,COSPSI,SINPSI, &
       SINTHE,CC,SS,SC1,SC2
  !
  !  ACTION: CHAR*4: 'INIT','ROTA','TRAN','ROTR','BIAS','ALL','BACK','MAIN',
  !                  'SIDE','ACCM','ACCR','ACCT','ACCB',
  !  SEGMOV: CHAR*4 SEGID of segment to be trans/rotated (Only 1 supported).
  !  RESMOV: CHAR*4 RESID of segment to be trans/rotated (Only 1 or 'all' supported)
  !  DCOSMX: Max change in cosine of Euler angle theta.
  !  DPHIMX: Max change in Euler angle phi.
  !  DPSIMX: Max change in Euler angle psi.
  !  DXYZMX: Max change in x, y, and z (All 3 changed each step).
  !  SIGSC:  Scale all default (Abagyan & Totrov's) Gaussian FWHMs by SIGSC,
  !          If 0 => No Gaussian biasing
  !  ISEED:  Seed for random number generator.
  !
  !
  ACTION = NEXTA4(COMLYN,COMLEN)
  !
  !  The first (INITialization) call to this routine must specify parameters
  !  used in all subsequent random rotations and translations.
  !
  IWRIGL = 0
  IPRINT = 2
  IF (ACTION.EQ.'INIT') THEN
     SEGMOV = NEXTA4(comlyn,comlen)
     RESMOV = NEXTA4(comlyn,comlen)
     DCOSMX = NEXTF(comlyn,comlen)
     IF (DCOSMX .GT. TWO) DCOSMX = TWO
     DPHIMX = NEXTF(comlyn,comlen)
     DPSIMX = NEXTF(comlyn,comlen)
     DXYZMX = NEXTF(comlyn,comlen)
     SIGSC  = NEXTF(comlyn,comlen)
     ISEED  = NEXTI(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,20) SEGMOV,RESMOV,DCOSMX, &
          DXYZMX,DPHIMX,DPSIMX,SIGSC,ISEED
20   FORMAT(/' MCMA> SEGMov =  ',A4,6X,'RESMov =  ',A4/7X, &
          'DCOSmx =',F7.3,'     DXYZmx =',F7.3/7X, &
          'DPHImx =',F7.3,'     DPSImx =',F7.3/7X, &
          'SIGSc  =',F7.3,'     ISEED  =',I8)
     CALL GETMOV
     !
     !     Call RESET to get initial center of mass, Euler angles,
     !     and coords (X0,Y0,Z0) relative to cm for the ligand to be docked,
     !     if any, (ie, for SEGMOV,RESMOV).
     !
     CALL RESET(MTOT,X0,Y0,Z0,XCM,YCM,ZCM,OLDXCM,OLDYCM,OLDZCM, &
          OLDCOS,OLDPHI,OLDPSI,IFIRST,ILAST)
     !
     !       Fill arrays used in biased dihedral jumps, and check IC
     !       table for compatibility with these moves.
     !
     CALL GET2DG
     CALL CHEKIC
     !
  ELSE IF (ACTION(1:3).EQ.'ALL') THEN
     !
     !       Modify all torsions in IC table, using 2D-Gaussian zones,
     !       where defined, for biased MC.
     !       NOTE: By not calling NEWXYZ after MOVEIC, this ALL move allows
     !       user-specified "initcoord", desirable when closing loops.  PJS 17Oct03
     !
     !C        IWRIGL = NEXTI(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,120) ACTION
     DO I = 1,NMOVES
        K = MVINDX(I)
        CALL MOVEIC(K,IWRIGL,IPRINT)
     enddo
     !
  ELSE IF (ACTION.EQ.'BIAS') THEN
     !
     !       Modify NJUMPS unpaired torsions or pairs of torsions in IC table,
     !       using 2D-Gaussian zones, where defined, for biased MC.
     !
     NJUMPS = NEXTI(comlyn,comlen)
     !C        IWRIGL = NEXTI(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,120) ACTION,NJUMPS
120  FORMAT(/' MCMA> ',A,I4)
     DO L = 1,NJUMPS
        UNIFRM = RANDOM(ISEED)
        I = NINT(UNIFRM/BIN2D + HALF)
        IF (I .EQ. 0) I = 1
        IF (I .GT. NMOVES) I = NMOVES
        K = MVINDX(I)
        CALL MOVEIC(K,IWRIGL,IPRINT)
        CALL NEWXYZ(K,IWRIGL)
     enddo
     !
  ELSE IF (ACTION.EQ.'SIDE') THEN
     !
     !       Modify NJUMPS unpaired or pairs of side-chain torsions in IC table,
     !       using 2D-Gaussian zones, where defined, for biased MC.
     !
     NJUMPS = NEXTI(comlyn,comlen)
     !C        IWRIGL = NEXTI(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,120) ACTION,NJUMPS
     DO L = 1,NJUMPS
        UNIFRM = RANDOM(ISEED)
        I = NINT(UNIFRM*DFLOAT(NSIDE) + HALF)
        IF (I .GT. NSIDE) I = NSIDE
        IF (I .EQ. 0) I = 1
        K = ICSIDE(I)
        CALL MOVEIC(K,IWRIGL,IPRINT)
        CALL NEWXYZ(K,IWRIGL)
     enddo
     !
  ELSE IF (ACTION.EQ.'MAIN') THEN
     !
     !       A concerted main-chain move to promote secondary structure.
     !       Change NJUMPS consecutive (phi,psi) pairs to the SAME torsional values.
     !
     NJUMPS = NEXTI(comlyn,comlen)
     !C        IWRIGL = NEXTI(comlyn,comlen)
     NANNL = NEXTI(comlyn,comlen)
     TMAX  = NEXTF(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,160) ACTION,NJUMPS,NANNL,TMAX
160  FORMAT(/' MCMA> ',A,I4,'; NANNL=',I8,'; TMAX=',F8.2)
     I = 1
     IF(NJUMPS.GT. 0) THEN
        CALL MAINCH(NJUMPS,I,IWRIGL,ISEG)
     ELSE
        !         Anneal segment number -NJUMPS, without any torsional jumps
        ISEG = MAX(1,-NJUMPS)
     END IF
     IF(NANNL .NE. 0) CALL RELAX(NANNL,TMAX,ISEG)
     !
  ELSE IF (ACTION.EQ.'BACK') THEN
     !
     !       Change NJUMPS consecutive (phi,psi), BIASING EACH RESIDUE INDIVIDUALLY.
     !
     NJUMPS = NEXTI(comlyn,comlen)
     !C        IWRIGL = NEXTI(comlyn,comlen)
     NANNL = NEXTI(comlyn,comlen)
     TMAX  = NEXTF(comlyn,comlen)
     IF (PRNLEV .GE. 2) WRITE(OUTU,160) ACTION,NJUMPS,NANNL,TMAX
     I = 0
     IF(NJUMPS.GT. 0) THEN
        CALL MAINCH(NJUMPS,I,IWRIGL,ISEG)
     ELSE
        !         Anneal segment number -NJUMPS, without any torsional jumps
        ISEG = MAX(1,-NJUMPS)
     END IF
     IF(NANNL .NE. 0) CALL RELAX(NANNL,TMAX,ISEG)
     !
  ELSE IF (ACTION.EQ.'ROTA') THEN
     !
     !C      IF (PRNLEV .GE. 2) WRITE(OUTU,180) XCM,YCM,ZCM
     !C180   FORMAT(/' MCMA> XCM,YCM,ZCM of atoms to rotate:',3F10.4/)
     !
     !       Modify Euler angles.  ISEED value is changed by RANDOM.
     !
     NEWCOS = OLDCOS + (TWO*RANDOM(ISEED)-ONE)*DCOSMX
     NEWPHI = OLDPHI + (TWO*RANDOM(ISEED)-ONE)*DPHIMX
     NEWPSI = OLDPSI + (TWO*RANDOM(ISEED)-ONE)*DPSIMX
     !C
     !C      Allen and Tildesley rotate theta by pi when NEWCOS is outside (-1,1):
     !C      NEWCOS = NEWCOS - DNINT(NEWCOS/TWO)*TWO   Allen and Tild., p133
     !C      --OR--
     !C      IF (NEWCOS .LT. -ONE) THEN IF
     !C        NEWCOS = NEWCOS + TWO
     !C      ELSE IF (NEWCOS .GT. ONE) THEN
     !C        NEWCOS = NEWCOS - TWO
     !C      END IF
     !
     !  This doesn't rotate theta by pi:
     !
     IF (NEWCOS .LT. MINONE) THEN
        !C        NEWCOS = MINONE + (MINONE-NEWCOS)
        NEWCOS = MINTWO - NEWCOS
        NEWPHI = NEWPHI + PI
     ELSE IF (NEWCOS .GT. ONE) THEN
        !C        NEWCOS = ONE - (NEWCOS-ONE)
        NEWCOS = TWO - NEWCOS
        NEWPHI = NEWPHI + PI
     END IF
     IF (NEWPHI .LT. -PI) THEN
        NEWPHI = NEWPHI + TWOPI
     ELSE IF (NEWPHI .GT. PI) THEN
        NEWPHI = NEWPHI - TWOPI
     END IF
     IF (NEWPSI .LT. -PI) THEN
        NEWPSI = NEWPSI + TWOPI
     ELSE IF (NEWPSI .GT. PI) THEN
        NEWPSI = NEWPSI - TWOPI
     END IF
     !C      IF (PRNLEV .GE. 2) WRITE(OUTU,185) NEWCOS,NEWPHI,NEWPSI
     !C185   FORMAT(/' MCMA> NEWCOS,PHI,PSI:',3F8.4/)
     !
     !   Apply rotation matrix (Goldstein, 4-47) to t=0 x,y,z values relative to cm.
     !
     COSPHI = COS(NEWPHI)
     SINPHI = SIN(NEWPHI)
     COSPSI = COS(NEWPSI)
     SINPSI = SIN(NEWPSI)
     SINTHE = SQRT ( ONE - NEWCOS * NEWCOS )
     CC = COSPHI*COSPSI
     SS = SINPHI*SINPSI
     SC1 = SINPHI*COSPSI
     SC2 = SINPSI*COSPHI
     DO I = IFIRST,ILAST
        X(I)= OLDXCM+ (CC-NEWCOS*SS)*X0(I) - (SC2+NEWCOS*SC1)*Y0(I) &
             +  SINTHE*SINPHI*Z0(I)
        Y(I)= OLDYCM+ (SC1+NEWCOS*SC2)*X0(I) - (SS-NEWCOS*CC)*Y0(I) &
             -  SINTHE*COSPHI*Z0(I)
        Z(I)= OLDZCM+ SINTHE*(SINPSI*X0(I)+COSPSI*Y0(I))+ NEWCOS*Z0(I)
     enddo
  ELSE IF (ACTION.EQ.'TRAN') THEN
     DX = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     DY = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     DZ = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     XCM = OLDXCM + DX
     YCM = OLDYCM + DY
     ZCM = OLDZCM + DZ
     DO I = IFIRST,ILAST
        X(I) = XCOMP(I) + DX
        Y(I) = YCOMP(I) + DY
        Z(I) = ZCOMP(I) + DZ
     enddo
     !                                 Do ROtation AND TRanslation
  ELSE IF (ACTION.EQ.'ROTR') THEN
     !
     !       Modify Euler angles.  ISEED value is changed by RANDOM.
     !
     NEWCOS = OLDCOS + (TWO*RANDOM(ISEED)-ONE)*DCOSMX
     NEWPHI = OLDPHI + (TWO*RANDOM(ISEED)-ONE)*DPHIMX
     NEWPSI = OLDPSI + (TWO*RANDOM(ISEED)-ONE)*DPSIMX
     !
     !  This doesn't rotate theta by pi:
     !
     IF (NEWCOS .LT. MINONE) THEN
        NEWCOS = MINTWO - NEWCOS
        NEWPHI = NEWPHI + PI
     ELSE IF (NEWCOS .GT. ONE) THEN
        NEWCOS = TWO - NEWCOS
        NEWPHI = NEWPHI + PI
     END IF
     IF (NEWPHI .LT. -PI) THEN
        NEWPHI = NEWPHI + TWOPI
     ELSE IF (NEWPHI .GT. PI) THEN
        NEWPHI = NEWPHI - TWOPI
     END IF
     IF (NEWPSI .LT. -PI) THEN
        NEWPSI = NEWPSI + TWOPI
     ELSE IF (NEWPSI .GT. PI) THEN
        NEWPSI = NEWPSI - TWOPI
     END IF
     !
     !   Apply rotation AND translation.
     !
     COSPHI = COS(NEWPHI)
     SINPHI = SIN(NEWPHI)
     COSPSI = COS(NEWPSI)
     SINPSI = SIN(NEWPSI)
     SINTHE = SQRT ( ONE - NEWCOS * NEWCOS )
     CC = COSPHI*COSPSI
     SS = SINPHI*SINPSI
     SC1 = SINPHI*COSPSI
     SC2 = SINPSI*COSPHI
     DX = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     DY = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     DZ = (TWO*RANDOM(ISEED)-ONE)*DXYZMX
     DO I = IFIRST,ILAST
        X(I)= OLDXCM + (CC-NEWCOS*SS)*X0(I) - (SC2+NEWCOS*SC1)*Y0(I) &
             +  SINTHE*SINPHI*Z0(I) + DX
        Y(I)= OLDYCM + (SC1+NEWCOS*SC2)*X0(I) - (SS-NEWCOS*CC)*Y0(I) &
             -  SINTHE*COSPHI*Z0(I) + DY
        Z(I)= OLDZCM + SINTHE*(SINPSI*X0(I)+COSPSI*Y0(I))+NEWCOS*Z0(I) &
             + DZ
     enddo
     XCM = OLDXCM + DX
     YCM = OLDYCM + DY
     ZCM = OLDZCM + DZ
  ELSE IF (ACTION.EQ.'ACCM') THEN
     !       After an ACCepted Minimization, RESET the cm,
     !       Euler angles, and coords (X0,Y0,Z0) in cm frame.
     CALL RESET(MTOT,X0,Y0,Z0,XCM,YCM,ZCM,OLDXCM,OLDYCM,OLDZCM, &
          OLDCOS,OLDPHI,OLDPSI,IFIRST,ILAST)
  ELSE IF (ACTION(1:3).EQ.'ACC') THEN
     DO I = IFIRST,ILAST
        XCOMP(I) = X(I)
        YCOMP(I) = Y(I)
        ZCOMP(I) = Z(I)
     enddo
     IF (ACTION(4:4).EQ.'T') THEN
        OLDXCM = XCM
        OLDYCM = YCM
        OLDZCM = ZCM
     ELSE IF (ACTION(4:4).EQ.'R') THEN
        OLDCOS = NEWCOS
        OLDPHI = NEWPHI
        OLDPSI = NEWPSI
     ELSE IF (ACTION(4:4).EQ.'B') THEN
        OLDXCM = XCM
        OLDYCM = YCM
        OLDZCM = ZCM
        OLDCOS = NEWCOS
        OLDPHI = NEWPHI
        OLDPSI = NEWPSI
     END IF
  ELSE
     IF (PRNLEV .GE. 2) WRITE(OUTU,700) ACTION
700  FORMAT(/' MCMA> WARNING: Unrecognized ACTION = ',A4, &
          '.  Nothing done!')
  ENDIF
  !CC    IF (PRNLEV .GE. 2) WRITE(OUTU,710)
  !CC710 FORMAT(/' MCMA> Exiting routine MCMA.')
  RETURN
END SUBROUTINE MCMA

SUBROUTINE ERELAX(NANNL,TMAX)
  !
  ! *****   Evaluate energy of relaxed conformation.
  !  ***    Do minimization/annealing and evaluate energy
  !   *     Annealing: heat to ~TMAX; cool to ~TMIN.
  !         22Aug05: Use a 1-fs (2-fs) time step if TMAX >0 (<0).
  !
  use chm_kinds
  use dimens_fcm
  use comand
  use number
  use clcg_mod, only: random

  implicit none
  INTEGER NANNL,I,NSTEPS,IHTFRQ,NABNRS,MDSEED
  real(chm_real) TMAX,DELTAT,TMIN
  LOGICAL LUSED,ONEFS
  !
  IF(NANNL .LT. 0) THEN
     !       NANNL<0: Do -NANNL changes to all side-chains moves, changing
     !                one side-chain (or pair) at a time and doing TMAX steps of ABNR
     NABNRS = NINT(TMAX)
     IF(NABNRS .GT. 0) THEN
        COMLYN ='MINI SD NSTE 5 NPRI 5 TOLG 0.01 INBF -1 STEP 0.005'
        COMLEN = 50
        CALL MAINCOMX(COMLYN,COMLEN,LUSED)
        !
        WRITE(COMLYN,20) NABNRS
20      FORMAT('MINI ABNR NSTE',I5,' NPRI 50 TOLG 0.01 INBF -1')
        COMLEN = 45
        CALL MAINCOMX(COMLYN,COMLEN,LUSED)
     END IF
     COMLYN = 'ENER'
     COMLEN = 4
     CALL MAINCOMX(COMLYN,COMLEN,LUSED)
  ELSE IF(NANNL .GT. 0) THEN
     !       NANNL>0: Do NANNL MD steps of annealing to max temperature = TMAX
     !                10% (90%) of MD steps for heating (cooling).
     !  First, minimize energy.
     !
     COMLYN ='MINI SD NSTE 50 NPRI 50 TOLG 5. INBF -1 STEP 0.005'
     COMLEN = 50
     CALL MAINCOMX(COMLYN,COMLEN,LUSED)
     COMLYN ='MINI ABNR NSTE 500 NPRI 100 TOLG 1. INBF -1'
     COMLEN = 43
     CALL MAINCOMX(COMLYN,COMLEN,LUSED)
     !
     !  Heat from 300 K to TMAX in 5 jumps for a total of NSTEPS heating steps.
     !  Note below, FIRS = 400. to quicken heating.  ASSUMES SHAKE!
     !
     ONEFS = TMAX .GT. ZERO
     IF(TMAX .LT. ZERO) TMAX = -TMAX
     NSTEPS = NANNL/10
     DELTAT = 0.2*(TMAX - 300.)
     IHTFRQ = NINT(0.2*DFLOAT(NSTEPS))
     IF(5*IHTFRQ .GT. NSTEPS) IHTFRQ = IHTFRQ-1
     WRITE(COMLYN(1:43),50) NSTEPS
     MDSEED = IDINT(RANDOM(ISEED)*1.0E7)
     IF(ONEFS) THEN
        WRITE(COMLYN(1:43),45) NSTEPS
     ELSE
        WRITE(COMLYN(1:43),50) NSTEPS
     ENDIF
45   FORMAT('DYNA LEAP VERL STAR NSTE',I7,' TIME 0.001 ')
50   FORMAT('DYNA LEAP VERL STAR NSTE',I7,' TIME 0.002 ')
     COMLYN(44:75) = 'IUNC -1 IUNR -1 IUNW -1 KUNI -1 '
     WRITE(COMLYN(76:114),51) IHTFRQ
51   FORMAT('INBF -1 IHTF',I6,' TSTR  10. FIRS 400. ')
     !C 51   FORMAT('INBF -1 IHTF',I6,' TSTR  50. FIRS  50. ')
     WRITE(COMLYN(115:149),52) TMAX,DELTAT
52   FORMAT('FINA',F7.1,' TEMI',F7.1,' ECHE 1.E30 ')
     WRITE(COMLYN(150:163),53) MDSEED
53   FORMAT('ISEE',I10)
     COMLEN = 163
     CALL MAINCOMX(COMLYN,COMLEN,LUSED)
     !
     !  Cool from TMAX to TMIN = 150 K in 10 jumps for total of NSTEPS of heat/cool.
     !
     TMIN = 150.
     NSTEPS = NANNL - NSTEPS
     DELTAT = 0.1*(TMIN - TMAX)
     IHTFRQ = NINT(0.1*DFLOAT(NSTEPS))
     MDSEED = IDINT(RANDOM(ISEED)*1.0E7)
     IF(10*IHTFRQ .GT. NSTEPS) IHTFRQ = IHTFRQ-1
     IF(ONEFS) THEN
        WRITE(COMLYN(1:43),45) NSTEPS
     ELSE
        WRITE(COMLYN(1:43),50) NSTEPS
     ENDIF
     COMLYN(44:75) = 'IUNC -1 IUNR -1 IUNW -1 KUNI -1 '
     WRITE(COMLYN(76:118),61) IHTFRQ,TMAX,TMAX
61   FORMAT('INBF -1 IHTF',I6,' TSTR',F7.1,' FIRS',F7.1,' ')
     WRITE(COMLYN(119:153),52) TMIN,DELTAT
     WRITE(COMLYN(154:167),53) MDSEED
     COMLEN = 167
     CALL MAINCOMX(COMLYN,COMLEN,LUSED)
  END IF
  !
  RETURN
END SUBROUTINE ERELAX

SUBROUTINE MAINCH(NJUMPS,ISAME,IWRIGL,ISEG)
  !
  ! ***** Select residue IRES at random.  Change (phi,psi) of residues
  !  ***  IRES to LRES = IRES+NJUMPS-1.
  !   *   ISAME= 1 Each residue assigned same phi-psi, biased for residue IRES.
  !            = 0 Each residue assigned its own biased phi-psi.
  !       IWRIGL=0 implies nonlocal "thrashing" move.
  !

  use chm_kinds
  use intcor_module
  use dimens_fcm
  use number
  use bases_fcm
  use stream
  use psf
  use clcg_mod, only: random
  implicit none
  INTEGER NJUMPS,ISAME,IWRIGL,ISEG,J,K,L,IRES,LRES,IPRINT
  real(chm_real) UNIFRM,PHI1,PHI2
  !
  IPRINT=2
10 FORMAT(/' MAINCH> ',A4,' IC:',I6,' New Angle:',F8.2)
11 FORMAT(9X,A4,' IC:',I6,' New Angle:',F8.2)
  !
  !     Select residue IRES at random.  Change (phi,psi) of residues
  !     IRES to LRES = IRES+NJUMPS-1.
  !
  J = NRES-NJUMPS+1
145 UNIFRM = RANDOM(ISEED)
  IRES = NINT(UNIFRM*DFLOAT(J) + HALF)
  IF (IRES .GT. J) IRES = J
  IF (IRES .EQ. 0) IRES = 1
  K = ICPSI(IRES)
  IF(K .EQ. 0) GO TO 145
  ISEG = SEGNUM(K)
  !
  IF(ISAME .EQ. 1) THEN
     J = MOV2D(K)
     IF(J .GT. 100) J = J-100
     CALL BIAS2(J,PHI1,PHI2)
  END IF
  LRES = IRES+NJUMPS-1
  DO L = IRES,LRES
     K = ICPSI(L)
     IF(K .EQ. 0) THEN
        IF (PRNLEV .GE. 2) WRITE(OUTU,154) L,L
154     FORMAT(/' MAINCH> WARNING: Psi torsion of residue',I6, &
             ' is not in MC IC table; residue',I6, &
             ' torsions unchanged!')
     ELSE IF(SEGNUM(K) .NE. ISEG) THEN
        IF (PRNLEV .GE. 2) WRITE(OUTU,155) IRES,L,L
155     FORMAT(/' MAINCH> WARNING: Residues',I6,' and',I6, &
             ' are in different segments; residue',I6, &
             ' torsions unchanged!')
     ELSE IF(K .EQ. 1) THEN
        !           The phi torsion of residue L is not in the MC IC table; change psi.
        IF(ISAME .EQ. 0) THEN
           J = MOV2D(K)-100
           CALL BIAS2(J,PHI1,PHI2)
        END IF
        CALL NEWIC(K,PHI2,icr_struct%PIC)
        IF (PRNLEV .GE. 2) WRITE(OUTU,10) ACTION,K,PHI2
        CALL NEWXYZ(K,IWRIGL)
     ELSE
        IF(ISAME .EQ. 0) THEN
           !             Change phi,psi of residue L, biasing based on residue L.
           K = K-1
           CALL MOVEIC(K,IWRIGL,IPRINT)
        ELSE
           !             Change phi,psi of residue L, biasing based on residue IRES.
           IF(IWRIGL .EQ. 0) THEN
              CALL NEWICS(K-1,K,PHI1,PHI2,icr_struct%PIC)

              IF (PRNLEV .GE. 2) THEN
                 WRITE(OUTU,10) ACTION,K-1,PHI1
                 WRITE(OUTU,11) ACTION,K,PHI2
              END IF
           ELSE
              !               Wriggle phi,psi of residue L, biased based on residue IRES.
           END IF
        END IF
        CALL NEWXYZ(K,IWRIGL)
     END IF
  enddo
  RETURN
END SUBROUTINE MAINCH

SUBROUTINE RELAX(NANNL,TMAX,ISEG)
  !
  ! ***** NANNL<0: Evaluate -NANNL different side-chain arrangements.
  !  ***           TMAX = # ABNR steps after each side-chain randomization.
  !   *   NANNL>0: Do NANNL MD steps of annealing to max temperature = TMAX
  !                10% (90%) of MD steps for heating (cooling)
  !

  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use coord
  use coordc
  use stream
  use psf
  use energym

  implicit none
  INTEGER NANNL,ISEG,IRES,LRES,ATOM1,ATOM2,I,J,K,L,N,IWRIGL,IPRINT
  real(chm_real) TMAX,EMIN,ABNRS
  !     IRES,LRES and ATOM1,ATOM2 assigned 1st,last RES/ATOM in segment
  IRES = NICTOT(ISEG)+1
  LRES = NICTOT(ISEG+1)
  ATOM1 = IBASE(IRES)+1
  ATOM2 = IBASE(LRES+1)
  !
  IF(NANNL .LT. 0) THEN
     IWRIGL = 0
     IPRINT = 10
     !
     !     First, energy-minimize coords following main-chain change, and
     !     save the energy and coordinates.
     !
     ABNRS = 5.*TMAX
     IF(ABNRS .LT. 150.) ABNRS = 150.
     CALL ERELAX(NANNL,ABNRS)
     EMIN = EPROP(EPOT)
     IF (PRNLEV .GE. 2) WRITE(OUTU,201) EMIN
     DO 20 I = ATOM1,ATOM2
        X0(I) = X(I)
        Y0(I) = Y(I)
20      Z0(I) = Z(I)
        !
        !     Now, consider -NANNL randomizations of each
        !     side-chain move, one at a time.
        !
        DO L = 1,-NANNL
           DO N = 1,NSIDE
              K = ICSIDE(N)
              CALL MOVEIC(K,IWRIGL,IPRINT)
              CALL NEWXYZ(K,IWRIGL)
              !
              CALL ERELAX(NANNL,TMAX)
              IF(EPROP(EPOT) .LT. EMIN) THEN
                 EMIN = EPROP(EPOT)
                 IF (PRNLEV .GE. 2) WRITE(OUTU,205) L,N,EMIN
                 DO I = ATOM1,ATOM2
                    X0(I) = X(I)
                    Y0(I) = Y(I)
                    Z0(I) = Z(I)
                 enddo
              ELSE
                 DO  I = ATOM1,ATOM2
                    X(I) = X0(I)
                    Y(I) = Y0(I)
                    Z(I) = Z0(I)
                 enddo
              END IF
           enddo
           IF (PRNLEV .GE. 2) WRITE(OUTU,206) L,EMIN
        enddo
        DO  I = ATOM1,ATOM2
           X(I) = X0(I)
           Y(I) = Y0(I)
           Z(I) = Z0(I)
        enddo
        !
201     FORMAT(/' RELAX> Initial side-chain conformation:', &
             ' E =',1PE13.5,' kcal/mol.')
205     FORMAT(/' RELAX> Side-chain packing #, move #',I5,I5, &
             ': new EMIN =',1PE13.5,'.')
206     FORMAT(/' RELAX> Side-chain packing #',I5, &
             ' EMIN =',1PE13.5,' kcal/mol.')
     ELSE
        !
        !      Perform MD simulated annealing.  Because MD overwrites
        !      comparison coords, save them and restore afterwards.
        !
        DO I = ATOM1,ATOM2
           X0(I) = XCOMP(I)
           Y0(I) = YCOMP(I)
           Z0(I) = ZCOMP(I)
        enddo
        !
        CALL ERELAX(NANNL,TMAX)
        !
        DO I = ATOM1,ATOM2
           XCOMP(I) = X0(I)
           YCOMP(I) = Y0(I)
           ZCOMP(I) = Z0(I)
        enddo
     END IF
     IF (PRNLEV .GE. 2) WRITE(OUTU,400)
400  FORMAT(/' RELAX> Exiting routine RELAX.')
     RETURN
END SUBROUTINE RELAX

SUBROUTINE MOVEIC(K,IWRIGL,IPRINT)
  !
  !  *****
  !   ***  Use appropriate biasing to change IC entry K (and perhaps K+1).
  !    *
  !
  use chm_kinds
  use intcor_module
  use dimens_fcm
  use number
  use bases_fcm
  use stream
  use clcg_mod, only: random

  implicit none
  INTEGER K,IWRIGL,IPRINT,J
  real(chm_real) PHI1,PHI2
  IF (MOV2D(K) .EQ. -1) THEN
     PHI1 = THR6TY * ( RANDOM(ISEED) - HALF )
     CALL NEWIC(K,PHI1,icr_struct%PIC)
     IF (PRNLEV .GE. IPRINT) WRITE(OUTU,10) ACTION,K,PHI1
  ELSE IF(MOV2D(K) .GT. 100) THEN
     !
     !       First IC entry is the second angle (psi or chi2) in a pair
     !
     J = MOV2D(K)-100
     CALL BIAS2(J,PHI1,PHI2)
     CALL NEWIC(K,PHI2,icr_struct%PIC)
     IF (PRNLEV .GE. IPRINT) WRITE(OUTU,10) ACTION,K,PHI2
  ELSE
     J = MOV2D(K)
     IF (K.LT.icr_struct%LENIC .AND. J.EQ.MOV2D(K+1)) THEN
        IF(J.GT.20 .OR. IWRIGL.EQ.0) THEN
           !
           !           chi1/chi2 pair, or phi/psi not to be wriggled
           !
           CALL BIAS2(J,PHI1,PHI2)
           CALL NEWICS(K,K+1,PHI1,PHI2,icr_struct%PIC)
           !..##ENDIF

           IF (PRNLEV .GE. IPRINT) THEN
              WRITE(OUTU,10) ACTION,K,PHI1
              WRITE(OUTU,11) ACTION,K+1,PHI2
           END IF
        ELSE
           !
           !           phi/psi pair to be wriggled
           !
        END IF
10      FORMAT(/' MOVEIC> ',A4,' IC:',I6,' New Angle:',F8.2)
11      FORMAT(9X,A4,' IC:',I6,' New Angle:',F8.2)
     ELSE IF (J.GT.MAX2D .AND. J.LE.MAXTOR) THEN
        IF(J.LT.MAXTOR .OR. IWRIGL.EQ.0) THEN
           !
           !           chi1 of CYS,SER,THR,VAL, or pre-PRO omega not to be wriggled
           !
           CALL BIAS1(J,PHI1)
           CALL NEWIC(K,PHI1,icr_struct%PIC)

           IF (PRNLEV .GE. IPRINT) WRITE(OUTU,10) ACTION,K,PHI1
        ELSE
           !
           !           pre-PRO omega (J=MAXTOR) to be wriggled
           !
        END IF
     ELSE
        IF (PRNLEV .GE. IPRINT) WRITE(OUTU,105) K,K+1
105     FORMAT(' WARNING!!  Unrecognized moves for IC #s',2I4)
     END IF
  END IF
  RETURN
END SUBROUTINE MOVEIC

SUBROUTINE WRIGL(IC1,IC2,IWRIGL,IPRINT)
  !
  !  ***** Change IC entry number IC1 (and perhaps IC2) by a finite (possibly large) amount,
  !   ***  by "wriggling" over several neighboring phi/psi angles by small amounts so as
  !    *   to minimize the movement of distant atoms.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use stream
  implicit none
  INTEGER IC1,IC2,IWRIGL,IPRINT,NBONDS
  !
  !   Determine NBONDS given IC1,IC2 and IWRIGL
  !
  IF(IWRIGL .EQ. 1) THEN
  ENDIF
  !
  !   Create vectors and arrays needed for SVD call
  !
  !   Do SVD
  !
  !   Update IC table
  !
  RETURN
END SUBROUTINE WRIGL

SUBROUTINE BIAS1(IMOVE,PHI1)
  !
  ! *****
  !  ***    Get 1 dihedral angle or 2 (a phi/psi or chi1/chi2 pair)
  !   *     PHI1 and PHI2 are in degrees.  IC table not changed.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use clcg_mod, only: bmgaus, random
  implicit none
  INTEGER IMOVE,I,IZONE
  real(chm_real) :: UNIFRM, PHI1
  IF (SIGSC .GT. ZERO) THEN
     UNIFRM = RANDOM(ISEED)
     DO I = 1,NZONES(IMOVE)
        IF (UNIFRM .LE. ZONBIN(I,IMOVE)) THEN
           IZONE = I
           GO TO 20
        END IF
     END DO
20   PHI1= MEAN1(IZONE,IMOVE) + BMGAUS(SIGMA1(IZONE,IMOVE),ISEED)
  ELSE
     PHI1= THR6TY * ( RANDOM(ISEED) - HALF )
  END IF
  RETURN
END SUBROUTINE BIAS1

SUBROUTINE BIAS2(IMOVE,PHI1,PHI2)
  use chm_kinds
  use dimens_fcm
  use number
  use clcg_mod, only: bmgaus, random

  implicit none
  INTEGER IMOVE,I,IZONE
  real(chm_real) :: UNIFRM, PHI1, PHI2
  IF (SIGSC .GT. ZERO) THEN
     UNIFRM = RANDOM(ISEED)
     DO I = 1,NZONES(IMOVE)
        IF (UNIFRM .LE. ZONBIN(I,IMOVE)) THEN
           IZONE = I
           GO TO 20
        END IF
     END DO
20   PHI1= MEAN1(IZONE,IMOVE) + BMGAUS(SIGMA1(IZONE,IMOVE),ISEED)
     PHI2= MEAN2(IZONE,IMOVE) + BMGAUS(SIGMA2(IZONE,IMOVE),ISEED)
  ELSE
     PHI1= THR6TY * ( RANDOM(ISEED) - HALF )
     PHI2= THR6TY * ( RANDOM(ISEED) - HALF )
  END IF
  RETURN
END SUBROUTINE BIAS2

SUBROUTINE NEWXYZ(IC,IWRIGL)
  !
  ! *****
  !  ***   Initialize relevant xyzs, save new ICs, and build xyzs.
  !   *
  !
  use chm_kinds
  use intcor_module
  use intcor2, only: bildc
  use dimens_fcm
  use number
  use psf
  use coord
  use coordc
  use bases_fcm

  implicit none
  INTEGER IC,IWRIGL,I
  !                         Initialize coords affected by this move
  IF(IWRIGL .EQ. 0) THEN
     DO I = MVATM0(IC),MVATMF(IC)
        X(I)=ANUM
        Y(I)=ANUM
        Z(I)=ANUM
     enddo
  ELSE
     !       Initialize coordinates of atoms affected by wriggling move.
  END IF
  !                         Do an "ic save overwrite"
  I=icr_struct%LENIC
  I=I+ics_struct%LENIC
  IF (ics_struct%INTLEN.LT.I) THEN
     CALL reintc_new(I,ics_struct)
  endif
  CALL INTCPY(ics_struct%LENIC,ics_struct%B1ic,ics_struct%B2ic, &
       ics_struct%T1ic,ics_struct%T2ic, &
       ics_struct%PIC, ics_struct%IAR, &
       ics_struct%JAR, ics_struct%KAR, &
       ics_struct%LAR, ics_struct%TAR, &
       icr_struct%LENIC,icr_struct%B1ic,icr_struct%B2ic, &
       icr_struct%T1ic,icr_struct%T2ic, &
       icr_struct%PIC, icr_struct%IAR, &
       icr_struct%JAR, icr_struct%KAR, &
       icr_struct%LAR, icr_struct%TAR, &
       .FALSE.,.TRUE.,.FALSE.,ZERO)
  IF (ics_struct%LENIC+500.GT.ics_struct%INTLEN) THEN
     I=ics_struct%LENIC+100
     CALL reintc_new(I,ics_struct)
  endif
  !
  !    Build XYZs from 'SAVEd' IC table, ie, "ic build save"
  !
  CALL BILDC(1,ics_struct%LENIC,X,Y,Z, &
       ics_struct%B1ic,ics_struct%B2ic, &
       ics_struct%T1ic,ics_struct%T2ic, &
       ics_struct%PIC, ics_struct%IAR, &
       ics_struct%JAR, ics_struct%KAR, &
       ics_struct%LAR, ics_struct%TAR, &
       NATOM)
  RETURN
END SUBROUTINE NEWXYZ

SUBROUTINE GETMOV
  !
  ! ***** Identify all atom #s belonging to SEGMOV,RESMOV.
  !  ***  First find the segment number, ISEG, and then residue number, IRES
  !   *   Next 9 lines hacked from chutil.src
  !       From psf.f90: IBASE(1)=0, IBASE(IRES+1) gives last atom of IRESth residue
  !                     NICTOT(1)=0, NICTOT(NSEG+1) gives NRES.
  !
  use dimens_fcm
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use number
  implicit none
  INTEGER ISEG,IRES,RES1ST,RESLST,I
  ISEG=1
  DO WHILE (SEGID(ISEG).NE.SEGMOV)
     ISEG=ISEG+1
  ENDDO
  IF (INDEX(RESMOV,'all').GT.0 .OR. INDEX(RESMOV,'ALL').GT.0) THEN
     RES1ST = NICTOT(ISEG)+1
     RESLST = NICTOT(ISEG+1)
     IFIRST = IBASE(RES1ST)+1
     ILAST  = IBASE(RESLST+1)
  ELSE
     ISEG=1
     DO WHILE (SEGID(ISEG).NE.SEGMOV)
        ISEG=ISEG+1
     ENDDO
     IRES=NICTOT(ISEG)+1
     DO WHILE (RESID(IRES).NE.RESMOV)
        IRES=IRES+1
     ENDDO
     IFIRST = IBASE(IRES)+1
     ILAST  = IBASE(IRES+1)
  END IF
  IF (PRNLEV .GE. 2) WRITE(OUTU,5) IFIRST,ILAST
5 FORMAT(/' GETMOV> FIRST,LAST atom #s to rotate/translate:',2I6/)
  !
  !     Get mass of residue to move.
  !
  MTOT = ZERO
  DO I = IFIRST,ILAST
     MTOT = MTOT + AMASS(I)
  enddo
  MTOT = ONE/MTOT
  RETURN
END SUBROUTINE GETMOV

SUBROUTINE GET2DG
  !
  ! *****  Initialize the 1- and 2-dimensional Gaussians used to bias MC jumps.
  !  ***   Parameters taken from Abagyan and Totrov (1994), J Mol Biol 235,983-1002.
  !   *
  !
  use chm_kinds
  use dimens_fcm
  implicit none
  INTEGER I,J
  !
  !       Set biasing parameters in columns of P,MEAN,SIGMA arrays as follows.
  !             Residues           phi/psi columns    chi1/chi2 columns
  !       ALA,ARG,ASN,ASP,CYS          1- 5             21-23 (No ala,cys)
  !       GLN,GLU,GLY,HIS,ILE          6-10             24-27 (No gly)
  !       LEU,LYS,MET,PHE,PRO         11-15             28-31 (No pro)
  !       SER,THR,TRP,TYR,VAL         16-20             32-33 (No ser,thr,val)
  !
  !                              phi/psi  ALA
  NZONES(1)   = 5
  P(1,1)      = 0.54
  MEAN1(1,1)  = -63.2
  MEAN2(1,1)  = -38.5
  SIGMA1(1,1) = 9.6
  SIGMA2(1,1) = 10.2
  P(2,1)      = 0.31
  MEAN1(2,1)  = -107.8
  MEAN2(2,1)  = 144.4
  SIGMA1(2,1) = 35.6
  SIGMA2(2,1) = 16.6
  P(3,1)      = 0.08
  MEAN1(3,1)  = -92.6
  MEAN2(3,1)  = -5.1
  SIGMA1(3,1) = 18.1
  SIGMA2(3,1) = 14.0
  P(4,1)      = 0.03
  MEAN1(4,1)  = -108.7
  MEAN2(4,1)  = 72.1
  SIGMA1(4,1) = 31.7
  SIGMA2(4,1) = 19.7
  P(5,1)      = 0.02
  MEAN1(5,1)  = 54.1
  MEAN2(5,1)  = 43.9
  SIGMA1(5,1) = 15.7
  SIGMA2(5,1) = 19.7
  !                                       ARG
  NZONES(2)   = 5
  P(1,2)      = 0.50
  MEAN1(1,2)  = -64.0
  MEAN2(1,2)  = -40.4
  SIGMA1(1,2) = 9.1
  SIGMA2(1,2) = 10.6
  P(2,2)      = 0.33
  MEAN1(2,2)  = -111.4
  MEAN2(2,2)  = 142.2
  SIGMA1(2,2) = 29.1
  SIGMA2(2,2) = 17.9
  P(3,2)      = 0.10
  MEAN1(3,2)  = -98.7
  MEAN2(3,2)  = -5.8
  SIGMA1(3,2) = 17.8
  SIGMA2(3,2) = 16.1
  P(4,2)      = 0.04
  MEAN1(4,2)  = -119.0
  MEAN2(4,2)  = 72.0
  SIGMA1(4,2) = 25.5
  SIGMA2(4,2) = 21.0
  P(5,2)      = 0.02
  MEAN1(5,2)  = 61.3
  MEAN2(5,2)  = 35.3
  SIGMA1(5,2) = 8.7
  SIGMA2(5,2) = 16.7
  !                                       ASN
  NZONES(3)   = 5
  P(1,3)      = 0.29
  MEAN1(1,3)  = -108.6
  MEAN2(1,3)  = 140.0
  SIGMA1(1,3) = 30.1
  SIGMA2(1,3) = 24.4
  P(2,3)      = 0.29
  MEAN1(2,3)  = -65.0
  MEAN2(2,3)  = -38.6
  SIGMA1(2,3) = 11.1
  SIGMA2(2,3) = 13.0
  P(3,3)      = 0.19
  MEAN1(3,3)  = -100.4
  MEAN2(3,3)  = 3.2
  SIGMA1(3,3) = 19.3
  SIGMA2(3,3) = 16.4
  P(4,3)      = 0.11
  MEAN1(4,3)  = -112.8
  MEAN2(4,3)  = 71.2
  SIGMA1(4,3) = 25.9
  SIGMA2(4,3) = 19.6
  P(5,3)      = 0.11
  MEAN1(5,3)  = 55.7
  MEAN2(5,3)  = 40.1
  SIGMA1(5,3) = 11.0
  SIGMA2(5,3) = 15.5
  !                                       ASP
  NZONES(4)   = 5
  P(1,4)      = 0.38
  MEAN1(1,4)  = -65.6
  MEAN2(1,4)  = -38.5
  SIGMA1(1,4) = 11.6
  SIGMA2(1,4) = 12.2
  P(2,4)      = 0.32
  MEAN1(2,4)  = -98.4
  MEAN2(2,4)  = 138.1
  SIGMA1(2,4) = 31.4
  SIGMA2(2,4) = 24.4
  P(3,4)      = 0.16
  MEAN1(3,4)  = -98.6
  MEAN2(3,4)  = -1.1
  SIGMA1(3,4) = 18.0
  SIGMA2(3,4) = 16.7
  P(4,4)      = 0.07
  MEAN1(4,4)  = -105.2
  MEAN2(4,4)  = 74.2
  SIGMA1(4,4) = 27.4
  SIGMA2(4,4) = 21.6
  P(5,4)      = 0.05
  MEAN1(5,4)  = 56.0
  MEAN2(5,4)  = 42.6
  SIGMA1(5,4) = 11.7
  SIGMA2(5,4) = 17.8
  !                                       CYS
  NZONES(5)   = 5
  P(1,5)      = 0.49
  MEAN1(1,5)  = -108.4
  MEAN2(1,5)  = 138.7
  SIGMA1(1,5) = 32.3
  SIGMA2(1,5) = 19.0
  P(2,5)      = 0.31
  MEAN1(2,5)  = -63.2
  MEAN2(2,5)  = -38.3
  SIGMA1(2,5) = 11.3
  SIGMA2(2,5) = 10.5
  P(3,5)      = 0.11
  MEAN1(3,5)  = -99.5
  MEAN2(3,5)  = -8.2
  SIGMA1(3,5) = 20.5
  SIGMA2(3,5) = 20.5
  P(4,5)      = 0.05
  MEAN1(4,5)  = -122.1
  MEAN2(4,5)  = 80.8
  SIGMA1(4,5) = 23.8
  SIGMA2(4,5) = 17.1
  P(5,5)      = 0.02
  MEAN1(5,5)  = 60.2
  MEAN2(5,5)  = 37.7
  SIGMA1(5,5) = 10.3
  SIGMA2(5,5) = 19.7
  !                                       GLN
  NZONES(6)   = 5
  P(1,6)      = 0.48
  MEAN1(1,6)  = -64.2
  MEAN2(1,6)  = -38.7
  SIGMA1(1,6) = 9.4
  SIGMA2(1,6) = 9.6
  P(2,6)      = 0.35
  MEAN1(2,6)  = -105.5
  MEAN2(2,6)  = 140.2
  SIGMA1(2,6) = 29.9
  SIGMA2(2,6) = 17.9
  P(3,6)      = 0.10
  MEAN1(3,6)  = -98.6
  MEAN2(3,6)  = -5.8
  SIGMA1(3,6) = 17.6
  SIGMA2(3,6) = 17.1
  P(4,6)      = 0.03
  MEAN1(4,6)  = 57.6
  MEAN2(4,6)  = 38.7
  SIGMA1(4,6) = 12.0
  SIGMA2(4,6) = 20.1
  P(5,6)      = 0.02
  MEAN1(5,6)  = -113.2
  MEAN2(5,6)  = 75.2
  SIGMA1(5,6) = 28.6
  SIGMA2(5,6) = 13.9
  !                                       GLU
  NZONES(7)   = 5
  P(1,7)      = 0.55
  MEAN1(1,7)  = -64.7
  MEAN2(1,7)  = -38.7
  SIGMA1(1,7) = 9.5
  SIGMA2(1,7) = 10.4
  P(2,7)      = 0.29
  MEAN1(2,7)  = -105.5
  MEAN2(2,7)  = 137.7
  SIGMA1(2,7) = 29.0
  SIGMA2(2,7) = 17.1
  P(3,7)      = 0.11
  MEAN1(3,7)  = -96.5
  MEAN2(3,7)  = -7.6
  SIGMA1(3,7) = 18.6
  SIGMA2(3,7) = 15.2
  P(4,7)      = 0.02
  MEAN1(4,7)  = -106.7
  MEAN2(4,7)  = 71.7
  SIGMA1(4,7) = 25.4
  SIGMA2(4,7) = 18.9
  P(5,7)      = 0.02
  MEAN1(5,7)  = 60.1
  MEAN2(5,7)  = 37.3
  SIGMA1(5,7) = 10.6
  SIGMA2(5,7) = 21.8
  !                                       GLY
  NZONES(8) = 5
  P(1,8)      = 0.41
  MEAN1(1,8)  = -184.1
  MEAN2(1,8)  = 178.1
  SIGMA1(1,8) = 77.9
  SIGMA2(1,8) = 30.3
  P(2,8)      = 0.22
  MEAN1(2,8)  = 92.5
  MEAN2(2,8)  = 0.2
  SIGMA1(2,8) = 14.7
  SIGMA2(2,8) = 13.9
  P(3,8)      = 0.17
  MEAN1(3,8)  = -62.8
  MEAN2(3,8)  = -39.8
  SIGMA1(3,8) = 10.6
  SIGMA2(3,8) = 12.8
  P(4,8)      = 0.11
  MEAN1(4,8)  = 68.6
  MEAN2(4,8)  = 31.4
  SIGMA1(4,8) = 12.0
  SIGMA2(4,8) = 13.9
  P(5,8)      = 0.05
  MEAN1(5,8)  = -99.8
  MEAN2(5,8)  = -3.4
  SIGMA1(5,8) = 20.7
  SIGMA2(5,8) = 19.5
  !                                       HIS
  NZONES(9) = 5
  P(1,9)      = 0.38
  MEAN1(1,9)  = -112.3
  MEAN2(1,9)  = 144.1
  SIGMA1(1,9) = 32.4
  SIGMA2(1,9) = 20.0
  P(2,9)      = 0.31
  MEAN1(2,9)  = -65.0
  MEAN2(2,9)  = -39.7
  SIGMA1(2,9) = 9.7
  SIGMA2(2,9) = 11.3
  P(3,9)      = 0.18
  MEAN1(3,9)  = -99.3
  MEAN2(3,9)  = -2.0
  SIGMA1(3,9) = 18.2
  SIGMA2(3,9) = 15.6
  P(4,9)      = 0.07
  MEAN1(4,9)  = -122.2
  MEAN2(4,9)  = 64.9
  SIGMA1(4,9) = 19.6
  SIGMA2(4,9) = 18.5
  P(5,9)      = 0.05
  MEAN1(5,9)  = 58.3
  MEAN2(5,9)  = 43.1
  SIGMA1(5,9) = 9.3
  SIGMA2(5,9) = 15.2
  !                                       ILE
  NZONES(10) = 5
  P(1,10)      = 0.52
  MEAN1(1,10)  = -109.2
  MEAN2(1,10)  = 132.2
  SIGMA1(1,10) = 22.1
  SIGMA2(1,10) = 15.8
  P(2,10)      = 0.39
  MEAN1(2,10)  = -65.8
  MEAN2(2,10)  = -42.4
  SIGMA1(2,10) = 10.8
  SIGMA2(2,10) = 10.0
  P(3,10)      = 0.07
  MEAN1(3,10)  = -101.8
  MEAN2(3,10)  = -9.6
  SIGMA1(3,10) = 15.5
  SIGMA2(3,10) = 19.2
  P(4,10)      = 0.02
  MEAN1(4,10)  = -117.4
  MEAN2(4,10)  = 78.8
  SIGMA1(4,10) = 16.2
  SIGMA2(4,10) = 21.5
  P(5,10)      = 0.01
  MEAN1(5,10)  = 41.7
  MEAN2(5,10)  = 46.0
  SIGMA1(5,10) = 20.1
  SIGMA2(5,10) = 9.4
  !                                       LEU
  NZONES(11) = 5
  P(1,11)      = 0.48
  MEAN1(1,11)  = -64.6
  MEAN2(1,11)  = -40.0
  SIGMA1(1,11) = 9.0
  SIGMA2(1,11) = 10.1
  P(2,11)      = 0.38
  MEAN1(2,11)  = -101.5
  MEAN2(2,11)  = 136.6
  SIGMA1(2,11) = 26.4
  SIGMA2(2,11) = 16.2
  P(3,11)      = 0.09
  MEAN1(3,11)  = -95.2
  MEAN2(3,11)  = -7.4
  SIGMA1(3,11) = 15.4
  SIGMA2(3,11) = 16.5
  P(4,11)      = 0.03
  MEAN1(4,11)  = -107.7
  MEAN2(4,11)  = 76.3
  SIGMA1(4,11) = 23.3
  SIGMA2(4,11) = 19.9
  P(5,11)      = 0.01
  MEAN1(5,11)  = 58.0
  MEAN2(5,11)  = 37.7
  SIGMA1(5,11) = 9.8
  SIGMA2(5,11) = 20.5
  !                                       LYS
  NZONES(12) = 5
  P(1,12)      = 0.46
  MEAN1(1,12)  = -63.7
  MEAN2(1,12)  = -39.1
  SIGMA1(1,12) = 10.1
  SIGMA2(1,12) = 10.6
  P(2,12)      = 0.35
  MEAN1(2,12)  = -105.2
  MEAN2(2,12)  = 140.0
  SIGMA1(2,12) = 30.4
  SIGMA2(2,12) = 17.6
  P(3,12)      = 0.12
  MEAN1(3,12)  = -98.3
  MEAN2(3,12)  = -8.9
  SIGMA1(3,12) = 17.6
  SIGMA2(3,12) = 16.5
  P(4,12)      = 0.03
  MEAN1(4,12)  = 55.2
  MEAN2(4,12)  = 41.7
  SIGMA1(4,12) = 9.1
  SIGMA2(4,12) = 13.5
  P(5,12)      = 0.02
  MEAN1(5,12)  = -108.1
  MEAN2(5,12)  = 72.4
  SIGMA1(5,12) = 23.9
  SIGMA2(5,12) = 19.8
  !                                       MET
  NZONES(13) = 5
  P(1,13)      = 0.52
  MEAN1(1,13)  = -65.5
  MEAN2(1,13)  = -39.4
  SIGMA1(1,13) = 8.6
  SIGMA2(1,13) = 9.9
  P(2,13)      = 0.35
  MEAN1(2,13)  = -113.3
  MEAN2(2,13)  = 141.2
  SIGMA1(2,13) = 28.8
  SIGMA2(2,13) = 17.3
  P(3,13)      = 0.07
  MEAN1(3,13)  = -93.1
  MEAN2(3,13)  = -3.0
  SIGMA1(3,13) = 13.1
  SIGMA2(3,13) = 15.4
  P(4,13)      = 0.04
  MEAN1(4,13)  = -94.6
  MEAN2(4,13)  = 76.6
  SIGMA1(4,13) = 21.5
  SIGMA2(4,13) = 16.3
  P(5,13)      = 0.02
  MEAN1(5,13)  = 54.2
  MEAN2(5,13)  = 40.2
  SIGMA1(5,13) = 12.6
  SIGMA2(5,13) = 22.9
  !                                       PHE
  NZONES(14) = 5
  P(1,14)      = 0.46
  MEAN1(1,14)  = -110.4
  MEAN2(1,14)  = 141.4
  SIGMA1(1,14) = 29.8
  SIGMA2(1,14) = 17.7
  P(2,14)      = 0.35
  MEAN1(2,14)  = -62.8
  MEAN2(2,14)  = -42.5
  SIGMA1(2,14) = 9.5
  SIGMA2(2,14) = 10.8
  P(3,14)      = 0.12
  MEAN1(3,14)  = -102.3
  MEAN2(3,14)  = -4.4
  SIGMA1(3,14) = 16.4
  SIGMA2(3,14) = 17.1
  P(4,14)      = 0.05
  MEAN1(4,14)  = -116.2
  MEAN2(4,14)  = 75.5
  SIGMA1(4,14) = 21.9
  SIGMA2(4,14) = 19.0
  P(5,14)      = 0.01
  MEAN1(5,14)  = 65.1
  MEAN2(5,14)  = 29.6
  SIGMA1(5,14) = 9.7
  SIGMA2(5,14) = 10.2
  !                                       PRO
  NZONES(15) = 2
  P(1,15)      = 0.51
  MEAN1(1,15)  = -66.5
  MEAN2(1,15)  = 146.4
  SIGMA1(1,15) = 10.4
  SIGMA2(1,15) = 14.8
  P(2,15)      = 0.44
  MEAN1(2,15)  = -62.6
  MEAN2(2,15)  = -27.4
  SIGMA1(2,15) = 12.2
  SIGMA2(2,15) = 15.5
  !                                       SER
  NZONES(16) = 5
  P(1,16)      = 0.42
  MEAN1(1,16)  = -107.8
  MEAN2(1,16)  = 148.8
  SIGMA1(1,16) = 34.1
  SIGMA2(1,16) = 17.8
  P(2,16)      = 0.35
  MEAN1(2,16)  = -64.9
  MEAN2(2,16)  = -36.8
  SIGMA1(2,16) = 11.8
  SIGMA2(2,16) = 13.2
  P(3,16)      = 0.15
  MEAN1(3,16)  = -96.9
  MEAN2(3,16)  = -4.7
  SIGMA1(3,16) = 19.1
  SIGMA2(3,16) = 16.2
  P(4,16)      = 0.03
  MEAN1(4,16)  = -123.7
  MEAN2(4,16)  = 71.7
  SIGMA1(4,16) = 28.2
  SIGMA2(4,16) = 22.1
  P(5,16)      = 0.01
  MEAN1(5,16)  = 58.2
  MEAN2(5,16)  = 37.4
  SIGMA1(5,16) = 12.6
  SIGMA2(5,16) = 20.8
  !                                       THR
  NZONES(17) = 5
  P(1,17)      = 0.49
  MEAN1(1,17)  = -111.7
  MEAN2(1,17)  = 145.2
  SIGMA1(1,17) = 25.9
  SIGMA2(1,17) = 19.8
  P(2,17)      = 0.31
  MEAN1(2,17)  = -66.2
  MEAN2(2,17)  = -40.2
  SIGMA1(2,17) = 12.6
  SIGMA2(2,17) = 11.9
  P(3,17)      = 0.15
  MEAN1(3,17)  = -103.9
  MEAN2(3,17)  = -6.1
  SIGMA1(3,17) = 17.7
  SIGMA2(3,17) = 16.4
  P(4,17)      = 0.02
  MEAN1(4,17)  = -121.7
  MEAN2(4,17)  = 56.4
  SIGMA1(4,17) = 17.8
  SIGMA2(4,17) = 20.6
  P(5,17)      = 0.01
  MEAN1(5,17)  = 48.3
  MEAN2(5,17)  = 35.9
  SIGMA1(5,17) = 14.3
  SIGMA2(5,17) = 32.2
  !                                       TRP
  NZONES(18) = 5
  P(1,18)      = 0.43
  MEAN1(1,18)  = -105.8
  MEAN2(1,18)  = 139.6
  SIGMA1(1,18) = 29.8
  SIGMA2(1,18) = 18.9
  P(2,18)      = 0.42
  MEAN1(2,18)  = -64.0
  MEAN2(2,18)  = -40.8
  SIGMA1(2,18) = 11.1
  SIGMA2(2,18) = 10.7
  P(3,18)      = 0.11
  MEAN1(3,18)  = -100.1
  MEAN2(3,18)  = -3.4
  SIGMA1(3,18) = 18.7
  SIGMA2(3,18) = 20.2
  P(4,18)      = 0.03
  MEAN1(4,18)  = -96.0
  MEAN2(4,18)  = 70.7
  SIGMA1(4,18) = 17.2
  SIGMA2(4,18) = 17.6
  P(5,18)      = 0.02
  MEAN1(5,18)  = 63.5
  MEAN2(5,18)  = 28.8
  SIGMA1(5,18) = 9.3
  SIGMA2(5,18) = 12.8
  !                                       TYR
  NZONES(19) = 5
  P(1,19)      = 0.48
  MEAN1(1,19)  = -114.0
  MEAN2(1,19)  = 142.5
  SIGMA1(1,19) = 29.1
  SIGMA2(1,19) = 18.0
  P(2,19)      = 0.33
  MEAN1(2,19)  = -63.5
  MEAN2(2,19)  = -42.3
  SIGMA1(2,19) = 9.6
  SIGMA2(2,19) = 10.4
  P(3,19)      = 0.12
  MEAN1(3,19)  = -103.0
  MEAN2(3,19)  = -2.8
  SIGMA1(3,19) = 16.8
  SIGMA2(3,19) = 16.2
  P(4,19)      = 0.03
  MEAN1(4,19)  = -114.9
  MEAN2(4,19)  = 78.6
  SIGMA1(4,19) = 21.9
  SIGMA2(4,19) = 14.8
  P(5,19)      = 0.03
  MEAN1(5,19)  = 61.8
  MEAN2(5,19)  = 32.9
  SIGMA1(5,19) = 11.6
  SIGMA2(5,19) = 16.7
  !                                       VAL
  NZONES(20) = 5
  P(1,20)      = 0.55
  MEAN1(1,20)  = -112.7
  MEAN2(1,20)  = 135.5
  SIGMA1(1,20) = 23.4
  SIGMA2(1,20) = 16.4
  P(2,20)      = 0.36
  MEAN1(2,20)  = -65.0
  MEAN2(2,20)  = -41.9
  SIGMA1(2,20) = 9.4
  SIGMA2(2,20) = 9.9
  P(3,20)      = 0.06
  MEAN1(3,20)  = -106.5
  MEAN2(3,20)  = -11.0
  SIGMA1(3,20) = 18.5
  SIGMA2(3,20) = 19.9
  P(4,20)      = 0.02
  MEAN1(4,20)  = -108.1
  MEAN2(4,20)  = 81.5
  SIGMA1(4,20) = 24.1
  SIGMA2(4,20) = 19.9
  P(5,20)      = 0.01
  MEAN1(5,20)  = 36.0
  MEAN2(5,20)  = 42.0
  SIGMA1(5,20) = 21.5
  SIGMA2(5,20) = 17.3
  !                            chi1/chi2  ARG
  NZONES(21) = 5
  P(1,21)      = 0.43
  MEAN1(1,21)  = -67.5
  MEAN2(1,21)  = -176.6
  SIGMA1(1,21) = 15.4
  SIGMA2(1,21) = 20.1
  P(2,21)      = 0.24
  MEAN1(2,21)  = -174.3
  MEAN2(2,21)  = 179.9
  SIGMA1(2,21) = 17.1
  SIGMA2(2,21) = 18.3
  P(3,21)      = 0.12
  MEAN1(3,21)  = -63.7
  MEAN2(3,21)  = -73.5
  SIGMA1(3,21) = 18.1
  SIGMA2(3,21) = 20.7
  P(4,21)      = 0.09
  MEAN1(4,21)  = 63.7
  MEAN2(4,21)  = 174.8
  SIGMA1(4,21) = 17.2
  SIGMA2(4,21) = 21.1
  P(5,21)      = 0.07
  MEAN1(5,21)  = -173.9
  MEAN2(5,21)  = 71.9
  SIGMA1(5,21) = 20.5
  SIGMA2(5,21) = 18.7
  !                                       ASN
  NZONES(22) = 4
  P(1,22)      = 0.33
  MEAN1(1,22)  = -70.2
  MEAN2(1,22)  = -40.2
  SIGMA1(1,22) = 14.4
  SIGMA2(1,22) = 32.4
  P(2,22)      = 0.29
  MEAN1(2,22)  = -169.7
  MEAN2(2,22)  = -39.0
  SIGMA1(2,22) = 16.9
  SIGMA2(2,22) = 88.7
  P(3,22)      = 0.21
  MEAN1(3,22)  = -69.3
  MEAN2(3,22)  = 136.1
  SIGMA1(3,22) = 16.6
  SIGMA2(3,22) = 38.8
  P(4,22)      = 0.16
  MEAN1(4,22)  = 62.9
  MEAN2(4,22)  = -30.1
  SIGMA1(4,22) = 13.0
  SIGMA2(4,22) = 81.5
  !                                       ASP
  NZONES(23) = 3
  P(1,23)      = 0.51
  MEAN1(1,23)  = -70.1
  MEAN2(1,23)  = 159.1
  SIGMA1(1,23) = 14.2
  SIGMA2(1,23) = 33.5
  P(2,23)      = 0.31
  MEAN1(2,23)  = -170.9
  MEAN2(2,23)  = -173.4
  SIGMA1(2,23) = 16.0
  SIGMA2(2,23) = 42.7
  P(3,23)      = 0.18
  MEAN1(3,23)  = 62.5
  MEAN2(3,23)  = 174.9
  SIGMA1(3,23) = 13.7
  SIGMA2(3,23) = 40.0
  !                                       GLN
  NZONES(24) = 6
  P(1,24)      = 0.36
  MEAN1(1,24)  = -67.7
  MEAN2(1,24)  = 179.9
  SIGMA1(1,24) = 15.3
  SIGMA2(1,24) = 17.1
  P(2,24)      = 0.21
  MEAN1(2,24)  = -173.0
  MEAN2(2,24)  =  178.6
  SIGMA1(2,24) = 18.1
  SIGMA2(2,24) = 19.0
  P(3,24)      = 0.15
  MEAN1(3,24)  = -64.8
  MEAN2(3,24)  = -66.8
  SIGMA1(3,24) = 15.8
  SIGMA2(3,24) = 19.5
  P(4,24)      = 0.11
  MEAN1(4,24)  = -172.8
  MEAN2(4,24)  = 68.4
  SIGMA1(4,24) = 21.2
  SIGMA2(4,24) = 16.8
  P(5,24)      = 0.07
  MEAN1(5,24)  = 64.3
  MEAN2(5,24)  = -179.2
  SIGMA1(5,24) = 20.6
  SIGMA2(5,24) = 20.6
  P(6,24)      = 0.05
  MEAN1(6,24)  = -73.6
  MEAN2(6,24)  = 74.1
  SIGMA1(6,24) = 22.8
  SIGMA2(6,24) = 19.9
  !                                       GLU
  NZONES(25) = 7
  P(1,25)      = 0.35
  MEAN1(1,25)  = -67.5
  MEAN2(1,25)  = 179.3
  SIGMA1(1,25) = 16.3
  SIGMA2(1,25) = 18.4
  P(2,25)      = 0.24
  MEAN1(2,25)  = -174.1
  MEAN2(2,25)  = -179.9
  SIGMA1(2,25) = 19.7
  SIGMA2(2,25) = 18.9
  P(3,25)      = 0.15
  MEAN1(3,25)  = -66.6
  MEAN2(3,25)  = -66.4
  SIGMA1(3,25) = 20.6
  SIGMA2(3,25) = 20.2
  P(4,25)      = 0.07
  MEAN1(4,25)  = 59.8
  MEAN2(4,25)  = -178.1
  SIGMA1(4,25) = 23.3
  SIGMA2(4,25) = 21.5
  P(5,25)      = 0.07
  MEAN1(5,25)  = -66.8
  MEAN2(5,25)  = 76.7
  SIGMA1(5,25) = 21.7
  SIGMA2(5,25) = 18.1
  P(6,25)      = 0.06
  MEAN1(6,25)  = -166.5
  MEAN2(6,25)  = 65.9
  SIGMA1(6,25) = 20.4
  SIGMA2(6,25) = 18.2
  P(7,25)      = 0.03
  MEAN1(7,25)  = 55.1
  MEAN2(7,25)  = -82.6
  SIGMA1(7,25) = 22.6
  SIGMA2(7,25) = 15.9
  !                                       HIS
  NZONES(26) = 6
  P(1,26)      = 0.30
  MEAN1(1,26)  = -65.1
  MEAN2(1,26)  = -82.0
  SIGMA1(1,26) = 14.3
  SIGMA2(1,26) = 31.0
  P(2,26)      = 0.24
  MEAN1(2,26)  = -65.8
  MEAN2(2,26)  = 109.8
  SIGMA1(2,26) = 12.1
  SIGMA2(2,26) = 34.8
  P(3,26)      = 0.19
  MEAN1(3,26)  = -176.5
  MEAN2(3,26)  = 78.3
  SIGMA1(3,26) = 11.9
  SIGMA2(3,26) = 31.3
  P(4,26)      = 0.15
  MEAN1(4,26)  = -170.1
  MEAN2(4,26)  = -101.5
  SIGMA1(4,26) = 13.9
  SIGMA2(4,26) = 33.6
  P(5,26)      = 0.07
  MEAN1(5,26)  = 62.7
  MEAN2(5,26)  = -86.9
  SIGMA1(5,26) = 12.0
  SIGMA2(5,26) = 20.2
  P(6,26)      = 0.05
  MEAN1(6,26)  = 58.7
  MEAN2(6,26)  = 96.1
  SIGMA1(6,26) = 15.0
  SIGMA2(6,26) = 31.9
  !                                       ILE
  NZONES(27) = 6
  P(1,27)      = 0.58
  MEAN1(1,27)  = -64.2
  MEAN2(1,27)  = 168.4
  SIGMA1(1,27) = 10.1
  SIGMA2(1,27) = 14.4
  P(2,27)      = 0.13
  MEAN1(2,27)  = 62.2
  MEAN2(2,27)  = 169.0
  SIGMA1(2,27) = 12.6
  SIGMA2(2,27) = 14.4
  P(3,27)      = 0.13
  MEAN1(3,27)  = -57.6
  MEAN2(3,27)  = -62.4
  SIGMA1(3,27) = 10.6
  SIGMA2(3,27) = 15.4
  P(4,27)      = .07
  MEAN1(4,27)  = -175.3
  MEAN2(4,27)  = 167.7
  SIGMA1(4,27) = 20.6
  SIGMA2(4,27) = 16.0
  P(5,27)      = .04
  MEAN1(5,27)  = -70.4
  MEAN2(5,27)  = 72.8
  SIGMA1(5,27) = 20.3
  SIGMA2(5,27) = 31.3
  P(6,27)      = .03
  MEAN1(6,27)  = -165.3
  MEAN2(6,27)  = 71.4
  SIGMA1(6,27) = 26.5
  SIGMA2(6,27) = 16.3
  !                                       LEU
  NZONES(28) = 5
  P(1,28)      = 0.51
  MEAN1(1,28)  = -66.3
  MEAN2(1,28)  = 175.5
  SIGMA1(1,28) = 12.2
  SIGMA2(1,28) = 12.9
  P(2,28)      = 0.26
  MEAN1(2,28)  = -177.9
  MEAN2(2,28)  = 65.5
  SIGMA1(2,28) = 11.9
  SIGMA2(2,28) = 12.9
  P(3,28)      = .10
  MEAN1(3,28)  = -99.6
  MEAN2(3,28)  = 44.3
  SIGMA1(3,28) = 19.3
  SIGMA2(3,28) = 25.3
  P(4,28)      = .06
  MEAN1(4,28)  = -159.8
  MEAN2(4,28)  = -179.3
  SIGMA1(4,28) = 19.0
  SIGMA2(4,28) = 30.6
  P(5,28)      = .05
  MEAN1(5,28)  = -104.4
  MEAN2(5,28)  = -56.5
  SIGMA1(5,28) = 40.3
  SIGMA2(5,28) = 30.0
  !                                       LYS
  NZONES(29) = 6
  P(1,29)      = 0.39
  MEAN1(1,29)  = -68.9
  MEAN2(1,29)  = -178.3
  SIGMA1(1,29) = 16.8
  SIGMA2(1,29) = 21.4
  P(2,29)      = 0.26
  MEAN1(2,29)  = -173.3
  MEAN2(2,29)  = 178.6
  SIGMA1(2,29) = 17.4
  SIGMA2(2,29) = 21.9
  P(3,29)      = 0.12
  MEAN1(3,29)  = -64.1
  MEAN2(3,29)  = -71.1
  SIGMA1(3,29) = 17.0
  SIGMA2(3,29) = 23.3
  P(4,29)      = 0.07
  MEAN1(4,29)  = 62.4
  MEAN2(4,29)  = -179.9
  SIGMA1(4,29) = 18.8
  SIGMA2(4,29) = 21.9
  P(5,29)      = 0.07
  MEAN1(5,29)  = -174.3
  MEAN2(5,29)  = 76.6
  SIGMA1(5,29) = 20.4
  SIGMA2(5,29) = 19.2
  P(6,29)      = 0.04
  MEAN1(6,29)  = -87.6
  MEAN2(6,29)  = 75.6
  SIGMA1(6,29) = 19.7
  SIGMA2(6,29) = 27.8
  !                                       MET
  NZONES(30) = 5
  P(1,30)      = 0.34
  MEAN1(1,30)  = -69.9
  MEAN2(1,30)  = -178.0
  SIGMA1(1,30) = 13.6
  SIGMA2(1,30) = 15.9
  P(2,30)      = 0.24
  MEAN1(2,30)  = -64.0
  MEAN2(2,30)  = -66.5
  SIGMA1(2,30) = 11.9
  SIGMA2(2,30) = 16.4
  P(3,30)      = 0.19
  MEAN1(3,30)  = -173.6
  MEAN2(3,30)  = 178.0
  SIGMA1(3,30) = 17.1
  SIGMA2(3,30) = 17.8
  P(4,30)      = 0.09
  MEAN1(4,30)  = -170.7
  MEAN2(4,30)  = 76.1
  SIGMA1(4,30) = 13.4
  SIGMA2(4,30) = 18.7
  P(5,30)      = 0.08
  MEAN1(5,30)  = 62.7
  MEAN2(5,30)  = -175.2
  SIGMA1(5,30) = 17.0
  SIGMA2(5,30) = 17.7
  !                                       PHE
  NZONES(31) = 3
  P(1,31)      = 0.52
  MEAN1(1,31)  = -66.8
  MEAN2(1,31)  = 98.7
  SIGMA1(1,31) = 11.9
  SIGMA2(1,31) = 30.0
  P(2,31)      = 0.34
  MEAN1(2,31)  = -177.4
  MEAN2(2,31)  = 76.9
  SIGMA1(2,31) = 12.4
  SIGMA2(2,31) = 19.0
  P(3,31)      = 0.13
  MEAN1(3,31)  = 63.1
  MEAN2(3,31)  = 91.1
  SIGMA1(3,31) = 11.6
  SIGMA2(3,31) = 13.2
  !                                       TRP
  NZONES(32) = 6
  P(1,32)      = 0.37
  MEAN1(1,32)  = -67.0
  MEAN2(1,32)  = 98.4
  SIGMA1(1,32) = 11.4
  SIGMA2(1,32) = 15.9
  P(2,32)      = 0.20
  MEAN1(2,32)  = -179.4
  MEAN2(2,32)  = 71.4
  SIGMA1(2,32) = 11.4
  SIGMA2(2,32) = 24.8
  P(3,32)      = 0.16
  MEAN1(3,32)  = -69.0
  MEAN2(3,32)  = -38.5
  SIGMA1(3,32) = 13.2
  SIGMA2(3,32) = 47.5
  P(4,32)      = 0.12
  MEAN1(4,32)  = 179.6
  MEAN2(4,32)  = -101.1
  SIGMA1(4,32) = 14.4
  SIGMA2(4,32) = 14.0
  P(5,32)      = 0.10
  MEAN1(5,32)  = 62.0
  MEAN2(5,32)  = -87.9
  SIGMA1(5,32) = 12.6
  SIGMA2(5,32) = 9.2
  P(6,32)      = 0.05
  MEAN1(6,32)  = 61.0
  MEAN2(6,32)  = 84.3
  SIGMA1(6,32) = 14.5
  SIGMA2(6,32) = 13.9
  !                                       TYR
  NZONES(33) = 3
  P(1,33)      = 0.53
  MEAN1(1,33)  = -66.1
  MEAN2(1,33)  = 99.8
  SIGMA1(1,33) = 11.9
  SIGMA2(1,33) = 26.6
  P(2,33)      = 0.35
  MEAN1(2,33)  = 179.7
  MEAN2(2,33)  = 76.3
  SIGMA1(2,33) = 11.8
  SIGMA2(2,33) = 20.6
  P(3,33)      = 0.12
  MEAN1(3,33)  = 64.3
  MEAN2(3,33)  = 87.4
  SIGMA1(3,33) = 12.3
  SIGMA2(3,33) = 15.7
  !                            chi1 only  CYS
  NZONES(34) = 3
  P(1,34)      = 0.16
  MEAN1(1,34)  = 63.1
  SIGMA1(1,34) = 18.5
  P(2,34)      = 0.30
  MEAN1(2,34)  = -177.1
  SIGMA1(2,34) = 12.9
  P(3,34)      = 0.54
  MEAN1(3,34)  = -64.8
  SIGMA1(3,34) = 13.7
  !                                       SER
  NZONES(35) = 3
  P(1,35)      = 0.43
  MEAN1(1,35)  = 63.6
  SIGMA1(1,35) = 16.1
  P(2,35)      = 0.24
  MEAN1(2,35)  = -179.0
  SIGMA1(2,35) = 19.9
  P(3,35)      = 0.33
  MEAN1(3,35)  = -64.8
  SIGMA1(3,35) = 18.7
  !                                       THR
  NZONES(36) = 3
  P(1,36)      = 0.42
  MEAN1(1,36)  = 62.7
  SIGMA1(1,36) = 13.4
  P(2,36)      = 0.10
  MEAN1(2,36)  = -179.2
  SIGMA1(2,36) = 25.3
  P(3,36)      = 0.48
  MEAN1(3,36)  = -60.5
  SIGMA1(3,36) = 14.2
  !                                       VAL
  NZONES(37) = 3
  P(1,37)      = 0.10
  MEAN1(1,37)  = 61.7
  SIGMA1(1,37) = 26.5
  P(2,37)      = 0.67
  MEAN1(2,37)  = 174.7
  SIGMA1(2,37) = 11.7
  P(3,37)      = 0.23
  MEAN1(3,37)  = -60.9
  SIGMA1(3,37) = 16.6
  !                            omega preceding PRO (standard devs. from John K.)
  NZONES(38) = 2
  P(1,38)      = 0.94
  MEAN1(1,38)  = 180.0
  SIGMA1(1,38) = 3.9
  P(2,38)      = 0.06
  MEAN1(2,38)  = 0.0
  SIGMA1(2,38) = 5.9
  !
  !       For Jth 2D-move type, determine bins for random
  !       selection of 1 of the NZONES 2D Gaussian zones.
  !
  DO J = 1,MAX2D
     SUMP(J) = 0.
     DO I = 1,NZONES(J)
        SUMP(J) = SUMP(J) + P(I,J)
        ZONBIN(I,J) = SUMP(J)
     enddo
     DO I = 1,NZONES(J)
        SIGMA1(I,J) = SIGSC*SIGMA1(I,J)
        SIGMA2(I,J) = SIGSC*SIGMA2(I,J)
        ZONBIN(I,J) = ZONBIN(I,J)/SUMP(J)
     enddo
  enddo
  !                  chi1 for CYS,SER,THR,VAL or pre-proline omega
  DO J = MAX2D+1,MAXTOR
     SUMP(J) = 0.
     DO I = 1,NZONES(J)
        SUMP(J) = SUMP(J) + P(I,J)
        ZONBIN(I,J) = SUMP(J)
     enddo
     DO I = 1,NZONES(J)
        SIGMA1(I,J) = SIGSC*SIGMA1(I,J)
        ZONBIN(I,J) = ZONBIN(I,J)/SUMP(J)
     enddo
  enddo
  RETURN
END SUBROUTINE GET2DG

SUBROUTINE CHEKIC
  !
  ! ***** Check that MC IC table is appropriate: every phi/chi1, should
  !  ***  be immediately followed by its corresponding  psi/chi2, if biased
  !   *   MC with 2D Gaussians is to be done.  Get BIN2D = 1/NMOVES, where NMOVES
  !       is the number of independent (1 or 2D) moves. Output data.
  !

  use chm_kinds
  use intcor_module
  use dimens_fcm
  use psf
  use number
  use stream
  use bases_fcm
  use chutil, only: getres

  implicit none
  INTEGER I,J,PSIRES,NUM0,NUMF
  LOGICAL*2 WARN1
  !
  ! First, for each entry in MC IC table, determine the TORsion TYPe
  ! and the pointer (MOV2D) to the correct column of the arrays
  ! P,MEAN1,SIGMA1,etc. used for biasing MC moves.  Also, for each IC
  ! find the SEGment NUMber, and the first and last atom numbers of the
  ! atoms to be moved: MVATM0,MVATMF.  ICPSI(PSIRES) is the # of IC entry
  ! that contains the psi torsion of residue # PSIRES.
  !
  DO I = 1,MAXRES
     ICPSI(I) = 0
  enddo
  IF (PRNLEV .GE. 2) WRITE(OUTU,45)
45 FORMAT(' IC #    Torsion Type   Move Index', &
       '    Atom#-RESId-TYPE of First,Last Atom Moved')
  do i = 1, icr_struct%lenic
     CALL IDENIC(I,icr_struct%IAR, icr_struct%JAR, &
          icr_struct%KAR, icr_struct%LAR, &
          TORTYP(I),MOV2D(I),MVATM0(I),MVATMF(I), &
          PSIRES,SEGNUM(I))
     IF(PSIRES .NE. 0) ICPSI(PSIRES) = I
  enddo
  !
  ! Next, check ICs are in appropriate order. MVINDX(n) is assigned the IC number
  ! of the torsion involved in Monte Carlo move n; 1 <= n <= NMOVES.  If move n
  ! involves a pair of torsions, MVINDX(n) is the first torsion.  Similarly,
  ! ICSIDE(I) is the # of IC entry that is the first IC of side-chain move # I,
  ! where 0 <= I <= NSIDE.
  !
  NMOVES = 0
  NSIDE = 0
  I = 0
  WARN1 = .FALSE.
  !
55 I = I+1
  J = MOD(TORTYP(I),10)
  NMOVES = NMOVES + 1
  MVINDX(NMOVES) = I
  NUM0 = MVATM0(I)
  NUMF = MVATMF(I)
  IF (J.EQ.1 .OR. (J.EQ.3 .AND. MOV2D(I).LE.MAX2D)) THEN
     !         Ith IC entry is a phi or chi1 angle.
     IF (I .LT. icr_struct%LENIC) THEN
        IF (MOV2D(I) .NE. MOV2D(I+1)) THEN
           MOV2D(I) = -1
           NSIDE = NSIDE+1
           ICSIDE(NSIDE) = I
           IF (PRNLEV .GE. 2) THEN
              WRITE(OUTU,61) I,I+1
              WRITE(OUTU,60) I,TORTXT(TORTYP(I)),MOV2D(I), &
                   NUM0,RESID(GETRES(NUM0,IBASE,NRES)),ATYPE(NUM0), &
                   NUMF,RESID(GETRES(NUMF,IBASE,NRES)),ATYPE(NUMF)
           END IF
           GO TO 55
        ELSE
           !             IC's I and I+1 are a phi/psi or chi1/chi2 pair.
           IF (PRNLEV .GE. 2) THEN
              WRITE(OUTU,60) I,TORTXT(TORTYP(I)),MOV2D(I), &
                   NUM0,RESID(GETRES(NUM0,IBASE,NRES)),ATYPE(NUM0), &
                   NUMF,RESID(GETRES(NUMF,IBASE,NRES)),ATYPE(NUMF)
              NUM0 = MVATM0(I+1)
              NUMF = MVATMF(I+1)
              WRITE(OUTU,60) I+1,TORTXT(TORTYP(I+1)),MOV2D(I+1), &
                   NUM0,RESID(GETRES(NUM0,IBASE,NRES)),ATYPE(NUM0), &
                   NUMF,RESID(GETRES(NUMF,IBASE,NRES)),ATYPE(NUMF)
           END IF
           IF(J .EQ. 3) THEN
              NSIDE = NSIDE+1
              ICSIDE(NSIDE) = I
           END IF
           I = I+1
           IF (I .LT. icr_struct%LENIC) GO TO 55
        END IF
     ELSE
        !                                                    I = LENIC
        IF(J .EQ. 3) THEN
           NSIDE = NSIDE+1
           ICSIDE(NSIDE) = I
        END IF
        MOV2D(I) = -1
        IF (PRNLEV .GE. 2) THEN
           WRITE(OUTU,63)
           WRITE(OUTU,60) I,TORTXT(TORTYP(I)),MOV2D(I), &
                NUM0,RESID(GETRES(NUM0,IBASE,NRES)),ATYPE(NUM0), &
                NUMF,RESID(GETRES(NUMF,IBASE,NRES)),ATYPE(NUMF)
        END IF
     END IF
  ELSE
     IF (J .GE. 5) THEN
        MOV2D(I) = -1
     ELSE IF (I.EQ.1 .AND. (J.EQ.2 .OR. J.EQ.4)) THEN
        MOV2D(I) = 100 + MOV2D(I)
        WARN1 = .TRUE.
     END IF
     IF(J .GE. 3) THEN
        NSIDE = NSIDE+1
        ICSIDE(NSIDE) = I
     END IF
     IF (PRNLEV .GE. 2) WRITE(OUTU,60) &
          I,TORTXT(TORTYP(I)),MOV2D(I), &
          NUM0,RESID(GETRES(NUM0,IBASE,NRES)),ATYPE(NUM0), &
          NUMF,RESID(GETRES(NUMF,IBASE,NRES)),ATYPE(NUMF)
     IF (I .LT. icr_struct%LENIC) GO TO 55
  END IF
  IF (PRNLEV.GE.2 .AND. WARN1) WRITE(OUTU,62)
60 FORMAT(I6,6X,A,I7,9X,2(I6,1X,A4,1X,A4))
61 FORMAT(' WARNING: IC Entries',I4,' and',I4,' are ', &
       'incompatible for 2D-Gaussian biased Monte Carlo.')
62 FORMAT(' WARNING: First IC Entry lacks partner entry ', &
       'appropriate for 2D-Gaussian biased Monte Carlo.')
63 FORMAT(' WARNING:  Last IC Entry lacks partner entry ', &
       'appropriate for 2D-Gaussian biased Monte Carlo.')
  !
  BIN2D = ONE/DFLOAT(NMOVES)
  IF (PRNLEV .GE. 2) WRITE(OUTU,64) NMOVES,NSIDE
64 FORMAT(/1X,I4,' independent MC moves exist in IC table;'/ &
       1X,I4,' independent MC side-chain moves exist', &
       ' in IC table.'/)
  !
  RETURN
END SUBROUTINE CHEKIC

SUBROUTINE NEWICS(ICNUM1,ICNUM2,PHI1,PHI2,PIC)
  !
  ! *****
  !  ***   NEWIC(S): Change value of dihedral angle(s) in IC table
  !   *
  !
  use chm_kinds
  use number
  implicit none
  INTEGER ICNUM1,ICNUM2
  real(chm_real) PHI1,PHI2,PIC(*)
  IF (PHI1.GT. ONE8TY) PHI1=PHI1-THR6TY
  IF (PHI1.LT.-ONE8TY) PHI1=PHI1+THR6TY
  IF (PHI2.GT. ONE8TY) PHI2=PHI2-THR6TY
  IF (PHI2.LT.-ONE8TY) PHI2=PHI2+THR6TY
  PIC(ICNUM1) = PHI1
  PIC(ICNUM2) = PHI2
  RETURN
END SUBROUTINE NEWICS

SUBROUTINE NEWIC(ICNUM1,PHI1,PIC)
  use chm_kinds
  use number
  implicit none
  INTEGER ICNUM1
  real(chm_real) PHI1,PIC(*)
  IF (PHI1.GT. ONE8TY) PHI1=PHI1-THR6TY
  IF (PHI1.LT.-ONE8TY) PHI1=PHI1+THR6TY
  PIC(ICNUM1) = PHI1
  RETURN
END SUBROUTINE NEWIC

SUBROUTINE IDENIC(ICNUM,IAR,JAR,KAR,LAR,TORTYP,MOV2D, &
     ATNUM0,ATNUMF,PSIRES,ISEG)
  !
  ! *****  Identify IC entry as a specific type of dihedral angle,
  !  ***   e.g., TORTYP=GLY phi, LEU chi1, etc.  All IC entries are
  !   *    assumed to be proper torsions involving only heavy atoms.
  !
  !      phi:  i-1 C  i N   i CA    i C, not defined for i=1 residue
  !      psi:    i N  i CA  i C   i+1 N,
  !                             or  i OT2  for i=last residue
  !     chi1:    i N  i CA  i CB  i *
  !     chi2:    i CA i CB  i *   i *
  !     chi3:    i CB i *   i *   i *
  !
  ! TORTYP Torsion TORTYP Torsion TORTYP Torsion TORTYP Torsion TORTYP Torsion
  !   11 Ala phi     12 Ala psi
  !   21 Arg phi     22 Arg psi     23 Arg chi1     24 Arg chi2  25-27 Arg chi3-5
  !   31 Asn phi     32 Asn psi     33 Asn chi1     34 Asn chi2
  !   41 Asp phi     42 Asp psi     43 Asp chi1     44 Asp chi2
  !   51 Cys phi     52 Cys psi     53 Cys chi1
  !   61 Gln phi     62 Gln psi     63 Gln chi1     64 Gln chi2    65  Gln chi3
  !   71 Glu phi     72 Glu psi     73 Glu chi1     74 Glu chi2    75  Glu chi3
  !   81 Gly phi     82 Gly psi
  !   91 His phi     92 His psi     93 His chi1     94 His chi2
  !  101 Ile phi    102 Ile psi    103 Ile chi1    104 Ile chi2
  !  111 Leu phi    112 Leu psi    103 Leu chi1    104 Leu chi2
  !  121 Lys phi    122 Lys psi    103 Lys chi1    104 Lys chi2  105,6 Lys chi3,4
  !  131 Met phi    132 Met psi    133 Met chi1    134 Met chi2   135  Met chi3
  !  141 Phe phi    142 Phe psi    143 Phe chi1    144 Phe chi2
  !  151 Pro phi    152 Pro psi    150 Omega angle preceding pro
  !  161 Ser phi    162 Ser psi    163 Ser chi1
  !  171 Thr phi    172 Thr psi    173 Thr chi1
  !  181 Trp phi    182 Trp psi    183 Trp chi1    184 Trp chi2
  !  191 Tyr phi    192 Tyr psi    193 Tyr chi1    194 Tyr chi2
  !  201 Val phi    202 Val psi    203 Val chi1
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use chutil, only: atomid, getres

  implicit none
  INTEGER ICNUM,IAR(*),JAR(*),KAR(*),LAR(*),TORTYP, &
       MOV2D,ATNUM0,ATNUMF,ATN0,ATNF,PSIRES,ISEG,KRES,LRES, &
       I,J,K,L,NATM
  CHARACTER(len=4) ISID,IRID,IREN,IAT,JSID,JRID,JREN,JAT, &
       KSID,KRID,KREN,KAT,LSID,LRID,LREN,LAT,ATMNAM
  I = IAR(ICNUM)
  J = JAR(ICNUM)
  K = KAR(ICNUM)
  L = LAR(ICNUM)
  CALL ATOMID(I,ISID,IRID,IREN,IAT)
  CALL ATOMID(J,JSID,JRID,JREN,JAT)
  CALL ATOMID(K,KSID,KRID,KREN,KAT)
  CALL ATOMID(L,LSID,LRID,LREN,LAT)
  MOV2D = -1
  PSIRES = 0
  !
  !  Find segment number, ISEG, and number of the residue that contains
  !  atom #K, KRES.  Set ATNUM0 to the number of the first atom in KRES.
  !
  ISEG=1
  IF (NSEG.EQ.0) GOTO 999
  DO WHILE (SEGID(ISEG).NE.ISID)
     ISEG=ISEG+1
     IF (ISEG.GT.NSEG) GOTO 999
  ENDDO
  KRES = GETRES(K,IBASE,NRES)
  ATNUM0 = IBASE(KRES) + 1
  !
  !  Initialize TORTYP to 1,2,3,4,5,... for phi,psi,chi1,chi2,chi3,...
  !
  IF (ISID.EQ.JSID .AND. JSID.EQ.KSID .AND. KSID.EQ.LSID .AND. &
       IRID.EQ.JRID .AND. JRID.EQ.KRID .AND. KRID.EQ.LRID) THEN
     !
     !       Four atoms in same RESIDue.
     !
     ATNUMF = IBASE(KRES+1)
     IF (IAT.EQ.'N   '.AND. JAT.EQ.'CA  ') THEN
        IF (KAT.EQ.'CB  ') THEN
           !                                 This is a chi1 angle.
           TORTYP = 3
        ELSE IF (KAT.EQ.'C   ') THEN
           !                                 This is the C-terminal psi angle.
           TORTYP = 2
           PSIRES = KRES
        END IF
     ELSE IF (IAT.EQ.'CA  '.AND. JAT.EQ.'CB  ') THEN
        !                                 This is a chi2 angle.
        TORTYP = 4
     ELSE IF (IAT.EQ.'CB  ') THEN
        !                                 This is a chi3 angle.
        TORTYP = 5
     ELSE IF (IAT.EQ.'CG  ') THEN
        !                                 This is a chi4 angle.
        TORTYP = 6
     ELSE IF (IAT.EQ.'CD  ') THEN
        !                                 This is a chi5 angle.
        TORTYP = 7
     ELSE IF (JAT.EQ.'N   '.AND. KAT.EQ.'CA  ') THEN
        !                                 This is the N-terminal phi angle.
        TORTYP = 1
     END IF
     CALL RESTYP(KREN,TORTYP,MOV2D)
  ELSE
     ATN0 = ATNUM0
     !
     !       The four IC atoms are in two RESIDues.  Determine if C- or N-terminal
     !       residues are to be moved (which involves fewer atoms).  First, consider
     !       C terminal atoms.  Set LRES to # of last residue in segment ISEG.
     !
     LRES = NICTOT(ISEG+1)
     ATNF = IBASE(LRES+1)
     I = ATNF-ATN0+1
     !
     !       Now, consider N-terminal atoms.  Last atom moved is
     !       last atom of KRES and first is first atom in segment.
     !       Set LRES to # of first residue in segment ISEG.
     !
     ATNUMF = IBASE(KRES+1)
     LRES = NICTOT(ISEG)+1
     ATNUM0 = IBASE(LRES)+1
     NATM = ATNUMF-ATNUM0+1
     IF (I .LT. NATM) THEN
        NATM = I
        ATNUM0 = ATN0
        ATNUMF = ATNF
     END IF
     !
     IF (IAT.EQ.'C   '.AND. JAT.EQ.'N   '.AND. KAT.EQ.'CA  '.AND. &
          LAT.EQ.'C   ') THEN
        !                                 This is a phi angle.
        TORTYP = 1
        CALL RESTYP(KREN,TORTYP,MOV2D)
     ELSE IF (IAT.EQ.'N   '.AND. JAT.EQ.'CA  '.AND. KAT.EQ.'C   ' &
          .AND. LAT.EQ.'N   ') THEN
        !                                 This is a psi angle.
        TORTYP = 2
        CALL RESTYP(JREN,TORTYP,MOV2D)
        PSIRES = KRES
     ELSE IF (IAT.EQ.'CA  '.AND. JAT.EQ.'C   '.AND. KAT.EQ.'N   ' &
          .AND. LAT.EQ.'CA  ') THEN
        !                                 This is an omega angle.
        TORTYP = 0
        CALL RESTYP(LREN,TORTYP,MOV2D)
        IF (TORTYP.NE.150 .AND. PRNLEV.GE.2) WRITE(OUTU,190) ICNUM
190     FORMAT(' IC entry #',I4,' is omega angle that does NOT', &
             ' precede proline!')
     ELSE
        IF (PRNLEV .GE. 2) WRITE(OUTU,200) ICNUM
200     FORMAT(' IC entry #',I4,' NOT recognized!')
     END IF
  END IF
  RETURN
999 IF (PRNLEV.GE.2) WRITE(OUTU,1000) ICNUM
1000 FORMAT(' Segment for IC #',I4,' NOT found. Abort!')
  RETURN
END SUBROUTINE IDENIC

SUBROUTINE RESTYP(RESNAM,TORTYP,MOV2D)
  !
  !  MOV2D points to the appropriate columns of P,MEAN1,etc.
  !            Residues        phi/psi columns    chi1/chi2 columns
  !      ALA,ARG,ASN,ASP,CYS       1- 5           21-23 (No ala,cys)
  !      GLN,GLU,GLY,HIS,ILE       6-10           24-27 (No gly)
  !      LEU,LYS,MET,PHE,PRO      11-15           28-31 (No pro)
  !      SER,THR,TRP,TYR,VAL      16-20           32-33 (No ser,thr,val)
  !
  !      chi1 only for CYS,SER,THR,VAL: MOV2D = 34,35,36,37
  !                Omega preceding PRO: MOV2D = 38
  !
  use chm_kinds
  implicit none
  CHARACTER(len=4) RESNAM
  character(len=60), PARAMETER :: RES = &
       'ALAARGASNASPCYSGLNGLUGLYHISILELEULYSMETPHEPROSERTHRTRPTYRVAL'
  INTEGER I,TORTYP,MOV2D
  I = INDEX(RES,RESNAM(1:3))
  IF (I.EQ.0 .AND. (RESNAM .EQ. 'HSE ' .OR. &
       RESNAM .EQ. 'HSD ' .OR. &
       RESNAM .EQ. 'HSC ' .OR. &
       RESNAM .EQ. 'HSP ')) THEN
     I = 9
  ELSE
     I = (I + 2) / 3
  END IF
  IF (TORTYP .EQ. 0) THEN
     MOV2D = 38
  ELSE IF (TORTYP .LE. 2) THEN
     MOV2D = I
  ELSE IF (TORTYP .LE. 4) THEN
     IF (RESNAM .EQ. 'ARG ') THEN
        MOV2D = 21
     ELSE IF (RESNAM .EQ. 'ASN ') THEN
        MOV2D = 22
     ELSE IF (RESNAM .EQ. 'ASP ') THEN
        MOV2D = 23
     ELSE IF (RESNAM .EQ. 'GLN ') THEN
        MOV2D = 24
     ELSE IF (RESNAM .EQ. 'GLU ') THEN
        MOV2D = 25
     ELSE IF (RESNAM .EQ. 'HIS ' .OR. &
          RESNAM .EQ. 'HSE ' .OR. &
          RESNAM .EQ. 'HSD ' .OR. &
          RESNAM .EQ. 'HSC ' .OR. &
          RESNAM .EQ. 'HSP ') THEN
        MOV2D = 26
     ELSE IF (RESNAM .EQ. 'ILE ') THEN
        MOV2D = 27
     ELSE IF (RESNAM .EQ. 'LEU ') THEN
        MOV2D = 28
     ELSE IF (RESNAM .EQ. 'LYS ') THEN
        MOV2D = 29
     ELSE IF (RESNAM .EQ. 'MET ') THEN
        MOV2D = 30
     ELSE IF (RESNAM .EQ. 'PHE ') THEN
        MOV2D = 31
     ELSE IF (RESNAM .EQ. 'TRP ') THEN
        MOV2D = 32
     ELSE IF (RESNAM .EQ. 'TYR ') THEN
        MOV2D = 33
        !                       CYS,SER,THR,VAL: chi1 only
     ELSE IF (RESNAM .EQ. 'CYS ') THEN
        MOV2D = 34
     ELSE IF (RESNAM .EQ. 'SER ') THEN
        MOV2D = 35
     ELSE IF (RESNAM .EQ. 'THR ') THEN
        MOV2D = 36
     ELSE IF (RESNAM .EQ. 'VAL ') THEN
        MOV2D = 37
     END IF
  END IF
  TORTYP = TORTYP + I*10
  RETURN
END SUBROUTINE RESTYP

CHARACTER(len=10) FUNCTION TORTXT(TORTYP)
  !
  use chm_kinds
  implicit none
  INTEGER TORTYP,I
  !
  IF (TORTYP .LT. 20) THEN
     TORTXT(1:4) =  'ALA '
  ELSE IF (TORTYP .LT. 30) THEN
     TORTXT(1:4) =  'ARG '
  ELSE IF (TORTYP .LT. 40) THEN
     TORTXT(1:4) =  'ASN '
  ELSE IF (TORTYP .LT. 50) THEN
     TORTXT(1:4) =  'ASP '
  ELSE IF (TORTYP .LT. 60) THEN
     TORTXT(1:4) =  'CYS '
  ELSE IF (TORTYP .LT. 70) THEN
     TORTXT(1:4) =  'GLN '
  ELSE IF (TORTYP .LT. 80) THEN
     TORTXT(1:4) =  'GLU '
  ELSE IF (TORTYP .LT. 90) THEN
     TORTXT(1:4) =  'GLY '
  ELSE IF (TORTYP .LT.100) THEN
     TORTXT(1:4) =  'HIS '
  ELSE IF (TORTYP .LT.110) THEN
     TORTXT(1:4) =  'ILE '
  ELSE IF (TORTYP .LT.120) THEN
     TORTXT(1:4) =  'LEU '
  ELSE IF (TORTYP .LT.130) THEN
     TORTXT(1:4) =  'LYS '
  ELSE IF (TORTYP .LT.140) THEN
     TORTXT(1:4) =  'MET '
  ELSE IF (TORTYP .LT.150) THEN
     TORTXT(1:4) =  'PHE '
  ELSE IF (TORTYP .LT.160) THEN
     TORTXT(1:4) =  'PRO '
  ELSE IF (TORTYP .LT.170) THEN
     TORTXT(1:4) =  'SER '
  ELSE IF (TORTYP .LT.180) THEN
     TORTXT(1:4) =  'THR '
  ELSE IF (TORTYP .LT.190) THEN
     TORTXT(1:4) =  'TRP '
  ELSE IF (TORTYP .LT.200) THEN
     TORTXT(1:4) =  'TYR '
  ELSE IF (TORTYP .LT.210) THEN
     TORTXT(1:4) =  'VAL '
  END IF
  I = TORTYP - ( TORTYP/10 ) * 10
  IF (I .EQ. 0) THEN
     TORTXT(5:10) = 'omega '
  ELSE IF (I .EQ. 1) THEN
     TORTXT(5:10) = 'phi   '
  ELSE IF (I .EQ. 2) THEN
     TORTXT(5:10) = 'psi   '
  ELSE IF (I .EQ. 3) THEN
     TORTXT(5:10) = 'chi 1 '
  ELSE IF (I .EQ. 4) THEN
     TORTXT(5:10) = 'chi 2 '
  ELSE IF (I .EQ. 5) THEN
     TORTXT(5:10) = 'chi 3 '
  ELSE IF (I .EQ. 6) THEN
     TORTXT(5:10) = 'chi 4 '
  ELSE IF (I .EQ. 7) THEN
     TORTXT(5:10) = 'chi 5 '
  END IF
  RETURN
END FUNCTION TORTXT

SUBROUTINE RESET(MTOT,X0,Y0,Z0,XCM,YCM,ZCM,OLDXCM,OLDYCM,OLDZCM, &
     OLDCOS,OLDPHI,OLDPSI,IFIRST,ILAST)
  use chm_kinds
  use dimens_fcm
  use comand
  use psf
  use stream
  use number
  use consta
  use coord
  use coordc

  implicit none
  real(chm_real) MTOT,X0(*),Y0(*),Z0(*),XCM,YCM,ZCM,OLDXCM,OLDYCM,OLDZCM, &
       OLDCOS,OLDPHI,OLDPSI
  INTEGER IFIRST,ILAST,I
  !
  OLDXCM = ZERO
  OLDYCM = ZERO
  OLDZCM = ZERO
  DO I = IFIRST,ILAST
     OLDXCM = OLDXCM + AMASS(I)*X(I)
     OLDYCM = OLDYCM + AMASS(I)*Y(I)
     OLDZCM = OLDZCM + AMASS(I)*Z(I)
  enddo
  OLDXCM = OLDXCM*MTOT
  OLDYCM = OLDYCM*MTOT
  OLDZCM = OLDZCM*MTOT
  XCM = OLDXCM
  YCM = OLDYCM
  ZCM = OLDZCM
  !
  !     Reset: Choose theta = phi = psi = 0.
  !     Get x,y,z's relative to cm.
  !     Also, store ALL accepted coords in COMP set.
  !
  OLDCOS = ONE
  OLDPHI = ZERO
  OLDPSI = ZERO
  DO I = IFIRST,ILAST
     X0(I) = X(I)-XCM
     Y0(I) = Y(I)-YCM
     Z0(I) = Z(I)-ZCM
  enddo
  DO I = 1,NATOM
     XCOMP(I) = X(I)
     YCOMP(I) = Y(I)
     ZCOMP(I) = Z(I)
  enddo
  !
  RETURN
END SUBROUTINE RESET
#endif /* (mcma_main)*/
end module mcmamod

