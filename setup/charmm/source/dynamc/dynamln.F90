#if KEY_MTS==1
!
!     LN Algorithm
!
!     PERFORMS MOLECULAR DYNAMICS INTEGRATION WITH NO FRILLS SUCH AS
!     TEMPERATURE ADJUSTMENT, LIST UPDATES, AND THE LIKE
!     THIS ROUTINE HANDLES THE ACTUAL INTEGRATION ALGORITHM.  IT ALSO
!     KEEPS TRACK OF AVERAGES AND FLUCTUATIONS DURING THE RUN.
!     AUTHORS: ADRIAN SANDU
!     (BASED ON VELOCITY VERLET MULTIPLE TIME SCALES CODE
!      OF MASA WATANABE)
!
!   The algorithm relies on existing CHARMM force splitting routines under
!   the MTS command. The  LN slow forces are incorporated via extrapolation as
!   opposed to "impulses" as in the MTS-RESPA method. This alleviates severe
!   resonance problems and permits larger outer timesteps to be used for
!   additional speedup.
!
!   The LN algorithm is compatible with SHAKE and with the use of boundary
!   conditions. Any other combinations of options have not been tested.
!
!   There are several parameters that are set for a LN simulation:
!
!   1. Langevin parameter gamma (FBETA in CHARMM notation),
!      the damping constant:
!
!        * Recommended value = 5 to 20 ps^(-1)
!
!        Too small a value will render the simulation unstable. On the other
!        hand, the larger gamma is, the greater the overdamping of the
!        low-frequency modes. The above recommendation reflects a balance
!        found by experimentation. Gamma can also be simulation-goal dependent.
!
!   2. Timestep Protocol for force splitting:
!
!      Dt(fast)   = inner TIMESTEP  for updating the "fast" forces
!
!        * Recommended value = 0.5 -- 1 fs (no shake)
!                               1  -- 2 fs (with shake)
!
!      Dt(medium) = K1*Dt(fast), update  frequency for "medium" forces
!
!        * Recommended value =  1  -- 3 fs
!
!      Dt(long)   = K2*Dt(medium), update frequency for "slow" forces
!
!        * Recommended value =  6  -- 200 fs
!
!        Larger computational savings can be realized with a larger Dt(long).
!        However, the speedup is limited and reaches an asymptotic value
!        since the evaluation of medium forces becomes increasingly costly.
!
!        The asymptotic maximum speedup can be reached for outer timesteps of
!        24 or 48 fs, for example, but the precise value depends on the timestep
!        protocol employed and the application system. This should be tested
!        carefully by the user for the problem at hand.
!
!   3. Definition of the force splitting classes:
!
!        Recommended Protocol --:
!
!        * Fast forces = BOND 1, ANGL 1, DIHE 1
!
!        * Medium forces = Nonbond cutoff
!               cutoff distance = 6 A - 8 A
!               healing region  = 1 A - 3 A
!               buffer region   = 1 A - 3 A
!          SLFG RSCUT [cutoff distance] RHEA [healing region]
!                 BUFF [buffer region]
!
!        * Longrange forces = remaining terms
!
!   4. GROUP electrostatics works much better than ATOM electrostatics.
!      Use of the latter is discouraged based on our test problems.
!
!
!*******************************************************************
!PLEASE NOTE:  All the LN parameters above can be sensitive to the
!-----------   specific protocol used for the dynamics simulations
!              and are problem dependent (see discussion of results
!              in the LN papers).
!
!              For further guidance, feel free to contact Tamar
!              Schlick at the email:    schlick@nyu.edu
!*******************************************************************

!
        SUBROUTINE DYNAMLN(VX,VY,VZ,VK,XNEW,YNEW,ZNEW, &
                          XOLD,YOLD,ZOLD,AMASS,NPRIV,NDEGF,IGVOPT, &
                          IMOVE,ISKP,FREEAT,NFREAT,NATOMX, &
                          BNBND,BIMAG, &
                          ISTART,ISTOP,IPRFRQ,IDYNPR,JHTEMP, &
                          GAMMA,RFD,RFT,QAVER,NAVER,XAVE,YAVE,ZAVE &
                          ,XMM,YMM,ZMM,XML,YML,ZML &
                          ,XLD,YLD,ZLD &
                         )
!
!
  use chm_kinds
  use chm_types
  use dimens_fcm
  use reawri
#if KEY_BLOCK==1 /*ldm*/
  use block_fcm, only: nblock, prhybh
  use lambdam,only: qldm, ldm_init_dynam, &
       ldm_prop2_dynamvv, &
       nsavl, msld_writld, &
       ldm_reset_dynam, &
       qmld
#endif 
  use ctitla
  use cnst_fcm
  use contrl
  use coord
  use deriv
  use cvio
  use dynio
  use holonom
  use energym
  use averfluc
  use heurist
  use number
  use consta
  use shake
  use stream
  use icfix
  use icpert
  use nose_mod
  use dmcons
  use rgym
  use tbmts
  use prssre
      implicit none
!
!
      real(chm_real) VX(*),VY(*),VZ(*),VK(*),XNEW(*),YNEW(*),ZNEW(*)
      real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
      real(chm_real) AMASS(*), ZZ
      INTEGER NPRIV,NDEGF,IGVOPT
      INTEGER IMOVE(*),ISKP(*)
      INTEGER FREEAT(*),NFREAT,NATOMX
!!      INTEGER BNBND(*),BIMAG(*)
      type(nonbondDataStructure) BNBND
      type(imageDataStructure) BIMAG

      INTEGER ISTART,ISTOP,IPRFRQ,IDYNPR
      real(chm_real) GAMMA(*),RFD(*),RFT(*)
      LOGICAL QAVER
      INTEGER NAVER
      real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
!      real(chm_real) XMI(*),YMI(*),ZMI(*)
      real(chm_real) XMM(*),YMM(*),ZMM(*)
! A. Sandu - long range forces
      real(chm_real) XML(*),YML(*),ZML(*)
      real(chm_real) XLD(*),YLD(*),ZLD(*)
      real(chm_real) JHTEMP
!

      real(chm_real) DELTA2,FACT,DNUM,DNM1,DNP1,DNPM1
      real(chm_real) DELTAS,TIME,TEMPI,AKMATI,ALPHA,TOTEPR
      INTEGER NATOM,IST1,ISTEP,ISTPSA,NUMSTP,I,J,K
      INTEGER NIT2,NIT
      LOGICAL OK
      real(chm_real)  TMPN(MAXNOS),EPTKE1(MAXNOS),EPTKX(MAXNOS)
      real(chm_real)  SNHV1(MAXNOS)
      real(chm_real)  ECSUM,FACT1,SA1X,SA2X,SA3X
      real(chm_real)  RAVL,TOTKEN,TOTKEO,SAX
      real(chm_real)  SS1X,SS2X,SS3X,SNHF1
      real(chm_real)  QK1,QK2,QV1,QV2,FACT2
      real(chm_real)  EMTS(LENENT)
      INTEGER NNQ, NMTS0
      LOGICAL DUMM, QOK
!
      INTEGER FSTEP,SSTEP,AVG,FLUC
      PARAMETER(FSTEP=0,SSTEP=1,AVG=2,FLUC=3)
!
      real(chm_real) FLUCTD
!
!     TIMFAC is the conversion factor from AKMA time to picoseconds.
!
      DATA TOTEPR/-9999.0D0/
!
      DUMM=.FALSE.
!
! A. Sandu - fill the Langevin std array here, if needed
!     DO I=1,NATOM
!       RFD(I) = SQRT(2*FBETA(I)*TIMFAC*KBOLTZ*300.*AMASS(I)/DELTA)
!     ENDDO
!
! A. Sandu - check if proper flags for LN
!            if not, abort integration
      IF (.NOT. QTBMTS) THEN
        WRITE(OUTU,*) '********************************************'
        WRITE(OUTU,*) '*   (USE <LANG> OPTION INSTEAD OF <LNX>)   *'
        WRITE(OUTU,*) '********************************************'
        CALL WRNDIE(-2, 'DYNAMLN', 'LN CALLED WITHOUT MTS')
      ENDIF
!
      NATOM=NATOMX
!
! First (N)
!
      DELTAS=HALF*DELTA
      DELTA2=DELTA*DELTAS
!
! Second (M)
!
      SS1X=DELTA*NMTS1
      SS2X=HALF*SS1X
      SS3X=SS1X*SS2X
!
!  Multitime Dt=N*M*dt
!
      IF (.NOT. QTBMTS) NMTS=1
      NMTS0=NMTS
!
      SA1X=DELTA*NMTS0
      SA2X=HALF*SA1X
      SA3X=SA1X*SA2X
!
      IST1=ISTART-1
!
      IF(QAVER) THEN
        IF(NAVER == 0) THEN
          DO I=1,NATOM
            XAVE(I)=ZERO
            YAVE(I)=ZERO
            ZAVE(I)=ZERO
          ENDDO
        ENDIF
      ENDIF
!
       DO I=1,LENENT
         EMTS(I)=ZERO
       ENDDO
!
!
#if KEY_BLOCK==1 /*ldm*/
!:    set BLCOEF = LAMBDA before update the lambdas or call energy
!:    lambda(1) == 1 by the design see block command
      if(qldm) call ldm_init_dynam(nblock)
#endif /*  LDM*/
!
      IF(IGVOPT >= 3 .AND. (IDYNPR == 0.OR.JHSTRT.EQ.0)) THEN
!
!  Multiple Time Scale (Initial Energy)
!
      IF (QTBMTS) THEN
!
      IF(NMTS2 > 1) THEN
        ENE2=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        DO I=1,LENENT
          EMTS(I)=ETERM(I)
        ENDDO
        DO I=1,NATOM
          XMM(I)=DX(I)
          YMM(I)=DY(I)
          ZMM(I)=DZ(I)
        ENDDO
      ENDIF
!
      ENE3=.TRUE.
      CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
      DO I=1,LENENT
        EMTS(I)=EMTS(I)+ETERM(I)
      ENDDO
      DO I=1,NATOM
        XML(I)=DX(I)
        YML(I)=DY(I)
        ZML(I)=DZ(I)
      ENDDO
!
      ENE1=.TRUE.
      CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
      ECSUM=ZERO
      DO I=1,LENENT
        ETERM(I)=ETERM(I)+EMTS(I)
        ECSUM=ETERM(I)+ECSUM
      ENDDO
      EPROP(EPOT)=ECSUM
!
        ELSE
!       Get previous energy for printing (just as a check)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        ENDIF
!
      CALL GRAM(DX,DY,DZ,XMM,YMM,ZMM,XML,YML,ZML)
      ENDIF
!
      IF(IGVOPT == 2) THEN
!
!       This is the 2-step VERLET
        DO I=1,NATOM
          XOLD(I)=X(I)
          YOLD(I)=Y(I)
          ZOLD(I)=Z(I)
        ENDDO
!
       IF(QHOLO) THEN
!         Do shake just in case coordinates don't fit constraints
          CALL HOLONOMA(X,Y,Z,XOLD,YOLD,ZOLD,.TRUE.,.TRUE.,QOK)
!
        IF (QTBMTS) THEN
          DUMM =.TRUE.
          QTBMTS = .FALSE.
          SLFG = .FALSE.
        ENDIF
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        IF(DUMM) THEN
          DUMM = .FALSE.
          QTBMTS = .TRUE.
          SLFG = .TRUE.
        ENDIF
!
        DO I = 1, NATOM
        IF(IMOVE(I) == 0) THEN
!           FACT1=2.0*DELTAS/AMASS(I)
           FACT1=DELTAS/AMASS(I)
           VX(I)=VX(I)-FACT1*DX(I)
           VY(I)=VY(I)-FACT1*DY(I)
           VZ(I)=VZ(I)-FACT1*DZ(I)
        ENDIF
        ENDDO
!--------------------------------------------
        DO I = 1, NATOM
         IF(IMOVE(I) == 0) THEN
           XNEW(I)=X(I)+DELTA*VX(I)
           YNEW(I)=Y(I)+DELTA*VY(I)
           ZNEW(I)=Z(I)+DELTA*VZ(I)
         ENDIF
        ENDDO
!
        CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE, &
                    ISKP,NATOM,DELTA)
!
        DO I = 1, NATOM
           X(I)=XNEW(I)
           Y(I)=YNEW(I)
           Z(I)=ZNEW(I)
        ENDDO
       ENDIF
!
! Multiple Time Scale (Initial Energy)
!
      IF (QTBMTS) THEN
!
!- MEDIUM FORCE
       IF(NMTS2 > 1) THEN
        ENE2=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        DO I=1,LENENT
          EMTS(I)=ETERM(I)
        ENDDO
        DO I=1,NATOM
          XMM(I)=DX(I)
          YMM(I)=DY(I)
          ZMM(I)=DZ(I)
        ENDDO
       ELSE
        DO I=1,NATOM
          XMM(I)=ZERO
          YMM(I)=ZERO
          ZMM(I)=ZERO
        ENDDO
       ENDIF
!
!- SLOW VERING FORCE
        ENE3=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
        DO I=1,LENENT
          EMTS(I)=EMTS(I)+ETERM(I)
        ENDDO
        DO I=1,NATOM
          XML(I)=DX(I)
          YML(I)=DY(I)
          ZML(I)=DZ(I)
        ENDDO
!
! - FASTEST VERING FORCE
       ENE1=.TRUE.
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       ECSUM=ZERO
       DO I=1,LENENT
          ETERM(I)=ETERM(I)+EMTS(I)
          ECSUM=ETERM(I)+ECSUM
       ENDDO
       EPROP(EPOT)=ECSUM
!
       ELSE
         CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       ENDIF
       CALL GRAM(DX,DY,DZ,XMM,YMM,ZMM,XML,YML,ZML)
!
        IGVOPT=3
!
        IF (QTBMTS) THEN
          DO I=1,NATOM
           IF(IMOVE(I) == 0) THEN
             FACT1=DELTAS/AMASS(I)
             FACT=SS2X/AMASS(I)
             FACT2=SA2X/AMASS(I)
           ENDIF
          ENDDO
        ELSE
!
        DO I=1,NATOM
         IF(IMOVE(I) == 0) THEN
          FACT1=DELTAS/AMASS(I)
!          VX(I)=VX(I)-FACT1*DX(I)
!          VY(I)=VY(I)-FACT1*DY(I)
!          VZ(I)=VZ(I)-FACT1*DZ(I)
         ENDIF
        ENDDO
!
        ENDIF
#if KEY_BLOCK==1 /*ldm*/
      if(qldm) call ldm_prop2_dynamvv(nblock, delta)
#endif /*  LDM*/

      ENDIF
!
      IF (ISTOP < ISTART) CALL WRNDIE(-2, 'DYNAMLN', 'ISTOP >= ISTART')
      IF(MOD(IST1,IPRFRQ) == 0) THEN
        call avfl_reset()
        FITA = ZERO
      ENDIF
!
       IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
          IF(QNOSE) THEN
          DO I=1,NOBL
          EPTKX(I)=ZERO
          ENDDO
          ENDIF
!
         TOTKEN=ZERO
         TEMPI=ZERO
          DO I=1,NATOM
            IF(IMOVE(I) == 0) THEN
              RAVL = AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
              TEMPI=TEMPI+RAVL
            IF(QNOSE) THEN
              J = INLCKP(I)
              EPTKX(J)=EPTKX(J)+RAVL
            ENDIF
            ENDIF
          ENDDO
!
!       WRITE OUT INITIAL CONDITIONS
!
        ISTEP=ISTART-1
        TIME=TIMFAC*DELTA*NMTS0*NPRIV
!       . Calculate the pressures.
        CALL PRSEXT
        CALL PRSINT(NATOM,AMASS,VX,VY,VZ,ONE,[ZERO], [ZERO],[ZERO],ZERO)
!
        QK1=ZERO
        QK2=ZERO
        IF(QNOSE) THEN
        DO I = 1, NOBL
        SNHF(I)=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
        QK1=QK1+0.5*SQM(I)*SNHV(I)*SNHV(I)
        QK2=QK2+NDGN(I)*KBOLTZ*SNH(I)*RTMPR(I)
        ENDDO
        ENDIF
!
        EPROP(TOTKE)=TEMPI/TWO
        EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK1+QK2
!
        IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK1+QK2
        ELSE
        EPROP(HFCTE)=ZERO
        EPROP(EHFC)=ZERO
        ENDIF
        EPROP(TEMPS)=TEMPI/(KBOLTZ*NDEGF)
        CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .TRUE., &
          ISTEP, TIME, ZERO, .TRUE.)
      ENDIF
!
!     INITIALIZE ACCUM VARIABLES
      IF(JHSTRT == 0) THEN
        DO I=1,NATOM
          VK(I)=ZERO
        ENDDO
        JHTEMP=ZERO
        FITP=ZERO
        call avfl_reset_lt()
      ENDIF
!
!-------------------------------------------------------
!     THIS IS THE MAIN LOOP FOR DYNAMICS.
!-------------------------------------------------------
!
        DO ISTEP=ISTART,ISTOP
!
        DO I=1,LENENT
          EMTS(I)=ZERO
        ENDDO
!
        JHSTRT=JHSTRT+1
!
        NPRIV=NPRIV+1
!
! FBS MOD ************************************
#if KEY_DMCONS==1
         DSTEP = NPRIV
#endif 
#if KEY_RGYCONS==1
         DSTEPRG = NPRIV
#endif 
! END FBS MOD ********************************
!
         IF(QAVER) THEN
          NAVER=NAVER+1
          DO I=1,NATOM
            XAVE(I)=XAVE(I)+X(I)
            YAVE(I)=YAVE(I)+Y(I)
            ZAVE(I)=ZAVE(I)+Z(I)
          ENDDO
        ENDIF
!
        IF(NSAVC > 0) THEN
          IF (MOD(ISTEP,NSAVC) == 0) THEN
            IF (QAVER) THEN
              DNUM=ONE
              DNUM=DNUM/NAVER
              DO I=1,NATOM
                XAVE(I)=XAVE(I)*DNUM
                YAVE(I)=YAVE(I)*DNUM
                ZAVE(I)=ZAVE(I)*DNUM
              ENDDO
              CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                          (/ ZERO /), .FALSE., &  
#endif
                          NATOM,FREEAT,NFREAT,NPRIV, &
                          ISTEP,NDEGF, &
                          SA1X,NSAVC,NSTEP,TITLEA,NTITLA,IUNCRD,.FALSE., &
                          .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
              NAVER=0
              DO I=1,NATOM
                XAVE(I)=ZERO
                YAVE(I)=ZERO
                ZAVE(I)=ZERO
              ENDDO
            ELSE
        CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                    (/ ZERO /), .FALSE., &  
#endif
                    NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                    SA1X,NSAVC,NSTEP,TITLEA,NTITLA,IUNCRD,.FALSE., &
                    .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
            ENDIF
          ENDIF
        ENDIF
#if KEY_BLOCK==1 /*ldm*/
      if(nsavl > 0 .and. iolev.gt.0) then
         if(mod(istep,nsavl) == 0) then
            if (qmld) then
               call msld_writld(nblock,npriv, &
                  istep,nstep, &
                  delta)
            else
               call writld(nblock,npriv, &
                  istep,nstep, &
                  delta)
            endif
         endif
      endif
#endif 
!
! -- Remember coordinates, it will help when updating the NB lists
! -- at the end of the outer timestep
         DO I=1,NATOM
           XOLD(I)=X(I)
           YOLD(I)=Y(I)
           ZOLD(I)=Z(I)
         ENDDO
!
!----------------------------------------------------|
!                                                    |
!              START OF MEDIUM CYCLE                 |
!                                                    |
!----------------------------------------------------|
!
       DO NIT2 = 1, NMTS2
!
! ---- Predict position at midpoint of next medium timestep -----
!
       DO I=1,NATOM
         XNEW(I)=X(I)
         YNEW(I)=Y(I)
         ZNEW(I)=Z(I)
         X(I)=X(I)+SS2X*VX(I)
         Y(I)=Y(I)+SS2X*VY(I)
         Z(I)=Z(I)+SS2X*VZ(I)
       ENDDO
!
! ---- Compute medium force at predicted positions ---
!
       ENE2=.TRUE.
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
!
! ---- Restore positions and save computed medium force ---
!
       DO I=1,NATOM
         X(I)=XNEW(I)
         Y(I)=YNEW(I)
         Z(I)=ZNEW(I)
         XMM(I)=DX(I)
         YMM(I)=DY(I)
         ZMM(I)=DZ(I)
       ENDDO
!
!----------------------------------------------------
!
!   START OF INNER CYCLE
!
!----------------------------------------------------

       DO NIT = 1, NMTS1
!
! --- First half step updates positions ---
!
           DO I=1,NATOM
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 981
           IF (IMOVE(I) == 0) THEN
             XNEW(I)=X(I)+DELTAS*VX(I)
             YNEW(I)=Y(I)+DELTAS*VY(I)
             ZNEW(I)=Z(I)+DELTAS*VZ(I)
           ENDIF
981        CONTINUE
           ENDDO
!
! --- Adjust the computed positions to satisfy
!     holonomic constraints (e.g. SHAKE), if necessary
!
         IF (QHOLO) THEN
            CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE, &
              ISKP,NATOM,DELTAS)
         ENDIF
!
! --- Load computed positions in default arrays ---
!
         DO I=1,NATOM
         IF (QTBMTS .AND. IMTS(I) < 0) GOTO 982
         IF (IMOVE(I) == 0) THEN
            X(I)=XNEW(I)
            Y(I)=YNEW(I)
            Z(I)=ZNEW(I)
         ENDIF
982      CONTINUE
         ENDDO
!
! --- Compute the fast force ---
!
        ENE1=.TRUE.
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
!
! --- Add the stochastic force to the fast force ---
!
        IF (FBETA(1) /= ZERO) THEN
          CALL DLNGVV(RFD,NATOM,RFT,DX,DY,DZ,ISEED)
        ENDIF
!
!
! --- Update velocities. Formula includes the Langevin damping
!     force (- gamma V) implicitly. The medium and longrange
!     forces are kept constant inside the inner loop,
!     but are properly updated outside it. ---
!
          DO I=1,NATOM
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 983
           IF(IMOVE(I) == 0) THEN
             FACT1 = DELTA/AMASS(I)
             FACT2 = ONE/(ONE + DELTA*TIMFAC*FBETA(I))
             VX(I)=(VX(I)-FACT1*(DX(I)+XMM(I)+XML(I)))*FACT2
             VY(I)=(VY(I)-FACT1*(DY(I)+YMM(I)+YML(I)))*FACT2
             VZ(I)=(VZ(I)-FACT1*(DZ(I)+ZMM(I)+ZML(I)))*FACT2
           ENDIF
983        CONTINUE
          ENDDO
!
! --- Second halfstep to update positions ---
!
           DO I=1,NATOM
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 984
           IF (IMOVE(I) == 0) THEN
              XNEW(I)=X(I)+DELTAS*VX(I)
              YNEW(I)=Y(I)+DELTAS*VY(I)
              ZNEW(I)=Z(I)+DELTAS*VZ(I)
           ENDIF
984        CONTINUE
           ENDDO
!
! --- Adjust the computed positions to satisfy
!     holonomic constraints (e.g. SHAKE), if necessary ---
!
         IF (QHOLO) THEN
            CALL DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW,AMASS,IMOVE, &
              ISKP,NATOM,DELTAS)
         ENDIF
!
! --- Load computed positions in default arrays ---
!
         DO I=1,NATOM
           IF (QTBMTS .AND. IMTS(I) < 0) GOTO 985
           IF (IMOVE(I) == 0) THEN
            X(I)=XNEW(I)
            Y(I)=YNEW(I)
            Z(I)=ZNEW(I)
          ENDIF
985       CONTINUE
         ENDDO
!
        ENDDO
!----------------------------------------------------
!
!   END OF INNER CYCLE
!
!----------------------------------------------------

!
        ENDDO
!----------------------------------------------------
!
!   END OF MEDIUM CYCLE
!
!----------------------------------------------------


! Do list updates if appropriate (Image centering within dynamics
! doesn't work with the old integrator).
!
         CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN, &
                            1,XOLD,YOLD,ZOLD,VX,VY,VZ)
!
! --- Update the longrange forces ---
!
         ENE3=.TRUE.
         CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
         DO I=1,NATOM
           XML(I)=DX(I)
           YML(I)=DY(I)
           ZML(I)=DZ(I)
         ENDDO
         DO I=1,LENENT
           EMTS(I)=ETERM(I)
         ENDDO
!
! -- Recompute medium and fast forces at the end of each outer timestep.
! -- This is some extra work for reporting purposes only.
!
         ENE2=.TRUE.
         CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
!
         DO I=1,LENENT
           EMTS(I)=EMTS(I)+ETERM(I)
         ENDDO
!
         ENE1=.TRUE.
         CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
!
! ---- Collect all energy terms for reporting ---
!
         ECSUM=ZERO
         DO I=1,LENENT
           ETERM(I)=ETERM(I)+EMTS(I)
           ECSUM=ETERM(I)+ECSUM
         ENDDO
         EPROP(EPOT)=ECSUM
         CALL GRAM(DX,DY,DZ,XMM,YMM,ZMM,XML,YML,ZML)
!---------------------------------------------------------------
!
!
!==========================================================
        NNQ=0
8818    CONTINUE
        NNQ=NNQ+1
!
!
        IF(QNOSE) THEN
!
! NOSE-HOOVER
!
        TEMPI=ZERO
        DO I = 1,NOBL
        EPTKE1(I)=EPTKX(I)
        EPTKX(I)=ZERO
        ENDDO
!
        DO I=1,NATOM
        IF(IMOVE(I) == 0) THEN
        RAVL=AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
        TEMPI=TEMPI+RAVL
        J = INLCKP(I)
        EPTKX(J)=EPTKX(J)+RAVL
        ENDIF
        ENDDO
!
        DO I=1,NOBL
        SNHF1=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
        SNHV(I)=SNHV1(I)+SA2X*SNHF1/SQM(I)
        ENDDO
         QV2=ZERO
         DO I=1,NOBL
         QV2=QV2+ABS(EPTKE1(I)-EPTKX(I))
         ENDDO
         QV2=QV2/NOBL
         IF(QV2 <= TOLNS) GOTO 8919
         IF(NNQ < NSCYC) GOTO 8818
        ENDIF
!
!-------------------------------------------------------
8919     CONTINUE
         TEMPI=ZERO
         TOTKEN=ZERO
          IF(QNOSE) THEN
          DO I=1,NOBL
          EPTKX(I)=ZERO
          ENDDO
          ENDIF
!
          DO I=1,NATOM
            IF(IMOVE(I) == 0) THEN
            RAVL=AMASS(I)*(VX(I)**2 + VY(I)**2 + VZ(I)**2)
            TEMPI=TEMPI+RAVL
            VK(I)=VK(I)+RAVL
            IF(QNOSE) THEN
            J = INLCKP(I)
            EPTKX(J)=EPTKX(J)+RAVL
            ENDIF
          ENDIF
          ENDDO
!
          IF(QNOSE) THEN
          DO I = 1,NOBL
          SNHF(I)=EPTKX(I)-NDGN(I)*KBOLTZ*RTMPR(I)
          SNHV(I)=SNHV1(I)+SA2X*SNHF(I)/SQM(I)
          ENDDO
          ENDIF
!
!       . Calculate the pressures.
        CALL PRSEXT
        CALL PRSINT(NATOM,AMASS,VX,VY,VZ,ONE,[ZERO],[ZERO],[ZERO],ZERO)
!
        QK1=ZERO
        QK2=ZERO
        IF(QNOSE) THEN
        DO I=1,NOBL
        QK1=QK1+0.5*SQM(I)*SNHV(I)*SNHV(I)
        QK2=QK2+NDGN(I)*KBOLTZ*RTMPR(I)*SNH(I)
        ENDDO
        ENDIF
!
        EPROP(TOTKE)=TEMPI/TWO
        EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)+QK2+QK1
!
        IF(QNOSE) THEN
        EPROP(HFCTE)=EPROP(EPOT)+EPROP(TOTKE)
        EPROP(EHFC)=QK2+QK1
        ENDIF
!
!       CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.
!
        IF ((.NOT.QNOSE).AND.(JHSTRT > 2)) THEN
!
! A. Sandu: Scale the energy change by NMTS2 to avoid dummy reports.
!
          IF (ABS(TOTEPR-EPROP(TOTE))/NMTS2 > MAX(ECHECK,0.1D0*EPROP &
            (TOTKE))) THEN
            IF (TOTEPR  /=  -9999.0) THEN
              WRITE (OUTU,2000) ECHECK,TOTEPR,EPROP(TOTE),EPROP(TOTKE)
 2000         FORMAT(' TOTAL ENERGY CHANGE EXCEDED'/G12.2, &
                ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY IN THE ' &
                  //'LAST STEP'/ &
                ' PREVIOUS EPROP(EPOT) =',G14.4,' CURRENT EPROP(EPOT)' &
                  //' =',G14.4, &
                ' KINETIC =',G14.4)
              CALL WRNDIE(-2,'<DYNAMC>','ENERGY CHANGE TOLERENCE ' &
                //'EXCEEDED')
            ENDIF
          ENDIF
        ENDIF
!
        TOTEPR=EPROP(TOTE)
        TIME=TIMFAC*DELTA*NMTS0*NPRIV
        AKMATI=DELTA*NMTS0*NPRIV
        EPROP(TEMPS)=TEMPI/(KBOLTZ*NDEGF)
!
        IF(QNOSE) THEN
        DO I=1,NOBL
        TMPN(I)=EPTKX(I)/(KBOLTZ*NDGN(I))
        ENDDO
        ENDIF
!
        JHTEMP=JHTEMP+EPROP(TEMPS)
        call avfl_update(EPROP, ETERM, EPRESS)
        ISTPSA=MOD(IST1,IPRFRQ)+ISTEP-IST1
        FITA=FITA+ISTPSA*EPROP(TOTE)
        FITP=FITP+JHSTRT*EPROP(TOTE)
!
        IF(NSAVV > 0) THEN
          IF(MOD(ISTEP,NSAVV) == 0) THEN
            CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                        (/ ZERO /), .FALSE., &  
#endif
                        NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF, &
                        SA1X,NSAVV,NSTEP,TITLEA,NTITLA,IUNVEL,.TRUE., &
                        .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
          ENDIF
        ENDIF
!
        IF((IUNOS >= 0).AND.QNOSE) THEN
          IF(MOD(ISTEP,NSNOS) == 0) THEN
           WRITE(IUNOS,166) NPRIV,TIME,EPROP(TOTE)
           WRITE(IUNOS,167) (SNH(I),TMPN(I),I=1,NOBL)
167        FORMAT(2F16.6)
!          machine dependent call to flush output buffers
           CALL GFLUSH(IUNOS)
166     FORMAT(I16,2F16.6)
          ENDIF
        ENDIF
!
        IF(MOD(ISTEP,NPRINT) == 0 .AND. PRNLEV >= 2) THEN
          CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
            ISTEP, TIME, ZERO, .TRUE.)
!
          IF(KUNIT >= 0 .AND. IOLEV > 0) THEN
            WRITE(KUNIT,160) NPRIV,TIME,EPROP(TOTE),EPROP(TOTKE), &
              EPROP(EPOT), &
              EPROP(EPOT)-EPROP(TOTKE),EPROP(TEMPS), &
#if KEY_CMAP==1
              ETERM(BOND),ETERM(ANGLE),ETERM(DIHE)+ETERM(CMAP), &
#else /**/
              ETERM(BOND),ETERM(ANGLE),ETERM(DIHE), &
#endif 
              ETERM(IMDIHE),ETERM(VDW),ETERM(ELEC), &
              ETERM(HBOND),ETERM(CHARM) &
#if KEY_DMCONS==1
              ,ETERM(DMC) &  
#endif
#if KEY_RGYCONS==1
              ,ETERM(RGY) &  
#endif
              ;
  160       FORMAT(I16,4F16.4,/,5F16.4,/,5F16.4)
!           machine dependent call to flush output buffers
            CALL GFLUSH(KUNIT)
          ENDIF
!
!
        ENDIF
#if KEY_BLOCK==1
!.ab.Print out HybridH. Use the #BLOCK#.
!.ab.Not clear that this integrator is suited...
        IF(IOLEV >= 0) CALL PRHYBH()
!.ab.
#endif 

        ENDDO
!----------------------------------------------------------------
! Main loop End
!----------------------------------------------------------------
#if KEY_BLOCK==1 /*ldm*/
!     reset lambdas to time t: the reason is that at the end of
!     dcntrl.src, when ORIG is true, x is reset to xold
!     set BLDOLD to bxlamb(t+dt) for restart file
      if(qldm) call ldm_reset_dynam(nblock)
#endif /*  LDM*/
      IF (ISTPSA < IPRFRQ) RETURN
!
!     Now do long time fluctuations
!
      call avfl_update_lt()
      NUMSTP = ISTPSA
      DNUM   = NUMSTP
      DNM1   = DNUM - ONE
      DNP1   = DNUM + ONE
      DNPM1  = DNM1 * DNP1
      IF(NUMSTP > 1) THEN
        DRIFTA = (TWO*FITA-DNP1*EPRPA(TOTE))*SIX/(DNPM1*DNUM)
        EAT0A  = EPRPA(TOTE)/DNUM-DRIFTA*DNP1/TWO
        IF ((EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2) > ZERO) THEN
          CORRA=DNPM1/(EPRP2A(TOTE)/DNUM-(EPRPA(TOTE)/DNUM)**2)/TWELVE
          CORRA=DRIFTA*SQRT(CORRA)
        ELSE
          CORRA=ZERO
        ENDIF
!
!       . Compute statistics.
        call avfl_compute(DNUM)
      ENDIF
      if (PRNLEV >= 2) then
         call avfl_print_aver(NUMSTP, TIME)
         call avfl_print_fluc(NUMSTP, TIME)
      endif
      DRIFTP=DRIFTA
      EAT0P=EAT0A
      CORRP=CORRA
      IF (JHSTRT <= NUMSTP) GOTO 190
      NUMSTP=JHSTRT
      DNUM=NUMSTP
      DNM1=DNUM-ONE
      DNP1=DNUM+ONE
      DNPM1=DNM1*DNP1
      IF(NUMSTP > 1) THEN
        DRIFTP = (TWO*FITP - DNP1*EPRPP(TOTE))*SIX/(DNPM1*DNUM)
        EAT0P  = EPRPP(TOTE)/DNUM-DRIFTP*DNP1/TWO
        IF ((EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2) > ZERO) THEN
          CORRP=DNPM1/(EPRP2P(TOTE)/DNUM-(EPRPP(TOTE)/DNUM)**2)/TWELVE
          CORRP=DRIFTP*SQRT(CORRP)
        ELSE
          CORRP=ZERO
        ENDIF
      ENDIF
      call avfl_compute_lt(DNUM)
      if (PRNLEV >= 2) then
         call avfl_print_aver_lt(NUMSTP, TIME)
         call avfl_print_fluc_lt(NUMSTP, TIME)
      endif
!
 190  CONTINUE
      IF(NUMSTP <= 1 .OR. JHSTRT.LE.1) RETURN
!
      WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
  195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
        /5X,'EPROP(EPOT) AT STEP 0  : ',1P,2G17.8, &
        /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
!
#else /**/
      SUBROUTINE DYNAMLN_DUM()
#endif 
      RETURN
      END

