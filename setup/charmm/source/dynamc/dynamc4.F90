module dyn4
  use chm_types
  use dimens_fcm
  implicit none

contains

#if KEY_FOURD==1 /*4ddynamc*/
  SUBROUTINE DYNAMC4(VX, VY, VZ, VK, XNEW, YNEW, ZNEW, XOLD, &
       YOLD, ZOLD, AMASS, FDNEW, FDOLD, FDAVE, VFD, &
       NPRIV, NPRIVOLD, NDEGF, NDEG4, IGVOPT, &
       IMOVE, ISKP, FREEAT, NFREAT, NATOMX, BNBND, &
       BIMAG, ISTART, ISTOP, IPRFRQ, IDYNPR, JHTEMP, &
       GAMMA, QAVER, NAVER, XAVE, YAVE, ZAVE,  &
       QKUHEAD,RUNOK, &
       IEQ4, IHT4, TIN4, FSTT4, FNLT4, TWH4, TWL4, &
       ICH4, IASORS, ISCVEL, IASVEL &
       )
    !-----------------------------------------------------------------------
    !     PERFORMS MOLECULAR DYNAMICS INTEGRATION IN 4 DIMINSIONS
    !     NOTE:although shake still exists here, it is
    !          not yet supported in 4D.
    !
    !     This routine handles the actual integration algorithms.  It also
    !     keeps track of averages and fluctuations during the run.
    !
    !     Authors: Everybody
    !     Overhauled by Bernie Brooks
    !     Langevin Dynamics included by Charlie Brooks and Axel Brunger
    !     4D:Elan Eisenmesser
    !
    use vector
    use number
#if KEY_SCCDFTB==1
    use blockscc_fcm  
#endif
#if KEY_SCCDFTB==1
    use sccdftb   
#endif
    use reawri
    use ctitla
    use contrl
    use coord
    use coordc
    use cvio
    use deriv
    use dynio
    use energym
    use averfluc
    use avfl_ucell
    use fourdm
    use image
    use consta
    use shake
    use stream
    use tsms_mod
#if KEY_FLUCQ==1
    use flucq      
#endif
    use phmd
#if KEY_ADUMB==1
    use umb       
#endif
#if KEY_GAMUS==1
    use gamusmodule 
#endif
#if KEY_PARALLEL==1
    use parallel       
#endif
#if KEY_BLOCK==1
    use block_fcm          
#endif
    use holonom,only:holonoma
    use machutil,only:eclock,die
    use heurist,only:updeci
    use prssre

!QC_Ito_UW0616
#if KEY_RXNCOR==1
  use rxncom, only: UMBMDSTEP
#endif

#if KEY_SCCDFTB==1
    real(chm_real) dvdltmp         
#endif
    !
    real(chm_real) VX(*),VY(*),VZ(*),VK(*),XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) XOLD(*),YOLD(*),ZOLD(*)
    real(chm_real) AMASS(*)
    real(chm_real) FDNEW(*),FDOLD(*),FDAVE(*),VFD(*)
    INTEGER NPRIV,NPRIVOLD,NDEGF,IGVOPT,ndegt
    INTEGER IMOVE(*),ISKP(*)
    INTEGER FREEAT(*),NFREAT,NATOMX
    !!      INTEGER BNBND(*),BIMAG(*)
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG

    INTEGER ISTART,ISTOP,IPRFRQ,IDYNPR
    real(chm_real) GAMMA(*)
    LOGICAL QAVER,QKUHEAD
    INTEGER NAVER
    real(chm_real) XAVE(*),YAVE(*),ZAVE(*)
    real(chm_real) JHTEMP,TOTTEMP,TOTVEL
    LOGICAL RUNOK,OK
    INTEGER IASORS,ISCVEL,IASVEL
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(10),TIMMER
#endif 
    !
    real(chm_real) DELTA2,FACT,DNUM,DNM1,DNP1,DNPM1
    real(chm_real) DELTAS,TIME,RVAL,ALPHA,TEMPI,EWRK
    !  For individual T's
    real(chm_real) tot3d,tot4d,totp3d
    real(chm_real) TEMX,TEMY,TEMZ,TEMFD
    real(chm_real) VXI,VYI,VZI
    !
    INTEGER NATOM,IST1,ISTEP,ISTPSA,NUMSTP,I,NATOM2,NATOM3
    INTEGER ATFRST,ATLAST
    !
    real(chm_real) FLUCTD,TOTKEO,TOTKEN,GNORM
    CHARACTER(len=40) STR
    INTEGER STRLN
    LOGICAL QOK
    ! 4-D variable:
    INTEGER J4STRT, ICH4, VFDI
    real(chm_real) J4TEMP,TEMNEW4,AVETEM4,HEAT4,DELTEM
    real(chm_real) FSTT4,TIN4,FNLT4,TWH4,TWL4
    INTEGER IEQ4,IHT4,J,NDEG4
    real(chm_real) TOTK4O,TOTK4N,TEMPI4,RVAL4
    !
    SAVE
    !
    ! TIMFAC is the conversion factor from AKMA time to picoseconds.
    !
    !=======================================================================
    ! Startup code
    !
#if KEY_BLOCK==1
    !.ab.Block.f90 is there, can use QHYBH, other wrndie below...
    IF (QHYBH) THEN
       IF (PRNLEV > 5) WRITE(OUTU,*) ' Entering DYNAMC4.'
       CALL WRNDIE(-6,'<DYNAMC4>','HYBH under development: not ready.')
    ENDIF
    !.ab.
#endif 
    NATOM=NATOMX
    NATOM2=NATOM+NATOM
    NATOM3=NATOM2+NATOM
    !
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (parfmain)*/
#else /* (paramain)*/
    ATFRST=1
    ATLAST=NATOM
#endif /* (paramain)*/
    !
    !
    DELTAS=HALF*DELTA*DELTA
    DELTA2=DELTA*DELTA
    IST1=ISTART-1

    ! Initialize 4-d heating parameters:
    IF(FCOUNT == 0) THEN
       IF((INC4D == 0).OR.(DEC4D.EQ.0)) THEN
          IGVOPT=2 
          CALL INRDYN(X,Y,Z,FDIM,VX,VY,VZ,VFD,NATOM)
       ENDIF
       IF(TSTRUC /= FMARK) THEN
          TEMNEW4=MAX(TWO*FSTT4-TSTRUC,ZERO)
       ELSE
          TEMNEW4=1.25*FSTT4
       ENDIF
       !CC        TEMNEW4=FSTT4
       IF(REST4 == 0) &
            CALL ASSVEL4(TEMNEW4,FDIM,VFD,AMASS,ISEED,NATOM,IMOVE, &
            IMOVE4,IASVEL)
       J4STRT = 0
       J4TEMP = ZERO
    ENDIF
    !
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN
#endif 
          XCOMP(I)=X(I)
          YCOMP(I)=Y(I)
          ZCOMP(I)=Z(I)
          FDCOMP(I)=FDIM(I)

#if KEY_PARASCAL==1
       ENDIF
#endif 
    ENDDO
    !
    IF(QAVER) THEN
       IF(NAVER == 0) THEN
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XAVE(I)=ZERO
                YAVE(I)=ZERO
                ZAVE(I)=ZERO
                FDAVE(I)=ZERO
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
       ENDIF
    ENDIF
#if KEY_TSM==1
    IF(QTSM) CALL WRNDIE(-3,'<DYNAMC>', &
         'DIM4 and TSM are not compatible')
#endif 
#if KEY_PHMD==1
    IF (QPHMD) CALL WRNDIE(-3,'<DYNAMC>', &
         'PHMD not supported in DIM4')
#endif 
    !
    !=======================================================================
    !
    ! This is the 3-step VERLET
    IF(IGVOPT >= 3) THEN
       ! Restore old step (just as a total energy check).
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             FACT=ONE/GAMMA(I+NATOM3)
             XNEW(I)=VX(I)*FACT-XOLD(I)
             YNEW(I)=VY(I)*FACT-YOLD(I)
             ZNEW(I)=VZ(I)*FACT-ZOLD(I)
             FDNEW(I)=VFD(I)*FACT-FDOLD(I)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
       IF(IDYNPR == 0.OR.JHSTRT.EQ.0) THEN
          !
          ! Get previous energy for printing (just as a check)
          CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
          !
          IF(QHOLO) THEN
             ! Add shake "force" to the internal virial.
             DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                   FACT=GAMMA(I+NATOM)
                   ALPHA=GAMMA(I+NATOM2)
                   IF (GAMMA(I+NATOM) /= ZERO) THEN
                      RVAL=ONE/GAMMA(I+NATOM)
                   ELSE
                      RVAL=ZERO
                   ENDIF
                   X(I)=(ALPHA*XNEW(I)-DX(I)*FACT-XOLD(I))*RVAL
                   Y(I)=(ALPHA*YNEW(I)-DY(I)*FACT-YOLD(I))*RVAL
                   Z(I)=(ALPHA*ZNEW(I)-DZ(I)*FACT-ZOLD(I))*RVAL
                   !                 FDIM(I)=(ALPHA*FDNEW(I)-DFDIM(I)*FACT-FDOLD(I))*RVAL
#if KEY_PARASCAL==1
                ENDIF
#endif 
             ENDDO
             CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,X,Y,Z)
          ENDIF
       ENDIF
    ENDIF
    !
    !=======================================================================
    !
    IF(IGVOPT == 2) THEN
       !
       ! This is the 2-step VERLET
       !    X,Y,Z          - current coordinates
       !    XOLD,YOLD,ZOLD - ignored
       !    VX,VY,VZ       - velocity to use
       !
       IF(QHOLO) THEN
          ! Do holonomic constraints just in case coordinates don't fit.
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
#if KEY_PARALLEL==1
          CALL VDGBR(X,Y,Z,0)
#endif 
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XCOMP(I)=X(I)
                YCOMP(I)=Y(I)
                ZCOMP(I)=Z(I)
                IF(QSHAK4) FDCOMP(I)=FDIM(I)
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
       ENDIF

       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       !
       IF(ILANG == 1) CALL DLNGV(GAMMA,ISEED)
       IGVOPT=3
       !
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             IF (IMOVE(I) == 0) THEN
                FACT=DELTAS*GAMMA(I+NATOM2)/AMASS(I)
                ALPHA=2.0*GAMMA(I+NATOM2)*GAMMA(I+NATOM3)*DELTA2
                XOLD(I)=VX(I)*ALPHA-DX(I)*FACT
                YOLD(I)=VY(I)*ALPHA-DY(I)*FACT
                ZOLD(I)=VZ(I)*ALPHA-DZ(I)*FACT
                IF (IMOVE4(I) == 0) THEN
                   FDOLD(I)=VFD(I)*ALPHA-DFDIM(I)*FACT
                ELSE
                   FDOLD(I)=ZERO
                ENDIF
                XNEW(I)=VX(I)*ALPHA+DX(I)*FACT
                YNEW(I)=VY(I)*ALPHA+DY(I)*FACT
                ZNEW(I)=VZ(I)*ALPHA+DZ(I)*FACT
                IF (IMOVE4(I) == 0) THEN
                   FDNEW(I)=VFD(I)*ALPHA+DFDIM(I)*FACT
                ELSE
                   FDNEW(I)=ZERO
                ENDIF
             ELSE
                XOLD(I)=ZERO
                YOLD(I)=ZERO
                ZOLD(I)=ZERO
                FDOLD(I)=ZERO
                XNEW(I)=ZERO
                YNEW(I)=ZERO
                ZNEW(I)=ZERO
                FDNEW(I)=ZERO
             ENDIF
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
       ! Process holonomic constraints if requested
       IF(QHOLO) THEN
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                X(I)=XCOMP(I)-XNEW(I)
                Y(I)=YCOMP(I)-YNEW(I)
                Z(I)=ZCOMP(I)-ZNEW(I)
                IF(QSHAK4) FDIM(I)=FDCOMP(I)-FDNEW(I)
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XNEW(I)=XCOMP(I)-X(I) 
                YNEW(I)=YCOMP(I)-Y(I)
                ZNEW(I)=ZCOMP(I)-Z(I)
                X(I)=XCOMP(I)+XOLD(I)
                Y(I)=YCOMP(I)+YOLD(I)
                Z(I)=ZCOMP(I)+ZOLD(I)
                !
                IF(QSHAK4) THEN
                   FDNEW(I)=FDCOMP(I)-FDIM(I)
                   FDIM(I)=FDCOMP(I)+FDOLD(I)
                ENDIF
                !
                IF (GAMMA(I+NATOM) /= ZERO) THEN
                   FACT=MINONE/GAMMA(I+NATOM)
                ELSE
                   FACT=ZERO
                ENDIF
                VX(I)=ALPHA*XNEW(I)-DX(I)*FACT
                VY(I)=ALPHA*YNEW(I)-DY(I)*FACT
                VZ(I)=ALPHA*ZNEW(I)-DZ(I)*FACT
                IF(QSHAK4) THEN
                   VFD(I)=(FDOLD(I)-VFD(I))*FACT
                ENDIF
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XOLD(I)=X(I)-XCOMP(I)
                YOLD(I)=Y(I)-YCOMP(I)
                ZOLD(I)=Z(I)-ZCOMP(I)
                !              FDOLD(I)=FDIM(I)-FDCOMP(I)
                IF (GAMMA(I+NATOM) /= ZERO) THEN
                   FACT=MINONE/GAMMA(I+NATOM)
                ELSE
                   FACT=ZERO
                ENDIF
                VX(I)=(XOLD(I)-VX(I))*FACT
                VY(I)=(YOLD(I)-VY(I))*FACT
                VZ(I)=(ZOLD(I)-VZ(I))*FACT
                IF(QSHAK4) VFD(I)=(FDOLD(I)-VFD(I))*FACT
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          ! Add shake "force" to the internal virial.
          ! ---->don't know if I need to add the 4th D into this yet??? ---Eisenmesser
          CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ)
          !
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                FACT=GAMMA(I+NATOM3)
                VX(I)=(XNEW(I)+XOLD(I))*FACT
                VY(I)=(YNEW(I)+YOLD(I))*FACT
                VZ(I)=(ZNEW(I)+ZOLD(I))*FACT
                IF(QSHAK4) VFD(I)=(FDNEW(I)+FDOLD(I))*FACT
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
       ENDIF
       !
       EPROP(PEPR) = FMARK
       EPROP(KEPR2) = FMARK
       IF(QCNSTP) THEN
          EPROP(XTLKP2) = FMARK
          EPROP(XTLKEP) = FMARK
          EPROP(XTLPEP) = FMARK
       ENDIF
       !
    ENDIF
    !
    !=======================================================================
    ! Generic code for both 3-step and 2-step continuation
    !
    ! compute initial portion of the kinetic energy
    ! compute temperature
    !
    IF (IDYNPR == 0 .OR. JHSTRT.EQ.0) THEN
       TOTKEO=ZERO
       TOTKEN=ZERO
       TEMPI=ZERO
       TOTK4O=ZERO
       TOTK4N=ZERO
       TEMPI4=ZERO
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             TOTKEO=TOTKEO+AMASS(I)*(XOLD(I)**2+YOLD(I)**2+ZOLD(I)**2)
             TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
             TEMPI=TEMPI+AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
             !    4-D kinetic and tem
             TOTK4O=TOTK4O+AMASS(I)*(FDOLD(I)**2)
             TOTK4N=TOTK4N+AMASS(I)*(FDNEW(I)**2)
             TEMPI4=TEMPI4+AMASS(I)*(VFD(I)**2)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       TOTKEO=TOTKEO+TOTK4O
       TOTKEN=TOTKEN+TOTK4N
       !
       ! . Calculate the RMS gradient and work.
       EWRK = ZERO
       RVAL = ZERO
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             EWRK=EWRK+DX(I)*VX(I)+DY(I)*VY(I)+DZ(I)*VZ(I) &
                  +DFDIM(I)*VFD(I)
             RVAL=RVAL+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I) &
                  +DFDIM(I)*DFDIM(I)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       EWRK=EWRK*DELTA
       !
#if KEY_PARALLEL==1
       GCARR(1)=TOTKEO
       GCARR(2)=TOTKEN
       GCARR(3)=TEMPI
       GCARR(4)=EWRK
       GCARR(5)=RVAL
       CALL GCOMB(GCARR,5)
       TOTKEO=GCARR(1)
       TOTKEN=GCARR(2)
       TEMPI=GCARR(3)
       EWRK=GCARR(4)
       RVAL=GCARR(5)
#endif 
       !      
       !      write(6,88) rval,ndegf,ndeg4
       !  88  format(' rval=',D20.8,'   ndegfs=',2I10)
       !
       EPROP(GRMS) = SQRT(RVAL/(NDEGF+NDEG4))
       !C         EPROP(WRKE) = EWRK
       !
       ! Calculate the pressures.
       CALL PRSEXT
       CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
            XNEW,YNEW,ZNEW,DELTA)
       !
       ! WRITE OUT INITIAL CONDITIONS
       !
       ISTEP=ISTART-1
       TIME=TIMFAC*DELTA*NPRIV
       !
       TOTKEN = HALF*TOTKEN/DELTA2
       TOTKEO = HALF*TOTKEO/DELTA2
       EPROP(HFCKE) = HALF * (TOTKEO+TOTKEN)
       EPROP(TOTKE) = (TEMPI+TEMPI4)*HALF
       IF(EPROP(KEPR2) == FMARK) EPROP(KEPR2)=TOTKEN + EWRK
       IF(EPROP(PEPR) == FMARK) EPROP(PEPR)=EPROP(EPOT) - EWRK
       EPROP(EHFC) = (TOTKEN - EPROP(TOTKE) + EPROP(PEPR) - &
            EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEO)/TWELVE
       !
       EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
       EPROP(TOTE)  = EPROP(EPOT) + EPROP(TOTKE)
       EPROP(TEMPS) = TEMPI / (KBOLTZ*NDEGF)
       !
#if KEY_FLUCQ==1
       ! Calculate charge velocities, and the FlucQ contribution to energy
       IF (QFLUC) CALL FQCNEW(0,NPRINT,ATFRST,ATLAST)
#endif 
       IF(QCNSTP) THEN
          IF(EPROP(XTLPEP) == FMARK) EPROP(XTLPEP) = EPROP(PEPR)
          IF(EPROP(XTLKP2) == FMARK) EPROP(XTLKP2) = EPROP(KEPR2)
       ENDIF
       EPROP(TEPR)  = EPROP(HFCTE)
       EPROP(PEPR)  = EPROP(EPOT)
       EPROP(KEPR2) = TOTKEN
       EPROP(KEPR)  = TOTKEO
       !    4-D kinetic and temp
       EPROP(TOTK4) = TEMPI4*HALF
       EPROP(TOT4)  = EPROP(EPOT4) + EPROP(TOTK4)
       EPROP(TEM4) = TEMPI4/ (KBOLTZ*NDEG4)
       !
       IF(PRNLEV > 2) THEN
          CALL PRINTE(OUTU, EPROP, ETERM,'DYNA','DYN', &
               .TRUE.,ISTEP, TIME, ZERO, .TRUE.)
          !adm jr.
          IF (QCNSTP) THEN
             GNORM = SQRT(DOTVEC(DXTL,DXTL,XDIM) / XDIM)
             CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                  .TRUE.,EPRESS)
          ENDIF
       ENDIF
    ENDIF
    !
    ! Initialize accum variables
    IF(ISTOP < ISTART) CALL DIE
    IF(MOD(IST1,IPRFRQ) == 0) THEN
       call avfl_reset()
       FITA = ZERO
       ! Unit cell and gradient terms
       if (QCNSTP) call avfl_ucell_reset()
    ENDIF
    ! Initialize accum variables
    IF (JHSTRT == 0) THEN
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             VK(I)=ZERO
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       JHTEMP=ZERO
       FITP=ZERO
       call avfl_reset_lt()
       ! Unit cell and gradient terms for long term averages
       if (QCNSTP) call avfl_ucell_reset_lt()
    ENDIF
    !
    !=======================================================================
    ! THIS IS THE MAIN LOOP FOR DYNAMICS.
    !
    ! within 3 step Verlet at this point;
    !    X,Y,Z          - scratch array for coordinates
    !    XCOMP,YCOMP,ZCOMP - current coordinates  (Xn-1)
    !    XOLD,YOLD,ZOLD - next step vector        (Un-half)
    !    XNEW,YNEW,ZNEW - previous step vector    (Un-3halves)
    !    VX,VY,VZ       - Filled with velocities  (Vn-1)
    !    GAMMA(NATOM,1) - RDF (std.dev. of random force)
    !                   SQRT(2.0*AMASS(I)*GAM*KBT)/DELTA
    !    GAMMA(NATOM,2) - BETA  ( dx scale factor)
    !                   DELTA*DELTA/((ONE+GAM*HALF)*AMASS(I))
    !    GAMMA(NATOM,3) - ALPHA ( x-xold scale factor)
    !                   (ONE-GAM*HALF)/(ONE+GAM*HALF)
    !    GAMMA(NATOM,4) - Velocity compute scale factor
    !                   HALF*SQRT(ONE+GAM*HALF)/DELTA
    !
    DO ISTEP=ISTART,ISTOP
!QC_Ito_UW0616
#if KEY_RXNCOR==1
  UMBMDSTEP=ISTEP
#endif

       !
       FCOUNT=ISTEP
#if KEY_PARALLEL==1
       CALL PSYNC()
       TIMMER=ECLOCK()
#endif 
       JHSTRT=JHSTRT+1
       NPRIV=NPRIV+1
       !
#if KEY_FLUCQ==1
       ! Compute new charges
       IF (QFLUC) CALL FQCHPR(ISTEP)
#endif 
       ! Compute new atom positions
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             X(I)=XCOMP(I)+XOLD(I)
             Y(I)=YCOMP(I)+YOLD(I)
             Z(I)=ZCOMP(I)+ZOLD(I)
             FDIM(I)=FDCOMP(I)+FDOLD(I)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
       ! If constant pressure is enabled
       IF (QCNSTP) THEN
          !
          !RCZ 92/06/01 - modify REFP if pressure was requested to be linearly
          !               modified i.e. if REFPI /= REFPF
          ! (in progress
          ! - I am not quite sure how to calculate FACT in the case of restart)
          IF(QCNSTPL) THEN
             FACT=NPRIV-NPRIVOLD
             IF(NSTEP /= 0) FACT=FACT/NSTEP
             DO I=1,6
                REFP(I)=(ONE-FACT)*REFPI(I)+FACT*REFPF(I)
             ENDDO
             IF(PRNLEV > 5) THEN
                WRITE(OUTU,'(A,3I5,F10.5)') &
                     ' ISTART,ISTOP,ISTEP,FACT=',ISTART,ISTOP,ISTEP,FACT
                WRITE(OUTU,'(A,6F8.1)') ' REFP=',REFP
             ENDIF
          ENDIF
          !
          CALL PSCALE(X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
               XNEW,YNEW,ZNEW,XCOMP,YCOMP,ZCOMP,GAMMA,ISKP,NDEGF,NPRIV)
       ENDIF
       !
#if KEY_PARALLEL==1
       TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
       CALL PSYNC()
       TIMMER=ECLOCK()
       CALL VDGBR(X,Y,Z,0)
       TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
       CALL PSYNC()
       TIMMER=ECLOCK()
#endif 
       !
       IF(QAVER) THEN
          NAVER=NAVER+1
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XAVE(I)=XAVE(I)+X(I)
                YAVE(I)=YAVE(I)+Y(I)
                ZAVE(I)=ZAVE(I)+Z(I)
                FDAVE(I)=FDAVE(I)+FDIM(I)
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
       ENDIF
       !
       IF(NSAVC > 0) THEN
          IF (MOD(ISTEP,NSAVC) == 0) THEN
             IF(QAVER) THEN
                DNUM=ONE / NAVER
                DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                   IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                      XAVE(I)=XAVE(I)*DNUM
                      YAVE(I)=YAVE(I)*DNUM
                      ZAVE(I)=ZAVE(I)*DNUM
                      FDAVE(I)=FDAVE(I)*DNUM
#if KEY_PARASCAL==1
                   ENDIF
#endif 
                ENDDO
                !
#if KEY_PARALLEL==1
                CALL VDGBR(XAVE,YAVE,ZAVE,1)
                CALL VDGBRE(FDAVE,IPARPT)
#endif 
                IF(IOLEV > 0) THEN
                   CALL WRITCV(XAVE,YAVE,ZAVE, &
#if KEY_CHEQ==1
                        (/ ZERO /), .FALSE., &  
#endif
                        NATOM,FREEAT,NFREAT, &
                        NPRIV,ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                        NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), DIM4,FDAVE)
                ENDIF
                NAVER=0
                DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                   IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                      XAVE(I)=ZERO
                      YAVE(I)=ZERO
                      ZAVE(I)=ZERO
                      FDAVE(I)=ZERO
#if KEY_PARASCAL==1
                   ENDIF
#endif 
                ENDDO
             ELSE
                IF(IOLEV > 0) THEN
                   CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                        (/ ZERO /), .FALSE., &  
#endif
                        NATOM,FREEAT,NFREAT,NPRIV, &
                        ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                        NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), DIM4,FDIM)
                ENDIF
             ENDIF
          ENDIF
       ENDIF
       !
       ! Do list updates if appropriate
       CALL UPDECI(ISTEP-1,X,Y,Z,WMAIN, &
            2,XOLD,YOLD,ZOLD,(/zero/),(/zero/),(/zero/))
       !
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       !
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             XCOMP(I)=X(I)
             YCOMP(I)=Y(I)
             ZCOMP(I)=Z(I)
             FDCOMP(I)=FDIM(I)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
#if KEY_PARALLEL==1
       CALL PSYNC()
       TIMMER=ECLOCK()
#endif 
       !
       ! compute u(n+1/2)
       IF(ILANG == 1) CALL DLNGV(GAMMA,ISEED)
       !
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             FACT=GAMMA(I+NATOM)
             ALPHA=GAMMA(I+NATOM2)
             XNEW(I)=ALPHA*XOLD(I)-DX(I)*FACT
             YNEW(I)=ALPHA*YOLD(I)-DY(I)*FACT
             ZNEW(I)=ALPHA*ZOLD(I)-DZ(I)*FACT
             IF (IMOVE4(I) == 0) FDNEW(I)=ALPHA*FDOLD(I)-DFDIM(I)*FACT
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
       ! Save coordinates for SHAKE (if present).
       IF(QHOLO) THEN
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                VX(I)=XNEW(I)
                VY(I)=YNEW(I)
                VZ(I)=ZNEW(I)
                X(I)=XNEW(I)+XCOMP(I)
                Y(I)=YNEW(I)+YCOMP(I)
                Z(I)=ZNEW(I)+ZCOMP(I)
                IF(QSHAK4) THEN
                   VFD(I)=FDNEW(I)
                   FDIM(I)=FDNEW(I)+FDCOMP(I)
                ENDIF
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          ! Process holonomic contraints if present (not the most efficient way).
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
          !
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XNEW(I)=X(I)-XCOMP(I)
                YNEW(I)=Y(I)-YCOMP(I)
                ZNEW(I)=Z(I)-ZCOMP(I)
                IF(QSHAK4) FDNEW(I)=FDIM(I)-FDCOMP(I)
                IF (GAMMA(I+NATOM) /= ZERO) THEN
                   FACT=MINONE/GAMMA(I+NATOM)
                ELSE
                   FACT=ZERO
                ENDIF
                VX(I)=(XNEW(I)-VX(I))*FACT
                VY(I)=(YNEW(I)-VY(I))*FACT
                VZ(I)=(ZNEW(I)-VZ(I))*FACT
                IF(QSHAK4) VFD(I)=(FDNEW(I)-VFD(I))*FACT
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          ! Add shake "force" to the internal virial.
          ! ----->Again, not sure if I need to add 4th D?????--Eisenmesser
          CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XCOMP,YCOMP,ZCOMP,VX,VY,VZ)
       ENDIF
       !
       ! Constant temperature scaling if requested.
       IF(QCNSTT) THEN
          TEMPI=ZERO
          TOTKEN=ZERO
          TEMPI4=ZERO
          TOTK4N=ZERO
          DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                FACT=GAMMA(I+NATOM3)
                VXI=(XNEW(I)+XOLD(I))*FACT
                VYI=(YNEW(I)+YOLD(I))*FACT
                VZI=(ZNEW(I)+ZOLD(I))*FACT
                VFDI=(FDNEW(I)+FDOLD(I))*FACT
                TEMPI=TEMPI+AMASS(I)*(VXI**2+VYI**2+VZI**2)
                TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
                TEMPI4=TEMPI4+AMASS(I)*(VFDI**2)
                TOTK4N=TOTK4N+AMASS(I)*(FDNEW(I)**2)
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          TOTKEN=TOTKEN+TOTK4N
          TEMPI = TEMPI / (KBOLTZ*NDEGF)
          TOTKEN = TOTKEN / (KBOLTZ*(NDEGF+NDEG4)*DELTA2)
          TEMPI4= TEMPI4/ (KBOLTZ*NDEG4)
          TOTK4N = TOTK4N / (KBOLTZ*NDEG4*DELTA2)
#if KEY_PARALLEL==1
          GCARR(1)=TOTKEN
          GCARR(2)=TEMPI
          CALL GCOMB(GCARR,2)
          TOTKEN=GCARR(1)
          TEMPI=GCARR(2)
#endif 
          IF(TOTKEN < ONE) TOTKEN=ONE
          !C         FACT=SQRT(ONE+(TIMEST/TCOUPL)*(TREF-TEMPI)/(TOTKEN))
          !C    Not sure this is correct:
          !C         FACT=SQRT(ONE+(TIMEST/TCOUPL)*(TREF-TEMPI-TEMPI4)/(TOTKEN))
          FACT=SQRT(ONE+(TIMEST/TCOUPL)*(TREF-TEMPI)/(TOTKEN-TOTK4N))
          !
          DO I = ATFRST,ATLAST
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                XNEW(I) = FACT * XNEW(I)
                YNEW(I) = FACT * YNEW(I)
                ZNEW(I) = FACT * ZNEW(I)
                FDNEW(I) = FACT * FDNEW(I)
#if KEY_PARASCAL==1
             ENDIF
#endif 
          ENDDO
          !
          ! Process holonomic constraints if present (not the most efficient way).
          IF(QHOLO) THEN
             DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                   X(I)=XNEW(I)+XCOMP(I)
                   Y(I)=YNEW(I)+YCOMP(I)
                   Z(I)=ZNEW(I)+ZCOMP(I)
                   IF(QSHAK4) FDIM(I)=FDNEW(I)+FDCOMP(I)
#if KEY_PARASCAL==1
                ENDIF
#endif 

             ENDDO
             CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
             DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                   XNEW(I)=X(I)-XCOMP(I)
                   YNEW(I)=Y(I)-YCOMP(I)
                   ZNEW(I)=Z(I)-ZCOMP(I)
                   IF(QSHAK4) FDNEW(I)=FDIM(I)-FDCOMP(I)
#if KEY_PARASCAL==1
                ENDIF
#endif 
             ENDDO
          ENDIF
       ENDIF
       !
       ! compute the velocities
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             FACT=GAMMA(I+NATOM3)
             VX(I)=(XNEW(I)+XOLD(I))*FACT
             VY(I)=(YNEW(I)+YOLD(I))*FACT
             VZ(I)=(ZNEW(I)+ZOLD(I))*FACT
             VFD(I)=(FDNEW(I)+FDOLD(I))*FACT
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
       ! compute temperature (2*ke) and  virial contribution (ke)
       TEMPI=ZERO
       TOTKEN=ZERO
       TEMPI4=ZERO
       TOTK4N=ZERO
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             RVAL=AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
             RVAL4=AMASS(I)*(VFD(I)**2)
             VK(I)=VK(I)+RVAL+RVAL4
             TEMPI=TEMPI+RVAL
             TEMPI4=TEMPI4+RVAL4
             TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2)
             TOTK4N=TOTK4N+AMASS(I)*(FDNEW(I)**2)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       TOTKEN=TOTKEN+TOTK4N
       !
       ! . Calculate the RMS gradient and work.
       EWRK = ZERO
       RVAL = ZERO
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             EWRK = EWRK+DX(I)*VX(I)+DY(I)*VY(I)+DZ(I)*VZ(I)+ &
                  DFDIM(I)*VFD(I)
             RVAL = RVAL+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)+ &
                  DFDIM(I)*DFDIM(I)
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       EWRK=EWRK*DELTA
       !
#if KEY_PARALLEL==1
       GCARR(1)=TOTKEN
       GCARR(2)=TEMPI
       GCARR(3)=EWRK
       GCARR(4)=RVAL
       CALL GCOMB(GCARR,4)
       TOTKEN=GCARR(1)
       TEMPI=GCARR(2)
       EWRK=GCARR(3)
       RVAL=GCARR(4)
       !
#endif 
       !
       !
       !      write(6,88) rval,ndegf,ndeg4
       !
       !
       EPROP(GRMS) = SQRT(RVAL/(NDEGF+NDEG4))
       !      EPROP(GRMS) = SQRT(RVAL/NDEGF)
       !C      EPROP(WRKE) = EWRK
       !
       ! Calculate the pressures.
       CALL PRSEXT
       CALL PRSINT(NATOM,AMASS,XOLD,YOLD,ZOLD,DELTA, &
            XNEW,YNEW,ZNEW,DELTA)
       !
       TOTKEN = HALF*TOTKEN/DELTA2
       EPROP(HFCKE) = HALF * (TOTKEN + EPROP(KEPR))
       EPROP(TOTKE)= (TEMPI+TEMPI4)*HALF
       ! compute total energy with correct high frequency correction
       !C      EPROP(EHFC)  = -(EPROP(TOTKE) - EPROP(HFCKE))/THREE
       EPROP(EHFC) = (EPROP(KEPR) - EPROP(TOTKE) + EPROP(PEPR) - &
            EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEN)/TWELVE
       !
       !      EPROP(GRMS)  = (EPROP(KEPR2) + 4*EPROP(PEPR) +6*EPROP(KEPR)+
       !     &                4*EPROP(EPOT) + TOTKEN )/8.0
       !
       EPROP(HFCTE) = EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
       EPROP(TOTE)  = EPROP(EPOT) + EPROP(TOTKE)
       !
       EPROP(KEPR2) = EPROP(KEPR)
       EPROP(KEPR)  = TOTKEN
       EPROP(TOTK4) = TEMPI4*HALF
       EPROP(TOT4)  = EPROP(EPOT4) + EPROP(TOTK4)
#if KEY_FLUCQ==1
       ! Calculate charge velocities, and the FlucQ contribution to energy
       IF (QFLUC) CALL FQCNEW(ISTEP,NPRINT,ATFRST,ATLAST)
#endif 
       !
       !     CHECK THE CHANGE IN TOTAL ENERGY TO BE SURE EVERYTHING IS OK.
       !
       IF(.NOT.DIM4) THEN
          IF (JHSTRT > 2) THEN
             IF (ABS(EPROP(TEPR)-EPROP(HFCTE)) >  &
                  MAX(ECHECK,PTONE*EPROP(HFCKE))) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,2000) &
                     ECHECK,EPROP(TEPR),EPROP(HFCTE),EPROP(HFCKE)
2000            FORMAT(' TOTAL ENERGY CHANGE EXCEDED'/G12.2, &
                     ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY IN THE LAST STEP'/ &
                     ' PREVIOUS E =',G14.4,' CURRENT E =',G14.4,' KINETIC =',G14.4)
                !
                ! Write out restart file with bad coordinates in it. Use the previous
                !  position instead of the velocity.  Then get out!
                DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
                   IF(JPBLOCK(I) == MYNOD) THEN
#endif 
                      VX(I)=XCOMP(I)-XOLD(I)
                      VY(I)=YCOMP(I)-YOLD(I)
                      VZ(I)=ZCOMP(I)-ZOLD(I)
                      VFD(I)=FDCOMP(I)-FDOLD(I)
                      X(I)=XCOMP(I)
                      Y(I)=YCOMP(I)
                      Z(I)=ZCOMP(I)
                      FDIM(I)=FDCOMP(I)
#if KEY_PARASCAL==1
                   ENDIF
#endif 
                ENDDO
#if KEY_PARALLEL==1
                CALL VDGBR(XOLD,YOLD,ZOLD,1)
                CALL VDGBR(VX,VY,VZ,1)
#endif 
                IF(IUNWRI > 0 .AND. IOLEV.GT.0) THEN
                   CALL WRIDYN(IUNWRI,NATOM,X,Y,Z,XOLD,YOLD,ZOLD,VX,VY,VZ, &
#if KEY_CHEQ==1
                        (/ ZERO /), (/ ZERO /), (/ ZERO /), .FALSE., & 
#endif
                        ! PJ 06/2005
#if KEY_PIPF==1
                        uind0, uind0, uind0, .FALSE.,    & 
#endif
#if KEY_PIPF==1
                        0, (/ ZERO /), (/ ZERO /),       & 
#endif
#if KEY_DYNVV2==1
                        .FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), & 
#endif
                        NPRIV,JHSTRT,NDEGF,NSTEP, &
                        NSAVC,NSAVV,ZERO,ZERO,0,-1 &
#if KEY_BLOCK==1 /*ldm*/
                        ,.FALSE., .FALSE., &
                        0, (/ ZERO /), (/ ZERO /), (/ ZERO /), 0 &
#endif 
                   ,VFD,FDOLD &
#if KEY_SCCDFTB==1
                   ,qlamda,qpkac,icntdyn,iavti,dvdl,dvdlav,dtmp1 &
#endif 
                   )
                ENDIF
                !
                CALL WRNDIE(-2,'<DYNAMC>', &
                     'ENERGY CHANGE TOLERANCE EXCEEDED')
                RUNOK=.FALSE.
                RETURN
             ENDIF
          ENDIF
       ENDIF
       !
       !
       EPROP(TEPR) = EPROP(HFCTE)
       EPROP(PEPR) = EPROP(EPOT)
       !
       TIME   = TIMFAC*DELTA*NPRIV
       EPROP(TEMPS) = TEMPI / (KBOLTZ*NDEGF)
       EPROP(TEM4) = TEMPI4 / (KBOLTZ*NDEG4)
       JHTEMP = JHTEMP + EPROP(TEMPS)
       J4TEMP = J4TEMP + EPROP(TEM4)
       J4STRT = J4STRT + 1
       call avfl_update(EPROP, ETERM, EPRESS)
       ! Unit cell and gradient terms
       if (QCNSTP) call avfl_ucell_update()
       !
       ISTPSA=MOD(IST1,IPRFRQ)+ISTEP-IST1
       FITA = FITA + ISTPSA * EPROP(HFCTE)
       FITP = FITP + JHSTRT * EPROP(HFCTE)
       !
       IF(NSAVV > 0 .AND. IOLEV.GT.0) THEN
          IF(MOD(ISTEP,NSAVV) == 0) THEN
             CALL WRITCV(VX,VY,VZ, &
#if KEY_CHEQ==1
                  (/ ZERO /), .FALSE., &  
#endif
                  NATOM,FREEAT,NFREAT,NPRIV,ISTEP, &
                  NDEGF,DELTA,NSAVV,NSTEP,TITLEA,NTITLA, &
                  IUNVEL,.TRUE.,.FALSE., (/ 0 /), DIM4,VFD)
          ENDIF
       ENDIF
       !
       ! . Adaptive umbrella sampling, write out umbrella coordinate
#if KEY_ADUMB==1
       IF((NSAVC > 0).AND.(WCUNUM.GT.0)) THEN
          IF (MOD(ISTEP,NSAVC) == 0) THEN
             CALL UMWRUC(ISTEP)
          ENDIF
       ENDIF
#endif 
      !GAMUS, write out reaction coordinates
#if KEY_GAMUS==1
      IF((DOGAMUS).AND.(IGAMUSW.GT.0))THEN
         IF(MOD(ISTEP,IGAMUSF).EQ.0)THEN
            CALL GAMUSW(GAMUSQ,GAMUSDQ)
         ENDIF
      ENDIF
#endif 
       !
       IF(MOD(ISTEP,NPRINT) == 0) THEN
          IF(PRNLEV > 0) THEN
             CALL PRINTE(OUTU, EPROP, ETERM, 'DYNA', 'DYN', .FALSE., &
                  ISTEP, TIME, ZERO, .TRUE.)
             !adm jr.
             IF (QCNSTP) THEN
                GNORM = SQRT(DOTVEC(DXTL,DXTL,XDIM) / XDIM)
                CALL PRNXTLD(OUTU,'DYNA',XTLTYP,XUCELL,.TRUE.,GNORM, &
                     .TRUE.,EPRESS)
             ENDIF
          ENDIF
          !
          CALL WRETERM(NPRIV,TIME,QKUHEAD)
       ENDIF
       !
       DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) == MYNOD) THEN
#endif 
             FACT=XOLD(I)
             XOLD(I)=XNEW(I)
             XNEW(I)=FACT
             FACT=YOLD(I)
             YOLD(I)=YNEW(I)
             YNEW(I)=FACT
             FACT=ZOLD(I)
             ZOLD(I)=ZNEW(I)
             ZNEW(I)=FACT
             FACT=FDOLD(I)
             FDOLD(I)=FDNEW(I)
             FDNEW(I)=FACT
#if KEY_PARASCAL==1
          ENDIF
#endif 
       ENDDO
       !
#if KEY_PARALLEL==1
       TMERI(TIMDCNTRL)=TMERI(TIMDCNTRL)+ECLOCK()-TIMMER
#endif 
       !
#if KEY_SCCDFTB==1
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 

          if(qsccb.and.(qlamda.or.qpkac)) then
             if((istep >= sccpass).and.(mod((istep-sccpass),sccstep) == 0)) &
                  then
                icntdyn=icntdyn+1
                dvdltmp=  dvdlb+dvdla+dvdlp+dvdle+dvdlv+dvdlscc+dvdlub &
                     +dvdlip
                dvdl=dvdl+dvdltmp
                if(mod(icntdyn,tiav) == 0) then
                   iavti=iavti+1
                   dtmp=dvdl/icntdyn
                   dtmp1=dtmp-dtmp1
                   write(outpka,*) '<dvdl> at ',iavti,'th average = ',dtmp
                   dtmp1=dtmp
                endif
             endif
          endif
#if KEY_PARALLEL==1
       ENDIF
#endif 

#endif 
    ENDDO
    !
    ! THIS IS THE END OF THE MAIN LOOP ! DO ISTEP=ISTART,ISTOP
    !=======================================================================
    !
    ! Before & after the back projection  
    ! the optimal 3-D projection calculated:
    IF((FCOUNT == INC4D).OR.(FCOUNT.EQ.DEC4D)) THEN
       IGVOPT=2 
       CALL INRDYN(X,Y,Z,FDIM,VX,VY,VZ,VFD,NATOM)
    ENDIF
    IF(IHT4 /= 0) THEN
       IF (SIGN(ONE,TIN4) /= SIGN(ONE,FNLT4-FSTT4)) TIN4=-TIN4
       AVETEM4=J4TEMP/J4STRT
       IF(MOD(FCOUNT,IHT4) == 0) THEN
          IF(TIN4 /= 0) THEN
             I=(FNLT4-FSTT4)/TIN4
          ELSE
             I=0
          ENDIF
          J=FCOUNT/IHT4
          TEMNEW4=FSTT4+J*TIN4
          IF(J > I) THEN
             TEMNEW4=FNLT4
             IHT4=0
             IF(PRNLEV >= 2) WRITE(OUTU,2030) NPRIV,FNLT4
2030         FORMAT(' ** INFO ** DYNAMICS CONTROL CHANGED FROM', &
                  ' HEATING IN 4-D TO EQUILIBRATION AT STEP',I6/ &
                  ' 4-DIMENSION EQUILIBRATION TEMPERATURE =',F10.2)
          ENDIF
          IF(TEMNEW4 < 0.0) TEMNEW4=0.0
          !
          IF(IASORS /= 0) THEN
             IF(TEMNEW4 > AVETEM4) TEMNEW4=TEMNEW4+(TEMNEW4-AVETEM4)*0.5
             CALL ASSVEL4(TEMNEW4,FDIM,VFD,AMASS,ISEED,NATOM, &
                  IMOVE,IMOVE4,IASVEL)
          ELSE
             HEAT4=SQRT(TEMNEW4/AVETEM4)
             IF (AVETEM4 == ZERO) HEAT4=ZERO
             CALL SCAVEL4(HEAT4,FDIM,VFD,VK,AMASS,NDEG4,ISCVEL,NATOM, &
                  IMOVE,IMOVE4,AVETEM4,J4STRT,IGVOPT)
          ENDIF
          J4STRT=0
          J4TEMP=ZERO
          TEMPI4=ZERO
          DO I=1,NATOM
             TEMPI4=TEMPI4+AMASS(I)*(VFD(I)**2)
          ENDDO
          EPROP(TOTK4) = TEMPI4*HALF
          EPROP(TOT4)  = EPROP(EPOT4) + EPROP(TOTK4)
          EPROP(TEM4)  = TEMPI4 / (KBOLTZ*NDEG4) 
       ENDIF
       !
       ! Equilibiration is done here.  the velocities are overscaled
       ! by a factor of two to give a correct reequilibration of
       ! harmonic systems when both momentum and pe modes are considered.
    ELSE IF(IEQ4 > 0) THEN
       AVETEM4=J4TEMP/J4STRT
       IF(MOD(FCOUNT,IEQ4) == 0) THEN
          DELTEM=AVETEM4-FNLT4
          IF (ICH4 == 0.OR.DELTEM <= TWL4.OR.DELTEM >= TWH4) THEN
             IF(IASORS /= 0) THEN
                TEMNEW4=FNLT4
                CALL ASSVEL4(TEMNEW4,FDIM,VFD,AMASS,ISEED,NATOM, &
                     IMOVE,IMOVE4,IASVEL)
             ELSE
                HEAT4=SQRT(MAX(TWO*FNLT4/AVETEM4-ONE,ZERO))
                CALL SCAVEL4(HEAT4,FDIM,VFD,VK,AMASS,NDEG4,ISCVEL,NATOM, &
                     IMOVE,IMOVE4,AVETEM4,J4STRT,IGVOPT)
             ENDIF
          ENDIF
          J4STRT=0
          J4TEMP=ZERO
       ENDIF
    ENDIF
#if KEY_PARALLEL==1
    !C      CALL VDGBR(XOLD,YOLD,ZOLD,1)
    !C      CALL VDGBR(VX,VY,VZ,1)
#endif 
    !
    DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) == MYNOD) THEN
#endif 
          X(I)=XCOMP(I)
          Y(I)=YCOMP(I)
          Z(I)=ZCOMP(I)
          FDIM(I)=FDCOMP(I)
#if KEY_PARASCAL==1
       ENDIF
#endif 
    ENDDO
    RUNOK=.TRUE.
    IF (ISTPSA < IPRFRQ) RETURN
    !
    !     Now do long time fluctuations
    !
    call avfl_update_lt()
    ! Unit cell and gradient long terms
    if (QCNSTP) call avfl_ucell_update_lt()
    !
    NUMSTP = ISTPSA
    DNUM   = NUMSTP
    DNM1   = DNUM - ONE
    DNP1   = DNUM + ONE
    DNPM1  = DNM1 * DNP1
    IF(NUMSTP > 1) THEN
       DRIFTA = (TWO*FITA-DNP1*EPRPA(HFCTE))*SIX/(DNPM1*DNUM)
       EAT0A  = EPRPA(HFCTE)/DNUM-DRIFTA*DNP1/TWO
       IF((EPRP2A(HFCTE)/DNUM-(EPRPA(HFCTE)/DNUM)**2) > ZERO) THEN
          CORRA=DNPM1/(EPRP2A(HFCTE)/DNUM-(EPRPA(HFCTE)/DNUM)**2)/ &
               TWELVE
          CORRA=DRIFTA*SQRT(CORRA)
       ELSE
          CORRA=ZERO
       ENDIF
       !
       ! . Compute statistics.
       call avfl_compute(DNUM)
       ! Calculate average unit cell and gradient terms
       if (QCNSTP) call avfl_ucell_compute(DNUM)
       !
    ENDIF
    ! . Print out the results.
    IF(PRNLEV >= 2) THEN
       call avfl_print_aver(NUMSTP, TIME, tag='DYNAMC>')
       if (QCNSTP) call avfl_ucell_print_aver(NUMSTP)
       call avfl_print_fluc(NUMSTP, TIME, tag='DYNAMC>')
       if (QCNSTP) call avfl_ucell_print_fluc(NUMSTP)
    ENDIF
    !
    DRIFTP = DRIFTA
    EAT0P  = EAT0A
    CORRP  = CORRA
    IF (JHSTRT <= NUMSTP) GOTO 190
    NUMSTP = JHSTRT
    DNUM   = NUMSTP
    DNM1   = DNUM - ONE
    DNP1   = DNUM + ONE
    DNPM1  = DNM1 * DNP1
    IF(NUMSTP > 1) THEN
       DRIFTP = (TWO*FITP - DNP1*EPRPP(HFCTE))*SIX/(DNPM1*DNUM)
       EAT0P  = EPRPP(HFCTE)/DNUM-DRIFTP*DNP1/TWO
       IF ((EPRP2P(HFCTE)/DNUM-(EPRPP(HFCTE)/DNUM)**2) > ZERO) THEN
          CORRP=DNPM1/(EPRP2P(HFCTE)/DNUM-(EPRPP(HFCTE)/DNUM)**2)/ &
               TWELVE
          CORRP=DRIFTP*SQRT(CORRP)
       ELSE
          CORRP=ZERO
       ENDIF
    ENDIF
    !
    call avfl_compute_lt(DNUM)
    ! Unit cell and gradient long terms
    if (QCNSTP) call avfl_ucell_compute_lt(DNUM)
    !
    ! . Print out the results.
    IF(PRNLEV >= 2) THEN
       call avfl_print_aver_lt(NUMSTP, TIME, tag='DYNAMC>')
       if (QCNSTP) call avfl_ucell_print_aver_lt(NUMSTP)
       call avfl_print_fluc_lt(NUMSTP, TIME, tag='DYNAMC>')
       if (QCNSTP) call avfl_ucell_print_fluc_lt(NUMSTP)
    ENDIF
    !
190 CONTINUE
    IF(NUMSTP <= 1 .OR. JHSTRT.LE.1 .OR. PRNLEV < 3) RETURN
    WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195 FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
         /5X,'E AT STEP 0            : ',1P,2G17.8, &
         /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
    !
    RETURN
  END SUBROUTINE DYNAMC4

  SUBROUTINE SCAVEL4(SCALE,FDIM,VFD,VK,AMASS, &
       NDEG4,ISCVEL,NATOM,IMOVE,IMOVE4,AVETEM4,J4STRT,IGVOPT)
    !
    !     SCALES VELOCITIES IN 4-D FOR A DYNAMICS RUN.
    !
    !     Authors: S. Swaminathan
    !              Bernie Brooks
    !
    use consta
    use number
    use stream
    real(chm_real) SCALE,AVETEM4
    real(chm_real) FDIM(*)
    real(chm_real) VFD(*),VK(*)
    real(chm_real) AMASS(*),F  
    INTEGER NDEG4,ISCVEL,NATOM
    INTEGER IMOVE(*),IMOVE4(*),J4STRT,IGVOPT
    !

    real(chm_real) T4OLD,SCALF,T4NEW
    INTEGER I
    !
    T4OLD=ZERO
    DO I=1,NATOM
       IF(IMOVE4(I) == 0) THEN
          T4OLD=T4OLD+(VFD(I)**2)*AMASS(I)
          SCALF=SCALE
          IF(ISCVEL /= 0) THEN
             F=SQRT(AVETEM4*(KBOLTZ*J4STRT)/VK(I))
             SCALF=SCALF*F
          ENDIF
          VFD(I)=VFD(I)*SCALF
       ENDIF
    ENDDO


    IGVOPT=2
    T4OLD=T4OLD/(KBOLTZ*NDEG4)
    T4NEW=T4OLD*SCALE*SCALE
    IF(PRNLEV >= 2) WRITE(OUTU,10) SCALE,T4OLD,T4NEW
10  FORMAT(/,5X,'VELOCITIES  IN  4-D  SCALED  -',F10.5,/, &
         5X,'OLD 4-D TEMPERATURE          -',F10.4,/, &
         5X,'NEW 4-D TEMPERATURE          -',F10.4)
    IF (ISCVEL /= 0 .AND. PRNLEV >= 2) WRITE(OUTU,15)
15  FORMAT(5X,'VELOCITIES SCALED DIFFERENTLY FOR EACH ATOM')
    RETURN
  END SUBROUTINE SCAVEL4

  SUBROUTINE ASSVEL4(TEMNEW4,FDIM,VFD, &
       AMASS,ISEED,NATOM,IMOVE,IMOVE4,IASVEL)
    !
    !     PERFORMS VELOCITY ASSIGNMENTS IN 4-D FOR A DYNAMICS RUN.
    !
    !     Authors: S. Swaminathan
    !              Bernie Brooks
    !
    use number
    use consta
    use stream
    !
    real(chm_real) TEMNEW4
    INTEGER ISEED
    real(chm_real) FDIM(*)
    real(chm_real) VFD(*)
    real(chm_real) AMASS(*)
    INTEGER NATOM,IASVEL
    INTEGER IMOVE(*),IMOVE4(*)
    !
    INTEGER I
    real(chm_real) BOLTZ
    !
    real(chm_real) SD,VEL
    !
5   FORMAT(/10X,'SEED FOR RANDOM NUMBER GENERATOR IS ',I12, &
         /10X,'GAUSSIAN OPTION                  IS ',I12, &
         /10X,'4-D  VEL    ASSIGNED  AT  4-D  TEMP ',F12.4)
    IF(PRNLEV >= 2) WRITE(OUTU,5) ISEED,IASVEL,TEMNEW4
    BOLTZ=TEMNEW4*KBOLTZ
    !
    DO I=1,NATOM
       IF(IMOVE4(I) == 0) THEN
          SD=BOLTZ/AMASS(I)
          SD=SQRT(SD)
          CALL GAUSSI(ZERO,SD,VEL,ISEED,IASVEL)
          VFD(I)=VEL
       ELSE
          VFD(I)=ZERO
       ENDIF
    ENDDO
    return
  end SUBROUTINE ASSVEL4
  !
#else /* (4ddynamc)*/
  SUBROUTINE DYNAMC4_NULL
    RETURN
  END SUBROUTINE DYNAMC4_NULL
#endif /* (4ddynamc)*/

end module dyn4

