  SUBROUTINE PRSET(COMLYN,COMLEN,TBATH)
    !
    ! Setup the variables... for the Langevin Piston method
    !       by Bernard R. Brooks and Scott Feller - NIH - June, 1994
    !
    use chm_kinds
    use number
    use dimens_fcm
    use consta
    use contrl
    use energym
    use image
    use eimg
    use mmfp
    use reawri
    use stream
    use string
    use prssre
    implicit none
    !
    INTEGER COMLEN
    character(len=*) COMLYN
    real(chm_real) TBATH, SF(6)
    real(chm_real) PREF, PREFI, PREFF
    !
    real(chm_real) PMASS(3,3), PMASSX, NSURF
    INTEGER I
    !
    ! out of common block
    real(chm_real) PCOMPR, PCOUPL, PGAMMA
    !      PCOMPR  - Isotropic Compressibility
    !      PCOUPL  - Pressure coupling coefficient
    !      PGAMMA  - Pressure friction constant
    !
    real(chm_real)  ROOMP
    PARAMETER (ROOMP = 1.0D0)
    !
    IF(XDIM == 0) THEN
       CALL WRNDIE(-3,'<PRSET>', &
            'CRYStal must be used for constant pressure simulations')
       QCNSTP=.FALSE.
       RETURN
    ENDIF
    !
    IF(PRNLEV >= 2) &
         WRITE(OUTU,'(A)') ' DCNTRL> Constant pressure requested.'
    !
    !-----------------------------------------------------------------------
    IF(INDXA(COMLYN,COMLEN,'BERE')  >  0) THEN
       ! Parse options for berendsen method
       PCOMPR = GTRMF(COMLYN,COMLEN,'COMP',ZERO)
       PCOUPL = GTRMF(COMLYN,COMLEN,'PCOU',ZERO)
       IF(PCOUPL == ZERO) PCOUPL=RBIG
       !
       GAM = ZERO
       PALPHA = ZERO
       PVFACT =  ZERO
       CALL GETVOL(EPROP(VOLUME))
       ! The meaning of this should be transparent...
       PBFACT = TIMEST*PCOMPR/(PCOUPL*ATMOSP*EPROP(VOLUME))
       !C (note: units must cancel - BRB)
       !
       CALL GETXTL(SF,XTLABC,XTLTYP,XTLREF)
       DO I=1,XDIM
          PWMAT(I)=ZERO
          PWINV(I)=SF(I)**2
          PRFWD(I)=ZERO
       ENDDO
       !
       IF(PRNLEV >= 2) WRITE(OUTU,'(2(8X,A,F12.7,A,/))') &
            ' Isothermal compressibility = ',PCOMPR,' atm**-1.', &
            ' Pressure coupling constant = ',PCOUPL,' ps.'
       !
    ELSE 
       !-----------------------------------------------------------------------
       ! Parse options for langevin piston method
       !
       ! Parse piston masses
       !
       PMASSX= GTRMF(COMLYN,COMLEN,'PMAS',ZERO)
       PMASS(1,1)= GTRMF(COMLYN,COMLEN,'PMXX',PMASSX)
       PMASS(2,2)= GTRMF(COMLYN,COMLEN,'PMYY',PMASSX)
       PMASS(3,3)= GTRMF(COMLYN,COMLEN,'PMZZ',PMASSX)
       PMASS(1,2)= GTRMF(COMLYN,COMLEN,'PMXY',PMASSX)
       PMASS(1,3)= GTRMF(COMLYN,COMLEN,'PMXZ',PMASSX)
       PMASS(2,3)= GTRMF(COMLYN,COMLEN,'PMYZ',PMASSX)
       PMASS(2,1)= PMASS(1,2)
       PMASS(3,1)= PMASS(1,3)
       PMASS(3,2)= PMASS(2,3)
       CALL LATTRN(XTLTYP,PMASS,PWMAT,XTLREF)
       DO I=1,XDIM
          IF(PWMAT(I) > ZERO) THEN
             PWINV(I)=ONE/PWMAT(I)
          ELSE 
             PWINV(I)=ZERO
          ENDIF
       ENDDO
       !
       PGAMMA = GTRMF(COMLYN,COMLEN,'PGAM',ZERO)
       KBT = KBOLTZ*TBATH
       GAM = TIMFAC*PGAMMA*DELTA
       PALPHA = (ONE-GAM*HALF)/(ONE+GAM*HALF)
       PBFACT = DELTA*DELTA/((ONE+GAM*HALF))
       PVFACT =  HALF/DELTA
       !
       ! Generate random force scale factor
       DO I=1,XDIM
          PRFWD(I)=SQRT(TWO*PWINV(I)*GAM*KBT)/DELTA
       ENDDO
    ENDIF
    !-----------------------------------------------------------------------
    !
    ! Parse reference pressure matrices
    !
    !RCZ 92/06/01 Initial (PREFI) and final (PREFF) pressures introduced
    !             It will allow for gradual modification of the pressure
    !             during the simulation
    PREFI= GTRMF(COMLYN,COMLEN,'PREFI',FMARK)
    PREFF= GTRMF(COMLYN,COMLEN,'PREFF',FMARK)
    PREF = GTRMF(COMLYN,COMLEN,'PREF',ROOMP)
    IF(PREFI == FMARK) PREFI=PREF
    IF(PREFF == FMARK) PREFF=PREF
    REFP(RPXX)= GTRMF(COMLYN,COMLEN,'PRXX',PREF)
    REFP(RPYY)= GTRMF(COMLYN,COMLEN,'PRYY',PREF)
    REFP(RPZZ)= GTRMF(COMLYN,COMLEN,'PRZZ',PREF)
    REFP(RPXY)= GTRMF(COMLYN,COMLEN,'PRXY',ZERO)
    REFP(RPXZ)= GTRMF(COMLYN,COMLEN,'PRXZ',ZERO)
    REFP(RPYZ)= GTRMF(COMLYN,COMLEN,'PRYZ',ZERO)
    REFPI(RPXX)= GTRMF(COMLYN,COMLEN,'PIXX',PREFI)
    REFPI(RPYY)= GTRMF(COMLYN,COMLEN,'PIYY',PREFI)
    REFPI(RPZZ)= GTRMF(COMLYN,COMLEN,'PIZZ',PREFI)
    REFPI(RPXY)= GTRMF(COMLYN,COMLEN,'PIXY',ZERO)
    REFPI(RPXZ)= GTRMF(COMLYN,COMLEN,'PIXZ',ZERO)
    REFPI(RPYZ)= GTRMF(COMLYN,COMLEN,'PIYZ',ZERO)
    REFPF(RPXX)= GTRMF(COMLYN,COMLEN,'PFXX',PREFF)
    REFPF(RPYY)= GTRMF(COMLYN,COMLEN,'PFYY',PREFF)
    REFPF(RPZZ)= GTRMF(COMLYN,COMLEN,'PFZZ',PREFF)
    REFPF(RPXY)= GTRMF(COMLYN,COMLEN,'PFXY',ZERO)
    REFPF(RPXZ)= GTRMF(COMLYN,COMLEN,'PFXZ',ZERO)
    REFPF(RPYZ)= GTRMF(COMLYN,COMLEN,'PFYZ',ZERO)
    !RCZ 92/06/01 - find out if pressure is going to be modified
    PREF=ZERO
    DO I=1,6
       PREF=PREF+ABS(REFPI(I)-REFPF(I))
    ENDDO
    QCNSTPL=(PREF > PTONE)
    !
    IF(PRNLEV >= 2) THEN
       IF(QCNSTPL) THEN
          WRITE(OUTU,'(8X,A,3F10.1,A,/)') &
               ' Initial   pressure tensor (XX,YY,ZZ)= ', &
               REFPI(RPXX),REFPI(RPYY),REFPI(RPZZ),' atm.', &
               ' Initial   pressure tensor (XY,XZ,YZ)= ', &
               REFPI(RPXY),REFPI(RPXZ),REFPI(RPYZ),' atm.'
          WRITE(OUTU,'(8X,A,3F10.1,A,/)') &
               ' Final     pressure tensor (XX,YY,ZZ)= ', &
               REFPF(RPXX),REFPF(RPYY),REFPF(RPZZ),' atm.', &
               ' Final     pressure tensor (XY,XZ,YZ)= ', &
               REFPF(RPXY),REFPF(RPXZ),REFPF(RPYZ),' atm.'
       ELSE
          WRITE(OUTU,'(8X,A,3F10.1,A,/)') &
               ' Reference pressure tensor (XX,YY,ZZ)= ', &
               REFP(RPXX),REFP(RPYY),REFP(RPZZ),' atm.', &
               ' Reference pressure tensor (XY,XZ,YZ)= ', &
               REFP(RPXY),REFP(RPXZ),REFP(RPYZ),' atm.'
       ENDIF
    ENDIF
    !
    ! Define which pressure to couple (default to internal pressure)
    !
    PTYPE = .NOT.(INDXA(COMLYN,COMLEN,'PINT')  >  0)
    PTYPE = (INDXA(COMLYN,COMLEN,'PEXT')  >  0)
    !
    ! Unit number, freqency for pressure tensor 
    IUPTEN = GTRMI(COMLYN,COMLEN,'IUPT',-1)
    IPTFRQ = GTRMI(COMLYN,COMLEN,'IPTF',-1)
    IF(PRNLEV >= 2.AND.IUPTEN>0) THEN
      IF (IPTFRQ>0) THEN
        WRITE(OUTU,'(8X,A,I4,A,I8,A)') &
          'Writing P tensor to unit ',IUPTEN, &
          ' every ',IPTFRQ,' steps.'
      ELSE
        CALL WRNDIE(-3,'<PRSET>', &
          'IUPTEN set w/o specifying IPTFRQ')
      ENDIF
    ENDIF
    !
    ! How many iterations to achieve consistency
    PITER = 3
    PITER = GTRMI(COMLYN,COMLEN,'PITE',PITER)
    !
    !-----------------------------------------------------------------------
    !
    ! Zero the velocity of the pressure pistons
    IF(IREST  ==  0) THEN
       DO I=1,6
          HDOT(I) = GTRMF(COMLYN,COMLEN,'HDOT',ZERO)
       ENDDO
    ENDIF
    ! Do constant surface tension
    QSURF = (INDXA(COMLYN,COMLEN,'SURF')  >  0)
    SURFT = GTRMF(COMLYN,COMLEN,'TENS',ZERO)
    QCONZ = (INDXA(COMLYN,COMLEN,'ZCON')  >  0)
    NSURF = GTRMF(COMLYN,COMLEN,'NSUR',TWO)
    SURFT = NSURF*SURFT
    ! enable dynamic adjustment of IMXCEN for P21 symmetry
    QP21XCEN = .FALSE.
    QP21XCEN = (INDXA(COMLYN,COMLEN,'P21X')  >  0)
    IF (QP21XCEN.and.(prnlev>2)) &
         WRITE(OUTU,'(8X,A,I5,F12.1,/)') XTLTYP,XNSYMM,PMASS(1,1)
    ! require tetragonal, non-zero PMXX, one added symm op (e.g. P21)
    QP21XCEN = QP21XCEN.AND.(XTLTYP.EQ.'TETR').AND.(PMASS(1,1).GT.0.) &
       .AND.(XNSYMM.EQ.2)
    IF(QP21XCEN) IMXCEN = 0.25*XTLABC(1)
    !
    RETURN
  END SUBROUTINE PRSET

  SUBROUTINE PSCALE(X,Y,Z,XNEW,YNEW,ZNEW,VX,VY,VZ, &
       XOLD,YOLD,ZOLD,XCOMP,YCOMP,ZCOMP,GAMMA,ISKP,NDEGF,NPRIV)
    !
    ! The Langevin Piston Method
    !
    !    by Bernard R. Brooks and Scott Feller - NIH - June, 1994
    !    Based on earlier versions by Martin Field and Ryszard Czerminsky
    !
    use chm_kinds
    use dimens_fcm
    use number
    use consta
    use deriv
    use energym
    use image
    use eimg
    use psf
    use reawri
    use stream
    use mmfp
    use shake
    use clcg_mod
    use holonom,only:holonoma
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:natoml,atoml,q_domdec 
#endif
    use prssre
    implicit none
    !
    real(chm_real) X(*),Y(*),Z(*),XNEW(*),YNEW(*),ZNEW(*)
    real(chm_real) VX(*),VY(*),VZ(*),XOLD(*),YOLD(*),ZOLD(*)
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),GAMMA(*)
    INTEGER ISKP(*)
    INTEGER NDEGF,NPRIV
    !
    !     Local variables.
    !
    INTEGER I, JIT, NJIT, NATOM3
    LOGICAL OK
    real(chm_real)  FACT, SFACT, SFXX, SFYY, SFZZ, SVXX, SVYY, SVZZ
    real(chm_real)  CURPRS,REFPA
    real(chm_real)  SF(6),XTLINV(6),WRK1(3,3),WRK2(3,3),WRK3(3,3), &
         SH(6)
    real(chm_real)  VXF, VYF, VZF
    real(chm_real)  VNP(6), VNM(6), XNP(6), XNM(6)
    real(chm_real)  EGEOI,EGEOF
    real(chm_real)  DELP(3,3), DELPX(6), DELPR(3,3)
    real(chm_real)  HDOTP(6), HDAV(6), AVABC(6)
    real(chm_real)  PMAX, DELTA2, HALFD, RVAL, HDABC(6)
    real(chm_real)  VCELL, RVC, SURFI
    real(chm_real)  XTLOLD(6), HDOLD(6), GCARR(10)
    real(chm_real)  TOTKEN, TOTKEO, HFCTEN, EPDIFF
    real(chm_real)  PNHVP, PNHFP, REFKE
    LOGICAL QOK
    integer i00, i01, ia
    !
    SAVE
    !
    !=======================================================================
    IF(XDIM == 0) RETURN
    !=======================================================================
    !
    ! Calculate the volume at step n-1
    CALL GETVOL(EPROP(VOLUME))
    !
    REFKE = HALF*NDEGF*KBOLTZ*REFT
    NATOM3=NATOM*3
    DELTA2=DELTA**2
    HALFD=HALF/DELTA
    VCELL = PT25*PATMOS/EPROP(VOLUME)/DELTA2
    SFACT = 98.6923
    !                           SFACT converts from dyne/cm to Atm*Angstroms
    !
    !                put h at n-1 into a holding array 
    DO I=1,6
       XTLOLD(I) = XTLABC(I)
       HDOLD(I) = HDOT(I)
    ENDDO
    !
    !  Copy the thermal piston velocity and acceleration at previous step
    PNHVP = PNHV
    PNHFP = PNHF
    !
    SURFI = HALF*XTLOLD(6)*(EPRESS(PIZZ)-HALF*( &
         EPRESS(PIYY)+EPRESS(PIXX)))/SFACT
    !
    IF(QCONZ) THEN
       REFP(RPZZ) = EPRESS(PIZZ)
    ENDIF
    !
    IF(QSURF) THEN
       REFP(RPXX) = REFP(RPZZ) - SFACT*SURFT/XTLABC(6)
       REFP(RPYY) = REFP(RPXX)
    ENDIF
    !
    RVAL = ZERO
    !
    ! Compute pressure difference matrix.
    IF(PTYPE) THEN
       ! use  external pressure
       DELP(1,1) = (EPRESS(PEXX)-REFP(RPXX))
       DELP(1,2) = (EPRESS(PEXY)-REFP(RPXY))
       DELP(1,3) = (EPRESS(PEXZ)-REFP(RPXZ))
       DELP(2,1) = (EPRESS(PEYX)-REFP(RPXY))
       DELP(2,2) = (EPRESS(PEYY)-REFP(RPYY))
       DELP(2,3) = (EPRESS(PEYZ)-REFP(RPYZ))
       DELP(3,1) = (EPRESS(PEZX)-REFP(RPXZ))
       DELP(3,2) = (EPRESS(PEZY)-REFP(RPYZ))
       DELP(3,3) = (EPRESS(PEZZ)-REFP(RPZZ))
    ELSE
       ! use internal pressure
       DELP(1,1) = (EPRESS(PIXX)-REFP(RPXX))
       DELP(1,2) = (EPRESS(PIXY)-REFP(RPXY))
       DELP(1,3) = (EPRESS(PIXZ)-REFP(RPXZ))
       DELP(2,1) = (EPRESS(PIYX)-REFP(RPXY))
       DELP(2,2) = (EPRESS(PIYY)-REFP(RPYY))
       DELP(2,3) = (EPRESS(PIYZ)-REFP(RPYZ))
       DELP(3,1) = (EPRESS(PIZX)-REFP(RPXZ))
       DELP(3,2) = (EPRESS(PIZY)-REFP(RPYZ))
       DELP(3,3) = (EPRESS(PIZZ)-REFP(RPZZ))
       DELPR(1,1) = ZERO
       DELPR(1,2) = ZERO
       DELPR(1,3) = ZERO
       DELPR(2,1) = ZERO
       DELPR(2,2) = ZERO
       DELPR(2,3) = ZERO
       DELPR(3,1) = ZERO
       DELPR(3,2) = ZERO
       DELPR(3,3) = ZERO

#if KEY_DOMDEC==1
       if (q_domdec) then
          i00 = 1
          i01 = natoml
       else
#endif 
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
          i00 = iparpt(mynod)+1
          i01 = iparpt(mynodp)
#elif KEY_SPACDEC==1
          i00=1
          i01=natom
#elif KEY_PARASCAL==1
          i00=1
          i01=natom
#endif 
#else /**/
          i00=1
          i01=natom
#endif 
#if KEY_DOMDEC==1
       endif  
#endif

!$omp parallel do schedule(static) private(ia, i, rvc) reduction(+:delpr,rval)
       do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
          if (q_domdec) then
             i = atoml(ia)
          else
#endif /* (domdec)*/
             i = ia
#if KEY_DOMDEC==1
          endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
          ! Calculate negative of the velocity contribution to the pressure
          RVC=VCELL*AMASS(I)
          DELPR(1,1) = DELPR(1,1) - RVC*XNEW(I)**2
          DELPR(1,2) = DELPR(1,2) - RVC*XNEW(I)*YNEW(I)
          DELPR(1,3) = DELPR(1,3) - RVC*XNEW(I)*ZNEW(I)
          DELPR(2,1) = DELPR(2,1) - RVC*YNEW(I)*XNEW(I)
          DELPR(2,2) = DELPR(2,2) - RVC*YNEW(I)**2
          DELPR(2,3) = DELPR(2,3) - RVC*YNEW(I)*ZNEW(I)
          DELPR(3,1) = DELPR(3,1) - RVC*ZNEW(I)*XNEW(I)
          DELPR(3,2) = DELPR(3,2) - RVC*ZNEW(I)*YNEW(I)
          DELPR(3,3) = DELPR(3,3) - RVC*ZNEW(I)**2
          RVAL = RVAL + AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
          VX(I) = XNEW(I)
          VY(I) = YNEW(I)
          VZ(I) = ZNEW(I)
       ENDDO
!$omp end parallel do
#if KEY_PARALLEL==1
       gcarr(1:3) = delpr(1:3,1)
       gcarr(4:6) = delpr(1:3,2)
       gcarr(7:9) = delpr(1:3,3)
       gcarr(10) = rval
!!$       CALL GCOMB(DELPR,9)
!!$       CALL GCOMB(RVAL,1)
       call gcomb(gcarr, 10)
       delpr(1:3,1) = gcarr(1:3)
       delpr(1:3,2) = gcarr(4:6)
       delpr(1:3,3) = gcarr(7:9)
       rval = gcarr(10)
#endif 
       DELPR(1,1) = DELPR(1,1) + DELP(1,1)
       DELPR(1,2) = DELPR(1,2) + DELP(1,2)
       DELPR(1,3) = DELPR(1,3) + DELP(1,3)
       DELPR(2,1) = DELPR(2,1) + DELP(2,1)
       DELPR(2,2) = DELPR(2,2) + DELP(2,2)
       DELPR(2,3) = DELPR(2,3) + DELP(2,3)
       DELPR(3,1) = DELPR(3,1) + DELP(3,1)
       DELPR(3,2) = DELPR(3,2) + DELP(3,2)
       DELPR(3,3) = DELPR(3,3) + DELP(3,3)
    ENDIF
    !
    ! This loop does iterations to ensure than the corrected velocity is
    ! in agreement with the pressure.  For starters we try 3 iterations.
    ! This should be replaced with a tolerance criterion.
    !
    NJIT=PITER
    IF(PTYPE) NJIT=1
    jit_loop: DO JIT=1,NJIT
       ! 
       DO I=1,6
          XTLABC(I) = XTLOLD(I)
          HDOT(I) = HDOLD(I)
       ENDDO
       !
       ! Compute inverse of h matrix.
       ! We have h at step n-1, this gives h inverse at step n-1
       !
       CALL INVT33S(XTLINV,XTLABC,OK)
       CALL MULNXNFL(WRK1,DELP,XTLINV,3)
       CALL LATTRN(XTLTYP,WRK1,DELPX,XTLREF)
       !
       ! Update the HDOT matrix.
       ! This sets hdotp to hdot at step n-3/2, hdoti is hdot at n-1/2
       ! The force on the piston comes from P at step n-1
       !
       FACT=EPROP(VOLUME)*PBFACT*ATMOSP
       !
       DO I=1,XDIM
          HDOTP(I)=HDOT(I)
          HDOT(I) = PALPHA*HDOT(I) +  &
               PWINV(I)*DELPX(I)*FACT + &
               PBFACT*BMGAUS(PRFWD(I),ISEED)
       ENDDO
       !
       !     Make sure that every process has exactly the same value
#if KEY_PARALLEL==1
       CALL PSND8(HDOT,XDIM)                        
#endif
       !
       DO I=1,XDIM
          HDAV(I) = (HDOTP(I)+HDOT(I))*PVFACT*DELTA
       ENDDO
       !
       ! Propogate the h matrix
       ! This calculates h at step n, using the h velocity we just found
       CALL GETXTL(SF,XTLABC,XTLTYP,XTLREF)
       !
       DO I=1,XDIM
          SF(I)=SF(I)+HDOT(I)
       ENDDO
       !
       !     Make sure that every process has exactly the same value
#if KEY_PARALLEL==1
       CALL PSND8(SF,XDIM)                          
#endif
       !
       DO I=1,XDIM
          SH(I)=SF(I)-HALF*HDOT(I)
       ENDDO
       !
       IF(QCONZ) THEN
          SF(XDIM) = XTLOLD(1)*XTLOLD(3)*XTLOLD(6)/SF(1)**2 
       ENDIF
       !
       ! warning: the isotropic and constant volume options
       ! work for tetragonal and orhtorombic only
       !
       !     this puts the new hvalues into the xtlabc matrix
       CALL SETXTL(HDAV,AVABC,XTLTYP,XTLREF)
       CALL MULNXNLL(WRK2,AVABC,XTLINV,3)
       !
       CALL SETXTL(HDOT,HDABC,XTLTYP,XTLREF)
       CALL PUTXTL(SH,XTLABC,XTLTYP,XTLREF)
       CALL INVT33S(XTLINV,XTLABC,OK)
       CALL MULNXNLL(WRK3,HDABC,XTLINV,3)
       CALL PUTXTL(SF,XTLABC,XTLTYP,XTLREF)
       !
       ! Calculate the thermal piston velocity and position
       IF(QNPT) THEN
          RVAL=HALF*RVAL
          PNHF = TWO*DELTA*(RVAL-REFKE)/TMASS
          IF(PNHFP == ZERO) THEN
             PNHFP = PNHF
          ENDIF
          PNHV = PNHVP + HALF*(PNHF + PNHFP)
          !     Make sure all processes have the same value
#if KEY_PARALLEL==1
          CALL PSND8(PNHV,1)                          
#endif
          !
          RVAL = ZERO
       ENDIF
       !
       !=======================================================================
       ! Scale the coordinates and velocities.
       !
       IF(.NOT.PTYPE) THEN
          ! Calculate corrected pressure difference for next iteration
          DELP(1,1) = ZERO
          DELP(1,2) = ZERO
          DELP(1,3) = ZERO
          DELP(2,1) = ZERO
          DELP(2,2) = ZERO
          DELP(2,3) = ZERO
          DELP(3,1) = ZERO
          DELP(3,2) = ZERO
          DELP(3,3) = ZERO

#if KEY_DOMDEC==1
          if (q_domdec) then
             i00 = 1
             i01 = natoml
          else
#endif 
             i00=1
             i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
             I00 = 1 + IPARPT(MYNOD)
             i01 = IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
          endif  
#endif

!$omp parallel do schedule(static) &
!$omp& private(ia, i, vxf, vyf, vzf, sfxx, sfyy, sfzz, svxx, svyy, svzz, rvc) &
!$omp& reduction(+:rval, delp)
          do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
             if (q_domdec) then
                i = atoml(ia)
             else
#endif /* (domdec)*/
                i = ia
#if KEY_DOMDEC==1
             endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
             ! Modify the forward half step velocity by -(v+f)*hdot/h
             !
             VXF = (XOLD(I) + VX(I)) * HALFD
             VYF = (YOLD(I) + VY(I)) * HALFD
             VZF = (ZOLD(I) + VZ(I)) * HALFD
             !
             SFXX = WRK2(1,1)*VXF &
                  + WRK2(1,2)*VYF &
                  + WRK2(1,3)*VZF
             SFYY = WRK2(2,1)*VXF &
                  + WRK2(2,2)*VYF &
                  + WRK2(2,3)*VZF
             SFZZ = WRK2(3,1)*VXF &
                  + WRK2(3,2)*VYF &
                  + WRK2(3,3)*VZF
             !
             IF(QCONZ) THEN
                SFZZ = -( WRK2(1,1) + WRK2(2,2) )*VZF
             ENDIF
             !
             VX(I) = XNEW(I) - DELTA*SFXX 
             VY(I) = YNEW(I) - DELTA*SFYY
             VZ(I) = ZNEW(I) - DELTA*SFZZ 
             !
             IF(QNPT) THEN
                VX(I) = VX(I) - DELTA2*VXF*PNHV
                VY(I) = VY(I) - DELTA2*VYF*PNHV
                VZ(I) = VZ(I) - DELTA2*VZF*PNHV
                !brb                  VXF = VXF - HALF*(SFXX + VXF*PNHV*DELTA)
                !brb                  VYF = VYF - HALF*(SFYY + VYF*PNHV*DELTA)
                !brb                  VZF = VZF - HALF*(SFZZ + VZF*PNHV*DELTA) 
                RVAL = RVAL + AMASS(I)*(VXF**2 + VYF**2 + VZF**2) 
             ENDIF
             !            
             VXF = (XCOMP(I) + X(I)) * HALF
             VYF = (YCOMP(I) + Y(I)) * HALF
             VZF = (ZCOMP(I) + Z(I)) * HALF
             !             
             SVXX = WRK3(1,1)*VXF &
                  + WRK3(1,2)*VYF &
                  + WRK3(1,3)*VZF
             SVYY = WRK3(2,1)*VXF &
                  + WRK3(2,2)*VYF &
                  + WRK3(2,3)*VZF
             SVZZ = WRK3(3,1)*VXF &
                  + WRK3(3,2)*VYF &
                  + WRK3(3,3)*VZF
             !
             IF(QCONZ) THEN
                SVZZ = -(WRK3(1,1) + WRK3(2,2))*VZF
             ENDIF
             !
             X(I) = VX(I) + SVXX + XCOMP(I)
             Y(I) = VY(I) + SVYY + YCOMP(I)
             Z(I) = VZ(I) + SVZZ + ZCOMP(I)
             !
             ! Calculate positive of the velocity contribution to the pressure
             !  to get new corrected pressure
             RVC=VCELL*AMASS(I)
             DELP(1,1)=DELP(1,1) + RVC*VX(I)**2
             DELP(1,2)=DELP(1,2) + RVC*VX(I)*VY(I)
             DELP(1,3)=DELP(1,3) + RVC*VX(I)*VZ(I)
             DELP(2,1)=DELP(2,1) + RVC*VY(I)*VX(I)
             DELP(2,2)=DELP(2,2) + RVC*VY(I)**2
             DELP(2,3)=DELP(2,3) + RVC*VY(I)*VZ(I)
             DELP(3,1)=DELP(3,1) + RVC*VZ(I)*VX(I)
             DELP(3,2)=DELP(3,2) + RVC*VZ(I)*VY(I)
             DELP(3,3)=DELP(3,3) + RVC*VZ(I)**2
          ENDDO
!$omp end parallel do
#if KEY_PARALLEL==1
!!$          CALL GCOMB(DELP,9)
!!$          CALL GCOMB(RVAL,1)
          gcarr(1:3) = delp(1:3,1)
          gcarr(4:6) = delp(1:3,2)
          gcarr(7:9) = delp(1:3,3)
          gcarr(10) = rval
          call gcomb(gcarr, 10)
          delp(1:3,1) = gcarr(1:3)
          delp(1:3,2) = gcarr(4:6)
          delp(1:3,3) = gcarr(7:9)
          rval = gcarr(10)
#endif 
          DELP(1,1) = DELPR(1,1) + DELP(1,1)
          DELP(1,2) = DELPR(1,2) + DELP(1,2)
          DELP(1,3) = DELPR(1,3) + DELP(1,3)
          DELP(2,1) = DELPR(2,1) + DELP(2,1)
          DELP(2,2) = DELPR(2,2) + DELP(2,2)
          DELP(2,3) = DELPR(2,3) + DELP(2,3)
          DELP(3,1) = DELPR(3,1) + DELP(3,1)
          DELP(3,2) = DELPR(3,2) + DELP(3,2)
          DELP(3,3) = DELPR(3,3) + DELP(3,3)
       ENDIF
    ENDDO jit_loop
    !
    !  Now we have well converged values and we calculate the energies and
    !  move back to the main dynamics loop.  First update the velocities
    !
    IF(.NOT.PTYPE) THEN
       IF(QHOLO) THEN
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,.TRUE.,.TRUE.,QOK)
          !        
#if KEY_DOMDEC==1
          if (q_domdec) then
             i00 = 1
             i01 = natoml
          else
#endif 
             i00=1
             i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
             I00=1+IPARPT(MYNOD)
             i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
          endif  
#endif

!$omp parallel do schedule(static) private(ia, i, vxf, vyf, vzf, svxx, svyy, svzz)
          do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
             if (q_domdec) then
                i = atoml(ia)
             else
#endif /* (domdec)*/
                i = ia
#if KEY_DOMDEC==1
             endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
             VXF = (XCOMP(I) + X(I)) * HALF
             VYF = (YCOMP(I) + Y(I)) * HALF
             VZF = (ZCOMP(I) + Z(I)) * HALF
             !             
             SVXX = WRK3(1,1)*VXF &
                  + WRK3(1,2)*VYF &
                  + WRK3(1,3)*VZF
             SVYY = WRK3(2,1)*VXF &
                  + WRK3(2,2)*VYF &
                  + WRK3(2,3)*VZF
             SVZZ = WRK3(3,1)*VXF &
                  + WRK3(3,2)*VYF &
                  + WRK3(3,3)*VZF
             !
             IF(QCONZ) THEN
                SVZZ = -(WRK3(1,1) + WRK3(2,2))*VZF
             ENDIF
             !
             XCOMP(I) = XCOMP(I) + SVXX
             YCOMP(I) = YCOMP(I) + SVYY
             ZCOMP(I) = ZCOMP(I) + SVZZ
          ENDDO
!$omp end parallel do
          !
          CALL HOLONOMA(XCOMP,YCOMP,ZCOMP,X,Y,Z,.TRUE.,.TRUE.,QOK)
          !
#if KEY_DOMDEC==1
          if (q_domdec) then
             i00 = 1
             i01 = natoml
          else
#endif 
             i00=1
             i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
             I00=1+IPARPT(MYNOD)
             i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
          endif  
#endif

!$omp parallel do schedule(static) private(ia, i)
          do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
             if (q_domdec) then
                i = atoml(ia)
             else
#endif /* (domdec)*/
                i = ia
#if KEY_DOMDEC==1
             endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
             VX(I) = X(I) - XCOMP(I)
             VY(I) = Y(I) - YCOMP(I)
             VZ(I) = Z(I) - ZCOMP(I)
          ENDDO
!$omp end parallel do
       ENDIF
       !

#if KEY_DOMDEC==1
       if (q_domdec) then
          i00 = 1
          i01 = natoml
       else
#endif 
          i00=1
          i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
          I00=1+IPARPT(MYNOD)
          i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
       endif  
#endif

!$omp parallel do schedule(static) private(ia, i)
       do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
          if (q_domdec) then
             i = atoml(ia)
          else
#endif /* (domdec)*/
             i = ia
#if KEY_DOMDEC==1
          endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
          IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
          XNEW(I) = VX(I)
          YNEW(I) = VY(I)
          ZNEW(I) = VZ(I)
          VX(I) = (XOLD(I)+XNEW(I)) * GAMMA(I+NATOM3)
          VY(I) = (YOLD(I)+YNEW(I)) * GAMMA(I+NATOM3)
          VZ(I) = (ZOLD(I)+ZNEW(I)) * GAMMA(I+NATOM3)
       ENDDO
!$omp end parallel do
    ENDIF
    !
    !
    !=======================================================================
    RVAL = 0.0
    TOTKEN = 0.0

#if KEY_DOMDEC==1
    if (q_domdec) then
       i00 = 1
       i01 = natoml
    else
#endif 
       i00=1
       i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
       I00=1+IPARPT(MYNOD)
       i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
    endif  
#endif

!$omp parallel do schedule(static) private(ia, i) reduction(+:rval, totken)
    do ia=i00,i01
#if KEY_DOMDEC==1 /*domdec*/
       if (q_domdec) then
          i = atoml(ia)
       else
#endif /* (domdec)*/
          i = ia
#if KEY_DOMDEC==1
       endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
       IF(JPBLOCK(I) /= MYNOD) cycle
#endif 
#endif 
       RVAL = RVAL + HALF*AMASS(I)*(VX(I)**2+VY(I)**2+VZ(I)**2)
       TOTKEN=TOTKEN+AMASS(I)*(XNEW(I)**2+YNEW(I)**2+ZNEW(I)**2) 
    ENDDO
!$omp end parallel do
#if KEY_PARALLEL==1
    GCARR(1)=RVAL
    GCARR(2)=TOTKEN
    CALL GCOMB(GCARR,2)
    RVAL=GCARR(1)
    TOTKEN=GCARR(2)
#endif 
    !=======================================================================
    !
    IF(QNPT) THEN
       PNHF = TWO*DELTA*(RVAL-REFKE)/TMASS
       PNHV = PNHVP + HALF*(PNHF + PNHFP)
       !     Make sure all processes have the same value
#if KEY_PARALLEL==1
       CALL PSND8(PNHV,1)                                 
#endif
       !
    ENDIF
    !
    ! Calulate the Piston potential energy
    !
    IF(QCONZ) THEN
       REFP(RPZZ) = ZERO
    ENDIF
    IF(QSURF) THEN
       PMAX = REFP(RPZZ)
    ELSE
       PMAX=THIRD*(REFP(RPXX)+REFP(RPYY)+REFP(RPZZ))
    ENDIF
    !
    EPROP(XTLPE) = EPROP(VOLUME)*PMAX*ATMOSP - &
         SURFT*SFACT*ATMOSP*EPROP(VOLUME)/XTLOLD(6)
    !
    ! Calulate the piston kinetic energy, make HDOTP be hdot at n-1
    EPROP(XTLKE) = ZERO
    DO I=1,XDIM
       EPROP(XTLKE) = EPROP(XTLKE) + PWMAT(I)*HDAV(I)**2
       TOTKEN = TOTKEN + PWMAT(I)*HDOT(I)**2 
    ENDDO
    !
    EPROP(XTLKE) = HALF*EPROP(XTLKE)/DELTA2
    TOTKEN = HALF*TOTKEN/DELTA2
    !
    EPROP(XTLTE)=EPROP(HFCTE)+EPROP(XTLKE)+EPROP(XTLPE)
    !
    ! Calculate the new volume
    CALL GETVOL(EPROP(VOLUME))
    !
    ! Calculate the high frequency corrected total energy
    !
    IF(EPROP(XTLKEP) == FMARK) THEN
       EPDIFF = EPROP(XTLPE) - EPROP(VOLUME)*PMAX*ATMOSP   
       EPROP(XTLPEP) = EPROP(XTLPEP) + EPROP(XTLPE) + EPDIFF
       TOTKEO = ZERO
       DO I=1,XDIM
          TOTKEO = TOTKEO + PWMAT(I)*HDOTP(I)**2 
       ENDDO
       TOTKEO = HALF*TOTKEO/DELTA2
       EPROP(XTLKP2) = EPROP(XTLKP2) + TOTKEO - EPDIFF 
       EPROP(XTLKEP) = EPROP(KEPR2) + TOTKEO
    ENDIF
    !
    HFCTEN = (EPROP(XTLKEP) - RVAL - EPROP(XTLKE) + &
         EPROP(XTLPEP) - EPROP(EPOT) - EPROP(XTLPE) )/THREE + &
         (EPROP(XTLKP2) - TOTKEN)/12.0
    !
    EPROP(XTLKP2) = EPROP(XTLKEP)
    EPROP(XTLKEP) = TOTKEN
    EPROP(XTLPEP) = EPROP(EPOT) + EPROP(XTLPE)
    !
    ! Calulate the pressure piston temperature
    IF(QCONZ) THEN
       EPROP(XTLTEM) = TWO*EPROP(XTLKE)/(KBOLTZ*(XDIM-1))
    ELSE
       EPROP(XTLTEM) = TWO*EPROP(XTLKE)/(KBOLTZ*XDIM)
    ENDIF
    !
    ! Now add in the contributions from the thermal piston
    EPROP(XTLKE) = EPROP(XTLKE) + HALF*TMASS*PNHV**2
    EPROP(XTLPE) = EPROP(XTLPE) + PNH*TWO*REFKE
    !
    ! Redefine piston total conserved energy (temporary fix)
    EPROP(XTLTE) = EPROP(EPOT) + EPROP(XTLKE) + EPROP(XTLPE) &
         + RVAL + HFCTEN
    !
    IF(QNPT) THEN
       EPROP(XTLTE)=EPROP(EPOT)+EPROP(XTLKE)+EPROP(XTLPE)+RVAL
    ENDIF
    !
    !     we need to refill the pressure array with the corrected values
    EPRESS(PIXX) = REFP(RPXX) + DELP(1,1)
    EPRESS(PIXY) = REFP(RPXY) + DELP(1,2)
    EPRESS(PIXZ) = REFP(RPXZ) + DELP(1,3)
    EPRESS(PIYX) = REFP(RPXY) + DELP(2,1)
    EPRESS(PIYY) = REFP(RPYY) + DELP(2,2)
    EPRESS(PIYZ) = REFP(RPYZ) + DELP(2,3)
    EPRESS(PIZX) = REFP(RPXZ) + DELP(3,1)
    EPRESS(PIZY) = REFP(RPYZ) + DELP(3,2)
    EPRESS(PIZZ) = REFP(RPZZ) + DELP(3,3)
    !
    ! output pressure tensor when IUPTEN postive; rv apr2008
    IF (0.EQ.MOD(NPRIV,IPTFRQ)) THEN
      IF(IUPTEN > 0.AND.IOLEV > 0)  &
        WRITE(IUPTEN,955) TIMEST*REAL(NPRIV),(EPRESS(I),I=PIXX,PIZZ)
    ENDIF
955 FORMAT(F13.3,1X,9F16.4)
    !
    ! update IMXCEN for P21 when dynamic adjustment enabled
    IF(QP21XCEN) IMXCEN = 0.25*XTLABC(1)
    !
    ! For a fix, I am putting surface tension in the piston KE
    ! spot for now.  This should be fixed.
    SURFI = HALF*XTLOLD(6)*(EPRESS(PIZZ)-HALF*( &
         EPRESS(PIYY)+EPRESS(PIXX)))/SFACT
    EPROP(XTLKE) = SURFI
    !
    IF(QNPT) THEN
       !                         Update the thermal piston coordinate
       !                         using the velocity verlet algorithm
       PNH = PNH + PNHV*DELTA + HALF*DELTA*PNHF
       !     Make sure all processes have the same value
#if KEY_PARALLEL==1
       CALL PSND8(PNH,1)                                     
#endif
       !
    ENDIF
    !
    !=======================================================================
    RETURN
  END SUBROUTINE PSCALE

SUBROUTINE SCAVEL(SCALE,X,Y,Z,VX,VY,VZ,VK, &
     AMASS,NDEGF,ISCVEL,IGVOPT,NATOM,IMOVE, &
     AVETEM,JHSTRT,QSHAKE,IDGF2)
  !
  !     SCALES VELOCITIES FOR A DYNAMICS RUN.
  !
  !     Authors: S. Swaminathan
  !              Bernie Brooks
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use stream
  use parallel
#if KEY_TSM==1
  use tsms_mod
#endif 
  implicit none
  real(chm_real) SCALE
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) VX(*),VY(*),VZ(*),VK(*)
  real(chm_real) AMASS(*)
  INTEGER NDEGF,ISCVEL,IGVOPT,NATOM
  INTEGER IMOVE(*)
  real(chm_real) AVETEM
  LOGICAL QSHAKE
  INTEGER JHSTRT,IDGF2(*)
  !

  real(chm_real) TOLD,SCALF,TNEW,VS,VA,AVET,NUMT
  INTEGER I,IDEGF
  !
  TOLD=ZERO
  TNEW=ZERO
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif 
#if KEY_PARALLEL==1
  IF(ISCVEL /= 0) CALL VDGBRE(VK,IPARPT)
#endif 
  !
  ! Compute approximate (scaled) average temperature
  IF(ISCVEL /= 0) THEN
     AVET=ZERO
     NUMT=0
     DO I=1,NATOM
        IF(IMOVE(I) == 0) THEN
           IDEGF=6
           IF(QSHAKE) IDEGF=IDGF2(I)
           IF(IDEGF > 0) THEN
              IF(VK(I) < ZERO) VK(I)=ZERO
              AVET=AVET+VK(I)/IDEGF
              NUMT=NUMT+1
           ENDIF
        ENDIF
     ENDDO
     IF(NUMT > 0) AVET=AVET/NUMT
  ENDIF
  !
  DO I=1,NATOM
     IF(IMOVE(I) == 0) THEN
        VA=(VX(I)**2+VY(I)**2+VZ(I)**2)*AMASS(I)
        TOLD=TOLD+VA
        SCALF=SCALE
        IF(ISCVEL /= 0) THEN
           IDEGF=6
           IF(QSHAKE) IDEGF=IDGF2(I)
           IF(IDEGF > 0 .AND. VK(I).GT.RSMALL) THEN
              VS=SQRT(AVET*IDEGF/VK(I))
              IF(VS > FIFTY) VS=FIFTY
              SCALF=SCALF*VS
           ENDIF
        ENDIF
        VX(I)=VX(I)*SCALF
        VY(I)=VY(I)*SCALF
        VZ(I)=VZ(I)*SCALF
        TNEW=TNEW+VA*SCALF**2
     ENDIF
  ENDDO
  !
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) CALL BACK0(VX,VY,VZ)
#endif 
  IGVOPT=2
  !
  TOLD=TOLD/(KBOLTZ*NDEGF)
  TNEW=TNEW/(KBOLTZ*NDEGF)
  !
  IF(PRNLEV >= 2) WRITE(OUTU,10) SCALE,TOLD,TNEW
10 FORMAT(/,5X,'VELOCITIES HAVE BEEN SCALED  -',F10.5,/, &
       5X,'OLD TEMPERATURE              -',F10.4,/, &
       5X,'NEW TEMPERATURE              -',F10.4)
  IF (ISCVEL /= 0 .AND. PRNLEV >= 2) WRITE(OUTU,15)
15 FORMAT(5X,'VELOCITIES SCALED DIFFERENTLY FOR EACH ATOM')
  RETURN
END SUBROUTINE SCAVEL

SUBROUTINE STOPRT(X,Y,Z,VX,VY,VZ, &
     AMASS,IGVOPT,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
     ,BACKLS &   
#endif
     )
  !
  !     STOPS THE ROTATION AND TRANSLATION OF A MOLECULE DURING A DYNAMICS
  !
  !     Authors: S. Swaminathan
  !              Bernie Brooks
  !
  use chm_kinds
  use stream
  use contrl, only: qsegsrt
  use psf, only: nseg,nictot,ibase
  use replica_ltm,only: qrep
#if KEY_RPATH==1
  use pathm, only: natrep      
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
#if KEY_DOMDEC==1
  use parallel,only:plnod0  
#endif
  implicit none
  real(chm_real) X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
  real(chm_real) AMASS(*)
  INTEGER IGVOPT,NATOM
  INTEGER IMOVE(*)
  LOGICAL QALWRT(6)
#if KEY_TSM==1
  INTEGER BACKLS(*)
#endif 
  !
  !
  real(chm_real) XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
  !
  ! local
  integer :: endseg,iseg,istart,iend,isiz,iallow
  !
  IGVOPT=2
  endseg=1
  if(qsegsrt) then
#if KEY_DOMDEC==1
     if (q_domdec) call wrndie(-5,'STOPRT',&
          'SEGMENT BASED STOP OF ROT/TRANS NOT IMPLEMENTED IN DOMDEC.')
#endif 
     endseg=nseg
  endif
  DO iseg = 1, endseg
     istart = 1
     iend   = natom
     ! for segment based stop of rot/trans
     ! calculate the offsets into R, V, M arrays:
     if(qsegsrt)then
        istart = ibase(nictot(iseg)+1)+1
        iend   = ibase(nictot(iseg+1)+1)
        ! In the case of replica this covers most of the cases!
        ! But to make it realy work correctly then qalwrt() must
        !   be set for each segment separately
#if KEY_REPLICA==1
        do iallow=1,6
           if(qrep) qalwrt(iallow)=.false.
        enddo
#endif 
        if(prnlev > 2) WRITE(OUTU,'(A,1X,I0)') &
             'STOPRT>Stopping rotation/translation for segment',iseg
     endif
     isiz = iend - istart + 1
#if KEY_REPLICA==1
#if KEY_RPATH==1
     if(qrep.and.qsegsrt.and.(natrep /= isiz)) call wrndie(-5,'STOPRT',&
          'For individual replica STOPRT, whole system must be replicated.')
#endif 
#endif 
     CALL CENMSS(X(istart:iend),Y(istart:iend),Z(istart:iend),&
          VX(istart:iend),VY(istart:iend),VZ(istart:iend),&
          AMASS(istart:iend),XCM,YCM,ZCM, &
          VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,isiz,IMOVE(istart:iend),QALWRT &
#if KEY_TSM==1
          ,BACKLS(istart:iend) &    
#endif
          )
     CALL ROTSTP(X(istart:iend),Y(istart:iend),Z(istart:iend),&
          VX(istart:iend),VY(istart:iend),VZ(istart:iend),&
          AMASS(istart:iend),XCM,YCM,ZCM, &
          VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,isiz,IMOVE(istart:iend),QALWRT &
#if KEY_TSM==1
          ,BACKLS(istart:iend) &  
#endif
          )
#if KEY_DOMDEC==1
     if ((q_domdec .and. plnod0 >= 5) .or. (.not.q_domdec .and. prnlev >= 5)) then
#else /**/
     IF(PRNLEV >= 5) THEN
#endif /*        */
        CALL CENMSS(X(istart:iend),Y(istart:iend),Z(istart:iend),&
             VX(istart:iend),VY(istart:iend),VZ(istart:iend),&
             AMASS(istart:iend),XCM,YCM,ZCM, &
             VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,isiz,IMOVE(istart:iend),QALWRT &
#if KEY_TSM==1
             ,BACKLS(istart:iend) &    
#endif
             )
     ENDIF
  enddo
  !
  RETURN
END SUBROUTINE STOPRT

SUBROUTINE CENMSS(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
     VXCM,VYCM, &
     VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
     ,BACKLS &     
#endif
     )
  !
  !     DETERMINES THE CENTER OF MASS, CENTER OF MASS MOTION, AND ANGULAR
  !     MOMENTUM OF THE MOLECULE.
  !
  !     Authors: S. Swaminathan
  !              Bernie Brooks
  !
  use chm_kinds
  use dimens_fcm
  use number
  use parallel,only: gcomb
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
#if KEY_DOMDEC==1
  use domdec_common,only:natoml,atoml,q_domdec 
#endif
  !
  !
  implicit none
  real(chm_real) X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
  real(chm_real) XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
  INTEGER NATOM
  real(chm_real) AMASS(*)
  INTEGER IMOVE(*)
  LOGICAL QALWRT(6)
#if KEY_TSM==1
  INTEGER BACKLS(*)
#endif 
  !
  real(chm_real) TMASS,AMASSI,EKCM,XI,YI,ZI,VXI,VYI,VZI
  INTEGER I,NUMFR
#if KEY_DOMDEC==1
  real(chm_real) cmdata(10)  
#endif
  integer ia,atfrst,atlast
  !
  VXCM=ZERO
  VYCM=ZERO
  VZCM=ZERO
  AXCM=ZERO
  AYCM=ZERO
  AZCM=ZERO
  XCM=ZERO
  YCM=ZERO
  ZCM=ZERO
  TMASS=ZERO

#if KEY_DOMDEC==1
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif 
     atfrst = 1
     atlast = natom
#if KEY_DOMDEC==1
  endif  
#endif

#if KEY_TSM==1
  IF(.NOT.QTSM) THEN
#endif 
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0) THEN
           AMASSI=AMASS(I)
           TMASS=TMASS+AMASSI
           XI=X(I)
           YI=Y(I)
           ZI=Z(I)
           VXI=VX(I)
           VYI=VY(I)
           VZI=VZ(I)
           VXCM=VXCM+VXI*AMASSI
           VYCM=VYCM+VYI*AMASSI
           VZCM=VZCM+VZI*AMASSI
           XCM=XCM+XI*AMASSI
           YCM=YCM+YI*AMASSI
           ZCM=ZCM+ZI*AMASSI
           AXCM=AXCM+(YI*VZI-ZI*VYI)*AMASSI
           AYCM=AYCM+(ZI*VXI-XI*VZI)*AMASSI
           AZCM=AZCM+(XI*VYI-YI*VXI)*AMASSI
        ENDIF
     ENDDO
#if KEY_TSM==1
  ELSE
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0.AND.BACKLS(I).EQ.0) THEN
           AMASSI=AMASS(I)
           TMASS=TMASS+AMASSI
           XI=X(I)
           YI=Y(I)
           ZI=Z(I)
           VXI=VX(I)
           VYI=VY(I)
           VZI=VZ(I)
           VXCM=VXCM+VXI*AMASSI
           VYCM=VYCM+VYI*AMASSI
           VZCM=VZCM+VZI*AMASSI
           XCM=XCM+XI*AMASSI
           YCM=YCM+YI*AMASSI
           ZCM=ZCM+ZI*AMASSI
           AXCM=AXCM+(YI*VZI-ZI*VYI)*AMASSI
           AYCM=AYCM+(ZI*VXI-XI*VZI)*AMASSI
           AZCM=AZCM+(XI*VYI-YI*VXI)*AMASSI
        ENDIF
     ENDDO
  ENDIF
#endif 
  !
  NUMFR=0
  IF(QALWRT(4)) NUMFR=NUMFR+1
  IF(QALWRT(5)) NUMFR=NUMFR+1
  IF(QALWRT(6)) NUMFR=NUMFR+1
  IF(NUMFR > 0 .AND. NUMFR < 3) THEN
     !     We have a system with axial symmetry.  Determine which axis
     !     and assign centers to appropriate axis.
     IF(.NOT.QALWRT(4)) THEN
        YCM=ZERO
        ZCM=ZERO
     ENDIF
     IF(.NOT.QALWRT(5)) THEN
        XCM=ZERO
        ZCM=ZERO
     ENDIF
     IF(.NOT.QALWRT(6)) THEN
        XCM=ZERO
        YCM=ZERO
     ENDIF
  ENDIF
  !
#if KEY_DOMDEC==1
  if (q_domdec) then
     cmdata(1) = xcm
     cmdata(2) = ycm
     cmdata(3) = zcm
     cmdata(4) = vxcm
     cmdata(5) = vycm
     cmdata(6) = vzcm
     cmdata(7) = axcm
     cmdata(8) = aycm
     cmdata(9) = azcm
     cmdata(10) = tmass
     call gcomb(cmdata, 10)
     xcm = cmdata(1)
     ycm = cmdata(2)
     zcm = cmdata(3)
     vxcm = cmdata(4)
     vycm = cmdata(5)
     vzcm = cmdata(6)
     axcm = cmdata(7)
     aycm = cmdata(8)
     azcm = cmdata(9)
     tmass = cmdata(10)
  endif
#endif 
  AXCM=AXCM-(YCM*VZCM-ZCM*VYCM)/TMASS
  AYCM=AYCM-(ZCM*VXCM-XCM*VZCM)/TMASS
  AZCM=AZCM-(XCM*VYCM-YCM*VXCM)/TMASS
  XCM=XCM/TMASS
  YCM=YCM/TMASS
  ZCM=ZCM/TMASS
  VXCM=VXCM/TMASS
  VYCM=VYCM/TMASS
  VZCM=VZCM/TMASS
  EKCM=VXCM**2+VYCM**2+VZCM**2
  EKCM=EKCM*TMASS/TWO
#if KEY_TSM==1
  IF(QTSM.AND.PIGSET) THEN
     CALL PIGCVSET(X,Y,Z)
     CALL BACK0(VX,VY,VZ)
  ENDIF
#endif 
  IF(PRNLEV >= 4) WRITE(OUTU,101) XCM,YCM,ZCM,VXCM,VYCM,VZCM, &
       AXCM,AYCM,AZCM,EKCM
101 FORMAT(/5X,'DETAILS ABOUT CENTRE OF MASS',/, &
       5X,'POSITION          :',1P,3G17.8,/, &
       5X,'VELOCITY          :',1P,3G17.8,/, &
       5X,'ANGULAR MOMENTUM  :',1P,3G17.8,/, &
       5X,'KINETIC ENERGY    :',1P,G17.8)
  !
  RETURN
END SUBROUTINE CENMSS

SUBROUTINE ROTSTP(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
     VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT &
#if KEY_TSM==1
     ,BACKLS &    
#endif
     )
  !
  !     STOPS THE ROTATION OF THE SYSTEM DURING MOLECULAR DYNAMICS.
  !
  !     Authors: S. Swaminathan
  !              Bernie Brooks
  !
  use chm_kinds
  use dimens_fcm
  use number
  use parallel,only: gcomb
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
#if KEY_DOMDEC==1
  use domdec_common,only:natoml,atoml,q_domdec  
#endif
  !
  implicit none
  real(chm_real) X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
  real(chm_real) AMASS(*)
  real(chm_real) XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
  real(chm_real) XI,YI,ZI
  INTEGER NATOM
  INTEGER IMOVE(*)
  LOGICAL QALWRT(6)
#if KEY_TSM==1
  INTEGER BACKLS(*)
#endif 
  !
  real(chm_real) TCM(3,3),AMASSI
  real(chm_real) XX,XY,XZ,YY,YZ,ZZ,OXCM,OYCM,OZCM
  real(chm_real) U(3,3),SCR(24),AMOM(6)
  INTEGER I,J,K,NUMFR
#if KEY_DOMDEC==1
  real(chm_real) cmdata(6)  
#endif
  integer ia,atfrst,atlast
  !
  XX=ZERO
  XY=ZERO
  XZ=ZERO
  YY=ZERO
  YZ=ZERO
  ZZ=ZERO

#if KEY_DOMDEC==1
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif 
     atfrst = 1
     atlast = natom
#if KEY_DOMDEC==1
  endif  
#endif

#if KEY_TSM==1
  IF(.NOT.QTSM) THEN     
#endif
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0) THEN
           XI=X(I)-XCM
           YI=Y(I)-YCM
           ZI=Z(I)-ZCM
           AMASSI=AMASS(I)
           XX=XX+XI*XI*AMASSI
           XY=XY+XI*YI*AMASSI
           XZ=XZ+XI*ZI*AMASSI
           YY=YY+YI*YI*AMASSI
           YZ=YZ+YI*ZI*AMASSI
           ZZ=ZZ+ZI*ZI*AMASSI
        ENDIF
     ENDDO
#if KEY_TSM==1
  ELSE
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0.AND.BACKLS(I).EQ.0) THEN
           XI=X(I)-XCM
           YI=Y(I)-YCM
           ZI=Z(I)-ZCM
           AMASSI=AMASS(I)
           XX=XX+XI*XI*AMASSI
           XY=XY+XI*YI*AMASSI
           XZ=XZ+XI*ZI*AMASSI
           YY=YY+YI*YI*AMASSI
           YZ=YZ+YI*ZI*AMASSI
           ZZ=ZZ+ZI*ZI*AMASSI
        ENDIF
     ENDDO
  ENDIF
#endif 
  !
#if KEY_DOMDEC==1
  if (q_domdec) then
     cmdata(1) = xx
     cmdata(2) = xy
     cmdata(3) = xz
     cmdata(4) = yy
     cmdata(5) = yz
     cmdata(6) = zz
     call gcomb(cmdata, 6)
     xx = cmdata(1)
     xy = cmdata(2)
     xz = cmdata(3)
     yy = cmdata(4)
     yz = cmdata(5)
     zz = cmdata(6)
  endif
#endif 
  AMOM(1)=YY+ZZ
  AMOM(2)=-XY
  AMOM(3)=-XZ
  AMOM(4)=XX+ZZ
  AMOM(5)=-YZ
  AMOM(6)=XX+YY
  !
  CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),SCR(1), &
       SCR(16),SCR(19),SCR(22),0)
  !
  DO I=1,3
     IF(ABS(SCR(I)) < TENM5) SCR(I)=SIGN(TENM5,SCR(I))
     SCR(I)=ONE/SCR(I)
  ENDDO
  !
  DO I=1,3
     DO J=1,3
        TCM(I,J)=0.0
        DO K=1,3
           TCM(I,J)=TCM(I,J)+U(I,K)*U(J,K)*SCR(K)
        ENDDO
     ENDDO
  ENDDO
  !
  !cc      IF(PRNLEV >= 2) WRITE(OUTU,386) QALWRT
  !cc 386  FORMAT(' STOPRT FLAGS:',6L4)
  !
  ! Keep motion that is allowed.
  IF(QALWRT(1)) VXCM=ZERO
  IF(QALWRT(2)) VYCM=ZERO
  IF(QALWRT(3)) VZCM=ZERO
  NUMFR=0
  IF(QALWRT(4)) NUMFR=NUMFR+1
  IF(QALWRT(5)) NUMFR=NUMFR+1
  IF(QALWRT(6)) NUMFR=NUMFR+1
  !
  IF(NUMFR == 0) THEN  ! remove all rotation
     OXCM=AXCM*TCM(1,1)+AYCM*TCM(1,2)+AZCM*TCM(1,3)
     OYCM=AXCM*TCM(2,1)+AYCM*TCM(2,2)+AZCM*TCM(2,3)
     OZCM=AXCM*TCM(3,1)+AYCM*TCM(3,2)+AZCM*TCM(3,3)
  ELSE
     OXCM=ZERO
     OYCM=ZERO
     OZCM=ZERO
     IF(.NOT.QALWRT(4)) OXCM=AXCM/(YY+ZZ)
     IF(.NOT.QALWRT(5)) OYCM=AYCM/(XX+ZZ)
     IF(.NOT.QALWRT(6)) OZCM=AZCM/(XX+YY)
  ENDIF
  !
#if KEY_TSM==1
  IF(.NOT.QTSM) THEN
#endif 
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0) THEN
           XI=X(I)-XCM
           YI=Y(I)-YCM
           ZI=Z(I)-ZCM
           VX(I)=VX(I)-VXCM-OYCM*ZI+OZCM*YI
           VY(I)=VY(I)-VYCM-OZCM*XI+OXCM*ZI
           VZ(I)=VZ(I)-VZCM-OXCM*YI+OYCM*XI
        ENDIF
     ENDDO
#if KEY_TSM==1
  ELSE
     do ia=atfrst,atlast
#if KEY_DOMDEC==1
        if (q_domdec) then
           i = atoml(ia)
        else
#endif 
           i = ia
#if KEY_DOMDEC==1
        endif  
#endif
        IF(IMOVE(I) == 0.AND.BACKLS(I).EQ.0) THEN
           XI=X(I)-XCM
           YI=Y(I)-YCM
           ZI=Z(I)-ZCM
           VX(I)=VX(I)-VXCM-OYCM*ZI+OZCM*YI
           VY(I)=VY(I)-VYCM-OZCM*XI+OXCM*ZI
           VZ(I)=VZ(I)-VZCM-OXCM*YI+OYCM*XI
        ENDIF
     ENDDO
  ENDIF
  !
  IF (QTSM.AND.PIGSET) THEN
     CALL PIGCVSET(X,Y,Z)
     CALL BACK0(VX,VY,VZ)
  ENDIF
#endif 
  RETURN
END SUBROUTINE ROTSTP

SUBROUTINE GAUSSI(AM,SD,V,IG,IGAUSS)
  !
  !     GAUSS WILL SUPPLY A NORMALLY DISTRIBUTED RANDOM NUMBER WITH
  !     A GIVEN MEAN AND STANDARD DEVIATION.
  !
  !     AM = THE DESIRED MEAN OF THE NORMAL DISTRIBUTION
  !     SD = THE DESIRED STANDARD DEVIATION OF THE NORMAL DISTRIBUTION
  !     V = VALUE OF THE COMPUTED NORMAL RANDOM VARIABLE
  !     IG = RANDOM NUMBER GENERATOR SEED
  !
  use clcg_mod,only:random
  use chm_kinds
  use exfunc
  use number
  use consta
  !
  use stream
  implicit none
  real(chm_real) AM,SD,V
  INTEGER IG,IGAUSS
  !
  real(chm_real) Y,A,B
#if KEY_JUNK==1
  INTEGER I
#endif 
  !
  IF(IGAUSS <= 0) then   !GOTO 101
     !
     ! random distribution of negative & positive values of sd in vel
     V=SD
     Y = RANDOM(IG)
     IF(Y <= 0.5) V=-SD
     RETURN
  endif
  !
  A=SQRT(MINTWO*LOG(RANDOM(IG)))
  B=TWO*PI*RANDOM(IG)
  V=SD*A*COS(B)+AM
  !
#if KEY_JUNK==1
  ! This code is needed to reproduce old dynamics runs.
  ! Keep this code for testing purposes only. BRB - 11/11/91
  A=0.E0
  DO I=1,12
     A=A+RANDOM(IG)
  ENDDO
  V=(A-6.E0)*SD+AM
#endif 
  RETURN
END SUBROUTINE GAUSSI

