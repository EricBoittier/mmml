#if KEY_FLUCQ==1 /*flucq*/
SUBROUTINE FQSETT(DT)
  !
  !     Sets FQ timestep to DT (used to equate the FQ
  !     timestep to the dynamics timestep)
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use flucq
  implicit none
  real(chm_real) DT
  IF (QFLUC) FQTSTP=DT
  RETURN
END SUBROUTINE FQSETT


SUBROUTINE FQZERO
  !
  !     Zeroes charge and thermostatting velocities;
  !     sets thermostatting scale factor to 1
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds

  use dimens_fcm
  use number
  use flucq
  use psf
  implicit none
  integer i
  IF (QFLUC) THEN
     do i=1,natom
        FQOLDQ(i)=CG(i)
        FQNEWQ(i)=CG(i)
     enddo
     FQCMV2=ZERO
     FQNHS=ONE
     FQNHSO=ONE
  ENDIF
  RETURN
END SUBROUTINE FQZERO


SUBROUTINE FQCHPR(ISTEP)
  !
  !     Propagates CHARMM charges during dynamics;
  !     CG(old) = CG(current) and CG(current) = CG(new)
  !     Performs a similar update for charge Nose-Hoover parameter S
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use psf
  use flucq
  implicit none
  INTEGER i,ISTEP
  ! N.B. Only 'real' atom charges are propagated; image charges
  ! are updated by TRANSO
  !     write(6,*) 'FQCHPR> Charge propagation at step ',ISTEP

  do i=1,natom
     FQOLDQ(i)=CG(i)
     CG(i)=FQNEWQ(i)
  enddo
  FQNHSO = FQNHS
  FQNHS  = FQNHSN
  RETURN
END SUBROUTINE FQCHPR


SUBROUTINE FQCNEW(ISTEP,NPRINT,ATFRST,ATLAST)
  !
  !     Calculates new CHARMM charges during dynamics, using the forces from
  !     a previous energy calculation
  !     ISTEP: current dynamics step number
  !     NPRINT: frequency to print within dynamics
  !     ATFRST,ATLAST: range of atoms to propagate (used by parallel
  !                    implementations of dynamics, currently ignored here)
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only:fqseln,fqcfor
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  use psf
  use flucq
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  INTEGER ISTEP,NPRINT,ATFRST,ATLAST
  !     write(6,*) 'FQCNEW> New charge calculation at step ',ISTEP
  IF (PRNLEV.GE.6) WRITE(OUTU,*) &
       'FQCNEW> Generating new charges within MM system...'
#if KEY_PARALLEL==1
  ! Current implementation generates charges on master node only
  IF (MYNOD.EQ.0) THEN
#endif 
     call FQCNW2(FQTSTP,ISTEP,NPRINT,FQSCAL,FQCFOR, &
          FQOLDQ,FQNEWQ, &
          FQSELN,FQCDGF,FQTEMP,ATFRST,ATLAST, &
          FQNHS,FQNHSO,FQNHSN,FQNHM,FQCMV2, &
          FQTCOU,FQUNIT,FQNHTL,FQNHMX)
#if KEY_PARALLEL==1
  ENDIF
  ! Update other nodes with the new charges and parameters
  call PSND8(FQNEWQ,NATOM)
  CALL PSND8(FQNHSN,1)
#endif 
  RETURN
END SUBROUTINE FQCNEW


SUBROUTINE FQCNW2(FQTSTP,ISTEP,NPRINT,FQSCAL,FQCFOR,FQOLDQ, &
     FQNEWQ,FQSELN,FQCDGF,FQTEMP, &
     ATFRST,ATLAST, &
     FQNHS,FQNHSO,FQNHSN,FQNHM,FQCMV2, &
     FQTCOU,FQUNIT,FQNHTL,FQNHMX)
  !
  !     Does the work of routine FQCNEW; uses a Verlet algorithm
  !     to calculate the new charges, applying Nose-Hoover thermostatting if
  !     the mass FQNHM is non-zero.
  !     The new charges are used to measure the kinetic energy, and thus
  !     to update the FlucQ energy properties (and thus, the total energy).
  !
  !     FQTSTP: FQ timestep, ps
  !     ISTEP:  Dynamics step number
  !     NPRINT: Frequency to print within dynamics
  !     FQSCAL: Frequency to scale velocities within dynamics (<=0 no scaling)
  !     FQCFOR: Charge force array
  !     FQOLDQ: Old charges
  !     FQNEWQ: New charges (updated on output)
  !     FQSELN: Selection of FQ atoms
  !     FQCDGF: Number of charge degrees of freedom
  !     FQTEMP: Thermostat charge temperature (K)
  !     ATFRST: First atom to propagate (ignored)
  !     ATLAST: Last atom to propagate (ignored)
  !     FQNHS:  Nose-Hoover thermostatting scale factor
  !     FQNHSO: Old scale factor, from last timestep
  !     FQNHSN: New scale factor (updated on output)
  !     FQNHM:  Thermostatting flucutation parameter (mass)
  !     FQCMV2: Kinetic energy measure (updated on output)
  !     FQTCOU: Thermostat coupling constant
  !     FQUNIT: Unit to write thermostatting data to
  !     FQNHTL: Nose-Hoover convergence tolerance
  !     FQNHMX: Maximum number of Nose-Hoover iterations
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use psf
  use param
  use stream
  use coord
  use consta
  use number
  use energym

  implicit none
  real(chm_real) FQTSTP
  INTEGER ISTEP,NPRINT,FQSCAL
  real(chm_real) FQCFOR(*),FQOLDQ(*),FQNEWQ(*)
  INTEGER FQSELN(*)
  real(chm_real) FQTEMP
  INTEGER ATFRST,ATLAST
  INTEGER FQCDGF
  real(chm_real) FQNHS,FQNHSO,FQNHSN,FQNHM,FQCMV2,FQTCOU,FQNHTL
  INTEGER FQUNIT,FQNHMX
  INTEGER I,J,NUM,NHITR
  real(chm_real) LAMBDA,NHSDIF
  real(chm_real) MV2TMP,NHKIN,NHPOT,FQNHA,CMASS,SCLFAC,MV2DES,VELFAC

  ! Do Nose-Hoover iterations if necessary
  IF (FQNHM.NE.ZERO) THEN
     DO NHITR=1,FQNHMX

        ! Obtain the acceleration on the Nose-Hoover scaling constant using
        ! mv**2 from the last timestep or previous iteration
        FQNHA=(FQCMV2-FQCDGF*KBOLTZ*FQTEMP)/FQNHM

        ! Get a first guess for the new scaling constant (by standard Verlet)
        ! and calculate the difference with the old (i.e. velocity*2dt)
        FQNHSN=TWO*FQNHS-FQNHSO+FQTSTP**2*FQNHA
        NHSDIF=FQNHSN-FQNHSO

        ! Propagate the charges by standard Verlet, including the calculated
        ! scaling constant, and get a new estimate for mv**2
        MV2TMP=ZERO
        DO I=1,NATOM
           IF (FQSELN(I).GT.0) THEN
              CMASS=FQCHMA(IAC(I))
              FQNEWQ(I)=(TWO*CG(I)-FQOLDQ(I)*(ONE-0.25d0*NHSDIF)- &
                   FQTSTP**2*FQCFOR(I)/CMASS)/(ONE+0.25d0*NHSDIF)
              MV2TMP=MV2TMP+CMASS*(FQNEWQ(I)-FQOLDQ(I))**2
           ENDIF
        ENDDO
        MV2TMP=MV2TMP/(TWO*FQTSTP)**2

        ! If the new value of mv**2 is close enough to the old, we have reached
        ! self-consistency and can continue. Otherwise, do another Nose iteration
        IF (ABS(MV2TMP-FQCMV2).LT.FQNHTL) THEN
           ! Add the Nose-Hoover contributions to the energies
           NHPOT=FQCDGF*KBOLTZ*FQTEMP*FQNHS
           NHKIN=HALF*FQNHM*(NHSDIF/(TWO*FQTSTP))**2
           EPROP(EHFC)=EPROP(EHFC)+NHPOT+NHKIN
           EPROP(TOTE)=EPROP(TOTE)+NHPOT+NHKIN
           IF (FQUNIT.NE.-1) THEN
              WRITE(FQUNIT,10) NHITR,FQNHS,NHSDIF/(TWO*FQTSTP), &
                   FQNHA,FQCMV2/TWO
10            FORMAT(' FQCHP2> ',I4,' iterations; S= ',F9.3, &
                   ' dS/dt= ',F9.3,' d2S/dt2= ',F9.3,' KE=',F9.3)
           ENDIF
           FQCMV2=MV2TMP
           GOTO 1000
        ENDIF
        FQCMV2=MV2TMP
     ENDDO
     CALL WRNDIE(-3,'<FQCNW2>', &
          'Maximum Nose-Hoover iterations exceeded')
  ELSE IF (FQTCOU.NE.ZERO) THEN
     ! Calculate desired value of mv**2 from the desired temperature FQTEMP
     MV2DES=FQTEMP*FQCDGF*KBOLTZ
     DO NHITR=1,FQNHMX
        IF (FQCMV2.EQ.ZERO) FQCMV2=0.001d0
        VELFAC = FQTCOU*(MV2DES/FQCMV2-1)*FQTSTP
        ! Propagate the charges by standard Verlet, including the calculated
        ! scaling constant, and get a new estimate for mv**2
        MV2TMP=ZERO
        DO I=1,NATOM
           IF (FQSELN(I).GT.0) THEN
              CMASS=FQCHMA(IAC(I))
              FQNEWQ(I)=(TWO*CG(I)-FQOLDQ(I)*(ONE+0.25d0*VELFAC)- &
                   FQTSTP**2*FQCFOR(I)/CMASS)/(ONE-0.25d0*VELFAC)
              MV2TMP=MV2TMP+CMASS*(FQNEWQ(I)-FQOLDQ(I))**2
           ENDIF
        ENDDO
        MV2TMP=MV2TMP/(TWO*FQTSTP)**2
        IF (ABS(MV2TMP-FQCMV2).LT.FQNHTL) THEN
           IF (FQUNIT.NE.-1) THEN
              WRITE(FQUNIT,20) NHITR,MV2DES/FQCMV2
20            FORMAT(' FQCHP2> ',I4,' iterations; ratio= ',F9.3)
           ENDIF
           FQCMV2=MV2TMP
           GOTO 1000
        ENDIF
        FQCMV2=MV2TMP
     ENDDO
     CALL WRNDIE(-3,'<FQCNW2>', &
          'Maximum thermostat iterations exceeded')
  ELSE
     ! No thermostatting - just do standard Verlet
     FQCMV2=ZERO
     DO I=1,NATOM
        IF (FQSELN(I).GT.0) THEN
           CMASS=FQCHMA(IAC(I))
           FQNEWQ(I)=TWO*CG(I)-FQOLDQ(I)- &
                FQTSTP**2*FQCFOR(I)/CMASS
           FQCMV2=FQCMV2+CMASS*(FQNEWQ(I)-FQOLDQ(I))**2
        ENDIF
     ENDDO
     FQCMV2=FQCMV2/(TWO*FQTSTP)**2
  ENDIF

  ! Do velocity scaling if requested
  IF (FQSCAL.GT.0.AND.FQCMV2.NE.ZERO) THEN
     IF (MOD(ISTEP,FQSCAL).EQ.0) THEN
        SCLFAC=FQCDGF*KBOLTZ*FQTEMP/FQCMV2
        FQCMV2=ZERO
        DO I=1,NATOM
           IF (FQSELN(I).GT.0) THEN
              CMASS=FQCHMA(IAC(I))
              FQNEWQ(I)=FQOLDQ(I)+(FQNEWQ(I)-FQOLDQ(I))*SCLFAC
              FQCMV2=FQCMV2+CMASS*(FQNEWQ(I)-FQOLDQ(I))**2
           ENDIF
        ENDDO
        FQCMV2=FQCMV2/(TWO*FQTSTP)**2
     ENDIF
  ENDIF

1000 CONTINUE

  ! Update fluctuating charge kinetic energy, and total energy counters
  EPROP(FQKIN)=HALF*FQCMV2
  EPROP(TOTE)=EPROP(TOTE)+EPROP(FQKIN)
  EPROP(HFCTE)=EPROP(HFCTE)+EPROP(FQKIN)

  RETURN
END SUBROUTINE FQCNW2

SUBROUTINE FQEXAC(COMLYN,COMLEN,NEEDUP)
  !
  !     Solves exact charges by Langevin dynamics
  !     NEEDUP: Set if the nonbond etc. lists must be updated first
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only: fqseln,fqcfor
  use chm_kinds
  use exfunc
  use dimens_fcm
  use psf
  use flucq
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  LOGICAL NEEDUP
  ! N.B. We pinch the "new charge" array to use as the
  ! "charge velocity" array in these routines; OK as this is only used
  ! during dynamics
  call FQEXC2(COMLYN,COMLEN,FQCFOR,FQNEWQ,FQOLDQ, &
       FQSELN,FQCDGF,NEEDUP)
  RETURN
END SUBROUTINE FQEXAC

SUBROUTINE FQEXC2(COMLYN,COMLEN,FQCFOR,VC,FQOLDQ,FQSELN, &
     FQCDGF,NEEDUP)
  !
  !     Does the work of FQEXAC
  !     FQCFOR: Charge forces; updated by each energy calculation
  !     VC:     Charge velocities
  !     FQOLDQ: Old charges
  !     FQSELN: Selection of FQ atoms
  !     FQCDGF: Number of charge degrees of freedom
  !     NEEDUP: Set if the nonbond etc. lists must be updated
  !
  !     Charges, velocites, and forces are updated by this routine
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use chm_types
  use exfunc
  use dimens_fcm
  use energym
  use stream
  use string
  use image
  use bases_fcm
  use coord
  use number
  use deriv
  use psf
  use param
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  real(chm_real) FQCFOR(*),VC(*),FQOLDQ(*)
  INTEGER FQSELN(*)
  INTEGER FQCDGF
  LOGICAL NEEDUP
  INTEGER I,NITER
  real(chm_real) TQ,C0,C1,C2
  real(chm_real) DT,ZDT,TQDES,ZETA
  real(chm_real) FQTFAC,CMASS
  LOGICAL PRIRES
  !
  real(chm_real) R1600
  R1600=1600.D0
  !
  DT=GTRMF(COMLYN,COMLEN,'TIME',PT001)
  ZETA=GTRMF(COMLYN,COMLEN,'ZETA',R1600)
  TQDES=GTRMF(COMLYN,COMLEN,'TQDE',TENM6)
  PRIRES=(INDXA(COMLYN,COMLEN,'PRIN').GT.0)
  ZDT=ZETA*DT

  ! Calculate conversion factor kinetic energy -> temperature
  FQTFAC=HALF*4184.0d0/8.314d0/FQCDGF

  IF (PRNLEV.GE.5) WRITE(OUTU,10) DT,ZDT,TQDES
10 FORMAT(' FQEXAC> Solving for exact charges with ', &
       'Timestep= ',D12.4,/,'Zeta= ',D12.4, &
       ' Desired charge temperature= ',D12.4)

  DO I=1,NATOM
     IF (FQSELN(I).LT.0) FQSELN(I)=1
  ENDDO

  C0 = EXP(-ZDT)
  C1 = (ONE-C0)/ZDT
  C2 = (ONE-C1)/ZDT

  ! First, update the CHARMM bond lists if necessary
  IF (NEEDUP) CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
       .TRUE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])

  ! Start with zeroed forces and velocities; equate old charges to
  ! current charges

  do i=1,natom
     FQOLDQ(i)=CG(i)
     VC(i)=ZERO
     FQCFOR(i)=ZERO
  enddo
  NITER=0
100 NITER=NITER+1
#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     DO I=1,NATOM
        IF (FQSELN(I).GT.0) THEN
           CMASS=FQCHMA(IAC(I))
           CG(I)=CG(I)+C1*DT*VC(I)-C2*DT**2*FQCFOR(I)/CMASS
           VC(I)=C0*VC(I)+(C2-C1)*DT*FQCFOR(I)/CMASS
        ENDIF
     ENDDO
#if KEY_PARALLEL==1
  ENDIF

  CALL PSND8(CG,NATOM)
#endif 

  CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)

#if KEY_PARALLEL==1
  IF (MYNOD.EQ.0) THEN
#endif 
     TQ=ZERO
     DO I=1,NATOM
        IF (FQSELN(I).GT.0) THEN
           CMASS=FQCHMA(IAC(I))
           VC(I)=VC(I)-C2*DT*FQCFOR(I)/CMASS
           TQ=TQ+CMASS*VC(I)**2
        ENDIF
     ENDDO
     TQ=TQ*HALF*FQTFAC
#if KEY_PARALLEL==1
  ENDIF

  CALL PSND8(TQ,1)
  CALL PSND8(CG,NATOM)
#endif 

  IF (PRNLEV.GE.5) WRITE(OUTU,30) NITER,TQ
30 FORMAT(' FQEXAC> Iteration ',I9,' Charge temp. ',D12.4,' K')
  IF (TQ.GT.TQDES) GOTO 100

  IF (PRNLEV.GE.5) WRITE(OUTU,40) NITER
40 FORMAT(' FQEXAC> Charges solved in ',I9,' iterations')
  IF (PRIRES.AND.PRNLEV.GE.2) THEN
     DO I=1,NATOM
        IF (FQSELN(I).GT.0) WRITE(OUTU,50) I,CG(I)
     ENDDO
50   FORMAT(' FQEXAC> Atom ',I6,' charge ',F10.6)
  ENDIF
  CALL FQZERO
#if KEY_PARALLEL==1
  CALL PSYNC()
#endif 
  RETURN
END SUBROUTINE FQEXC2


SUBROUTINE FQRUPD
  !
  !     Updates FlucQ parameters on parallel nodes after reading
  !     a restart file
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use flucq
  use parallel,only: psnd4,psnd8
  implicit none
#if KEY_PARALLEL==1
  IF (QFLUC) THEN
     CALL PSND8(CG,NATOM)
     call PSND8(FQOLDQ,NATOM)
     ! this is bug: it is declared as real*8
!     CALL PSND4(FQNHSO,1)
!     CALL PSND4(FQNHS,1)
     CALL PSND8(FQNHSO,1)
     CALL PSND8(FQNHS,1)
  ENDIF
#endif 
  RETURN
END SUBROUTINE FQRUPD

SUBROUTINE FQRWRI(U)
  !
  !     Writes current and old charges, as well as Nose-Hoover
  !     thermostat parameters, to a restart file on unit U
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only: fqseln
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use flucq
  implicit none
  INTEGER U
  INTEGER I

  WRITE(U,'(/A)') ' !FLUCQ OLDNHS, NHS'
  WRITE(U,'(2D22.15)') FQNHSO,FQNHS

  WRITE(U,'(/A)') ' !FLUCQ OLDCG, CG'
  DO I=1,NATOM
     IF (FQSELN(I).NE.0) THEN
        WRITE(U,'(2D22.15)') FQOLDQ(I),CG(I)
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE FQRWRI


SUBROUTINE FQRREA(U)
  !
  !     Reads current and old charges, as well as Nose-Hoover
  !     thermostat parameters, from a restart file on unit U
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only: fqseln
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use flucq
  implicit none
  INTEGER U
  INTEGER I
  CHARACTER(len=80) LINE
  real(chm_real) TMP

  READ(U,'(/A)',END=9) LINE
  READ(U,'(2D22.15)') FQNHSO,FQNHS

  READ(U,'(/A)',END=9) LINE
  DO I=1,NATOM
     IF (FQSELN(I).NE.0) THEN
        READ(U,'(2D22.15)') TMP,CG(I)
        FQOLDQ(I)=TMP
     ENDIF
  ENDDO
  RETURN

9 CALL WRNDIE(-3,'<FQRREA>','EOF during read')
  RETURN
END SUBROUTINE FQRREA


SUBROUTINE FQCWRI(U)
  !
  !     Writes all FQ charges to the trajectory file on unit U
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use dimens_fcm
  use psf
  implicit none
  INTEGER U
  INTEGER I
#if KEY_SINGLE==1
  WRITE(U) (CG(I),I=1,NATOM)
#else /**/
  WRITE(U) (SNGL(CG(I)),I=1,NATOM)
#endif 
  RETURN
END SUBROUTINE FQCWRI


SUBROUTINE FQCREA(U,TEMP)
  !
  !     Reads all FQ charges from the trajectory file on unit U
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use dimens_fcm
  use psf
  implicit none
  INTEGER U
  REAL(chm_real) TEMP(NATOM)
  INTEGER I
  READ(U) TEMP
  DO I=1,NATOM
     CG(I)=TEMP(I)
  ENDDO
  RETURN
END SUBROUTINE FQCREA
#endif /* (flucq)*/

subroutine flucq_dyn_stub()
  stop
end subroutine flucq_dyn_stub

