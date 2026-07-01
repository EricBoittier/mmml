#if KEY_FLUCQ==1 /*flucq*/

! These routines take care of calculating the FLUCQ charge forces,
! as well as the polarisation energy term (charge kinetic energy is
! determined by the dynamics routines in dynamc.src). This energy
! term is calculated from the charge forces in array FQCFOR, which
! are built up a) by the various QM codes (if QM/MM interactions are
! present - see qmmm.src) followed by b) the nonbond electrostatic terms
! (subroutine FQELEC), and finished with c) the internal terms (subroutine
! FQENER, below).

SUBROUTINE FQENER(FQPOL,X,Y,Z,QFQPOL)
  !
  !     Calculates FQ polarisation energy term and charge accelerations, using
  !     the electrostatic contributions from CHARMM nonbonded routines,
  !     QM/MM contributions from qmmm.src (called by QM codes) and internal
  !     terms.
  !
  !     FQPOL:  FQ polarisation energy (updated on output)
  !     X,Y,Z:  Coordinate arrays
  !     QFQPOL: Flag - whether to calculate FQPOL
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm, only: fqjr,fqseln,fqcfor
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  use flucq
  use psf

  implicit none
  real(chm_real) FQPOL
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QFQPOL
  CALL FQENR2(FQPOL,NATOM, &
       X,Y,Z,QFQPOL,FQSELN, &
       FQJR,FQCFOR, &
       FQTSTP,FQREFE,IMOVE,FQFIXD)
  RETURN
END SUBROUTINE FQENER


SUBROUTINE FQENR2(FQPOL,NATM,X,Y,Z, &
     QFQPOL,FQSELN,FQJR,FQCFOR, &
     FQTSTP,FQREFE,IMVE,FQFIXD)
  !
  !     Does the work for FQENER
  !
  !     FQPOL:  FQ polarisation energy (updated on output)
  !     NATM:   Total number of atoms
  !     X,Y,Z:  Coordinate arrays
  !     QFQPOL: Flag - whether to calculate FQPOL
  !     FQSELN: FQ atom selection
  !     FQJR:   Slater atom-atom interaction terms for constrained bonds
  !     FQCFOR: Charge forces (updated on output)
  !     FQTSTP: FQ timestep
  !     FQREFE: FQ reference polarisation energy
  !     IMVE:   Array of flags - whether atoms can move
  !     FQFIXD: Whether bond lengths are fixed
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm, only: fqint
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  use code
  use psf
  use param
  use image
  use number
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  real(chm_real) FQPOL
  INTEGER NATM
  real(chm_real) X(*),Y(*),Z(*)
  LOGICAL QFQPOL
  INTEGER FQSELN(*)
  real(chm_real) FQJR(*),FQCFOR(*)
  real(chm_real) FQTSTP,FQREFE
  INTEGER IMVE(*)
  LOGICAL FQFIXD

  INTEGER I,J,A1,A2
  real(chm_real) JR

  !      real(chm_real) FQINT

  ! First, convert electrostatic energies to electronegativities
  ! (since these terms are linear in charge, just divide by charge;
  ! this could create complications if charge==0, but we ignore that here)
  J=NATOM
  IF (NTRANS.GT.0) J=NATIM
  DO I=1,J
     IF (CG(I).NE.ZERO) THEN
        FQCFOR(I)=FQCFOR(I)/CG(I)
     ELSE
        FQCFOR(I)=ZERO
     ENDIF
  ENDDO

#if KEY_PARALLEL==1
  ! Do intramolecular stuff on one processor when running in parallel
  IF (MYNOD.EQ.0) THEN
#endif 

     ! Calculate bond contributions; if FQFIXD is set (the default), assume
     ! all bonds are maintained at their parameterised equilibrium values,
     ! otherwise recalculate the Slater interaction integrals. N.B. This is
     ! not the most efficient way of handling non-constrained bonds - should
     ! be improved (e.g. by using calculated distances from standard CHARMM
     ! bond energy routines)
     IF (FQFIXD) THEN
        DO I=1,NBOND
           A1=IB(I)
           A2=JB(I)
           IF (FQSELN(A1).GT.0.AND.FQSELN(A2).GT.0.AND. &
                (IMVE(A1).LE.0.OR.IMVE(A2).LE.0).AND. &
                ICB(I).NE.0) THEN
              JR=FQJR(ICB(I))
              FQPOL=FQPOL+JR*CG(A1)*CG(A2)
              FQCFOR(A1) = FQCFOR(A1)+JR*CG(A2)
              FQCFOR(A2) = FQCFOR(A2)+JR*CG(A1)
           ENDIF
        ENDDO
     ELSE
        DO I=1,NBOND
           A1=IB(I)
           A2=JB(I)
           IF (FQSELN(A1).GT.0.AND.FQSELN(A2).GT.0.AND. &
                (IMVE(A1).LE.0.OR.IMVE(A2).LE.0).AND. &
                ICB(I).NE.0) THEN
              JR=FQINT(SQRT((X(A1)-X(A2))**2+(Y(A1)-Y(A2))**2+ &
                   (Z(A1)-Z(A2))**2),IAC(A1),IAC(A2))
              FQPOL=FQPOL+JR*CG(A1)*CG(A2)
              FQCFOR(A1) = FQCFOR(A1)+JR*CG(A2)
              FQCFOR(A2) = FQCFOR(A2)+JR*CG(A1)
           ENDIF
        ENDDO
     ENDIF

     ! Calculate self-interaction terms
     DO I=1,NATOM
        IF (FQSELN(I).GT.0.AND.IMVE(I).LE.0) THEN
           FQPOL=FQPOL+FQCHI(IAC(I))*CG(I)+ &
                HALF*FQJZ(IAC(I))*CG(I)*CG(I)
           FQCFOR(I) = FQCFOR(I)+FQCHI(IAC(I))+FQJZ(IAC(I))*CG(I)
        ENDIF
     ENDDO

     ! If the polarisation term is required, subtract the reference energy;
     ! otherwise just zero the term
     IF (QFQPOL) THEN
        FQPOL=FQPOL-FQREFE
     ELSE
        FQPOL=ZERO
     ENDIF

#if KEY_PARALLEL==1
  ENDIF
#endif 
  RETURN
END SUBROUTINE FQENR2


SUBROUTINE FQFORC
  !
  !     Calculates the charge forces from the electronegativity arrays,
  !     and informs all parallel nodes
  !
  !     Author: Ben Webb, 2000
  use flucqm, only: fqseln,fqcfor
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  use flucq
  use param
  use psf
  implicit none
  IF (FQGRUP) THEN
     CALL FQFOR2(FQSELN,FQCFOR,FQCHMA,NGRP,IGPBS)
  ELSE
     CALL FQFOR2(FQSELN,FQCFOR,FQCHMA,NRES,IBASE)
  ENDIF
  RETURN
END SUBROUTINE FQFORC


SUBROUTINE FQFOR2(FQSELN,FQCFOR,FQCHMA,NUMR,RBASE)
  !
  !     Does the work for FQFORC
  !
  !     FQSELN: FQ atom selection
  !     FQCFOR: Electronegativities (input); Charge forces (output)
  !     FQCHMA: Charge masses
  !     NUMR  : Number of residues/groups
  !     RBASE : Index into residue/group array
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  use number
  use psf
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none
  INTEGER FQSELN(*)
  real(chm_real) FQCFOR(*),FQCHMA(*)
  INTEGER NUMR,RBASE(*)

  INTEGER I,J
  real(chm_real) LAMBDA,MASINV,TMASS
#if KEY_PARALLEL==1
  ! Get electronegativities from other nodes
  CALL GCOMB(FQCFOR,NATOM)
  ! Only do averaging on node 0
  IF (MYNOD.EQ.0) THEN
#endif 

     ! Now, given dE/dq for each atom, we calculate the charge force
     ! by mass-weighted averaging over the residue (or group)
     DO I=1,NUMR
        LAMBDA=ZERO
        TMASS=ZERO
        DO J=RBASE(I)+1,RBASE(I+1)
           IF (FQSELN(J).GT.0) THEN
              MASINV=ONE/FQCHMA(IAC(J))
              LAMBDA=LAMBDA+FQCFOR(J)*MASINV
              TMASS=TMASS+MASINV
           ELSE
              FQCFOR(J)=ZERO
           ENDIF
        ENDDO
        IF (TMASS.GT.ZERO) LAMBDA=LAMBDA/TMASS
        DO J=RBASE(I)+1,RBASE(I+1)
           IF (FQSELN(J).GT.0) THEN
              FQCFOR(J)=FQCFOR(J)-LAMBDA
           ELSE
              FQCFOR(J)=ZERO
           ENDIF
        ENDDO
     ENDDO
#if KEY_PARALLEL==1
  ENDIF
  ! Send resultant accelerations to all nodes
  CALL PSND8(FQCFOR,NATOM)
#endif 
  RETURN
END SUBROUTINE FQFOR2
#endif /* (flucq)*/

subroutine fluq_ene_stub
  return
end subroutine fluq_ene_stub

