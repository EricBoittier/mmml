#if KEY_CFF==1
SUBROUTINE EANGANGFS(ETT)

  ! Function
  !     This routine computes the theta*theta energy and 1st derivatives
  !     for all theta*theta interactions.

  use chm_kinds
  use dimens_fcm
  use energym
  use number

  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) ITHETA,ITHETB,ITHTH,ITT1,ITT2,ITT3,ITT4,ITH,IVPN,ISGN1, &
       ISGN2
  real(chm_real) CTHTH,DETTDA,DETTDT,DIFTHA,DIFTHB,THEANA,THEANB,ETT
  ! BIOVIA Code Start : Bug fix
  integer(chm_int4) IBOND1,IBOND2,IBOND3,IBOND4,ICB1,ICB2,ICB3,ICB4
  ! BIOVIA Code End

  IF(.NOT.QETERM(BNDBND)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES_CFF

  !-----loop over all theta*theta angles

  DO ITHTH=1,NTHTH

     !-----now find pointers to the 2 thetas and 3 bonds in the
     !     theta*theta interaction.
     ITHETA = ITTTW(1,ITHTH)
     ITHETB = ITTTW(2,ITHTH)

     ! BIOVIA Code Start : Bug fix
     !-----get pointers to the 2 bonds in the theta angle
     IBOND1 = ITBW(1,ITHETA)
     ICB1 = ICB(IBOND1)
     IF (ICB1 == 0) CYCLE

     IBOND2 = ITBW(2,ITHETA)
     ICB2 = ICB(IBOND2)
     IF (ICB2 == 0) CYCLE

     IBOND3 = ITBW(1,ITHETB)
     ICB3 = ICB(IBOND3)
     IF (ICB3 == 0) CYCLE

     IBOND4 = ITBW(2,ITHETB)
     ICB4 = ICB(IBOND4)
     IF (ICB4 == 0) CYCLE
 
     ! BIOVIA Code End
     
     !-----get the theta angle from the theta angle previously computed.
     THEANA = TH(ITHETA)

     DIFTHA = THEANA - CTHET2(ICT(ITHETA))

     !-----next get the 2nd theta angle from previously computed values
     THEANB = TH(ITHETB)

     DIFTHB = THEANB - CTHET2(ICT(ITHETB))

     CTHTH = CTT(ICTT(ITHTH))

     !-----compute the theta*theta energy and add it to the total
     ETT = ETT + CTHTH*DIFTHA*DIFTHB

     DETTDT = CTHTH*DIFTHB
     DETTDA = CTHTH*DIFTHA

     ISGN1 = ITTFLG(1,ITHTH)
     ISGN2 = ITTFLG(2,ITHTH)
     ITT1 = ITTW(1,ITHTH)
     DX(ITT1) = DX(ITT1) + DTHDX(2+ISGN1,ITHETA)*DETTDT
     DY(ITT1) = DY(ITT1) + DTHDY(2+ISGN1,ITHETA)*DETTDT
     DZ(ITT1) = DZ(ITT1) + DTHDZ(2+ISGN1,ITHETA)*DETTDT
     ITT2 = ITTW(2,ITHTH)
     DX(ITT2) = DX(ITT2) + DTHDX(2-ISGN2,ITHETB)*DETTDA
     DY(ITT2) = DY(ITT2) + DTHDY(2-ISGN2,ITHETB)*DETTDA
     DZ(ITT2) = DZ(ITT2) + DTHDZ(2-ISGN2,ITHETB)*DETTDA
     ITT3 = ITTW(3,ITHTH)
     DX(ITT3) = DX(ITT3) + DTHDX(2-ISGN1,ITHETA)*DETTDT &
          + DTHDX(2+ISGN2,ITHETB)*DETTDA
     DY(ITT3) = DY(ITT3) + DTHDY(2-ISGN1,ITHETA)*DETTDT &
          + DTHDY(2+ISGN2,ITHETB)*DETTDA
     DZ(ITT3) = DZ(ITT3) + DTHDZ(2-ISGN1,ITHETA)*DETTDT &
          + DTHDZ(2+ISGN2,ITHETB)*DETTDA
     ITT4 = ITTW(4,ITHTH)
     DX(ITT4) = DX(ITT4) + DTHDX(2,ITHETA)*DETTDT &
          + DTHDX(2,ITHETB)*DETTDA
     DY(ITT4) = DY(ITT4) + DTHDY(2,ITHETA)*DETTDT &
          + DTHDY(2,ITHETB)*DETTDA
     DZ(ITT4) = DZ(ITT4) + DTHDZ(2,ITHETA)*DETTDT &
          + DTHDZ(2,ITHETB)*DETTDA
  ENDDO

  RETURN
END SUBROUTINE EANGANGFS

SUBROUTINE EANGLFS_CFF(ET)

  ! Function

  use chm_kinds
  use dimens_fcm
  use energym
  use consta
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) ICTI,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA
  real(chm_real) COSTHE,DETD,DTHETA,ET,ETHET,VLEN1,VLEN2, &
       VX1,VX2,VY1,VY2,VZ1,VZ2,RSNTHE,RVLN1,RVLN12, &
       RVLN2,RVLN22,SINTH2,SINTHE,SMALLA,VL1DV2,VL2DV1, &
       DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32,DTHTA2, &
       DTHET1,DTHET2,DTHET3,DTHET4,DTHET5,DTHET6,DTHET7,DTHET8,DTHET9

  DATA SMALLA/1.0D-10/

  IF(.NOT.QETERM(ANGLE)) RETURN
  IF (.NOT.QETERM(BOND)) CALL BONDLEN_CFF

  !-----loop over all valence angles in this molecule and compute the
  !     valence angle, energy and derivatives.

  DO ITHETA=1,NTHETA

     !-----save the atom numbers of the 3 atoms involved in this angle.
     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)

     !-----compute the 2 vectors of the valence angle,
     !     thevec(,1) (atoms 2->1) and thevec(,2) (atoms 2->3)a.
     VX1 = X(ITHE1) - X(ITHE2)
     VX2 = X(ITHE3) - X(ITHE2)
     VY1 = Y(ITHE1) - Y(ITHE2)
     VY2 = Y(ITHE3) - Y(ITHE2)
     VZ1 = Z(ITHE1) - Z(ITHE2)
     VZ2 = Z(ITHE3) - Z(ITHE2)

     !-----find the cosine of the angle between these 2 vectors,
     !     the valence angle, and save it in theang.
     !     the length of these 2 vectors is also returned in vlen1 and vlen2.

     VLEN1 = BL(ITBW(1,ITHETA))
     VLEN2 = BL(ITBW(2,ITHETA))

     COSTHE = (VX1*VX2 + VY1*VY2 + VZ1*VZ2)/(VLEN1*VLEN2)
     COSTHE = MIN(ONE,MAX(-ONE,COSTHE))

     COSTH(ITHETA) = COSTHE
     TH(ITHETA) = ACOS(COSTHE)

     !-----compute the 1st (and possibly 2nd) derivatives of the angle
     !     (NOT! the cosine) w.r.t the 2 vectors forming the angle, vectr1
     !     and vectr2. the resulting derivatives are returned in dvec and
     !     d2vec. isn, the "sign" of the angle is made positive, as the
     !     valence angle (unlike the torsion angle) is never negative.
     !
     !     difang computes the first (and possibly 2nd) derivatives
     !     of the cosine of an angle (or possibly the angle itself,
     !     depending on iarc) w.r.t the 2 vectors defining the angle.
     !     this routine is used to find the derivatives of the valence
     !     angle w.r.t. the vector components and also for the derivatives
     !     of the torsion angle.
     !     the derivatives for the cosine of the angle are found by
     !     computing the derivative of the dot product divided by the length
     !     of the 2 vectors w.r.t the 2 vectors forming the angle.
     !     the derivatives for the angle are found by computing the
     !     derivative of the angle w.r.t the cosine of the angle and
     !     then multipying this by the derivative of the cosine of the
     !     angle w.r.t the 2 vectors forming the angle.
     !     this requires the 2 vectors in the angle, vectr1 and vectr2, and
     !     the length of these 2 vectors, vlen1 and vlen2 and the cosine
     !     of the angle.
     !     NOTE. This routine produces discontinuities in the derivatives
     !           if the angle is very close to zero and derivatives for
     !           the angle are required, instead of the cosine.
     !
     !-----compute the first derivatives of the cosine of the angle
     !     or the angle w.r.t. the 2 vectors forming tha angle.
     !     vl1dv2 is the length of vectr1 divided by the length of vectr2.
     !     vl2dv1 is the length of vectr2 divided by the length of vectr1.
     RVLN1 = ONE/VLEN1
     RVLN2 = ONE/VLEN2
     RVLN12 = RVLN1*RVLN1
     RVLN22 = RVLN2*RVLN2
     VL1DV2 = VLEN1*RVLN2
     VL2DV1 = VLEN2*RVLN1

     DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
     DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
     DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
     DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
     DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
     DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)

     !-----compute the derivatives for the angle instead of the cosine of
     !     the angle.
     !     first, compute the sine of the angle (from the cosine)
     !     MAKE SURE the angle is not zero, if it is make it a small
     !     number. (this gives rise to a discontinuity in the derivatives
     !     below the value of small which can cause problems).
     !     make the sign of the sine correspond to the sign of isn.
     SINTH2 = ABS(ONE - COSTHE*COSTHE)
     SINTH2 = MAX(SMALLA,SINTH2)
     SINTHE = SQRT(SINTH2)

     RSNTHE = ONE/SINTHE

     !-----finally compute the first derivatives for the angle, not the
     !     cosine.
     !     NOTE. This must come last as dvec is also used in the computation
     !           of the 2nd derivatives of the angle and the original
     !           dvec (for the cosine of the angle) is needed there.
     DVEC11 = -DVEC11*RSNTHE
     DVEC21 = -DVEC21*RSNTHE
     DVEC31 = -DVEC31*RSNTHE
     DVEC12 = -DVEC12*RSNTHE
     DVEC22 = -DVEC22*RSNTHE
     DVEC32 = -DVEC32*RSNTHE

     !-----compute the 1st derivatives of the angle w.r.t the cartesian
     !     coordinates from the derivatives w.r.t the vectors forming the
     !     angle.
     DTHDX(1,ITHETA) = DVEC11
     DTHDY(1,ITHETA) = DVEC21
     DTHDZ(1,ITHETA) = DVEC31
     DTHDX(2,ITHETA) = -DVEC11 - DVEC12
     DTHDY(2,ITHETA) = -DVEC21 - DVEC22
     DTHDZ(2,ITHETA) = -DVEC31 - DVEC32
     DTHDX(3,ITHETA) = DVEC12
     DTHDY(3,ITHETA) = DVEC22
     DTHDZ(3,ITHETA) = DVEC32

     !-----compute the theta energy and the 1st derivative
     !     of the energy w.r.t the valence angle.
     !
     ICTI=ICT(ITHETA)
     DTHETA = TH(ITHETA) - CTHET2(ICTI)
     DTHTA2 = DTHETA * DTHETA
     ETHET = (CTHET1(ICTI) + CTHET6(ICTI)*DTHETA)*DTHTA2 &
          + CTHET7(ICTI)*DTHTA2*DTHTA2

     ET = ET + ETHET

     DETD  = TWO*CTHET1(ICTI)*DTHETA + THREE*CTHET6(ICTI)*DTHTA2 &
          + FOUR*CTHET7(ICTI)*DTHTA2*DTHETA
     DX(ITHE1) = DX(ITHE1) + DTHDX(1,ITHETA)*DETD
     DY(ITHE1) = DY(ITHE1) + DTHDY(1,ITHETA)*DETD
     DZ(ITHE1) = DZ(ITHE1) + DTHDZ(1,ITHETA)*DETD
     DX(ITHE2) = DX(ITHE2) + DTHDX(2,ITHETA)*DETD
     DY(ITHE2) = DY(ITHE2) + DTHDY(2,ITHETA)*DETD
     DZ(ITHE2) = DZ(ITHE2) + DTHDZ(2,ITHETA)*DETD
     DX(ITHE3) = DX(ITHE3) + DTHDX(3,ITHETA)*DETD
     DY(ITHE3) = DY(ITHE3) + DTHDY(3,ITHETA)*DETD
     DZ(ITHE3) = DZ(ITHE3) + DTHDZ(3,ITHETA)*DETD
  ENDDO
  RETURN
END SUBROUTINE EANGLFS_CFF


SUBROUTINE EBNDANGFS(EBB,EBT)

  ! Function
  !     Compute the bond*theta cross-term energy and its 1st
  !     derivative w.r.t. cartesian coordinates
  !     each theta angle also represents 2 bond*theta cross-terms, 1 for
  !     each bond. thus, this routine loops over all theta angles and
  !     computes the 2 bond*theta cross-terms in each angle.

  use chm_kinds
  use dimens_fcm
  use energym
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) IBOND1,IBOND2,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA,ISGN,ICTP
  real(chm_real) BNDLN1,BNDLN2,CBTA,CBTB,CTMP,DEBTDT,DTHET,DBDX1, &
       DBDY1,DBDZ1,DBDX2,DBDY2,DBDZ2, &
       KBB,DEBDA,DEBDB,DIFB1,DIFB2,EBB,EBT
  ! BIOVIA Code Start : Bug fix
  INTEGER(chm_int4) ICB1,ICB2
  ! BIOVIA Code End

  IF(.NOT.QETERM(STRSTR) .AND. .NOT.QETERM(STRB)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES_CFF

  !-----loop over all valence angles.
  !     there are 2 bond*theta cross-terms for each valence angle, 1 for
  !     each bond.

  DO ITHETA=1,NTHETA

     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)

     !-----get pointers to the 2 bonds in the theta angle
     IBOND1 = ITBW(1,ITHETA)
     IBOND2 = ITBW(2,ITHETA)

     BNDLN1 = BL(IBOND1)
     BNDLN2 = BL(IBOND2)
     ! BIOVIA Code Start : Bug fix
     ICB1 = ICB(IBOND1)
     ICB2 = ICB(IBOND2)
     IF(ICB1 == 0 .OR. ICB2 == 0) CYCLE 
     ! BIOVIA Code End

     ISGN = ITFLG(1,ITHETA)
     DBDX1 = DBDX(IBOND1)*ISGN
     DBDY1 = DBDY(IBOND1)*ISGN
     DBDZ1 = DBDZ(IBOND1)*ISGN
     ISGN = ITFLG(2,ITHETA)
     DBDX2 = DBDX(IBOND2)*ISGN
     DBDY2 = DBDY(IBOND2)*ISGN
     DBDZ2 = DBDZ(IBOND2)*ISGN

     !-----compute the difference of the 2 bond lengths from the equilibrium
     !     values for the 2 bonds taken from their BOND potential parameters.
     !     these are found using itbw to find the bond and then using icb.
     ! BIOVIA Code Start : Bug fix
     DIFB1 = BNDLN1 - CBOND2(ICB1)
     DIFB2 = BNDLN2 - CBOND2(ICB2)
     ! BIOVIA Code End

     !-----compute the difference of the theta angle from its equilibrium
     !     value - dthet
     ICTP = ICT(ITHETA)
     DTHET = TH(ITHETA) - CTHET2(ICTP)

     !-----the theta parameters have the outer 2 atoms ordered so that the
     !     atom with the lowest kind is first. the 4th entry in each theta
     !     parameter refers to the bond*theta force constant for the first
     !     bond in the reordered angle, the 5th to the second bond.
     !     so reorder the bond*theta parameters if the outer 2 atoms of the
     !     theta angle are not in the correct order (this saves having to
     !     reorder the theta angle)
     CBTA = CTHET4(ICTP)
     CBTB = CTHET5(ICTP)
     IF (ITE(IAC(ITHE1))  >  ITE(IAC(ITHE3))) THEN

        CTMP = CBTA
        CBTA = CBTB
        CBTB = CTMP
     ENDIF

     !-----take derivatives of the energy w.r.t. the bond a and its cartesian
     !     coordinates and add this contribution to the total derivative.

     DEBTDT = CBTA*DIFB1 + CBTB*DIFB2

     !-----the bond*bond cross-term potential constant is 3rd entry in the
     !     valence angle potential entry for this theta angle.
     KBB = CTHET3(ICTP)

     !-----now compute bond*bond energy and add it to total bond*bond energy
     EBB = EBB + KBB*DIFB1*DIFB2

     EBT = EBT + DEBTDT * DTHET

     DEBDB = KBB*DIFB1 + CBTB*DTHET
     DEBDA = KBB*DIFB2 + CBTA*DTHET

     DX(ITHE1) = DX(ITHE1) + DTHDX(1,ITHETA)*DEBTDT + DBDX1*DEBDA
     DY(ITHE1) = DY(ITHE1) + DTHDY(1,ITHETA)*DEBTDT + DBDY1*DEBDA
     DZ(ITHE1) = DZ(ITHE1) + DTHDZ(1,ITHETA)*DEBTDT + DBDZ1*DEBDA
     DX(ITHE2) = DX(ITHE2) + DTHDX(2,ITHETA)*DEBTDT - DBDX1*DEBDA &
          - DBDX2*DEBDB
     DY(ITHE2) = DY(ITHE2) + DTHDY(2,ITHETA)*DEBTDT - DBDY1*DEBDA &
          - DBDY2*DEBDB
     DZ(ITHE2) = DZ(ITHE2) + DTHDZ(2,ITHETA)*DEBTDT - DBDZ1*DEBDA &
          - DBDZ2*DEBDB
     DX(ITHE3) = DX(ITHE3) + DTHDX(3,ITHETA)*DEBTDT + DBDX2*DEBDB
     DY(ITHE3) = DY(ITHE3) + DTHDY(3,ITHETA)*DEBTDT + DBDY2*DEBDB
     DZ(ITHE3) = DZ(ITHE3) + DTHDZ(3,ITHETA)*DEBTDT + DBDZ2*DEBDB
  ENDDO
  RETURN
END SUBROUTINE EBNDANGFS


SUBROUTINE EBNDBNDFS(EBBL)

  ! Function
  !     Compute the bond*bond cross-term energy for the two outer bonds
  !     in a torsion.

  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) IBOND1,IBOND2,ICOOR,IPRINT,IPHI1,IPHI2,IPHI3,IPHI4, &
       IBB,IBBA,IATOM,INDEX,LC,LI,LJ,JCOOR,NUMITR
  real(chm_real) BNDLN1,BNDLN2,DBDX1,DBDY1,DBDZ1,DBDX2,DBDY2, &
       DBDZ2,DBDC2,VX1,VX2,VY1,VY2,VZ1,VZ2, &
       RVLN1,RVLN2,DBDC1,KBB,DEBDA,DEBDB,DIFB1,DIFB2,EBBL
  ! BIOVIA Code Start : Bug fix
  integer(chm_int4) ICB1,ICB2
  ! BIOVIA Code End

  !-----set up timing call

  DO IBB=1,NBB
     IBBA = IBBW(IBB)
     IPHI1 = IP(IBBA)
     IPHI2 = JP(IBBA)
     IPHI3 = KP(IBBA)
     IPHI4 = LP(IBBA)

     !-----get pointers to the 2 bonds in the torsion
     IBOND1 = IPBW(1,IBBA)
     IBOND2 = IPBW(3,IBBA)
     ! BIOVIA Code Start : Bug fix
     ICB1 = ICB(IBOND1)
     ICB2 = ICB(IBOND2)
     IF (ICB1 == 0 .OR. ICB2 == 0) CYCLE
     ! BIOVIA Code End

     BNDLN1 = BL(IBOND1)
     BNDLN2 = BL(IBOND2)

     IF (IPHI1 == IB(IBOND1)) THEN
        DBDX1 = DBDX(IBOND1)
        DBDY1 = DBDY(IBOND1)
        DBDZ1 = DBDZ(IBOND1)
     ELSE
        DBDX1 = -DBDX(IBOND1)
        DBDY1 = -DBDY(IBOND1)
        DBDZ1 = -DBDZ(IBOND1)
     ENDIF
     IF (IPHI3 == IB(IBOND2)) THEN
        DBDX2 = DBDX(IBOND2)
        DBDY2 = DBDY(IBOND2)
        DBDZ2 = DBDZ(IBOND2)
     ELSE
        DBDX2 = -DBDX(IBOND2)
        DBDY2 = -DBDY(IBOND2)
        DBDZ2 = -DBDZ(IBOND2)
     ENDIF

     !-----compute the difference of the 2 bond lengths from the equilibrium
     !     values for the 2 bonds taken from their BOND potential parameters.
     !     these are found using itbw to find the bond and then using icb.
     DIFB1 = BNDLN1 - CBOND2(ICB(IBOND1))
     DIFB2 = BNDLN2 - CBOND2(ICB(IBOND2))

     KBB = CBB2(ICP(IBBA))

     !-----now compute bond*bond energy and add it to total bond*bond energy
     EBBL = EBBL + KBB*DIFB1*DIFB2

     DEBDB = KBB*DIFB1
     DEBDA = KBB*DIFB2

     DX(IPHI1) = DX(IPHI1) + DBDX1*DEBDA
     DX(IPHI2) = DX(IPHI2) - DBDX1*DEBDA
     DX(IPHI3) = DX(IPHI3) + DBDX2*DEBDB
     DX(IPHI4) = DX(IPHI4) - DBDX2*DEBDB
     DY(IPHI1) = DY(IPHI1) + DBDY1*DEBDA
     DY(IPHI2) = DY(IPHI2) - DBDY1*DEBDA
     DY(IPHI3) = DY(IPHI3) + DBDY2*DEBDB
     DY(IPHI4) = DY(IPHI4) - DBDY2*DEBDB
     DZ(IPHI1) = DZ(IPHI1) + DBDZ1*DEBDA
     DZ(IPHI2) = DZ(IPHI2) - DBDZ1*DEBDA
     DZ(IPHI3) = DZ(IPHI3) + DBDZ2*DEBDB
     DZ(IPHI4) = DZ(IPHI4) - DBDZ2*DEBDB
  ENDDO
  RETURN
END SUBROUTINE EBNDBNDFS


SUBROUTINE EBONDFS_CFF(EB)
  !
  ! Function
  !     this routine computes the bond energy and first derivatives
  !     of the bond energy w.r.t the cartesian coordinates of the atoms
  !     in each bond for all bonds in a molecule.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  use fast
  implicit none

  INTEGER(chm_int4) IBOND,IBW1,IBW2,ICBP
  real(chm_real) BINV,BNDLEN,DEBDB,DXX,EB,EBND,DX2,DX3, &
       VECTX,VECTY,VECTZ, &
       DBDX1,DBDX2,DBDX3

  !-----loop over all bonds in this molecule and compute the bond length,
  !     energy and derivatives.

  DO IBOND=1,NBOND

     IBW1 = IB(IBOND)
     IBW2 = JB(IBOND)

     !-----compute the bond length and derivatives of the bond length w.r.t
     !     the cartesians.

     !-----compute the vector between the 2 atoms, whose cartesian
     !     components are in x1 and x2.
     VECTX = X(IBW1) - X(IBW2)
     VECTY = Y(IBW1) - Y(IBW2)
     VECTZ = Z(IBW1) - Z(IBW2)

     !-----the distance between the 2 atoms is simply the dot product of
     !     this vector.

     BNDLEN = VECTX*VECTX + VECTY*VECTY + VECTZ*VECTZ
     BINV = ONE/SQRT(BNDLEN)
     BNDLEN = BNDLEN*BINV
     BL(IBOND)=BNDLEN

     !-----compute the 1st derivatives of the distance/bond length.
     DBDX1 = VECTX*BINV
     DBDX2 = VECTY*BINV
     DBDX3 = VECTZ*BINV

     DBDX(IBOND) = DBDX1
     DBDY(IBOND) = DBDX2
     DBDZ(IBOND) = DBDX3

     !-----now compute the bond energy and the derivative of the bond energy
     !     w.r.t the cartesian coordinates. save the energy of each bond
     !     in ebond.
     ICBP=ICB(IBOND)
     ! BIOVIA Code Start : Bug fix
     IF (ICBP == 0) CYCLE
     ! BIOVIA Code End

     DXX = BNDLEN - CBOND2(ICBP)
     DX2 = DXX * DXX
     DX3 = DX2 * DXX
     EBND = CBOND1(ICBP)*DX2 + CBOND3(ICBP)*DX3 &
          + CBOND4(ICBP)*DX3*DXX

     EB = EB + EBND

     DEBDB = TWO*CBOND1(ICBP)*DXX + THREE*CBOND3(ICBP)*DX2 &
          + FOUR*CBOND4(ICBP)*DX3
     DX(IBW1) = DX(IBW1) + DBDX1*DEBDB
     DX(IBW2) = DX(IBW2) - DBDX1*DEBDB
     DY(IBW1) = DY(IBW1) + DBDX2*DEBDB
     DY(IBW2) = DY(IBW2) - DBDX2*DEBDB
     DZ(IBW1) = DZ(IBW1) + DBDX3*DEBDB
     DZ(IBW2) = DZ(IBW2) - DBDX3*DEBDB
  ENDDO
  !     write(6,980)eb
  ! 980 format("Bond:             ",f15.6)
  RETURN
END SUBROUTINE EBONDFS_CFF


SUBROUTINE EOPLNFS_CFF(EOPL)

  ! Function

  !     This routine computes the out of plane energy and its derivatives
  !     w.r.t. cartesian coordinates for all out of plane angles

  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) ICOPT,IOP1,IOP2,IOP3,IOP4,IVPN,IOPLN,IPLN
  real(chm_real) COP1,DCM1,DCM2,DCM3,DEO,EOPL,F1,F2,G1,G2,VLN1X2, &
       VLN2X3,VX1X2,VY1X2,VZ1X2,VX2X3,VY2X3,VZ2X3,VEC1X,VEC1Y,VEC1Z, &
       VEC2X,VEC2Y,VEC2Z,VEC3X,VEC3Y,VEC3Z,VLN3X1,VX3X1,VY3X1,VZ3X1, &
       VLEN1,VLEN2,VLEN3,CHI,CHI1,CHI2,CHI3,X2,Y2,Z2, &
       CM,THETA1,THETA2,THETA3,ABM,TEMP1,TEMP2,TEMP3, &
       DSX1,DSX2,DSX3,DSX4,DSX5,DSX6,DSX7,DSX8,DSX9,DSX10,DSX11,DSX12, &
       ESX1,ESX2,ESX3,ESX4,ESX5,ESX6,ESX7,ESX8,ESX9,ESX10,ESX11,ESX12, &
       FSX1,FSX2,FSX3,FSX4,FSX5,FSX6,FSX7,FSX8,FSX9,FSX10,FSX11,FSX12
  LOGICAL NEWT

  NEWT = .TRUE.

  !-----loop over all out of planes in this molecule and compute
  !     the out of plane angle, energy and derivatives.

  DO IOPLN=1,NIMPHI
     ICOPT = ICI(IOPLN)
     ! BIOVIA Code Start : Bug fix
     IF (ICOPT == 0) CYCLE
     ! BIOVIA Code End

     !   calculate the 1st derivative (w.r.t. the cartesian
     !   coordinates) of the (averaged) Wilson angle for the out of plane
     !   coordinate defined by atoms IOP1, IOP2, IOP3, and IOP4, where IOP2
     !   is the out of plane atom.
     !   The out-of-plane coordinate is returned by chi.

     IOP2 = JM(IOPLN)
     X2 = X(IOP2)
     Y2 = Y(IOP2)
     Z2 = Z(IOP2)

     !   Loop over the 3 Wilson angles which are to be averaged.
     !   Compute the three vectors in the out of plane

     IOP4 = LM(IOPLN)
     VEC1X = X(IOP4) - X2
     VEC1Y = Y(IOP4) - Y2
     VEC1Z = Z(IOP4) - Z2
     IOP3 = KM(IOPLN)
     VEC2X = X(IOP3) - X2
     VEC2Y = Y(IOP3) - Y2
     VEC2Z = Z(IOP3) - Z2
     IOP1 = IM(IOPLN)
     VEC3X = X(IOP1) - X2
     VEC3Y = Y(IOP1) - Y2
     VEC3Z = Z(IOP1) - Z2
     !   Compute the cross products 1X2, 2X3 and 3X1.
     VX1X2 = VEC1Y*VEC2Z - VEC1Z*VEC2Y
     VY1X2 = VEC1Z*VEC2X - VEC1X*VEC2Z
     VZ1X2 = VEC1X*VEC2Y - VEC1Y*VEC2X
     VX2X3 = VEC2Y*VEC3Z - VEC2Z*VEC3Y
     VY2X3 = VEC2Z*VEC3X - VEC2X*VEC3Z
     VZ2X3 = VEC2X*VEC3Y - VEC2Y*VEC3X
     VX3X1 = VEC3Y*VEC1Z - VEC3Z*VEC1Y
     VY3X1 = VEC3Z*VEC1X - VEC3X*VEC1Z
     VZ3X1 = VEC3X*VEC1Y - VEC3Y*VEC1X
     !   Compute the lengths of the three bonds and the three cross products.
     VLEN1 = SQRT(VEC1X*VEC1X + VEC1Y*VEC1Y + VEC1Z*VEC1Z)
     VLEN2 = SQRT(VEC2X*VEC2X + VEC2Y*VEC2Y + VEC2Z*VEC2Z)
     VLEN3 = SQRT(VEC3X*VEC3X + VEC3Y*VEC3Y + VEC3Z*VEC3Z)
     VLN1X2 = SQRT(VX1X2*VX1X2 + VY1X2*VY1X2 + VZ1X2*VZ1X2)
     VLN2X3 = SQRT(VX2X3*VX2X3 + VY2X3*VY2X3 + VZ2X3*VZ2X3)
     VLN3X1 = SQRT(VX3X1*VX3X1 + VY3X1*VY3X1 + VZ3X1*VZ3X1)
     !   Calculate the out of plane distance for each angle
     CHI1 = (VX1X2*VEC3X + VY1X2*VEC3Y + VZ1X2*VEC3Z)/VLN1X2
     CHI2 = (VX3X1*VEC2X + VY3X1*VEC2Y + VZ3X1*VEC2Z)/VLN3X1
     CHI3 = (VX2X3*VEC1X + VY2X3*VEC1Y + VZ2X3*VEC1Z)/VLN2X3
     !   Get the derivatives of the first out of plane distance (chi1).
     ABM = ONE/VLN1X2
     F1 = (VX1X2 * VEC3X + VY1X2 * VEC3Y + VZ1X2 * VEC3Z)*ABM*ABM*ABM
     TEMP1 = VEC2X - VEC1X
     TEMP2 = VEC2Y - VEC1Y
     TEMP3 = VEC2Z - VEC1Z
     CM  = ONE/VLEN3
     THETA1 = ASIN (CHI1*CM)
     F2 = CHI1*CM
     DCM1 = VEC3X*CM
     DCM2 = VEC3Y*CM
     DCM3 = VEC3Z*CM
     G1 = CM * THIRD / COS(THETA1)
     G2 = G1 * F2
     ABM = ABM * G1
     F1 = F1 * G1
     !
     DSX1  = VX1X2 * ABM - G2*DCM1
     DSX2  = VY1X2 * ABM - G2*DCM2
     DSX3  = VZ1X2 * ABM - G2*DCM3
     DSX4  = (TEMP3*VEC3Y - TEMP2*VEC3Z - VX1X2) * ABM &
          - (TEMP3*VY1X2 - TEMP2*VZ1X2) * F1 + G2*DCM1
     DSX5  = (TEMP1*VEC3Z - TEMP3*VEC3X - VY1X2) * ABM &
          - (TEMP1*VZ1X2 - TEMP3*VX1X2) * F1 + G2*DCM2
     DSX6  = (TEMP2*VEC3X - TEMP1*VEC3Y - VZ1X2) * ABM &
          - (TEMP2*VX1X2 - TEMP1*VY1X2) * F1 + G2*DCM3
     DSX7  = (VEC1Z*VEC3Y - VEC1Y*VEC3Z) * ABM &
          - (VEC1Z*VY1X2 - VEC1Y*VZ1X2) * F1
     DSX8  = (VEC1X*VEC3Z - VEC1Z*VEC3X) * ABM &
          - (VEC1X*VZ1X2 - VEC1Z*VX1X2) * F1
     DSX9  = (VEC1Y*VEC3X - VEC1X*VEC3Y) * ABM &
          - (VEC1Y*VX1X2 - VEC1X*VY1X2) * F1
     DSX10 = (VEC2Y*VEC3Z - VEC2Z*VEC3Y) * ABM &
          - (VEC2Y*VZ1X2 - VEC2Z*VY1X2) * F1
     DSX11 = (VEC2Z*VEC3X - VEC2X*VEC3Z) * ABM &
          - (VEC2Z*VX1X2 - VEC2X*VZ1X2) * F1
     DSX12 = (VEC2X*VEC3Y - VEC2Y*VEC3X) * ABM &
          - (VEC2X*VY1X2 - VEC2Y*VX1X2) * F1
     !
     !   Get the derivatives of the second out of plane distance (chi2).
     ABM = ONE/VLN3X1
     F1 = (VX3X1 * VEC2X + VY3X1 * VEC2Y + VZ3X1 * VEC2Z)*ABM*ABM*ABM
     TEMP1 = VEC1X - VEC3X
     TEMP2 = VEC1Y - VEC3Y
     TEMP3 = VEC1Z - VEC3Z
     CM  = ONE/VLEN2
     THETA2 = ASIN (CHI2*CM)
     F2 = CHI2*CM
     DCM1 = VEC2X*CM
     DCM2 = VEC2Y*CM
     DCM3 = VEC2Z*CM
     G1 = CM * THIRD / COS(THETA2)
     G2 = G1 * F2
     ABM = ABM * G1
     F1 = F1 * G1
     !
     ESX1  = VX3X1 * ABM - G2*DCM1
     ESX2  = VY3X1 * ABM - G2*DCM2
     ESX3  = VZ3X1 * ABM - G2*DCM3
     ESX4  = (TEMP3*VEC2Y - TEMP2*VEC2Z - VX3X1) * ABM &
          - (TEMP3*VY3X1 - TEMP2*VZ3X1) * F1 + G2*DCM1
     ESX5  = (TEMP1*VEC2Z - TEMP3*VEC2X - VY3X1) * ABM &
          - (TEMP1*VZ3X1 - TEMP3*VX3X1) * F1 + G2*DCM2
     ESX6  = (TEMP2*VEC2X - TEMP1*VEC2Y - VZ3X1) * ABM &
          - (TEMP2*VX3X1 - TEMP1*VY3X1) * F1 + G2*DCM3
     ESX7  = (VEC3Z*VEC2Y - VEC3Y*VEC2Z) * ABM &
          - (VEC3Z*VY3X1 - VEC3Y*VZ3X1) * F1
     ESX8  = (VEC3X*VEC2Z - VEC3Z*VEC2X) * ABM &
          - (VEC3X*VZ3X1 - VEC3Z*VX3X1) * F1
     ESX9  = (VEC3Y*VEC2X - VEC3X*VEC2Y) * ABM &
          - (VEC3Y*VX3X1 - VEC3X*VY3X1) * F1
     ESX10 = (VEC1Y*VEC2Z - VEC1Z*VEC2Y) * ABM &
          - (VEC1Y*VZ3X1 - VEC1Z*VY3X1) * F1
     ESX11 = (VEC1Z*VEC2X - VEC1X*VEC2Z) * ABM &
          - (VEC1Z*VX3X1 - VEC1X*VZ3X1) * F1
     ESX12 = (VEC1X*VEC2Y - VEC1Y*VEC2X) * ABM &
          - (VEC1X*VY3X1 - VEC1Y*VX3X1) * F1
     !
     !   Get the derivatives of the third out of plane distance (chi3).
     ABM = ONE/VLN2X3
     F1 = (VX2X3 * VEC1X + VY2X3 * VEC1Y + VZ2X3 * VEC1Z)*ABM*ABM*ABM
     TEMP1 = VEC3X - VEC2X
     TEMP2 = VEC3Y - VEC2Y
     TEMP3 = VEC3Z - VEC2Z
     CM  = ONE/VLEN1
     THETA3 = ASIN (CHI3*CM)
     F2 = CHI3*CM
     DCM1 = VEC1X*CM
     DCM2 = VEC1Y*CM
     DCM3 = VEC1Z*CM
     G1 = CM * THIRD / COS(THETA3)
     G2 = G1 * F2
     ABM = ABM * G1
     F1 = F1 * G1
     !
     FSX1  = VX2X3 * ABM - G2*DCM1
     FSX2  = VY2X3 * ABM - G2*DCM2
     FSX3  = VZ2X3 * ABM - G2*DCM3
     FSX4  = (TEMP3*VEC1Y - TEMP2*VEC1Z - VX2X3) * ABM &
          - (TEMP3*VY2X3 - TEMP2*VZ2X3) * F1 + G2*DCM1
     FSX5  = (TEMP1*VEC1Z - TEMP3*VEC1X - VY2X3) * ABM &
          - (TEMP1*VZ2X3 - TEMP3*VX2X3) * F1 + G2*DCM2
     FSX6  = (TEMP2*VEC1X - TEMP1*VEC1Y - VZ2X3) * ABM &
          - (TEMP2*VX2X3 - TEMP1*VY2X3) * F1 + G2*DCM3
     FSX7  = (VEC2Z*VEC1Y - VEC2Y*VEC1Z) * ABM &
          - (VEC2Z*VY2X3 - VEC2Y*VZ2X3) * F1
     FSX8  = (VEC2X*VEC1Z - VEC2Z*VEC1X) * ABM &
          - (VEC2X*VZ2X3 - VEC2Z*VX2X3) * F1
     FSX9  = (VEC2Y*VEC1X - VEC2X*VEC1Y) * ABM &
          - (VEC2Y*VX2X3 - VEC2X*VY2X3) * F1
     FSX10 = (VEC3Y*VEC1Z - VEC3Z*VEC1Y) * ABM &
          - (VEC3Y*VZ2X3 - VEC3Z*VY2X3) * F1
     FSX11 = (VEC3Z*VEC1X - VEC3X*VEC1Z) * ABM &
          - (VEC3Z*VX2X3 - VEC3X*VZ2X3) * F1
     FSX12 = (VEC3X*VEC1Y - VEC3Y*VEC1X) * ABM &
          - (VEC3X*VY2X3 - VEC3Y*VX2X3) * F1
     !
     !   Divide the coordinate and its derivatives by 3, since 3 angles are
     !   used to compute a symmetrized (i.e., averaged) coordinate.
     !
     CHI= (THETA1+THETA2+THETA3)*THIRD
     OPLN(IOPLN) = CHI
     COP1 = COPLN1(ICOPT)*CHI
     !
     !-----sum the out of plane energy into the total energy.
     EOPL = EOPL + COP1*CHI
     DEO = TWO*COP1
     !
     DX(IOP1) = DX(IOP1)+(DSX1  + ESX10 + FSX7)*DEO
     DY(IOP1) = DY(IOP1)+(DSX2  + ESX11 + FSX8)*DEO
     DZ(IOP1) = DZ(IOP1)+(DSX3  + ESX12 + FSX9)*DEO
     DX(IOP2) = DX(IOP2)+(DSX4  + ESX4  + FSX4)*DEO
     DY(IOP2) = DY(IOP2)+(DSX5  + ESX5  + FSX5)*DEO
     DZ(IOP2) = DZ(IOP2)+(DSX6  + ESX6  + FSX6)*DEO
     DX(IOP3) = DX(IOP3)+(DSX7  + ESX1  + FSX10)*DEO
     DY(IOP3) = DY(IOP3)+(DSX8  + ESX2  + FSX11)*DEO
     DZ(IOP3) = DZ(IOP3)+(DSX9  + ESX3  + FSX12)*DEO
     DX(IOP4) = DX(IOP4)+(DSX10 + ESX7  + FSX1)*DEO
     DY(IOP4) = DY(IOP4)+(DSX11 + ESX8  + FSX2)*DEO
     DZ(IOP4) = DZ(IOP4)+(DSX12 + ESX9  + FSX3)*DEO
  ENDDO
  !     write(6,983)eopl
  ! 983 format("OutOfPlane:       ",f15.6)
  RETURN
END SUBROUTINE EOPLNFS_CFF


SUBROUTINE EPHIFS_CFF(EP,ETP,EBP,EMBP,ETTP)

  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use deriv
  use energym
  use code
  use coord
  use param
  use psf
  use cff_fcm
  use stream
  implicit none

  INTEGER(chm_int4) ICPT,IPHI,IPHI1,IPHI2,IPHI3,IPHI4,IPH,IFILL,NPHM, &
       ITHTA,ITHTB,IVPN,ISGN1,ISGN2,ISGN3,ISGN4,IBOND1,IBOND2,IBOND3
  real(chm_real) COSPHI,CPH1,CPH2,EP,PHI1,PHI2,PHI3,EPH,EBP,ETP, &
       ETTP,EMBP,XSIGN,SMALLA,VLN1X2,VLN3X2,VXIJ,VXKL,VXKJ, &
       VYIJ,VYKJ,VYKL,VZIJ,VZKJ,VZKL,VX1X2,VY1X2,VZ1X2,VX3X2,VY3X2, &
       VZ3X2,C1,C2,C3,C4,C5,C6,COSV12,COSV23,DOTP12,DOTP23,SINV12, &
       SINV23,VLEN1,VLEN2,VLEN3,CTTP,DTHE1,DTHE2,DTTPDP,DTTPDT, &
       DTTPDA,DPHI,DPHI1,DPHI2,DPHI3,DPHI4,DPHI5,DPHI6,DPHI7, &
       DPHI8,DPHI9,DPHI10,DPHI11,DPHI12,DIFB1,DIFB2,DIFB3, &
       SV12,SV112,SV23,SV223,CPH3,S1,S2,S3,COSP1,COSP2,COSP3, &
       ENGB1,ENGB2,ENGB3,ENGT1,ENGT2,SINP1,SINP2,SINP3, &
       DENG,DBDX1,DBDY1,DBDZ1,DBDX2,DBDY2,DBDZ2,DBDX3,DBDY3,DBDZ3
  real(chm_real) DENGB1,DENGB2,DENGB3,DENGT1,DENGT2

  DATA SMALLA/0.0001D0/


  IF(.NOT.QETERM(DIHE)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES_CFF

  DO IPHI=1,NPHI
     ! BIOVIA Code Start : Bug fix
     ICPT = ICP(IPHI)
     IF (ICPT == 0) CYCLE
     ! BIOVIA Code End
     !-----compute the 3 vectors in the torsion angle.
     IPHI1 = IP(IPHI)
     IPHI2 = JP(IPHI)
     IPHI3 = KP(IPHI)
     IPHI4 = LP(IPHI)
     VXIJ = X(IPHI1) - X(IPHI2)
     VYIJ = Y(IPHI1) - Y(IPHI2)
     VZIJ = Z(IPHI1) - Z(IPHI2)
     VXKJ = X(IPHI3) - X(IPHI2)
     VYKJ = Y(IPHI3) - Y(IPHI2)
     VZKJ = Z(IPHI3) - Z(IPHI2)
     VXKL = X(IPHI3) - X(IPHI4)
     VYKL = Y(IPHI3) - Y(IPHI4)
     VZKL = Z(IPHI3) - Z(IPHI4)

     !-----compute the cross product of vectors 1 X 2 and 3 X 2.
     VX1X2 = VYIJ*VZKJ - VZIJ*VYKJ
     VY1X2 = -VXIJ*VZKJ + VZIJ*VXKJ
     VZ1X2 = VXIJ*VYKJ - VYIJ*VXKJ
     VX3X2 = VYKL*VZKJ - VZKL*VYKJ
     VY3X2 = -VXKL*VZKJ + VZKL*VXKJ
     VZ3X2 = VXKL*VYKJ - VYKL*VXKJ

     !-----the torsion angle is given by the angle between the 2 vectors
     !     from the above cross product. (i.e. the scalar product of the 2
     !     vectors).
     !     NOTE. The direction of the 2 cross-products is such that
     !           when the torsion angle is 0 the above angle is 180,
     !           and when it is 180 the above angle is 0.
     !
     VLN1X2 = MAX(SMALLA,VX1X2*VX1X2 + VY1X2*VY1X2 + VZ1X2*VZ1X2)
     VLN3X2 = MAX(SMALLA,VX3X2*VX3X2 + VY3X2*VY3X2 + VZ3X2*VZ3X2)
     VLN1X2 = SQRT(VLN1X2)
     VLN3X2 = SQRT(VLN3X2)
     COSPHI = (VX1X2*VX3X2 + VY1X2*VY3X2 + VZ1X2*VZ3X2) &
          / (VLN1X2*VLN3X2)
     COSPHI = MIN(ONE,MAX(-ONE,COSPHI))

     XSIGN = VXKJ*(VY1X2*VZ3X2 - VZ1X2*VY3X2) &
          + VYKJ*(-VX1X2*VZ3X2 + VZ1X2*VX3X2) &
          + VZKJ*(VX1X2*VY3X2 - VY1X2*VX3X2)

     !-----save the torsion angle (on range 0-360) in torsion angle array.
     !     NOTE. because phiang is now 180 - or + phiang, cosphi does not
     !           correspond to phiang after this point. cos(phiang) is
     !           actually -cosphi.

     PHI1 = PI - ACOS(COSPHI)*SIGN(ONE,-XSIGN)
     PH(IPHI) = PHI1
     PHI2 = PHI1 * TWO
     PHI3 = PHI1 * THREE

     !-----compute the length of the 3 vectors of the torsion angle.
     VLEN1 = BL(IPBW(1,IPHI))
     VLEN2 = BL(IPBW(2,IPHI))
     VLEN3 = BL(IPBW(3,IPHI))
     DOTP12 = VXIJ*VXKJ + VYIJ*VYKJ + VZIJ*VZKJ
     DOTP23 = VXKJ*VXKL + VYKJ*VYKL + VZKJ*VZKL

     !-----find the cosine of the angle between vectr1 (atoms 1-->2) and
     !     vectr2 (atoms 3-->2) and between vectr2 and vectr3 (atoms 3-->4)
     !     (by cos alpha = dot product/(vector1 length*vector2 length)
     !     from the cosine get the sine of these 2 angles.
     COSV12 = DOTP12/(VLEN1*VLEN2)
     COSV23 = DOTP23/(VLEN2*VLEN3)
     SINV12 = SQRT(ONE - COSV12*COSV12)
     SINV23 = SQRT(ONE - COSV23*COSV23)
     SV12 = SINV12*VLN1X2
     SV23 = SINV23*VLN3X2
     SV112 = VLEN1*SV12
     C1 = ONE/SV112
     C2 = -(VLEN2 - VLEN1*COSV12)/(VLEN2*SV112)
     SV223 = VLEN2*SV23
     C3 = COSV23/SV223
     C4 = -(VLEN2 - VLEN3*COSV23)/(VLEN3*SV223)
     C5 = COSV12/(VLEN2*SV12)
     C6 = ONE/(VLEN3*SV23)

     !-----calculate the energy and derivatives for the barrier.

     ICPT = ICP(IPHI)
     CPH1 = CPHI1(ICPT)
     CPH2 = CPHI2(ICPT)
     CPH3 = CPHI3(ICPT)
     CTTP = -CPHI4(ICPT)
     S1 = CSGN1(ICPT)
     S2 = CSGN2(ICPT)
     S3 = CSGN3(ICPT)
     EPH=CPH1*(ONE-S1*COS(PHI1)) + CPH2*(ONE-S2*COS(PHI2)) &
          + CPH3*(ONE-S3*COS(PHI3))

     !-----get pointers to the 2 valence angles and 3 bonds in the torsion
     IBOND1 = IPBW(1,IPHI)
     IBOND2 = IPBW(2,IPHI)
     IBOND3 = IPBW(3,IPHI)
     ITHTA = IPTW(1,IPHI)
     ITHTB = IPTW(2,IPHI)

     ! BIOVIA Code Start : Bug fix
     DIFB1 = 0
     DIFB2 = 0
     DIFB3 = 0
     IF (ICB(IBOND1) > 0) DIFB1 = BL(IBOND1) - CBOND2(ICB(IBOND1))
     IF (ICB(IBOND2) > 0) DIFB2 = BL(IBOND2) - CBOND2(ICB(IBOND2))
     IF (ICB(IBOND3) > 0) DIFB3 = BL(IBOND3) - CBOND2(ICB(IBOND3))

     DTHE1 = 0
     DTHE2 = 0
     IF (ICT(ITHTA) > 0) DTHE1 = TH(ITHTA) - CTHET2(ICT(ITHTA))
     IF (ICT(ITHTB) > 0) DTHE2 = TH(ITHTB) - CTHET2(ICT(ITHTB))
     ! BIOVIA Code End
     COSP1 = COS(PHI1)
     COSP2 = COS(PHI2)
     COSP3 = COS(PHI3)

     !-----compute the bond*phi energy

     ENGB1 = CBP11(ICPT)*COSP1 + CBP12(ICPT)*COSP2+CBP13(ICPT)*COSP3
     ENGB2 = CBP21(ICPT)*COSP1 + CBP22(ICPT)*COSP2+CBP23(ICPT)*COSP3
     ENGB3 = CBP31(ICPT)*COSP1 + CBP32(ICPT)*COSP2+CBP33(ICPT)*COSP3

     !-----add the energy to the total theta*theta*phi energy

     EP = EP + EPH
     EBP = EBP + DIFB1*ENGB1 + DIFB3*ENGB3
     EMBP = EMBP + DIFB2*ENGB2
     ENGT1 = CTP11(ICPT)*COSP1 + CTP12(ICPT)*COSP2+CTP13(ICPT)*COSP3
     ENGT2 = CTP21(ICPT)*COSP1 + CTP22(ICPT)*COSP2+CTP23(ICPT)*COSP3
     ETP = ETP + DTHE1*ENGT1 + DTHE2*ENGT2
     ETTP = ETTP + CTTP*COSPHI*DTHE1*DTHE2

     DPHI=CPH1*S1*SIN(PHI1) + TWO*CPH2*S2*SIN(PHI2) &
          + THREE*CPH3*S3*SIN(PHI3)
     SINP1 = -SIN(PHI1)
     SINP2 = -TWO*SIN(PHI2)
     SINP3 = -THREE*SIN(PHI3)
     DENGB1 = CBP11(ICPT)*SINP1+CBP12(ICPT)*SINP2+CBP13(ICPT)*SINP3
     DENGB2 = CBP21(ICPT)*SINP1+CBP22(ICPT)*SINP2+CBP23(ICPT)*SINP3
     DENGB3 = CBP31(ICPT)*SINP1+CBP32(ICPT)*SINP2+CBP33(ICPT)*SINP3
     DENGT1 = CTP11(ICPT)*SINP1+CTP12(ICPT)*SINP2+CTP13(ICPT)*SINP3
     DENGT2 = CTP21(ICPT)*SINP1+CTP22(ICPT)*SINP2+CTP23(ICPT)*SINP3
     DENG = DIFB1*DENGB1 + DIFB2*DENGB2 + DIFB3*DENGB3 &
          + DTHE1*DENGT1 + DTHE2*DENGT2
     IF (IPHI1 == IB(IBOND1)) THEN
        DBDX1 = DBDX(IBOND1)
        DBDY1 = DBDY(IBOND1)
        DBDZ1 = DBDZ(IBOND1)
     ELSE
        DBDX1 = -DBDX(IBOND1)
        DBDY1 = -DBDY(IBOND1)
        DBDZ1 = -DBDZ(IBOND1)
     ENDIF
     IF (IPHI2 == IB(IBOND2)) THEN
        DBDX2 = DBDX(IBOND2)
        DBDY2 = DBDY(IBOND2)
        DBDZ2 = DBDZ(IBOND2)
     ELSE
        DBDX2 = -DBDX(IBOND2)
        DBDY2 = -DBDY(IBOND2)
        DBDZ2 = -DBDZ(IBOND2)
     ENDIF
     IF (IPHI3 == IB(IBOND3)) THEN
        DBDX3 = DBDX(IBOND3)
        DBDY3 = DBDY(IBOND3)
        DBDZ3 = DBDZ(IBOND3)
     ELSE
        DBDX3 = -DBDX(IBOND3)
        DBDY3 = -DBDY(IBOND3)
        DBDZ3 = -DBDZ(IBOND3)
     ENDIF

     DTTPDP = CTTP*SIN(PHI1)*DTHE1*DTHE2 + DPHI + DENG
     DTTPDT = CTTP*COSPHI*DTHE2 + ENGT1
     DTTPDA = CTTP*COSPHI*DTHE1 + ENGT2

     !-----compute the 1st derivatives of the angle w.r.t the cartesian
     !     coordinates from the derivatives w.r.t the vectors forming the
     !     angle.
     ISGN1 = 2 + IPHFLG(1,IPHI)
     ISGN2 = 2 - IPHFLG(1,IPHI)
     ISGN3 = 2 + IPHFLG(2,IPHI)
     ISGN4 = 2 - IPHFLG(2,IPHI)

     DX(IPHI1) = DX(IPHI1) &
          + VX1X2*C1*DTTPDP + DTHDX(ISGN1,ITHTA)*DTTPDT &
          + DBDX1*ENGB1
     DY(IPHI1) = DY(IPHI1) &
          + VY1X2*C1*DTTPDP + DTHDY(ISGN1,ITHTA)*DTTPDT &
          + DBDY1*ENGB1
     DZ(IPHI1) = DZ(IPHI1) &
          + VZ1X2*C1*DTTPDP + DTHDZ(ISGN1,ITHTA)*DTTPDT &
          + DBDZ1*ENGB1
     DX(IPHI2) = DX(IPHI2) &
          + (C2*VX1X2 - VX3X2*C3)*DTTPDP &
          + DTHDX(2,ITHTA)*DTTPDT + DTHDX(ISGN3,ITHTB)*DTTPDA &
          - DBDX1*ENGB1 + DBDX2*ENGB2
     DY(IPHI2) = DY(IPHI2) &
          + (C2*VY1X2 - VY3X2*C3)*DTTPDP &
          + DTHDY(2,ITHTA)*DTTPDT + DTHDY(ISGN3,ITHTB)*DTTPDA &
          - DBDY1*ENGB1 + DBDY2*ENGB2
     DZ(IPHI2) = DZ(IPHI2) &
          + (C2*VZ1X2 - VZ3X2*C3)*DTTPDP &
          + DTHDZ(2,ITHTA)*DTTPDT + DTHDZ(ISGN3,ITHTB)*DTTPDA &
          - DBDZ1*ENGB1 + DBDZ2*ENGB2
     DX(IPHI3) = DX(IPHI3) &
          + (C4*VX3X2 - VX1X2*C5)*DTTPDP &
          + DTHDX(ISGN2,ITHTA)*DTTPDT + DTHDX(2,ITHTB)*DTTPDA &
          - DBDX2*ENGB2 + DBDX3*ENGB3
     DY(IPHI3) = DY(IPHI3) &
          + (C4*VY3X2 - VY1X2*C5)*DTTPDP &
          + DTHDY(ISGN2,ITHTA)*DTTPDT + DTHDY(2,ITHTB)*DTTPDA &
          - DBDY2*ENGB2 + DBDY3*ENGB3
     DZ(IPHI3) = DZ(IPHI3) &
          + (C4*VZ3X2 - VZ1X2*C5)*DTTPDP &
          + DTHDZ(ISGN2,ITHTA)*DTTPDT + DTHDZ(2,ITHTB)*DTTPDA &
          - DBDZ2*ENGB2 + DBDZ3*ENGB3
     DX(IPHI4) = DX(IPHI4) &
          + VX3X2*C6*DTTPDP + DTHDX(ISGN4,ITHTB)*DTTPDA &
          - DBDX3*ENGB3
     DY(IPHI4) = DY(IPHI4) &
          + VY3X2*C6*DTTPDP + DTHDY(ISGN4,ITHTB)*DTTPDA &
          - DBDY3*ENGB3
     DZ(IPHI4) = DZ(IPHI4) &
          + VZ3X2*C6*DTTPDP + DTHDZ(ISGN4,ITHTB)*DTTPDA &
          - DBDZ3*ENGB3
  ENDDO
  if(prnlev > 6) then
     write(6,982)ep
982  format("Torsion:          ",f15.6)
     write(6,987)ebp
987  format("EndBondTorsion:   ",f15.6)
     write(6,988)embp
988  format("MiddleBondTorsion:",f15.6)
     write(6,989)etp
989  format("AngleTorsion:     ",f15.6)
     write(6,990)ettp
990  format("AngleAngleTorsion:",f15.6)
  endif
  RETURN
END SUBROUTINE EPHIFS_CFF


SUBROUTINE BONDLEN_CFF

  ! Function
  !     This routine computes the bond lengths and derivatives wrt bond
  !     length in cases where bond energies are not calculated. It uses 
  !     the global coordinate arrays to do the calculation

  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  use fast
  implicit none

  INTEGER(chm_int4) IBOND,IBW1,IBW2
  real(chm_real) BINV,BNDLEN,VECTX,VECTY,VECTZ

  !-----loop over all bonds in this molecule and compute the bond length
  !     and derivatives.

  DO IBOND=1,NBOND

     IBW1 = IB(IBOND)
     IBW2 = JB(IBOND)

     !-----compute the bond length and derivatives of the bond length w.r.t
     !     the cartesians.

     !-----compute the vector between the 2 atoms, whose cartesian
     !     components are in x1 and x2.
     VECTX = X(IBW1) - X(IBW2)
     VECTY = Y(IBW1) - Y(IBW2)
     VECTZ = Z(IBW1) - Z(IBW2)

     !-----the distance between the 2 atoms is simply the dot product of
     !     this vector.

     BNDLEN = VECTX*VECTX + VECTY*VECTY + VECTZ*VECTZ
     BINV = ONE/SQRT(BNDLEN)
     BL(IBOND)=BNDLEN*BINV

     !-----compute the 1st derivatives of the distance/bond length.

     DBDX(IBOND) = VECTX*BINV
     DBDY(IBOND) = VECTY*BINV
     DBDZ(IBOND) = VECTZ*BINV
  ENDDO
  RETURN
END SUBROUTINE BONDLEN_CFF


SUBROUTINE ANGLES_CFF

  ! Function
  !     This routine calculates the theta angle properties using the global
  !     coordinate arrays. The angle properties computed are:
  !     theta angle and its cosine, the 1st derivatives of the angle w.r.t 
  !     the cartesian coordinates

  use chm_kinds
  use dimens_fcm
  use energym
  use consta
  use number
  use deriv
  use code
  use coord
  use param
  use psf
  use cff_fcm
  implicit none

  INTEGER(chm_int4) ICTI,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA
  real(chm_real) COSTHE,DETD,DTHETA,ET,ETHET,VLEN1,VLEN2, &
       VX1,VX2,VY1,VY2,VZ1,VZ2,RSNTHE,RVLN1,RVLN12, &
       RVLN2,RVLN22,SINTH2,SINTHE,SMALLA,VL1DV2,VL2DV1, &
       DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32,DTHTA2, &
       DTHET1,DTHET2,DTHET3,DTHET4,DTHET5,DTHET6,DTHET7,DTHET8,DTHET9

  DATA SMALLA/1.0D-10/

  IF (.NOT.QETERM(BOND)) CALL BONDLEN_CFF

  !-----loop over all valence angles in this molecule and compute the
  !     valence angle, energy and derivatives.

  DO ITHETA=1,NTHETA

     !-----save the atom numbers of the 3 atoms involved in this angle.
     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)

     !-----compute the 2 vectors of the valence angle,
     !     thevec(,1) (atoms 2->1) and thevec(,2) (atoms 2->3)a.
     VX1 = X(ITHE1) - X(ITHE2)
     VX2 = X(ITHE3) - X(ITHE2)
     VY1 = Y(ITHE1) - Y(ITHE2)
     VY2 = Y(ITHE3) - Y(ITHE2)
     VZ1 = Z(ITHE1) - Z(ITHE2)
     VZ2 = Z(ITHE3) - Z(ITHE2)

     !-----find the cosine of the angle between these 2 vectors,
     !     the valence angle, and save it in theang.
     !     the length of these 2 vectors is also returned in vlen1 and vlen2.

     VLEN1 = BL(ITBW(1,ITHETA))
     VLEN2 = BL(ITBW(2,ITHETA))

     COSTHE = (VX1*VX2 + VY1*VY2 + VZ1*VZ2)/(VLEN1*VLEN2)
     COSTHE = MIN(ONE,MAX(-ONE,COSTHE))

     COSTH(ITHETA) = COSTHE
     TH(ITHETA) = ACOS(COSTHE)

     !-----compute the first derivatives of the cosine of the angle
     !     or the angle w.r.t. the 2 vectors forming tha angle.
     !     vl1dv2 is the length of vectr1 divided by the length of vectr2.
     !     vl2dv1 is the length of vectr2 divided by the length of vectr1.
     RVLN1 = ONE/VLEN1
     RVLN2 = ONE/VLEN2
     RVLN12 = RVLN1*RVLN1
     RVLN22 = RVLN2*RVLN2
     VL1DV2 = VLEN1*RVLN2
     VL2DV1 = VLEN2*RVLN1

     DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
     DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
     DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
     DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
     DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
     DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)

     !-----compute the derivatives for the angle instead of the cosine of
     !     the angle.
     !     first, compute the sine of the angle (from the cosine)
     !     MAKE SURE the angle is not zero, if it is make it a small
     !     number. (this gives rise to a discontinuity in the derivatives
     !     below the value of small which can cause problems).
     !     make the sign of the sine correspond to the sign of isn.
     SINTH2 = ABS(ONE - COSTHE*COSTHE)
     SINTH2 = MAX(SMALLA,SINTH2)
     SINTHE = SQRT(SINTH2)

     RSNTHE = ONE/SINTHE

     !-----finally compute the first derivatives for the angle, not the
     !     cosine.
     !     NOTE. This must come last as dvec is also used in the computation
     !           of the 2nd derivatives of the angle and the original
     !           dvec (for the cosine of the angle) is needed there.
     DVEC11 = -DVEC11*RSNTHE
     DVEC21 = -DVEC21*RSNTHE
     DVEC31 = -DVEC31*RSNTHE
     DVEC12 = -DVEC12*RSNTHE
     DVEC22 = -DVEC22*RSNTHE
     DVEC32 = -DVEC32*RSNTHE

     !-----compute the 1st derivatives of the angle w.r.t the cartesian
     !     coordinates from the derivatives w.r.t the vectors forming the
     !     angle.
     DTHDX(1,ITHETA) = DVEC11
     DTHDY(1,ITHETA) = DVEC21
     DTHDZ(1,ITHETA) = DVEC31
     DTHDX(2,ITHETA) = -DVEC11 - DVEC12
     DTHDY(2,ITHETA) = -DVEC21 - DVEC22
     DTHDZ(2,ITHETA) = -DVEC31 - DVEC32
     DTHDX(3,ITHETA) = DVEC12
     DTHDY(3,ITHETA) = DVEC22
     DTHDZ(3,ITHETA) = DVEC32

  ENDDO
  RETURN
END SUBROUTINE ANGLES_CFF

SUBROUTINE BONDLEN1_CFF(X, Y, Z)

  ! Function
  !     this routine computes the bond lengths and derivatives wrt bond
  !     length in cases where bond energies are not calculated using 
  !     coordinate arrays passed in.

  use chm_kinds
  use dimens_fcm
  use number
  use deriv
  use code
  use param
  use psf
  use cff_fcm
  use fast
  use stream
  implicit none

  real(chm_real) X(*),Y(*),Z(*)      

  INTEGER(chm_int4) IBOND,IBW1,IBW2
  real(chm_real) BINV,BNDLEN,VECTX,VECTY,VECTZ

  !-----loop over all bonds in this molecule and compute the bond length
  !     and derivatives.

  DO IBOND=1,NBOND

     IBW1 = IB(IBOND)
     IBW2 = JB(IBOND)

     !-----compute the bond length and derivatives of the bond length w.r.t
     !     the cartesians.
     !
     !-----compute the vector between the 2 atoms, whose cartesian
     !     components are in x1 and x2.
     VECTX = X(IBW1) - X(IBW2)
     VECTY = Y(IBW1) - Y(IBW2)
     VECTZ = Z(IBW1) - Z(IBW2)


     !-----the distance between the 2 atoms is simply the dot product of
     !     this vector.

     BNDLEN = VECTX*VECTX + VECTY*VECTY + VECTZ*VECTZ
     BINV = ONE/SQRT(BNDLEN)
     BL(IBOND)=BNDLEN*BINV

     !-----compute the 1st derivatives of the distance/bond length.

     DBDX(IBOND) = VECTX*BINV
     DBDY(IBOND) = VECTY*BINV
     DBDZ(IBOND) = VECTZ*BINV
  ENDDO
  RETURN
END SUBROUTINE BONDLEN1_CFF

SUBROUTINE ANGLES1_CFF(X,Y,Z)
  !
  ! Function
  !     This routine calculates the theta angle properties using the 
  !     coordinate arrays passed in. The angle properties computed are:
  !     theta angle and its cosine, the 1st derivatives of the angle w.r.t 
  !     the cartesian coordinates
  !
  use chm_kinds
  use dimens_fcm
  use energym
  use consta
  use number
  use deriv
  use code
  use param
  use psf
  use cff_fcm
  implicit none

  real(chm_real) X(*),Y(*),Z(*)
  INTEGER(chm_int4) ICTI,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA
  real(chm_real) COSTHE,DETD,DTHETA,ET,ETHET,VLEN1,VLEN2, &
       VX1,VX2,VY1,VY2,VZ1,VZ2,RSNTHE,RVLN1,RVLN12, &
       RVLN2,RVLN22,SINTH2,SINTHE,SMALLA,VL1DV2,VL2DV1, &
       DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32,DTHTA2, &
       DTHET1,DTHET2,DTHET3,DTHET4,DTHET5,DTHET6,DTHET7,DTHET8,DTHET9

  DATA SMALLA/1.0D-10/
  IF (.NOT.QETERM(BOND)) CALL BONDLEN1_CFF(X,Y,Z)

  !-----loop over all valence angles in this molecule and compute the
  !     valence angle, energy and derivatives.

  DO ITHETA=1,NTHETA

     !-----save the atom numbers of the 3 atoms involved in this angle.
     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)

     !-----compute the 2 vectors of the valence angle,
     !     thevec(,1) (atoms 2->1) and thevec(,2) (atoms 2->3)a.
     VX1 = X(ITHE1) - X(ITHE2)
     VX2 = X(ITHE3) - X(ITHE2)
     VY1 = Y(ITHE1) - Y(ITHE2)
     VY2 = Y(ITHE3) - Y(ITHE2)
     VZ1 = Z(ITHE1) - Z(ITHE2)
     VZ2 = Z(ITHE3) - Z(ITHE2)

     !-----find the cosine of the angle between these 2 vectors,
     !     the valence angle, and save it in theang.
     !     the length of these 2 vectors is also returned in vlen1 and vlen2.

     VLEN1 = BL(ITBW(1,ITHETA))
     VLEN2 = BL(ITBW(2,ITHETA))

     COSTHE = (VX1*VX2 + VY1*VY2 + VZ1*VZ2)/(VLEN1*VLEN2)
     COSTHE = MIN(ONE,MAX(-ONE,COSTHE))

     COSTH(ITHETA) = COSTHE
     TH(ITHETA) = ACOS(COSTHE)

     !-----compute the first derivatives of the cosine of the angle
     !     or the angle w.r.t. the 2 vectors forming tha angle.
     !     vl1dv2 is the length of vectr1 divided by the length of vectr2.
     !     vl2dv1 is the length of vectr2 divided by the length of vectr1.
     RVLN1 = ONE/VLEN1
     RVLN2 = ONE/VLEN2
     RVLN12 = RVLN1*RVLN1
     RVLN22 = RVLN2*RVLN2
     VL1DV2 = VLEN1*RVLN2
     VL2DV1 = VLEN2*RVLN1

     DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
     DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
     DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
     DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
     DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
     DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)

     !-----compute the derivatives for the angle instead of the cosine of
     !     the angle.
     !     first, compute the sine of the angle (from the cosine)
     !     MAKE SURE the angle is not zero, if it is make it a small
     !     number. (this gives rise to a discontinuity in the derivatives
     !     below the value of small which can cause problems).
     !     make the sign of the sine correspond to the sign of isn.
     SINTH2 = ABS(ONE - COSTHE*COSTHE)
     SINTH2 = MAX(SMALLA,SINTH2)
     SINTHE = SQRT(SINTH2)

     RSNTHE = ONE/SINTHE

     !-----finally compute the first derivatives for the angle, not the
     !     cosine.
     !     NOTE. This must come last as dvec is also used in the computation
     !           of the 2nd derivatives of the angle and the original
     !           dvec (for the cosine of the angle) is needed there.
     DVEC11 = -DVEC11*RSNTHE
     DVEC21 = -DVEC21*RSNTHE
     DVEC31 = -DVEC31*RSNTHE
     DVEC12 = -DVEC12*RSNTHE
     DVEC22 = -DVEC22*RSNTHE
     DVEC32 = -DVEC32*RSNTHE

     !-----compute the 1st derivatives of the angle w.r.t the cartesian
     !     coordinates from the derivatives w.r.t the vectors forming the
     !     angle.
     DTHDX(1,ITHETA) = DVEC11
     DTHDY(1,ITHETA) = DVEC21
     DTHDZ(1,ITHETA) = DVEC31
     DTHDX(2,ITHETA) = -DVEC11 - DVEC12
     DTHDY(2,ITHETA) = -DVEC21 - DVEC22
     DTHDZ(2,ITHETA) = -DVEC31 - DVEC32
     DTHDX(3,ITHETA) = DVEC12
     DTHDY(3,ITHETA) = DVEC22
     DTHDZ(3,ITHETA) = DVEC32

  ENDDO
  return
end SUBROUTINE ANGLES1_CFF

#endif 

SUBROUTINE NULL_efCFF
  RETURN
END SUBROUTINE NULL_efCFF

