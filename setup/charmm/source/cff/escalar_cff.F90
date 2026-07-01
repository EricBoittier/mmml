#if KEY_CFF==1
! BIOVIA Code Start : Bug fix
SUBROUTINE EANGANG(ETT,ICB,ICT,X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD)
! BIOVIA Code End
  !
  ! Function
  !     This routine computes the theta*theta energy and 1st derivatives
  !     and (if qsecd is true) 2nd derivatives for all theta*theta
  !     interactions.
  !
  use chm_kinds
  !
  use dimens_fcm
  use energym
  use param
  use number
  use cff_fcm
  implicit none
  ! BIOVIA Code Start : Bug fix
  INTEGER ITHETA,ITHETB,ITHTH,ITT1,ITT2,ITT3,ITT4,ITH,IVPN,ISGN1, &
       ISGN2,LC(12),DERIVS,ICB(*),ICT(*)
  ! BIOVIA Code End
  real(chm_real) CTHTH,DETTDA,DETTDT,DIFTHA,DIFTHB,THEANA,THEANB,ETT
  INTEGER*4 IBD1,IBD2,IBD3,IBD4,IBOND1,IBOND2,IBOND3,ICOOR, &
       IATOM,INDEX,LI,LJ,JCOOR,ISGN
  real(chm_real) COSTHA,COSTHB,D2ETDT,VX1,VX2,VX3,VY1, &
       VY2,VY3,VZ1,VZ2,VZ3,CRST,CRVL4,CTHST2,RSNTHE,RVL123, &
       RVL124,RVL213,RVLN1,RVLN12,RVLN13,RVLN2,RVLN22,RVLN23,RVLN3, &
       RVLN32,RVLN33,SINTH2,SINTHE,SMALLA,VL1DV2,VL2DV1,VLEN1,VLEN2, &
       VLEN3,DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32, &
       DD(*),D2THDX(9,9),D2VEC(6,6),DTHEC1(12),DTHEC2(12)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  ! BIOVIA Code Start : Bug fix
  integer(chm_int4) IBOND4,ICB1,ICB2,ICB3,ICB4
  ! BIOVIA Code End
  !
  DATA SMALLA/1.0D-10/
  !
  IF(.NOT.QETERM(BNDBND)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES1_CFF(X,Y,Z)
  !
  !-----loop over all theta*theta angles
  !
  DO ITHTH=1,NTHTH
     !
     !-----now find pointers to the 2 thetas and 3 bonds in the
     !     theta*theta interaction.
     ITHETA = ITTTW(1,ITHTH)
     ITHETB = ITTTW(2,ITHTH)
     ! BIOVIA Code Start : Bug fix
     !-----get pointers to the 2 bonds in the theta angle
     IBOND1 = ITBW(1,ITHETA)
     ICB1 = ICB(IBOND1)
     IF (ICB1 == 0) CYCLE
     !
     IBOND2 = ITBW(2,ITHETA)
     ICB2 = ICB(IBOND2)
     IF (ICB2 == 0) CYCLE
     !
     IBOND3 = ITBW(3,ITHETB)
     ICB3 = ICB(IBOND3)
     IF (ICB3 == 0) CYCLE
     !
     IBOND4 = ITBW(4,ITHETB)
     ICB4 = ICB(IBOND4)
     IF (ICB4 == 0) CYCLE
     ! BIOVIA Code End
     !
     !-----get the theta angle from the theta angle previously computed.
     THEANA = TH(ITHETA)
     !
     DIFTHA = THEANA - CTHET2(ICT(ITHETA))
     !
     !-----next get the 2nd theta angle from previously computed values
     THEANB = TH(ITHETB)
     !
     DIFTHB = THEANB - CTHET2(ICT(ITHETB))
     !
     CTHTH = CTT(ICTT(ITHTH))
     !
     !-----compute the theta*theta energy and add it to the total
     ETT = ETT + CTHTH*DIFTHA*DIFTHB
     IF (DERIVS > 0) THEN
        !
        !-----now take derivatives of the energy w.r.t. the first angle and
        !     add the derivatives w.r.t. cartesians to total derivative array.
        DETTDT = CTHTH*DIFTHB
        DETTDA = CTHTH*DIFTHA
        !
        !-----add in the first derivatives.
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
        !
        IF (QSECD) THEN
           DTHEC1(10) = DTHDX(2,ITHETA)
           DTHEC1(11) = DTHDY(2,ITHETA)
           DTHEC1(12) = DTHDZ(2,ITHETA)
           DTHEC1(1) = DTHDX(2+ISGN1,ITHETA)
           DTHEC1(2) = DTHDY(2+ISGN1,ITHETA)
           DTHEC1(3) = DTHDZ(2+ISGN1,ITHETA)
           DTHEC1(7) = DTHDX(2-ISGN1,ITHETA)
           DTHEC1(8) = DTHDY(2-ISGN1,ITHETA)
           DTHEC1(9) = DTHDZ(2-ISGN1,ITHETA)
           DTHEC2(10) = DTHDX(2,ITHETB)
           DTHEC2(11) = DTHDY(2,ITHETB)
           DTHEC2(12) = DTHDZ(2,ITHETB)
           DTHEC2(7) = DTHDX(2+ISGN2,ITHETB)
           DTHEC2(8) = DTHDY(2+ISGN2,ITHETB)
           DTHEC2(9) = DTHDZ(2+ISGN2,ITHETB)
           DTHEC2(4) = DTHDX(2-ISGN2,ITHETB)
           DTHEC2(5) = DTHDY(2-ISGN2,ITHETB)
           DTHEC2(6) = DTHDZ(2-ISGN2,ITHETB)

           !
           IBD1 = ITBW(1,ITHETA)
           IBD2 = ITBW(2,ITHETA)
           IBD3 = ITBW(1,ITHETB)
           IBD4 = ITBW(2,ITHETB)
           IF (IBD1  ==  IBD3 .OR. IBD1  ==  IBD4) THEN
              IBOND1 = IBD2
              IBOND3 = IBD1
              IBOND2 = IBD4
              IF (IBD1  /=  IBD3) IBOND2 = IBD3
           ELSE
              !
              IBOND1 = IBD1
              IBOND3 = IBD2
              IBOND2 = IBD3
              IF (IBD2  /=  IBD4) IBOND2 = IBD4
           ENDIF
           !
           VLEN1 = BL(IBOND1)
           VLEN2 = BL(IBOND2)
           VLEN3 = BL(IBOND3)
           !
           VX1 = X(ITT1) - X(ITT4)
           VX2 = X(ITT2) - X(ITT4)
           VX3 = X(ITT3) - X(ITT4)
           VY1 = Y(ITT1) - Y(ITT4)
           VY2 = Y(ITT2) - Y(ITT4)
           VY3 = Y(ITT3) - Y(ITT4)
           VZ1 = Z(ITT1) - Z(ITT4)
           VZ2 = Z(ITT2) - Z(ITT4)
           VZ3 = Z(ITT3) - Z(ITT4)
           COSTHA = COSTH(ITHETA)
           RVLN1 = ONE/VLEN1
           RVLN3 = ONE/VLEN3
           RVLN12 = RVLN1*RVLN1
           RVLN32 = RVLN3*RVLN3
           VL1DV2 = VLEN1*RVLN3
           VL2DV1 = VLEN3*RVLN1
           !
           DVEC11 = RVLN12*(VL1DV2*VX3 - COSTHA*VX1)
           DVEC12 = RVLN32*(VL2DV1*VX1 - COSTHA*VX3)
           DVEC21 = RVLN12*(VL1DV2*VY3 - COSTHA*VY1)
           DVEC22 = RVLN32*(VL2DV1*VY1 - COSTHA*VY3)
           DVEC31 = RVLN12*(VL1DV2*VZ3 - COSTHA*VZ1)
           DVEC32 = RVLN32*(VL2DV1*VZ1 - COSTHA*VZ3)
           !
           !-----compute the derivatives for the angle instead of the cosine of
           !     the angle.
           !     first, compute the sine of the angle (from the cosine)
           !     MAKE SURE the angle is not zero, if it is make it a small
           !     number. (this gives rise to a discontinuity in the derivatives
           !     below the value of small which can cause problems).
           !     make the sign of the sine correspond to the sign of isn.
           SINTH2 = ONE - COSTHA*COSTHA
           IF (ABS(SINTH2)  <  SMALLA) SINTH2 = SMALLA
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           !-----now compute the second derivatives.
           !     icmp1 refers to the components of the first vector,
           !     icmp2 to the components of the second vector, these being
           !     from 4-6 in d2vec.
           !     first, get the 2nd derivatives of the first vector squared
           !     and the second vector squared.
           D2VEC(1,1) = -RVLN12*(TWO*VX1*DVEC11 &
                + COSTHA*(ONE - RVLN12*VX1*VX1))
           D2VEC(4,4) = -RVLN32*(TWO*VX3*DVEC12 &
                + COSTHA*(ONE - RVLN32*VX3*VX3))
           D2VEC(2,2) = -RVLN12*(TWO*VY1*DVEC21 &
                + COSTHA*(ONE - RVLN12*VY1*VY1))
           D2VEC(5,5) = -RVLN32*(TWO*VY3*DVEC22 &
                + COSTHA*(ONE - RVLN32*VY3*VY3))
           D2VEC(3,3) = -RVLN12*(TWO*VZ1*DVEC31 &
                + COSTHA*(ONE - RVLN12*VZ1*VZ1))
           D2VEC(6,6) = -RVLN32*(TWO*VZ3*DVEC32 &
                + COSTHA*(ONE - RVLN32*VZ3*VZ3))
           !
           RVLN13 = RVLN12*RVLN1
           RVLN33 = RVLN32*RVLN3
           RVL123 = RVLN33*RVLN1
           RVL213 = RVLN13*RVLN3
           RVL124 = RVLN12*RVLN32
           !
           !-----now setup the 2nd derivatives for the first and second
           !     vectors, the mixed terms.
           D2VEC(1,4) = -RVLN1*(RVLN33*VX3*VX3 &
                - RVLN3 + VX1*DVEC12*RVLN1)
           D2VEC(2,5) = -RVLN1*(RVLN33*VY3*VY3 &
                - RVLN3 + VY1*DVEC22*RVLN1)
           D2VEC(3,6) = -RVLN1*(RVLN33*VZ3*VZ3 &
                - RVLN3 + VZ1*DVEC32*RVLN1)
           !
           !-----now add in more components of the second derivative mixed terms.
           CRVL4 = COSTHA*RVL124
           CRST = -VX3*VY3*RVL123 - VX1*VY1*RVL213
           D2VEC(1,5) = CRST + VX1*VY3*CRVL4
           D2VEC(2,4) = CRST + VY1*VX3*CRVL4
           CRST = -VY3*VZ3*RVL123 - VY1*VZ1*RVL213
           D2VEC(2,6) = CRST + VY1*VZ3*CRVL4
           D2VEC(3,5) = CRST + VZ1*VY3*CRVL4
           CRST = -VZ3*VX3*RVL123 - VZ1*VX1*RVL213
           D2VEC(3,4) = CRST + VZ1*VX3*CRVL4
           D2VEC(1,6) = CRST + VX1*VZ3*CRVL4
           !
           !-----more components of the mixed terms.
           D2VEC(1,2) = -RVLN12*(VX1*DVEC21 &
                + VY1*DVEC11 - COSTHA*VX1*VY1*RVLN12)
           D2VEC(4,5) = -RVLN32*(VX3*DVEC22 &
                + VY3*DVEC12 - COSTHA*VX3*VY3*RVLN32)
           D2VEC(1,3) = -RVLN12*(VX1*DVEC31 &
                + VZ1*DVEC11 - COSTHA*VX1*VZ1*RVLN12)
           D2VEC(4,6) = -RVLN32*(VX3*DVEC32 &
                + VZ3*DVEC12 - COSTHA*VX3*VZ3*RVLN32)
           D2VEC(2,3) = -RVLN12*(VY1*DVEC31 &
                + VZ1*DVEC21 - COSTHA*VY1*VZ1*RVLN12)
           D2VEC(5,6) = -RVLN32*(VY3*DVEC32 &
                + VZ3*DVEC22 - COSTHA*VY3*VZ3*RVLN32)
           !
           CTHST2 = COSTHA/SINTH2
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           !
           D2VEC(2,1)=D2VEC(1,2)
           D2VEC(6,5)=D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THDX(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THDX(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo
              !
              D2THDX(ICOOR,7) = D2VEC(ICOOR,4)
              D2THDX(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THDX(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THDX(ICOOR,8) = D2VEC(ICOOR,5)
              D2THDX(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THDX(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THDX(ICOOR,9) = D2VEC(ICOOR,6)
              D2THDX(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THDX(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
           enddo
           !
           D2THDX(4,4) = -D2THDX(1,4) - D2THDX(4,7)
           D2THDX(4,5) = -D2THDX(1,5) - D2THDX(4,8)
           D2THDX(4,6) = -D2THDX(1,6) - D2THDX(4,9)
           D2THDX(5,5) = -D2THDX(2,5) - D2THDX(5,8)
           D2THDX(5,6) = -D2THDX(2,6) - D2THDX(5,9)
           D2THDX(6,6) = -D2THDX(3,6) - D2THDX(6,9)
           !
           !-----loop over all atoms involved in the interaction, compute
           !     the total coordinate number of each atom, i.e. the atom number * 3
           !     and add it to the list of coordinates stored in lcoor3
           !     for each atom found in the list of atoms stored in latom there
           !     are 3 coordinates (x,y,z) to be added to the second derivative
           !     matrix.
           !
           IATOM = (ITT1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (ITT4 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (ITT3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           !
           !-----now add all contributions to the second derivative array
           !     indxdd computes where the element of the second derivative
           !     matrix is to be stored in the linear array dd. this is used
           !     as the second derivative matrix is symmetric and so only one
           !     diagonal half of the mtrix need be stored.
           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2THDX(JCOOR,ICOOR)*DETTDT
              enddo
           enddo

           !-----compute the first derivatives of the cosine of the angle
           !     or the angle w.r.t. the 2 vectors forming tha angle.
           !     vl1dv2 is the length of vectr1 divided by the length of vectr2.
           !     vl2dv1 is the length of vectr2 divided by the length of vectr1.
           COSTHB = COSTH(ITHETB)
           RVLN2 = ONE/VLEN2
           RVLN3 = ONE/VLEN3
           RVLN22 = RVLN2*RVLN2
           RVLN32 = RVLN3*RVLN3
           VL1DV2 = VLEN2*RVLN3
           VL2DV1 = VLEN3*RVLN2
           !
           DVEC11 = RVLN22*(VL1DV2*VX3 - COSTHB*VX2)
           DVEC12 = RVLN32*(VL2DV1*VX2 - COSTHB*VX3)
           DVEC21 = RVLN22*(VL1DV2*VY3 - COSTHB*VY2)
           DVEC22 = RVLN32*(VL2DV1*VY2 - COSTHB*VY3)
           DVEC31 = RVLN22*(VL1DV2*VZ3 - COSTHB*VZ2)
           DVEC32 = RVLN32*(VL2DV1*VZ2 - COSTHB*VZ3)
           !
           !-----compute the derivatives for the angle instead of the cosine of
           !     the angle.
           !     first, compute the sine of the angle (from the cosine)
           !     MAKE SURE the angle is not zero, if it is make it a small
           !     number. (this gives rise to a discontinuity in the derivatives
           !     below the value of small which can cause problems).
           !     make the sign of the sine correspond to the sign of isn.
           SINTH2 = ONE - COSTHB*COSTHB
           IF (ABS(SINTH2)  <  SMALLA) SINTH2 = SMALLA
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           !-----now compute the second derivatives.
           !     icmp1 refers to the components of the first vector,
           !     icmp2 to the components of the second vector, these being
           !     from 4-6 in d2vec.
           !     first, get the 2nd derivatives of the first vector squared
           !     and the second vector squared.
           D2VEC(1,1) = -RVLN22*(TWO*VX2*DVEC11 &
                + COSTHB*(ONE - RVLN22*VX2*VX2))
           D2VEC(4,4) = -RVLN32*(TWO*VX3*DVEC12 &
                + COSTHB*(ONE - RVLN32*VX3*VX3))
           D2VEC(2,2) = -RVLN22*(TWO*VY2*DVEC21 &
                + COSTHB*(ONE - RVLN22*VY2*VY2))
           D2VEC(5,5) = -RVLN32*(TWO*VY3*DVEC22 &
                + COSTHB*(ONE - RVLN32*VY3*VY3))
           D2VEC(3,3) = -RVLN22*(TWO*VZ2*DVEC31 &
                + COSTHB*(ONE - RVLN22*VZ2*VZ2))
           D2VEC(6,6) = -RVLN32*(TWO*VZ3*DVEC32 &
                + COSTHB*(ONE - RVLN32*VZ3*VZ3))
           !
           RVLN23 = RVLN22*RVLN2
           RVLN33 = RVLN32*RVLN3
           RVL123 = RVLN33*RVLN2
           RVL213 = RVLN23*RVLN3
           RVL124 = RVLN22*RVLN32
           !
           !-----now setup the 2nd derivatives for the first and second
           !     vectors, the mixed terms.
           D2VEC(1,4) = -RVLN2*(RVLN33*VX3*VX3 &
                - RVLN3 + VX2*DVEC12*RVLN2)
           D2VEC(2,5) = -RVLN2*(RVLN33*VY3*VY3 &
                - RVLN3 + VY2*DVEC22*RVLN2)
           D2VEC(3,6) = -RVLN2*(RVLN33*VZ3*VZ3 &
                - RVLN3 + VZ2*DVEC32*RVLN2)
           !
           !-----now add in more components of the second derivative mixed terms.
           CRVL4 = COSTHB*RVL124
           CRST = -VX3*VY3*RVL123 - VX2*VY2*RVL213
           D2VEC(1,5) = CRST + VX2*VY3*CRVL4
           D2VEC(2,4) = CRST + VY2*VX3*CRVL4
           CRST = -VY3*VZ3*RVL123 - VY2*VZ2*RVL213
           D2VEC(2,6) = CRST + VY2*VZ3*CRVL4
           D2VEC(3,5) = CRST + VZ2*VY3*CRVL4
           CRST = -VZ3*VX3*RVL123 - VZ2*VX2*RVL213
           D2VEC(3,4) = CRST + VZ2*VX3*CRVL4
           D2VEC(1,6) = CRST + VX2*VZ3*CRVL4
           !
           !-----more components of the mixed terms.
           D2VEC(1,2) = -RVLN22*(VX2*DVEC21 &
                + VY2*DVEC11 - COSTHB*VX2*VY2*RVLN22)
           D2VEC(4,5) = -RVLN32*(VX3*DVEC22 &
                + VY3*DVEC12 - COSTHB*VX3*VY3*RVLN32)
           D2VEC(1,3) = -RVLN22*(VX2*DVEC31 &
                + VZ2*DVEC11 - COSTHB*VX2*VZ2*RVLN22)
           D2VEC(4,6) = -RVLN32*(VX3*DVEC32 &
                + VZ3*DVEC12 - COSTHB*VX3*VZ3*RVLN32)
           D2VEC(2,3) = -RVLN22*(VY2*DVEC31 &
                + VZ2*DVEC21 - COSTHB*VY2*VZ2*RVLN22)
           D2VEC(5,6) = -RVLN32*(VY3*DVEC32 &
                + VZ3*DVEC22 - COSTHB*VY3*VZ3*RVLN32)
           !
           CTHST2 = COSTHB/SINTH2
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           !
           D2VEC(2,1)=D2VEC(1,2)
           D2VEC(6,5)=D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THDX(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THDX(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo
              !
              D2THDX(ICOOR,7) = D2VEC(ICOOR,4)
              D2THDX(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THDX(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THDX(ICOOR,8) = D2VEC(ICOOR,5)
              D2THDX(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THDX(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THDX(ICOOR,9) = D2VEC(ICOOR,6)
              D2THDX(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THDX(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
              !
           enddo
           !
           D2THDX(4,4) = -D2THDX(1,4) - D2THDX(4,7)
           D2THDX(4,5) = -D2THDX(1,5) - D2THDX(4,8)
           D2THDX(4,6) = -D2THDX(1,6) - D2THDX(4,9)
           D2THDX(5,5) = -D2THDX(2,5) - D2THDX(5,8)
           D2THDX(5,6) = -D2THDX(2,6) - D2THDX(5,9)
           D2THDX(6,6) = -D2THDX(3,6) - D2THDX(6,9)
           !
           !-----loop over all atoms involved in the interaction, compute
           !     the total coordinate number of each atom, i.e. the atom number * 3
           !     and add it to the list of coordinates stored in lcoor3
           !     for each atom found in the list of atoms stored in latom there
           !     are 3 coordinates (x,y,z) to be added to the second derivative
           !     matrix.
           !
           IATOM = (ITT2 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           !
           !-----now add all contributions to the second derivative array
           !     indxdd computes where the element of the second derivative
           !     matrix is to be stored in the linear array dd. this is used
           !     as the second derivative matrix is symmetric and so only one
           !     diagonal half of the mtrix need be stored.
           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2THDX(JCOOR,ICOOR)*DETTDA
              enddo
           enddo
           !
           !-----setup dttdxm, an array which contains the 1st derivatives for the
           !     2 theta angles, angle a in dttdxm(,,1) and angle b in dttdxm(,,2).
           !     ddmix creates the mixed 2nd derivative terms which are the
           !     product of the 1st derivatives of the 2 angles. it adds these
           !     terms to the total 2nd derivative array. the 12 in the
           !     argument list is the no. of coordinates in the theta*theta term,
           !     from the no. of atoms in the theta*theta term (4) * the no. of
           !     cartesian coordinates (3). the 4 is the no. of atoms in the
           !     cross-term and the 1 and 2 where the derivatives are in dttdxm for
           !     theta angle a and b respectively.
           DTHEC1(4) = ZERO
           DTHEC1(5) = ZERO
           DTHEC1(6) = ZERO
           DTHEC2(1) = ZERO
           DTHEC2(2) = ZERO
           DTHEC2(3) = ZERO
           !
           D2ETDT = CTHTH
           !
           IATOM = (ITT1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (ITT2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (ITT3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           IATOM = (ITT4 - 1)*3
           LC(10) = IATOM + 1
           LC(11) = IATOM + 2
           LC(12) = IATOM + 3
           !
           DO ICOOR=1,12
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 !
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 !
                 DD(INDEX) = DD(INDEX) &
                      + D2ETDT*(DTHEC1(ICOOR)*DTHEC2(JCOOR) &
                      + DTHEC1(JCOOR)*DTHEC2(ICOOR))
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO

  !      write(6,*) 'EANGANG> energy=', ETT

  RETURN
END SUBROUTINE EANGANG


SUBROUTINE EANGLE_CFF(ET,IT,JT,KT,ICT,NT,X,Y,Z,DX,DY,DZ,DD,DERIVS, &
     QSECD)
  !
  ! Function
  !
  use chm_kinds
  !
  use dimens_fcm
  use energym
  use param
  use number
  use cff_fcm
  !
  implicit none
  INTEGER*4 ICTI,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA,IT(*),JT(*),KT(*), &
       ICT(*),NT,DERIVS
  real(chm_real) COSTHE,DETD,DTHETA,ET,ETHET,VLEN1,VLEN2, &
       VX1,VX2,VY1,VY2,VZ1,VZ2,RSNTHE,RVLN1,RVLN12, &
       RVLN2,RVLN22,SINTH2,SINTHE,SMALLA,VL1DV2,VL2DV1, &
       DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32,DTHTA2, &
       DTHET1,DTHET2,DTHET3,DTHET4,DTHET5,DTHET6,DTHET7,DTHET8,DTHET9
  LOGICAL QSECD
  !
  INTEGER*4 IATOM,INDEX,LC(9),LI,LJ,JCOOR,ICOOR
  real(chm_real) D2ETDT,D2THDX(9,9),DTHDS(9),CRST,CRVL4, &
       CTHST2,RVL123,RVL124,RVL213,RVLN13, &
       RVLN23,D2VEC(6,6),DD(*),X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  !
  DATA SMALLA/1.0D-10/
  !
  IF(.NOT.QETERM(ANGLE)) RETURN
  IF (.NOT.QETERM(BOND)) CALL BONDLEN1_CFF(X,Y,Z)
  !
  !-----loop over all valence angles in this molecule and compute the
  !     valence angle, energy and derivatives.
  !
  DO ITHETA=1,NT
     !
     !-----save the atom numbers of the 3 atoms involved in this angle.
     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)
     !
     !-----compute the 2 vectors of the valence angle,
     !     thevec(,1) (atoms 2->1) and thevec(,2) (atoms 2->3)a.
     VX1 = X(ITHE1) - X(ITHE2)
     VX2 = X(ITHE3) - X(ITHE2)
     VY1 = Y(ITHE1) - Y(ITHE2)
     VY2 = Y(ITHE3) - Y(ITHE2)
     VZ1 = Z(ITHE1) - Z(ITHE2)
     VZ2 = Z(ITHE3) - Z(ITHE2)
     !
     !-----find the cosine of the angle between these 2 vectors,
     !     the valence angle, and save it in theang.
     !     the length of these 2 vectors is also returned in vlen1 and vlen2.
     !
     VLEN1 = BL(ITBW(1,ITHETA))
     VLEN2 = BL(ITBW(2,ITHETA))
     !
     COSTHE = (VX1*VX2 + VY1*VY2 + VZ1*VZ2)/(VLEN1*VLEN2)
     COSTHE = MIN(ONE,MAX(-ONE,COSTHE))
     !
     COSTH(ITHETA) = COSTHE
     TH(ITHETA) = ACOS(COSTHE)
     !
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
     !
     DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
     DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
     DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
     DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
     DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
     DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)
     !
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
     !
     RSNTHE = ONE/SINTHE
     !
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
     !
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
     !
     !-----compute the theta energy and the 1st and 2nd derivatives
     !     of the energy w.r.t the valence angle. save each theta
     !     energy in etheta.
     !
     ICTI=ICT(ITHETA)
     ! BIOVIA Code Start : Bug fix
     IF (ICTI == 0) CYCLE
     ! BIOVIA Code End
     DTHETA = TH(ITHETA) - CTHET2(ICTI)
     DTHTA2 = DTHETA * DTHETA
     ETHET = (CTHET1(ICTI) + CTHET6(ICTI)*DTHETA)*DTHTA2 &
          + CTHET7(ICTI)*DTHTA2*DTHTA2
     !
     ET = ET + ETHET
     !
     IF (DERIVS > 0) THEN
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
        !
        IF (QSECD) THEN
           DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
           DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
           DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
           DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
           DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
           DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)
           !
           SINTH2 = ABS(ONE - COSTHE*COSTHE)
           SINTH2 = MAX(SINTH2,SMALLA)
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           RVLN13 = RVLN12*RVLN1
           RVLN23 = RVLN22*RVLN2
           RVL123 = RVLN23*RVLN1
           RVL213 = RVLN13*RVLN2
           RVL124 = RVLN12*RVLN22
           CRVL4 = COSTHE*RVL124
           CTHST2 = COSTHE/SINTH2
           !
           D2VEC(1,1) = -RVLN12*(TWO*VX1*DVEC11 &
                + COSTHE*(ONE - RVLN12*VX1*VX1))
           D2VEC(4,4) = -RVLN22*(TWO*VX2*DVEC12 &
                + COSTHE*(ONE - RVLN22*VX2*VX2))
           D2VEC(2,2) = -RVLN12*(TWO*VY1*DVEC21 &
                + COSTHE*(ONE - RVLN12*VY1*VY1))
           D2VEC(5,5) = -RVLN22*(TWO*VY2*DVEC22 &
                + COSTHE*(ONE - RVLN22*VY2*VY2))
           D2VEC(3,3) = -RVLN12*(TWO*VZ1*DVEC31 &
                + COSTHE*(ONE - RVLN12*VZ1*VZ1))
           D2VEC(6,6) = -RVLN22*(TWO*VZ2*DVEC32 &
                + COSTHE*(ONE - RVLN22*VZ2*VZ2))
           !
           D2VEC(1,4) = -RVLN1*(RVLN23*VX2*VX2-RVLN2+VX1*DVEC12*RVLN1)
           D2VEC(2,5) = -RVLN1*(RVLN23*VY2*VY2-RVLN2+VY1*DVEC22*RVLN1)
           D2VEC(3,6) = -RVLN1*(RVLN23*VZ2*VZ2-RVLN2+VZ1*DVEC32*RVLN1)
           !
           CRST = -VX2*VY2*RVL123 - VX1*VY1*RVL213
           D2VEC(1,5) = CRST + VX1*VY2*CRVL4
           D2VEC(2,4) = CRST + VY1*VX2*CRVL4
           CRST = -VY2*VZ2*RVL123 - VY1*VZ1*RVL213
           D2VEC(2,6) = CRST + VY1*VZ2*CRVL4
           D2VEC(3,5) = CRST + VZ1*VY2*CRVL4
           CRST = -VZ2*VX2*RVL123 - VZ1*VX1*RVL213
           D2VEC(3,4) = CRST + VZ1*VX2*CRVL4
           D2VEC(1,6) = CRST + VX1*VZ2*CRVL4
           !
           D2VEC(1,2) = -RVLN12*(VX1*DVEC21 + VY1*DVEC11 &
                - COSTHE*VX1*VY1*RVLN12)
           D2VEC(4,5) = -RVLN22*(VX2*DVEC22 + VY2*DVEC12 &
                - COSTHE*VX2*VY2*RVLN22)
           D2VEC(1,3) = -RVLN12*(VX1*DVEC31 + VZ1*DVEC11 &
                - COSTHE*VX1*VZ1*RVLN12)
           D2VEC(4,6) = -RVLN22*(VX2*DVEC32 + VZ2*DVEC12 &
                - COSTHE*VX2*VZ2*RVLN22)
           D2VEC(2,3) = -RVLN12*(VY1*DVEC31 + VZ1*DVEC21 &
                - COSTHE*VY1*VZ1*RVLN12)
           D2VEC(5,6) = -RVLN22*(VY2*DVEC32 + VZ2*DVEC22 &
                - COSTHE*VY2*VZ2*RVLN22)
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           !
           D2VEC(2,1)=D2VEC(1,2)
           D2VEC(6,5)=D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THDX(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THDX(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo
              !
              D2THDX(ICOOR,7) = D2VEC(ICOOR,4)
              D2THDX(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THDX(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THDX(ICOOR,8) = D2VEC(ICOOR,5)
              D2THDX(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THDX(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THDX(ICOOR,9) = D2VEC(ICOOR,6)
              D2THDX(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THDX(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
           enddo
           !
           D2THDX(4,4) = -D2THDX(1,4) - D2THDX(4,7)
           D2THDX(4,5) = -D2THDX(1,5) - D2THDX(4,8)
           D2THDX(4,6) = -D2THDX(1,6) - D2THDX(4,9)
           D2THDX(5,5) = -D2THDX(2,5) - D2THDX(5,8)
           D2THDX(5,6) = -D2THDX(2,6) - D2THDX(5,9)
           D2THDX(6,6) = -D2THDX(3,6) - D2THDX(6,9)
           !
           D2ETDT = TWO*CTHET1(ICTI) + (6.0*CTHET6(ICTI) &
                + TWELVE*CTHET7(ICTI)*DTHETA)*DTHETA
           !
           IATOM = (ITHE1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (ITHE2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (ITHE3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           DTHDS(1) = DTHDX(1,ITHETA)
           DTHDS(2) = DTHDY(1,ITHETA)
           DTHDS(3) = DTHDZ(1,ITHETA)
           DTHDS(4) = DTHDX(2,ITHETA)
           DTHDS(5) = DTHDY(2,ITHETA)
           DTHDS(6) = DTHDZ(2,ITHETA)
           DTHDS(7) = DTHDX(3,ITHETA)
           DTHDS(8) = DTHDY(3,ITHETA)
           DTHDS(9) = DTHDZ(3,ITHETA)
           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2THDX(JCOOR,ICOOR)*DETD &
                      + DTHDS(ICOOR)*DTHDS(JCOOR)*D2ETDT
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO
  !

  !      write(6,*),'EANGLE_CFF> energy=',ET

  RETURN
END SUBROUTINE EANGLE_CFF


SUBROUTINE EBNDANG(EBB,EBT,IT,JT,KT,ICT,NT,ICB,IAC, &
     X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD)
  !
  ! Function
  !     Compute the bond*theta cross-term energy and its 1st and
  !     (if qsecd is true) 2nd derivatives w.r.t. cartesian coordinates
  !     each theta angle also represents 2 bond*theta cross-terms, 1 for
  !     each bond. thus, this routine loops over all theta angles and
  !     computes the 2 bond*theta cross-terms in each angle.
  !     the bond*theta potential parameters are also stored in the theta
  !     parameters.
  !
  use chm_kinds
  !
  use dimens_fcm
  use energym
  use param
  use number
  use cff_fcm
  !
  implicit none
  INTEGER IBOND1,IBOND2,ITHE1,ITHE2,ITHE3,ITHETA,ITHTA,ISGN,ICTP, &
       ICOOR,IATOM,INDEX,LI,LJ,JCOOR,IT(*),JT(*),KT(*), &
       ICT(*),DERIVS,LC(9),NT,ICB(*),IAC(*)
  real(chm_real) BNDLN1,BNDLN2,CBTA,CBTB,CTMP,DEBTDT,DTHET,DBDX1, &
       DBDY1,DBDZ1,DBDX2,DBDY2,DBDZ2, &
       KBB,DEBDA,DEBDB,DIFB1,DIFB2,EBB,EBT
  real(chm_real) COSTHE,VX1,VX2,VY1,VY2,VZ1,VZ2,TDIAG,TMIX,CRST, &
       CRVL4,CTHST2,RSNTHE,RVL123,RVL124,RVL213, &
       RVLN1,RVLN12,RVLN13,RVLN2,RVLN22,RVLN23,SINTH2,SINTHE,SMALLA, &
       VL1DV2,VL2DV1,DVEC11,DVEC12,DVEC21,DVEC22,DVEC31,DVEC32,DD(*), &
       D2BDC1(6,6),D2THC1(9,9),DBDC1(9),DBDC2(9),D2BDC2(6,6), &
       D2VEC(6,6),DTHEC(9)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  ! BIOVIA Code Start : Bug fix
  integer(chm_int4) ICB1,ICB2
  ! BIOVIA Code End
  !
  DATA SMALLA/1.0D-10/
  !
  IF(.NOT.QETERM(STRSTR) .AND. .NOT.QETERM(STRB)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES1_CFF(X,Y,Z)
  !
  !-----loop over all valence angles.
  !     there are 2 bond*theta cross-terms for each valence angle, 1 for
  !     each bond.
  !
  DO ITHETA=1,NT
     !
     ITHE1 = IT(ITHETA)
     ITHE2 = JT(ITHETA)
     ITHE3 = KT(ITHETA)
     !
     !-----get pointers to the 2 bonds in the theta angle
     IBOND1 = ITBW(1,ITHETA)
     IBOND2 = ITBW(2,ITHETA)
     !
     BNDLN1 = BL(IBOND1)
     BNDLN2 = BL(IBOND2)
     ! BIOVIA Code Start : Bug fix
     ICB1 = ICB(IBOND1)
     ICB2 = ICB(IBOND2)
     IF(ICB1 == 0 .OR. ICB2 == 0) CYCLE
     ! BIOVIA Code End
     !
     ISGN = ITFLG(1,ITHETA)
     DBDX1 = DBDX(IBOND1)*ISGN
     DBDY1 = DBDY(IBOND1)*ISGN
     DBDZ1 = DBDZ(IBOND1)*ISGN
     ISGN = ITFLG(2,ITHETA)
     DBDX2 = DBDX(IBOND2)*ISGN
     DBDY2 = DBDY(IBOND2)*ISGN
     DBDZ2 = DBDZ(IBOND2)*ISGN
     !
     !-----compute the difference of the 2 bond lengths from the equilibrium
     !     values for the 2 bonds taken from their BOND potential parameters.
     !     these are found using itbw to find the bond and then using icb.
     ! BIOVIA Code Start : Bug fix
     DIFB1 = BNDLN1 - CBOND2(ICB1)
     DIFB2 = BNDLN2 - CBOND2(ICB2)
     ! BIOVIA Code End
     !
     !-----compute the difference of the theta angle from its equilibrium
     !     value - dthet
     ICTP = ICT(ITHETA)
     ! BIOVIA Code Start : Bug fix
     IF (ICTP == 0) CYCLE 
     ! BIOVIA Code End
     DTHET = TH(ITHETA) - CTHET2(ICTP)
     !
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
        !
        CTMP = CBTA
        CBTA = CBTB
        CBTB = CTMP
     ENDIF
     !
     !-----take derivatives of the energy w.r.t. the bond a and its cartesian
     !     coordinates and add this contribution to the total derivative.
     !
     DEBTDT = CBTA*DIFB1 + CBTB*DIFB2
     !
     !-----the bond*bond cross-term potential constant is 3rd entry in the
     !     valence angle potential entry for this theta angle.
     KBB = CTHET3(ICTP)
     !
     !-----now compute bond*bond energy and add it to total bond*bond energy
     EBB = EBB + KBB*DIFB1*DIFB2
     !
     EBT = EBT + DEBTDT * DTHET
     !
     IF (DERIVS > 0) THEN
        DEBDB = KBB*DIFB1 + CBTB*DTHET
        DEBDA = KBB*DIFB2 + CBTA*DTHET
        !
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
        !
        IF (QSECD) THEN
           RVLN1 = ONE/BNDLN1
           RVLN2 = ONE/BNDLN2
           !
           VX1 = X(ITHE1) - X(ITHE2)
           VX2 = X(ITHE3) - X(ITHE2)
           VY1 = Y(ITHE1) - Y(ITHE2)
           VY2 = Y(ITHE3) - Y(ITHE2)
           VZ1 = Z(ITHE1) - Z(ITHE2)
           VZ2 = Z(ITHE3) - Z(ITHE2)
           DBDC1(1) = DBDX1
           DBDC1(2) = DBDY1
           DBDC1(3) = DBDZ1
           DBDC1(4) = -DBDX1
           DBDC1(5) = -DBDY1
           DBDC1(6) = -DBDZ1
           DBDC1(7) = ZERO
           DBDC1(8) = ZERO
           DBDC1(9) = ZERO
           DBDC2(1) = ZERO
           DBDC2(2) = ZERO
           DBDC2(3) = ZERO
           DBDC2(4) = -DBDX2
           DBDC2(5) = -DBDY2
           DBDC2(6) = -DBDZ2
           DBDC2(7) = DBDX2
           DBDC2(8) = DBDY2
           DBDC2(9) = DBDZ2
           !
           DTHEC(1) = DTHDX(1,ITHETA)
           DTHEC(2) = DTHDY(1,ITHETA)
           DTHEC(3) = DTHDZ(1,ITHETA)
           DTHEC(4) = DTHDX(2,ITHETA)
           DTHEC(5) = DTHDY(2,ITHETA)
           DTHEC(6) = DTHDZ(2,ITHETA)
           DTHEC(7) = DTHDX(3,ITHETA)
           DTHEC(8) = DTHDY(3,ITHETA)
           DTHEC(9) = DTHDZ(3,ITHETA)
           !
           COSTHE = COSTH(ITHETA)
           RVLN12 = RVLN1*RVLN1
           RVLN22 = RVLN2*RVLN2
           VL1DV2 = BNDLN1*RVLN2
           VL2DV1 = BNDLN2*RVLN1
           !
           DVEC11 = RVLN12*(VL1DV2*VX2 - COSTHE*VX1)
           DVEC12 = RVLN22*(VL2DV1*VX1 - COSTHE*VX2)
           DVEC21 = RVLN12*(VL1DV2*VY2 - COSTHE*VY1)
           DVEC22 = RVLN22*(VL2DV1*VY1 - COSTHE*VY2)
           DVEC31 = RVLN12*(VL1DV2*VZ2 - COSTHE*VZ1)
           DVEC32 = RVLN22*(VL2DV1*VZ1 - COSTHE*VZ2)
           !
           !-----compute the derivatives for the angle instead of the cosine of
           !     the angle.
           !     first, compute the sine of the angle (from the cosine)
           !     MAKE SURE the angle is not zero, if it is make it a small
           !     number. (this gives rise to a discontinuity in the derivatives
           !     below the value of small which can cause problems).
           !     make the sign of the sine correspond to the sign of isn.
           SINTH2 = ONE - COSTHE*COSTHE
           IF (ABS(SINTH2)  <  SMALLA) SINTH2 = SMALLA
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           !-----now compute the second derivatives.
           !     icmp1 refers to the components of the first vector,
           !     icmp2 to the components of the second vector, these being
           !     from 4-6 in d2vec.
           !     first, get the 2nd derivatives of the first vector squared
           !     and the second vector squared.
           D2VEC(1,1) = -RVLN12*(TWO*VX1*DVEC11 &
                + COSTHE*(ONE - RVLN12*VX1*VX1))
           D2VEC(4,4) = -RVLN22*(TWO*VX2*DVEC12 &
                + COSTHE*(ONE - RVLN22*VX2*VX2))
           D2VEC(2,2) = -RVLN12*(TWO*VY1*DVEC21 &
                + COSTHE*(ONE - RVLN12*VY1*VY1))
           D2VEC(5,5) = -RVLN22*(TWO*VY2*DVEC22 &
                + COSTHE*(ONE - RVLN22*VY2*VY2))
           D2VEC(3,3) = -RVLN12*(TWO*VZ1*DVEC31 &
                + COSTHE*(ONE - RVLN12*VZ1*VZ1))
           D2VEC(6,6) = -RVLN22*(TWO*VZ2*DVEC32 &
                + COSTHE*(ONE - RVLN22*VZ2*VZ2))
           !
           RVLN13 = RVLN12*RVLN1
           RVLN23 = RVLN22*RVLN2
           RVL123 = RVLN23*RVLN1
           RVL213 = RVLN13*RVLN2
           RVL124 = RVLN12*RVLN22
           !
           !-----now setup the 2nd derivatives for the first and second
           !     vectors, the mixed terms.
           D2VEC(1,4) = -RVLN1*(RVLN23*VX2*VX2 &
                - RVLN2 + VX1*DVEC12*RVLN1)
           D2VEC(2,5) = -RVLN1*(RVLN23*VY2*VY2 &
                - RVLN2 + VY1*DVEC22*RVLN1)
           D2VEC(3,6) = -RVLN1*(RVLN23*VZ2*VZ2 &
                - RVLN2 + VZ1*DVEC32*RVLN1)
           !
           !-----now add in more components of the second derivative mixed terms.
           CRVL4 = COSTHE*RVL124
           CRST = -VX2*VY2*RVL123 - VX1*VY1*RVL213
           D2VEC(1,5) = CRST + VX1*VY2*CRVL4
           D2VEC(2,4) = CRST + VY1*VX2*CRVL4
           CRST = -VY2*VZ2*RVL123 - VY1*VZ1*RVL213
           D2VEC(2,6) = CRST + VY1*VZ2*CRVL4
           D2VEC(3,5) = CRST + VZ1*VY2*CRVL4
           CRST = -VZ2*VX2*RVL123 - VZ1*VX1*RVL213
           D2VEC(3,4) = CRST + VZ1*VX2*CRVL4
           D2VEC(1,6) = CRST + VX1*VZ2*CRVL4
           !
           !-----more components of the mixed terms.
           D2VEC(1,2) = -RVLN12*(VX1*DVEC21 &
                + VY1*DVEC11 - COSTHE*VX1*VY1*RVLN12)
           D2VEC(4,5) = -RVLN22*(VX2*DVEC22 &
                + VY2*DVEC12 - COSTHE*VX2*VY2*RVLN22)
           D2VEC(1,3) = -RVLN12*(VX1*DVEC31 &
                + VZ1*DVEC11 - COSTHE*VX1*VZ1*RVLN12)
           D2VEC(4,6) = -RVLN22*(VX2*DVEC32 &
                + VZ2*DVEC12 - COSTHE*VX2*VZ2*RVLN22)
           D2VEC(2,3) = -RVLN12*(VY1*DVEC31 &
                + VZ1*DVEC21 - COSTHE*VY1*VZ1*RVLN12)
           D2VEC(5,6) = -RVLN22*(VY2*DVEC32 &
                + VZ2*DVEC22 - COSTHE*VY2*VZ2*RVLN22)
           !
           CTHST2 = COSTHE/SINTH2
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           !
           D2VEC(2,1)=D2VEC(1,2)
           D2VEC(6,5)=D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THC1(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THC1(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo

              D2THC1(ICOOR,7) = D2VEC(ICOOR,4)
              D2THC1(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THC1(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THC1(ICOOR,8) = D2VEC(ICOOR,5)
              D2THC1(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THC1(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THC1(ICOOR,9) = D2VEC(ICOOR,6)
              D2THC1(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THC1(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
           enddo
           !
           D2THC1(4,4) = -D2THC1(1,4) - D2THC1(4,7)
           D2THC1(4,5) = -D2THC1(1,5) - D2THC1(4,8)
           D2THC1(4,6) = -D2THC1(1,6) - D2THC1(4,9)
           D2THC1(5,5) = -D2THC1(2,5) - D2THC1(5,8)
           D2THC1(5,6) = -D2THC1(2,6) - D2THC1(5,9)
           D2THC1(6,6) = -D2THC1(3,6) - D2THC1(6,9)
           !
           !-----d2bdx is dimensioned 6,6 which represents a 3,2 by 3,2 array.
           !     the components for atom 1 are stored from 1-3 and the components
           !     for atom 2 in 4-6. lat1 is the start index for atom a in d2bdx,
           !     lat2 is the start index for atom 2. i.e. lat2 + icoor is a
           !     component of atom 2. an index with no added component indicates
           !     this is for atom 1.
           !     first, we have the diagonal terms of each atom with itself.
           !     we also include the mixed term of the e.g. x coordinate of
           !     atom 1 and the x coordinate of atom 2 here.
           !     NOTE. this saves time for the (lat1,lat2) component by
           !           changing the sign of tdiag as it should be dbdx(,1)*
           !           dbdx(,2) but dbdx(,2) is simply -dbdx(,1).
           !
           TDIAG = (ONE - DBDX1*DBDX1)*RVLN1
           D2BDC1(1,1) = TDIAG
           D2BDC1(4,4) = TDIAG
           D2BDC1(1,4) = -TDIAG
           TDIAG = (ONE - DBDY1*DBDY1)*RVLN1
           D2BDC1(2,2) = TDIAG
           D2BDC1(5,5) = TDIAG
           D2BDC1(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ1*DBDZ1)*RVLN1
           D2BDC1(3,3) = TDIAG
           D2BDC1(6,6) = TDIAG
           D2BDC1(3,6) = -TDIAG
           TDIAG = (ONE - DBDX2*DBDX2)*RVLN2
           D2BDC2(1,1) = TDIAG
           D2BDC2(4,4) = TDIAG
           D2BDC2(1,4) = -TDIAG
           TDIAG = (ONE - DBDY2*DBDY2)*RVLN2
           D2BDC2(2,2) = TDIAG
           D2BDC2(5,5) = TDIAG
           D2BDC2(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ2*DBDZ2)*RVLN2
           D2BDC2(3,3) = TDIAG
           D2BDC2(6,6) = TDIAG
           D2BDC2(3,6) = -TDIAG
           !
           !-----now the mixed terms for each atom separately, i.e. x of atom 1
           !     with y of atom 1 etc. in terms of coordinates this is 1,2 and
           !     1,3 and 2,3. this also includes the mixed terms x of atom 1 and
           !     y of atom 2 etc.
           TMIX = -DBDX1*DBDY1*RVLN1
           D2BDC1(1,2) = TMIX
           D2BDC1(4,5) = TMIX
           D2BDC1(1,5) = -TMIX
           TMIX = -DBDX1*DBDZ1*RVLN1
           D2BDC1(1,3) = TMIX
           D2BDC1(4,6) = TMIX
           D2BDC1(1,6) = -TMIX
           TMIX = -DBDY1*DBDZ1*RVLN1
           D2BDC1(2,3) = TMIX
           D2BDC1(5,6) = TMIX
           D2BDC1(2,6) = -TMIX
           TMIX = -DBDX2*DBDY2*RVLN2
           D2BDC2(1,2) = TMIX
           D2BDC2(4,5) = TMIX
           D2BDC2(1,5) = -TMIX
           TMIX = -DBDX2*DBDZ2*RVLN2
           D2BDC2(1,3) = TMIX
           D2BDC2(4,6) = TMIX
           D2BDC2(1,6) = -TMIX
           TMIX = -DBDY2*DBDZ2*RVLN2
           D2BDC2(2,3) = TMIX
           D2BDC2(5,6) = TMIX
           D2BDC2(2,6) = -TMIX
           !
           !-----the following terms are the same: y1,z2 and z1,y2, x1,y2 and
           !  y1,x2,
           !     and x1,z2 and z1,x2, as the order of the derivatives does not
           !     matter in this case..
           D2BDC1(3,5) = D2BDC1(2,6)
           D2BDC1(2,4) = D2BDC1(1,5)
           D2BDC1(3,4) = D2BDC1(1,6)
           D2BDC2(3,5) = D2BDC2(2,6)
           D2BDC2(2,4) = D2BDC2(1,5)
           D2BDC2(3,4) = D2BDC2(1,6)
           !
           IATOM = (ITHE1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (ITHE2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (ITHE3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           !
           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) &
                      + D2THC1(JCOOR,ICOOR)*DEBTDT &
                      + CBTA*(DBDC1(ICOOR)*DTHEC(JCOOR) &
                      + DBDC1(JCOOR)*DTHEC(ICOOR)) &
                      + CBTB*(DBDC2(ICOOR)*DTHEC(JCOOR) &
                      + DBDC2(JCOOR)*DTHEC(ICOOR)) &
                      + KBB*(DBDC1(ICOOR)*DBDC2(JCOOR) &
                      + DBDC1(JCOOR)*DBDC2(ICOOR))
              enddo
           enddo

           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2BDC1(JCOOR,ICOOR)*DEBDA
              enddo
           enddo
           !
           IATOM = (ITHE3 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           !
           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2BDC2(JCOOR,ICOOR)*DEBDB
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO

  !      write(6,*) 'EBNDANG>energy? = ',EBB
  !      write(6,*) 'EBNDANG>energy? = ',EBT

  RETURN
END SUBROUTINE EBNDANG

SUBROUTINE EBNDBND(EBBL,IP,JP,KP,LP,ICP,IB,ICB, &
     X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD)
  !
  ! Function
  !     Compute the bond*bond cross-term energy for the two outer bonds
  !     in a torsion.
  !     (if qsecd is true) 2nd derivatives w.r.t. cartesian coordinates
  !
  use chm_kinds
  !
  use dimens_fcm
  use param
  use number
  use cff_fcm
  !
  implicit none
  INTEGER IBOND1,IBOND2,ICOOR,IPRINT,IPHI1,IPHI2,IPHI3,IPHI4, &
       IBB,IBBA,IATOM,INDEX,LC(12),LI,LJ,JCOOR,NUMITR,DERIVS
  INTEGER IP(*),JP(*),KP(*),LP(*),ICP(*),IB(*),ICB(*)
  real(chm_real) BNDLN1,BNDLN2,DBDX1,DBDY1,DBDZ1,DBDX2,DBDY2, &
       DBDZ2,VX1,VX2,VY1,VY2,VZ1,VZ2,TDIAG,TMIX, &
       RVLN1,RVLN2,KBB,DEBDA,DEBDB,DIFB1,DIFB2,EBBL,DD(*)
  real(chm_real) D2BDC1(6,6),DBDC1(12),DBDC2(12),D2BBDC2(6,6)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  !
  !-----set up timing call
  !
  DO IBB=1,NBB
     IBBA = IBBW(IBB)
     IPHI1 = IP(IBBA)
     IPHI2 = JP(IBBA)
     IPHI3 = KP(IBBA)
     IPHI4 = LP(IBBA)
     !
     !-----get pointers to the 2 bonds in the torsion
     IBOND1 = IPBW(1,IBBA)
     IBOND2 = IPBW(3,IBBA)
     ! BIOVIA Code Start : Bug fix
     IF (ICB(IBOND1) == 0 .OR. ICB(IBOND2) == 0) CYCLE
     ! BIOVIA Code End
     !
     BNDLN1 = BL(IBOND1)
     BNDLN2 = BL(IBOND2)
     !
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
     !
     !-----compute the difference of the 2 bond lengths from the equilibrium
     !     values for the 2 bonds taken from their BOND potential parameters.
     !     these are found using itbw to find the bond and then using icb.
     DIFB1 = BNDLN1 - CBOND2(ICB(IBOND1))
     DIFB2 = BNDLN2 - CBOND2(ICB(IBOND2))
     !
     KBB = CBB2(ICP(IBBA))
     !
     !-----now compute bond*bond energy and add it to total bond*bond energy
     EBBL = EBBL + KBB*DIFB1*DIFB2
     !
     IF (DERIVS > 0) THEN
        DEBDB = KBB*DIFB1
        DEBDA = KBB*DIFB2
        !
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
        !
        IF (QSECD) THEN
           RVLN1 = ONE/BNDLN1
           RVLN2 = ONE/BNDLN2
           !
           VX1 = X(IPHI1) - X(IPHI2)
           VX2 = X(IPHI3) - X(IPHI4)
           VY1 = Y(IPHI1) - Y(IPHI2)
           VY2 = Y(IPHI3) - Y(IPHI4)
           VZ1 = Z(IPHI1) - Z(IPHI2)
           VZ2 = Z(IPHI3) - Z(IPHI4)
           DBDC1(1) = DBDX1
           DBDC1(2) = DBDY1
           DBDC1(3) = DBDZ1
           DBDC1(4) = -DBDX1
           DBDC1(5) = -DBDY1
           DBDC1(6) = -DBDZ1
           DBDC1(7) = ZERO
           DBDC1(8) = ZERO
           DBDC1(9) = ZERO
           DBDC1(10) = ZERO
           DBDC1(11) = ZERO
           DBDC1(12) = ZERO
           DBDC2(1) = ZERO
           DBDC2(2) = ZERO
           DBDC2(3) = ZERO
           DBDC2(4) = ZERO
           DBDC2(5) = ZERO
           DBDC2(6) = ZERO
           DBDC2(7) = DBDX2
           DBDC2(8) = DBDY2
           DBDC2(9) = DBDZ2
           DBDC2(10) = -DBDX2
           DBDC2(11) = -DBDY2
           DBDC2(12) = -DBDZ2
           !
           !-----d2bdx is dimensioned 6,6 which represents a 3,2 by 3,2 array.
           !     the components for atom 1 are stored from 1-3 and the components
           !     for atom 2 in 4-6. lat1 is the start index for atom a in d2bdx,
           !     lat2 is the start index for atom 2. i.e. lat2 + icoor is a
           !     component of atom 2. an index with no added component indicates
           !     this is for atom 1.
           !     first, we have the diagonal terms of each atom with itself.
           !     we also include the mixed term of the e.g. x coordinate of
           !     atom 1 and the x coordinate of atom 2 here.
           !     NOTE. this saves time for the (lat1,lat2) component by
           !           changing the sign of tdiag as it should be dbdx(,1)*
           !           dbdx(,2) but dbdx(,2) is simply -dbdx(,1).
           !
           TDIAG = (ONE - DBDX1*DBDX1)*RVLN1
           D2BDC1(1,1) = TDIAG
           D2BDC1(4,4) = TDIAG
           D2BDC1(1,4) = -TDIAG
           TDIAG = (ONE - DBDY1*DBDY1)*RVLN1
           D2BDC1(2,2) = TDIAG
           D2BDC1(5,5) = TDIAG
           D2BDC1(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ1*DBDZ1)*RVLN1
           D2BDC1(3,3) = TDIAG
           D2BDC1(6,6) = TDIAG
           D2BDC1(3,6) = -TDIAG
           TDIAG = (ONE - DBDX2*DBDX2)*RVLN2
           D2BBDC2(1,1) = TDIAG
           D2BBDC2(4,4) = TDIAG
           D2BBDC2(1,4) = -TDIAG
           TDIAG = (ONE - DBDY2*DBDY2)*RVLN2
           D2BBDC2(2,2) = TDIAG
           D2BBDC2(5,5) = TDIAG
           D2BBDC2(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ2*DBDZ2)*RVLN2
           D2BBDC2(3,3) = TDIAG
           D2BBDC2(6,6) = TDIAG
           D2BBDC2(3,6) = -TDIAG
           !
           !-----now the mixed terms for each atom separately, i.e. x of atom 1
           !     with y of atom 1 etc. in terms of coordinates this is 1,2 and
           !     1,3 and 2,3. this also includes the mixed terms x of atom 1 and
           !     y of atom 2 etc.
           TMIX = -DBDX1*DBDY1*RVLN1
           D2BDC1(1,2) = TMIX
           D2BDC1(4,5) = TMIX
           D2BDC1(1,5) = -TMIX
           TMIX = -DBDX1*DBDZ1*RVLN1
           D2BDC1(1,3) = TMIX
           D2BDC1(4,6) = TMIX
           D2BDC1(1,6) = -TMIX
           TMIX = -DBDY1*DBDZ1*RVLN1
           D2BDC1(2,3) = TMIX
           D2BDC1(5,6) = TMIX
           D2BDC1(2,6) = -TMIX
           TMIX = -DBDX2*DBDY2*RVLN2
           D2BBDC2(1,2) = TMIX
           D2BBDC2(4,5) = TMIX
           D2BBDC2(1,5) = -TMIX
           TMIX = -DBDX2*DBDZ2*RVLN2
           D2BBDC2(1,3) = TMIX
           D2BBDC2(4,6) = TMIX
           D2BBDC2(1,6) = -TMIX
           TMIX = -DBDY2*DBDZ2*RVLN2
           D2BBDC2(2,3) = TMIX
           D2BBDC2(5,6) = TMIX
           D2BBDC2(2,6) = -TMIX
           !
           !-----the following terms are the same: y1,z2 and z1,y2, x1,y2 and
           !  y1,x2,
           !     and x1,z2 and z1,x2, as the order of the derivatives does not
           !     matter in this case..
           D2BDC1(3,5) = D2BDC1(2,6)
           D2BDC1(2,4) = D2BDC1(1,5)
           D2BDC1(3,4) = D2BDC1(1,6)
           D2BBDC2(3,5) = D2BBDC2(2,6)
           D2BBDC2(2,4) = D2BBDC2(1,5)
           D2BBDC2(3,4) = D2BBDC2(1,6)
           !
           IATOM = (IPHI1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (IPHI2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (IPHI3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           IATOM = (IPHI4 - 1)*3
           LC(10) = IATOM + 1
           LC(11) = IATOM + 2
           LC(12) = IATOM + 3
           !
           DO ICOOR=1,12
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) &
                      + KBB*(DBDC1(ICOOR)*DBDC2(JCOOR) &
                      + DBDC1(JCOOR)*DBDC2(ICOOR))
              enddo
           enddo
           !
           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2BDC1(JCOOR,ICOOR)*DEBDA
              enddo
           enddo

           LC(1) = LC(7)
           LC(2) = LC(8)
           LC(3) = LC(9)
           LC(4) = LC(10)
           LC(5) = LC(11)
           LC(6) = LC(12)

           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2BBDC2(JCOOR,ICOOR)*DEBDB
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO

  !      write(6,*) 'EBNDBND>energy = ',EBBL

  RETURN
END SUBROUTINE EBNDBND


SUBROUTINE EBOND_CFF(EB,IB,JB,ICB,NB, &
     X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD)
  !
  ! Function
  !     this routine computes the bond energy and first derivatives
  !     of the bond energy w.r.t the cartesian coordinates of the atoms
  !     in each bond for all bonds in a molecule.
  !
  use chm_kinds
  !
  use dimens_fcm
  use param
  use number
  use cff_fcm
  implicit none
  !
  INTEGER IBOND,IBW1,IBW2,ICBP,IB(*),JB(*),ICB(*),NB
  INTEGER ICOOR,IBND,INDEX,LI,LJ,LC(6),JCOOR,IATOM,DERIVS
  real(chm_real) BINV,BNDLEN,DEBDB,DXX,EB,EBND,DX2,DX3,VECTX,VECTY, &
       VECTZ,DBDX1,DBDX2,DBDX3
  real(chm_real) D2EBDB,D2BDX(6,6),DBDC(6),TDIAG,TMIX,DD(*)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  !
  !-----loop over all bonds in this molecule and compute the bond length,
  !     energy and derivatives.
  !
  DO IBOND=1,NB
     !
     IBW1 = IB(IBOND)
     IBW2 = JB(IBOND)
     !
     !-----compute the bond length and derivatives of the bond length w.r.t
     !     the cartesians.
     !
     !-----compute the vector between the 2 atoms, whose cartesian
     !     components are in x1 and x2.
     VECTX = X(IBW1) - X(IBW2)
     VECTY = Y(IBW1) - Y(IBW2)
     VECTZ = Z(IBW1) - Z(IBW2)
     !
     !-----the distance between the 2 atoms is simply the dot product of
     !     this vector.
     !
     BNDLEN = VECTX*VECTX + VECTY*VECTY + VECTZ*VECTZ
     BINV = ONE/SQRT(BNDLEN)
     BNDLEN = BNDLEN*BINV
     BL(IBOND)=BNDLEN
     !
     !-----compute the 1st derivatives of the distance/bond length.
     DBDX1 = VECTX*BINV
     DBDX2 = VECTY*BINV
     DBDX3 = VECTZ*BINV
     !
     DBDX(IBOND) = DBDX1
     DBDY(IBOND) = DBDX2
     DBDZ(IBOND) = DBDX3
     !
     !-----now compute the bond energy and the derivative of the bond energy
     !     w.r.t the cartesian coordinates. save the energy of each bond
     !     in ebond.
     ICBP=ICB(IBOND)
     ! BIOVIA Code Start : Bug fix
     IF (ICBP == 0) CYCLE
     ! BIOVIA Code End
     !
     DXX = BNDLEN - CBOND2(ICBP)
     DX2 = DXX * DXX
     DX3 = DX2 * DXX
     EBND = CBOND1(ICBP)*DX2 + CBOND3(ICBP)*DX3 &
          + CBOND4(ICBP)*DX3*DXX
     !
     EB = EB + EBND
     !
     IF (DERIVS > 0) THEN
        DEBDB = TWO*CBOND1(ICBP)*DXX + THREE*CBOND3(ICBP)*DX2 &
             + FOUR*CBOND4(ICBP)*DX3
        DX(IBW1) = DX(IBW1) + DBDX1*DEBDB
        DX(IBW2) = DX(IBW2) - DBDX1*DEBDB
        DY(IBW1) = DY(IBW1) + DBDX2*DEBDB
        DY(IBW2) = DY(IBW2) - DBDX2*DEBDB
        DZ(IBW1) = DZ(IBW1) + DBDX3*DEBDB
        DZ(IBW2) = DZ(IBW2) - DBDX3*DEBDB
        !
        IF (QSECD) THEN
           TDIAG = (ONE - DBDX1*DBDX1)*BINV
           D2BDX(1,1) = TDIAG
           D2BDX(4,4) = TDIAG
           D2BDX(1,4) = -TDIAG
           TDIAG = (ONE - DBDX2*DBDX2)*BINV
           D2BDX(2,2) = TDIAG
           D2BDX(5,5) = TDIAG
           D2BDX(2,5) = -TDIAG
           TDIAG = (ONE - DBDX3*DBDX3)*BINV
           D2BDX(3,3) = TDIAG
           D2BDX(6,6) = TDIAG
           D2BDX(3,6) = -TDIAG
           TMIX = -DBDX1*DBDX2*BINV
           D2BDX(1,2) = TMIX
           D2BDX(4,5) = TMIX
           D2BDX(1,5) = -TMIX
           TMIX = -DBDX1*DBDX3*BINV
           D2BDX(1,3) = TMIX
           D2BDX(4,6) = TMIX
           D2BDX(1,6) = -TMIX
           TMIX = -DBDX2*DBDX3*BINV
           D2BDX(2,3) = TMIX
           D2BDX(5,6) = TMIX
           D2BDX(2,6) = -TMIX
           D2BDX(3,5) = D2BDX(2,6)
           D2BDX(2,4) = D2BDX(1,5)
           D2BDX(3,4) = D2BDX(1,6)
           !
           D2EBDB = TWO*CBOND1(ICBP) + (6.0*CBOND3(ICBP) &
                + TWELVE*CBOND4(ICBP)*DXX)*DXX
           !
           IATOM = (IBW1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (IBW2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           !
           DBDC(1)=DBDX1
           DBDC(2)=DBDX2
           DBDC(3)=DBDX3
           DBDC(4)=-DBDX1
           DBDC(5)=-DBDX2
           DBDC(6)=-DBDX3
           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX)+D2BDX(JCOOR,ICOOR)*DEBDB &
                      + DBDC(ICOOR)*DBDC(JCOOR)*D2EBDB
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO

  !      write(6,*) 'EBOND_CFF>energy = ',EB

  RETURN
END SUBROUTINE EBOND_CFF


SUBROUTINE EOPLN_CFF(EOPL,IM,JM,KM,LM,ICI,NI, &
     X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD)
  !
  ! Function
  use chm_kinds

  use dimens_fcm
  use number
  use param
  use cff_fcm
  !
  implicit none
  INTEGER ICOPT,IOP1,IOP2,IOP3,IOP4,IOPLN,IPLN,DERIVS
  INTEGER IM(*),JM(*),KM(*),LM(*),ICI(*),NI, &
       ICOOR,INDEX,I,J,JCOOR,IATOM,LC(12),LI,LJ,INDX1(12),INDX2(12)
  real(chm_real) COP1,DCM1,DCM2,DCM3,DEO,EOPL,F1,F2,G1,G2,VLN1X2, &
       VLN2X3,VX1X2,VY1X2,VZ1X2,VX2X3,VY2X3,VZ2X3,VEC1X,VEC1Y,VEC1Z, &
       VEC2X,VEC2Y,VEC2Z,VEC3X,VEC3Y,VEC3Z,VLN3X1,VX3X1,VY3X1,VZ3X1, &
       VLEN1,VLEN2,VLEN3,C13,CHI,CHI1,CHI2,CHI3,X2,Y2,Z2, &
       CM,THETA1,THETA2,THETA3,ABM,TEMP1,TEMP2,TEMP3, &
       DSX1,DSX2,DSX3,DSX4,DSX5,DSX6,DSX7,DSX8,DSX9,DSX10,DSX11,DSX12, &
       ESX1,ESX2,ESX3,ESX4,ESX5,ESX6,ESX7,ESX8,ESX9,ESX10,ESX11,ESX12, &
       FSX1,FSX2,FSX3,FSX4,FSX5,FSX6,FSX7,FSX8,FSX9,FSX10,FSX11,FSX12
  !
  real(chm_real) D2EODO,DEODO,DD(*),DOPN1,DOPN2,DOPN3, &
       DOPN4,DOPN5,DOPN6,DOPN7,DOPN8,DOPN9,DOPN10,DOPN11,DOPN12
  real(chm_real) DOPNC(12),DTHET(12),D2THET(12,12), &
       AVDSX(12),AVD2(12,12),D2(12,12)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  LOGICAL NEWT
  PARAMETER (C13 = ONE/THREE)
  DATA INDX1/10,11,12,4,5,6,1,2,3,7,8,9/
  DATA INDX2/7,8,9,4,5,6,10,11,12,1,2,3/
  NEWT = .TRUE.
  !
  !-----loop over all out of planes in this molecule and compute
  !     the out of plane angle, energy and derivatives.
  !
  DO IOPLN=1,NI
     ICOPT = ICI(IOPLN)
     ! BIOVIA Code Start : Bug fix
     IF (ICOPT == 0) CYCLE
     ! BIOVIA Code End
     !
     !   calculate the 1st and 2nd derivatives (w.r.t. the cartesian
     !   coordinates) of the (averaged) Wilson angle for the out of plane
     !   coordinate defined by atoms IOP1, IOP2, IOP3, and IOP4, where IOP2
     !   is the out of plane atom.
     !   The out-of-plane coordinate is returned by chi.
     !
     IOP2 = JM(IOPLN)
     X2 = X(IOP2)
     Y2 = Y(IOP2)
     Z2 = Z(IOP2)
     !
     !   Loop over the 3 Wilson angles which are to be averaged.
     !   Compute the three vectors in the out of plane
     !
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
     G1 = CM * C13 / COS(THETA1)
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
     G1 = CM * C13 / COS(THETA2)
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
     G1 = CM * C13 / COS(THETA3)
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
     CHI= (THETA1+THETA2+THETA3)*C13
     OPLN(IOPLN) = CHI
     COP1 = COPLN1(ICOPT)*CHI
     !
     !-----sum the out of plane energy into the total energy.
     EOPL = EOPL + COP1*CHI
     IF (DERIVS > 0) THEN
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
        IF (QSECD) THEN
           !
           !   Get the derivatives of the first out of plane distance (chi1).
           CALL OPDIF(VEC1X,VEC1Y,VEC1Z,VEC2X,VEC2Y,VEC2Z,VEC3X,VEC3Y, &
                VEC3Z,VX1X2,VY1X2,VZ1X2,VLN1X2,DOPNC,D2,NEWT)
           !   Get the derivatives of the out-of-plane coordinate (angle).  The 1st
           !   and 2nd derivatives are returned in arrays dthet and d2thet.
           CALL OPANG(VEC3X,VEC3Y,VEC3Z,VLEN3,CHI1,DOPNC,D2,AVDSX,AVD2, &
                NEWT)
           !   Get the derivatives of the second out of plane distance (chi2).
           CALL OPDIF(VEC3X,VEC3Y,VEC3Z,VEC1X,VEC1Y,VEC1Z,VEC2X,VEC2Y, &
                VEC2Z,VX3X1,VY3X1,VZ3X1,VLN3X1,DOPNC,D2,NEWT)
           !   Get the derivatives of the out-of-plane coordinate (angle).  The 1st
           !   and 2nd derivatives are returned in arrays dthet and d2thet.
           CALL OPANG(VEC2X,VEC2Y,VEC2Z,VLEN2,CHI2,DOPNC,D2,DTHET, &
                D2THET,NEWT)
           !   Average the derivatives.

           DO I=1,12
              AVDSX(I) = AVDSX(I) + DTHET(INDX1(I))
           enddo
           DO  I=1,12
              DO  J=1,I
                 AVD2(J,I) = AVD2(J,I) + D2THET(INDX1(J),INDX1(I))
              enddo
           enddo
           !   Get the derivatives of the third out of plane distance (chi3).
           CALL OPDIF(VEC2X,VEC2Y,VEC2Z,VEC3X,VEC3Y,VEC3Z,VEC1X,VEC1Y, &
                VEC1Z,VX2X3,VY2X3,VZ2X3,VLN2X3,DOPNC,D2,NEWT)
           !   Get the derivatives of the out-of-plane coordinate (angle).  The 1st
           !   and 2nd derivatives are returned in arrays dthet and d2thet.
           CALL OPANG(VEC1X,VEC1Y,VEC1Z,VLEN1,CHI3,DOPNC,D2,DTHET, &
                D2THET,NEWT)
           !   Average the derivatives.
           !
           DO  I=1,12
              DOPNC(I) = (AVDSX(I) + DTHET(INDX2(I))) * C13
           enddo
           ICOOR = 0
           DO  I=1,12
              DO  J=1,I
                 AVD2(J,I) = (AVD2(J,I) + D2THET(INDX2(J),INDX2(I)))*C13
                 ICOOR = ICOOR + 1
              enddo
           enddo

           !   Divide the coordinate and its derivatives by 3, since 3 angles are
           !   used to compute a symmetrized (i.e., averaged) coordinate.
           CHI= OPLN(IOPLN)
           COP1 = COPLN1(ICOPT)
           !
           D2EODO = TWO*COP1
           !
           !-----loop over all atoms involved in the interaction, compute
           !     the total coordinate number of each atom, i.e. the atom number * 3
           !     and add it to the list of coordinates stored in lcoor3
           !     for each atom found in the list of atoms stored in latom there
           !     are 3 coordinates (x,y,z) to be added to the second derivative
           !     matrix.
           !
           IATOM = (IOP1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (IOP2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (IOP3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           IATOM = (IOP4 - 1)*3
           LC(10) = IATOM + 1
           LC(11) = IATOM + 2
           LC(12) = IATOM + 3
           !
           !-----now add all contributions to the second derivative array
           !     indxdd computes where the element of the second derivative
           !     matrix is to be stored in the linear array dd. this is used
           !     as the second derivative matrix is symmetric and so only one
           !     diagonal half of the mtrix need be stored.
           DO ICOOR=1,12
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2EODO*(AVD2(JCOOR,ICOOR)*CHI &
                      + DOPNC(ICOOR)*DOPNC(JCOOR))
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO

  !      write(6,*) 'EOPLN_CFF>energy = ',EOPL

  RETURN
END SUBROUTINE EOPLN_CFF


!========================================================================
SUBROUTINE OPANG(CX,CY,CZ,OCM,CHI,DSX,D2,DTHET,D2THET,NEWT) 
  !
  use chm_kinds
  use number
  !
  implicit none
  INTEGER*4 I,J
  real(chm_real) CX,CY,CZ,OCM,CHI,DSX(12),D2(12,12), &
       DTHET(12),D2THET(12,12),CM,CM2,CM3,THETA,C1, &
       C2,C3,C4,C5,C6,CTHET,F1,F2,F3,F4,DCM1,DCM2,DCM3, &
       G1,G2,G3,G4,G5,G6
  LOGICAL NEWT

  CM  = ONE/OCM
  CM2 = CM*CM

  !-   CALCULATING CONSTANT TERMS

  THETA = ASIN (CHI*CM)
  CTHET = ONE/ COS(THETA)
  F1 = CHI*CM

  !-   CALCULATING FIRST PARTIAL DERIVSATIVES OF BOND C

  DCM1 = CX*CM
  DCM2 = CY*CM
  DCM3 = CZ*CM
  G1 = CTHET * CM
  G2 = G1 * F1

  !-   CALCULATING FIRST PARTIAL DERIVSATIVES OF THETA

  DTHET(1)  = G1*DSX(1) - G2*DCM1
  DTHET(2)  = G1*DSX(2) - G2*DCM2
  DTHET(3)  = G1*DSX(3) - G2*DCM3
  DTHET(4)  = G1*DSX(4) + G2*DCM1
  DTHET(5)  = G1*DSX(5) + G2*DCM2
  DTHET(6)  = G1*DSX(6) + G2*DCM3
  DTHET(7)  = G1*DSX(7)
  DTHET(8)  = G1*DSX(8)
  DTHET(9)  = G1*DSX(9)
  DTHET(10) = G1*DSX(10)
  DTHET(11) = G1*DSX(11)
  DTHET(12) = G1*DSX(12)
  !
  IF (NEWT) THEN
     !
     !-   CALCULATING SECOND PARTIAL DERIVSATIVES OF BOND C

     CM3 = CM2*CM
     F2 = SIN(THETA)*CTHET
     CTHET = CTHET * CM

     !-   diagonal terms
     G1 = (ONE - DCM1*DCM1)*CM
     G2 = (ONE - DCM2*DCM2)*CM
     G3 = (ONE - DCM3*DCM3)*CM

     !-   cross terms w/ different coordinates

     G4 = -DCM1*DCM2*CM
     G5 = -DCM1*DCM3*CM
     G6 = -DCM2*DCM3*CM

     C1 = DSX(1) - F1*DCM1
     C2 = DSX(2) - F1*DCM2
     C3 = DSX(3) - F1*DCM3
     C4 = DSX(4) + F1*DCM1
     C5 = DSX(5) + F1*DCM2
     C6 = DSX(6) + F1*DCM3

     !-   CALCULATING SECOND PARTIAL DERIVSATIVES OF THETA

     D2THET(1,1) = CTHET*(D2(1,1) - CM*(DSX(1)*DCM1 + DSX(1)*DCM1) &
          - F1*(G1 - TWO*CM*DCM1*DCM1) + F2*DTHET(1)*C1) 

     F3 = F2*DTHET(2)
     D2THET(1,2) = CTHET*(D2(1,2) - CM*(DSX(1)*DCM2 + DSX(2)*DCM1) &
          - F1*(G4 - TWO*CM*DCM2*DCM1) + F3*C1)
     D2THET(2,2) = CTHET*(D2(2,2) - CM*(DSX(2)*DCM2 + DSX(2)*DCM2) &
          - F1*(G2 - TWO*CM*DCM2*DCM2) + F3*C2) 

     F3 = F2*DTHET(3)
     F4 = TWO*CM*DCM3
     D2THET(1,3) = CTHET*(D2(1,3) - CM*(DSX(1)*DCM3 + DSX(3)*DCM1) &
          - F1*(G5 - F4*DCM1) + F3*C1) 
     D2THET(2,3) = CTHET*(D2(2,3) - CM*(DSX(2)*DCM3 + DSX(3)*DCM2) &
          - F1*(G6 - F4*DCM2) + F3*C2) 
     D2THET(3,3) = CTHET*(D2(3,3) - CM*(DSX(3)*DCM3 + DSX(3)*DCM3) &
          - F1*(G3 - F4*DCM3) + F3*C3) 

     F3 = F2*DTHET(4)
     F4 = TWO*CM*DCM1
     D2THET(1,4) = CTHET*(D2(1,4) + CM*(DSX(1)*DCM1 - DSX(4)*DCM1) &
          + F1*(G1 - F4*DCM1) + F3*C1) 
     D2THET(2,4) = CTHET*(D2(2,4) + CM*(DSX(2)*DCM1 - DSX(4)*DCM2) &
          + F1*(G4 - F4*DCM2) + F3*C2) 
     D2THET(3,4) = CTHET*(D2(3,4) + CM*(DSX(3)*DCM1 - DSX(4)*DCM3) &
          + F1*(G5 - F4*DCM3) + F3*C3) 
     D2THET(4,4) = CTHET*(D2(4,4) + CM*(DSX(4)*DCM1 + DSX(4)*DCM1) &
          - F1*(G1 - F4*DCM1) + F3*C4) 

     F3 = F2*DTHET(5)
     F4 = TWO*CM*DCM2
     D2THET(1,5) = CTHET*(D2(1,5) + CM*(DSX(1)*DCM2 - DSX(5)*DCM1) &
          + F1*(G4 - F4*DCM1) + F3*C1)
     D2THET(2,5) = CTHET*(D2(2,5) + CM*(DSX(2)*DCM2 - DSX(5)*DCM2) &
          + F1*(G2 - F4*DCM2) + F3*C2)
     D2THET(3,5) = CTHET*(D2(3,5) + CM*(DSX(3)*DCM2 - DSX(5)*DCM3) &
          + F1*(G6 - F4*DCM3) + F3*C3)
     D2THET(4,5) = CTHET*(D2(4,5) + CM*(DSX(4)*DCM2 + DSX(5)*DCM1) &
          - F1*(G4 - F4*DCM1) + F3*C4)
     D2THET(5,5) = CTHET*(D2(5,5) + CM*(DSX(5)*DCM2 + DSX(5)*DCM2) &
          - F1*(G2 - F4*DCM2) + F3*C5)

     F3 = F2*DTHET(6)
     F4 = TWO*CM*DCM3
     D2THET(1,6) = CTHET*(D2(1,6) + CM*(DSX(1)*DCM3 - DSX(6)*DCM1) &
          + F1*(G5 - F4*DCM1) + F3*C1)
     D2THET(2,6) = CTHET*(D2(2,6) + CM*(DSX(2)*DCM3 - DSX(6)*DCM2) &
          + F1*(G6 - F4*DCM2) + F3*C2)
     D2THET(3,6) = CTHET*(D2(3,6) + CM*(DSX(3)*DCM3 - DSX(6)*DCM3) &
          + F1*(G3 - F4*DCM3) + F3*C3)
     D2THET(4,6) = CTHET*(D2(4,6) + CM*(DSX(4)*DCM3 + DSX(6)*DCM1) &
          - F1*(G5 - F4*DCM1) + F3*C4)
     D2THET(5,6) = CTHET*(D2(5,6) + CM*(DSX(5)*DCM3 + DSX(6)*DCM2) &
          - F1*(G6 - F4*DCM2) + F3*C5)
     D2THET(6,6) = CTHET*(D2(6,6) + CM*(DSX(6)*DCM3 + DSX(6)*DCM3) &
          - F1*(G3 - F4*DCM3) + F3*C6)

     F3 = F2*DTHET(7)
     F4 = CM*DSX(7)
     D2THET(1,7) = CTHET*(D2(1,7) - F4*DCM1 + F3*C1)
     D2THET(2,7) = CTHET*(D2(2,7) - F4*DCM2 + F3*C2)
     D2THET(3,7) = CTHET*(D2(3,7) - F4*DCM3 + F3*C3)
     D2THET(4,7) = CTHET*(D2(4,7) + F4*DCM1 + F3*C4)
     D2THET(5,7) = CTHET*(D2(5,7) + F4*DCM2 + F3*C5)
     D2THET(6,7) = CTHET*(D2(6,7) + F4*DCM3 + F3*C6)
     D2THET(7,7) = CTHET*(D2(7,7) + F3*DSX(7))

     F3 = F2*DTHET(8)
     F4 = CM*DSX(8)
     D2THET(1,8) = CTHET*(D2(1,8) - F4*DCM1 + F3*C1)
     D2THET(2,8) = CTHET*(D2(2,8) - F4*DCM2 + F3*C2)
     D2THET(3,8) = CTHET*(D2(3,8) - F4*DCM3 + F3*C3)
     D2THET(4,8) = CTHET*(D2(4,8) + F4*DCM1 + F3*C4)
     D2THET(5,8) = CTHET*(D2(5,8) + F4*DCM2 + F3*C5)
     D2THET(6,8) = CTHET*(D2(6,8) + F4*DCM3 + F3*C6)
     D2THET(7,8) = CTHET*(D2(7,8) + F3*DSX(7))
     D2THET(8,8) = CTHET*(D2(8,8) + F3*DSX(8))

     F3 = F2*DTHET(9)
     F4 = CM*DSX(9)
     D2THET(1,9) = CTHET*(D2(1,9) - F4*DCM1 + F3*C1)
     D2THET(2,9) = CTHET*(D2(2,9) - F4*DCM2 + F3*C2)
     D2THET(3,9) = CTHET*(D2(3,9) - F4*DCM3 + F3*C3)
     D2THET(4,9) = CTHET*(D2(4,9) + F4*DCM1 + F3*C4)
     D2THET(5,9) = CTHET*(D2(5,9) + F4*DCM2 + F3*C5)
     D2THET(6,9) = CTHET*(D2(6,9) + F4*DCM3 + F3*C6)
     D2THET(7,9) = CTHET*(D2(7,9) + F3*DSX(7))
     D2THET(8,9) = CTHET*(D2(8,9) + F3*DSX(8))
     D2THET(9,9) = CTHET*(D2(9,9) + F3*DSX(9))

     F3 = F2*DTHET(10)
     F4 = CM*DSX(10)
     D2THET(1,10)  = CTHET*(D2(1,10)  - F4*DCM1 + F3*C1)
     D2THET(2,10)  = CTHET*(D2(2,10)  - F4*DCM2 + F3*C2)
     D2THET(3,10)  = CTHET*(D2(3,10)  - F4*DCM3 + F3*C3)
     D2THET(4,10)  = CTHET*(D2(4,10)  + F4*DCM1 + F3*C4)
     D2THET(5,10)  = CTHET*(D2(5,10)  + F4*DCM2 + F3*C5)
     D2THET(6,10)  = CTHET*(D2(6,10)  + F4*DCM3 + F3*C6)
     D2THET(7,10)  = CTHET*(D2(7,10)  + F3*DSX(7))
     D2THET(8,10)  = CTHET*(D2(8,10)  + F3*DSX(8))
     D2THET(9,10)  = CTHET*(D2(9,10)  + F3*DSX(9))
     D2THET(10,10) = CTHET*(D2(10,10) + F3*DSX(10))

     F3 = F2*DTHET(11)
     F4 = CM*DSX(11)
     D2THET(1,11)  = CTHET*(D2(1,11)  - F4*DCM1 + F3*C1)
     D2THET(2,11)  = CTHET*(D2(2,11)  - F4*DCM2 + F3*C2)
     D2THET(3,11)  = CTHET*(D2(3,11)  - F4*DCM3 + F3*C3)
     D2THET(4,11)  = CTHET*(D2(4,11)  + F4*DCM1 + F3*C4)
     D2THET(5,11)  = CTHET*(D2(5,11)  + F4*DCM2 + F3*C5)
     D2THET(6,11)  = CTHET*(D2(6,11)  + F4*DCM3 + F3*C6)
     D2THET(7,11)  = CTHET*(D2(7,11)  + F3*DSX(7))
     D2THET(8,11)  = CTHET*(D2(8,11)  + F3*DSX(8))
     D2THET(9,11)  = CTHET*(D2(9,11)  + F3*DSX(9))
     D2THET(10,11) = CTHET*(D2(10,11) + F3*DSX(10))
     D2THET(11,11) = CTHET*(D2(11,11) + F3*DSX(11))

     F3 = F2*DTHET(12)
     F4 = CM*DSX(12)
     D2THET(1,12)  = CTHET*(D2(1,12)  - F4*DCM1 + F3*C1)
     D2THET(2,12)  = CTHET*(D2(2,12)  - F4*DCM2 + F3*C2)
     D2THET(3,12)  = CTHET*(D2(3,12)  - F4*DCM3 + F3*C3)
     D2THET(4,12)  = CTHET*(D2(4,12)  + F4*DCM1 + F3*C4)
     D2THET(5,12)  = CTHET*(D2(5,12)  + F4*DCM2 + F3*C5)
     D2THET(6,12)  = CTHET*(D2(6,12)  + F4*DCM3 + F3*C6)
     D2THET(7,12)  = CTHET*(D2(7,12)  + F3*DSX(7))
     D2THET(8,12)  = CTHET*(D2(8,12)  + F3*DSX(8))
     D2THET(9,12)  = CTHET*(D2(9,12)  + F3*DSX(9))
     D2THET(10,12) = CTHET*(D2(10,12) + F3*DSX(10))
     D2THET(11,12) = CTHET*(D2(11,12) + F3*DSX(11))
     D2THET(12,12) = CTHET*(D2(12,12) + F3*DSX(12))

     DO I = 1,11
        DO J = I+1,12
           D2THET(J,I) = D2THET(I,J)
        enddo
     enddo
  ENDIF

  RETURN
END SUBROUTINE OPANG


!========================================================================
SUBROUTINE OPDIF(AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,ABX,ABY,ABZ,OABM, &
     DSX,D2,NEWT)
  !
  use chm_kinds
  use number
  !
  implicit none
  real(chm_real) AX,AY,AZ,BX,BY,BZ,CX,CY,CZ,ABX,ABY,ABZ,ABM, &
       DSX(12),D2(12,12), &
       ABM2,ABM3,DC,TEMP1,TEMP2,TEMP3,CONST1,CONST2,CONST3,F1,DABM1, &
       OABM,DABM2,DABM3,DABM4,DABM5,DABM6,DABM7,DABM8,DABM9,A1,A2,A3,A4, &
       DD1,DD2,DD3,DD4,DD5,DD6,DD7,DD8,DD9
  LOGICAL NEWT

  ABM = ONE/OABM
  ABM2 = ABM*ABM

  !-  CALCULATING CONSTANT TERMS

  DC = ABX * CX + ABY * CY + ABZ * CZ
  TEMP1 = BX - AX
  TEMP2 = BY - AY
  TEMP3 = BZ - AZ

  F1 = DC * ABM2

  !-  CALCULATING THE 1ST PARTIALS OF THE DOT PRODUCT  (AxB).C
  !-  W/ RESPECT TO THE 4 ATOMS

  DD1 = TEMP3*CY - TEMP2*CZ - ABX
  DD2 = AZ*CY - AY*CZ
  DD3 = BY*CZ - BZ*CY
  DD4 = TEMP1*CZ - TEMP3*CX - ABY
  DD5 = AX*CZ - AZ*CX
  DD6 = BZ*CX - BX*CZ
  DD7 = TEMP2*CX - TEMP1*CY - ABZ
  DD8 = AY*CX - AX*CY
  DD9 = BX*CY - BY*CX

  !-  CALCULATING AND STORING THE PARTIAL DERIVSATIVES OF THE MAGNITUDE
  !-  OF AxB W/ RESPECT TO THE 4 ATOMS.

  DABM1 = (TEMP3*ABY - TEMP2*ABZ) * ABM
  DABM2 = (AZ*ABY - AY*ABZ) * ABM
  DABM3 = (BY*ABZ - BZ*ABY) * ABM
  DABM4 = (TEMP1*ABZ - TEMP3*ABX) * ABM
  DABM5 = (AX*ABZ - AZ*ABX) * ABM
  DABM6 = (BZ*ABX - BX*ABZ) * ABM
  DABM7 = (TEMP2*ABX - TEMP1*ABY) * ABM
  DABM8 = (AY*ABX - AX*ABY) * ABM
  DABM9 = (BX*ABY - BY*ABX) * ABM

  !-  CALCULATING THE 1ST PARTIALS OF CHI W/ RESPECT TO THE 4 ATOMS 

  DSX(1) = ABX * ABM
  DSX(2) = ABY * ABM
  DSX(3) = ABZ * ABM
  DSX(4) = DD1 * ABM - DABM1 * F1
  DSX(5) = DD4 * ABM - DABM4 * F1
  DSX(6) = DD7 * ABM - DABM7 * F1
  DSX(7) = DD2 * ABM - DABM2 * F1
  DSX(8) = DD5 * ABM - DABM5 * F1
  DSX(9) = DD8 * ABM - DABM8 * F1
  DSX(10) = DD3 * ABM - DABM3 * F1
  DSX(11) = DD6 * ABM - DABM6 * F1
  DSX(12) = DD9 * ABM - DABM9 * F1
  IF (NEWT) THEN
     ABM3 = ABM2*ABM
     CONST1 = -ABX * ABM2
     CONST2 = -ABY * ABM2
     CONST3 = -ABZ * ABM2

     !-  CALCULATING THE 2ND PARTIALS OF CHI W/ RESPECT TO THE 1ST
     !-  PARTIALS OF X2, Y2, Z2

     A3 = -DC * ABM3
     A4 = 3.0 * DC * ABM3

     D2(1,1)  = ZERO
     D2(1,2)  = ZERO
     D2(1,3)  = ZERO
     D2(1,4)  = CONST1 * DABM1
     D2(1,5)  = -ABM * TEMP3 + CONST1 * DABM4
     D2(1,6)  = ABM * TEMP2 + CONST1 * DABM7
     D2(1,7)  = CONST1 * DABM2
     D2(1,8)  = -ABM * AZ + CONST1 * DABM5
     D2(1,9)  = ABM * AY + CONST1 * DABM8
     D2(1,10) = CONST1 * DABM3
     D2(1,11) = ABM * BZ + CONST1 * DABM6
     D2(1,12) = -ABM * BY + CONST1 * DABM9

     D2(2,2)  = ZERO
     D2(2,3)  = ZERO
     D2(2,4)  = ABM * TEMP3 + CONST2 * DABM1
     D2(2,5)  = CONST2 * DABM4
     D2(2,6)  = -ABM * TEMP1 + CONST2 * DABM7
     D2(2,7)  = ABM * AZ + CONST2 * DABM2
     D2(2,8)  = CONST2 * DABM5
     D2(2,9)  = -ABM * AX + CONST2 * DABM8
     D2(2,10) = -ABM * BZ + CONST2 * DABM3
     D2(2,11) = CONST2 * DABM6
     D2(2,12) = ABM * BX + CONST2 * DABM9

     D2(3,3)  = ZERO
     D2(3,4)  = -ABM * TEMP2 + CONST3 * DABM1
     D2(3,5)  = ABM * TEMP1 + CONST3 * DABM4
     D2(3,6)  = CONST3 * DABM7
     D2(3,7)  = -ABM * AY + CONST3 * DABM2
     D2(3,8)  = ABM * AX + CONST3 * DABM5
     D2(3,9)  = CONST3 * DABM8
     D2(3,10) = ABM * BY + CONST3 * DABM3
     D2(3,11) = -ABM * BX + CONST3 * DABM6
     D2(3,12) = CONST3 * DABM9

     A1 = A4*DABM1 - ABM2*DD1
     A2 = -ABM2*DABM1
     D2(4,4)  = A1*DABM1 + A2*DD1 + A3*(TEMP3*TEMP3+TEMP2*TEMP2)
     D2(4,5)  = A1*DABM4 + A2*DD4 - A3 * (TEMP1 * TEMP2)
     D2(4,6)  = A1*DABM7 + A2*DD7 - A3 * (TEMP1 * TEMP3)
     D2(4,7)  = A1*DABM2 + A2*DD2 + A3 * (AZ*TEMP3 + AY*TEMP2)
     D2(4,8)  = A1*DABM5 + A2*DD5 - A3*(AX*TEMP2 + ABZ) + ABM*(AZ-CZ)
     D2(4,9)  = A1*DABM8 + A2*DD8 + A3*(ABY - AX*TEMP3) + ABM*(CY-AY)
     D2(4,10) = A1*DABM3 + A2*DD3 - A3 * (BZ*TEMP3 + BY*TEMP2)
     D2(4,11) = A1*DABM6 + A2*DD6 + A3*(BX*TEMP2 + ABZ) + ABM*(CZ-BZ)
     D2(4,12) = A1*DABM9 + A2*DD9 + A3*(BX*TEMP3 - ABY) + ABM*(BY-CY)

     A1 = A4*DABM4 - ABM2*DD4
     A2 = -ABM2*DABM4
     D2(5,5)  = A1*DABM4 + A2*DD4 + A3*(TEMP3*TEMP3+TEMP1*TEMP1)
     D2(5,6)  = A1*DABM7 + A2*DD7 - A3*(TEMP2*TEMP3)
     D2(5,7)  = A1*DABM2 + A2*DD2 + A3*(ABZ - AY*TEMP1) + ABM*(CZ-AZ)
     D2(5,8)  = A1*DABM5 + A2*DD5 + A3*(AX*TEMP1 + AZ*TEMP3)
     D2(5,9)  = A1*DABM8 + A2*DD8 - A3*(AY*TEMP3 + ABX) + ABM*(AX-CX)
     D2(5,10) = A1*DABM3 + A2*DD3 + A3*(BY*TEMP1 - ABZ) + ABM*(BZ-CZ)
     D2(5,11) = A1*DABM6 + A2*DD6 - A3*(BZ*TEMP3 + BX*TEMP1)
     D2(5,12) = A1*DABM9 + A2*DD9 + A3*(BY*TEMP3 + ABX) + ABM*(CX-BX)

     A1 = A4*DABM7 - ABM2*DD7
     A2 = -ABM2*DABM7
     D2(6,6)  = A1*DABM7 + A2*DD7 + A3*(TEMP2*TEMP2 + TEMP1*TEMP1) 
     D2(6,7)  = A1*DABM2 + A2*DD2 - A3*(AZ*TEMP1 + ABY) + ABM*(AY-CY)
     D2(6,8)  = A1*DABM5 + A2*DD5 + A3*(ABX - AZ*TEMP2) + ABM*(CX-AX)
     D2(6,9)  = A1*DABM8 + A2*DD8 + A3*(AY*TEMP2 + AX*TEMP1)
     D2(6,10) = A1*DABM3 + A2*DD3 + A3*(BZ*TEMP1 + ABY) + ABM*(CY-BY)
     D2(6,11) = A1*DABM6 + A2*DD6 + A3*(BZ*TEMP2 - ABX) + ABM*(BX-CX)
     D2(6,12) = A1*DABM9 + A2*DD9 - A3*(BX*TEMP1 + BY*TEMP2)

     A1 = A4*DABM2 - ABM2*DD2
     A2 = -ABM2*DABM2
     D2(7,7)  = A1*DABM2 + A2*DD2 + A3*(AY*AY + AZ*AZ)
     D2(7,8)  = A1*DABM5 + A2*DD5 - A3*(AX*AY)
     D2(7,9)  = A1*DABM8 + A2*DD8 - A3*(AX*AZ)
     D2(7,10) = A1*DABM3 + A2*DD3 - A3*(AY*BY + AZ*BZ)
     D2(7,11) = A1*DABM6 + A2*DD6 + A3*(AY*BX - ABZ) - ABM*CZ 
     D2(7,12) = A1*DABM9 + A2*DD9 + A3*(AZ*BX + ABY) + ABM*CY

     A1 = A4*DABM5 - ABM2*DD5
     A2 = -ABM2*DABM5
     D2(8,8)  = A1*DABM5 + A2*DD5 + A3*(AX*AX + AZ*AZ)
     D2(8,9)  = A1*DABM8 + A2*DD8 - A3*(AY*AZ)
     D2(8,10) = A1*DABM3 + A2*DD3 + A3*(AX*BY + ABZ) + ABM*CZ
     D2(8,11) = A1*DABM6 + A2*DD6 - A3*(AX*BX + AZ*BZ)
     D2(8,12) = A1*DABM9 + A2*DD9 + A3*(AZ*BY - ABX) - ABM*CX

     A1 = A4*DABM8 - ABM2*DD8
     A2 = -ABM2*DABM8
     D2(9,9)  = A1*DABM8 + A2*DD8 + A3*(AX*AX + AY*AY)
     D2(9,10) = A1*DABM3 + A2*DD3 + A3*(AX*BZ - ABY) - ABM*CY
     D2(9,11) = A1*DABM6 + A2*DD6 + A3*(AY*BZ + ABX) + ABM*CX
     D2(9,12) = A1*DABM9 + A2*DD9 - A3*(AX*BX + AY*BY)

     A1 = A4*DABM3 - ABM2*DD3
     A2 = -ABM2*DABM3
     D2(10,10) = A1*DABM3 + A2*DD3 + A3*(BY*BY + BZ*BZ)
     D2(10,11) = A1*DABM6 + A2*DD6 - A3*(BY*BX)
     D2(10,12) = A1*DABM9 + A2*DD9 - A3*(BX*BZ)

     A1 = A4*DABM6 - ABM2*DD6
     A2 = -ABM2*DABM6
     D2(11,11) = A1*DABM6 + A2*DD6 + A3*(BX*BX + BZ*BZ)
     D2(11,12) = A1*DABM9 + A2*DD9 - A3*(BY*BZ)

     A1 = A4*DABM9 - ABM2*DD9
     A2 = -ABM2*DABM9
     D2(12,12) = A1*DABM9 + A2*DD9 + A3*(BX*BX + BY*BY)

  ENDIF
  RETURN
END SUBROUTINE OPDIF


SUBROUTINE EPHI_CFF(EP,IP,JP,KP,LP,ICP,NP,ICT,IB,ICB, &
     X,Y,Z,DX,DY,DZ,DD,DERIVS,QSECD, &
     ETP,EBP,EMBP,ETTP)
  !
  use chm_kinds
  use energym
  use number
  use consta
  !
  use stream
  use dimens_fcm
  use param
  use cff_fcm
  !
  implicit none
  INTEGER ICPT,IPHI,IPHI1,IPHI2,IPHI3,IPHI4,IPH,IFILL,NPHM, &
       ITHTA,ITHTB,ISGN1,ISGN2,ISGN3,ISGN4,IBOND1,IBOND2,IBOND3,IB(*)
  INTEGER IP(*),JP(*),KP(*),LP(*),ICP(*),NP,LC(12),DERIVS,ICT(*)
  real(chm_real) COSPHI,CPH1,CPH2,EP,PHI1,PHI2,PHI3, &
       EPH,EBP,ETP,ETTP,EMBP, &
       XSIGN,SMALLA,VLN1X2,VLN3X2,VXIJ,VXKL,VXKJ,EPL,ETTPL,EBPL, &
       VYIJ,VYKJ,VYKL,VZIJ,VZKJ,VZKL,VX1X2,VY1X2,VZ1X2,VX3X2,VY3X2, &
       VZ3X2,C1,C2,C3,C4,C5,C6,COSV12,COSV23,DOTP12,DOTP23,SINV12, &
       SINV23,VLEN1,VLEN2,VLEN3,CTTP,DTHE1,DTHE2,DTTPDP,DTTPDT, &
       DTTPDA,DPHI,DPHI1,DPHI2,DPHI3,DPHI4,DPHI5,DPHI6,DPHI7, &
       DPHI8,DPHI9,DPHI10,DPHI11,DPHI12,DIFB1,DIFB2,DIFB3, &
       SV12,SV112,SV23,SV223,CPH3,S1,S2,S3,COSP1,COSP2,COSP3, &
       ENGB1,ENGB2,ENGB3,ENGT1,ENGT2,SINP1,SINP2,SINP3, &
       DENG,DBDX1,DBDY1,DBDZ1,DBDX2,DBDY2,DBDZ2,DBDX3,DBDY3,DBDZ3
  !
  INTEGER IATOM,LI,LJ,JCOOR,I,J,J3,J6,J9,K,ICOOR,INDEX,ICB(*)
  real(chm_real) D2PHI,V1,V2,V3,V4,A2,AI,B2,BI,V2I, &
       X1,X2,X3,X4,XA,XB,XX,DD(*), &
       COSTH1,COSTH2,CSINPH,D2TTP,D2TTPX,THETA1,THETA2, &
       D1TTPX,D3TTPX,CRST,CRVL4,CTHST2,RSNTHE,RVL123,RVL124, &
       RVL213,RVLN1,RVLN12,RVLN13,RVLN2,RVLN22,RVLN23,SINTH2,SINTHE, &
       SMALL1,VL1DV2,VL2DV1,BNDLN1,BNDLN2,DVEC11,DVEC12,DVEC21, &
       DVEC22,DVEC31,DVEC32,BNDLN3,DENGB1, &
       DENGB2,DENGB3,DENGT1,DENGT2,TDIAG,TMIX, &
       RVLN3,D2NGB1,D2NGB2,D2NGB3,D2NGT1, &
       D2NGT2,D2ENG,AA,AC
  real(chm_real) D2PHDX(12,12),DPHIC(12), &
       BM(12),AB(12),BC(12),AD(12),BD(12),A(3,12),B1(3,12), &
       DTHEC1(12),DTHEC2(12),D2THC1(9,9),D2VEC(6,6),D2THC2(9,9), &
       DBDC1(6),DBDC2(12),DBDC3(12),D2BDC1(6,6),D2BDC2(6,6),D2BDC3(6,6)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QSECD
  !
  DATA SMALLA/0.0001D0/
  DATA SMALL1/1.0D-10/
  IF(.NOT.QETERM(DIHE)) RETURN
  IF(.NOT.QETERM(ANGLE)) CALL ANGLES1_CFF(X,Y,Z)
  !
  DO IPHI=1,NP
     !
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
     !
     !-----compute the cross product of vectors 1 X 2 and 3 X 2.
     VX1X2 = VYIJ*VZKJ - VZIJ*VYKJ
     VY1X2 = -VXIJ*VZKJ + VZIJ*VXKJ
     VZ1X2 = VXIJ*VYKJ - VYIJ*VXKJ
     VX3X2 = VYKL*VZKJ - VZKL*VYKJ
     VY3X2 = -VXKL*VZKJ + VZKL*VXKJ
     VZ3X2 = VXKL*VYKJ - VYKL*VXKJ
     !
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
     !
     XSIGN = VXKJ*(VY1X2*VZ3X2 - VZ1X2*VY3X2) &
          + VYKJ*(-VX1X2*VZ3X2 + VZ1X2*VX3X2) &
          + VZKJ*(VX1X2*VY3X2 - VY1X2*VX3X2)
     !
     !-----save the torsion angle (on range 0-360) in torsion angle array.
     !     NOTE. because phiang is now 180 - or + phiang, cosphi does not
     !           correspond to phiang after this point. cos(phiang) is
     !           actually -cosphi. This is important in phien and difph.
     !
     PHI1 = PI - ACOS(COSPHI)*SIGN(ONE,-XSIGN)
     PH(IPHI) = PHI1
     PHI2 = PHI1 * TWO
     PHI3 = PHI1 * THREE
     !
     !-----compute the length of the 3 vectors of the torsion angle.
     VLEN1 = BL(IPBW(1,IPHI))
     VLEN2 = BL(IPBW(2,IPHI))
     VLEN3 = BL(IPBW(3,IPHI))
     DOTP12 = VXIJ*VXKJ + VYIJ*VYKJ + VZIJ*VZKJ
     DOTP23 = VXKJ*VXKL + VYKJ*VYKL + VZKJ*VZKL
     !
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
     !
     !-----calculate the energy and derivatives for the barrier.
     !
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
     !
     !-----get pointers to the 2 valence angles and 3 bonds in the torsion
     IBOND1 = IPBW(1,IPHI)
     IBOND2 = IPBW(2,IPHI)
     IBOND3 = IPBW(3,IPHI)
     ITHTA = IPTW(1,IPHI)
     ITHTB = IPTW(2,IPHI)
     !
     ! BIOVIA Code Start : Bug fix
     DIFB1 = 0
     DIFB2 = 0
     DIFB3 = 0
     IF (ICB(IBOND1) > 0) DIFB1 = BL(IBOND1) - CBOND2(ICB(IBOND1))
     IF (ICB(IBOND2) > 0) DIFB2 = BL(IBOND2) - CBOND2(ICB(IBOND2))
     IF (ICB(IBOND3) > 0) DIFB3 = BL(IBOND3) - CBOND2(ICB(IBOND3))
     !
     DTHE1 = 0
     DTHE2 = 0
     IF (ICT(ITHTA) > 0) DTHE1 = TH(ITHTA) - CTHET2(ICT(ITHTA))
     IF (ICT(ITHTB) > 0) DTHE2 = TH(ITHTB) - CTHET2(ICT(ITHTB))
     ! BIOVIA Code End
     COSP1 = COS(PHI1)
     COSP2 = COS(PHI2)
     COSP3 = COS(PHI3)
     !
     !-----compute the bond*phi energy
     !
     ENGB1 = CBP11(ICPT)*COSP1 + CBP12(ICPT)*COSP2+CBP13(ICPT)*COSP3
     ENGB2 = CBP21(ICPT)*COSP1 + CBP22(ICPT)*COSP2+CBP23(ICPT)*COSP3
     ENGB3 = CBP31(ICPT)*COSP1 + CBP32(ICPT)*COSP2+CBP33(ICPT)*COSP3
     !
     !-----add the energy to the total theta*theta*phi energy
     !
     EP = EP + EPH
     EBP = EBP + DIFB1*ENGB1 + DIFB3*ENGB3
     EMBP = EMBP + DIFB2*ENGB2
     ENGT1 = CTP11(ICPT)*COSP1 + CTP12(ICPT)*COSP2+CTP13(ICPT)*COSP3
     ENGT2 = CTP21(ICPT)*COSP1 + CTP22(ICPT)*COSP2+CTP23(ICPT)*COSP3
     ETP = ETP + DTHE1*ENGT1 + DTHE2*ENGT2
     ETTP = ETTP + CTTP*COSPHI*DTHE1*DTHE2
     !
     IF (DERIVS > 0) THEN
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
        !
        DTTPDP = CTTP*SIN(PHI1)*DTHE1*DTHE2 + DPHI + DENG
        DTTPDT = CTTP*COSPHI*DTHE2 + ENGT1
        DTTPDA = CTTP*COSPHI*DTHE1 + ENGT2
        !
        !-----compute the 1st derivatives of the angle w.r.t the cartesian
        !     coordinates from the derivatives w.r.t the vectors forming the
        !     angle.
        ISGN1 = 2 + IPHFLG(1,IPHI)
        ISGN2 = 2 - IPHFLG(1,IPHI)
        ISGN3 = 2 + IPHFLG(2,IPHI)
        ISGN4 = 2 - IPHFLG(2,IPHI)
        !
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
        !
        IF (QSECD) THEN
           DO I=1,12
              BM(I)=ZERO
              AB(I)=ZERO
              BC(I)=ZERO
              AD(I)=ZERO
              BD(I)=ZERO
              A(1,I)=ZERO
              A(2,I)=ZERO
              A(3,I)=ZERO
              B1(1,I)=ZERO
              B1(2,I)=ZERO
              B1(3,I)=ZERO
           ENDDO
           DTHEC1(10)   = ZERO
           DTHEC1(11)   = ZERO
           DTHEC1(12)   = ZERO
           DTHEC2(1)    = ZERO
           DTHEC2(2)    = ZERO
           DTHEC2(3)    = ZERO
           DBDC2(1)     = ZERO
           DBDC2(2)     = ZERO
           DBDC2(3)     = ZERO
           DBDC2(10)    = ZERO
           DBDC2(11)    = ZERO
           DBDC2(12)    = ZERO
           D2PHDX(1,10) = ZERO
           D2PHDX(2,10) = ZERO
           D2PHDX(3,10) = ZERO
           D2PHDX(1,11) = ZERO
           D2PHDX(2,11) = ZERO
           D2PHDX(3,11) = ZERO
           D2PHDX(1,12) = ZERO
           D2PHDX(2,12) = ZERO
           D2PHDX(3,12) = ZERO
           !
           BNDLN1 = BL(IBOND1)
           BNDLN2 = BL(IBOND2)
           BNDLN3 = BL(IBOND3)
           !
           THETA1 = TH(ITHTA)
           THETA2 = TH(ITHTB)
           !
           COSTH1 = COS(THETA1)
           COSTH2 = COS(THETA2)
           !
           PHI1 = PH(IPHI)
           PHI2 = PHI1 * TWO
           PHI3 = PHI1 * THREE
           !
           !
           !-----calculate the energy and derivatives for the barrier.
           !
           CPH1=CPHI1(ICPT)
           CPH2=CPHI2(ICPT)
           CPH3=CPHI3(ICPT)
           S1=CSGN1(ICPT)
           S2=CSGN2(ICPT)
           S3=CSGN3(ICPT)
           DPHI=CPH1*S1*SIN(PHI1) + TWO*CPH2*S2*SIN(PHI2) &
                + THREE*CPH3*S3*SIN(PHI3)
           D2PHI=CPH1*S1*COS(PHI1) + FOUR*S2*CPH2*COS(PHI2) &
                + NINE*CPH3*S3*COS(PHI3)
           !
           CTTP = CPHI4(ICPT)
           D2TTP = -CTTP*COSP1*DTHE1*DTHE2
           CSINPH = CTTP*SINP1
           DTTPDP = CSINPH*DTHE1*DTHE2
           DTTPDT = CTTP*COSP1*DTHE2 + ENGT1
           DTTPDA = CTTP*COSP1*DTHE1 + ENGT2
           !
           !-----compute the cross product of vectors 1 X 2 and 3 X 2.
           VX1X2 = VYIJ*VZKJ - VZIJ*VYKJ
           VY1X2 = -VXIJ*VZKJ + VZIJ*VXKJ
           VZ1X2 = VXIJ*VYKJ - VYIJ*VXKJ
           VX3X2 = VYKL*VZKJ - VZKL*VYKJ
           VY3X2 = -VXKL*VZKJ + VZKL*VXKJ
           VZ3X2 = VXKL*VYKJ - VYKL*VXKJ
           !
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
           !
           !-----compute the length of the 3 vectors of the torsion angle.
           VLEN1 = BL(IPBW(1,IPHI))
           VLEN2 = BL(IPBW(2,IPHI))
           VLEN3 = BL(IPBW(3,IPHI))
           DOTP12 = VXIJ*VXKJ + VYIJ*VYKJ + VZIJ*VZKJ
           DOTP23 = VXKJ*VXKL + VYKJ*VYKL + VZKJ*VZKL
           !
           COSV12 = DOTP12/(VLEN1*VLEN2)
           COSV23 = DOTP23/(VLEN2*VLEN3)
           SINV12 = SQRT(ONE - COSV12*COSV12)
           SINV23 = SQRT(ONE - COSV23*COSV23)
           C1 = ONE/(VLN1X2*VLEN1*SINV12)
           C2 = -(VLEN2 - VLEN1*COSV12)/(VLEN1*VLEN2*SINV12*VLN1X2)
           C3 = COSV23/(VLN3X2*VLEN2*SINV23)
           C4 = -(VLEN2 - VLEN3*COSV23)/(VLEN3*VLEN2*SINV23*VLN3X2)
           C5 = COSV12/(VLN1X2*VLEN2*SINV12)
           C6 = ONE/(VLN3X2*VLEN3*SINV23)
           !
           DPHIC(1) = VX1X2*C1
           DPHIC(2) = VY1X2*C1
           DPHIC(3) = VZ1X2*C1
           DPHIC(4) = C2*VX1X2 - VX3X2*C3
           DPHIC(5) = C2*VY1X2 - VY3X2*C3
           DPHIC(6) = C2*VZ1X2 - VZ3X2*C3
           DPHIC(7) = C4*VX3X2 - VX1X2*C5
           DPHIC(8) = C4*VY3X2 - VY1X2*C5
           DPHIC(9) = C4*VZ3X2 - VZ1X2*C5
           DPHIC(10) = VX3X2*C6
           DPHIC(11) = VY3X2*C6
           DPHIC(12) = VZ3X2*C6
           !
           !-----compute the 1st derivatives of the angle w.r.t the cartesian
           !     coordinates from the derivatives w.r.t the vectors forming the
           !     angle.
           ISGN1 = 2 + IPHFLG(1,IPHI)
           ISGN2 = 2 - IPHFLG(1,IPHI)
           ISGN3 = 2 + IPHFLG(2,IPHI)
           ISGN4 = 2 - IPHFLG(2,IPHI)
           DTHEC1(1) = DTHDX(ISGN1,ITHTA)
           DTHEC1(2) = DTHDY(ISGN1,ITHTA)
           DTHEC1(3) = DTHDZ(ISGN1,ITHTA)
           DTHEC1(4) = DTHDX(2,ITHTA)
           DTHEC1(5) = DTHDY(2,ITHTA)
           DTHEC1(6) = DTHDZ(2,ITHTA)
           DTHEC1(7) = DTHDX(ISGN2,ITHTA)
           DTHEC1(8) = DTHDY(ISGN2,ITHTA)
           DTHEC1(9) = DTHDZ(ISGN2,ITHTA)
           !
           DTHEC2(4) = DTHDX(ISGN3,ITHTB)
           DTHEC2(5) = DTHDY(ISGN3,ITHTB)
           DTHEC2(6) = DTHDZ(ISGN3,ITHTB)
           DTHEC2(7) = DTHDX(2,ITHTB)
           DTHEC2(8) = DTHDY(2,ITHTB)
           DTHEC2(9) = DTHDZ(2,ITHTB)
           DTHEC2(10) = DTHDX(ISGN4,ITHTB)
           DTHEC2(11) = DTHDY(ISGN4,ITHTB)
           DTHEC2(12) = DTHDZ(ISGN4,ITHTB)
           !
           !   calculate the 2nd derivatives of phi.
           !
           BI=ONE/VLEN2
           !   Determine 1st der. of the magnitude of bond length vlen2.
           BM(7)=VXKJ*BI
           BM(4)=-BM(7)
           BM(8)=VYKJ*BI
           BM(5)=-BM(8)
           BM(9)=VZKJ*BI
           BM(6)=-BM(9)
           !   Determine 1st der. of the dot products of vectr1 and vectr2
           !   and also vectr2 and vectr3.
           AB(1)=VXKJ
           AB(4)=-(VXIJ+VXKJ)
           AB(7)=VXIJ
           BC(4)=-VXKL
           BC(7)=VXKJ+VXKL
           BC(10)=-VXKJ
           AB(2)=VYKJ
           AB(5)=-(VYIJ+VYKJ)
           AB(8)=VYIJ
           BC(5)=-VYKL
           BC(8)=VYKJ+VYKL
           BC(11)=-VYKJ
           AB(3)=VZKJ
           AB(6)=-(VZIJ+VZKJ)
           AB(9)=VZIJ
           BC(6)=-VZKL
           BC(9)=VZKJ+VZKL
           BC(12)=-VZKJ
           !   Determine the 1st derivatives of the components of vec1x2 and
           !   vec3x2.
           A(1,2)=VZKJ
           A(2,1)=-VZKJ
           A(1,3)=-VYKJ
           A(3,1)=VYKJ
           A(2,3)=VXKJ
           A(3,2)=-VXKJ
           A(1,8)=-VZIJ
           A(2,7)=VZIJ
           A(1,9)=VYIJ
           A(3,7)=-VYIJ
           A(2,9)=-VXIJ
           A(3,8)=VXIJ
           B1(1,5)=VZKL
           B1(2,4)=-VZKL
           B1(1,6)=-VYKL
           B1(3,4)=VYKL
           B1(2,6)=VXKL
           B1(3,5)=-VXKL
           DO J=1,3
              J3=J+3
              J6=J+6
              J9=J+9
              DO I=1,3
                 B1(I,J9)=-A(I,J)
                 B1(I,J6)=-(B1(I,J9)+B1(I,J3))
                 A(I,J3)=-(A(I,J)+A(I,J6))
              enddo
           enddo

           !   Compute 1st derivatives of vln1x2 and vln3x2 w.r.t. cartesian
           !   coordinates of all four atoms.
           AI=ONE/VLN1X2
           BI=ONE/VLN3X2
           K=0
           DO J=1,4
              K=K+1
              AD(K)=AI*(VY1X2*A(2,K)+VZ1X2*A(3,K))
              BD(K)=BI*(VY3X2*B1(2,K)+VZ3X2*B1(3,K))
              K=K+1
              AD(K)=AI*(VX1X2*A(1,K)+VZ1X2*A(3,K))
              BD(K)=BI*(VX3X2*B1(1,K)+VZ3X2*B1(3,K))
              K=K+1
              AD(K)=AI*(VX1X2*A(1,K)+VY1X2*A(2,K))
              BD(K)=BI*(VX3X2*B1(1,K)+VY3X2*B1(2,K))
           enddo
           !   Compute 1st three rows of 2nd derivative matrix (i.e., all terms
           !   involving a derivative w.r.t. a cartesian coordinate of atom 1).
           A2=AI*AI
           XA=-TWO*AI*VLEN2
           B2=BI*BI
           XB=-TWO*BI*VLEN2
           !   Loop over 1st three rows of 2nd der. matrix.
           !   Loop over columns of 2nd der. matrix.
           V1 =VX1X2
           D2PHDX(1,1:9)=(V1*(BM(1:9)+AD(1:9)*XA)+A(1,1:9)*VLEN2)*A2
           V1 =VY1X2
           D2PHDX(2,2:9)=(V1*(BM(2:9)+AD(2:9)*XA)+A(2,2:9)*VLEN2)*A2
           V1 =VZ1X2
           D2PHDX(3,3:9)=(V1*(BM(3:9)+AD(3:9)*XA)+A(3,3:9)*VLEN2)*A2
           !   Compute last 3 rows of 2nd der. matrix (i.e., terms involving atom
           !   4 coordinates only).  Upper diagonal only.
           V1 = VX3X2*XB
           D2PHDX(10,10)=(V1*BD(10)+B1(1,10)*VLEN2)*B2
           D2PHDX(10,11)=(V1*BD(11)+B1(1,11)*VLEN2)*B2
           D2PHDX(10,12)=(V1*BD(12)+B1(1,12)*VLEN2)*B2
           V1 = VY3X2*XB
           D2PHDX(11,11)=(V1*BD(11)+B1(2,11)*VLEN2)*B2
           D2PHDX(11,12)=(V1*BD(12)+B1(2,12)*VLEN2)*B2
           V1 = VZ3X2*XB
           D2PHDX(12,12)=(V1*BD(12)+B1(3,12)*VLEN2)*B2
           !   Compute cross terms between coordinates of atom 2 (or 3) and atom 4.
           DO I=4,9
              V1 = BM(I) + BD(I)*XB
              D2PHDX(I,10)=(VX3X2*V1+B1(1,I)*VLEN2)*B2
              D2PHDX(I,11)=(VY3X2*V1+B1(2,I)*VLEN2)*B2
              D2PHDX(I,12)=(VZ3X2*V1+B1(3,I)*VLEN2)*B2
           enddo
           !   Compute scalar product of vectr1 and vectr2 and of vectr2 and
           !   vectr3.
           V2I=ONE/VLEN2
           X1=(DOTP12*V2I-VLEN2)*A2
           X2=-(DOTP12*V2I*V2I+ONE)
           X3=-V2I*B2
           X4=-DOTP23*X3*V2I*BI
           XA=-TWO*AI
           XB=TWO*VLEN2
           !   Compute upper diagonal remainder of rows 4-6 (i.e., terms involving
           !   derivatives w.r.t. coordinates of atom 2).
           V1 = VX1X2*XA
           V2 = VX3X2
           V3 = VX1X2*A2
           V4 = VX3X2*X4
           DO J=4,9
              XX=(A(1,J)+V1*AD(J))*X1+V3*(V2I*AB(J)+X2*BM(J)) &
                   +X3*(B1(1,J)*DOTP23+V2*BC(J))
              D2PHDX(4,J)=XX+V4*(XB*BD(J)+VLN3X2*BM(J))
           enddo
           V1 = VY1X2*XA
           V2 = VY3X2
           V3 = VY1X2*A2
           V4 = VY3X2*X4
           DO J=5,9
              XX=(A(2,J)+V1*AD(J))*X1+V3*(V2I*AB(J)+X2*BM(J)) &
                   +X3*(B1(2,J)*DOTP23+V2*BC(J))
              D2PHDX(5,J)=XX+V4*(XB*BD(J)+VLN3X2*BM(J))
           enddo
           V1 = VZ1X2*XA
           V2 = VZ3X2
           V3 = VZ1X2*A2
           V4 = VZ3X2*X4
           DO J=6,9
              XX=(A(3,J)+V1*AD(J))*X1+V3*(V2I*AB(J)+X2*BM(J)) &
                   +X3*(B1(3,J)*DOTP23+V2*BC(J))
              D2PHDX(6,J)=XX+V4*(XB*BD(J)+VLN3X2*BM(J))
           enddo
           !   Compute upper diagonal remainder of rows 7-9 (i.e., terms involving
           !   derivatives w.r.t. coordinates of atom 3).
           XA=-TWO*BI
           X1=(DOTP23*V2I-VLEN2)*B2
           X2=-(DOTP23*V2I*V2I+ONE)
           X3=-V2I*A2
           X4=-DOTP12*X3*V2I*AI
           V1 = XA*VX3X2
           V2 = B2*VX3X2
           V3 = VX1X2
           V4 = X4*VX1X2
           XX=(B1(1,7)+V1*BD(7))*X1+V2*(V2I*BC(7)+X2*BM(7)) &
                +X3*(A(1,7)*DOTP12+V3*AB(7))
           D2PHDX(7,7)=XX+V4*(XB*AD(7)+VLN1X2*BM(7))
           XX=(B1(1,8)+V1*BD(8))*X1+V2*(V2I*BC(8)+X2*BM(8)) &
                +X3*(A(1,8)*DOTP12+V3*AB(8))
           D2PHDX(7,8)=XX+V4*(XB*AD(8)+VLN1X2*BM(8))
           XX=(B1(1,9)+V1*BD(9))*X1+V2*(V2I*BC(9)+X2*BM(9)) &
                +X3*(A(1,9)*DOTP12+V3*AB(9))
           D2PHDX(7,9)=XX+V4*(XB*AD(9)+VLN1X2*BM(9))
           V1 = XA*VY3X2
           V2 = B2*VY3X2
           V3 = VY1X2
           V4 = X4*VY1X2
           XX=(B1(2,8)+V1*BD(8))*X1+V2*(V2I*BC(8)+X2*BM(8)) &
                +X3*(A(2,8)*DOTP12+V3*AB(8))
           D2PHDX(8,8)=XX+V4*(XB*AD(8)+VLN1X2*BM(8))
           XX=(B1(2,9)+V1*BD(9))*X1+V2*(V2I*BC(9)+X2*BM(9)) &
                +X3*(A(2,9)*DOTP12+V3*AB(9))
           D2PHDX(8,9)=XX+V4*(XB*AD(9)+VLN1X2*BM(9))
           V1 = XA*VZ3X2
           V2 = B2*VZ3X2
           V3 = VZ1X2
           V4 = X4*VZ1X2
           XX=(B1(3,9)+V1*BD(9))*X1+V2*(V2I*BC(9)+X2*BM(9)) &
                +X3*(A(3,9)*DOTP12+V3*AB(9))
           D2PHDX(9,9)=XX+V4*(XB*AD(9)+VLN1X2*BM(9))
           !
           RVLN1 = ONE/BNDLN1
           RVLN2 = ONE/BNDLN2
           RVLN3 = ONE/BNDLN3
           TDIAG = (ONE - DBDX1*DBDX1)*RVLN1
           D2BDC1(1,1) = TDIAG
           D2BDC1(4,4) = TDIAG
           D2BDC1(1,4) = -TDIAG
           TDIAG = (ONE - DBDY1*DBDY1)*RVLN1
           D2BDC1(2,2) = TDIAG
           D2BDC1(5,5) = TDIAG
           D2BDC1(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ1*DBDZ1)*RVLN1
           D2BDC1(3,3) = TDIAG
           D2BDC1(6,6) = TDIAG
           D2BDC1(3,6) = -TDIAG
           TDIAG = (ONE - DBDX2*DBDX2)*RVLN2
           D2BDC2(1,1) = TDIAG
           D2BDC2(4,4) = TDIAG
           D2BDC2(1,4) = -TDIAG
           TDIAG = (ONE - DBDY2*DBDY2)*RVLN2
           D2BDC2(2,2) = TDIAG
           D2BDC2(5,5) = TDIAG
           D2BDC2(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ2*DBDZ2)*RVLN2
           D2BDC2(3,3) = TDIAG
           D2BDC2(6,6) = TDIAG
           D2BDC2(3,6) = -TDIAG
           TDIAG = (ONE - DBDX3*DBDX3)*RVLN3
           D2BDC3(1,1) = TDIAG
           D2BDC3(4,4) = TDIAG
           D2BDC3(1,4) = -TDIAG
           TDIAG = (ONE - DBDY3*DBDY3)*RVLN3
           D2BDC3(2,2) = TDIAG
           D2BDC3(5,5) = TDIAG
           D2BDC3(2,5) = -TDIAG
           TDIAG = (ONE - DBDZ3*DBDZ3)*RVLN3
           D2BDC3(3,3) = TDIAG
           D2BDC3(6,6) = TDIAG
           D2BDC3(3,6) = -TDIAG
           !
           !-----now the mixed terms for each atom separately, i.e. x of atom 1
           !     with y of atom 1 etc. in terms of coordinates this is 1,2 and
           !     1,3 and 2,3. this also includes the mixed terms x of atom 1 and
           !     y of atom 2 etc.
           TMIX = -DBDX1*DBDY1*RVLN1
           D2BDC1(1,2) = TMIX
           D2BDC1(4,5) = TMIX
           D2BDC1(1,5) = -TMIX
           TMIX = -DBDX1*DBDZ1*RVLN1
           D2BDC1(1,3) = TMIX
           D2BDC1(4,6) = TMIX
           D2BDC1(1,6) = -TMIX
           TMIX = -DBDY1*DBDZ1*RVLN1
           D2BDC1(2,3) = TMIX
           D2BDC1(5,6) = TMIX
           D2BDC1(2,6) = -TMIX
           TMIX = -DBDX2*DBDY2*RVLN2
           D2BDC2(1,2) = TMIX
           D2BDC2(4,5) = TMIX
           D2BDC2(1,5) = -TMIX
           TMIX = -DBDX2*DBDZ2*RVLN2
           D2BDC2(1,3) = TMIX
           D2BDC2(4,6) = TMIX
           D2BDC2(1,6) = -TMIX
           TMIX = -DBDY2*DBDZ2*RVLN2
           D2BDC2(2,3) = TMIX
           D2BDC2(5,6) = TMIX
           D2BDC2(2,6) = -TMIX
           TMIX = -DBDX3*DBDY3*RVLN3
           D2BDC3(1,2) = TMIX
           D2BDC3(4,5) = TMIX
           D2BDC3(1,5) = -TMIX
           TMIX = -DBDX3*DBDZ3*RVLN3
           D2BDC3(1,3) = TMIX
           D2BDC3(4,6) = TMIX
           D2BDC3(1,6) = -TMIX
           TMIX = -DBDY3*DBDZ3*RVLN3
           D2BDC3(2,3) = TMIX
           D2BDC3(5,6) = TMIX
           D2BDC3(2,6) = -TMIX
           !
           !-----the following terms are the same: y1,z2 and z1,y2, x1,y2 and
           !  y1,x2,
           !     and x1,z2 and z1,x2, as the order of the derivatives does not
           !     matter in this case..
           D2BDC1(3,5) = D2BDC1(2,6)
           D2BDC1(2,4) = D2BDC1(1,5)
           D2BDC1(3,4) = D2BDC1(1,6)
           D2BDC2(3,5) = D2BDC2(2,6)
           D2BDC2(2,4) = D2BDC2(1,5)
           D2BDC2(3,4) = D2BDC2(1,6)
           D2BDC3(3,5) = D2BDC3(2,6)
           D2BDC3(2,4) = D2BDC3(1,5)
           D2BDC3(3,4) = D2BDC3(1,6)
           !
           COSP1 = -COSP1
           COSP2 = -4.0*COSP2
           COSP3 = -9.0*COSP3
           D2NGB1=CBP11(ICPT)*COSP1+CBP12(ICPT)*COSP2+CBP13(ICPT)*COSP3
           D2NGB2=CBP21(ICPT)*COSP1+CBP22(ICPT)*COSP2+CBP23(ICPT)*COSP3
           D2NGB3=CBP31(ICPT)*COSP1+CBP32(ICPT)*COSP2+CBP33(ICPT)*COSP3
           D2NGT1=CTP11(ICPT)*COSP1+CTP12(ICPT)*COSP2+CTP13(ICPT)*COSP3
           D2NGT2=CTP21(ICPT)*COSP1+CTP22(ICPT)*COSP2+CTP23(ICPT)*COSP3
           D2ENG = DIFB1*D2NGB1 + DIFB2*D2NGB2 + DIFB3*D2NGB3 &
                + DTHE1*D2NGT1 + DTHE2*D2NGT2
           DBDC1(1) = DBDX1
           DBDC1(2) = DBDY1
           DBDC1(3) = DBDZ1
           DBDC1(4) =-DBDX1
           DBDC1(5) =-DBDY1
           DBDC1(6) =-DBDZ1
           DBDC2(4) = DBDX2
           DBDC2(5) = DBDY2
           DBDC2(6) = DBDZ2
           DBDC2(7) =-DBDX2
           DBDC2(8) =-DBDY2
           DBDC2(9) =-DBDZ2
           DBDC3(7) = DBDX3
           DBDC3(8) = DBDY3
           DBDC3(9) = DBDZ3
           DBDC3(10)=-DBDX3
           DBDC3(11)=-DBDY3
           DBDC3(12)=-DBDZ3
           !
           !-----compute the first derivatives of the cosine of the angle
           !     or the angle w.r.t. the 2 vectors forming tha angle.
           !     vl1dv2 is the length of vectr1 divided by the length of vectr2.
           !     vl2dv1 is the length of vectr2 divided by the length of vectr1.
           RVLN1 = ONE/BNDLN1
           RVLN2 = ONE/BNDLN2
           RVLN12 = RVLN1*RVLN1
           RVLN22 = RVLN2*RVLN2
           VL1DV2 = BNDLN1*RVLN2
           VL2DV1 = BNDLN2*RVLN1
           !
           DVEC11 = RVLN12*(VL1DV2*VXKJ - COSTH1*VXIJ)
           DVEC12 = RVLN22*(VL2DV1*VXIJ - COSTH1*VXKJ)
           DVEC21 = RVLN12*(VL1DV2*VYKJ - COSTH1*VYIJ)
           DVEC22 = RVLN22*(VL2DV1*VYIJ - COSTH1*VYKJ)
           DVEC31 = RVLN12*(VL1DV2*VZKJ - COSTH1*VZIJ)
           DVEC32 = RVLN22*(VL2DV1*VZIJ - COSTH1*VZKJ)
           !
           !-----compute the derivatives for the angle instead of the cosine of
           !     the angle.
           !     first, compute the sine of the angle (from the cosine)
           !     MAKE SURE the angle is not zero, if it is make it a small
           !     number. (this gives rise to a discontinuity in the derivatives
           !     below the value of small which can cause problems).
           !     make the sign of the sine correspond to the sign of isn.
           SINTH2 = ONE - COSTH1*COSTH1
           IF (ABS(SINTH2)  <  SMALL1) SINTH2 = SMALL1
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           !-----now compute the second derivatives.
           !     icmp1 refers to the components of the first vector,
           !     icmp2 to the components of the second vector, these being
           !     from 4-6 in d2vec.
           !     first, get the 2nd derivatives of the first vector squared
           !     and the second vector squared.
           D2VEC(1,1) = -RVLN12*(TWO*VXIJ*DVEC11 &
                + COSTH1*(ONE - RVLN12*VXIJ*VXIJ))
           D2VEC(4,4) = -RVLN22*(TWO*VXKJ*DVEC12 &
                + COSTH1*(ONE - RVLN22*VXKJ*VXKJ))
           D2VEC(2,2) = -RVLN12*(TWO*VYIJ*DVEC21 &
                + COSTH1*(ONE - RVLN12*VYIJ*VYIJ))
           D2VEC(5,5) = -RVLN22*(TWO*VYKJ*DVEC22 &
                + COSTH1*(ONE - RVLN22*VYKJ*VYKJ))
           D2VEC(3,3) = -RVLN12*(TWO*VZIJ*DVEC31 &
                + COSTH1*(ONE - RVLN12*VZIJ*VZIJ))
           D2VEC(6,6) = -RVLN22*(TWO*VZKJ*DVEC32 &
                + COSTH1*(ONE - RVLN22*VZKJ*VZKJ))
           !
           RVLN13 = RVLN12*RVLN1
           RVLN23 = RVLN22*RVLN2
           RVL123 = RVLN23*RVLN1
           RVL213 = RVLN13*RVLN2
           RVL124 = RVLN12*RVLN22
           !
           !-----now setup the 2nd derivatives for the first and second
           !     vectors, the mixed terms.
           D2VEC(1,4) = -RVLN1*(RVLN23*VXKJ*VXKJ &
                - RVLN2 + VXIJ*DVEC12*RVLN1)
           D2VEC(2,5) = -RVLN1*(RVLN23*VYKJ*VYKJ &
                - RVLN2 + VYIJ*DVEC22*RVLN1)
           D2VEC(3,6) = -RVLN1*(RVLN23*VZKJ*VZKJ &
                - RVLN2 + VZIJ*DVEC32*RVLN1)
           !
           !-----now add in more components of the second derivative mixed terms.
           CRVL4 = COSTH1*RVL124
           CRST = -VXKJ*VYKJ*RVL123 - VXIJ*VYIJ*RVL213
           D2VEC(1,5) = CRST + VXIJ*VYKJ*CRVL4
           D2VEC(2,4) = CRST + VYIJ*VXKJ*CRVL4
           CRST = -VYKJ*VZKJ*RVL123 - VYIJ*VZIJ*RVL213
           D2VEC(2,6) = CRST + VYIJ*VZKJ*CRVL4
           D2VEC(3,5) = CRST + VZIJ*VYKJ*CRVL4
           CRST = -VZKJ*VXKJ*RVL123 - VZIJ*VXIJ*RVL213
           D2VEC(3,4) = CRST + VZIJ*VXKJ*CRVL4
           D2VEC(1,6) = CRST + VXIJ*VZKJ*CRVL4
           !
           !-----more components of the mixed terms.
           D2VEC(1,2) = -RVLN12*(VXIJ*DVEC21 &
                + VYIJ*DVEC11 - COSTH1*VXIJ*VYIJ*RVLN12)
           D2VEC(4,5) = -RVLN22*(VXKJ*DVEC22 &
                + VYKJ*DVEC12 - COSTH1*VXKJ*VYKJ*RVLN22)
           D2VEC(1,3) = -RVLN12*(VXIJ*DVEC31 &
                + VZIJ*DVEC11 - COSTH1*VXIJ*VZIJ*RVLN12)
           D2VEC(4,6) = -RVLN22*(VXKJ*DVEC32 &
                + VZKJ*DVEC12 - COSTH1*VXKJ*VZKJ*RVLN22)
           D2VEC(2,3) = -RVLN12*(VYIJ*DVEC31 &
                + VZIJ*DVEC21 - COSTH1*VYIJ*VZIJ*RVLN12)
           D2VEC(5,6) = -RVLN22*(VYKJ*DVEC32 &
                + VZKJ*DVEC22 - COSTH1*VYKJ*VZKJ*RVLN22)
           !
           CTHST2 = COSTH1/SINTH2
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,1) = D2VEC(1,2)
           D2VEC(6,5) = D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THC1(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THC1(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo
              !
              D2THC1(ICOOR,7) = D2VEC(ICOOR,4)
              D2THC1(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THC1(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THC1(ICOOR,8) = D2VEC(ICOOR,5)
              D2THC1(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THC1(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THC1(ICOOR,9) = D2VEC(ICOOR,6)
              D2THC1(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THC1(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
           enddo

           D2THC1(4,4) = -D2THC1(1,4) - D2THC1(4,7)
           D2THC1(4,5) = -D2THC1(1,5) - D2THC1(4,8)
           D2THC1(4,6) = -D2THC1(1,6) - D2THC1(4,9)
           D2THC1(5,5) = -D2THC1(2,5) - D2THC1(5,8)
           D2THC1(5,6) = -D2THC1(2,6) - D2THC1(5,9)
           D2THC1(6,6) = -D2THC1(3,6) - D2THC1(6,9)
           !
           !-----now add all contributions to the second derivative array
           !     indxdd computes where the element of the second derivative
           !     matrix is to be stored in the linear array dd. this is used
           !     as the second derivative matrix is symmetric and so only one
           !     diagonal half of the mtrix need be stored.
           !
           RVLN1 = ONE/BNDLN2
           RVLN2 = ONE/BNDLN3
           RVLN12 = RVLN1*RVLN1
           RVLN22 = RVLN2*RVLN2
           VL1DV2 = BNDLN2*RVLN2
           VL2DV1 = BNDLN3*RVLN1
           !
           DVEC11 = RVLN12*(-VL1DV2*VXKL + COSTH2*VXKJ)
           DVEC12 = RVLN22*(-VL2DV1*VXKJ + COSTH2*VXKL)
           DVEC21 = RVLN12*(-VL1DV2*VYKL + COSTH2*VYKJ)
           DVEC22 = RVLN22*(-VL2DV1*VYKJ + COSTH2*VYKL)
           DVEC31 = RVLN12*(-VL1DV2*VZKL + COSTH2*VZKJ)
           DVEC32 = RVLN22*(-VL2DV1*VZKJ + COSTH2*VZKL)
           !
           !-----now compute the second derivatives.
           !     icmp1 refers to the components of the first vector,
           !     icmp2 to the components of the second vector, these being
           !     from 4-6 in d2vec.
           !     first, get the 2nd derivatives of the first vector squared
           !     and the second vector squared.
           D2VEC(1,1) = RVLN12*(TWO*VXKJ*DVEC11 &
                - COSTH2*(ONE - RVLN12*VXKJ*VXKJ))
           D2VEC(4,4) = RVLN22*(TWO*VXKL*DVEC12 &
                - COSTH2*(ONE - RVLN22*VXKL*VXKL))
           D2VEC(2,2) = RVLN12*(TWO*VYKJ*DVEC21 &
                - COSTH2*(ONE - RVLN12*VYKJ*VYKJ))
           D2VEC(5,5) = RVLN22*(TWO*VYKL*DVEC22 &
                - COSTH2*(ONE - RVLN22*VYKL*VYKL))
           D2VEC(3,3) = RVLN12*(TWO*VZKJ*DVEC31 &
                - COSTH2*(ONE - RVLN12*VZKJ*VZKJ))
           D2VEC(6,6) = RVLN22*(TWO*VZKL*DVEC32 &
                - COSTH2*(ONE - RVLN22*VZKL*VZKL))
           !
           RVLN13 = RVLN12*RVLN1
           RVLN23 = RVLN22*RVLN2
           RVL123 = RVLN23*RVLN1
           RVL213 = RVLN13*RVLN2
           RVL124 = RVLN12*RVLN22
           !
           !-----now setup the 2nd derivatives for the first and second
           !     vectors, the mixed terms.
           D2VEC(1,4) = -RVLN1*(RVLN23*VXKL*VXKL &
                - RVLN2 - VXKJ*DVEC12*RVLN1)
           D2VEC(2,5) = -RVLN1*(RVLN23*VYKL*VYKL &
                - RVLN2 - VYKJ*DVEC22*RVLN1)
           D2VEC(3,6) = -RVLN1*(RVLN23*VZKL*VZKL &
                - RVLN2 - VZKJ*DVEC32*RVLN1)
           !
           !-----now add in more components of the second derivative mixed terms.
           CRVL4 = COSTH2*RVL124
           CRST = -VXKL*VYKL*RVL123 - VXKJ*VYKJ*RVL213
           D2VEC(1,5) = CRST + VXKJ*VYKL*CRVL4
           D2VEC(2,4) = CRST + VYKJ*VXKL*CRVL4
           CRST = -VYKL*VZKL*RVL123 - VYKJ*VZKJ*RVL213
           D2VEC(2,6) = CRST + VYKJ*VZKL*CRVL4
           D2VEC(3,5) = CRST + VZKJ*VYKL*CRVL4
           CRST = -VZKL*VXKL*RVL123 - VZKJ*VXKJ*RVL213
           D2VEC(3,4) = CRST + VZKJ*VXKL*CRVL4
           D2VEC(1,6) = CRST + VXKJ*VZKL*CRVL4
           !
           !-----more components of the mixed terms.
           D2VEC(1,2) = RVLN12*(VXKJ*DVEC21 &
                + VYKJ*DVEC11 + COSTH2*VXKJ*VYKJ*RVLN12)
           D2VEC(4,5) = RVLN22*(VXKL*DVEC22 &
                + VYKL*DVEC12 + COSTH2*VXKL*VYKL*RVLN22)
           D2VEC(1,3) = RVLN12*(VXKJ*DVEC31 &
                + VZKJ*DVEC11 + COSTH2*VXKJ*VZKJ*RVLN12)
           D2VEC(4,6) = RVLN22*(VXKL*DVEC32 &
                + VZKL*DVEC12 + COSTH2*VXKL*VZKL*RVLN22)
           D2VEC(2,3) = RVLN12*(VYKJ*DVEC31 &
                + VZKJ*DVEC21 + COSTH2*VYKJ*VZKJ*RVLN12)
           D2VEC(5,6) = RVLN22*(VYKL*DVEC32 &
                + VZKL*DVEC22 + COSTH2*VYKL*VZKL*RVLN22)
           !
           !-----compute the derivatives for the angle instead of the cosine of
           !     the angle.
           !     first, compute the sine of the angle (from the cosine)
           !     MAKE SURE the angle is not zero, if it is make it a small
           !     number. (this gives rise to a discontinuity in the derivatives
           !     below the value of small which can cause problems).
           !     make the sign of the sine correspond to the sign of isn.
           SINTH2 = ONE - COSTH2*COSTH2
           IF (ABS(SINTH2)  <  SMALL1) SINTH2 = SMALL1
           SINTHE = SQRT(SINTH2)
           !
           RSNTHE = ONE/SINTHE
           !
           CTHST2 = COSTH2/SINTH2
           !
           D2VEC(1,1) = -(D2VEC(1,1)+DVEC11*DVEC11*CTHST2)*RSNTHE
           D2VEC(1,2) = -(D2VEC(1,2)+DVEC11*DVEC21*CTHST2)*RSNTHE
           D2VEC(1,3) = -(D2VEC(1,3)+DVEC11*DVEC31*CTHST2)*RSNTHE
           D2VEC(1,4) = -(D2VEC(1,4)+DVEC11*DVEC12*CTHST2)*RSNTHE
           D2VEC(1,5) = -(D2VEC(1,5)+DVEC11*DVEC22*CTHST2)*RSNTHE
           D2VEC(1,6) = -(D2VEC(1,6)+DVEC11*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,2) = -(D2VEC(2,2)+DVEC21*DVEC21*CTHST2)*RSNTHE
           D2VEC(2,3) = -(D2VEC(2,3)+DVEC21*DVEC31*CTHST2)*RSNTHE
           D2VEC(2,4) = -(D2VEC(2,4)+DVEC21*DVEC12*CTHST2)*RSNTHE
           D2VEC(2,5) = -(D2VEC(2,5)+DVEC21*DVEC22*CTHST2)*RSNTHE
           D2VEC(2,6) = -(D2VEC(2,6)+DVEC21*DVEC32*CTHST2)*RSNTHE
           D2VEC(3,3) = -(D2VEC(3,3)+DVEC31*DVEC31*CTHST2)*RSNTHE
           D2VEC(3,4) = -(D2VEC(3,4)+DVEC31*DVEC12*CTHST2)*RSNTHE
           D2VEC(3,5) = -(D2VEC(3,5)+DVEC31*DVEC22*CTHST2)*RSNTHE
           D2VEC(3,6) = -(D2VEC(3,6)+DVEC31*DVEC32*CTHST2)*RSNTHE
           D2VEC(4,4) = -(D2VEC(4,4)+DVEC12*DVEC12*CTHST2)*RSNTHE
           D2VEC(4,5) = -(D2VEC(4,5)+DVEC12*DVEC22*CTHST2)*RSNTHE
           D2VEC(4,6) = -(D2VEC(4,6)+DVEC12*DVEC32*CTHST2)*RSNTHE
           D2VEC(5,5) = -(D2VEC(5,5)+DVEC22*DVEC22*CTHST2)*RSNTHE
           D2VEC(5,6) = -(D2VEC(5,6)+DVEC22*DVEC32*CTHST2)*RSNTHE
           D2VEC(6,6) = -(D2VEC(6,6)+DVEC32*DVEC32*CTHST2)*RSNTHE
           D2VEC(2,1) = D2VEC(1,2)
           D2VEC(6,5) = D2VEC(5,6)
           DO ICOOR=1,3
              DO JCOOR=ICOOR,3
                 D2THC2(ICOOR,JCOOR) = D2VEC(ICOOR,JCOOR)
                 D2THC2(6+ICOOR,6+JCOOR) = D2VEC(3+ICOOR,3+JCOOR)
              enddo

              D2THC2(ICOOR,7) = D2VEC(ICOOR,4)
              D2THC2(ICOOR,4) = -D2VEC(1,ICOOR) - D2VEC(ICOOR,4)
              D2THC2(3+ICOOR,7) = -D2VEC(1,3+ICOOR) - D2VEC(4,3+ICOOR)
              D2THC2(ICOOR,8) = D2VEC(ICOOR,5)
              D2THC2(ICOOR,5) = -D2VEC(2,ICOOR) - D2VEC(ICOOR,5)
              D2THC2(3+ICOOR,8) = -D2VEC(2,3+ICOOR) - D2VEC(3+ICOOR,5)
              D2THC2(ICOOR,9) = D2VEC(ICOOR,6)
              D2THC2(ICOOR,6) = -D2VEC(ICOOR,3) - D2VEC(ICOOR,6)
              D2THC2(3+ICOOR,9) = -D2VEC(3,3+ICOOR) - D2VEC(3+ICOOR,6)
           enddo

           D2THC2(4,4) = -D2THC2(1,4) - D2THC2(4,7)
           D2THC2(4,5) = -D2THC2(1,5) - D2THC2(4,8)
           D2THC2(4,6) = -D2THC2(1,6) - D2THC2(4,9)
           D2THC2(5,5) = -D2THC2(2,5) - D2THC2(5,8)
           D2THC2(5,6) = -D2THC2(2,6) - D2THC2(5,9)
           D2THC2(6,6) = -D2THC2(3,6) - D2THC2(6,9)
           !
           D1TTPX = CSINPH*DTHE2
           D2TTPX = CSINPH*DTHE1
           D3TTPX = -CTTP*COSP1
           IATOM = (IPHI1 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (IPHI2 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (IPHI3 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3
           IATOM = (IPHI4 - 1)*3
           LC(10) = IATOM + 1
           LC(11) = IATOM + 2
           LC(12) = IATOM + 3
           !
           DO ICOOR=1,12
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) &
                      + D2PHDX(JCOOR,ICOOR)*(DTTPDP+DENG+DPHI) &
                      + DPHIC(ICOOR)*DPHIC(JCOOR)*(D2TTP+D2ENG+D2PHI) &
                      + (D1TTPX+DENGT1)*(DPHIC(ICOOR)*DTHEC1(JCOOR) &
                      + DPHIC(JCOOR)*DTHEC1(ICOOR)) &
                      + (D2TTPX+DENGT2)*(DPHIC(ICOOR)*DTHEC2(JCOOR) &
                      + DPHIC(JCOOR)*DTHEC2(ICOOR)) &
                      + D3TTPX*(DTHEC1(ICOOR)*DTHEC2(JCOOR) &
                      + DTHEC1(JCOOR)*DTHEC2(ICOOR)) &
                      + DENGB2*(DPHIC(ICOOR)*DBDC2(JCOOR) &
                      + DPHIC(JCOOR)*DBDC2(ICOOR))
              enddo
           enddo

           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2THC1(JCOOR,ICOOR)*DTTPDT
              enddo
           enddo
           !
           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) &
                      + D2BDC1(JCOOR,ICOOR)*ENGB1 &
                      + DENGB1*(DPHIC(ICOOR)*DBDC1(JCOOR) &
                      + DPHIC(JCOOR)*DBDC1(ICOOR))
              enddo
           enddo

           DO ICOOR=7,12
              LI = LC(ICOOR)
              AA = DENGB1*DPHIC(ICOOR)
              AC = DENGB3*DBDC3(ICOOR)
              DO JCOOR=1,6
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + AA*DBDC1(JCOOR)+AC*DPHIC(JCOOR)
              enddo
              DO JCOOR=7,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) &
                      + D2BDC3(JCOOR-6,ICOOR-6)*ENGB3 &
                      + DENGB3*(DPHIC(ICOOR)*DBDC3(JCOOR) &
                      + DPHIC(JCOOR)*DBDC3(ICOOR))
              enddo
           enddo
           !
           IATOM = (IPHI2 - 1)*3
           LC(1) = IATOM + 1
           LC(2) = IATOM + 2
           LC(3) = IATOM + 3
           IATOM = (IPHI3 - 1)*3
           LC(4) = IATOM + 1
           LC(5) = IATOM + 2
           LC(6) = IATOM + 3
           IATOM = (IPHI4 - 1)*3
           LC(7) = IATOM + 1
           LC(8) = IATOM + 2
           LC(9) = IATOM + 3

           DO ICOOR=1,6
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2BDC2(JCOOR,ICOOR)*ENGB2
              enddo
           enddo

           DO ICOOR=1,9
              LI = LC(ICOOR)
              DO JCOOR=1,ICOOR
                 LJ = LC(JCOOR)
                 IF (LJ >= LI) THEN
                    INDEX = (LI-1)*NUMAT3 - LI*(LI-1)/2 + LJ
                 ELSE
                    INDEX = (LJ-1)*NUMAT3 - LJ*(LJ-1)/2 + LI
                 ENDIF
                 DD(INDEX) = DD(INDEX) + D2THC2(JCOOR,ICOOR)*DTTPDA
              enddo
           enddo
        ENDIF
     ENDIF
  ENDDO
  return
end SUBROUTINE EPHI_CFF

#endif 

SUBROUTINE NULL_escalar_CFF
  RETURN
END SUBROUTINE NULL_escalar_CFF

