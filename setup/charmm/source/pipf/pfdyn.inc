  !   MAIN LOOP BEGIN
  DO I=IFRSTA,NATOM
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) THEN

        !         I1=ITC(IAC(I))
        !         IACI=IOFF(I1)

        !
        ! not sure about thie section (group-group?)
        !         IF (ELECFG) THEN
        !           CGT=CGF*CG(I)
        !           ELCFG=(CGT.NE.0.0)
        !         ELSE
        !           CGT=ZERO
        !         ENDIF
        !

        ! CHARGE FACTOR I
        CGI = CG(I)*CCELEC
        I1=ITC(IAC(I))
        !
        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNB(NB)

              !             J1=ITC(IAC(J))
              !             IF (I1.LT.J1) THEN
              !               IC=IOFF(J1)+I1+MAXROW
              !             ELSE
              !               IC=IACI+J1+MAXROW
              !             ENDIF

           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNB(NB)

              !             J1=ITC(IAC(J))
              !             IF (I1.LT.J1) THEN
              !               IC=IOFF(J1)+I1
              !             ELSE
              !               IC=IACI+J1
              !             ENDIF

           ENDIF

           !- debug
           !           if (natom .le. npfpr) then
           !              write(*,*) '--- P-P pair: I,J = ', i, j
           !           else
           !              write(*,*) '--- P-I pair: I,J = ', i, j
           !           end if

           !      CHARGE FACTOR FOR J
           CGJ = CG(J)*CCELEC
           J1=ITC(IAC(J))

           XX = X(I)-X(J)
           YY = Y(I)-Y(J)
           ZZ = Z(I)-Z(J)

#if KEY_PBOUND==1 /*pbound*/
           If(qBoun) then
              If(qCUBoun.or.qTOBoun) then
                 XX = BOXINV * XX
                 YY = BOYINV * YY
                 ZZ = BOZINV * ZZ
                 xx = xx - nint(xx)
                 yy = yy - nint(yy)
                 zz = zz - nint(zz)
!!$                 IF(XX.GT.  HALF) XX = XX - ONE
!!$                 IF(XX.LT. -HALF) XX = XX + ONE
!!$                 IF(YY.GT.  HALF) YY = YY - ONE
!!$                 IF(YY.LT. -HALF) YY = YY + ONE
!!$                 IF(ZZ.GT.  HALF) ZZ = ZZ - ONE
!!$                 IF(ZZ.LT. -HALF) ZZ = ZZ + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(XX) + &
                         ABS(YY) + &
                         ABS(ZZ)))
                    XX = XX - SIGN( CORR,  XX  )
                    YY = YY - SIGN( CORR,  YY  )
                    ZZ = ZZ - SIGN( CORR,  ZZ  )
                 Endif
                 XX = XSIZE * XX
                 YY = YSIZE * YY
                 ZZ = ZSIZE * ZZ
              Else
                 Call PBMove(XX, YY, ZZ)
              Endif
           Endif
#endif /*      (pbound)*/

           RIJ2 = XX*XX+YY*YY+ZZ*ZZ

#ifdef PIPF_CTOF  /* ecut1 */
           !
           !      APPLY AN ENERGY CUT-OFF
           !
           IF (RIJ2 .LT. PFCTOF2) THEN
#endif  /* ecut1 */

              !
              !           R2 = 1.D+00/(XX*XX+YY*YY+ZZ*ZZ)
              !
              R2 = 1.D+00/RIJ2
              R1 = SQRT(R2)
              R3 = R1*R2
              !
              !      PERMENENT ELECTRIC FIELD
              !      E0 FOLLOW EQ. (5) in CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !               E^i = - sum(T^iA Q^A) and T^iA = - a/R^3 (a denotes a^i-a^j)
              !                a       A   a             a
#ifndef PIPF_DAMP  /* damp1 */
              !       NO DAMPING (DEFAULT)
                 QR3I = CG(I)*R3
                 QR3J = CG(J)*R3
#else  /* damp2 */
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD3 = 1.0D0 - EXP(-AU3)
                 QR3I = CG(I)*R3*FLMD3
                 QR3J = CG(J)*R3*FLMD3
#endif  /* damp2 */

              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION, DO NOTHING
              ELSE
                 EFIELD(1,I) = EFIELD(1,I)+XX*QR3J
                 EFIELD(2,I) = EFIELD(2,I)+YY*QR3J
                 EFIELD(3,I) = EFIELD(3,I)+ZZ*QR3J
                 EFIELD(1,J) = EFIELD(1,J)-XX*QR3I
                 EFIELD(2,J) = EFIELD(2,J)-YY*QR3I
                 EFIELD(3,J) = EFIELD(3,J)-ZZ*QR3I
              ENDIF
              !
              !      COMPUTE THE INDUCED FIELD
              !      SINCE WE USE NONBONDED PAIR (I<J), WE NEED TO INCLUDE
              !      CONTRIBUTION TO J FOR I>J PAIR EXPLICITLY
              !
              !      DIPOLE TENSOR
              !
              !         SIGN CONVENTION FLOLLOWS THE DEFINITION AS EQ. (6)
              !         DESCRIBED IN CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !              T^ij  = d^i d^i (|Ri-Rj|^-1)
              !               ab      a   b

#ifndef PIPF_DAMP  /* damp3 */
              !        NO DAMPING (DEFAULT)
                 RR3 = 3.0D+00*R3*R2
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
#else  /* damp4 */
                 !        OR APPLY A DAMPING FACTOR
                 !        (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
                 RR3 = 3.0D+00*R3*R2*FLMD5
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
                 R3 = R3 * FLMD3
#endif  /* damp4 */
              !         ACCORDING TO THIS CONVENTION, WE HAVE:
              !
              !              T^ij  = - T^ji,   T^ij = T^ji , and T^ij = -T^ji
              !               a         a       ab     ab         abc     abc
              !
              !         AND SO ON (OPPSITE SIGN FOR ODD ORDER OF DIFFERENTIATION)
              !         WE MAP TXWWT TO A 2-D ARRAY, ONLY HAVE 6xNB TERMS.
              !         NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION, EXPLICITLY ZERO OUT TXWW
                 TXWW(1) = 0.0D0
                 TXWW(2) = 0.0D0
                 TXWW(3) = 0.0D0
                 TXWW(4) = 0.0D0
                 TXWW(5) = 0.0D0
                 TXWW(6) = 0.0D0
              ELSE
                 TXWW(1) = RR3XX*XX-R3
                 TXWW(2) = RR3XX*YY
                 TXWW(3) = RR3YY*YY-R3
                 TXWW(4) = RR3XX*ZZ
                 TXWW(5) = RR3YY*ZZ
                 TXWW(6) = RR3ZZ*ZZ-R3
              ENDIF
              !
              ! INDUCED ELECTRIC FIELD
              !
              EIND(1,I) = EIND(1,I)+TXWW(1)*UIND(1,J) &
                   +TXWW(2)*UIND(2,J)+TXWW(4)*UIND(3,J)
              EIND(2,I) = EIND(2,I)+TXWW(2)*UIND(1,J) &
                   +TXWW(3)*UIND(2,J)+TXWW(5)*UIND(3,J)
              EIND(3,I) = EIND(3,I)+TXWW(4)*UIND(1,J) &
                   +TXWW(5)*UIND(2,J)+TXWW(6)*UIND(3,J)

              EIND(1,J) = EIND(1,J)+TXWW(1)*UIND(1,I) &
                   +TXWW(2)*UIND(2,I)+TXWW(4)*UIND(3,I)
              EIND(2,J) = EIND(2,J)+TXWW(2)*UIND(1,I) &
                   +TXWW(3)*UIND(2,I)+TXWW(5)*UIND(3,I)
              EIND(3,J) = EIND(3,J)+TXWW(4)*UIND(1,I) &
                   +TXWW(5)*UIND(2,I)+TXWW(6)*UIND(3,I)
              !
              !      WE COMPUTE THE NUCLEAR GRADIENT ASSOCIATED WITH THE INDUCTION
              !      ENERGY (see dpipf for details of these three terms):
              !
              !          (I) + (II): permenent field gradient
              !          (III)     : induced filed gradient
              !
              !      (I) Q^K * PI^K   = Q^K * sum (T^KA u^A)     (r denotes lamda)
              !                  r             A    rb   b
              !
              DX(I) = DX(I) - CGI*(TXWW(1)*UIND(1,J) &
                   +TXWW(2)*UIND(2,J)+TXWW(4)*UIND(3,J))
              DY(I) = DY(I) - CGI*(TXWW(2)*UIND(1,J) &
                   +TXWW(3)*UIND(2,J)+TXWW(5)*UIND(3,J))
              DZ(I) = DZ(I) - CGI*(TXWW(4)*UIND(1,J) &
                   +TXWW(5)*UIND(2,J)+TXWW(6)*UIND(3,J))

              DX(J) = DX(J) - CGJ*(TXWW(1)*UIND(1,I) &
                   +TXWW(2)*UIND(2,I)+TXWW(4)*UIND(3,I))
              DY(J) = DY(J) - CGJ*(TXWW(2)*UIND(1,I) &
                   +TXWW(3)*UIND(2,I)+TXWW(5)*UIND(3,I))
              DZ(J) = DZ(J) - CGJ*(TXWW(4)*UIND(1,I) &
                   +TXWW(5)*UIND(2,I)+TXWW(6)*UIND(3,I))
              !
              !     (II) u^K * E^K   = u^K * [- sum (T^KA Q^A)]    (r denotes lamda)
              !           a     ar      a        A    ar

              DX(I) = DX(I) + CGJ*(TXWW(1)*UIND(1,I) &
                   +TXWW(2)*UIND(2,I)+TXWW(4)*UIND(3,I))
              DY(I) = DY(I) + CGJ*(TXWW(2)*UIND(1,I) &
                   +TXWW(3)*UIND(2,I)+TXWW(5)*UIND(3,I))
              DZ(I) = DZ(I) + CGJ*(TXWW(4)*UIND(1,I) &
                   +TXWW(5)*UIND(2,I)+TXWW(6)*UIND(3,I))

              DX(J) = DX(J) + CGI*(TXWW(1)*UIND(1,J) &
                   +TXWW(2)*UIND(2,J)+TXWW(4)*UIND(3,J))
              DY(J) = DY(J) + CGI*(TXWW(2)*UIND(1,J) &
                   +TXWW(3)*UIND(2,J)+TXWW(5)*UIND(3,J))
              DZ(J) = DZ(J) + CGI*(TXWW(4)*UIND(1,J) &
                   +TXWW(5)*UIND(2,J)+TXWW(6)*UIND(3,J))
              !
              !    (III) nuclear gradient due to induced field (NOT sure for
              !          unconverged dipole this is still the correct formular)
              !
              !      NOTE: the nuclear force share the same form no matter
              !            whether the dipole is induced (SCF determined) or
              !            present as physical dipole in the sysetm (dynamical
              !            determined, so we should be able to use the same formula)
              !
              !      Conclusion: the nuclear force now can be tested using TSFR
              !                  option for PIPF DYNA. The numerical results
              !                  show the validity of these analytical evaluations.
              !
              R2 = 5.0D0*R1*R1
              R5 = 3.0D0*CCELEC*R1**5.0D0

              !
              ! OPTIONALLY EXCLUDE 1-4 POLARIZATION BY ZERO OUT COMMEN FACTOR R5.
              ! NO NEED FOR HANDLING FIRST TWO TERMS, SINCE THE COMMON FACTOR TXWW
              ! HAS BEEN CONSISTENT EXCLUSED IN ETINIT FOR 1-4 PAIRS.
              !
              IF (QPFEX .AND. JNB(NB).LT.0) R5 = 0.0D0

              DPR1 = UIND(1,I)*XX + UIND(2,I)*YY + UIND(3,I)*ZZ
              DPR2 = UIND(1,J)*XX + UIND(2,J)*YY + UIND(3,J)*ZZ
              DPR3 = UIND(1,I)*UIND(1,J) + UIND(2,I)*UIND(2,J) +    &
                   UIND(3,I)*UIND(3,J)

#ifndef PIPF_DAMP  /* damp5 */
              !      NO DAMPING (DEFAULT)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*XX - &
                      DPR3*XX - DPR2*UIND(1,I) - DPR1*UIND(1,J))
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*YY - &
                      DPR3*YY - DPR2*UIND(2,I) - DPR1*UIND(2,J))
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*ZZ - &
                      DPR3*ZZ - DPR2*UIND(3,I) - DPR1*UIND(3,J))

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*XX - &
                      DPR3*XX - DPR1*UIND(1,J) - DPR2*UIND(1,I))
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*YY - &
                      DPR3*YY - DPR1*UIND(2,J) - DPR2*UIND(2,I))
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*ZZ - &
                      DPR3*ZZ - DPR1*UIND(3,J) - DPR2*UIND(3,I))
#else  /* damp6 */
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 FLMD7 = 1.0D0 - (1.0D0+AU3+0.6D0*AU3*AU3)*EXP(-AU3)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*XX*FLMD7 - &
                      (DPR3*XX+DPR2*UIND(1,I)+DPR1*UIND(1,J))*FLMD5)
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*YY*FLMD7 - &
                      (DPR3*YY+DPR2*UIND(2,I)+DPR1*UIND(2,J))*FLMD5)
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*ZZ*FLMD7 - &
                      (DPR3*ZZ+DPR2*UIND(3,I)+DPR1*UIND(3,J))*FLMD5)

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*XX*FLMD7 - &
                      (DPR3*XX+DPR1*UIND(1,J)+DPR2*UIND(1,I))*FLMD5)
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*YY*FLMD7 - &
                      (DPR3*YY+DPR1*UIND(2,J)+DPR2*UIND(2,I))*FLMD5)
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*ZZ*FLMD7 - &
                      (DPR3*ZZ+DPR1*UIND(3,J)+DPR2*UIND(3,I))*FLMD5)
#endif  /* damp6 */
              !       MARK J AS A DIPOLE CENTER
              IPOL(J) = 1
#ifdef PIPF_CTOF  /* ecut2 */
              ! END THE ENERGY CUT-OFF
           END IF
#endif  /* ecut2 */
        ENDDO ! END JPR LOOP
        IPOL(I) = 1  !       MARK I AS A DIPOLE CENTER
     ENDIF
  ENDDO ! END MAIN LOOP
