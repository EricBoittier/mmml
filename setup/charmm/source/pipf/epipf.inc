  !   MAIN LOOP BEGIN
  DO I=IFRSTA,NATOM
     I1=ITC(IAC(I))
     NPR=INBLO(I)-ITEMP
     ITEMP=INBLO(I)
     IF(NPR.GT.0) THEN
        DO JPR=1,NPR
           NB=NB+1
           IF (JNB(NB).LT.0) THEN
              J=-JNB(NB)
           ELSE
              J=JNB(NB)
           ENDIF

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
#endif /* (ecut1)*/

              !
              !           R2 = 1.D+00/(XX*XX+YY*YY+ZZ*ZZ)
              !
              R2 = 1.D+00/RIJ2
              R1 = SQRT(R2)
              R3 = R1*R2
              !
              !      0TH-ORDER ELECTRIC FIELD 
              !      FOLLOW EQ. (5) in CHEM. PHYS. LETT. 1990, 166, 180.
              !
              !               E^i = - sum(T^iA Q^A) and T^iA = - a/R^3 (a denotes a^i-a^j)
              !                a       A   a             a
              !
#ifndef PIPF_DAMP  /* damp1 */
              !
              !       NO DAMPING (DEFAULT)
              !
                 QR3I = CG(I)*R3
                 QR3J = CG(J)*R3
#else  /* damp2 */
                 !
                 !       OR APPLY A DAMPING FACTOR 
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 !
                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD3 = 1.0D0 - EXP(-AU3)
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)   
                 QR3I = CG(I)*R3*FLMD3
                 QR3J = CG(J)*R3*FLMD3
#endif /* (damp2)*/
              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 !
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION, DO NOTHING
                 !
              ELSE
                 EZERO(1,I) = EZERO(1,I)+XX*QR3J
                 EZERO(2,I) = EZERO(2,I)+YY*QR3J
                 EZERO(3,I) = EZERO(3,I)+ZZ*QR3J
                 EZERO(1,J) = EZERO(1,J)-XX*QR3I
                 EZERO(2,J) = EZERO(2,J)-YY*QR3I
                 EZERO(3,J) = EZERO(3,J)-ZZ*QR3I
              END IF
              !
              !      DIPOLE TENSOR 
              !
              !         SIGN CONVENTION FLOLLOWS THE DEFINITION AS EQ. (6) 
              !         DESCRIBED IN CHEM. PHYS. LETT. 1990, 166, 180. 
              !
              !              T^ij  = d^i d^i (|Ri-Rj|^-1)
              !               ab      a   b
#ifndef PIPF_DAMP  /* damp3 */
              !
              !        NO DAMPING (DEFAULT)
              !
                 RR3 = 3.0D+00*R3*R2
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
#else  /* damp4 */
                 !
                 !        OR APPLY A DAMPING FACTOR
                 !        (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 !
                 RR3 = 3.0D+00*R3*R2*FLMD5
                 RR3XX = RR3*XX
                 RR3YY = RR3*YY
                 RR3ZZ = RR3*ZZ
                 R3 = R3 * FLMD3
#endif /* (damp4)*/
              !         ACCORDING TO THIS CONVENTION, WE HAVE:
              !
              !              T^ij  = - T^ji,   T^ij = T^ji , and T^ij = -T^ji
              !               a         a       ab     ab         abc     abc
              !
              !         AND SO ON (OPPSITE SIGN FOR ODD ORDER OF DIFFERENTIATION)
              !         WE MAP TXWWT TO A 2-D ARRAY, ONLY HAVE 6xNB TERMS.
              !         NOTE:  TXWW(1:6,NB) = TXWW(1:6,I,J) = TXWW(1:6,J,I)
              !
              IF (QPFEX .AND. JNB(NB).LT.0) THEN
                 ! OPTIONALLY EXCLUDE 1-4 POLARIZATION
                 TXWWT(1,NB) = 0.0D0
                 TXWWT(2,NB) = 0.0D0
                 TXWWT(3,NB) = 0.0D0
                 TXWWT(4,NB) = 0.0D0
                 TXWWT(5,NB) = 0.0D0
                 TXWWT(6,NB) = 0.0D0
              ELSE
                 TXWWT(1,NB) = RR3XX*XX-R3
                 TXWWT(2,NB) = RR3XX*YY
                 TXWWT(3,NB) = RR3YY*YY-R3
                 TXWWT(4,NB) = RR3XX*ZZ
                 TXWWT(5,NB) = RR3YY*ZZ
                 TXWWT(6,NB) = RR3ZZ*ZZ-R3
              ENDIF
              !
              !     IF (WIX.GE.RL2) SSPOL = SSPOL*RUL*(RU2-WIX)
              !
#ifdef PIPF_CTOF  /* ecut2 */
              !
              ! END ENERGY THE CUT-OFF
              !
           END IF
#endif /* (ecut2)*/
        ENDDO
     ENDIF
  ENDDO
