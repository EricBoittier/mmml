  !   MAIN LOOP BEGIN
  DO I=IFRSTA,NATOM - 1
     I1=ITC(IAC(I))
     DO J=I + 1,NATOM
        NB = NB + 1
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
!!$              IF(XX.GT.  HALF) XX = XX - ONE
!!$              IF(XX.LT. -HALF) XX = XX + ONE
!!$              IF(YY.GT.  HALF) YY = YY - ONE
!!$              IF(YY.LT. -HALF) YY = YY + ONE
!!$              IF(ZZ.GT.  HALF) ZZ = ZZ - ONE
!!$              IF(ZZ.LT. -HALF) ZZ = ZZ + ONE
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
           R2 = 1.D+00/RIJ2
           R1 = SQRT(R2)
           R3 = R1*R2
#ifndef PIPF_DAMP  /* damp1 */
           !
           !       NO DAMPING (DEFAULT)
           !
#else /* damp2 */
              !
              !       OR APPLY A DAMPING FACTOR
              !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
              U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
              AU3 = DPFAC*U3
              FLMD3 = 1.0D0 - EXP(-AU3)
              FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
#endif  /* damp1 damp2 */
           
#ifndef PIPF_DAMP  /* damp3 */
           !
           !        NO DAMPING (DEFAULT)
              RR3 = 3.0D+00*R3*R2
              RR3XX = RR3*XX
              RR3YY = RR3*YY
              RR3ZZ = RR3*ZZ
#else  /* damp4 */
              !
              !        OR APPLY A DAMPING FACTOR
              !        (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
              RR3 = 3.0D+00*R3*R2*FLMD5
              RR3XX = RR3*XX
              RR3YY = RR3*YY
              RR3ZZ = RR3*ZZ
              R3 = R3 * FLMD3
#endif  /* damp2 damp4 */
           !         ACCORDING TO THIS CONVENTION, WE HAVE:
           !
           !              T^ij  = - T^ji,   T^ij = T^ji , and T^ij = -T^ji
           !               a         a       ab     ab         abc     abc
           !
           TTSR(1,NB) = RR3XX*XX-R3
           TTSR(2,NB) = RR3XX*YY
           TTSR(3,NB) = RR3YY*YY-R3
           TTSR(4,NB) = RR3XX*ZZ
           TTSR(5,NB) = RR3YY*ZZ
           TTSR(6,NB) = RR3ZZ*ZZ-R3
           !
#ifdef PIPF_CTOF  /* ecut2 */
           ! END ENERGY THE CUT-OFF
        END IF
#endif /* ecut2 */
     ENDDO
  ENDDO
