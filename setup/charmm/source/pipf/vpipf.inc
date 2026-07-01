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

           IDXTSL = NATOM*(I-1)+J 
           IDXTSU = NATOM*(J-1)+I 

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
              R5 = R2*R3
              R7 = R2*R5
              R9 = R2*R7
#ifndef PIPF_DAMP  /* damp1 */
              !       NO DAMPING (DEFAULT)
                 FLMD3 = 1.0D0
                 FLMD5 = 1.0D0
                 FLMD7 = 1.0D0
                 FLMD9 = 1.0D0
#else  /* damp2 */
                 !       OR APPLY A DAMPING FACTOR 
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 EXPAU3 = EXP(-AU3)
                 A2U6 = AU3*AU3
                 A3U9 = A2U6*AU3
                 FLMD3 = 1.0D0 - EXPAU3
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXPAU3 
                 FLMD7 = 1.0D0 - (1.0D0+AU3+0.6D0*A2U6)*EXPAU3 
                 FLMD9 = 1.0D0 - (1.0D0+AU3+(18.0D0*A2U6+9.0D0*A3U9)/ &
                      35.0D0)*EXPAU3 
                 !              FLMD9 = 1.0D0 - ((18.0DO*A2U6+9.0D0*A3U9)/
                 !    &                        35.0D0)*EXPAU3 
                 !              QR3I = CG(I)*R3*FLMD3
                 !              QR3J = CG(J)*R3*FLMD3
#endif /* (damp2)*/
              DO K1 = 1,3
                 IF (K1.EQ.1) THEN 
                    RK1 = XX
                 ELSEIF (K1.EQ.2) THEN
                    RK1 = YY
                 ELSE
                    RK1 = ZZ
                 ENDIF
                 DO K2 = 1,3
                    IF (K2.EQ.1) THEN 
                       RK2 = XX
                    ELSEIF (K2.EQ.2) THEN
                       RK2 = YY
                    ELSE
                       RK2 = ZZ
                    ENDIF
                    IF (K2.EQ.K1) THEN
                       DTA1 = 1.0D0
                    ELSE 
                       DTA1 = 0.0D0
                    ENDIF
                    IF (QPFEX .AND. JNB(NB).LT.0) THEN
                       !  DO NOTHING, EXCLUDED
                    ELSE
                       NDTS(K1,K2,IDXTSL) = 3.0D0*RK1*RK2*R5*FLMD5 &
                            - DTA1*R3*FLMD3
                       NDTS(K1,K2,IDXTSU) = NDTS(K1,K2,IDXTSL)
                    ENDIF
                    DO K3 = 1,3
                       IF (K3.EQ.1) THEN 
                          RK3 = XX
                       ELSEIF (K3.EQ.2) THEN
                          RK3 = YY
                       ELSE
                          RK3 = ZZ
                       ENDIF
                       IF (K3.EQ.K1) THEN
                          DTA2 = 1.0D0
                       ELSE 
                          DTA2 = 0.0D0
                       ENDIF
                       IF (K3.EQ.K2) THEN
                          DTA4 = 1.0D0
                       ELSE
                          DTA4 = 0.0D0
                       ENDIF
                       IF (QPFEX .AND. JNB(NB).LT.0) THEN
                          !  DO NOTHING, EXCLUDED
                       ELSE 
                          !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                          RDTS(K1,K2,K3,IDXTSL) =  &
                               -15.0D0*RK1*RK2*RK3*R7*FLMD7  &
                               + 3.0D0*(RK1*DTA4+RK2*DTA2+RK3*DTA1)*R5*FLMD5 
                          RDTS(K1,K2,K3,IDXTSU) = -RDTS(K1,K2,K3,IDXTSL)
                       ENDIF
                       DO K4 = 1,3
                          IF (K4.EQ.1) THEN 
                             RK4 = XX
                          ELSEIF (K4.EQ.2) THEN
                             RK4 = YY
                          ELSE
                             RK4 = ZZ
                          ENDIF
                          IF (K4.EQ.K1) THEN
                             DTA3 = 1.0D0
                          ELSE
                             DTA3 = 0.0D0
                          ENDIF
                          IF (K4.EQ.K2) THEN
                             DTA5 = 1.0D0
                          ELSE
                             DTA5 = 0.0D0
                          ENDIF
                          IF (K4.EQ.K3) THEN
                             DTA6 = 1.0D0
                          ELSE
                             DTA6 = 0.0D0
                          ENDIF
                          IF (QPFEX .AND. JNB(NB).LT.0) THEN
                             !  DO NOTHING, EXCLUDED
                          ELSE 
                             !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
                             FTHTS(K1,K2,K3,K4,IDXTSL) =  &
                                  105.0D0*RK1*RK2*RK3*RK4*R9*FLMD9  &
                                  - 15.0D0*(RK1*RK2*DTA6+RK1*RK3*DTA5+RK1*RK4 &
                                  *DTA4+RK2*RK3*DTA3+RK2*RK4*DTA2+RK3*RK4*DTA1) &
                                  *R7*FLMD7 &
                                  + 3.0D0*(DTA1*DTA6+DTA2*DTA5+DTA3*DTA4) &
                                  *R5*FLMD5
                             FTHTS(K1,K2,K3,K4,IDXTSU)= &
                                  FTHTS(K1,K2,K3,K4,IDXTSL)
                          ENDIF
                       ENDDO
                    ENDDO
                 ENDDO
              ENDDO
#ifdef PIPF_CTOF  /* ecut2 */
              ! END ENERGY THE CUT-OFF
           END IF
#endif  /* ecut2 */
        ENDDO
     ENDIF
  ENDDO
