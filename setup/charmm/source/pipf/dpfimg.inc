  !=======================================================================
  !   Main loop begin (P-P interaction)
  !=======================================================================
  NB = 0
  ITEMP = 0
  DO I=IFRSTAPP,NATOMPP
     NPR=INBLOPP(I)-ITEMP
     ITEMP=INBLOPP(I)
     IF(NPR.GT.0) THEN

        ! CHARGE FACTOR I
        CGI = CG(I)*CCELEC
        I1=ITC(IAC(I))

        DO JPR=1,NPR
           NB=NB+1
           IF (JNBPP(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNBPP(NB)
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNBPP(NB)
           ENDIF

           !      CHARGE FACTOR FOR J
           CGJ = CG(J)*CCELEC
           J1=ITC(IAC(J))
           !
           DXI=X(I)-X(J)
           DYI=Y(I)-Y(J)
           DZI=Z(I)-Z(J)

#if KEY_PBOUND==1 /*pbound*/
           If(qBoun) then                             
              If(qCUBoun.or.qTOBoun) then
                 DXI = BOXINV * DXI
                 DYI = BOYINV * DYI
                 DZI = BOZINV * DZI
                 dxi = dxi - nint(dxi)
                 dyi = dyi - nint(dyi)
                 dzi = dzi - nint(dzi)
!!$                 IF(DXI.GT.  HALF) DXI = DXI - ONE
!!$                 IF(DXI.LT. -HALF) DXI = DXI + ONE
!!$                 IF(DYI.GT.  HALF) DYI = DYI - ONE
!!$                 IF(DYI.LT. -HALF) DYI = DYI + ONE
!!$                 IF(DZI.GT.  HALF) DZI = DZI - ONE
!!$                 IF(DZI.LT. -HALF) DZI = DZI + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(DXI) + &
                         ABS(DYI) + &
                         ABS(DZI)))
                    DXI = DXI - SIGN( CORR,  DXI  )
                    DYI = DYI - SIGN( CORR,  DYI  )
                    DZI = DZI - SIGN( CORR,  DZI  )
                 Endif
                 DXI = XSIZE * DXI
                 DYI = YSIZE * DYI
                 DZI = ZSIZE * DZI
              Else
                 Call PBMove(DXI, DYI, DZI)
              Endif
           Endif
#endif /*      (pbound)*/
           RIJ = SQRT(DXI*DXI+DYI*DYI+DZI*DZI)

#ifdef PIPF_CTOF  /* ecut1 */
           IF (RIJ .LT. PFCTOF) THEN  ! APPLY AN ENERGY CUT-OFF
#endif  /* ecut1 */
              !
              !      WE FOLLOW THE EQ. (30) IN CHEM. PHYS. LETT. 1990, 166,180. NOTE THAT
              !      CHARMM ARRAY DX, DY, DZ ACTUALLY CONTAINS GRADIENT (DV/DX) INSTEAD OF 
              !      FORCE (-DV/DX)
              !
              !      FIRST, WE COMPUTE THE GRADIENT ASSOCIATED WITH PERMENENT CHARGE
              !      (THE SECOND TERM IN EQ. 30)
              !
              !          Q^K * PI^K   = Q^K * sum (T^KA u^A)     (r denotes lamda)
              !                  r             A    rb   b
              !
              DX(I) = DX(I) - CGI*(TXWWPP(1,NB)*DMUIND(1,J) &
                   +TXWWPP(2,NB)*DMUIND(2,J)+TXWWPP(4,NB)*DMUIND(3,J))
              DY(I) = DY(I) - CGI*(TXWWPP(2,NB)*DMUIND(1,J) &
                   +TXWWPP(3,NB)*DMUIND(2,J)+TXWWPP(5,NB)*DMUIND(3,J))
              DZ(I) = DZ(I) - CGI*(TXWWPP(4,NB)*DMUIND(1,J) &
                   +TXWWPP(5,NB)*DMUIND(2,J)+TXWWPP(6,NB)*DMUIND(3,J))

              DX(J) = DX(J) - CGJ*(TXWWPP(1,NB)*DMUIND(1,I) &
                   +TXWWPP(2,NB)*DMUIND(2,I)+TXWWPP(4,NB)*DMUIND(3,I))  
              DY(J) = DY(J) - CGJ*(TXWWPP(2,NB)*DMUIND(1,I)           &
                   +TXWWPP(3,NB)*DMUIND(2,I)+TXWWPP(5,NB)*DMUIND(3,I))
              DZ(J) = DZ(J) - CGJ*(TXWWPP(4,NB)*DMUIND(1,I) &
                   +TXWWPP(5,NB)*DMUIND(2,I)+TXWWPP(6,NB)*DMUIND(3,I))
              !
              !      NEXT, WE COMPUTE THE GRADIENT ASSOCIATED WITH PERMENENT FIELD GRADIENT
              !      (THE THIRD TERM IN EQ. 30)
              !
              !          u^K * E^K   = u^K * [- sum (T^KA Q^A)]    (r denotes lamda)
              !           a     ar      a        A    ar   

              DX(I) = DX(I) + CGJ*(TXWWPP(1,NB)*DMUIND(1,I) &
                   +TXWWPP(2,NB)*DMUIND(2,I)+TXWWPP(4,NB)*DMUIND(3,I))
              DY(I) = DY(I) + CGJ*(TXWWPP(2,NB)*DMUIND(1,I) &
                   +TXWWPP(3,NB)*DMUIND(2,I)+TXWWPP(5,NB)*DMUIND(3,I))
              DZ(I) = DZ(I) + CGJ*(TXWWPP(4,NB)*DMUIND(1,I) &
                   +TXWWPP(5,NB)*DMUIND(2,I)+TXWWPP(6,NB)*DMUIND(3,I))

              DX(J) = DX(J) + CGI*(TXWWPP(1,NB)*DMUIND(1,J)   &
                   +TXWWPP(2,NB)*DMUIND(2,J)+TXWWPP(4,NB)*DMUIND(3,J))        
              DY(J) = DY(J) + CGI*(TXWWPP(2,NB)*DMUIND(1,J) &
                   +TXWWPP(3,NB)*DMUIND(2,J)+TXWWPP(5,NB)*DMUIND(3,J))
              DZ(J) = DZ(J) + CGI*(TXWWPP(4,NB)*DMUIND(1,J) &
                   +TXWWPP(5,NB)*DMUIND(2,J)+TXWWPP(6,NB)*DMUIND(3,J)) 

              !
              !      FINALLY WE COMPUTE GRADIENT DUE TO INDUCED FIELD GRADIENT
              !      (THE FOURTH TERM IN EQ. 30) 
              !              
              !          u^K * PI^K = u^K * sum (T^KA u^A)      (r denotes lamda)
              !           a      ar    a     A    arb  b    
              !
              !      THE ABOVE EXPRESSION IS EQUAVALENT TO THE SECOND TERM OF EQ. (11)
              !      IN MOLE. PHYS. 1989, 68, 563. SO INSTEAD OF FORMULATE THIS TERM
              !      IN TERMS OF RANK 3 POLAR TENSOR, ONE DIRECTLY EVALUATE THIS
              !      TERM ACCORDING TO EQ. (12) IN MOLE. PHYS. 1989, 68, 563. 
              !

              !
              !           R1 = 1.0D0/SQRT(DXI*DXI+DYI*DYI+DZI*DZI)
              !
              R1 = 1.0D0/RIJ
              R2 = 5.0D0*R1*R1
              R5 = 3.0D0*CCELEC*R1**5.0D0 
              !
              ! OPTIONALLY EXCLUDE 1-4 POLARIZATION BY ZERO OUT COMMEN FACTOR R5.
              ! NO NEED FOR HANDLING FIRST TWO TERMS, SINCE THE COMMON FACTOR TXWW
              ! HAS BEEN CONSISTENT EXCLUSED IN ETINIT FOR 1-4 PAIRS.
              !
              IF (QPFEX .AND. JNBPP(NB).LT.0) R5 = 0.0D0

              DPR1 = DMUIND(1,I)*DXI + DMUIND(2,I)*DYI + DMUIND(3,I)*DZI
              DPR2 = DMUIND(1,J)*DXI + DMUIND(2,J)*DYI + DMUIND(3,J)*DZI
              DPR3 = DMUIND(1,I)*DMUIND(1,J) + DMUIND(2,I)*DMUIND(2,J) +  &
                   DMUIND(3,I)*DMUIND(3,J)  

#ifndef PIPF_DAMP  /* damp1 */
              !      NO DAMPING (DEFAULT)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*DXI - &
                      DPR3*DXI - DPR2*DMUIND(1,I) - DPR1*DMUIND(1,J))  
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*DYI - &
                      DPR3*DYI - DPR2*DMUIND(2,I) - DPR1*DMUIND(2,J))
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*DZI - &
                      DPR3*DZI - DPR2*DMUIND(3,I) - DPR1*DMUIND(3,J))

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*DXI - &
                      DPR3*DXI - DPR1*DMUIND(1,J) - DPR2*DMUIND(1,I))    
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*DYI - &
                      DPR3*DYI - DPR1*DMUIND(2,J) - DPR2*DMUIND(2,I))
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*DZI - &
                      DPR3*DZI - DPR1*DMUIND(3,J) - DPR2*DMUIND(3,I))
#else /* damp2 */
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 R3 = R1*R1*R1
                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
                 FLMD7 = 1.0D0 - (1.0D0+AU3+0.6D0*AU3*AU3)*EXP(-AU3)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*DXI*FLMD7 - &
                      (DPR3*DXI+DPR2*DMUIND(1,I)+DPR1*DMUIND(1,J))*FLMD5)
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*DYI*FLMD7 - &
                      (DPR3*DYI+DPR2*DMUIND(2,I)+DPR1*DMUIND(2,J))*FLMD5)
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*DZI*FLMD7 - &
                      (DPR3*DZI+DPR2*DMUIND(3,I)+DPR1*DMUIND(3,J))*FLMD5)

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*DXI*FLMD7 - &
                      (DPR3*DXI+DPR1*DMUIND(1,J)+DPR2*DMUIND(1,I))*FLMD5)
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*DYI*FLMD7 - &
                      (DPR3*DYI+DPR1*DMUIND(2,J)+DPR2*DMUIND(2,I))*FLMD5)
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*DZI*FLMD7 - &
                      (DPR3*DZI+DPR1*DMUIND(3,J)+DPR2*DMUIND(3,I))*FLMD5)
#endif /* damp1 damp2 */

#ifdef PIPF_CTOF /* ecut2 */
           END IF  ! END THE ENERGY CUT-OFF
#endif  /* ecut2 */
        ENDDO  ! end JPR loop
     ENDIF
  ENDDO  ! end main loop

  !=======================================================================
  !   Main loop begin (P-I interaction)
  !=======================================================================
  NB = 0
  ITEMP = 0
  DO I=IFRSTAIP,NATOMIP
     NPR=INBLOIP(I)-ITEMP
     ITEMP=INBLOIP(I)
     IF(NPR.GT.0) THEN

        ! CHARGE FACTOR I
        CGI = CG(I)*CCELEC
        I1 = ITC(IAC(IMATTR(I)))

        DO JPR=1,NPR
           NB=NB+1
           IF (JNBIP(NB).LT.0) THEN
              CGT2=CGT*E14FAC
              E14F = E14M1
              J=-JNBIP(NB)
           ELSE
              CGT2=CGT
              E14F=ZERO
              J=JNBIP(NB)
           ENDIF

           !      CHARGE FACTOR FOR J
           CGJ = CG(J)*CCELEC
           J1 = ITC(IAC(J))
           !
           DXI=X(I)-X(J)
           DYI=Y(I)-Y(J)
           DZI=Z(I)-Z(J)

#if KEY_PBOUND==1 /*pbound*/
           If(qBoun) then
              If(qCUBoun.or.qTOBoun) then
                 DXI = BOXINV * DXI
                 DYI = BOYINV * DYI
                 DZI = BOZINV * DZI
                 dxi = dxi - nint(dxi)
                 dyi = dyi - nint(dyi)
                 dzi = dzi - nint(dzi)
!!$                 IF(DXI.GT.  HALF) DXI = DXI - ONE
!!$                 IF(DXI.LT. -HALF) DXI = DXI + ONE
!!$                 IF(DYI.GT.  HALF) DYI = DYI - ONE
!!$                 IF(DYI.LT. -HALF) DYI = DYI + ONE
!!$                 IF(DZI.GT.  HALF) DZI = DZI - ONE
!!$                 IF(DZI.LT. -HALF) DZI = DZI + ONE
                 If (qTOBoun) Then
                    CORR = HALF * AINT ( R75 * (ABS(DXI) + &
                         ABS(DYI) + &
                         ABS(DZI)))
                    DXI = DXI - SIGN( CORR,  DXI  )
                    DYI = DYI - SIGN( CORR,  DYI  )
                    DZI = DZI - SIGN( CORR,  DZI  )
                 Endif
                 DXI = XSIZE * DXI
                 DYI = YSIZE * DYI
                 DZI = ZSIZE * DZI
              Else
                 Call PBMove(DXI, DYI, DZI)
              Endif
           Endif
#endif /*      (pbound)*/
           RIJ = SQRT(DXI*DXI+DYI*DYI+DZI*DZI)

#ifdef PIPF_CTOF  /* ecut3 */
           IF (RIJ .LT. PFCTOF) THEN  ! APPLY AN ENERGY CUT-OFF
#endif  /* ecut3 */

              !
              !      WE FOLLOW THE EQ. (30) IN CHEM. PHYS. LETT. 1990, 166,180. NOTE THAT
              !      CHARMM ARRAY DX, DY, DZ ACTUALLY CONTAINS GRADIENT (DV/DX) INSTEAD OF
              !      FORCE (-DV/DX)
              !
              !      FIRST, WE COMPUTE THE GRADIENT ASSOCIATED WITH PERMENENT CHARGE
              !      (THE SECOND TERM IN EQ. 30)
              !
              !          Q^K * PI^K   = Q^K * sum (T^KA u^A)     (r denotes lamda)
              !                  r             A    rb   b
              !
              DX(I) = DX(I) - CGI*(TXWWIP(1,NB)*DMUIND(1,J) &
                   +TXWWIP(2,NB)*DMUIND(2,J)+TXWWIP(4,NB)*DMUIND(3,J))
              DY(I) = DY(I) - CGI*(TXWWIP(2,NB)*DMUIND(1,J) &
                   +TXWWIP(3,NB)*DMUIND(2,J)+TXWWIP(5,NB)*DMUIND(3,J))
              DZ(I) = DZ(I) - CGI*(TXWWIP(4,NB)*DMUIND(1,J) &
                   +TXWWIP(5,NB)*DMUIND(2,J)+TXWWIP(6,NB)*DMUIND(3,J))

              DX(J) = DX(J) - CGJ*(TXWWIP(1,NB)*DMUIND(1,I) &
                   +TXWWIP(2,NB)*DMUIND(2,I)+TXWWIP(4,NB)*DMUIND(3,I))
              DY(J) = DY(J) - CGJ*(TXWWIP(2,NB)*DMUIND(1,I) &
                   +TXWWIP(3,NB)*DMUIND(2,I)+TXWWIP(5,NB)*DMUIND(3,I))
              DZ(J) = DZ(J) - CGJ*(TXWWIP(4,NB)*DMUIND(1,I) &
                   +TXWWIP(5,NB)*DMUIND(2,I)+TXWWIP(6,NB)*DMUIND(3,I))
              !
              !      NEXT, WE COMPUTE THE GRADIENT ASSOCIATED WITH PERMENENT FIELD GRADIENT
              !      (THE THIRD TERM IN EQ. 30)
              !
              !
              !          u^K * E^K   = u^K * [- sum (T^KA Q^A)]    (r denotes lamda)
              !           a     ar      a        A    ar

              DX(I) = DX(I) + CGJ*(TXWWIP(1,NB)*DMUIND(1,I) &
                   +TXWWIP(2,NB)*DMUIND(2,I)+TXWWIP(4,NB)*DMUIND(3,I))
              DY(I) = DY(I) + CGJ*(TXWWIP(2,NB)*DMUIND(1,I) &
                   +TXWWIP(3,NB)*DMUIND(2,I)+TXWWIP(5,NB)*DMUIND(3,I))
              DZ(I) = DZ(I) + CGJ*(TXWWIP(4,NB)*DMUIND(1,I) &
                   +TXWWIP(5,NB)*DMUIND(2,I)+TXWWIP(6,NB)*DMUIND(3,I))

              DX(J) = DX(J) + CGI*(TXWWIP(1,NB)*DMUIND(1,J) &
                   +TXWWIP(2,NB)*DMUIND(2,J)+TXWWIP(4,NB)*DMUIND(3,J))
              DY(J) = DY(J) + CGI*(TXWWIP(2,NB)*DMUIND(1,J) &
                   +TXWWIP(3,NB)*DMUIND(2,J)+TXWWIP(5,NB)*DMUIND(3,J))
              DZ(J) = DZ(J) + CGI*(TXWWIP(4,NB)*DMUIND(1,J) &
                   +TXWWIP(5,NB)*DMUIND(2,J)+TXWWIP(6,NB)*DMUIND(3,J))

              !
              !      FINALLY WE COMPUTE GRADIENT DUE TO INDUCED FIELD GRADIENT
              !      (THE FOURTH TERM IN EQ. 30)
              !
              !          u^K * PI^K = u^K * sum (T^KA u^A)      (r denotes lamda)
              !           a      ar    a     A    arb  b
              !
              !      THE ABOVE EXPRESSION IS EQUAVALENT TO THE SECOND TERM OF EQ. (11)
              !      IN MOLE. PHYS. 1989, 68, 563. SO INSTEAD OF FORMULATE THIS TERM
              !      IN TERMS OF RANK 3 POLAR TENSOR, ONE DIRECTLY EVALUATE THIS
              !      TERM ACCORDING TO EQ. (12) IN MOLE. PHYS. 1989, 68, 563.
              !
              !
              !           R1 = 1.0D0/SQRT(DXI*DXI+DYI*DYI+DZI*DZI)
              !
              R1 = 1.0D0/RIJ
              R2 = 5.0D0*R1*R1
              R5 = 3.0D0*CCELEC*R1**5.0D0
              !
              ! OPTIONALLY EXCLUDE 1-4 POLARIZATION BY ZERO OUT COMMEN FACTOR R5.
              ! NO NEED FOR HANDLING FIRST TWO TERMS, SINCE THE COMMON FACTOR TXWW
              ! HAS BEEN CONSISTENT EXCLUSED IN ETINIT FOR 1-4 PAIRS.
              !
              IF (QPFEX .AND. JNBIP(NB).LT.0) R5 = 0.0D0

              DPR1 = DMUIND(1,I)*DXI + DMUIND(2,I)*DYI + DMUIND(3,I)*DZI
              DPR2 = DMUIND(1,J)*DXI + DMUIND(2,J)*DYI + DMUIND(3,J)*DZI
              DPR3 = DMUIND(1,I)*DMUIND(1,J) + DMUIND(2,I)*DMUIND(2,J) + &
                   DMUIND(3,I)*DMUIND(3,J)
#ifndef PIPF_DAMP  /* damp3 */
                 ! NO DAMPING (DEFAULT)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*DXI - &
                      DPR3*DXI - DPR2*DMUIND(1,I) - DPR1*DMUIND(1,J))
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*DYI - &
                      DPR3*DYI - DPR2*DMUIND(2,I) - DPR1*DMUIND(2,J))
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*DZI - &
                      DPR3*DZI - DPR2*DMUIND(3,I) - DPR1*DMUIND(3,J))

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*DXI - &
                      DPR3*DXI - DPR1*DMUIND(1,J) - DPR2*DMUIND(1,I))
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*DYI - &
                      DPR3*DYI - DPR1*DMUIND(2,J) - DPR2*DMUIND(2,I))
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*DZI - &
                      DPR3*DZI - DPR1*DMUIND(3,J) - DPR2*DMUIND(3,I))
#else  /* damp4 */
                 !       OR APPLY A DAMPING FACTOR
                 !       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
                 R3 = R1*R1*R1
                 U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
                 AU3 = DPFAC*U3
                 FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
                 FLMD7 = 1.0D0 - (1.0D0+AU3+0.6D0*AU3*AU3)*EXP(-AU3)
                 DX(I) = DX(I) + R5*(R2*DPR1*DPR2*DXI*FLMD7 - &
                      (DPR3*DXI+DPR2*DMUIND(1,I)+DPR1*DMUIND(1,J))*FLMD5)
                 DY(I) = DY(I) + R5*(R2*DPR1*DPR2*DYI*FLMD7 - &
                      (DPR3*DYI+DPR2*DMUIND(2,I)+DPR1*DMUIND(2,J))*FLMD5)
                 DZ(I) = DZ(I) + R5*(R2*DPR1*DPR2*DZI*FLMD7 - &
                      (DPR3*DZI+DPR2*DMUIND(3,I)+DPR1*DMUIND(3,J))*FLMD5)

                 DX(J) = DX(J) - R5*(R2*DPR1*DPR2*DXI*FLMD7 - &
                      (DPR3*DXI+DPR1*DMUIND(1,J)+DPR2*DMUIND(1,I))*FLMD5)
                 DY(J) = DY(J) - R5*(R2*DPR1*DPR2*DYI*FLMD7 - &
                      (DPR3*DYI+DPR1*DMUIND(2,J)+DPR2*DMUIND(2,I))*FLMD5)
                 DZ(J) = DZ(J) - R5*(R2*DPR1*DPR2*DZI*FLMD7 - &
                      (DPR3*DZI+DPR1*DMUIND(3,J)+DPR2*DMUIND(3,I))*FLMD5)   
#endif  /* damp3 damp4 */

#ifdef PIPF_CTOF  /* ecut4 */
           END IF  ! END THE ENERGY CUT-OFF
#endif  /* ecut4 */
        ENDDO  ! end JPR loop
     ENDIF
  ENDDO  ! end main loop
