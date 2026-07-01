!=======================================================================
!   Main loop begin
!=======================================================================
      DO I=IFRSTA,NATOM
        NPR=INBLO(I)-ITEMP
        ITEMP=INBLO(I)
        IF(NPR.GT.0) THEN

! CHARGE FACTOR I
          CGI = CG(I)*CCELEC
          I1=ITC(IAC(I))
!
!         IACI=IOFF(I1)
!

!
! not sure about thie section (group-group?)
!         IF (ELECFG) THEN
!           CGT=CGF*CG(I)
!           ELCFG=(CGT.NE.0.0)
!         ELSE
!           CGT=ZERO
!         ENDIF
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
!!$                  IF(DXI.GT.  HALF) DXI = DXI - ONE
!!$                  IF(DXI.LT. -HALF) DXI = DXI + ONE
!!$                  IF(DYI.GT.  HALF) DYI = DYI - ONE
!!$                  IF(DYI.LT. -HALF) DYI = DYI + ONE
!!$                  IF(DZI.GT.  HALF) DZI = DZI - ONE
!!$                  IF(DZI.LT. -HALF) DZI = DZI + ONE
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
!      APPLY AN ENERGY CUT-OFF
          IF (RIJ .LT. PFCTOF) THEN
#endif  /* ecut1 */

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
            DX(I) = DX(I) - CGI*(TXWW(1,NB)*DMUIND(1,J) &
                    +TXWW(2,NB)*DMUIND(2,J)+TXWW(4,NB)*DMUIND(3,J))
            DY(I) = DY(I) - CGI*(TXWW(2,NB)*DMUIND(1,J) &
                    +TXWW(3,NB)*DMUIND(2,J)+TXWW(5,NB)*DMUIND(3,J))
            DZ(I) = DZ(I) - CGI*(TXWW(4,NB)*DMUIND(1,J) &
                    +TXWW(5,NB)*DMUIND(2,J)+TXWW(6,NB)*DMUIND(3,J))

            DX(J) = DX(J) - CGJ*(TXWW(1,NB)*DMUIND(1,I) &
                    +TXWW(2,NB)*DMUIND(2,I)+TXWW(4,NB)*DMUIND(3,I))
            DY(J) = DY(J) - CGJ*(TXWW(2,NB)*DMUIND(1,I)           &
                    +TXWW(3,NB)*DMUIND(2,I)+TXWW(5,NB)*DMUIND(3,I))
            DZ(J) = DZ(J) - CGJ*(TXWW(4,NB)*DMUIND(1,I) &
                    +TXWW(5,NB)*DMUIND(2,I)+TXWW(6,NB)*DMUIND(3,I))
!
!      NEXT, WE COMPUTE THE GRADIENT ASSOCIATED WITH PERMENENT FIELD GRADIENT
!      (THE THIRD TERM IN EQ. 30)
!
!          u^K * E^K   = u^K * [- sum (T^KA Q^A)]    (r denotes lamda)
!           a     ar      a        A    ar   

            DX(I) = DX(I) + CGJ*(TXWW(1,NB)*DMUIND(1,I) &
                    +TXWW(2,NB)*DMUIND(2,I)+TXWW(4,NB)*DMUIND(3,I))
            DY(I) = DY(I) + CGJ*(TXWW(2,NB)*DMUIND(1,I) &
                    +TXWW(3,NB)*DMUIND(2,I)+TXWW(5,NB)*DMUIND(3,I))
            DZ(I) = DZ(I) + CGJ*(TXWW(4,NB)*DMUIND(1,I) &
                    +TXWW(5,NB)*DMUIND(2,I)+TXWW(6,NB)*DMUIND(3,I))

            DX(J) = DX(J) + CGI*(TXWW(1,NB)*DMUIND(1,J)   &
                    +TXWW(2,NB)*DMUIND(2,J)+TXWW(4,NB)*DMUIND(3,J))        
            DY(J) = DY(J) + CGI*(TXWW(2,NB)*DMUIND(1,J) &
                    +TXWW(3,NB)*DMUIND(2,J)+TXWW(5,NB)*DMUIND(3,J))
            DZ(J) = DZ(J) + CGI*(TXWW(4,NB)*DMUIND(1,J) &
                    +TXWW(5,NB)*DMUIND(2,J)+TXWW(6,NB)*DMUIND(3,J)) 

!
!      FINALLY WE COMPUTE GRADIENT DUE TO INDUCED FIELD GRADIENT
!      (THE FOURTH TERM IN EQ. 30) 
!              
!          u^K * PI^K = u^K * sum (T^KA u^A)      (r denotes lamda)
!           a      ar    a     A    arb  b    
!
!           R1 = 1.0D0/SQRT(DXI*DXI+DYI*DYI+DZI*DZI)
!
!           R1 = 1.0D0/RIJ
!           R2 = 5.0D0*R1*R1
!           R5 = 3.0D0*CCELEC*R1**5.0D0
!
! ##IF .not.damp B (damp1)
!      NO DAMPING (DEFAULT)
!
!
!              TXXX = 3.0D0*DXI - DXI*DXI*DXI*R2 
!              TYXX = DYI - DXI*DXI*DYI*R2 
!              TYYX = DXI - DXI*DYI*DYI*R2 
!              TZXX = DZI - DXI*DXI*DZI*R2 
!              TZYX = - DXI*DYI*DZI*R2 
!              TZZX = DXI - DXI*DZI*DZI*R2 
!              TYYY = 3.0D0*DYI - DYI*DYI*DYI*R2
!              TZYY = DZI - DYI*DYI*DZI*R2
!              TZZY = DYI - DYI*DZI*DZI*R2 
!              TZZZ = 3.0D0*DZI - DZI*DZI*DZI*R2 
!...##ENDIF (damp1)
!...##IF damp B (damp2)
!
!       OR APPLY A DAMPING FACTOR (DAMPING CORRECT, PASSED NUMERICAL TEST)
!       (Thole, ChemPhys 59, 341, 1981. Ren&Ponder, JPC B 107, 5935,2003)
!
!
!              R3 = R1*R1*R1
!              U3 = (1.0D0/R3)/SQRT(ALP(I1)*ALP(J1))
!              AU3 = DPFAC*U3
!              FLMD5 = 1.0D0 - (1.0D0+AU3)*EXP(-AU3)
!              FLMD7 = 1.0D0 - (1.0D0+AU3+0.6D0*AU3*AU3)*EXP(-AU3)
!              TXXX = 3.0D0*DXI*FLMD5 - DXI*DXI*DXI*R2*FLMD7 
!              TYXX = DYI*FLMD5 - DXI*DXI*DYI*R2*FLMD7
!              TYYX = DXI*FLMD5 - DXI*DYI*DYI*R2*FLMD7
!              TZXX = DZI*FLMD5 - DXI*DXI*DZI*R2*FLMD7
!              TZYX = - DXI*DYI*DZI*R2*FLMD7
!              TZZX = DXI*FLMD5 - DXI*DZI*DZI*R2*FLMD7
!              TYYY = 3.0D0*DYI*FLMD5 - DYI*DYI*DYI*R2*FLMD7
!              TZYY = DZI*FLMD5 - DYI*DYI*DZI*R2*FLMD7
!              TZZY = DYI*FLMD5 - DYI*DZI*DZI*R2*FLMD7
!              TZZZ = 3.0D0*DZI*FLMD5 - DZI*DZI*DZI*R2*FLMD7
!...##ENDIF (damp2)
!
!           DX(I) = DX(I) - R5 * (
!    & (TXXX*DMUIND(1,J)+TYXX*DMUIND(2,J)+TZXX*DMUIND(3,J))*DMUIND(1,I)+   
!    & (TYXX*DMUIND(1,J)+TYYX*DMUIND(2,J)+TZYX*DMUIND(3,J))*DMUIND(2,I)+   
!    & (TZXX*DMUIND(1,J)+TZYX*DMUIND(2,J)+TZZX*DMUIND(3,J))*DMUIND(3,I))   
!
!           DY(I) = DY(I) - R5 * (
!    & (TYXX*DMUIND(1,J)+TYYX*DMUIND(2,J)+TZYX*DMUIND(3,J))*DMUIND(1,I)+   
!    & (TYYX*DMUIND(1,J)+TYYY*DMUIND(2,J)+TZYY*DMUIND(3,J))*DMUIND(2,I)+
!    & (TZYX*DMUIND(1,J)+TZYY*DMUIND(2,J)+TZZY*DMUIND(3,J))*DMUIND(3,I))
!
!           DZ(I) = DZ(I) - R5 * (
!    & (TZXX*DMUIND(1,J)+TZYX*DMUIND(2,J)+TZZX*DMUIND(3,J))*DMUIND(1,I)+   
!    & (TZYX*DMUIND(1,J)+TZYY*DMUIND(2,J)+TZZY*DMUIND(3,J))*DMUIND(2,I)+
!    & (TZZX*DMUIND(1,J)+TZZY*DMUIND(2,J)+TZZZ*DMUIND(3,J))*DMUIND(3,I))
!
!           DX(J) = DX(J) + R5 * (
!    & (TXXX*DMUIND(1,I)+TYXX*DMUIND(2,I)+TZXX*DMUIND(3,I))*DMUIND(1,J)+
!    & (TYXX*DMUIND(1,I)+TYYX*DMUIND(2,I)+TZYX*DMUIND(3,I))*DMUIND(2,J)+
!    & (TZXX*DMUIND(1,I)+TZYX*DMUIND(2,I)+TZZX*DMUIND(3,I))*DMUIND(3,J))
!
!           DY(J) = DY(J) + R5 * (
!    & (TYXX*DMUIND(1,I)+TYYX*DMUIND(2,I)+TZYX*DMUIND(3,I))*DMUIND(1,J)+
!    & (TYYX*DMUIND(1,I)+TYYY*DMUIND(2,I)+TZYY*DMUIND(3,I))*DMUIND(2,J)+
!    & (TZYX*DMUIND(1,I)+TZYY*DMUIND(2,I)+TZZY*DMUIND(3,I))*DMUIND(3,J))
!
!           DZ(J) = DZ(J) + R5 * (
!    & (TZXX*DMUIND(1,I)+TZYX*DMUIND(2,I)+TZZX*DMUIND(3,I))*DMUIND(1,J)+
!    & (TZYX*DMUIND(1,I)+TZYY*DMUIND(2,I)+TZZY*DMUIND(3,I))*DMUIND(2,J)+
!    & (TZZX*DMUIND(1,I)+TZZY*DMUIND(2,I)+TZZZ*DMUIND(3,I))*DMUIND(3,J))
!
 
!
!      THE ABOVE EXPRESSION IS EQUAVALENT TO THE SECOND TERM OF EQ. (11)
!      IN MOLE. PHYS. 1989, 68, 563. SO INSTEAD OF FORMULATE THIS TERM
!      IN TERMS OF RANK 3 POLAR TENSOR, ONE DIRECTLY EVALUATE THIS
!      TERM ACCORDING TO EQ. (12) IN MOLE. PHYS. 1989, 68, 563. 
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
            IF (QPFEX .AND. JNB(NB).LT.0) R5 = 0.0D0
 
            DPR1 = DMUIND(1,I)*DXI + DMUIND(2,I)*DYI + DMUIND(3,I)*DZI
            DPR2 = DMUIND(1,J)*DXI + DMUIND(2,J)*DYI + DMUIND(3,J)*DZI
            DPR3 = DMUIND(1,I)*DMUIND(1,J) + DMUIND(2,I)*DMUIND(2,J) +  &
                   DMUIND(3,I)*DMUIND(3,J)  
 
!
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
 
#else  /* damp2 */
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
#endif  /* damp1 damp2 */

#ifdef PIPF_CTOF  /* ecut2 */
           END IF  ! END THE ENERGY CUT-OFF
#endif  /* ecut2 */
          ENDDO  ! end JPR loop
        ENDIF
      ENDDO  ! end main loop
