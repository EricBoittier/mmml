module mcarmupt
#if KEY_MC==1
contains

  SUBROUTINE ARMUPT(IDX,IARMF,AP,AA,AB,NTRY,NACC,ALIM,AMAX,RMX, &
       ANISO)
    !
    !       ARM optimization of the move step size.
    !
    !       For explanation of both ARM and DOMC, see Bouzida, D., Kumar, S. 
    !       and Swendsen, R. H.  (1992)  Efficient Monte Carlo Methods for the 
    !       Computer Simulation of Biological Molecules.  Phys. Rev. A 45, 
    !       8894-8901.
    !      
    !       Updates the NTRY array and if NTRY(IDX) equals the
    !       ARM frequency, updates the step size.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    INTEGER IDX, IARMF, NTRY(*), NACC(*)
    real(chm_real) :: RMX(:), AP, AA, AB, AMAX
    LOGICAL ALIM, ANISO
    !       Local Variables
    INTEGER I, IADD
    real(chm_real)  RP

    !       AP is the ideal probability of acceptance.
    !       Use AP negative as a flag to skip DOMC for a move group
    IF (AP .LT. 0.0) RETURN
    !
    NTRY(IDX) = NTRY(IDX) + 1
    IF (NTRY(IDX) .EQ. IARMF) THEN
       RP = FLOAT(NACC(IDX))/NTRY(IDX)
       RP = LOG(AA*AP+AB)/LOG(AA*RP+AB)
       IF (ANISO) THEN
          IADD = (IDX - 1)*9
          DO I = 1, 9
             RMX(IADD+I) = RMX(IADD+I)*RP
          ENDDO
       ELSE
          RMX(IDX) = RMX(IDX)*RP
       ENDIF
       !         ALIM should be TRUE only for isotropic 
       IF (ALIM) RMX(IDX) = MIN(RMX(IDX),AMAX)
       NACC(IDX) = 0
       NTRY(IDX) = 0
    ENDIF

    RETURN
  END SUBROUTINE ARMUPT

  SUBROUTINE DOMCUP(IDX,IDOMCF,AP,AA,AB,NTRY,NACC, &
       EAVE,DAVE,ALIM,AMAX,DX,DE, &
       DOMCF,ANISO,BETA,RMX)
    !
    !       DOMC optimization of the move step size.
    !       If it is anisotropic, just calls DOMCAI and returns.
    !       If it is   isotropic, deals with it here.
    !      
    !       Updates the NTRY array and if NTRY(IDX) equals the
    !       DOMC frequency, updates the step size.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    implicit none
    INTEGER IDX, IDOMCF, NTRY(*), NACC(*)
    real(chm_real) :: RMX(:), AP, AA, AB, AMAX
    real(chm_real)  EAVE(*),DAVE(*), DOMCF, BETA
    real(chm_real)  DX(3), DE
    LOGICAL ALIM, ANISO

    !       F is E^2 of the effective harmonic
    !       Use F negative as a flag to skip DOMC for a move group
    IF (DOMCF .LT. 0.0) RETURN

    NTRY(IDX) = NTRY(IDX) + 1
    IF (ANISO) THEN
       !
       !         Pass the arrays so they start at the IDX of interest.
       !
       CALL DOMCAI(IDX,IDOMCF,NTRY(IDX), &
            EAVE((IDX-1)*6+1),DAVE((IDX-1)*15+1), &
            DX,DE,DOMCF,BETA, &
            AP,AA,AB,NACC(IDX), &
            RMX((IDX-1)*9+1:))
       RETURN
    ENDIF

    EAVE(IDX) = EAVE(IDX) + DE
    DAVE(IDX) = DAVE(IDX) + DX(1)*DX(1)
    IF (NTRY(IDX) .EQ. IDOMCF) THEN
       IF (EAVE(IDX) .GT. ZERO) THEN
          RMX(IDX) = DOMCF*SQRT(DAVE(IDX)/(BETA*EAVE(IDX)))
          IF (ALIM) RMX(IDX) = MIN(RMX(IDX),AMAX)
          NTRY(IDX) = 0
          NACC(IDX) = 0
       ELSE
          NTRY(IDX) = NTRY(IDX) - 1
          CALL ARMUPT(IDX,IDOMCF,AP,AA,AB,NTRY,NACC,ALIM,AMAX,RMX, &
               ANISO)
       ENDIF
       EAVE(IDX) = ZERO
       DAVE(IDX) = ZERO
    ENDIF

    RETURN
  END SUBROUTINE DOMCUP

  SUBROUTINE DOMCAI(IDX,IDOMCF,NTRY,EAVE,DAVE,DX,DE, &
       DOMCF,BETA,AP, AA,AB,NACC,RMX)
    !
    !       Anisotropic DOMC optimization of the move step size.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    use number
    implicit none
    INTEGER IDX, IDOMCF, NTRY, NACC
    real(chm_real) EAVE(6), DAVE(15), DX(3), DE, DOMCF
    real(chm_real) BETA, RMX(:)
    real(chm_real) AP, AA, AB
    !       Local Variables
    INTEGER I,J,INDX(6),NROT
    real(chm_real) A(6,6), D, K(6), EVLU(3), EV(3,3), RF
    real(chm_real) SCR(24), BMAT(6,6)
    LOGICAL NEGEV
    !
    CALL DOMCAD(DE,DX,EAVE,DAVE)
    IF (NTRY .EQ. IDOMCF) THEN
       !
       !         1. Set up the A matrix
       !            This is the set of 6 linear equations to be solved.
       !         2. Solve the linear equations using LU decomposition
       !            Note: EAVE gets replaced by K elements!
       !            See Numerical Recipes (1989) p. 37
       !         3. Set up the K matrix given the solution
       !            This is the K of the DOMC appendix.
       !         4. Find the K eigenvalues and eigenvectors
       !         5. Find the new RMX values (9) (The D matrix)
       !

       CALL SETUPA(A,DAVE)
       CALL MCDCMP(A,6,6,INDX,D)
       CALL MCBKSB(A,6,6,INDX,EAVE)
       CALL SETUPK(K,EAVE)

       !         Note: K is destroyed by DIAGQ!
       CALL DIAGQ(3,3,K,EV,SCR(4),SCR(7),SCR(10), &
            SCR(13),EVLU,SCR(16),SCR(19),SCR(22),0)

       NEGEV = .FALSE.
       DO I = 1, 3
          IF (EVLU(I) .LE. 0) NEGEV = .TRUE.
       ENDDO
       IF (NEGEV) THEN
          !           Negative eigenvalue.  Use the ARM instead.
          CALL ARMAIS(AP,AA,AB,NTRY,NACC,RMX)
       ELSE
          DO I = 1, 3
             RF = DOMCF*SQRT(2.0/(BETA*EVLU(I)))
             DO J = 1, 3
                RMX((J-1)*3 + I) = RF*EV(J,I)
             ENDDO
          ENDDO
          CALL SETUPK(K,EAVE)
       ENDIF

       !     
       !         Reset the bookkeeping arrays
       !
       NTRY = 0
       NACC = 0
       DO I =  1, 6
          EAVE(I) = ZERO
       ENDDO
       DO I =  1, 15
          DAVE(I) = ZERO
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE DOMCAI

  SUBROUTINE DOMCAD(DE,DX,EAVE,DAVE)
    !
    !       Adds the statistics to the appropriate place in the
    !       anisotropic DOMC bookkeeping  arrays.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    real(chm_real) DE, DX(3), EAVE(6), DAVE(15)

    EAVE(1) = EAVE(1) + DE*DX(1)*DX(1)
    EAVE(2) = EAVE(2) + DE*DX(1)*DX(2)
    EAVE(3) = EAVE(3) + DE*DX(2)*DX(2)
    EAVE(4) = EAVE(4) + DE*DX(2)*DX(3)
    EAVE(5) = EAVE(5) + DE*DX(3)*DX(3)
    EAVE(6) = EAVE(6) + DE*DX(3)*DX(1)

    DAVE( 1) = DAVE( 1) + DX(1)*DX(1)*DX(1)*DX(1)
    DAVE( 2) = DAVE( 2) + DX(1)*DX(1)*DX(1)*DX(2)
    DAVE( 3) = DAVE( 3) + DX(1)*DX(1)*DX(2)*DX(2)
    DAVE( 4) = DAVE( 4) + DX(1)*DX(1)*DX(2)*DX(3)
    DAVE( 5) = DAVE( 5) + DX(1)*DX(1)*DX(3)*DX(3)
    DAVE( 6) = DAVE( 6) + DX(1)*DX(1)*DX(3)*DX(1)
    DAVE( 7) = DAVE( 7) + DX(1)*DX(2)*DX(2)*DX(2)
    DAVE( 8) = DAVE( 8) + DX(1)*DX(2)*DX(2)*DX(3)
    DAVE( 9) = DAVE( 9) + DX(1)*DX(2)*DX(3)*DX(3)
    DAVE(10) = DAVE(10) + DX(2)*DX(2)*DX(2)*DX(2)
    DAVE(11) = DAVE(11) + DX(2)*DX(2)*DX(2)*DX(3)
    DAVE(12) = DAVE(12) + DX(2)*DX(2)*DX(3)*DX(3)
    DAVE(13) = DAVE(13) + DX(2)*DX(3)*DX(3)*DX(3)
    DAVE(14) = DAVE(14) + DX(3)*DX(3)*DX(3)*DX(3)
    DAVE(15) = DAVE(15) + DX(3)*DX(3)*DX(3)*DX(1)

    RETURN
  END SUBROUTINE DOMCAD

  SUBROUTINE SETUPA(A,DAVE)
    !
    !       Puts the DAVE elements in the right places on the A 
    !       matrix for DOMC.
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    real(chm_real) DAVE(15), A(6,6)
    !
    INTEGER I, J, IM1

    A(1,1) = DAVE( 1)*0.5
    A(1,2) = DAVE( 2)
    A(1,3) = DAVE( 3)*0.5
    A(1,4) = DAVE( 4)
    A(1,5) = DAVE( 5)*0.5
    A(1,6) = DAVE( 6)
    A(2,1) = DAVE( 2)*0.5
    A(2,2) = DAVE( 3)
    A(2,3) = DAVE( 7)*0.5
    A(2,4) = DAVE( 8)
    A(2,5) = DAVE( 9)*0.5
    A(2,6) = DAVE( 4)
    A(3,1) = DAVE( 3)*0.5
    A(3,2) = DAVE( 7)
    A(3,3) = DAVE(10)*0.5
    A(3,4) = DAVE(11)
    A(3,5) = DAVE(12)*0.5
    A(3,6) = DAVE( 8)
    A(4,1) = DAVE( 4)*0.5
    A(4,2) = DAVE( 8)
    A(4,3) = DAVE(11)*0.5
    A(4,4) = DAVE(12)
    A(4,5) = DAVE(13)*0.5
    A(4,6) = DAVE( 9)
    A(5,1) = DAVE( 5)*0.5
    A(5,2) = DAVE( 9)
    A(5,3) = DAVE(12)*0.5
    A(5,4) = DAVE(13)
    A(5,5) = DAVE(14)*0.5
    A(5,6) = DAVE(15)
    A(6,1) = DAVE( 6)*0.5
    A(6,2) = DAVE( 4)
    A(6,3) = DAVE( 8)*0.5
    A(6,4) = DAVE( 9)
    A(6,5) = DAVE(15)*0.5
    A(6,6) = DAVE( 5)

    RETURN
  END SUBROUTINE SETUPA

  SUBROUTINE SETUPK(K,EAVE)
    !
    !       Puts the EAVE elements in the right places on the K 
    !       matrix for DOMC.
    !       This routine could be eliminated by ordering them right
    !       in the first place!
    !
    !       Aaron R. Dinner
    !
    use chm_kinds
    implicit none
    real(chm_real) EAVE(6), K(6)

    K(1) = EAVE(1)
    K(2) = EAVE(2)
    K(3) = EAVE(6)
    K(4) = EAVE(3)
    K(5) = EAVE(4)
    K(6) = EAVE(5)

    RETURN
  END SUBROUTINE SETUPK

  SUBROUTINE ARMAIS(AP,AA,AB,NTRY,NACC,RMX)
    !
    !       ARM scaling of anisotropic moves.
    !
    !       Aaron R. Dinner
    !      
    use chm_kinds
    implicit none
    INTEGER NTRY, NACC
    real(chm_real)  RMX(9), AP, AA, AB, AMAX
    !       Local Variables
    INTEGER I
    real(chm_real)  RP, RF

    RP = FLOAT(NACC)/NTRY
    RF = LOG(AA*AP+AB)/LOG(AA*RP+AB)
    DO I = 1, 9
       RMX(I) = RMX(I)*RF
    ENDDO
    RETURN
  END SUBROUTINE ARMAIS

#endif 
end module mcarmupt

