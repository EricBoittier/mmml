SUBROUTINE SETLEPS(COMLYN,COMLEN)
  !
  !  routine to evaluate leps corrections to semiempirical energies
  !
  !            AB + C ===== A + BC     (1)
  !            DE + F ----> D + EF     (2)
  !----------------------------------------------------------------------
  !
  use chm_kinds
  use leps
  use stream
  use string
  use exfunc
  use number, only : zero, one
  implicit none
  !
  CHARACTER COMLYN*(*)
  INTEGER   COMLEN
  INTEGER   I
  real(chm_real),parameter:: FMARK=99999.9D0
  integer, parameter :: IMARK=99999

  !   DTM Assume 1D SVB correction and regular exponential coupling term
  SVB_DIM = 1
  GCROSS  =.FALSE.
  G1CROSS =.FALSE.
  G2CROSS =.FALSE.
  !   Assume 2D surface is simple sum of terms unless VC12 keyword is used
  SCROSS  =.FALSE.
  !   Test keyword
  TEST    =.FALSE.
  ! Default is to compute 1st derivatives 
  NDER    = 1

  !
  ! get the 3 atoms A B C number from input deck  
  !
  !   mDTM Check if 1D or 2D SVB. Check if Gaussian cross term.

  QSVB = (INDXA(COMLYN,COMLEN,'SVB').GT.0)
  IF (INDXA(COMLYN,COMLEN,'SURF').GT.0) THEN
     SVB_DIM = 2
     if(prnlev.ge.2) WRITE(OUTU,*) 'SETLEPS> 2D SVB Surface employed'
  ENDIF
  GCROSS  =(INDXA(COMLYN,COMLEN,'GCOU').GT.0)
  G1CROSS =(INDXA(COMLYN,COMLEN,'G1CO').GT.0)
  G2CROSS =(INDXA(COMLYN,COMLEN,'G2CO').GT.0)
  TEST    =(INDXA(COMLYN,COMLEN,'TEST').GT.0)
  IF (GCROSS .OR. (G1CROSS .AND. G2CROSS)) THEN
     if(prnlev.ge.2) WRITE(OUTU,*) 'SETLEPS> Gaussian term(s) employed'
     G1CROSS = .TRUE.
     G2CROSS = .TRUE.
  ELSE
     if(prnlev.ge.2) then
        IF (G1CROSS) THEN
           IF (SVB_DIM .EQ. 2) THEN
              WRITE(OUTU,*) 'SETLEPS> Gaussian R1 term employed'
              WRITE(OUTU,*) 'SETLEPS> Exponent R2 term employed'
           ELSE
              WRITE(OUTU,*) 'SETLEPS> Gaussian term employed'
           ENDIF
        ELSE IF (G2CROSS) THEN
           IF (SVB_DIM .EQ. 2) THEN
              WRITE(OUTU,*) 'SETLEPS> Exponent R1 term employed'
              WRITE(OUTU,*) 'SETLEPS> Gaussian R2 term employed'
           ELSE
              WRITE(OUTU,*) 'SETLEPS> Exponential term employed'
           ENDIF
        ELSE
           WRITE(OUTU,*) 'SETLEPS> Exponential term(s) employed'
        ENDIF
     end if
  ENDIF
  NTA = GTRMI (COMLYN,COMLEN,'LEPA', IMARK)
  NTB = GTRMI (COMLYN,COMLEN,'LEPB', IMARK)
  NTC = GTRMI (COMLYN,COMLEN,'LEPC', IMARK)

  ! get the 3 atoms D E F number from input deck !05/29/03                                                                                      
  ! 
  NTD = GTRMI (COMLYN,COMLEN,'LEPD', IMARK)
  NTE = GTRMI (COMLYN,COMLEN,'LEPE', IMARK)
  NTF = GTRMI (COMLYN,COMLEN,'LEPF', IMARK)

  IF((NTA.EQ.IMARK).OR.(NTB.EQ.IMARK)) THEN
     if(prnlev.ge.2) WRITE(OUTU,212)
     CALL WRNDIE(-3,'<SETLEPS>','Wrong LEPA or LEPB input.')
  END IF
  !   Add SVB between A and B only
  SVB_AB = (NTC.EQ.IMARK) 
  if(prnlev.ge.2) then
     IF (SVB_AB) THEN
        WRITE(OUTU,*) 'SETLEPS> A-B correction only'
     ELSE
        WRITE(OUTU,*) 'SETLEPS> A-B-C correction'
     ENDIF
  end if
  !
  IF (SVB_DIM .EQ. 2) THEN
     IF((NTD.EQ.IMARK).OR.(NTE.EQ.IMARK)) THEN
        if(prnlev.ge.2) WRITE(OUTU,213)
        CALL WRNDIE(-3,'<SETLEPS>','Wrong LEPD or LEPE input.')
     END IF
     !   Add SVB between D and E only
     SVB_DE = (NTF.EQ.IMARK)
     if(prnlev.ge.2) then
        IF (SVB_DE) THEN
           WRITE(OUTU,*) 'SETLEPS> D-E correction only'
        ELSE 
           WRITE(OUTU,*) 'SETLEPS> D-E-F correction'
        ENDIF
     end if
  ENDIF

  !
  !
  ! Initialize values
  !
  DE_1(1:6) = zero
  RE_1(1:6) = zero
  BE_1(1:6) = zero

  !   mDTM coupling term between 2 reaction coordinates
  VCROSS    = zero

  ! Get the parameters
  ! for the semiempirical surface
  ! 
  ! Bond energies
  ! 

  DE_1(1) = GTRMF (COMLYN,COMLEN, 'D1AB', FMARK)
  IF (.NOT. SVB_AB) THEN
     DE_1(2) = GTRMF (COMLYN,COMLEN, 'D1BC', FMARK)
     DE_1(3) = GTRMF (COMLYN,COMLEN, 'D1AC', FMARK)
  ENDIF
  IF (SVB_DIM .EQ. 2) THEN
     DE_1(4) = GTRMF (COMLYN,COMLEN, 'D1DE', FMARK)
     IF (.NOT. SVB_DE) THEN
        DE_1(5) = GTRMF (COMLYN,COMLEN, 'D1EF', FMARK)
        DE_1(6) = GTRMF (COMLYN,COMLEN, 'D1DF', FMARK)
     ENDIF
     !   mDTM coupling term between 2 reaction coordinates
     VCROSS = GTRMF(COMLYN,COMLEN,'VC12',FMARK)
     IF (VCROSS .NE. FMARK) THEN
        SCROSS = .TRUE.
        if(prnlev.ge.2) WRITE(OUTU,*) 'SETLEPS> 2D coupling term employed'
     ENDIF
  ENDIF

  !
  ! Equilbirium bond lengths
  !
  RE_1(1) = GTRMF (COMLYN,COMLEN, 'R1AB', FMARK)
  IF (.NOT. SVB_AB) THEN
     RE_1(2) = GTRMF (COMLYN,COMLEN, 'R1BC', FMARK)
     RE_1(3) = GTRMF (COMLYN,COMLEN, 'R1AC', FMARK)
  ENDIF
  IF (SVB_DIM .EQ. 2) THEN
     RE_1(4) = GTRMF (COMLYN,COMLEN, 'R1DE', FMARK)
     IF (.NOT. SVB_DE) THEN
        RE_1(5) = GTRMF (COMLYN,COMLEN, 'R1EF', FMARK)
        RE_1(6) = GTRMF (COMLYN,COMLEN, 'R1DF', FMARK)
     ENDIF
  ENDIF
  !
  ! Beta exponents
  !
  BE_1(1) = GTRMF (COMLYN,COMLEN, 'B1AB', FMARK)
  IF (.NOT. SVB_AB) THEN
     BE_1(2) = GTRMF (COMLYN,COMLEN, 'B1BC', FMARK)
     BE_1(3) = GTRMF (COMLYN,COMLEN, 'B1AC', FMARK)
  ENDIF
  IF (SVB_DIM .EQ. 2) THEN
     BE_1(4) = GTRMF (COMLYN,COMLEN, 'B1DE', FMARK)
     IF (.NOT. SVB_DE) THEN
        BE_1(5) = GTRMF (COMLYN,COMLEN, 'B1EF', FMARK)
        BE_1(6) = GTRMF (COMLYN,COMLEN, 'B1DF', FMARK)
     ENDIF
  ENDIF
  !
  !
  !
  NDER = GTRMI(COMLYN,COMLEN,'NDER',1)
  !     
  ! JG 5/02 extra parameters not needed if SVB 
  if(.not.QSVB) then
     !--------------------------------------------------------------
     ! commented on 05/29/03, for SVB
     !
     ! Sato parameters
     !
     SA_1(1) = GTRMF (COMLYN,COMLEN, 'S1AB', FMARK)
     SA_1(2) = GTRMF (COMLYN,COMLEN, 'S1BC', FMARK)
     SA_1(3) = GTRMF (COMLYN,COMLEN, 'S1AC', FMARK)
     !--------------------------------------------------------------
     !
     ! get the potential parameters for the dyades AB BC AC for the reference
     ! surface 
     !
     ! Bond energies
     !
     DE_2(1) = GTRMF (COMLYN,COMLEN, 'D2AB', FMARK)
     DE_2(2) = GTRMF (COMLYN,COMLEN, 'D2BC', FMARK)
     DE_2(3) = GTRMF (COMLYN,COMLEN, 'D2AC', FMARK)

     !         DE_2(4) = GTRMF (COMLYN,COMLEN, 'D2DE', FMARK)
     !         DE_2(5) = GTRMF (COMLYN,COMLEN, 'D2EF', FMARK)
     !         DE_2(6) = GTRMF (COMLYN,COMLEN, 'D2DF', FMARK)

     !
     ! Equilbirium bond lengths
     !
     RE_2(1) = GTRMF (COMLYN,COMLEN, 'R2AB', FMARK)
     RE_2(2) = GTRMF (COMLYN,COMLEN, 'R2BC', FMARK)
     RE_2(3) = GTRMF (COMLYN,COMLEN, 'R2AC', FMARK)

     !         RE_2(4) = GTRMF (COMLYN,COMLEN, 'R2DE', FMARK)
     !         RE_2(5) = GTRMF (COMLYN,COMLEN, 'R2EF', FMARK)
     !         RE_2(6) = GTRMF (COMLYN,COMLEN, 'R2DF', FMARK)
     !
     !
     ! Beta exponents
     !
     BE_2(1) = GTRMF (COMLYN,COMLEN, 'B2AB', FMARK)
     BE_2(2) = GTRMF (COMLYN,COMLEN, 'B2BC', FMARK)
     BE_2(3) = GTRMF (COMLYN,COMLEN, 'B2AC', FMARK)

     !         BE_2(4) = GTRMF (COMLYN,COMLEN, 'B2DE', FMARK)
     !         BE_2(5) = GTRMF (COMLYN,COMLEN, 'B2EF', FMARK)
     !         BE_2(6) = GTRMF (COMLYN,COMLEN, 'B2DF', FMARK)

     !--------------------------------------------------------------
     ! commented on 05/29/03, for SVB
     !
     ! Sato parameters
     !
     SA_2(1) = GTRMF (COMLYN,COMLEN, 'S2AB', FMARK)
     SA_2(2) = GTRMF (COMLYN,COMLEN, 'S2BC', FMARK)
     SA_2(3) = GTRMF (COMLYN,COMLEN, 'S2AC', FMARK)
     !--------------------------------------------------------------
     !
     ! Write out the parameters to define both surfaces
     !
     IF (PRNLEV.GE.2) THEN
        WRITE(OUTU,211)
        WRITE(OUTU,210)
        WRITE(OUTU,211)   
     ENDIF
  end if      ! .not.QSVB
  !
  !  Echo the potential parameters
  ! SVB term : 
  if(PRNLEV.GE.2) then
     IF (QSVB) THEN
        IF (SVB_DIM .EQ. 2) THEN
           WRITE (OUTU,105) (DE_1(I),I=1,6), (RE_1(I),I=1,6),(BE_1(I),I=1,6), VCROSS 
        ELSE 
           WRITE (OUTU,106) (DE_1(I),I=1,3), (RE_1(I),I=1,3),(BE_1(I),I=1,3)
        ENDIF
     ELSE
        WRITE (OUTU,100) DE_1, RE_1, BE_1, SA_1
        WRITE (OUTU,101) DE_2, RE_2, BE_2, SA_2
     END IF
  end if
  !
100 FORMAT(/,1X,'**','Semiempirical Potential Energy Surface',1X,'**',//,1X,T5,'LEPS potential energy surface', &
       //,1X,T5,'Parameters:', &
       /, 2X, T5, 'Bond', T46, 'C4-H', T58, 'H-Cb', T69, 'C4-Cb', &
       /, 2X, T5, 'Dissociation energies (kcal/mol):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Equilibrium bond lengths (Angstroms):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Morse beta parameters (Angstroms**-1):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Sato parameters:',T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'*****')
  !
101 FORMAT(/,1X,'**','Reference Potential Energy Surface',1X,'**',//,1X,T5,'LEPS potential energy surface', &
       //,1X,T5,'Parameters:', &
       /, 2X, T5, 'Bond', T46, 'C4-H', T58, 'H-Cb', T69, 'C4-Cb', &
       /, 2X, T5, 'Dissociation energies (kcal/mol):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Equilibrium bond lengths (Angstroms):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Morse beta parameters (Angstroms**-1):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Sato parameters:',T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'*****') 
  !
105 FORMAT(/,1X,'**','Semiempirical Potential Energy Surface',1X,'**',//,1X,T5,'SVB 2D correction term', &
       //,1X,T5,'Parameters:', &
       /, 2X, T5, 'Term', T47, 'M1  ', T58, 'M2  ', T69, 'V12 ', &
       /, 2X, T5, 'Energies for reaction (1)(kcal/mol):',T44, F10.5, T55, F10.5, T66, F10.5, & 
       /, 2X, T5, 'Energies for reaction (2)(kcal/mol):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Equil bond lengths for (1)(Angstroms):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Equil bond lengths for (2)(Angstroms):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Exponential parameters for (1)(A**-1):',T44, F10.5, T55, F10.5, T66, F10.5,     & 
       /, 2X, T5, 'Exponential parameters for (2)(A**-1):',T44, F10.5, T55, F10.5, T66, F10.5, &
       /, 2X, T5, 'Coupling term for (1) & (2)(kcal/mol):',T44, F10.5//,1X,'********')
  !
106 FORMAT(/,1X,'**','Semiempirical Potential Energy Surface',1X,'**',//,1X,T5,'SVB 1D correction term', &
       //,1X,T5,'Parameters:', &
       /, 2X, T5, 'Term', T47, 'M1  ', T58, 'M2  ', T69, 'V12 ', &
       /, 2X, T5, 'Energies for reaction (kcal/mol):',T44, F10.5, T55, F10.5, T66, F10.5, & 
       /, 2X, T5, 'Equil bond lengths for (Angstroms):',T44, F10.5, T55, F10.5, T66, F10.5, & 
       /, 2X, T5, 'Exponential parameters for (A**-1):',T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'********')

211 FORMAT(/,'****************************************************',/)
210 FORMAT(' SETLEPS> SETUP OF THE HIGH AND LOW LEVEL LEPS SURFACES')
212 FORMAT(' SETLEPS> Check defintion of NTA NTB or NTB: bad values')
213 FORMAT(' SETLEPS> Check defintion of NTD NTE or NTF: bad values',/, 1X, &
       'Use SURF keyword and define D,E,F for 2D SVB')
  !
  RETURN 
END SUBROUTINE SETLEPS
!
!-----------------------------------------------------------------------------
!
SUBROUTINE CORRECT_LEPS(E_LEPS,DA_LEPS,DB_LEPS,DC_LEPS)
  !
  ! distances in atomic units 
  ! energies come in atomic units
  !
  !-----------------------------------------------------------------------------
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : zero, one
  use leps
  implicit none
  real(chm_real) E_LEPS,DA_LEPS(3),DB_LEPS(3),DC_LEPS(3)
  !
  real(chm_real) :: EC_AM1, EC_REF   
  real(chm_real) :: xyz_ab(3),xyz_cb(3),xyz_ac(3),r_ab2,r_cb2,r_ac2,rR_leps(3)
  real(chm_real) :: dxyz_ab(3),dxyz_cb(3),dxyz_ac(3)
  real(chm_real) :: DEDR_1(3),DEDR_2(3),DEDR_C(3)
  !
  real(chm_real),parameter :: SCAL=1.0d0        ! THIS IS A SCALING FACTOR FOR THE GRADIENTS
  real(chm_real),parameter :: F1  = TOKCAL, &   ! 627.5095D0
       F2  = BOHRR , &   ! 0.529177106D0
       r_F2= one/F2, &
       F3  = F1/F2

  !
  !====      real(chm_real) :: X(3),COUL(3), EXCH(3)
  !====      real(chm_real), parameter :: R2 = 1.41421356D0
  !
  !
  !====      real(chm_real) R, ENERGY, DEDR
  !====      COMMON /PT1CM/ R(3), ENERGY, DEDR(3)
  !
  !
  !               <--------
  ! GET DISTANCES NTA - NTB   , THEY SHOULD GO IN ATOMIC UNITS
  !               NTC - NTB 
  !               NTA - NTC 

  xyz_ab(1:3) = XLA(1:3) - XLB(1:3)
  xyz_cb(1:3) = XLC(1:3) - XLB(1:3)
  xyz_ac(1:3) = XLA(1:3) - XLC(1:3)
  r_ab2       = xyz_ab(1)*xyz_ab(1)+xyz_ab(2)*xyz_ab(2)+xyz_ab(3)*xyz_ab(3)
  r_cb2       = xyz_cb(1)*xyz_cb(1)+xyz_cb(2)*xyz_cb(2)+xyz_cb(3)*xyz_cb(3)
  r_ac2       = xyz_ac(1)*xyz_ac(1)+xyz_ac(2)*xyz_ac(2)+xyz_ac(3)*xyz_ac(3)
  R_leps(1) = SQRT( r_ab2 )
  R_leps(2) = SQRT( r_cb2 )
  R_leps(3) = SQRT( r_ac2 )
  rR_leps(1:3)=one/R_leps(1:3)

  dxyz_ab(1:3)=xyz_ab(1:3)*rR_leps(1)
  dxyz_cb(1:3)=xyz_cb(1:3)*rR_leps(2)
  dxyz_ac(1:3)=xyz_ac(1:3)*rR_leps(3)

  R_leps(1:3) = R_leps(1:3)*r_F2     
  !
  ! GET THE AM1 LEPS ENERGY
  !
  CALL PREPOT('AM1')
  CALL POT

  EC_AM1 = ENERGY_leps
  DEDR_1(1:3)=DEDR_leps(1:3)
  !
  ! GET THE REFERENCE LEPS ENERGY
  !
  CALL PREPOT('REF')
  CALL POT 

  EC_REF = ENERGY_leps 
  DEDR_2(1:3)=DEDR_leps(1:3)
  !
  ! GET THE CORRECTION AS, VCOR = V_LEPS_REF - V_LEPS_AM1 
  !
  E_LEPS = EC_REF - EC_AM1
  E_LEPS = E_LEPS * F1
  DEDR_C(1:3)= one*(DEDR_2(1:3)-DEDR_1(1:3))*F3*SCAL
  !      WRITE(6,*) ' CORR  EC_REF EC_AM1 ' , E_LEPS, EC_REF ,EC_AM1
  !
  ! GET (DR/DXI) CONTRIBUTION TO THE DERIVATIVES
  !
  DA_LEPS(1:3)= DEDR_C(1)*dxyz_ab(1:3)+DEDR_C(3)*dxyz_ac(1:3)
  DB_LEPS(1:3)=-one*(DEDR_C(2)*dxyz_cb(1:3)+DEDR_C(1)*dxyz_ab(1:3))
  DC_LEPS(1:3)=-DEDR_C(3)*dxyz_ac(1:3)+DEDR_C(2)*dxyz_cb(1:3)

  !      WRITE(*,40) R_leps(1)*F2 , R_leps(2)*F2, R_leps(3)*F3, ENERGY_leps*F1
  !      WRITE(*,40) ' DERIVATIVES ',DEDR_leps(1)*F3,DEDR_leps(2)*F3,DEDR_leps(3)*F3
  ! 40   FORMAT(2X,A,3F20.5) 

  RETURN
END SUBROUTINE CORRECT_LEPS

!
SUBROUTINE PREPOT(FLAG)
  !
  !   System:           BrH2
  !   Functional form:  LEPS (London-Erying-Polyani-Sato)
  !                     
  !   Calling Sequence: 
  !      PREPOT - initializes the potential's variables and
  !               must be called once before any calls to POT
  !      POT    - driver for the evaluation of the energy and the derivatives 
  !               of the energy with respect to the coordinates for a given 
  !               geometric configuration
  !
  !   Units: 
  !      energies    - hartrees
  !      coordinates - bohrs
  !      derivatives - hartrees/bohr
  !
  !   Zero of energy: 
  !      The classical potential energy is set equal to zero when the Br
  !      atom is infinitely far from the H2 diatomic with R(H2) set equal to
  !      the H2 equilibrium diatomic value.
  !
  !   Parameters:
  !      Set in the BLOCK DATA subprogram PTPARM
  !
  !   Coordinates:
  !      Internal, Definition: R(1) = R(Br-H)
  !                            R(2) = R(H-H)
  !                            R(3) = R(Br-H)
  !
  !                            r(1) =r(Cn - H)
  !                            r(2) =r(H  - Cs)
  !                            r(3) =r(Cn - Cs)
  !
  !   Common Blocks (used between the calling program and this potential):
  !      /PT1CM/ R_leps(3), ENERGY_leps, DEDR_leps(3)
  !        passes the coordinates, ground state electronic energy, and 
  !        derivatives of the ground electronic state energy with respect 
  !        to the coordinates.
  !      /PT2CM/ NSURF_leps, NDUM_leps(21)
  !        passes the control flags where
  !        NDER_leps  = 0 => no derivatives are computed
  !                   = 1 => derivatives of the energy for the ground electronic 
  !                          state with respect to the coordinates are computed
  !        NDUM_leps  - not used 
  !      /PT4CM/ IPTPRT_leps, IDUM_leps(19)
  !        IPTPRT_leps passes the FORTRAN unit number used for potential output
  !        IDUM_leps   not used
  !      /PT5CM/ EASYAB_leps, EASYBC_leps, EASYAC_leps
  !        passes the energy in the three asymptotic valleys for an A + BC system.
  !        The energy in the AB valley, EASYAB_leps, is equal to the energy of the 
  !        C atom "infinitely" far from the AB diatomic and R_leps(AB) set equal to 
  !        Re(AB), the equilibrium bond length for the AB diatomic.  
  !        In this potential the AB valley represents H infinitely far from
  !        the BrH diatomic and R_leps(BrH) equal to Re(BrH).  Similarly, the terms
  !        EASYBC_leps and EASYAC_leps represent the energies in the H2 and the BrH 
  !        valleys, respectively.
  !
  !   Default Parameter Values:
  !      Variable      Default value
  !      NSURF_leps        0
  !      NDER_leps         1 
  !      IPTPRT_leps       6
  !
  !*****
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : zero, one, two, three, half, PT25
  use leps
  use stream
  implicit none
  CHARACTER(LEN=3):: FLAG
  !
  integer :: I
  real(chm_real),parameter :: CKCAL=TOKCAL, &      ! 627.5095D0
       CANGS=BOHRR,  &      ! 0.529177106D0
       r_CKCAL=one/CKCAL, r_CANGS=one/CANGS

  !====      real(chm_real) :: X(3), COUL(3), EXCH(3)
  !====      real(chm_real),parameter :: R2 = 1.41421356D0
  !====
  !====!
  !====      real(chm_real) EASYAB, EASYBC, EASYAC, DE, RE, BETA, Z
  !====      real(chm_real) ZPO,OP3Z,ZP3,TZP3,TOP3Z, DO4Z, B
  !====!
  !====      COMMON /PT5CM/ EASYAB, EASYBC, EASYAC
  !====      COMMON /SATOCM/ DE(3), RE(3), BETA(3), Z(3)
  !====      COMMON /LEPSCM/ ZPO(3), OP3Z(3), ZP3(3), TZP3(3), TOP3Z(3), &
  !====                      DO4Z(3), B(3)
  !
  IF(FLAG.EQ.'AM1') THEN
     DE_leps(1:3)  = DE_1(1:3) 
     RE_leps(1:3)  = RE_1(1:3)
     BETA_leps(1:3)= BE_1(1:3)
     Z_leps(1:3)   = SA_1(1:3)
  ELSE IF (FLAG.EQ.'REF') THEN
     DE_leps(1:3)  = DE_2(1:3)
     RE_leps(1:3)  = RE_2(1:3)
     BETA_leps(1:3)= BE_2(1:3)
     Z_leps(1:3)   = SA_2(1:3)
  ELSE 
     WRITE(OUTU,*) ' PREPOT > THERE IS A MISTAKE WITH THE SETUP IN LEPS'
  END IF
  !
!!!   ECHO THE POTENTIAL PARAMETERS
!!!        WRITE (IPTPRT_leps, 100) DE_leps, RE_leps, BETA_leps, Z_leps
!!!
!!!
!!!100   FORMAT(/,1X,'*****','POTENTIAL ENERGY SURFACE',1X,'*****', &
!!!            //,1X,T5,'HYDRIDE TRANSFER LEPS POTENTIAL ENERGY SURFACE', &
!!!            //,1X,T5,'PARAMETERS:', &
!!!              /, 2X, T5, 'BOND', T46, 'C4-H', T58, 'H-CB', T69, 'C4-CB', &
!!!              /, 2X, T5, 'DISSOCIATION ENERGIES (KCAL/MOL):',T44, F10.5, T55, F10.5, T66, F10.5, &
!!!              /, 2X, T5, 'EQUILIBRIUM BOND LENGTHS (ANGSTROMS):',T44, F10.5, T55, F10.5, T66, F10.5, &
!!!              /, 2X, T5, 'MORSE BETA PARAMETERS (ANGSTROMS**-1):',T44, F10.5, T55, F10.5, T66, F10.5, &
!!!              /, 2X, T5, 'SATO PARAMETERS:',T44, F10.5, T55, F10.5, T66, F10.5,//,1X,'*****')

  !
  do I = 1,3
     !   CONVERT TO ATOMIC UNITS
     DE_leps(I)    = DE_leps(I)*r_CKCAL       ! /CKCAL
     RE_leps(I)    = RE_leps(I)*r_CANGS       ! /CANGS
     BETA_leps(I)  = BETA_leps(I)*CANGS
     !   COMPUTE USEFUL CONSTANTS
     ZPO_leps(I)   = one + Z_leps(I)
     OP3Z_leps(I)  = one + three*Z_leps(I)
     TOP3Z_leps(I) = two*OP3Z_leps(I)
     ZP3_leps(I)   = Z_leps(I) + three
     TZP3_leps(I)  = two*ZP3_leps(I)
     DO4Z_leps(I)  = DE_leps(I)*PT25/ZPO_leps(I)   ! /4.0D0
     B_leps(I)     = BETA_leps(I)*DO4Z_leps(I)*two
  end do
  !
  !   INITIALIZE THE ASYMPTOTIC ENERGY VALUES
  EASYAB_leps = DE_leps(1)
  EASYBC_leps = DE_leps(2)
  EASYAC_leps = DE_leps(3)
  !
  RETURN
END SUBROUTINE PREPOT
!*****
!
SUBROUTINE POT
  !
  !   System:          ABC
  !   Functional form: LEPS (London-Erying-Polyani-Sato)
  !   References:      S. Sato
  !                    J. Chem. Phys. 592, 2465 (1955)
  !                    
  !
  !   The potential parameters must be passed through the common blocks
  !   PT1CM, PT5CM, PT2CM, PT4CM, SATOCM, and LEPSCM.  
  !   All information passed through the common blocks PT1CM, PT5CM, 
  !   SATOCM, and LEPSCM must be in Hartree atomic units.
  !
  !        For the reaction: A + BC -> AB + C we write:
  !                          R(1) = R(A-B)
  !                          R(2) = R(B-C)
  !                          R(3) = R(C-A)
  !
  !   NOTE: The potential energy at the reactant asympotote, that is at 
  !         A infinitely far from the BC diatomic, BC diatomic at its
  !         equilibrium configuration, is set equal to zero.
  !
  !        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
  !
  use chm_kinds
  use number, only : zero, one, two, three
  use leps
  implicit none

  real(chm_real) :: RAD, S, OUL, r_RAD
  real(chm_real) :: X(3), COUL(3), EXCH(3)
  real(chm_real), parameter :: R2 = 1.41421356D0, r_R2=one/R2
  !
  INTEGER I

  !====      INTEGER IPTPRT
  !====      real(chm_real) R, ENERGY, DEDR, EASYAB, EASYBC, EASYAC, DE, RE, BETA, Z
  !====      real(chm_real) ZPO, OP3Z, ZP3, TZP3, TOP3Z, DO4Z, B
  !====!
  !====      COMMON /PT4CM/ IPTPRT
  !====      COMMON /PT1CM/ R(3), ENERGY, DEDR(3)
  !====      COMMON /PT5CM/ EASYAB, EASYBC, EASYAC
  !====      COMMON /SATOCM/ DE(3), RE(3), BETA(3), Z(3) 
  !====      COMMON /LEPSCM/ ZPO(3), OP3Z(3), ZP3(3), TZP3(3), TOP3Z(3),  &
  !====                      DO4Z(3), B(3)
  !
  !   INITIALIZE THE VARIABLE USED IN THE CALCULATION OF THE ENERGY.
  !
  ENERGY_leps = zero
  !
  !   CHECK THE VALUE OF NDER 
  IF (NDER .GT. 1) THEN
     WRITE (IPTPRT_leps, 900) NDER
     CALL WRNDIE(-5,'<POT>','POT 1: Wrong NDER, only 1st derivatives are coded.')
  ENDIF
  !
  !   COMPUTE THE ENERGY.
  !
  do I = 1,3
     X(I)       = EXP(-BETA_leps(I)*(R_leps(I)-RE_leps(I)))
     COUL(I)    = DO4Z_leps(I)*(ZP3_leps(I)*X(I)-TOP3Z_leps(I))*X(I)
     EXCH(I)    = DO4Z_leps(I)*(OP3Z_leps(I)*X(I)-TZP3_leps(I))*X(I)
     ENERGY_leps= ENERGY_leps + COUL(I)
  end do
  !
  RAD = SQRT((EXCH(1)-EXCH(2))**2 + (EXCH(2)-EXCH(3))**2 + (EXCH(3)-EXCH(1))**2)
  r_RAD=one/RAD
  !
  ENERGY_leps = ENERGY_leps - RAD*r_R2 + EASYBC_leps     ! /R2
  !
  !   COMPUTE THE DERIVATIVES OF THE ENERGY WITH RESPECT TO THE INTERNAL
  !   COORDINATES.
  !
  IF (NDER .EQ. 1) THEN
     S = EXCH(1) + EXCH(2) + EXCH(3)
     do I = 1,3
        DEDR_leps(I)=B_leps(I)*X(I)*( (three*EXCH(I)-S)*r_R2*(OP3Z_leps(I)*X(I)-ZP3_leps(I))*r_RAD &
             -ZP3_leps(I)*X(I)+OP3Z_leps(I) ) ! /R2, /RAD
     end do
  ENDIF
  !
900 FORMAT(/,1X,T5,'ERROR: POT HAS BEEN CALLED WITH NDER = ', I5, &
       /,1X,T12,'ONLY THE FIRST DERIVATIVES, NDER = 1, ARE ','CODED IN THIS POTENTIAL')
  !
  RETURN
END SUBROUTINE POT
!*****
!
!
!-----------------------------------------------------------------------------
!
SUBROUTINE QM_SVB1D(E_LEPS,DA_LEPS,DB_LEPS,DC_LEPS)
  !
  ! distances in atomic units 
  ! energies come in atomic units
  !
  !-----------------------------------------------------------------------------
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : zero, one, two
  use leps
  implicit none
  real(chm_real) E_LEPS,DA_LEPS(3),DB_LEPS(3),DC_LEPS(3)
  !
  real(chm_real) ENERGY
  real(chm_real) DEDR(3),DE(3),RE(3),BETA(3)
  INTEGER :: I,J
  real(chm_real) :: r_r(3),r(3),xyz_ab(3),xyz_cb(3),xyz_ac(3),r_ab2,r_cb2,r_ac2
  real(chm_real) :: dxyz_ab(3),dxyz_cb(3),dxyz_ac(3)
  real(chm_real) :: DEDR_1(3),DEDR_C(3)
  !
  real(chm_real), parameter :: SCAL=1.0D0, &        ! THIS IS A SCALING FACTOR FOR THE GRADIENTS
       F1 = TOKCAL, &       ! 627.5095D0
       F2 = BOHRR, &        ! 0.529177106D0
       F3 = F1/F2, &
       r_F1=one/F1, r_F2=one/F2
  !
  DEDR(1:3) = zero
  !
  !               <--------
  ! GET DISTANCES NTA - NTB   , THEY SHOULD GO IN ATOMIC UNITS
  !               NTC - NTB 
  !               NTA - NTC 

  !   SIMPLE A-B CORRECTION
  IF (SVB_AB) THEN
     xyz_ab(1:3) = XLA(1:3) - XLB(1:3)           
     r_ab2 = xyz_ab(1)*xyz_ab(1)+xyz_ab(2)*xyz_ab(2)+xyz_ab(3)*xyz_ab(3)

     r(1) = SQRT(r_ab2) 
     r(2) = zero
     r(3) = zero
     r_r(1)=one/r(1)

     dxyz_ab(1:3) = xyz_ab(1:3)*r_r(1)
     dxyz_cb(1:3) = zero
     dxyz_ac(1:3) = zero

     r(1) = r(1)*r_F2

  ELSE
     xyz_ab(1:3) = XLA(1:3) - XLB(1:3)
     xyz_cb(1:3) = XLC(1:3) - XLB(1:3)
     xyz_ac(1:3) = XLA(1:3) - XLC(1:3)
     r_ab2 = xyz_ab(1)*xyz_ab(1)+xyz_ab(2)*xyz_ab(2)+xyz_ab(3)*xyz_ab(3)
     r_cb2 = xyz_cb(1)*xyz_cb(1)+xyz_cb(2)*xyz_cb(2)+xyz_cb(3)*xyz_cb(3)
     r_ac2 = xyz_ac(1)*xyz_ac(1)+xyz_ac(2)*xyz_ac(2)+xyz_ac(3)*xyz_ac(3)

     r(1) = SQRT(r_ab2)
     r(2) = SQRT(r_cb2)
     r(3) = SQRT(r_ac2)

     !   CDTM 1ST REACTION COORDINATE
     if (G1CROSS) r(3) = r(1) - r(2)
     r_r(1:3)=one/r(1:3)

     !   CDTM FOR DERIVATIVES
     dxyz_ab(1:3) = xyz_ab(1:3)*r_r(1)
     dxyz_cb(1:3) = xyz_cb(1:3)*r_r(2)
     dxyz_ac(1:3) = xyz_ac(1:3)*r_r(3)
     r(1:3) = r(1:3)*r_F2
  ENDIF

  !
  ! GET THE SVB ENERGY CORRECTION
  !
  CALL PREPOT_SVB1D(DE,RE,BETA)
  CALL POT_SVB1D(ENERGY,DEDR,DE,R,RE,BETA)

  !      WRITE(*,20)'DTM> R1-R6,E',R(1),R(2),ENERGY
  !20    FORMAT(2X,A,4F10.5)

  DEDR_1(1:3)= DEDR(1:3)

  !
  ! GET THE CORRECTION , 
  !
  E_LEPS = ENERGY
  E_LEPS = E_LEPS * F1

  !CC   PRINTING FOR TESTS
  IF (TEST) THEN 
     IF (SVB_AB) THEN
        WRITE(*,30) 'SVB ENERGY CORRECTION (R,E): ',R(1)*F2,E_LEPS
     ELSE
        WRITE(*,30) 'SVB ENERGY CORRECTION (R,E): ',(R(1)-R(2))*F2,E_LEPS
     ENDIF
  ENDIF
30 FORMAT(2X,A,2F10.5)

  DEDR_C(1:3)=DEDR_1(1:3)*F3*SCAL
  !
  ! GET (DR/DXI) CONTRIBUTION TO THE DERIVATIVES
  !
  IF (G1CROSS) THEN
     DA_LEPS(1:3) = DEDR_C(1)*dxyz_ab(1:3) + DEDR_C(3)*dxyz_ab(1:3)
     DB_LEPS(1:3) =-one*((DEDR_C(2)-DEDR_C(3))*dxyz_cb(1:3) + (DEDR_C(1)+DEDR_C(3))*dxyz_ab(1:3))
     DC_LEPS(1:3) =-DEDR_C(3)*dxyz_cb(1:3) + DEDR_C(2)*dxyz_cb(1:3)
  ELSE
     DA_LEPS(1:3) = DEDR_C(1)*dxyz_ab(1:3) + DEDR_C(3)*dxyz_ac(1:3)
     DB_LEPS(1:3) =-one*(DEDR_C(2)*dxyz_cb(1:3) + DEDR_C(1)*dxyz_ab(1:3))
     DC_LEPS(1:3) =-DEDR_C(3)*dxyz_ac(1:3) + DEDR_C(2)*dxyz_cb(1:3)
  ENDIF

  !        WRITE(*,50) 'DTM> DIST ', R(1)*F2 , R(2)*F2, R(3)*F2
  !        WRITE(*,40) ' DERIVATIVES ',DEDR(1)*F3,DEDR(2)*F3,DEDR(3)*F3
  !CC   DERIVATIVES PRINT
  IF (TEST) THEN
     WRITE(*,40)'LEPS> DAXYZ ',(DA_LEPS(I),I=1,3)
     WRITE(*,40)'LEPS> DBXYZ ',(DB_LEPS(I),I=1,3)
     WRITE(*,40)'LEPS> DCXYZ ',(DC_LEPS(I),I=1,3)
     !         WRITE(*,60)'LEPS> ENERGY ', ENERGY*F1 
  ENDIF

40 FORMAT(2X,A,3F20.5) 
50 FORMAT(2X,A,4F15.5)
60 FORMAT(2X,A,1F15.5)
  RETURN
END SUBROUTINE QM_SVB1D

SUBROUTINE QM_SVB2D(E_LEPS,DA_LEPS_local,DB_LEPS_local,DC_LEPS_local,DD_LEPS_local,DE_LEPS_local,DF_LEPS_local)
  !
  ! distances in atomic units 
  ! energies come in atomic units
  !
  !-----------------------------------------------------------------------------
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : zero, one
  use leps
  implicit none
  real(chm_real) E_LEPS,DA_LEPS_local(3),DB_LEPS_local(3),DC_LEPS_local(3)
  real(chm_real) DD_LEPS_local(3),DE_LEPS_local(3),DF_LEPS_local(3)
  !
  real(chm_real) ENERGY, VC12, DEDR(6), DE(6), RE(6), BETA(6)

  integer :: I, J
  real(chm_real),parameter :: SCAL=1.0D0        ! THIS IS A SCALING FACTOR FOR THE GRADIENTS
  real(chm_real),parameter :: F1 = TOKCAL, &    ! 627.5095D0
       F2 = BOHRR, &     ! 0.529177106D0
       F3 = F1/F2, &
       r_F1=one/F1, r_F2=one/F2
  real(chm_real) :: xyz_ab(3),xyz_cb(3),xyz_ac(3),r_ab2,r_cb2,r_ac2, &
       dxyz_ab(3),dxyz_cb(3),dxyz_ac(3), &
       xyz_de(3),xyz_fe(3),xyz_df(3),r_de2,r_fe2,r_df2, &
       dxyz_de(3),dxyz_fe(3),dxyz_df(3), &
       r(6),r_r(6),DEDR_1(6),DEDR_C(6)


  !====!
  !====!      real(chm_real) R, DEDR, I
  !====!      real(chm_real) EASYAB, EASYBC, EASYAC, EASYDE, EASYEF, EASYDF
  !====!      real(chm_real) DE, RE, BETA, Z                     
  !====!
  !====!
  !====!      INTEGER IPTPRT, NSURF, NDUM 
  !====!
  !====!      COMMON /PT4CM_2D/ IPTPRT
  !====!      COMMON /PT1CM_2D/ R(6), ENERGY, DEDR(6)
  !====!      COMMON /PT2CM_2D/ NSURF, NDUM(21)
  !====!      COMMON /PT5CM_2D/ EASYAB,EASYBC,EASYAC,EASYDE,EASYEF,EASYDF
  !====!      COMMON /SATOCM_2D/ DE(6), RE(6), BETA(6), Z(6),VC12
  !====!

  DEDR(1:6) = zero
  !
  !               <--------
  ! GET DISTANCES NTA - NTB   , THEY SHOULD GO IN ATOMIC UNITS
  !               NTC - NTB 
  !               NTA - NTC 


  !   SIMPLE A-B CORRECTION
  IF (SVB_AB) THEN
     xyz_ab(1:3)=XLA(1:3) - XLB(1:3)
     r_ab2 = xyz_ab(1)*xyz_ab(1)+xyz_ab(2)*xyz_ab(2)+xyz_ab(3)*xyz_ab(3)
     r(1)  = SQRT(r_ab2)
     r(2:3)= zero

     r_r(1)= one/r(1)
     r(1)  = r(1)*r_F2

     dxyz_ab(1:3)= xyz_ab(1:3)*r_r(1)
     dxyz_cb(1:3)= zero
     dxyz_ac(1:3)= zero

     !   A-B-C CORRECTION
  ELSE
     xyz_ab(1:3)= XLA(1:3) - XLB(1:3)
     xyz_cb(1:3)= XLC(1:3) - XLB(1:3)
     xyz_ac(1:3)= XLA(1:3) - XLC(1:3)
     r_ab2 = xyz_ab(1)*xyz_ab(1)+xyz_ab(2)*xyz_ab(2)+xyz_ab(3)*xyz_ab(3)
     r_cb2 = xyz_cb(1)*xyz_cb(1)+xyz_cb(2)*xyz_cb(2)+xyz_cb(3)*xyz_cb(3)
     r_ac2 = xyz_ac(1)*xyz_ac(1)+xyz_ac(2)*xyz_ac(2)+xyz_ac(3)*xyz_ac(3)

     r(1) = SQRT(r_ab2)
     r(2) = SQRT(r_cb2)
     r(3) = SQRT(r_ac2)
     !   CDTM 1ST REACTION COORDINATE
     if (G1CROSS) r(3) = r(1) - r(2)
     r_r(1:3)=one/r(1:3)

     !   CDTM FOR DERIVATIVES
     dxyz_ab(1:3) = xyz_ab(1:3)*r_r(1)
     dxyz_cb(1:3) = xyz_cb(1:3)*r_r(2)
     dxyz_ac(1:3) = xyz_ac(1:3)*r_r(3)

     r(1:3) = r(1:3)*r_F2
  ENDIF

  !               <--------
  ! GET DISTANCES NTD - NTE   , THEY SHOULD GO IN ATOMIC UNITS
  !               NTE - NTF
  !               NTD - NTF

  !   SIMPLE D-E CORRECTION
  IF (SVB_DE) THEN
     xyz_de(1:3) = XLD(1:3) - XLE(1:3)
     r_de2 = xyz_de(1)*xyz_de(1)+xyz_de(2)*xyz_de(2)+xyz_de(3)*xyz_de(3)
     r(4)  = sqrt(r_de2)
     r(5:6)= zero
     r_r(4)= one/r(4)

     dxyz_de(1:3) = xyz_de(1:3)*r_r(4)
     dxyz_fe(1:3) = zero
     dxyz_df(1:3) = zero
     r(4)  = r(4)*r_F2

  ELSE
     xyz_de(1:3) = XLD(1:3) - XLE(1:3)
     xyz_fe(1:3) = XLF(1:3) - XLE(1:3)
     xyz_df(1:3) = XLD(1:3) - XLF(1:3)
     r_de2 = xyz_de(1)*xyz_de(1)+xyz_de(2)*xyz_de(2)+xyz_de(3)*xyz_de(3)
     r_fe2 = xyz_fe(1)*xyz_fe(1)+xyz_fe(2)*xyz_fe(2)+xyz_fe(3)*xyz_fe(3)
     r_df2 = xyz_df(1)*xyz_df(1)+xyz_df(2)*xyz_df(2)+xyz_df(3)*xyz_df(3)
     r(4) = sqrt(r_de2)
     r(5) = sqrt(r_fe2)
     r(6) = sqrt(r_df2)

     !   CDTM 2ND REACTION COORDINATE
     if (G2CROSS) r(6) = r(4) - r(5)
     r_r(4:6) = one/r(4:6)

     !   CDTM FOR DERIVATIVES
     dxyz_de(1:3) = xyz_de(1:3)*r_r(4)
     dxyz_fe(1:3) = xyz_fe(1:3)*r_r(5)
     dxyz_df(1:3) = xyz_df(1:3)*r_r(6)

     r(4:6) = r(4:6)*r_F2
  ENDIF

  !
  ! GET THE SVB ENERGY CORRECTION
  !
  CALL PREPOT_SVB2D(DE,RE,BETA)
  CALL POT_SVB2D(ENERGY,DEDR,DE,R,RE,BETA)

  !       WRITE(*,20)'DTM> R1-R6,E',R(1),R(2),R(3),R(4),R(5),R(6),ENERGY
  !20     FORMAT(2X,A,7F10.5)

  DEDR_1(1:6)= DEDR(1:6)
  !
  !
  ! GET THE CORRECTION , 
  !
  E_LEPS = ENERGY
  E_LEPS = E_LEPS * F1

  !CC   PRINTING FOR TESTS
  IF (TEST) THEN
     IF (SVB_AB .AND. SVB_DE) THEN
        WRITE(*,30) 'SVB ENERGY CORRECTION (R1,R2,E): ',R(1)*F2,R(4)*F2,E_LEPS
     ELSE IF (SVB_AB) THEN
        WRITE(*,30) 'SVB ENERGY CORRECTION (R1,R2,E): ',R(1)*F2,(R(4)-R(5))*F2,E_LEPS
     ELSE IF (SVB_DE) THEN
        WRITE(*,30) 'SVB ENERGY CORRECTION (R1,R2,E): ',(R(1)-R(2))*F2,R(4)*F2,E_LEPS
     ELSE
        WRITE(*,30) 'SVB ENERGY CORRECTION (R1,R2,E): ',(R(1)-R(2))*F2,(R(4)-R(5))*F2,E_LEPS
     ENDIF
  ENDIF
30 FORMAT(2X,A,3F10.5)

  DEDR_C(1:6)= DEDR_1(1:6)*F3*SCAL
  !
  ! GET (DR/DXI) CONTRIBUTION TO THE DERIVATIVES
  !
  IF (G1CROSS) THEN
     DA_LEPS_local(1:3) = DEDR_C(1)*dxyz_ab(1:3) + DEDR_C(3)*dxyz_ab(1:3)
     DB_LEPS_local(1:3) =-one*((DEDR_C(2)-DEDR_C(3))*dxyz_cb(1:3) + (DEDR_C(1)+DEDR_C(3))*dxyz_ab(1:3))
     DC_LEPS_local(1:3) =-DEDR_C(3)*dxyz_cb(1:3) + DEDR_C(2)*dxyz_cb(1:3)
  ELSE
     DA_LEPS_local(1:3) = DEDR_C(1)*dxyz_ab(1:3) + DEDR_C(3)*dxyz_ac(1:3)
     DB_LEPS_local(1:3) =-one*(DEDR_C(2)*dxyz_cb(1:3) + DEDR_C(1)*dxyz_ab(1:3))
     DC_LEPS_local(1:3) =-DEDR_C(3)*dxyz_ac(1:3) + DEDR_C(2)*dxyz_cb(1:3)
  ENDIF

  IF (G2CROSS) THEN
     DD_LEPS_local(1:3) = DEDR_C(4)*dxyz_de(1:3) + DEDR_C(6)*dxyz_de(1:3)
     DE_LEPS_local(1:3) =-one*((DEDR_C(5)-DEDR_C(6))*dxyz_fe(1:3) + (DEDR_C(4)+DEDR_C(6))*dxyz_de(1:3))
     DF_LEPS_local(1:3) =-DEDR_C(6)*dxyz_fe(1:3) + DEDR_C(5)*dxyz_fe(1:3)
  ELSE
     DD_LEPS_local(1:3) = DEDR_C(4)*dxyz_de(1:3) + DEDR_C(6)*dxyz_df(1:3)
     DE_LEPS_local(1:3) =-one*(DEDR_C(5)*dxyz_fe(1:3) + DEDR_C(4)*dxyz_de(1:3))
     DF_LEPS_local(1:3) =-DEDR_C(6)*dxyz_df(1:3) + DEDR_C(5)*dxyz_fe(1:3)
  ENDIF

  !        WRITE(*,50) 'DTM> DIST ', R(1)*F2 , R(2)*F2, R(3)*F2
  !        WRITE(*,40) ' DERIVATIVES ',DEDR(1)*F3,DEDR(2)*F3,DEDR(3)*F3
  !CC   DERIVATIVES TEST
  IF (TEST) THEN
     WRITE(*,40)'LEPS> DAXYZ ',(DA_LEPS_local(I),I=1,3)
     WRITE(*,40)'LEPS> DBXYZ ',(DB_LEPS_local(I),I=1,3)
     WRITE(*,40)'LEPS> DCXYZ ',(DC_LEPS_local(I),I=1,3)
     WRITE(*,40)'LEPS> DDXYZ ',(DD_LEPS_local(I),I=1,3)
     WRITE(*,40)'LEPS> DEXYZ ',(DE_LEPS_local(I),I=1,3)
     WRITE(*,40)'LEPS> DFXYZ ',(DF_LEPS_local(I),I=1,3)
     !         WRITE(*,60)'LEPS> ENERGY ', ENERGY*F1 
  ENDIF

40 FORMAT(2X,A,3F20.5) 
50 FORMAT(2X,A,4F15.5)
60 FORMAT(2X,A,1F15.5)
  RETURN
END SUBROUTINE QM_SVB2D

!
SUBROUTINE PREPOT_SVB1D(DE,RE,BETA)
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : one
  use leps
  use stream
  implicit none
  real(chm_real) DE(3),RE(3),BETA(3)
  real(chm_real),parameter :: CKCAL=TOKCAL, &   ! 627.5095D0
       CANGS=BOHRR, &    ! 0.529177106D0   
       r_CKCAL=one/CKCAL, r_CANGS=one/CANGS
  !
  DE(1:3)  = DE_1(1:3)
  RE(1:3)  = RE_1(1:3)
  BETA(1:3)= BE_1(1:3)
  !
  !   CONVERT TO ATOMIC UNITS
  DE(1:3)  = DE(1:3)*r_CKCAL
  RE(1:3)  = RE(1:3)*r_CANGS
  BETA(1:3)= BETA(1:3)*CANGS
  !
  RETURN
END SUBROUTINE PREPOT_SVB1D

SUBROUTINE PREPOT_SVB2D(DE,RE,BETA)
  !
  use chm_kinds
  use consta, only : BOHRR,TOKCAL
  use number, only : one
  use leps
  use stream
  implicit none
  !
  real(chm_real) DE(6),RE(6),BETA(6)
  real(chm_real) :: VC12
  real(chm_real),parameter :: CKCAL=TOKCAL, &   ! 627.5095D0
       CANGS=BOHRR, &    ! 0.529177106D0
       r_CKCAL=one/CKCAL, r_CANGS=one/CANGS
  INTEGER I
  !
  VC12 = VCROSS
  DE(1:6)  = DE_1(1:6)
  RE(1:6)  = RE_1(1:6)
  BETA(1:6)= BE_1(1:6)
  !
  !   CONVERT TO ATOMIC UNITS
  DE(1:6)  = DE(1:6)*r_CKCAL
  RE(1:6)  = RE(1:6)*r_CANGS
  BETA(1:6)= BETA(1:6)*CANGS
  VC12 = VC12*r_CKCAL
  !
  RETURN
END SUBROUTINE PREPOT_SVB2D
!*****
!

SUBROUTINE POT_SVB1D(ENERGY,DEDR,DE,R,RE,BETA)
  !
  !   System:          ABC
  !   Functional form: LEPS (London-Erying-Polyani-Sato)
  !   References:      S. Sato
  !                    J. Chem. Phys. 592, 2465 (1955)
  !                    
  !
  !   The potential parameters must be passed through the common blocks
  !   PT1CM, PT5CM, PT2CM, PT4CM, SATOCM.  
  !   All information passed through the common blocks PT1CM, PT5CM, 
  !   SATOCM  must be in Hartree atomic units.
  !
  !        For the reaction: A + BC -> AB + C we write:
  !                          R(1) = R(A-B)
  !                          R(2) = R(B-C)
  !                          R(3) = R(C-A)
  !
  !   NOTE: The potential energy at the reactant asympotote, that is at 
  !         A infinitely far from the BC diatomic, BC diatomic at its
  !         equilibrium configuration, is set equal to zero.
  !
  use chm_kinds
  use number, only : zero, one, two, three, four, half
  use leps
  implicit none

  real(chm_real) ENERGY,DEDR(3),DE(3),R(3),RE(3),BETA(3)
  !
  real(chm_real) VC
  real(chm_real) DE2(3)
  real(chm_real) TSQRTAB1,TSQRTAM1,TSQRTAB2,TSQRTAM2,TEXP3,TEXP6
  real(chm_real) SQRTAB
  real(chm_real) EXPO(2),VAB(3),DVAB(3),DVR_AB(3),TEXP(2)
  real(chm_real) DEDR_AB(3)
  INTEGER I,L
  !
  !   INITIALIZE THE VARIABLE USED IN THE CALCULATION OF THE ENERGY.
  !
  ENERGY = zero
  !
  !   CHECK THE VALUE OF NDER 
  IF (NDER .GT. 1) THEN
     WRITE (6, 900) NDER
     CALL WRNDIE(-5,'<POT_SVB1D>','POT 1: Wrong NDER, only 1st derivatives are coded.')
  ENDIF
  !    MOD MG
  DE2(1:3) =DE(1:3)
  DEDR(1:3)=zero
  !
  !   COMPUTE THE ENERGY.
  !   MDTM ALL AM1 TERMS REMOVED. 06/03/04
  !  
  !   SIMPLE A-B CORRECTION
  IF (SVB_AB) THEN
     IF (G1CROSS) THEN
        VC =-DE2(1)*EXP(-BETA(1)*(R(1)-RE(1))**2)
     ELSE
        VC = DE2(1)*(EXP(-2*BETA(1)*(R(1)-RE(1)))- two*EXP(-BETA(1)*(R(1)-RE(1))))
     ENDIF
  ELSE
     !   A-B-C CORRECTION
     do L=1,2
        EXPO(L)=(EXP(-2*BETA(L)*(R(L)-RE(L))))- two*EXP(-BETA(L)*(R(L)-RE(L)))
        VAB(L) =DE2(L)*EXPO(L)
     end do
     !   MDTM GAUSSIAN CROSS TERM. 06/03/04
     IF (G1CROSS) THEN
        VAB(3)=DE2(3)*EXP(-BETA(3)*(R(3)-RE(3))**2)
     ELSE 
        VAB(3)=DE2(3)*EXP(-BETA(3)*(R(3)-RE(3)))
     ENDIF
     VC = half*(VAB(1)+VAB(2))-half*SQRT((VAB(1)-VAB(2))**2+four*VAB(3)**2 )+DE2(1)
     !   END A-B-C CORRECTION
  ENDIF
  ENERGY=VC
  !
  !   COMPUTE THE DERIVATIVES OF THE ENERGY WITH RESPECT TO THE INTERNAL
  !   COORDINATES (RAB,RBC,RAC OR RZ).
  !
  IF (NDER .EQ. 1) THEN
     !   SIMPLE A-B CORRECTION
     IF (SVB_AB) THEN
        IF (G1CROSS) THEN
           DEDR_AB(1)=-two*DE2(1)*BETA(1)*(R(1)-RE(1))*EXP(-BETA(1)*(R(1)-RE(1))**2)
           DEDR(1)   =-DEDR_AB(1)
        ELSE
           TEXP(1)   =EXP(-BETA(1)*(R(1)-RE(1)))
           DEDR_AB(1)=two*DE2(1)*BETA(1)*TEXP(1)*(one-TEXP(1))
           DEDR(1)   =DEDR_AB(1)
        ENDIF
     ELSE
        !   A-B-C CORRECTION
        TSQRTAB1=SQRT(((VAB(1)-VAB(2))**2+four*(VAB(3)**2)))
        IF (TSQRTAB1 .NE. zero)THEN
           TSQRTAB1=one/TSQRTAB1
        ELSE
           CALL WRNDIE(-5,'<POT_SVB1D>','SVB 1D DIV BY 0 IN 1ST REACTION DERIVATIVES.')
           RETURN
        ENDIF
        DVAB(1)=half*(one-((TSQRTAB1)*(VAB(1)-VAB(2))))
        DVAB(2)=half*(one+((TSQRTAB1)*(VAB(1)-VAB(2))))
        !            DVAB(3)=-two*TSQRTAB1*VAB(3)
        do I=1,2
           TEXP(I)=EXP(-BETA(I)*(R(I)-RE(I)))
           DVR_AB(I)=two*DE2(I)*BETA(I)*TEXP(I)*(one-TEXP(I))
        end do

        IF (.NOT. G1CROSS) THEN
           DVAB(3)=-two*TSQRTAB1*VAB(3)
           TEXP3=-BETA(3)*EXP(-BETA(3)*(R(3)-RE(3)))
           DVR_AB(3)=DE2(3)*TEXP3
        ELSE
           DVAB(3)=-two*TSQRTAB1*VAB(3)
           TEXP3=-BETA(3)*EXP(-BETA(3)*(R(3)-RE(3))**2)*two*(R(3)-RE(3))
           DVR_AB(3)=DE2(3)*TEXP3
           !   CDTM THE FOLLOWING LINES DO DERIVATIVES ONLY BY RAB AND RBC
!!!!            DEDR_AB(1)=DVAB(1)*DVR_AB(1) + four*VAB(3)*DE2(3)*BETA(3)*(R(3)-RE(3))*EXP(-BETA(3)*(R(3)-RE(3))**2)/TSQRTAB1
!!!!            DEDR(1)=DEDR_AB(1)
!!!!            DEDR_AB(2)=DVAB(2)*DVR_AB(2) - four*VAB(3)*DE2(3)*BETA(3)*(R(3)-RE(3))*EXP(-BETA(3)*(R(3)-RE(3))**2)/TSQRTAB1
!!!!            DEDR(2)=DEDR_AB(2)
        ENDIF
        DO I=1,3
           DEDR_AB(I)=DVAB(I)*DVR_AB(I)
           DEDR(I)=DEDR_AB(I)
        END DO
        !   END A-B-C CORRECTION
     ENDIF    ! SVB_AB
     !             DEDR(I)=DEDR_AB(I)
     !   END DERIVATIVES
  ENDIF       ! NDER .EQ. 1
  !
900 FORMAT(/,1X,T5,'ERROR: POT HAS BEEN CALLED WITH NDER = ', I5, &
       /,1X,T12,'ONLY THE FIRST DERIVATIVES, NDER = 1, ARE ','CODED IN THIS POTENTIAL')
  !
  RETURN
END SUBROUTINE POT_SVB1D
!*****
!

SUBROUTINE POT_SVB2D(ENERGY,DEDR,DE,R,RE,BETA)
  !
  !   System:          ABC
  !   Functional form: LEPS (London-Erying-Polyani-Sato)
  !   References:      S. Sato
  !                    J. Chem. Phys. 592, 2465 (1955)
  !                    
  !
  !   The potential parameters must be passed through the common blocks
  !   PT1CM, PT5CM, PT2CM, PT4CM, SATOCM.  
  !   All information passed through the common blocks PT1CM, PT5CM, 
  !   SATOCM  must be in Hartree atomic units.
  !
  !        For the reaction: A + BC -> AB + C we write:
  !                          R(1) = R(A-B)
  !                          R(2) = R(B-C)
  !                          R(3) = R(C-A)
  !
  !   NOTE: The potential energy at the reactant asympotote, that is at 
  !         A infinitely far from the BC diatomic, BC diatomic at its
  !         equilibrium configuration, is set equal to zero.
  !
  use chm_kinds
  use number, only : zero, one, two, three, four, five, half
  use leps
  implicit none

  real(chm_real) ENERGY,DEDR(6),DE(6),R(6),RE(6), BETA(6)
  !
  real(chm_real) VC,VC1,VC2,VC12
  real(chm_real) EASYAB, EASYBC, EASYAC, EASYDE, EASYEF, EASYDF
  real(chm_real) TSQRTAB1,TSQRTAM1,TSQRTAB2,TSQRTAM2,TEXP3,TEXP6
  real(chm_real) SQRTAB,DEDR_TMP
  INTEGER L,I
  real(chm_real) EXPO(6),VAB(6),DVAB(6),DVR_AB(6),TEXP(6) 
  real(chm_real) VAM(6),DVAM(6),DVR_AM(6),DE2(6)
  real(chm_real) DEDR_AB(6), DEDR_AM(6) 
  !
  !   INITIALIZE THE VARIABLE USED IN THE CALCULATION OF THE ENERGY.
  !
  ENERGY = zero
  !
  !   CHECK THE VALUE OF NDER 
  IF (NDER .GT. 1) THEN
     WRITE (6, 900) NDER
     CALL WRNDIE(-5,'<POT_SVB2D>','POT 1: Wrong NDER, only 1st derivatives are coded.')
  ENDIF
  !    MOD MG !3 CHANGED TO 6 FOR 2 REACTIONS, 05/30/03
  DE2(1:6)=DE(1:6)
  DEDR(1:6)=zero
  !
  !   COMPUTE THE ENERGY.
  !   MDTM ALL AM1 TERMS REMOVED. 06/03/04
  !  
  !CC 1ST REACTION    
  !   SIMPLE A-B CORRECTION
  IF (SVB_AB) THEN
     IF (G1CROSS) THEN
        VC1 =-DE2(1)*EXP(-BETA(1)*(R(1)-RE(1))**2)
     ELSE
        VC1 = DE2(1)*(EXP(-2*BETA(1)*(R(1)-RE(1)))-two*EXP(-BETA(1)*(R(1)-RE(1))))
     ENDIF
  ELSE
     !   A-B-C CORRECTION
     do L=1,2
        EXPO(L)=EXP(-2*BETA(L)*(R(L)-RE(L))) - two*EXP(-BETA(L)*(R(L)-RE(L)))
        VAB(L)=DE2(L)*EXPO(L)
     end do
     !   MDTM GAUSSIAN CROSS TERM. 06/03/04
     IF (G1CROSS) THEN
        VAB(3)=DE2(3)*EXP(-BETA(3)*(R(3)-RE(3))**2)
     ELSE 
        VAB(3)=DE2(3)*EXP(-BETA(3)*(R(3)-RE(3)))
     ENDIF
     VC1 = half*(VAB(1)+VAB(2))-half*SQRT((VAB(1)-VAB(2))**2+four*VAB(3)**2)+DE2(1)
     !   END A-B-C CORRECTION
  ENDIF

  !CC 2ND REACTION
  !   SIMPLE D-E CORRECTION
  IF (SVB_DE) THEN
     IF (G2CROSS) THEN
        VC2 = -DE2(4)*EXP(-BETA(4)*(R(4)-RE(4))**2)
     ELSE
        VC2 = DE2(4)*(EXP(-two*BETA(4)*(R(4)-RE(4)))-two*EXP(-BETA(4)*(R(4)-RE(4))))
     ENDIF
  ELSE
     !   D-E-F CORRECTION
     do L=4,5
        EXPO(L)=EXP(-two*BETA(L)*(R(L)-RE(L)))-two*EXP(-BETA(L)*(R(L)-RE(L)))
        VAB(L)=DE2(L)*EXPO(L)
     end do
     !   MDTM GAUSSIAN TERM. 06/03/04
     IF (G2CROSS) THEN
        VAB(6)=DE2(6)*EXP(-BETA(6)*(R(6)-RE(6))**2)
     ELSE
        VAB(6)=DE2(6)*EXP(-BETA(6)*(R(6)-RE(6)))
     ENDIF
     VC2 = half*(VAB(4)+VAB(5))-half*SQRT((VAB(4)-VAB(5))**2+four*VAB(6)**2)+DE2(4)
     !   END D-E-F CORRECTION
  ENDIF

  !CC COMBINE ENERGY FOR 2 REACTION COORDINATES
  !   OLD TERM
  !       VC=VC1+VC2
  !   MDTM ADD CROSS TERM (VC12) TO ACCOUNT FOR INTERACTION BETWEEN REACTION COORDINATES
  !   CURRENTLY ONLY A CONSTANT, BUT COULD IN PRINCIPLE BE DISTANCE DEPENDENT
  !           Q = SQRT(R(3)**2 + R(6)**2)
  !           Q0 = zero
  !           QCONS = one
  !           VC12  = QCONS * EXP(-ALPHA*(Q-Q0)**2)
  IF (SCROSS) THEN
     VC = half*(VC1+VC2)-half*SQRT((VC1-VC2)**2+four*VC12**2)
  ELSE
     VC=VC1+VC2
  ENDIF
  ENERGY=VC
  !
  !   COMPUTE THE DERIVATIVES OF THE ENERGY WITH RESPECT TO THE INTERNAL
  !   COORDINATES (RAB,RBC,RAC OR RZ).
  !
  IF (NDER .EQ. 1) THEN
     !   MDTM ADD DERIVATIVE OF CROSS TERM VIA DEDR_TMP
     IF (SCROSS) THEN
        SQRTAB = SQRT((VC1-VC2)**2+four*VC12**2)
        IF (SQRTAB .NE. zero) THEN
           DEDR_TMP = half*(one-(VC1-VC2)/SQRTAB)
        ELSE
           CALL WRNDIE(-5,'<POT_SVB2D>','SVB 2D DIV BY 0 IN DR CROSS TERM.')
           RETURN
        ENDIF
     ELSE
        DEDR_TMP = one
     ENDIF

     !CC 1ST REACTION
     !   SIMPLE A-B CORRECTION
     IF (SVB_AB) THEN
        IF (G1CROSS) THEN
           DEDR_AB(1)=-two*DE2(1)*BETA(1)*(R(1)-RE(1))*EXP(-BETA(1)*(R(1)-RE(1))**2)
           DEDR(1)   =-DEDR_AB(1)*DEDR_TMP
        ELSE
           TEXP(1)   =EXP(-BETA(1)*(R(1)-RE(1)))
           DEDR_AB(1)=two*DE2(1)*BETA(1)*TEXP(1)*(one-TEXP(1))
           DEDR(1)   =DEDR_AB(1)*DEDR_TMP
        ENDIF
     ELSE
        !   A-B-C CORRECTION
        TSQRTAB1=SQRT(((VAB(1)-VAB(2))**2+four*(VAB(3)**2)))
        IF (TSQRTAB1 .NE. zero)THEN
           TSQRTAB1=one/TSQRTAB1
        ELSE
           CALL WRNDIE(-5,'<POT_SVB2D>','SVB 2D DIV BY 0 IN 1ST REACTION DERIVATIVES.')
           RETURN
        ENDIF
        DVAB(1)=half*(one-((TSQRTAB1)*(VAB(1)-VAB(2))))
        DVAB(2)=half*(one+((TSQRTAB1)*(VAB(1)-VAB(2))))
        !             DVAB(3)=-two*TSQRTAB1*VAB(3)
        do I=1,2
           TEXP(I)  =EXP(-BETA(I)*(R(I)-RE(I)))
           DVR_AB(I)=two*DE2(I)*BETA(I)*TEXP(I)*(one-TEXP(I))
        end do
        IF (.NOT. G1CROSS) THEN
           DVAB(3)  =-two*TSQRTAB1*VAB(3)
           TEXP3    =-BETA(3)*EXP(-BETA(3)*(R(3)-RE(3)))
           DVR_AB(3)= DE2(3)*TEXP3
        ELSE
           DVAB(3)  =-two*TSQRTAB1*VAB(3)
           TEXP3    =-BETA(3)*EXP(-BETA(3)*(R(3)-RE(3))**2)*two*(R(3)-RE(3))
           DVR_AB(3)= DE2(3)*TEXP3
           !   CDTM THE FOLLOWING LINES DO DERIVATIVES ONLY BY RAB AND RBC
!!!!!!!            DEDR_AB(1)=DVAB(1)*DVR_AB(1)+four*VAB(3)*DE2(3)*BETA(3)*(R(3)-RE(3))*EXP(-BETA(3)*(R(3)-RE(3))**2)/TSQRTAB1
!!!!!!!            DEDR(1)=DEDR_AB(1)
!!!!!!!            DEDR_AB(2)=DVAB(2)*DVR_AB(2)-four*VAB(3)*DE2(3)*BETA(3)*(R(3)-RE(3))*EXP(-BETA(3)*(R(3)-RE(3))**2)/TSQRTAB1
!!!!!!!            DEDR(2)=DEDR_AB(2)
        ENDIF
        DO I=1,3
           DEDR_AB(I)=DVAB(I)*DVR_AB(I)
           DEDR(I)   =DEDR_AB(I)*DEDR_TMP
        END DO
        !   END A-B-C CORRECTION
     ENDIF      ! SVB_AB
     !             DEDR(I)=DEDR_AB(I)
     ! 2ND REACTION
     !   MDTM ADD DERIVATIVE OF CROSS TERM VIA DEDR_TMP
     IF (SCROSS) THEN
        DEDR_TMP = half*(one+(VC1-VC2)/SQRTAB)
     ENDIF

     !   SIMPLE D-E CORRECTION
     IF (SVB_DE) THEN
        IF (G2CROSS) THEN
           DEDR_AB(4)=-two*DE2(4)*BETA(4)*(R(4)-RE(4))*EXP(-BETA(4)*(R(4)-RE(4))**2)
           DEDR(4)   =-DEDR_AB(4)*DEDR_TMP
        ELSE
           TEXP(4)   =EXP(-BETA(4)*(R(4)-RE(4)))
           DEDR_AB(4)=two*DE2(4)*BETA(4)*TEXP(4)*(one-TEXP(4))
           DEDR(4)   =DEDR_AB(4)*DEDR_TMP
        ENDIF
     ELSE
        !   D-E-F CORRECTION
        TSQRTAB2=SQRT(((VAB(4)-VAB(5))**2+four*(VAB(6)**2)))
        IF (TSQRTAB2 .NE. 0.0) THEN
           TSQRTAB2=1.0/TSQRTAB2
        ELSE
           CALL WRNDIE(-5,'<POT_SVB2D>','SVB 2D DIV BY 0 IN 2ND REACTION DERIVATIVES.')
           RETURN
        ENDIF

        DVAB(4) =half*(one-((TSQRTAB2)*(VAB(4)-VAB(5))))
        DVAB(5) =half*(one+((TSQRTAB2)*(VAB(4)-VAB(5))))
        !             DVAB(6)=-two*TSQRTAB2*VAB(6)
        do I=4,5
           TEXP(I)  =EXP(-BETA(I)*(R(I)-RE(I)))
           DVR_AB(I)=two*DE2(I)*BETA(I)*TEXP(I)*(1-TEXP(I))
        end do
        IF (.NOT. G2CROSS) THEN
           DVAB(6)  =-two*TSQRTAB2*VAB(6)
           TEXP6    =-BETA(6)*EXP(-BETA(6)*(R(6)-RE(6)))
           DVR_AB(6)= DE2(6)*TEXP6
        ELSE
           DVAB(6)  =-two*TSQRTAB2*VAB(6)
           TEXP6    =-BETA(6)*EXP(-BETA(6)*(R(6)-RE(6))**2)*two*(R(6)-RE(6))
           DVR_AB(6)= DE2(6)*TEXP6
!!!!!!!            DEDR_AB(4)=DVAB(4)*DVR_AB(4)+four*VAB(6)*DE2(6)*BETA(6)*(R(6)-RE(6))*EXP(-BETA(6)*(R(6)-RE(6))**2)/TSQRTAB2
!!!!!!!            DEDR(4)=DEDR_AB(4)
!!!!!!!            DEDR_AB(5)=DVAB(5)*DVR_AB(5)-four*VAB(6)*DE2(6)*BETA(6)*(R(6)-RE(6))*EXP(-BETA(6)*(R(6)-RE(6))**2)/TSQRTAB2
!!!!!!!            DEDR(5)=DEDR_AB(5)
        ENDIF
        DO I=4,6
           DEDR_AB(I)=DVAB(I)*DVR_AB(I)
           DEDR(I)   =DEDR_AB(I)*DEDR_TMP
        END DO
        !   END D-E-F CORRECTION
     ENDIF    ! SVB_DE
     !             DEDR(I)=DEDR_AB(I)
     !   END NDER
  ENDIF       ! NDER .EQ. 1
  !
900 FORMAT(/,1X,T5,'Error: POT has been called with NDER = ', I5, &
       /,1X,T12,'only the first derivatives, NDER = 1, are ','coded in this potential')
  !
  RETURN
END SUBROUTINE POT_SVB2D
!*****
!

