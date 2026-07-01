module rgyr_mod
  use chm_kinds
  implicit none

contains
  SUBROUTINE RGYR(NATOMS,X,Y,Z,W,AM,ISLCT,FACT,LMASS,LWEIG)
    !-----------------------------------------------------------------------
    !     Compute the radius of gyration, center of mass,
    !     and total mass of the selected subset of either the main or
    !     the comparison structure. The results are output to unit OUTU.
    !     If keyword WEIG is given the weighting array, which the user
    !     must fill with CORMAN or SCALAR commands, is used for the
    !     weighting. This is indicated by LWEIG=.TRUE. and is taken care
    !     of in CORMAN's call to RGYR.
    !     If keyword MASS is given, then the mass-weighted radius of
    !     gyration, etc are computed, otherwise unit weighting per
    !     atom is used (giving rms distance from the geometric center).
    !     The default option is to do e geometric RGYR calculation.
    !     A constant offset to be subtracted from the weights can be
    !     specified with keyword FACT.
    !
    !     SYNTAX:
    !
    !     COOR RGYR  [FACT <real>] {[MASS]} [COMP]  [<atom-selection>]
    !     {[WEIG]}
    !
    !     1983-09-01/LN
    !     1985-01-05/ Default revised /LN
    !
    use number
    use stream
    use param_store, only: set_param

    implicit none

    INTEGER NATOMS
    real(chm_real) X(*),Y(*),Z(*),W(*)
    real(chm_real) AM(*)
    INTEGER ISLCT(*)
    real(chm_real) FACT
    LOGICAL LMASS,LWEIG
    real(chm_real) XCM,YCM,ZCM
    INTEGER NMISS,I
    real(chm_real) TMASS,WW,RG,AR
    !
    !     Center-of-mass:
    !
    XCM=0.0
    YCM=0.0
    ZCM=0.0
    TMASS=0.0
    NMISS=0
    !
    DO I=1,NATOMS
       IF (ISLCT(I) /= 1) THEN
       ELSE IF (X(I) == ANUM) THEN
          NMISS=NMISS+1
       ELSE
          IF (LMASS.AND.LWEIG) THEN
             WW=W(I)*AM(I)
          ELSE IF (LMASS) THEN
             WW=AM(I)
          ELSE IF (LWEIG) THEN
             WW=W(I)
          ELSE
             WW=1.0
          ENDIF
          WW=WW-FACT
          XCM=WW*X(I)+XCM
          YCM=WW*Y(I)+YCM
          ZCM=WW*Z(I)+ZCM
          TMASS=WW+TMASS
       ENDIF
    enddo
    !
    IF(NMISS == NATOMS) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,25)
       RETURN
    ENDIF
25  FORMAT(/' RGYR: *** ERROR ***  All coordinates were missing'/)
    !
    IF(TMASS <= 0.0) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,35) TMASS
       IF(TMASS == 0.0) RETURN
    ENDIF
35  FORMAT(/' RGYR: *** WARNING *** Net "mass"=',F12.5/)
    !
    XCM=XCM/TMASS
    YCM=YCM/TMASS
    ZCM=ZCM/TMASS
    IF(NMISS /= 0 .AND. WRNLEV >= 2) WRITE(OUTU,45) NMISS
45  FORMAT(/' RGYR:   There were',I5,' missing coordinates.'/)
    !
    !     Radius of gyration:
    !
    RG=0.0
    DO I=1,NATOMS
       IF (ISLCT(I) /= 1) THEN
       ELSE IF (X(I) == ANUM) THEN
       ELSE
          IF (LMASS.AND.LWEIG) THEN
             WW=W(I)*AM(I)
          ELSE IF (LMASS) THEN
             WW=AM(I)
          ELSE IF (LWEIG) THEN
             WW=W(I)
          ELSE
             WW=1.0
          ENDIF
          WW=WW-FACT
          RG=RG+WW*((X(I)-XCM)**2+(Y(I)-YCM)**2+(Z(I)-ZCM)**2)
       ENDIF
    enddo
    AR=ABS(RG/TMASS)
    !
    !     Compute an RG with the same sign as RG/TMASS:
    !
    RG=SQRT(AR)*RG/(TMASS*AR)
    call set_param('RGYR',RG)
    call set_param('MASS',TMASS)
    call set_param('XCM ',XCM)
    call set_param('YCM ',YCM)
    call set_param('ZCM ',ZCM)
    !
    IF(PRNLEV >= 2) THEN
       IF(LWEIG) WRITE(OUTU,71)
       IF(LMASS) WRITE(OUTU,72)
       IF(.NOT. (LMASS.OR.LWEIG)) WRITE(OUTU,73)
       WRITE(OUTU,74) RG,TMASS,XCM,YCM,ZCM
    ENDIF
71  FORMAT(/' RGYR:'/)
72  FORMAT(/' RGYR:  Mass weighted results:'/)
73  FORMAT(/' RGYR:  Geometric results:'/)
74  FORMAT(/'       Radius of gyration=',F12.5,5X, &
         'Net "mass"=',F12.3/ &
         '       Center-of-"mass" = ' ,3F12.5/)
    !
    RETURN
  END SUBROUTINE RGYR

  SUBROUTINE LSQP(N,X,Y,Z,W,AMASS,ISLCT,LVERB,LMASS,LWEIG,QAXIS)
    !-----------------------------------------------------------------------
    !     FIT A LEAST SQUARES PLANE TO SELECTED COORDINATES IN X,Y,Z.
    !     ALGORITHM FROM  BLOW ACTA CRYST (1960)13,168.
    !
    !     SYNTAX:  COOR LSQP  { [MASS] } [VERBOSE] [ATOM_SELECTION] [COMP]
    !     { [WEIG] }
    !
    !     23-FEB-1984 /LENNART NILSSON
    !
    use number
    use stream
    use corsubs, only: &
      axisx, axisy, axisz, axisr, &
      axiscx, axiscy, axiscz, qaxisc, lsqp2
    use chutil, only: atomid
    use param_store, only: set_param

    INTEGER N
    real(chm_real) X(*),Y(*),Z(*),W(*)
    real(chm_real) AMASS(*)
    INTEGER ISLCT(*)
    LOGICAL LVERB,LWEIG,LMASS
    INTEGER QAXIS
    !
    !
    real(chm_real) LAMDA,A(3,3),SUM(3),DC(4),R(3),EV(3)
    real(chm_real) XCM,YCM,ZCM
    INTEGER I,J,L
    real(chm_real) SSQ,DIST
    CHARACTER(len=8) CA,CB,CC,CD
    !
    LVERB=(LVERB .AND. PRNLEV >= 2)
    !
    !     get direction cosines and distance from origin of a plane
    !     also compute center of mass of selected atoms
    !
    CALL LSQP2(N,X,Y,Z,AMASS,LMASS, &
         ISLCT,LWEIG,W,.FALSE.,LVERB,XCM,YCM,ZCM,A,EV)
    !
    IF(EV(2) < PT001) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,125) EV(2)
125    FORMAT(/' LSQP: *** ERROR *** EV(2)=',G12.4)
       IF(WRNLEV >= 2) WRITE(OUTU,135)
135    FORMAT('      CHECK ATOMS FOR COLLINEARITY')
       RETURN
    ENDIF
    !
    DC(1)=A(1,3)
    DC(2)=A(2,3)
    DC(3)=A(3,3)
    DC(4)=DC(1)*XCM+DC(2)*YCM+DC(3)*ZCM
    !
    !     compute ssq  from plane for input atoms
    !     and print individual atom distances (if lverb on)
    SSQ=0.0
    !
    DO I=1,N
       IF(ISLCT(I) == 1) THEN
          !         non-weighted distances
          DIST=ABS((X(I)*DC(1)+Y(I)*DC(2)+Z(I)*DC(3)-DC(4)))
          IF(LVERB) THEN
             CALL ATOMID(I,CA,CB,CC,CD)
             WRITE(OUTU,235) I,CA(1:idleng),CB(1:idleng), &
                  CC(1:idleng),CD(1:idleng),DIST
235          FORMAT(10X,'ATOM',I5,4(2X,A),' IS',F8.4, &
                  ' ANGSTROMS FROM PLANE')
          ENDIF
          SSQ=SSQ+DIST**2
       ENDIF
    ENDDO
    !
    IF(PRNLEV >= 2) WRITE(OUTU,245) DC,SSQ,XCM,YCM,ZCM
245 FORMAT(/' LSQP: EQUATION FOR LEAST-SQUARES-PLANE:   ', &
         F7.4,'*X',SP,F7.4,'*Y',F7.4,'*Z = ',F9.2// &
         '      SSQ =',G13.6/ &
         '      CENTER-OF-MASS =',3F10.3/)
    !
    IF(QAXIS > 0) THEN
       QAXISC=.TRUE.
       AXISCX= XCM
       AXISCY= YCM
       AXISCZ= ZCM
       AXISR = ONE
       AXISX = A(1,QAXIS)
       AXISY = A(2,QAXIS)
       AXISZ = A(3,QAXIS)
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
    ENDIF
    !
    RETURN
  END SUBROUTINE LSQP


  SUBROUTINE CORINER(NATM,X,Y,Z,AMASS,ISLCT,LENTRO)
    ! Modifications:
    ! Victor Anisimov, 2004: Rotational and Translational Entropy
    ! Randy Bin Lin,   2011: Standard State Defined for Translational Entropy

    use number
    use stream
    ! Rotational Entropy
    use entropy
    ! for PI
    use consta
    use param_store, only: set_param

    implicit none

    INTEGER NATM, ISLCT(NATM)
    real(chm_real) X(NATM), Y(NATM), Z(NATM), AMASS(NATM)
    LOGICAL LENTRO

    real(chm_real) XCM,YCM,ZCM, XX,XY,XZ,YY,YZ,ZZ, AMS,TMS
    real(chm_real) A(6), B(3,3), EV(3), SCR(21)
    INTEGER I,J

    XCM = ZERO
    YCM = ZERO
    ZCM = ZERO
    TMS = ZERO
    !
    DO I=1,NATM
       IF(ISLCT(I) == 1) THEN
          AMS=AMASS(I)
          TMS = TMS + AMS
          XCM = XCM + X(I)*AMS
          YCM = YCM + Y(I)*AMS
          ZCM = ZCM + Z(I)*AMS
       ENDIF
    ENDDO

    IF (TMS > ZERO) THEN
       XCM = XCM / TMS
       YCM = YCM / TMS
       ZCM = ZCM / TMS
    ENDIF

    XX = ZERO
    XY = ZERO
    XZ = ZERO
    YY = ZERO
    YZ = ZERO
    ZZ = ZERO

    DO I=1,NATM
       IF (ISLCT(I) == 1) THEN
          AMS = AMASS(I)
          XX = XX + AMS*(X(I)-XCM)*(X(I)-XCM)
          XY = XY + AMS*(X(I)-XCM)*(Y(I)-YCM)
          XZ = XZ + AMS*(X(I)-XCM)*(Z(I)-ZCM)
          YY = YY + AMS*(Y(I)-YCM)*(Y(I)-YCM)
          YZ = YZ + AMS*(Y(I)-YCM)*(Z(I)-ZCM)
          ZZ = ZZ + AMS*(Z(I)-ZCM)*(Z(I)-ZCM)
       ENDIF
    ENDDO

    A(1) = YY + ZZ
    A(2) = -XY
    A(3) = -XZ
    A(4) = XX + ZZ
    A(5) = -YZ
    A(6) = XX + YY

    !  diagonalize the matrix
    CALL DIAGQ(3,3,A,B,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
         SCR(16),SCR(19),SCR(1),0)
    !
    !  smallest eigenvalue,EV(1), is principal intertia vector,B(*,1)
    !  secondary vector is B(*,2), tertiary is B(*,3)
    if (prnlev > 2) then
       WRITE(OUTU,'(/A)') ' Principal Moments of Inertia, amu*A^2'
       WRITE (OUTU,101) (EV(J),J=1,3)
101    FORMAT(' Sorted Eigenvalues: ',3G16.7)
       WRITE (OUTU,102) (B(J,1),J=1,3)
102    FORMAT(' Principal axis, X= ',F8.4,3X,'Y= ',F8.4,3X,'Z= ',F8.4)
       WRITE (OUTU,103) (B(J,2),J=1,3)
103    FORMAT(' Secondary axis, X= ',F8.4,3X,'Y= ',F8.4,3X,'Z= ',F8.4)
       WRITE (OUTU,104) (B(J,3),J=1,3)
104    FORMAT(' Tertiary axis,  X= ',F8.4,3X,'Y= ',F8.4,3X,'Z= ',F8.4)
    endif
#if KEY_CONSHELIX==1
    call set_param('PXXX',B(1,1))
    call set_param('PYYY',B(2,1))
    call set_param('PZZZ',B(3,1))
#endif 
    !
    ! Rotational component of Entropy:
    ! S_rot = R { ln[ (sqrt(PI*I_A*I_B*I_C)/sigma) * (8PI^2*k_B*T/h^2)^1.5] + 1.5} 
    !
    ! Translational component of Entropy for Ideal Gas as Standard State
    ! S_tran = R { ln[ (2PI*M*k_B*T/h^2)^1.5 * k_B*T/P] + 2.5}
    ! Translational component of Entropy for 1M Solution as Standard State
    ! S_tran = R { ln[ (2PI*M*k_B*T/h^2)^1.5 / rho    ] + 2.5}
    ! where rho = 1M, therefore the difference for translational component
    ! of entropy between the two states are a constant, R ln[ Vg / Vs]
    ! where Vg is the molar volume of ideal gas which is 2.445D-3 m^3
    ! where Vs is the molar volume of solution which is 1.000D-3 m^3
    !
    ! The units are selected so as to render expression under logarithm 
    ! (representing rotational partition function) dimensionless
    !
    ! CRC Handbook, 84th edition, 2003-2004
    !      A = 1 * 10^-10             m                Angstrom 
    !      h = 6.62606876 * 10^-34    J*s              Planck constant
    !      R = 8.314472               J / (mol * K)    Molar gas constant
    !      J = 1.112650056 * 10^-17   kg * m^2 / s^2   Energy, Joule
    !    k_B = 1.3806503 * 10^-23     J / K            Boltzmann constant
    !    N_A = 6.02214199 * 10^23     mol^-1           Avogadro constant
    !    amu = 10^-3 / N_A            kg               Atomic mass unit
    !      I =                        kg * m^2         Principal moment of inertia
    !     Pa =                        N / m^2 = kg/(m*s^2)   Pascal
    !      N =                        kg*m/s^2         Force, Newton
    !
    ! Conversion:
    ! [I] = amu*A^2= (10^-3/N_A)*(10^-10)^2 = 10^-23/N_A kg*m^2
    ! 1 cal = 4.184 J
    !
    ! Fundamental physical constants:
    !     NA = 6.02214199D0
    !     KB = 1.3806503D0
    !     H  = 6.62606876D0
    !     P  = 1.01325D0
    !
    !     Srot = 1.9872065D0 * 
    !    1 ( LOG(SQRT(EV(1)*EV(2)*EV(3)*TK*TK*TK) / SIGMA) +
    !    2   LOG(PI*PI*PI*SQRT(PI*KB*KB*KB*0.512D0/(NA*NA*NA))/(H*H*H)) +
    !    3   1.5D0)
    !
    !     ideal gas as standard volume
    !     Stran = 1.9872065D0 * 
    !    1 ( LOG(SQRT(TMS*T)*TMS*TK*TK) +
    !    2   LOG(SQRT(80.0D0*PI*KB/NA) * PI*KB*KB / (NA*H*H*H*P)) +
    !    3   2.5D0)
    !
    !     1M solution as standard volume
    !     Stran = 1.9872065D0 * 
    !    1 ( LOG(SQRT(TMS*T)*TMS*TK*TK) +
    !    2   LOG(SQRT(80.0D0*PI*KB/NA) * PI*KB*KB / (NA*H*H*H*P)) +
    !    3   2.5D0 + LOG(Vs/Vg) )
    !
    IF(LENTRO) THEN
       if (prnlev > 2) WRITE(OUTU,'(1X,A,F16.5)') 'Molecular Weight, amu = ', TMS
       SROT = 1.9872065D0 *  &
            (LOG(SQRT(EV(1)*EV(2)*EV(3)*TK*TK*TK)/SIGMA) - 2.7105283523D0)
       IF (SSTAN.EQ.'SOLU'.OR.SSTAN.EQ.'    ') THEN
          STRAN = 1.9872065D0 *  &
                ( ( LOG(SQRT(TMS*TK)*TMS*TK*TK) - 1.1648678461D0 ) &
                + LOG(1.D-3/2.445D-2))
       ELSEIF (SSTAN.EQ.'GAS ') THEN
          STRAN = 1.9872065D0 *  &
                ( LOG(SQRT(TMS*TK)*TMS*TK*TK) - 1.1648678461D0)
       ELSE
          if (prnlev > 2) WRITE(OUTU,'(/1X,A)') 'Unsupported standard state  '
          STRAN = 0.D0
       ENDIF
       !
       ! Header printing
       if (prnlev > 2) WRITE(OUTU,'(3(/1X,A),2(/1X,A,F8.2))') &
            'ENTROPY>', &
            'Parameters used in the calculation:', &
            '  Num. trivial modes =        6 (non-linear molecule assumed)', &
            '  Temperature, K     = ',TK, &
            '  Symmetry number    = ',SIGMA
       ! Print out standard state
       IF (SSTAN.EQ.'SOLU'.OR.SSTAN.EQ.'    ') THEN
          if (prnlev > 2) WRITE(OUTU,'(/1X,A)') 'Standard state is 1M solution'
       ELSEIF (SSTAN.EQ.'GAS ') THEN
          if (prnlev > 2) WRITE(OUTU,'(/1X,A)') 'Standard state is ideal gas'
       ELSE
          if (prnlev > 2) WRITE(OUTU,'(/1X,A)') 'Unsupported standard state  '
       ENDIF
       if (prnlev > 2) WRITE(OUTU,'(/1X,A,2(/1X,A,F10.3))') &
            'Entropy:           cal/(mol*K)', &
            '  Rotational    = ', SROT, &
            '  Translational = ', STRAN
       SSUM=SROT+STRAN
       !
       ! Save Entropy components in CHARMM variables
       call set_param('SROT',SROT)
       call set_param('STRA',STRAN)
       call set_param('SVIB',ZERO)
       call set_param('SSUM',SSUM)
    ENDIF
    !
    RETURN
  END SUBROUTINE CORINER
end module rgyr_mod
