module exelecm
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="exelec.src"
  !CHARMM Element source/fcm/exelec.fcm 1.1
  !========================================================================
  !     EXELEC.FCM contains the extended electrostatics and reaction
  !     field data structure.
  !
  !     Extended electrostatics :
  !
  !     QEXTND            The extended electrostatics flag.
  !     QXGRAD            Are field gradients to be calculated?
  !     QXQUAD            Are quadrupole terms to be included in the
  !                       potential and field calculation?
  !     CTEXNB            The extended electrostatics cutoff distance.
  !
  !     ATPOT             The potential at each atom due to groups outside
  !                       the cutoff.
  !     ATFX, ATFY, ATFZ  The field at each atom.
  !     ATGXX, ATGYY,     The field gradient at each atom.
  !     ATGZZ, ATGXY,
  !     ATGYZ, ATGZX
  !
  !     Reaction fields :
  !
  !     QRXNFL            The reaction field flag.
  !     RXNMOD            The reaction field mode (ENERGY or NBONDS).
  !     RXNORD            The order of the reaction field calculation.
  !     RXNSHL            The size of the reaction field shell.
  !     EPSEXT            The external dielectric constant.
  !
  !     LEGPN, LEGPND     The Legendre polynomials and their derivatives.
  !     ERXMNT, MPMMNT,   Local reaction field arrays.
  !     RXMMNT
  !
  ! (OLD) RXNFLD    INT*4   Reaction field calculation option
  !
  !========================================================================
  ! . Extended electrostatics.
  real(chm_real),dimension(:),allocatable :: ATPOT, ATFX, ATFY, ATFZ, &
       ATGXX, ATGYY, ATGZZ, ATGXY, ATGYZ, ATGZX

  !
  LOGICAL   QEXTND, QXGRAD, QXQUAD
  !
  real(chm_real)    CTEXNB

  ! . Reaction fields.
  character(len=8) RXNMOD
  !
  INTEGER   RXNORD,siz_legpn
  real(chm_real),allocatable,dimension(:) :: LEGPN, LEGPND, ERXMNT
  complex(chm_cmpx),allocatable,dimension(:) :: MPMMNT, RXMMNT
  !
  LOGICAL   QRXNFL
  !
  real(chm_real)    EPSEXT, RXNSHL

contains

  subroutine exelec_init()
    qextnd=.false.
    qxgrad=.false.
    qxquad=.false.
    return
  end subroutine exelec_init


  SUBROUTINE EXELEC(EEXTND, NATOM, X, Y, Z, DX, DY, DZ, QXGRAD, &
       WRNMXD, ATSX, ATSY, ATSZ, ATPOT, ATFX, ATFY, &
       ATFZ, ATGXX, ATGYY, ATGZZ, ATGXY, ATGYZ, &
       ATGZX, UNIT)
    !
    !     EXELEC calculates the electrostatic interactions for each atom
    !     with other atoms that are outside its cutoff.
    !
    !     By David J. States
    !
  use chm_kinds
  use number
  use stream
  use dimens_fcm
#if KEY_GCMC==1
  use gcmc
#endif 
#if KEY_PARALLEL==1
  use parallel      
#endif
    implicit none
    ! . Passed variables.
    INTEGER NATOM, UNIT
    LOGICAL QXGRAD
    real(chm_real)  ATSX(*),  ATSY(*),  ATSZ(*),  ATPOT(*), ATFX(*), &
         ATFY(*), ATFZ(*), ATGXX(*), ATGYY(*), ATGZZ(*), &
         ATGXY(*), ATGYZ(*), ATGZX(*), DX(*), DY(*), DZ(*), &
         EEXTND, WRNMXD, X(*), Y(*), Z(*)
    !
    INTEGER ATFRST,ATLAST
    ! . Local variables.
    INTEGER I
    !C      INTEGER IMAX
    real(chm_real)  DR, DRMAX, DRSUM, ETEMP, XD, YD, ZD
    !
    DRSUM = ZERO
    DRMAX = ZERO
    !C      IMAX  = 0
    !
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
    ! Define the atom bounds for this processor.
    ATFRST=1+IPARPT(MYNOD)
    ATLAST=IPARPT(MYNODP)
#elif KEY_SPACDEC==1 /*parfmain*/
    atfrst=1
    atlast=natom
#elif KEY_PARASCAL==1 /*parfmain*/
    atfrst=1
    atlast=natom
#endif /* (parfmain)*/
#else /* (paramain)*/
    atfrst=1
    atlast=natom
#endif /* (paramain)  PARALLEL*/

    loop10: DO I=ATFRST,ATLAST

#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#elif KEY_SPACDEC==1 /*parfmain*/
#elif KEY_PARASCAL==1 /*parfmain*/
       IF(JPBLOCK(I) /= MYNOD) GOTO 100
#endif /* (parfmain)*/
#else /* (paramain)*/
#endif /* (paramain)  PARALLEL*/

#if KEY_GCMC==1
       if(qgcmc) then
          IF (.NOT. GCMCON(I)) cycle loop10
       endif
#endif 
       XD = X(I) - ATSX(I)
       YD = Y(I) - ATSY(I)
       ZD = Z(I) - ATSZ(I)
       DR = XD*XD + YD*YD + ZD*ZD
       IF (WRNMXD  >  ZERO) THEN
          DRSUM = DRSUM + DR
          IF (DR  >  DRMAX) THEN
             DRMAX = DR
             !C               IMAX  = I
          ENDIF
       ENDIF
       ETEMP = ATPOT(I) + ATFX(I)*XD + ATFY(I)*YD + ATFZ(I)*ZD
       DX(I) = DX(I) + ATFX(I)
       DY(I) = DY(I) + ATFY(I)
       DZ(I) = DZ(I) + ATFZ(I)
       IF (QXGRAD) THEN
          ETEMP  = ETEMP + (ATGXX(I)*XD*XD + ATGYY(I)*YD*YD + &
               ATGZZ(I)*ZD*ZD + ATGXY(I)*XD*YD + &
               ATGYZ(I)*YD*ZD + ATGZX(I)*ZD*XD) / TWO
          DX(I) = DX(I) + TWO*XD*ATGXX(I)+YD*ATGXY(I)+ZD*ATGZX(I)
          DY(I) = DY(I) + TWO*YD*ATGYY(I)+XD*ATGXY(I)+ZD*ATGYZ(I)
          DZ(I) = DZ(I) + TWO*ZD*ATGZZ(I)+XD*ATGZX(I)+YD*ATGYZ(I)
       ENDIF
       EEXTND = EEXTND + ETEMP
    enddo loop10
    !
    IF (WRNMXD  >  ZERO .AND. NATOM .GT. 0) THEN
       IF (DRMAX  >  WRNMXD * WRNMXD .AND. WRNLEV >= 2) THEN
          DRMAX = SQRT(DRMAX)
          DRSUM = SQRT(DRSUM / NATOM)
          WRITE (UNIT,'(2A)') ' EXELEC> Atom displacement from', &
               ' last update exceeds warning distance.'
          WRITE (UNIT,'(A,F9.2,A,I5)') &
               '         Maximum distance = ', DRMAX, ' Atom = ', I
          WRITE (UNIT,'(A,F9.2)') &
               '         Average distance = ', DRSUM
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE EXELEC
  
  SUBROUTINE ERXNFL(ERXN, NATOM, CG, X, Y, Z, DX, DY, DZ, &
       EATRXN, FX, FY, FZ, MODE, ORDER, EINT, &
       EEXT, SHELL, MPMMNT, RXMMNT, ERXMNT, &
       LEGPN, LEGPND)
    !
    !     ERXNFL calculates the reaction field energy for an array of
    !     charges by first doing a multipole moment expansion to the
    !     specified order and then using it to calculate reaction fields
    !     at each atom in a second pass. The contributions of each moment
    !     to the reaction field energy are summed in the second pass and
    !     are included in the atom derivatives in a third pass.
    !
    !     A spherical model is used with a radius of SQRT(5/3)*R_RMS + SHELL
    !     (the SQRT(5/3) factor gives the correct outer radius of a sphere
    !     given its root mean square radius). If atoms are outside the
    !     sphere their radius is scaled to place them between the sphere and
    !     the outer shell in a manner that yields a constant limiting
    !     reaction force. The calculation follows the method of Kirkwood
    !     (JCP 3 p1 1934). For a discussion of Legendre expanions in
    !     electrostatics see Jackson (1975). Recurrence formulas for the
    !     Legendre polynomials are derived from those found in Abramowitz
    !     and Stegun.
    !
    !     The order must be greater than or equal to 1 for execution.
    !
    !     Mode is an integer flag that determines the output of ERXNFL.
    !
    !     0 - Evaluate the total energy and derivatives (for use in ENERGY).
    !     1 - Evaluate the energy and add atom by atom to the energy and
    !         derivatives (for use in NBONDS).
    !
    !     Storage:                       
    !
    !     CG, X, Y, Z        Each NATOM.
    !     DX, DY, DZ         Natom - only if mode = 0.
    !     EATRXN             Natom - only if mode = 1.
    !     FX,FY,FZ           Natom - only if mode = 1.
    !
    !     LEGPN, LEGPND,     Dimension (ORDER+1)**2
    !     ERXMNT
    !     
    !     MPMMNT, RXMMNT     Dimension (ORDER+1)
    !
    !     By David J. States
    !              
  use chm_kinds
  use consta
  use number
  use stream
    implicit none
    ! . Passed variables.
    INTEGER    MODE, NATOM, ORDER
    complex(chm_cmpx) :: MPMMNT(*), RXMMNT(*)
    real(chm_real)     CG(NATOM), DX(NATOM), DY(NATOM), DZ(NATOM), &
         EATRXN(NATOM), EEXT, EINT, ERXMNT(*), ERXN, &
         FX(NATOM), FY(NATOM), FZ(NATOM), LEGPN(*), &
         LEGPND(*), SHELL, X(NATOM), Y(NATOM), Z(NATOM)
    ! . Local variables.
    INTEGER    I, L, LB, LM, M
    complex(chm_cmpx) :: CTEMP, CTEMPC, EIMPHI, EIPHI
    real(chm_real)     CGR, CMX, CMY, CMZ, CONSTN, COSPHI, &
         COSTH, D2R, DCOST, DP, DR, DRCALC, DRO, DXSUM, &
         DYSUM, DZSUM, EATOM, EMMNT, FACTRL, HALFSH, &
         RCALC, RI, RISQ, RLIM, RO, RRMS, RXYSQ,  &
         SINPHI, SINTH, TEMP, &
         XD, YD, ZD, XX, YY, ZZ     
    !
    ERXN = ZERO
    IF (EINT  ==  EEXT .OR. ORDER  <  1) RETURN
    ! . Set up the centre of mass and the transition radius.
    CMX  = ZERO
    CMY  = ZERO
    CMZ  = ZERO
    XX   = ZERO
    YY   = ZERO
    ZZ   = ZERO
    loop10: DO I = 1,NATOM
       CMX = CMX + X(I)
       CMY = CMY + Y(I)
       CMZ = CMZ + Z(I)
       XX  = XX + X(I)*X(I)
       YY  = YY + Y(I)*Y(I)
       ZZ  = ZZ + Z(I)*Z(I)
    enddo loop10
    CMX  = CMX / NATOM
    CMY  = CMY / NATOM
    CMZ  = CMZ / NATOM
    RRMS = SQRT((XX+YY+ZZ)/NATOM - CMX*CMX - CMY*CMY - CMZ*CMZ)
    RLIM = SQRT(FIVE/THREE) * RRMS - SHELL
    RO   = RLIM + TWO*SHELL
    !
    HALFSH = SHELL/TWO
    DXSUM  = ZERO
    DYSUM  = ZERO
    DZSUM  = ZERO
    LB = 1
    loop30: DO L = 0,ORDER
       LB = LB + L + L
       loop20: DO M = -L,L
          MPMMNT(LB+M) = cmplx(ZERO, ZERO, chm_cmpx)
       enddo loop20
    enddo loop30
    ! . Sum the multipole expansion atom by atom.
    loop70: DO  I = 1,NATOM
       ! . Calculate the Legendre polynomials.
       CALL LPDCAL(.FALSE., I, ORDER, CMX, CMY, CMZ, COSPHI, &
            COSTH, DRCALC, HALFSH, LEGPN, LEGPND, &
            RCALC, RI, RISQ, RLIM, RXYSQ, SHELL, &
            SINPHI, SINTH, XD, YD, ZD, X, Y, Z)
       CGR = CG(I)/RCALC
       LB = 1
       loop60: DO L = 0,ORDER
          LB = LB + L + L
          EIPHI  = cmplx(COSPHI,  SINPHI, chm_cmpx)
          EIMPHI = cmplx(COSPHI, -SINPHI, chm_cmpx)
          CGR    = CGR * RCALC
          loop40: DO M = 0,L
             LM = LB + M
             EIMPHI = EIMPHI * EIPHI
             MPMMNT(LM) = MPMMNT(LM) + EIMPHI*(LEGPN(LM)*CGR)
          enddo loop40
          EIPHI  = cmplx(COSPHI, -SINPHI, chm_cmpx)
          EIMPHI = cmplx(ONE, ZERO, chm_cmpx)
          loop50: DO M = -1,-L,-1
             LM = LB + M
             EIMPHI = EIMPHI * EIPHI
             MPMMNT(LM) = MPMMNT(LM) + EIMPHI*(LEGPN(LM)*CGR)
          enddo loop50
       enddo loop60
    enddo loop70
    ! . Calculate the reaction field from the summer MPMMNT.
    LB = 1
    loop100: DO L = 0,ORDER
       LB = LB + L + L
       CONSTN = HALF * CCELEC*(L+1) * &
            (EINT-EEXT)/(EINT*((L+1)*EEXT+L*EINT)*RO**(2*L+1))
       FACTRL = ONE
       CTEMP  = MPMMNT(LB)
       CTEMPC = CTEMP*CONSTN
       ERXMNT(LB) = DBLE(CTEMP)  * DBLE(CTEMPC) + &
            aimag(CTEMP) * aimag(CTEMPC)
       RXMMNT(LB) = CTEMPC
       loop80: DO  M = 1,L
          LM = LB + M
          FACTRL = FACTRL/((1+L-M)*(L+M))
          CTEMP  = MPMMNT(LM)
          CTEMPC = (FACTRL*CONSTN)*CTEMP
          ERXMNT(LM) = DBLE(CTEMP)  * DBLE(CTEMPC) + &
               aimag(CTEMP) * aimag(CTEMPC)
          RXMMNT(LM) = CTEMPC
       enddo loop80
       FACTRL = ONE
       loop90: DO M = -1,-L,-1
          LM = LB + M
          FACTRL = FACTRL/((1+L+M)*(L-M))
          CTEMP  = MPMMNT(LM)
          CTEMPC = (FACTRL*CONSTN)*CTEMP
          ERXMNT(LM) = DBLE(CTEMP)  * DBLE(CTEMPC) + &
               aimag(CTEMP) * aimag(CTEMPC)
          RXMMNT(LM) = CTEMPC
       enddo loop90
    enddo loop100
    DRO = ZERO
    ! . Calculate the derivative with respect to the overall radius.
    LB  = 1
    loop120: DO L = 0,ORDER
       LB = LB + L + L
       TEMP = (2*L+1) / (RO**(2*L+2))
       loop110: DO  M = -L,L
          DRO = DRO - ERXMNT(LB+M)*TEMP
       enddo loop110
    enddo loop120
    DRO = DRO / (NATOM*(RRMS)**ONEPT5)
    ! . Calculate the reaction potential and the fields at each atom.
    loop160: DO I = 1,NATOM
       ! . Calculate the Legendre polynomials and their derivatives.
       CALL LPDCAL(.TRUE., I, ORDER, CMX, CMY, CMZ, COSPHI, &
            COSTH, DRCALC, HALFSH, LEGPN, LEGPND, &
            RCALC, RI, RISQ, RLIM, RXYSQ, SHELL, &
            SINPHI, SINTH, XD, YD, ZD, X, Y, Z)
       !
       CGR   = CG(I) / RCALC
       DR    = DRO*RI
       D2R   = ZERO
       DCOST = ZERO
       DP    = ZERO
       EATOM = ZERO
       LB  =  1
       loop150: DO L = 0,ORDER
          LB = LB + L + L
          CGR   = CGR * RCALC
          TEMP  = CGR * DBLE(RXMMNT(LB))
          EMMNT = LEGPN(LB) * TEMP
          DCOST = DCOST + LEGPND(LB) * TEMP
          EIPHI  = cmplx(COSPHI, SINPHI, chm_cmpx)
          EIMPHI = cmplx(ONE, ZERO, chm_cmpx)
          loop130: DO M = 1,L
             LM = LB + M
             EIMPHI = EIMPHI * EIPHI
             CTEMP  = RXMMNT(LM)
             TEMP   = CGR * (DBLE(CTEMP)  * DBLE(EIMPHI) + &
                  aimag(CTEMP) * aimag(EIMPHI))
             EMMNT = EMMNT + TEMP * LEGPN(LM)
             DCOST = DCOST + TEMP * LEGPND(LM)
             DP = DP + LEGPN(LM) * M * CGR * &
                  (-DBLE(RXMMNT(LM)) * aimag(EIMPHI) + &
                  aimag(RXMMNT(LM)) * DBLE(EIMPHI))
          enddo loop130
          EIPHI  = cmplx(COSPHI, -SINPHI, chm_cmpx)
          EIMPHI = cmplx(ONE, ZERO, chm_cmpx)
          loop140: DO M =  -1,-L,-1
             LM = LB + M
             EIMPHI = EIMPHI * EIPHI
             CTEMP  = RXMMNT(LM)
             TEMP   = CGR * (DBLE(CTEMP)  *  DBLE(EIMPHI) + &
                  aimag(CTEMP) * aimag(EIMPHI))
             EMMNT  = EMMNT + TEMP*LEGPN(LM)
             DCOST  = DCOST + TEMP*LEGPND(LM)
             DP = DP - LEGPN(LM)*M*CGR* &
                  (DBLE(RXMMNT(LM))  * aimag(EIMPHI) - &
                  aimag(RXMMNT(LM)) *  DBLE(EIMPHI))
          enddo loop140
          TEMP = L*EMMNT/RCALC
          DR = DR + TEMP
          IF (RI  >=  RLIM) THEN
             TEMP = TEMP*(L - 1)/RCALC
             D2R = D2R + TEMP
          ENDIF
          EATOM = EATOM + EMMNT
       enddo loop150
       !
       IF (RI  >=  RLIM) THEN
          EATOM = EATOM + TWO * DR *(RI - RCALC)
          DR    = DR    + TWO * D2R*(RI - RCALC)
       ENDIF
       ERXN = ERXN + EATOM
       IF (MODE  ==  1) THEN
          EATRXN(I) = EATRXN(I) + EATOM
          TEMP = TWO*(DR*SINTH*COSPHI+DCOST*COSTH*XD/RISQ-DP*YD/RXYSQ)
          FX(I) = FX(I) + TEMP
          DXSUM = DXSUM + TEMP
          TEMP = TWO*(DR*SINTH*SINPHI+DCOST*COSTH*YD/RISQ+DP*XD/RXYSQ)
          FY(I) = FY(I) + TEMP
          DYSUM = DYSUM + TEMP
          TEMP = TWO*(DR*COSTH + DCOST*RXYSQ/(RI*RISQ))
          FZ(I) = FZ(I) + TEMP
          DZSUM = DZSUM + TEMP
       ELSE IF (MODE  ==  0) THEN
          TEMP = TWO*(DR*SINTH*COSPHI+DCOST*COSTH*XD/RISQ-DP*YD/RXYSQ)
          DX(I) = DX(I) + TEMP
          DXSUM = DXSUM + TEMP
          TEMP = TWO*(DR*SINTH*SINPHI+DCOST*COSTH*YD/RISQ+DP*XD/RXYSQ)
          DY(I) = DY(I) + TEMP
          DYSUM = DYSUM + TEMP
          TEMP = TWO*(DR*COSTH + DCOST*RXYSQ/(RI*RISQ))
          DZ(I) = DZ(I) + TEMP
          DZSUM = DZSUM + TEMP
       ENDIF
    enddo loop160
    ! . Correct for an overall imbalance in the forces applied.
    IF (MODE  ==  0) THEN
       DXSUM = DXSUM/NATOM
       DYSUM = DYSUM/NATOM
       DZSUM = DZSUM/NATOM
       DO I = 1,NATOM
          DX(I) = DX(I) - DXSUM
          DY(I) = DY(I) - DYSUM
          DZ(I) = DZ(I) - DZSUM
       enddo
    ELSE IF (MODE  ==  1) THEN
       DXSUM = DXSUM/NATOM
       DYSUM = DYSUM/NATOM
       DZSUM = DZSUM/NATOM
       DO I = 1,NATOM
          FX(I) = FX(I) - DXSUM
          FY(I) = FY(I) - DYSUM
          FZ(I) = FZ(I) - DZSUM
       enddo
    ENDIF
    !
    RETURN
  END SUBROUTINE ERXNFL
  
  SUBROUTINE LPDCAL(QDERIV, I, ORDER, CMX, CMY, CMZ, COSPHI, &
       COSTH, DRCALC, HALFSH, LEGPN, LEGPND, &
       RCALC, RI, RISQ, RLIM, RXYSQ, SHELL, &
       SINPHI, SINTH, XD, YD, ZD, X, Y, Z)
    !
    !     Calculate the Legendre polynomials and their derivatives.
    !     The derivatives are calculated only if QDERIV is .TRUE.
    !
  use chm_kinds
  use number
    implicit none
    ! . Passed variables.
    INTEGER  I, ORDER
    LOGICAL  QDERIV
    real(chm_real)   CMX, CMY, CMZ, COSPHI, COSTH, DRCALC, HALFSH, &
         LEGPN(*), LEGPND(*), RCALC, RI, RISQ, RLIM, &
         RXYSQ, SHELL, SINPHI, SINTH, XD, YD, ZD, &
         X(*),Y(*),Z(*)
    ! . Local variables.
    INTEGER  L, LB, LBP, LBPP, M
    real(chm_real)   BOTTOM, BOTSQ, FACTOR, RXY, SINNGM, SINTSQ, &
         TEMP, TEMPSQ
    ! . Calculate the Legendre polynomials.
    XD = X(I) - CMX
    YD = Y(I) - CMY
    ZD = Z(I) - CMZ
    RISQ   = XD*XD+YD*YD+ZD*ZD
    RI     = SQRT(RISQ)
    RXYSQ  = XD*XD + YD*YD
    RXY    = SQRT(RXYSQ)
    COSTH  = ZD/RI
    SINTH  = RXY/RI
    SINTSQ = SINTH * SINTH
    COSPHI = XD/RXY
    SINPHI = YD/RXY
    IF (RI  <=  RLIM) THEN
       RCALC  = RI
       DRCALC = ONE
    ELSE
       TEMP   = RI - RLIM
       TEMPSQ = TEMP*TEMP
       BOTTOM = TEMP + HALFSH
       BOTSQ  = BOTTOM*BOTTOM
       RCALC  = RLIM + HALFSH*(TEMP/BOTTOM + TEMPSQ/BOTSQ)
       DRCALC = HALFSH/BOTSQ + SHELL*TEMP/(BOTSQ*BOTTOM)
    ENDIF
    LEGPN(1) = ONE
    LEGPN(3) = COSTH
    LBP = 1
    LB  = 3
    DO L = 2,ORDER
       LBPP = LBP
       LBP  = LB
       LB   = LB + L + L
       LEGPN(LB) = ((2*L-1)*COSTH*LEGPN(LBP)-(L-1)*LEGPN(LBPP))/L
    enddo
    !
    TEMP   = ONE
    SINNGM = ONE
    LB = 1
    loop20: DO L = 1,ORDER
       LB     = LB + L + L
       TEMP   = TEMP * (1-2*L)*SINTH
       SINNGM = -SINNGM
       LEGPN(LB+L) = TEMP
       IF (SINNGM  >  ZERO) THEN
          LEGPN(LB-L) =  TEMP
       ELSE
          LEGPN(LB-L) = -TEMP
       ENDIF
    enddo loop20
    !
    LB = 3
    loop40: DO L = 2,ORDER
       LBP = LB
       LB  = LB + L + L
       SINNGM = ONE
       loop30: DO M = 1,L-1
          TEMP = COSTH*LEGPN(LBP+M)-SINTH*(L-1+M)*LEGPN(LBP+M-1)
          LEGPN(LB+M) = TEMP
          SINNGM = -SINNGM
          IF (SINNGM  >  0) THEN
             LEGPN(LB-M) = TEMP
          ELSE
             LEGPN(LB-M) = -TEMP
          ENDIF
       enddo loop30
    enddo loop40
    ! . Calculate the derivatives of the Legendre polynomials.
    IF (.NOT. QDERIV) GOTO 999
    LEGPND(1) = ZERO
    IF (SINTH  >  PT001) THEN
       LB = 1
       loop60: DO L = 1,ORDER
          LBP = LB
          LB  = LB + L + L
          LEGPND(LB) = -(COSTH*LEGPN(LB)-LEGPN(LBP))*L/SINTSQ
          SINNGM = ONE
          loop50: DO  M = 1,L-1
             TEMP = -(L*COSTH*LEGPN(LB+M)-(L+M)*LEGPN(LBP+M))/SINTSQ
             SINNGM = -SINNGM
             LEGPND(LB+M) = TEMP
             IF (SINNGM  >  ZERO) THEN
                LEGPND(LB-M) = TEMP
             ELSE
                LEGPND(LB-M) = -TEMP
             ENDIF
          enddo loop50
       enddo loop60
       SINNGM =  ONE
       FACTOR = -COSTH/SINTSQ
       LB = 1
       DO L = 1,ORDER
          LB = LB + L + L
          FACTOR = FACTOR*(1-2*L)*SINTH
          TEMP   = L*FACTOR
          SINNGM = -SINNGM
          LEGPND(LB+L) = TEMP
          IF (SINNGM  >  ZERO) THEN
             LEGPND(LB-L) = TEMP
          ELSE
             LEGPND(LB-L) = -TEMP
          ENDIF
       enddo
    ELSE
       LB = 1
       DO L = 1,ORDER
          LB = LB + L + L
          LEGPND(LB) = ZERO
          DO M = 1,ORDER
             LEGPND(LB+M) = ZERO
             LEGPND(LB-M) = ZERO
          enddo
       enddo
    ENDIF
    !
999 RETURN
  end SUBROUTINE LPDCAL

end module exelecm

