module erfcd_mod

  use chm_kinds
  use ewald_1m,only:ewnpts
  implicit none

  !   EWNPTS       - number of points on the erfc lookup table
  !   EWLDT        - heap pointer for the erfc lookup table
  !   EWRDEL       - reciprocal of spacing on the erfc lookup table
  real(chm_real),allocatable,dimension(:),save :: EWLDT
  real(chm_real),save :: ewrdel
!  integer,save :: ERFMOD ,EWNPTS
! 
! if this variable is activated here then it is a problem in a lot of
! CHARMM routines (testcases show different results!) Maybe we need to
! discard this #if ... #endif for all QM methods. QCHEM is already out!
#if KEY_QTURBO==1
  integer,save :: ERFMOD        
#endif

  private :: erfcd_one

contains

  subroutine allocate_ewldt()
    use memory
    call chmalloc('ewald.src','allocate_ewldt','ewldt',ewnpts,crl=ewldt)
    return
  end subroutine allocate_ewldt

  subroutine deallocate_ewldt()
    use memory
    call chmdealloc('ewald.src','deallocate_ewldt','ewldt',ewnpts,crl=ewldt)
    return
  end subroutine deallocate_ewldt


  !==========================================================================
  !          ERFCD
  !==========================================================================
  SUBROUTINE ERFCD(R,KAPPAX,ERFCX,DRFC,METHOD)
    implicit none
    real(chm_real), intent(in) :: R, KAPPAX
    real(chm_real), intent(out) :: ERFCX, DRFC
    integer, intent(in) :: METHOD

    if (R /= R) call wrndie(-4, '<ERFCD>', 'R is NaN')
    ! optimizing compiler will inline this call
    call erfcd_one(R,KAPPAX,ERFCX,DRFC,EWLDT,METHOD)
  END SUBROUTINE ERFCD
  
  !==========================================================================
  !          ERFCD_one
  !==========================================================================
  SUBROUTINE ERFCD_one(R,KAPPAX,ERFCX,DRFC,erfct_1,METHOD)
    !
    !--- Calculate the erfc function, and its derivative
    !---            By Steve Bogusz and Bernard Brooks - July 1995
    !
    use number
    implicit none
    real(chm_real), intent(in) :: R, KAPPAX
    real(chm_real), intent(out) :: ERFCX, DRFC
    real(chm_real), intent(in) :: erfct_1(:)
    integer, intent(in) :: METHOD

    !--- METHOD 0 and -1:
    !---     Lookup table methods using either linear or
    !---     cubic spline interpolations
    !
    REAL(KIND=CHM_REAL) XVAL,REM,VAL0,VAL1,VAL2,D1,D2
    INTEGER IXVAL
    !-------------------------------------------------------------------
    !--- METHOD 1:
    !---     Approximation to the complementary error function using in
    !---     real space part of the Ewald summation for electrostatics
    !---     
    !---     Reference:
    !---     
    !---     Abramowitz and Stegun, Handboook of Mathematical Functions,
    !---     National Bureau of Standards, 1964.  Forumula 7.1.26
    !
    REAL(KIND=CHM_REAL)  A1,A2,A3,A4,A5,P,RSQPI2
    PARAMETER (RSQPI2 = TWO*5.6418958354775628695D-1)
    PARAMETER (A1 = 0.254829592D0, A2=-0.284496736D0 )
    PARAMETER (A3 = 1.421413741D0, A4=-1.453152027D0 )
    PARAMETER (A5 = 1.061405429D0, P= 0.3275911D0   )
    !
    REAL(KIND=CHM_REAL) X,T,XSQ,TP
    !
    !-------------------------------------------------------------------
    !--- METHOD 2 and 3:
    !
    !---     Calculates ERFc using method in chapter 6.2 of 
    !---     Numerical Recipes 2nd ed.
    !
    REAL(KIND=CHM_REAL) RSQPI,AP,SUM,DEL,XX,B,C,D,H,EPS,AN
    PARAMETER (RSQPI = 5.6418958354775628695D-1)
    INTEGER  N,I,ITMAX
    !      !---      PARAMETER (ITMAX = 100, EPS=3.D-7, FPMIN=1.D-30)
    PARAMETER (ITMAX = 100)
    !
    !-------------------------------------------------------------------
    !--- METHOD 4:
    !
    !---     Calculates ERFC using 2nd method in chapter 6.2 of 
    !---     Numerical Recipes 2nd ed. (based on Chebyshev)
    !
    REAL(KIND=CHM_REAL)  B1,B2,B3,B4,B5,B6,B7,B8,B9,B10
    PARAMETER (B1 = -1.26551223D0, B2=1.0000236D0 )
    PARAMETER (B3 = .37409196D0, B4=.09678418D0 )
    PARAMETER (B5 = -.18628806D0, B6=.27886807D0 )
    PARAMETER (B7 = -1.13520398D0, B8=1.48851587D0 )
    PARAMETER (B9 = -.82215223D0, B10=.17087277D0 )
    !
    !-------------------------------------------------------------------
    !--- begin
    X = R*KAPPAX
    !
    !-------------------------------------------------------------------
    IF(METHOD <= 0) THEN
       !
       !--- METHOD0 and -1: Lookup table methods
       XVAL = X*EWRDEL
       IXVAL = XVAL+HALF
       REM = XVAL-IXVAL
       IXVAL=IXVAL+2
       IXVAL = MIN(IXVAL,EWNPTS-1)
       VAL0 = ERFCT_1(IXVAL-1)
       VAL1 = ERFCT_1(IXVAL)
       VAL2 = ERFCT_1(IXVAL+1)
       D1 = (VAL0-VAL2)*HALF
       ERFCX = VAL1-D1*REM
       IF(METHOD < 0) THEN
          !---        Cubic spline interpolation
          D2 = (VAL1+VAL1-VAL0-VAL2)*REM
          ERFCX = ERFCX-HALF*D2*REM
          D1 = D1+D2
       ENDIF
       DRFC = D1*EWRDEL*KAPPAX
       RETURN
    ENDIF
    !-------------------------------------------------------------------
    !
    XX = X*X
    XSQ = EXP(-XX)
    DRFC = XSQ*KAPPAX*RSQPI2
    !
    !-------------------------------------------------------------------
    !
    IF(METHOD == 1) THEN
       !
       !--- METHOD1: Abramowitz and Stegun
       T = ONE/(ONE+P*X)
       TP = T*(A1 + T*(A2 + T*(A3 + T*(A4 + T*A5))))
       ERFCX = TP*XSQ
       !
       !-------------------------------------------------------------------
       !
    ELSE IF(METHOD == 2) THEN
       !
       !--- METHOD2: Exact high precision
       EPS = TWO*RPRECI
       IF (XX < 3.5) GOTO 200
       GOTO 300
       !
       !-------------------------------------------------------------------       
    ELSE IF(METHOD == 3) THEN
       !
       !--- METHOD3: Exact low precision
       EPS = 3.0D-7
       IF (XX < 1.5) GOTO 200
       GOTO 300
       !
       !-------------------------------------------------------------------       
       !
    ELSEIF(METHOD == 4) THEN
       !
       !---      Calculates ERFC using 2nd method in chapter 6.2 of 
       !---      Numerical Recipes 2nd ed. (based on Chebyshev)
       ! METHOD4:
       T = ONE/(ONE+HALF*X)
       ERFCX = T*EXP(-XX+B1+T*(B2+T*(B3+T*(B4+T*(B5+T*(B6+ &
            T*(B7+T*(B8+T*(B9+T*B10)))))))))
       !
    ENDIF
    !-------------------------------------------------------------------       
    RETURN
    !
    !---     Calculates ERFC using method in chapter 6.2 of 
    !---     Numerical Recipes 2nd ed. - (hacked for efficiency)
    !--- METHOD 2 and 3:
    !
100 CONTINUE
    XX = X*X
    XSQ = EXP(-XX)
    DRFC = XSQ*KAPPAX*RSQPI2
    EPS = TWO*RPRECI
    IF(XX > 3.5) GOTO 300
    !
    !---  Converge for small values of XX
200 CONTINUE
    AP = HALF
    SUM = TWO
    DEL = SUM
    DO N = 1,ITMAX
       AP = AP+ONE
       DEL = DEL*(XX/AP)
       SUM = SUM+DEL
       IF(DEL < SUM*EPS) THEN
          ERFCX = ONE-SUM*X*XSQ*RSQPI
          RETURN
       ENDIF
    ENDDO
    CALL WRNDIE(-4,'<ERFCD>', &
         'Maximum number of iterations reached (2A)')
    RETURN
    !
    !---  Converge for large values of XX
300 CONTINUE
    B = XX+HALF
    C = HALF*RBIGST
    D = ONE/B
    H = D
    DO I = 1,ITMAX
       AN = -I*(I-HALF)
       B = B+TWO
       D = AN*D+B
       !      !---          IF(D < FPMIN) D = FPMIN  ! can't happen for our case...
       C = B+AN/C
       !      !---          IF(C < FPMIN) C = FPMIN  ! can't happen for our case...
       D = ONE/D
       DEL = D*C
       H = H*DEL
       IF(ABS(DEL-ONE) < EPS) THEN
          ERFCX = H*X*XSQ*RSQPI
          RETURN
       ENDIF
    ENDDO
    CALL WRNDIE(-4,'<ERFCD>', &
         'Maximum number of iterations reached (2B)')
    RETURN
  END  SUBROUTINE ERFCD_one
  
  SUBROUTINE FERFCT()
    !
    !--- Fill the ERFC lookup table.
    !---            By Steve Bogusz and Bernard Brooks - July 1995
    !
    use chm_kinds
    use number
    use consta
    use stream
    implicit none

    real(kind=chm_real) :: &
         del,drfc,test1,test2,ssq1,ssq2,err,errmx1,errmx2,x,tmp(1)
    integer i

    del = one/ewrdel
    x = zero
    !--- generate the table
    do i = 3,ewnpts-3
       x = x+del
       call erfcd_one(x,one,ewldt(i),drfc,tmp,2)
    enddo

    ewldt(ewnpts-2)=zero
    ewldt(ewnpts-1)=zero
    ewldt(ewnpts)=zero
    ewldt(2)=one
    ewldt(1)=one+del*two/sqrt(pi)

    !--- now check the table near midpoints...
    ssq1 = zero
    ssq2 = zero
    errmx1 = zero
    errmx2 = zero
    x = zero-(half+tenm5)*del
    do i = 1,ewnpts-5
       x = x+del
       call erfcd_one(x,one,test2,drfc,tmp,2)
       call erfcd_one(x,one,test1,drfc,ewldt,0)
       err = abs(test1-test2)
       ssq1 = ssq1+err*err
       if(err > errmx1) errmx1 = err

       call erfcd_one(x,one,test1,drfc,ewldt,-1)
       err = abs(test1-test2)
       ssq2 = ssq2+err*err
       if(err > errmx2) errmx2 = err
    enddo
    ssq1 = ssq1/(ewnpts-5)
    ssq1 = sqrt(ssq1)
    ssq2 = ssq2/(ewnpts-5)
    ssq2 = sqrt(ssq2)

    if(prnlev > 3) write(outu,55) 'linear inter',ssq1,errmx1
    if(prnlev > 3) write(outu,55) 'cubic spline',ssq2,errmx2
55  format(' fill erfc table: ',a,' has rms error = ',d12.6, &
         ' maximum error = ',d12.6)
    return
  end SUBROUTINE FERFCT
end module erfcd_mod

