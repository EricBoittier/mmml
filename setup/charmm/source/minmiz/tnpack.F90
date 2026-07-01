!***********************************************************************
! ----------------------------------------------------------
! This is the file for testing and using TNPACK,
! our Fortran package for minimizing multivariate
! functions by a truncated Newton algorithm.
!
! A full description is available in the manuscripts
! entitled:
!
!     "TNPACK --- A Truncated Newton Minimization
!      Package for Large-Scale Problems":
!      I. Algorithm and usage.
! and
!     "TNPACK --- A Truncated Newton Minimization
!      Package for Large-Scale Problems":
!      II. Implementation Examples.
!
! by   Tamar Schlick (a)   and   Aaron Fogelson (b)
!
! (a) Courant Institue of Mathematical Sciences
!     251 Mercer Street
!     New York University
!     New York,  New York   10012
!     --------------------------------
!     [INTERNET: schlick@acfclu.nyu.edu]
!     [BITNET:   schlick@nyuacf]
!     (212) 998 - 3116
!
! (b) University of Utah
!     Department of Mathematics
!     Salt Lake City,  Utah 84112
!     -----------------------------------------
!     [INTERNET: fogelson@foglsun.math.utah.edu]
!     (801) 581 - 8150
!
!*********************************************************************
!     TNPACK was incorporated into and modified for CHARMM by
!     Bernard R. Brooks (Summer 92) and Philippe Derreumaux (1992-3).
!     The contents of this file include TNPACK subprograms (listed
!     below in the original documentation) and additional routines
!     (SCHEDU, PARMSD, BESTSE, and HDPNEW).
!     See also the file TNDRIV.src for the driver and related
!     subroutines.
!
!     Special segments involving TNPACK modifications are bracketed
!     separately below.
!***********************************************************************
!                                                                      *
!                  -    "TNPACK"   -                                   *
!                                                                      *
!          A  PACKAGE  OF  ROUTINES  FOR  A                            *
!       LARGE-SCALE TRUNCATED NEWTON ALGORITHM                         *
!                                                                      *
!                copyright   (c)   1990  by                            *
!                                                                      *
!            Tamar Schlick   &    Aaron Fogelson                       *
!                                                                      *
!***********************************************************************
! PLEASE NOTE:                                                         *
!                                                                      *
! (i)   Double Precision is used througout.                            *
! (ii)  Documentation for the program is given separately; here        *
!       only brief comments are provided.                              *
! (iii) It is essential that the user become familiar with all input   *
!       options and parameters before using this package. Entry to     *
!       TNPACK from the user's driver must first be done through a     *
!       call to routine SETLIS. SETLIS sets sample input options and   *
!       parameters, some of which the user may then modify to suit     *
!       the problem. Routine TNMIN should then be called.              *
! (iv)  For runs on Cray computers, the Yale Sparse Matrix Package     *
!       parameter RATIO must be changed from 2 to 1 - routine SDRVMD   *
!                                                                      *
!***********************************************************************
!                                                                      *
! TNPACK COMPONENTS:                                                   *
!                                                                      *
! (A)  USER-SUPPLIED SUBROUTINES (generic names):                      *
!      -----------------------------------------                       *
!                                                                      *
!  *   CALFGH    - calculates F,G,H and M (preconditioner) at X        *
!  *   CALHDP    - calculates HD, a Hessian/vector product             *
!  *   CALPAT    - determines the sparsity pattern of M                *
!                                                                      *
! (B) TRUNCATED NEWTON SUBROUTINES:                                    *
!     ----------------------------                                     *
!                                                                      *
!  *   TNMIN              -    the TN driver routine                   *
!  *   OUTER              -    the outer Newton iteration              *
!  *   INNER              -    the inner PCG iteration                 *
!  *   OURHDP             -    computes Hessian/vector products        *
!  *   SETLIS             -    sets sample parameters for minimization *
!  *   CHKLIS             -    checks input options and parameters     *
!  *   DIVWRK             -    divides work space                      *
!  *   MLINES, NEWSTP     -    perform the line search                 *
!                                                                      *
! (C)  YSMP (YALE SPARSE MATRIX PACKAGE) SUBROUTINES:                  *
!      ---------------------------------------------                   *
!                                                                      *
!  *   SDRVMD             -    the driver for solving Mz=r             *
!  *   ODRV               -    the driver for finding M's reordering   *
!  *   MD,MDI,MDM,MDP,MDU -    the minimum-degree reordering routines  *
!  *   SRO                -    prepares for symmetric reordering of M  *
!  *   SSF                -    performs symbolic factorization of M    *
!  *   SNFMOD             -    performs numerical factorization of M   *
!  *   SNS                -    performs numerical solution of Mz=r     *
!                                                                      *
! (D)  OTHER SUBPROGRAMS:                                              *
!      ----------------                                                *
!                                                                      *
!  *   DNRM2                (blas functions)                           *
!                                                                      *
!***********************************************************************
module tnpack
  use chm_kinds
  implicit none

  ! TN001
  integer :: MP, ICALLS
  ! TN002
  integer :: IEXIT, IHD, ITT
  real(chm_real) :: PRODK, ETAR, ETAQ
  ! TN003
  integer :: MAXFEV
  real(chm_real) :: FTOL, GTOL
  ! TN004
  integer :: IXP, IXY, IXD, IXHD, IXZ, IXA, IXRSP, &
       IXD1, IXX1, IXX2, IXX3, IXG1, IXG2, IXG3
  ! TN005
  integer :: IXPER, IXIPER, IXIA, IXJA, IXISP
  ! TN006
  real(chm_real) :: EPSMCH, SQEPS2, SQRTN

contains

#if KEY_TNPACK==0 /*tnpack*/
SUBROUTINE TNMIN(N,X,F,G,OPLIST,PLIST,INFORM, &
     NZ,W,LW,IW,LIW, &
     CALFGH,CALPAT,CALHDP)
  INTEGER N,OPLIST(20),INFORM,NZ,LW,LIW,IW(LIW)
  real(chm_real) F,X(N),G(N),PLIST(20),W(LW)
  EXTERNAL CALFGH, CALPAT, CALHDP

  CALL WRNDIE(-3,'<TNDRIV>','TNPACK is NOT compiled.')
  RETURN
END SUBROUTINE TNMIN
#else /* (tnpack)*/

SUBROUTINE TNMIN(N,X,F,G,OPLIST,PLIST,INFORM, &
     NZ,W,LW,IW,LIW, &
     CALFGH,CALPAT,CALHDP)
  !
  ! --------------------------------------------------------
  ! TNMIN:  Truncated Newton Interface Routine
  ! --------------------------------------------------------
  !
  use euler
  use stream
  implicit none
  !
  ! passed variables
  INTEGER N,OPLIST(20),INFORM,NZ,LW,LIW,IW(LIW)
  real(chm_real) F,X(N),G(N),PLIST(20),W(LW)
  EXTERNAL CALFGH, CALPAT, CALHDP

  INTEGER MARK,I,NSP
  LOGICAL QWRIT
  !
  ! --------------------------------------------------------
  ! check parameters and divide work space
  ! --------------------------------------------------------
  !
  IF (INFORM  /=  -10) THEN
     IF(PRNLEV > 2) WRITE (OUTU,800)
     INFORM = -1
     RETURN
  ELSE
     INFORM = 0
  ENDIF
  !
  MARK = 0
  IF(QEULER .AND. JCALDY  <=  3) THEN
     CALL CHKLIS(N,OPLIST,PLIST,MARK)
  ENDIF

  IF (MARK  <  0) THEN
     INFORM = -2
     RETURN
  ENDIF
  !
  MP = OPLIST(11)
  !
  ! Protect writes to MP
  IF(MP == OUTU) THEN
     QWRIT=PRNLEV > 2
  ELSE
     QWRIT=IOLEV > 0
  ENDIF
  !
  IEULER = 0
  IF (QEULER) GO TO 1417
  IF (JCALDY  <=  1 ) THEN  !! M.D.
     IF (ICALLS  ==  0) THEN
        IF(QWRIT) THEN
           WRITE (MP,805) N,(OPLIST(I),I=1,6)
           WRITE (MP,806) (OPLIST(I),I=7,16)
           WRITE (MP,807) (PLIST(I),I=1,7)
        ENDIF
     ELSE
        IF(QWRIT) WRITE (MP,808) ICALLS+1
     ENDIF
  ENDIF   !! M.D.
1417 CONTINUE
  ICALLS = ICALLS + 1
  !
  MARK = 0
  IF (QEULER .AND. JCALDY  >  1) GO TO 8200
  CALL DIVWRK(N,NZ,LW,LIW,MARK)
8200 CONTINUE

  IF (MARK  <  0) THEN
     INFORM = -3
     RETURN
  ENDIF
  !
  NSP = 3*N + 4*MAX(N,NZ)  

  ! =======  CHARMM SEGMENT ## 2 =========================
  ! Add seven more starting indices for work vectors W :
  ! W(IXD1),...,W(IXG3)

  CALL OUTER(N,X,F,G,OPLIST,PLIST,INFORM,NZ,NSP, &
       W(IXP),W(IXY),W(IXD),W(IXHD),W(IXZ),W(IXA),W(IXRSP), &
       IW(IXPER),IW(IXIPER),IW(IXIA),IW(IXJA),IW(IXISP), &
       CALFGH,CALPAT,CALHDP,W(IXD1),W(IXX1),W(IXX2), &
       W(IXX3),W(IXG1),W(IXG2),W(IXG3))
  ! =====================================================
  !
800 FORMAT(/2X,'SETLIS MUST BE CALLED BEFORE TNMIN! ')
805 FORMAT(/'  PARAMETER INFO, ENTERING TNPACK'/ &
       8X,' N = ',I8/ &
       5X,'OPLIST:'/ &
       8X,' 1) IPCG   = ',I8,8X,'(preconditioning option)'/ &
       8X,' 2) IPRINT = ',I8,8X,'(printing option)'/ &
       8X,' 3) IPFREQ = ',I8,8X,'(printing frequency)'/ &
       8X,' 4) MXITN  = ',I8,8X,'(max Newton itns.)'/ &
       8X,' 5) MXITCG = ',I8,8X,'(max PCG itns.)'/ &
       8X,' 6) MAXNF  = ',I8,8X,'(max F&G evals.)')
806 FORMAT(8X,' 7) IORDER = ',I8,8X, &
       '(M-reordering option - calc. per.)'/8X, &
       ' 8) IPERPR = ',I8,8X,'(M-reordering printing option)'/8X, &
       ' 9) IPKNOW = ',I8,8X,'(M-reordering option -', &
       ' per. known)'/8x, &
       '10) IHD    = ',I8,8X,'(Numeric HD option, IHD=1 OURHD)'/8x, &
       '11) MP     = ',I8,8X,'(printing unit)'/8x, &
       '12) IEXIT  = ',I8,8X,'(descent option for neg.', &
       ' curvature)'/8x, &
       '13) ITT    = ',I8,8X,'(truncation test option)'/8x, &
       '14) MAXFEV = ',I8,8X,'(max F evals. in line search)'/8x, &
       '15) SCHED  = ',I8,8X,'(if zero, the sched sub is ignored)'/8x, &
       '16) SEARCH = ',I8,8X,'(if 0 search direction is the def one)')
807 FORMAT(5x,'PLIST:'/8X,' 1) EPSF   = ',1PE10.3,6X, &
       '(controls conv. test, F accuracy)'/8x, &
       ' 2) EPSG   = ',1PE10.3,6X,'(controls conv. test, G accuracy)'/ &
       8X,' 3) ETAR   = ',1PE10.3,6X,'(controls res.-based truncation)'/ &
       8X,' 4) ETAQ   = ',1PE10.3,6X,'(controls quad.-model based ', &
       'truncation)'/ &
       8X,' 5) PRODK  = ',1PE10.3,6X,'(controls PCG neg curv. test)'/ &
       8X,' 6) FTOL   = ',1PE10.3,6X,'(controls F test in line search)'/ &
       8X,' 7) GTOL   = ',1PE10.3,6X,'(controls G test in line ', &
       'search)'/)
808 FORMAT(/6X,'ICALLS =',i8)
  !
  RETURN
END SUBROUTINE TNMIN
!***********************************************************************
!***********************************************************************
SUBROUTINE OUTER(N,X,F,G,OPLIST,PLIST,INFORM,NZ,NSP, &
     P,Y,D,HD,Z,A,RSP,PER,IPER,IA,JA,ISP, &
     CALFGH,CALPAT,CALHDP, &
     D1,X1,X2,X3,G11,G22,G33)
  !
  ! --------------------------------------------------------
  ! OUTER:  Outer Newton iteration of the TN method
  ! --------------------------------------------------------
  use number
  use contrl
  use stream
  use euler
  implicit none

  !
  ! argument-list variables:
  INTEGER N,NZ,NSP,INFORM,OPLIST(20),PER(N),IPER(N), &
       IA(N+1),JA(N+NZ),ISP(NSP)
  real(chm_real) F,X(N),G(N),PLIST(20),P(N),Y(N),D(N),HD(N),Z(N), &
       A(N+NZ),RSP(NSP)
  !
  ! other variables:
  INTEGER ESP,ILINE,IORDER,IPCG,IPERPR,IPFREQ,IPKNOW,IPRINT, &
       ITR,ITRCG,ITRMAJ,MAXNF,MODE,MXITCG,MXITN,NFEV, &
       NFUN,NOUT,OFLAG,SFLAG,OPATH,SPATH
  real(chm_real) GNORM,XNORM,XSNORM,RNORM,LAMBDA,FOLD, &
       ZDUM(1),RDUM(1),EPSF,EPSG,EPSF2,EPSF3,ONEF
  LOGICAL T1,T2,T3,T4,T123,T5
  !
  INTEGER LENA,I,IG,IPOINT
  real(chm_real) PERCNT,DNRM2

  ! =======  CHARMM SEGMENT ## 3 =========================
  LOGICAL OKPG,OKSCH,OKSD,OKIL6
  INTEGER IDUM,JDUM,KERROR,ITRIND,ITRINP
  real(chm_real), PARAMETER :: HAF = 0.5D0, ZER = 0.0D0
  real(chm_real) ABS,GNMOLD
  real(chm_real) D1(N),X1(N),X2(N),X3(N),G11(N),G22(N),G33(N)
  ! =====================================================
  INTEGER ICGOLD
  real(chm_real)  ADUM
  LOGICAL QWRIT
  !
  ! --------------------------------------------------------
  ! subroutines and functions called:
  !        calfgh, calpat, calhdp (user-supplied)
  !        inner, mlines, odrv, sdrvmd, dnrm2
  ! --------------------------------------------------------
  !
  EXTERNAL CALFGH, CALPAT, CALHDP
  !
  ! --------------------------------------------------------
  ! Initialize
  ! --------------------------------------------------------
  !
  OKSD = .true.
  IPCG   = OPLIST(1)
  IPRINT = OPLIST(2)
  IPFREQ = OPLIST(3)
  MXITN  = OPLIST(4)
  MXITCG = OPLIST(5)
  MAXNF  = OPLIST(6)
  IORDER = OPLIST(7)
  IPERPR = OPLIST(8)
  IPKNOW = OPLIST(9)
  IHD    = OPLIST(10)
  MP     = OPLIST(11)
  IEXIT  = OPLIST(12)
  ITT    = OPLIST(13)
  MAXFEV = OPLIST(14)
  !
  EPSF   = PLIST(1)
  EPSG   = PLIST(2)
  ETAR   = PLIST(3)
  ETAQ   = PLIST(4)
  PRODK  = PLIST(5)
  FTOL   = PLIST(6)
  GTOL   = PLIST(7)
  !
  ITRMAJ = 0
  ITRCG  = 0
  NFUN   = 0
  EPSF2  = SQRT (EPSF)
  EPSF3  = EPSF ** (1./3.)/100
  ! Protect writes to MP
  IF(MP == OUTU) THEN
     QWRIT=PRNLEV > 2
  ELSE
     QWRIT=IOLEV > 0
  ENDIF
  !
  ! --------------------------------------------------------
  ! determine the pattern of the preconditioner M
  ! --------------------------------------------------------
  !
  IF (IPCG  ==  1) THEN

     !  skip the determination of the pattern for further steps of
     !  molecular dynamics

     IF(JCALDY  <=  1) THEN
        CALL CALPAT(N,X,A,IA,JA,N+NZ)
     ENDIF
     LENA = IA(N+1) - 1
     PERCNT = (DBLE(LENA) / DBLE(N*(N+1)/2)) * 100.0
     IF (LENA  >  (N+NZ)) THEN
        INFORM = -4
        IF(QWRIT) WRITE (MP,810) LENA,N+NZ
        GOTO 1000
     ENDIF
     IF (ICALLS  <=  1 .AND. JCALDY  ==  1) THEN
        IF(QWRIT) WRITE (MP,815) LENA,PERCNT !! JCALDY pointer in M.D.
     ENDIF
  ENDIF
  IF(.NOT.QEULER) THEN
     IF(QWRIT) WRITE (MP,815) LENA,PERCNT 
  ENDIF
  !
  ! --------------------------------------------------------
  ! compute f, g, H, and M at x0
  ! --------------------------------------------------------
  !
  NOUT = 2
  IF (IPCG  ==  1) NOUT = 3 
  CALL CALFGH(N,X,F,G,A,IA,JA,N+NZ,NOUT)
  NFUN = NFUN+1
  !
  IF (IPRINT  >=  1 .AND. QWRIT ) THEN
     WRITE (MP,820)
     WRITE (MP,822) (X(I), I = 1,N)
     IF (IPRINT  >=  2) THEN
        WRITE (MP,824)
        WRITE (MP,822) (G(I), I = 1,N)
     ENDIF
  ENDIF
  !
  ! --------------------------------------------------------
  ! check for convergence at the first iteration
  ! --------------------------------------------------------
  !
  XNORM = DNRM2(N,X,1) / SQRTN
  GNORM = DNRM2(N,G,1) / SQRTN
  IF (GNORM < (EPSG * (ONE + ABS(F)))) THEN
     IF(QWRIT) WRITE (MP,828) ITRMAJ,F,GNORM
     INFORM = 0
     IF(QWRIT) WRITE (MP,829) INFORM
     ITRCG = 0
     GOTO 1000
  ENDIF
  !
  ! --------------------------------------------------------
  ! when reordering M, call ODRV to: i) compute the permutation
  ! arrays and then prepare for symmetric reordering of M, or
  ! ii) just prepare for the symmetric reordering of M when the
  ! permutation arrays are known (IPKNOW=1). Then call SDRVMD
  ! to factorize M symbolically.
  ! --------------------------------------------------------
  !
  IF (IPCG  ==  1) THEN
     !
     IF (IPKNOW  ==  1) THEN
        IF(JCALDY <=  2) THEN
           IF(PRNLEV > 2) WRITE(OUTU,*)' IPKNOW ',IPKNOW
        ENDIF
        OPATH = 5
     ELSE
        OPATH = 4
        DO I = 1,N
           PER(I) = I
           IPER(I) = I
        ENDDO
     ENDIF
     !
     OFLAG = 0
     IF (IORDER == 1) THEN
        WRITE(OUTU,*)' reordering of M required'
        CALL ODRV(N,IA,JA,A,PER,IPER,NSP,ISP,OPATH,OFLAG)
        IF (IPERPR == 1 .AND. IPKNOW /= 1) THEN
           IF(QWRIT) WRITE (MP,840)
           IF(QWRIT) WRITE (MP,842) (PER(I), I = 1, N)
        ENDIF
     ENDIF
     !
     SPATH = 4
     IF (OFLAG  ==  0) CALL SDRVMD(N,PER,IPER,IA,JA,A,RDUM,ZDUM, &
          NSP,ISP,RSP,ESP,SPATH,SFLAG,NZ)
     IF (OFLAG /= 0 .OR. SFLAG.NE.0) THEN
        IF(QWRIT) WRITE (MP,850) OFLAG,SFLAG
        GOTO 1000
     ENDIF
     !
  ENDIF
  !
  IF(JCALDY  >=  3) GO TO 4020
  IF(QWRIT) WRITE (MP,828) ITRMAJ,F,GNORM
4020 CONTINUE 
  XNORM = DNRM2(N,X,1)
  !
  ! --------------------------------------------------------
  !  MAIN LOOP BEGINS
  ! --------------------------------------------------------
  !
10 CONTINUE
  ITRMAJ = ITRMAJ+1
  MODE = 0
  ! --------------------------------------------------------
  ! print x and f if specified
  ! --------------------------------------------------------
  !
  IF (IPFREQ  >  0) THEN
     IF (MOD(ITRMAJ,IPFREQ)  ==  0) THEN
        IF(QWRIT) WRITE (MP,860) ITRMAJ
        IF(QWRIT) WRITE (MP,822) (X(I), I = 1,N)
     ENDIF
  ENDIF
  !
  ! --------------------------------------------------------
  ! check if either the number of maximum-allowed Newton
  ! iterations or function evaluations has been exceeded
  ! --------------------------------------------------------
  !
  IF (ITRMAJ  >=  MXITN) THEN
     INFORM = -5
     IF(QWRIT) WRITE (MP,870) ITRMAJ,MXITN
     GOTO 1000
  ENDIF
  !
  IF (NFUN  >  MAXNF) THEN
     INFORM = -6
     IF(QWRIT) WRITE (MP,880)
     GOTO 1000
  ENDIF
  ! --------------------------------------------------------
  ! call INNER to solve for the search vector p
  ! --------------------------------------------------------
  !
  ! =============================================================
  ! ======   CHARMM SEGMENT ## 4 for implicit  scheme  =========
  IF (QEULER) THEN
     IF (ITRMAJ  <=  1 )  MXITCG = 1 
     IF (ITRMAJ  >  2 )  MXITCG = OPLIST(5)
  ENDIF
  ! =============================================================
  CALL INNER(N,MODE,MXITCG,ITRMAJ,ITR,IPCG,NZ,NSP, &
       X,G,P,Y,PER,IPER,IA,JA,A,ISP,RSP, &
       D,HD,Z,XNORM,CALFGH,CALHDP,OKSD,GNORM,D1)
  RNORM = DNRM2(N,Y,1)
  ! --------------------------------------------------------
  ! save old value of f for convergence tests
  ! --------------------------------------------------------
  !
  FOLD = F
  ITRCG = ITRCG + ITR
  !
  ! --------------------------------------------------------
  ! call line search
  ! --------------------------------------------------------
  !
  ! =====  CHARMM SEGMENT ##  5  ==============================
  ICGOLD = ITRCG
  GNMOLD = GNORM
  IF(OPLIST(16)  ==  0 .OR. (OKSD)) GO TO 888
  CALL BESTSE(MODE,N,X,F,G,P,LAMBDA,ILINE,Y,NFUN, &
       NFEV,CALFGH,OKSD,ITRMAJ,XSNORM,OKIL6,RNORM,OKPG,GNORM, &
       FOLD,GNMOLD,D1,X1,X2,X3,G11,G22,G33)
  IF(MODE  ==  2) GO TO 2345
888 CONTINUE
  ! ==========================================================
  LAMBDA = ONE
  ! =====  CHARMM SEGMENT ##  6  ==============================
  OKIL6 = .FALSE.
  KERROR = 0
777 CONTINUE
  OKPG = .FALSE.
  GNORM = GNMOLD
  ! ============================================================

  ILINE = 0
  CALL MLINES(N,X,F,G,P,LAMBDA,ILINE,Y,NFEV,CALFGH,MODE, &
       OKSD,ITRMAJ,XSNORM,OKIL6,OKPG,GNORM,KERROR)
  NFUN = NFUN + NFEV
  !
  ! =====  CHARMM SEGMENT ##  7  ==============================
  !   avoid exit due to rounding errors and update number of 
  !   function evaluations when finite differences are used for Hd.

  IF(IHD  ==  1 .AND. GNORM/SQRTN  <=  0.5D0) NFUN = NFUN+1
  IF(KERROR  ==  1) GO TO 2345
  IF(ILINE  ==  6 .OR. ILINE .EQ. 7) THEN
     KERROR = KERROR + 1
     IF(PRNLEV  >=  9 .AND. QWRIT) WRITE(MP,1906)
     !     IF(jcaldy  >  3950 ) WRITE(MP,1906)
1906 FORMAT ('    rounding errors --> other search direction ')
     DO I=1,N
        X(I) = Y(I)
     ENDDO
     CALL CALFGH(N,X,F,G,ADUM,IDUM,JDUM,N,1)
     NFUN = NFUN + 1 
     GNORM =GNMOLD
     DO I=1,N
        P(I)=-G(I)/GNORM
     ENDDO
     OKIL6 = .TRUE.
     LAMBDA = ONE
     GO TO 777
  ENDIF
2345 CONTINUE
  ! =============================================================
  !
  ! --------------------------------------------------------
  ! summarize progress and prepare for convergence tests
  ! --------------------------------------------------------
  !
  GNORM = DNRM2(N,G,1) / SQRTN
  XNORM = DNRM2(N,X,1) / SQRTN
  GNOMI = GNORM
  IF(QEULER.AND. JCALDY  >  395000) THEN
     !       IF(GNORM  <=  0.0005 .AND. ITRMAJ  >=  2)
     IF(QWRIT) WRITE (MP,885) ITRMAJ,F,GNORM,ITR,MODE,RNORM, &
          LAMBDA,NFUN,ITRCG
  ELSE
     IF(QWRIT) WRITE (MP,885) ITRMAJ,F,GNORM,ITR,MODE,RNORM, &
          LAMBDA,NFUN,ITRCG
  ENDIF
  !
  XSNORM =  LAMBDA * (DNRM2(N,P,1) / SQRTN)
  !
  ! --------------------------------------------------------
  ! call a routine MONIT, as desired, to print further info
  ! --------------------------------------------------------
  !
  !     IF (ICALLS  ==  1) CALL MONIT(ITRMAJ,ICALLS,F,GNORM)
  !
  ! --------------------------------------------------------
  ! four convergence tests are performed:
  ! tests 1 & 2 check for convergence of the f and x sequences
  ! tests 3 & 4 check that |g| is suff. small
  ! --------------------------------------------------------
  !
  ONEF = ONE + ABS(F)
  T1 = (FOLD-F)  <  (EPSF  * ONEF)
  T5 = (FOLD-F)  >=  (EPSF  * ONEF)
  T2 =  XSNORM   <  (EPSF2 * (ONE + XNORM ) )
  T3 =  GNORM    <  (EPSF3 * ONEF)
  T4 =  GNORM    <  (EPSG  * ONEF)
  T123 = T1 .AND. (T2 .AND. T3)
  !
  ! -------------------------------------------------------------------
  ! to check conv. tests for difficult problems, set prnlev  >=  9 
  !
  IF(PRNLEV  >=  9 .AND. QWRIT) THEN
     WRITE (MP,*) '       t1:',FOLD-F,EPSF*ONEF,' ',t1
     WRITE (MP,*) '       t2:',XSNORM,EPSF2*(ONE + XNORM),' ',t2
     WRITE (MP,*) '       t3:',GNORM,EPSF3*ONEF,' ',t3
     WRITE (MP,*) '       t4:',GNORM,EPSG*ONEF,' ',t4
  ELSE 
  ENDIF
  ! -------------------------------------------------------------------
  IF (ILINE  ==  1) THEN
     IF (T123 .OR. T4) INFORM = 1
  ELSE
     IF (T3 .OR. T4) THEN
        INFORM = 2
     ELSE
        INFORM = 3
     ENDIF
     IF (ILINE  ==  0) THEN
        IF(QWRIT) WRITE (MP,900)
     ELSE IF (ILINE  ==  2) THEN
        IF(QWRIT) WRITE (MP,902)
     ELSE IF (ILINE  ==  3) THEN
        IF(QWRIT) WRITE (MP,903)
     ELSE IF (ILINE  ==  4) THEN
        IF(QWRIT) WRITE (MP,904)
     ELSE IF (ILINE  ==  5) THEN
        IF(QWRIT) WRITE (MP,905)
     ELSE IF (ILINE  ==  6) THEN
        IF(QWRIT) WRITE (MP,906)
     ELSE IF (ILINE  ==  7) THEN
        IF(QWRIT) WRITE (MP,907)
        IF(QEULER) THEN
           IF(PRNLEV > 2) WRITE(OUTU,*) ' CONTINUE FOR DYNAMICS'
        ENDIF
     ENDIF
  ENDIF
  !
  IF (INFORM == 1 .OR. ILINE /= 1) THEN
     !     IF(PRNLEV  >=  9) THEN  !! M.D.
     IF(QWRIT) THEN
        WRITE (MP,920) ITRCG
        WRITE (MP,922) INFORM
        IF (ILINE /= 1 .AND. INFORM.NE.1) WRITE (MP,924)
        IF (T1) WRITE (MP,931)
        IF (T2) WRITE (MP,932)
        IF (T3) WRITE (MP,933)
        IF (T4) WRITE (MP,934)
        WRITE (MP,936)
        !         ENDIF !! M.D. 
        IF (IPRINT  >=  1) THEN
           WRITE (MP,940)
           WRITE (MP,822) (X(I), I = 1,N)
           IF (IPRINT  >=  2) THEN
              WRITE (MP,942)
              WRITE (MP,822) (G(I), I = 1,N)
           ENDIF
        ENDIF
     ENDIF
     ! ====  small kick for dynamics if necessary ======
     IF (QEULER) THEN
        IF (ILINE  ==  3 .OR. ILINE .EQ. 4 .OR. ILINE .EQ. 5 &
             .OR. ILINE  ==  6 .OR. ILINE .EQ. 7 &
             .OR. ILINE  ==  8 ) THEN
           IF(PRNLEV > 2) WRITE(OUTU,*) &
                ' FURTHER MINIMIZATION REQUESTED '
           IEULER = 1 
        ENDIF
     ENDIF
     GOTO 1000
  ENDIF
  !
  ! --------------------------------------------------------
  ! line search OK and conv. criteria are not satisfied;
  ! compute H and M, prepare for symmetric reordering of M,
  ! and continue main loop
  ! --------------------------------------------------------
  !
  ! =====  CHARMM SEGMENT ##  8  ==============================
  ! call schedule if required 
  IF(QEULER .AND. ITRMAJ  ==  1) THEN
     !       IF(F  >  1.0E+04) OPLIST(15)=1
     !       IF(F  <=  1.0E+04) OPLIST(15)=0
  ENDIF
  IF(OPLIST(15)  ==  1) THEN
     CALL SCHEDU(ITRMAJ,N,F,IPCG,IEXIT,MXITCG,IHD,OPLIST, &
          OKSD,IA,JA,A,NZ,FOLD,ITRIND,ITRINP,GNORM,RNORM,MODE, &
          OKSCH,ITT)
  ENDIF
  ! ========================================================

  IF (IPCG == 0 .AND. IHD.EQ.1) GO TO 10 
  IF (IPCG == 0 .AND. IHD.EQ.0) THEN
     NOUT = 4
  ELSE IF (IPCG == 1 .AND. IHD.EQ.0) THEN
     NOUT = 5 
  ELSE 
     NOUT = 6 
  ENDif
  CALL CALFGH(N,X,F,G,A,IA,JA,N+NZ,NOUT)
  !
  !
  IF (IPCG == 1 .AND. IORDER.EQ.1) THEN
     OFLAG = 0
     OPATH = 5
     IF(PRNLEV > 2) WRITE(OUTU,*) &
          '   reordering of M before next Newton Iter.'
     CALL ODRV(N,IA,JA,A,PER,IPER,NSP,ISP,OPATH,OFLAG)
     IF (OFLAG /= 0) THEN
        IF(QWRIT) WRITE (MP,950) OFLAG
        GOTO 1000
     ENDIF
  ENDIF
  !
  GOTO 10
  !
  ! --------------------------------------------------------
  !  MAIN LOOP ENDS
  ! --------------------------------------------------------
  !

1000 continue
  !
810 FORMAT(/1X,' INSUFFICIENT STORAGE FOR M. LENGTH of M  = ',I8/1X, &
       ' STORAGE AVAILABLE = N+NZ = ',I8/)
815 FORMAT(/1X,'*****   upper-M has ',I8,' nonzeros (',F6.2, &
       ' % )  ***** '/)
820 FORMAT(/'  INITIAL X:  '/)
822 FORMAT(6(3X,F10.6))
824 FORMAT(/'  INITIAL G:  '/)
828 FORMAT(2X,'____________________________'/ &
       '  ITN.',1X,'    F    ',5X,'   |G|   ',2X,' #CG ',3X,'mode',2X, &
       '  |R|   ',2X,'  LAMBDA',3X,' NFG'/I5,1PE13.4,1PE12.2)
829 FORMAT(' INFORM = ',I5/' (|G(X0)| sufficiently small)'/1X, &
       '____________________________'/)
840 FORMAT(/1X,' PERMUTATION ARRAY FROM ODRV: '/)
842 FORMAT(12(1X,I5))
850 FORMAT(/1X,' OFLAG =  ',I6,' SFLAG =  ',I6 /1X,'INSUFF. STORAGE', &
       ' FOR YSMP (INCREASE NZ) OR DUPLICATE ENTRY IN A'/)
860 FORMAT(/'  X AT ITRMAJ',I5,':'/)
870 FORMAT(/1X,' MXITN (Max Newton iterations) EXCEEDED!',2(I8) /)
880 FORMAT(/1X,' MAXNF (Max function evaluations) EXCEEDED!' /)
885 FORMAT(I5,1PE13.6,1PE12.2,1X,I5,4X,I2,1X,1PE11.2,1PE11.2,I5,2x,i5)
900 FORMAT(/1X,' IMPROPER INPUT PARAMETERS DURING THE LINE SEARCH.'/)
902 FORMAT(/1X,' RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY', &
       ' IN THE LINE SEARCH'/'IS AT MOST XTOL.'/)
903 FORMAT(/1X,' NUMBER OF CALLS TO FUNCTION IN THE', &
       ' LINE SEARCH HAS REACHED MAXFEV.')
904 FORMAT(/1X,' THE STEP IN THE LINE SEARCH IS TOO SMALL.'/)
905 FORMAT(/1X,' THE STEP IN THE LINE SEARCH IS TOO LARGE.'/)
906 FORMAT(/1X,' ROUNDING ERRORS PREVENT FURTHER PROGRESS IN '/ &
       '  THE LINE SEARCH.'/)
907 FORMAT(/1X,' P IS NOT A DESCENT DIRECTION.'/)
920 FORMAT(/1X,'____________________________',2X,'#CG,tot'/29X,I7)
922 FORMAT(' INFORM = ',I5)
924 FORMAT(/' CHECK LS PARAMETERS, TRY RELAXING CONV. CRITERIA.' &
       /' AND TRY A DIFFERENT TRUNCATION PARAMETER).'/)
931 FORMAT(' CONV. TEST 1 SATISFIED ')
932 FORMAT(' CONV. TEST 2 SATISFIED ')
933 FORMAT(' CONV. TEST 3 SATISFIED ')
934 FORMAT(' CONV. TEST 4 SATISFIED ')
936 FORMAT(1X,'____________________________')
940 FORMAT(/'  FINAL X:  '/)
942 FORMAT(/'  FINAL G:  '/)
950 FORMAT(/1X,' IN TNMIN, after ODRV for new M, OFLAG =  ',I6 /1X, &
       'INSUFFICIENT STORAGE FOR YSMP. INCREASE NZ.'/)
  !
  RETURN
END SUBROUTINE OUTER
!
!***********************************************************************
!***********************************************************************
SUBROUTINE INNER(N,MODE,MXITCG,ITRMAJ,ITR,IPCG,NZ,NSP, &
     X,G,P,RES,PER,IPER,IA,JA,A,ISP,RSP, &
     D,HD,Z,XNORM,CALFGH,CALHDP,OKSD,GNORM, &
     D1)
  !
  ! --------------------------------------------------------
  ! INNER:  Inner PCG Iteration
  ! --------------------------------------------------------
  use stream
  !
  implicit none
  !
  ! argument-list variables:
  INTEGER N,MODE,MXITCG,ITRMAJ,ITR,IPCG,NZ,NSP, &
       PER(N),IPER(N),IA(N+1),JA(N+NZ),ISP(NSP)
  real(chm_real) X(N),G(N),P(N),RES(N),A(N+NZ),RSP(NSP), &
       D(N),HD(N),Z(N),XNORM
  !
  ! other variables:
  INTEGER SFLAG,SPATH,ESP,I
  real(chm_real) GNORM,DELTA,ETA,ALPHA,RNORM, &
       PRODCT,BETA,RTZ,RTZOLD,RITR,RMAJ,QOLD,QNEW,PTR,PTG
  real(chm_real)   DDOT,DNRM2
  EXTERNAL DDOT,DNRM2
  !
  ! =====  CHARMM SEGMENT ##  9  ==============================
  real(chm_real) GDP
  real(chm_real) D1(N)
  character(len=80) stri
  INTEGER IUT
  LOGICAL TIHD,OKSD
  LOGICAL QWRIT
  ! =============================================================

  ! --------------------------------------------------------
  ! subroutines and functions called:
  !        calhdp (user-supplied), ourhdp,
  !        sdrvmd, dcopy, daxpy, dnrm2, ddot
  ! --------------------------------------------------------
  !
  EXTERNAL CALFGH,CALHDP
  !
  ! --------------------------------------------------------
  ! STEP 1 - (a)  compute |g|, set p=0, res= -g,
  !          (b)  if preconditioning, perform the numerical
  !               factorization of M and solve Mz=res;
  !               else (M=I) set z=res,
  !          (c)  set d=z and compute delta=|d|**2
  ! --------------------------------------------------------
  !
  ! Protect writes to MP
  IF(MP == OUTU) THEN
     QWRIT=PRNLEV > 2
  ELSE
     QWRIT=IOLEV > 0
  ENDIF
  ! =====  CHARMM SEGMENT ##  10 ==============================
  !  Implement SD method  
  IUT = 0
  IF(OKSD) THEN
     ITR = 1
     DO I=1,N
        P(I) = - G(I)
     ENDDO
     MODE = 3
     GNORM = DNRM2(N,G,1)
     GO TO 1000
  ENDIF
  ! =============================================================
  ITR = 0
  DO I = 1, N
     P(I) = 0.0D0
     RES(I) = -G(I)
  ENDDO
  GNORM = DNRM2(N,G,1)
  QOLD = 0.D0
  !
  IF (IPCG  ==  1) THEN
     SPATH = 2  
     CALL SDRVMD(N,PER,IPER,IA,JA,A,RES,Z, &
          NSP,ISP,RSP,ESP,SPATH,SFLAG,NZ)
     IF (SFLAG > 0) THEN
        IF(QWRIT) WRITE (MP,850) SFLAG
        GOTO 1000
     ENDIF
  ELSE
#if KEY_GAMESS==1
     CALL DCOPY(int8(N),RES,1_8,Z,1_8)
#else
     CALL DCOPY(N,RES,1,Z,1)
#endif
  ENDIF
  !
#if KEY_GAMESS==1  
  CALL DCOPY(int8(N),Z,1_8,D,1_8)
  DELTA = DDOT(int8(N),D,1_8,D,1_8)
#else
  CALL DCOPY(N,Z,1,D,1)
  DELTA = DDOT(N,D,1,D,1)
#endif
  RMAJ = ITRMAJ
  !
  ! --------------------------------------------------------
  ! STEP 2 -  MAIN LOOP BEGINS:
  !          (a) compute the Hessian-vector product Hd, and
  !          (b) if d*Hd is small, exit with a descent direction
  !              chosen according to IEXIT. (On ITR 1, p=-g or
  !              p=d= -[M**(-1)]g, and on ITR>1, p= -g,
  !              current p, or current d)
  ! --------------------------------------------------------
  !
10 CONTINUE
  IF (ITR  >=  MXITCG) THEN
     MODE = 3
     GOTO 1000
  ENDIF
  IF (IHD  ==  1) THEN 
     CALL OURHDP(N,D,HD,X,G,RSP(1),ITR,XNORM,CALFGH)
     TIHD = .FALSE.
  ENDIF
  ! =====  CHARMM SEGMENT ## 11  ==============================
  IF(IHD  ==  0) THEN
     IF(IPCG  ==  1) THEN
        CALL HDPNEW(ITR,N,D,HD,ITRMAJ)
     ELSE 
        CALL CALHDP(N,D,HD,X,G)
     ENDIF
     TIHD = .TRUE.
  ENDIF
  ! =============================================================  
  !
  ITR = ITR + 1
  RITR = ITR
  !
#if KEY_GAMESS==1  
  PRODCT = DDOT(int8(N),D,1_8,HD,1_8)
#else
  PRODCT = DDOT(N,D,1,HD,1)
#endif
  IF(PRNLEV  >=  9) THEN
     IF(PRODCT  <=  0.0D0 .AND. (TIHD)) &
          WRITE(OUTU,*)'           dT H d  by USER ',prodct
     IF(PRODCT  <=  0.0D0 .AND. .NOT.TIHD) &
          WRITE(OUTU,*)'           dT H d  by OURHD ',prodct
  ENDIF

  IF (PRODCT  <=  (DELTA*PRODK) ) THEN
     IF (ITR  ==  1) THEN
        MODE = 1
        ! =====  CHARMM SEGMENT ##  12  ==============================
        IUT = IUT + 1
        IF(IUT  ==  1) GO TO 707
        ! =============================================================
        IF (IEXIT  >  3) THEN
#if KEY_GAMESS==1
           CALL DCOPY(int8(N),D,1_8,P,1_8)
#else
           CALL DCOPY(N,D,1,P,1)
#endif
        ELSE
           ! =====  CHARMM SEGMENT ##  13  ==============================
#if KEY_GAMESS==1
           IF(IEXIT  /=  2) CALL DCOPY(int8(N),RES,1_8,P,1_8)
           IF(IEXIT  ==  2) CALL DCOPY(int8(N),D,1_8,P,1_8)
#else
           IF(IEXIT  /=  2) CALL DCOPY(N,RES,1,P,1)
           IF(IEXIT  ==  2) CALL DCOPY(N,D,1,P,1)
#endif
           ! =============================================================
        ENDIF
     ELSE
        MODE = 2
        IF (IEXIT == 0 .OR. IEXIT.EQ.3) THEN
           DO I = 1, N
              P(I) = -G(I)
           ENDDO
        ELSE IF (IEXIT == 2 .OR. IEXIT.EQ.5) THEN
#if KEY_GAMESS==1
           CALL DCOPY(int8(N),D,1_8,P,1_8)
#else
           CALL DCOPY(N,D,1,P,1)
#endif
        ENDIF
     ENDIF
     GOTO 1000
  ENDIF
  !
  ! --------------------------------------------------------
  ! STEP 3 - (a)  update  p and  res, and
  !          (b)  check for the PCG termination criterion
  !               (residual-based or quadratic-model based)
  ! --------------------------------------------------------
  !
707 CONTINUE
#if KEY_GAMESS==1  
  RTZ = DDOT(int8(N),RES,1_8,Z,1_8)
#else
  RTZ = DDOT(N,RES,1,Z,1)
#endif
  ALPHA = RTZ/PRODCT

  ! =====  CHARMM SEGMENT ##  14  ==============================
  !     Force more CG itns. even when negative curvature is found
  !     at the first iteration (a CG itn. is much cheaper than
  !     a Newton one).
  IF(MODE  ==  1 .AND. IUT.EQ. 1) THEN
     ALPHA = - ALPHA
     IF(PRNLEV  >= 9) &
          WRITE(OUTU,*)'      mode = 1, alpha ---> -alpha '
     IUT = 0 
  ENDIF
  ! ==============================================================

  RTZOLD = RTZ
#if KEY_GAMESS==1
  CALL DAXPY(int8(N),ALPHA,D,1_8,P,1_8)
#else
  CALL DAXPY(N,ALPHA,D,1,P,1)
#endif
  ! =====  CHARMM SEGMENT ##  15  ==============================
  DO I=1,N
     D1(I) = P(I)
  ENDDO
  ! =============================================================

#if KEY_GAMESS==1
  CALL DAXPY(int8(N),-ALPHA,HD,1_8,RES,1_8)
#else
  CALL DAXPY(N,-ALPHA,HD,1,RES,1)
#endif
  !
  IF (ITT  ==  1) THEN
#if KEY_GAMESS==1  
     PTR = DDOT(int8(N),P,1_8,RES,1_8)
     PTG = DDOT(int8(N),P,1_8, G ,1_8)
#else
     PTR = DDOT(N,P,1,RES,1)
     PTG = DDOT(N,P,1, G ,1)
#endif
     QNEW = 0.5D0 * (PTR + PTG)
     IF (RITR*(1.D0 - (QOLD/QNEW))  <=  ETAQ) THEN
        MODE = 0
        GOTO 1000
     ENDIF
     QOLD = QNEW
     !
  ELSE
     !
     RNORM = DNRM2(N,RES,1)
     ETA = MIN(ETAR/RMAJ,GNORM)
     IF (RNORM  <=  (GNORM*ETA)) THEN
        MODE = 0
        GOTO 1000
     ENDIF
     !
  ENDIF
  !
  ! --------------------------------------------------------
  ! STEP 4 - Truncation criterion was not satisfied; continue PCG.
  !          (a)  if preconditioning, solve Mz=res by re-using
  !               the factors of M; else (M=I) set z=res,
  !          (b)  update d and delta, and
  !          (c)  go on to the next PCG iteration.
  ! --------------------------------------------------------
  !
  IF (IPCG  ==  1) THEN
     SPATH = 3
     CALL SDRVMD(N,PER,IPER,IA,JA,A,RES,Z, &
          NSP,ISP,RSP,ESP,SPATH,SFLAG,NZ)
     IF (SFLAG  >  0) THEN
        IF(QWRIT) WRITE (MP,850) SFLAG
        GOTO 1000
     ENDIF
  ELSE
#if KEY_GAMESS==1
     CALL DCOPY(int8(N),RES,1_8,Z,1_8)
#else
     CALL DCOPY(N,RES,1,Z,1)
#endif
  ENDIF
  !
#if KEY_GAMESS==1
  RTZ = DDOT(int8(N),RES,1_8,Z,1_8)
#else
  RTZ = DDOT(N,RES,1,Z,1)
#endif
  BETA = RTZ/RTZOLD
  DO I = 1, N
     D(I) = BETA*D(I) + Z(I)
  ENDDO
#if KEY_GAMESS==1
  DELTA = DDOT(int8(N),D,1_8,D,1_8)
#else
  DELTA = DDOT(N,D,1,D,1)
#endif
  !
  GOTO 10
  !
  ! --------------------------------------------------------
  !  MAIN LOOP ENDS
  ! --------------------------------------------------------
  !
1000 CONTINUE
  !
850 FORMAT(/1X,' IN TNPCG, SFLAG =  ',I6 /1X,'INSUFF. STORAGE &
       &FOR YSMP (INCREASE NZ) OR DUPLICATE ENTRY IN A'/)
  !
  RETURN
END SUBROUTINE INNER
!***********************************************************************
!***********************************************************************
SUBROUTINE OURHDP (N,D,HD,X,G,Y,ITR,XNORM,CALFGH)
  !
  ! ------------------------------------------------------
  ! OURHDP: Our numeric Hessian/vector multiplication
  ! ------------------------------------------------------
  !
  use number
  implicit none
  real(chm_real), PARAMETER :: DMAX=0.1D0, &
       DIMAX=ONE/DMAX, FRAC=0.01D0
  INTEGER N,ITR
  real(chm_real) D(N),HD(N),X(N),G(N),Y(N),XNORM, &
       DELH,DELHI,DNUM,DDEN,DNORM, GNORM
  SAVE DNUM
  EXTERNAL CALFGH
  !
  INTEGER I,IADUM,JADUM
  real(chm_real) ADUM,FNEW
  ! =====  CHARMM SEGMENT ##  16  ===========================
  real(chm_real) DDOT,DNRM2
  EXTERNAL DDOT,DNRM2
  real(chm_real) DMMAX,DELH1
  ! =======================================================
  !
  ! --------------------------------------------------------
  ! DELH is set to:  2*SQRT(EPS)*(1+|X|) / |D|, with division
  ! precautions and upper/lower limits
  ! --------------------------------------------------------
  IF (ITR  ==  0) DNUM = SQEPS2 * (ONE + XNORM*SQRTN)
  DNORM = DNRM2(N,D,1)
  DDEN = MAX(DNUM*DIMAX, DNORM)
  DELH = MAX(DNUM/DDEN, DNUM*FRAC)

  ! =====  CHARMM SEGMENT ##  17  ===========================
  !   Hd product is switched between forward and central
  !   finite differences depending on the value of the gradient norm. 
  !   This balances accuracy with computation. 
  !
  DMMAX = ABS(D(1))
  DO I=1,N
     DMMAX = MAX(DMMAX,ABS(D(I)))
  ENDDO
  DELH1 = SQEPS2/DMMAX
  DELH = MIN(DELH1,DELH)
  DELHI = ONE/DELH

  DO I = 1, N
     Y(I) = X(I) + DELH*D(I)
  ENDDO

  CALL CALFGH(N,Y,FNEW,HD,ADUM,IADUM,JADUM,N,1)
  GNORM = DNRM2(N,G,1)
  IF(GNORM/SQRT(DBLE(N))  <=  0.5D0) THEN
     DO I=1,N
        Y(I) = X(I) - DELH*D(I)
     ENDDO
     CALL CALFGH(N,Y,FNEW,X,ADUM,IADUM,JADUM,N,1)
     DO I = 1, N
        HD(I) = (HD(I)-X(I))* DELHI/TWO
        X(I) = Y(I) + DELH*D(I)
     ENDDO
  ELSE 
     DO I = 1, N
        HD(I) = ( HD(I) - G(I) ) * DELHI
     ENDDO
  ENDIF
  ! ==========================================================
  RETURN
END SUBROUTINE OURHDP
!***********************************************************************
!***********************************************************************
SUBROUTINE SETLIS(N,OPLIST,PLIST,INFORM)
  !
  ! --------------------------------------------------------
  ! SETLIS:  Set sample values for the TNMIN-call list of
  !          options and parameters (OPLIST & PLIST).
  !          The user should then change some of these items
  !          - to suit the problem - before the TNMIN call
  ! --------------------------------------------------------
  use stream
  implicit none
  !
  INTEGER N,OPLIST(20),INFORM
  real(chm_real) PLIST(20)
  !
  real(chm_real)   DDOT,DNRM2
  EXTERNAL DDOT,DNRM2
  !
  ICALLS = 0
  !
  ! --------------------------------------------------------
  ! Brief Description of parameters (see manual for details;
  ! M below refers to the preconditioner):
  ! --------------------------------------------------------
  ! INFORM     -   diagnostic/status parameter
  ! ICALLS     -   counter for number of TNPACK calls (multiple
  !                calls may be made for a series of minimizations)
  ! OPLIST(1)  -   preconditioning option (0 - No, 1 - Yes)
  ! OPLIST(2)  -   general output controler (0,1,2)
  ! OPLIST(3)  -   output controler, frequency of printing X (0,1,...)
  ! OPLIST(4)  -   limit of total Newton iterations
  ! OPLIST(5)  -   limit of PCG iterations in each Newton iteration
  ! OPLIST(6)  -   limit of total function evaluations
  ! OPLIST(7)  -   M-reordering option (0 - Do not reorder, 1 - Reorder)
  ! OPLIST(8)  -   printing option of M's reordering (permutation array)
  !                (0 - Do not print, 1 - Print)
  ! OPLIST(9)  -   option for specifying whether the permutation array
  !                for reordering M is known (0 - Not known, 1 - Known).
  !                (This option is useful for multiple TNPACK minimiza-
  !                tions, involving reorderings of M, if the sparsity
  !                structure of M does not change. The user can reset
  !                OPLIST(9) from 0 to 1 before the second TNMIN call).
  ! OPLIST(10) -   Hessian/vector multiplication option (0 - use a user-
  !                supplied routine, 1 - use our default finite-difference
  !                design routine); IHD in COMMON/TN002
  ! OPLIST(11) -   unit number for printing; MP in COMMON/TN001
  ! OPLIST(12) -   option selector for combination of exit directions in
  !                case of detection of negative curvature for PCG itns. 1
  !                and >1 (0: -G/-G, 1: -G/P, 2: -G/D, 3: -Y/-G, 4: -Y/P,
  !                5: -Y/D, where Y=-[M**(-1)]G); IEXIT in COMMON/TN002
  ! OPLIST(13) -   option selector for PCG truncation test
  !                (0: residual-based, 1: quadratic-model based);
  !                ITT in COMMON/TN002
  ! OPLIST(14) -   limit for number of function calls in the line search
  !                (at each Newton iteration); MAXFEV in COMMON/TN003
  ! PLIST(1)   -   desired   F   accuracy in min. conv. test
  ! PLIST(2)   -   desired ||G|| accuracy in min. conv. test
  ! PLIST(3)   -   truncation parameter for residual test;
  !                ETAR in COMMON/TN002
  ! PLIST(4)   -   truncation parameter for quadratic-model test;
  !                ETAQ in COMMON/TN002
  ! PLIST(5)   -   tolerance for 'negative curvature' test;
  !                PRODK in COMMON/TN002
  ! PLIST(6)   -   line-search conv. parameter, controlling function
  !                decrease; FTOL in COMMON/TN003
  ! PLIST(7)   -   line-search conv. parameter, controlling reduction
  !                of derivative magnitude; GTOL in COMMON/TN003
  ! --------------------------------------------------------
  !
  IF (N  <=  0) THEN
     IF(WRNLEV >= -2) WRITE (OUTU,800) N
     RETURN
  ENDIF
  !
  INFORM = -10
  !
  OPLIST(1) = 0
  OPLIST(2) = 0
  OPLIST(3) = 0
  OPLIST(4) = 100 * N
  OPLIST(5) = N
  OPLIST(6) = 100 * N
  OPLIST(7) = 0
  OPLIST(8) = 0
  OPLIST(9) = 0
  OPLIST(10)= 0 
  OPLIST(11)= 6
  OPLIST(12)= 2 
  OPLIST(13)= 0 
  OPLIST(14)= 60
  !
  MP = OPLIST(11)
  EPSMCH = epsilon(EPSMCH)
  SQEPS2 = 2.D0 * SQRT(EPSMCH)

  SQRTN  = SQRT( DBLE(N) )
  !
  PLIST(1)  = 1.0D-08
  PLIST(2)  = SQRT(EPSMCH)
  PLIST(3)  = 0.5D0
  PLIST(4)  = 0.5D0
  PLIST(5)  = 1.0D-10
  PLIST(6)  = 1.0D-04
  PLIST(7)  = 0.9D0
  !
800 FORMAT(/2X,' N <= 0, N = ',I10)
  RETURN
END SUBROUTINE SETLIS
!
!***********************************************************************
!***********************************************************************
SUBROUTINE CHKLIS(N,OPLIST,PLIST,MARK)
  !
  ! --------------------------------------------------------
  ! CHKLIS:  Check input options & parameters (OPLIST,PLIST)
  ! --------------------------------------------------------
  use number
  use euler
  use stream
  implicit none
  !
  INTEGER N,OPLIST(20),MARK
  real(chm_real) PLIST(20)
  real(chm_real) SQEPS
  !
  SQEPS = SQRT(EPSMCH)
  !
  IF (N  <=  0) THEN
     IF(PRNLEV > 2) WRITE (OUTU,800) N
     MARK = -1
     RETURN
  ENDIF
  !
  IF (OPLIST(1) /= 0 .AND. OPLIST(1).NE.1) THEN
     OPLIST(1) = 0
     IF(PRNLEV > 2) WRITE (OUTU,801)
  ENDIF
  !
  IF (OPLIST(2) < 0 .OR. OPLIST(2) > 2) THEN
     OPLIST(2) = 0
     IF(PRNLEV > 2) WRITE (OUTU,802)
  ENDIF
  !
  IF (OPLIST(3)  <  0) THEN
     OPLIST(3) = 0
     IF(PRNLEV > 2) WRITE (OUTU,803)
  ENDIF
  !
  IF (OPLIST(4) <= 0 .OR. OPLIST(5).LE.0 .OR. OPLIST(6).LE.0) THEN
     IF(PRNLEV > 2) WRITE (OUTU,804)
     MARK = -4
     RETURN
  ENDIF
  !
  IF (OPLIST(1)  ==  0) THEN
     OPLIST(7) = 0
     OPLIST(8) = 0
     OPLIST(9) = 0
  ELSE
     IF (OPLIST(7) /= 0 .AND. OPLIST(7).NE.1) THEN
        OPLIST(7) = 0
        IF(PRNLEV > 2) WRITE (OUTU,807)
     ENDIF
     IF (OPLIST(7) == 0 .AND. &
          (OPLIST(8) /= 0 .OR. OPLIST(9).NE.0)) THEN
        OPLIST(8) = 0
        IF (QEULER) THEN
           OPLIST(9) = 1
        ELSE
           OPLIST(9) = 0
        ENDIF
     ENDIF
  ENDIF
  !
  IF (OPLIST(10) /= 0 .AND. OPLIST(10).NE.1) THEN
     OPLIST(10) = 1
     IF(PRNLEV > 2) WRITE (OUTU,810)
  ENDIF
  !
  IF (OPLIST(11)  <=  0) THEN
     IF(PRNLEV > 2) WRITE (OUTU,811)
     MARK = -11
     RETURN
  ENDIF
  !
  IF (OPLIST(12) < 0 .OR. OPLIST(12) > 5) THEN
     OPLIST(12) = 1
     IF(PRNLEV > 2) WRITE (OUTU,812) OPLIST(12)
  ENDIF
  !
  IF (OPLIST(13) /= 0 .AND. OPLIST(13).NE.1) THEN
     OPLIST(13) = 0
     IF(PRNLEV > 2) WRITE (OUTU,813)
  ENDIF
  !
  IF (OPLIST(14) <= 0 .OR. OPLIST(14) > 80) THEN
     IF(PRNLEV > 2) WRITE (OUTU,814)
  ENDIF
  !
  IF (PLIST(1)  <  EPSMCH) THEN
     PLIST(1) = EPSMCH*100.D0 
     IF(PRNLEV > 2) WRITE (OUTU,821) PLIST(1)
  ENDIF
  !
  IF (PLIST(2)  <  EPSMCH) THEN
     PLIST(2) = SQEPS*10.D0 
     IF(PRNLEV > 2) WRITE (OUTU,822) PLIST(2),SQEPS
  ENDIF
  !
  IF (OPLIST(13)  ==  0) THEN
     IF (PLIST(3) < ZERO .OR. PLIST(3) > ONE) THEN
        PLIST(3) = 0.5D0
        IF(PRNLEV > 2) WRITE (OUTU,823) PLIST(3)
     ELSE IF (PLIST(3)  <  1.D-4) THEN
        IF(PRNLEV > 2) WRITE (OUTU,833)
     ENDIF
  ENDIF
  !
  IF (OPLIST(13)  ==  1) THEN
     IF (PLIST(4) < ZERO .OR. PLIST(4) > ONE) THEN
        PLIST(4) = 0.5D0
        IF(PRNLEV > 2) WRITE (OUTU,824) PLIST(4)
     ELSE IF (PLIST(4)  <  1.D-4) THEN
        IF(PRNLEV > 2) WRITE (OUTU,834)
     ENDIF
  ENDIF
  !
  IF (PLIST(5)  <  EPSMCH) THEN
     PLIST(5) = SQEPS*0.1D0
     IF(PRNLEV > 2) WRITE (OUTU,825) PLIST(5)
  ENDIF
  !
  IF (PLIST(6) < ZERO .OR. PLIST(7).LT.ZERO .OR. &
       PLIST(6) > ONE  .OR. PLIST(7).GT.ONE  .OR. &
       PLIST(6) >= PLIST(7)) THEN
     PLIST(6) = 1.0D-04
     PLIST(7) = 0.9D0
     IF(PRNLEV > 2) WRITE (OUTU,826)
  ENDIF
  !
800 FORMAT(/2X,'N <= 0, N = ',I10)
801 FORMAT(/2X,'OPLIST(1)  OUT-OF-RANGE, reset to 0 (no PCG)')
802 FORMAT(/2X,'OPLIST(2)  OUT-OF-RANGE, reset to 0')
803 FORMAT(/2X,'OPLIST(3)  OUT-OF-RANGE, reset to 0')
804 FORMAT(/2X,'OPLIST (4,5, and/or 6) <= 0')
807 FORMAT(/2X,'OPLIST(7)  OUT-OF-RANGE, reset to 0 (no reordering)')
810 FORMAT(/2X,'OPLIST(10) OUT-OF-RANGE, reset to 1 (our Hd routine)')
811 FORMAT(/2X,'OPLIST(11), PRINTING UNIT, <= 0')
812 FORMAT(/2X,'OPLIST(12)  OUT-OF-RANGE, reset to',I10)
813 FORMAT(/2X,'OPLIST(13)  OUT-OF-RANGE, reset to 0 (residual test)')
814 FORMAT(/2X,'OPLIST(14) <=0 or unreasonably high, reset to 20')
821 FORMAT(/2X,'PLIST(1) < EPSMCH, reset to',1PE15.5)
822 FORMAT(/2X,'PLIST(2) < EPSMCH, reset to',1PE15.5/2X, &
       'However, around SQRT(EPSMCH) is recommended:',1PE15.5)
823 FORMAT(/2X,'PLIST(3)  OUT-OF-RANGE, reset to',1PE15.5)
824 FORMAT(/2X,'PLIST(4)  OUT-OF-RANGE, reset to',1PE15.5)
833 FORMAT(/2X,'NOTE: A SMALL INPUT VALUE FOR PLIST(3) MAY NOT' &
       /2X,'EXPLOIT THE BENEFIT OF TRUNCATION')
834 FORMAT(/2X,'NOTE: A SMALL INPUT VALUE FOR PLIST(4) MAY NOT' &
       /2X,'EXPLOIT THE BENEFIT OF TRUNCATION')
825 FORMAT(/2X,'PLIST(5) < EPSMCH, reset to',1PE15.5)
826 FORMAT(/2X,'PLIST(6 and/or 7)  OUT-OF-RANGE, both were reset'/2X, &
       'to default values')
  !
  RETURN
END SUBROUTINE CHKLIS
!
!***********************************************************************
!***********************************************************************
SUBROUTINE DIVWRK(N,NZ,LW,LIW,MARK)
  !
  ! --------------------------------------------------------
  ! DIVWRK:  Compute starting indices for work vectors W,IW
  ! --------------------------------------------------------
  use stream
  implicit none
  !
  INTEGER N,NZ,LW,LIW,MARK,NSP,NEEDW,NEEDIW
  INTEGER NZH
  !
  ! --------------------------------------------------------
  ! W is partitioned to vectors:
  !   P(n),Y(n),D(n),HD(n),Z(n),A(n+nz),RSP(nsp),
  !   D1(n),X1(n),X2(n),X3(n),G11(n),G22(n),G33(n)
  ! IW is partitioned to vectors:
  !   PER(n),IPER(n),IA(n+1),JA(n+nz),ISP(nsp),
  ! where   nz  =  # of nonzeros in strict upper-triang. of M,
  ! and     nsp =  3*n + 4*max(n,nz).
  ! --------------------------------------------------------
  !
  NSP    = 3*N + 4*MAX(N,NZ)
  NEEDW  = 6*N + NZ + NSP + 7*N   !!  seven arrays are added
  NEEDIW = 4*N + NZ + NSP + 1
  IF (N <= 0 .OR. NZ < 0 .OR. &
       LW < NEEDW .OR. LIW.LT.NEEDIW) THEN
     IF(PRNLEV > 2) WRITE (OUTU,800) N, NZ, LW, NEEDW, LIW, NEEDIW
     MARK = -1
     RETURN
  ENDIF
  !
  IXP    = 1
  IXY    = IXP    + N
  IXD    = IXY    + N
  IXHD   = IXD    + N
  IXZ    = IXHD   + N
  IXA    = IXZ    + N
  IXRSP  = IXA    + N + NZ
  ! =======  CHARMM SEGMENT ## 19 =========================
  ! Definition of the seven new indices. 
  IXD1   = IXRSP + N
  IXX1   = IXD1  + N
  IXX2   = IXX1  + N
  IXX3   = IXX2  + N
  IXG1   = IXX3  + N
  IXG2   = IXG1  + N
  IXG3   = IXG2  + N
  ! =======================================================
  !
  IXPER  = 1
  IXIPER = 1      + N
  IXIA   = IXIPER + N
  IXJA   = IXIA   + N + 1
  IXISP  = IXJA   + N + NZ

  !
800 FORMAT(/2X,'ERROR in DIMENSIONS of N,NZ,LW,or LIW'/2X, &
       'N = ',I10,'NZ = ',I10/2X,'LW = ',I10,'(NEED ',I10,')', &
       'LIW = ', I10,'(NEED ',I10,')')
  !
  RETURN
END SUBROUTINE DIVWRK
!***********************************************************************
!***********************************************************************
SUBROUTINE MLINES(N,X,F,G,S,STP,INFO,WA,NFEV,CALFGH,MODE, &
     OKSD,ITRMAJ,XSNORM,OKIL6,OKPG,GNORM,KERROR)
  !
  ! --------------------------------------------------------
  ! MLINES:  Line-search algorithm of More' and Thuente
  ! --------------------------------------------------------
  !
  use number
  use euler
  use stream
  implicit none

  ! argument-list variables
  INTEGER N,INFO,NFEV,MODE
  real(chm_real) F,G(N),S(N),STP,WA(N),X(N)
  !
  ! other variables
  INTEGER INFOC
  real(chm_real) DG,DGM,DGINIT,DGTEST,DGX,DGXM,DGY, &
       DGYM, FINIT, FTEST1, FM, FX, FXM, FY, FYM, &
       STX,STY,STMIN,STMAX, &
       WIDTH,WIDTH1
  LOGICAL BRACKT,STAGE1
  real(chm_real) ADUM
  INTEGER IDUM,JDUM 
  SAVE
  ! =====  CHARMM SEGMENT ##  20  ===========================
  LOGICAL OKMOD,OKSD,OKIL6,OKPG
  INTEGER ITRMAJ,KERROR
  character(len=80) stri
  real(chm_real) XSNORM,GNORM,FTOL1
  real(chm_real) DNRM2,DDOT
  EXTERNAL DNRM2,DDOT
  ! =====================================================

  ! --------------------------------------------------------
  ! subroutines and functions called:
  !       calfgh       (user-supplied)
  !       newstp
  !       daxpy, dcopy, ddot  (blas)
  ! --------------------------------------------------------
  EXTERNAL CALFGH
  !
  real(chm_real) :: P5=0.5_chm_real,P66=0.66_chm_real,XTRAPF=4.0_chm_real
  real(chm_real) :: XTOL=1.0E-17_chm_real,STPMIN=1.0E-20_chm_real,STPMAX=1.0E+20_chm_real
  INFOC = 1
  ! --------------------------------------------------------
  !    CHECK THE INPUT PARAMETERS FOR ERRORS.
  ! --------------------------------------------------------
  IF (N <= 0 .OR. STP.LE.ZERO .OR. FTOL < ZERO .OR. &
       GTOL < ZERO .OR. XTOL.LT.ZERO .OR. STPMIN.LT.ZERO &
       .OR. STPMAX < STPMIN .OR. MAXFEV <= 0) THEN
     IF(WRNLEV > 0) WRITE(OUTU,*) &
          n,stp,ftol,gtol,xtol,stpmin,stpmax,maxfev
  ENDIF
  IF (N <= 0 .OR. STP.LE.ZERO .OR. FTOL < ZERO .OR. &
       GTOL < ZERO .OR. XTOL.LT.ZERO .OR. STPMIN.LT.ZERO &
       .OR. STPMAX < STPMIN .OR. MAXFEV <= 0) RETURN
  ! --------------------------------------------------------
  !     COMPUTE THE INITIAL GRADIENT IN THE SEARCH DIRECTION
  !     AND CHECK THAT S IS A DESCENT DIRECTION.
  ! --------------------------------------------------------
#if KEY_GAMESS==1
  DGINIT = DDOT(int8(N),G,1_8,S,1_8)
#else
  DGINIT = DDOT(N,G,1,S,1)
#endif
  !
  ! --------------------------------------------------------
  !     INITIALIZE LOCAL VARIABLES.
  ! --------------------------------------------------------
  BRACKT = .FALSE.
  STAGE1 = .TRUE.
  NFEV = 0
  FINIT = F
  DGTEST = FTOL*DGINIT
  WIDTH = STPMAX - STPMIN
  WIDTH1 = WIDTH/P5
#if KEY_GAMESS==1
  CALL DCOPY(int8(N),X,1_8,WA,1_8) 
#else
  CALL DCOPY(N,X,1,WA,1) 
#endif
  ! =====  CHARMM SEGMENT ##  21  ======================
  !   scale the search direction before computing lambda
  FTOL1 = FTOL
  CALL SCADIR(N,MODE,OKMOD,OKIL6,OKSD,ITRMAJ, &
       GNORM,XSNORM,G,S,DGINIT,FTOL1,DGTEST,OKPG)
  ! ====================================================

  IF (DGINIT  >=  ZERO) THEN
     IF(PRNLEV  >=  9) THEN
        WRITE (OUTU,*) '  GTP in MLINES = ',DGINIT
        WRITE (OUTU,*) &
             '      the next direction is not a descent one'
     ENDIF
     INFO = 7
     RETURN
  ENDIF
  ! --------------------------------------------------------
  !     THE VARIABLES STX, FX, DGX CONTAIN THE VALUES OF THE STEP,
  !     FUNCTION, AND DIRECTIONAL DERIVATIVE AT THE BEST STEP.
  !     THE VARIABLES STY, FY, DGY CONTAIN THE VALUE OF THE STEP,
  !     FUNCTION, AND DERIVATIVE AT THE OTHER ENDPOINT OF
  !     THE INTERVAL OF UNCERTAINTY.
  !     THE VARIABLES STP, F, DG CONTAIN THE VALUES OF THE STEP,
  !     FUNCTION, AND DERIVATIVE AT THE CURRENT STEP.
  ! --------------------------------------------------------
  STX = ZERO
  FX = FINIT
  DGX = DGINIT
  STY = ZERO
  FY = FINIT
  DGY = DGINIT
  ! --------------------------------------------------------
  !     START OF ITERATION.
  ! --------------------------------------------------------
30 CONTINUE
  ! --------------------------------------------------------
  !        SET THE MINIMUM AND MAXIMUM STEPS TO CORRESPOND
  !        TO THE PRESENT INTERVAL OF UNCERTAINTY.
  ! --------------------------------------------------------
  IF (BRACKT) THEN
     STMIN = MIN(STX,STY)
     STMAX = MAX(STX,STY)
  ELSE
     STMIN = STX
     STMAX = STP + XTRAPF*(STP - STX)
  END IF
  ! --------------------------------------------------------
  !        FORCE THE STEP TO BE WITHIN THE BOUNDS STPMAX AND STPMIN.
  ! --------------------------------------------------------
  STP = MAX(STP,STPMIN)
  STP = MIN(STP,STPMAX)
  ! --------------------------------------------------------
  !        IF AN UNUSUAL TERMINATION IS TO OCCUR THEN LET
  !        STP BE THE LOWEST POINT OBTAINED SO FAR.
  ! --------------------------------------------------------
  IF ((BRACKT .AND. (STP <= STMIN .OR. STP >= STMAX)) &
       .OR. NFEV >= MAXFEV-1 .OR. INFOC == 0 &
       .OR. (BRACKT .AND. STMAX-STMIN <= XTOL*STMAX)) STP = STX
  ! --------------------------------------------------------
  !        EVALUATE THE FUNCTION AND GRADIENT AT STP
  !        AND COMPUTE THE DIRECTIONAL DERIVATIVE.
  ! --------------------------------------------------------

  ! =====  CHARMM SEGMENT ##  22 ========================
  IF(KERROR  ==  1) GO TO 707
  IF(STP  <=  1.0D-4 .AND. .NOT.QEULER) THEN
     INFO = 6
     RETURN
  ENDIF
  IF(STP  <=  1.0D-4 .AND. QEULER) THEN
     INFO = 6
     RETURN
  ENDIF
707 CONTINUE
  ! =====================================================
#if KEY_GAMESS==1
  CALL DCOPY(int8(N),WA,1_8,X,1_8)
  CALL DAXPY(int8(N),STP,S,1_8,X,1_8)
#else
  CALL DCOPY(N,WA,1,X,1)
  CALL DAXPY(N,STP,S,1,X,1)
#endif
  CALL CALFGH(N,X,F,G,ADUM,IDUM,JDUM,N,1)

  INFO = 0
  NFEV= NFEV + 1
#if KEY_GAMESS==1
  DG = DDOT(int8(N),G,1_8,S,1_8)
#else
  DG = DDOT(N,G,1,S,1)
#endif
  FTEST1 = FINIT + STP*DGTEST
  ! --------------------------------------------------------
  !        TEST FOR CONVERGENCE.
  ! --------------------------------------------------------
  IF ((BRACKT .AND. (STP <= STMIN .OR. STP >= STMAX)) &
       .OR. INFOC == 0) INFO = 6
  IF (STP == STPMAX .AND. &
       F <= FTEST1 .AND. DG.LE.DGTEST) INFO = 5
  IF (STP == STPMIN .AND. &
       (F > FTEST1 .OR. DG >= DGTEST)) INFO = 4
  IF (NFEV  >=  MAXFEV) INFO = 3
  IF (BRACKT .AND. STMAX-STMIN <= XTOL*STMAX) INFO = 2

  ! =====  CHARMM SEGMENT ##  23  ===========================
  !       IF (F <= FTEST1 .AND.(ABS(DG).LE.GTOL*(-DGINIT)))INFO=1 
  IF (F <= FTEST1 .AND.(ABS(DG).LE.GTOL*(-DGINIT).OR. &
       GTOL*ABS(DG) >= (-DGINIT) ))INFO = 1

  IF(PRNLEV  >=  9) THEN
     IF(F <= FTEST1 .AND.GTOL*ABS(DG) >= (-DGINIT)) &
          WRITE(OUTU,*)'c2'
  ELSE 
  ENDIF
  ! =========================================================

  ! --------------------------------------------------------
  !        CHECK FOR TERMINATION.
  ! -------------------------------------------------------
  IF (INFO  /=  0) RETURN
  ! --------------------------------------------------------
  !        IN THE FIRST STAGE WE SEEK A STEP FOR WHICH THE MODIFIED
  !        FUNCTION HAS A NONPOSITIVE VALUE AND NONNEGATIVE DERIVATIVE.
  ! --------------------------------------------------------
  IF (STAGE1 .AND. F <= FTEST1 .AND. &
       DG >= MIN(FTOL,GTOL)*DGINIT) STAGE1 = .FALSE.
  ! --------------------------------------------------------
  !        A MODIFIED FUNCTION IS USED TO PREDICT THE STEP ONLY IF
  !        WE HAVE NOT OBTAINED A STEP FOR WHICH THE MODIFIED
  !        FUNCTION HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE
  !        DERIVATIVE, AND IF A LOWER FUNCTION VALUE HAS BEEN
  !        OBTAINED BUT THE DECREASE IS NOT SUFFICIENT.
  ! --------------------------------------------------------
  IF (STAGE1 .AND. F <= FX .AND. F > FTEST1) THEN
     ! --------------------------------------------------------
     !           DEFINE THE MODIFIED FUNCTION AND DERIVATIVE VALUES.
     ! --------------------------------------------------------
     FM = F - STP*DGTEST
     FXM = FX - STX*DGTEST
     FYM = FY - STY*DGTEST
     DGM = DG - DGTEST
     DGXM = DGX - DGTEST
     DGYM = DGY - DGTEST
     ! --------------------------------------------------------- 
     !           CALL NEWSTP TO UPDATE THE INTERVAL OF UNCERTAINTY
     !           AND TO COMPUTE THE NEW STEP.
     ! --------------------------------------------------------
     CALL NEWSTP(STX,FXM,DGXM,STY,FYM,DGYM,STP,FM,DGM, &
          BRACKT,STMIN,STMAX,INFOC)
     ! --------------------------------------------------------
     !           RESET THE FUNCTION AND GRADIENT VALUES FOR F.
     ! --------------------------------------------------------
     FX = FXM + STX*DGTEST
     FY = FYM + STY*DGTEST
     DGX = DGXM + DGTEST
     DGY = DGYM + DGTEST
  ELSE
     ! --------------------------------------------------------
     !           CALL NEWSTP TO UPDATE THE INTERVAL OF UNCERTAINTY
     !           AND TO COMPUTE THE NEW STEP.
     ! --------------------------------------------------------
     CALL NEWSTP(STX,FX,DGX,STY,FY,DGY,STP,F,DG, &
          BRACKT,STMIN,STMAX,INFOC)
  END IF
  ! --------------------------------------------------------
  !        FORCE A SUFFICIENT DECREASE IN THE SIZE OF THE
  !        INTERVAL OF UNCERTAINTY.
  ! --------------------------------------------------------
  IF (BRACKT) THEN
     IF (ABS(STY-STX)  >=  P66*WIDTH1) &
          STP = STX + P5*(STY - STX)
     WIDTH1 = WIDTH
     WIDTH = ABS(STY-STX)
  END IF
  ! --------------------------------------------------------
  !        END OF ITERATION.
  ! --------------------------------------------------------
  GO TO 30
END SUBROUTINE MLINES
!
!***********************************************************************
!***********************************************************************
SUBROUTINE NEWSTP(STX,FX,DX,STY,FY,DY,STP,FP,DP,BRACKT, &
     STPMIN,STPMAX,INFO)
  !
  ! --------------------------------------------------------
  ! NEWSTP:  Compute safeguarded step for linesearch; update
  !          interval of uncertainty
  ! --------------------------------------------------------
  implicit none
  !
  ! argument-list variables:
  INTEGER INFO
  real(chm_real) DP,DX,DY,FP,FX,FY,STP,STPMIN,STPMAX, &
       STX,STY
  LOGICAL BRACKT
  !
  ! other variables:
  real(chm_real) GAMMA,P,Q,R,S,SGND,STPC,STPF,STPQ,THETA
  LOGICAL BOUND
  !
  INFO = 0
  ! --------------------------------------------------------
  !     CHECK THE INPUT PARAMETERS FOR ERRORS.
  ! --------------------------------------------------------
  IF ((BRACKT .AND. (STP <= MIN(STX,STY) .OR. &
       STP >= MAX(STX,STY))) .OR. &
       DX*(STP-STX) >= 0.0 .OR. STPMAX < STPMIN) RETURN
  ! --------------------------------------------------------
  !     DETERMINE IF THE DERIVATIVES HAVE OPPOSITE SIGN.
  ! --------------------------------------------------------
  SGND = DP*(DX/ABS(DX))
  ! --------------------------------------------------------
  !     FIRST CASE. A HIGHER FUNCTION VALUE.
  !     THE MINIMUM IS BRACKETED. IF THE CUBIC STEP IS CLOSER
  !     TO STX THAN THE QUADRATIC STEP, THE CUBIC STEP IS TAKEN,
  !     ELSE THE AVERAGE OF THE CUBIC AND QUADRATIC STEPS IS TAKEN.
  ! --------------------------------------------------------
  IF (FP  >  FX) THEN
     INFO = 1
     BOUND = .TRUE.
     THETA = 3*(FX - FP)/(STP - STX) + DX + DP
     S = MAX(ABS(THETA),ABS(DX),ABS(DP))
     GAMMA = S*SQRT((THETA/S)**2 - (DX/S)*(DP/S))
     IF (STP  <  STX) GAMMA = -GAMMA
     P = (GAMMA - DX) + THETA
     Q = ((GAMMA - DX) + GAMMA) + DP
     R = P/Q
     STPC = STX + R*(STP - STX)
     STPQ = STX + ((DX/((FX-FP)/(STP-STX)+DX))/2)*(STP - STX)
     IF (ABS(STPC-STX)  <  ABS(STPQ-STX)) THEN
        STPF = STPC
     ELSE
        STPF = STPC + (STPQ - STPC)/2
     END IF
     BRACKT = .TRUE.
     ! --------------------------------------------------------
     !     SECOND CASE. A LOWER FUNCTION VALUE AND DERIVATIVES OF
     !     OPPOSITE SIGN. THE MINIMUM IS BRACKETED. IF THE CUBIC
     !     STEP IS CLOSER TO STX THAN THE QUADRATIC (SECANT) STEP,
     !     THE CUBIC STEP IS TAKEN, ELSE THE QUADRATIC STEP IS TAKEN.
     ! --------------------------------------------------------
  ELSE IF (SGND  <  0.0) THEN
     INFO = 2
     BOUND = .FALSE.
     THETA = 3*(FX - FP)/(STP - STX) + DX + DP
     S = MAX(ABS(THETA),ABS(DX),ABS(DP))
     GAMMA = S*SQRT((THETA/S)**2 - (DX/S)*(DP/S))
     IF (STP  >  STX) GAMMA = -GAMMA
     P = (GAMMA - DP) + THETA
     Q = ((GAMMA - DP) + GAMMA) + DX
     R = P/Q
     STPC = STP + R*(STX - STP)
     STPQ = STP + (DP/(DP-DX))*(STX - STP)
     IF (ABS(STPC-STP)  >  ABS(STPQ-STP)) THEN
        STPF = STPC
     ELSE
        STPF = STPQ
     END IF
     BRACKT = .TRUE.
     ! --------------------------------------------------------
     !     THIRD CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
     !     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DECREASES.
     !     THE CUBIC STEP IS ONLY USED IF THE CUBIC TENDS TO INFINITY
     !     IN THE DIRECTION OF THE STEP OR IF THE MINIMUM OF THE CUBIC
     !     IS BEYOND STP. OTHERWISE THE CUBIC STEP IS DEFINED TO BE
     !     EITHER STPMIN OR STPMAX. THE QUADRATIC (SECANT) STEP IS ALSO
     !     COMPUTED AND IF THE MINIMUM IS BRACKETED THEN THE THE STEP
     !     CLOSEST TO STX IS TAKEN, ELSE THE STEP FARTHEST AWAY IS TAKEN.
     ! --------------------------------------------------------
  ELSE IF (ABS(DP)  <  ABS(DX)) THEN
     INFO = 3
     BOUND = .TRUE.
     THETA = 3*(FX - FP)/(STP - STX) + DX + DP
     S = MAX(ABS(THETA),ABS(DX),ABS(DP))
     ! --------------------------------------------------------
     !        THE CASE GAMMA = 0 ONLY ARISES IF THE CUBIC DOES NOT TEND
     !        TO INFINITY IN THE DIRECTION OF THE STEP.
     ! --------------------------------------------------------
     GAMMA = S*SQRT(MAX(0.0D0,(THETA/S)**2 - (DX/S)*(DP/S)))
     IF (STP  >  STX) GAMMA = -GAMMA
     P = (GAMMA - DP) + THETA
     Q = (GAMMA + (DX - DP)) + GAMMA
     R = P/Q
     IF (R < 0.0 .AND. GAMMA /= 0.0) THEN
        STPC = STP + R*(STX - STP)
     ELSE IF (STP  >  STX) THEN
        STPC = STPMAX
     ELSE
        STPC = STPMIN
     END IF
     STPQ = STP + (DP/(DP-DX))*(STX - STP)
     IF (BRACKT) THEN
        IF (ABS(STP-STPC)  <  ABS(STP-STPQ)) THEN
           STPF = STPC
        ELSE
           STPF = STPQ
        END IF
     ELSE
        IF (ABS(STP-STPC)  >  ABS(STP-STPQ)) THEN
           STPF = STPC
        ELSE
           STPF = STPQ
        END IF
     END IF
     ! --------------------------------------------------------
     !     FOURTH CASE. A LOWER FUNCTION VALUE, DERIVATIVES OF THE
     !     SAME SIGN, AND THE MAGNITUDE OF THE DERIVATIVE DOES
     !     NOT DECREASE. IF THE MINIMUM IS NOT BRACKETED, THE STEP
     !     IS EITHER STPMIN OR STPMAX, ELSE THE CUBIC STEP IS TAKEN.
     ! --------------------------------------------------------
     !
  ELSE
     INFO = 4
     BOUND = .FALSE.
     IF (BRACKT) THEN
        THETA = 3*(FP - FY)/(STY - STP) + DY + DP
        S = MAX(ABS(THETA),ABS(DY),ABS(DP))
        GAMMA = S*SQRT((THETA/S)**2 - (DY/S)*(DP/S))
        IF (STP  >  STY) GAMMA = -GAMMA
        P = (GAMMA - DP) + THETA
        Q = ((GAMMA - DP) + GAMMA) + DY
        R = P/Q
        STPC = STP + R*(STY - STP)
        STPF = STPC
     ELSE IF (STP  >  STX) THEN
        STPF = STPMAX
     ELSE
        STPF = STPMIN
     END IF
  END IF
  ! --------------------------------------------------------
  !     UPDATE THE INTERVAL OF UNCERTAINTY. THIS UPDATE DOES NOT
  !     DEPEND ON THE NEW STEP OR THE CASE ANALYSIS ABOVE.
  ! --------------------------------------------------------
  IF (FP  >  FX) THEN
     STY = STP
     FY = FP
     DY = DP
  ELSE
     IF (SGND  <  0.0) THEN
        STY = STX
        FY = FX
        DY = DX
     END IF
     STX = STP
     FX = FP
     DX = DP
  END IF
  ! --------------------------------------------------------
  !     COMPUTE THE NEW STEP AND SAFEGUARD IT.
  ! --------------------------------------------------------
  STPF = MIN(STPMAX,STPF)
  STPF = MAX(STPMIN,STPF)
  STP = STPF
  IF (BRACKT .AND. BOUND) THEN
     IF (STY  >  STX) THEN
        STP = MIN(STX+0.66*(STY-STX),STP)
     ELSE
        STP = MAX(STX+0.66*(STY-STX),STP)
     END IF
  END IF
  RETURN
END SUBROUTINE NEWSTP
!***********************************************************************
!                                                                1/15/81
!***********************************************************************
!  ODRV -- DRIVER FOR SPARSE MATRIX REORDERING ROUTINES
!***********************************************************************
SUBROUTINE ODRV &
     (N, IA,JA,A, P,IP, NSP,ISP, PATH, FLAG)
  !
  !  DESCRIPTION
  !
  !    ODRV FINDS A MINIMUM DEGREE ORDERING OF THE ROWS AND COLUMNS OF A
  !    SYMMETRIC MATRIX M STORED IN (IA,JA,A) FORMAT (SEE BELOW).  FOR THE
  !    REORDERED MATRIX, THE WORK AND STORAGE REQUIRED TO PERFORM GAUSSIAN
  !    ELIMINATION IS (USUALLY) SIGNIFICANTLY LESS.
  !
  !    IF ONLY THE NONZERO ENTRIES IN THE UPPER TRIANGLE OF M ARE BEING
  !    STORED, THEN ODRV SYMMETRICALLY REORDERS (IA,JA,A), (OPTIONALLY)
  !    WITH THE DIAGONAL ENTRIES PLACED FIRST IN EACH ROW.  THIS IS TO
  !    ENSURE THAT IF M(I,J) WILL BE IN THE UPPER TRIANGLE OF M WITH
  !    RESPECT TO THE NEW ORDERING, THEN M(I,J) IS STORED IN ROW I (AND
  !    THUS M(J,I) IS NOT STORED);  WHEREAS IF M(I,J) WILL BE IN THE
  !    STRICT LOWER TRIANGLE OF M, THEN M(J,I) IS STORED IN ROW J (AND
  !    THUS M(I,J) IS NOT STORED).
  !
  !
  !  STORAGE OF SPARSE MATRICES
  !
  !    THE NONZERO ENTRIES OF THE MATRIX M ARE STORED ROW-BY-ROW IN THE
  !    ARRAY A.  TO IDENTIFY THE INDIVIDUAL NONZERO ENTRIES IN EACH ROW,
  !    WE NEED TO KNOW IN WHICH COLUMN EACH ENTRY LIES.  THESE COLUMN
  !    INDICES ARE STORED IN THE ARRAY JA;  I.E., IF  A(K) = M(I,J),  THEN
  !    JA(K) = J.  TO IDENTIFY THE INDIVIDUAL ROWS, WE NEED TO KNOW WHERE
  !    EACH ROW STARTS.  THESE ROW POINTERS ARE STORED IN THE ARRAY IA;
  !    I.E., IF M(I,J) IS THE FIRST NONZERO ENTRY (STORED) IN THE I-TH ROW
  !    AND  A(K) = M(I,J),  THEN  IA(I) = K.  MOREOVER, IA(N+1) POINTS TO
  !    THE FIRST LOCATION FOLLOWING THE LAST ELEMENT IN THE LAST ROW.
  !    THUS, THE NUMBER OF ENTRIES IN THE I-TH ROW IS  IA(I+1) - IA(I),
  !    THE NONZERO ENTRIES IN THE I-TH ROW ARE STORED CONSECUTIVELY IN
  !
  !            A(IA(I)),  A(IA(I)+1),  ..., A(IA(I+1)-1),
  !
  !    AND THE CORRESPONDING COLUMN INDICES ARE STORED CONSECUTIVELY IN
  !
  !            JA(IA(I)), JA(IA(I)+1), ..., JA(IA(I+1)-1).
  !
  !    SINCE THE COEFFICIENT MATRIX IS SYMMETRIC, ONLY THE NONZERO ENTRIES
  !    IN THE UPPER TRIANGLE NEED BE STORED.  FOR EXAMPLE, THE MATRIX
  !
  !             ( 1  0  2  3  0 )
  !             ( 0  4  0  0  0 )
  !         M = ( 2  0  5  6  0 )
  !             ( 3  0  6  7  8 )
  !             ( 0  0  0  8  9 )
  !
  !    COULD BE STORED AS
  !
  !            \ 1  2  3  4  5  6  7  8  9 10 11 12 13
  !         ---+--------------------------------------
  !         IA \ 1  4  5  8 12 14
  !         JA \ 1  3  4  2  1  3  4  1  3  4  5  4  5
  !          A \ 1  2  3  4  2  5  6  3  6  7  8  8  9
  !
  !    OR (SYMMETRICALLY) AS
  !
  !            \ 1  2  3  4  5  6  7  8  9
  !         ---+--------------------------
  !         IA \ 1  4  5  7  9 10
  !         JA \ 1  3  4  2  3  4  4  5  5
  !          A \ 1  2  3  4  5  6  7  8  9          .
  !
  !
  !  PARAMETERS
  !
  !    N    - ORDER OF THE MATRIX
  !
  !    IA   - INTEGER ONE-DIMENSIONAL ARRAY CONTAINING POINTERS TO DELIMIT
  !           ROWS IN JA AND A;  DIMENSION = N+1
  !
  !    JA   - INTEGER ONE-DIMENSIONAL ARRAY CONTAINING THE COLUMN INDICES
  !           CORRESPONDING TO THE ELEMENTS OF A;  DIMENSION = NUMBER OF
  !           NONZERO ENTRIES IN (THE UPPER TRIANGLE OF) M
  !
  !    A    - REAL ONE-DIMENSIONAL ARRAY CONTAINING THE NONZERO ENTRIES IN
  !           (THE UPPER TRIANGLE OF) M, STORED BY ROWS;  DIMENSION =
  !           NUMBER OF NONZERO ENTRIES IN (THE UPPER TRIANGLE OF) M
  !
  !    P    - INTEGER ONE-DIMENSIONAL ARRAY USED TO RETURN THE PERMUTATION
  !           OF THE ROWS AND COLUMNS OF M CORRESPONDING TO THE MINIMUM
  !           DEGREE ORDERING;  DIMENSION = N
  !
  !    IP   - INTEGER ONE-DIMENSIONAL ARRAY USED TO RETURN THE INVERSE OF
  !           THE PERMUTATION RETURNED IN P;  DIMENSION = N
  !
  !    NSP  - DECLARED DIMENSION OF THE ONE-DIMENSIONAL ARRAY ISP;  NSP
  !           MUST BE AT LEAST  3N+4K,  WHERE K IS THE NUMBER OF NONZEROES
  !           IN THE STRICT UPPER TRIANGLE OF M
  !
  !    ISP  - INTEGER ONE-DIMENSIONAL ARRAY USED FOR WORKING STORAGE;
  !           DIMENSION = NSP
  !
  !    PATH - INTEGER PATH SPECIFICATION;  VALUES AND THEIR MEANINGS ARE -
  !             1  FIND MINIMUM DEGREE ORDERING ONLY
  !             2  FIND MINIMUM DEGREE ORDERING AND REORDER SYMMETRICALLY
  !                  STORED MATRIX (USED WHEN ONLY THE NONZERO ENTRIES IN
  !                  THE UPPER TRIANGLE OF M ARE BEING STORED)
  !             3  REORDER SYMMETRICALLY STORED MATRIX AS SPECIFIED BY
  !                  INPUT PERMUTATION (USED WHEN AN ORDERING HAS ALREADY
  !                  BEEN DETERMINED AND ONLY THE NONZERO ENTRIES IN THE
  !                  UPPER TRIANGLE OF M ARE BEING STORED)
  !             4  SAME AS 2 BUT PUT DIAGONAL ENTRIES AT START OF EACH ROW
  !             5  SAME AS 3 BUT PUT DIAGONAL ENTRIES AT START OF EACH ROW
  !
  !    FLAG - INTEGER ERROR FLAG;  VALUES AND THEIR MEANINGS ARE -
  !               0    NO ERRORS DETECTED
  !              9N+K  INSUFFICIENT STORAGE IN MD
  !             10N+1  INSUFFICIENT STORAGE IN ODRV
  !             11N+1  ILLEGAL PATH SPECIFICATION
  !
  !
  !  CONVERSION FROM REAL TO real(chm_real)
  !
  !    CHANGE THE REAL DECLARATIONS IN ODRV AND SRO TO real(chm_real)
  !    DECLARATIONS.
  !
  !-----------------------------------------------------------------------
  implicit none
  !
  INTEGER  IA(N+1), JA(1),  P(1), IP(1),  ISP(1),  PATH,  FLAG, &
       V, L, HEAD,  TMP, Q
  !....   REAL  A(1)
  real(chm_real)  A(1)
  LOGICAL  DFLAG
  INTEGER NEXT,N,MAX,NSP
  !
  !----INITIALIZE ERROR FLAG AND VALIDATE PATH SPECIFICATION
  FLAG = 0
  IF (PATH < 1 .OR. 5.LT.PATH)  GO TO 111
  !
  !----ALLOCATE STORAGE AND FIND MINIMUM DEGREE ORDERING
  IF ((PATH-1) * (PATH-2) * (PATH-4)  /=  0)  GO TO 1
  MAX = (NSP-N)/2
  V    = 1
  L    = V     +  MAX
  HEAD = L     +  MAX
  NEXT = HEAD  +  N
  IF (MAX < N)  GO TO 110
  !
  CALL  MD &
       (N, IA,JA, MAX,ISP(V),ISP(L), ISP(HEAD),P,IP, ISP(V), FLAG)
  IF (FLAG /= 0)  GO TO 100
  !
  !----ALLOCATE STORAGE AND SYMMETRICALLY REORDER MATRIX
1 IF ((PATH-2) * (PATH-3) * (PATH-4) * (PATH-5)  /=  0)  GO TO 2
  TMP = (NSP+1) -      N
  Q   = TMP     - (IA(N+1)-1)
  IF (Q < 1)  GO TO 110
  !
  DFLAG = PATH == 4 .OR. PATH.EQ.5
  CALL SRO &
       (N,  IP,  IA, JA, A,  ISP(TMP),  ISP(Q),  DFLAG)
  !
2 RETURN
  !
  ! ** ERROR -- ERROR DETECTED IN MD
100 RETURN
  ! ** ERROR -- INSUFFICIENT STORAGE
110 FLAG = 10*N + 1
  RETURN
  ! ** ERROR -- ILLEGAL PATH SPECIFIED
111 FLAG = 11*N + 1
  RETURN
END SUBROUTINE ODRV
!***********************************************************************
!  SSF --  SYMBOLIC UT-D-U FACTORIZATION OF SPARSE SYMMETRIC MATRIX
!***********************************************************************
SUBROUTINE  SSF &
     (N,NZ,P,IP, IA,JA,A, IJU,JU,IU,JUMAX, Q, MARK, JL, FLAG)
  !
  !  ADDITIONAL PARAMETERS
  !
  !    Q     - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !    MARK  - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !    JL    - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !
  !  DEFINITIONS OF INTERNAL PARAMETERS (DURING K-TH STAGE OF ELIMINATION)
  !
  !    Q CONTAINS AN ORDERED LINKED LIST REPRESENTATION OF THE NONZERO
  !      STRUCTURE OF THE K-TH ROW OF U --
  !        Q(K) IS THE FIRST COLUMN WITH A NONZERO ENTRY
  !        Q(I) IS THE NEXT COLUMN WITH A NONZERO ENTRY AFTER COLUMN I
  !      IN EITHER CASE, Q(I) = N+1 INDICATES THE END OF THE LIST
  !
  !    JL CONTAINS LISTS OF ROWS TO BE MERGED INTO UNELIMINATED ROWS --
  !        I GE K => JL(I) IS THE FIRST ROW TO BE MERGED INTO ROW I
  !        I LT K => JL(I) IS THE ROW FOLLOWING ROW I IN SOME LIST OF ROWS
  !      IN EITHER CASE, JL(I) = 0 INDICATES THE END OF A LIST
  !
  !    MARK(I) IS THE LAST ROW STORED IN JU FOR WHICH U(MARK(I),I) NE 0
  !
  !    JUMIN AND JUPTR ARE THE INDICES IN JU OF THE FIRST AND LAST
  !      ELEMENTS IN THE LAST ROW SAVED IN JU
  !
  !    LUK IS THE NUMBER OF NONZERO ENTRIES IN THE K-TH ROW
  !
  !-----------------------------------------------------------------------
  implicit none
  !
  INTEGER JUMIN,JUMAX,JUPTR,JMIN,JMAX,LMAX,I,J,K,M,N,LUI,LUK,NZ
  INTEGER  P(N), IP(N),IA(N+1),JA(N+NZ), IJU(*), JU(*), IU(*), &
       Q(N),  MARK(N),  JL(N),  FLAG,  TAG, VJ, QM
  real(chm_real) A(N+NZ)
  LOGICAL  CLIQUE
  !
  !----INITIALIZATION
  JUMIN = 1
  JUPTR = 0
  IU(1) = 1
  DO K=1,N
     MARK(K) = 0
     JL(K) = 0
  ENDDO
  !
  !----FOR EACH ROW K
  DO K=1,N
     LUK = 0
     Q(K) = N+1
     !
     TAG = MARK(K)
     CLIQUE = .FALSE.
     IF (JL(K) /= 0)  CLIQUE = JL(JL(K)) == 0
     !
     !------INITIALIZE NONZERO STRUCTURE OF K-TH ROW TO ROW P(K) OF M
     JMIN = IA(P(K))
     JMAX = IA(P(K)+1) - 1
     IF (JMIN > JMAX)  GO TO 4
     DO J=JMIN,JMAX
        VJ = IP(JA(J))
        IF (VJ <= K)  GO TO 3
        !
        QM = K
2       M = QM
        QM = Q(M)
        IF (QM < VJ)  GO TO 2
        IF (QM == VJ)  GO TO 102
        LUK = LUK+1
        Q(M) = VJ
        Q(VJ) = QM
        IF (MARK(VJ) /= TAG)  CLIQUE = .FALSE.
        !
3       CONTINUE
     ENDDO
     !
     !------IF EXACTLY ONE ROW IS TO BE MERGED INTO THE K-TH ROW AND THERE IS
     !------A NONZERO ENTRY IN EVERY COLUMN IN THAT ROW IN WHICH THERE IS A
     !------NONZERO ENTRY IN ROW P(K) OF M, THEN DO NOT COMPUTE FILL-IN, JUST
     !------USE THE COLUMN INDICES FOR THE ROW WHICH WAS TO HAVE BEEN MERGED
4    IF (.NOT.CLIQUE)  GO TO 5
     IJU(K) = IJU(JL(K)) + 1
     LUK = IU(JL(K)+1) - (IU(JL(K))+1)
     GO TO 17
     !
     !------MODIFY NONZERO STRUCTURE OF K-TH ROW BY COMPUTING FILL-IN
     !------FOR EACH ROW I TO BE MERGED IN
5    LMAX = 0
     IJU(K) = JUPTR
     !
     I = K
6    I = JL(I)
     IF (I == 0)  GO TO 10
     !
     !--------MERGE ROW I INTO K-TH ROW
     LUI = IU(I+1) - (IU(I)+1)
     JMIN = IJU(I) +  1
     JMAX = IJU(I) + LUI
     QM = K
     !
     DO J=JMIN,JMAX
        VJ = JU(J)
7       M = QM
        QM = Q(M)
        IF (QM < VJ)  GO TO 7
        IF (QM == VJ)  GO TO 8
        LUK = LUK+1
        Q(M) = VJ
        Q(VJ) = QM
        QM = VJ
8       CONTINUE
     ENDDO
     !
     !--------REMEMBER LENGTH AND POSITION IN JU OF LONGEST ROW MERGED
     IF (LUI <= LMAX)  GO TO 9
     LMAX = LUI
     IJU(K) = JMIN
     !
9    GO TO 6
     !
     !------IF THE K-TH ROW IS THE SAME LENGTH AS THE LONGEST ROW MERGED,
     !------THEN USE THE COLUMN INDICES FOR THAT ROW
10   IF (LUK == LMAX)  GO TO 17
     !
     !------IF THE TAIL OF THE LAST ROW SAVED IN JU IS THE SAME AS THE HEAD
     !------OF THE K-TH ROW, THEN OVERLAP THE TWO SETS OF COLUMN INDICES --
     !--------SEARCH LAST ROW SAVED FOR FIRST NONZERO ENTRY IN K-TH ROW ...
     I = Q(K)
     IF (JUMIN > JUPTR)  GO TO 12
     DO JMIN=JUMIN,JUPTR
        IF (JU(JMIN)-I.eq.0) goto 13
        IF (JU(JMIN)-I.gt.0) goto 12
     ENDDO
12   GO TO 15
     !
     !--------... AND THEN TEST WHETHER TAIL MATCHES HEAD OF K-TH ROW
13   IJU(K) = JMIN
     DO J=JMIN,JUPTR
        IF (JU(J) /= I)  GO TO 15
        I = Q(I)
        IF (I > N)  GO TO 17
     ENDDO
     JUPTR = JMIN - 1
     !
     !------SAVE NONZERO STRUCTURE OF K-TH ROW IN JU
15   I = K
     JUMIN = JUPTR +  1
     JUPTR = JUPTR + LUK
     IF (JUPTR > JUMAX)  GO TO 106
     DO J=JUMIN,JUPTR
        I = Q(I)
        JU(J) = I
        MARK(I) = K
     ENDDO
     IJU(K) = JUMIN
     !
     !------ADD K TO ROW LIST FOR FIRST NONZERO ELEMENT IN K-TH ROW
17   IF (LUK <= 1)  GO TO 18
     I = JU(IJU(K))
     JL(K) = JL(I)
     JL(I) = K
     !
18   IU(K+1) = IU(K) + LUK
  ENDDO
  !
  FLAG = 0
  RETURN
  !
  ! ** ERROR -- DUPLICATE ENTRY IN A
102 FLAG = 2*N + P(K)
  RETURN
  ! ** ERROR -- INSUFFICIENT STORAGE FOR JU
106 FLAG = 6*N + K
  RETURN
END SUBROUTINE SSF
!
!***********************************************************************
!  SNS -- SOLUTION OF SPARSE SYMMETRIC POSITIVE DEFINITE SYSTEM OF
!         LINEAR EQUATIONS  MX = B  GIVEN UT-D-U FACTORIZATION OF M
!***********************************************************************
SUBROUTINE  SNS &
     (N, P, D, IJU,JU,IU,U, Z, B, TMP)
  !
  implicit none
  !
  INTEGER P(*), IJU(*), JU(*), IU(*)
  real(chm_real) D(*), U(*), Z(*), B(*), TMP(*), TMPK, SUM
  INTEGER MU,JMIN,JMAX,I,J,K,N

  !
  !  ADDITIONAL PARAMETERS
  !
  !    TMP   - REAL ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !  --------------------------------------------------------------------
  !
  !----SET TMP TO PERMUTED B
  DO K=1,N
     TMP(K) = B(P(K))
  ENDDO
  !
  !----SOLVE  UT D Y = B  BY FORWARD SUBSTITUTION
  DO K=1,N
     TMPK = TMP(K)
     JMIN = IU(K)
     JMAX = IU(K+1) - 1
     IF (JMIN > JMAX)  GO TO 3
     MU = IJU(K) - JMIN
     DO J=JMIN,JMAX
        TMP(JU(MU+J)) = TMP(JU(MU+J)) + U(J) * TMPK
     ENDDO
3    TMP(K) = TMPK * D(K)
  ENDDO
  !
  !----SOLVE  U X = Y  BY BACK SUBSTITUTION
  K = N
  DO I=1,N
     SUM = TMP(K)
     JMIN = IU(K)
     JMAX = IU(K+1) - 1
     IF (JMIN > JMAX)  GO TO 5
     MU = IJU(K) - JMIN
     DO J=JMIN,JMAX
        SUM = SUM + U(J) * TMP(JU(MU+J))
     ENDDO
5    TMP(K) = SUM
     Z(P(K)) = SUM
     K = K-1
  ENDDO
  !
  RETURN
END SUBROUTINE SNS
!***********************************************************************
!***********************************************************************
!***********************************************************************
!  SRO -- SYMMETRIC REORDERING OF SPARSE SYMMETRIC MATRIX
!***********************************************************************
SUBROUTINE  SRO &
     (N, IP, IA,JA,A, Q, R, DFLAG)
  !
  !  DESCRIPTION
  !
  !    THE NONZERO ENTRIES OF THE MATRIX M ARE ASSUMED TO BE STORED
  !    SYMMETRICALLY IN (IA,JA,A) FORMAT (I.E., NOT BOTH M(I,J) AND M(J,I)
  !    ARE STORED IF I NE J).
  !
  !    SRO DOES NOT REARRANGE THE ORDER OF THE ROWS, BUT DOES MOVE
  !    NONZEROES FROM ONE ROW TO ANOTHER TO ENSURE THAT IF M(I,J) WILL BE
  !    IN THE UPPER TRIANGLE OF M WITH RESPECT TO THE NEW ORDERING, THEN
  !    M(I,J) IS STORED IN ROW I (AND THUS M(J,I) IS NOT STORED);  WHEREAS
  !    IF M(I,J) WILL BE IN THE STRICT LOWER TRIANGLE OF M, THEN M(J,I) IS
  !    STORED IN ROW J (AND THUS M(I,J) IS NOT STORED).
  !
  !
  !  ADDITIONAL PARAMETERS
  !
  !    Q     - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !    R     - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = NUMBER OF
  !            NONZERO ENTRIES IN THE UPPER TRIANGLE OF M
  !
  !    DFLAG - LOGICAL VARIABLE;  IF DFLAG = .TRUE., THEN STORE NONZERO
  !            DIAGONAL ELEMENTS AT THE BEGINNING OF THE ROW
  !
  !-----------------------------------------------------------------------
  implicit none
  !
  INTEGER  IP(1),  IA(N+1), JA(1),  Q(1), R(1)
  !       REAL  A(1),  AK
  real(chm_real)  A(1),  AK
  INTEGER JMIN,JMAX,I,J,K,N,JDUMMY,JAK,ILAST
  LOGICAL  DFLAG
  !
  !
  !--PHASE 1 -- FIND ROW IN WHICH TO STORE EACH NONZERO
  !----INITIALIZE COUNT OF NONZEROES TO BE STORED IN EACH ROW
  DO I=1,N
     Q(I) = 0
  ENDDO
  !
  !----FOR EACH NONZERO ELEMENT A(J)
  DO I=1,N
     JMIN = IA(I)
     JMAX = IA(I+1) - 1
     IF (JMIN > JMAX)  GO TO 3
     DO J=JMIN,JMAX
        !
        !--------FIND ROW (=R(J)) AND COLUMN (=JA(J)) IN WHICH TO STORE A(J) ...
        K = JA(J)
        IF (IP(K) < IP(I))  JA(J) = I
        IF (IP(K) >= IP(I))  K = I
        R(J) = K
        !
        !--------... AND INCREMENT COUNT OF NONZEROES (=Q(R(J)) IN THAT ROW
        Q(K) = Q(K) + 1
     ENDDO
3    CONTINUE
  ENDDO
  !
  !
  !--PHASE 2 -- FIND NEW IA AND PERMUTATION TO APPLY TO (JA,A)
  !----DETERMINE POINTERS TO DELIMIT ROWS IN PERMUTED (JA,A)
  DO I=1,N
     IA(I+1) = IA(I) + Q(I)
     Q(I) = IA(I+1)
  ENDDO
  !
  !----DETERMINE WHERE EACH (JA(J),A(J)) IS STORED IN PERMUTED (JA,A)
  !----FOR EACH NONZERO ELEMENT (IN REVERSE ORDER)
  ILAST = 0
  JMIN = IA(1)
  JMAX = IA(N+1) - 1
  J = JMAX
  DO JDUMMY=JMIN,JMAX
     I = R(J)
     IF (.NOT.DFLAG .OR. JA(J) /= I .OR. I == ILAST)  GO TO 5
     !
     !------IF DFLAG, THEN PUT DIAGONAL NONZERO AT BEGINNING OF ROW
     R(J) = IA(I)
     ILAST = I
     GO TO 6
     !
     !------PUT (OFF-DIAGONAL) NONZERO IN LAST UNUSED LOCATION IN ROW
5    Q(I) = Q(I) - 1
     R(J) = Q(I)
     !
6    J = J-1
  ENDDO
  !
  !
  !--PHASE 3 -- PERMUTE (JA,A) TO UPPER TRIANGULAR FORM (WRT NEW ORDERING)
  DO J=JMIN,JMAX
7    IF (R(J) == J)  GO TO 8
     K = R(J)
     R(J) = R(K)
     R(K) = K
     JAK = JA(K)
     JA(K) = JA(J)
     JA(J) = JAK
     AK = A(K)
     A(K) = A(J)
     A(J) = AK
     GO TO 7
8    CONTINUE
  ENDDO
  !
  RETURN
END SUBROUTINE SRO
!
!***********************************************************************
!***********************************************************************
!  MD -- MINIMUM DEGREE ALGORITHM (BASED ON ELEMENT MODEL)
!***********************************************************************
SUBROUTINE  MD &
     (N, IA,JA, MAX, V,L, HEAD,LAST,NEXT, MARK, FLAG)
  !
  !  DESCRIPTION
  !
  !    MD FINDS A MINIMUM DEGREE ORDERING OF THE ROWS AND COLUMNS OF A
  !    SYMMETRIC MATRIX M STORED IN (IA,JA,A) FORMAT.
  !
  !
  !  ADDITIONAL PARAMETERS
  !
  !    MAX  - DECLARED DIMENSION OF THE ONE-DIMENSIONAL ARRAYS V AND L;
  !           MAX MUST BE AT LEAST  N+2K,  WHERE K IS THE NUMBER OF
  !           NONZEROES IN THE STRICT UPPER TRIANGLE OF M
  !
  !    V    - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = MAX
  !
  !    L    - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = MAX
  !
  !    HEAD - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !    LAST - INTEGER ONE-DIMENSIONAL ARRAY USED TO RETURN THE PERMUTATION
  !           OF THE ROWS AND COLUMNS OF M CORRESPONDING TO THE MINIMUM
  !           DEGREE ORDERING;  DIMENSION = N
  !
  !    NEXT - INTEGER ONE-DIMENSIONAL ARRAY USED TO RETURN THE INVERSE OF
  !           THE PERMUTATION RETURNED IN LAST;  DIMENSION = N
  !
  !    MARK - INTEGER ONE-DIMENSIONAL WORK ARRAY (MAY BE THE SAME AS V);
  !           DIMENSION = N
  !
  !    FLAG - INTEGER ERROR FLAG;  VALUES AND THEIR MEANINGS ARE -
  !             0      NO ERRORS DETECTED
  !             11N+1  INSUFFICIENT STORAGE IN MD
  !
  !
  !  DEFINITIONS OF INTERNAL PARAMETERS
  !
  !    ---------+---------------------------------------------------------
  !    V(S)     \ VALUE FIELD OF LIST ENTRY
  !    ---------+---------------------------------------------------------
  !    L(S)     \ LINK FIELD OF LIST ENTRY  (0 => END OF LIST)
  !    ---------+---------------------------------------------------------
  !    L(VI)    \ POINTER TO ELEMENT LIST OF UNELIMINATED VERTEX VI
  !    ---------+---------------------------------------------------------
  !    L(EJ)    \ POINTER TO BOUNDARY LIST OF ACTIVE ELEMENT EJ
  !    ---------+---------------------------------------------------------
  !    HEAD(D)  \ VJ => VJ HEAD OF D-LIST D
  !             \  0 => NO VERTEX IN D-LIST D
  !
  !
  !             \                  VI UNELIMINATED VERTEX
  !             \          VI IN EK           \       VI NOT IN EK
  !    ---------+-----------------------------+---------------------------
  !    NEXT(VI) \ UNDEFINED BUT NONNEGATIVE   \ VJ => VJ NEXT IN D-LIST
  !             \                             \  0 => VI TAIL OF D-LIST
  !    ---------+-----------------------------+---------------------------
  !    LAST(VI) \ (NOT SET UNTIL MDP)         \ -D => VI HEAD OF D-LIST D
  !             \-VK => COMPUTE DEGREE        \ VJ => VJ LAST IN D-LIST
  !             \ EJ => VI PROTOTYPE OF EJ    \  0 => VI NOT IN ANY D-LIST
  !             \  0 => DO NOT COMPUTE DEGREE \
  !    ---------+-----------------------------+---------------------------
  !    MARK(VI) \ MARK(VK)                    \ NONNEGATIVE TAG < MARK(VK)
  !
  !
  !             \                   VI ELIMINATED VERTEX
  !             \      EI ACTIVE ELEMENT      \           OTHERWISE
  !    ---------+-----------------------------+---------------------------
  !    NEXT(VI) \ -J => VI WAS J-TH VERTEX    \ -J => VI WAS J-TH VERTEX
  !             \       TO BE ELIMINATED      \       TO BE ELIMINATED
  !    ---------+-----------------------------+---------------------------
  !    LAST(VI) \  M => SIZE OF EI = M        \ UNDEFINED
  !    ---------+-----------------------------+---------------------------
  !    MARK(VI) \ -M => OVERLAP COUNT OF EI   \ UNDEFINED
  !             \       WITH EK = M           \
  !             \ OTHERWISE NONNEGATIVE TAG   \
  !             \       < MARK(VK)            \
  !
  !-----------------------------------------------------------------------
  implicit none
  !
  INTEGER  IA(N+1), JA(1),  V(1), L(1),  HEAD(1), LAST(1), NEXT(1), &
       MARK(1),  FLAG,  TAG, DMIN, VK,EK, TAIL
  EQUIVALENCE  (VK,EK)
  INTEGER K,N,MAX
  !
  !----INITIALIZATION
  TAG = 0
  CALL  MDI &
       (N, IA,JA, MAX,V,L, HEAD,LAST,NEXT, MARK,TAG, FLAG)
  IF (FLAG /= 0)  RETURN
  !
  K = 0
  DMIN = 1
  !
  !----WHILE  K < N  DO
1 IF (K >= N)  GO TO 4
  !
  !------SEARCH FOR VERTEX OF MINIMUM DEGREE
2 IF (HEAD(DMIN) > 0)  GO TO 3
  DMIN = DMIN + 1
  GO TO 2
  !
  !------REMOVE VERTEX VK OF MINIMUM DEGREE FROM DEGREE LIST
3 VK = HEAD(DMIN)
  HEAD(DMIN) = NEXT(VK)
  IF (HEAD(DMIN) > 0)  LAST(HEAD(DMIN)) = -DMIN
  !
  !------NUMBER VERTEX VK, ADJUST TAG, AND TAG VK
  K = K+1
  NEXT(VK) = -K
  LAST(EK) = DMIN - 1
  TAG = TAG + LAST(EK)
  MARK(VK) = TAG
  !
  !------FORM ELEMENT EK FROM UNELIMINATED NEIGHBORS OF VK
  CALL  MDM &
       (VK,TAIL, V,L, LAST,NEXT, MARK)
  !
  !------PURGE INACTIVE ELEMENTS AND DO MASS ELIMINATION
  CALL  MDP &
       (K,EK,TAIL, V,L, HEAD,LAST,NEXT, MARK)
  !
  !------UPDATE DEGREES OF UNELIMINATED VERTICES IN EK
  CALL  MDU &
       (EK,DMIN, V,L, HEAD,LAST,NEXT, MARK)
  !
  GO TO 1
  !
  !----GENERATE INVERSE PERMUTATION FROM PERMUTATION
4 DO K=1,N
     NEXT(K) = -NEXT(K)
     LAST(NEXT(K)) = K
  ENDDO
  !
  RETURN
END SUBROUTINE MD
!
!***********************************************************************
!  MDI -- INITIALIZATION
!***********************************************************************
SUBROUTINE  MDI &
     (N, IA,JA, MAX,V,L, HEAD,LAST,NEXT, MARK,TAG, FLAG)
  !
  implicit none
  INTEGER  IA(N+1), JA(1),  V(1), L(1),  HEAD(1), LAST(1), NEXT(1), &
       MARK(1), TAG,  FLAG,  SFS, VI,DVI, VJ
  !
  INTEGER JMIN,JMAX,J,N,MAX
  !
  !----INITIALIZE DEGREES, ELEMENT LISTS, AND DEGREE LISTS
  DO VI=1,N
     MARK(VI) = 1
     L(VI) = 0
     HEAD(VI) = 0
  ENDDO
  SFS = N+1
  !
  !----CREATE NONZERO STRUCTURE
  !----FOR EACH NONZERO ENTRY A(VI,VJ) IN STRICT UPPER TRIANGLE
  DO VI=1,N
     JMIN = IA(VI)
     JMAX = IA(VI+1) - 1
     IF (JMIN > JMAX)  GO TO 3
     DO J=JMIN,JMAX
        VJ = JA(J)
        IF (VI >= VJ)  GO TO 2
        IF (SFS >= MAX)  GO TO 101
        !
        !------ENTER VJ IN ELEMENT LIST FOR VI
        MARK(VI) = MARK(VI) + 1
        V(SFS) = VJ
        L(SFS) = L(VI)
        L(VI) = SFS
        SFS = SFS+1
        !
        !------ENTER VI IN ELEMENT LIST FOR VJ
        MARK(VJ) = MARK(VJ) + 1
        V(SFS) = VI
        L(SFS) = L(VJ)
        L(VJ) = SFS
        SFS = SFS+1
2       CONTINUE
     ENDDO
3    CONTINUE
  ENDDO
  !
  !----CREATE DEGREE LISTS AND INITIALIZE MARK VECTOR
  DO VI=1,N
     DVI = MARK(VI)
     NEXT(VI) = HEAD(DVI)
     HEAD(DVI) = VI
     LAST(VI) = -DVI
     IF (NEXT(VI) > 0)  LAST(NEXT(VI)) = VI
     MARK(VI) = TAG
  ENDDO
  !
  RETURN
  !
  ! ** ERROR -- INSUFFICIENT STORAGE
101 FLAG = 9*N + VI
  RETURN
END SUBROUTINE MDI
!
!***********************************************************************
!  MDM -- FORM ELEMENT FROM UNELIMINATED NEIGHBORS OF VK
!***********************************************************************
SUBROUTINE  MDM &
     (VK,TAIL, V,L, LAST,NEXT, MARK)
  !
  implicit none
  !
  INTEGER  VK, TAIL,  V(1), L(1),   LAST(1), NEXT(1),   MARK(1), &
       TAG, S,LS,VS,ES, B,LB,VB, BLP,BLPMAX
  EQUIVALENCE  (VS, ES)
  !
  !----INITIALIZE TAG AND LIST OF UNELIMINATED NEIGHBORS
  TAG = MARK(VK)
  TAIL = VK
  !
  !----FOR EACH VERTEX/ELEMENT VS/ES IN ELEMENT LIST OF VK
  LS = L(VK)
1 S = LS
  IF (S == 0)  GO TO 5
  LS = L(S)
  VS = V(S)
  IF (NEXT(VS) < 0)  GO TO 2
  !
  !------IF VS IS UNELIMINATED VERTEX, THEN TAG AND APPEND TO LIST OF
  !------UNELIMINATED NEIGHBORS
  MARK(VS) = TAG
  L(TAIL) = S
  TAIL = S
  GO TO 4
  !
  !------IF ES IS ACTIVE ELEMENT, THEN ...
  !--------FOR EACH VERTEX VB IN BOUNDARY LIST OF ELEMENT ES
2 LB = L(ES)
  BLPMAX = LAST(ES)
  DO BLP=1,BLPMAX
     B = LB
     LB = L(B)
     VB = V(B)
     !
     !----------IF VB IS UNTAGGED VERTEX, THEN TAG AND APPEND TO LIST OF
     !----------UNELIMINATED NEIGHBORS
     IF (MARK(VB) >= TAG)  GO TO 3
     MARK(VB) = TAG
     L(TAIL) = B
     TAIL = B
3    CONTINUE
  ENDDO
  !
  !--------MARK ES INACTIVE
  MARK(ES) = TAG
  !
4 GO TO 1
  !
  !----TERMINATE LIST OF UNELIMINATED NEIGHBORS
5 L(TAIL) = 0
  !
  RETURN
END SUBROUTINE MDM
!
!***********************************************************************
!  MDP -- PURGE INACTIVE ELEMENTS AND DO MASS ELIMINATION
!***********************************************************************
SUBROUTINE  MDP &
     (K,EK,TAIL, V,L, HEAD,LAST,NEXT, MARK)
  !
  implicit none
  !
  INTEGER  EK, TAIL,  V(1), L(1),  HEAD(1), LAST(1), NEXT(1), &
       MARK(1),  TAG, FREE, LI,VI,LVI,EVI, S,LS,ES, ILP,ILPMAX
  INTEGER I,K
  !
  !----INITIALIZE TAG
  TAG = MARK(EK)
  !
  !----FOR EACH VERTEX VI IN EK
  LI = EK
  ILPMAX = LAST(EK)
  IF (ILPMAX <= 0)  GO TO 12
  DO ILP=1,ILPMAX
     I = LI
     LI = L(I)
     VI = V(LI)
     !
     !------REMOVE VI FROM DEGREE LIST
     IF (LAST(VI) == 0)  GO TO 3
     IF (LAST(VI) > 0)  GO TO 1
     HEAD(-LAST(VI)) = NEXT(VI)
     GO TO 2
1    NEXT(LAST(VI)) = NEXT(VI)
2    IF (NEXT(VI) > 0)  LAST(NEXT(VI)) = LAST(VI)
     !
     !------REMOVE INACTIVE ITEMS FROM ELEMENT LIST OF VI
3    LS = VI
4    S = LS
     LS = L(S)
     IF (LS == 0)  GO TO 6
     ES = V(LS)
     IF (MARK(ES) < TAG)  GO TO 5
     FREE = LS
     L(S) = L(LS)
     LS = S
5    GO TO 4
     !
     !------IF VI IS INTERIOR VERTEX, THEN REMOVE FROM LIST AND ELIMINATE
6    LVI = L(VI)
     IF (LVI /= 0)  GO TO 7
     L(I) = L(LI)
     LI = I
     !
     K = K+1
     NEXT(VI) = -K
     LAST(EK) = LAST(EK) - 1
     GO TO 11
     !
     !------ELSE ...
     !--------CLASSIFY VERTEX VI
7    IF (L(LVI) /= 0)  GO TO 9
     EVI = V(LVI)
     IF (NEXT(EVI) >= 0)  GO TO 9
     IF (MARK(EVI) < 0)  GO TO 8
     !
     !----------IF VI IS PROTOTYPE VERTEX, THEN MARK AS SUCH, INITIALIZE
     !----------OVERLAP COUNT FOR CORRESPONDING ELEMENT, AND MOVE VI TO END
     !----------OF BOUNDARY LIST
     LAST(VI) = EVI
     MARK(EVI) = -1
     L(TAIL) = LI
     TAIL = LI
     L(I) = L(LI)
     LI = I
     GO TO 10
     !
     !----------ELSE IF VI IS DUPLICATE VERTEX, THEN MARK AS SUCH AND ADJUST
     !----------OVERLAP COUNT FOR CORRESPONDING ELEMENT
8    LAST(VI) = 0
     MARK(EVI) = MARK(EVI) - 1
     GO TO 10
     !
     !----------ELSE MARK VI TO COMPUTE DEGREE
9    LAST(VI) = -EK
     !
     !--------INSERT EK IN ELEMENT LIST OF VI
10   V(FREE) = EK
     L(FREE) = L(VI)
     L(VI) = FREE
11   CONTINUE
  ENDDO
  !
  !----TERMINATE BOUNDARY LIST
12 L(TAIL) = 0
  !
  RETURN
END SUBROUTINE MDP
!
!***********************************************************************
!  MDU -- UPDATE DEGREES OF UNELIMINATED VERTICES IN EK
!***********************************************************************
SUBROUTINE  MDU &
     (EK,DMIN, V,L, HEAD,LAST,NEXT, MARK)
  !
  implicit none
  !
  INTEGER  EK, DMIN,  V(1), L(1),  HEAD(1), LAST(1), NEXT(1), &
       MARK(1),  TAG, VI,EVI,DVI, S,VS,ES, B,VB, ILP,ILPMAX, &
       BLP,BLPMAX
  EQUIVALENCE  (VS, ES)
  INTEGER I
  !
  !----INITIALIZE TAG
  TAG = MARK(EK) - LAST(EK)
  !
  !----FOR EACH VERTEX VI IN EK
  I = EK
  ILPMAX = LAST(EK)
  IF (ILPMAX <= 0)  GO TO 11
  DO ILP=1,ILPMAX
     I = L(I)
     VI = V(I)
     IF (LAST(VI).eq.0) goto 10
     IF (LAST(VI).gt.0) goto 8
     !
     !------IF VI NEITHER PROTOTYPE NOR DUPLICATE VERTEX, THEN MERGE ELEMENTS
     !------TO COMPUTE DEGREE
     TAG = TAG + 1
     DVI = LAST(EK)
     !
     !--------FOR EACH VERTEX/ELEMENT VS/ES IN ELEMENT LIST OF VI
     S = L(VI)
2    S = L(S)
     IF (S == 0)  GO TO 9
     VS = V(S)
     IF (NEXT(VS) < 0)  GO TO 3
     !
     !----------IF VS IS UNELIMINATED VERTEX, THEN TAG AND ADJUST DEGREE
     MARK(VS) = TAG
     DVI = DVI + 1
     GO TO 5
     !
     !----------IF ES IS ACTIVE ELEMENT, THEN EXPAND
     !------------CHECK FOR OUTMATCHED VERTEX
3    IF (MARK(ES) < 0)  GO TO 6
     !
     !------------FOR EACH VERTEX VB IN ES
     B = ES
     BLPMAX = LAST(ES)
     DO BLP=1,BLPMAX
        B = L(B)
        VB = V(B)
        !
        !--------------IF VB IS UNTAGGED, THEN TAG AND ADJUST DEGREE
        IF (MARK(VB) >= TAG)  GO TO 4
        MARK(VB) = TAG
        DVI = DVI + 1
4       CONTINUE
     ENDDO
     !
5    GO TO 2
     !
     !------ELSE IF VI IS OUTMATCHED VERTEX, THEN ADJUST OVERLAPS BUT DO NOT
     !------COMPUTE DEGREE
6    LAST(VI) = 0
     MARK(ES) = MARK(ES) - 1
7    S = L(S)
     IF (S == 0)  GO TO 10
     ES = V(S)
     IF (MARK(ES) < 0)  MARK(ES) = MARK(ES) - 1
     GO TO 7
     !
     !------ELSE IF VI IS PROTOTYPE VERTEX, THEN CALCULATE DEGREE BY
     !------INCLUSION/EXCLUSION AND RESET OVERLAP COUNT
8    EVI = LAST(VI)
     DVI = LAST(EK) + LAST(EVI) + MARK(EVI)
     MARK(EVI) = 0
     !
     !------INSERT VI IN APPROPRIATE DEGREE LIST
9    NEXT(VI) = HEAD(DVI)
     HEAD(DVI) = VI
     LAST(VI) = -DVI
     IF (NEXT(VI) > 0)  LAST(NEXT(VI)) = VI
     IF (DVI < DMIN)  DMIN = DVI
     !
10   CONTINUE
  ENDDO
  !
11 RETURN
END SUBROUTINE MDU
!***********************************************************************
!***********************************************************************
!     Modified Routines  of the Yale Sparse Marix Package (YSMP):
!
!         SDRVMD   - a modified form of the SDRV driver.
!         SNFMOD   - a modified form of the SNF routine.
!
!               copyright (c) 1990 by Tamar Schlick
!
!***********************************************************************
!        Yale's SDRV solves a linear system for (symmetric) positive-
!   definite matrices. It calls the following routines:
!                 SSF (for symbolic factorization)
!                 SNF (for numerical factorization) and
!                 SNS (for numerical solution).
!        Our goal is to solve large sparse symmetric linear systems for
!   matrices that are not necessarily pos-def. Thus, we
!   replace SNF by SNFMOD so that a modified-Cholesky (MCF), rather than
!   a Cholesky, factorization is performed. In SDRV, we replace
!   the statememt "CALL SNF" by "CALL SNFMOD".
!        In Yale's SDRV, the diagnostic parameter FLAG is set to zero in
!   the non pos-def case (and control is returned to the main program).
!   Here, instead, we set FLAG in SNFMOD to a negative integer if the
!   matrix is not sufficiently pos-def. Specifically, FLAG is set to minus
!   the position of the diagonal element in the original matrix whose
!   modification was of the largest magnitude. Recall that in MCF we
!   produce matrices E,D, and U so that  M + E = UT-D-U  where E and D are
!   diagonal and U is unit upper-triangular. FLAG records the index k
!   for the largest modification in E: ( E(P(k)) = max over i {E(i)} ).
!
!   All modifications to the original YSMP code are indicated.
!                                                                1/15/81
!***********************************************************************
!  SDRV -- DRIVER FOR SPARSE SYMMETRIC POSITIVE DEFINITE MATRIX ROUTINES
!***********************************************************************
!
! ====================  change #1  (replacement) =====================1
! WAS:  SUBROUTINE SDRV
! =====================================================================
SUBROUTINE SDRVMD &
     ! ==================================================================end
     (N, P,IP, IA,JA,A, B, Z, NSP,ISP,RSP,ESP, PATH, FLAG,NZ)
  !
  !  DESCRIPTION
  !
  ! ====================  change #2  (replacement) =====================2
  ! WAS: SDRV SOLVES SPARSE SYMMETRIC POSITIVE DEFINITE SYSTEMS OF LINEAR
  ! =====================================================================
  !    SDRVMD SOLVES SPARSE SYMMETRIC SYSTEMS OF LINEAR
  ! ==================================================================end
  !    EQUATIONS.  THE SOLUTION PROCESS IS DIVIDED INTO THREE STAGES --
  !
  !      SSF - THE COEFFICIENT MATRIX M IS FACTORED SYMBOLICALLY TO
  !            DETERMINE WHERE FILLIN WILL OCCUR DURING THE NUMERIC
  !            FACTORIZATION.
  !
  ! ====================  change #3  (replacement) =====================3
  ! WAS: SNF - M IS FACTORED NUMERICALLY INTO THE PRODUCT UT-D-U, WHERE
  ! =====================================================================
  !      SNFMOD - M+E IS FACTORED NUMERICALLY BY THE GILL/MURRAY/WRIGHT
  !            MODIFIED CHOLESKY FACTORIZATION: M + E = UT-D-U, WHERE
  !            E IS DIAGONAL,
  ! ==================================================================end
  !            D IS DIAGONAL AND U IS UNIT UPPER TRIANGULAR.
  !
  !      SNS - THE LINEAR SYSTEM  MX = B  IS SOLVED USING THE UT-D-U
  !            FACTORIZATION FROM SNF.
  !
  !    FOR SEVERAL SYSTEMS WITH THE SAME COEFFICIENT MATRIX, SSF AND SNF
  !    NEED BE DONE ONLY ONCE (FOR THE FIRST SYSTEM);  THEN SNS IS DONE
  !    ONCE FOR EACH ADDITIONAL RIGHT-HAND SIDE.  FOR SEVERAL SYSTEMS
  !    WHOSE COEFFICIENT MATRICES HAVE THE SAME NONZERO STRUCTURE, SSF
  !    NEED BE DONE ONLY ONCE (FOR THE FIRST SYSTEM);  THEN SNF AND SNS
  !    ARE DONE ONCE FOR EACH ADDITIONAL SYSTEM.
  !
  !
  !  STORAGE OF SPARSE MATRICES
  !
  !    THE NONZERO ENTRIES OF THE MATRIX M ARE STORED ROW-BY-ROW IN THE
  !    ARRAY A.  TO IDENTIFY THE INDIVIDUAL NONZERO ENTRIES IN EACH ROW,
  !    WE NEED TO KNOW IN WHICH COLUMN EACH ENTRY LIES.  THESE COLUMN
  !    INDICES ARE STORED IN THE ARRAY JA;  I.E., IF  A(K) = M(I,J),  THEN
  !    JA(K) = J.  TO IDENTIFY THE INDIVIDUAL ROWS, WE NEED TO KNOW WHERE
  !    EACH ROW STARTS.  THESE ROW POINTERS ARE STORED IN THE ARRAY IA;
  !    I.E., IF M(I,J) IS THE FIRST NONZERO ENTRY (STORED) IN THE I-TH ROW
  !    AND  A(K) = M(I,J),  THEN  IA(I) = K.  MOREOVER, IA(N+1) POINTS TO
  !    THE FIRST LOCATION FOLLOWING THE LAST ELEMENT IN THE LAST ROW.
  !    THUS, THE NUMBER OF ENTRIES IN THE I-TH ROW IS  IA(I+1) - IA(I),
  !    THE NONZERO ENTRIES IN THE I-TH ROW ARE STORED CONSECUTIVELY IN
  !
  !            A(IA(I)),  A(IA(I)+1),  ..., A(IA(I+1)-1),
  !
  !    AND THE CORRESPONDING COLUMN INDICES ARE STORED CONSECUTIVELY IN
  !
  !            JA(IA(I)), JA(IA(I)+1), ..., JA(IA(I+1)-1).
  !
  !    SINCE THE COEFFICIENT MATRIX IS SYMMETRIC, ONLY THE NONZERO ENTRIES
  !    IN THE UPPER TRIANGLE NEED BE STORED, FOR EXAMPLE, THE MATRIX
  !
  !             ( 1  0  2  3  0 )
  !             ( 0  4  0  0  0 )
  !         M = ( 2  0  5  6  0 )
  !             ( 3  0  6  7  8 )
  !             ( 0  0  0  8  9 )
  !
  !    COULD BE STORED AS
  !
  !            \ 1  2  3  4  5  6  7  8  9 10 11 12 13
  !         ---+--------------------------------------
  !         IA \ 1  4  5  8 12 14
  !         JA \ 1  3  4  2  1  3  4  1  3  4  5  4  5
  !          A \ 1  2  3  4  2  5  6  3  6  7  8  8  9
  !
  !    OR (SYMMETRICALLY) AS
  !
  !            \ 1  2  3  4  5  6  7  8  9
  !         ---+--------------------------
  !         IA \ 1  4  5  7  9 10
  !         JA \ 1  3  4  2  3  4  4  5  5
  !          A \ 1  2  3  4  5  6  7  8  9          .
  !
  !
  !  REORDERING THE ROWS AND COLUMNS OF M
  !
  !    A SYMMETRIC PERMUTATION OF THE ROWS AND COLUMNS OF THE COEFFICIENT
  !    MATRIX M (E.G., WHICH REDUCES FILLIN OR ENHANCES NUMERICAL
  !    STABILITY) MUST BE SPECIFIED.  THE SOLUTION Z IS RETURNED IN THE
  !    ORIGINAL ORDER.
  !
  !    TO SPECIFY THE TRIVIAL ORDERING (I.E., THE IDENTITY PERMUTATION),
  !    SET  P(I) = IP(I) = I,  I=1,...,N.  IN THIS CASE, P AND IP CAN BE
  !    THE SAME ARRAY.
  !
  !    IF A NONTRIVIAL ORDERING (I.E., NOT THE IDENTITY PERMUTATION) IS
  !    SPECIFIED AND M IS STORED SYMMETRICALLY (I.E., NOT BOTH M(I,J) AND
  !    M(J,I) ARE STORED FOR I NE J), THEN ODRV SHOULD BE CALLED (WITH
  !    PATH = 3 OR 5) TO SYMMETRICALLY REORDER (IA,JA,A) BEFORE CALLING
  !    SDRV.  THIS IS TO ENSURE THAT IF M(I,J) WILL BE IN THE UPPER
  !    TRIANGLE OF M WITH RESPECT TO THE NEW ORDERING, THEN M(I,J) IS
  !    STORED IN ROW I (AND THUS M(J,I) IS NOT STORED);  WHEREAS IF M(I,J)
  !    WILL BE IN THE STRICT LOWER TRIANGLE OF M, THEN M(J,I) IS STORED IN
  !    ROW J (AND THUS M(I,J) IS NOT STORED).
  !
  !
  !  PARAMETERS
  !
  !    N    - NUMBER OF VARIABLES/EQUATIONS
  !
  !    P    - INTEGER ONE-DIMENSIONAL ARRAY SPECIFYING A PERMUTATION OF
  !           THE ROWS AND COLUMNS OF M;  DIMENSION = N
  !
  !    IP   - INTEGER ONE-DIMENSIONAL ARRAY CONTAINING THE INVERSE OF THE
  !           PERMUTATION SPECIFIED IN P;  I.E., IP(P(I)) = I, I=1,...,N;
  !           DIMENSION = N
  !
  !    IA   - INTEGER ONE-DIMENSIONAL ARRAY CONTAINING POINTERS TO DELIMIT
  !           ROWS IN JA AND A;  DIMENSION = N+1
  !
  !    JA   - INTEGER ONE-DIMENSIONAL ARRAY CONTAINING THE COLUMN INDICES
  !           CORRESPONDING TO THE ELEMENTS OF A;  DIMENSION = NUMBER OF
  !           NONZERO ENTRIES IN M STORED
  !
  !    A    - REAL ONE-DIMENSIONAL ARRAY CONTAINING THE NONZERO ENTRIES IN
  !           THE COEFFICIENT MATRIX M, STORED BY ROWS;  DIMENSION =
  !           NUMBER OF NONZERO ENTRIES IN M STORED
  !
  !    B    - REAL ONE-DIMENSIONAL ARRAY CONTAINING THE RIGHT-HAND SIDE B;
  !           B AND Z CAN BE THE SAME ARRAY;  DIMENSION = N
  !
  !    Z    - REAL ONE-DIMENSIONAL ARRAY CONTAINING THE SOLUTION X;  Z AND
  !           B CAN BE THE SAME ARRAY;  DIMENSION = N
  !
  !    NSP  - DECLARED DIMENSION OF THE ONE-DIMENSIONAL ARRAYS ISP AND
  !           RSP;  NSP MUST BE (SUBSTANTIALLY) LARGER THAN  3N+2K,  WHERE
  !           K = NUMBER OF NONZERO ENTRIES IN THE UPPER TRIANGLE OF M
  !
  !    ISP  - INTEGER ONE-DIMENSIONAL ARRAY USED FOR WORKING STORAGE;  ISP
  !           AND RSP SHOULD BE EQUIVALENCED;  DIMENSION = NSP
  !
  !    RSP  - REAL ONE-DIMENSIONAL ARRAY USED FOR WORKING STORAGE;  RSP
  !           AND ISP SHOULD BE EQUIVALENCED;  DIMENSION = NSP
  !
  !    ESP  - INTEGER VARIABLE;  IF SUFFICIENT STORAGE WAS AVAILABLE TO
  !           PERFORM THE SYMBOLIC FACTORIZATION (SSF), THEN ESP IS SET TO
  !           THE AMOUNT OF EXCESS STORAGE PROVIDED (NEGATIVE IF
  !           INSUFFICIENT STORAGE WAS AVAILABLE TO PERFORM THE NUMERIC
  !           FACTORIZATION (SNF))
  !
  !    PATH - INTEGER PATH SPECIFICATION;  VALUES AND THEIR MEANINGS ARE -
  !             1  PERFORM SSF, SNF, AND SNS
  !             2  PERFORM SNF AND SNS (ISP/RSP IS ASSUMED TO HAVE BEEN
  !                  SET UP IN AN EARLIER CALL TO SDRV (FOR SSF))
  !             3  PERFORM SNS ONLY (ISP/RSP IS ASSUMED TO HAVE BEEN SET
  !                  UP IN AN EARLIER CALL TO SDRV (FOR SSF AND SNF))
  !             4  PERFORM SSF
  !             5  PERFORM SSF AND SNF
  !             6  PERFORM SNF ONLY (ISP/RSP IS ASSUMED TO HAVE BEEN SET
  !                  UP IN AN EARLIER CALL TO SDRV (FOR SSF))
  !
  !    FLAG - INTEGER ERROR FLAG;  VALUES AND THEIR MEANINGS ARE -
  !
  !               0     NO ERRORS DETECTED
  !              2N+K   DUPLICATE ENTRY IN A  --  ROW = K
  !              6N+K   INSUFFICIENT STORAGE IN SSF  --  ROW = K
  !              7N+1   INSUFFICIENT STORAGE IN SNF
  !              8N+K   ZERO PIVOT  --  ROW = K
  !             10N+1   INSUFFICIENT STORAGE IN SDRV
  !             11N+1   ILLEGAL PATH SPECIFICATION
  !
  ! ====================  change #4  (insertion) =======================4
  !              <0     MATRIX NOT SUFF. POS-DEF (detected in SNFMOD)
  !                     FLAG IS SET TO MINUS THE INDEX (IN THE ORIGINAL
  !                     MATRIX) OF THE LARGEST ADDITION in E
  ! ==================================================================end
  !
  !
  !  CONVERSION FROM REAL TO real(chm_real)
  !
  !    CHANGE THE REAL DECLARATIONS IN SDRV, SNF, AND SNS TO DOUBLE
  !    PRECISION DECLARATIONS;  AND CHANGE THE VALUE IN THE DATA STATEMENT
  !    FOR THE INTEGER VARIABLE RATIO (IN SDRV) FROM 1 TO 2.
  !
  !  NOTE: FOR CRAY, SET RATIO to 1!
  !-----------------------------------------------------------------------
  implicit none
  !
  INTEGER JUMAX,IL,JL,IU,JU,N,IJU,NSP,NZ
  INTEGER  P(N), IP(N), IA(N+1), JA(N+NZ), ISP(NSP), ESP, PATH, FLAG, &
       Q, MARK, D, U, TMP, UMAX
  real(chm_real) A(N+NZ), B(N), Z(N), RSP(NSP)
  INTEGER :: RATIO=2
  !
  !----VALIDATE PATH SPECIFICATION
  IF (PATH < 1 .OR. 6.LT.PATH)  GO TO 111
  !
  !----ALLOCATE STORAGE AND FACTOR M SYMBOLICALLY TO DETERMINE FILL-IN
  IJU   = 1
  IU    = IJU     +  N
  JL    = IU      + N+1
  JU    = JL      +  N
  Q     = (NSP+1) -  N
  MARK  = Q       -  N
  JUMAX = MARK    - JU
  !
  IF ((PATH-1) * (PATH-4) * (PATH-5)  /=  0)  GO TO 1
  IF (JUMAX <= 0)  GO TO 110
  CALL SSF &
       (N,NZ,P,IP,IA,JA,A,ISP(IJU), ISP(JU), ISP(IU), JUMAX, &
       ISP(Q), ISP(MARK), ISP(JL), FLAG)
  IF (FLAG /= 0)  GO TO 100
  !
  !----ALLOCATE STORAGE AND FACTOR M NUMERICALLY
1 IL   = JU      + ISP(IJU+(N-1))
  TMP  = ((IL-1)+(RATIO-1)) / RATIO  +  1
  D    = TMP     + N
  U    = D       + N
  UMAX = (NSP+1) - U
  ESP  = UMAX    - (ISP(IU+N)-1)
  !
  IF ((PATH-1) * (PATH-2) * (PATH-5) * (PATH-6)  /=  0)  GO TO 2
  IF (UMAX <= 0)  GO TO 110
  !
  ! ====================  change #5  (replacement) =====================5
  !  WAS:   CALL SNF
  ! ==================================================================end
  !
  CALL SNFMOD &
       (N,NZ,  P, IP,  IA, JA, A, &
       RSP(D),  ISP(IJU), ISP(JU), ISP(IU), RSP(U), UMAX, &
       ISP(IL),  ISP(JL),  FLAG)
  !
  ! ====================  change #6  (replacement) =====================6
  !  WAS:         IF (FLAG /= 0)  GO TO 100
  ! ==================================================================end
  !
  IF (FLAG > 0)  GO TO 100
  !
  !----SOLVE SYSTEM OF LINEAR EQUATIONS  MX = B
2 IF ((PATH-1) * (PATH-2) * (PATH-3)  /=  0)  GO TO 3
  IF (UMAX <= 0)  GO TO 110
  CALL SNS &
       (N,  P,  RSP(D), ISP(IJU), ISP(JU), ISP(IU), RSP(U),  Z, B, &
       RSP(TMP))
  !
3 RETURN
  !
  ! ** ERROR -- ERROR DETECTED IN SSF, SNF, OR SNS
100 RETURN
  ! ** ERROR -- INSUFFICIENT STORAGE
110 FLAG = 10*N + 1
  RETURN
  ! ** ERROR -- ILLEGAL PATH SPECIFICATION
111 FLAG = 11*N + 1
  RETURN
END SUBROUTINE SDRVMD
!
!***********************************************************************
!***********************************************************************
! NUMERICAL FACTORIZATION OF SYMMETRIC MATRICES
!***********************************************************************
!
! ====================  change #1  (replacement) =====================1
! WAS:
!  SNF -- NUMERICAL UT-D-U FACTORIZATION OF SPARSE SYMMETRIC POSITIVE
!         DEFINITE MATRIX
!       SUBROUTINE  SNF
! =====================================================================
!
!  SNFMOD -- NUMERICAL FACTORIZATION OF SPARSE SYMMETRIC MATRICES M BY
!         THE GILL/MURRAY/WRIGHT MODIFIED CHOLESKY FACTORIZATION (GMW
!         MCF) WITHOUT PIVOTING.  THE FACTORIZATION PRODUCES U,D, AND
!         E SO THAT   M + E = UT-D-U,  WHERE  E AND D ARE DIAGONAL
!         MATRICES. THIS ROUTINE IS A MODIFICATION OF THE YSMP
!         routine SNF. ALL CHANGES ARE INDICATED.
!
SUBROUTINE  SNFMOD &
     ! ==================================================================end
     (N,NZ, P,IP, IA,JA,A, D, IJU,JU,IU,U,UMAX, IL, JL, FLAG)
  !
  !  ADDITIONAL PARAMETERS
  !
  !    IL    - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !    JL    - INTEGER ONE-DIMENSIONAL WORK ARRAY;  DIMENSION = N
  !
  !
  !  DEFINITIONS OF INTERNAL PARAMETERS (DURING K-TH STAGE OF ELIMINATION)
  !
  !    (D(I),I=K,N) CONTAINS THE K-TH ROW OF U (EXPANDED)
  !
  !    IL(I) POINTS TO THE FIRST NONZERO ELEMENT IN COLUMNS K,...,N OF
  !      ROW I OF U
  !
  !    JL CONTAINS LISTS OF ROWS TO BE ADDED TO UNELIMINATED ROWS --
  !      I GE K => JL(I) IS THE FIRST ROW TO BE ADDED TO ROW I
  !      I LT K => JL(I) IS THE ROW FOLLOWING ROW I IN SOME LIST OF ROWS
  !      IN EITHER CASE, JL(I) = 0 INDICATES THE END OF A LIST
  !
  !-----------------------------------------------------------------------
  use number
  implicit none
  !
  INTEGER NEXTI,JUMUJ,MU,JMIN,JMAX,I,J,K,N,ILI,NZ
  INTEGER P(n), IP(n), IA(n+1), JA(n+nz), IJU(*), JU(*), IU(*), &
       UMAX, IL(*), JL(*), FLAG, VJ
  real(chm_real) A(n+nz), D(n), U(*), DK, UKIDI
  !
  ! ====================  change #2  (insertion) =======================2
  INTEGER IROW, KK,KKMIN,KKMAX
  real(chm_real) GAMMA,XI,XIN,EPS,DEL,EPS1,ELT,ELT2,ELTNEW, &
       BOUND,W,EMAX,EK,DK1
  !
  FLAG = 0
  EMAX = ZERO
  ! ==================================================================end
  !
  !----CHECK FOR SUFFICIENT STORAGE FOR U
  IF (IU(N+1)-1  >  UMAX)  GO TO 107
  !
  !----INITIALIZATION
  DO K=1,N
     D(K) = 0
     JL(K) = 0
  ENDDO
  ! ====================  change #3  (insertion) =======================3
  ! Calculate GAMMA and XI, the largest magnitudes of the diag. and  off-
  ! diag. elements, respectively. When the diag. elts. are stored first
  ! in A in each row (PATH = 4 or 5 in ODRV), GAMMA=max(GAMMA,A(IA(i))),
  ! i=1,...,IA(n+1)-1. We assume that this IS the case. (If this were
  ! later changed, then for each row I we would have to loop through KK
  ! = KKMIN,..., KKMAX  where KKMIN = IA(I), KKMAX = IA(I+1)-1, and test
  ! whether I = JA(KK), ie. row index = column index. If this equality
  ! holds, the element is a diagonal). Then calculate DEL and BOUND:
  ! DEL =  max ( max(XI,GAMMA)*EPS, EPS) where EPS is a small given
  ! number, and  BOUND = max ( XI/N, GAMMA, EPS).
  ! =====================================================================
  EPS = 1.0D-06
  GAMMA = ZERO
  XI    = ZERO
  DO IROW = 1, N
     GAMMA = MAX ( GAMMA, ABS (A (IA(IROW)) ) )
     KKMIN = IA( IROW ) + 1
     KKMAX = IA(IROW+1) - 1
     IF (KKMIN  >  KKMAX) GOTO 21
     DO KK = KKMIN, KKMAX
        XI    = MAX ( XI   , ABS (A(KK)) )
     ENDDO
21   CONTINUE
  ENDDO
  EPS1 = MAX (GAMMA,XI) * EPS
  DEL  = MAX (EPS,EPS1)
  XIN = N
  XIN = XI/XIN
  BOUND=MAX (GAMMA,XIN,EPS)
  !
  ! ==================================================================end
  !
  !----FOR EACH ROW K
  DO K=1,N
     !
     !------INITIALIZE K-TH ROW WITH ELEMENTS NONZERO IN ROW P(K) OF M
     JMIN = IA(P(K))
     JMAX = IA(P(K)+1) - 1
     IF (JMIN > JMAX) GO TO 5
     DO J=JMIN,JMAX
        VJ = IP(JA(J))
        IF (K <= VJ)  D(VJ) = A(J)
     ENDDO
     !
     !------MODIFY K-TH ROW BY ADDING IN THOSE ROWS I WITH U(I,K) NE 0
     !------FOR EACH ROW I TO BE ADDED IN
5    DK = D(K)
     I = JL(K)
6    IF (I == 0)  GO TO 9
     NEXTI = JL(I)
     !
     !--------COMPUTE MULTIPLIER AND UPDATE DIAGONAL ELEMENT
     ILI = IL(I)
     UKIDI = - U(ILI) * D(I)
     DK = DK + UKIDI * U(ILI)
     U(ILI) = UKIDI
     !
     !--------ADD MULTIPLE OF ROW I TO K-TH ROW ...
     JMIN = ILI     + 1
     JMAX = IU(I+1) - 1
     IF (JMIN > JMAX)  GO TO 8
     MU = IJU(I) - IU(I)
     DO J=JMIN,JMAX
        D(JU(MU+J)) = D(JU(MU+J)) + UKIDI * U(J)
     ENDDO
     !
     !--------... AND ADD I TO ROW LIST FOR NEXT NONZERO ENTRY
     IL(I) = JMIN
     J = JU(MU+JMIN)
     JL(I) = JL(J)
     JL(J) = I
     !
8    I = NEXTI
     GO TO 6
     !
     ! ====================  change #4  (replacement) =====================4
     ! WAS:
     !------CHECK FOR ZERO PIVOT
     !  9      IF (DK == 0)  GO TO 108
     ! =====================================================================
     ! STATEMENT 9 ABOVE WILL BE MODIFIED TO RESET Dk IN THE EVENT THE
     ! THE MATRIX IS NOT SUFF. POSITIVE-DEFINITE. NOTE THAT EVEN WHEN Dk>0,
     ! IT MAY BE MODIFIED IF THE MATRIX IS NOT POS. DEF!
     !
     ! Dk is set as:  Dk = MAX ( ABS(Dk), DEL, (ELT**2)/BOUND), where
     ! ELT is the largest magnitude among the elements in the Kth row of U.
     ! This restriction guarantees that all elts. of D are strictly positive
     ! and that the elts. of the factors satisfy a uniform bound.
     ! [   Recall that we work with the auxiliary quantities  Vik = Uik * Dk.
     !     The bound we want to impose on the elts. of U,
     !          ( max(Uik)**2 )  * Dk  <=  BOUND, is equivalent to
     !          ( max(Vik)**2 )  / Dk  <=  BOUND, or
     !          Dk   >=    (max(Vik)**2) / BOUND.)
     !     The value for ELT = max(Vik), max over i for fixed k, is found by
     !     looping through the appropriate elements of U. These elements
     !     are currently stored in part of D.  ]
     !
     ! =====================================================================

     !
9    W = DK
     ! =====================================================================
     ELT=ZERO
     !
     JMIN=IU(K)
     JMAX=IU(K+1)-1
     MU=IJU(K)-JMIN
     IF(JMIN > JMAX)GO TO 28
     DO J=JMIN,JMAX
        ELTNEW = ABS ( D (JU(MU+J)) )
        ELT = MAX ( ELT , ELTNEW )
     ENDDO
28   CONTINUE
     !
     !
     ELT2 = ELT * ELT
     DK=MAX ( ABS(DK) , DEL, ELT2 / BOUND)
     EK = DK - W
     !
     ! =====  CHARMM SEGMENT ##  24  ===========================
     !  Modify MC factorization to avoid large modifications.
     DK1 = 1.0D0/DK
     IF(EK  /=  ZERO .AND. DK1  <=  5.0E-2)  THEN
        DK = W
     ENDIF
     ! =========================================================
     !
     IF (EK  >  EMAX) THEN
        EMAX = EK
        FLAG = -P(K)
     ENDIF
     !
     ! ==================================================================end
     !
     !------SAVE DIAGONAL ELEMENT
     D(K) = 1 / DK
     !------SAVE NONZERO ENTRIES IN K-TH ROW OF U ...
     JMIN = IU(K)
     JMAX = IU(K+1) - 1
     IF (JMIN > JMAX)  GO TO 11
     MU = IJU(K) - JMIN
     DO J=JMIN,JMAX
        JUMUJ = JU(MU+J)
        U(J) = D(JUMUJ)
        D(JUMUJ) = 0
     ENDDO
     !
     !------... AND ADD K TO ROW LIST FOR FIRST NONZERO ENTRY IN K-TH ROW
     IL(K) = JMIN
     I = JU(MU+JMIN)
     JL(K) = JL(I)
     JL(I) = K
11   CONTINUE
  ENDDO
  !
  ! uncomment next line to check for the largest diagonal modification
  !          IF (FLAG  <  0) WRITE (6,*) '      NMAX, EMAX', FLAG, EMAX
  !            stop
  ! ====================  change #5  (deletion) ========================5
  ! WAS:     FLAG = 0
  ! ==================================================================end
  RETURN
  !
  ! ** ERROR -- INSUFFICIENT STORAGE FOR U
107 CONTINUE
  FLAG = 7*N + 1
  RETURN
  ! ** ERROR -- ZERO PIVOT
  !108    CONTINUE
  !       FLAG = 8*N + K
  !       RETURN
END SUBROUTINE SNFMOD
! -------------------------------------------------------
!       SUBROUTINE SCHEDULE
!
! =====  CHARMM SEGMENT ##  25  ===========================
!       date : 20 January 1993.
!       This subroutine limits TNPACK operations so that not
!       too much time is wasted on expensive operations far away
!       from local minima.
!       Progress can be estimated from the energy value for the
!       system. We use the following heuristics.
!
!       1. the sharp switching function (cutnb 8.0 A)
!       For BPTI (59 residues, 1740 variables by using the extended
!       atomic representation), E_min is about -900,-1000 kcals/mol.
!       Consequently, if the energy is higher than -750*n/1740
!       where n is the number of variables, SD method and then ten
!       Truncated Newton iterations without preconditioning will
!       be first carried out.
!       The SD method is turned on until the energy difference
!       between two sucessive steps is less than:
!       3.5*n/1740 (minimization of the potential energy function)
!       100*n/1740 (minimization of the "dynamics" function).
!
!       2. the long-range shifting function (cutnb 13.0, ctofnb 12.0 A)
!
!       -750 is replaced by -2000 ,
!       For N<3000 variables , 20 Nopreco TN iterations
!       For N>3000 variables , 40 Nopreco TN iterations can be carried
!       out.

!       Note that NOPRECO CG runs with the quadratic residual test, and
!       either the finite difference in the gradient or explicit Hd 
!       product.
!       Later, TN with PRECO is picked out and, residual truncation test
!       and explicit Hd product are selected.


SUBROUTINE SCHEDU(ITRMAJ,N,F,IPCG,IEXIT,MXITCG,IHD,OPLIST, &
     OKSD,IA,JA,A,NZ,FOLD,ITRIND,ITRINP,GNORM,RNORM,MODE,OKSCH,ITT)
  use dimens_fcm
  use euler
  use inbnd
  use stream
  implicit none

  LOGICAL OKSD,OKSCH
  INTEGER ITRMAJ,N,NZ,IPCG,IEXIT,MXITCG,IHD,OPLIST(20)
  INTEGER IA(N+1),JA(N+NZ),ITRIND,ITRINP,MODE,ITT
  real(chm_real) GNORM,RNORM,F,A(N+NZ),FOLD,FLT,FPI,FREF
  LOGICAL F130,F3444

  !      PARAMETER (FREF = -750.0d0)  !! cutnb=8.0, switch elec
  !      PARAMETER (FREF = -2000.0d0) !! cutnb=13.0 shift elec , eps=1

  F130=.FALSE.
  F3444=.FALSE.
  !  Test on energy

  IF (CUTNB  <  9.0) FREF = -750.0D0
  IF (CUTNB  >=  9.0) FREF = -2000.0D0

  IF(ABS(F) < ABS(FREF*N/1740.) .OR. F > (FREF*N/1740.)) THEN

     OKSCH = .TRUE.
     IF(ITRMAJ  ==  1) GO TO 120
     !  Acceptable decrease in the energy between two sucessive SD steps
     !  is defined.
     IF(N <=  3000) FPI=3.5D0*n/1740. !Minimization Potential Energy
     IF(N >  3000) FPI=1.0D0*n/1740. !Minimization Potential Energy
     IF(QEULER) FPI = 100.0D0*N/1740.     ! Dynamics function

     IF(ABS(F-FOLD)  <=  FPI .OR. .NOT.OKSD) F130=.TRUE.
120  CONTINUE 
     IF(GNORM  <=  0.05) F3444=.TRUE.
     !  SD method is turned on
     IF(ITRMAJ  <=  1 .AND. .NOT. (F130.OR.F3444)) THEN 
        IF(WRNLEV > 2) THEN
           WRITE(OUTU,*)' '
           WRITE(OUTU,*) &
                '       too far away for quadratic approximation, SD'
           WRITE(OUTU,*) &
                '       is turned on until new conditions noticed'
           WRITE(OUTU,*)' '
        ENDIF
     ENDIF

     IF (.NOT. (F130.OR.F3444)) THEN
        CALL PARMSD(IPCG,IEXIT,MXITCG,IHD,ITRIND,ITRMAJ,OKSD,OPLIST)
        GOTO 3445
     ENDIF

  ENDIF  ! energy test

  IF (F3444) GOTO 3444

  !   TN Without preconditioning is now chosen, otherwise continue 3444
  IF(.NOT.OKSCH .OR. GNORM  <=  0.1) GO TO 3444
  FLT = OPLIST(5)
  IF(ITRMAJ  ==  ITRIND+1) THEN
     IF(WRNLEV > 2) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) &
             '         ITN performed without preconditioning'
        WRITE(OUTU,*) ' '
     ENDIF
  ELSE
     !   Test on the variation of the energy and number of TN without PRECO.

     IF(CUTNB  >=  9.0) GO TO 1016
     IF(ABS(F-FOLD)/FLT  <=  1.5D0 .AND. ITRMAJ  >  ITRIND+10) &
          GO TO 3444  !! cutnb = 8.0A

1016 CONTINUE
     IF(CUTNB  <  9.0) GO TO 1017 

     IF(N  <=  3000) THEN
        IF(ABS(F-FOLD)/FLT  <=  3.0D0 .AND. ITRMAJ  >  ITRIND+20) &
             GO TO 3444  !! cutnb = 12.0A
     ELSE
        IF(ABS(F-FOLD)/FLT  <=  3.0D0 .AND. ITRMAJ  >  ITRIND+40) &
             GO TO 3444  !! cutnb = 12.0A
     ENDIF
  ENDIF

1017 CONTINUE
  OKSD=.FALSE.
  OKSCH=.TRUE.
  IPCG = 0
  IEXIT = 1
  MXITCG = OPLIST(5) * 2
  IHD = 1 
  ITT = 1
  ITRINP = ITRMAJ 
  GO TO 3077


3444 CONTINUE
  !   TN with Preconditioning is chosen.

  IF(ITRMAJ  ==  ITRINP+1) THEN 
     IF(WRNLEV > 2) THEN
        WRITE(OUTU,*) ' '
        WRITE(OUTU,*) &
             ' the initial conditions for minimization are applied '
        WRITE(OUTU,*) ' '
     ENDIF
  ENDIF

  OKSD =.FALSE.
  OKSCH =.FALSE.
  MXITCG = OPLIST(5) + OPLIST(5) 
  IPCG = OPLIST(1)
  IEXIT = OPLIST(12) 
  IHD = OPLIST(10)
  ITT = oplist(13)

3077 CONTINUE 

3445 CONTINUE

  RETURN 
END SUBROUTINE SCHEDU
! ***********************************************************

SUBROUTINE PARMSD(IPCG,IEXIT,MXITCG,IHD,ITRIND,ITRMAJ, &
     OKSD,OPLIST)

  ! =====  CHARMM SEGMENT ##  26  ===========================
  !     date : 20 January 1993.
  !     Parameters for SD runs.

  INTEGER IPCG,IEXIT,MXITCG,IHD,ITRIND,ITRMAJ,OPLIST(20)
  LOGICAL OKSD

  IPCG = 0
  IEXIT = 0
  MXITCG = 1
  IHD = 1 
  ITRIND = ITRMAJ
  OKSD = .TRUE.

  RETURN
END SUBROUTINE PARMSD

! --------------------------------------------------------
SUBROUTINE BESTSE(MODE,N,X,F,G,P,LAMBDA,ILINE,Y,NFUN, &
     NFEV,CALFGH,OKSD,ITRMAJ,XSNORM,OKIL6,RNORM,OKPG,GNORM, &
     FOLD,GNMOLD,D1,X1,X2,X3,G11,G22,G33)

  ! =====  CHARMM SEGMENT ##  27  ===========================
  !     date : 6 February 1993.
  !     This subroutine is designed to compute the best search
  !     according to the current program status
  !     (the mode parameter from the inner loop).
  !     Candidates are as follows:

  !     if mode = 0               P only (skip)
  !             = 1               D only
  !             = 2 and GNORM>1   D, P and -G  
  !             = 2 and GNORM<1   D and P 
  !             = 3               P only (skip)

  use number
  use stream
  use contrl
  implicit none

  INTEGER N,I,IU,MODE,ILINE,NFEV,NFUN,ILIN1(3),ITRMAJ,KERROR
  real(chm_real) X(N),Y(N),G(N),P(N),ABS
  real(chm_real) LAMBDA,F,FOLD,GNORM,GNMOLD,XSNORM,RNORM
  real(chm_real) X1(N),X2(N),X3(N),G11(N),D1(N),G22(N),G33(N)
  real(chm_real) FTEST1(3),LAMB1(3)
  LOGICAL OKPG,OKIL6,OKSD,OKPN
  EXTERNAL CALFGH

  DO I=1,3

     IF(MODE  ==  0 .OR. MODE .EQ. 3 .OR. MODE .EQ. 1) RETURN

     IF(MODE  ==  2 .AND. GNORM/SQRT(DBLE(N))  <=  ONE &
          .AND. I  ==  3) GO TO 5000


     ! manipulate the coordinates to start always with the original ones.
     IF(MODE  ==  3) THEN
        IF(I  ==  3) THEN
           DO IU=1,N
              X(IU) = Y(IU)
           ENDDO
        ENDIF
        GO TO 921
     ENDIF  ! mode eq 3

     IF(I  ==  2 .OR. I .EQ. 3) THEN
        DO IU=1,N
           X(IU) = Y(IU)
        ENDDO
     ENDIF   ! i eq 1 or 2
921  CONTINUE

     ! Now the candidates are examined in the following order : 
     ! D, P and -G.

     LAMBDA = ONE
     ILINE = 0
     OKIL6 = .FALSE.
     OKPG = .FALSE.

     ! ------------------------------------------------------------------
     !     First candidate: D
     IF(I  ==  1) THEN 
        F = FOLD
        GNORM = GNMOLD
        KERROR = 0

        !     determine lambda.
        CALL MLINES(N,X,F,G,P,LAMBDA,ILINE,Y,NFEV,CALFGH,MODE, &
             OKSD,ITRMAJ,XSNORM,OKIL6,OKPG,GNORM,KERROR)

        IF(PRNLEV  >=  9) WRITE(OUTU,3041) F,LAMBDA,NFEV
3041    format('    d direction ',E14.8,' lambda =',F7.5,' NFEV = ',i4)

        !     Save the function, the coordinate and gradient vectors,and lambda.
        NFUN = NFUN + NFEV
        FTEST1(I) = F
        DO IU=1,N
           X1(IU) = X(IU)
           G11(IU) = G(IU)
        ENDDO
        LAMB1(1) = LAMBDA
        ILIN1(1) = ILINE
     ENDIF  !  I  ==  1 

     ! ------------------------------------------------------------------
     !     Second candidate: current  P direction. 
     IF(I  ==  2) THEN 
        F = FOLD
        GNORM = GNMOLD
        IF(MODE  ==  2) THEN
           DO IU=1,N
              P(IU) = D1(IU)
           ENDDO
        ENDIF
        OKPG = .TRUE.
        KERROR = 0

        !     determine lambda.
        CALL MLINES(N,X,F,G,P,LAMBDA,ILINE,Y,NFEV,CALFGH,MODE, &
             OKSD,ITRMAJ,XSNORM,OKIL6,OKPG,GNORM,KERROR)

        IF(PRNLEV  >=  9) WRITE(OUTU,3042) F,LAMBDA,NFEV
3042    format('    p direction ',E14.8,' lambda =',F7.5,' NFEV = ',i4)
        !   Save the function, the coordinate and gradient vectors,and lambda.
        NFUN = NFUN + NFEV
        FTEST1(I) = F
        DO IU=1,N
           X2(IU) = X(IU)
           G22(IU) = G(IU)
        ENDDO
        LAMB1(2) = LAMBDA
        ILIN1(2) = ILINE
     ENDIF  !  I  ==  2 

     ! ------------------------------------------------------------------
     !     Third candidate: -G direction. 
     IF(I  ==  3) THEN 
        F = FOLD
        GNORM = GNMOLD
        DO IU=1,N
           P(IU) = -G(IU)
        ENDDO
        KERROR = 0

        CALL MLINES(N,X,F,G,P,LAMBDA,ILINE,Y,NFEV,CALFGH,MODE, &
             OKSD,ITRMAJ,XSNORM,OKIL6,OKPG,GNORM,KERROR)

        IF(PRNLEV  >=  9) WRITE(OUTU,3040) F,LAMBDA,NFEV
3040    format('   -g direction ',E14.8,' lambda =',F7.5,' NFEV = ',i4)
        !   Save the function, the coordinate and gradient vectors,and lambda.
        NFUN = NFUN + NFEV
        FTEST1(I) = F
        DO IU=1,N
           X3(IU) = X(IU)
           G33(IU) = G(IU)
        ENDDO
        LAMB1(3) = LAMBDA
        ILIN1(3) = ILINE
     ENDIF  !  I  ==  3

5000 CONTINUE
  ENDDO

  ! Determine the best search direction and save the corresponding data.
  ! Note that when mode = 2 and gnorm <1, only two search candidates
  ! are considered,and we must consider the case where the energy passes
  ! from positive to negative energy.

  IF(FOLD  <  0.0D0) F=ABS(FTEST1(1))
  IF(FOLD  >=  0.0D0) F=FTEST1(1)

  IF(MODE == 2 .AND. GNORM/SQRT(DBLE(N)) <= ONE) THEN
     DO I=1,2
        IF(FOLD  <  0.0D0) F=MAX(ABS(FTEST1(I)),F)
        IF(FOLD  >=  0.0D0) F=MIN(FTEST1(I),F)
     ENDDO
  ELSE
     DO I=1,3
        IF(FOLD  <  0.0D0) F=MAX(ABS(FTEST1(I)),F)
        IF(FOLD  >=  0.0D0) F=MIN(FTEST1(I),F)
     ENDDO
  ENDIF

  OKPN = .FALSE.
555 CONTINUE

  IF(F  ==  ABS(FTEST1(1))) THEN
     DO I=1,N
        X(I) = X1(I)
        G(I) = G11(I)
     ENDDO
     LAMBDA = LAMB1(1)
     F = FTEST1(1)
     ILINE = ILIN1(1)
  ENDIF

  IF(F  ==  ABS(FTEST1(2))) THEN
     DO I=1,N
        X(I) = X2(I)
        G(I) = G22(I)
     ENDDO
     LAMBDA = LAMB1(2)
     F = FTEST1(2)
     ILINE = ILIN1(2)
  ENDIF

  IF(MODE == 2 .AND. GNORM/SQRT(DBLE(N)) <= ONE) GO TO 254 
  IF(F  ==  ABS(FTEST1(3))) THEN
     DO I=1,N
        X(I) = X3(I)
        G(I) = G33(I)
     ENDDO
     LAMBDA = LAMB1(3)
     F = FTEST1(3)
     ILINE = ILIN1(3)
  ENDIF
254 CONTINUE

  IF(OKPN) RETURN
  IF(F  /=  ABS(FTEST1(3)) .AND. F .NE. ABS(FTEST1(2)) &
       .AND. F  /=  ABS(FTEST1(1)) ) THEN
     F  = -F
     OKPN = .TRUE.
     GO TO 555
  ELSE
  ENDIF

  RETURN
END SUBROUTINE BESTSE
!-----------------------------------------------------------
SUBROUTINE SCADIR(N,MODE,OKMOD,OKIL6,OKSD,ITRMAJ, &
     GNORM,XSNORM,G,S,DGINIT,FTOL,DGTEST,OKPG)

  ! =====  CHARMM SEGMENT ##  28  ===========================
  !    date : 3 February 1993.
  !    Scale the search direction to reduce the number of linesearch 
  !    iterations, depending on TNPACK options and the value of MODE
  !    (MODE=0: Truncation criterion satisfied; MODE=2: Negative curvature
  !    after first CG iteration, MODE=3: maximum CG iterations exceeded).
  !
  !    The search direction s is scaled :
  !
  !    1.  If SD method used:
  !            at ITN = 1     s = -g/gnorm
  !            else           s = -g/gnorm*xsnorm
  !                           where xsnorm = x(k) - x(k-1)
  !
  !    2.  If TN method: 
  !        if mode = 0       no scaling
  !                  1       scaling depends on gnorm and whether 
  !                          the cosine of the angle between s and g
  !                          is inferior to 9.0d-3 ("TRUE" below):

  !                      (a) if True and gnorm>1.  s=-g/gnorm*xsnorm
  !                      (b) if True and gnorm>1.  s = - g
  !                      (c) if False and gnorm>0.1 s = s/snorm
  !                      (d) if False and gnorm<0.1 no scaling
  !
  !                   2       s = s/snorm
  !
  !                   3       no scaling

  ! Note that the definition of the following logical variables:
  !  -  okil6  no scaling
  !  -  oksd   SD method active

  use stream
  implicit none

  LOGICAL OKMOD,OKIL6,OKSD,OKPG
  INTEGER N,MODE,IU,ITRMAJ,I
  real(chm_real) :: G(N),S(N),DGINIT,FTOL,DGTEST,PNORM,COSPHI,MIN6=9.0E-3_chm_real
  real(chm_real) GST,GNORM,XSNORM
  real(chm_real) DNRM2,DDOT
  EXTERNAL DNRM2,DDOT

  OKMOD = .TRUE.  !! DUMMY LOGICAL VARIABLE
  IF(OKIL6) GO TO 7233

  IF(OKSD) THEN
     IF(ITRMAJ  /=  1) THEN
        DO IU=1,N
           S(IU) = - G(IU)/GNORM*XSNORM   ! SD and ITN =1
        ENDDO
#if KEY_GAMESS==1
        DGINIT = DDOT(int8(N),G,1_8,S,1_8)
#else
        DGINIT = DDOT(N,G,1,S,1)
#endif
        DGTEST = FTOL*DGINIT
        GO TO 7233
     ELSE
        DO IU=1,N
           S(IU) = - G(IU)/GNORM          ! SD and ITN >1
        ENDDO
#if KEY_GAMESS==1
        DGINIT = DDOT(int8(N),G,1_8,S,1_8)
#else
        DGINIT = DDOT(N,G,1,S,1)
#endif
        DGTEST = FTOL*DGINIT
        GO TO 7233
     ENDIF
  ENDIF

  PNORM = DNRM2(N,S,1)
  IF(OKMOD) THEN

     IF(MODE  ==  2) THEN
        DO I=1,N
           S(I) = S(I)/PNORM !  MODE = 2
        ENDDO
        DGINIT = DGINIT/PNORM 
        DGTEST = FTOL*DGINIT
     ENDIF
     IF(MODE  ==  1) THEN
        !          Compute the cosine angle.
        COSPHI = DGINIT /(GNORM*PNORM)
        IF((ABS(COSPHI) <= MIN6).OR.(DGINIT > 0.0)  ) THEN
           GST = GNORM/SQRT(DBLE(N))
           DO IU=1,N
              IF(GST > 1.0D0) S(IU) = -G(IU)/GNORM*XSNORM ! see .a
              IF(GST <= 1.0D0) S(IU) = -G(IU) ! see .b
           ENDDO
           !              Do some printings if necessary
           IF(PRNLEV  >=  9) THEN
              IF(GST  >  1.0D0) WRITE(OUTU,*) &
                   '          the direction is reset to  -g/!g!*!xs!'
              IF(GST  <=  1.0D0) WRITE(OUTU,*) &
                   '          the direction is reset to  -g'
           ENDIF
#if KEY_GAMESS==1
           DGINIT = DDOT(int8(N),G,1_8,S,1_8)
#else
           DGINIT = DDOT(N,G,1,S,1)
#endif
           DGTEST = FTOL*DGINIT
           GO TO 7233 
        ELSE
        ENDIF
        DO I=1,N
           IF(GNORM/SQRT(DBLE(N))  >=  0.1) S(I)=S(I)/PNORM ! see .c
           IF(GNORM/SQRT(DBLE(N))  <  0.1) GO TO 1717      ! see .d
        ENDDO
1717    CONTINUE
        DGINIT = DGINIT/PNORM
        IF(GNORM/SQRT(DBLE(N))  <  0.1) DGINIT=DGINIT*PNORM 
        DGTEST = FTOL*DGINIT
     ENDIF                   !! mode 1 

  ENDIF                     !! okmod 

7233 CONTINUE 
  RETURN 
END SUBROUTINE SCADIR
!---------------------------------------------------------------
SUBROUTINE HDPNEW(ITR,N,D,HD,ITRMAJ)

  ! =====  CHARMM SEGMENT ##  29 ===========================
  !      Compute the Hessian/vector d product from
  !      the pattern of the Hessian matrix.
  !      date : 22 February 1993.

  use contrl
  use number
  use euler
  implicit none

  INTEGER N,I,K,J,JI,NUM,ITR,ITRMAJ
  real(chm_real) D(N),HD(N)

  !yw...Use HEAP space for IAHES, JAHES and AHES, 12-Aug-95
  IF (ITR  ==  0) CALL MTOA1(N,IDDF,IAHES,JAHES,AHES,NZHE)

  IF (IAHES(N+1) > NZHE+1) CALL WRNDIE(-1, &
       '<HDPNEW>',' The number of nonzeros in H is too large.')

  K=0
  DO I=1,N
     HD(I) = ZERO
  ENDDO

  DO I=1,N
     K=K+1
     HD(I) = HD(I) + AHES(K) * D(I)
     NUM = IAHES(I+1) - IAHES(I) - 1
     IF(I  ==  N) GO TO 100
     DO JI=1,NUM
        K=K+1
        J = JAHES(K)
        HD(I) = HD(I) + AHES(K) * D(J) ! upper triangle
        HD(J) = HD(J) + AHES(K) * D(I) ! lower triangle
     ENDDO
  ENDDO
100 CONTINUE 
  !
  RETURN 
END SUBROUTINE HDPNEW
#endif /* (tnpack)*/

end module tnpack

#if KEY_GAMESS==0 && KEY_GAMUS==0 /*nogamess2*/
!***********************************************************************
FUNCTION DNRM2 (N,DX,INCX)
  use chm_kinds
  use number
  implicit none
  !
  real(chm_real) :: dnrm2
  INTEGER NEXT
  real(chm_real) DX(*), HITEST, SUM, XMAX
  INTEGER NN,INCX,I,J,N
  ! --------------------------------------------------------
  !  EUCLIDEAN NORM OF THE N-VECTOR STORED IN DX() WITH STORAGE
  !  INCREMENT INCX .
  !  IF    N  <=  0   RETURN WITH RESULT = 0.
  !  IF    N  >=  1   THEN INCX MUST BE .GE. 1
  !        C.L.LAWSON, 1978 JAN 08
  !  FOUR PHASE METHOD     USING TWO BUILT-IN CONSTANTS THAT ARE
  !  HOPEFULLY APPLICABLE TO ALL MACHINES.
  !      CUTLO = MAXIMUM OF  DSQRT(U/EPS)  OVER ALL KNOWN MACHINES.
  !      CUTHI = MINIMUM OF  DSQRT(V)      OVER ALL KNOWN MACHINES.
  !  WHERE
  !      EPS = SMALLEST NO. SUCH THAT EPS + 1.  >  1.
  !      U   = SMALLEST POSITIVE NO.   (UNDERFLOW LIMIT)
  !      V   = LARGEST  NO.            (OVERFLOW  LIMIT)
  !
  !  BRIEF OUTLINE OF ALGORITHM..
  !  PHASE 1    SCANS ZERO COMPONENTS.
  !  MOVE TO PHASE 2 WHEN A COMPONENT IS NONZERO AND  <=  CUTLO
  !  MOVE TO PHASE 3 WHEN A COMPONENT IS  >  CUTLO
  !  MOVE TO PHASE 4 WHEN A COMPONENT IS  >=  CUTHI/M
  !                  WHERE M = N FOR X() REAL AND M = 2*N FOR COMPLEX.
  !  VALUES FOR CUTLO AND CUTHI..
  !        FROM THE ENVIRONMENTAL PARAMETERS LISTED IN THE IMSL CONVERTER
  !        DOCUMENT THE LIMITING VALUES ARE AS FOLLOWS..
  !    CUTLO, S.P.   U/EPS = 2**(-102) FOR  HONEYWELL.  CLOSE SECONDS ARE
  !                UNIVAC AND DEC AT 2**(-103)
  !                THUS CUTLO = 2**(-51) = 4.44089E-16
  !    CUTHI, S.P.   V = 2**127 FOR UNIVAC, HONEYWELL, AND DEC.
  !                THUS CUTHI = 2**(63.5) = 1.30438E19
  !    CUTLO, D.P.   U/EPS = 2**(-67) FOR HONEYWELL AND DEC.
  !                THUS CUTLO = 2**(-33.5) = 8.23181D-11
  !    CUTHI, D.P.   SAME AS S.P.  CUTHI = 1.30438D19
  !  DATA CUTLO, CUTHI / 8.232D-11, 1.304D19 /
  !  DATA CUTLO, CUTHI / 4.441E-16, 1.304E19 /
  !
  real(chm_real) :: CUTLO=8.232e-11_chm_real, CUTHI=1.304e+19_chm_real
  !
  !_mfc changing assigned/computed goto to select case
  integer casenum

  IF (N  <  0) then
     DNRM2  = ZERO
     return
  endif
  !     
  !                         ASSIGN 30 TO NEXT
  casenum=30
  SUM = ZERO
  NN = N * INCX
  ! ------------------
  ! BEGIN MAIN LOOP
  ! ------------------
  I = 1
20 continue
  !   20    GO TO NEXT,(30, 50, 70, 110)
  !      phase: select case (casenum)
  !         case(30)
  IF(CASENUM == 30)THEN
     IF (DABS(DX(I))  <=  CUTLO) then   !GO TO 85
        casenum=50
        XMAX = ZERO
        ! ------------------
        ! PHASE 1.  SUM IS ZERO
        ! ------------------
        IF (DX(I)  ==  ZERO) GO TO 200
        ! ------------------
        ! PREPARE FOR PHASE 2.
        ! ------------------
        casenum=70
        XMAX = DABS(DX(I))
        SUM = SUM + (DX(I)/XMAX)**2
        GO TO 200
     endif

     !         case(50)
  ELSE IF(CASENUM == 50)THEN
     ! ------------------
     ! PHASE 1.  SUM IS ZERO
     ! ------------------
     IF (DX(I)  ==  ZERO) GO TO 200
     IF (DABS(DX(I))  <=  CUTLO) then   ! GO TO 85
        ! ------------------
        ! PREPARE FOR PHASE 2.
        ! ------------------
        casenum=70
        XMAX = DABS(DX(I))
        SUM = SUM + (DX(I)/XMAX)**2
        GO TO 200
     endif

     !         case(70)
  ELSE IF(CASENUM == 70)THEN
     ! ------------------
     ! SUM IS SMALL.
     ! SCALE TO AVOID DESTRUCTIVE UNDERFLOW.
     ! ------------------
     IF (DABS(DX(I))  >  CUTLO) then
        ! ------------------
        ! PREPARE FOR PHASE 3.
        ! ------------------
        SUM = (SUM * XMAX) * XMAX
     else
        IF (DABS(DX(I))  <=  XMAX) then
           SUM = SUM + (DX(I)/XMAX)**2
        else
           SUM = ONE + SUM * (XMAX / DX(I))**2
           XMAX = DABS(DX(I))
        endif
        GO TO 200
     endif

     ! ------------------
     ! COMMON CODE FOR PHASES 2 AND 4.
     ! IN PHASE 4 SUM IS LARGE.  SCALE TO AVOID OVERFLOW.
     ! ------------------
     !         case(110)
  ELSE IF(CASENUM == 110)THEN
     IF (DABS(DX(I))  <=  XMAX) then
        SUM = SUM + (DX(I)/XMAX)**2
     else
        SUM = ONE + SUM * (XMAX / DX(I))**2
        XMAX = DABS(DX(I))
     endif
     GO TO 200
     !      end select phase
  ENDIF
  ! ------------------
  ! FOR REAL OR D.P. SET HITEST = CUTHI/N
  ! FOR COMPLEX      SET HITEST = CUTHI/(2*N)
  ! ------------------
85 HITEST = CUTHI/N
  ! ------------------
  ! PHASE 3.  SUM IS MID-RANGE.  NO SCALING.
  ! ------------------
  DO J =I,NN,INCX
     IF (DABS(DX(J))  >=  HITEST) then
        !         ------------------
        !         PREPARE FOR PHASE 4.
        !         ------------------
        I = J
        casenum=110
        SUM = (SUM / DX(I)) / DX(I)
        XMAX = DABS(DX(I))
        SUM = SUM + (DX(I)/XMAX)**2
        GO TO 200
     endif
     SUM = SUM + DX(J)**2
     DNRM2 = DSQRT(SUM)
  enddo
  return
  !
200 CONTINUE
  I = I + INCX
  IF (I  <=  NN) GO TO 20
  ! ------------------
  ! END OF MAIN LOOP.
  ! COMPUTE SQUARE ROOT AND ADJUST FOR SCALING.
  ! ------------------
  DNRM2 = XMAX * DSQRT(SUM)
  RETURN
END FUNCTION DNRM2
#endif /* (nogamess2)*/

