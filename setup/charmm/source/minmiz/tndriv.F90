#if KEY_TNPACK==0 /*tnpack*/
SUBROUTINE TNDRIV(COMLYN,COMLEN,QSTART)
  CALL WRNDIE(-3,'<TNDRIV>','TNPACK is NOT compiled.')
  RETURN
end SUBROUTINE TNDRIV
#else /* (tnpack)*/

SUBROUTINE TNDRIV(COMLYN,COMLEN,QSTART)
  !-----------------------------------------------------------------------
  !     TNDRIV, written by B. Brooks and P. Derreumaux,
  !     sets up and performs Truncated-Newton minimization
  !     by the program TNPACK of Schlick & Fogelson.
  !     Complete details of the minimizer are given in the
  !     two ACM Transactions on Mathematical Software papers,
  !     Volume 18, pages 46-111, 1992, by T. Schlick and
  !     A. Fogelson, entitled:
  !     "TNPACK --- A Truncated Newton Minimization
  !     Package for Large-Scale Problems":
  !     I. Algorithm ans usage,
  !     and
  !     II. Implementation Examples.
  !
  !    Algorithmic questions can be addressed to T. Schlick,
  !    Courant Institute of Mathematical Sciences,
  !    251 Mercer Street, New York University,
  !    New York,  NY 10012,
  !    or via e-mail: schlick@acfclu.nyu.edu
  !
  !    Segments specific to CHARMM implementations are
  !    bracketed separately below.
  !
#if KEY_FLUCQ==1
  use flucqm, only: fqseln       
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use tnpack
  use contrl
  use coord
  use egrad
  use energym
  use psf
  use stream
  use string
  use euler
#if KEY_TSM==1
  use tsms_mod
  use tsmh
#endif 
#if KEY_FLUCQ==1
  use flucq
#endif 
#if KEY_DHDGB==1
  use dhdgb,only:totals
#endif
  use memory
  implicit none
  !
  !     Passed variables.
  !
  real(chm_real),allocatable,dimension(:) :: VARB
  real(chm_real),allocatable,dimension(:) :: GRAD
  real(chm_real),allocatable,dimension(:) :: VREF
  real(chm_real),allocatable,dimension(:) :: W
  integer,allocatable,dimension(:) :: IW
  INTEGER       COMLEN
  CHARACTER(len=*) COMLYN
  LOGICAL QSTART
  !
  !     Local variables.
  !
  INTEGER  CONVRG, I, NSTEP, NVAR, NCALLS, NCGCYC
  real(chm_real) ::  STEP, TOLGRD=1.0e-08_chm_real, TOLFUN, TOLSTP
  !
  INTEGER OPLIST(20),INFORM
  real(chm_real) PLIST(20)
  !
  !     Pointers.
  !
  real(chm_real) FUNC
  INTEGER LW, NW, NZ, LIW, NAT3, NTOT
  !
  EXTERNAL CALFGH, CALPAT, CALHDP
  SAVE
  !
#if KEY_DHDGB==1
  real(chm_real) DUM_FHDGB(TOTALS)
#endif
  ! =========  SEGMENT  ## 1  ============================
  LOGICAL QPRECON,QIORDER,QPERMUTM
  LOGICAL QHDPROD,QTRUNCA,QSEARCH,QSCHED,QPRECI
  ! =====================================================

  !     Do some initialisation and parse the command line.

  CONVRG = 0
  if(allocated(iddf)) then
     call wrndie(-3,'<TNDRIV>','IDDF already allocated.')
  end if
  ! =========== SEGMENT ## 2 =============================
  !   --  The following setup defines minimization parameters
  !       from the commands defined in the CHARMM input file.
  !       Suggested default parameters are set for:
  !       preconditioning, preconditioner reordering,
  !       truncation tests (residual and quadratic options),
  !       Hessian/vector multiplication,
  !       termination tests (convergence criteria for TOLGRD
  !       and function accuracy, TOLFUN), and
  !       use of a scheduling subroutine, and
  !       a subroutine that chooses the best search direction
  !       in case of negative curvature.

  IF (JCALDY  >  3) GO TO 6000
  NPRINT = GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
  NSTEP  = GTRMI(COMLYN,COMLEN,'NSTE',100)
  NCGCYC = GTRMI(COMLYN,COMLEN,'NCGC',100)
  STEP   = GTRMF(COMLYN,COMLEN,'STEP',PT02)
  TOLFUN = GTRMF(COMLYN,COMLEN,'TOLE',ZERO)
  TOLGRD = GTRMF(COMLYN,COMLEN,'TOLG',TOLGRD)
  TOLSTP = GTRMF(COMLYN,COMLEN,'TOLS',ZERO)
  QPRECON = (INDXA(COMLYN,COMLEN,'PREC') > 0)
  QPRECON = (INDXA(COMLYN,COMLEN,'NOPR') <= 0)
  QHDPROD = (INDXA(COMLYN,COMLEN,'USER') <= 0)
  QHDPROD = (INDXA(COMLYN,COMLEN,'OURH') > 0)
  QTRUNCA = (INDXA(COMLYN,COMLEN,'REST') <= 0)
  QTRUNCA = (INDXA(COMLYN,COMLEN,'QUAT') > 0)
  QSCHED  = (INDXA(COMLYN,COMLEN,'NOSC') <= 0)
  QSCHED  = (INDXA(COMLYN,COMLEN,'SCHE') > 0)
  QSEARCH  = (INDXA(COMLYN,COMLEN,'DEFS') <= 0)
  QSEARCH  = (INDXA(COMLYN,COMLEN,'SEAR') > 0)
  QPRECI  = (INDXA(COMLYN,COMLEN,'LOWP') <= 0)
  QPRECI  = (INDXA(COMLYN,COMLEN,'HIGP') > 0)
  QIORDER  = (INDXA(COMLYN,COMLEN,'IORD') > 0)
  QIORDER  = (INDXA(COMLYN,COMLEN,'NOOR') <= 0)
  QPERMUTM  = (INDXA(COMLYN,COMLEN,'PERM') > 0)
  QPERMUTM  = (INDXA(COMLYN,COMLEN,'NOPM') <= 0)
6000 CONTINUE
  ! ========================================================

  IF (NPRINT  <  0 .OR. NPRINT  >  NSTEP) NPRINT = 0
  IF (PRNLEV < 2) NPRINT=0
  !
  CALL XTRANE(COMLYN,COMLEN,'STEEPD')
  IF (COMLEN  >  0) CALL DIEWRN(-2)
  !
  !     Determine the number of variables and do some printing.
  !
#if KEY_TSM==1
  CALL CALCNVAR(QTSM,BACKLS,NVAR)
#else /**/
  CALL CALCNVAR(.FALSE.,(/0/),NVAR)
#endif 
  !
  IF (JCALDY  ==  0) THEN  !! M.D.
     IF (NPRINT  /=  0) THEN
        WRITE (OUTU,'(/)')
        WRITE (OUTU,'(A)') &
             ' STEEPD> An energy minimization has been requested.'
        WRITE (OUTU,'(/,A,I12,A,I12,/,2(A,F12.7,A,F12.7,/))') &
             ' NSTEP  = ',NSTEP, '   NPRINT = ',NPRINT, &
             ' STEP   = ',STEP,  '   TOLFUN = ',TOLFUN, &
             ' TOLGRD = ',TOLGRD,'   TOLSTP = ',TOLSTP
     ENDIF
  ENDIF
  !
  !     Allocate storage.
  !
  call chmalloc('tndriv.src','TNDRIV','VARB',NVAR,crl=VARB)
  call chmalloc('tndriv.src','TNDRIV','GRAD',NVAR,crl=GRAD)
  call chmalloc('tndriv.src','TNDRIV','VREF',NVAR,crl=VREF)
  call chmalloc('tndriv.src','TNDRIV','IAHES',SIZE1,intg=IAHES)
  call chmalloc('tndriv.src','TNDRIV','JAHES',NZHE,intg=JAHES)
  call chmalloc('tndriv.src','TNDRIV','AHES',NZHE,crl=AHES)
  !
  !     Fill the variable array with the coordinates to be optimised.
  !
  CALL GETVR1(.TRUE.,NATOM,VARB,IMOVE,X,Y,Z,.FALSE., &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/), & 
#endif
       'NONE',(/ZERO/),(/ZERO/), &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &  
#endif
       .FALSE.,(/ZERO/),(/0/)  &
#if KEY_DHDGB==1
!AP/MF
     ,.FALSE.,(/ZERO/),TOTALS &
#endif
   )
  !
  IF(QSTART) THEN
     !
     !========  SEGMENT ## 3 ========================================
     IF(JCALDY  <  3) THEN
        CALL SETLIS(NVAR,OPLIST,PLIST,INFORM)
     ELSE
        INFORM = - 10
        GO TO 7000
     ENDIF

     OPLIST(1) = 0
     IF(QPRECON) OPLIST(1) = 1
     OPLIST(2) = 0
     OPLIST(3) = 0
     OPLIST(4) = NSTEP
     OPLIST(5) = NCGCYC
     OPLIST(6) = NSTEP*NCGCYC
     OPLIST(7)= 0
     IF(QIORDER) OPLIST(7) = 1
     OPLIST(9)= 0
     IF(QPERMUTM) OPLIST(9) = 1
     OPLIST(10)= 0
     IF(QHDPROD) OPLIST(10) = 1
     OPLIST(13) = 0
     IF(QTRUNCA) OPLIST(13) = 1

     OPLIST(11)= OUTU
     OPLIST(12) = 2
     OPLIST(15) = 0
     IF(QSCHED) OPLIST(15) = 1
     OPLIST(16) = 0
     IF(QSEARCH) OPLIST(16) = 1
     !
     !        Parameters for Dynamics (QEULER)
     IF(QEULER) OPLIST(5) = 20
     IF(QEULER) OPLIST(4) = 400
     IF(QEULER) TOLGRD = 1.0E-7
     !
     PLIST(1)  = TOLGRD

     IF(QPRECI) THEN
        PLIST(1) = PLIST(1)*1.0D-8 !cutnb 8 1.0d-10
        PLIST(2) = PLIST(2)*1.0D-4 ! 1.0d-4
     ELSE
     ENDIF
     ! =========================================================

     PLIST(3)  = 0.5D0
     PLIST(4)  = 0.50D0
     PLIST(5)  = 1.0D-10
     PLIST(6)  = 1.0D-04
     PLIST(7)  = 0.9D0
     !
     QSTART=.FALSE.
  ELSE
     OPLIST(9) = 0
  ENDIF
  !
  ! ========= SEGMENT ## 4 =========================
  ! Estimate the number of nonzeros for the preconditioner
  ! M by approximating some average number of neighbors
  ! per atom. For small molecules, use the total number of
  ! entries in a triangular part of the symmetric matrix
  ! (diagonals included).
7000 CONTINUE
  NZ=30*NVAR+16
  NAT3 = NATOM*3
  IF(NAT3  <=  300) NZ = (NAT3*(NAT3+1))/2
  ! =================================================
  LW = 9*NVAR + NZ + 4*MAX(NVAR,NZ) + 1
  ! =========  SEGMENT  ## 5 ========================
  ! Increase the size of W. Seven vectors:
  ! D1(N),X1(N),X2(N),X3(n),G11(N),G22(N),G33(N)
  ! are added to the original list,
  LW = LW + 7*NVAR
  ! skip allocation for further steps of dynamics
  IF(QEULER .AND. JCALDY  >  1) GO TO 9232
  ! ==================================================
  call chmalloc('tndriv.src','TNDRIV','W',LW,crl=W)
  W(1:LW) = ZERO
  LIW = 7*NVAR + NZ + 4*MAX(NVAR,NZ) +1
  call chmalloc('tndriv.src','TNDRIV','IW',LIW,intg=IW)
9232 CONTINUE
  !
  CALL TNMIN(NVAR,VARB,FUNC,GRAD, &
       OPLIST,PLIST,INFORM, &
       NZ,W,LW,IW,LIW, &
       CALFGH,CALPAT,CALHDP)
  !
  !     Print some results.
  !
  IF (NPRINT  /=  0) THEN
     !
     !       Test for convergence.
     !
     IF (CONVRG  ==  1) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' STEEPD> Minimization exiting with step tolerance (', &
             TOLSTP,') satisfied.'
     ELSE IF (CONVRG  ==  2) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             ' STEEPD> Minimization exiting with gradient tolerance (', &
             TOLGRD,') satisfied.'
     ELSE IF (CONVRG  ==  3) THEN
        WRITE (OUTU,'(/,A,F10.7,A,/)') &
             TOLFUN,') satisfied.'
     ELSE IF (CONVRG  ==  4) THEN
        WRITE (OUTU,'(/,A,I10,A,/)') &
             ' STEEPD> Minimization exiting with number of steps limit (', &
             NSTEP,') exceeded.'
     ENDIF
     !yw   11-May-91 Save the last step characteristics
     !       EPROP(PJNK1)=NCALLS
     !       EPROP(PJNK2)=STEP
     ! If QEULER true skip next printing
     IF(.NOT.QEULER) THEN
        IF (NCALLS < NSTEP) THEN
           CALL PRINTE(OUTU, EPROP, ETERM, 'STPD', 'MIN', .TRUE., &
                NCALLS, ZERO, STEP, .TRUE.)
        ENDIF
     ENDIF                   ! QEULER
     !
  ENDIF
  !
  !     Fill the coordinate arrays with the optimised variables.
  !
  CALL PUTVR1(.TRUE.,NATOM,VARB,IMOVE,X,Y,Z, &
#if KEY_CHEQ==1
       .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), & 
#endif
       .FALSE.,'NONE',(/ZERO/),(/ZERO/),.TRUE., &
#if KEY_FLUCQ==1
       QFLUC,CG,FQSELN, &                    
#endif
       .FALSE.,(/ZERO/),(/0/) &
#if KEY_DHDGB==1
       ,QFHDGB=.FALSE.,SDEF=DUM_FHDGB, TOTALS=TOTALS &
#endif
  )

  !     Clear up any temporary storage space.
  call chmdealloc('tndriv.src','TNDRIV','GRAD',NVAR,crl=GRAD)
  call chmdealloc('tndriv.src','TNDRIV','VARB',NVAR,crl=VARB)
  call chmdealloc('tndriv.src','TNDRIV','VREF',NVAR,crl=VREF)

  ! =========  SEGMENT  ## 5B ========================
  ! do not clear if M.D
  IF(.NOT.QEULER) THEN
     IF(allocated (iddf)) THEN
        NAT3=NATOM*3
        NTOT=(NAT3*(NAT3+1))/2
        call chmdealloc('tndriv.src','TNDRIV','IDDF',NTOT,crl=IDDF)
     ENDIF
     call chmdealloc('tndriv.src','TNDRIV','IAHES',SIZE1,intg=IAHES)
     call chmdealloc('tndriv.src','TNDRIV','JAHES',NZHE,intg=JAHES)
     call chmdealloc('tndriv.src','TNDRIV','AHES',NZHE,crl=AHES)
     call chmdealloc('tndriv.src','TNDRIV','W',LW,crl=W)
     call chmdealloc('tndriv.src','TNDRIV','IW',LIW,intg=IW)
  ENDIF
  !
  RETURN
END SUBROUTINE TNDRIV
!***********************************************************************
SUBROUTINE CALFGH(N,X,F,G,A,IA,JA,NZ,NOUT)
  !
  !     CALFGH is the routine required by TNPACK to evaluate
  !     the function and derivatives according to NOUT.
  !     See original TNPACK documentation for details.
  !
  !     The output at X is:
  !
  !     NOUT = 0  F only
  !     NOUT = 1  F,G
  !     NOUT = 2  F,G,H
  !     NOUT = 3  F,G,H
  !     NOUT = 4      H
  !     NOUT = 5      H,M (option 1)
  !     NOUT = 6      H,M (option 1)
  !     NOUT = 10       M (option 2)
  !
  !     where options 1, 2 are:
  !     1: IA and JA are known for M, and all the second derivatives
  !        of the potential (with the exception of the hydrogen
  !        bonding term) are included in the matrix positions that arise
  !        from the local energy terms.

  !     2: IA and JA are computed, only the local potential terms
  !        contribute to M.

  !        Note that IA and JA are the one dimensional arrays that store
  !        M in a sparse format, see routine MTOA1.
  use chm_kinds
  use number
  !
  use contrl
  use stream
  use egrad
  use energym
  use euler
  implicit none
  !
  INTEGER N,NOUT,IA(N+1),JA(*),NZ
  real(chm_real) F,X(N),G(N),A(*),F1
  !
  INTEGER :: ERSTAT,ICALL,ISTEP=0,I
  LOGICAL QLOCAL
  SAVE
  !
  ! =======   SEGMENT ##6 ============================
  real(chm_real) :: X0(1)
  CHARACTER(len=80) STRI
  ! ==================================================

  ICALL=0
  IF(NOUT < 0) RETURN
  ICALL=1
  ISTEP=ISTEP+1

  IF(NOUT ==  0) THEN
     CALL EGRAD1(N,X,X0,F,A,ISTEP,ICALL,ERSTAT)
     GO TO 900
  ENDIF

  IF(NOUT  ==  10) GO TO 600
  GO TO (100,200,300,400,500,500) NOUT


  ! NOUT = 1
100 CALL EGRAD1(N,X,X0,F,G,ISTEP,ICALL,ERSTAT)
  GO TO 900

  ! NOUT = 2
200 QLOCAL=.FALSE.
  QLOC1 = QLOCAL
  CALL EGRAD2(N,X,F,G,QLOCAL,ISTEP,ICALL,ERSTAT)
  GO TO 900

  ! NOUT = 3
300 QLOCAL=.FALSE.
  QLOC1 = QLOCAL
  CALL EGRAD2(N,X,F,G,QLOCAL,ISTEP,ICALL,ERSTAT)
  GO TO 900

  ! NOUT = 4
400 QLOCAL=.FALSE.
  QLOC1 = QLOCAL
  CALL EGRAD2(N,X,F1,A,QLOCAL,ISTEP,ICALL,ERSTAT)
  GO TO 900

  ! ================= SEGMENT ## 7 ===================================
  ! NOUT = 5,6
500 QLOCAL=.FALSE.
  QLOC1 = QLOCAL
  CALL EGRAD2(N,X,F1,A,QLOCAL,ISTEP,ICALL,ERSTAT)
  CALL MTOA2(N,IDDF, IA,JA,A,NZ)
  GO TO 900
  ! =================================================================

  ! NOUT = 10
600 QLOCAL=.TRUE.
  QLOC1 = QLOCAL
  !     prnlev=5
  !     call wrttim(stri)
  CALL EGRAD2(N,X,F1,A,QLOCAL,ISTEP,ICALL,ERSTAT)
  !     call wrttim(stri)
  CALL MTOA1(N, IDDF, IA,JA,A,NZ)
900 CONTINUE
  IF(PRNLEV > 9) THEN
     WRITE(OUTU,45) 'X IN CALFGH:', (X(I),I=1,N)
     WRITE(OUTU,45) 'G IN CALFGH:', (G(I),I=1,N)
45   FORMAT(A/,(3F10.5))
  ENDIF
  IF(PRNLEV > 7) THEN
     CALL PRINTE(OUTU, EPROP, ETERM, 'TN', 'MIN', &
          .TRUE., NOUT, ZERO, ZERO, .TRUE.)
  ENDIF
  !
  RETURN
END SUBROUTINE CALFGH
!***********************************************************************
SUBROUTINE CALHDP(N,D,HD,X,G)
  !
  !     CALHDP is the routine required by TNPACK to calculate
  !     the Hessian/vector product HD.
  !     See original TNPACK documentation for details.
  !
  !      D  - input vector
  !      HD - resultant product vector
  !      X  - current coordinate
  !      G  - gradient
  !
  !
  use chm_kinds
  use contrl
  implicit none
  !
  INTEGER N
  real(chm_real) D(N),HD(N),X(N),G(N)
  !
  INTEGER NATOM

  NATOM=N/3

  CALL RALEG2(D,HD,NATOM,IDDF)
  RETURN
END SUBROUTINE CALHDP
!***********************************************************************
SUBROUTINE CALPAT(N,X,A,IA,JA,NZ)
  !
  !     CALHDP is the routine required by TNPACK to calculate
  !     the sparsity pattern of the preconditioner M.
  !     See original TNPACK documentation for details.
  !
  use chm_kinds
  use number
  use stream
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: GRAD
  INTEGER N,IA(N+1),JA(*),NZ
  real(chm_real) X(N),A(*)
  !
  INTEGER I,J,LENA,NOUT
  real(chm_real) FY
  !
  !
  ! ======  SEGMENT ##  8 ====================================
  A(1:n) = X(1:n)
  ! ==========================================================
  !
  call chmalloc('tndriv.src','CALPAT','GRAD',N,crl=GRAD)
  NOUT = 10
  CALL CALFGH(N,A,FY,GRAD,A,IA,JA,NZ,NOUT)
  call chmdealloc('tndriv.src','CALPAT','GRAD',N,crl=GRAD)
  IF(PRNLEV > 9) THEN
     WRITE(OUTU,*) 'IA ', (IA(I), I=1,N+1)
     LENA = IA(N+1) - 1
     WRITE(OUTU,*) 'LENA=', LENA
     WRITE(OUTU,*) 'JA ', (JA(I), I=1,LENA)
     WRITE(OUTU,*) 'A  ', ( A(I), I=1,LENA)
  ENDIF
  !
  RETURN
END SUBROUTINE CALPAT


!***********************************************************************
SUBROUTINE MTOA1(N, DDF, IA, JA, A, NZ)
  !
  ! Convert the storage form of a sparse symmetric matrix from DP (Double
  ! Precision) arrays {MATD,MATL} to the group {A,IA,JA} by reading
  ! the LOWER-triangle and storing all diagonals and nonzero off-diagonals.
  ! In the present format, MATD holds the diagonals, and MATL contains the
  ! lower triangle of H, stored row-by-row. The new format is required for
  ! the YSMP (Yale Sparse Matrix Package). Array  A is DP and will hold
  ! all diagonals and nonzero entries of the UPPER-triangle. Arrays IA and
  ! JA will be the associated row and columm pointers, respectively. IA(I)
  ! for rows I=1,n  will specify the location in A of the diag. element in
  ! row I. IA(n+1) will be set to the location of the last entry in A
  ! PLUS 1. JA(K) for K=1,LENA (LENA = IA(n+1)-1) will specify the column
  ! for element A(K).
  ! MTOA1 was adapted from the TNPACK package.

  use chm_kinds
  use stream
  implicit none


  real(chm_real) DDF(*), A(*)
  INTEGER N, IA(*), JA(*), NZ
  !
  real(chm_real) EPS, ELT
  INTEGER I,J,K,IPT,LENA
  !
  EPS=1.0D-10
  !
  ! Go through all the elements in MATD and MATL: read the upper-triangle
  ! column-by-column (same as lower traingle row-by-row) starting at the
  ! diag. of each row I. For off-diag. (I,J) elements, compte INDEX, its
  ! location in MATL, and then store it only if it is greater than EPS.
  !

  K=0
  IPT=0
  DO J=1,N
     DO I=J,N
        IF (I  ==  J) THEN
           K=K+1
           IPT=IPT+1
           IA(I)=K
           JA(K)=I
           A(K)=DDF(IPT)
        ELSE
           IPT=IPT+1
           ELT=DDF(IPT)
           IF (ABS(ELT)  >  EPS) THEN
              K=K+1
              JA(K)=I
              A(K)=ELT
           ENDIF
        ENDIF
     ENDDO
  ENDDO
100 CONTINUE
  IA(N+1)=K+1
  IF(PRNLEV > 9) THEN
     WRITE(OUTU,*) 'IA ', (IA(I), I=1,N+1)
     LENA = IA(N+1) - 1
     WRITE(OUTU,*) 'LENA=', LENA
     WRITE(OUTU,*) 'JA ', (JA(I), I=1,LENA)
     WRITE(OUTU,*) 'A  ', ( A(I), I=1,LENA)
  ENDIF
  RETURN
END SUBROUTINE MTOA1
! **********************************************************************
SUBROUTINE MTOA2(N, DDF, IA, JA, A, NZ)

  ! Convert the storage form of a sparse symmetric matrix from DP
  ! arrays {MATD,MATL} to DP {A} when the nonzero pattern is known and
  ! available in {IA,JA}. The {A,IA,JA} triplet is used in the YSMP.
  ! In the present format, MATD holds the diag. elts., and MATL contains
  ! the lower triangle, stored row-by-row. Array A will hold all
  ! diagonals and the nonzero entries of the UPPER-triangle.
  ! MTOA2 was adapted from the TNPACK package.
  !
  use chm_kinds
  use stream
  implicit none


  real(chm_real)  DDF(*), A(*)
  INTEGER N, IA(*), JA(*), NZ
  !
  INTEGER K,INDEX,NROWI,I,J,JI,NZX,LENA
  real(chm_real)  EPS

  EPS=1.0D-10

  ! Go through the pointer arrays IA and JA, and for each K find the
  ! corresponding (I,J) pair of M. Note that a symmetric reordering
  ! may have been performed so (I,J) may be in either the lower (I>J) or
  ! upper (I<J) triangle. Also note that some entered entries may be
  ! zero; NZX is used to count the number of zeros.

  NZX = 0
  K=0
  DO I=1,N
     K= K+1
     INDEX= ( (N*(N+1) - (N-I+1)*(N-I+2))/2 ) - (I - 1)
     A(K)=DDF(INDEX+I)
     NROWI=IA(I+1)-IA(I)-1
     DO JI = 1, NROWI
        K=K+1
        J=JA(K)
        A(K)=DDF(INDEX+J)
        IF(ABS(A(K)) < EPS) NZX = NZX +1
     ENDDO
  ENDDO
100 continue
  IF (K /=  (IA(N+1)-1)) WRITE (OUTU, *) 'ERROR IN MTOA2'

  IF(PRNLEV > 9) THEN
     WRITE(OUTU,*) 'IA ', (IA(I), I=1,N+1)
     LENA = IA(N+1) - 1
     WRITE(OUTU,*) 'LENA=', LENA
     WRITE(OUTU,*) 'JA ', (JA(I), I=1,LENA)
     WRITE(OUTU,*) 'A  ', ( A(I), I=1,LENA)
  ENDIF
  RETURN
END SUBROUTINE MTOA2
#endif /* (tnpack)*/

subroutine tndriv_dummy()
  return
end subroutine tndriv_dummy

