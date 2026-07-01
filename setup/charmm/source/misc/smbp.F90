SUBROUTINE NULL_SMBP2
RETURN
END SUBROUTINE NULL_SMBP2

#if KEY_PBEQ==1
#if KEY_SMBP==1
!CHARMM Element source/misc/smbp.src
      SUBROUTINE SPRC_PROJ(NATOM,X,Y,Z,CG,PHI,SURFCRD,SURFCHR,NQM,QMLST, &
                           QPRIN,QDOSCC,QDOQM,QBSPL)
!
!-----------------------------------------------------------------------
!
! Calculates the projection and optimization of the surface charges
! for the QM-part of SMBP
! 
!-----------------------------------------------------------------------
!
!
      use chm_kinds
      use memory
      use number
      use stream
      use string
      use dimens_fcm
      use consta
      use comand
      use parallel
      use pbeq,only: numsurf,nclxog,nclyog,nclzog,dcelog,&
                     tranxog,tranyog,tranzog,xbcenog,ybcenog,zbcenog
      implicit none
!
      real(chm_real)  X(*),Y(*),Z(*),CG(*),SURFCRD(*),SURFCHR(*)
      real(chm_real4) PHI(*)
      INTEGER QMLST(*) 
      INTEGER NATOM,NQM
      LOGICAL QPRIN,QBSPL
      LOGICAL QDOSCC,QDOQM
! local
      real(chm_real), allocatable, dimension (:,:) :: DIST,AMAT
      real(chm_real), allocatable, dimension (:) :: PHIVEC,BVEC
      real(chm_real)  ESMBP,FACTOR,SUMSURFCHR
      real(chm_real)  XS,YS,ZS
      real(chm_real)  POT,POTA
      real(chm_real)  ERROR,MAXERR,MAE
      INTEGER I,II,IL,J,N
      INTEGER OLDUSD
      LOGICAL QCHECK_SURFACE_CHRG
! JZ debug
      real(chm_real), allocatable, dimension (:) :: CGSAVE 
      real(chm_real)  EDBG,MINONEFOUR
      INTEGER I1,DBGPRNT,NTMP
      PARAMETER (DBGPRNT=1)

      QCHECK_SURFACE_CHRG = .TRUE. 

      call chmalloc('smbp.src','SPRC_PROJ','PHIVEC',NQM,crl=PHIVEC)
      call chmalloc('smbp.src','SPRC_PROJ','BVEC',NUMSURF,crl=BVEC)
      call chmalloc('smbp.src','SPRC_PROJ','DIST',NUMSURF,NQM,crl=DIST)
      call chmalloc('smbp.src','SPRC_PROJ','AMAT',NUMSURF,NUMSURF,crl=AMAT)
      call chmalloc('smbp.src','SPRC_PROJ','CGSAVE',NATOM,crl=CGSAVE)
!
!     1. Charge positions are in SURFCRD; Surface-charge array
!        has same order
!        Obtain A (matrix) and b (vector) for Conjugate Gradient (Aq = b) 
!
!
!     1.1 Obtain vector of phi^QM_tot at positions of QM-atoms
!         and store it in PHIVEC
      CALL SMBP_SPHE_BUILD_POT_QM_POS(NQM,QMLST,X,Y,Z,CG, &
                 NCLXOG,NCLYOG,NCLZOG,DCELOG, &
                 PHI,PHIVEC,TRANXOG,TRANYOG,TRANZOG, &
                 XBCENOG,YBCENOG,ZBCENOG,ONE,QBSPL)
!
!     1.2 Calculate matrix of inverse QM-Surface-charge distances 
!         and store it in DIST
      CALL SMBP_SPHE_BUILD_DIST(NQM,QMLST,NATOM,X,Y,Z,CG,SURFCRD,DIST)

!
!     1.3 Calculate A and b 
!
      CALL SMBP_BUILD_A_B(NQM,QMLST,NATOM,PHIVEC,DIST,AMAT,BVEC)

!
!     1.4 Perform Conjugate Gradient minimization of f(q) = 1/2 q^T A q - b^T q + c
!         by solving A q = b 
!
      CALL FILLR8(SURFCHR,NUMSURF,ZERO)
      CALL SMBP_CONJ_GRAD(AMAT,BVEC,SURFCHR,NUMSURF)

      IF (PRNLEV .ge. 5) THEN
        IF (PRNLEV .ge. 10) WRITE(OUTU,100)'Printout of surface charges'
        SUMSURFCHR = ZERO
        DO I = 1, NUMSURF
          IF (PRNLEV .ge. 10) & 
            WRITE(OUTU,103)'surface charge(',I,') = ', SURFCHR(I)
          SUMSURFCHR = SUMSURFCHR + SURFCHR(I)
        ENDDO
        WRITE(OUTU,101)'Sum of surface charges = ', SUMSURFCHR
      ENDIF

!     2. Check if optimized surface charges are able to reproduce PHIVEC
      IF (QCHECK_SURFACE_CHRG) THEN
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,100)''
          WRITE(OUTU,100)'Checking surface charges'
        ENDIF
        N = NUMSURF
        MAXERR = ZERO
        MAE    = ZERO
        DO I = 1, NQM
          POT  = PHIVEC(I)
          POTA = ZERO
          DO J = 1, N
            POTA = POTA + SURFCHR(J)*DIST(J,I)
          ENDDO
          ERROR  = ABS(POT-POTA)
          MAXERR = MAX(MAXERR,ERROR)
          MAE    = MAE + ERROR
        ENDDO 
        MAE = MAE/NQM
        IF (MAE .gt. 1.e-3) THEN 
          WRITE(OUTU,102)'Maximum Error       = ', MAXERR
          WRITE(OUTU,102)'Mean Absolute Error = ', MAE
          CALL WRNDIE(-3,'<SMBP>','Large error in surface charges')
        ENDIF
        IF (PRNLEV .ge. 5) THEN
          WRITE(OUTU,102)'Maximum Error       = ', MAXERR
          WRITE(OUTU,102)'Mean Absolute Error = ', MAE
          WRITE(OUTU,100)''
        ENDIF
      ENDIF

      call chmdealloc('smbp.src','SPRC_PROJ','CGSAVE',NATOM,crl=CGSAVE)
      call chmdealloc('smbp.src','SPRC_PROJ','PHIVEC',NQM,crl=PHIVEC)
      call chmdealloc('smbp.src','SPRC_PROJ','BVEC',NUMSURF,crl=BVEC)
      call chmdealloc('smbp.src','SPRC_PROJ','DIST',NUMSURF,NQM,crl=DIST)
      call chmdealloc('smbp.src','SPRC_PROJ','AMAT',NUMSURF,NUMSURF,crl=AMAT)


100   FORMAT(3X,A)
101   FORMAT(3X,A,F20.6)
102   FORMAT(3X,A,E15.2)
103   FORMAT(3X,A,I4,A,F15.4)
      RETURN
      END SUBROUTINE SPRC_PROJ

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP_SPHE_BUILD_POT_QM_POS(NQM,QMLST,X,Y,Z,CG, &
                 NCLX,NCLY,NCLZ,DCEL, &
                 PHI1,PHIVEC,TRANX,TRANY,TRANZ,XBCEN,YBCEN,ZBCEN,FACTOR,QBSPL)
!-----------------------------------------------------------------------
!     This subroutine computes the electrostatic potential at
!     the position of each QM atom.
!
!     Taken from RFORCE2 
!
      use chm_kinds
      use consta
      use number
      use stream
      use pbeq,only: M3
#if KEY_GCMC==1
      use gcmc 
#endif
      implicit none
!
      real(chm_real)  x(*),y(*),z(*),cg(*)
      real(chm_real)  dcel,tranx,trany,tranz,xbcen,ybcen,zbcen
      real(chm_real)  phivec(nqm)
      real(chm_real4) phi1(*)
      real(chm_real)  FACTOR
      integer NQM,QMLST(*)
      integer NCLX,NCLY,NCLZ
      logical QPRIN,QBSPL
!   local
      real(chm_real)  chi,xi,yi,zi,ai,bi,ci,fi
      real(chm_real)  enelp1
      real(chm_real)  aisign,bisign,cisign,prefac
      real(chm_real)  xgrad,ygrad,zgrad
      integer ncyz,il,i,j,ix,iy,iz,n1,in1,n2,in2,n3,in3
! B-spline
      real(chm_real)  xc,yc,zc
      integer jx1,jx2,jy1,jy2,jz1,jz2,k,l,m,ipx,ipy,ipz,nfil
!
      ncyz=ncly*nclz
      enelp1=zero

      DO I = 1, NQM
        PHIVEC(I) = ZERO
#if KEY_GCMC==1
        J=QMLST(I)
        IF (.NOT. GCMCON(J)) THEN
          CALL WRNDIE(-5,'<SMBP>','All QM atoms must be active for GCMC')
        ENDIF
#endif 
      ENDDO

!
!
!==============================================================================
      IF(QBSPL) THEN
!
!     Main loop by atoms
!
         do il=1,nqm 
            i=qmlst(il)
            chi=cg(i)
            xi=x(i)+tranx-xbcen
            yi=y(i)+trany-ybcen
            zi=z(i)+tranz-zbcen
            nfil=1
            ix=int(xi/dcel)+1
            iy=int(yi/dcel)+1
            iz=int(zi/dcel)+1
            jx1=ix-nfil
            if(jx1.lt.1)jx1=1
            jx2=ix+nfil+1
            if(jx2.gt.nclx)jx2=nclx
            jy1=iy-nfil
            if(jy1.lt.1)jy1=1
            jy2=iy+nfil+1
            if(jy2.gt.ncly)jy2=ncly
            jz1=iz-nfil
            if(jz1.lt.1)jz1=1
            jz2=iz+nfil+1
            if(jz2.gt.nclz)jz2=nclz

            DO K=jx1,jx2
               IPX=(K-1)*NCyz
               XC=(K-1)*DCEL
!
               DO L=jy1,jy2
                  IPY=(L-1)*NCLz
                  YC=(L-1)*DCEL
!
                  DO M=jz1,jz2
                     IPZ=M+IPY+IPX
                     ZC=(M-1)*DCEL

                     ai=1.5-(xi-xc)/dcel
                     bi=1.5-(yi-yc)/dcel
                     ci=1.5-(zi-zc)/dcel
                     fi=M3(ai)*M3(bi)*M3(ci)
!
                     phivec(il)=phivec(il)+FACTOR*fi*phi1(ipz)

                  enddo
               enddo
            enddo
 490        continue
         enddo
!
      ELSE
!
!
!     Main loop by atoms
!
         do il=1,nqm 
            i=qmlst(il)
            chi=cg(i)
            xi=x(i)+tranx-xbcen
            yi=y(i)+trany-ybcen
            zi=z(i)+tranz-zbcen
            ix=int(xi/dcel)+1
            iy=int(yi/dcel)+1
            iz=int(zi/dcel)+1
!
!     Atom charge distribution by 8 adjacent grid points
!
            do n1=1,2
               in1=ix+n1-1
               ai=xi-(in1-1)*dcel
               aisign=sign(one,ai)
               ai=1.-abs(ai)/dcel
               in1=(in1-1)*ncyz

               do n2=1,2
                  in2=iy+n2-1
                  bi=yi-(in2-1)*dcel
                  bisign=sign(one,bi)
                  bi=1. - abs(bi)/dcel
                  in2=(in2-1)*nclz

                  do n3=1,2
                     in3=iz+n3-1
                     ci=zi-(in3-1)*dcel
                     cisign=sign(one,ci)
                     ci=1. - abs(ci)/dcel
                     fi=ai*bi*ci
                     in3=in1+in2+in3
!
!     Electrostatic Energy
!
                     phivec(il)=phivec(il)+FACTOR*fi*phi1(in3)

                  enddo
               enddo
            enddo
 500        continue
         enddo
!
      ENDIF


!==============================================================================
      RETURN
      END SUBROUTINE SMBP_SPHE_BUILD_POT_QM_POS


!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP_SPHE_BUILD_DIST(NQM,QMLST,NATOM,X,Y,Z,CG, &
                                      SURFCRD,DIST)
!-----------------------------------------------------------------------
!     This subroutine computes the distance matrix between QM-Atom positions
!     and those of the surface charges
!     
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use parallel
      use pbeq,only: numsurf
      implicit none
!
      real(chm_real)  X(*),Y(*),Z(*),CG(*)
      real(chm_real)  SURFCRD(*),DIST(NUMSURF,NQM)
      INTEGER QMLST(*) 
      INTEGER NQM,NATOM
!   local
      real(chm_real)  XS,YS,ZS,XQ,YQ,ZQ
      INTEGER I,J,IL,NUM
!
      INTEGER NTMP


!     Loop over Quantum-atoms and surface charges
      NUM = 1
      DO J = 1, NUMSURF
        XS = SURFCRD(NUM)
        YS = SURFCRD(NUM+1)
        ZS = SURFCRD(NUM+2)

        DO IL = 1, NQM
          I = QMLST(IL)
          XQ = X(I)
          YQ = Y(I)
          ZQ = Z(I)
       
            DIST(J,IL) = ONE/(SQRT((XS-XQ)*(XS-XQ) & 
                                 + (YS-YQ)*(YS-YQ) &
                                 + (ZS-ZQ)*(ZS-ZQ)))

        ENDDO
        NUM = NUM + 3
      ENDDO

      RETURN
      END SUBROUTINE SMBP_SPHE_BUILD_DIST

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      SUBROUTINE SMBP_BUILD_A_B(NQM,QMLST,NATOM,PHIVEC,DIST,AMAT,BVEC) 
!-----------------------------------------------------------------------
!     This subroutine computes the A-matrix and the b-vector for 
!     the Conjugate Gradient minimization
!     
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use parallel
      use pbeq,only: numsurf
      implicit none
!
      real(chm_real)  PHIVEC(NQM),DIST(NUMSURF,NQM)
      real(chm_real)  AMAT(NUMSURF,NUMSURF),BVEC(NUMSURF)
      INTEGER QMLST(*) 
      INTEGER NQM,NATOM
!   local
      real(chm_real), allocatable, dimension(:) ::  EVAL,EVEC,ACOPY,WORK1
      real(chm_real)  BSUM,ASUM
      INTEGER IER
      INTEGER I,J,K,IL
      LOGICAL QCHECK_POS_DEF_A

!     Initialisation
      QCHECK_POS_DEF_A = .TRUE.
!

!     Build b
      DO I = 1, NUMSURF
        BSUM = ZERO
        DO J = 1, NQM
          BSUM = BSUM + PHIVEC(J)*DIST(I,J)
        ENDDO
        BVEC(I) = BSUM
      ENDDO


!     Build A
      DO I = 1, NUMSURF
        DO J = 1, I
          ASUM = ZERO
          DO K = 1, NQM
            ASUM = ASUM + DIST(I,K)*DIST(J,K)
          ENDDO
          AMAT(I,J) = ASUM
          AMAT(J,I) = ASUM
        ENDDO
      ENDDO


!     If requested, check for positive definiteness of A
!     by calculating the EV and checking if they are all > 0
      IF (QCHECK_POS_DEF_A) THEN
        call chmalloc('smbp.src','SMBP_BUILD_A_B','EVAL',NUMSURF,crl=EVAL)
        call chmalloc('smbp.src','SMBP_BUILD_A_B','ACOPY',NUMSURF*NUMSURF,crl=ACOPY)
        call chmalloc('smbp.src','SMBP_BUILD_A_B','EVEC',NUMSURF*NUMSURF,crl=EVEC)
        call chmalloc('smbp.src','SMBP_BUILD_A_B','WORK1',NUMSURF,crl=WORK1)
        IER   = 0
#if KEY_GAMESS==1
        CALL DCOPY(int8(NUMSURF*NUMSURF),AMAT,1_8,ACOPY,1_8)
#else
        CALL DCOPY(NUMSURF*NUMSURF,AMAT,1,ACOPY,1)
#endif
        CALL FILLR8(EVEC,NUMSURF*NUMSURF,ZERO)
        CALL FILLR8(EVAL,NUMSURF,ZERO)
        CALL EIGRS(ACOPY,NUMSURF,11,EVAL,EVEC,NUMSURF,WORK1,IER) 
        IF (IER .GT. 128) THEN
            IER = IER - 128
            IF(WRNLEV.GE.2) WRITE (OUTU,'(A,I6,A)') &
            ' DIAGRS> Failed to converge on root number ', IER, '.'
        ENDIF
       
        DO I = 1,NUMSURF
!          IF ((ABS(EVAL(I)) .le. RSMALL) .or.    ! problems with numerics 
          IF (ABS(EVAL(I)) .gt. RSMALL .and. EVAL(I) .lt. ZERO) THEN
            CALL WRNDIE(-5,'<SMBP_BUILD_A_B>', &
                  'Distance matrix not positive definite!')
          ENDIF
        ENDDO
        call chmdealloc('smbp.src','SMBP_BUILD_A_B','EVAL',NUMSURF,crl=EVAL)
        call chmdealloc('smbp.src','SMBP_BUILD_A_B','ACOPY',NUMSURF*NUMSURF,crl=ACOPY)
        call chmdealloc('smbp.src','SMBP_BUILD_A_B','EVEC',NUMSURF*NUMSURF,crl=EVEC)
        call chmdealloc('smbp.src','SMBP_BUILD_A_B','WORK1',NUMSURF,crl=WORK1)
      ENDIF

      RETURN
      END SUBROUTINE SMBP_BUILD_A_B

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SMBP_CONJ_GRAD(AMAT,BVEC,QVEC,N)
!-----------------------------------------------------------------------
!     This is a Conjugate Gradient solver for QVEC 
!     in AMAT x QVEC = BVEC, using diagonal preconditioning
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use parallel
      use pbeq,only: cgthresh,cgmaxit
      use vector, only: snrmvec,dotpr,addctv,scalr8
      implicit none
!
      real(chm_real) AMAT(N,N),BVEC(N),QVEC(N)
! local
      real(chm_real), allocatable, dimension(:) :: M,RVEC,PVEC,SVEC,AP,AQ 
      real(chm_real) ALPHA,BETA,DELTA,ALPHAMIN,DELTANEW,PAP
      real(chm_real) ERROR,MAXERR,MAE,BNRM,RNRM,THRESH
      INTEGER I,J,N
      LOGICAL QCHECK_CG_ERR

      call chmalloc('smbp.src','SMBP_CONJ_GRAD','M',N,crl=M)
      call chmalloc('smbp.src','SMBP_CONJ_GRAD','RVEC',N,crl=RVEC)
      call chmalloc('smbp.src','SMBP_CONJ_GRAD','PVEC',N,crl=PVEC)
      call chmalloc('smbp.src','SMBP_CONJ_GRAD','SVEC',N,crl=SVEC)
      call chmalloc('smbp.src','SMBP_CONJ_GRAD','AP',N,crl=AP)
      call chmalloc('smbp.src','SMBP_CONJ_GRAD','AQ',N,crl=AQ)

      THRESH = CGTHRESH
      QCHECK_CG_ERR = .TRUE. 
!     Construct preconditioner first; we use the simplest ansatz,
!     the Jacobi (diagonal) preconditioner: M = diag(A)
      DO I = 1, N
        M(I) = ONE/AMAT(I,I)
      ENDDO


      IF (PRNLEV .ge. 10) THEN
        WRITE(OUTU,100)'Entering Conjugate Gradient Solver'
      ENDIF
!     Initialize
      CALL SNRMVEC(BVEC,BNRM,N)

!     Check if BVEC is zero, and if so, return zero for
!     the surface charges
      IF (BNRM .le. TENM14) THEN
        IF (PRNLEV .ge. 10) THEN
          WRITE(OUTU,100) 'Surface charges set to zero. Skipping CG solver!'
        ENDIF
        CALL FILLR8(QVEC,N,ZERO)
        RETURN
      ENDIF

      CALL OLDCHM_MATVEC(AQ,AMAT,QVEC,N)
      DO I = 1, N
        RVEC(I) = BVEC(I) - AQ(I)
      ENDDO         

#if KEY_GAMESS==1
      CALL DCOPY(int8(N),RVEC,1_8,PVEC,1_8)
#else
      CALL DCOPY(N,RVEC,1,PVEC,1)
#endif
      DO I = 1, N
        PVEC(I) = PVEC(I) * M(I)
      ENDDO         

      IF (PRNLEV .ge. 10) WRITE(OUTU,101)'User Threshold   = ', THRESH
      CALL DOTPR(RVEC,PVEC,N,DELTA)
      THRESH = THRESH * THRESH
      THRESH = THRESH * DELTA
      IF (PRNLEV .ge. 10) WRITE(OUTU,101)'==> CG Threshold = ', THRESH

!     Start loop for optimizing charges
      IF (PRNLEV .ge. 100) THEN
        WRITE(OUTU,100)''
        WRITE(OUTU,100)'Iteration |     Error'
        WRITE(OUTU,100)'---------------------'
      ENDIF

      ERROR = 1.0E0
      I = 1
10    CONTINUE

      CALL OLDCHM_MATVEC(AP,AMAT,PVEC,N)
      CALL DOTPR(PVEC,AP,N,PAP)
      ALPHA = DELTA/PAP

      CALL ADDCTV(QVEC,PVEC,N,ALPHA)
 
!     Remove accumulated floating-point errors
      IF (MOD(I-1,50) .eq. 0) THEN
        IF (PRNLEV .ge. 100) THEN
          WRITE(OUTU,100)'CALCULATING EXACT RESIDUAL'
        ENDIF
        CALL OLDCHM_MATVEC(AQ,AMAT,QVEC,N)
        DO J = 1, N
          RVEC(J) = BVEC(J) - AQ(J)
        ENDDO         
      ELSE
        ALPHAMIN = MINONE*ALPHA
        CALL ADDCTV(RVEC,AP,N,ALPHAMIN)
      ENDIF

      DO J = 1, N
        SVEC(J) = RVEC(J) * M(J)
      ENDDO

      CALL DOTPR(RVEC,SVEC,N,DELTANEW)
      BETA = DELTANEW/DELTA

      CALL SCALR8(PVEC,N,BETA)
      CALL ADDCTV(PVEC,SVEC,N,ONE) 

      CALL SNRMVEC(RVEC,RNRM,N)
      ERROR = RNRM/BNRM
      IF (PRNLEV .ge. 100) WRITE(OUTU,104) I, ' |', ERROR

      I     = I + 1
      DELTA = DELTANEW 
      IF (I .le. CGMAXIT .AND. ERROR .gt. THRESH) GOTO 10
20    CONTINUE
      IF (I .gt. CGMAXIT) THEN
        CALL WRNDIE(-3,'<SMBP_CONJ_GRAD>','CG solver not converged')
      ENDIF

      IF (PRNLEV .ge. 100) WRITE(OUTU,100)'---------------------'
      IF (PRNLEV .ge. 10) WRITE(OUTU,101)'Final Error          = ',ERROR
      IF (PRNLEV .ge. 10) WRITE(OUTU,102)'Number of Iterations = ',I-1


!     Printout of CG-errors
      IF (QCHECK_CG_ERR) THEN
        IF (PRNLEV .ge. 10) THEN
          WRITE(OUTU,100)''
          WRITE(OUTU,100)'!!! Checking CG-Errors !!!' 
        ENDIF
        CALL OLDCHM_MATVEC(AQ,AMAT,QVEC,N)
        IF (PRNLEV .ge. 100) THEN
          DO I = 1, N
            WRITE(OUTU,103) 'Error(', I, ') = ', ABS(AQ(I)-BVEC(I))
          ENDDO
        ENDIF
        MAXERR = ZERO
        MAE    = ZERO
        DO I = 1, N
          MAXERR = MAX(MAXERR,ABS(AQ(I)-BVEC(I)))
          MAE    = MAE + ABS(AQ(I)-BVEC(I))
        ENDDO
        MAE = MAE/N
        IF (MAE .gt. 1.e-3) THEN
          WRITE(OUTU,101)'Maximum CG Error       = ', MAXERR
          WRITE(OUTU,101)'Mean Absolute CG Error = ', MAE
          CALL WRNDIE(-3,'<SMBP>','Large error in CG solver')
        ENDIF
        IF (PRNLEV .ge. 10) THEN
          WRITE(OUTU,101)'Maximum CG Error       = ', MAXERR
          WRITE(OUTU,101)'Mean Absolute CG Error = ', MAE
        ENDIF
      ENDIF

      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','M',N,crl=M)
      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','RVEC',N,crl=RVEC)
      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','PVEC',N,crl=PVEC)
      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','SVEC',N,crl=SVEC)
      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','AP',N,crl=AP)
      call chmdealloc('smbp.src','SMBP_CONJ_GRAD','AQ',N,crl=AQ)

 100  FORMAT(3X,A)
 101  FORMAT(3X,A,E10.2)
 102  FORMAT(3X,A,I4)
 103  FORMAT(3X,A,I4,A,E10.2)
 104  FORMAT(3X,I9,A,E10.2)

      RETURN
      END SUBROUTINE SMBP_CONJ_GRAD

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SMBP_CALC_QM(EQM,NATOM,X,Y,Z,CG, &
                              SURFCRD,SURFCHR, &
                              NQM,QMLST,QMCHR,QPRIN,QDOSCC,QDOQM)
!-----------------------------------------------------------------------
!     This calls the appropriate QM-routine to calculate 
!     the QM-part of the SMBP
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use parallel
      use psf,only: amass,iac
      use pbeq,only: numsurf,qcchrg
      use gamess_fcm
#if KEY_QCHEM==1 || KEY_G09==1
      use gukini_mod, only: gukene 
#endif
#if KEY_SCCDFTB==1
      use sccdftb 
#endif
      implicit none
!
      real(chm_real) X(*),Y(*),Z(*),CG(*)
      real(chm_real) SURFCRD(*),SURFCHR(*),QMCHR(*)
      real(chm_real) EQM
      INTEGER NATOM,NQM
      INTEGER QMLST(NQM)
      LOGICAL QPRIN,QDOSCC,QDOQM 
! local
      real(chm_real), allocatable, dimension(:) :: SAVEX,SAVEY,SAVEZ,SAVECG,SAVEIG
      real(chm_real), allocatable, dimension(:) :: TMPDX,TMPDY,TMPDZ
      INTEGER I,NATSURF,NUM,NPTCOLD
      LOGICAL DOGRAD,LEXIST
! dummy variables for GUKENE hessian
      real(chm_real),dimension(1) :: DD1 
      INTEGER,DIMENSION(1) ::  IUPT,JUPT
      INTEGER NDD1
      LOGICAL QSECD

      call chmalloc('smbp.src','SMBP_CALC_QM','SAVEX',NATOM+NUMSURF,crl=SAVEX)
      call chmalloc('smbp.src','SMBP_CALC_QM','SAVEY',NATOM+NUMSURF,crl=SAVEY)
      call chmalloc('smbp.src','SMBP_CALC_QM','SAVEZ',NATOM+NUMSURF,crl=SAVEZ)
      call chmalloc('smbp.src','SMBP_CALC_QM','SAVECG',NATOM+NUMSURF,crl=SAVECG)
      call chmalloc('smbp.src','SMBP_CALC_QM','SAVEIG',NATOM+NUMSURF,crl=SAVEIG)
      call chmalloc('smbp.src','SMBP_CALC_QM','TMPDX',NATOM+NUMSURF,crl=TMPDX)
      call chmalloc('smbp.src','SMBP_CALC_QM','TMPDY',NATOM+NUMSURF,crl=TMPDY)
      call chmalloc('smbp.src','SMBP_CALC_QM','TMPDZ',NATOM+NUMSURF,crl=TMPDZ)

      QSECD = .false.
      EQM   = ZERO

      NATSURF = NATOM+NUMSURF
      IF (NATSURF .gt. MAXAIM) THEN
        CALL WRNDIE(-5,'<SMBP_CALC_QM>','Too many atoms/surface charges')
      ENDIF

!     Add coords of surface charges to X,Y,Z
!     and surface charges to CG
!     Save Coord and CG-Arrays beforehand
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATSURF),X,1_8,SAVEX,1_8)
      CALL DCOPY(int8(NATSURF),Y,1_8,SAVEY,1_8)
      CALL DCOPY(int8(NATSURF),Z,1_8,SAVEZ,1_8)
      CALL DCOPY(int8(NATSURF),CG,1_8,SAVECG,1_8)
#else
      CALL DCOPY(NATSURF,X,1,SAVEX,1)
      CALL DCOPY(NATSURF,Y,1,SAVEY,1)
      CALL DCOPY(NATSURF,Z,1,SAVEZ,1)
      CALL DCOPY(NATSURF,CG,1,SAVECG,1)
#endif

!     -------------------
      IF (QDOSCC) THEN
!     -------------------

#if KEY_SCCDFTB==1

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM) 
          NUM  = NUM+3
        ENDDO


!       Add surface charges to external charges (MM) array in SCCDFTB
        NPTCOLD = NPTC
        CALL UPSCCSURF(NUMSURF,SURFCRD,SURFCHR)

!       Calculate SCC-part and save mulliken charges in QMCHR
        DOGRAD = .FALSE. !!! No Gradients here
        CALL SCCTBENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,NATOM,DOGRAD)
#if KEY_GAMESS==1
        CALL DCOPY(int8(NQM),QMULIK,1_8,QMCHR,1_8)
#else
        CALL DCOPY(NQM,QMULIK,1,QMCHR,1)
#endif

!       Restore NPTC, important!!!
        NPTC = NPTCOLD

#endif 

!     -------------------
      ELSEIF (QDOQM) THEN 
!     -------------------

#if KEY_QCHEM==1

#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),IGMSEL,1_8,SAVEIG,1_8)
#else
        CALL DCOPY(NATSURF,IGMSEL,1,SAVEIG,1)
#endif

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM)
          IGMSEL(I) = 6 
          NUM  = NUM+3
        ENDDO

!       Call Q-Chem via GUKENE
!       Use wrapper because of AMASS and IAC (conflict with NATOM and CG when
!       including psf.fcm) 
        CALL GUKENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,AMASS,IAC,NATOM, &
                    NDD1,DD1,QSECD,IUPT,JUPT)

!        CALL GUKENE_SMBP(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,NATOM,NDD1, &
!                         DD1,QSECD,IUPT,JUPT)

!       Read charges from files (mulliken: 'charges.dat', MDC: 'mdc_charges.dat')
!       QCCHRG == 1: MDC; QCCHRG == 2: Mulliken

!       MDC first
        IF (QCCHRG .eq. 1) THEN     
          INQUIRE(FILE='mdc_charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading MDC charges'
            OPEN (UNIT=11, FILE='mdc_charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(3X,F19.16)')QMCHR(I)
            ENDDO
          ELSE 
            CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                        'File mdc_charges.dat not found')
          ENDIF
        ELSEIF (QCCHRG .eq. 2) THEN   
          INQUIRE(FILE='charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading Charges from Q-Chem' 
            OPEN (UNIT=11, FILE='charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(3X,F19.16)')QMCHR(I)
            ENDDO
          ELSE 
            CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                        'File charges.dat not found')
          ENDIF
        ELSE 
          CALL WRNDIE(-5,'<SMBP_CALC_QM>','QCCH must be 1 or 2')    
        ENDIF

#if KEY_GAMESS==1
        CALL DCOPY(NATSURF,SAVEIG,1,IGMSEL,1)
#else
        CALL DCOPY(NATSURF,SAVEIG,1,IGMSEL,1)
#endif

#elif KEY_G09==1

#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),IGMSEL,1_8,SAVEIG,1_8)
#else
        CALL DCOPY(NATSURF,IGMSEL,1,SAVEIG,1)
#endif

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM)
          IGMSEL(I) = 6 
          NUM  = NUM+3
        ENDDO


!       Call Gaussian via GUKENE
!       Use wrapper because of AMASS and IAC (conflict with NATOM and CG when
!       including psf.fcm) 
        CALL GUKENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,AMASS,IAC,NATOM, &
                    NDD1,DD1,QSECD,IUPT,JUPT)
!        CALL GUKENE_SMBP(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,NATOM,NDD1, &
!                         DD1,QSECD,IUPT,JUPT)

!       Read charges from files (mulliken: 'charges.dat', MDC: 'esp_charges.dat')
!       QCCHRG == 1: MDC; QCCHRG == 2: Mulliken

!       MDC first
        IF (QCCHRG .eq. 1) THEN     
          INQUIRE(FILE='esp_charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading ESP charges'
            OPEN (UNIT=11, FILE='esp_charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(E16.9)')QMCHR(I)
            ENDDO
          ELSE 
            CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                        'File esp_charges.dat not found')
          ENDIF
        ELSEIF (QCCHRG .eq. 2) THEN   
          INQUIRE(FILE='charges.dat',EXIST=LEXIST)
          IF (LEXIST) THEN
            IF (PRNLEV.ge.10) write(OUTU,100)'Reading Mulliken charges'
            OPEN (UNIT=11, FILE='charges.dat', status='old')
            REWIND(11)
            DO I=1, NQM
              READ(11,'(E16.9)')QMCHR(I)
            ENDDO
          ELSE 
            CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                        'File charges.dat not found')
          ENDIF
        ELSE 
          CALL WRNDIE(-5,'<SMBP_CALC_QM>','QCCH must be 1 or 2')    
        ENDIF

 
#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),SAVEIG,1_8,IGMSEL,1_8)
#else
        CALL DCOPY(NATSURF,SAVEIG,1,IGMSEL,1)
#endif

#else /**/
        CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                    'QDOQM: SMBP currently implemented  &
&                     for QM = Q-CHEM || Gaussian only')
#endif 


!     -------------------
      ELSE
!     -------------------

        CALL WRNDIE(-5,'<SMBP_CALC_QM>', &
                    'Either QDOSCC or QDOQM must be true')

!     -------------------
      ENDIF
!     -------------------

!     Restore Coords and CG-Arrays, just to make sure
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATSURF),SAVEX,1_8,X,1_8)
      CALL DCOPY(int8(NATSURF),SAVEY,1_8,Y,1_8)
      CALL DCOPY(int8(NATSURF),SAVEZ,1_8,Z,1_8)
      CALL DCOPY(int8(NATSURF),SAVECG,1_8,CG,1_8)
#else
      CALL DCOPY(NATSURF,SAVEX,1,X,1)
      CALL DCOPY(NATSURF,SAVEY,1,Y,1)
      CALL DCOPY(NATSURF,SAVEZ,1,Z,1)
      CALL DCOPY(NATSURF,SAVECG,1,CG,1)
#endif

      call chmdealloc('smbp.src','SMBP_CALC_QM','SAVEX',NATOM+NUMSURF,crl=SAVEX)
      call chmdealloc('smbp.src','SMBP_CALC_QM','SAVEY',NATOM+NUMSURF,crl=SAVEY)
      call chmdealloc('smbp.src','SMBP_CALC_QM','SAVEZ',NATOM+NUMSURF,crl=SAVEZ)
      call chmdealloc('smbp.src','SMBP_CALC_QM','SAVECG',NATOM+NUMSURF,crl=SAVECG)
      call chmdealloc('smbp.src','SMBP_CALC_QM','SAVEIG',NATOM+NUMSURF,crl=SAVEIG)
      call chmdealloc('smbp.src','SMBP_CALC_QM','TMPDX',NATOM+NUMSURF,crl=TMPDX)
      call chmdealloc('smbp.src','SMBP_CALC_QM','TMPDY',NATOM+NUMSURF,crl=TMPDY)
      call chmdealloc('smbp.src','SMBP_CALC_QM','TMPDZ',NATOM+NUMSURF,crl=TMPDZ)

 100  FORMAT(3X,A)
      RETURN
      END SUBROUTINE SMBP_CALC_QM

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SMBP_CALC_QM_GRAD(NTRB,LSTRB,NATOM,X,Y,Z,DX,DY,DZ,CG, &
                                   SURFCRD,SURFCHR,NUMSURF, &
                                   QMGX,QMGY,QMGZ, &
                                   NQM,QMLST,QMCHR,QPRIN,QDOSCC,QDOQM)
!-----------------------------------------------------------------------
!     This calls the appropriate QM-routine to calculate 
!     the QM gradient for the SMBP
!
      use chm_kinds
      use memory
      use number
      use stream
      use dimens_fcm
      use consta
      use parallel
      use gamess_fcm
      use psf,only: amass,iac
#if KEY_QCHEM==1 || KEY_G09==1
      use gukini_mod, only: gukene 
#endif
#if KEY_SCCDFTB==1
      use sccdftb 
#endif
      implicit none
!
      real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CG(*)
      real(chm_real)  SURFCRD(*),SURFCHR(*),QMCHR(*)
      real(chm_real)  QMGX(*),QMGY(*),QMGZ(*)
      INTEGER NATOM,NQM,NTRB,NUMSURF
      INTEGER QMLST(NQM),LSTRB(*)
      LOGICAL QPRIN,QDOSCC,QDOQM 
! local
      real(chm_real), allocatable, dimension(:) :: SAVEX,SAVEY,SAVEZ,SAVECG,SAVEIG
      real(chm_real), allocatable, dimension(:) :: TMPDX,TMPDY,TMPDZ,SAVEQMUL
      real(chm_real)  EQM
      INTEGER I,NATSURF,NUM,NPTCOLD
      LOGICAL DOGRAD,LEXIST
! dummy variables for GUKENE hessian
      real(chm_real),dimension(1) :: DD1 
      INTEGER,DIMENSION(1) ::  IUPT,JUPT
      INTEGER NDD1
      LOGICAL QSECD

      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEX',NATOM+NUMSURF,crl=SAVEX)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEY',NATOM+NUMSURF,crl=SAVEY)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEZ',NATOM+NUMSURF,crl=SAVEZ)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVECG',NATOM+NUMSURF,crl=SAVECG)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEIG',NATOM+NUMSURF,crl=SAVEIG)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDX',NATOM+NUMSURF,crl=TMPDX)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDY',NATOM+NUMSURF,crl=TMPDY)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDZ',NATOM+NUMSURF,crl=TMPDZ)
      call chmalloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEQMUL',NQM,crl=SAVEQMUL)

      QSECD = .false.

      NATSURF = NATOM+NUMSURF
      IF (NATSURF .gt. MAXAIM) THEN
        CALL WRNDIE(-5,'<SMBP_CALC_QM_GRAD>','Too many atoms/surface charges')
      ENDIF
      CALL FILLR8(QMGX,NTRB,ZERO)
      CALL FILLR8(QMGY,NTRB,ZERO)
      CALL FILLR8(QMGZ,NTRB,ZERO)
      CALL FILLR8(TMPDX,NATSURF,ZERO)
      CALL FILLR8(TMPDY,NATSURF,ZERO)
      CALL FILLR8(TMPDZ,NATSURF,ZERO)

!     Add coords of surface charges to X,Y,Z
!     and surface charges to CG
!     Save Coords, CG and QMULIK (for charges guess) beforehand
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATSURF),X,1_8,SAVEX,1_8)
      CALL DCOPY(int8(NATSURF),Y,1_8,SAVEY,1_8)
      CALL DCOPY(int8(NATSURF),Z,1_8,SAVEZ,1_8)
      CALL DCOPY(int8(NATSURF),CG,1_8,SAVECG,1_8)
#else
      CALL DCOPY(NATSURF,X,1,SAVEX,1)
      CALL DCOPY(NATSURF,Y,1,SAVEY,1)
      CALL DCOPY(NATSURF,Z,1,SAVEZ,1)
      CALL DCOPY(NATSURF,CG,1,SAVECG,1)
#endif

!     -------------------
      IF (QDOSCC) THEN
!     -------------------

#if KEY_SCCDFTB==1

!       Save QMULIK
#if KEY_GAMESS==1
        CALL DCOPY(int8(NQM),QMULIK,1_8,SAVEQMUL,1_8)
#else
        CALL DCOPY(NQM,QMULIK,1,SAVEQMUL,1)
#endif

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM) 
          NUM  = NUM+3
        ENDDO


!       Add surface charges to external charges (MM) array in SCCDFTB
        NPTCOLD = NPTC
        CALL UPSCCSURF(NUMSURF,SURFCRD,SURFCHR)

!       Initialize gradients and calculate SCC-part 
        DOGRAD = .TRUE. 
        CALL SCCTBENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,NATOM,DOGRAD)
        
        DO I = 1, NTRB
          QMGX(I) = TMPDX(LSTRB(I)) * MINONE 
          QMGY(I) = TMPDY(LSTRB(I)) * MINONE 
          QMGZ(I) = TMPDZ(LSTRB(I)) * MINONE 
        ENDDO

!       Restore NPTC, important!!!
        NPTC = NPTCOLD

!       Restore QMULIK
#if KEY_GAMESS==1
        CALL DCOPY(int8(NQM),SAVEQMUL,1_8,QMULIK,1_8)
#else
        CALL DCOPY(NQM,SAVEQMUL,1,QMULIK,1)
#endif


#endif 

!     -------------------
      ELSEIF (QDOQM) THEN 
!     -------------------

#if KEY_QCHEM==1
        
!       Save IGMSEL        
#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),IGMSEL,1_8,SAVEIG,1_8)
#else
        CALL DCOPY(NATSURF,IGMSEL,1,SAVEIG,1)
#endif

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM)
          IGMSEL(I) = 6 
          NUM  = NUM+3
        ENDDO


!       Call Q-Chem via GUKENE
!       Use wrapper because of AMASS and IAC (conflict with NATOM and CG when
!       including psf.fcm) 
        CALL GUKENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,AMASS,IAC,NATOM, &
                    NDD1,DD1,QSECD,IUPT,JUPT)
!        CALL GUKENE_SMBP(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,NATOM,NDD1, &
!                         DD1,QSECD,IUPT,JUPT)

        DO I = 1, NTRB
          QMGX(I) = TMPDX(LSTRB(I)) * MINONE 
          QMGY(I) = TMPDY(LSTRB(I)) * MINONE 
          QMGZ(I) = TMPDZ(LSTRB(I)) * MINONE 
        ENDDO


#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),SAVEIG,1_8,IGMSEL,1_8)
#else
        CALL DCOPY(NATSURF,SAVEIG,1,IGMSEL,1)
#endif

#elif KEY_G09==1

!       Save IGMSEL        
#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),IGMSEL,1_8,SAVEIG,1_8)
#else
        CALL DCOPY(NATSURF,IGMSEL,1,SAVEIG,1)
#endif

        NUM = 1
        DO I = NATOM+1, NATSURF
          X(I)  = SURFCRD(NUM)
          Y(I)  = SURFCRD(NUM+1)
          Z(I)  = SURFCRD(NUM+2)
          CG(I) = SURFCHR(I-NATOM)
          IGMSEL(I) = 6 
          NUM  = NUM+3
        ENDDO


!       Call Gaussian via GUKENE
!       Use wrapper because of AMASS and IAC (conflict with NATOM and CG when
!       including psf.fcm) 
        CALL GUKENE(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,AMASS,IAC,NATOM, &
                    NDD1,DD1,QSECD,IUPT,JUPT)
!        CALL GUKENE_SMBP(EQM,X,Y,Z,TMPDX,TMPDY,TMPDZ,CG,NATOM,NDD1, &
!                         DD1,QSECD,IUPT,JUPT)

        DO I = 1, NTRB
          QMGX(I) = TMPDX(LSTRB(I)) * MINONE 
          QMGY(I) = TMPDY(LSTRB(I)) * MINONE 
          QMGZ(I) = TMPDZ(LSTRB(I)) * MINONE 
        ENDDO


#if KEY_GAMESS==1
        CALL DCOPY(int8(NATSURF),SAVEIG,1_8,IGMSEL,1_8)
#else
        CALL DCOPY(NATSURF,SAVEIG,1,IGMSEL,1)
#endif

#else /**/
        CALL WRNDIE(-5,'<SMBP_CALC_QM_GRAD>', &
                    'QDOQM: SMBP currently implemented  &
&                     for QM = Q-CHEM || Gaussian only')
#endif 

!     -------------------
      ELSE
!     -------------------

        CALL WRNDIE(-5,'<SMBP_CALC_QM_GRAD>', &
                    'Either QDOSCC or QDOQM must be true')

!     -------------------
      ENDIF
!     -------------------

      ! Restore coords and CG
#if KEY_GAMESS==1
      CALL DCOPY(int8(NATSURF),SAVEX,1_8,X,1_8)
      CALL DCOPY(int8(NATSURF),SAVEY,1_8,Y,1_8)
      CALL DCOPY(int8(NATSURF),SAVEZ,1_8,Z,1_8)
      CALL DCOPY(int8(NATSURF),SAVECG,1_8,CG,1_8)
#else
      CALL DCOPY(NATSURF,SAVEX,1,X,1)
      CALL DCOPY(NATSURF,SAVEY,1,Y,1)
      CALL DCOPY(NATSURF,SAVEZ,1,Z,1)
      CALL DCOPY(NATSURF,SAVECG,1,CG,1)
#endif

      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEX',NATOM+NUMSURF,crl=SAVEX)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEY',NATOM+NUMSURF,crl=SAVEY)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEZ',NATOM+NUMSURF,crl=SAVEZ)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVECG',NATOM+NUMSURF,crl=SAVECG)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEIG',NATOM+NUMSURF,crl=SAVEIG)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDX',NATOM+NUMSURF,crl=TMPDX)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDY',NATOM+NUMSURF,crl=TMPDY)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','TMPDZ',NATOM+NUMSURF,crl=TMPDZ)
      call chmdealloc('smbp.src','SMBP_CALC_QM_GRAD','SAVEQMUL',NQM,crl=SAVEQMUL)

 100  FORMAT(3X,A)
      RETURN
      END SUBROUTINE SMBP_CALC_QM_GRAD

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!      SUBROUTINE GUKENE_SMBP(EQM,X,Y,Z,DX,DY,DZ,CGX,NATOMX, &
!                             NDD1,DD1,QSECD,IUPT,JUPT)
!!-----------------------------------------------------------------------
!!     Wrapper for calling GUKENE 
!!
!      use chm_kinds
!      use psf
!      use gukini_mod, only: gukene
!      implicit none
!
!      real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CGX(*),DD1(*)
!      real(chm_real) EQM
!      INTEGER IUPT(*),JUPT(*)
!      INTEGER NATOMX,NDD1
!      LOGICAL QSECD 
!
#if KEY_IF==1 || KEY_QCHEM==1 || KEY_G09==1

#endif
!      CALL GUKENE(EQM,X,Y,Z,DX,DY,DZ,CGX,AMASS,IAC,NATOMX, &
!                  NDD1,DD1,QSECD,IUPT,JUPT)
#if KEY_ENDIF==1

#endif
!
!      RETURN
!      END SUBROUTINE GUKENE_SMBP

#endif /* SMBP*/
#endif /* PBEQ */


