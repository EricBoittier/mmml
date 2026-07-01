!  =====================================================================
!  QC: The following were modified based on phi and phi1 
!  which would be used if we chose to specify KMAX, KSQMAX instead of
!  letting the code (phi,phi1) determines these values.
!  Basically try to be consistent with CHARMM
!  =====================================================================
!  evaluation of the potential phi 
!
!   phi(r) = 4*pi/Omega ( Summe{G neq 0} e^{-G^2/{4 alpha^2}}/{G^2} cos(G r)
!            +Summe{R, R neq r} (1-erf(alpha*|R-r|))/|R-r|
!            -pi/(Omega*alpha^2)
!
!
!   INPUT Parameter:
!   REAL*8 r(3)           position of evaluation of potential
!   REAL*8 basis(3,3)     basis of cell
!   REAL*8 recbasis(3,3)      basis of reciprocal cell
!   REAL*8 alpha          convergence parameter
!   REAL*8 vol            cell volume
!   REAL*8 tol            tolerance for convergence of "last shell"
!                       (can often be used as a criterion for global
!                        convergence) 
!   OUTPUT:
!   REAL*8 potential      value of Ewald potential
!  
!  =====================================================================

        subroutine phiscc(x,basis,recbasis,alpha,vol,tol,
     $                    xE,nE,ZE,qsccewc,qsccews,potential,
     $                    erfct,del,pts)
        use sccdftb, only: kxvec,kyvec,kzvec,kvec,nkvec
        use sccdftbsrc, only: qsccnb,sccfnb,qsccs,qsccsh
        IMPLICIT NONE
!       QC: Phiscc differs from phiscc0 in that:
!       For each QM(with coordinates x), sums over ALL mm atoms
!       includes both reciprocal and real sums

!       external terfc
!       REAL*8 terfc
        real*8 erfct(*)
        real*8 xval,rem,val0,val1,val2,d1,d2,del,erfcx
        integer ixval,pts
        REAL*8 x(3), basis(3,3), recbasis(3,3), alpha, vol, tol
        REAL*8 xE(3,*),ZE(*)
        integer nE
        REAL*8  qsccewc(*),qsccews(*)
        REAL*8 potential
        REAL*8 reciprocal,rspace,cterm
        REAL*8 r(3)
        integer ireal 
        REAL*8 G(3),rh(3),help,norm,lastshell
        REAL*8 MPI
        INTEGER nrezi, nreal, nmax, nmin
        INTEGER i,j,k,icrap
!       --------------------------------------------------------- 
!       integer maxkscc 
!       parameter(maxkscc=5000)
!       real*8 kxvec(maxkscc)
!       real*8 kyvec(maxkscc)
!       real*8 kzvec(maxkscc)
!       real*8 kvec (maxkscc)
!       integer nkvec
!       common /sccewk/ kxvec,kyvec,kzvec,kvec,nkvec
!       SCC Nonbond 
!       real*8  sccfnb  
!       logical qsccnb,qsccs,qsccsh
!       common /scccut/qsccnb,sccfnb,qsccs,qsccsh


        MPI = 3.14159265358979323846 
        nmax = 20
        nmin = 2

        reciprocal = 0.0
        CALL ksumscc(x,kxvec,kyvec,kzvec,kvec,nkvec,
     $               qsccewc,qsccews,reciprocal)

!       write(*,*) "DEBUG>PHISCC>X,RECI "x(1),x(2),x(3),reciprocal
 667    CONTINUE 
!       evaluate  real space term (sum over R)   
!       /* sum over R until tolerance is reached */
        potential = 0.0

!       goto 666
        DO ireal=1,nE
          r(1)=x(1)-xE(1,ireal)
          r(2)=x(2)-xE(2,ireal)
          r(3)=x(3)-xE(3,ireal)

        nreal = 0
        rspace = 0.0
         lastshell = tol+1d-8  
        DO WHILE ((nreal .le. nmax) .and. ((nreal .le. nmin) 
     &            .or. (abs(lastshell) .gt.  tol)))
         lastshell = 0.0  
         DO i=-nreal,nreal
          DO j=-nreal,nreal
           DO k=-nreal,nreal
!            /*only R belonging to outer shells are new ones */
             IF((nreal .eq. abs(i)) .or. (nreal .eq. abs(j)) .or.
     &         (nreal. eq. abs(k)) ) THEN
                rh(1)=r(1)-(i*basis(1,1)+j*basis(2,1)+k*basis(3,1))
                rh(2)=r(2)-(i*basis(1,2)+j*basis(2,2)+k*basis(3,2))
                rh(3)=r(3)-(i*basis(1,3)+j*basis(2,3)+k*basis(3,3))
                norm=sqrt(rh(1)*rh(1)+rh(2)*rh(2)+rh(3)*rh(3))
                IF (norm .gt. 1.0d-20) THEN 
!               erfc=1-erf   
!                 help   = terfc(alpha*norm)/norm
                  xval = norm*alpha*del
                  ixval =int(xval+0.5)
                  rem = xval-ixval
                  ixval=ixval+2
                  ixval=min(ixval,pts-1)
                  VAL0 = ERFCT(IXVAL-1)
                  VAL1 = ERFCT(IXVAL)
                  VAL2 = ERFCT(IXVAL+1)
                  D1 = (VAL0-VAL2)*0.5
                  D2 = (VAL1+VAL1-VAL0-VAL2)*REM
                  ERFCX = VAL1-(D1+0.5*D2)*REM
                  help=erfcx/norm
                  rspace = rspace + help
                  lastshell = lastshell + help
                ELSE 
                   lastshell = tol+1d-8  
              END IF
             END IF
            END DO
           END DO
          END DO
         nreal = nreal + 1
        END DO

!       write(*,*) "PHISCC> NREAL:",nreal

!       stop if tolerance not reached
        IF ( abs(lastshell)  .gt. tol ) THEN
         STOP "tolerance in phi not reached in real space"     
        END IF
        
        potential = potential + rspace*ZE(ireal) 

        ENDDO  ! ireal
 666    CONTINUE 

!       evaluate constant term pi/(Omega*alpha^2)  ---> FOR EACH PAIR
        cterm = 0.0 
        do ireal=1,nE
          cterm =cterm -MPI/(vol*alpha*alpha)*ZE(ireal)
        enddo

!       set the value of the potential
        potential = potential + reciprocal + cterm
!       write(*,'(1x,"PHI> ",4F10.6)') alpha,reciprocal,rspace,cterm
        return
        END 
 
!  =====================================================================
!  evaluation of the derivative of the potential phi 
!
!   INPUT Parameter:
!   REAL*8 r(3)           position of evaluation of potential
!   REAL*8 basis(3,3)     basis of cell
!   REAL*8 recbasis(3,3)      basis of reciprocal cell
!   REAL*8 alpha          convergence parameter
!   REAL*8 vol            cell volume
!   REAL*8 tol            tolerance for convergence of "last shell"
!                       (can often be used as a criterion for global
!                        convergence) 
!  =====================================================================
 
        subroutine phi1scc(x,nn,qmat,qzero,izp,xE,nE,zE,
     $      qsccewc,qsccews,qscqewc,qscqews,
     $      erfct,del,pts,
     $      basis,alpha,tol,gr,DXTBMM,DYTBMM,DZTBMM)
        use sccdftb, only: kxvec,kyvec,kzvec,kvec,nkvec
        use sccdftbsrc, only: qsccnb,sccfnb,qsccs,qsccsh,maxtyp !MG_UW1210 (ldep)
        use parallel

        IMPLICIT NONE
!       external terfc
!       REAL*8 terfc 
        real*8 erfct(*)
        real*8 xval,rem,val0,val1,val2,d1,d2,del,erfcx,drfc
        integer ixval,pts
!       integer MXCHMPT 
!       parameter(MXCHMPT=25120)
        integer nn,nE,izp(*)
        REAL*8 x(3,*),xE(3,*),zE(*)
        REAL*8 qmat(*),qzero(maxtyp,4) !MG_UW1210 (ldep)
        REAL*8 qsccewc(*),qsccews(*),qscqewc(*),qscqews(*)
        REAL*8 gr(3,*),DXTBMM(*),DYTBMM(*),DZTBMM(*)

        REAL*8 basis(3,3), alpha
        integer istart,ifinish
!       ---------------------------------------------------
        REAL*8 r(3)
        REAL*8 reciprocal(3),rspace(3),MPI 
        REAL*8 rh(3),norm,help,tol,lastshell 
        INTEGER i,j,k,l, iqm,imm, izpj, nreal, nmax, nmin ,icrap
!       ---------------------------------------------------
!       integer maxkscc 
!       parameter(maxkscc=5000)
!       real*8 kxvec(maxkscc)
!       real*8 kyvec(maxkscc)
!       real*8 kzvec(maxkscc)
!       real*8 kvec (maxkscc)
!       integer nkvec
!       common /sccewk/ kxvec,kyvec,kzvec,kvec,nkvec
!       SCC Nonbond 
!       real*8  sccfnb  
!       logical qsccnb,qsccs,qsccsh
!       common /scccut/qsccnb,sccfnb,qsccs,qsccsh
!       ---------------------------------------------------

        nmax = 20
        nmin = 2
        MPI =  3.14159265358979323846

!       goto 667
!       ...... First do reciprocal contribution of all MM to QM ...... 
        do i=1,nn
          izpj=izp(i)
          do l=1,3
           reciprocal(l) = 0.0  
          enddo
          call kfsumscc(x(1,i),kxvec,kyvec,kzvec,kvec,nkvec,
     $         qsccewc,qsccews,reciprocal,1)
          do l=1,3
            gr(l,i)=gr(l,i) - reciprocal(l)*(qmat(i)-qzero(izpj,4))
          enddo
        enddo

!       ...... Next do reciprocal contribution of all QM to MM ...... 
#if KEY_PARALLEL==1
        istart=1 + immlsc(mynod)
        ifinish=   immlsc(mynodp)
#else /**/
        istart=1
        ifinish=nE
#endif 
!       do i=1,nE
        do i=istart,ifinish
          do l=1,3
            reciprocal(l) = 0.0  
          enddo
          call kfsumscc(xE(1,i),kxvec,kyvec,kzvec,kvec,nkvec,
     $         qscqewc,qscqews,reciprocal,0)
!         do l=1,3
!           DFTBMM(i,l)=DFTBMM(i,l) + reciprocal(l)*zE(i)
            DXTBMM(i)=DXTBMM(i) + reciprocal(1)*zE(i)
            DYTBMM(i)=DYTBMM(i) + reciprocal(2)*zE(i)
            DZTBMM(i)=DZTBMM(i) + reciprocal(3)*zE(i)
!         enddo
        enddo
!       LOOKS LIKE THE DFTBMM IN OTHER NODES ARE ZERO, SO WE CAN SIMPLY
!       COMBINE TH RESULTS TOGETHER
#if KEY_PARALLEL==1
!       CALL GCOMB(DFTBMM(1,1),nE)
!       CALL GCOMB(DFTBMM(1,2),nE)
!       CALL GCOMB(DFTBMM(1,3),nE)
        CALL GCOMB(DXTBMM,nE)
        CALL GCOMB(DYTBMM,nE)
        CALL GCOMB(DZTBMM,nE)
#endif /*         */
 
  667   CONTINUE 
!       ...... Now loop over QM/MM pairs for the real space part ...... 
!       NOT PARALLELED 
!       goto 669
!       /* evaluate  real space term (sum over R) */
!       /* sum over R until tolerance is reached */
       do imm=1,nE
!      do imm=istart,ifinish
        do iqm=1,nn
          izpj=izp(iqm)
            do l=1,3
              r(l)=x(l,iqm) - xE(l,imm)
              rspace(l) = 0.0  
            enddo

          nreal = 0
          lastshell = tol+1d-8
          DO WHILE ((nreal .le. nmax) .and. ((nreal .le. nmin)
     &              .or. (abs(lastshell) .gt.  tol)))
            lastshell = 0.0
            DO i=-nreal,nreal
             DO j=-nreal,nreal
              DO k=-nreal,nreal
!            /*only R belonging to outer shells are new ones */
              IF((nreal .eq. abs(i)) .or. (nreal .eq. abs(j)) .or.
     &           (nreal. eq. abs(k)) ) THEN
              rh(1)=r(1)-(i*basis(1,1)+j*basis(2,1)+k*basis(3,1)) 
              rh(2)=r(2)-(i*basis(1,2)+j*basis(2,2)+k*basis(3,2)) 
              rh(3)=r(3)-(i*basis(1,3)+j*basis(2,3)+k*basis(3,3)) 
 
              norm=sqrt(rh(1)*rh(1)+rh(2)*rh(2)+rh(3)*rh(3)) 
 
!             help = (-2/sqrt(MPI)*exp(-alpha*alpha*norm*norm)*
!    &             alpha*norm - terfc(alpha*norm))/(norm*norm*norm) 
            xval = norm*alpha*del
            ixval =int(xval+0.5)
            rem = xval-ixval
            ixval=ixval+2
            ixval=min(ixval,pts-1)
            val0 = erfct(ixval-1)
            val1 = erfct(ixval)
            val2 = erfct(ixval+1)
            d1 = (val0-val2)*0.5
            d2 = (val1+val1-val0-val2)*REM
            ERFCX = VAL1-(D1+0.5*D2)*REM
!           help=erfcx/norm
            drfc = (d1+d2)*alpha*del
            help=-(drfc*norm+erfcx)/norm/norm/norm
 
              rspace(1) = rh(1)*help + rspace(1)  
              rspace(2) = rh(2)*help + rspace(2)
              rspace(3) = rh(3)*help + rspace(3)
   
              lastshell = lastshell + help
              END IF
            END DO ! k
           END DO ! j
          END DO ! i
          nreal = nreal + 1
!         QC: the same here:if we have cutoff, limit to the primary cell
          if (qsccnb) goto 669
         END DO ! real sum

!         stop if tolerance not reached
          IF ( abs(lastshell)  .gt. tol ) THEN
           STOP "tolerance in phi1 not reached in real space"
          END IF

 669      CONTINUE 

!         do l=1,3
!           rspace(l) = rspace(l)*(qmat(iqm)-qzero(izpj),4)*zE(imm)
!           gr(l,iqm) = gr(l,iqm) - rspace(l)
!           DFTBMM(imm,l)=DFTBMM(imm,l) + rspace(l)

             rspace(1) = rspace(1)*(qmat(iqm)-qzero(izpj,4))*zE(imm)
             gr(1,iqm) = gr(1,iqm) - rspace(1)
             DXTBMM(imm)=DXTBMM(imm) + rspace(1)
             rspace(2) = rspace(2)*(qmat(iqm)-qzero(izpj,4))*zE(imm)
             gr(2,iqm) = gr(2,iqm) - rspace(2)
             DYTBMM(imm)=DYTBMM(imm) + rspace(2)
             rspace(3) = rspace(3)*(qmat(iqm)-qzero(izpj,4))*zE(imm)
             gr(3,iqm) = gr(3,iqm) - rspace(3)
             DZTBMM(imm)=DZTBMM(imm) + rspace(3)

!         enddo

        enddo ! iqm
        enddo ! imm 

        END 


