!       ******************* Reciprocal case only ***********************

        subroutine phiscc2a(r,recbasis,alpha,vol,nE,zE,qsccewc,qsccews,
     &                      potential)

        use sccdftb, only: kxvec,kyvec,kzvec,kvec,nkvec
        IMPLICIT NONE
!       phiscc2a differs from phiscc in that we do NOT do real space
!       sum because it's done separately using cutoff

        REAL*8 r(3), recbasis(3,3), alpha, vol
        integer nE
        real*8  zE(*)
        REAL*8 qsccewc(*),qsccews(*)
        REAL*8 potential
        REAL*8 reciprocal,cterm
        REAL*8 MPI
!       ----------------------------------------------
!       integer maxkscc 
!       parameter(maxkscc=5000)
!       real*8 kxvec(maxkscc)
!       real*8 kyvec(maxkscc)
!       real*8 kzvec(maxkscc)
!       real*8 kvec (maxkscc)
!       integer nkvec
!       common /sccewk/ kxvec,kyvec,kzvec,kvec,nkvec
        integer i

        MPI = 3.14159265358979323846 
        reciprocal = 0.0

        CALL ksumscc(r,kxvec,kyvec,kzvec,kvec,nkvec,
     $               qsccewc,qsccews,reciprocal)
!       write(*,*) "DEBUG>PHISCC2>X,RECI",r(1),r(2),r(3),reciprocal

!       evaluate constant term pi/(Omega*alpha^2)  --> for each pair
        cterm      = 0.0
        do i=1,nE
          cterm = cterm -MPI/(vol*alpha*alpha)*zE(i)
        enddo

        potential = reciprocal + cterm
        RETURN
        END
 
        subroutine phiscc2f(x,nn,qmat,qzero,izp,xE,nE,zE,
     $      qsccewc,qsccews,qscqewc,qscqews,
     $      gr,DXTBMM,DYTBMM,DZTBMM)

        ! some more parameters necessary for PME (PZ): MG_QC_UW1206 
        use sccdftb, only: kxvec,kyvec,kzvec,kvec,nkvec,boxsiz,maxptc
        use sccdftbsrc, only: qsccpme,fftx,ffty,fftz,fftorder,ksgrd,
     &                  nndim,maxtyp

        use parallel
        IMPLICIT NONE
        external getalpha ! PZ
        integer nn,nE,izp(*)
!       integer MXCHMPT 
!       parameter(MXCHMPT=25120)

        REAL*8 x(3,*),xE(3,*),zE(*)
        REAL*8 qmat(*),qzero(maxtyp,4)
        REAL*8 qsccewc(*),qsccews(*),qscqewc(*),qscqews(*)
        REAL*8 gr(3,*),DXTBMM(*),DYTBMM(*),DZTBMM(*)

        integer i,j,l,izpj
        REAL*8 reciprocal(3)
        REAL*8 MPI 
!       ----------------------------------------------------------
!       integer maxkscc 
!       parameter(maxkscc=5000)
!       real*8 kxvec(maxkscc)
!       real*8 kyvec(maxkscc)
!       real*8 kzvec(maxkscc)
!       real*8 kvec (maxkscc)
!       integer nkvec
!       common /sccewk/ kxvec,kyvec,kzvec,kvec,nkvec

        integer istart,ifinish
        real*8 qmchg(nndim),ksgrd_mm(3,maxptc) ! PZ MG_QC_UW1206
        real*8 recbasis(3,3),vol,alpha,getalpha

        MPI =  3.14159265358979323846

!       ...... First do reciprocal contribution of all MM to QM ......
!TMP    TIMING
!       CALL WRTTIM('BEFORE KFSUM')
        if(qsccpme) then ! PZ
        !write(6,*) "invoked qsccpme"
        do i=1,nn
          izpj=izp(i)
          do l=1,3
            gr(l,i)=gr(l,i) - ksgrd(3*i+l-3)*(qmat(i)-qzero(izpj,4))
          enddo
        enddo
! ASC: Add Parallel - for QM gradient, because ksgrd is not summed up
#if KEY_PARALLEL==1
        CALL GCOMBR(gr,nn*3)
#endif
        else ! if qsccpme 
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
        endif ! if qsccpme 
!       CALL WRTTIM('AFTER  KFSUM')

!       ...... Next do reciprocal contribution of all QM to MM ......
#if KEY_PARALLEL==1
        istart=1 + immlsc(mynod)
        ifinish=   immlsc(mynodp)
#else /**/
        istart=1
        ifinish=nE
#endif 

        if(qsccpme) then ! PZ
          CALL REZVOL(boxsiz,recbasis,vol)
          alpha=getalpha(boxsiz)
          do i=1,nn
            izpj=izp(i)
            qmchg(i)=qmat(i)-qzero(izpj,4)
          enddo
          call do_pme_ksp_grad_mm(nn,nE,x,xE,zE,qmchg,ksgrd_mm,recbasis
     &                            ,vol,alpha)
          do i=1,nE
            DXTBMM(i) = DXTBMM(i) - ksgrd_mm(1,i)*zE(i)
            DYTBMM(i) = DYTBMM(i) - ksgrd_mm(2,i)*zE(i)
            DZTBMM(i) = DZTBMM(i) - ksgrd_mm(3,i)*zE(i)
          enddo
        else ! if qsccpme
!       do i=1,nE
        do i=istart,ifinish
          do l=1,3
            reciprocal(l) = 0.0
          enddo
!       ...... Do the vector differently ...... 
!       ...... But notice that not all nodes know about the vectors 
!       ...... Has to be careful in 
          call kfsumscc(xE(1,i),kxvec,kyvec,kzvec,kvec,nkvec,
     $         qscqewc,qscqews,reciprocal,0)
!         do l=1,3
!           DFTBMM(i,l)=DFTBMM(i,l) + reciprocal(l)*zE(i)
            DXTBMM(i)=DXTBMM(i) + reciprocal(1)*zE(i)
            DYTBMM(i)=DYTBMM(i) + reciprocal(2)*zE(i)
            DZTBMM(i)=DZTBMM(i) + reciprocal(3)*zE(i)
!         enddo
        enddo
        endif ! if qsccpme ! PZ

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
!       CALL WRTTIM('AFTER  KFSUM2')

        RETURN
        END 


