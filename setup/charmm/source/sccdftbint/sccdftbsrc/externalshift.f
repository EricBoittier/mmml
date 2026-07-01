!      subroutine externalshift(nn,x,izp,shiftE,qsccewc,qsccews)
       subroutine externalshift(nn,x,    shiftE,shiftE2,qsccewc,qsccews,
     $                          erfct,del,pts)
       use sccdftbsrc, izp=>izp2
       use sccdftb, xe=>cptc, ze=>zptc, ext=>extflag,ne=>nptc
       implicit REAL*8 (A-H,O-Z)
!      include 'maxima.inc'
       integer n !,nE,izp(NNDIM)
       real*8 x(3,NNDIM)
       real*8 qsccewc(*),qsccews(*)
!      xiao: add erfc for ewald
       real*8  erfct(*)
       real*8 xval,rem,val0,val1,val2,d1,d2,del,erfcx
       integer ixval,pts
! MG+Guanhua_QC_UW1005: KO
       real*8 kaltmp,kbetmp
       integer izpj
!! qmat will be passed from argument
!       real*8 qmat(NNDIM)
       real*8 exptmp,uhubre,uhubre2,mmuhubre,qcharge,uhdertmp


!      integer MXCHMPT
!      parameter(MXCHMPT=25120)
!      real*8 xE(3,MXCHMPT),ZE(MXCHMPT)
       real*8 shiftE(NNDIM),dif(3),r,r2
       real*8 shiftE2(NNDIM) ! MG+Guanhua_QC_UW1206: KO

!      character*2 EXT
!      common /mcharge/ qzeroscc(MAXTYP), uhubb(MAXTYP)
!      common /extchr/ xE, ZE, nE, EXT
!     --------------------------------------------------------------
!      integer MAXPTR
!      parameter(MAXPTR=5000)
!      REAL*8 CPTR,ZPTR
!      INTEGER NPTR
!      INTEGER IMM2LST 
!      COMMON/EXTCH2/CPTR(3,MAXPTR),ZPTR(MAXPTR),NPTR,IMM2LST(MAXPTR)
!      REAL*8 DXTBM2,DYTBM2,DZTBM2
!      COMMON/CHRMTB/DXTBM2(MAXPTR),DYTBM2(MAXPTR),DZTBM2(MAXPTR)
!     ----------------------------------------------------------------
!      SCC Nonbond
!      real*8  sccfnb
!      logical qsccnb,qsccs,qsccsh
!      common /scccut/qsccnb,sccfnb,qsccs,qsccsh
!      --------------------------------------------------------------
!      QC_UW04: Add ewald potential if periodic
!      logical period
!      common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, 
!    $              nlat(3)     

       REAL*8 recbasis(3,3), vol

!      If choose to optimize para
!      logical LSCOEWD 
!      real*8 kappascc
!      integer kmxscc,ksqmxscc
!      common /sccewd/ kappascc,kmxscc,ksqmxscc,LSCOEWD
!      --------------------------------------------------------------
       real*8 sccftmp
       sccftmp=sccfnb/0.529177249d0

       if (EXT.eq.'CH') then
        if (period) then
!         get reciprocal lattice vectors and cell volume
          CALL REZVOL(boxsiz,recbasis,vol)

!         choose good convergence parameter alpha(another option is to
!         use the alpha provided by CHARMM)
          alpha = getalpha(boxsiz)

        if (LSCOEWD) then
!       ============== Ewald potential from all MM atoms =============  
!       ...... Optimize everything, iterate ......
        do j=1,nn 
        shiftE(j) = 0.0
        shiftE2(j)= 0.0d0 ! MG_QC_UW1206 for KO
        do k=1,nE
          r2=0.0
          do i=1,3 
            dif(i) = x(i,j) - xE(i,k)
            r2=r2 + dif(i)**2
          enddo 
!         set tolerance for convergence
          tol = 1.0d-8
!         in principle can use phi!!!          
          call phi(dif,boxsiz,recbasis,alpha,vol,tol,phivalue,
     $             erfct,del,pts)
          shiftE(j) = shiftE(j) + phivalue*ZE(k)
        enddo !k
        enddo !j

        else
!       --------------------------------------------------------------
        if (.not.qsccnb) then
!       ...... Use input parameters for ewald, but enforce real space
!       ...... convergence 
          do j=1,nn 
          shiftE(j) = 0.0
          shiftE2(j) = 0.0d0 ! MG_QC_UW1206 for KO
!Q06      do k=1,nE
!Q06        r2=0.0
!Q06        do i=1,3 
!Q06          dif(i) = x(i,j) - xE(i,k)
!Q06          r2=r2 + dif(i)**2
!Q06        enddo 

!         set tolerance for convergence
            tol = 1.0d-8
!Q06        call phiscc(dif,boxsiz,recbasis,alpha,vol,tol,phivalue)
            call phiscc(x(1,j),boxsiz,recbasis,alpha,vol,tol,
     $                  xE,nE,ZE,qsccewc,qsccews,phivalue,
     $                  erfct,del,pts)
!Q06        shiftE(j) = shiftE(j) + phivalue*ZE(k)
            shiftE(j) = phivalue
!Q06      enddo !k
          enddo !j
!       --------------------------------------------------------------
        else
!         ...... do reciprocal space for ALL pt charges ......
!         ...... UW04_06: First pre-assemble terms related to 
!         ...... MM point charges ......
           

          do j=1,nn 
          shiftE(j) = 0.0
          shiftE2(j) = 0.0d0 ! MG_QC_UW1206 for KO
!Q06      do k=1,nE
!Q06        r2=0.0
!Q06        do i=1,3 
!Q06          dif(i) = x(i,j) - xE(i,k)
!Q06        enddo 
!Q06        call phiscc2a(dif,recbasis,alpha,vol,phivalue)
!Q06        shiftE(j) = shiftE(j) + phivalue*ZE(k)
!Q06      enddo !k
          if(.NOT.qsccpme) then ! PZ
            call phiscc2a(x(1,j),recbasis,alpha,vol,nE,zE,
     $                    qsccewc,qsccews,phivalue)
            shiftE(j) = phivalue
          endif
          enddo !j
          if(qsccpme) then ! PZ
          !write(*,*) "nn",nn
          !write(*,*) "xE",xE
          !write(*,*) "nE",nE
          !write(*,*) "zE",zE
          !write(*,*) "recbasis,vol,shiftE",recbasis,vol,shiftE
          !write(*,*) "ksgrd",ksgrd
          !write(*,*) "alpha",alpha
            call sccpme2a(x,nn,xE,nE,zE,recbasis,vol,shiftE,ksgrd,alpha)
          endif
!         ..... then do the real space sum over only those within
!         ..... minimum image < sccfnb ......
!         DEBUG DEBUG
!         goto 700
          do j=1,nn 
          do k=1,NPTR
            r2=0.0
            do i=1,3 
              dif(i) = x(i,j) - CPTR(i,k)
              r2=r2 + dif(i)**2
            enddo 
            gamma =  sqrt(r2)
!           if (gamma.le.sccftmp) 
!    $      shiftE(j) = shiftE(j) + terfc(alpha*gamma)*ZPTR(k)/gamma
!    Xiao
            if (gamma.le.sccftmp) then
            xval = gamma*alpha*del
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
            shiftE(j) = shiftE(j) + erfcx*ZPTR(k)/gamma
            endif

          enddo !k
          enddo !j
  700   CONTINUE 
        endif ! for nonbond cut in eWald

        endif ! for eWALD
!       ============== Normal case (No PBC) ================= 
        else
        do j=1,nn 
        shiftE(j) = 0.0d0
        shiftE2(j) = 0.0d0 ! MG+Guanhua_QC_UW1206: KO
        izpj=izp(j)

        do k=1,nE
          r2=0.0d0
          do i=1,3 
            dif(i) = x(i,j) - xE(i,k)
            r2=r2 + dif(i)**2
          enddo 


! MG+Guanhua_QC_UW1005: KO
          if (lcdko) then
            kaltmp=kalpha(izpj)
            kbetmp=kbeta(izpj)
            qcharge=qmat(j)-qzeroscc(izpj,4) !MG_UW1210 (ldep)
            uhdertmp=uhder(izpj,1)
            uhub=uhubb(izpj,1)+uhdertmp*qcharge !MG_UW1210 (ldep)
            uhubre=1.0d0/uhub
            uhubre2=uhubre*uhubre
            exptmp=dexp(-kbetmp*dsqrt(r2))
! Guanhua: add mm contribution
            if (nmmtype .ne. 0) then
              if (mmuhub(k).gt.0.0000001) then
                mmuhubre=1.0d0/mmuhub(k)
              else
                mmuhubre=0.0d0
              endif
              gamma=1.0d0/dsqrt(r2+kaltmp*(uhubre+mmuhubre)**2
     &              *exptmp)
              shiftE(j) = shiftE(j) + gamma*ZE(k)*sccg(r2)
              shiftE2(j)=shiftE2(j)
     $        +sccg(r2)*ZE(k)*gamma**3*qcharge
     $        *uhubre2*uhdertmp*(uhubre+mmuhubre)
     $        *kaltmp*exptmp
            else
              gamma=1.0d0/dsqrt(r2+kaltmp*uhubre2
     &              *exptmp)
              shiftE(j) = shiftE(j) + gamma*ZE(k)*sccg(r2)
              shiftE2(j)=shiftE2(j)
     $        +sccg(r2)*ZE(k)*gamma**3*qcharge
     $        *uhubre2*uhubre*uhdertmp*kaltmp*exptmp
            endif
          else
            gamma =  1.0d0/dsqrt(r2)
            shiftE(j) = shiftE(j) + gamma*ZE(k)*sccg(r2)
          endif
        enddo !k
        enddo !j

        endif ! for period
       endif ! for CHARMM

! end CH!, now field EF
       if (EXT.eq.'EF') then
       do i=1,nn
        shiftE(i) = 0.0
        shiftE2(j)= 0.0d0 ! MG_QC_UW1206 for KO
        do j=1,3
         shiftE(i) = shiftE(i) + ZE(j)*(x(j,i)-xE(1,j))
        enddo 
! 2* shiftE, da der gleiche Vekrtor wie fuer CH genommen.
! der wird bei Addition of shift halbiert, was
! im Falle des Feldes nicht noetig ist
         shiftE(i) =2*shiftE(i) 
       enddo
       endif 
      end

