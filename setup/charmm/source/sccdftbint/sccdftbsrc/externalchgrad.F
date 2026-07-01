!      subroutine externalchgrad(nn,x,izp,gr,qmat,
       subroutine externalchgrad(nn,x,    gr,     
     $                           qsccewc,qsccews,qscqewc,qscqews,
     $                           erfct,del,pts)
!      QC: since we worry about both gradients, it's not possible to
!      sum over ALL mm atoms as we did in externalshift.f

       use sccdftbsrc, izp=>izp2
       use sccdftb, xe=>cptc, ze=>zptc, ext=>extflag,ne=>nptc

       implicit REAL*8 (A-H,O-Z)
!      include 'maxima.inc'
       integer n !,nE,izp(NNDIM)
       integer izpj ! MG_QC_UW1206 KO
       real*8 x(3,NNDIM)
       real*8 qsccewc(*),qsccews(*)
       real*8 qscqewc(*),qscqews(*)
       real*8  erfct(*)
       real*8 xval,rem,val0,val1,val2,d1,d2,del,erfcx,drfc
       integer ixval,pts

!      integer MXCHMPT
!      parameter(MXCHMPT=25120)
!      real*8 xE(3,MXCHMPT ),ZE(MXCHMPT )

       real*8 dif(3),r,r2,gr(3,NNDIM),dgr

!      real*8  qmat(NNDIM)
!      character*2 EXT
!      common /mcharge/ qzeroscc(MAXTYP), uhubb(MAXTYP)
!      integer mysccid
!      common /extchr/ xE, ZE, nE, EXT,mysccid
!      QC: we add the counter force on the point charges from the
!      SCCTB atoms
!      REAL*8 DFTBMM
!      COMMON/CHRMTB/DFTBMM(MXCHMPT,3)
!      --------------------------------------------------------------
!      QC_UW_04: Add ewald potential if periodic
!      logical period
!      common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, 
!    $              nlat(3)     
       REAL*8 recbasis(3,3), vol,long_deriv(3)
!      If choose to optimize para
!      logical LSCOEWD 
!      real*8 kappascc
!      integer kmxscc,ksqmxscc
!      common /sccewd/ kappascc,kmxscc,ksqmxscc,LSCOEWD
!      --------------------------------------------------------------
!      QC_UW_04: Add non-bond cut in eWald 
! 
!      integer MAXPTR
!      parameter(MAXPTR=5000)
!      REAL*8 CPTR,ZPTR
!      INTEGER NPTR
!      INTEGER IMM2LST 
!      COMMON/EXTCH2/CPTR(3,MAXPTR),ZPTR(MAXPTR),NPTR,IMM2LST(MAXPTR)
!      REAL*8 DFTBM2
!      LOGICAL LSKIMG
!      COMMON/CHRMT2/DFTBM2(MAXPTR,3),LSKIMG
!
!     ----------------------------------------------------------------
!     SCC Nonbond
!     real*8  sccfnb
!     logical qsccnb,qsccs,qsccsh
!     common /scccut/qsccnb,sccfnb,qsccs,qsccsh
!      --------------------------------------------------------------

! MG+Guanhua_QC_UW1206:KO
      real*8 uhub,kaltmp,kbetmp
      real*8 exptmp,uhubre,qcharge,rtmp,mmuhubre

      real*8 MPI 
      real*8 sccftmp 
      MPI=3.14159265358979323846
      sccftmp=sccfnb/0.529177249d0

       do j = 1,nn
!        write(*,*) "exterchrgd> qmat: ",j,qmat(j)
         do i=1,3
           gr(i,j) = 0.0 
         enddo
       enddo

       if (EXT.eq.'CH') then
       if (period) then
!       ===================== QC: Ewald summation=================

!      get reciprocal lattice vectors and cell volume
       CALL REZVOL(boxsiz,recbasis,vol)
       alpha = getalpha(boxsiz)

       if (LSCOEWD) then
!      ...... Optimize everything, iterate till convergence ......
!      ...... QC: If parallel, only headnode does this (for now) ...... 
       if(mysccid.ne.0) return
       do j = 1,nn  
        izpj = izp(j)
        do k =  1,nE
          r2 = 0.0
          do i = 1,3 
             dif(i) = x(i,j) - xE(i,k)
             r2 = r2 + dif(i)**2
!            write(*,*) x(i,j), xE(i,k)
          enddo 
          r=sqrt(r2)
!           set tolerance for convergence
            tol = 1.0d-8
            CALL PHI1(dif,boxsiz,recbasis,alpha,vol,tol,long_deriv,
     $                erfct,del,pts)
!           do l=1,3

              dgr = long_deriv(1)*(qmat(j)-qzeroscc(izpj,4))*ZE(k) !MG_UW1210 (ldep)
              gr(1,j)=gr(1,j) - dgr
              DXTBMM(k)=DXTBMM(k) + dgr 

              dgr = long_deriv(2)*(qmat(j)-qzeroscc(izpj,4))*ZE(k)
              gr(2,j)=gr(2,j) - dgr
              DYTBMM(k)=DYTBMM(k) + dgr 

              dgr = long_deriv(3)*(qmat(j)-qzeroscc(izpj,4))*ZE(k)
              gr(3,j)=gr(3,j) - dgr
              DZTBMM(k)=DZTBMM(k) + dgr 

!           enddo
         enddo !k
        enddo  !j
        else
!       -------------------- QC: Ewald with specified para ----------
!       ...... FIRST compute the sum over QM atoms of contributions
!       ...... to the ewald force
!       CALL WRTTIM('BEFORE SCQEWCS')
        if(.not.qsccpme) then ! PZ
                              ! not needed for PME method anymore
        call SCQEWCS(QSCQEWC,QSCQEWS,qzeroscc,izp)
        endif ! .not.qsccpme
!       CALL WRTTIM('AFTER  SCQEWCS')
        if (.not.qsccnb) then
!       ...... include everything in real space sum, iterate till conv
!         do j = 1,nn  
!           izpj = izp(j)
!           do k =  1,nE
!            r2 = 0.0
!            do i = 1,3 
!               dif(i) = x(i,j) - xE(i,k)
!               r2 = r2 + dif(i)**2
!            enddo 
!            r=sqrt(r2)
!            CALL PHI1SCC(dif,boxsiz,recbasis,alpha,vol,tol,long_deriv)
             tol = 1.0d-8
             CALL PHI1SCC(x,nn,qmat,qzeroscc,izp,xE,nE,zE,
     $       qsccewc,qsccews,qscqewc,qscqews,
     $       erfct,del,pts,
     $       boxsiz,alpha,tol,gr,DXTBMM,DYTBMM,DZTBMM)
!            do l=1,3
!             dgr = long_deriv(l)*(qmat(j)-qzeroscc(izpj),4)*ZE(k) !MG_UW1210 (ldep)
!             gr(l,j)=gr(l,j) - dgr
!             DFTBMM(k,l)=DFTBMM(k,l) + dgr 
!           enddo
!          enddo !k
!         enddo  !j
        else
!         ...... include all reciprocal space terms ......
!         do j = 1,nn  
!           izpj = izp(j)
!           do k =  1,nE
!            r2 = 0.0
!            do i = 1,3 
!               dif(i) = x(i,j) - xE(i,k)
!            enddo 
!            CALL PHISCC2F(dif,recbasis,alpha,vol,long_deriv)
!            CALL WRTTIM('BEFORE PHISCC2')
             CALL PHISCC2F(x,nn,qmat,qzeroscc,izp,xE,nE,zE,
     $       qsccewc,qsccews,qscqewc,qscqews,
     $       gr,DXTBMM,DYTBMM,DZTBMM)
!            CALL WRTTIM('AFTER  PHISCC2')
!            do l=1,3
!             dgr = long_deriv(l)*(qmat(j)-qzeroscc(izpj,4))*ZE(k) !MG_UW1210 (ldep)
!             gr(l,j)=gr(l,j) - dgr
!             DFTBMM(k,l)=DFTBMM(k,l) + dgr 
!           enddo
!          enddo !k
!         enddo  !j
!TMP      ...... do the real space terms ......
!         goto 770
!         write(*,*) "NPTR: >",NPTR
!         QC: slavenodes do not have to compute real part
          if (mysccid.ne.0) return
!         CALL WRTTIM('BEFORE REALL  ')
          do j = 1,nn  
           izpj = izp(j)
           do k =  1,NPTR
            r2 = 0.0
            do i = 1,3 
               dif(i) = x(i,j) - CPTR(i,k)
               r2 = r2 + dif(i)**2
            enddo 
            r=sqrt(r2)
            if (r.le.sccftmp) then
! Xiao:
            xval = r*alpha*del
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
            drfc = (d1+d2)*alpha*del
            gamma=-(drfc*r+erfcx)/r/r/r

!             gamma=(-2.d0/sqrt(MPI)*exp(-alpha*alpha*r*r)*
!    &               alpha*r - terfc(alpha*r))/(r*r*r)

!             do l=1,3 
                dgr = dif(1)*gamma*
     $              (qmat(j)-qzeroscc(izpj,4))*ZPTR(k) !MG_UW1210 (ldep)
                gr(1,j) = gr(1,j) - dgr
                DXTBM2(k)=DXTBM2(k) + dgr 

                dgr = dif(2)*gamma*
     $              (qmat(j)-qzeroscc(izpj,4))*ZPTR(k)
                gr(2,j) = gr(2,j) - dgr
                DYTBM2(k)=DYTBM2(k) + dgr 

                dgr = dif(3)*gamma*
     $              (qmat(j)-qzeroscc(izpj,4))*ZPTR(k)
                gr(3,j) = gr(3,j) - dgr
                DZTBM2(k)=DZTBM2(k) + dgr 

!             enddo 

            endif!within cut
           enddo !k
           enddo  !j
  770      CONTINUE 
!         CALL WRTTIM('AFTER  REALL  ')
        endif ! eWald non-bond
        endif ! eWald option

!       ===================== QC: Normal case (no PBC) ---------------
        else 
!       QC_Haibo_UW_041705: we need to modify this for PARASCC - i.e. 
!       when slave nodes do DIFFERENT SCC job as the master
#if KEY_PARASCC==0
         if (mysccid.ne.0) return 
#endif 
         do j = 1,nn  
           izpj = izp(j)
           do k =  1,nE
            r2 = 0.0
            do i = 1,3 
               dif(i) = x(i,j) - xE(i,k)
               r2 = r2 + dif(i)**2
!              write(*,*) x(i,j), xE(i,k)
            enddo 
            r=sqrt(r2)

! MG+Guanhua_QC_UW1206: KO
            if (lcdko) then
              kaltmp=kalpha(izpj)
              kbetmp=kbeta(izpj)
              qcharge=qmat(j)-qzeroscc(izpj,4) !MG_UW1210 (ldep)
              uhub=uhubb(izpj,1)+uhder(izpj,1)*qcharge !MG_UW1210 (ldep)
              rtmp=dsqrt(r2)
              exptmp=dexp(-kbetmp*rtmp)
              uhubre=1.0d0/uhub
! Guanhua: add MM contributions
              if (nmmtype .ne. 0) then
              if (mmuhub(k).gt.0.0000001) then
                mmuhubre=1.0d0/mmuhub(k)
              else
                mmuhubre=0.0d0
              endif
              gamma=1.0d0/sqrt(r2+kaltmp*(uhubre+mmuhubre)**2*exptmp)
!              do l=1,3
                dgr = dif(1)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $            *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*(uhubre+mmuhubre)**2
     $            *exptmp)-dif(1)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(1,j) = gr(1,j) + dgr
                DXTBMM(k)=DXTBMM(k) - dgr

                dgr = dif(2)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $            *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*(uhubre+mmuhubre)**2
     $            *exptmp)-dif(2)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(2,j) = gr(2,j) + dgr
                DYTBMM(k)=DYTBMM(k) - dgr

                dgr = dif(3)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $            *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*(uhubre+mmuhubre)**2
     $            *exptmp)-dif(3)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(3,j) = gr(3,j) + dgr
                DZTBMM(k)=DZTBMM(k) - dgr
!              enddo
              else
              gamma=1.0d0/dsqrt(r2+kaltmp*uhubre**2*exptmp)
!              do l=1,3
                dgr = dif(1)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $             *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*uhubre**2
     $             *exptmp)-dif(1)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(1,j) = gr(1,j) + dgr
                DXTBMM(k)=DXTBMM(k) - dgr

                dgr = dif(2)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $             *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*uhubre**2
     $             *exptmp)-dif(2)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(2,j) = gr(2,j) + dgr
                DYTBMM(k)=DYTBMM(k) - dgr

                dgr = dif(3)*(gamma**3)*qcharge*ZE(k)*sccg(r2)
     $             *(1.0d0-kaltmp*kbetmp/2.0d0/rtmp*uhubre**2
     $             *exptmp)-dif(3)*gamma*qcharge*ZE(k)*dsccg(r2)/rtmp
                gr(3,j) = gr(3,j) + dgr
                DZTBMM(k)=DZTBMM(k) - dgr
!              enddo
              endif
            else

!           uhub=uhubb(izp(j),1) !MG_UW1210 (ldep)
!           gamma= gamE(r2,uhubb(izpj,1),uhubb(izpj),1) !MG_UW1210 (ldep)
!           gamma = 1.0/sqrt(r2 + (0.5/uhub + 0.5/uhub)**2)

            gamma=1.0d0/r
!           do l=1,3 
!             dgr = dif(l)*(gamma**3)*
!    $              (qmat(j)-qzeroscc(izpj,4))*ZE(k) !MG_UW1210 (ldep)
!           enddo 


              dgr = dif(1)*(gamma**3)*
     $              (qmat(j)-qzeroscc(izpj,4))*ZE(k)*sccg(r2)-   !MG_UW1210 (ldep)
     $ dif(1)*(gamma**2)*(qmat(j)-qzeroscc(izpj,4))*ZE(k)*dsccg(r2)
              gr(1,j) = gr(1,j) + dgr
              DXTBMM(k)=DXTBMM(k) - dgr 

              dgr = dif(2)*(gamma**3)*
     $              (qmat(j)-qzeroscc(izpj,4))*ZE(k)*sccg(r2)-
     $ dif(2)*(gamma**2)*(qmat(j)-qzeroscc(izpj,4))*ZE(k)*dsccg(r2)
              gr(2,j) = gr(2,j) + dgr
              DYTBMM(k)=DYTBMM(k) - dgr 

              dgr = dif(3)*(gamma**3)*
     $              (qmat(j)-qzeroscc(izpj,4))*ZE(k)*sccg(r2)-
     $ dif(3)*(gamma**2)*(qmat(j)-qzeroscc(izpj,4))*ZE(k)*dsccg(r2)
              gr(3,j) = gr(3,j) + dgr
              DZTBMM(k)=DZTBMM(k) - dgr 
            endif
           enddo !k
          enddo  !j
      endif ! CHARMM

!   external field (QC: didn't care)
      else 
        do j = 1,nn
         izpj = izp(j)
          do i=1,3 
           dgr= -(qmat(j)-qzeroscc(izpj,4))*ZE(i) !MG_UW1210 (ldep)
           gr(i,j) = gr(i,j) + dgr
          enddo
         enddo
       endif
      end

      REAL*8 FUNCTION SCCG(RQMMM2)
      use sccdftbsrc, only: qsccnb,sccfnb,qsccs,qsccsh
      implicit none
!     function to return the scaling factor for QM charge
!     At the moment we do not use switch -- people say it's poor!
!     now we just focus on" atom based shift."
!     CSHFT: "Energy-based" Shift
!     CSHIFT:"Force-based"  Shift 
      REAL*8 RQMMM2,g2,g3,S,C2ROF2,CHROF2,RECOF2,MIN2OF,R1,C2OFNB
      logical CSHFT, CSHIFT
      real*8 mintwo,half,one,rsmall,ctofnb,r2
      parameter (one=1.0d0,half=0.5d0,rsmall=1.0d-10,mintwo=-2.0d0)

!      real*8  sccfnb
!      logical qsccnb,qsccs,qsccsh
!      common /scccut/qsccnb,sccfnb,qsccs,qsccsh


      if(qsccnb) then
      cshft=qsccs
      cshift=qsccsh
      ctofnb =sccfnb/(0.529177249d0)

      s=RQMMM2
      IF (S .LT. RSMALL) S=RSMALL
      C2OFNB=CTOFNB*CTOFNB

         if(s.lt.c2ofnb) then
      if(cshft) then
      C2ROF2=MINTWO/C2OFNB
      CHROF2=-HALF/C2OFNB
      G2=S*C2ROF2
      G3=G2*S*CHROF2
         sccg=1.0d0+G2+G3
      else if(cshift) then
      s=RQMMM2
      r2=1.0d0/s
      r1=dsqrt(r2)
      RECOF2 = ONE/C2OFNB
      MIN2OF = MINTWO/CTOFNB
         sccg=1.0d0 + S*(RECOF2 + MIN2OF*R1)
      endif
!     WRITE(*,*) "SCMMCG> ",CSHFT,CSHIFT,ctofnb,RQMMM2,SCMMCG
         else
         sccg=0.0d0
         endif
      else
      sccg=1.0d0
      endif
      RETURN
      END

      REAL*8 FUNCTION DSCCG(RQMMM2)
      use sccdftbsrc, only: qsccnb,sccfnb,qsccs,qsccsh
      implicit none
!     function to return the Derivative of scaling factor.
!     At the moment we do not use switch -- people say it's poor!
!     now we just focus on" atom based shift."
      REAL*8 RQMMM2,g2,g3,S,C2ROF2,CHROF2,RECOF2,MIN2OF,R1,C2OFNB
      logical CSHFT, CSHIFT
      real*8 mintwo,half,one,rsmall,ctofnb,r2
      parameter (one=1.0d0,half=0.5d0,rsmall=1.0d-10,mintwo=-2.0d0)

!      real*8  sccfnb
!      logical qsccnb,qsccs,qsccsh
!      common /scccut/qsccnb,sccfnb,qsccs,qsccsh


      if(qsccnb) then
      cshft=qsccs
      cshift=qsccsh
      ctofnb =sccfnb/(0.529177249d0)

      s=RQMMM2
      r2=1.0d0/s
      r1=dsqrt(r2)

      IF (S .LT. RSMALL) S=RSMALL
      C2OFNB=CTOFNB*CTOFNB

         if(s.lt.c2ofnb) then
      if(cshft) then
      C2ROF2=MINTWO/C2OFNB
      CHROF2=-HALF/C2OFNB
      G2=S*C2ROF2
      G3=G2*S*CHROF2
         dsccg=2.0d0*s*r1*C2ROF2+4.0d0*s*s*r1*C2ROF2*CHROF2
      else if(cshift) then
      RECOF2 = ONE/C2OFNB
      MIN2OF = MINTWO/CTOFNB
         dsccg=2.0d0*s*r1*RECOF2+MIN2OF
      endif
!     WRITE(*,*) "SCMMCG> ",CSHFT,CSHIFT,ctofnb,RQMMM2,SCMMCG
         else
         dsccg=0.0d0
         endif
      else
      dsccg=0.0d0
      endif
      RETURN
      END



