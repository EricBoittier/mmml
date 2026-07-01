!=======================================================================
!      get qdiff and shift-vectors 
!      (necessary for Hubbard contribution to H matrix elements, total 
!       energy and forces)
!
!      INPUT:
!      integer     nn                     number of atoms (in one cell)
!      real*8      qmat(NNDIM)            mulliken charges 
!      real*8      qzero(MAXTYP,4)        neutral atomic/ldep charges
!      integer     izp(NNDIM)             map from atoms to atom types
!      real*8      gammamat(NNDIM,NNDIM,3,3)  gamma_{ij} for every atom 
!                                         pairing i and j (includes 
!                                         Ewald+shortrange if  
!                                         periodicity is switched on)
!      real*8      gammader(NNDIM,NNDIM,3,3)  Gamma_{ij)=dU_{ij}/dq_i for 
!                                         every atom pairing
!      integer     indl(NNDIM+1)          index for l-dependent charges
!
!      OUTPUT:
!      real*8  qdiff(NNDIM)   net charge of atoms
!      real*8  shift(NNDIM,3)   shift(i) = \sum_{j}\Delta q_j \gamma_{ij}
!      real*8  shift3(NNDIM,3)  shift3(i) = \sum_{j}\Delta q_i \Delta q_j
!                                          \Gamma_{ij}
!      real*8  shift3A(NNDIM,3) shift3A(i)= \sum_{j}\Delta q_j \Delta q_j
!                                          \Gamma_{ji}
!      real*8  spinshift(NNDIM,3)  ...(i,li)=sum_{l_i}
!
!  for l-dependence output is slightly different:
!      real*8  qdiff(NNDIM),qldiff(NNDIM) atomic and l-dep net charges
!      real*8  shift
!      real*8  shift3  
!      real*8  shift3A  
!      real*8  shift3B  


!======================================================================

! MG+Guanhua_QC_UW1205: substantial changes
! MG_QC_UW1207: spin-polarization added; lcolspin
! MG_UW1210: added ldep
       SUBROUTINE HAMILSHIFT(nat,izp,gammamat,gammader,
     &      qmat,qzero,shift,shift3,shift3A,shift3B,indl,spinshift)
       use sccdftbsrc, only: nndim,maxtp,qdiff,lscc3rd,lcolspin,qlup,
     &                       qldown,wspin,lmax,lldep,ql !MG_UW1210 (ldep)
       IMPLICIT NONE
!      INCLUDE 'maxima.inc'
       integer nat,i,j,li,lj,indlj
       integer izp(NNDIM),indl(NNDIM+1)
       real*8 shift(NNDIM,3),qmat(NNDIM),qzero(MAXTP,4)
       real*8 shift3(NNDIM,3),shift3A(NNDIM,3),spinshift(NNDIM,3)
       real*8 shift3B(NNDIM),qldiff(NNDIM,3)
       real*8 gammamat(NNDIM,NNDIM,3,3),gammader(NNDIM,NNDIM,3,3)
       
!  zero shift-vectors
       do i=1,nat
         shift3B(i) = 0.0d0
         do li=1,3
           shift(i,li)   = 0.0d0
           shift3(i,li)  = 0.0d0
           shift3A(i,li) = 0.0d0
           spinshift(i,li)=0.0d0
         enddo
       enddo

       if (lldep) then
! get qldiff 
         do i=1,nat
           qdiff(i) = qmat(i) - qzero(izp(i),4)
           do li=1,lmax(izp(i))
             qldiff(i,li) = ql(indl(i)+li) - qzero(izp(i),li)
           enddo
         enddo

! shift for 2nd order contributions
         do i=1,nat
         do j=1,nat
         do li=1,lmax(izp(i))
         do lj=1,lmax(izp(j))
           shift(i,li) = shift(i,li) + qldiff(j,lj)*gammamat(i,j,li,lj)
         enddo
         enddo
         enddo
         enddo

! shift for 3rd order contributions
         if (lscc3rd) then
           do i=1,nat
           do li=1,lmax(izp(i))
           do j=1,nat
           do lj=1,lmax(izp(j))
            shift3(i,li) =shift3(i,li) +qldiff(j,lj)*gammader(i,j,li,lj)
            shift3A(i,li)=shift3A(i,li)+qldiff(j,lj)*gammader(j,i,lj,li)
     &                                 *qdiff(j)
            shift3B(i)   =shift3B(i)   +qldiff(j,lj)*gammader(i,j,li,lj)
     &                                 *qldiff(i,li)
           enddo
           enddo
            shift3(i,li)  = shift3(i,li) * qdiff(i)
           enddo
           enddo
         endif

! if not ldep
       else 
! get qdiff
       do i=1,nat
         qdiff(i) = qmat(i) - qzero(izp(i),4)
       enddo

!  shift for 2nd order contributions
       do i=1,nat
         do j=1,nat
           shift(i,1) = shift(i,1) + qdiff(j)*gammamat(i,j,1,1)
         enddo
       enddo

!  shift for 3rd order contributions
       if (lscc3rd) then
         do i=1,nat
           do j=1,nat
          shift3(i,1) = shift3(i,1) +qdiff(j)*gammader(i,j,1,1)
          shift3A(i,1)= shift3A(i,1)+qdiff(j)*qdiff(j)*gammader(j,i,1,1)
           enddo
           shift3(i,1)  = shift3(i,1) * qdiff(i)
         enddo
       endif
       endif !  ldep else

!  shift for spin-polarized-formalism
       if (lcolspin) then
         do i=1,nat
           do li=1,lmax(izp(i))
             do lj=1,lmax(izp(i))
               indlj=indl(i)+lj
               spinshift(i,li) = spinshift(i,li)
     &           +wspin(izp(i),li,lj)*(qlup(indlj)-qldown(indlj))
             enddo
           enddo
         enddo
       endif

       END



!=======================================================================
! get the gamma and Gamma contribution to the gradient
! -F_{kx}= 0.5d0*\Delta q_k\sum_{a!=k}\Delta q_a(dgamma_{ak}/dR_{kx}+
!          dgamma_{ka}/dR_{kx})+1/3 \Delta q_k\sum_{a!=k}\Delta q_a (
!          \Delta q_a dGamma_{ak}/dR_{kx}+\Delta q_k dGamma_{ak}/dR_{kx}
!          )
! For periodic systems: all expressions with gamma or Gamma also run
! over a sum of the cells (there is a cutoff)
!
! INPUT:
! integer  nn            number of atoms (in one cell)
! integer  nmov          number of movable atoms (in one cell)
! real*8   x(3,NNDIM)    coordinates
! real*8   izp(NNDIM)    map from atoms to atom types      
! real*8   uhubb(MAXTYP,3) Hubbard parameters
! real*8   uhder(MAXTYP,3) Hubbard derivatives
! real*8   basis         basis of cell
! logical  period        .true. if periodic boundary conditions
! logical  izpxh(MAXTYP) .true. for atom types which need extra term
!                         in gamma^h if switched on
! real*8   zeta          parameter for gamma^h (see Gaus JCTC 2011)
! real*8   qdiff(NNDIM)  atomic net charges (Mulliken)
!
! OUTPUT:
! real*8   hgrad(3,NNDIM) gradient contribution 
!
!======================================================================

! MG+Guanhua_QC_UW1205: substantial changes, MG_UW1210 (ldep)
       SUBROUTINE GAMMAGRAD(nat,nmov,rat,izp,uhubb,uhder,
     &                      qmat,qzero,ql,indl,basis,period,hgrad,
     &                      erfct,del,pts)
       use sccdftb, only: LSCOEWD
       use sccdftbsrc, only: nndim,maxtp,qdiff,lscc3rd,lscchb,kl1,izpxh,
     &                       lldep,lmax
       IMPLICIT NONE
!      INCLUDE 'maxima.inc'
       integer nat,nmov,izp(NNDIM),indl(NNDIM+1)
       real*8 qmat(NNDIM),qzero(MAXTP,4),ql(3*NNDIM)
       real*8 uhubb(MAXTP,3),rat(3,*),basis(3,3)
       real*8 uhder(MAXTP,3)
!       logical lscc3rd
!       common /scc3/ lscc3rd
! Guanhua_QC_UW1103
       real*8 r(3)!,kl1
!       logical izpxh(MAXTYP),xhgammahlp
       logical xhgammahlp
!       logical lscchb
!       common /scchb/ lscchb,kl1,izpxh
       integer j,k,ix,i,li,lk,lj
       external    getalpha
       real*8      recbasis(3,3),vol,tol,tmp(3),tmp3(3),getalpha
       real*8      alpha,long_deriv(3),short_deriv(3),norm
       real*8      dgdrkj,dgdr3kj,dgdrjk,dgdr3jk,hlp
       real*8      dgdr(NNDIM,NNDIM),dgdr3(NNDIM,NNDIM)
       real(8), allocatable :: SRdgdr(:, :, :), SRdgdr3(:, :, :)
       real*8      SRdgdrkj(3),SRdgdr3kj(3),SRdgdrjk(3),SRdgdr3jk(3)
       real*8      qldiff(NNDIM,3)
       real*8  erfct(*),del
       integer pts

       logical period

       real*8 hgrad(3,*)
!       common /varhelp/ Qdiff

       allocate(SRdgdr(3,NNDIM,NNDIM))
       allocate(SRdgdr3(3,NNDIM,NNDIM))
       
       DO k=1,nat
        DO i=1,3
         hgrad(i,k) = 0.0d0
        END DO
       END DO

!      set net charges Qdiff = qmat - qzero 
       DO i=1,nat 
        qdiff(i) = qmat(i) - qzero(izp(i),4)       
       END DO  

! first gradient for l-dependet Hubbards 
       ! not very efficient, but comprehensible formula hgrad=...
       if (lldep) then
! get qldiff
         do i=1,nat
           do li=1,lmax(izp(i))
             qldiff(i,li) = ql(indl(i)+li) - qzero(izp(i),li)
           enddo
         enddo
! gradient for periodic systems 
         if (period) then
           call rezvol(basis,recbasis,vol)
           alpha = getalpha(basis)
           tol = 1.0d-8
           do k=1,nmov
           do j=1,nat
           if (k.ne.j) then
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
! QC: UW_1112 I'll make it to call phi1scc0 always
! although to be safe, we probably should include the LSCOEWD flag. --> corrected MG_121120
             if (LSCOEWD) then 
               call phi1(r,basis,recbasis,alpha,vol,tol,long_deriv,
     &                erfct,del,pts)
             else
               call phi1scc0(r,basis,recbasis,alpha,vol,tol,long_deriv,
     &                erfct,del,pts)
             endif
! QC: UW_1112
             if ((izpxh(izp(k))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             do lk=1,lmax(izp(k))
             do lj=1,lmax(izp(j))
               call shortrange1(r,basis,uhubb(izp(k),lk),
     &              uhubb(izp(j),lj),uhder(izp(k),lk),xhgammahlp,kl1,
     &              tol,SRdgdrkj,SRdgdr3kj)
               call shortrange1(r,basis,uhubb(izp(j),lj),
     &              uhubb(izp(k),lk),uhder(izp(j),lj),xhgammahlp,kl1,
     &              tol,SRdgdrjk,SRdgdr3jk)
               do ix=1,3
                 hgrad(ix,k) = hgrad(ix,k) + 0.5d0*qldiff(k,lk)
     &                  *qldiff(j,lj)*( SRdgdrjk(ix)+SRdgdrkj(ix)
     &                   +2.0d0*long_deriv(ix) )
                 if (lscc3rd) then
                 hgrad(ix,k) = hgrad(ix,k) + qldiff(k,lk)/3.0d0
     &                  *qldiff(j,lj)*( qdiff(j)*SRdgdr3jk(ix)
     &                                 +qdiff(k)*SRdgdr3kj(ix) )
                 endif
               enddo
             enddo
             enddo
           endif
           enddo
           enddo

! gradient for non-periodic systems (cluster) and l-dependent Hubbard
         else 
           do k=1,nmov
           do j=1,nat
           if (j.ne.k) then
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
             norm = dsqrt(r(1)**2 + r(2)**2 + r(3)**2)
             if ((izpxh(izp(k))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             do lk=1,lmax(izp(k))
             do lj=1,lmax(izp(j))
               call gammaall1(norm,uhubb(izp(k),lk),uhubb(izp(j),lj),
     &              uhder(izp(k),lk),xhgammahlp,kl1,dgdrkj,dgdr3kj)
               call gammaall1(norm,uhubb(izp(j),lj),uhubb(izp(k),lk),
     &              uhder(izp(j),lj),xhgammahlp,kl1,dgdrjk,dgdr3jk)
               hlp = 0.5d0*qldiff(k,lk)*qldiff(j,lj)*(dgdrjk+dgdrkj)
               if (lscc3rd) then
                 hlp = hlp + qldiff(k,lk)*qldiff(j,lj)/3.0d0
     &               *( qdiff(j)*dgdr3jk+qdiff(k)*dgdr3kj )
               endif
               do ix=1,3
                 hgrad(ix,k) = hgrad(ix,k) + hlp*r(ix)/norm
               enddo
             enddo
             enddo
           endif
           enddo
           enddo
            
         endif ! cluster/period


! now gradient for atomic Hubbards (original)
       else 
       if (period) then
         do k=1,nat
           do j=1,nat
           if (k.ne.j) then
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
             if ((izpxh(izp(k))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             call shortrange1(r,basis,uhubb(izp(k),1),uhubb(izp(j),1),
     &            uhder(izp(k),1),xhgammahlp,kl1,tol,SRdgdrkj,SRdgdr3kj)
             do ix=1,3
               SRdgdr(ix,k,j)  = SRdgdrkj(ix)
               SRdgdr3(ix,k,j) = SRdgdr3kj(ix)
             enddo
           endif
           enddo
         enddo

         ! get reciprocal lattice vectors and cell volume
         call rezvol(basis,recbasis,vol)
         ! choose good convergence parameter alpha
         alpha = getalpha(basis)
         ! set tolerance for convergence
         tol = 1.0d-8
         do k=1,nmov
           do ix=1,3
             tmp(ix)=0.0d0
             tmp3(ix)=0.0d0
           enddo
           do j=1,nat
           if (k.ne.j) then
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
!            QC: is this the problem? UW_1112
             if (LSCOEWD) then
               call phi1(r,basis,recbasis,alpha,vol,tol,long_deriv,
     &                erfct,del,pts)
             else
               call phi1scc0(r,basis,recbasis,alpha,vol,tol,long_deriv,
     &                erfct,del,pts)
             endif
             do ix=1,3
               tmp(ix)  = tmp(ix)  + qdiff(j)*( SRdgdr(ix,k,j)
     &                      -SRdgdr(ix,j,k)+2.0d0*long_deriv(ix)  )
               tmp3(ix) = tmp3(ix) + qdiff(j)*( qdiff(k)*SRdgdr3(ix,k,j)
     &                    -qdiff(j)*SRdgdr3(ix,j,k)     )
             enddo
           endif
           enddo
           if (lscc3rd) then
             hgrad(1,k) = qdiff(k)*(0.5d0*tmp(1)+tmp3(1)/3.0d0)
             hgrad(2,k) = qdiff(k)*(0.5d0*tmp(2)+tmp3(2)/3.0d0)
             hgrad(3,k) = qdiff(k)*(0.5d0*tmp(3)+tmp3(3)/3.0d0)
           else
             hgrad(1,k) = qdiff(k)*0.5d0*tmp(1)
             hgrad(2,k) = qdiff(k)*0.5d0*tmp(2)
             hgrad(3,k) = qdiff(k)*0.5d0*tmp(3)
           endif
         enddo

       else ! no periodicity
 
         do k=1,nat
           do j=1,nat
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
             norm = dsqrt(r(1)**2 + r(2)**2 + r(3)**2)
             if ((izpxh(izp(k))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             call gammaall1(norm,uhubb(izp(k),1),uhubb(izp(j),1),
     &            uhder(izp(k),1),xhgammahlp,kl1,dgdrkj,dgdr3kj)
             dgdr(k,j)  = dgdrkj
             dgdr3(k,j) = dgdr3kj
           enddo
         enddo

         do k=1,nmov
           do ix=1,3
             tmp(ix)=0.0d0
             tmp3(ix)=0.0d0
           enddo
           do j=1,nat
           if (j.ne.k) then
             r(1)=rat(1,k)-rat(1,j)
             r(2)=rat(2,k)-rat(2,j)
             r(3)=rat(3,k)-rat(3,j)
             norm = dsqrt(r(1)**2 + r(2)**2 + r(3)**2)
             do ix=1,3
               tmp(ix) = tmp(ix) + qdiff(j) * ( dgdr(j,k) + dgdr(k,j) )
     &                                           * r(ix)/norm
               tmp3(ix) = tmp3(ix) + qdiff(j)*( qdiff(j)*dgdr3(j,k) +
     &                    qdiff(k)*dgdr3(k,j) ) * r(ix)/norm
             enddo
           endif
           enddo
           if (lscc3rd) then
             hgrad(1,k) = qdiff(k)*(0.5d0*tmp(1)+tmp3(1)/3.0d0)
             hgrad(2,k) = qdiff(k)*(0.5d0*tmp(2)+tmp3(2)/3.0d0)
             hgrad(3,k) = qdiff(k)*(0.5d0*tmp(3)+tmp3(3)/3.0d0)
           else
             hgrad(1,k) = qdiff(k)*0.5d0*tmp(1)
             hgrad(2,k) = qdiff(k)*0.5d0*tmp(2)
             hgrad(3,k) = qdiff(k)*0.5d0*tmp(3)
           endif
         enddo

       endif
       endif !no ldep

      END

!     ================ QC: Add GSBP shift terms =====================

      SUBROUTINE GSBPSHIFT(nat,atomtype,qmat,qzero,gamags,omegags,shift)
      use sccdftbsrc, only: nndim,maxtp,lldep,lmax,izp=>izp2
      IMPLICIT NONE
!     INCLUDE 'maxima.inc'
      integer nat
      integer atomtype(NNDIM)
      real*8 shift(NNDIM,3),qmat(NNDIM),qzero(MAXTP,4)
      real*8 gamags(*),omegags(*)

!     Local
      integer i,j,indk,ik
      real*8 factor

!     QC: Pay attention to unit here, both OMEGAGS and GAMAGS are in
!     the unit consistent with kcal/mol. switch to atomic unit@@CHK!!

      factor = 0.529177249D0

      DO i=1,nat
! >>>>>>>> OMEGA <<<<<<<<<<<
!       Watch out for signs here: since qmat is positive in the code
!       the sign should be flipped here - similar to CHARMM contribution
        if (lldep) then ! MG_UW1210 (ldep)
          do ik=1,lmax(izp(i))
            shift(i,ik)=shift(i,ik)-OMEGAGS(i)*factor
          enddo
        else
          shift(i,1) = shift(i,1) - OMEGAGS(i)*factor
        endif !lldep
! >>>>>>>> GAMAMS<<<<<<<<<<<
        DO j=1,nat
          if (j .gt. i) then
            indk= j*(j-1)/2 + i
          else
            indk= i*(i-1)/2 + j
          endif
!         Watch out for signs here, similar to gamma 
          if (lldep) then !MG_UW1210 (ldep) 
            do ik=1,lmax(izp(i))
            shift(i,ik) = shift(i,ik) + (qmat(j) - qzero(atomtype(j),4))
     $                        *GAMAGS(indk)*factor
            enddo
          else
            shift(i,1) = shift(i,1) + (qmat(j) - qzero(atomtype(j),4))
     $                        *GAMAGS(indk)*factor
          endif
        ENDDO
      ENDDO
      RETURN
      END

!     ============== XIAO ZHU & QC: Add PB shift terms ================

      SUBROUTINE PBSHIFT(nat,rf_qm,rf_mm,shift)
      use sccdftbsrc, only: nndim,lmax,izp=>izp2,lldep
      IMPLICIT NONE
!     INCLUDE 'maxima.inc'
      integer nat
      real*8 shift(NNDIM,3)
      real*8 rf_qm(*),rf_mm(*)

!     Local
      integer i,ik
      real*8 factor

!     XIAO ZHU & QC: Pay attention to unit here, both rf_mm and rf_qm
!     are in
!     the unit consistent with kcal/mol. switch to atomic unit@@CHK!!

      factor = 0.529177249D0

      if (lldep) then !MG_UW1210
        DO i=1,nat
          do ik=1,lmax(izp(i))
            shift(i,ik) = shift(i,ik) - rf_mm(i)*factor
            shift(i,ik) = shift(i,ik) - rf_qm(i)*factor
          enddo
        ENDDO
      else ! no ldep
        DO i=1,nat
          shift(i,1) = shift(i,1) - rf_mm(i)*factor
          shift(i,1) = shift(i,1) - rf_qm(i)*factor
        ENDDO
      endif !no ldep

      RETURN
      END


