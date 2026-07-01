!======================================================================
!   Build symmetric matrix gammamat containing Ewald potentials + short
!         range terms for periodic systems
!   Build non-symmetric matrix gammader
!
!   INPUT:
!   integer nn            number of atoms
!   real*8  x(3,NNDIM)    position of atoms
!   integer izp(NNDIM)    map from atoms to atom types
!   real*8  uhubb(MAXTYP,3) Hubbard parameters (l-dep)
!   real*8  uhder(MAXTYP,3) Hubbard derivatives dU/dq (l-dep)
!   real*8  zeta          parameter for gamma^h (see Gaus JCTC 2011)
!   real*8  basis(3,3)    basis of cell
!   logical period        .true. if periodic boundary conditions
!   logical izpxh(MAXTYP) .true. for atom types which need extra term
!                         in gamma^h if switched on
!
!   OUTPUT:
!   real*8 gammamat(*,*,3,3) matrix containing Ewald potentials + short 
!                        range terms  (for molecules the simple 
!                        gamma/gamma^h for all atom-pairings (ldep!))
!   real*8 gammader(*,*,3,3) matrix containing Gamma=dgamma/dq for DFTB3 
!                        for all atom-pairings (ldep!)
!                        (for periodic systems: only short range terms)
!
!   Note that code is made efficient (but still easily readable) 
!   for DFTB3, but allows also running DFTB2, therefore gammader 
!   is calculated by default in this function but of course may 
!   be zeroed or controlled by a subroutine calling get_gammamat 
!   or using the OUTPUT.
!      !!! NOTE THAT phi(ri - rj) = phi(rj - ri) !!!
!      !!! NOTE THAT shortrange(ri - rj) = shortrange(rj - ri) !!!
!
!=====================================================================

! MG+Guanhua_QC_UW1205: substantial changes, MG_UW1210 (ldep)
       subroutine get_gammamat(nat,rat,izp,uhubb,uhder,basis,period,
     &                  lldep,lmax,gammamat,gammader)
       use sccdftbsrc, only: nndim, maxtyp, lscchb, kl1, izpxh
       use sccdftb,    only: kappascc,kmxscc,ksqmxscc,LSCOEWD
       use erfcd_mod,  only: EWNPTS,EWLDT,EWRDEL
       IMPLICIT NONE
!      INCLUDE 'maxima.inc'
       INTEGER nat
       REAL*8 rat(3,*),basis(3,3),tol,gammamat(NNDIM,NNDIM,3,3)
! MG+Guanhua_QC_UW1205
       integer izp(nndim),lmax(maxtyp)
       real*8 uhubb(MAXTYP,3),uhder(MAXTYP,3)
       logical xhgammahlp,lldep
       real*8 gder,gammader(NNDIM,NNDIM,3,3)
       INTEGER i,j,li,lj
       REAL*8 phivalue,r(3)
       REAL*8 recbasis(3,3), vol
       REAL*8 alpha
       EXTERNAL getalpha
       REAL*8 getalpha
       REAL*8 gval,norm
       LOGICAL period
!      QC: If chose to specify ewald parameters as stated in
!      sccdftbini.src, then make alpha kappascc
!       logical LSCOEWD         
!       real*8 kappascc
!       integer kmxscc,ksqmxscc
!       common /sccewd/ kappascc,kmxscc,ksqmxscc,LSCOEWD

       if (period) then
! get reciprocal lattice vectors and cell volume
         call rezvol(basis,recbasis,vol)
! choose good convergence parameter alpha
         alpha = getalpha(basis)
         tol   = 1.0d-8

         do i=1,nat
           do j=1,nat
             r(1)=rat(1,i)-rat(1,j)
             r(2)=rat(2,i)-rat(2,j)
             r(3)=rat(3,i)-rat(3,j)

             IF (LSCOEWD) THEN 
!    xiao
               CALL phi(r,basis,recbasis,alpha,vol,tol,phivalue, 
     $           ewldt,ewrdel,ewnpts)
             ELSE
!    xiao
               CALL phiscc0(r,basis,recbasis,alpha,vol,tol,phivalue,
     $           ewldt,ewrdel,ewnpts)
             ENDIF

            if ((izpxh(izp(i))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             if (lldep) then
                 do li=1,lmax(izp(i))
                 do lj=1,lmax(izp(j))
                   call shortrange(r,basis,uhubb(izp(i),li),
     &                             uhubb(izp(j),lj),uhder(izp(i),li),
     &                             xhgammahlp,kl1,tol,gval,gder)
                   gammamat(i,j,li,lj) = phivalue + gval
                   gammader(i,j,li,lj) = gder
                 enddo
                 enddo
             else !no ldep
               call shortrange(r,basis,uhubb(izp(i),1),uhubb(izp(j),1),
     &                uhder(izp(i),1),xhgammahlp,kl1,tol,gval,gder)
               gammamat(i,j,1,1) = phivalue + gval
               gammader(i,j,1,1) = gder
             endif
           enddo
         enddo

       ELSE ! non periodic

         do i=1,nat
           do j=1,nat
             r(1)=rat(1,i)-rat(1,j)
             r(2)=rat(2,i)-rat(2,j)
             r(3)=rat(3,i)-rat(3,j)
             norm   = dsqrt(r(1)**2+r(2)**2+r(3)**2)
             if ((izpxh(izp(i))).or.(izpxh(izp(j)))) then
               xhgammahlp=.true.
             else
               xhgammahlp=.false.
             endif
             if (lldep) then
               do li=1,lmax(izp(i))
               do lj=1,lmax(izp(j))
                 call gammaall(norm,uhubb(izp(i),li),uhubb(izp(j),lj),
     &                uhder(izp(i),li),xhgammahlp,kl1,gval,gder)
                 gammamat(i,j,li,lj) = gval
                 gammader(i,j,li,lj) = gder
               enddo
               enddo
             else
               call gammaall(norm,uhubb(izp(i),1),uhubb(izp(j),1),
     &              uhder(izp(i),1),xhgammahlp,kl1,gval,gder)
               gammamat(i,j,1,1) = gval
               gammader(i,j,1,1) = gder
             endif
           enddo
         enddo

       ENDIF

       END

