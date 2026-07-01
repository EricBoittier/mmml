module mare
#if KEY_GAMUS==1
use chm_kinds
use dimens_fcm
use number
!MARE from Maragakis et al. PRL 96, 100602 (2006)

!Fortran translation and code clean up by Justin Spiriti





contains

!f(x) = fermi function 1/(1+exp(x))
!calculates r=sum(log(1-f(x))), s=sum(f(x)), t=sum(f'(x))

subroutine fermisum(n,x,r,s,t)
use chm_kinds
use number
implicit none
     real(chm_real) x(*),r,s,t
     integer n
     integer i
     real(chm_real) f,limit
     parameter(limit=-log(rpreci))
     r=zero
     s=zero
     t=zero
     do i=1,n
        !if (x(i)>ten) f(x), f'(x), and log(1-f(x)) are all zero
        if (x(i)<-limit) then
             r=r+x(i)  !log(1-f(x)) approx = x
             s=s+one  !f(x) approx 1; f'(x) approx zero
        else if (x(i)<limit) then !functions should be ok
             f=one/(one+exp(x(i)))
             r=r+log(one-f)
             s=s+f
             t=t-one/(two+two*cosh(x(i)))
        end if
     end do
end subroutine fermisum


!Equations 6, 7, and 8.  Includes "init_rst" and "energy" from C++ code.
subroutine deriv(nsim, maxwork, nwork, work, lnz, m, lnp, derlnp, derderlnp, frm, dfrm)
use chm_kinds
use number
implicit none
integer nsim,maxwork,nwork(1:nsim,1:nsim)
real(chm_real) work(1:maxwork,1:nsim,1:nsim),lnz(1:nsim)
real(chm_real) lnp
real(chm_real) derlnp(1:nsim),derderlnp(1:nsim,1:nsim)
real(chm_real) m(1:nsim,1:nsim),frm(1:maxwork),dfrm(1:maxwork)
real(chm_real) rij, sij, tij, am
integer i,j,nw,k
     lnp = zero
     derlnp = zero
     derderlnp = zero
     do i=1,nsim
          do j=1,nsim
              nw = nwork(i,j)
              am = lnz(i) - lnz(j) + m(i,j)
              frm = zero
              frm(1:nw) = work(1:nw,i,j) - am
              !do k=1,nw
              !   if (abs(frm(k))>9d0) write(*,*) i,j,work(k,i,j),am
              !end do
              !dfrm(1:nw) = frm(1:nw)
              !call fermi(frm(1:nw),nw)
              !call derfermi(dfrm(1:nw),nw)
              !rij = sum(log(one - frm(1:nw)))
              !sij = sum(frm(1:nw)) !eq. 7
              !tij = sum(dfrm(1:nw)) !eq. 8
              call fermisum(nw,frm,rij,sij,tij) 
              lnp = lnp + rij !summation, eq. 6
              derlnp(i) = derlnp(i) - sij 
              derlnp(j) = derlnp(j) + sij
              derderlnp(i,i) = derderlnp(i,i) + tij !eq. 8
              derderlnp(j,j) = derderlnp(j,j) + tij
              derderlnp(i,j) = derderlnp(i,j) - tij
              derderlnp(j,i) = derderlnp(j,i) - tij
          end do
     end do
end subroutine deriv
             
!This does only lnp.
subroutine objective(nsim, maxwork, nwork, work, lnz, m, lnp,  frm)
use chm_kinds
use number
implicit none
integer nsim,maxwork,nwork(1:nsim,1:nsim)
real(chm_real) work(1:maxwork,1:nsim,1:nsim),lnz(1:nsim)
real(chm_real) lnp
real(chm_real) m(1:nsim,1:nsim),frm(1:maxwork)
real(chm_real) rij, am, sij,tij
integer i,j,nw
     lnp = zero
     do i=1,nsim
          do j=1,nsim
              nw = nwork(i,j)
              am = lnz(i) - lnz(j) + m(i,j)
              frm = zero
              frm(1:nw) = work(1:nw,i,j) - am
              call fermisum(nw,frm,rij,sij,tij)
              lnp = lnp + rij !summation, eq. 6
          end do
     end do
end subroutine objective



!Only one protocol per (i,j) pair of simulations.
subroutine do_mare(nsim,maxwork,nwork,work,lnz)
use chm_kinds
use number
use memory
use stream
use clcg_mod,only: random
use reawri, only: iseed
implicit none
integer nsim,maxwork,nwork(1:nsim,1:nsim)
real(chm_real) work(1:maxwork,1:nsim,1:nsim),lnz(1:nsim)
real(chm_real),allocatable :: m(:,:),r(:,:),s(:,:),t(:,:)
real(chm_real),allocatable :: derlnp(:),derderlnp(:,:),frm(:),dfrm(:),dz(:),lnz2(:)
integer i,j,k, iter,info, n_gam
real(chm_real) lnp, lnp2, gam, lam2, tol, zeta, eta, lnp_min, dz_max, mstep
parameter (zeta=0.2d0, eta=half, lnp_min = -rbig, dz_max=twenty)

     call chmalloc('mare.src','mare','m',nsim,nsim,crl=m)
     !call chmalloc('mare.src','mare','r',nsim,nsim,crl=r)
     !call chmalloc('mare.src','mare','s',nsim,nsim,crl=s)
     !call chmalloc('mare.src','mare','t',nsim,nsim,crl=t)
     call chmalloc('mare.src','mare','derlnp',nsim,crl=derlnp)
     call chmalloc('mare.src','mare','derderlnp',nsim,nsim,crl=derderlnp)
     call chmalloc('mare.src','mare','frm',maxwork,crl=frm)
     call chmalloc('mare.src','mare','dfrm',maxwork,crl=dfrm)
     call chmalloc('mare.src','mare','dz',nsim-1,crl=dz)
     call chmalloc('mare.src','mare','lnz2',nsim,crl=lnz2)

     !Equation 5 
     do i=1,nsim
          do j=1,nsim
             m(i,j) = -log(real(nwork(i,j))) + log(real(nwork(j,i)))
          end do
     end do
     
     tol = 0.1 * sqrt(RPRECI) !machine epsilon, from number_ltm.src
     
     !lnz will already have been initialized to an initial guess.
     !damped Newton loop
     do iter=1,1000
          !Calculate derivatives and Hessian of likelihood
          call deriv(nsim, maxwork, nwork, work, lnz, m, lnp, derlnp, derderlnp, frm, dfrm)
          if (prnlev>=6) then
              write(outu,*) "lnp: ",lnp
              write(outu,*) "lnz: ",lnz
              write(outu,*) "derivatives: ",derlnp 
          endif
          !Find the solution to -H Dz = F, restricted to ln Z_2 through ln Z_nsim
          dz(1:nsim-1) = derlnp(2:nsim)
          call DPOSV('U',nsim-1,1,-derderlnp(2:nsim,2:nsim),nsim-1,dz,nsim-1,info)
          if (info<0) then
             if (prnlev>=2) write(outu,*) "LAPACK error: matrix ",i," info ",info
             call wrndie(-10,'<mare>','LAPACK error')
          endif
          !write(outu,*) "dz: ",dz        
          !convergence test
          lam2 = dot_product(dz(1:nsim-1),derlnp(2:nsim))
          if (lam2/two <= tol) exit

          !heuristic for singularities of objective funciton
          if (lnp<lnp_min) then
              if (prnlev>=2) write(outu,*) "singular lnp: max step move"
              mstep = maxval(abs(dz(1:nsim-1)))
              lnz(2:nsim) = lnz(2:nsim) + dz(1:nsim-1)/mstep
              cycle
          endif
       
          !backtracking
          gam = one
          lnz2(1) = zero
          lnz2(2:nsim) = lnz(2:nsim) + gam * dz(1:nsim-1)
          call objective(nsim, maxwork, nwork, work, lnz2, m, lnp2, frm)
          n_gam = 0
          !write(outu,*) "gam , lnp2, lam2 = ",gam,lnp2,lam2
          do while ((lnp2 - lnp < zeta*gam*lam2) .and. (n_gam <= 100))
               gam = gam * eta
               lnz2(2:nsim) = lnz(2:nsim) + gam * dz(1:nsim-1)
               call objective(nsim, maxwork, nwork, work, lnz2, m, lnp2, frm)
               !write(outu,*) "gam , lnp2, lam2 = ",gam,lnp2,lam2
               n_gam = n_gam + 1
               if (n_gam>20) then !small random kick out of a rare difficult loop
                   do i=1,nsim-1
                      lnz2(i) = lnz2(i) + pt01 * (random(iseed) - half)
                   end do
               endif
          end do

          lnz(1:nsim) = lnz2(1:nsim)
     end do

     if (lam2/2 > tol) call wrndie(-2,'<do_mare>','MARE failed to converge')
     if (prnlev>=2) write(outu,*) "GAMUSREWEIGHT> MARE convergence required ",iter," iterations"    
     call chmdealloc('mare.src','mare','m',nsim,nsim,crl=m)
     !call chmdealloc('mare.src','mare','r',nsim,nsim,crl=r)
     !call chmdealloc('mare.src','mare','s',nsim,nsim,crl=s)
     !call chmdealloc('mare.src','mare','t',nsim,nsim,crl=t)
     call chmdealloc('mare.src','mare','derlnp',nsim,crl=derlnp)
     call chmdealloc('mare.src','mare','derderlnp',nsim,nsim,crl=derderlnp)
     call chmdealloc('mare.src','mare','frm',maxwork,crl=frm)
     call chmdealloc('mare.src','mare','dfrm',maxwork,crl=dfrm)
     call chmdealloc('mare.src','mare','dz',nsim-1,crl=dz)
     call chmdealloc('mare.src','mare','lnz2',nsim,crl=lnz2)

end subroutine do_mare
#endif
end module mare
