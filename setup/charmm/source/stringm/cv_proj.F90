! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_PROJ.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `PROJ` : length of projection along vector between two reference structures
      module cv_proj
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use cv_common
      use cv_drmsd, only : cv_drmsd_add, cv_drmsd_list, tol
!
      implicit none
      private
      ! subroutines
      public cv_proj_add
      public cv_proj_calc
      public cv_proj_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_proj_add(ind_o,ind_f,r_o,r_f,r1_o,r1_f, &
     & ow,fw,k,gamma,weight)
       real(chm_real) :: k, gamma, weight
       integer, pointer :: ind_o(:), ind_f(:)
       real(chm_real), pointer :: r_o(:,:), r_f(:,:), r1_o(:,:), r1_f(:,:), &
     & ow(:), fw(:)
       logical :: cv_proj_add
! use drmsd routine since the storage scheme is identical
       cv_proj_add=cv_drmsd_add(ind_o,ind_f,r_o,r_f,r1_o,r1_f, &
     & ow,fw,k,gamma,weight,.true.)
!
       end function cv_proj_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_proj_calc(i,x,y,z,mass,fx,fy,fz, &
     & calctheta,deriv,addforce,fext)
       use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
       use consta
!
       real(chm_real) :: x(:), y(:), z(:), &
     & fx(:), fy(:), fz(:), mass(:)
       real(chm_real), optional :: fext ! `external` force for planar dynamics (see smcv_addforce in smcv_master)
       integer, intent(in) :: i
       logical :: calctheta ! whether or not to calculate theta(x); if not, use theta=cv%r(i,instant)
                            ! note that if calctheta=.false., we do not calculate the derivatives!
       logical :: deriv ! whether or not to calculate derivatives w.r.t. x
       logical :: addforce ! whether or not to add forces on simulation atoms
! locals
       logical :: qdiffrot
       integer :: j, k, ii, jj, ind, norient, nforced, obeg, oend, p, q
       real(chm_real) :: d, f, w, r1(3), r2(3), r_com(3), theta, u(3,3), u1(3,3)
       real(chm_real) :: d1, d2, d3, rho, rho1
       real(chm_real) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
       real(chm_real) :: b11, b12, b13, b21, b22, b23, b31, b32, b33
       integer, pointer :: priv(:)
       integer, pointer :: ind_o(:), ind_f(:)
       real(chm_real), pointer :: r_o(:,:), rt_o(:,:), rt1_o(:,:), ow(:), &
     & r_f(:,:), rt_f(:,:), rt1_f(:,:), fw(:), &
     & rt_rot_f(:,:), rt1_rot_f(:,:), M(:,:)
       real(chm_real), dimension(:,:,:,:), pointer :: ugrad, ugrad1, ugrad2
! real(chm_real), dimension(:,:,:,:), pointer, target :: ugrad, ugrad1, ugrad2
!
       norient=cv%priv(i)%p(1)
       nforced=cv%priv(i)%p(2)
       qdiffrot=(cv%priv(i)%p(3).ne.0)
! qdiffrot=.true. ! aa
!
! write(0,*) norient, nforced, qdiffrot, deriv !aa
!
       if (qdiffrot) then ! extract all indices and coordinates
        allocate(ind_o(norient)) ! indices into atom map
        allocate(rt_o(norient,3), rt1_o(norient,3), r_o(norient,3))
        allocate(ow(norient))
        allocate(ind_f(nforced))
        allocate(rt_f(nforced,3), r_f(nforced,3), rt_rot_f(nforced,3), &
     & rt1_f(nforced,3), rt1_rot_f(nforced,3))
        allocate(fw(nforced))
! indices
        ii=4; jj=ii+norient-1; ind_o=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+nforced-1; ind_f=cv%priv(i)%p(ii:jj)
! o. coordinates
        jj=0;
        ii=jj+1; jj=ii+norient-1; rt_o(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt_o(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt_o(:,3)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,3)=cv%priv(i)%pr(ii:jj)
! f. coordinates
        ii=jj+1; jj=ii+nforced-1; rt_f(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+nforced-1; rt_f(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+nforced-1; rt_f(:,3)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+nforced-1; rt1_f(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+nforced-1; rt1_f(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+nforced-1; rt1_f(:,3)=cv%priv(i)%pr(ii:jj)
! o. weights
        ii=jj+1; jj=ii+norient-1; ow(:)=cv%priv(i)%pr(ii:jj)
! f. weights
        ii=jj+1; jj=ii+nforced-1; fw(:)=cv%priv(i)%pr(ii:jj)
       else ! some (not all) contributions coming from rotations are zero
        allocate(ind_o(norient))
        allocate(rt_o(norient,3), rt1_o(norient,3), r_o(norient,3))
        allocate(ow(norient))
! indices
        ii=4; jj=ii+norient-1; ind_o=cv%priv(i)%p(ii:jj)
! o. coordinates
        jj=0;
        ii=jj+1; jj=ii+norient-1; rt_o(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt_o(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt_o(:,3)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,1)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,2)=cv%priv(i)%pr(ii:jj)
        ii=jj+1; jj=ii+norient-1; rt1_o(:,3)=cv%priv(i)%pr(ii:jj)
! o. weights (skip forced atom coordinates)
        ii=jj+6*nforced+1; jj=ii+norient-1; ow(:)=cv%priv(i)%pr(ii:jj)
! assume orientation atoms/weights are the same as forced atoms/weights
        ind_f=>ind_o
        rt_f=>rt_o
        r_f=>r_o
        rt1_f=>rt1_o
        allocate(rt_rot_f(nforced,3)) ! only needed for forced set
        allocate(rt1_rot_f(nforced,3)) ! only needed for forced set
        fw=>ow
       endif ! diffrot
!
       if (calctheta) then
! load coordinates
! orient
        do k=1,norient
         ind=cv%amap%i(ind_o(k))
         r_o(k,1)=x(ind)
         r_o(k,2)=y(ind)
         r_o(k,3)=z(ind)
        enddo
! translate forced atoms to centroid
! v%rcurrent_com=matmul(transpose(v%rcurrent_o), v%orientWeights) ! shorthand (might be slower)
! conventional way (might be faster)
        r_com(:)=0d0;
        do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*r_o(k,j)
        enddo ; enddo
!
        r_o(:,1)=r_o(:,1)-r_com(1)
        r_o(:,2)=r_o(:,2)-r_com(2)
        r_o(:,3)=r_o(:,3)-r_com(3)
!
        obeg=1; oend=norient
!
        if (qdiffrot) then ! also use forced atoms (otherwise, they are the same -- see above!)
!
         do k=1,nforced
          ind=cv%amap%i(ind_f(k))
          r_f(k,1)=x(ind)
          r_f(k,2)=y(ind)
          r_f(k,3)=z(ind)
         enddo
!
         r_f(:,1)=r_f(:,1)-r_com(1)
         r_f(:,2)=r_f(:,2)-r_com(2)
         r_f(:,3)=r_f(:,3)-r_com(3)
!
        endif ! qdiffrot
!
! compute rotation matrix (and its gradient, if needed)
        if (deriv) then
         allocate(ugrad (3,3,3,norient), &
     & ugrad1(3,3,3,norient), &
     & ugrad2(3,3,3,norient) )
         call RMSBestFit(rt_o,r_o,ow,u,obeg,oend,ugrad)
         call RMSBestFit(rt1_o,r_o,ow,u1,obeg,oend,ugrad1)
        else
         call RMSBestFit(rt_o,r_o,ow,u)
         call RMSBestFit(rt1_o,r_o,ow,u1)
        endif ! deriv
!
! rotate target structures to overlap with current
! conventional way (might be faster)
        rt_rot_f=0d0;
        rt1_rot_f=0d0;
        do k=1,3; do j=1,3
          rt_rot_f(:,j)=rt_rot_f(:,j)+rt_f(:,k)*u(j,k)
          rt1_rot_f(:,j)=rt1_rot_f(:,j)+rt1_f(:,k)*u1(j,k)
        enddo; enddo
! compute theta
        rho=0d0
        rho1=0d0
!
        if (deriv) then
! initialize derivative arrays
         priv=>cv%priv(i)%amap_ptr ! this should copy the pointers statically; the pointers point to the _same_ data
         do jj=2,priv(1)+1 ! only a subset of indices needs to be considered
          j=priv(jj)
!
          do ii=1,2
          cv%gradx(i,j,ii)=0d0;cv%grady(i,j,ii)=0d0;cv%gradz(i,j,ii)=0d0
          enddo
         enddo
!
         r1=0d0;
         do k=1,3 ; do j=1,nforced
                     d = r_f(j,k) -rt_rot_f(j,k)
                     d1= rt1_rot_f(j,k)-rt_rot_f(j,k)
!
                     rho1=rho1 + d1*d1 *fw(j) ! denominator
                     d1 = d1 * fw(j)
                     rho =rho + d *d1 ! numerator
!
! COM contribution to gradient for orientation atoms:
                     r1(k)=r1(k)+d1 ! will only be used if (qdiffrot)
! derivative components from the forcing atoms:
                     ind=ind_f(j) ; cv%grad(i,ind,1,k)=d1 ! note the use of alternative gradient pointer (see cv_common.src)
         enddo ; enddo
        else ! derivative calculation not requested
         do k=1,3 ; do j=1,nforced
                    d = r_f(j,k) -rt_rot_f(j,k)
                    d1= rt1_rot_f(j,k)-rt_rot_f(j,k)
!
                    rho =rho + d *d1 *fw(j) ! numerator
                    rho1=rho1 + d1*d1 *fw(j) ! denominator
         enddo ; enddo
        endif ! deriv
!
        if (rho1.le.tol) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a very unlikely event
        theta=rho*rho1
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta
!
        if (deriv) then
!
         w=1d0-2d0*theta
         if (qdiffrot) then
!
          do k=obeg, oend
            ind=ind_o(k)
! COM contribution to gradients on orientation atoms
            cv%gradx(i,ind,1)=cv%gradx(i,ind,1)-r1(1)*ow(k)
            cv%grady(i,ind,1)=cv%grady(i,ind,1)-r1(2)*ow(k)
            cv%gradz(i,ind,1)=cv%gradz(i,ind,1)-r1(3)*ow(k)
! in this loop we also compute the gradients of [transpose(A) B ]
            do j=1,3
! multiplications `by hand`
             M=>ugrad(:,:,j,k)
             a11=M(1,1); a21=M(2,1); a31=M(3,1);
             a12=M(1,2); a22=M(2,2); a32=M(3,2);
             a13=M(1,3); a23=M(2,3); a33=M(3,3);
             M=>ugrad1(:,:,j,k)
             b11=M(1,1); b21=M(2,1); b31=M(3,1);
             b12=M(1,2); b22=M(2,2); b32=M(3,2);
             b13=M(1,3); b23=M(2,3); b33=M(3,3);
             M=>ugrad2(:,:,j,k)
!
             M(1,1)=a11*u1(1,1) + a21*u1(2,1) + a31*u1(3,1) + &
     & u(1,1)*b11 + u(2,1)*b21 + u(3,1)*b31
             M(2,1)=a12*u1(1,1) + a22*u1(2,1) + a32*u1(3,1) + &
     & u(1,2)*b11 + u(2,2)*b21 + u(3,2)*b31
             M(3,1)=a13*u1(1,1) + a23*u1(2,1) + a33*u1(3,1) + &
     & u(1,3)*b11 + u(2,3)*b21 + u(3,3)*b31
!
             M(1,2)=a11*u1(1,2) + a21*u1(2,2) + a31*u1(3,2) + &
     & u(1,1)*b12 + u(2,1)*b22 + u(3,1)*b32
             M(2,2)=a12*u1(1,2) + a22*u1(2,2) + a32*u1(3,2) + &
     & u(1,2)*b12 + u(2,2)*b22 + u(3,2)*b32
             M(3,2)=a13*u1(1,2) + a23*u1(2,2) + a33*u1(3,2) + &
     & u(1,3)*b12 + u(2,3)*b22 + u(3,3)*b32
!
             M(1,3)=a11*u1(1,3) + a21*u1(2,3) + a31*u1(3,3) + &
     & u(1,1)*b13 + u(2,1)*b23 + u(3,1)*b33
             M(2,3)=a12*u1(1,3) + a22*u1(2,3) + a32*u1(3,3) + &
     & u(1,2)*b13 + u(2,2)*b23 + u(3,2)*b33
             M(3,3)=a13*u1(1,3) + a23*u1(2,3) + a33*u1(3,3) + &
     & u(1,3)*b13 + u(2,3)*b23 + u(3,3)*b33
! aa: this is what we are doing above:
! ugrad2(:,:,j,k)=matmul(transpose(ugrad(:,:,j,k)),u1)+
! & matmul(transpose(u),ugrad1(:,:,j,k))
           enddo ! j
          enddo !
! contribution from quadratics (3 terms)
          do j=1, nforced
           r1=fw(j)*r_f(j,:)
           r2=w*fw(j)*rt_f(j,:)
!
           do k=obeg, oend
            ind=ind_o(k)
!
            do p=1,3
             do q=1,3
!
              d =r1(p)
              d1=rt_f(j,q)
              d2=rt1_f(j,q)
              d3=r2(p)*d2
!
              cv%gradx(i,ind,1)=cv%gradx(i,ind,1) + &
     & d * ( ugrad1(p,q,1,k) * d2 - ugrad(p,q,1,k) * d1 ) - &
     & d3 * ( ugrad2(p,q,1,k) )
!
              cv%grady(i,ind,1)=cv%grady(i,ind,1) + &
     & d * ( ugrad1(p,q,2,k) * d2 - ugrad(p,q,2,k) * d1 ) - &
     & d3 * ( ugrad2(p,q,2,k) )
!
              cv%gradz(i,ind,1)=cv%gradz(i,ind,1) + &
     & d * ( ugrad1(p,q,3,k) * d2 - ugrad(p,q,3,k) * d1 ) - &
     & d3 * ( ugrad2(p,q,3,k) )
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
          enddo ! j (forcing atoms)
!
         else ! not qdiffrot
          do k=obeg, oend
! compute the gradients of [transpose(A) B ]
           do j=1,3
! multiplications `by hand`
             M=>ugrad(:,:,j,k)
             a11=M(1,1); a21=M(2,1); a31=M(3,1);
             a12=M(1,2); a22=M(2,2); a32=M(3,2);
             a13=M(1,3); a23=M(2,3); a33=M(3,3);
             M=>ugrad1(:,:,j,k)
             b11=M(1,1); b21=M(2,1); b31=M(3,1);
             b12=M(1,2); b22=M(2,2); b32=M(3,2);
             b13=M(1,3); b23=M(2,3); b33=M(3,3);
             M=>ugrad2(:,:,j,k)
!
             M(1,1)=a11*u1(1,1) + a21*u1(2,1) + a31*u1(3,1) + &
     & u(1,1)*b11 + u(2,1)*b21 + u(3,1)*b31
             M(2,1)=a12*u1(1,1) + a22*u1(2,1) + a32*u1(3,1) + &
     & u(1,2)*b11 + u(2,2)*b21 + u(3,2)*b31
             M(3,1)=a13*u1(1,1) + a23*u1(2,1) + a33*u1(3,1) + &
     & u(1,3)*b11 + u(2,3)*b21 + u(3,3)*b31
!
             M(1,2)=a11*u1(1,2) + a21*u1(2,2) + a31*u1(3,2) + &
     & u(1,1)*b12 + u(2,1)*b22 + u(3,1)*b32
             M(2,2)=a12*u1(1,2) + a22*u1(2,2) + a32*u1(3,2) + &
     & u(1,2)*b12 + u(2,2)*b22 + u(3,2)*b32
             M(3,2)=a13*u1(1,2) + a23*u1(2,2) + a33*u1(3,2) + &
     & u(1,3)*b12 + u(2,3)*b22 + u(3,3)*b32
!
             M(1,3)=a11*u1(1,3) + a21*u1(2,3) + a31*u1(3,3) + &
     & u(1,1)*b13 + u(2,1)*b23 + u(3,1)*b33
             M(2,3)=a12*u1(1,3) + a22*u1(2,3) + a32*u1(3,3) + &
     & u(1,2)*b13 + u(2,2)*b23 + u(3,2)*b33
             M(3,3)=a13*u1(1,3) + a23*u1(2,3) + a33*u1(3,3) + &
     & u(1,3)*b13 + u(2,3)*b23 + u(3,3)*b33
! aa: this is what we are doing above:
! ugrad2(:,:,j,k)=matmul(transpose(ugrad(:,:,j,k)),u1)+
! & matmul(transpose(u),ugrad1(:,:,j,k))
           enddo ! j
          enddo ! k
! contribution from quadratics
! aa
! write(0,*) 'U: ', u
! write(0,*) 'U1: ', u1
!
          do j=1, nforced
           r2=w*fw(j)*rt_f(j,:)
!
           do k=obeg, oend
            ind=ind_o(k)
!
            do p=1,3
             do q=1,3
              d3=r2(p)*rt1_f(j,q)
              cv%gradx(i,ind,1)=cv%gradx(i,ind,1) - d3 * ugrad2(p,q,1,k)
              cv%grady(i,ind,1)=cv%grady(i,ind,1) - d3 * ugrad2(p,q,2,k)
              cv%gradz(i,ind,1)=cv%gradz(i,ind,1) - d3 * ugrad2(p,q,3,k)
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
          enddo ! j (forcing atoms)
!
         endif ! qdiffrot
!
! scale derivatives
         priv=>cv%priv(i)%amap_ptr
         do ii=2, priv(1)+1
          jj=priv(ii)
          j=cv%amap%i(jj) ! psf index
          w=sqrt(1d0/mass(j))
          cv%gradx(i,jj,1)=cv%gradx(i,jj,1)*rho1
          cv%gradx(i,jj,2)=cv%gradx(i,jj,1)*w
!
          cv%grady(i,jj,1)=cv%grady(i,jj,1)*rho1
          cv%grady(i,jj,2)=cv%grady(i,jj,1)*w
!
          cv%gradz(i,jj,1)=cv%gradz(i,jj,1)*rho1
          cv%gradz(i,jj,2)=cv%gradz(i,jj,1)*w
         enddo
        endif ! deriv
       else
        theta=cv%r(i,instant) ! assume valid theta has been computed in cv%r(i,instant)
       endif ! calctheta
!
! NOTE that the forces calculated here are NOT acting on the CV, as in
! the evolution subroutine, but on the simulation atoms
       d=cv%r(i,zcur)-theta ! zcur contains reference coords (combination of main+comp)
!
       f=cv%k(i)*d ! note that f is not divided by 1/sqrt(mass)
       cv%r(i,forces2)=f
       if (addforce) then
        if (present(fext)) f=fext ! override for planar sampling
        priv=>cv%priv(i)%amap_ptr
        do ii=2, priv(1)+1
         jj=priv(ii)
         j=cv%amap%i(jj) ! psf index
         fx(j)=fx(j)-f*cv%gradx(i,jj,1)
         fy(j)=fy(j)-f*cv%grady(i,jj,1)
         fz(j)=fz(j)-f*cv%gradz(i,jj,1)
        enddo
       endif ! addforce
! free memory
       deallocate(ind_o,r_o,rt_o,rt1_o,ow,rt_rot_f,rt1_rot_f)
       if (qdiffrot) then
        deallocate(ind_f,r_f,rt_f,rt1_f,fw)
       endif
       if (deriv) deallocate(ugrad, ugrad1, ugrad2)
!
       end subroutine cv_proj_calc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_proj_list(i)
       integer :: i
       call cv_drmsd_list(i,.true.) ! call drmsd routine
       end subroutine cv_proj_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_proj
!
