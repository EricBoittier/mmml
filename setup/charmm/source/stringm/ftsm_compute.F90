! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! finite-temperature string method calculations
      module ftsm_compute
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
!
      use chm_kinds
      use ftsm_var
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3 ! , only : RMSBestFit, rmsd
      implicit none
!
      private
!=====================================================================
! SUBROUTINES
      public ftsmv1_calc
      public ftsm_calc_M
      public ftsm_calc_J
      public ftsm_addforce
      public ftsm_compute_fe_fd
!
      contains
!======================================================================
! compute RC value and derivatives
! renamed from ftsm to ftsmv1
      subroutine ftsmv1_calc(x,y,z,deriv,t)
!
      use number
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use parallel, only: psnd4, psnd8
!
!
      real(chm_real), intent(in) :: x(:), y(:), z(:)
      logical :: deriv, qgrp, qcombine
      real(chm_real), intent(in), optional :: t
      real(chm_real) :: s, oms ! this value indicates how much of the old reference set to take
!
      integer :: ind, i, j, k, obeg, oend, p, q, ierror
!
      real(chm_real) :: u (3,3), u1(3,3), u2(3,3), u3(3,3)
      real(chm_real), pointer :: r_com(:)
      real(chm_real) :: rho, rho1, rho2, r1(3), r2(3), r3(3), d, d1, d2, d3, &
     & d4, d5, d6, w
      real(chm_real) :: omdpar
      real(chm_real) :: a11, a12, a13, a21, a22, a23, a31, a32, a33
      real(chm_real) :: b11, b12, b13, b21, b22, b23, b31, b32, b33
      real(chm_real), dimension(:,:,:,:), pointer :: ugrad, ugrad1, ugrad2
      real(chm_real), pointer, dimension(:,:) :: roi,rol,ror,roc, &
     & rol_old,roc_old,ror_old,rol_cur,roc_cur,ror_cur
      real(chm_real), pointer, dimension(:,:) :: rfi,rfl,rfr,rfc,rfl_rot,rfr_rot&
     & ,rfc_rot,rfl_old,rfc_old,rfr_old,rfl_cur,rfc_cur,rfr_cur
      real(chm_real), pointer, dimension(:,:) :: fopar, foprp, ffpar, ffprp
      real(chm_real), pointer, dimension(:,:) :: rfpar, rfprp
      real(chm_real), pointer :: M(:,:)
!
      integer*4, pointer :: orient_count(:), orient_displ(:) ! if declared as integer*8 parallelization will break
!
      real(chm_real) :: tol
!
      real(chm_real), pointer :: ow(:), fw(:), wgts(:)
      integer, pointer :: inds(:)
!
      interface
       subroutine hypercube_allgatherv(message,count,displ,type, &
     & comm, ierror, rank, size)
      use chm_kinds
       real(chm_real) :: message(*)
       integer :: ierror
       integer*4 :: comm, rank, size, type, count(size), displ(size)
       end subroutine hypercube_allgatherv
      end interface
!
      tol=RSMALL
!
      if (present(t)) then ; s=min(max(t,zero),one); else ; s=one ; endif ! limit the range of s for clarity; the code works for s>1 b/c of qcombine below
!
      qcombine=s.lt.one
!
      qgrp=( SIZE_LOCAL.gt.1.and.MPI_COMM_LOCAL.ne.MPI_COMM_NULL &
     & .and.calc_bestfit_grad_para)
! shorthand
      ow=>orientWeights
      r_com=>rcom(:,instant)
      roi=>r_o(:,:,instant);
      rol=>r_o(:,:,left); ror=>r_o(:,:,right); roc=>r_o(:,:,center);
      fopar=>r_o(:,:,fpar); foprp=>r_o(:,:,fperp) ! parallel forces
!--------------------------------------------------------------------------------------
      fw=>forcedWeights
      rfi=>r_f(:,:,instant);
      rfl=>r_f(:,:,left); rfr=>r_f(:,:,right); rfc=>r_f(:,:,center);
      rfl_rot=>r_f(:,:,left_rot); rfr_rot=>r_f(:,:,right_rot); rfc_rot=>r_f(:,:,center_rot);
      ffpar=>r_f(:,:,fpar); ffprp=>r_f(:,:,fperp) ! forces
      rfpar=>r_f(:,:,vpar); rfprp=>r_f(:,:,vperp) ! displacement vectors
!
! load coordinates
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, nforced
    rfi(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     rfi(k,1)=rfi(k,1)+(d*x(ind));
     rfi(k,2)=rfi(k,2)+(d*y(ind));
     rfi(k,3)=rfi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    !
    rfi(k,1)=x(ind)
    rfi(k,2)=y(ind)
    rfi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
!
      if (qorient) then
       if (qdiffrot) then
!
!
!
!
!
!
! | macro ftsm_load_fcor.def
! v
! ==========================
  if (ftsm_com_on) then
   do k=1, norient
    roi(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     !
     roi(k,1)=roi(k,1)+(d*x(ind));
     roi(k,2)=roi(k,2)+(d*y(ind));
     roi(k,3)=roi(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    !
    roi(k,1)=x(ind)
    roi(k,2)=y(ind)
    roi(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
       endif ! qdiffrot (otherwise rfi and roi point to the same thing)
!
! translate forced atoms to centroid
       r_com(:)=0d0;
       do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j)
       enddo ; enddo
!
       rfi(:,1)=rfi(:,1)-r_com(1)
       rfi(:,2)=rfi(:,2)-r_com(2)
       rfi(:,3)=rfi(:,3)-r_com(3)
!
       if (qdiffrot) then ! also use orientation atoms (otherwise, they are the same -- see above!)
         roi(:,1)=roi(:,1)-r_com(1)
         roi(:,2)=roi(:,2)-r_com(2)
         roi(:,3)=roi(:,3)-r_com(3)
       endif ! qdiffrot
!
      else
       u = Id3
       u1= Id3
       u2= Id3
       u3= Id3
      endif ! qorient
!
      if (qcombine) then ! use a combination of old and new reference structures for mild restart
       oms=one-s
!
       rfl_old=>r_f(:,:,left_old); rfr_old=>r_f(:,:,right_old);
       rfc_old=>r_f(:,:,center_old);
       rfl_cur=>r_f(:,:,left_cur); rfr_cur=>r_f(:,:,right_cur);
       rfc_cur=>r_f(:,:,center_cur);
!
       rol_old=>r_o(:,:,left_old); ror_old=>r_o(:,:,right_old);
       roc_old=>r_o(:,:,center_old);
       rol_cur=>r_o(:,:,left_cur); ror_cur=>r_o(:,:,right_cur);
       roc_cur=>r_o(:,:,center_cur);
!
       if (proj_on) then
        if (qorient) then
         call RMSBestFit(rol,roi,ow,u)
         call RMSBestFit(rol_old,roi,ow,u1)
         call RMSBestFit(ror,roi,ow,u2)
         call RMSBestFit(ror_old,roi,ow,u3)
! combine rotated structures
! left
!
         u=u*s; u1=u1*oms;
         u2=u2*s; u3=u3*oms;
         rfl_cur=0d0; rfr_cur=0d0;
         do k=1,3; do j=1,3
           rfl_cur(:,j)=rfl_cur(:,j) + rfl(:,k) * u(j,k) + &
     & rfl_old(:,k) * u1(j,k)
           rfr_cur(:,j)=rfr_cur(:,j) + rfr(:,k) * u2(j,k) + &
     & rfr_old(:,k) * u3(j,k)
         enddo; enddo
!
         if (qdiffrot) then
          rol_cur=0d0; ror_cur=0d0
          do k=1,3; do j=1,3
           rol_cur(:,j)=rol_cur(:,j) + rol(:,k) * u(j,k) + &
     & rol_old(:,k) * u1(j,k)
           ror_cur(:,j)=ror_cur(:,j) + ror(:,k) * u2(j,k) + &
     & ror_old(:,k) * u3(j,k)
          enddo; enddo
         endif
!
        else ! not qorient
!
         do j=1,3
           rfl_cur(:,j)=s*rfl(:,j)+oms*rfl_old(:,j)
           rfr_cur(:,j)=s*rfr(:,j)+oms*rfr_old(:,j)
         enddo;
!
        endif ! qorient
! point to combined reference structures
        rfl=>rfl_cur
        rfr=>rfr_cur
        rol=>rol_cur
        ror=>ror_cur
!
       endif ! proj
!
       if (.not.proj_on .or. fe_curvature) then
        rfc_cur=0d0;
        if (qorient) then
         call RMSBestFit(roc,roi,ow,u)
         call RMSBestFit(roc_old,roi,ow,u1)
! combine rotated structures
! left
!
         u=u*s; u1=u1*oms;
         do k=1,3; do j=1,3
           rfc_cur(:,j)=rfc_cur(:,j) + rfc(:,k) * u(j,k) + &
     & rfc_old(:,k) * u1(j,k)
         enddo; enddo
!
         if (qdiffrot) then
          roc_cur=0d0;
          do k=1,3; do j=1,3
           roc_cur(:,j)=roc_cur(:,j) + roc(:,k) * u(j,k) + &
     & roc_old(:,k) * u1(j,k)
          enddo; enddo
         endif
!
        else ! not qorient
!
         do j=1,3
           rfc_cur(:,j)=s*rfc(:,j)+oms*rfc_old(:,j)
         enddo;
!
        endif ! qorient
! point to combined reference structures
        rfc=>rfc_cur
        roc=>roc_cur
!
       endif ! (not proj) / curvature
      endif ! qcombine
!
!! write(600+me_global,*) matmul(ow,rol) ! correct (0)
!! write(600+me_global,*) matmul(ow,roc)
!! write(600+me_global,*) matmul(ow,ror)
!! write(600+me_global,*) matmul(ow,roi)
!
      if (qorient) then
! compute rotation matrices (and gradients, if needed)
!
       if (deriv) then
!%%%%%%%%%%%%%%%%%%% set up indices %%%%%%%%%%%%%%%%%%%
         if (qgrp) then
!
          j=ceiling(1.0d0*norient/SIZE_LOCAL)
!
          allocate(orient_displ(SIZE_LOCAL), orient_count(SIZE_LOCAL))
!
          do i=1,SIZE_LOCAL
           orient_displ(i)=min((i-1)*j,norient-1)
           orient_count(i)=max(0,min(j,norient-j*(i-1)))
          enddo
!
          obeg=orient_displ(ME_LOCAL+1) + 1
          oend=obeg - 1 + orient_count(ME_LOCAL+1)
!
         else ! not qgrp
          obeg=1; oend=norient
         endif ! qgrp
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         if (proj_on) then
          allocate(ugrad (3,3,3,norient), &
     & ugrad1(3,3,3,norient), &
     & ugrad2(3,3,3,norient) )
!
          call RMSBestFit(rol,roi,ow,u, obeg,oend,ugrad)
          call RMSBestFit(ror,roi,ow,u1,obeg,oend,ugrad1)
! fe_curvature (do not need gradient)
          if (fe_curvature) call RMSBestFit(roc,roi,ow,u2) ! here, u2 corresponds to the center image
!
         elseif (qdiffrot) then ! not proj_on but need derivatives of rotations
          allocate(ugrad (3,3,3,norient))
          call RMSBestFit(roc,roi,ow,u2, obeg,oend,ugrad)
         else ! no need to calculate gradients
          call RMSBestFit(roc,roi,ow,u2)
         endif ! proj_on
!
       else ! not deriv
         if (proj_on) then
          call RMSBestFit(rol,roi,ow,u)
          call RMSBestFit(ror,roi,ow,u1)
          if (fe_curvature) call RMSBestFit(roc,roi,ow,u2)
         else ! not proj_on
          call RMSBestFit(roc,roi,ow,u2) ! orientation between center image and current coords
         endif ! proj_on
       endif ! deriv
! rotate target structures to overlap with current
! conventional way (might be faster)
       if (proj_on .and. fe_curvature) then
        rfr_rot=0d0; rfl_rot=0d0; rfc_rot=0d0;
        do k=1,3; do j=1,3
           rfl_rot(:,j)=rfl_rot(:,j)+rfl(:,k)*u (j,k)
           rfr_rot(:,j)=rfr_rot(:,j)+rfr(:,k)*u1(j,k)
           rfc_rot(:,j)=rfc_rot(:,j)+rfc(:,k)*u2(j,k)
        enddo; enddo
!
       elseif (proj_on) then
        rfr_rot=0d0; rfl_rot=0d0;
        do k=1,3; do j=1,3
           rfl_rot(:,j)=rfl_rot(:,j)+rfl(:,k)*u (j,k)
           rfr_rot(:,j)=rfr_rot(:,j)+rfr(:,k)*u1(j,k)
        enddo; enddo
!
       else ! not proj_on
        rfc_rot=0d0
        do k=1,3; do j=1,3
           rfc_rot(:,j)=rfc_rot(:,j)+rfc(:,k)*u2(j,k)
        enddo; enddo
       endif ! proj_on
!
      else ! not qorient
       if (proj_on) then
        rfl_rot=>rfl; rfr_rot=>rfr ! no rotation
        if (fe_curvature) rfc_rot=>rfc
       else
        rfc_rot=>rfc
       endif
      endif ! qorient
!
      if (proj_on) then
! compute projection (delta)
       rho=0d0
       rho1=0d0
       rho2=0d0
       fe_curv=0d0
!
       if (deriv) then
!
        r1=0d0;
        do k=1,3 ; do j=1,nforced
                     d = rfi(j,k) -rfl_rot(j,k)
                     d1= rfr_rot(j,k)-rfl_rot(j,k)
                     d2= rfc_rot(j,k)-rfl_rot(j,k) ! for curvature
!
                     rfpar(j,k) = d ! displacement to first reference
                     rfprp(j,k) = d-d1 ! displacement to second reference
                     fe_curv = fe_curv + fw(j) * ( d1-2d0*d2 ) * (d-d2 ) ! curvature
!
                     rho1=rho1 + d1*d1 * fw(j) ! denominator in dpar
                     d1 = d1 * fw(j)
                     rho =rho + d *d1 ! numerator
! COM contribution to gradient for orientation atoms:
                     r1(k)=r1(k)+d1 ! will only be used if (qdiffrot)
! derivative components from the forcing atoms:
                     ffpar(j,k)=d1 !
        enddo ; enddo
        if (rho1.le.tol) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a rather unlikely event
        dpar=rho*rho1 ! length of projection along line connecting left and right structures
        rho=sqrt(rho1)! for normalization of perpendicular derivatives
        fe_curv = fe_curv * rho1 * 4d0 ! for internal pts will need x 4 because rho1 is (2)^2 times the string length increment ds
        omdpar=1d0-dpar
        d5=dpar*omdpar
! the next loop is required for the perpendicular component
        r2=0d0;
        do k=1,3 ; do j=1,nforced
                     d1 = dpar * rfprp(j,k) + omdpar * rfpar(j,k)
                     rfprp(j,k)=d1 ! true perpendicular component
                     rho2=rho2 + d1*d1 *fw(j) ! squared length of perpendicular vector
!
                     d1=d1 * fw(j)
! COM contribution to gradient for orientation atoms:
                     r2(k)=r2(k) + d1
! derivative components from the forcing atoms:
                     ffprp(j,k)=d1
        enddo ; enddo
!
        d6=d5+rho1*rho2 ! rho1*rho2 is the normalized perp. component squared
        dperp=sqrt(rho2) ! unnormalized perp component
        if (dperp.le.tol) then; rho2=1d0; else; rho2=1d0/dperp; endif ! a rather unlikely event
        dperp=dperp*rho ! normalize perp component
!
        if (qorient) then ! this part nonzero only if orientation is on
!
         w=omdpar-dpar ! (1-2d)
         if (qdiffrot) then
!
          do k=obeg, oend
! COM contribution to gradients on orientation atoms
            fopar(k,:)=-r1*ow(k) ! parallel
            foprp(k,:)=-r2*ow(k) ! perpendicular
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
! aa: this is what we are doing above: At`B + AtB` = (AtB)`
! ugrad2(:,:,j,k)=matmul(transpose(ugrad(:,:,j,k)),u1)+
! & matmul(transpose(u),ugrad1(:,:,j,k))
           enddo ! j
          enddo ! k
! contribution from quadratics (3 terms)
          do j=1, nforced
           r1=fw(j)*rfi(j,:)
           r2=fw(j)*rfl(j,:)
! r3=d5*r2
           r3=d6*r2 ! for normalized distance
           r2=w*r2
!
           do k=obeg, oend
!
            do p=1,3
             do q=1,3
!
              d =r1(p)
              d1=rfl(j,q)
              d2=rfr(j,q)
              d3=r2(p)*d2
              d4=r3(p)*d2
! not sure how to compute this more efficiently
              fopar(k,:)=fopar(k,:) + &
     & d * ( ugrad1(p,q,:,k) * d2 - ugrad(p,q,:,k) * d1 ) - &
     & d3 * ( ugrad2(p,q,:,k) )
!
              foprp(k,:)=foprp(k,:) - &
     & d * ( dpar * ugrad1(p,q,:,k) * d2 + &
     & omdpar * ugrad (p,q,:,k) * d1 ) + &
     & d4 * ( ugrad2(p,q,:,k) )
!
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
          enddo ! j (forcing atoms)
!
! scale orientation atoms derivatives
          fopar=rho1*fopar
          foprp=rho*rho2*foprp
!
         else ! not qdiffrot
! NOTE: in this case, the forces on the f and o atoms are stored in the same location, so that below we are adding to the f forces
! in particular, we do not initialize fopar/foprp because they point to ffpar/ffprp, which are already initialized
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
! aa: this is what we are doing above: At'B + AtB' = (AtB)`
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
           r2=fw(j)*rfl(j,:)
! r3=d5*r2
           r3=d6*r2
           r2=w*r2
!
           do k=obeg, oend
!
            do p=1,3
             do q=1,3
              d3=r2(p)*rfr(j,q)
              d4=r3(p)*rfr(j,q)
              fopar(k,:)=fopar(k,:) - d3 * ugrad2(p,q,:,k) ! parallel
              foprp(k,:)=foprp(k,:) + d4 * ugrad2(p,q,:,k) ! perpendicular
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
          enddo ! j (forcing atoms)
!
         endif ! qdiffrot
!
! NOTE: when diffrot false, fopar and ffpar point to the same thing; gather below should still work fine
         if (qgrp) then ! gather orientation forces
          if (mod(SIZE_LOCAL,2).eq.0.and.allgather_method.eq.hypercube_) then ! use hypercube allgather
           call hypercube_allgatherv( &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror, ME_LOCAL, SIZE_LOCAL )
           call hypercube_allgatherv( &
     & foprp, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror, ME_LOCAL, SIZE_LOCAL )
          elseif (allgather_method.eq.allgather_) then
! ALLGATHER
           call MPI_ALLGATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror)
           call MPI_ALLGATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & foprp, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror)
          elseif (allgather_method.eq.gather_bcast_.or.mod(SIZE_LOCAL,2).eq.1) then ! default method when cores are not in multiples of two
! GATHER + BCAST
           if (ME_LOCAL.eq.0) then
            call MPI_GATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
            call MPI_GATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & foprp, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
           else
            call MPI_GATHERV(fopar(obeg,1), &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
            call MPI_GATHERV(foprp(obeg,1), &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & foprp, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
           endif ! ME_LOCAL
! send to slaves
           call PSND8m(r_o(1,1,fpar),6*norient) ! perp follows par in memory, so send both by doubling data count
          endif ! hypercube_allgather
         endif ! qgrp
!
! free memory
         deallocate(ugrad, ugrad1, ugrad2)
         if (qgrp) deallocate(orient_count, orient_displ)
!
        endif ! qorient
! scale forcing atoms derivatives
        ffpar=rho1*ffpar
        ffprp=rho*rho2*ffprp
!----------------------------------------------------------------------------------------------
       else ! derivative calculation not requested
         do k=1,3 ; do j=1,nforced
                    d = rfi(j,k) -rfl_rot(j,k)
                    d1= rfr_rot(j,k)-rfl_rot(j,k)
                    d2= rfc_rot(j,k)-rfl_rot(j,k) ! for curvature
!
                    rfpar(j,k) = d ! displacement to first reference
                    rfprp(j,k) = d-d1 ! displacement to second reference
                    fe_curv = fe_curv + fw(j) * ( d1-2d0*d2 ) * (d-d2 ) ! curvature
!
                    rho =rho + d *d1 *fw(j) ! numerator
                    rho1=rho1 + d1*d1 *fw(j) ! denominator
         enddo ; enddo
         if (rho1.le.tol) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a very unlikely event
         dpar=rho*rho1 ! length projection along line connecting left and right structures
         fe_curv = fe_curv * rho1 * 4d0 ! for internal pts, will need x 4 because rho1 is (2)^2 times the string length increment ds
         omdpar=1d0-dpar
! the next loop is required for the perpendicular component
         do k=1,3 ; do j=1,nforced
                     d1 = dpar * rfprp(j,k) + omdpar * rfpar(j,k) !
                     rfprp(j,k)=d1 ! true perpendicular component
                     rho2=rho2 + d1*d1 *fw(j) ! length of perpendicular vector
         enddo ; enddo
!
        dperp=sqrt(rho2*rho1) ! normalized perpendicular length
!
       endif ! deriv
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      else ! not proj_on
! NOTE: will put forces into parallel force array; fperp is unused
! compute the RMSD
        drms=rmsd(rfi, rfc_rot, fw)
!
        if (deriv) then
! to scale derivatives by 1/drms
         if (drms.gt.tol) then
           rho=1d0/drms
         else
           rho=1d0 ! avoid singularity at near zero separation
         endif
!
         if (qdiffrot.and.qorient) then
! compute COM contribution to gradient for orientation atoms:
! and add forces on forcing atoms
          r1(:)=0d0;
          do k=1, nforced
           ffpar(k,:) = rho * fw(k) * (rfi(k,:)-rfc_rot(k,:))
           r1(:) = r1(:) + ffpar(k,:)
          enddo
          do j=obeg, oend
           fopar(j,:)=-r1(:)*ow(j)
          enddo
!
          do j=1, nforced
           r2=fw(j)*rfi(j,:) * rho
           do k=obeg, oend
            do p=1,3
             do q=1,3
              d=r2(p)*rfc(j,q)
!
              fopar(k,:)=fopar(k,:) - ugrad(p,q,:,k) * d
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
          enddo ! j (forcing atoms)
!
          deallocate(ugrad)
!
          if (qgrp) then ! gather orientation forces
           if (mod(SIZE_LOCAL,2).eq.0.and.allgather_method.eq.hypercube_) then ! use hypercube allgather
            call hypercube_allgatherv( &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror, ME_LOCAL, SIZE_LOCAL )
           elseif (allgather_method.eq.allgather_) then
!ALLGATHER
            call MPI_ALLGATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & MPI_COMM_LOCAL, ierror)
           elseif (allgather_method.eq.gather_bcast_.or.mod(SIZE_LOCAL,2).eq.1) then ! default method when cores are not in multiples of two
! GATHER + BCAST
            if (ME_LOCAL.eq.0) then
             call MPI_GATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
            else
             call MPI_GATHERV(fopar(obeg,1), &
     & orient_count(ME_LOCAL+1),MPI_RTMD_TYPE, &
     & fopar, orient_count, orient_displ, MPI_RTMD_TYPE, &
     & 0, MPI_COMM_LOCAL, ierror)
            endif
! send to slaves
            call PSND8m(fopar,3*norient)
           endif
          endif ! qgrp
!===========================================================================
         else ! not qdiffrot .or. not qorient
! apply forces to the forcing atoms
          do j=1,nforced
           ffpar(j,:)=rho * fw(j) * (rfi(j,:)-rfc_rot(j,:))
          enddo
         endif ! qdiffrot.and.qorient
!
         if (qgrp) deallocate(orient_count, orient_displ)
        endif ! deriv
!
      endif ! proj_on
!
      end subroutine ftsmv1_calc
!
!===========================================================================
      subroutine ftsm_addforce(fx,fy,fz,addforce)
      use number
! note: boundary points require special treatment: x0.25 force constant and x2 dperp0
! this is because distance between replicas is halved for the endpoints, but still corresponds to 1 (as for the inner pts)
      real(chm_real) :: fx(:), fy(:), fz(:)
      real(chm_real), pointer, dimension(:,:) :: ffpar, ffprp, fopar, foprp
      real(chm_real) :: pre1, pre2
      real(chm_real) :: fac, fac2, t, omt, wgt, dx, dy, dz, kbt=0.6d0 ! for now, hardwired at room temperature
!
      integer, pointer :: inds(:)
      real(chm_real), pointer :: wgts(:)
      integer :: i, j, ind
      logical :: addforce ! whether to add forces to running force average
      logical :: qendpoint
!
      qendpoint=(mestring.eq.0.or.mestring.eq.nstring-1)
!
      ffpar=>r_f(:,:,fpar); ffprp=>r_f(:,:,fperp) ! forcing forces
      fopar=>r_o(:,:,fpar); foprp=>r_o(:,:,fperp) ! orientation forces
!
      if (addforce) then
       num_force_samples=num_force_samples+1
       omt=1d0/num_force_samples
       t=1d0-omt
      else
       t=1d0; omt=0d0
      endif
!
      if (ftsm_scaledist) then
! lowering the force constants for endpoints is necessary to get the equivalent fluctuations
! can think of either scaling the force constants, or scaling the distances (_and gradients_) by 0.5 and keeping the same force constans
! perpendicular component normalized by physical interimage dist. (which is halved for endpoints); so it appears too large, and is halved below
!
! for the internal points, integrating the force through one inter-image distance is equivalent to multiplying
! the force by 0.5 (0.5 is the scaled distance between adjacent inner images)
! for the endpoints, the distance between adjacent images is 1, so take the force (scaled down) and multiply by 1
       if (qendpoint) then ; fac=0.5d0 ; fac2=1d0 ; fe_curv=0d0 ! no curvature at endpoints
                      else ; fac=1.0d0 ; fac2=0.5d0
       endif
! no need to scale -- using dimensional distances
      else
       fac=one ; fac2=one ;
       if (qendpoint) fe_curv=zero ; ! the endpoint curvature from ftsmv2_compute is scaled by 4 (incorrect) but we do not use it here
      endif
!
      if (proj_on) then
! restraint force parallel to string
       pre1 = kpara * fac * fac * ( dpar-dpar0 ) ! scale down the force constant of endpoints (one for d, one for gradients)
       pre2 = kperp * fac * max ( fac * dperp - dperp0, zero ) ! ignore negative values; dperp0 criterion in inner-replica d-metric
! force limiting
       if (ftsm_flim_coeff.gt.zero) then
        pre1=sign( min(abs(pre1),sqrt(two*fac*kbt*kpara*ftsm_flim_coeff)),pre1) ! limit pre1 magnitude taking care of sign
        pre2=min(pre2,sqrt(two*fac*kbt*kperp*ftsm_flim_coeff)) ! pre2 > 0
       endif
! update force: for inner points, multiply force by 0.5; for boundary use 1d0; the FE is then simply the sum of the forces along the string
       avforce(1) = t * avforce(1) + omt * pre1 * fac2 ! force acting on dpar0 is pre1 (integrate negative to get f.e.)
       avforce(2) = t * avforce(2) - omt * pre1 * fac2 * (fe_curv * fac * fac) ! force due to curvature (note: either fac=1 or curv=0)
       avforce(3) = t * avforce(3) + omt * pre2 * fac2 ! force acting on dperp0 is pre2 ( NOTE: this is not a true force along the path)
!
      else ! .not. proj_on
       pre1 = krms * ( drms-drms0 )
       if (qrms_upper_bound) pre1=max(pre1,zero) ! apply if force exceeds threshold
! force limiting
       if (ftsm_flim_coeff.gt.zero) then
        pre1=sign( min(abs(pre1),sqrt(two*fac*kbt*krms*ftsm_flim_coeff)),pre1) ! limit pre1 magnitude taking care of sign
       endif
       pre2=zero
! update force (which will not likely be used)
       avforce(1) = t * avforce(1) + omt * pre1 ! force acting on dpar0 is pre1 (integrate negative to get f.e.)
       avforce(2:3)=zero ! set to zero in case user runs rmsd dynamics after proj dynamics
      endif ! proj_on
! ! even if I am treating it here as such; it is an orthogonal `correction`
      if (pre2 .gt. zero) then
       if (ftsm_com_on) then
        do i=1, nforced
          dx=pre1 * ffpar(i,1) + pre2 * ffprp(i,1)
          dy=pre1 * ffpar(i,2) + pre2 * ffprp(i,2)
          dz=pre1 * ffpar(i,3) + pre2 * ffprp(i,3)
          inds=>iatoms_f%v(i)%i
          wgts=>wgts_f%v(i)%r
          do j=1, iatoms_f%v(i)%last
           ind=inds(j)
           wgt=wgts(j)
           fx(ind)=fx(ind)+(wgt*dx);
           fy(ind)=fy(ind)+(wgt*dy);
           fz(ind)=fz(ind)+(wgt*dz);
          enddo ! j
        enddo ! i
       else ! ftsm_com_on
        do i=1, nforced
          ind=iatom_f(i)
          fx(ind) = fx(ind) + pre1 * ffpar(i,1) + pre2 * ffprp(i,1)
          fy(ind) = fy(ind) + pre1 * ffpar(i,2) + pre2 * ffprp(i,2)
          fz(ind) = fz(ind) + pre1 * ffpar(i,3) + pre2 * ffprp(i,3)
        enddo
       endif ! ftsm_com_on
!
       if (qorient.and.qdiffrot) then
        if (ftsm_com_on) then
         do i=1, norient
           dx=pre1 * fopar(i,1) + pre2 * foprp(i,1)
           dy=pre1 * fopar(i,2) + pre2 * foprp(i,2)
           dz=pre1 * fopar(i,3) + pre2 * foprp(i,3)
           inds=>iatoms_o%v(i)%i
           wgts=>wgts_o%v(i)%r
           do j=1, iatoms_o%v(i)%last
            ind=inds(j)
            wgt=wgts(j)
            fx(ind)=fx(ind)+(wgt*dx);
            fy(ind)=fy(ind)+(wgt*dy);
            fz(ind)=fz(ind)+(wgt*dz);
           enddo ! j
         enddo ! i
        else ! ftsm_com_on
         do i=1, norient
           ind=iatom_o(i)
           fx(ind) = fx(ind) + pre1 * fopar(i,1) + pre2 * foprp(i,1)
           fy(ind) = fy(ind) + pre1 * fopar(i,2) + pre2 * foprp(i,2)
           fz(ind) = fz(ind) + pre1 * fopar(i,3) + pre2 * foprp(i,3)
         enddo
        endif ! ftsm_com_on
       endif ! qorient && qdiffrot
!
      else ! parallel force only
       if (ftsm_com_on) then
        do i=1, nforced
          dx=pre1 * ffpar(i,1)
          dy=pre1 * ffpar(i,2)
          dz=pre1 * ffpar(i,3)
          inds=>iatoms_f%v(i)%i
          wgts=>wgts_f%v(i)%r
          do j=1, iatoms_f%v(i)%last
           ind=inds(j)
           wgt=wgts(j)
           fx(ind)=fx(ind)+(wgt*dx);
           fy(ind)=fy(ind)+(wgt*dy);
           fz(ind)=fz(ind)+(wgt*dz);
          enddo ! j
        enddo ! i
       else ! ftsm_com_on
        do i=1, nforced
          ind=iatom_f(i)
          fx(ind) = fx(ind) + pre1 * ffpar(i,1)
          fy(ind) = fy(ind) + pre1 * ffpar(i,2)
          fz(ind) = fz(ind) + pre1 * ffpar(i,3)
        enddo
       endif ! ftsm_com_on
!
       if (qorient.and.qdiffrot) then
        if (ftsm_com_on) then
         do i=1, norient
           dx=pre1 * fopar(i,1)
           dy=pre1 * fopar(i,2)
           dz=pre1 * fopar(i,3)
           inds=>iatoms_o%v(i)%i
           wgts=>wgts_o%v(i)%r
           do j=1, iatoms_o%v(i)%last
            ind=inds(j)
            wgt=wgts(j)
            fx(ind)=fx(ind)+(wgt*dx);
            fy(ind)=fy(ind)+(wgt*dy);
            fz(ind)=fz(ind)+(wgt*dz);
           enddo ! j
         enddo ! i
        else ! ftsm_com_on
         do i=1, norient
           ind=iatom_o(i)
           fx(ind) = fx(ind) + pre1 * fopar(i,1)
           fy(ind) = fy(ind) + pre1 * fopar(i,2)
           fz(ind) = fz(ind) + pre1 * fopar(i,3)
         enddo
        endif ! ftsm_com_on
       endif ! qorient
      endif ! pre2>0
!
      end subroutine ftsm_addforce
!============================================================================
      subroutine ftsm_compute_fe_fd()
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use number
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real) :: avforces(3,nstring), qcurv
      integer :: ierror, i
      character(len=len("FTSM_COMPUTE_FE_FD>") ),parameter::whoami="FTSM_COMPUTE_FE_FD>";!macro
!
      if (proj_on) then
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and. &
     & SIZE_STRNG.gt.1) then
        call MPI_GATHER(avforce, 3, mpifloat, &
     & avforces, 3, mpifloat, &
     & 0, MPI_COMM_STRNG, ierror)
        fe(1)=0d0
!
        if (fe_curvature) then ; qcurv=one ; else ; qcurv=zero ; endif
!
        if (ftsm_scaledist) then
         do i=2, nstring
          fe(i) =fe(i-1) - half * ( avforces(1,i-1) + avforces(1,i) + qcurv*( avforces(2,i-1) + avforces(2,i) ) )
! fe(i-1)=fe(i-1) - ( avforces(2,i-1) - avforces(2,1) )
         enddo
! fe(nstring)=fe(nstring) - ( avforces(2,nstring) - avforces(2,1))
        else ! integrate along curve using arclength in ds
         do i=2, nstring
          fe(i) =fe(i-1) - ds(i-1) * sqrt3 * half * ( avforces(1,i-1) + avforces(1,i) + qcurv*( avforces(2,i-1) + avforces(2,i) ) )
         enddo
        endif
!
       endif ! qroot
      else ! explicitly set to zero for rmsd dynamics
       fe=zero
      endif ! proj_on
!
! send free energy to slaves -- no need at present time
! if (MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.
! & SIZE_LOCAL.gt.1) then
! call call PSND8(fe,nstring)
! endif
!
      end subroutine ftsm_compute_fe_fd
!===========================================================================
      subroutine ftsm_calc_M(x,y,z,mass,t)
! 5.2013 : TODO : debug the case qdiffrot=1
! debugging output not cleaned up
      use number
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real), dimension(:), intent(in) :: x, y, z, mass
      real(chm_real), intent(in), optional :: t
! local vars
      real(chm_real) :: s ! this value indicates how much of the old reference set to take
      real(chm_real) :: tol ! small number tolerance
      real(chm_real) :: u(3,3), ug2(3,3), ug3(3,3,3), ug3i(3,3,3), ugrad_dyad_ave(3,3,3,4) ! for rotations (with extra space)
      real(chm_real), dimension(:,:), pointer :: roi, rfi, roc
      real(chm_real), dimension(:,:,:,:), pointer :: ugrad
      real(chm_real), pointer :: ow(:)
      real(chm_real), pointer :: r_com(:)
      logical :: qgrp, qcombine
      integer :: i, a, b, c, d, e, j, k, n ! indices used in the derivation
      real(chm_real) :: rj(3), rk(3) ! used in the analytical derivation
      integer*4, pointer :: send_count(:), send_displ(:) ! if declared as integer*8 parallelization will break
      integer :: obeg, oend, fbeg, fend, indo, indf
      real(chm_real), pointer :: Mt(:,:,:,:)
      real(chm_real) :: Mtemp(3,3,nforced,nforced) ! scratch matrix
      real(chm_real) :: oom(nforced)
      tol=RSMALL
!
      if (present(t)) then ; s=min(max(t,zero),one); else ; s=one ; endif
      qcombine=s.lt.one
!
      qgrp=( SIZE_LOCAL.gt.1.and.MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.calc_bestfit_grad_para)
! shorthand
      ow=>orientWeights
      r_com=>rcom(:,instant)
      roi=>r_o(:,:,instant);
      rfi=>r_f(:,:,instant);
!
      if (qcombine.and.qorient) then ! straight average
       r_o(:,:,center_cur) = s * r_o(:,:,center) + (one-s) * r_o(:,:,center_old)
       roc=>r_o(:,:,center_cur)
      else
       roc=>r_o(:,:,center);
      endif
!
      Mt=>Mtensor(:,:,:,:,1);
!--------------------------------------------------------------------------------------
! load coordinates -- simple form (currently no support for FTSM COM)
!
      if (.not. restraint_force_on) then ! if restraints are on coordinates already loaded
       do k=1,nforced
        indf=iatom_f(k); rfi(k,1)=x(indf); rfi(k,2)=y(indf); rfi(k,3)=z(indf)
       enddo
!
       if (qorient) then
        if (qdiffrot) then
         do k=1,norient ! when qorient false, norient zero
          indo=iatom_o(k); roi(k,1)=x(indo); roi(k,2)=y(indo); roi(k,3)=z(indo)
         enddo
        endif ! qdiffrot (otherwise rfi and roi point to the same thing)
!
! translate forced atoms to centroid
        r_com(:)=zero;
        do j=1,3 ; do k=1, norient;
           r_com(j) = r_com(j)+ow(k)*roi(k,j)
        enddo ; enddo
!
        rfi(:,1)=rfi(:,1)-r_com(1); rfi(:,2)=rfi(:,2)-r_com(2); rfi(:,3)=rfi(:,3)-r_com(3)
!
        if (qdiffrot) then ! also use orientation atoms (otherwise, they are the same -- see above!)
         roi(:,1)=roi(:,1)-r_com(1); roi(:,2)=roi(:,2)-r_com(2); roi(:,3)=roi(:,3)-r_com(3)
        endif ! qdiffrot
!
       endif ! qorient
      endif ! (.not. restraint_force_on)
!
! load masses
      do i=1,nforced
       oom(i) = mass(iatom_f(i)) ; if (oom(i).gt.tol) then ; oom(i)=one/oom(i) ; else ; oom(i) = one ; endif
      enddo
!
! aa
! write(0,*) 'qgrp, qorie :', qgrp, qorient, calc_bestfit_grad_para, ME_LOCAL, SIZE_LOCAL
! write(0,*) '1/m :', oom
!
      Mt=zero
      Mtemp=zero
!
      if (qorient) then
! compute rotation matrix gradient
!--------------------- indices for parallelization
        if (qgrp) then
         j=ceiling(one*norient/SIZE_LOCAL)
         obeg=min(j*ME_LOCAL,norient-1) + 1
         oend=obeg - 1 + max(0,min(j,norient-j*ME_LOCAL ))
        else ! not qgrp
         obeg=1; oend=norient
        endif ! qgrp
! allocate memory
        allocate(ugrad(3,3,3,norient))
! need to calculate the matrix to fit roi onto roc and its derivatives
        call RMSBestFit(roc,roi,ow,u,obeg,oend,ugrad) ! fitting roc onto roi, and computing derivatives w.r.t. roi;
! ==> *** to get the desired matrix & derivative take TRANSPOSE (which corresponds to an opposite rotation)
! compute entries in M that require only one derivative
! this loop gooes over atoms that are both orientation and forcing atoms
        if (qdiffrot) then
         do i=1, nboth ! overlapping atoms
          indo=iatom_both(2,i) ! orientation index
          if ( (indo.ge.obeg).and.(indo.le.oend) ) then ! have the corresponding derivative
           ug3=ugrad(:,:,:,indo) ! prefetch
! precompute matrix derivative term A . grad A (3 components)
           ug3i=zero
           do c=1,3 ! dot product variable
            do b=1,3 ; do a=1,3 ; do e=1,3
             ug3i(e,a,b) = ug3i(e,a,b) + u(c,a)*ug3(e,b,c) ! note transposition
            enddo ; enddo ; enddo
           enddo ! c
! now compute sum over all forcing atoms
           j=iatom_both(1,i) ! forcing index
           do k=1, nforced
            rk = oom(j) * rfi(k,:)
            do a=1,3 ; do b=1,3
             s = dot_product(ug3i(:,a,b),rk)
             Mt(a,b,j,k) = Mt(a,b,j,k) + s
             Mt(b,a,k,j) = Mt(b,a,k,j) + s
            enddo ; enddo
           enddo ! k
          endif ! orientation index in range
         enddo ! i -- overlapping atoms
!------------------------------------------------------------------
        else ! .not. qdiffrot : orientation indices/coordinates the same and can be used interchangeably
         do j=obeg, oend
           ug3=ugrad(:,:,:,j)
! precompute matrix derivative term A . grad A (3 components)
           ug3i=zero
           do c=1,3 ! dot product variable
            do b=1,3 ; do a=1,3 ; do e=1,3
             ug3i(e,a,b) = ug3i(e,a,b) + u(c,a)*ug3(e,b,c) ! note transposition
            enddo ; enddo ; enddo
           enddo ! c
! now compute sum over all forcing atoms
           do k=1, nforced
            rk = oom(j)*rfi(k,:)
            do a=1,3 ; do b=1,3
             s = dot_product(ug3i(:,a,b),rk)
             Mt(a,b,j,k) = Mt(a,b,j,k) + s
             Mt(b,a,k,j) = Mt(b,a,k,j) + s
            enddo ; enddo
           enddo ! k
         enddo ! i -- orientation atoms
        endif ! qdiffrot
!
! compute terms that involve sums of matrix derivatives
!
        ugrad_dyad_ave=zero
        ug3=zero
!
        do n=obeg, oend
         do c=1,3
          ug2=ugrad(:,:,c,n) ! prefetch
!
          do a=1,3 ; do b=1,3 ; do d=1,3 ;
           ug3(d,b,a) = ug3(d,b,a) + oom(n) * ow(n) * u(c,b)*ug2(d,a) ! note the transposition
           do e=1,3
! dyadic product
           ugrad_dyad_ave(e,d,b,a) = ugrad_dyad_ave(e,d,b,a) + oom(n) * ug2(d,a)*ug2(e,b) ! note the transposition
           enddo
          enddo; enddo; enddo
!
         enddo ! c component of orientation atom
        enddo ! orientation index
!
!aa
! if (ME_STRNG.eq.0) then
! write(100,*) r_o(:,:,center_old);
! write(100,*) r_o(:,:,center);
! write(100,*) r_o(:,:,center_cur);
!
! write(100,*) roc;
! write(100,*) roi;
! write(100,*) ow;
! write(100,*) ugrad;
! close(100)
! write(200,*) 'TRIPLET: ',ug3
! write(200,*) 'DYAD: ',ugrad_dyad_ave
! write(200,*) 'OOM: ',oom
! write(200,*) 'U: ',u
! close(200)
! endif
!
!
! now need to reduce matrix sums on all slaves
!
        if (qgrp) then
         ugrad_dyad_ave(:,:,:,4)=ug3
         i=3**3 * 4 ! number of elements
         call MPI_ALLREDUCE(MPI_IN_PLACE, ugrad_dyad_ave, i, mpifloat, MPI_SUM, MPI_COMM_LOCAL, j)
         ug3=ugrad_dyad_ave(:,:,:,4)
! also sum up partial tensor Mt on the root processor
         i=(3*nforced)**2
         if (ME_LOCAL.eq.0) then ! use the same array on root
          call MPI_REDUCE(MPI_IN_PLACE, Mt, i, mpifloat, MPI_SUM, 0, MPI_COMM_LOCAL, j)
         else
          call MPI_REDUCE(Mt, MPI_UNDEFINED, i, mpifloat, MPI_SUM, 0, MPI_COMM_LOCAL, j)
         endif ! root
! aa -- debug
! if (ME_STRNG.eq.0) then
! write(50+ME_STRNG,'(30F12.6)') ( (Mt(:,a,:,k), a=1,3), k=1,nforced ) ; close(50+me_strng)
! endif
! Mt=zero;
! return
!aa
!
! now can compute the other terms
! (1) recompute send indices
         j=ceiling(one*nforced/SIZE_LOCAL)
         allocate(send_displ(SIZE_LOCAL), send_count(SIZE_LOCAL))
         do i=1,SIZE_LOCAL
          send_displ(i)=min((i-1)*j,nforced-1)
          send_count(i)=max(0,min(j,nforced-j*(i-1)))
         enddo
!
         fbeg=send_displ(ME_LOCAL+1) + 1
         fend=fbeg - 1 + send_count(ME_LOCAL+1)
!
         i=9*nforced
         send_displ=i*send_displ
         send_count=i*send_count
!
! (2) compute quadratics
         do k=fbeg, fend
          rk=rfi(k,:)
! precompute
          ug2=zero
          do d=1,3
           ug2=ug2 + ug3(d,:,:)*rk(d)
          enddo
!
          do j=1, nforced
           rj=rfi(j,:)
!
           do a=1,3 ; do b=1,3
!
             s=zero
             do d=1,3 ; do e=1,3
              s=s+rj(d)*rk(e)*ugrad_dyad_ave(e,d,b,a)
             enddo ; enddo
!
! last contribution from rotations; zero if the orientation weights are proportional to mass
!
             do d=1,3
              s = s - ug3(d,b,a)*rj(d) ! could be computed once outside of k-loop for all k`s, but this suffices for now
             enddo
             s = s - ug2(a,b) ! precomputed before j-loop
             Mtemp(a,b,j,k) = s
!
           enddo ; enddo ! a and b
          enddo ; ! j
         enddo ; ! k
! (3) gather partial M-matrices
         if (ME_LOCAL.eq.0) then
          call MPI_GATHERV(MPI_IN_PLACE, &
     & send_count(ME_LOCAL+1),mpifloat, &
     & Mtemp, send_count, send_displ, mpifloat, &
     & 0, MPI_COMM_LOCAL, i)
         else
          call MPI_GATHERV(Mtemp(1,1,1,fbeg), &
     & send_count(ME_LOCAL+1),mpifloat, &
     & MPI_UNDEFINED, send_count, send_displ, mpifloat,&
     & 0, MPI_COMM_LOCAL, i)
         endif
!
! aa
! if (ME_STRNG.eq.0) then
! write(60+ME_STRNG,'(30F12.6)') ( (Mtemp(:,a,:,k), a=1,3), k=1,nforced ) ; close(60+me_strng)
! write(30+me_strng,*) ug3 ; close(30+me_strng) ! should be zero for mass-weighting
! write(40+me_strng,*) ugrad_dyad_ave(:,:,:,1:3) ; close(40+me_strng) ! not necessarily zero
! endif
! aa
         deallocate(send_count, send_displ)
!
        else ! not qgrp
! aa
! if (ME_STRNG.eq.0) then
! write(500+ME_STRNG,'(30F12.6)') ( (Mt(:,a,:,k), a=1,3), k=1,nforced ) ; close(500+me_strng)
! endif
! Mt=zero
! return
! aa
         do k=1, nforced
          rk=rfi(k,:)
          do j=1, k-1
           rj=rfi(j,:)
           do a=1,3 ; do b=1,3
             s=zero
             do d=1,3 ; do e=1,3
              s=s+rj(d)*rk(e)*ugrad_dyad_ave(e,d,b,a)
             enddo ; enddo
             Mtemp(a,b,j,k) = s
             Mtemp(b,a,k,j) = s
           enddo ; enddo ! a and b
          enddo ; ! j
! ---------- diagonal atom submatrices (there is still a little bit of duplication left (a<->b))
          do a=1,3 ; do b=1,3
             s=zero
             do d=1,3 ; do e=1,3
              s=s+rk(d)*rk(e)*ugrad_dyad_ave(e,d,b,a)
             enddo ; enddo
             Mtemp(a,b,k,k) = s
          enddo ; enddo ! a and b
! --------
         enddo ; ! k
!
! (3) compute remaining contribution from rotation derivatives; zero if the orientation weights are proportional to mass
         do j=1, nforced
          rj = rfi(j,:)
          do a=1,3 ; do b=1,3
           s = dot_product(ug3(:,b,a),rj)
! all forcing atoms have a contribution
           do k=1,nforced
            Mtemp(a,b,j,k) = Mtemp(a,b,j,k) + s
            Mtemp(b,a,k,j) = Mtemp(b,a,k,j) + s
           enddo ! k
!
          enddo ; enddo
         enddo ! j
! aa -- debug
! if (ME_STRNG.eq.0) then
! write(600+ME_STRNG,'(30F12.6)') ( (Mtemp(:,a,:,k), a=1,3), k=1,nforced ) ; close(600+me_strng)
! write(300+me_strng,*) ug3 ; close(300+me_strng) ! should be zero for mass-weighting
! write(400+me_strng,*) ugrad_dyad_ave(:,:,:,1:3) ; close(400+me_strng) ! not necessarily zero
! endif
! Mtemp=zero
! aa
        endif ! qgrp
! add contribution from Mtemp
        Mt=Mt+Mtemp
!
        if (qdiffrot) then
         do i=1, nboth ! overlapping atoms
          indo=iatom_both(2,i) ! orientation index
          k=iatom_both(1,i) ! forcing index
!
          s = -ow(indo)*oom(k) ! this reduces to a constant for all atoms if the orientation weights are proportional to mass
          do a=1,3
           Mt(a,a,:,k) = Mt(a,a,:,k) + s
           Mt(a,a,k,:) = Mt(a,a,k,:) + s
          enddo
         enddo
!
        else ! not qdiffrot
!
         do k=1, nforced
          s = -ow(k)*oom(k) ! constant if the orientation weights are proportional to mass
          do a=1,3
           Mt(a,a,:,k) = Mt(a,a,:,k) + s
           Mt(a,a,k,:) = Mt(a,a,k,:) + s
          enddo
         enddo
!
        endif ! qdiffrot
! aa
! if (ME_STRNG.eq.0) then
! write(700+ME_STRNG,'(30F12.6)') ( (Mt(:,a,:,k), a=1,3), k=1,nforced ) ; close(700+me_strng)
! endif
! Mt=zero;
! aa
! sum w_n^2 /m_n :
        s=zero
        do i=1, norient ; s=s+ow(i)*ow(i)*oom(i) ; enddo ! s = 1/sum(mass) if the weights are normalized masses
        do a=1,3 ; Mt(a,a,:,:)=Mt(a,a,:,:) + s ; enddo
! aa
! if (ME_STRNG.eq.0) then
! write(800+ME_STRNG,'(30F12.6)') ( (Mt(:,a,:,k), a=1,3), k=1,nforced ) ; close(800+me_strng)
! endif
! aa
!
        if (associated(ugrad)) deallocate(ugrad)
!
      endif ! qorient
!
! remaining term : d_ab x d_jk / m
!
      do a=1,3 ; do j=1,nforced
        Mt(a,a,j,j) = Mt(a,a,j,j) + oom(j)
      enddo ; enddo
! aa
! if (ME_STRNG.eq.0) then
! write(900+ME_STRNG,'(30F12.6)') ( (Mt(:,a,:,k), a=1,3), k=1,nforced ) ; close(900+me_strng)
! endif
!
! aa
      end subroutine ftsm_calc_M
!===========================================================================
! much of this (again) taken from calc; this is OK because this routine is
! not needed for production simulations
!
      subroutine ftsm_calc_J(x,y,z,t)
!
      use number
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
      real(chm_real), dimension(:), intent(in) :: x, y, z
      real(chm_real), intent(in), optional :: t
! local vars
      real(chm_real) :: s, oms ! this value indicates how much of the old reference set to take
      real(chm_real) :: tol ! small number tolerance
      real(chm_real) :: u(3,3,3) ! three rotation matrices
      real(chm_real) :: r1(3) ! aux. vector
      real(chm_real), pointer, dimension(:,:) :: roi,rol,ror,roc, &
     & rol_old,roc_old,ror_old,rol_cur,roc_cur,ror_cur
      real(chm_real), pointer, dimension(:,:) :: rfi,rfl,rfr,rfc,rfl_rot &
     & ,rfr_rot,rfl_old,rfc_old,rfr_old,rfl_cur,rfc_cur,rfr_cur
      real(chm_real), dimension(:,:,:,:), pointer :: ugrad
      real(chm_real), pointer :: ow(:), fw(:), ug(:,:)
      real(chm_real), pointer :: r_com(:)
      logical :: qgrp, qcombine
      integer :: i, a, b, c ! indices used in the derivation
      integer*4, pointer :: orient_count(:), orient_displ(:) ! for integer*8 parallelization break with some compilers
      integer :: j, k, indo, indf, obeg, oend
      real(chm_real) :: jo(3,norient), jf(3,nforced) ! (Jacobian) derivative vectors (orientation and forced contributions)
      real(chm_real) :: d, d1, rho ! aux. vars
!
      tol=RSMALL
!
      if (present(t)) then ; s=min(max(t,zero),one); else ; s=one ; endif
      qcombine=s.lt.one
!
      qgrp=( SIZE_LOCAL.gt.1.and.MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.calc_bestfit_grad_para)
! shorthand
      ow=>orientWeights
      r_com=>rcom(:,instant)
      roi=>r_o(:,:,instant);
      rol=>r_o(:,:,left); ror=>r_o(:,:,right); roc=>r_o(:,:,center);
!--------------------------------------------------------------------------------------------
      fw=>forcedWeights
      rfi=>r_f(:,:,instant);
      rfl=>r_f(:,:,left); rfr=>r_f(:,:,right); rfc=>r_f(:,:,center);
      rfl_rot=>r_f(:,:,left_rot); rfr_rot=>r_f(:,:,right_rot)
!
!---- load coordinates ---------------------------------------------------------------------------
! write(ME_STRNG + 100 , *) ME_STRNG, ME_LOCAL, qcombine, restraint_force_on ! aa
!
      if (.not. restraint_force_on) then ! if restraints are on, coordinates already loaded (called after calc in ftsm)
       do k=1,nforced
        indf=iatom_f(k); rfi(k,1)=x(indf); rfi(k,2)=y(indf); rfi(k,3)=z(indf)
       enddo
!
       if (qorient) then
        if (qdiffrot) then
         do k=1,norient ! when qorient false, norient zero
          indo=iatom_o(k); roi(k,1)=x(indo); roi(k,2)=y(indo); roi(k,3)=z(indo)
         enddo
        endif ! qdiffrot (otherwise rfi and roi point to the same thing)
!
! translate forced atoms to centroid
        r_com(:)=zero;
        do j=1,3 ; do k=1, norient;
           r_com(j) = r_com(j)+ow(k)*roi(k,j)
        enddo ; enddo
!
        rfi(:,1)=rfi(:,1)-r_com(1); rfi(:,2)=rfi(:,2)-r_com(2); rfi(:,3)=rfi(:,3)-r_com(3)
!
        if (qdiffrot) then ! also use orientation atoms (otherwise, they are the same)
         roi(:,1)=roi(:,1)-r_com(1); roi(:,2)=roi(:,2)-r_com(2); roi(:,3)=roi(:,3)-r_com(3)
        endif ! qdiffrot
!
       endif ! qorient
      endif ! (.not. restraint_force_on)
!
!---------------------------------------------------------------------------------------------
      if (qcombine) then ! use a combination of old and new reference structures
       oms=one-s
!
       rfl_old=>r_f(:,:,left_old); rfr_old=>r_f(:,:,right_old); rfc_old=>r_f(:,:,center_old);
       rfl_cur=>r_f(:,:,left_cur); rfr_cur=>r_f(:,:,right_cur); rfc_cur=>r_f(:,:,center_cur);
!
       rol_old=>r_o(:,:,left_old); ror_old=>r_o(:,:,right_old); roc_old=>r_o(:,:,center_old);
       rol_cur=>r_o(:,:,left_cur); ror_cur=>r_o(:,:,right_cur); roc_cur=>r_o(:,:,center_cur);
!
       if (qorient) then
        call RMSBestFit(rol_old, rol, ow,u(:,:,1))
        call RMSBestFit(ror_old, ror, ow,u(:,:,2))
        call RMSBestFit(roc_old, roc, ow,u(:,:,3))
!
! combine rotated structures
!
        u = oms * u ! prescale rotation matrix
        rfl_cur=0d0; rfr_cur=0d0; rfc_cur=0d0;
        do k=1,3; do j=1,3
           rfl_cur(:,j)=rfl_cur(:,j) + s * rfl(:,k) + u(j,k,1) * rfl_old(:,k)
           rfr_cur(:,j)=rfr_cur(:,j) + s * rfr(:,k) + u(j,k,2) * rfr_old(:,k)
           rfc_cur(:,j)=rfc_cur(:,j) + s * rfc(:,k) + u(j,k,3) * rfc_old(:,k)
        enddo; enddo
!
        if (qdiffrot) then
          rol_cur=0d0; ror_cur=0d0; roc_cur=0d0;
          do k=1,3; do j=1,3
           rol_cur(:,j)=rol_cur(:,j) + s * rol(:,k) + u(j,k,1) * rol_old(:,k)
           ror_cur(:,j)=ror_cur(:,j) + s * ror(:,k) + u(j,k,2) * ror_old(:,k)
           roc_cur(:,j)=roc_cur(:,j) + s * roc(:,k) + u(j,k,3) * roc_old(:,k)
          enddo; enddo
        endif ! qdiffrot
!
       else ! not qorient
!
        do j=1,3
           rfl_cur(:,j) = s * rfl(:,j) + oms * rfl_old(:,j)
           rfr_cur(:,j) = s * rfr(:,j) + oms * rfr_old(:,j)
           rfc_cur(:,j) = s * rfc(:,j) + oms * rfc_old(:,j)
        enddo;
!
       endif ! qorient
! point to combined reference structures
       rfl=>rfl_cur
       rfr=>rfr_cur
       rfc=>rfc_cur
       rol=>rol_cur
       ror=>ror_cur
       roc=>roc_cur
!
      endif ! qcombine
!
! superpose structures onto roc
!%%%%%%%%%%%%%%%%%%% set up indices for gradients %%%%%%%%%%%%%%%%%%%
      if (qorient) then
! set up indices
       if (qgrp) then ! in parallel
        j=ceiling(one*norient/SIZE_LOCAL)
        allocate(orient_displ(SIZE_LOCAL), orient_count(SIZE_LOCAL))
        do i=1,SIZE_LOCAL
         orient_displ(i)=min((i-1)*j,norient-1)
         orient_count(i)=max(0,min(j,norient-j*(i-1)))
        enddo
        obeg=orient_displ(ME_LOCAL+1) + 1
        oend=obeg - 1 + orient_count(ME_LOCAL+1)
!
        orient_displ=3*orient_displ ! three components per atom index
        orient_count=3*orient_count
       else ! not qgrp
        obeg=1; oend=norient
       endif ! qgrp
!
       allocate(ugrad (3,3,3,norient))
!
       call RMSBestFit(rol,roc,ow,u(:,:,1))
       call RMSBestFit(ror,roc,ow,u(:,:,2))
       call RMSBestFit(roi,roc,ow,u(:,:,3),obeg,oend,ugrad) ! roi => roc superposition
! rotate target structures to overlap with current center
! to be able to compute tangent vector
       rfr_rot=0d0; rfl_rot=0d0;
       do k=1,3; do j=1,3
          rfl_rot(:,j)=rfl_rot(:,j)+rfl(:,k)*u(j,k,1)
          rfr_rot(:,j)=rfr_rot(:,j)+rfr(:,k)*u(j,k,2)
       enddo; enddo
!
      else ! not qorient
        rfl_rot=>rfl; rfr_rot=>rfr ! no rotation
! define trivial identity rotation
        u(:,:,3) = Id3
      endif ! qorient
!
! compute gradient vector
!
! first compute correction for absolute coords
      rho=zero ! tangent vector norm
      r1=zero
      jf=zero
      do j=1,nforced
        do a=1,3
         d = rfr_rot(j,a)-rfl_rot(j,a) ! tangent vector
         d1 = d * fw(j)
         rho =rho + d*d1 ! tangent normalization factor
!
         do c=1,3
          d = d1 * u(a,c,3)
          r1(c) = r1(c) + d ! correction sum
          jf(c,j) = jf(c,j) + d ! forced part of jacobian
         enddo ! c
        enddo !a
      enddo ! forced
!
! write(ME_STRNG + 100 , *) 'jf ',jf
! write(ME_STRNG + 100 , *) 'fw ',fw
! write(ME_STRNG + 100 , *) 'rho ',rho
!
      if (qorient) then
       jo=zero
! write(ME_STRNG + 100 , *) 'jo ',jo
! write(ME_STRNG + 100 , *) 'obeg, oend ',obeg, oend
! write(ME_STRNG + 200 , *) 'ugrad:', ugrad(:,:,:,:)
!
       do j=obeg, oend ! orientation indices
        do c=1,3 ! cartesian components
         ug=>ugrad(:,:,c,j)
! loop over all forced indices
         do i=1,nforced
          do a=1,3
           d = rfr_rot(i,a)-rfl_rot(i,a) ! tangent vector
           d1 = d * fw(i)
           do b=1,3
            jo(c,j) = jo(c,j) + d1 * ug(a,b) * rfi(i,b) ! quadratic part involving matrix derivative
           enddo
          enddo
         enddo ! nforced
         jo(c,j)=jo(c,j)-ow(j)*r1(c) ! for relative to absolute gradients `correction`
        enddo ! c
       enddo ! j
       deallocate(ugrad)
       nullify(ug)
!
! write(ME_STRNG + 100 , *) 'jo 2',jo
! gather entries from all processors
!
       if (qgrp) then
        if (ME_LOCAL.eq.0) then
         call MPI_GATHERV(MPI_IN_PLACE, &
     & orient_count(ME_LOCAL+1),mpifloat, &
     & jo, orient_count, orient_displ, mpifloat, &
     & 0, MPI_COMM_LOCAL, i)
        else
         call MPI_GATHERV(jo(1,obeg), &
     & orient_count(ME_LOCAL+1),mpifloat, &
     & MPI_UNDEFINED, orient_count, orient_displ, mpifloat,&
     & 0, MPI_COMM_LOCAL, i)
        endif
        deallocate(orient_count, orient_displ)
       endif ! qgrp
!
      endif ! qorient
!
! compute norm, but combine orientation and forced contributions for identical atoms (over nboth loop)
! orientation indices
      Jacobian(1)=zero
      do j=1,norient ; do i=1,3
       Jacobian(1) = Jacobian(1) + jo(i,j)*jo(i,j)
      enddo ; enddo
! forced indices
      do j=1,nforced ; do i=1,3
       Jacobian(1) = Jacobian(1) + jf(i,j)*jf(i,j)
      enddo ; enddo
      do j=1,nboth
       indf=iatom_both(1,j) ! forcing index
       indo=iatom_both(2,j) ! orientation index
       do i=1,3
        Jacobian(1) = Jacobian(1) + 2 * jo(i,indo) * jf(i,indf)
       enddo
      enddo
! tangent normalization :
      if (rho.le.tol) then ; rho=one ; else ; rho=one/rho ; endif
      Jacobian(1)=sqrt(Jacobian(1)*rho)
!
! write(ME_STRNG + 100 , *) 'jo ',jo
! write(ME_STRNG + 100 , *) 'jac ', Jacobian(1)
! write(ME_STRNG + 100 , *) 'nboth ', nboth
! write(ME_STRNG + 100 , *) 'norient ', norient
! compute vector norm
!
      end subroutine ftsm_calc_J
!================================================================
#endif /* automatically protect all code */
      end module ftsm_compute
