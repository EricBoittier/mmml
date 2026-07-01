! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! finite-temperature string method calculations
! version 2, in which a different, computationally simpler method is employed
! for projection calculations
!
! curvature version (compile time)
!#define FTSM_CURV_V2
!
!
      module ftsmv2_compute
!
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
      use chm_kinds
      use ftsm_var
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3 ! , only : RMSBestFit, rmsd
      implicit none
!
      private
!=====================================================================
! SUBROUTINES
      public ftsmv2_calc
!
      contains
!======================================================================
! compute RC value and derivatives
      subroutine ftsmv2_calc(x,y,z,deriv,t)
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
      real(chm_real), dimension(3,3) :: u, um, up & ! rotation matrices
     & , du &
     & ;
!
      real(chm_real), pointer :: r_com(:)
      real(chm_real) :: rho, rho1, rho2, r1(3), r2(3), r3(3), d, d1, d2
      real(chm_real) :: omdpar
      real(chm_real), dimension(:,:,:,:), pointer :: ugrad
      real(chm_real), pointer, dimension(:,:) :: roi,rol,ror,roc, &
     & rol_old,roc_old,ror_old,rol_cur,roc_cur,ror_cur
      real(chm_real), pointer, dimension(:,:) :: rfi,rfl,rfr,rfc,rfi_rot,&
     & rfl_old,rfc_old,rfr_old,rfl_cur,rfc_cur,rfr_cur, rfl_rot, rfr_rot, &

     & drf



      real(chm_real), pointer, dimension(:,:) :: fopar, foprp, ffpar, ffprp
      real(chm_real), pointer, dimension(:,:) :: ropar, roprp, rfpar, rfprp
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
      if (.not.proj_on) then
       ! warn -- invalid
       return
      endif
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
      ropar=>r_o(:,:,vpar); roprp=>r_o(:,:,vperp)
!--------------------------------------------------------------------------------------
      fw=>forcedWeights
      rfi=>r_f(:,:,instant);
      rfi_rot=>r_f(:,:,dummy);
      rfl=>r_f(:,:,left); rfr=>r_f(:,:,right); rfc=>r_f(:,:,center);
      rfl_rot=>r_f(:,:,left_rot); rfr_rot=>r_f(:,:,right_rot); ! used as swap arrays
      ffpar=>r_f(:,:,fpar); ffprp=>r_f(:,:,fperp)
      rfpar=>r_f(:,:,vpar); rfprp=>r_f(:,:,vperp)

      drf=>r_f(:,:,scratch); ! for curvature computation




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
       if (qdiffrot) then ! also use orientation atoms (otherwise, they are the same as forcing)
         roi(:,1)=roi(:,1)-r_com(1)
         roi(:,2)=roi(:,2)-r_com(2)
         roi(:,3)=roi(:,3)-r_com(3)
       endif ! qdiffrot
!
      else ! not qorient -- define u here
       u = Id3 ! rotation matrix
      endif ! qorient
!
      if (qcombine) then ! use a combination of old and new reference structures for mild restart
! combine rotated structures
       oms=one-s
!
       rfl_old=>r_f(:,:,left_old); rfr_old=>r_f(:,:,right_old); rfc_old=>r_f(:,:,center_old);
       rfl_cur=>r_f(:,:,left_cur); rfr_cur=>r_f(:,:,right_cur); rfc_cur=>r_f(:,:,center_cur);
!
       rfl_cur = s * rfl + oms * rfl_old ! left
       rfc_cur = s * rfc + oms * rfc_old ! center
       rfr_cur = s * rfr + oms * rfr_old ! right
!
       rol_old=>r_o(:,:,left_old); ror_old=>r_o(:,:,right_old); roc_old=>r_o(:,:,center_old);
       rol_cur=>r_o(:,:,left_cur); ror_cur=>r_o(:,:,right_cur); roc_cur=>r_o(:,:,center_cur);
!
       if (qorient.and.qdiffrot) then
        rol_cur = s * rol + oms * rol_old ! old left
        roc_cur = s * roc + oms * roc_old ! old center
        ror_cur = s * ror + oms * ror_old ! old right
       endif
! point to combined reference structures
       rfl=>rfl_cur
       rfc=>rfc_cur
       rfr=>rfr_cur
       rol=>rol_cur
       roc=>roc_cur
       ror=>ror_cur
      endif ! qcombine
!
      if (qorient) then
! compute rotation matrices (and gradients, if needed)
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
         allocate(ugrad(3,3,3,norient))
         call RMSBestFit(roc,roi,ow,u,obeg,oend,ugrad) ! u orients roc onto roi ; to orient roi=>roc use (u^T x roi^T)^T = roi x u ;
       else ! not deriv
         call RMSBestFit(roc,roi,ow,u)
       endif ! deriv
! compute adjacent rotation matrices for curvature
       if (fe_curvature) then
         call RMSBestFit(rol,roi,ow,um)
         call RMSBestFit(ror,roi,ow,up)
         du=up-um
         drf=fourth*matmul(rfi,du) ! note : x 0.25 because we normalize by dpar, rather than ds (=dpar/2) in the loop below
       else ! curvature
         drf=zero
       endif
       rfi_rot=matmul(rfi,u)
      else ! not qorient
       rfi_rot=>rfi
       drf=zero
      endif ! qorient
!
! compute projection (delta)
      rho=0d0
      rho1=0d0
      rho2=0d0
      fe_curv=0d0
!
      if (deriv) then
        r1=0d0;
        do k=1,3 ; do j=1,nforced;
!
                     d = rfi_rot(j,k) - rfl(j,k)
                     d1= rfr(j,k) - rfl(j,k)
                     d2= rfc(j,k) - rfl(j,k) ! for curvature
!
                     rfpar(j,k) = d ! displacement to first reference
                     rfprp(j,k) = d-d1 ! displacement to second reference : rfi_rot - rfr
                     fe_curv = fe_curv + fw(j) * ( ( d1-two*d2 ) * (d-d2 ) & ! curvature correction (see equation 21, l4 in ftsm paper)
& + d1 * drf(j,k) )
!
                     rho1=rho1 + d1*d1 * fw(j) ! denominator in dpar
                     d1 = d1 * fw(j)
                     rho =rho + d * d1 ! numerator
! COM contribution to gradient for orientation atoms:
                     r1(k)=r1(k)+d1 ! used if qdiffrot=T
! derivative components from the forcing atoms:
                     ffpar(j,k)=d1 ! to be rotated into the reference frame of roi
        enddo ; enddo
        if (rho1.le.tol) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a rather unlikely event
        dpar=rho*rho1 ! length of projection along line connecting left and right structures
        rho=sqrt(rho1)! for normalization of perpendicular derivatives (p.24 in notes)
        omdpar=1d0-dpar
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
        dperp=sqrt(rho2) ! unnormalized perp component
        if (dperp.le.tol) then; rho2=1d0; else; rho2=1d0/dperp; endif ! length of perpendicular component
!
! scale curvature : for internal pts will need x 4 because rho1 is (2)^2 times the string length increment ds
! for endpoints no scaling is needed, but the curvature correction is (supposed to be) ignored for them
        fe_curv = fe_curv * rho1 * four
! distance scaling
        if (ftsm_scaledist) then
         dperp = dperp * rho ! normalize perp component
        else
         dpar = dpar / rho ! multiply by || q - p || to unscale
         if (ftsm_reset_dpar0f) then
          if (mestring.eq.nstring-1) then ; dpar0f=one/rho ; elseif (mestring .gt. 0) then ; dpar0f=half/rho ; endif ; dpar0i=dpar0
          ftsm_reset_dpar0f=.false.
         endif
        endif
!
! for unscaled version, replace 1/||q - p||^2 by 1/||q-p||, and the latter by 1
        if (.not.ftsm_scaledist) then ; rho1=rho ; rho=one ; endif ! divide by rho to unscale
!
        if (qorient) then ! contributes only if orientation is on
! contribution from quadratics
! (not sure how to compute this more efficiently while preserving clarity)
         do k=obeg, oend
          ropar(k,:)=zero ; roprp(k,:)=zero ! cannot write to fopar directly if qdiffrot b/c then fopar:=ffpar
!
          do j=1, nforced
            do p=1,3
             d1=ffpar(j,p) ! these are forces, but they contain the displacements we need here
             d2=ffprp(j,p)
             do q=1,3
!
              r3=rfi(j,q) * ugrad(q,p,:,k) ! note the transposition in ugrad: p <==> q consistent with u vs. u^T
              ropar(k,:)=ropar(k,:) + d1*r3 ! gradient w.r.t. orientation atoms
              roprp(k,:)=roprp(k,:) + d2*r3
!
             enddo ! q
            enddo ! p
          enddo ! j (forcing atoms)
         enddo ! k (orientation atoms)
! set orientation atom derivatives
         rfr_rot=0d0; rfl_rot=0d0; ! used as swap arrays below
         if (qdiffrot) then
! now `rotate` forcing atom derivatives:
          do k=1,3; do j=1,3
           rfl_rot(:,j)=rfl_rot(:,j)+ffpar(:,k)*u (j,k)
           rfr_rot(:,j)=rfr_rot(:,j)+ffprp(:,k)*u (j,k)
          enddo; enddo;
          ffpar=rfl_rot
          ffprp=rfr_rot
!
          r3=matmul(u,r1) ; r1=r3
          r3=matmul(u,r2) ; r2=r3
          d1=rho*rho2
          do k=obeg, oend
           fopar(k,:) = rho1 * ( ropar(k,:)-r1*ow(k) ) ! scale here for qdiffrot
           foprp(k,:) = d1 * ( roprp(k,:)-r2*ow(k) )
          enddo
         else ! fopar/foprp point to ffpar/ffprp
! now `rotate` forcing atom derivatives:
          do k=1,3; do j=1,3
           rfl_rot(obeg:oend,j)=rfl_rot(obeg:oend,j)+ffpar(obeg:oend,k)*u (j,k)
           rfr_rot(obeg:oend,j)=rfr_rot(obeg:oend,j)+ffprp(obeg:oend,k)*u (j,k)
          enddo; enddo;
          do k=obeg, oend
           ffpar(k,:) = rfl_rot(k,:)
           ffprp(k,:) = rfr_rot(k,:)
           fopar(k,:) = fopar(k,:) + ropar(k,:) ! fopar and ffpar point to the same thing, but I prefer this notation
           foprp(k,:) = foprp(k,:) + roprp(k,:)
          enddo
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
         deallocate(ugrad)
         if (qgrp) deallocate(orient_count, orient_displ)
!
        endif ! qorient
! scale forcing atoms derivatives
        ffpar=rho1*ffpar
        ffprp=rho*rho2*ffprp
!
!----------------------------------------------------------------------------------------------
      else ! (deriv=F : derivative calculation not requested)
        do k=1,3 ; do j=1,nforced
                     d = rfi_rot(j,k) - rfl(j,k)
                     d1= rfr(j,k) - rfl(j,k)
                     d2= rfc(j,k) - rfl(j,k) ! for curvature
!
                     rfpar(j,k) = d ! displacement to first reference
                     rfprp(j,k) = d-d1 ! displacement to second reference : rfi_rot - rfr
                     fe_curv = fe_curv + fw(j) * ( ( d1-two*d2 ) * (d-d2 ) & ! curvature correction (see equation 21, l4 in ftsm paper)
& + d1 * drf(j,k) )
!
                     rho =rho + d *d1 *fw(j) ! numerator
                     rho1=rho1 + d1*d1 *fw(j) ! denominator
        enddo ; enddo
        if (rho1.le.tol) then ; rho1=1d0 ; else ; rho1=1d0/rho1 ; endif ! a very unlikely event
        dpar=rho*rho1 ! length projection along line connecting left and right structures
        omdpar=1d0-dpar
! perpendicular component
        do k=1,3 ; do j=1,nforced
                     d1 = dpar * rfprp(j,k) + omdpar * rfpar(j,k)
                     rfprp(j,k)=d1 ! true perpendicular component
                     rho2=rho2 + d1*d1 *fw(j) ! squared length of perpendicular vector
        enddo ; enddo
        dperp=sqrt(rho2*rho1) ! normalized perpendicular length
!
! scale curvature : for internal pts will need x 4 because rho1 is (2)^2 times the string length increment ds
! for endpoints no scaling is needed, but the curvature correction is (supposed to be) ignored for them
        fe_curv = fe_curv * rho1 * four
!
        if (.not.ftsm_scaledist) then ! scale to dimensional coordinates
         rho=one/sqrt(rho1)
         dpar = dpar * rho ! dimensionalize
         dperp = dperp * rho ! dimensionalize
         if (ftsm_reset_dpar0f) then
          if (mestring.eq.nstring-1) then ; dpar0f=one/rho ; elseif (mestring .gt. 0) then ; dpar0f=half/rho ; endif ; dpar0i=dpar0
          ftsm_reset_dpar0f=.false.
         endif
        endif
!
      endif ! deriv
!
      end subroutine ftsmv2_calc
!
!===========================================================================
#endif /* automatically protect all code */
end module ftsmv2_compute
! finite-temperature string method calculations
!
