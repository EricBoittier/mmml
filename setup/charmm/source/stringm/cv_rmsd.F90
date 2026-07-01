! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_RMSD.MOD
!
! ROUTINES FOR COLLECTIVE VARIABLE `RMSD` : rms distance between atom groups after best fit
      module cv_rmsd
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_PARALLEL==1)
!
      use cv_common
      use ivector
      use ivector_list
!
      implicit none
      private
      ! subroutines
      public cv_rmsd_add
      public cv_rmsd_calc
      public cv_rmsd_list
!
      contains
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       function cv_rmsd_add(ind_o,ind_f,r_o,r_f,ow,fw,k,gamma,weight) ! note: i is the atom index in the PSF
       use cv_types
       use stream
!
       real(chm_real) :: k, gamma, weight
       integer, pointer :: ind_o(:), ind_f(:)
       real(chm_real), pointer :: r_o(:,:), r_f(:,:), ow(:), fw(:)
! locals
       integer :: i, j, l, m, ind, num_int, norient, nforced, num_real, &
     & qdiffrot ! qdiffrot indicates that derivatives of rotation matrix need not be computed (oset=fset; ow=fw)
       logical :: found, cv_rmsd_add
       type (int_vector) :: unique_amap_ptr ! contains a unique list of atom map pointers (to speed up gradient computation, etc)
!
       character(len=len("CV_RMSD_ADD>") ),parameter::whoami="CV_RMSD_ADD>";!macro
!
       if (.not.(associated(r_o).and. &
     & associated(r_f).and. &
     & associated(ind_o).and. &
     & associated(ind_f).and. &
     & associated(ow).and. &
     & associated(fw))) then
        call wrndie(0,whoami,trim(' UNALLOCATED POINTER ARRAY. ABORT.'))
        cv_rmsd_add=.false.
        return
       endif
       norient=size(ind_o)
       nforced=size(ind_f)
! check whether the orientation/forcing sets identical
! note: in principle, the sets could be ordered differently, but be identical since order is immaterial
! however, the parent routine (smcv_add) produces a unique ordering, so we do not check for this
       qdiffrot=1
       if (norient.eq.nforced) then
        if (all(ind_o.eq.ind_f).and.all(ow.eq.fw)) qdiffrot=0
       endif
! check for duplicate CV (exact identical entry only)
       found=.false.
       do l=1, cv%num_cv
        if (cv%type(l).eq.rmsd) then
         found=(norient.eq.cv%priv(l)%p(1).and. &
     & nforced.eq.cv%priv(l)%p(2).and. &
     & qdiffrot.eq.cv%priv(l)%p(3))
         ind=4
         do j=1,norient
           if (found) found= &
     & (ind_o(j).eq.cv%priv(l)%p(ind))
           ind=ind+1
         enddo
!
         do j=1,nforced
           if (found) found= &
     & (ind_f(j).eq.cv%priv(l)%p(ind))
           ind=ind+1
         enddo
!
        endif
        if (found) exit
       enddo
!
       if (.not.found) then ! (if found -- do nothing)
        l=cv_common_add(k,gamma,weight,rmsd) ! get a new cv index
        if (l.gt.0) then
! allocate private data
! space needed:
         num_int = 3 + norient + nforced ! number of ints needed for storage
!
         allocate(cv%priv(l)%p(num_int));
         cv%priv(l)%p(1)=norient
         cv%priv(l)%p(2)=nforced
         cv%priv(l)%p(3)=qdiffrot
! now add slave CV indices
         ind=4
         do j=1,norient
           m=ind_o(j)
           if (m.le.0) then
             call wrndie(0,whoami,trim(' NONPOSITIVE ATOM INDEX.'))
           endif
           cv%priv(l)%p(ind)=int_vlist_uaddu(cv%amap,m,l) ! add indices into unique map
           m=int_vector_uadd(unique_amap_ptr,cv%priv(l)%p(ind))
           ind=ind+1
         enddo
!
         do j=1,nforced
           m=ind_f(j)
           if (m.le.0) then
             call wrndie(0,whoami,trim(' NONPOSITIVE ATOM INDEX.'))
           endif
           cv%priv(l)%p(ind)=int_vlist_uaddu(cv%amap,m,l) ! add indices into unique map
           m=int_vector_uadd(unique_amap_ptr,cv%priv(l)%p(ind))
           ind=ind+1
         enddo
! populate private real array
         num_real = 4 * ( norient + nforced ) ! number of reals needed for storage
         allocate(cv%priv(l)%pr(num_real));
         ind=1
         do j=1,3
          do i=1, norient
           cv%priv(l)%pr(ind)=r_o(i,j)
           ind=ind+1
          enddo
         enddo
!
         do j=1,3
          do i=1, nforced
           cv%priv(l)%pr(ind)=r_f(i,j)
           ind=ind+1
          enddo
         enddo
!
          do i=1, norient
           cv%priv(l)%pr(ind)=ow(i)
           ind=ind+1
          enddo
!
          do i=1, nforced
           cv%priv(l)%pr(ind)=fw(i)
           ind=ind+1
          enddo
!
         cv_rmsd_add=.true.
!
         m=unique_amap_ptr%last ! number of unique atoms this cv depends on
         allocate(cv%priv(l)%amap_ptr(m+1)) ! add one to include length of list
         cv%priv(l)%amap_ptr(1)=m;
         cv%priv(l)%amap_ptr(2:m+1)=unique_amap_ptr%i(1:m)
         call int_vector_done(unique_amap_ptr)
! write(0,*) whoami, size(cv%priv(l)%amap_ptr(:)) !aa
!
        else ! out of bounds
         call wrndie(0,whoami,trim(' ERROR ADDING RMSD CV. NOTHING DONE.'))
         cv_rmsd_add=.false.
        endif
       else ! found
         call wrndie(0,whoami,trim(' RMSD CV ALREADY PRESENT. NOTHING DONE.'))
         cv_rmsd_add=.false.
       endif
!
       end function cv_rmsd_add
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_rmsd_calc(i,x,y,z,mass,fx,fy,fz, &
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
       real(chm_real), parameter :: tol=1.0e-10
       logical :: qdiffrot
       integer :: j, k, ii, jj, ind, norient, nforced, obeg, oend, p, q
       real(chm_real) :: d, f, w, r1(3), r2(3), r_com(3), theta, u(3,3)
       integer, pointer :: priv(:)
       integer, pointer :: ind_o(:), ind_f(:)
       real(chm_real), pointer :: r_o(:,:), rt_o(:,:), ow(:), &
     & r_f(:,:), rt_f(:,:), fw(:), rt_rot_f(:,:)
       real(chm_real), dimension(:,:,:,:), pointer :: ugrad
!
       norient=cv%priv(i)%p(1)
       nforced=cv%priv(i)%p(2)
       qdiffrot=(cv%priv(i)%p(3).ne.0)
!
! write(0,*) norient, nforced, qdiffrot, deriv !aa
!
       if (qdiffrot) then ! extract all indices and coordinates
        allocate(ind_o(norient)) ! indices into atom map
        allocate(rt_o(norient,3), r_o(norient,3))
        allocate(ow(norient))
        allocate(ind_f(nforced))
        allocate(rt_f(nforced,3), r_f(nforced,3), rt_rot_f(nforced,3))
        allocate(fw(nforced))
! indices
        ii=4; jj=ii+norient-1; ind_o=cv%priv(i)%p(ii:jj)
        ii=jj+1; jj=ii+nforced-1; ind_f=cv%priv(i)%p(ii:jj)
! o. coordinates
        jj=0;
        do j=1,3
         ii=jj+1; jj=ii+norient-1; rt_o(:,j)=cv%priv(i)%pr(ii:jj)
        enddo
! f. coordinates
        do j=1,3
         ii=jj+1; jj=ii+nforced-1; rt_f(:,j)=cv%priv(i)%pr(ii:jj)
        enddo
! o. weights
        ii=jj+1; jj=ii+norient-1; ow(:)=cv%priv(i)%pr(ii:jj)
! f. weights
        ii=jj+1; jj=ii+nforced-1; fw(:)=cv%priv(i)%pr(ii:jj)
       else ! will not differentiate rotations
        allocate(ind_o(norient))
        allocate(rt_o(norient,3), r_o(norient,3))
        allocate(ow(norient))
! indices
        ii=4; jj=ii+norient-1; ind_o=cv%priv(i)%p(ii:jj)
! o. coordinates
        jj=0;
        do j=1,3
         ii=jj+1; jj=ii+norient-1; rt_o(:,j)=cv%priv(i)%pr(ii:jj)
        enddo
! o. weights (skip forced atom coordinates)
        ii=jj+3*nforced+1; jj=ii+norient-1; ow(:)=cv%priv(i)%pr(ii:jj)
! assume orientation atoms/weights are the same as forced atoms/weights
        ind_f=>ind_o
        rt_f=>rt_o
        r_f=>r_o
        allocate(rt_rot_f(nforced,3)) ! only needed for forced set
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
        if (qdiffrot.and.deriv) then
         allocate(ugrad(3,3,3,norient))
         call RMSBestFit(rt_o,r_o,ow,u,obeg,oend,ugrad)
        else
         call RMSBestFit(rt_o,r_o,ow,u)
        endif
!
! rotate target structure to overlap with current
! v%rtarget_rot_f=matmul(v%rtarget_f, transpose(u)) ! shorthand way
! conventional way (might be faster)
        rt_rot_f=0d0;
        do k=1,3; do j=1,3
          rt_rot_f(:,j)=rt_rot_f(:,j)+rt_f(:,k)*u(j,k)
        enddo; enddo
! compute the RMSD
        theta=rmsd(r_f, rt_rot_f, fw)
!
        cv%r(i,previnst)=cv%r(i,instant)
        cv%r(i,instant)=theta
!
! write(0,*) theta , 'THETA' !aa
        if (deriv) then
! initialize derivative arrays
! write(0,*) 'getting pointer:'
         priv=>cv%priv(i)%amap_ptr ! this should copy the pointers statically; the pointers point to the _same_ data
! write(0,*) size(cv%priv(i)%amap_ptr) !aa
! write(0,*) 'all atoms for this CV:', priv(1) !aa
         do jj=2,priv(1)+1 ! only a subset of indices needs to be considered
          j=priv(jj)
!
          do ii=1,2
          cv%gradx(i,j,ii)=0d0;cv%grady(i,j,ii)=0d0;cv%gradz(i,j,ii)=0d0
          enddo
         enddo
! write(0,*) 'Init deriv.'!aa
!
         if (qdiffrot) then
! compute COM contribution to gradient for orientation atoms:
! r1 = matmul(transpose(v%rcurrent_f-v%rtarget_rot_f), ! shorthand way
! & v%forcedWeights)
! conventional way (might be faster)
          r1(:)=0d0;
          do j=1,3; do k=1, nforced;
           r1(j)=r1(j) + fw(k) * (r_f(k,j)-rt_rot_f(k,j))
          enddo ; enddo
          do j=obeg, oend
           ind=ind_o(j)
           cv%gradx(i,ind,1)=-r1(1)*ow(j)
           cv%grady(i,ind,1)=-r1(2)*ow(j)
           cv%gradz(i,ind,1)=-r1(3)*ow(j)
          enddo
!
          do j=1, nforced
           r2=fw(j)*r_f(j,:)
           do k=obeg, oend
            ind=ind_o(k)
! note: am actually computing -forces here
            do p=1,3
             do q=1,3
              d=r2(p)*rt_f(j,q)
              cv%gradx(i,ind,1)=cv%gradx(i,ind,1) - ugrad(p,q,1,k) * d
              cv%grady(i,ind,1)=cv%grady(i,ind,1) - ugrad(p,q,2,k) * d
              cv%gradz(i,ind,1)=cv%gradz(i,ind,1) - ugrad(p,q,3,k) * d
             enddo ! q
            enddo ! p
           enddo ! k (orientation atoms)
! also add forces on the forcing atoms:
           ind=ind_f(j)
           cv%gradx(i,ind,1)=cv%gradx(i,ind,1) + &
     & (r_f(j,1)-rt_rot_f(j,1))*fw(j)
           cv%grady(i,ind,1)=cv%grady(i,ind,1) + &
     & (r_f(j,2)-rt_rot_f(j,2))*fw(j)
           cv%gradz(i,ind,1)=cv%gradz(i,ind,1) + &
     & (r_f(j,3)-rt_rot_f(j,3))*fw(j)
          enddo ! j (forcing atoms)
!
         else ! not qdiffrot
! apply forces to the forcing atoms
          do j=1,nforced
           ind=ind_f(j)
! write(0,*) 'map index:', j, ind
           cv%gradx(i,ind,1)=(r_f(j,1)-rt_rot_f(j,1))*fw(j)
           cv%grady(i,ind,1)=(r_f(j,2)-rt_rot_f(j,2))*fw(j)
           cv%gradz(i,ind,1)=(r_f(j,3)-rt_rot_f(j,3))*fw(j)
          enddo
         endif ! qdiffrot
!
! scale derivatives by 1/theta and populate mass-weighted der. arrays
         if (theta.gt.tol) then
          d=1d0/theta
         else
          d=1d0 ! avoid singularity at near zero separation
         endif
!
! write(0,*) d !aa
         priv=>cv%priv(i)%amap_ptr
         do ii=2, priv(1)+1
          jj=priv(ii)
          j=cv%amap%i(jj) ! psf index
          w=sqrt(1d0/mass(j))
          cv%gradx(i,jj,1)=cv%gradx(i,jj,1)*d
          cv%gradx(i,jj,2)=cv%gradx(i,jj,1)*w
!
          cv%grady(i,jj,1)=cv%grady(i,jj,1)*d
          cv%grady(i,jj,2)=cv%grady(i,jj,1)*w
!
          cv%gradz(i,jj,1)=cv%gradz(i,jj,1)*d
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
       deallocate(ind_o,r_o,rt_o,ow,rt_rot_f)
       if (qdiffrot) then
        deallocate(ind_f,r_f,rt_f,fw)
        if (deriv) deallocate(ugrad)
       endif
!
       end subroutine cv_rmsd_calc
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       subroutine cv_rmsd_list(i)
       use cv_types
       use multicom_aux;
       use mpi
       use chutil, only : atomid
       use stream
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
       integer :: i, j, type, ii, jj, iatom
       character(len=8) :: sid, rid, ren, ac
!
       integer :: norient, nforced
       logical :: qdiffrot
       integer, pointer, dimension(:) :: ind_o, ind_f
       character(len=len("CV_RMSD_LIST>") ),parameter::whoami="CV_RMSD_LIST>";!macro
!
       if (MPI_COMM_STRNG.eq.MPI_COMM_NULL) return ! only replica heads stay
!
! check type just in case
       type=cv%type(i)
       if (type.ne.rmsd) then
        call wrndie(0,whoami,trim(' WRONG CV TYPE RECEIVED.'))
       endif
!
       if (ME_STRNG.eq.0) then
        norient=cv%priv(i)%p(1)
        nforced=cv%priv(i)%p(2)
        qdiffrot=(cv%priv(i)%p(3).ne.0)
!
! note that I currently cannot tell whether mass-weighting is requested
! am also not printing the weights
!
        allocate(ind_o(norient))
        ii=4; jj=ii+norient-1; ind_o=cv%priv(i)%p(ii:jj)
        write(info,'(A)') '\t RMSD FROM REFERENCE, ORIENTATION ATOMS' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        do j=1, norient;
         iatom=cv%amap%i(ind_o(j))
         call atomid(iatom, sid, rid, ren, ac)
         write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        enddo
        deallocate(ind_o)
!
        if (qdiffrot) then
         allocate(ind_f(nforced))
         ii=jj+1; jj=ii+nforced-1; ind_f=cv%priv(i)%p(ii:jj)
         write(info,'(A)') '\t RMSD FROM REFERENCE, FORCED ATOMS' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         do j=1, nforced;
          iatom=cv%amap%i(ind_f(j)) ! actual psf index
          call atomid(iatom, sid, rid, ren, ac)
          write(info,667) '\t',j, iatom, sid, rid, ren, ac ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
         enddo
         deallocate(ind_f)
        else
         write(info,'(A)')'\t FORCED AND ORIENTATION ATOMS ARE THE SAME' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
        endif
       endif ! ME_STRING
!
 667 format(A,2I8,' ',4A)
!
       end subroutine cv_rmsd_list
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#endif
#endif /* automatically protect all code */
      end module cv_rmsd
!
