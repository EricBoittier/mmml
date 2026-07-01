! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! utility function module for finite-temperature string module
      module ftsm_util
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use number
      use ftsm_var
      implicit none
!
      private
      character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
!=====================================================================
! SUBROUTINES
      public ftsm_check
      public ftsm_copy
      public ftsm_swap
      public ftsm_fill
      public ftsm_lift
      public ftsm_save_com
      public ftsm_swap_bc
      public ftsm_define_rtmd_type
      public ftsm_update_overlap_coor
      public ftsm_qdiffrot
      public ftsm_compute_overlap_ind
      public ftsm_set_weights
      public ftsm_coor_wgt_alloc ! relevant for COM version : allocate coordinate and weight arrays after FTSM groups have been defined
      public ftsm_test_grad_fd
      public ftsm_test_parallel
!
!==================================================================================
      contains
!==================================================================================
      function ftsm_check(qorie) result (ok)
      use string
      integer :: error
      logical :: ok, qorie
!
      character(len=len("FTSM_CHECK>") ),parameter::whoami="FTSM_CHECK>";!macro
      error=0
!
      do
       if (.not.ftsm_initialized) then
        call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. ABORT.'))
        error=1
        exit
       endif
!
       if ( qorie ) then
        if ( norient.eq.0 .or. .not.associated(r_o) .or. .not.associated(orientWeights) ) then
         error=2
        elseif (.not.ftsm_com_on .and. .not.associated(iatom_o)) then
         error=2
        elseif (ftsm_com_on) then
         if (.not.associated(iatoms_o)) then
          error=2
         elseif (iatoms_o%last.eq.0) then
          error=2
         endif
        endif
        if (error.eq.2) then
         call wrndie(0,whoami,trim('NO ORIENTATION ATOMS FOUND. ABORT.'))
         exit
        endif
       endif ! qorie
!
       if ( nforced.eq.0 .or. .not.associated(r_f) .or. .not.associated(forcedWeights) ) then
        error=3
       elseif (.not.ftsm_com_on .and. .not.associated(iatom_f)) then
        error=3
       elseif (ftsm_com_on) then
        if (.not.associated(iatoms_f)) then
         error=3
        elseif (iatoms_f%last.eq.0) then
         error=3
        endif
       endif ! nforced
       if (error.eq.3) then
        call wrndie(0,whoami,trim('NO FORCING ATOMS FOUND. ABORT.'))
        exit
       endif
! no error
       exit
      enddo
!
      ok=error.eq.0
!
      end function ftsm_check
!=================================================================================
      subroutine ftsm_swap(c1,c2)
! use stream
      character(len=len("FTSM_SWAP>") ),parameter::whoami="FTSM_SWAP>";!macro
      integer :: c1, c2
      real(chm_real) :: com(3)
!
      if (.not.ftsm_check(qorient)) return
!
      if (c1.gt.num_sets.or.c2.gt.num_sets.or.c1.le.0.or.c2.le.0) then
       call wrndie(0,whoami,trim('INVALID COLUMN SPECIFIED. ABORT.'))
       return
      else
       if (qorient) then
        if (qdiffrot) then
         r_o(:,:,dummy)=r_o(:,:,c1)
         r_o(:,:,c1)=r_o(:,:,c2)
         r_o(:,:,c2)=r_o(:,:,dummy)
        endif
! swap COMs
        com =rcom(:,c1)
        rcom(:,c1)=rcom(:,c2)
        rcom(:,c2)=com
       endif
! forcing atoms
       r_f(:,:,dummy)=r_f(:,:,c1)
       r_f(:,:,c1) =r_f(:,:,c2)
       r_f(:,:,c2) =r_f(:,:,dummy)
!
      endif
!
      end subroutine ftsm_swap
!=================================================================================
      subroutine ftsm_copy(c1,c2)
      use number
      character(len=len("FTSM_COPY>") ),parameter::whoami="FTSM_COPY>";!macro
      integer :: c1, c2
!
      if (.not.ftsm_check(qorient)) return
!
      if (c1.gt.num_sets.or.c2.gt.num_sets.or.c1.le.0.or.c2.le.0) then
       call wrndie(0,whoami,trim('INVALID COLUMN SPECIFIED. ABORT.'))
       return
      else
       if (qorient) then
        if (qdiffrot) r_o(:,:,c2)=r_o(:,:,c1)
        rcom(:,c2) =rcom(:,c1)
       endif ! qorient
       r_f(:,:,c2)=r_f(:,:,c1)
      endif
!
      end subroutine ftsm_copy
!
!=================================================================================
      subroutine ftsm_fill(x,y,z,col)
      use number
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      character(len=len("FTSM_FILL>") ),parameter::whoami="FTSM_FILL>";!macro
!
      real(chm_real) :: x(:), y(:), z(:)
      integer, optional :: col
!
      real(chm_real), pointer :: wgts(:), rf(:,:), ro(:,:)
      real(chm_real) :: d
      integer, pointer :: inds(:)
!
      integer :: i,k,ind,c
      if (.not.ftsm_check(qorient)) return
!
      if (any(x.eq.anum).or.any(y.eq.anum).or.any(z.eq.anum)) then
       call wrndie(0,whoami,trim('COORDINATE SET HAS UNDEFINED VALUES. NOTHING DONE.'))
       return
      else
!
      if (present(col)) then
       if (col .gt. zero .and. col .le. num_sets) then
        c=col
       else
        call wrndie(0,whoami,trim('INVALID COLUMN SPECIFIED. ABORT.'))
        return
       endif
      else
       c=center
      endif
!
       rf=>r_f(:,:,c)
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
    rf(k,:)=zero
    inds=>iatoms_f%v(k)%i
    wgts=>wgts_f%v(k)%r
    do i=1, iatoms_f%v(k)%last
     ind=inds(i)
     d=wgts(i)
     if (ind>size(x).or.ind>size(y).or.ind>size(z)) then;call wrndie(0,whoami,trim('COORDINATE ARRAY BOUNDS EXCEEDED. ABORT.'));return;endif
     rf(k,1)=rf(k,1)+(d*x(ind));
     rf(k,2)=rf(k,2)+(d*y(ind));
     rf(k,3)=rf(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, nforced
    ind=iatom_f(k)
    if (ind>size(x).or.ind>size(y).or.ind>size(z)) then;call wrndie(0,whoami,trim('COORDINATE ARRAY BOUNDS EXCEEDED. ABORT.'));return;endif
    rf(k,1)=x(ind)
    rf(k,2)=y(ind)
    rf(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
!
       if (qorient.and.qdiffrot) then ! technically, qorient not needed because of norient below
        ro=>r_o(:,:,c)
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
    ro(k,:)=zero
    inds=>iatoms_o%v(k)%i
    wgts=>wgts_o%v(k)%r
    do i=1, iatoms_o%v(k)%last
     ind=inds(i)
     d=wgts(i)
     if (ind>size(x).or.ind>size(y).or.ind>size(z)) then;call wrndie(0,whoami,trim('COORDINATE ARRAY BOUNDS EXCEEDED. ABORT.'));return;endif
     ro(k,1)=ro(k,1)+(d*x(ind));
     ro(k,2)=ro(k,2)+(d*y(ind));
     ro(k,3)=ro(k,3)+(d*z(ind));
    enddo ! i
   enddo ! k
  else
   do k=1, norient
    ind=iatom_o(k)
    if (ind>size(x).or.ind>size(y).or.ind>size(z)) then;call wrndie(0,whoami,trim('COORDINATE ARRAY BOUNDS EXCEEDED. ABORT.'));return;endif
    ro(k,1)=x(ind)
    ro(k,2)=y(ind)
    ro(k,3)=z(ind)
   enddo
  endif
!===========================
! ^
! | macro ftsm_load_fcor.def
       endif ! qdiffrot
      endif ! x.eq.anum
! remove and save center of mass of orientation atoms
      call ftsm_save_com(c)
!
      if (c.eq.center) then
       call ftsm_swap_bc(.true.) ! .true. : send to slaves
!
       r_f(:,:,left_old:right_old)=r_f(:,:,left:right)
       r_f(:,:,center_new)=r_f(:,:,center)
       if (qorient.and.qdiffrot) then
        r_o(:,:,left_old:right_old)=r_o(:,:,left:right)
        r_o(:,:,center_new)=r_o(:,:,center)
       endif
      endif ! c
!
      end subroutine ftsm_fill
!=================================================================================
      subroutine ftsm_swap_bc(qsendo)
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use number
      use parallel, only: psnd4, psnd8
      logical :: qroot, qslave, qsend_o
      logical, optional :: qsendo
      real(chm_real), pointer, dimension(:,:) :: rlf, rcf, rrf, rlo, rco, rro
      integer :: me, ierror
      integer*4 :: stat(MPI_STATUS_SIZE)
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qslave=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
      if (present(qsendo)) then
        qsend_o=qsendo.and.qorient.and.qdiffrot
      else
        qsend_o=qorient.and.qdiffrot
      endif
!
! note that I can avoid sending the orientation coordinates, since they only evolve through the forced atoms
!
      me=mestring+1 ! slaves need me too
!
      if (qroot.and.nstring.gt.1) then
        rcf=>r_f(:,:,center)
        rlf=>r_f(:,:,left)
        rrf=>r_f(:,:,right)
!
        if (qdiffrot) then
         rco=>r_o(:,:,center)
         rlo=>r_o(:,:,left)
         rro=>r_o(:,:,right)
        endif
! replicas send to the right and receive from the left
        if (me.eq.1) then
         call MPI_SEND(rcf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & me, 0, MPI_COMM_STRNG, ierror)
         if (qsend_o) &
     & call MPI_SEND(rco, 3*norient, MPI_DOUBLE_PRECISION, &
     & me, 1, MPI_COMM_STRNG, ierror)
!
        elseif (me.eq.nstring) then
         call MPI_RECV(rlf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & mestring-1, 0, MPI_COMM_STRNG, stat, ierror)
         if (qsend_o) &
     & call MPI_RECV(rlo, 3*norient, MPI_DOUBLE_PRECISION, &
     & mestring-1, 1, MPI_COMM_STRNG, stat, ierror)
        else ! inner replicas
         call MPI_SENDRECV(rcf, 3*nforced, MPI_DOUBLE_PRECISION, me, 0, &
     & rlf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & mestring-1, 0, MPI_COMM_STRNG, stat,ierror)
         if (qsend_o) &
     & call MPI_SENDRECV(rco, 3*norient, MPI_DOUBLE_PRECISION, me, 1,&
     & rlo, 3*norient, MPI_DOUBLE_PRECISION, &
     & mestring-1, 1, MPI_COMM_STRNG, stat,ierror)
        endif ! me.eq.1
! replicas send to the left and receive from the right
        if (me.eq.nstring) then
         call MPI_SEND(rcf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & mestring-1, 0, MPI_COMM_STRNG, ierror)
         if (qsend_o) &
     & call MPI_SEND(rco, 3*norient, MPI_DOUBLE_PRECISION, &
     & mestring-1, 1, MPI_COMM_STRNG, ierror)
!
        elseif (me.eq.1) then
         call MPI_RECV(rrf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & me, 0, MPI_COMM_STRNG, stat, ierror)
         if (qsend_o) &
     & call MPI_RECV(rro, 3*norient, MPI_DOUBLE_PRECISION, &
     & me, 1, MPI_COMM_STRNG, stat, ierror)
        else ! inner replicas
         call MPI_SENDRECV(rcf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & mestring-1, 0, &
     & rrf, 3*nforced, MPI_DOUBLE_PRECISION, &
     & me, 0, MPI_COMM_STRNG, stat,ierror)
         if (qsend_o) &
     & call MPI_SENDRECV(rco, 3*norient, MPI_DOUBLE_PRECISION, &
     & mestring-1, 1, &
     & rro, 3*norient, MPI_DOUBLE_PRECISION, &
     & me, 1, MPI_COMM_STRNG, stat,ierror)
        endif ! me.eq.nstring
      endif ! qroot & nstring > 1
!
! send to slaves
      if (qslave) then
       call PSND8m(r_f(:,:,left),9*nforced) ! send three sets at once (see ftsm_var)
       if (qsend_o) call PSND8m(r_o(:,:,left),9*norient)
      endif
!
! duplicate endpoints for force calculations:
!
      if (me.eq.1) then
       r_f(:,:,left)=r_f(:,:,center)
       if (qsend_o) then
        r_o(:,:,left)=r_o(:,:,center)
       endif
      endif
!
      if (me.eq.nstring) then
       r_f(:,:,right)=r_f(:,:,center)
       if (qsend_o) then
        r_o(:,:,right)=r_o(:,:,center)
       endif
      endif
!
! update any orientation coordinates that have changes
      if (qorient.and.qdiffrot) call ftsm_update_overlap_coor(ithree)
!
      end subroutine ftsm_swap_bc
!=================================================================================
      subroutine ftsm_save_com(c)
      use number
! real(chm_real) :: r_com(3)
      real(chm_real) :: w
      real(chm_real), pointer :: ro_com(:)
      integer, optional :: c
      integer :: col, i
      real(chm_real), pointer, dimension(:,:) :: ro, rf
!
! compute and save COM of current reference structure
!
      if (qorient) then
!
        if (present(c)) then ; col=c ; else ; col=center ; endif
        if (col.le.num_sets.and.col.gt.0) then
         ro_com=>rcom(:,col)
!
! r_com=ro_com ! save old COM
         ro_com=zero ! will recompute COM using new weights
         ro => r_o(:,:,col)
         rf => r_f(:,:,col)
! compute new COM
         do i=1, norient
          w=orientWeights(i)
          ro_com(1)=ro_com(1) + w * ro(i,1)
          ro_com(2)=ro_com(2) + w * ro(i,2)
          ro_com(3)=ro_com(3) + w * ro(i,3)
         enddo
! translate orientation structure to centroid
         ro(:,1)=ro(:,1) - ro_com(1)
         ro(:,2)=ro(:,2) - ro_com(2)
         ro(:,3)=ro(:,3) - ro_com(3)
! translate forcing structure to centroid
         if (qdiffrot) then
          rf(:,1)=rf(:,1) - ro_com(1)
          rf(:,2)=rf(:,2) - ro_com(2)
          rf(:,3)=rf(:,3) - ro_com(3)
         endif
! centroid relative to original coords:
! ro_com = ro_com + r_com ! this is no longer useful; VO 1/2013
       endif
      endif
!
      end subroutine ftsm_save_com
!========================================================================================
      subroutine ftsm_lift(x, y, z)
!
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use stream
      use number
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
!
      real(chm_real) :: x(:), y(:), z(:)
!
! local variables
      real(chm_real) :: u(3,3)= Id3
      real(chm_real), pointer :: r_com(:), ow(:)
      real(chm_real), pointer, dimension(:,:) :: roi, roc, rfc, roc_rot, rfc_rot
!
      integer :: j, k
!
      integer :: natom, ind
!
      character(len=len("FTSM_LIFT>") ),parameter::whoami="FTSM_LIFT>";!macro
!
! check if the user has made an initialization call to the minimizer
!
      if (.not.ftsm_check(qorient)) return
!
      if (ftsm_com_on) return ! not compatible with FTSM-COM (in that case, lift will require hard work)
!
! create coordinate arrays
!
      ow=>orientWeights
      r_com=>rcom(:,instant)
      roi=>r_o(:,:,instant)
      roc=>r_o(:,:,center)
      rfc=>r_f(:,:,center)
      roc_rot=>r_o(:,:,center_rot)
      rfc_rot=>r_f(:,:,center_rot)
      natom=size(x,1)
! copy string coordinates to corresponding instantaneous coordinates
! first, align string coordinates with the instantaneous coordinates, if needed
!
      if (qorient) then
        do k=1,norient
          ind=iatom_o(k)
          roi(k,1)=x(ind)
          roi(k,2)=y(ind)
          roi(k,3)=z(ind)
        enddo
!
! translate forced atoms to centroid
!
        r_com(:)=0d0;
        do j=1,3 ; do k=1, norient;
          r_com(j) = r_com(j)+ow(k)*roi(k,j)
        enddo ; enddo
!
        roi(:,1)=roi(:,1)-r_com(1)
        roi(:,2)=roi(:,2)-r_com(2)
        roi(:,3)=roi(:,3)-r_com(3)
!
      endif ! qorient
!
      if (qorient) then ! orient center image w.r.t. instantaneous coordinates
       call RMSBestFit(roi,roc,ow,u) ! superpose roi onto roc (assuming string is COM-free)
!
       rfc_rot = matmul(rfc, u) ! apply transpose (=inverse) of u to rfc
       rfc_rot(:,1)=rfc_rot(:,1)+r_com(1)
       rfc_rot(:,2)=rfc_rot(:,2)+r_com(2)
       rfc_rot(:,3)=rfc_rot(:,3)+r_com(3)
!
       if (qdiffrot) then
! move to COM of the instantaneous coordinates
        roc_rot = matmul(roc, u) ! apply transpose (=inverse) of u to roc
        roc_rot(:,1)=roc_rot(:,1)+r_com(1)
        roc_rot(:,2)=roc_rot(:,2)+r_com(2)
        roc_rot(:,3)=roc_rot(:,3)+r_com(3)
! insert orientation coordinates into all-atom coordinate array
        do k=1,norient
         ind=iatom_o(k)
         x(ind)=roc_rot(k,1)
         y(ind)=roc_rot(k,2)
         z(ind)=roc_rot(k,3)
        enddo
       endif ! qdiffrot
      else ! no orientation
       rfc_rot=>rfc
      endif ! qorient
!
! insert forced coordinates into all-atom coordinate array
!
      do k=1,nforced
        ind=iatom_f(k)
        x(ind)=rfc_rot(k,1)
        y(ind)=rfc_rot(k,2)
        z(ind)=rfc_rot(k,3)
      enddo
!
      end subroutine ftsm_lift
!===========================================================================
      subroutine ftsm_define_rtmd_type()
      use sm_config, only: sizeofreal
      use mpi
      use number
!
      integer*4 :: error, norient_mpi
      integer(kind=MPI_ADDRESS_KIND) :: lb, extent
!
! create custom type for gathering array of shape (norient,3), in rows
! taken from rtmd code
!
      if (MPI_RTMD_TYPE.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_RTMD_TYPE, error)
      if (MPI_RTMD_TYPE_.ne.MPI_DATATYPE_NULL) &
     & call mpi_type_free(MPI_RTMD_TYPE_, error)
!
      if (norient.gt.0) then
       lb=0
       extent=sizeofreal
       norient_mpi=norient
       call mpi_type_vector(3, 1, norient_mpi, &
     & MPI_DOUBLE_PRECISION, MPI_RTMD_TYPE_, error)
! corresponding resized type
       call mpi_type_create_resized(MPI_RTMD_TYPE_,lb,extent, &
     & MPI_RTMD_TYPE, error)
       call mpi_type_commit(MPI_RTMD_TYPE, error)
      endif
!
      end subroutine ftsm_define_rtmd_type
!====================================================================================
      subroutine ftsm_compute_overlap_ind() ! compute overlap indices in iatom_both
      use ivector; use ivector_list; use rvector; use rvector_list
      integer :: i, j
      integer, pointer :: temp(:,:)
      type (int_vector), pointer :: vo, vf
      logical :: oflag(norient)
!
      if (associated(iatom_both)) deallocate(iatom_both)
      nboth=0
      if ( .not. ( qorient .and. qdiffrot .and. ftsm_check(qorient))) return
!
      allocate(temp(2,max(nforced, norient)))
!
      if (.not. ftsm_com_on) then ! this is a fast simple test because indices are ordered :
       i=1; j=1
       do while (i.le.nforced .and. j.le.norient)
!
        do while ( j.lt.norient .and. ( iatom_f(i) .gt. iatom_o(j) ) ) ! N.B.: indices assumed to be ordered
         j=j+1
        enddo
        if (iatom_f(i).eq.iatom_o(j)) then
         nboth=nboth+1;
         temp(1,nboth)=i; temp(2,nboth)=j;
         j=j+1; ! i-indices are unique, so safe to increment j
        endif
        i=i+1
       enddo
!
      else ! this is a slow (quadratic) test because the atom groups can be given in any order
       oflag=.false. ! tells whether an orientation index has already been matched to a forced index
       do i=1, nforced
        vf=>iatoms_f%v(i)
        do j=1, norient
         if (oflag(j)) continue ! atom group j has already been matched; either no match will be found or there is a duplicate, which is illegal
         vo=>iatoms_o%v(j)
         if ( int_vector_eq_ordered(vo,vf) ) then ! note that group indices are assumed to be ordered
          nboth=nboth+(1);
          temp(1,nboth)=i; temp(2,nboth)=j;
          oflag(j)=.true.
          exit ! j-loop
         endif
        enddo
       enddo
      endif
!
      if (nboth.gt.0) then
       allocate(iatom_both(2,nboth))
       iatom_both(:,1:nboth)=temp(:,1:nboth)
      endif
      deallocate(temp)
!
      end subroutine ftsm_compute_overlap_ind
!====================================================================================
      subroutine ftsm_qdiffrot() ! test whether the orientation and forced indices are equal
! NOTE: this routine will give qdiffrot=.true. if orient=.false. ; make sure this is what you want
! will also return qdiffrot = .true. if norient and nforced are both zero, unless index arrays are not associated
! if (not qorient) return false
      use ivector; use ivector_list; use rvector; use rvector_list
      integer :: i, j
      logical :: equal, found
      type (int_vector), pointer :: vo, vf
      logical :: oflag(norient)
!
      equal=qorient.and.(norient.eq.nforced)
!
      if (equal) then
       if (.not. ftsm_com_on) then ! this is a fast simple test because indices are ordered :
        equal=associated(iatom_f).and.associated(iatom_o)
        if (equal) equal=all(iatom_f.eq.iatom_o)
!
       else ! this is a slow (quadratic) test because the atom groups can be given in any order
        equal=associated(iatoms_o).and.associated(iatoms_f)
!
        if (equal) then
         equal=iatoms_o%last .eq. iatoms_f%last
         if (equal) then
          oflag=.false. ! whether an orientation index has already been matched to a forced index
!
          do i=1, nforced
           vf=>iatoms_f%v(i)
! find equivalent entry in orientation set
           found=.false.
           do j=1, norient
            if (oflag(j)) continue ! atom group j has already been matched; either no match will be found or there is a duplicate, which is illegal
            vo=>iatoms_o%v(j)
            if ( int_vector_eq_ordered(vo,vf) ) then ! note that group indices are assumed to be ordered
             found=.true.
             oflag(i)=.true.
             exit ! j-loop
            endif
           enddo
           equal=found
           if (.not.equal) exit
          enddo
         endif ! equal
        endif ! equal
!
       endif ! not ftsm_com_on
      endif ! equal
!
      qdiffrot=.not.equal
      ftsm_compute_qdiffrot=.false.
!
      end subroutine ftsm_qdiffrot
!=================================================================
      subroutine ftsm_update_overlap_coor(which)
      integer :: which
      integer :: i
! copy overlapping coordinates from one set to another
      if (which.eq.1) then
       do i=1, nboth
        r_o(iatom_both(2,i), :, center)=r_f(iatom_both(1,i), :, center)
       enddo
      elseif (which.eq.2) then
       do i=1, nboth
        r_f(iatom_both(1,i), :, center)=r_o(iatom_both(2,i), :, center)
       enddo
      elseif (which.eq.3) then ! special case of left:right sets all being updated
       do i=1, nboth
        r_o(iatom_both(2,i), :, left:right)= &
     & r_f(iatom_both(1,i), :, left:right)
       enddo
      endif
!
      end subroutine ftsm_update_overlap_coor
!========================================================================
      subroutine ftsm_coor_wgt_alloc(mass,n) ! massed needed for COM computation
      use stream
      use ivector; use ivector_list; use rvector; use rvector_list
!
      integer, intent(in) :: n
      real(chm_real), intent(in) :: mass(n)
      real(chm_real) :: wsum
      character(len=len("FTSM_COOR_WGT_ALLOC>") ),parameter::whoami="FTSM_COOR_WGT_ALLOC>";!macro
!
!
      integer :: i, j, k
      integer, pointer :: inds(:)
      real(chm_real), pointer :: w(:)
!
! do nothing if arrays already allocated
      if (.not.((.not.((associated(r_o).and.associated(orientWeights).or..not.qorient).and.associated(r_f).and.associated(forcedWeights))))) return
!
      if (.not.ftsm_initialized) then; call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. NOTHING DONE.')); return; endif;
!
      if (nforced.le.0) then
       call wrndie(0,whoami,trim('NO FORCING ATOMS SELECTED. ABORT.'))
       return
      endif
!
! forcing arrays
!
!
      if(associated(r_f,target=r_o)) then; nullify(r_f);else; if(associated(r_f))deallocate(r_f); endif ! nullify rf if rf=>ro, otherwise deallocate
      allocate(r_f(nforced,3,num_sets)) ; r_f=anum
      if(associated(forcedWeights))deallocate(forcedWeights)
      allocate(forcedWeights(nforced)) ; forcedWeights = one / nforced ;
      if (ftsm_com_on) then
       if (associated(wgts_f, target=wgts_o)) allocate(wgts_f)
       call real_vlist_reinit(wgts_f)
!
       do i=1, nforced
        inds=>iatoms_f%v(i)%i
        do j=1, iatoms_f%v(i)%last
         if (inds(j) > n) then;call wrndie(0,whoami,trim('MASS ARRAY BOUNDS EXCEEDED. ABORT.')); if(associated(r_f))deallocate(r_f); return; endif
         k=real_vlist_uadd(wgts_f, i, mass( inds(j) ))
        enddo
! append total mass to the end and scale weights
        j=wgts_f%v(i)%last
        w=>wgts_f%v(i)%r(1:j) ! note: cannot use j as the length of the list (j=len+1, but depends on implementation)
        wsum=sum(w(1:j))
        if (wsum.gt.RSMALL) then
         w=w/wsum
         k=real_vlist_uadd(wgts_f, i, wsum )
        else
         call wrndie(0,whoami,trim('SUM OF COM GROUP WEIGHTS IS VERY SMALL. ABORT.'))
         return
        endif
       enddo ! nforced
      endif ! ftsm_com_on
!
! qorient
      if (norient.lt.3) then
        if (norient.gt.0) then ! greater than 0 -- user added an insufficient number of o groups
         call wrndie(0,whoami,trim(' FEWER THAN THREE ATOMS SELECTED FOR ORIENTATION. WILL NOT ORIENT.'))
        endif
        qorient=.false.
        return
      else
        qorient=.true.
      endif
!
      if (qorient) then
       if(associated(r_o))deallocate(r_o)
       if(associated(orientWeights))deallocate(orientWeights)
       allocate(orientWeights(norient)) ; orientWeights = one / norient ;
       if (.not. associated(rcom)) allocate(rcom(3,num_sets))
       rcom=zero ! must be zero initially
       if (ftsm_com_on) call real_vlist_done(wgts_o)
!
       if (ftsm_compute_qdiffrot) call ftsm_qdiffrot()
       if (.not.qdiffrot) then
        r_o=>r_f
! COM version
        if (ftsm_com_on) then
         deallocate(wgts_o)
         wgts_o=>wgts_f
        endif
       else ! qdiffrot
        allocate(r_o(norient,3,num_sets)) ; r_o=anum
        if (ftsm_com_on) then
         call ftsm_compute_overlap_ind()
!
         do i=1, norient
          inds=>iatoms_o%v(i)%i
          do j=1, iatoms_o%v(i)%last
           if (inds(j) > n) then;call wrndie(0,whoami,trim('MASS ARRAY BOUNDS EXCEEDED. ABORT.')); if(associated(r_f))deallocate(r_f); return; endif
           k=real_vlist_uadd(wgts_o, i, mass( inds(j) ))
          enddo
! append total mass to the end and scale weights
          j=wgts_o%v(i)%last
          w=>wgts_o%v(i)%r(1:j)
          wsum=sum(w(1:j)) ; ! specify bounds just in case the compiler does not set them
          if (wsum.gt.RSMALL) then
           w=w/wsum
           k=real_vlist_uadd(wgts_o, i, wsum )
          else
           call wrndie(0,whoami,trim('SUM OF COM GROUP WEIGHTS IS VERY SMALL. ABORT.'))
           return
          endif
         enddo ! norient
        endif ! ftsm_com_on
       endif ! qdiffrot
       call ftsm_define_rtmd_type()
      endif ! qorient
!
      end subroutine ftsm_coor_wgt_alloc
!========================================================================
      subroutine ftsm_set_weights(w,n)
! NOTE : weights are assumed to be indexed by PSF atomid (so that n is effectively natom)
      use stream
      use number
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
!
      integer :: i, j, n
      real(chm_real) :: w(n), a
      integer, pointer :: inds(:)
      character(len=len("FTSM_SET_WEIGHTS>") ),parameter::whoami="FTSM_SET_WEIGHTS>";!macro
!
      if (qorient.and.ME_STRNG.eq.0.and.ME_LOCAL.eq.0) then
! presumably, need to recompute COM and send to neighbors (saves a few extra lines here)
       write(info,668) &
     & whoami,' A CHANGE IN THE ORIENTATION WEIGHTS REQUIRES ', &
     & whoami,' REDEFINING IMAGES (e.g. USING FILL)' ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
 668 format(/2A,/2A)
!
      if (norient.eq.0.and.nforced.eq.0) then
       call wrndie(0,whoami,trim('NO STRING ATOMS FOUND. NOTHING DONE.'))
       return
      endif
!
      if (associated(orientWeights)) then
       if (ftsm_com_on) then
        if (associated(iatoms_o) .and. associated(wgts_o)) then
         do i=1, norient
          inds=>iatoms_o%v(i)%i(1:iatoms_o%v(i)%last)
          if (any(inds.gt.n)) then
            call wrndie(0,whoami,trim('WEIGHT ARRAY BOUNDS EXCEEDED. ABORT.'))
            return
          else
           orientWeights(i)=sum(w(inds))
          endif
         enddo ! norient
        endif ! associated (iatoms_o)
       else
        if (associated(iatom_o)) then
         do i=1,norient
          j=iatom_o(i)
          if (j.le.n) then
           orientWeights(i)=w(j)
          else
           call wrndie(0,whoami,trim('WEIGHT ARRAY BOUNDS EXCEEDED. ABORT.'))
           return
          endif ! bounds check
         enddo
        endif ! associated
       endif ! ftsm_com_on
!
       a=sum(orientWeights);
       if (a.gt.RSMALL) then ; orientWeights=orientWeights/a;
       else
        call wrndie(0,whoami,trim('SUM OF ORIENTATION WEIGHTS IS VERY SMALL. ABORT.'))
        return
       endif
!
      endif ! orientWeights
!
      if (associated(forcedWeights)) then
       if (ftsm_com_on) then
        if (associated(iatoms_f) .and. associated(wgts_f)) then
         do i=1, nforced
          inds=>iatoms_f%v(i)%i(1:iatoms_f%v(i)%last)
          if (any(inds.gt.n)) then
            call wrndie(0,whoami,trim('WEIGHT ARRAY BOUNDS EXCEEDED. ABORT.'))
            return
          else
           forcedWeights(i)=sum(w(inds))
          endif
         enddo ! norient
        endif ! associated (iatoms_f)
       else
        if (associated(iatom_f)) then
         do i=1,nforced
          j=iatom_f(i)
          if (j.le.n) then
           forcedWeights(i)=w(j)
          else
           call wrndie(0,whoami,trim('WEIGHT ARRAY BOUNDS EXCEEDED. ABORT.'))
           return
          endif
         enddo
        endif
       endif ! ftsm_com_on
!
       a=sum(forcedWeights);
       if (a.gt.RSMALL) then ; forcedWeights=forcedWeights/a;
       else
        call wrndie(0,whoami,trim('SUM OF FORCING WEIGHTS IS VERY SMALL. ABORT.'))
        return
       endif
!
      endif ! forced weights associated
!
      end subroutine ftsm_set_weights
!================================================================
      function ftsm_test_grad_fd(x,y,z,h) result(error)
      use ftsm_compute
      use ftsmv2_compute
      real(chm_real) :: x(:), y(:), z(:)
      real(chm_real), intent(in) :: h
      real(chm_real), pointer :: error(:,:)
      integer :: i, j, k, jj
      real(chm_real) :: d, ap, am, bp, bm
      real(chm_real) :: s, wgt
! variables for ftsm_com_on
      real(chm_real), pointer, dimension(:,:) :: ffpar, ffprp, fopar, foprp
      real(chm_real), target :: ptrone(1) = one
      integer, pointer :: inds(:)
      real(chm_real), pointer :: wgts(:)
      integer :: jend, ind
!
      type (real_vlist) :: gpar, gprp ! parallel and perpendicular auxiliary gradient arrays
!
      character(len=len("FTSM_TEST_GRAD_FD>") ),parameter::whoami="FTSM_TEST_GRAD_FD>";!macro
!
      allocate(error(2,3))
      ap=0d0; am=0d0; bp=0d0; bm=0d0; error(1,1)=anum
!
      if (.not.ftsm_check(qorient)) return
!
      if (h.eq.0d0) then
       call wrndie(0,whoami,trim('COORDINATE PERTURBATION ZERO.'))
       return
      endif
!
      s=1d0
! s=0.5d0
! compute projection and derivatives analytically
      if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.true.,s); else ; call ftsmv1_calc(x,y,z,.true.,s); endif
! reorganize gradient data to perform FD test
! 1) use COM gradients to compute atomic gradients (e.g. derivatives w.r.t. atom (not COM) positions
! to do this, create an list that includes only atoms to which forces are applied
      ffpar=>r_f(:,:,fpar); ffprp=>r_f(:,:,fperp) ! forcing forces
      fopar=>r_o(:,:,fpar); foprp=>r_o(:,:,fperp) ! orientation forces
!
      do i=1, nforced
       if (ftsm_com_on) then
        jend=iatoms_f%v(i)%last
        inds=>iatoms_f%v(i)%i
        wgts=>wgts_f%v(i)%r
       else
        jend=1; inds=>iatom_f(i:); wgts=>ptrone
       endif
       do j=1, jend
         ind=inds(j)
         wgt=wgts(j)
         k=real_vlist_uadd (gpar, ind, wgt*ffpar(i,1)); ! create/add to psf-indexed list x-gradient contribution from this (ith) COM
         jj=real_vector_add(gpar%v(k), wgt*ffpar(i,2)); ! add y-gradient contribution (can add directly to list avoiding label search)
         jj=real_vector_add(gpar%v(k), wgt*ffpar(i,3)); ! add z-gradient contribution
!
         k=real_vlist_uadd(gprp, ind, wgt*ffprp(i,1)); ! same as above for perpendicular gradient component
         jj=real_vector_add(gprp%v(k), wgt*ffprp(i,2));
         jj=real_vector_add(gprp%v(k), wgt*ffprp(i,3));
       enddo ! j
      enddo ! nforced
!
      if (qorient.and.qdiffrot) then
       do i=1, norient
        if (ftsm_com_on) then
         jend=iatoms_o%v(i)%last
         inds=>iatoms_o%v(i)%i
         wgts=>wgts_o%v(i)%r
        else
         jend=1; inds=>iatom_o(i:); wgts=>ptrone
        endif
        do j=1, jend
         ind=inds(j)
         wgt=wgts(j)
         k=real_vlist_uadd (gpar, ind, wgt*fopar(i,1)); ! same as above but for orientation atoms
         jj=real_vector_add(gpar%v(k), wgt*fopar(i,2));
         jj=real_vector_add(gpar%v(k), wgt*fopar(i,3));
!
         k=real_vlist_uadd(gprp, ind, wgt*foprp(i,1));
         jj=real_vector_add(gprp%v(k), wgt*foprp(i,2));
         jj=real_vector_add(gprp%v(k), wgt*foprp(i,3));
        enddo ! j
       enddo ! norient
      endif ! qorient && qdiffrot
!
      error=zero
! 2) loop over all atoms involved in FTSM coordinates and compute finite differences
      do jj=1, gpar%last
! to get the total force, need to sum over the force list
       k=gpar%v(jj)%last
       j=gpar%i(jj) ! psf index
! x-derivatives *******************************************
       d=x(j) ; x(j)=d-h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif
       if (proj_on) then; am=dpar; bm=dperp; else; am=drms ; endif
       x(j)=d+h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif ! overwrite zold cv value array (warn about this elsewhere)
       if (proj_on) then; ap=dpar; bp=dperp; else; ap=drms ; endif
       x(j)=d ! restore correct value
!
       error(1,1) = max (error(1,1), &
     & abs(0.5d0/h*(ap-am)-sum(gpar%v(jj)%r(1:k:3))) )
       if (proj_on) &
     & error(2,1) = max (error(2,1), &
     & abs(0.5d0/h*(bp-bm)-sum(gprp%v(jj)%r(1:k:3))) )
! y-derivatives *******************************************
       d=y(j) ; y(j)=d-h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif
       if (proj_on) then; am=dpar; bm=dperp; else; am=drms ; endif
       y(j)=d+h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif ! overwrite zold cv value array (warn about this elsewhere)
       if (proj_on) then; ap=dpar; bp=dperp; else; ap=drms ; endif
       y(j)=d ! restore correct value
!
       error(1,2) = max (error(1,2), &
     & abs(0.5d0/h*(ap-am)-sum(gpar%v(jj)%r(2:k:3))) )
       if (proj_on) &
     & error(2,2) = max (error(2,2), &
     & abs(0.5d0/h*(bp-bm)-sum(gprp%v(jj)%r(2:k:3))) )
! z-derivatives *******************************************
       d=z(j) ; z(j)=d-h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif
       if (proj_on) then; am=dpar; bm=dperp; else; am=drms ; endif
       z(j)=d+h
       if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.false.,s); else ; call ftsmv1_calc(x,y,z,.false.,s); endif ! overwrite zold cv value array (warn about this elsewhere)
       if (proj_on) then; ap=dpar; bp=dperp; else; ap=drms ; endif
       z(j)=d ! restore correct value
!
       error(1,3) = max (error(1,3), &
     & abs(0.5d0/h*(ap-am)-sum(gpar%v(jj)%r(3:k:3))) )
       if (proj_on) &
     & error(2,3) = max (error(2,3), &
     & abs(0.5d0/h*(bp-bm)-sum(gprp%v(jj)%r(3:k:3))) )
      enddo ! all atoms with forces
!
      call real_vlist_done(gpar)
      call real_vlist_done(gprp)
!
      end function ftsm_test_grad_fd
!============================================================================
      function ftsm_test_parallel(x,y,z) result(error)
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use ftsm_compute
      use ftsmv2_compute
!
      real(chm_real) :: x(:), y(:), z(:)
      real(chm_real), pointer :: error(:,:)
      real(chm_real) :: am, bm
!
      logical :: qpara
      logical :: qgrp
!
      character(len=len("FTSM_TEST_PARALLEL>") ),parameter::whoami="FTSM_TEST_PARALLEL>";!macro
      allocate(error(2,4)) ! first column contains the CV values; then maximum derivative error (x,y,z)
      error(2,:)=0d0; error(1,:)=anum
!
      if (.not.ftsm_check(qorient)) return
!
      qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL) &
     & .and.(SIZE_LOCAL.gt.1)
      if (.not. qgrp) then ! quit if cannot run in parallel
       call wrndie(0,whoami,trim('CANNOT PERFORM TEST ON 1-PROCESSOR GROUPS'))
       return
      endif
! save values & force a serial calculation
      qpara=calc_bestfit_grad_para; calc_bestfit_grad_para=.false.
!
! 1) compute serially
      if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.true.); else ; call ftsmv1_calc(x,y,z,.true.); endif
! using left_cur array : it should not be used unless an additional parameter is passed to calc
! save gradients
      r_f(:,:,left_cur)=r_f(:,:,fpar); r_f(:,:,fpar)=0d0;
      if (qorient.and.qdiffrot) then
       r_o(:,:,left_cur)=r_o(:,:,fpar); r_o(:,:,fpar)=0d0
      endif
!
      if (proj_on) then
       r_f(:,:,right_cur)=r_f(:,:,fperp); r_f(:,:,fperp)=0d0
       if (qorient.and.qdiffrot) then
        r_o(:,:,right_cur)=r_o(:,:,fperp); r_o(:,:,fperp)=0d0
       endif
      endif
! write(me_GLOBAL+500,*) drms, dpar, dperp
! write(me_GLOBAL+500,*) r_o(:,2,left_cur)
! save projections
      if (proj_on) then; am=dpar; bm=dperp; else; am=drms ; endif
!
! 2) compute in (fully) parallel
      calc_bestfit_grad_para=.true.
!
      if (qver2.and.proj_on) then ; call ftsmv2_calc(x,y,z,.true.); else ; call ftsmv1_calc(x,y,z,.true.); endif
! write(me_GLOBAL+500,*) drms, dpar, dperp
! write(me_GLOBAL+500,*) r_o(:,2,fpar)
! close(me_GLOBAL+500)
! compute largest absolute error
! parallel component (or drms)
      error(1,1) = maxval ( abs (r_f(:,1,left_cur)-r_f(:,1,fpar) ) )
      error(1,2) = maxval ( abs (r_f(:,2,left_cur)-r_f(:,2,fpar) ) )
      error(1,3) = maxval ( abs (r_f(:,3,left_cur)-r_f(:,3,fpar) ) )
      if (proj_on) then
       error(2,1) = maxval ( abs (r_f(:,1,right_cur)-r_f(:,1,fperp) ) )
       error(2,2) = maxval ( abs (r_f(:,2,right_cur)-r_f(:,2,fperp) ) )
       error(2,3) = maxval ( abs (r_f(:,3,right_cur)-r_f(:,3,fperp) ) )
       error(2,4) = abs ( dperp - bm )
       error(1,4) = abs ( dpar - am )
      else
       error(1,4) = abs ( drms - am )
      endif
!
      if (qdiffrot.and.qorient) then
       error(1,1) = max(error(1,1), &
     & maxval ( abs (r_o(:,1,left_cur)-r_o(:,1,fpar) ) ) )
       error(1,2) = max(error(1,2), &
     & maxval ( abs (r_o(:,2,left_cur)-r_o(:,2,fpar) ) ) )
       error(1,3) = max(error(1,3), &
     & maxval ( abs (r_o(:,3,left_cur)-r_o(:,3,fpar) ) ) )
       if (proj_on) then
        error(2,1) = max(error(2,1), &
     & maxval ( abs (r_o(:,1,right_cur)-r_o(:,1,fperp) ) ))
        error(2,2) = max(error(2,2), &
     & maxval ( abs (r_o(:,2,right_cur)-r_o(:,2,fperp) ) ))
        error(2,3) = max(error(2,3), &
     & maxval ( abs (r_o(:,3,right_cur)-r_o(:,3,fperp) ) ))
       endif
      endif
! restore original option
      calc_bestfit_grad_para=qpara
!
      end function ftsm_test_parallel
!==========================================================================================
#endif
#endif /* automatically protect all code */
      end module ftsm_util
!
