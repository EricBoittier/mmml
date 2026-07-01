! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! I/O module for finite-temperature string module
      module ftsm_io
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ftsm_var
      implicit none
!
!=====================================================================
! SUBROUTINES
      public ftsm_write_dcd
      public ftsm_read_dcd
      public ftsm_write_cor
      public ftsm_read_cor
!
      contains
!======================================================================
      subroutine ftsm_write_dcd(ifile, col, ibeg, iend)
!
      use psf
#if (KEY_CVELOCI==1)
      use cveloci_mod
#endif
      use version
!
      use dimens_fcm
      use coord; use coordc
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use string
      use cvio, only : writcv, readcv
      use coorio_mod, only : cwrite, cread
      use ctitla

      use mpi
      use number
      use ftsm_util

      implicit none

! will use fixed atom arrays to print only the path atoms
!

      character(len=80) :: title(maxtit)
      integer :: ntitle, ncv




      real(chm_real), pointer :: r3(:,:) ! coordinates
      integer, pointer :: stringatoms(:), string_inds(:)

      integer :: header(20)=zero

      integer :: i, j, k
      integer :: ifile
      integer, optional :: col, ibeg, iend
      integer :: c, ierror, stat(MPI_STATUS_SIZE), ibg, ie, ind
      integer :: rtype=MPI_DOUBLE_PRECISION
      real(chm_real) :: r_com(3)
      logical :: qroot
!
      character(len=len("FTSM_WRITE_DCD>") ),parameter::whoami="FTSM_WRITE_DCD>";!macro
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
!

      title(1) = '* FINITE TEMPERATURE STRING IMAGE FILE'
      ntitle=1

!
      if (.not. ftsm_check(qorient)) return
      if (present(col)) then ; c=col; else; c=center; endif
      if (c.lt.1.or.c.gt.num_sets) then
       call wrndie(0,whoami,trim('INVALID COLUMN. ABORT.'))
       return
      endif
!
      if (present(ibeg)) then ; ibg=ibeg; else; ibg=1; endif
      if (present(iend)) then ; ie=iend; else; ie=nstring; endif
!
      if (ibg.lt.1.or.ibg.gt.ie) then
       call wrndie(0,whoami,trim('INVALID FRAMES REQUESTED. ABORT.'))
       return
      endif
!
      nullify(stringatoms, string_inds, r3)
!
      if (qroot) then
!
      call ftsm_update_overlap_coor(ione)
!
      if (ME_STRNG.eq.0) then
!



!
       allocate(stringatoms(natom), string_inds(natom))
       stringatoms=0
!
       if (ftsm_com_on) then
        do i=1, nforced ; stringatoms(iatoms_f%v(i)%i(1)) = 1 ; enddo ! put in index of first atom in each COM group
        if (qorient.and.qdiffrot) then
         do i=1, norient ; stringatoms(iatoms_o%v(i)%i(1)) = 1 ; enddo
        endif
       else
        stringatoms(iatom_f)=1
        if (qorient.and.qdiffrot) stringatoms(iatom_o)=1
       endif
!
       k=0
       do i = 1, natom
        if (stringatoms(i).gt.0) then; k=k+1; string_inds(k)=i; endif ! need this array for dcd writer below
       enddo

#if (KEY_CVELOCI==1) /*  exclude constant velocity atoms also */
       ncv=ncvel ; ncvel=0
#endif

! for first frame, output all coordinates: take from instantaneous set
       allocate(r3(natom,3)) ; r3=0d0
       if (qorient) then ; r_com=rcom(:,instant) ; else ; r_com=zero ; endif

       r3(:,1)=x(1:natom)-r_com(1) ! bring all coordinates to zero COM for convenience (assume rcom(:,instant) is reasonably accurate)
       r3(:,2)=y(1:natom)-r_com(2)
       r3(:,3)=z(1:natom)-r_com(3)





!
       do j=1,nforced
        if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
        r3(ind,:)=r_f(j,:,c)
       enddo
       if (qorient.and.qdiffrot) then
        do j=1,norient
         if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
         r3(ind,:)=r_o(j,:,c)
        enddo
       endif
! call trajectory writer

! (first, generate custom icontrol array)
       header(1)=nstring ! number of sets
       header(2)=1 ! step
       header(3)=1 ! interval
       header(4)=nstring ! total number of trajectory "frames"
       header(9)=natom-k ! number of free (non-fixed) atoms
       header(8)=header(9)*3 ! number of dof
       header(20)=vernum ! version
!
       call writcv(r3(:,1), r3(:,2), r3(:,3), &
#if (KEY_CHEQ==1)
     & r3(:,1), .false., &
#endif
     & natom, &
     & string_inds, k, ibg, ibg, 3*k, 0d0, 1, ie, title,ntitle,ifile, &
     & .false., .true., header, .false., r3(:,1)) ! whew...



!from dynio.src
! SUBROUTINE WRITCV(X,Y,Z,
! $ CG,QCG, !##CHEQ
! $ NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF,
! $ DELTA,NSAVC,NSTEP,TITLE,NTITLE,IUNCRD,QVEL,
! $ QINCT,JCNTRL,DIM4,FDIM)
!C
!C WRITES A SET OF COORDINATES FOR A SINGLE DYNAMICS STEP. THE FORMAT
!C FOR THE TRAJECTORY FILE VARIES WITH WHETHER ANY ATOMS ARE FIXED.
!C ICNTRL(9) STORES THE NUMBER OF FIXED ATOMS WHICH WILL BE ZERO FOR
!C ALL PREVIOUS TRAJECTORY FILES SO COMPATIBILITY IS ASSURED.
!C
!C Authors: S. Swaminathan
!C Robert Bruccoleri
!
! repeat a few times to write out entire string
       do i=2, nstring
!
        call MPI_RECV(r_f(:,:,dummy),3*nforced,rtype,i-1,i-1,MPI_COMM_STRNG,stat,ierror)
! extract coordinates
        do j=1,nforced
         if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
         r3(ind,:)=r_f(j,:,dummy)
        enddo
!
        if (qdiffrot.and.qorient) then
         call MPI_RECV(r_o(:,:,dummy),3*norient,rtype,i-1,i-1,MPI_COMM_STRNG,stat,ierror)
         do j=1,norient
          if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
          r3(ind,:)=r_o(j,:,dummy)
         enddo
        endif
!
! write next frame

        call writcv(r3(:,1), r3(:,2), r3(:,3), &
#if (KEY_CHEQ==1)
     & r3(:,1), .false., &
#endif
     & natom, &
     & string_inds, k, ibg, ibg+i-1, 3*k, 0d0, 1, ie, title, 1, ifile,&
     & .false., .true., header, .false., r3(:,1))



       enddo
!

#if (KEY_CVELOCI==1)
       ncvel=ncv
#endif

      else
       call MPI_SEND(r_f(:,:,c),3*nforced,rtype,0,ME_STRNG, &
     & MPI_COMM_STRNG, ierror)
       if (qdiffrot.and.qorient) &
     & call MPI_SEND(r_o(:,:,c),3*norient,rtype,0,ME_STRNG, &
     & MPI_COMM_STRNG, ierror)
      endif ! string root
!
      endif ! qroot
!
      if (associated(stringatoms)) deallocate(stringatoms)
      if (associated(string_inds)) deallocate(string_inds)
      if (associated(r3)) deallocate(r3)
!
      end subroutine ftsm_write_dcd
!===============================================================
      subroutine ftsm_read_dcd(ifile, col)

      use psf

      use dimens_fcm
      use coord; use coordc
#if (KEY_MULTICOM==1)
      use multicom_aux;
#endif
      use string
      use mpi
      use cvio, only : writcv, readcv
      use coorio_mod, only : cwrite, cread
      use ctitla
      use ftsm_util

      implicit none

      character(len=len("FTSM_READ_DCD>") ),parameter::whoami="FTSM_READ_DCD>";!macro
      real(chm_real), pointer :: r3(:,:)
      integer, pointer :: stringatoms(:), string_inds(:)
      integer :: i, j, k, ind, ifile
      integer, optional :: col
      integer :: c, ierror, stat(MPI_STATUS_SIZE)
      integer :: rtype=MPI_DOUBLE_PRECISION
!

      character(len=80) :: title(maxtit)
      real*4 :: trash4(natom) ! scratch array for ugly routine
      real(chm_real) :: trash8(natom) ! scratch array for ugly routine
! some dummy vars for coordinate read
      integer :: nfile, istep, istats, ndof, begin_, stop_, &
     & skip_, nsavv_, satoms, ntitle
      real(chm_real) :: delta
      logical :: qdim4, qcg






!
      if (.not. ftsm_check(qorient)) return
      if (present(col)) then ; c=col; else; c=center; endif
      if (c.lt.1.or.c.gt.num_sets) then
       call wrndie(0,whoami,trim('INVALID COLUMN. ABORT.'))
       return
      endif
!



!
      allocate(r3(natom,3))
      allocate(stringatoms(natom), string_inds(natom))
!
      r3=0d0; stringatoms=0
!
      if (ftsm_com_on) then
       do i=1, nforced ; stringatoms(iatoms_f%v(i)%i(1)) = 1 ; enddo ! put in index of first atom in each COM group
       if (qorient.and.qdiffrot) then
        do i=1, norient ; stringatoms(iatoms_o%v(i)%i(1)) = 1 ; enddo
       endif
      else
       stringatoms(iatom_f)=1
       if (qorient.and.qdiffrot) stringatoms(iatom_o)=1
      endif
!
      k=0
      do i = 1, natom
       if (stringatoms(i).gt.0) then ; k=k+1 ; string_inds(k)=i ; endif ! need this array for dcd reader below
      enddo
!
      if (ME_STRNG.eq.0) then
!
! call trajectory reader
!

       istats=1
       qcg=.false.
       qdim4=.false.
       begin_=0 ! note begin <=0 forces a strange "reset" with begin=istep (which is zero below); this is to support trajectories
                ! made with VMD
       skip_=1
       stop_=nstring
       ntitle=0
       istep=0
!
       call readcv(r3(:,1), r3(:,2), r3(:,3), &
#if (KEY_CHEQ==1)
     & trash8, qcg, &
#endif
     & trash4, natom, &
     & stringatoms, satoms, ifile, 1, ifile, nfile, &
     & istep, istats, ndof, delta, begin_, stop_, skip_, &
     & nsavv_, 'CORD', 'CORD', title, ntitle, qdim4, trash8, .false.)




!
       if ( &

& satoms &



& .eq.k) then
        if ( any(stringatoms(1:k).ne.string_inds(1:k)) ) &
     & call wrndie(0,whoami,trim('INVALID STRING ATOM INDICES. BEWARE.'))
       else
        call wrndie(0,whoami,trim('INCORRECT NUMBER OF STRING ATOMS. BEWARE.'))
       endif
!
!
! SUBROUTINE READCV(X,Y,Z,
! $ CG,QCG, !##CHEQ
! $ TEMP,NATOM,FREEAT,NFREAT,
! $ FIRSTU,NUNIT,IUNIT,NFILE,
! $ ISTEP,ISTATS,NDEGF,DELTA,
! $ BEGIN,STOP,SKIP,NSAVV,HDR1,HDR2,
! $ TITLE,NTITLE,DIM4,FDIM,Q_PLL)
! FREEAT WILL BE READ IF NFREAT IS NOT EQUAL TO NATOM.
! ISTATS IS A FLAG WITH THE FOLLOWING FUNCTIONS:
!CC ON CALL
!C 1 - OPEN A NEW UNIT FOR READING THE INFORMATION
!C 2 - USE THE UNIT THAT IS ALREADY OPEN ON IUNIT
!C ON RETURN
!C -1 - THE REQUESTED INFORMATION HAS BEEN READ
!C 1 - NOT DONE READING, BUT THIS FILE IS FINISHED
!C 2 - NOT DONE READING, AND THIS FILE IS IS NOT DONE.
!C HDR1 AND HDR2 ARE OPTIONS FOR THE FILE HEADER THAT IS READ.
!C RECORDS WILL BE USED IF MOD(ISTEP,SKIP)=0 AND BEGIN<=ISTEP<=STOP.
!C DJS 1/25/81
!C
!C Authors: S. Swaminathan
!C David Perahia
!C Dave States
!C Robert Bruccoleri
!C
!C Q_PLL mfc added logical variable to signal whether
!C the calling routine is being done in parallel or not.
!C When calling routine is not parallel, master hangs
!C trying to send data to slaves that are not receiving.
!C
!
!
       do j=1, nforced
        if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
        r_f(j,:,c)=r3(ind,:)
       enddo
!
       if (qorient.and.qdiffrot) then
        do j=1, norient
         if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
         r_o(j,:,c)=r3(ind,:)
        enddo
       endif
!
! repeat a few times to read entire string
       do i=2, nstring

        call readcv(r3(:,1), r3(:,2), r3(:,3), &
#if (KEY_CHEQ==1)
     & trash8, qcg, &
#endif
     & trash4, natom, &
     & stringatoms, satoms, ifile, 1, ifile, nfile, &
     & istep, istats, ndof, delta, begin_, stop_, skip_, &
     & nsavv_, 'CORD', 'CORD', title, ntitle, qdim4, trash8, .false.)



!
        call MPI_SEND(r3,3*natom,rtype,i-1,i-1, &
     & MPI_COMM_STRNG, ierror)
       enddo ! i
!
      else ! me_string == 0
!
       call MPI_RECV(r3,3*natom,rtype,0,ME_STRNG, &
     & MPI_COMM_STRNG,stat,ierror)
!
       do j=1, nforced
        if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
        r_f(j,:,c)=r3(ind,:)
       enddo
!
       if (qorient.and.qdiffrot) then
        do j=1, norient
         if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
         r_o(j,:,c)=r3(ind,:)
        enddo
       endif
      endif
!
      call ftsm_save_com(c) ! compute and remove center of mass
!
      if(associated(stringatoms))deallocate(stringatoms)
      if(associated(string_inds))deallocate(string_inds)
      if(associated(r3))deallocate(r3)



!
      end subroutine ftsm_read_dcd
!=====================================================
      subroutine ftsm_write_cor(ifile, col)

      use psf

      use coord; use coordc
      use dimens_fcm
      use string
      use cvio, only : writcv, readcv
      use coorio_mod, only : cwrite, cread
      use ctitla
      use number
      use ftsm_util

      implicit none

      character(len=80) :: title(maxtit)
      integer :: ntitle
! compatibility variables for coordinate reading/writing
      real(chm_real) :: wdum(natom+1)
      integer :: icntrl(20)=0, modew






      real(chm_real), pointer :: r3(:,:)
      integer, pointer :: stringatoms(:)
      integer :: j, ind
      integer, optional :: col
      integer :: c, ifile
!
      character(len=len("FTSM_WRITE_COR>") ),parameter::whoami="FTSM_WRITE_COR>";!macro
!

      ntitle = 1
      title(1) = '* FINITE TEMPERATURE STRING IMAGE FILE'

!
      if (.not. ftsm_check(qorient)) return
      if (present(col)) then ; c=col; else; c=center; endif
      if (c.lt.1.or.c.gt.num_sets) then
       call wrndie(0,whoami,trim('INVALID COLUMN. ABORT.'))
       return
      endif
!



!
      allocate(r3(natom,3),stringatoms(natom))
      r3=0d0; stringatoms=0
!
      if (ftsm_com_on) then
       do j=1, nforced ; stringatoms(iatoms_f%v(j)%i(1)) = 1 ; enddo ! put in index of first atom in each COM group
       if (qorient.and.qdiffrot) then
        do j=1, norient ; stringatoms(iatoms_o%v(j)%i(1)) = 1 ; enddo
       endif
      else
       stringatoms(iatom_f)=1
       if (qorient.and.qdiffrot) stringatoms(iatom_o)=1
      endif
!
      call ftsm_update_overlap_coor(ione)
      do j=1, nforced
       if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
       r3(ind,:)=r_f(j,:,c)
      enddo
!
      if (qorient.and.qdiffrot) then
       do j=1, norient
        if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
        r3(ind,:)=r_o(j,:,c)
       enddo
      endif
!

! call writer
! formatted coor card files
      modew=2
      wdum=0d0
!
      call cwrite(ifile,title,ntitle,icntrl, &
     & r3(:,1),r3(:,2),r3(:,3),wdum, &
     & res,atype,ibase, &
     & nres,natom,stringatoms,modew,0,0,.false.)



!
      if(associated(stringatoms))deallocate(stringatoms)
      if(associated(r3))deallocate(r3)
!
      end subroutine ftsm_write_cor
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ftsm_read_cor(ifile, col)
!
      use dimens_fcm
      use coord; use coordc
      use string
!

      use psf
      use ctitla

      use number
      use cvio, only : writcv, readcv
      use coorio_mod, only : cwrite, cread
      use ctitla
      use ftsm_util

      implicit none

      real(chm_real), pointer :: r3(:,:)
      integer, pointer :: stringatoms(:)
      integer :: j, ind
      integer, optional :: col
      integer :: c, ifile
!

      character(len=80) :: title(maxtit), cdummy=''
! compatibility variables for coordinate reading/writing
      real(chm_real) :: xdum(natom+1), ydum(natom+1), zdum(natom+1), &
     & wdum(natom+1)
!
      integer :: icntrl(20), moder, ntitle=0
      integer :: ifree(natom)



!
      character(len=len("FTSM_READ_COR>") ),parameter::whoami="FTSM_READ_COR>";!macro
!
      if (.not. ftsm_check(qorient)) return
      if (present(col)) then ; c=col; else; c=center; endif
      if (c.lt.1.or.c.gt.num_sets) then
       call wrndie(0,whoami,trim('INVALID COLUMN. ABORT.'))
       return
      endif
!



!
      allocate(r3(natom,3),stringatoms(natom))
      r3=0d0; stringatoms=0
!
      if (ftsm_com_on) then
       do j=1, nforced ; stringatoms(iatoms_f%v(j)%i(1)) = 1 ; enddo ! put in index of first atom in each COM group
       if (qorient.and.qdiffrot) then
        do j=1, norient ; stringatoms(iatoms_o%v(j)%i(1)) = 1 ; enddo
       endif
      else
       stringatoms(iatom_f)=1
       if (qorient.and.qdiffrot) stringatoms(iatom_o)=1
      endif
! call reader
! formatted coor card files

      moder=1
      xdum=anum; ydum=anum; zdum=anum; wdum=anum
!
      call cread(ifile, title, ntitle, icntrl, &
     & r3(:,1), r3(:,2), r3(:,3), & ! pass by reference ?
     & wdum, natom, moder, stringatoms, &
     & 0, res, nres, atype, ibase, 1, ifree, &
     & segid, resid, nictot, nseg, .false., .false., &
     & cdummy, 80, 0, .false.)
!
!
! from coor.io
! SUBROUTINE CREAD(IUNIT,TITLE,NTITL,ICNTRL,X,Y,Z,WMAIN,NATOM,
! & NINPUT,ISLCT,IOFFS,RES,NRES,TYPE,IBASE,
! & IFILE,FREEAT,SEGID,RESID,NICTOT,NSEG,LRSID,LFREE,LYN,MXLEN,
! & MODEL,OFFICIAL)
!



!
      do j=1, nforced
       if (ftsm_com_on) then ; ind=iatoms_f%v(j)%i(1) ; else ; ind=iatom_f(j) ; endif
       r_f(j,:,c)=r3(ind,:)
      enddo
!
      if (qorient.and.qdiffrot) then
       do j=1, norient
        if (ftsm_com_on) then ; ind=iatoms_o%v(j)%i(1) ; else ; ind=iatom_o(j) ; endif
        r_o(j,:,c)=r3(ind,:)
       enddo
      endif
!
      call ftsm_save_com(c) ! compute remove center of mass
!
      if (associated(stringatoms)) deallocate(stringatoms)
      if (associated(r3)) deallocate(r3)
!
      end subroutine ftsm_read_cor
!=================================================================================
#endif

#endif /* automatically protect all code */
      end module ftsm_io
!
