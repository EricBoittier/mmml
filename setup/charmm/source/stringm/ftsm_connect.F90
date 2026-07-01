! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! module for finite-temperature string module for reconnecting the path to minimize length
! compute distances between all pairs of string images; optimize order to produce a shorter total path length
!
      module ftsm_connect
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ivector_list ! to keep track of migrations
      use ftsm_var
      use tsp
      use multicom_aux;
      use consta
      use stream
      implicit none
!
      private
      character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
! code based loosely on ftsm_voronoi and ftsm_rex
!
      logical, save, public :: ftsm_connect_initialized=.false.
      real(chm_real), save, pointer, dimension (:,:,:) :: rall_f, & ! forcing (rall`s will not always be associated so beware)
     & rall_o ! orientation
!
      integer, dimension(:), pointer :: ftsm_connect_map ! holds map between original replica index and current index
      type (int_vlist), save, public :: ftsm_connect_log
!
      real(chm_real), save, private :: ftsm_connect_dist=-one
      real(chm_real), save, private :: ftsm_connect_itemp, ftsm_connect_dtemp, ftsm_connect_ftemp
      integer, save, private :: ftsm_connect_nmove
      integer, parameter, private :: ftsm_connect_nmove_default=1000 ;
      logical, save, private :: ftsm_connect_gettemp
!
!=================================================================================
! SUBROUTINES
      public ftsm_reconnect_parse
      public ftsm_reconnect_init
      public ftsm_reconnect
      public ftsm_reconnect_done
      public ftsm_reconnect_print_map
      public ftsm_reconnect_read_map
      public ftsm_reconnect_print_log
!
      contains
!=================================================================================
       subroutine ftsm_reconnect_parse(comlyn, comlen)
       use string
       character(len=*) :: comlyn
       integer :: comlen
       character(len=20) :: keyword
       logical :: qroot, qprint
       integer :: ifile, flen, i

       character(len=80) :: fname
       integer :: ierror

       character(len=len("FTSM_RECONNECT_PARSE>") ),parameter::whoami="FTSM_RECONNECT_PARSE>";!macro
!
       qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
       qprint=qroot.and.ME_STRNG.eq.0
!
       keyword=nexta8(comlyn,comlen)
!=================================================================================
       if ( ( keyword(1:4).eq.'INIT'(1:4) ) ) then
        call ftsm_reconnect_init(comlyn, comlen)
        return
       endif
!=================================================================================
       if ( .not. ftsm_connect_initialized ) then
        call ftsm_reconnect_init(comlyn, comlen)
       endif
!=================================================================================
       if ( ( keyword(1:4).eq.'DONE'(1:4) ) ) then
        if (ftsm_connect_initialized) call ftsm_reconnect_done()
        call ftsm_reconnect_init(comlyn, comlen)
       elseif ( ( keyword(1:4).eq.'RMAP'(1:4) ) ) then
!
! write cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        if (indxa(comlyn, comlen, 'PRIN').gt.0) then
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (qroot) then
           if (flen.gt.0) then
            if (qprint) then
             call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6011) whoami, fname(1:flen)
             write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6011 format(A,' WRITING FTSM PATH CONNECTIVITY MAP TO FILE ',A,'.')
             call ftsm_reconnect_print_map(ifile)
             call VCLOSE(ifile, 'KEEP', ierror)
            endif ! qprint
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
          endif ! qroot
! read ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'READ').gt.0) then
          ifile=gtrmi(comlyn, comlen, 'UNIT', -1)
          call gtrmwa(COMLYN, COMLEN, 'NAME', 4, FNAME, 80, FLEN)
! note: FNAME will be UPPER CASE
          if (flen.gt.0) then
            if (qprint) then
            call open_file(ifile, fname, 'FORMATTED', 'WRITE')
             write(info,6013) whoami, fname(1:flen) ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
            endif
!
 6013 format(A,' READING FTSM PATH CONNECTIVITY MAP FROM FILE ',A,'.')
            call ftsm_reconnect_read_map(ifile)
            if (qprint) then ; call VCLOSE(ifile, 'KEEP', ierror) ; endif
           else
            call wrndie(0,whoami,trim('FILE NAME NOT SPECIFIED. NOTHING DONE.'))
           endif ! flen
! clear ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        elseif (indxa(comlyn, comlen, 'CLEA').gt.0) then
         if (qprint) then
          write(info,6012) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6012 format(A,' RESETTING FTSM PATH CONNECTIVITY MAP.');
         endif
         if (associated(ftsm_connect_map)) ftsm_connect_map=(/(i,i=1,nstring)/)
        endif ! PRINT
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
       else ! assume initialization
        if (qprint) then
          write(info,6014) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info='';
 6014 format(A,' REINITIALIZING FTSM PATH RECONNECTION.');
        endif
        call ftsm_reconnect_init(comlyn, comlen)
       endif
!
       end subroutine ftsm_reconnect_parse
!=================================================================================
       subroutine ftsm_reconnect_init(comlyn, comlen)
       use string
       character(len=*) :: comlyn
       integer :: comlen
       integer :: i
       logical :: qprint, qroot
       character(len=len("FTSM_RECONNECT_INIT>") ),parameter::whoami="FTSM_RECONNECT_INIT>";!macro
!
       qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
       qprint=qroot.and.ME_STRNG.eq.0
!
       if (.not.ftsm_initialized) then
        call wrndie(0,whoami,trim('FTSM NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (ftsm_connect_initialized) call ftsm_reconnect_done()
!
       ftsm_connect_itemp=gtrmf(comlyn, comlen, 'ITEM', -abs(anum))
       ftsm_connect_ftemp=gtrmf(comlyn, comlen, 'FTEM', -abs(anum))
       ftsm_connect_dtemp=gtrmf(comlyn, comlen, 'DTEM', -abs(anum)) ! dtemp negated prior to tsp (it is a decrement)
       ftsm_connect_nmove=gtrmi(comlyn, comlen, 'NMOV', ftsm_connect_nmove_default)
! print summary
       if (qprint) then
        write(info(1),99) whoami
 99 format(A, ' WILL RECONNECT PATH OPTIMALLY USING MONTE CARLO SIMULATED ANNEALING.')
        if (ftsm_connect_itemp .ge. zero ) then
         write(info(2),100) whoami, 'SET TO '//ftoa(ftsm_connect_itemp)
         ftsm_connect_gettemp=.false.
        else
         write(info(2),100) whoami, 'WILL BE SET EMPIRICALLY.'
         ftsm_connect_gettemp=.true.
        endif
 100 format(A,' INITIAL REFERENCE DISTANCE IN PATH OPTIMIZATION BY MCSA ',A)
!====================================================================
        if (ftsm_connect_ftemp .ge. zero ) then
         write(info(3),101) whoami, 'SET TO '//ftoa(ftsm_connect_ftemp)
        else
         write(info(3),101) whoami, 'WILL BE SET TO ALGORITHM DEFAULT.'
        endif
 101 format(A,' FINAL REFERENCE DISTANCE IN PATH OPTIMIZATION BY MCSA ',A)
!=====================================================================
        if (ftsm_connect_dtemp .ge. zero ) then
         write(info(4),102) whoami, 'SET TO '//ftoa(ftsm_connect_dtemp)
        else
         write(info(4),102) whoami, 'WILL BE COMPUTED AUTOMATICALLY.'
        endif
        ftsm_connect_dtemp=-ftsm_connect_dtemp
 102 format(A,' REFERENCE DISTANCE DECREMENT IN PATH OPTIMIZATION BY MCSA ',A)
!=====================================================================
         write(info(5),103) whoami, ftsm_connect_nmove
 103 format(A,' NUMBER OF ITERATIONS IN PATH OPTIMIZATION BY MCSA SET TO ',I9)
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
! allocate data for all replicas, if not already allocated elsewhere (voronoi is not the only module that is expected to use rall)
!
       if (.not. associated(rall_f)) allocate(rall_f(nforced,3,nstring))
       if (qorient) then
         if (qdiffrot) then
          if (.not. associated(rall_o)) allocate(rall_o(norient,3,nstring))
         else
          rall_o =>rall_f
         endif !qdiffrot
       endif ! qorient
!
       allocate(ftsm_connect_map(nstring));
       ftsm_connect_map=(/(i, i=1,nstring)/)
       call int_vlist_reinit(ftsm_connect_log);
!
       ftsm_connect_initialized=.true. ! this must be set because it affects the behavior of voronoi_update
!
       end subroutine ftsm_reconnect_init
!=================================================================================
       subroutine ftsm_reconnect(itime)
       use mpi
       use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
       use clcg_mod, only: random; use reawri, only: iseed
       use parallel, only: psnd4, psnd8
!
       integer :: itime
       logical :: qstring, qgrp
       logical :: ready(1:2*nstring) ! test if communication completed for forcing and orientation atoms
       real(chm_real), pointer, dimension(:,:) :: rf, ro, rf2
       real(chm_real), dimension(3,3,nstring) :: u ! rotation matrices
       real(chm_real), dimension(nstring,nstring) :: dmat ! distance matrix
       integer :: i, j, k, ind
       integer, pointer :: newroute(:)
       integer :: oldroute(nstring)
       type (int_vector), pointer :: list
       real(chm_real) :: d
!
       integer*4 :: ierror
       integer*4, dimension(0:2*nstring-1) :: srequest, rrequest
       integer*4 :: stat(MPI_STATUS_SIZE)
       real*8 :: path_len(2) ! for determining nore index of optimap path
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
       character(len=len("FTSM_RECONNECT>") ),parameter::whoami="FTSM_RECONNECT>";!macro
!
       if (.not.ftsm_connect_initialized) then
        call wrndie(0,whoami,trim(' FTSM PATH RECONNECTION NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       srequest=MPI_REQUEST_NULL
       rrequest=MPI_REQUEST_NULL
!
       if (itime.lt.0) then
! current center coordinates
        ro=>r_o(:,:,center)
        rf=>r_f(:,:,center)
! will change evolving coordinates (otherwise evolution will be slow)
       else
        ro=>r_o(:,:,center_new)
        rf=>r_f(:,:,center_new)
       endif
       rf2=>r_f(:,:,scratch)
!
       qstring=(MPI_COMM_STRNG.ne.MPI_COMM_NULL)
       qgrp=(MPI_COMM_LOCAL.ne.MPI_COMM_NULL).and.(SIZE_LOCAL.gt.1)
!
! submit sends and receives to/from all nodes
!
       if (qstring.and.(SIZE_STRNG.gt.1)) then ! only roots exchange coordinates
        do i=0, nstring-1
         if (i.eq.ME_STRNG) then
! request arrays already initialized above -- skip two lines below
! rrequest(ME_STRNG)=MPI_REQUEST_NULL ! make sure mpi_waitany does not consider self-communications
! if (qorient.and.qdiffrot) rrequest(ME_STRNG+nstring)=MPI_REQUEST_NULL ! orientation atoms
          cycle ! skip self
         endif
!
         if (qorient) then ! send orientation atoms first
          call MPI_ISEND(ro, 3*norient, mpifloat, i, ME_STRNG, MPI_COMM_STRNG, srequest(i), ierror)
          call MPI_IRECV(rall_o(:,:,i+1), 3*norient, mpifloat, i, i, MPI_COMM_STRNG, rrequest(i), ierror)
!
          if (qdiffrot) then ! send forcing atoms next (with incremented request counters)
           call MPI_ISEND(rf, 3*nforced, mpifloat, i, ME_STRNG+nstring, MPI_COMM_STRNG, srequest(i+nstring), ierror)
           call MPI_IRECV(rall_f(:,:,i+1), 3*nforced, mpifloat, i, i+nstring, MPI_COMM_STRNG, rrequest(i+nstring), ierror)
          endif ! do nothing if .not.qdiffrot
         else ! not qorient : send forcing atoms only
          call MPI_ISEND(rf, 3*nforced, mpifloat, i, ME_STRNG+nstring, MPI_COMM_STRNG, srequest(i), ierror)
          call MPI_IRECV(rall_f(:,:,i+1), 3*nforced, mpifloat, i, i+nstring, MPI_COMM_STRNG, rrequest(i), ierror)
         endif
!
        enddo
! receive and process messages
        ready=.false.
        ready(ME_STRNG+1)=.true. ; ready(ME_STRNG+1+nstring)=.true. ! do not wait for (nonexistent) self-communications
        if (.not.(qorient.and.qdiffrot)) then ! if .not. qorient / .not. qdiffrot, only the first nstring elements are relevant
         ready(nstring+1:2*nstring)=.true. ! forcing atoms the same as orientation atoms
         k=nstring
        else
         k=2*nstring
        endif
!
        do while (any(.not.ready(1:k)))
         call MPI_WAITANY(k, rrequest, ind, stat, ierror) ! note that mpi does not know that rrequest is indexed from 0 not 1 (see above)
         ready(ind)=.true.
         if (qorient) then
          if (ind.le.nstring) then ! orientation atoms received
! NOTE : assuming here that COM has been subtracted, compute rotation matrix
           call RMSBestFit(ro, rall_o(:,:,ind), orientWeights, u(:,:,ind))
          endif
         endif
! determine if can compute metric
         j=mod(ind-1,nstring)+1
         if (ready(j).and.ready(j+nstring)) then ! both sets are ready ( if !qdiffrot or !qorient ready(j+nstring) always true)
          if (qorient) then ! optimally rotate received forcing atoms
           rf2 = matmul(rall_f(:,:,j),u(:,:,j))
          endif
! compute RMSD
          dmat(j,ME_STRNG+1) = rmsd(rf2, rf, forcedWeights);
         endif ! ready
        enddo ! any(not ready)
! gather distance matrix
! in principle, we could use the same asynchronous strategy as above, but the benefits will be much less,
! at the expense of longer code. Instead, will use a simple allgather
        dmat(ME_STRNG+1, ME_STRNG+1)=0 ! self distances
        call mpi_allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &
& dmat, nstring, mpifloat, MPI_COMM_STRNG, ierror)
       endif ! qstring
! send distance matrix to slaves
       if (qgrp) then
#if (KEY_SINGLE==1)
        call PSND4r(dmat,nstring**2) 
#endif
#if (KEY_SINGLE==0)
        call PSND8m(dmat,nstring**2) 
#endif
       endif
! compute initial distance before optimization
       oldroute=(/(i,i=1,nstring)/)
       ftsm_connect_dist=tsp_path_len(oldroute, dmat, nstring)
       if (qstring.and..not.string_noprint.and.ME_STRNG.eq.0) then
        write(info(1),'(A," UNOPTIMIZED PATH LENGTH IS ",'//real_format//')' ) whoami, ftsm_connect_dist
       endif
!
! call tsp solver on all cores
! one preliminary : if the random seeds are equal on all cpus, then all cpus are doing identical work
! therefore, here I will call the RNG a number of times that depends on the rank to randomize the outcomes
! this does not guarantee that the cpus are doing different work, but it is a compromise that does not
! require resetting the seeds explicitly; the clcg generator allows reading from many random number streams,
! but CHARMM is hardwired to read only from the first stream, regardless of the parameter passed to random(<g>)
       do i=1, ME_GLOBAL * 10
        d=random(iseed)
       enddo
!
       newroute=>tsp_anneal_path(dmat, oldroute, &
& ftsm_connect_nmove, ftsm_connect_itemp, ftsm_connect_ftemp, ftsm_connect_dtemp, ftsm_connect_gettemp)
! compute new distance
       if (.not. associated(newroute)) then
        call wrndie(0,whoami,trim('COULD NOT OPTIMIZE PATH. NULL POINTER RECEIVED. ABORT.'))
        return
       endif
!
       ftsm_connect_dist=tsp_path_len(newroute, dmat, nstring)
       path_len(1)=ftsm_connect_dist ; path_len(2) = dble(ME_LOCAL)
       if (qgrp) then
! determine local node with lowest dstance path
        call mpi_allreduce(MPI_IN_PLACE, path_len, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_LOCAL, ierror)
! send optimal route to root
        if (INT(path_len(2)).gt.0) then ! otherwise broadcasting from 0 to 0
         if (ME_LOCAL .eq. INT(path_len(2))) then ; call mpi_send(newroute, nstring, mpiint, 0, 1, MPI_COMM_LOCAL, ierror)
         elseif (ME_LOCAL.eq.0) then ; call mpi_recv(newroute, nstring, mpiint, INT(path_len(2)), 1, MPI_COMM_LOCAL, stat, ierror)
         endif
        endif
       endif
! gather within string communicator
       if (qstring.and.SIZE_STRNG.gt.1) then
        path_len(2)=ME_STRNG
        call mpi_allreduce(MPI_IN_PLACE, path_len, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_STRNG, ierror)
! broadcast optimal route within local group
        call mpi_bcast(newroute, nstring, mpiint, INT(path_len(2)), MPI_COMM_STRNG, ierror)
       endif
! broadcast to local cpus
       if (qgrp) then
        call mpi_bcast(newroute, nstring, mpiint, 0, MPI_COMM_LOCAL, ierror)
       endif
!
! update distance and route map
!
       ftsm_connect_dist=path_len(1)
       ftsm_connect_map=ftsm_connect_map(newroute);
!
! update coordinates
!
       ind=newroute(mestring+1)
       if (ind.ne.mestring+1) then ! only if coordinates have changed
        rf=rall_f(:,:,ind)
        if (qorient.and.qdiffrot) ro=rall_o(:,:,ind)
       endif
!
       if (qstring.and..not.string_noprint.and.ME_STRNG.eq.0) then
        write(info(2),'(A," OPTIMIZED PATH LENGTH IS ",'//real_format//')' ) whoami, ftsm_connect_dist
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
!
! add new route to log
       if (qstring) then
        i=int_vlist_add(ftsm_connect_log, itime)
        list=>ftsm_connect_log%v(i)
        do i=1, nstring ! faster to add elements via a pointer to list
         j=int_vector_add(list, ftsm_connect_map(i)-1) ! output with 0-based offsets
        enddo
       endif
!
       if(associated(newroute))deallocate(newroute)
!
       end subroutine ftsm_reconnect
!=================================================================================
       subroutine ftsm_reconnect_print_log(iunit,fmt)
      use stream
       integer :: iunit
       character(len=*), optional :: fmt
! local
       integer :: j, itime, llen
       integer, pointer :: list(:)
       character(80) :: frm
       character(len=len("FTSM_RECONNECT_PRINT_LOG>") ),parameter::whoami="FTSM_RECONNECT_PRINT_LOG>";!macro
!
! begin
!
       if (.not.ftsm_connect_initialized) then
        call wrndie(0,whoami,trim(' FTSM PATH RECONNECTION NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0) then ! only root prints
        if (.not.present(fmt)) then
         write(frm,'("(I10,",I5,"I5)")') nstring
        else
         frm=fmt
        endif
        do j=1, ftsm_connect_log%last
         itime=ftsm_connect_log%i(j) ! list label
         list=>ftsm_connect_log%v(j)%i ! data vector
         llen=ftsm_connect_log%v(j)%last ! must equal nstring, but just to be safe and avoid a segfault
         write(iunit,frm) itime, list(1:llen)
        enddo
!
       endif ! root
!
       call int_vlist_reinit(ftsm_connect_log) ! erase and reinitialize log
!
       end subroutine ftsm_reconnect_print_log
!=================================================================================
       subroutine ftsm_reconnect_done()
!
        if(associated(ftsm_connect_map))deallocate(ftsm_connect_map)
        call int_vlist_done(ftsm_connect_log)
        if(associated(rall_f))deallocate(rall_f)
        if(associated(rall_o))deallocate(rall_o)
        ftsm_connect_initialized=.false.
!
       end subroutine ftsm_reconnect_done
!=================================================================================
       subroutine ftsm_reconnect_print_map(iunit,fmt)
      use stream
! only root process should call
       integer :: iunit
       character(len=*), optional :: fmt
! local
       integer :: i
       character(80) :: frm
       character(len=len("FTSM_RECONNECT_PRINT_MAP>") ),parameter::whoami="FTSM_RECONNECT_PRINT_MAP>";!macro
! begin
       if (.not.ftsm_connect_initialized) then
        call wrndie(0,whoami,trim(' FTSM PATH RECONNECTION NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (.not.present(fmt)) then
        write(frm,'("(",I5,"I5)")') nstring
       else
        frm=fmt
       endif
       write(iunit,frm) (/ (i, i=0,nstring-1) /)
       write(iunit,frm) ftsm_connect_map(1:nstring)-1 ! 0 - offset
       end subroutine ftsm_reconnect_print_map
!===================================================================================================
       subroutine ftsm_reconnect_read_map(iunit)
      use stream
      use multicom_aux;
      use mpi
      use parallel, only: psnd4, psnd8
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
       integer :: iunit, ierror
       character(len=len("FTSM_RECONNECT_READ_MAP>") ),parameter::whoami="FTSM_RECONNECT_READ_MAP>";!macro
! begin
!
       if (.not.ftsm_connect_initialized) then
        call wrndie(0,whoami,trim(' FTSM PATH RECONNECTION NOT INITIALIZED. NOTHING DONE.'))
        return
       endif
!
       if (MPI_COMM_STRNG.ne.MPI_COMM_NULL) then
        if (ME_STRNG.eq.0) then
         read(iunit,*) ftsm_connect_map(1:nstring) ! first row contains indices 0 -- nstring-1
         read(iunit,*) ftsm_connect_map(1:nstring) ! second row is what we want
         ftsm_connect_map=ftsm_connect_map+(1); ! output with 0-offset, but we use 1-offset
         if (any(ftsm_connect_map.lt.0)) call wrndie(0,whoami,trim('READ NEGATIVE RANK.'))
        endif ! ME_
        if (SIZE_STRNG.gt.1) call mpi_bcast(ftsm_connect_map,nstring,mpiint,0,MPI_COMM_STRNG,ierror)
       endif ! MPI_COMM
! broadcast to slave nodes
       if (ME_LOCAL.ne.MPI_UNDEFINED.and.SIZE_LOCAL.gt.1) &
! & call MPI_BCAST(cv%rex_map, nstring, MPI_INTEGER,
! & 0,MPI_COMM_LOCAL,ierr)
#if (KEY_INTEGER8==0)
     & call PSND4(ftsm_connect_map,nstring) 
#endif
#if (KEY_INTEGER8==1)
     & call PSND8(ftsm_connect_map,nstring) 
#endif
!
       end subroutine ftsm_reconnect_read_map
!==================================================================================================
!
#endif
#endif /* automatically protect all code */
      end module ftsm_connect
!
