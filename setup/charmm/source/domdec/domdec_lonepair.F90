module domdec_lonepair

#if KEY_DOMDEC==1
#if KEY_LONEPAIR==1
  use chm_kinds
  use nblist_types,only:intarray_t
  use hash,only:hashtable_t
  implicit none
  private

  ! true if lone pairs are used
  logical q_lonepr

  ! numlp saved from the call to init_lonepr
  integer numlp_save

  ! Start of the lonepr list for each atom lonepr_start(1:natom)
  integer, allocatable, dimension(:) :: loneprind_start
  ! List of lone pair indices (ilp) loneprind(1:numlp)
  ! Indices for atom i are in loneprind(loneprind_start(i)+1:loneprind_start(i+1))
  integer, allocatable, dimension(:) :: loneprind

  ! Number of lone pairs that this node takes care of
  integer nloneprlist
  ! List of lone pairs that this node takes care of
  integer, allocatable, dimension(:) :: loneprlist

  ! -----------------------------------------------------
  ! Communications
  ! -----------------------------------------------------
  ! Number of atoms to send and receive
  integer nsend(2,3)
  integer nrecv(2,3)
  ! Atom indices to send and receive
  type(intarray_t) send_ind(2,3)
  type(intarray_t) recv_ind(2,3)
  ! Nodes for sending and receiving
  integer send_node(2,3)
  integer recv_node(2,3)
  ! Communication buffer for sending and receiving
  real(chm_real), pointer, dimension(:) :: sendbuf
  real(chm_real), pointer, dimension(:) :: recvbuf
  ! List of required atom indices
  integer, allocatable, dimension(:) :: reqind
  ! Hash table for storing requested atoms
  type(hashtable_t) hashtable
  ! -----------------------------------------------------

  ! Public subroutines
  public init_lonepr, uninit_lonepr
  public check_lonepr_total, check_lonepr

  public setup_lonepr_comm
  public lonepr_comm_coord, lonepr_zero_force, lonepr_comm_force

  ! Public variables
  public q_lonepr
  public nloneprlist, loneprlist

#endif
#endif

contains

#if KEY_DOMDEC==1
#if KEY_LONEPAIR==1
  ! *
  ! * Setups lone pair communications
  ! *
  subroutine setup_lonepr_comm(numlp, lpnhost, lphptr, lphost)
    use stream,only:outu
    use memory
    use hash,only:get_hash, set_hash, and_hash, or_hash, reinit_hash, set_or_hash
    use domdec_common,only:homezone, nx, ny, nz, q_test, homeix, homeiy, homeiz, atoml, natoml
    use mpi,only:mpi_success, mpi_integer, mpi_status_ignore
    use parallel,only:comm_charmm, mynod
    implicit none
    ! Input / Output
    integer, intent(in) :: numlp, lpnhost(:), lphptr(:), lphost(:)
    ! Parameters
    integer, parameter :: COORD_TAG = 1, NCOORD_TAG = 2
    ! Variables
    integer ii, ilp, ipt, n, ind(9), d, dir, numdir, nc, dummy, i, j, sendtmp(2), recvtmp(2)
    integer nreqind, nreqind_val, nreq_this, nreq(2,3), nreq0(2,3), reqpos(2,3), pos, val
    integer nsend0, nrecv0(2,3), len, ierror, bit_mask
    logical ok

    ! Build loneprlist
    nloneprlist = 0
    if (numlp < natoml) then
       do ilp=1,numlp
          ipt = lphptr(ilp)
          if (homezone(lphost(ipt)) == 1) then
             ! Add lone pair to the list
             nloneprlist = nloneprlist + 1
             if (size(loneprlist) < nloneprlist) then
                call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','loneprlist',&
                     int(nloneprlist*1.5),intg=loneprlist)
             endif
             loneprlist(nloneprlist) = ilp
          endif
       enddo
    else
       do i=1,natoml
          ii = atoml(i)
          do j=loneprind_start(ii)+1,loneprind_start(ii+1)
             ilp = loneprind(j)
             ipt = lphptr(ilp)
             if (homezone(lphost(ipt)) == 1) then
                ! Add lone pair to the list
                nloneprlist = nloneprlist + 1
                if (size(loneprlist) < nloneprlist) then
                   call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','loneprlist',&
                        int(nloneprlist*1.5),intg=loneprlist)
                endif
                loneprlist(nloneprlist) = ilp
             endif
          enddo
       enddo
    endif

    ! Loop through lone pair list and build the requested atoms into the hashtable
    ! NOTE: this should be looping over the atom list (atoml), if there are less atoms per node
    !       than lone pairs
    nreqind = 0
    nloneprlist = 0
    do ilp=1,numlp
       ipt = lphptr(ilp)
       n = lpnhost(ilp)
       ind(1:n+1) = lphost(ipt:ipt+n)
       if (homezone(ind(1)) == 1) then
          ! Add lone pair to the list
          nloneprlist = nloneprlist + 1
          if (size(loneprlist) < nloneprlist) then
             call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','loneprlist',&
                  int(nloneprlist*1.5),intg=loneprlist)
          endif
          loneprlist(nloneprlist) = ilp
          if (any(homezone(ind(1:n+1)) /= 1)) then
             ! Lone pair is in this node, but this node does not contain all the host atoms
             ! Request host atoms this node does not have
             ! NOTE: lone pairs can share host atoms, therefore we have to check if this host atom
             !       has already been requested
             bit_mask = b'1'
             do i=2,n+1
                if (homezone(ind(i)) /= 1 .and. .not.get_hash(hashtable, ind(i), dummy)) then
                   nreqind = nreqind + 1
                   ! Re-allocate reqind if needed
                   if (size(reqind) < nreqind) then
                      call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','reqind',&
                           int(nreqind*1.5),intg=reqind)
                   endif
                   reqind(nreqind) = ind(i)
                   ! Mark requested atoms by setting 1st bit
                   call set_hash(hashtable, ind(i), bit_mask)
                endif
             enddo
          endif
       endif
    enddo

    ! nreq_this = number of atoms this node requests
    nreq_this = nreqind

    ! reqind(1:nreq_this) = atom indices that this node requires

    ! Communicate to get the total list of required atoms
    nreq = 0
    nreq0 = 0
    do d=3,1,-1
       if (d == 1) then
          nc = nx
       elseif (d == 2) then
          nc = ny
       else
          nc = nz
       endif
       numdir = 2
       if (nc == 1) then
          numdir = 0
       elseif (nc == 2) then
          numdir = 1
       endif
       ! Value that we send
       nreqind_val = nreqind
       do dir=1,numdir
          ! Send and receive the number of requested atom
          sendtmp(1:2) = (/ nreqind_val, nreq_this /)
          call mpi_sendrecv(sendtmp, 2, mpi_integer, recv_node(dir,d), NCOORD_TAG,&
               recvtmp, 2, mpi_integer, send_node(dir,d), NCOORD_TAG, &
               comm_charmm, mpi_status_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_sendrecv for atom numbers')
          endif
          nreq(dir,d) = recvtmp(1)
          nreq0(dir,d) = recvtmp(2)
          ! Re-allocate reqind if needed
          if (size(reqind) < nreqind + max(1,nreq(dir,d))) then
             call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','reqind',&
                  int((nreqind + max(1,nreq(dir,d)))*1.5),intg=reqind)
          endif
          ! Send and receive the requested atom indices
          call mpi_sendrecv(reqind, nreqind_val, mpi_integer, recv_node(dir,d), COORD_TAG,&
               reqind(nreqind+1), nreq(dir,d), mpi_integer, send_node(dir,d), COORD_TAG, &
               comm_charmm, mpi_status_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_sendrecv for atom indices')
          endif
          ! Store position where required atoms from other nodes are stored
          reqpos(dir,d) = nreqind
          ! Add the received atoms to the total send count
          nreqind = nreqind + nreq(dir,d)
       enddo
    enddo

    ! send_ind
    do d=1,3
       do dir=1,2
          ! Assume that half of the required atoms are on this node
          len = min(20,nreq(dir,d)/2)
          if (allocated(send_ind(dir,d)%array)) then
             if (size(send_ind(dir,d)%array) < len) then
                call chmdealloc('domdec_lonepair.src','setup_lonepr_comm','send_ind',&
                     size(send_ind(dir,d)%array),intg=send_ind(dir,d)%array)
             endif
          endif
          if (.not.allocated(send_ind(dir,d)%array)) then
             call chmalloc('domdec_lonepair.src','setup_lonepr_comm','send_ind',&
                  len,intg=send_ind(dir,d)%array)
          endif
       enddo
    enddo

    ! Now we have the list of requested atoms in reqind(1:nreqind)
    nsend = 0
    nrecv = 0
    nrecv0 = 0
    do d=1,3
       if (d == 1) then
          nc = nx
       elseif (d == 2) then
          nc = ny
       else
          nc = nz
       endif
       numdir = 2
       if (nc == 1) then
          numdir = 0
       elseif (nc == 2) then
          numdir = 1
       endif
       do dir=numdir,1,-1
          pos = reqpos(dir,d)
          ! requested atoms are in:
          ! reqind(pos+1:pos+nreq(dir,d))
          !
          ! Atoms that only node (dir,d) requested are in:
          ! reqind(pos+1:pos+nreq0(dir,d))
          nsend0 = 0
          bit_mask = b'100'
          do i=1,nreq(dir,d)
             val = 0
             ok = get_hash(hashtable, reqind(pos+i), val)
             if ((homezone(reqind(pos+i)) == 1) .or. (iand(val,b'10') == b'10')) then
                ! We have this atom or it was received from other nodes
                if (i <= nreq0(dir,d) .or. .not.ok .or. iand(val,b'100') /= b'100') then
                   ! Node (dir,d) requested the atom, or the atom has not been sent yet
                   ! => add it to the send list
                   nsend(dir,d) = nsend(dir,d) + 1
                   ! Re-allocate send_ind if needed
                   if (size(send_ind(dir,d)%array) < nsend(dir,d)) then
                      call chmrealloc('domdec_lonepair.src','setup_lonepr_comm','send_ind',&
                           int(nsend(dir,d)*1.5),intg=send_ind(dir,d)%array)
                   endif
                   send_ind(dir,d)%array(nsend(dir,d)) = reqind(pos+i)
                   ! Mark atoms that have been sent by value setting 3rd bit
                   if (ok) then
                      call or_hash(hashtable, reqind(pos+i), bit_mask)
                   else
                      call set_hash(hashtable, reqind(pos+i), bit_mask)
                   endif
                   ! Count the number of atoms node (dir,d) requested
                   if (i <= nreq0(dir,d)) then
                      nsend0 = nsend0 + 1
                   endif
                endif
             endif
          enddo
          ! Clear hash table 3rd bit
          bit_mask = b'11'
          do i=1,nsend(dir,d)
             call and_hash(hashtable, send_ind(dir,d)%array(i), bit_mask)
          enddo
          ! Send and receive the number of atoms
          sendtmp(1:2) = (/ nsend(dir,d), nsend0 /)
          call mpi_sendrecv(sendtmp, 2, mpi_integer, send_node(dir,d), NCOORD_TAG, &
               recvtmp, 2, mpi_integer, recv_node(dir,d), NCOORD_TAG, &
               comm_charmm, mpi_status_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_sendrecv for atom numbers (2)')
          endif
          nrecv(dir,d) = recvtmp(1)
          nrecv0(dir,d) = recvtmp(2)
          ! Re-allocate recv_ind if needed
          if (allocated(recv_ind(dir,d)%array)) then
             if (size(recv_ind(dir,d)%array) < nrecv(dir,d)) then
                call chmdealloc('domdec_lonepair.src','setup_lonepr_comm','recv_ind',&
                     size(recv_ind(dir,d)%array),intg=recv_ind(dir,d)%array)
             endif
          endif
          if (.not.allocated(recv_ind(dir,d)%array)) then
             call chmalloc('domdec_lonepair.src','setup_lonepr_comm','recv_ind',&
                  int(nrecv(dir,d)*1.5)+1,intg=recv_ind(dir,d)%array)
          endif
          ! Send and receive the atom indices
          call mpi_sendrecv(send_ind(dir,d)%array, nsend(dir,d), mpi_integer, send_node(dir,d), &
               NCOORD_TAG, &
               recv_ind(dir,d)%array, nrecv(dir,d), mpi_integer, recv_node(dir,d), NCOORD_TAG, &
               comm_charmm, mpi_status_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_sendrecv for atom indices (2)')
          endif          
       enddo
       ! Mark the atoms that were received from other nodes by setting the 2nd bit
       bit_mask = b'10'
       do dir=1,numdir
          do i=1,nrecv(dir,d)
             call set_or_hash(hashtable, recv_ind(dir,d)%array(i), bit_mask)
          enddo
       enddo
    enddo

    ! Check that the number of received atoms matches the original requested count
    if (nreq_this /= sum(nrecv0)) then
       write (outu,'(a,3i3)') 'mynod,nreq_this,sum(nrecv0)=',mynod,nreq_this,sum(nrecv0)
       call wrndie(-5,'<domdec_lonepair>','Original requested atom count not matched')
    endif

    ! Additional test checks that we are receiving the correct atom indices
    if (q_test) then
       call test_setup_lonepr_comm(nreq_this, reqind)
    endif

    ! Optimizes (and clears) the hash
    call reinit_hash(hashtable)

    !-----------------------
    ! Do memory (re-)allocation
    !-----------------------

    ! sendbuf
    len = 0
    do d=1,3
       len = max(len, sum(nsend(1:2,d))*3)
    enddo
    if (associated(sendbuf)) then
       if (size(sendbuf) < len) then
          call chmdealloc('domdec_lonepair.src','setup_lonepr_comm','sendbuf',&
               size(sendbuf),mcrlp=sendbuf)
       endif
    endif
    if (.not.associated(sendbuf)) then
       call chmalloc('domdec_lonepair.src','setup_lonepr_comm','sendbuf',&
            len,mcrlp=sendbuf)
    endif

    ! recvbuf
    len = 0
    do d=1,3
       len = max(len, sum(nrecv(1:2,d))*3)
    enddo
    if (associated(recvbuf)) then
       if (size(recvbuf) < len) then
          call chmdealloc('domdec_lonepair.src','setup_lonepr_comm','recvbuf',&
               size(recvbuf),mcrlp=recvbuf)
       endif
    endif
    if (.not.associated(recvbuf)) then
       call chmalloc('domdec_lonepair.src','setup_lonepr_comm','recvbuf',&
            len,mcrlp=recvbuf)
    endif

    return
  end subroutine setup_lonepr_comm

  ! *
  ! * Test for setup_lonepr_comm
  ! *
  subroutine test_setup_lonepr_comm(nreq_this, reqind)
    use stream,only:prnlev, outu
    implicit none
    ! Input
    integer, intent(in) :: nreq_this, reqind(:)
    ! Variables
    integer i
    logical ok

    ! Make sure that we receive all the atoms in reqind(1:nreq_this)
    do i=1,nreq_this
       ok = .false.
       call check()
       if (.not.ok) then
          write (outu,'(a)') 'test_setup_lonepr_comm FAILED'
       endif
    enddo

    if (prnlev > 2) write (outu,'(a)') 'test_setup_lonepr_comm OK'

    return
  contains
    subroutine check
      implicit none
      integer d, dir, j
      do d=1,3
         do dir=1,2
            do j=1,nrecv(dir,d)
               if (recv_ind(dir,d)%array(j) == reqind(i)) then
                  ok = .true.
                  return
               endif
            enddo
         enddo
      enddo
      return
    end subroutine check
  end subroutine test_setup_lonepr_comm
  
  ! *
  ! * Communicate lone pair coordinates
  ! *
  subroutine lonepr_comm_coord(x, y, z)
    use mpi,only:mpi_real8, mpi_success, mpi_statuses_ignore
    use parallel,only:comm_charmm
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    ! Parameters
    integer, parameter :: COORD_TAG = 1
    ! Variables
    integer d, dir, i, j, k
    integer nrequest, request(4)
    integer ierror

    ! Loop through x, y, and z dimensions
    do d=1,3
       ! Start receiving
       nrequest = 0
       j = 1
       do dir=1,2
          if (nrecv(dir,d) > 0) then
             nrequest = nrequest + 1
             call mpi_irecv(recvbuf(j), nrecv(dir,d)*3, mpi_real8, recv_node(dir,d), &
                  COORD_TAG, comm_charmm, request(nrequest), ierror)
             if (ierror /= mpi_success) then
                call wrndie(-5,'<domdec_lonepair>','Error receiving coordinates')
             endif
             j = j + nrecv(dir,d)*3
          endif
       enddo

       ! Copy coordinates into sendbuf
       j = 1
       do dir=1,2
          do k=1,nsend(dir,d)
             i = send_ind(dir,d)%array(k)
             sendbuf(j)   = x(i)
             sendbuf(j+1) = y(i)
             sendbuf(j+2) = z(i)
             j = j + 3
          enddo
       enddo
       
       ! Send coordinates
       j = 1
       do dir=1,2
          if (nsend(dir,d) > 0) then
             nrequest = nrequest + 1
             call mpi_isend(sendbuf(j), nsend(dir,d)*3, mpi_real8, send_node(dir,d), &
                  COORD_TAG, comm_charmm, request(nrequest), ierror)
             if (ierror /= mpi_success) then
                call wrndie(-5,'<domdec_lonepair>','Error sending coordinates')
             endif
             j = j + nsend(dir,d)*3
          endif
       enddo
       
       ! Wait for communcation to finish
       if (nrequest > 0) then
          call mpi_waitall(nrequest, request, mpi_statuses_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_waitall')
          endif
       endif

       ! Copy coordinates into global arrays
       j = 1
       do dir=1,2
          do k=1,nrecv(dir,d)
             i = recv_ind(dir,d)%array(k)
             x(i) = recvbuf(j)
             y(i) = recvbuf(j+1)
             z(i) = recvbuf(j+2)
             j = j + 3
          enddo
       enddo       

    enddo

    return
  end subroutine lonepr_comm_coord

  ! *
  ! * Zeros forces on atoms that are in the recv_ind buffer
  ! * NOTE: Due to a potential compiler optimizer bug in Intel 14 and 15 Fortran
  ! * compilers, we have to input nd and ndir as input parameters to prevent
  ! * the compiler from messing up the code.
  ! *
  subroutine lonepr_zero_force(nd, ndir, forcex, forcey, forcez)
    use number,only:zero
    use parallel,only:mynod
    implicit none
    ! Input / Output
    integer, intent(in) :: nd, ndir
    real(chm_real), intent(out) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer d, dir, i, k

    do d=1,nd
       do dir=1,ndir
          do k=1,nrecv(dir,d)
             i = recv_ind(dir,d)%array(k)
             forcex(i) = zero
             forcey(i) = zero
             forcez(i) = zero
          enddo
       enddo
    enddo

    return
  end subroutine lonepr_zero_force

  ! *
  ! * Communicate lone pair forces
  ! *
  subroutine lonepr_comm_force(forcex, forcey, forcez)
    use mpi,only:mpi_real8, mpi_success, mpi_statuses_ignore
    use parallel,only:comm_charmm
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)
    ! Parameters
    integer, parameter :: COORD_TAG = 1
    ! Variables
    integer d, dir, i, j, k
    integer nrequest, request(4)
    integer ierror

    ! Loop through x, y, and z dimensions
    do d=3,1,-1

       ! Start receiving forces
       nrequest = 0
       j = 1
       do dir=1,2
          if (nsend(dir,d) > 0) then
             nrequest = nrequest + 1
             call mpi_irecv(sendbuf(j), nsend(dir,d)*3, mpi_real8, send_node(dir,d), &
                  COORD_TAG, comm_charmm, request(nrequest), ierror)
             if (ierror /= mpi_success) then
                call wrndie(-5,'<domdec_lonepair>','Error receiving forces')
             endif
             j = j + nsend(dir,d)*3
          endif
       enddo

       ! Copy forces into recvbuf
       j = 1
       do dir=1,2
          do k=1,nrecv(dir,d)
             i = recv_ind(dir,d)%array(k)
             recvbuf(j)   = forcex(i)
             recvbuf(j+1) = forcey(i)
             recvbuf(j+2) = forcez(i)
             j = j + 3
          enddo
       enddo
       
       ! Send forces
       j = 1
       do dir=1,2
          if (nrecv(dir,d) > 0) then
             nrequest = nrequest + 1
             call mpi_isend(recvbuf(j), nrecv(dir,d)*3, mpi_real8, recv_node(dir,d), &
                  COORD_TAG, comm_charmm, request(nrequest), ierror)
             if (ierror /= mpi_success) then
                call wrndie(-5,'<domdec_lonepair>','Error sending forces')
             endif
             j = j + nrecv(dir,d)*3
          endif
       enddo
       
       ! Wait for communcation to finish
       if (nrequest > 0) then
          call mpi_waitall(nrequest, request, mpi_statuses_ignore, ierror)
          if (ierror /= mpi_success) then
             call wrndie(-5,'<domdec_lonepair>','Error in mpi_waitall')
          endif
       endif

       ! Copy forces into global arrays
       j = 1
       do dir=1,2
          do k=1,nsend(dir,d)
             i = send_ind(dir,d)%array(k)
             forcex(i) = forcex(i) + sendbuf(j)
             forcey(i) = forcey(i) + sendbuf(j+1)
             forcez(i) = forcez(i) + sendbuf(j+2)
             j = j + 3
          enddo
       enddo       

    enddo

    return
  end subroutine lonepr_comm_force

  ! *
  ! * Setups loneprind and loneprind_start
  ! *
  subroutine setup_loneprind(numlp, lpnhost, lphptr, lphost)
    use memory,only:chmalloc, chmdealloc
    use nblist_util,only:cumsum_exclusive
    use psf,only:natom
    implicit none
    ! Input
    integer, intent(in) :: numlp, lpnhost(:), lphptr(:), lphost(:)
    ! Variables
    integer ilp, ipt, i
    integer, allocatable, dimension(:) :: pos

    ! Allocate loneprind_start
    if (allocated(loneprind_start)) then
       if (size(loneprind_start) /= natom+1) then
          call chmdealloc('domdec_lonepair.src','setup_loneprind','loneprind_start',&
               size(loneprind_start),intg=loneprind_start)
       endif
    endif
    if (.not.allocated(loneprind_start)) then
       call chmalloc('domdec_lonepair.src','setup_loneprind','loneprind_start',&
            natom+1,intg=loneprind_start)
    endif

    ! Count the number of lone pairs for each atom
    loneprind_start(1:natom+1) = 0
    do ilp=1,numlp
       ipt = lphptr(ilp)
       i = lphost(ipt)
       loneprind_start(i) = loneprind_start(i) + 1
    enddo

    ! Calculate the start for each atom
    call cumsum_exclusive(loneprind_start, natom+1)

    ! Allocate loneprind
    if (allocated(loneprind)) then
       if (size(loneprind) /= loneprind_start(natom+1)) then
          call chmdealloc('domdec_lonepair.src','setup_loneprind','loneprind',&
               size(loneprind),intg=loneprind)
       endif
    endif
    if (.not.allocated(loneprind)) then
       call chmalloc('domdec_lonepair.src','setup_loneprind','loneprind',&
            loneprind_start(natom+1),intg=loneprind)
    endif

    call chmalloc('domdec_lonepair.src','setup_loneprind','pos',&
         natom,intg=pos)

    ! Set lone pair indices into loneprind(loneprind_start(i)+1:loneprind_start(i+1))
    pos(1:natom) = 0
    do ilp=1,numlp
       ipt = lphptr(ilp)
       i = lphost(ipt)
       pos(i) = pos(i) + 1
       loneprind(loneprind_start(i) + pos(i)) = ilp
    enddo

    call chmdealloc('domdec_lonepair.src','setup_loneprind','pos',&
         natom,intg=pos)

    return
  end subroutine setup_loneprind

  ! *
  ! * Initializes lone pairs
  ! *
  subroutine init_lonepr(numlp, lpnhost, lphptr, lphost)
    use memory,only:chmalloc
    use groupxfast,only:invgroup
    use domdec_common,only:nodeindfunc, ndirect, homeix, homeiy, homeiz
    use hash,only:init_hash
    implicit none
    ! Input
    integer, intent(in) :: numlp, lpnhost(:), lphptr(:), lphost(:)
    ! Variables
    integer i, j, d, dir, ipt, n
    integer numlp_group, nkey, loneprlist_len

    numlp_save = numlp

    if (numlp == 0) then
       q_lonepr = .false.
       return
    else
       q_lonepr = .true.
    endif

    ! Allocate loneprlist
    loneprlist_len = max(100, numlp/ndirect)
    if (.not.allocated(loneprlist)) then
       allocate(loneprlist(loneprlist_len))
    endif

    ! Setup and allocate loneprind and loneprind_start
    call setup_loneprind(numlp, lpnhost, lphptr, lphost)

    ! Setup (send_node, recv_node)
    send_node(1,1) = nodeindfunc(homeix+1,homeiy,homeiz)
    send_node(2,1) = nodeindfunc(homeix-1,homeiy,homeiz)
    send_node(1,2) = nodeindfunc(homeix,homeiy+1,homeiz)
    send_node(2,2) = nodeindfunc(homeix,homeiy-1,homeiz)
    send_node(1,3) = nodeindfunc(homeix,homeiy,homeiz+1)
    send_node(2,3) = nodeindfunc(homeix,homeiy,homeiz-1)

    recv_node(1,1) = nodeindfunc(homeix-1,homeiy,homeiz)
    recv_node(2,1) = nodeindfunc(homeix+1,homeiy,homeiz)
    recv_node(1,2) = nodeindfunc(homeix,homeiy-1,homeiz)
    recv_node(2,2) = nodeindfunc(homeix,homeiy+1,homeiz)
    recv_node(1,3) = nodeindfunc(homeix,homeiy,homeiz-1)
    recv_node(2,3) = nodeindfunc(homeix,homeiy,homeiz+1)

    ! Setup hashtable, but first calculate the number of lone pairs that consist of more than
    ! one groups (numlp_group) and use that to estimate the size of the hash table
    numlp_group = 0
    do i=1,numlp
       ipt = lphptr(i)
       n = lpnhost(i)
       if (any(invgroup(lphost(ipt+1:ipt+n)) /= invgroup(lphost(ipt)))) then
          numlp_group = numlp_group + 1
       endif
    enddo
    ! Estimate the number of split groups that would be between two nodes
    nkey = numlp_group/max(8,ndirect)
    call init_hash(hashtable, nkey)

    if (.not.allocated(reqind)) then
       call chmalloc('domdec_lonepair.src','init_lonepr','reqind',&
            nkey,intg=reqind)
    endif

    return
  end subroutine init_lonepr

  ! *
  ! * Uninitializes lone pairs
  ! *
  subroutine uninit_lonepr()
    use memory,only:chmdealloc
    use hash,only:uninit_hash
    implicit none
    integer i, d, dir

    q_lonepr = .false.

    ! loneprlist
    if (allocated(loneprlist)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','send_ind',&
            size(loneprlist),intg=loneprlist)
    endif

    ! loneprind_start
    if (allocated(loneprind_start)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','loneprind_start',&
            size(loneprind_start),intg=loneprind_start)
    endif

    ! loneprind
    if (allocated(loneprind)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','loneprind',&
            size(loneprind),intg=loneprind)
    endif

    ! (send_ind, recv_ind)
    do d=1,3
       do dir=1,2
          if (allocated(send_ind(dir,d)%array)) then
             call chmdealloc('domdec_lonepair.src','uninit_lonepr','send_ind',&
                  size(send_ind(dir,d)%array),intg=send_ind(dir,d)%array)
          endif
          if (allocated(recv_ind(dir,d)%array)) then
             call chmdealloc('domdec_lonepair.src','uninit_lonepr','recv_ind',&
                  size(recv_ind(dir,d)%array),intg=recv_ind(dir,d)%array)
          endif
       enddo
    enddo

    ! sendbuf
    if (associated(sendbuf)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','sendbuf',&
            size(sendbuf),mcrlp=sendbuf)
    endif

    ! recvbuf
    if (associated(recvbuf)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','recvbuf',&
            size(recvbuf),mcrlp=recvbuf)
    endif

    call uninit_hash(hashtable)

    ! reqind
    if (allocated(reqind)) then
       call chmdealloc('domdec_lonepair.src','uninit_lonepr','reqind',&
            size(reqind),intg=reqind)
    endif

    return
  end subroutine uninit_lonepr

  ! *
  ! * Checks that there are correct number of lone pairs
  ! *
  subroutine check_lonepr()
    implicit none
    ! Variables
    integer nlonepr_sum
    integer tbl(1)

    if (.not.q_lonepr) return

    tbl(1) = nloneprlist
    call igcomb(tbl, 1)
    nlonepr_sum = tbl(1)

    call check_lonepr_total(nlonepr_sum)

    return
  end subroutine check_lonepr

  ! *
  ! * Checks the lone pair total number
  ! *
  subroutine check_lonepr_total(nlonepr_sum)
    use stream,only:outu,prnlev
    implicit none
    ! Input
    integer, intent(in) :: nlonepr_sum

    if (.not.q_lonepr) return

    ! Check the numbers:
    if (nlonepr_sum /= numlp_save) then
       if (prnlev > 2) then
          write (outu,'(a,i8)') 'correct: numlp',numlp_save
          write (outu,'(a,i8)') 'actual:  numlp',nlonepr_sum
          call wrndie(-5,'<domdec_lonepair>','Incorrect number of lone pairs')
       endif
    endif
    
    return
  end subroutine check_lonepr_total

#endif
#endif

end module domdec_lonepair
