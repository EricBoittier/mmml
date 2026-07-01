module domdec_d2d_comm

  ! *
  ! * Domain decomposition direct-to-direct communication
  ! *
  ! *

#if KEY_DOMDEC==1 /*domdec_main*/
#if KEY_PARALLEL==1 /*parallel*/
!#if KEY_CMPI==0 /*not_cmpi*/
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  type crlpointer_t
     real(chm_real), pointer :: p(:)
  end type crlpointer_t

  integer, parameter :: COORDBUF=1, FORCEBUF=2

  ! Array to store cumulative positions from threads, 0:nthread
  integer, allocatable, dimension(:) :: cumpos

  ! Send and receive buffers for update_groupl() -routine
  type(crlpointer_t), allocatable, dimension(:) :: neighsendbuf
  real(chm_real), pointer, dimension(:) :: neighrecvbuf
  ! Destination node index 
  integer, allocatable, dimension(:) :: neighind_dest

  ! Indicates if mpi_iprobe has already received this message
  logical, allocatable, dimension(:) :: recv_flag

  integer, allocatable, dimension(:) :: reqbuf

  ! For the new communication method:
  integer, allocatable, dimension(:) :: x_send_node, y_send_node, z_send_node
  integer, allocatable, dimension(:) :: x_recv_node, y_recv_node, z_recv_node
  real(chm_real), allocatable, dimension(:) :: xtmp, ytmp, ztmp

  ! Communication buffers
  integer, allocatable, dimension(:,:) :: x_send_ncoord, y_send_ncoord, z_send_ncoord
  integer, allocatable, dimension(:,:) :: x_send_ngroup, y_send_ngroup, z_send_ngroup

  integer, allocatable, dimension(:) :: x_send_atomind, y_send_atomind, z_send_atomind
  integer, allocatable, dimension(:) :: x_recv_atomind, y_recv_atomind, z_recv_atomind

  integer, allocatable, dimension(:,:) :: x_recv_ncoord, y_recv_ncoord, z_recv_ncoord
  integer, allocatable, dimension(:,:) :: x_recv_ngroup, y_recv_ngroup, z_recv_ngroup

  type(crlpointer_t), allocatable, dimension(:) :: x_send_buf, y_send_buf, z_send_buf
  type(crlpointer_t), allocatable, dimension(:) :: x_recv_buf, y_recv_buf, z_recv_buf

  ! The following are for copy_to_all and copy_to_root routines
  real(chm_real), allocatable, dimension(:) :: send_xbuf, send_ybuf, send_zbuf
  real(chm_real), allocatable, dimension(:) :: recv_xbuf, recv_ybuf, recv_zbuf
  integer, allocatable, dimension(:) :: send_indbuf, recv_indbuf

  ! Buffers for copy_to_all -routines
  integer, allocatable, dimension(:) :: nrecv, disp, disp2, recvcount

  ! Public subroutines
  public transfer_coord, transfer_force, update_groupl, init_d2d_comm, uninit_d2d_comm, &
       copy_to_all, calc_max_atom_group, deallocate_copy
  public test_groupl_atoml

  ! Public Variables

  interface copy_to_all
     module procedure copy_to_all1
     module procedure copy_to_all3
  end interface

contains

  ! *
  ! * Calculate esimate for maximum number of atoms and groups in each subbox
  ! *
  subroutine calc_max_atom_group(max_atom_box, max_group_box)
    use number
    use consta
    use psf,only:natom
    use groupxfast,only:ngroup
    use inbnd,only:cutnb
    use domdec_common,only:boxx, boxy, boxz, frx, fry, frz, set_box
    implicit none
    ! Input / Output
    integer max_atom_box, max_group_box
    ! Variables
    real(chm_real) cut, volim, voltot, volfr
    real(chm_real) sx, sy, sz

    ! Estimate for the actual cut-off
    cut = cutnb + 3.5d0

    call set_box()

    ! Sub-box sizes
    sx = frx*boxx
    sy = fry*boxy
    sz = frz*boxz

    ! Total volume
    voltot = boxx*boxy*boxz
    ! Import space volume
    volim = sx*sy*sz + &         ! zone I
         cut*sy*sz + &           ! zone FX
         cut*sx*sz + &           ! zone FY
         cut*sx*sy + &           ! zone FZ
         sx*fourth*pi*cut**2 + & ! zone EX
         sy*fourth*pi*cut**2 + & ! zone EY
         sz*fourth*pi*cut**2 + & ! zone EZ
         sixth*pi*cut**3         ! zone C

    ! Fraction of volume taken by the import volume
    volfr = volim/voltot

    max_atom_box = min(natom,max(100,int(volfr*natom)))
    max_group_box = min(ngroup,max(100,int(volfr*ngroup)))

!!$    max_atom_box = 8*natom/ndirect
!!$    if (max_atom_box > natom) max_atom_box = natom
!!$
!!$    max_group_box = 8*ngroup/ndirect
!!$    if (max_group_box > ngroup) max_group_box = ngroup

    return
  end subroutine calc_max_atom_group

  ! *
  ! * Initializes communication, reallocates tables if needed
  ! *
  subroutine init_d2d_comm(x, y, z)
    use number,only:zero
    use stream,only:outu
    use memory,only:chmalloc, chmdealloc
    use psf,only:natom
    use parallel,only:mpi_integer_size, mpi_real8_size
    use domdec_dlb,only:init_subboxes
    use domdec_common
    implicit none
    ! Input
    real(chm_real) x(*), y(*), z(*)
    ! Variables
    integer i, ix, iy, iz
    integer max_atom_box, max_grp_box
    integer x_send_buflen, y_send_buflen, z_send_buflen
    integer x_recv_buflen, y_recv_buflen, z_recv_buflen
    integer x_send_atomind_len, y_send_atomind_len, z_send_atomind_len
    integer x_recv_atomind_len, y_recv_atomind_len, z_recv_atomind_len
    integer neighsendbuflen, neighrecvbuflen
    logical q_dealloc

    ! Determine nx_comm, ny_comm, nz_comm
    call init_subboxes()

    ! (xtmp, ytmp, ztmp)
    if (allocated(xtmp)) then
       if (size(xtmp) /= natom) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','xtmp',natom,crl=xtmp)
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','ytmp',natom,crl=ytmp)
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','ztmp',natom,crl=ztmp)
       endif
    endif
    if (.not.allocated(xtmp)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','xtmp',natom,crl=xtmp)
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','ytmp',natom,crl=ytmp)
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','ztmp',natom,crl=ztmp)
    endif

    xtmp(1:natom) = zero
    ytmp(1:natom) = zero
    ztmp(1:natom) = zero

    ! (x_recv_node, y_recv_node, z_recv_node)
    if (allocated(x_recv_node)) then
       if (size(x_recv_node) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_node',size(x_recv_node),&
               intg=x_recv_node)
       endif
    endif
    if (.not.allocated(x_recv_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_node',nx_comm,&
            intg=x_recv_node)
    endif

    if (allocated(y_recv_node)) then
       if (size(y_recv_node) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_node',size(y_recv_node),&
               intg=y_recv_node)
       endif
    endif
    if (.not.allocated(y_recv_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_node',ny_comm,&
            intg=y_recv_node)
    endif

    if (allocated(z_recv_node)) then
       if (size(z_recv_node) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_node',size(z_recv_node),&
               intg=z_recv_node)
       endif
    endif
    if (.not.allocated(z_recv_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_node',nz_comm,&
            intg=z_recv_node)
    endif

    ix = homeix
    do i=1,nx_comm
       ix = ix + 1
       x_recv_node(i) = nodeindfunc(ix,homeiy,homeiz)
    enddo
    iy = homeiy
    do i=1,ny_comm
       iy = iy + 1
       y_recv_node(i) = nodeindfunc(homeix,iy,homeiz)
    enddo
    iz = homeiz
    do i=1,nz_comm
       iz = iz + 1
       z_recv_node(i) = nodeindfunc(homeix,homeiy,iz)
    enddo

    ! (x_send_node, y_send_node, z_send_node)
    if (allocated(x_send_node)) then
       if (size(x_send_node) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_send_node',size(x_send_node),&
               intg=x_send_node)
       endif
    endif
    if (.not.allocated(x_send_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_send_node',nx_comm,&
            intg=x_send_node)
    endif

    if (allocated(y_send_node)) then
       if (size(y_send_node) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_send_node',size(y_send_node),&
               intg=y_send_node)
       endif
    endif
    if (.not.allocated(y_send_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_send_node',ny_comm,&
            intg=y_send_node)
    endif

    if (allocated(z_send_node)) then
       if (size(z_send_node) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_send_node',size(z_send_node),&
               intg=z_send_node)
       endif
    endif
    if (.not.allocated(z_send_node)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_send_node',nz_comm,&
            intg=z_send_node)
    endif

    ix = homeix
    do i=1,nx_comm
       ix = ix - 1
       if (ix == 0) ix = nx
       x_send_node(i) = nodeindfunc(ix,homeiy,homeiz)
    enddo
    iy = homeiy
    do i=1,ny_comm
       iy = iy - 1
       if (iy == 0) iy = ny
       y_send_node(i) = nodeindfunc(homeix,iy,homeiz)
    enddo
    iz = homeiz
    do i=1,nz_comm
       iz = iz - 1
       if (iz == 0) iz = nz
       z_send_node(i) = nodeindfunc(homeix,homeiy,iz)
    enddo
    ! ###########################################################################

    call calc_max_atom_group(max_atom_box, max_grp_box)

    neighsendbuflen = max_atom_box
    neighrecvbuflen = max_atom_box

    z_send_atomind_len = max_atom_box
    y_send_atomind_len = max_atom_box*2
    x_send_atomind_len = max_atom_box*4
    z_recv_atomind_len = max_atom_box
    y_recv_atomind_len = max_atom_box*2
    x_recv_atomind_len = max_atom_box*4

    z_send_buflen = max_atom_box*3
    y_send_buflen = min(max_atom_box*2,natom)*3
    x_send_buflen = min(max_atom_box*4,natom)*3
    z_recv_buflen = max_atom_box*3
    y_recv_buflen = min(max_atom_box*2,natom)*3
    x_recv_buflen = min(max_atom_box*4,natom)*3

    ! recv_flag
    if (allocated(recv_flag)) then
       if (size(recv_flag) /= max(26,nx_comm,ny_comm,nz_comm)) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','recv_flag',&
               size(recv_flag),log=recv_flag)
       endif
    endif
    if (.not.allocated(recv_flag)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','recv_flag',&
            max(26,nx_comm,ny_comm,nz_comm),log=recv_flag)
    endif

    ! reqbuf
    if (allocated(reqbuf)) then
       if (size(reqbuf) /= 2*max(nx_comm,ny_comm,nz_comm)) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','reqbuf',&
               size(reqbuf),intg=reqbuf)
       endif
    endif
    if (.not.allocated(reqbuf)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','reqbuf',&
            2*max(nx_comm,ny_comm,nz_comm),intg=reqbuf)
    endif

    ! neighsendbuf
    if (allocated(neighsendbuf)) then
       q_dealloc = .false.
       if (size(neighsendbuf) /= nneigh) then
          q_dealloc = .true.
       endif
       if (size(neighsendbuf) >= 1) then
          if (size(neighsendbuf(1)%p) /= neighsendbuflen) then
             q_dealloc = .true.
          endif
       endif
       if (q_dealloc) then
          do i=1,size(neighsendbuf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','neighsendbuf',&
                  size(neighsendbuf(i)%p),mcrlp=neighsendbuf(i)%p)
          enddo
          deallocate(neighsendbuf)
       endif
    endif
    if (.not.allocated(neighsendbuf)) then
       allocate(neighsendbuf(nneigh))
       do i=1,nneigh
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','neighsendbuf',&
               neighsendbuflen, mcrlp=neighsendbuf(i)%p)
       enddo
    endif

    ! neighrecvbuf
    if (associated(neighrecvbuf)) then
       if (size(neighrecvbuf) /= neighrecvbuflen) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','neighrecvbuf',size(neighrecvbuf),&
               mcrlp=neighrecvbuf)
       endif
    endif
    if (.not.associated(neighrecvbuf)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','neighrecvbuf',neighrecvbuflen,&
            mcrlp=neighrecvbuf)
    endif

    ! x_send_atomind
    if (allocated(x_send_atomind)) then
       if (size(x_send_atomind) /= x_send_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_send_atomind',&
               size(x_send_atomind),intg=x_send_atomind)
       endif
    endif
    if (.not.allocated(x_send_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_send_atomind',&
            x_send_atomind_len,intg=x_send_atomind)
    endif

    ! y_send_atomind
    if (allocated(y_send_atomind)) then
       if (size(y_send_atomind) /= y_send_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_send_atomind',&
               size(y_send_atomind),intg=y_send_atomind)
       endif
    endif
    if (.not.allocated(y_send_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_send_atomind',&
            y_send_atomind_len,intg=y_send_atomind)
    endif

    ! z_send_atomind
    if (allocated(z_send_atomind)) then
       if (size(z_send_atomind) /= z_send_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_send_atomind',&
               size(z_send_atomind),intg=z_send_atomind)
       endif
    endif
    if (.not.allocated(z_send_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_send_atomind',&
            z_send_atomind_len,intg=z_send_atomind)
    endif

    ! x_recv_atomind
    if (allocated(x_recv_atomind)) then
       if (size(x_recv_atomind) /= x_recv_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_atomind',&
               size(x_recv_atomind),intg=x_recv_atomind)
       endif
    endif
    if (.not.allocated(x_recv_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_atomind',&
            x_recv_atomind_len,intg=x_recv_atomind)
    endif

    ! y_recv_atomind
    if (allocated(y_recv_atomind)) then
       if (size(y_recv_atomind) /= y_recv_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_atomind',&
               size(y_recv_atomind),intg=y_recv_atomind)
       endif
    endif
    if (.not.allocated(y_recv_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_atomind',&
            y_recv_atomind_len,intg=y_recv_atomind)
    endif

    ! z_recv_atomind
    if (allocated(z_recv_atomind)) then
       if (size(z_recv_atomind) /= z_recv_atomind_len) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_atomind',&
               size(z_recv_atomind),intg=z_recv_atomind)
       endif
    endif
    if (.not.allocated(z_recv_atomind)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_atomind',&
            z_recv_atomind_len,intg=z_recv_atomind)
    endif

    ! x_send_ngroup
    if (allocated(x_send_ngroup)) then
       if (size(x_send_ngroup,2) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_send_ngroup',&
               4,size(x_send_ngroup,2),intg=x_send_ngroup)
       endif
    endif
    if (.not.allocated(x_send_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_send_ngroup',4,nx_comm,&
            intg=x_send_ngroup)
    endif

    ! y_send_ngroup
    if (allocated(y_send_ngroup)) then
       if (size(y_send_ngroup,2) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_send_ngroup',&
               2,size(y_send_ngroup,2),intg=y_send_ngroup)
       endif
    endif
    if (.not.allocated(y_send_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_send_ngroup',2,ny_comm,&
            intg=y_send_ngroup)
    endif

    ! z_send_ngroup
    if (allocated(z_send_ngroup)) then
       if (size(z_send_ngroup,2) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_send_ngroup',&
               1,size(z_send_ngroup,2),intg=z_send_ngroup)
       endif
    endif
    if (.not.allocated(z_send_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_send_ngroup',1,nz_comm,&
            intg=z_send_ngroup)
    endif

    ! x_recv_ngroup
    if (allocated(x_recv_ngroup)) then
       if (size(x_recv_ngroup,2) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_ngroup',&
               4,size(x_recv_ngroup,2),intg=x_recv_ngroup)
       endif
    endif
    if (.not.allocated(x_recv_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_ngroup',4,nx_comm,&
            intg=x_recv_ngroup)
    endif

    ! y_recv_ngroup
    if (allocated(y_recv_ngroup)) then
       if (size(y_recv_ngroup,2) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_ngroup',&
               2,size(y_recv_ngroup,2),intg=y_recv_ngroup)
       endif
    endif
    if (.not.allocated(y_recv_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_ngroup',2,ny_comm,&
            intg=y_recv_ngroup)
    endif

    ! z_recv_ngroup
    if (allocated(z_recv_ngroup)) then
       if (size(z_recv_ngroup,2) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_ngroup',&
               1,size(z_recv_ngroup,2),intg=z_recv_ngroup)
       endif
    endif
    if (.not.allocated(z_recv_ngroup)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_ngroup',1,nz_comm,&
            intg=z_recv_ngroup)
    endif

    ! x_send_ncoord
    if (allocated(x_send_ncoord)) then
       if (size(x_send_ncoord,2) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_send_ncoord',&
               4,size(x_send_ncoord,2),intg=x_send_ncoord)
       endif
    endif
    if (.not.allocated(x_send_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_send_ncoord',4,nx_comm,&
            intg=x_send_ncoord)
    endif

    ! y_send_ncoord
    if (allocated(y_send_ncoord)) then
       if (size(y_send_ncoord,2) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_send_ncoord',&
               2,size(y_send_ncoord,2),intg=y_send_ncoord)
       endif
    endif
    if (.not.allocated(y_send_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_send_ncoord',2,ny_comm,&
            intg=y_send_ncoord)
    endif

    ! z_send_ncoord
    if (allocated(z_send_ncoord)) then
       if (size(z_send_ncoord,2) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_send_ncoord',&
               1,size(z_send_ncoord,2),intg=z_send_ncoord)
       endif
    endif
    if (.not.allocated(z_send_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_send_ncoord',1,nz_comm,&
            intg=z_send_ncoord)
    endif

    ! x_recv_ncoord
    if (allocated(x_recv_ncoord)) then
       if (size(x_recv_ncoord,2) /= nx_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_ncoord',&
               4,size(x_recv_ncoord,2),intg=x_recv_ncoord)
       endif
    endif
    if (.not.allocated(x_recv_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_ncoord',4,nx_comm,&
            intg=x_recv_ncoord)
    endif

    ! y_recv_ncoord
    if (allocated(y_recv_ncoord)) then
       if (size(y_recv_ncoord,2) /= ny_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_ncoord',&
               2,size(y_recv_ncoord,2),intg=y_recv_ncoord)
       endif
    endif
    if (.not.allocated(y_recv_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_ncoord',2,ny_comm,&
            intg=y_recv_ncoord)
    endif

    ! z_recv_ncoord
    if (allocated(z_recv_ncoord)) then
       if (size(z_recv_ncoord,2) /= nz_comm) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_ncoord',&
               1,size(z_recv_ncoord,2),intg=z_recv_ncoord)
       endif
    endif
    if (.not.allocated(z_recv_ncoord)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_ncoord',1,nz_comm,&
            intg=z_recv_ncoord)
    endif

    ! (x_send_buf, x_recv_buf)
    if (allocated(x_send_buf)) then
       q_dealloc = .false.
       if (size(x_send_buf) /= nx_comm .or. size(x_recv_buf) /= nx_comm) then
          q_dealloc = .true.
       endif
       if (size(x_send_buf) >= 1) then
          if (size(x_send_buf(1)%p) /= x_send_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (size(x_recv_buf) >= 1) then
          if (size(x_recv_buf(1)%p) /= x_recv_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (q_dealloc) then
          do i=1,size(x_send_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_send_buf',&
                  size(x_send_buf(i)%p),mcrlp=x_send_buf(i)%p)
          enddo
          do i=1,size(x_recv_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_buf',&
                  size(x_recv_buf(i)%p),mcrlp=x_recv_buf(i)%p)
          enddo
          deallocate(x_send_buf)
          deallocate(x_recv_buf)
       endif
    endif
    if (.not.allocated(x_send_buf)) then
       allocate(x_send_buf(nx_comm))
       allocate(x_recv_buf(nx_comm))
       do i=1,nx_comm
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_send_buf',&
               x_send_buflen,mcrlp=x_send_buf(i)%p)
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','x_recv_buf',&
               x_recv_buflen,mcrlp=x_recv_buf(i)%p)
       enddo
    endif

    ! (y_send_buf, y_recv_buf)
    if (allocated(y_send_buf)) then
       q_dealloc = .false.
       if (size(y_send_buf) /= ny_comm .or. size(y_recv_buf) /= ny_comm) then
          q_dealloc = .true.
       endif
       if (size(y_send_buf) >= 1) then
          if (size(y_send_buf(1)%p) /= y_send_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (size(y_recv_buf) >= 1) then
          if (size(y_recv_buf(1)%p) /= y_recv_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (q_dealloc) then
          do i=1,size(y_send_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_send_buf',&
                  size(y_send_buf(i)%p),mcrlp=y_send_buf(i)%p)
          enddo
          do i=1,size(y_recv_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_buf',&
                  size(y_recv_buf(i)%p),mcrlp=y_recv_buf(i)%p)
          enddo
          deallocate(y_send_buf)
          deallocate(y_recv_buf)
       endif
    endif
    if (.not.allocated(y_send_buf)) then
       allocate(y_send_buf(ny_comm))
       allocate(y_recv_buf(ny_comm))
       do i=1,ny_comm
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_send_buf',&
               y_send_buflen,mcrlp=y_send_buf(i)%p)
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','y_recv_buf',&
               y_recv_buflen,mcrlp=y_recv_buf(i)%p)
       enddo
    endif

    ! (z_send_buf, z_recv_buf)
    if (allocated(z_send_buf)) then
       q_dealloc = .false.
       if (size(z_send_buf) /= nz_comm .or. size(z_recv_buf) /= nz_comm) then
          q_dealloc = .true.
       endif
       if (size(z_send_buf) >= 1) then
          if (size(z_send_buf(1)%p) /= z_send_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (size(z_recv_buf) >= 1) then
          if (size(z_recv_buf(1)%p) /= z_recv_buflen) then
             q_dealloc = .true.
          endif
       endif
       if (q_dealloc) then
          do i=1,size(z_send_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_send_buf',&
                  size(z_send_buf(i)%p),mcrlp=z_send_buf(i)%p)
          enddo
          do i=1,size(z_recv_buf)
             call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_buf',&
                  size(z_recv_buf(i)%p),mcrlp=z_recv_buf(i)%p)
          enddo
          deallocate(z_send_buf)
          deallocate(z_recv_buf)
       endif
    endif
    if (.not.allocated(z_send_buf)) then
       allocate(z_send_buf(nz_comm))
       allocate(z_recv_buf(nz_comm))
       do i=1,nz_comm
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_send_buf',&
               z_send_buflen,mcrlp=z_send_buf(i)%p)
          call chmalloc('domdec_d2d_comm.src','init_d2d_comm','z_recv_buf',&
               z_recv_buflen,mcrlp=z_recv_buf(i)%p)
       enddo
    endif

    ! cumpos
    if (allocated(cumpos)) then
       if (size(cumpos) /= nthread+1) then
          call chmdealloc('domdec_d2d_comm.src','init_d2d_comm','cumpos',size(cumpos),&
               intg=cumpos)
       endif
    endif
    if (.not.allocated(cumpos)) then
       call chmalloc('domdec_d2d_comm.src','init_d2d_comm','cumpos',nthread+1,lbou=0,&
            intg=cumpos)
    endif

    return
  end subroutine init_d2d_comm

  ! *
  ! * Uninitializes communication 
  ! *
  subroutine uninit_d2d_comm()
    use psf,only:natom
    use memory,only:chmdealloc
    use domdec_common,only:nx_comm, ny_comm, nz_comm
    use domdec_dlb,only:uninit_subboxes
    implicit none
    ! Variables
    integer i

    if (allocated(xtmp)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','xtmp',natom,crl=xtmp)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','ytmp',natom,crl=ytmp)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','ztmp',natom,crl=ztmp)
    endif

    if (allocated(x_recv_node)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_recv_node',nx_comm,&
            intg=x_recv_node)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_recv_node',ny_comm,&
            intg=y_recv_node)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_recv_node',nz_comm,&
            intg=z_recv_node)
    endif

    if (allocated(x_send_node)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_send_node',nx_comm,&
            intg=x_send_node)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_send_node',ny_comm,&
            intg=y_send_node)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_send_node',nz_comm,&
            intg=z_send_node)
    endif
    
    call uninit_subboxes()

    if (allocated(recv_flag)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','recv_flag',&
            size(recv_flag),log=recv_flag)
    endif

    if (allocated(reqbuf)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','reqbuf',&
            size(reqbuf),intg=reqbuf)
    endif

    if (allocated(neighsendbuf)) then
       do i=1,size(neighsendbuf)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','neighsendbuf',&
               size(neighsendbuf(i)%p), mcrlp=neighsendbuf(i)%p)
       enddo
       deallocate(neighsendbuf)
    endif

    if (associated(neighrecvbuf)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','neighrecvbuf',size(neighrecvbuf),&
            mcrlp=neighrecvbuf)
    endif

    if (allocated(neighind_dest)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','neighind_dest',size(neighind_dest),&
            intg=neighind_dest)
    endif

    if (allocated(x_send_atomind)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_send_atomind',&
            size(x_send_atomind),intg=x_send_atomind)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_send_atomind',&
            size(y_send_atomind),intg=y_send_atomind)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_send_atomind',&
            size(z_send_atomind),intg=z_send_atomind)
    endif

    if (allocated(x_recv_atomind)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_recv_atomind',&
            size(x_recv_atomind),intg=x_recv_atomind)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_recv_atomind',&
            size(y_recv_atomind),intg=y_recv_atomind)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_recv_atomind',&
            size(z_recv_atomind),intg=z_recv_atomind)
    endif

    if (allocated(x_send_ngroup)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_send_ngroup',&
            size(x_send_ngroup,1),size(x_send_ngroup,2),intg=x_send_ngroup)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_send_ngroup',&
            size(y_send_ngroup,1),size(y_send_ngroup,2),intg=y_send_ngroup)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_send_ngroup',&
            size(z_send_ngroup,1),size(z_send_ngroup,2),intg=z_send_ngroup)
    endif

    if (allocated(x_recv_ngroup)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_recv_ngroup',&
            size(x_recv_ngroup,1),size(x_recv_ngroup,2),intg=x_recv_ngroup)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_recv_ngroup',&
            size(y_recv_ngroup,1),size(y_recv_ngroup,2),intg=y_recv_ngroup)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_recv_ngroup',&
            size(z_recv_ngroup,1),size(z_recv_ngroup,2),intg=z_recv_ngroup)
    endif

    if (allocated(x_send_ncoord)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_send_ncoord',&
            size(x_send_ncoord,1),size(x_send_ncoord,2),intg=x_send_ncoord)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_send_ncoord',&
            size(y_send_ncoord,1),size(y_send_ncoord,2),intg=y_send_ncoord)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_send_ncoord',&
            size(z_send_ncoord,1),size(z_send_ncoord,2),intg=z_send_ncoord)
    endif

    if (allocated(x_recv_ncoord)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_recv_ncoord',&
            size(x_recv_ncoord,1),size(x_recv_ncoord,2),intg=x_recv_ncoord)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_recv_ncoord',&
            size(y_recv_ncoord,1),size(y_recv_ncoord,2),intg=y_recv_ncoord)
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_recv_ncoord',&
            size(z_recv_ncoord,1),size(z_recv_ncoord,2),intg=z_recv_ncoord)
    endif

    if (allocated(x_send_buf)) then
       do i=1,size(x_send_buf)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_send_buf',&
               size(x_send_buf(i)%p),mcrlp=x_send_buf(i)%p)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','x_recv_buf',&
               size(x_recv_buf(i)%p),mcrlp=x_recv_buf(i)%p)
       enddo
       do i=1,size(y_send_buf)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_send_buf',&
               size(y_send_buf(i)%p),mcrlp=y_send_buf(i)%p)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','y_recv_buf',&
               size(y_recv_buf(i)%p),mcrlp=y_recv_buf(i)%p)
       enddo
       do i=1,size(z_send_buf)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_send_buf',&
               size(z_send_buf(i)%p),mcrlp=z_send_buf(i)%p)
          call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','z_recv_buf',&
               size(z_recv_buf(i)%p),mcrlp=z_recv_buf(i)%p)
       enddo
       deallocate(x_send_buf)
       deallocate(y_send_buf)
       deallocate(z_send_buf)
       deallocate(x_recv_buf)
       deallocate(y_recv_buf)
       deallocate(z_recv_buf)
    endif

    if (allocated(cumpos)) then
       call chmdealloc('domdec_d2d_comm.src','uninit_d2d_comm','cumpos',size(cumpos),&
            intg=cumpos)
    endif

    return
  end subroutine uninit_d2d_comm

  ! *
  ! * Returns the z communication boundary for the node that is at (ix, iy, iz)
  ! *
  subroutine get_fz_boundary(ix, iy, iz, fz_z, cut, cut_grouped)
    use number
    use domdec_common
    use domdec_dlb
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: fz_z
    real(chm_real), intent(in) :: cut, cut_grouped
    ! Variables
    integer ixl, iyl, j, k
    real(chm_real) fz_z_top
    real(chm_real) ez_z_top, fx_z_top, fy_z_top
    real(chm_real) ex_y, ex_z, ey_x, ey_z, group_z, group_z_top
    real(chm_real) xd, yd, zd, dist, y0, ez_y, shy
    real(chm_real) cutsq, cutz, cutz_grouped, cutsq_grouped
    logical q_checkgrouped

    fz_z = get_nodebz(iz, iy, ix)

    if (q_load_balance_z) then
       cutsq = cut*cut
       cutz = cut/boxz
       cutsq_grouped = cut_grouped*cut_grouped
       cutz_grouped = cut_grouped/boxz
       ! These loops expand the FZ zone to include FZ-FX, FZ-FY, 
       ! and FZ-EZ interactions correctly.
       ! NOTE: Although they could be written as a single loop, 
       ! they are written out separately for clarity sake
       !
       ! This loop expands the FZ zone to include all FZ-EZ interactions correctly.
       ! We loop through: ix+1:ix+nx_comm, iy+1:iy+ny_comm
       !
       ! fz_z     = z coordinate of the FZ zone bottom
       ! fz_z_top = z coordinate of the FZ zone top
       ! ez_z_top = z coordinate of the EZ zone top
       !
       ! y0   = y coordinate of the node (iy,ix) top
       ! ez_y = bottom y coordinate of the cell (iyl,ixl) in EZ
       ! xd = x distance from the node top (x,y) -corner
       ! yd = positive y distance (if yd < 0, y-distance must be considered to be 0)
       xd = zero
       y0 = get_nodeby(iy,ix)
       ! Loop through all cells in the EZ zone
       do ixl=ix+1,ix+nx_comm
          ez_y = get_nodeby(iy, ixl)
          do iyl=iy+1,iy+ny_comm
             yd = ez_y - y0
             yd = max(zero, yd)              ! Enforces yd >= 0
             fz_z_top = fz_z + cutz
             ez_z_top = get_nodebz(iz, iyl, ixl)
             zd = fz_z_top - ez_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2 - (xd*boxx)**2)/boxz - cutz + ez_z_top
             endif
             ez_y = ez_y + nodefry_pbc(iyl, ixl)
          enddo
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FZ zone to include all FZ-FX interactions correctly.
       xd = zero
       ! Loop through all cells in the FX zone
       do ixl=ix+1,ix+nx_comm
          fz_z_top = fz_z + cutz
          fx_z_top = get_nodebz(iz, iy, ixl)
          if (ixl == ix+1) then
             ! Treat special case of neighboring cell separately
             if (fz_z_top - fx_z_top < cutz) fz_z = fx_z_top
          else
             zd = fz_z_top - fx_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + fx_z_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FZ zone to include all FZ-FY interactions correctly.
       yd = zero
       ! Loop through all cells in the FY zone
       do iyl=iy+1,iy+ny_comm
          fz_z_top = fz_z + cutz
          fy_z_top = get_nodebz(iz, iyl, ix)
          if (iyl == iy+1) then
             ! Treat special case of neighboring cell separately
             if (fz_z_top - fy_z_top < cutz) fz_z = fy_z_top
          else
             zd = fz_z_top - fy_z_top
             zd = max(zero, zd)              ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2)/boxz - cutz + fy_z_top
             endif
          endif
          yd = yd + nodefry_pbc(iyl, ix)
       enddo

       ! Coordinates from the FZ zone will be further communicated in the y and x
       ! directions to zones EX, EY, and C. Therefore, we have to expand FZ zone
       ! to include all coordinates that are needed in this future communications

       ! This loop expands the FZ zone to include coordinates in the EX zones of the
       ! nodes in iy-1:iy-ny_comm
       ! Also takes care of the expansion due to future communication to C zones
       ! (calls are made from get_ex_boundary to get_c_boundary)
       !
       do iyl=iy-1,iy-ny_comm,-1
          call get_ex_boundary(ix, iyl, iz, ex_y, ex_z, &
               group_z, q_checkgrouped, cut, cut_grouped)
          if (iyl == iy-1) then
             ! Treat neighboring cell case separately
             fz_z = max(fz_z, ex_z)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                group_z_top = group_z + cutz_grouped
                if (group_z_top > fz_z_top) fz_z = group_z_top - cutz
             endif
          else
             fz_z_top = fz_z + cutz
             zd = fz_z_top - ex_z
             zd = max(zero, zd)           ! Ensures zd >= 0
             ! Shift factor shy = 1 if we have crossed the box border in y-direction
             shy = zero
             if (iy-1 > 0 .and. iyl <= 0) shy = one
             !
             yd = get_nodeby(iy-1, ix) + shy - ex_y
             yd = max(zero, yd)           ! Ensures yd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (yd*boxy)**2)/boxz - cutz + ex_z
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                zd = fz_z_top - group_z
                zd = max(zero, zd)           ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (yd*boxy)**2
                if (dist < cutsq_grouped) then
                   fz_z = sqrt(cutsq_grouped - (yd*boxy)**2)/boxz - cutz + group_z
                endif
             endif
          endif
       enddo

       ! This loop expands the FZ zone to include coordinates in the EY zones of the
       ! nodes in ix-1:ix-nx_comm
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          call get_ey_boundary(ixl, iy, iz, ey_x, ey_z, group_z, q_checkgrouped, cut)
          if (ixl == ix-1) then
             ! Treat neighboring cell case separately
             fz_z = max(fz_z, ey_z)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                group_z_top = group_z + cutz_grouped
                if (group_z_top > fz_z_top) fz_z = group_z_top - cutz
             endif
          else
             fz_z_top = fz_z + cutz          
             zd = fz_z_top - ey_z
             zd = max(zero, zd)           ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fz_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + ey_z
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fz_z_top = fz_z + cutz
                zd = fz_z_top - group_z
                zd = max(zero, zd)           ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   fz_z = sqrt(cutsq_grouped - (xd*boxx)**2)/boxz - cutz + group_z
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif

    if (isnan(fz_z)) then
       call wrndie(-5,'<domdec_d2d_comm>','get_fz_boundary: NaN fz_z')
    endif

    if (fz_z < zero .or. fz_z > one) then
       call wrndie(-5,'<domdec_d2d_comm>','get_fz_boundary: fz_z out of range')
    endif

    return
  end subroutine get_fz_boundary

  ! *
  ! * Returns the y communication boundary for the node that is at iy
  ! *
  subroutine get_fy_boundary(ix, iy, iz, fy_y, cut, cut_grouped)
    use number
    use domdec_common
    use domdec_dlb
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: fy_y
    real(chm_real), intent(in) :: cut, cut_grouped
    ! Variables
    integer ixl, izl
    real(chm_real) fy_y_top
    real(chm_real) ey_y_top, fx_y_top, ez_x, ez_y, group_y, group_y_top
    real(chm_real) xd, yd, zd, dist, z0, z1
    real(chm_real) cuty, cutsq, cuty_grouped, cutsq_grouped
    logical q_checkgrouped

    fy_y = get_nodeby(iy, ix)

    if (q_load_balance_y) then
       cutsq = cut*cut
       cuty = cut/boxy
       cutsq_grouped = cut_grouped*cut_grouped
       cuty_grouped = cut_grouped/boxy
       ! These loops expand the FY zone to include FY-FX and FY-EY interactions correctly
       !
       ! This loop expands the FY zone to include FY-EY interactions correctly
       !
       ! fy_y     = y coordinate of the FY zone bottom
       ! fy_y_top = y coordinate of the FY zone top
       ! ey_y_top = y coordinate of the EY zone top
       !
       ! z0 = z coordinate of the node (iz,iy,ix) top
       ! z1 = z coordinate of the node (izl,iy,ixl) bottom
       !
       ! xd, zd   = x and z distance from node top (x,z) -corner
       xd = zero
       z0 = get_nodebz(iz,iy,ix)
       ! Loop through all cells in the EY zone
       do ixl=ix+1,ix+nx_comm
          z1 = get_nodebz(iz,iy,ixl)
          do izl=iz+1,iz+nz_comm
             zd = z1 - z0
             zd = max(zero, zd)               ! Ensures zd >= 0
             ey_y_top = get_nodeby(iy,ixl)
             fy_y_top = fy_y + cuty
             yd = fy_y_top - ey_y_top
             yd = max(zero, yd)               ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2 + (zd*boxz)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2 - (zd*boxz)**2)/boxy - cuty + ey_y_top
             endif
             z1 = z1 + nodefrz_pbc(izl,iy,ixl)
          enddo
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! This loop expands the FY zone to include FY-FX interactions correctly
       !
       ! fx_y_top = y coordinate of the FX zone top
       ! xd = x distance between FY and FX
       xd = zero
       do ixl=ix+1,ix+nx_comm
          fx_y_top = get_nodeby(iy,ixl)
          fy_y_top = fy_y + cuty
          if (ixl == ix+1) then
             ! Treat special case of neighboring cell separately (xd = 0)
             if (fy_y_top - fx_y_top < cuty) fy_y = fx_y_top
          else
             yd = fy_y_top - fx_y_top
             yd = max(zero, yd)              ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + fx_y_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! Coordinates in FY will be further communicated in x direction to zone EZ.
       ! Therefore, we have to expand FY zone to include all coordinates that are needed
       ! in this future communication

       ! This loop expands the FY zone to include coordinates in the EZ zones of the nodes
       ! ix-1:ix-nx_comm
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          call get_ez_boundary(ixl, iy, ez_x, ez_y, group_y, q_checkgrouped)
          if (ixl == ix-1) then
             ! Treat neighboring cell case (xd = 0) separately
             fy_y = max(fy_y, ez_y)
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fy_y_top = fy_y + cuty
                group_y_top = group_y + cuty_grouped
                if (group_y_top > fy_y_top) fy_y = group_y_top - cuty
             endif
          else
             fy_y_top = fy_y + cuty
             yd = fy_y_top - ez_y
             yd = max(zero, yd)          ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                fy_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + ez_y
             endif
             if (q_checkgrouped) then
                ! Check if grouped interactions expand FZ zone
                fy_y_top = fy_y + cuty
                yd = fy_y_top - group_y
                yd = max(zero, yd)          ! Ensures yd >= 0
                dist = (yd*boxy)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   fy_y = sqrt(cutsq_grouped - (xd*boxx)**2)/boxy - cuty + group_y
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif

    if (isnan(fy_y)) then
       call wrndie(-5,'<domdec_d2d_comm>','get_fy_boundary: NaN fy_y')
    endif

    if (fy_y < zero .or. fy_y > one) then
       call wrndie(-5,'<domdec_d2d_comm>','get_fy_boundary: fy_y out of range')
    endif

    return
  end subroutine get_fy_boundary

  ! *
  ! * Returns the (ex_y, ex_z) communication origin for the node that is at (ix,iy,iz)
  ! *
  subroutine get_ex_boundary(ix, iy, iz, ex_y, ex_z, group_z, q_checkgrouped, cut, cut_grouped)
    use number
    use domdec_common
    use domdec_dlb
    use domdec_grouped,only:q_grouped
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: ex_y, ex_z, group_z
    logical, intent(out) :: q_checkgrouped
    real(chm_real), intent(in) :: cut, cut_grouped
    ! Variables
    integer ixl
    real(chm_real) ex_y_top, ex_z_top
    real(chm_real) fx_y_top, fx_z_top
    real(chm_real) xd, zd, yd, dist
    real(chm_real) cuty, cutz, cutsq, cutsq_grouped, cutz_grouped, cuty_grouped
    real(chm_real) c_x, c_y, c_z, c_group_y, c_group_z, c_group_y_top, group_z_top
    logical q_c_checkgrouped

    ex_y = get_nodeby(iy, ix)
    ex_z = get_nodebz(iz, iy, ix)
    q_checkgrouped = .false.

    if (q_load_balance) then
       cutsq = cut*cut
       cuty = cut/boxy
       cutz = cut/boxz
       cutsq_grouped = cut_grouped*cut_grouped
       cuty_grouped = cut_grouped/boxy
       cutz_grouped = cut_grouped/boxz
       ! Determine grouped interaction origin z-coordinate
       group_z = ex_z
       ! This checks for the neighboring cell in the FY zone
       if (ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix))
       ! This checks for the neighboring cell in the FX zone
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz,iy,ix+1))
       ! This checks for the neighboring cell in the EZ zone
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix+1))
       if (q_grouped .and. group_z > ex_z) q_checkgrouped = .true.

       ! This loop expands the EX zone to include all FX-EX interactions
       ! (ex_y, ex_z) = y and z coordinates of the EX zone bottom
       ! (fx_y_top, fx_z_top) = y and z coordinates of the FX zone top
       ! xd = x distance between EX and FX boundary
       xd = zero
       ! Loop through all cells in the FX zone
       do ixl=ix+1,ix+nx_comm
          fx_y_top = get_nodeby(iy, ixl)
          fx_z_top = get_nodebz(iz, iy, ixl)
          ! In order to include all FX-EX interactions, the distance between FX
          ! zone top and the EX zone top must be smaller than cut
          ex_y_top = ex_y + cuty
          ex_z_top = ex_z + cutz
          if (ixl == ix + 1) then
             ! Treat special case xd == 0 separately
             if (ex_y_top - fx_y_top < cuty) ex_y = fx_y_top
             if (ex_z_top - fx_z_top < cutz) ex_z = fx_z_top
          else
             ! Check y-distance
             yd = ex_y_top - fx_y_top
             yd = max(zero, yd)             ! Ensures yd >= 0
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EX zone in y-direction
                ex_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + fx_y_top
             endif
             ! Check z-distance
             zd = ex_z_top - fx_z_top
             zd = max(zero, zd)             ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EX zone in z-direction
                ex_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + fx_z_top
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

       ! Part of the EX zone will be communicated further in the x-direction to 
       ! nodes at ix-1:ix-nx_comm as C zone
       ! This loop expands the EX zone so that enough is communicated
       !
       ! xd = x distance between EX and C
       xd = zero
       do ixl=ix-1,ix-nx_comm,-1
          ! We are at node (ixl, iy, iz)
          call get_c_boundary(ixl, iy, iz, c_x, c_y, c_z, c_group_y, c_group_z, &
               q_c_checkgrouped)
          if (ixl == ix-1) then
             ! Treat special case xd = 0 separately
             ex_y = max(ex_y, c_y)
             ex_z = max(ex_z, c_z)
             if (q_c_checkgrouped) then
                ! Increase EX grouped interaction zone if needed
                if (group_z < c_group_z) then
                   group_z = c_group_z
                   q_checkgrouped = .true.
                endif
                ! Increase EX in y-direction if needed
                ex_y_top = ex_y + cuty
                c_group_y_top = c_group_y + cuty_grouped
                if (c_group_y_top > ex_y_top) ex_y = c_group_y_top - cuty
             endif
          else
             ! If EX top z falls within C zone => expand EX in z-direction
             ex_z_top = ex_z + cutz
             zd = ex_z_top - c_z
             zd = max(zero,zd)                    ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ex_z = sqrt(cutsq - (xd*boxx)**2)/boxz - cutz + c_z
             endif
             ! If EX top y falls within C zone => expand EX in y-direction
             ex_y_top = ex_y + cuty
             yd = ex_y_top - c_y                  ! Ensures yd >= 0
             yd = max(zero, yd)
             dist = (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ex_y = sqrt(cutsq - (xd*boxx)**2)/boxy - cuty + c_y
             endif
             !
             if (q_c_checkgrouped) then
                ! Increase EX grouped interaction zone if needed
                group_z_top = group_z + cutz_grouped
                zd = group_z_top - c_group_z
                zd = max(zero,zd)                  ! Ensures zd >= 0
                dist = (zd*boxz)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   group_z = sqrt(cutsq_grouped - (xd*boxx)**2)/boxz - cutz_grouped + c_group_z
                   q_checkgrouped = .true.
                endif
                ! Increase EX in y-direction if needed
                ex_y_top = ex_y + cuty
                yd = ex_y_top - c_group_y
                yd = max(zero,yd)                  ! Ensures yd >= 0
                dist = (yd*boxy)**2 + (xd*boxx)**2
                if (dist < cutsq_grouped) then
                   ex_y = sqrt(cutsq_grouped - (xd*boxx)**2)/boxy - cuty + c_group_y
                endif
             endif
          endif
          xd = xd + nodefrx_pbc(ixl)
       enddo

    endif

    if (isnan(ex_y) .or. isnan(ex_z)) then
       call wrndie(-5,'<domdec_d2d_comm>','get_ex_boundary: ex_y or ex_z is NaN')
    endif

    if (ex_y < zero .or. ex_y > one .or. ex_z < zero .or. ex_z > one) then
       call wrndie(-5,'<domdec_d2d_comm>','get_ex_boundary: ex_y or ex_z out of range')
    endif


    return
  end subroutine get_ex_boundary

  ! *
  ! * Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
  ! *
  subroutine get_fx_boundary(ix, fx_x)
    use domdec_common
    use domdec_dlb
    implicit none
    ! Input / Output
    integer, intent(in) :: ix
    real(chm_real), intent(out) :: fx_x
    
    fx_x = get_nodebx(ix)

    return
  end subroutine get_fx_boundary

  ! *
  ! * Returns the (ez_x, ez_y) communication origin for the node that is at (ix,iy,iz)
  ! *
  subroutine get_ez_boundary(ix, iy, ez_x, ez_y, group_y, q_checkgrouped)
    use domdec_common
    use domdec_dlb
    use domdec_grouped,only:q_grouped
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy
    real(chm_real), intent(out) :: ez_x, ez_y, group_y
    logical, intent(out) :: q_checkgrouped
    ! Variables

    ez_x = get_nodebx(ix)
    ez_y = get_nodeby(iy, ix)
    q_checkgrouped = .false.

    if (q_load_balance) then
       group_y = ez_y
       if (nx_comm > 0) group_y = max(group_y, get_nodeby(iy, ix+1))
       if (q_grouped .and. group_y > ez_y) q_checkgrouped = .true.
    endif

    return
  end subroutine get_ez_boundary

  ! *
  ! * Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
  ! *
  subroutine get_ey_boundary(ix, iy, iz, ey_x, ey_z, group_z, q_checkgrouped, cut)
    use number
    use domdec_common
    use domdec_dlb
    use domdec_grouped,only:q_grouped
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: ey_x, ey_z, group_z
    logical, intent(out) :: q_checkgrouped
    real(chm_real), intent(in) :: cut
    ! Variables
    integer ixl, iyl
    real(chm_real) ey_z_top, ey_y_top
    real(chm_real) fy_z_top, fy_y
    real(chm_real) xd, yd, zd, dist, cutz, cutsq
    
    ey_x = get_nodebx(ix)
    ey_z = get_nodebz(iz, iy, ix)
    q_checkgrouped = .false.

    if (q_load_balance) then
       cutsq = cut*cut
       cutz = cut/boxz
       group_z = ey_z
       ! This checks if the neighboring node (iy,ix+1) in the EY zone is higher
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz, iy, ix+1))
       ! This checks if the neighboring node (iy+1,ix) in the FY zone is higher
       ! NOTE: only done if EY and FY overlap
       ! nodeby(iy,ix)   = low y coordinate of FY
       ! nodeby(iy,ix+1) = high y coordinate of EY
       if (nx_comm > 0 .and. ny_comm > 0) then
          if (get_nodeby(iy,ix) < get_nodeby(iy,ix+1)) then
             group_z = max(group_z, get_nodebz(iz, iy+1, ix))
          endif
       endif
       ! This checks if the corner node (iy+1, ix+1) in the C zone is higher
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz, iy+1, ix+1))
       if (q_grouped .and. group_z > ey_z) q_checkgrouped = .true.
       
       ! This loop expands the EY zone in z-direction to include all FY-EY interactions
       !
       ! ey_z     = z coordinate of the EY zone bottom
       ! ey_z_top = z coordinate of the EY zone top
       ! ey_y_top  = y coordinate of the EY zone top
       !
       ! fy_z_top = z coordinate of the FY zone top
       ! fy_y     = y coordinate of the FY zone bottom
       !
       ! yd = y distance between FY and EY boundary
       yd = zero
       fy_y = get_nodeby(iy, ix)
       ! Loop through all cells in the FY zone, from iy+1 to iy+ny_comm
       do iyl=iy+1,iy+ny_comm
          fy_z_top = get_nodebz(iz, iyl, ix)
          ! Loop through all x rows of EY zone
          ! (this has to be done because the rows have different y coordinates)
          xd = zero
          do ixl=ix+1,ix+nx_comm
             ey_y_top = get_nodeby(iy, ixl)
             ey_z_top = ey_z + cutz
             ! In order to include all FY-EY interactions, the distance between FY
             ! zone top and EY zone top must be smaller than cut
             yd = fy_y - ey_y_top
             yd = max(zero, yd)               ! Ensures yd >= 0
             zd = ey_z_top - fy_z_top
             zd = max(zero, zd)               ! Ensures zd >= 0
             dist = (zd*boxz)**2 + (yd*boxy)**2 + (xd*boxx)**2
             if (dist < cutsq) then
                ! Expand EY zone in z-direction
                ey_z = sqrt(cutsq - (yd*boxy)**2 - (xd*boxx)**2)/boxz - cutz + fy_z_top
             endif
             xd = xd + nodefrx_pbc(ixl)
          enddo
          fy_y = fy_y + nodefry_pbc(iyl,ix)
       enddo
    endif    

    if (isnan(ey_x) .or. isnan(ey_z)) then
       call wrndie(-5,'<domdec_d2d_comm>','get_ey_boundary: ey_x or ey_z is NaN')
    endif

    if (ey_x < zero .or. ey_x > one .or. ey_z < zero .or. ey_z > one) then
       call wrndie(-5,'<domdec_d2d_comm>','get_ey_boundary: ey_x or ey_z out of range')
    endif

    return
  end subroutine get_ey_boundary
  
  ! *
  ! * Returns the (x,y,z) communication origin for the node that is at (ix,iy,iz)
  ! *
  subroutine get_c_boundary(ix, iy, iz, c_x, c_y, c_z, group_y, group_z, q_checkgrouped)
    use number
    use domdec_common
    use domdec_dlb
    use domdec_grouped,only:q_grouped
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: c_x, c_y, c_z, group_y, group_z
    logical, intent(out) :: q_checkgrouped
    ! Variables

    c_x = get_nodebx(ix)
    c_y = get_nodeby(iy, ix)
    c_z = get_nodebz(iz, iy, ix)
    q_checkgrouped = .false.

    if (q_load_balance) then
       group_y = c_y
       group_z = c_z
       ! group_y = max between (iy,ix) and (iy,ix+1)
       if (nx_comm > 0) group_y = max(group_y, get_nodeby(iy,ix+1) )
       ! group_z = max between (iy,ix), (iy,ix+1), (iy+1,ix), (iy+1,ix+1)
       if (nx_comm > 0) group_z = max(group_z, get_nodebz(iz,iy,ix+1))
       if (ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix))
       if (nx_comm > 0 .and. ny_comm > 0) group_z = max(group_z, get_nodebz(iz,iy+1,ix+1))
       if (q_grouped .and. (group_y > c_y .or. group_z > c_z)) q_checkgrouped = .true.
    endif
    
    return
  end subroutine get_c_boundary

  ! *
  ! * Returns the z0 (lowest z coordinate cell within the c zone)
  ! *
  subroutine get_z0_for_c(ix, iy, iz, z0)
    use domdec_common
    use domdec_dlb
    implicit none
    ! Input / Output
    integer, intent(in) :: ix, iy, iz
    real(chm_real), intent(out) :: z0
    ! Variables
    integer ixl, iyl

    z0 = get_nodebz(iz, iy, ix)

    if (q_load_balance) then
       do ixl=ix,ix+nx_comm
          do iyl=iy,iy+ny_comm
             z0 = min(z0, get_nodebz(iz, iyl, ixl))
          enddo
       enddo
    endif

    return
  end subroutine get_z0_for_c

  ! *
  ! * Communicates coordinates between homeboxes
  ! * NOTE: this version uses mpi_iprobe
  ! *
  subroutine transfer_coord(x, y, z, rezone)
    use memory
    use number
    use mpi,only:mpi_real8, mpi_byte, mpi_success, mpi_statuses_ignore
    use psf,only:natom
    use parallel,only:comm_charmm, mpi_integer_size, mpi_real8_size
    use groupxfast,only:group, group_out, maxgrp_rad, groupcenter, groupbox, invgroup
    use inbnd,only:cutnb
    use stream,only:outu
    use domdec_common,only:natoml_tot, natoml, atoml, zonelist, homeix, homeiy, homeiz, &
         nx_comm, ny_comm, nz_comm, groupl, invx, invy, invz, box_a, box_b, box_c, &
         homezone, ny, nz, set_box, zonelist_atom, q_gpu, q_test
    use domdec_dlb,only:q_load_balance
    use domdec_grouped,only:q_grouped, rcut_grouped
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop  
#endif
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    logical, intent(in) :: rezone
    ! Variables
    integer i, j, k, l, ig, gtype
    real(chm_real) xj, yj, zj, xd, yd, zd, ydd, zdd
    real(chm_real) xf, yf, zf
    real(chm_real) shx, shy, shz
    real(chm_real) cutsq, cut, rcut_groupedsq, z0
    real(chm_real) y_grouped, z_grouped
    integer nreqbuf, ierror
    logical inzone, q_checkgrouped
    integer ncoord0, ngroup0
    integer x_send_count, y_send_count, z_send_count
    integer x_recv_count, y_recv_count, z_recv_count
    integer :: x_homezone_val(1:4) = (/ 5, 6, 7, 8 /)
    integer :: y_homezone_val(1:2) = (/ 3, 4 /)
    integer :: z_homezone_val(1) = (/ 2 /)
    integer ngroup(4)
    integer is, iq

    call set_box()

    if (q_load_balance .and. q_grouped) rcut_groupedsq = (rcut_grouped)**2

    cut = cutnb + two*maxgrp_rad
    cutsq = cut*cut !cutgrp*cutgrp
   
    ! Start receiving buffers from +z direction
    nreqbuf = 0
    if (rezone) then
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('calc group box size')
#endif
       ! Calculate group center and group box size
!$omp parallel do schedule(static) private(j, i)
       do j=1,zonelist(1)
          i = groupl(j)
          call calc_groupbox_loc(i, x, y, z)
       enddo
!$omp end parallel do
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
    else
       do i=1,nz_comm
          if (z_recv_ncoord(1,i) > 0) then
             ! Start receiving groups
             nreqbuf = nreqbuf + 1
             call mpi_irecv(z_recv_buf(i)%p, 3*z_recv_ncoord(1,i)*mpi_real8_size, mpi_byte, &
                  z_recv_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec_d2d_comm>','Error start receiving groups from +z direction')
          endif
       enddo
    endif

    ! Send coordinates to boxes at -z direction
    ncoord0 = 0
    do i=1,nz_comm
       z_send_count = 0
       if (rezone) then

          call get_fz_boundary(homeix, homeiy, homeiz-i, zf, cut, rcut_grouped)

          ! shift factor: if we go around the boundary, shift by one
          shz = zero
          if (homeiz - i <= 0) shz = one

          ngroup(1) = 0
          z_send_ncoord(1,i) = 0

          do ig=1,zonelist(1)
             j = groupl(ig)

             inzone = .false.

             zd = groupcenter(3,j)*invz + half
             zd = zd - floor(zd)

             zd = zd + shz - zf
             if (zd < zero) zd = zero         ! group is behind boundary
             zj = zd*box_c(3)
             if (zj**2 < cutsq) inzone = .true.

             if (inzone) then
                call add_group(group(j), ncoord0, ngroup(1), z_send_ncoord(1,i), z_send_atomind)
             endif
          enddo

          ! Pack atom indices
          call pack_atomind(1, ncoord0, ngroup(1:1), z_send_ncoord(:,i), z_send_atomind, &
               z_send_buf(i)%p, z_send_count)

       endif

       ! Pack coordinates
       call pack_coords(ncoord0, z_send_ncoord(1,i), z_send_atomind, x, y, z, &
            z_send_buf(i)%p, z_send_count)

       ! Send the coordinates
       if (z_send_count > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(z_send_buf(i)%p, z_send_count, mpi_byte, &
               z_send_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error sending coordinates from +z direction')
       endif

       ncoord0 = ncoord0 + z_send_ncoord(1,i)
    enddo

    ! Wait for coordinates from +z direction
    if (rezone .and. nz_comm > 0) then
       call recv_groups_coords(nz_comm, z_recv_node, z_recv_buf)
    endif

    ! Wait for mpi_isend and mpi_irecv to finish
    if (nreqbuf > 0) then
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','error in mpi_waitall from +z direction')
    endif

    ! Copy the coordinates from +z direction into correct arrays
    ncoord0 = 0
    do i=1,nz_comm
       z_recv_count = 0
       if (rezone) then
          call unpack_atomind(1, ncoord0, z_recv_ngroup(:,i), z_recv_ncoord(:,i), &
               z_recv_atomind, z_recv_buf(i)%p, z_recv_count)
          call set_homezone(1, ncoord0, z_recv_ncoord(:,i), z_recv_atomind, &
               z_homezone_val, homezone)
       endif
       call unpack_coords(ncoord0, z_recv_ncoord(1,i), z_recv_atomind, x, y, z, z_recv_buf(i)%p, &
            z_recv_count)
       ncoord0 = ncoord0 + z_recv_ncoord(1,i)
    enddo

    if (rezone) then
       call build_groups(1, nz_comm, z_recv_ngroup, z_recv_ncoord, &
            z_recv_atomind, x, y, z, groupl, atoml, zonelist(1:2), zonelist_atom(1:2))
    endif

    ! Start receiving buffers from +y direction
    nreqbuf = 0
    if (.not.rezone) then
       do i=1,ny_comm
          if (y_recv_ncoord(2,i) > 0) then
             ! Start receiving groups
             nreqbuf = nreqbuf + 1
             call mpi_irecv(y_recv_buf(i)%p, 3*y_recv_ncoord(2,i)*mpi_real8_size, mpi_byte,&
                  y_recv_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec_d2d_comm>','Error start receiving groups from +y direction')
          endif
       enddo
    endif

    ! Send coordinates to boxes in -y direction
    ncoord0 = 0
    do i=1,ny_comm
       y_send_count = 0
       if (rezone) then

          call get_fy_boundary(homeix, homeiy-i, homeiz, yf, cut, rcut_grouped)

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('y loop1')
#endif

          shy = zero
          if (homeiy - i <= 0) shy = one

          ! Copy the coordinates from home box to y_send_buf
          ngroup(1) = 0
          y_send_ncoord(1,i) = 0
          do ig=1,zonelist(1)                   ! (fy)
             j = groupl(ig)

             inzone = .false.

             yd = groupcenter(2,j)*invy + half
             yd = yd - floor(yd)

             yd = yd + shy - yf
             if (yd < zero) yd = zero         ! group is behind boundary
             yj = yd*box_b(2)
             if (yj**2 < cutsq) inzone = .true.

             if (inzone) then
                call add_group(group(j), ncoord0, ngroup(1), y_send_ncoord(1,i), y_send_atomind)
             endif
          enddo

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif

          ! Calculate the ex boundary for node at (homei, homeiy-i, homeiz)
          call get_ex_boundary(homeix, homeiy-i, homeiz, yf, zf, &
               z_grouped, q_checkgrouped, cut, rcut_grouped)


#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('y loop2')
#endif

          ! Copy the coordinates received from +z to y_send_buf (ex)
          ngroup(2) = ngroup(1)
          y_send_ncoord(2,i) = y_send_ncoord(1,i)

          ngroup0 = zonelist(1)
          do l=1,nz_comm
             shz = zero
             if (homeiz + l > nz) shz = one
             do j=1,z_recv_ngroup(1,l)   ! Loop over groups from +z box
                k = groupl(ngroup0 + j)

                inzone = .false.

                yd = groupcenter(2,k)*invy + half
                zd = groupcenter(3,k)*invz + half
                yd = yd - floor(yd)
                zd = zd - floor(zd)

                zdd = zd
                yd = yd + shy - yf
                zd = zd + shz - zf
                ! Calculate minimum image: (yd, zd) in (-0.5, 0.5)
                ! The following if-statement takes care of the cases where destination box has
                ! higher boundary than the current box i.e. prevents rounding of the ex zone
                ! in the lower right y,z -corner
                if (yd < zero) yd = zero        ! group is behind boundary
                if (zd < zero) zd = zero        ! group is behind boundary
                yj = yd*box_b(2) + zd*box_c(2)
                zj = zd*box_c(3)
                if (yj**2 + zj**2 < cutsq) then
                   inzone = .true.
                elseif (q_checkgrouped) then
                   call group_out(group(k), gtype)
                   if (gtype == 0) then
                      zd = zdd + shz - z_grouped
                      if (zd < zero) zd = zero
                      yj = yd*box_b(2) + zd*box_c(2)
                      zj = zd*box_c(3)
                      if (yj**2 + zj**2 < rcut_groupedsq) inzone = .true.
                   endif
                endif

                if (inzone) then
                   call add_group(group(k), ncoord0, ngroup(2), y_send_ncoord(2,i), y_send_atomind)
                endif
             enddo
             ngroup0 = ngroup0 + z_recv_ngroup(1,l)
          enddo

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('pack_groups_coords')
#endif

          ! Pack atom indices
          call pack_atomind(2, ncoord0, ngroup(1:2), y_send_ncoord(:,i), y_send_atomind, &
               y_send_buf(i)%p, y_send_count)

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif

       endif

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('pack_coords')
#endif

       ! Pack coordinates
       call pack_coords(ncoord0, y_send_ncoord(2,i), y_send_atomind, x, y, z, &
               y_send_buf(i)%p, y_send_count)

#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif

       ! Send the coordinates
       if (y_send_count > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(y_send_buf(i)%p, y_send_count, mpi_byte, &
               y_send_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error sending coordinates from +y direction')
       endif

       ncoord0 = ncoord0 + y_send_ncoord(2,i)
    enddo

    ! Wait for coordinates from +y direction
    if (rezone .and. ny_comm > 0) then
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('recv_groups_coords')
#endif
       call recv_groups_coords(ny_comm, y_recv_node, y_recv_buf)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
    endif

    if (nreqbuf > 0) then
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('mpi_waitall')
#endif
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error in mpi_waitall from +y direction')
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
    endif

    ! Copy the coordinates from +y direction into correct arrays
    ncoord0 = 0
    do i=1,ny_comm
       y_recv_count = 0
       if (rezone) then
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('unpack_atomind')
#endif
          call unpack_atomind(2, ncoord0, y_recv_ngroup(:,i), y_recv_ncoord(:,i), &
               y_recv_atomind, y_recv_buf(i)%p, y_recv_count)

#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_start('set_homezone')
#endif
          call set_homezone(2, ncoord0, y_recv_ncoord(:,i), y_recv_atomind, &
               y_homezone_val, homezone)
#if KEY_DOMDEC_GPU==1
          if (q_gpu) call range_stop()
#endif
       endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('unpack_coords')
#endif
       call unpack_coords(ncoord0, y_recv_ncoord(2,i), y_recv_atomind, x, y, z, y_recv_buf(i)%p, &
            y_recv_count)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()
#endif
       ncoord0 = ncoord0 + y_recv_ncoord(2,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('build_groups')
#endif
    if (rezone) then
       call build_groups(2, ny_comm, y_recv_ngroup, y_recv_ncoord, &
            y_recv_atomind, x, y, z, groupl, atoml, zonelist(2:4), zonelist_atom(2:4))
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif


    ! Start receiving buffers from +x direction
    nreqbuf = 0
    if (.not.rezone) then
       do i=1,nx_comm
          if (x_recv_ncoord(4,i) > 0) then
             ! Start receiving groups
             nreqbuf = nreqbuf + 1
             call mpi_irecv(x_recv_buf(i)%p, 3*x_recv_ncoord(4,i)*mpi_real8_size, mpi_byte, &
                  x_recv_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec_d2d_comm>','Error start receiving groups from +x direction')
          endif
       enddo
    endif

    ! Send coordinates to boxes in -x direction
    ncoord0 = 0
    do i=1,nx_comm
       x_send_count = 0
       if (rezone) then

          call get_fx_boundary(homeix-i, xf)

          shx = zero
          if (homeix - i <= 0) shx = one

          x_send_ncoord(1,i) = 0
          ngroup(1) = 0
          ! Copy the coordinates from home box to x_send_group
          do ig=1,zonelist(1)             ! (fx)
             j = groupl(ig)

             inzone = .false.

             xd = groupcenter(1,j)*invx + half
             xd = xd - floor(xd)

             xd = xd + shx - xf
             if (xd < zero) xd = zero         ! group is behind boundary
             xj = xd*box_a(1)
             if (xj**2 < cutsq) inzone = .true.

             if (inzone) then
                call add_group(group(j), ncoord0, ngroup(1), x_send_ncoord(1,i), x_send_atomind)
             endif

          enddo

          ngroup(2) = ngroup(1)
          x_send_ncoord(2,i) = x_send_ncoord(1,i)

          call get_ez_boundary(homeix-i, homeiy, xf, yf, y_grouped, q_checkgrouped)

          ! Copy the coordinates received from +y to x_send_group
          ngroup0 = zonelist(2)
          do l=1,ny_comm

             shy = zero
             if (homeiy + l > ny) shy = one

             do j=1,y_recv_ngroup(1,l)   ! Loop over coordinates from +y box (ez)
                k = groupl(ngroup0 + j)

                inzone = .false.

                xd = groupcenter(1,k)*invx + half
                yd = groupcenter(2,k)*invy + half
                xd = xd - floor(xd)
                yd = yd - floor(yd)

                ydd = yd
                xd = xd + shx - xf
                yd = yd + shy - yf
                ! Periodic boundaries: (xd, yd) in (-0.5, 0.5)
                if (xd < zero) xd = zero
                if (yd < zero) yd = zero
                ! The following if-statement takes care of the cases where destination box has
                ! higher boundary than the current box i.e. prevents rounding of the ez zone
                ! in the lower right x,y -corner
                ! If x coordinate is behind ez zone, make sure this group is not included
                xj = xd*box_a(1) + yd*box_b(1)
                yj = yd*box_b(2)
                if (xj**2 + yj**2 < cutsq) then
                   inzone = .true.
                elseif (q_checkgrouped) then
                   call group_out(group(k), gtype)
                   if (gtype == 0) then
                      yd = ydd + shy - y_grouped
                      ! Again, following statement prevents rounding
                      if (yd < zero) yd = zero
                      xj = xd*box_a(1) + yd*box_b(1)
                      yj = yd*box_b(2)
                      if (xj**2 + yj**2 < rcut_groupedsq) inzone = .true.
                   endif
                endif

                if (inzone) then
                   call add_group(group(k), ncoord0, ngroup(2), x_send_ncoord(2,i), x_send_atomind)
                endif
             enddo
             ngroup0 = ngroup0 + y_recv_ngroup(1,l)
          enddo

          ngroup(3) = ngroup(2)
          x_send_ncoord(3,i) = x_send_ncoord(2,i)

          call get_ey_boundary(homeix-i, homeiy, homeiz, xf, zf, z_grouped, q_checkgrouped, cut)

          ngroup0 = zonelist(1)
          do l=1,nz_comm

             shz = zero
             if (homeiz + l > nz) shz = one

             do j=1,z_recv_ngroup(1,l)   ! Loop over coordinates from +z box (ey)
                k = groupl(ngroup0 + j)

                inzone = .false.

                xd = groupcenter(1,k)*invx + half
                zd = groupcenter(3,k)*invz + half
                xd = xd - floor(xd)
                zd = zd - floor(zd)

                zdd = zd
                xd = xd + shx - xf
                zd = zd + shz - zf
                ! Minimum image: (xd, zd) in (-0.5, 0.5)
                if (xd < zero) xd = zero     ! group is behind boundary
                if (zd < zero) zd = zero     ! group is behind boundary
                ! The following if-statement takes care of the cases where destination box has
                ! higher boundary than the current box i.e. prevents rounding of the ey zone
                ! in the lower right x,z -corner
                ! If x coordinate is behind ey zone, make sure this group is not included
                xj = xd*box_a(1) + zd*box_c(1)
                zj = zd*box_c(3)
                if (xj**2 + zj**2 < cutsq) then
                   inzone = .true.
                elseif (q_checkgrouped) then
                   call group_out(group(k), gtype)
                   if (gtype == 0) then
                      zd = zdd + shz - z_grouped
                      ! Again, following statement prevents rounding
                      if (zd < zero) zd = zero
                      xj = xd*box_a(1) + zd*box_c(1)
                      zj = zd*box_c(3)
                      if (xj**2 + zj**2 < rcut_groupedsq) inzone = .true.
                   endif
                endif

                if (inzone) then
                   call add_group(group(k), ncoord0, ngroup(3), x_send_ncoord(3,i), x_send_atomind)
                endif
             enddo
             ngroup0 = ngroup0 + z_recv_ngroup(1,l)
          enddo

          ngroup(4) = ngroup(3)
          x_send_ncoord(4,i) = x_send_ncoord(3,i)

          call get_c_boundary(homeix-i, homeiy, homeiz, xf, yf, zf, &
               y_grouped, z_grouped, q_checkgrouped)
          call get_z0_for_c(homeix-i, homeiy, homeiz, z0)

          ngroup0 = zonelist(3)
          do l=1,ny_comm

             shy = zero
             if (homeiy + l > ny) shy = one

             ! Loop over coordinates from +y +z box (c)
             do j=1,y_recv_ngroup(2,l)-y_recv_ngroup(1,l)
                k = groupl(ngroup0 + j)

                inzone = .false.

                xd = groupcenter(1,k)*invx + half
                yd = groupcenter(2,k)*invy + half
                zd = groupcenter(3,k)*invz + half
                xd = xd - floor(xd)
                yd = yd - floor(yd)
                zd = zd - floor(zd)

                shz = zero
                if (zd < z0) shz = one

                ydd = yd
                zdd = zd
                xd = xd + shx - xf
                yd = yd + shy - yf
                zd = zd + shz - zf
                ! Minimum image: (xd, yd, zd) in (-0.5, 0.5)
                if (xd < zero) xd = zero
                if (yd < zero) yd = zero
                if (zd < zero) zd = zero

                ! The following if-statement takes care of the cases where destination box has
                ! higher boundary than the current box i.e. prevents rounding of the c zone
                ! in the lower right corner
                ! If x coordinate is behind c zone, make sure this group is not included
                xj = xd*box_a(1) + yd*box_b(1) + zd*box_c(1)
                yj = yd*box_b(2) + zd*box_c(2)
                zj = zd*box_c(3)

                if (xj**2 + yj**2 + zj**2 < cutsq) then
                   inzone = .true.
                elseif (q_checkgrouped) then
                   call group_out(group(k), gtype)
                   if (gtype == 0) then
                      yd = ydd + shy - y_grouped
                      zd = zdd + shz - z_grouped
                      ! Again, following statement prevents rounding
                      if (yd < zero) yd = zero
                      if (zd < zero) zd = zero
                      xj = xd*box_a(1) + yd*box_b(1) + zd*box_c(1)
                      yj = yd*box_b(2) + zd*box_c(2)
                      zj = zd*box_c(3)
                      if (xj**2 + yj**2 + zj**2 < rcut_groupedsq) inzone = .true.
                   endif
                endif

                if (inzone) then
                   call add_group(group(k), ncoord0, ngroup(4), x_send_ncoord(4,i), x_send_atomind)
                endif
             enddo
             ngroup0 = ngroup0 + (y_recv_ngroup(2,l) - y_recv_ngroup(1,l))
          enddo

          ! Pack atom indices
          call pack_atomind(4, ncoord0, ngroup(1:4), x_send_ncoord(:,i), x_send_atomind, &
               x_send_buf(i)%p, x_send_count)

       endif

       ! Pack coordinates
       call pack_coords(ncoord0, x_send_ncoord(4,i), x_send_atomind, x, y, z, &
            x_send_buf(i)%p, x_send_count)


       ! Send the coordinates
       if (x_send_count > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(x_send_buf(i)%p, x_send_count, mpi_byte,&
               x_send_node(i), coordbuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error sending coordinates from +x direction')
       endif

       ncoord0 = ncoord0 + x_send_ncoord(4,i)
    enddo

    ! Wait for coordinates from +x direction
    if (rezone .and. nx_comm > 0) then
       call recv_groups_coords(nx_comm, x_recv_node, x_recv_buf)
    endif

    if (nreqbuf > 0) then
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error in mpi_waitall from +x direction')
    endif

    ! Copy the coordinates from +x direction into correct arrays
    ncoord0 = 0
    do i=1,nx_comm
       x_recv_count = 0
       if (rezone) then
          call unpack_atomind(4, ncoord0, x_recv_ngroup(:,i), x_recv_ncoord(:,i), &
               x_recv_atomind, x_recv_buf(i)%p, x_recv_count)
          call set_homezone(4, ncoord0, x_recv_ncoord(:,i), x_recv_atomind, &
               x_homezone_val, homezone)
       endif
       call unpack_coords(ncoord0, x_recv_ncoord(4,i), x_recv_atomind, x, y, z, x_recv_buf(i)%p, &
            x_recv_count)
       ncoord0 = ncoord0 + x_recv_ncoord(4,i)
    enddo

    if (rezone) then
       call build_groups(4, nx_comm, x_recv_ngroup, x_recv_ncoord, &
            x_recv_atomind, x, y, z, groupl, atoml, zonelist(4:8), zonelist_atom(4:8))
    endif

    ! Done communicating

    if (rezone) natoml_tot = zonelist_atom(8)

    if (q_test) then
       call test_transfer_coord(natom, x, y, z, cutnb)
    endif

    call calc_recenter_disp(x, y, z, rezone)

    return
  end subroutine transfer_coord

  ! *
  ! * Adds group i to grouplist
  ! *
  subroutine add_group(groupi, ncoord0, ngroup, ncoord, atomind)
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: groupi, ncoord0
    integer, intent(inout) :: ngroup, ncoord
    integer, intent(inout), allocatable, dimension(:) :: atomind
    ! Variables
    integer is, iq, len, ii

    ngroup = ngroup + 1
    is = iand(groupi, Z'FFFFFFF')
    iq = is + iand(ishft(groupi, -28),7)
    if (ncoord + ncoord0 + iq-is+1 > size(atomind)) then
       len = int((ncoord + ncoord0 + iq-is+1)*1.5)
       call chmrealloc('domdec_d2d_comm.src','add_group','atomind',len,intg=atomind)
    endif
    do ii=is,iq
       ncoord = ncoord + 1
       atomind(ncoord + ncoord0) = ii
    enddo
    
    return
  end subroutine add_group

  ! *
  ! * Packs groups and coordinates
  ! * Out:
  ! * buf = packed data
  ! * count = number of bytes in buf
  ! *
  subroutine pack_atomind(n, ncoord0, ngroup, ncoord, atomind, bufp, count)
    use pack_mod,only:pack_int_double
    use memory,only:chmrealloc
    use parallel,only:mpi_integer_size, mpi_real8_size
    implicit none
    ! Input / Ouput
    integer, intent(in) :: n, ncoord0, ngroup(1:n), ncoord(1:n), atomind(:)
    real(chm_real), intent(inout), pointer :: bufp(:)
    integer, intent(inout) :: count
    ! Variables
    integer req_len, buf_align
    
    ! Reallocate y_send_group if neccessary
    ! NOTE: No need to reallocate in pack_coords, because this reallocation guarantees
    !       that the buffer is big enough
    buf_align = mod(ncoord(n)*mpi_integer_size, mpi_real8_size)
    req_len = (ncoord(n) + 2*n)*mpi_integer_size + buf_align + ncoord(n)*3*mpi_real8_size
    if (req_len > size(bufp)*mpi_real8_size) then
       call chmrealloc('domdec_d2d_comm.src','pack_groups_coords','bufp',&
            int(req_len*1.5/mpi_real8_size),mcrlp=bufp)
    endif
    
    ! Pack in number of groups and coordinates
    call pack_int_double(ngroup(1), n, bufp(1), count)
    call pack_int_double(ncoord(1), n, bufp(1), count)

    ! Pack atomind
    call pack_int_double(atomind(ncoord0+1), ncoord(n), bufp(1), count)
    if (count /= (ncoord(n) + 2*n)*mpi_integer_size) then
       call wrndie(-5,'<domdec_d2d_comm>','Invalid group index packing')
    endif
    
    ! Align to mpi_real8_size
    count = count + buf_align

    return
  end subroutine pack_atomind

  ! *
  ! * Packs coordinates
  ! * Out:
  ! * buf = packed data
  ! * count = number of bytes in buf
  ! *
  subroutine pack_coords(ncoord0, ncoord, atomind, x, y, z, bufp, count)
    use parallel,only:mpi_real8_size
    implicit none
    ! Input / Ouput
    integer, intent(in) :: ncoord0, ncoord, atomind(:)
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(inout), pointer :: bufp(:)
    integer, intent(inout) :: count
    ! Variables
    integer i, j, pos

    if (mod(count, mpi_real8_size) /= 0) then
       call wrndie(-5,'<domdec_d2d_comm>','Invalid alignment in pack_coords')
    endif

    pos = count / mpi_real8_size + 1
    do i=1,ncoord
       j = atomind(ncoord0 + i)
       bufp(pos)   = x(j)
       bufp(pos+1) = y(j)
       bufp(pos+2) = z(j)
       pos = pos + 3
    enddo

    count = count + ncoord*3*mpi_real8_size

    return
  end subroutine pack_coords

  ! *
  ! * Packs forces
  ! *
  subroutine pack_forces(ncoord0, ncoord, atomind, forcex, forcey, forcez, bufp, &
       xtmp, ytmp, ztmp)
    implicit none
    ! Input / Ouput
    integer, intent(in) :: ncoord0, ncoord, atomind(:)
    real(chm_real), intent(in) :: forcex(*), forcey(*), forcez(*)
    real(chm_real), intent(inout), pointer :: bufp(:)
    real(chm_real), intent(in), optional :: xtmp(*), ytmp(*), ztmp(*)
    ! Variables
    integer i, j, count

    count = 1
    if (present(xtmp) .and. present(ytmp) .and. present(ztmp)) then
       do i=1,ncoord
          j = atomind(i + ncoord0)
          bufp(count)   = xtmp(j) + forcex(j)
          bufp(count+1) = ytmp(j) + forcey(j)
          bufp(count+2) = ztmp(j) + forcez(j)
          count = count + 3
       enddo
    else
       do i=1,ncoord
          j = atomind(i + ncoord0)
          bufp(count)   = forcex(j)
          bufp(count+1) = forcey(j)
          bufp(count+2) = forcez(j)
          count = count + 3
       enddo
    endif

    return
  end subroutine pack_forces

  ! *
  ! * Receive groups and coordinates
  ! *
  subroutine recv_groups_coords(ncomm, node, buf)
    use mpi,only:mpi_success, mpi_status_size, mpi_byte
    use memory,only:chmrealloc
    use parallel,only:comm_charmm, mpi_real8_size
    implicit none
    ! Input / Output
    integer, intent(in) :: ncomm, node(1:ncomm)
    type(crlpointer_t), intent(inout) :: buf(1:ncomm)
    ! Variables
    integer i, j, len
    logical flag
    integer status(mpi_status_size), ierror

    recv_flag(1:ncomm) = .false.
    do i=1,ncomm
       flag = .false.
       j = 0
       do while (.not.flag)
          j = j + 1
          if (j > ncomm) j = 1
          if (.not.recv_flag(j)) then
             call mpi_iprobe(node(j), coordbuf, comm_charmm, flag, status, ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec_d2d_comm>','Error in mpi_iprobe')
          endif
       enddo
       recv_flag(j) = .true.
       ! Get the size in bytes
       call mpi_get_count(status, mpi_byte, len, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error in mpi_get_count')
       ! Resize buffer size if needed
       if (len > size(buf(j)%p)*mpi_real8_size) then
          call chmrealloc('domdec_d2d_comm.src','transfer_coord',&
               'buf',int(len*1.5/mpi_real8_size),mcrlp=buf(j)%p)
       endif
       call mpi_recv(buf(j)%p, len, mpi_byte,&
            node(j), coordbuf, comm_charmm, status, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error in mpi_recv')
    enddo
    
    return
  end subroutine recv_groups_coords

  ! *
  ! * Unpacks atom indices
  ! *
  subroutine unpack_atomind(n, ncoord0, ngroup, ncoord, atomind, bufp, count)
    use pack_mod,only:unpack_int_double
    use memory,only:chmrealloc
    use parallel,only:mpi_integer_size, mpi_real8_size
    implicit none
    ! Input / Output
    integer, intent(in) :: n, ncoord0
    integer, intent(inout) :: ngroup(1:n), ncoord(1:n)
    integer, intent(inout), allocatable, dimension(:) :: atomind
    real(chm_real), intent(in), pointer :: bufp(:)
    integer, intent(inout) :: count
    ! Variables
    integer i, j
    integer buf_align

    ! Unpack number of groups and coordinates from the beginning
    call unpack_int_double(bufp(1), count, ngroup(1), n)
    call unpack_int_double(bufp(1), count, ncoord(1), n)
    do i=1,n-1
       if (ngroup(i) > ngroup(i+1) .or. ngroup(i) < 0) then
          call wrndie(-5,'<domdec_d2d_comm>','Error in ngroup')
       endif
       if (ncoord(i) > ncoord(i+1) .or. ncoord(i) < 0) then
          call wrndie(-5,'<domdec_d2d_comm>','Error in ncoord')
       endif
    enddo

    ! Reallocate atomind if neccessary
    ! NOTE: no need to reallocate atomind in unpack_coords because the realloc here
    !       guarantees that the buffer if large enough
    if (ncoord(n) + ncoord0 > size(atomind)) then
       call chmrealloc('domdec_d2d_comm.src','unpack_atomind','atomind',&
            int((ncoord(n)+ncoord0)*1.5),intg=atomind)
    endif

    buf_align = mod(ncoord(n)*mpi_integer_size, mpi_real8_size)

    ! Unpack atomind
    call unpack_int_double(bufp(1), count, atomind(ncoord0+1), ncoord(n))
    if (count /= (ncoord(n) + 2*n)*mpi_integer_size) then
       call wrndie(-5,'<domdec_d2d_comm>','Error unpacking atom index')
    endif

    ! Align to mpi_real8_size
    count = count + buf_align

    return
  end subroutine unpack_atomind

  ! *
  ! * Sets homezone using atomind
  ! *
  subroutine set_homezone(n, ncoord0, ncoord, atomind, homezone_val, homezone)
    implicit none
    ! Input / Output
    integer, intent(in) :: n, ncoord0, ncoord(1:n), atomind(:), homezone_val(1:n)
    integer, intent(out) :: homezone(:)
    ! Variables
    integer i, j, k, ncoord_prev

    do j=1,n
       if (j == 1) then
          ncoord_prev = 0
       else
          ncoord_prev = ncoord(j-1)
       endif
       do i=ncoord_prev+1,ncoord(j)
          k = atomind(ncoord0 + i)
          homezone(k) = homezone_val(j)
       enddo
    enddo

    return
  end subroutine set_homezone

  ! *
  ! * Unpacks coordinates
  ! *
  subroutine unpack_coords(ncoord0, ncoord, atomind, x, y, z, bufp, count)
    use parallel,only:mpi_real8_size
    implicit none
    ! Input / Output
    integer, intent(in) :: ncoord0, ncoord, atomind(:)
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    real(chm_real), intent(in), pointer :: bufp(:)
    integer, intent(in) :: count
    ! Variables
    integer i, j, pos
    
    if (mod(count, mpi_real8_size) /= 0) then
       call wrndie(-5,'<domdec_d2d_comm>','Invalid alignment in unpack_coords')
    endif

    pos = count / mpi_real8_size + 1
    do i=1,ncoord
       j = atomind(ncoord0 + i)
       x(j) = bufp(pos)
       y(j) = bufp(pos+1)
       z(j) = bufp(pos+2)
       pos = pos + 3
    enddo

    return
  end subroutine unpack_coords

  ! *
  ! * Calculates group i center and box size
  ! *
  subroutine calc_groupbox_loc(i, x, y, z)
    use number,only:half
    use groupxfast,only:group, groupcenter, groupbox
    implicit none
    ! Input / Output
    integer, intent(in) :: i
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer is, iq
    real(chm_real) xmin, ymin, zmin, xmax, ymax, zmax

    is = iand(group(i), Z'FFFFFFF')
    iq = is + iand(ishft(group(i), -28),7)

    xmin = minval(x(is:iq))
    ymin = minval(y(is:iq))
    zmin = minval(z(is:iq))
    xmax = maxval(x(is:iq))
    ymax = maxval(y(is:iq))
    zmax = maxval(z(is:iq))

    groupcenter(1,i) = half*(xmin + xmax)
    groupcenter(2,i) = half*(ymin + ymax)
    groupcenter(3,i) = half*(zmin + zmax)

    groupbox(1,i) = xmax - groupcenter(1,i)
    groupbox(2,i) = ymax - groupcenter(2,i)
    groupbox(3,i) = zmax - groupcenter(3,i)

    return
  end subroutine calc_groupbox_loc

  ! *
  ! * Calculates group i center and box size
  ! *
  subroutine calc_groupbox_loc2(i, x, y, z, is, iq)
    use number,only:half
    use groupxfast,only:group, groupcenter, groupbox
    implicit none
    ! Input / Output
    integer, intent(in) :: i
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(out) :: is, iq
    ! Variables
    real(chm_real) xmin, ymin, zmin, xmax, ymax, zmax

    is = iand(group(i), Z'FFFFFFF')
    iq = is + iand(ishft(group(i), -28),7)

    xmin = minval(x(is:iq))
    ymin = minval(y(is:iq))
    zmin = minval(z(is:iq))
    xmax = maxval(x(is:iq))
    ymax = maxval(y(is:iq))
    zmax = maxval(z(is:iq))

    groupcenter(1,i) = half*(xmin + xmax)
    groupcenter(2,i) = half*(ymin + ymax)
    groupcenter(3,i) = half*(zmin + zmax)

    groupbox(1,i) = xmax - groupcenter(1,i)
    groupbox(2,i) = ymax - groupcenter(2,i)
    groupbox(3,i) = zmax - groupcenter(3,i)

    return
  end subroutine calc_groupbox_loc2

  ! *
  ! * Builds group list (groupl) and calculates groupbox
  ! *
  subroutine build_groups(n, ncomm, ngroup, ncoord, atomind, x, y, z, groupl, atoml, &
       zonelist, zonelist_atom)
    use groupxfast,only:invgroup
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: n, ncomm, ngroup(n,ncomm), ncoord(n, ncomm), atomind(:)
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(inout), allocatable, dimension(:) :: groupl
    integer, intent(inout), allocatable, dimension(:) :: atoml
    integer, intent(inout) :: zonelist(n+1), zonelist_atom(n+1)
    ! Variables
    integer i, j, k, l, igroup, igroup_prev, iatom, is, iq, groupl_len, atoml_len
    integer ncoord0, ncoord_prev, ngroup_prev, ii

    ! Reallocate groupl if neccessary
    groupl_len = zonelist(1)
    if (ncomm > 0) then
       groupl_len = groupl_len + sum(ngroup(n,1:ncomm))
    endif
    if (groupl_len > size(groupl)) then
       call chmrealloc('domdec_d2d_comm.src','build_groups','groupl',&
            int(1.5*groupl_len),intg=groupl)
    endif

    ! Reallocate atoml if neccessary
    atoml_len = zonelist_atom(1)
    if (ncomm > 0) then
       atoml_len = atoml_len + sum(ncoord(n,1:ncomm))
    endif
    if (atoml_len > size(atoml)) then
       call chmrealloc('domdec_d2d_comm.src','build_groups','atoml',&
            int(1.5*atoml_len),intg=atoml)
    endif

    do k=1,n
       igroup = zonelist(k)
       iatom = zonelist_atom(k)
       ncoord0 = 0
       do i=1,ncomm
          if (k == 1) then
             ncoord_prev = 0
             ngroup_prev = 0
          else
             ncoord_prev = ncoord(k-1,i)
             ngroup_prev = ngroup(k-1,i)
          endif
          igroup_prev = igroup
          j = ncoord_prev + 1
          do while (j <= ncoord(k,i))
             l = invgroup(atomind(ncoord0 + j))
             call calc_groupbox_loc2(l, x, y, z, is, iq)
             igroup = igroup + 1
             groupl(igroup) = l
             !atoml(iatom+1:iatom+iq-is+1) = (/ is:iq /)
             do ii=0,iq-is
                atoml(iatom+ii+1) = is + ii
             enddo
             j = j + (iq-is+1)
             iatom = iatom + (iq-is+1)
          enddo
          if (igroup-igroup_prev /= ngroup(k,i)-ngroup_prev) then
             call wrndie(-5,'<domdec_d2d_comm>','Invalid igroup count')
          endif
          ncoord0 = ncoord0 + ncoord(n,i)
          !iatom = iatom + (ncoord(k,i) - ncoord_prev)
       enddo
       zonelist(k+1) = igroup
       zonelist_atom(k+1) = iatom
    enddo

    if (zonelist(n+1) /= groupl_len) then
       call wrndie(-5,'<domdec_d2d_comm>','Invalid zonelist in build_groups')
    endif

    if (zonelist_atom(n+1) /= atoml_len) then
       call wrndie(-5,'<domdec_d2d_comm>','Invalid zonelist_atom in build_groups')
    endif

    return
  end subroutine build_groups

  ! *
  ! * Tests transfer_coord() -subroutine
  ! *
  subroutine test_transfer_coord(natom, x, y, z, rcut)
    use groupxfast,only:group_out, group
    use number,only:zero
    use stream,only:outu, prnlev
    use domdec_dlb,only:get_zone_corners
    use domdec_common,only:homezone, boxx, boxy, boxz, &
         zonelist, zonelist_atom, groupl, natoml_tot, &  
         hboxx, hboxy, hboxz, atoml
    use memory
    use parallel,only:mynod
    implicit none
    ! Input / Output
    integer, intent(in) :: natom
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    real(chm_real), intent(in) :: rcut
    ! Variables
    real(chm_real), allocatable, dimension(:) :: xt, yt, zt
    real(chm_real) x0, y0, z0, x1, y1, z1
    real(chm_real) dx, dy, dz
    real(chm_real) rcutsq, rsq
    integer i, izone, ig, ig_start, ig_end, iatom, is, iq, ii

    ! Check that zonelist and zonelist_atom match
    iatom = 0
    do izone=1,8
       if (izone > 1) then
          ig_start = zonelist(izone-1) + 1
       else
          ig_start = 1
       endif
       ig_end = zonelist(izone)       
       do ig=ig_start,ig_end
          call group_out(group(groupl(ig)), is, iq)
          do ii=0,iq-is
             if (atoml(iatom+ii+1) /= is + ii) then
                call wrndie(-5,'<domdec_d2d_comm>',&
                     'test_transfer_coord: groupl and atoml do not match')
             endif
          enddo
          iatom = iatom + (iq-is+1)
       enddo
       if (iatom /= zonelist_atom(izone)) then
          write (outu,'(a,i2,2i12)') 'izone,iatom,zonelist_atom(izone)=',&
               izone,iatom,zonelist_atom(izone)
          call wrndie(-5,'<domdec_d2d_comm>',&
               'test_transfer_coord: zonelist does not match zonelist_atom')
       endif
    enddo
    if (iatom /= natoml_tot) then
       call wrndie(-5,'<domdec_d2d_comm>','test_transfer_coord: Incorrect total number of atoms')
    endif

    rcutsq = rcut*rcut

    call chmalloc('domdec_d2d_comm.src','test_transfer_coord','xt',natom,crl=xt)
    call chmalloc('domdec_d2d_comm.src','test_transfer_coord','yt',natom,crl=yt)
    call chmalloc('domdec_d2d_comm.src','test_transfer_coord','zt',natom,crl=zt)

    xt(1:natom) = x(1:natom)
    yt(1:natom) = y(1:natom)
    zt(1:natom) = z(1:natom)
    call copy_to_all(xt, yt, zt)

    ! (x1, y1, z1) = right boundary of zone I
    call get_zone_corners(1, rcut, x0, y0, z0, x1, y1, z1)
    
    ! Make sure this node has all the coordinates
    do i=1,natom
       if (homezone(i) == 0) then
          ! Atom i is not on this node => Check to make sure it should not be!
          dx = (x(i) + hboxx) - x1
          dy = (y(i) + hboxy) - y1
          dz = (z(i) + hboxz) - z1
          if (dx < -hboxx) dx = dx + boxx
          if (dy < -hboxy) dy = dy + boxy
          if (dz < -hboxz) dz = dz + boxz
          if (dx < zero .or. dy < zero .or. dz < zero) cycle
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rcutsq) then
             write (outu,'(a,i3,i8)') 'mynod,i=',mynod,i
             write (outu,'(a,6f10.3)') 'x,y,z, x1,y1,z1=',x(i),y(i),z(i),x1,y1,z1
             write (outu,'(a,3f10.3)') 'hboxx,hboxy,hboxz=',hboxx,hboxy,hboxz
             write (outu,'(a,3f10.3)') 'dx,dy,dz=',dx,dy,dz
             call wrndie(-5,'<domdec_d2d_comm>','test_transfer_coord: Missing a coordinate')
          endif
       elseif (homezone(i) > 1) then
          if (isnan(x(i)) .or. isnan(y(i)) .or. isnan(z(i))) then
             call wrndie(-5,'<domdec_d2d_comm>','test_transfer_coord: NaN coordinate')
          endif
          ! Atom i is on this node => Check that the coordinates are set correctly
          if (x(i) /= xt(i) .or. y(i) /= yt(i) .or. z(i) /= zt(i)) then
             write (outu,'(a,i4,i8)') 'mynod,i=',mynod,i
             write (outu,'(a,3f10.3)') 'x,y,z   =',x(i),y(i),z(i)
             write (outu,'(a,3f10.3)') 'xt,yt,zt=',xt(i),yt(i),zt(i)
             call wrndie(-5,'<domdec_d2d_comm>','test_transfer_coord: Coordinates set incorrectly')
          endif
       endif
    enddo

    ! Copy coordinates to (x, y, z). Now all direct nodes have all the coorinates.
    x(1:natom) = xt(1:natom)
    y(1:natom) = yt(1:natom)
    z(1:natom) = zt(1:natom)

    call chmdealloc('domdec_d2d_comm.src','test_transfer_coord','xt',natom,crl=xt)
    call chmdealloc('domdec_d2d_comm.src','test_transfer_coord','yt',natom,crl=yt)
    call chmdealloc('domdec_d2d_comm.src','test_transfer_coord','zt',natom,crl=zt)

    if (prnlev > 2) write (outu,'(a)') 'test_transfer_coord OK'

    return
  end subroutine test_transfer_coord

  ! *
  ! * Calculate re-center displacements for ns_grid routine
  ! *
  subroutine calc_recenter_disp(x, y, z, rezone)
    use number,only:zero, two, rbig
    use groupxfast,only:group, groupcenter, groupbox, grouploc, &
         maxbox, groupsh, group_out
    use image,only:xtlabc
    use domdec_common,only:groupl, zonelist, homeix, homeiy, homeiz, frx, fry, frz, &
         nx_comm, ny_comm, box_a, box_b, box_c
    use domdec_dlb,only:q_load_balance, get_nodebx, get_nodeby, get_nodebz, topx, topy, topz, &
         write_nodeb
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    logical, intent(in) :: rezone
    ! Variables
    real(chm_real) xyzo(3)
    real(chm_real) xtlinv(6)
    integer i, j
    logical ok

    call invt33s(xtlinv,xtlabc,ok)

    ! Origin of home box in fractional coordinates
    if (q_load_balance) then
       xyzo(1:3) = zero
       if (homeix > 1) xyzo(1) = get_nodebx(homeix-1)
       if (homeiy > 1) then
          xyzo(2) = two
          do i=0,nx_comm
             xyzo(2) = min(xyzo(2),get_nodeby(homeiy-1,homeix+i))
          enddo
       endif
       if (homeiz > 1) then
          xyzo(3) = two
          do i=0,nx_comm
             do j=0,ny_comm
                xyzo(3) = min(xyzo(3),get_nodebz(homeiz-1,homeiy+j,homeix+i))
             enddo
          enddo
       endif
    else
       xyzo(1) = (homeix-1)*frx
       xyzo(2) = (homeiy-1)*fry
       xyzo(3) = (homeiz-1)*frz
    endif

    maxbox(1:3) = zero
    topx = -rbig
    topy = -rbig
    topz = -rbig

    if (rezone) then
!$omp parallel reduction(max:topx,topy,topz,maxbox)
       call calc_recenter_disp_kernel(1, zonelist(8), groupl, q_load_balance, xyzo, xtlinv, &
            box_a, box_b, box_c, groupcenter, groupbox, groupsh, grouploc,&
            topx, topy, topz, maxbox)
!$omp end parallel
    endif

    return
  end subroutine calc_recenter_disp

  ! *
  ! * Kernel for the above
  ! *
  subroutine calc_recenter_disp_kernel(jstart, jend, groupl, q_load_balance, &
       xyzo, xtlinv, box_a, box_b, box_c, groupcenter, groupbox, groupsh, grouploc, &
       topx, topy, topz, maxbox)
    use number,only:half
    use stream,only:outu
    use parallel,only:mynod
    use domdec_common,only:homeix, homeiy, homeiz, nx, ny, nz, nx_comm, ny_comm, nz_comm, &
         frx, fry, frz
    use domdec_dlb,only:nodebx, nodeby, nodebz, write_nodeb
    implicit none
    ! Input / Output
    integer, intent(in) :: jstart, jend, groupl(:)
    logical, intent(in) :: q_load_balance
    real(chm_real), intent(in) :: xyzo(3), xtlinv(6), groupcenter(:,:), groupbox(:,:)
    real(chm_real), intent(in) :: box_a(3), box_b(3), box_c(3)
    real(chm_real), intent(inout) :: groupsh(:,:)
    integer, intent(inout) :: grouploc(:)
    real(chm_real), intent(inout) :: topx, topy, topz, maxbox(3)
    ! Variables
    integer i, j
    real(chm_real) x, y, z, shx, shy, shz
    real(chm_real) groupdisp(3)
    integer ix, iy, iz, dix, diy, diz, bx, by, bz

!$omp do schedule(static)
    do j=jstart,jend
       i = groupl(j)
       ! Calculate groupcenter and groupbox
       if (isnan(groupcenter(1,i)) .or. isnan(groupcenter(1,i)) .or. &
            isnan(groupcenter(1,i))) then
          if (q_load_balance) call write_nodeb()
          write (outu,'(a,i4,2i8)') 'mynod,j,i=',mynod,j,i
          call wrndie(-5,'<domdec_d2d_comm>','NaN coordinate')
       endif
       maxbox(1) = max(maxbox(1),groupbox(1,i))
       maxbox(2) = max(maxbox(2),groupbox(2,i))
       maxbox(3) = max(maxbox(3),groupbox(3,i))
       
       ! (x, y, z) = fractional coordinates of the group center
       x = groupcenter(1,i)*xtlinv(1) + groupcenter(2,i)*xtlinv(2) + &
            groupcenter(3,i)*xtlinv(4) + half
       y = groupcenter(1,i)*xtlinv(2) + groupcenter(2,i)*xtlinv(3) + &
            groupcenter(3,i)*xtlinv(5) + half
       z = groupcenter(1,i)*xtlinv(4) + groupcenter(2,i)*xtlinv(5) + &
            groupcenter(3,i)*xtlinv(6) + half

!       x = groupcenter(1,i)*xtlinv(1) + half
!       y = groupcenter(2,i)*xtlinv(3) + half
!       z = groupcenter(3,i)*xtlinv(6) + half
       
       shx = ceiling(-(x - xyzo(1)))
       shy = ceiling(-(y - xyzo(2)))
       shz = ceiling(-(z - xyzo(3)))
       
       ! Calculate displacement in coordinate
       groupsh(1,i) = shx
       groupsh(2,i) = shy
       groupsh(3,i) = shz
       groupdisp(1) = groupsh(1,i)*box_a(1) + groupsh(2,i)*box_b(1) + groupsh(3,i)*box_c(1)
       groupdisp(2) = groupsh(2,i)*box_b(2) + groupsh(3,i)*box_c(2)
       groupdisp(3) = groupsh(3,i)*box_c(3)
       
       topx = max(topx, groupcenter(1,i) + groupdisp(1))
       topy = max(topy, groupcenter(2,i) + groupdisp(2))
       topz = max(topz, groupcenter(3,i) + groupdisp(3))

       x = x - floor(x)
       y = y - floor(y)
       z = z - floor(z)
       
       if (q_load_balance) then

          ix = homeix
          do while (x < nodebx(ix))
             ix = ix - 1
          enddo
          do while (x >= nodebx(ix+1))
             ix = ix + 1
          enddo
          dix = ix - homeix
          if (dix < -nx_comm) dix = dix + nx
          if (dix > nx_comm) dix = dix - nx
          if (dix < -nx_comm .or. dix > nx_comm) then
             call wrndie(-5,'<domdec_d2d_comm>','Coordinate outside communication region X')
          endif
          
          iy = homeiy
          do while (y < nodeby(iy,dix))
             iy = iy - 1
          enddo
          do while (y >= nodeby(iy+1,dix))
             iy = iy + 1
          enddo
          diy = iy - homeiy
          if (diy < -ny_comm) diy = diy + ny
          if (diy > ny_comm) diy = diy - ny
          if (diy < -ny_comm .or. diy > ny_comm) then
             call wrndie(-5,'<domdec_d2d_comm>','Coordinate outside communication region Y')
          endif
          
          iz = homeiz
          do while (z < nodebz(iz,diy,dix))
             iz = iz - 1
          enddo
          do while (z >= nodebz(iz+1,diy,dix))
             iz = iz + 1
          enddo
          diz = iz - homeiz
          if (diz < -nz_comm) diz = diz + nz
          if (diz > nz_comm) diz = diz - nz
          if (diz < -nz_comm .or. diz > nz_comm) then
             call wrndie(-5,'<domdec_d2d_comm>','Coordinate outside communication region Z')
          endif
       else
          dix = int(x/frx) + 1 - homeix
          diy = int(y/fry) + 1 - homeiy
          diz = int(z/frz) + 1 - homeiz
          
          bx = (iabs(dix) > 1)
          dix = dix - nx*sign(bx, dix)
          
          by = (iabs(diy) > 1)
          diy = diy - ny*sign(by, diy)
          
          bz = (iabs(diz) > 1)
          diz = diz - nz*sign(bz, diz)
       endif

       bx = (dix == 0)    ! These return -1 for "true" and 0 for "false"
       by = (diy == 0)
       bz = (diz == 0)

       grouploc(i) = ishft(iand(bz,1), 2) + ishft(iand(by,1), 1) + iand(bx,1)
       
    enddo
!$omp end do

    return
  end subroutine calc_recenter_disp_kernel

  ! *
  ! * Exports forces to nodes
  ! *
  subroutine transfer_force(forcex, forcey, forcez)
    use mpi,only:mpi_real8, mpi_success, mpi_statuses_ignore, mpi_status_ignore
    use parallel,only:comm_charmm
    use domdec_common,only:nx_comm, ny_comm, nz_comm, q_gpu, q_test
#if KEY_DOMDEC_GPU==1
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    real(chm_real) forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, ierror, nreqbuf, ncoord0
    integer j

    !call mpi_barrier(comm_charmm, ierror)

    if (q_test) then
       call test_transfer_force(1, forcex, forcey, forcez)
    endif

    ! Send forces to boxes in +x direction
    ncoord0 = 0
    do i=1,nx_comm
       call pack_forces(ncoord0, x_recv_ncoord(4,i), x_recv_atomind, &
            forcex, forcey, forcez, x_recv_buf(i)%p)
       ncoord0 = ncoord0 + x_recv_ncoord(4,i)
    enddo

    ! Send and receive forces in x direction
    nreqbuf = 0
    do i=1,nx_comm
       if (x_send_ncoord(4,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_irecv(x_send_buf(i)%p, 3*x_send_ncoord(4,i), mpi_real8,&
               x_send_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error start receiving forces from -x direction')
       endif
       if (x_recv_ncoord(4,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(x_recv_buf(i)%p, 3*x_recv_ncoord(4,i), mpi_real8,&
               x_recv_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','error start sending forces from -x direction')
       endif
    enddo

    ! wait for forces from -x direction
    if (nreqbuf > 0) then
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error receiving forces from -x direction')
    endif

    ! Put forces from -x direction into a temporary array
    ncoord0 = 0
    do i=1,nx_comm
       call unpack_force2(ncoord0, 1, x_send_ncoord(4,i), &
            x_send_atomind, x_send_buf(i)%p, xtmp, ytmp, ztmp)
       ncoord0 = ncoord0 + x_send_ncoord(4,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Pack (y)')
#endif

    ! Send forces to boxes in +y direction
    ncoord0 = 0
    do i=1,ny_comm
       call pack_forces(ncoord0, y_recv_ncoord(2,i), y_recv_atomind, &
            forcex, forcey, forcez, y_recv_buf(i)%p, xtmp, ytmp, ztmp)
       ncoord0 = ncoord0 + y_recv_ncoord(2,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Send and receive forces in y direction
    nreqbuf = 0
    do i=1,ny_comm
       if (y_send_ncoord(2,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_irecv(y_send_buf(i)%p, 3*y_send_ncoord(2,i), mpi_real8, &
               y_send_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error start receiving forces from -y direction')
       endif
       if (y_recv_ncoord(2,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(y_recv_buf(i)%p, 3*y_recv_ncoord(2,i), mpi_real8, &
               y_recv_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error start sending forces from -y direction')
       endif
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_waitall (y)')
#endif

    ! Wait for forces from -y direction
    if (nreqbuf > 0) then
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error receiving forces from -y direction')
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

!!$    do i=1,ny_comm
!!$       if (y_send_ncoord(2,i) > 0 .and. y_recv_ncoord(2,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Sendrecv (y)')
!!$#endif
!!$          call mpi_sendrecv(y_recv_buf(i)%p, 3*y_recv_ncoord(2,i), mpi_real8, &
!!$               y_recv_node(i), 100, &
!!$               y_send_buf(i)%p, 3*y_send_ncoord(2,i), mpi_real8, &
!!$               y_send_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_sendrecv')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       elseif (y_send_ncoord(2,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Rrecv (y)')
!!$#endif
!!$          call mpi_recv(y_send_buf(i)%p, 3*y_send_ncoord(2,i), mpi_real8, &
!!$               y_send_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_recv')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       elseif (y_recv_ncoord(2,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Send (y)')
!!$#endif
!!$          call mpi_send(y_recv_buf(i)%p, 3*y_recv_ncoord(2,i), mpi_real8, &
!!$               y_recv_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_send')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       endif
!!$    enddo


#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('unpack_force (y)')
#endif

    ! Put forces from -y direction into a temporary array
    ncoord0 = 0
    do i=1,ny_comm
       call unpack_force2(ncoord0, 1, y_send_ncoord(2,i), &
            y_send_atomind, y_send_buf(i)%p, xtmp, ytmp, ztmp)
       ncoord0 = ncoord0 + y_send_ncoord(2,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('pack_force (z)')
#endif

    ! Send forces to boxes in +z direction
    ncoord0 = 0
    do i=1,nz_comm
       call pack_forces(ncoord0, z_recv_ncoord(1,i), z_recv_atomind, &
            forcex, forcey, forcez, z_recv_buf(i)%p, xtmp, ytmp, ztmp)
       ncoord0 = ncoord0 + z_recv_ncoord(1,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Send and receive forces in z direction
    nreqbuf = 0
    do i=1,nz_comm
       if (z_send_ncoord(1,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_irecv(z_send_buf(i)%p, 3*z_send_ncoord(1,i), mpi_real8,&
               z_send_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error start receiving forces from -z direction')
       endif
       if (z_recv_ncoord(1,i) > 0) then
          nreqbuf = nreqbuf + 1
          call mpi_isend(z_recv_buf(i)%p, 3*z_recv_ncoord(1,i), mpi_real8,&
               z_recv_node(i), forcebuf, comm_charmm, reqbuf(nreqbuf), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec_d2d_comm>','Error start sending forces from -z direction')
       endif
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_waitall (z)')
#endif

    ! wait for forces from -z direction
    if (nreqbuf > 0) then
       call mpi_waitall(nreqbuf, reqbuf, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec_d2d_comm>','Error receiving forces from -z direction')
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

!!$    do i=1,nz_comm
!!$       if (z_send_ncoord(1,i) > 0 .and. z_recv_ncoord(1,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Sendrecv (z)')
!!$#endif
!!$          call mpi_sendrecv(z_recv_buf(i)%p, 3*z_recv_ncoord(1,i), mpi_real8, &
!!$               z_recv_node(i), 100, &
!!$               z_send_buf(i)%p, 3*z_send_ncoord(1,i), mpi_real8, &
!!$               z_send_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_sendrecv')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       elseif (z_send_ncoord(1,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Rrecv (z)')
!!$#endif
!!$          call mpi_recv(z_send_buf(i)%p, 3*z_send_ncoord(1,i), mpi_real8, &
!!$               z_send_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_recv')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       elseif (z_recv_ncoord(1,i) > 0) then
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_start('Send (z)')
!!$#endif
!!$          call mpi_send(z_recv_buf(i)%p, 3*z_recv_ncoord(1,i), mpi_real8, &
!!$               z_recv_node(i), 100, comm_charmm, mpi_status_ignore, ierror)
!!$          if (ierror /= mpi_success) call wrndie(-5, '<domdec_d2d_comm>','Error in mpi_send')
!!$#if KEY_DOMDEC_GPU==1
!!$          if (q_gpu) call range_stop()
!!$#endif
!!$       endif
!!$    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Unpack force')
#endif

    ! Put forces from -z direction into correct arrays
    ncoord0 = 0
    do i=1,nz_comm
       call unpack_force2(ncoord0, 1, z_send_ncoord(1,i), &
            z_send_atomind, z_send_buf(i)%p, forcex, forcey, forcez)
       ncoord0 = ncoord0 + z_send_ncoord(1,i)
    enddo

    ! Put forces from -y direction into correct arrays
    ncoord0 = 0
    do i=1,ny_comm
       call unpack_force2(ncoord0, 1, y_send_ncoord(2,i), &
            y_send_atomind, y_send_buf(i)%p, forcex, forcey, forcez)
       ncoord0 = ncoord0 + y_send_ncoord(2,i)
    enddo

    ! Put forces from -x direction into correct arrays
    ncoord0 = 0
    do i=1,nx_comm
       call unpack_force2(ncoord0, 1, x_send_ncoord(4,i), &
            x_send_atomind, x_send_buf(i)%p, forcex, forcey, forcez)
       ncoord0 = ncoord0 + x_send_ncoord(4,i)
    enddo

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Zero temp arrays')
#endif

    ! Zero temporary arrays
    if (nx_comm > 0) then
       call clear_force(sum(x_send_ncoord(4,1:nx_comm)), x_send_atomind, xtmp, ytmp, ztmp)
    endif

    if (ny_comm > 0) then
       call clear_force(sum(y_send_ncoord(2,1:ny_comm)), y_send_atomind, xtmp, ytmp, ztmp)
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    if (q_test) then
       call test_transfer_force(2, forcex, forcey, forcez)
    endif

    return
  end subroutine transfer_force

  ! *
  ! * Test correctness of transfer_force
  ! *
  subroutine test_transfer_force(f, forcex, forcey, forcez)
    use number,only:zero
    use stream,only:outu, prnlev
    use psf,only:natom
    use memory,only:chmalloc, chmdealloc
    use domdec_common,only:homezone
    use mpi,only:mpi_real8, mpi_sum, mpi_success
    use parallel,only:comm_charmm, mynod
    implicit none
    ! Input / Output
    integer, intent(in) :: f
    real(chm_real), intent(in) :: forcex(*), forcey(*), forcez(*)
    ! Variables
    integer i, ierror
    real(chm_real), allocatable, dimension(:) :: fxt, fyt, fzt
    real(chm_real), save, allocatable, dimension(:) :: fx, fy, fz
    real(chm_real) dx, dy, dz

    if (f == 1) then
       ! Calculate correct reference forces
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fx)
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fy)
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fz)
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fxt)
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fyt)
       call chmalloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fzt)

       do i=1,natom
          if (homezone(i) == 0) then
             fxt(i) = zero
             fyt(i) = zero
             fzt(i) = zero
          else
             fxt(i) = forcex(i)
             fyt(i) = forcey(i)
             fzt(i) = forcez(i)
          endif
       enddo

       call mpi_allreduce(fxt, fx, natom, mpi_real8, mpi_sum, comm_charmm, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_d2d_comm>','test_transfer_force, error calling mpi_allreduce')
       endif

       call mpi_allreduce(fyt, fy, natom, mpi_real8, mpi_sum, comm_charmm, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_d2d_comm>','test_transfer_force, error calling mpi_allreduce')
       endif

       call mpi_allreduce(fzt, fz, natom, mpi_real8, mpi_sum, comm_charmm, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_d2d_comm>','test_transfer_force, error calling mpi_allreduce')
       endif

       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fxt)
       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fyt)
       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fxt',natom,crl=fzt)

    else

       do i=1,natom
          if (homezone(i) == 1) then
             dx = abs(fx(i)-forcex(i))
             dy = abs(fy(i)-forcey(i))
             dz = abs(fz(i)-forcez(i))
             if (dx > 1.0e-8 .or. dy > 1.0e-8 .or. dz > 1.0e-8) then
                call wrndie(-5,'<domdec_d2d_comm>','test_transfer_force FAILED')
             endif
          endif
       enddo

       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fx)
       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fy)
       call chmdealloc('domdec_d2d_comm.src','test_transfer_force','fx',natom,crl=fz)

       if (prnlev > 2) write (outu,'(a)') 'test_transfer_force OK'

    endif

    return
  end subroutine test_transfer_force

  ! *
  ! * Unpacks and adds force to (fx, fy, fz)
  ! *
  subroutine unpack_force2(ncoord0, coord_start, coord_end, atomind, buf, fx, fy, fz)
    implicit none
    ! Input / Output
    integer, intent(in) :: ncoord0, coord_start, coord_end
    integer, intent(in) :: atomind(:)
    real(chm_real), intent(in) :: buf(:)
    real(chm_real), intent(inout) :: fx(*), fy(*), fz(*)
    ! Variables
    integer i, j, pos

    pos = 1
    do i=coord_start,coord_end
       j = atomind(ncoord0 + i)
       fx(j) = fx(j) + buf(pos)
       fy(j) = fy(j) + buf(pos+1)
       fz(j) = fz(j) + buf(pos+2)
       pos = pos + 3
    enddo

    return
  end subroutine unpack_force2

  ! *
  ! * Unpacks and adds force to (fx, fy, fz)
  ! *
  subroutine clear_force(n, atomind, fx, fy, fz)
    use number,only:zero
    implicit none
    ! Input / Output
    integer, intent(in) :: n, atomind(:)
    real(chm_real), intent(out) :: fx(*), fy(*), fz(*)
    ! Variables
    integer i, j

    do i=1,n
       j = atomind(i)
       fx(j) = zero
       fy(j) = zero
       fz(j) = zero
    enddo

    return
  end subroutine clear_force

  ! *
  ! *
  ! *
  subroutine get_neighsend_dlb(x, y, z, nneighsend, neighind_dest, ncoord_send)
    use stream,only:outu
    use number,only:zero, half
    use domdec_common,only:zonelist, groupl, invx, invy, invz, homeix, homeiy, homeiz, nx, ny, nz, &
         neighind
    use domdec_dlb,only:nodefrx, nodefry, nodefrz
    use groupxfast,only:groupcenter
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(inout) :: nneighsend(26), neighind_dest(:), ncoord_send(26)
    ! Variables
    integer ig, i, nod, is, iq
    integer ix, iy, iz, dix, diy, diz
    real(chm_real) xf, yf, zf
    real(chm_real) xd, yd, zd

!$omp do schedule(static)
    do ig=1,zonelist(1)
       i = groupl(ig)

       ! Calculate group center of mass
       call calc_groupbox_loc2(i, x, y, z, is, iq)

       xf = groupcenter(1,i)*invx + half
       yf = groupcenter(2,i)*invy + half
       zf = groupcenter(3,i)*invz + half
       xf = xf - floor(xf)
       yf = yf - floor(yf)
       zf = zf - floor(zf)

       ! x
       xd = xf
       ix = 0
       do while (xd > zero .and. ix < nx)
          ix = ix + 1
          xd = xd - nodefrx(ix)
       enddo
       dix = ix - homeix
       ! Make dix = -1,0,1
       if (dix > 1) then
          dix = dix - nx
       elseif (dix < -1) then
          dix = dix + nx
       endif
       if (abs(dix) > 1) then
          call wrndie(-5,'<domdec>','get_neighsend_dlb: group moved more than one box')
       endif
       ! y
       yd = yf
       iy = 0
       do while (yd > zero .and. iy < ny)
          iy = iy + 1
          yd = yd - nodefry(iy,dix)
       enddo
       diy = iy - homeiy
       if (diy > 1) then
          diy = diy - ny
       elseif (diy < -1) then
          diy = diy + ny
       endif
       if (abs(diy) > 1) then
          call wrndie(-5,'<domdec>','get_neighsend_dlb: group moved more than one box')
       endif
       ! z
       zd = zf
       iz = 0
       do while (zd > zero .and. iz < nz)
          iz = iz + 1
          zd = zd - nodefrz(iz,diy,dix)
       enddo
       diz = iz - homeiz
       if (diz > 1) then
          diz = diz - nz
       elseif (diz < -1) then
          diz = diz + nz
       endif
       if (abs(diz) > 1) then
          call wrndie(-5,'<domdec>','get_neighsend_dlb: group moved more than one box')
       endif

       if (dix /= 0 .or. diy /= 0 .or. diz /= 0) then
          if (abs(dix) > 1 .or. abs(diy) > 1 .or. abs(diz) > 1) then
             write (outu,'(a,6i8)') 'ix,iy,iz,home=',ix,iy,iz,homeix,homeiy,homeiz
             call wrndie(-5,'<domdec_d2d_comm>','group displacement too large')
          endif
          nod = neighind(dix,diy,diz)
          nneighsend(nod) = nneighsend(nod) + 1
          ncoord_send(nod) = ncoord_send(nod) + iq-is+1
       else
          nod = -1
       endif

       ! Store node destination to neighind_dest
       neighind_dest(ig) = nod
    enddo
!$omp end do

    return
  end subroutine get_neighsend_dlb

  ! *
  ! *
  ! *
  subroutine get_neighsend(x, y, z, nneighsend, neighind_dest, ncoord_send)
    use stream,only:outu
    use number,only:zero, half
    use domdec_common,only:zonelist, groupl, invx, invy, invz, homeix, homeiy, homeiz, nx, ny, nz, &
         neighind
    use groupxfast,only:groupcenter
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    integer, intent(inout) :: nneighsend(26), neighind_dest(:), ncoord_send(26)
    ! Variables
    integer ig, i, nod, is, iq
    integer ix, iy, iz, dix, diy, diz
    real(chm_real) xf, yf, zf

!$omp do schedule(static)
    do ig=1,zonelist(1)
       i = groupl(ig)

       ! Calculate group center of mass
       call calc_groupbox_loc2(i, x, y, z, is, iq)

       xf = groupcenter(1,i)*invx + half
       yf = groupcenter(2,i)*invy + half
       zf = groupcenter(3,i)*invz + half
       xf = xf - floor(xf)
       yf = yf - floor(yf)
       zf = zf - floor(zf)

       ix = int(xf*nx) + 1
       iy = int(yf*ny) + 1
       iz = int(zf*nz) + 1
       dix = ix - homeix
       diy = iy - homeiy
       diz = iz - homeiz
       if (dix > 1) then
          dix = dix - nx
       elseif (dix < -1) then
          dix = dix + nx
       endif
       if (diy > 1) then
          diy = diy - ny
       elseif (diy < -1) then
          diy = diy + ny
       endif
       if (diz > 1) then
          diz = diz - nz
       elseif (diz < -1) then
          diz = diz + nz
       endif

       if (dix /= 0 .or. diy /= 0 .or. diz /= 0) then
          if (abs(dix) > 1 .or. abs(diy) > 1 .or. abs(diz) > 1) then
             write (*,'(a,3i4)') 'dix,diy,diz=',dix,diy,diz
             write (*,'(a,3f10.3)') 'xf,yf,zf=',xf,yf,zf
             call wrndie(-5,'<domdec_d2d_comm>','group displacement too large')
          endif
          nod = neighind(dix,diy,diz)
          nneighsend(nod) = nneighsend(nod) + 1
          ncoord_send(nod) = ncoord_send(nod) + iq-is+1
       else
          nod = -1
       endif

       ! Store node destination to neighind_dest
       neighind_dest(ig) = nod
    enddo
!$omp end do

    return
  end subroutine get_neighsend

  ! *
  ! * Updates group list
  ! * NOTE: assumes that the groups have only moved between neighboring cubes
  ! * x2, y2, z2 are optional secondary particle lists that are also updated
  !
  subroutine update_groupl(x, y, z, x2, y2, z2)
    use memory
!!$    use pack_mod,only:pack_int_double, pack_double_double, unpack_int_double, unpack_double_double
    use iso_c_binding,only:c_loc
    use parallel,only:comm_charmm, mpi_integer_size, mpi_real8_size, mynod
    use groupxfast,only:group, group_out, ngroup, max_group_size
    use stream,only:outu
    use domdec_common,only:set_box, nneigh, zonelist, groupl, atoml, natoml, &
         homezone, neighlist, zonelist_atom, q_test
    use domdec_dlb,only:q_load_balance
    use mpi,only:mpi_integer, mpi_real8, mpi_success, mpi_packed, mpi_status_size, &
         mpi_statuses_ignore
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
#if KEY_LONEPAIR==1
    use domdec_lonepair,only:q_lonepr, setup_lonepr_comm
    use lonepr,only:numlp, lpnhost, lphptr, lphost
#endif
    use psf,only:natom
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: x(*), y(*), z(*)
    real(chm_real), intent(inout), optional :: x2(*), y2(*), z2(*)
    ! Variables
    logical secflag, flag
    integer i, j, k, ig, nod
    integer requests(26)
    integer is, iq, igroup, iatom
    integer neighsendbufpos(26), pos
    integer status(mpi_status_size), ierror
    integer neighrecvbufsize, ineigh, igrp
    integer nneighsend(26), nneighsend_tot, len
    integer ncoord_send(26), req_len
    integer ngroup_recv, ncoord_recv
    ! Temporary buffer used to pack the coordinates 
    real(chm_real) packbuf(max_group_size*6)

    call set_box()

    if (present(x2) .and. present(y2) .and. present(z2)) then
       secflag = .true.
    else
       secflag = .false.
    endif

    neighsendbufpos(1:nneigh) = 0
    nneighsend(1:nneigh) = 0
    ncoord_send(1:nneigh) = 0

    ! Allocate & reallocate (neighind_dest)
    call alloc_realloc()

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Get neighsend')
#endif    

    ! Calculate neighind_dest and ncoord_send (=total number of coordinates to be sent to neighbors)
!$omp parallel reduction(+:nneighsend, ncoord_send)
    if (q_load_balance) then
       call get_neighsend_dlb(x, y, z, nneighsend, neighind_dest, ncoord_send)
    else
       call get_neighsend(x, y, z, nneighsend, neighind_dest, ncoord_send)
    endif
!$omp end parallel

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif    

    ! nneigh = number of neighbors
    ! nneighsend(1:nneigh) contains the number of groups that are sent to each neighbor
    
    if (nneigh > 0) then
       nneighsend_tot = sum(nneighsend(1:nneigh))
       !
       ! Clear up homezone for groups 
       ! NOTE: We only need to clear the homezone for groups outside home box, because
       !       homezone for groups in home box is cleared in the loop below
       !       
!$omp parallel do schedule(static) private(j, i, is, iq)
       do j=zonelist(1)+1,zonelist(8)
          i = groupl(j)
          is = iand(group(i), Z'FFFFFFF')
          iq = is + iand(ishft(group(i), -28),7)
          homezone(is:iq) = 0
       enddo
!$omp end parallel do

       ! Reallocate neighsendbuf if neccessary
       do i=1,nneigh
          req_len = (nneighsend(i) + 2)*mpi_integer_size
          if (secflag) then
             req_len = req_len + 6*ncoord_send(i)*mpi_real8_size
          else
             req_len = req_len + 3*ncoord_send(i)*mpi_real8_size
          endif
          if (size(neighsendbuf(i)%p)*mpi_real8_size < req_len) then
             call chmrealloc('domdec.src','update_groupl','neighsendbuf',&
                  int(req_len*1.5/mpi_real8_size),mcrlp=neighsendbuf(i)%p)
          endif
       enddo

       ! Pack in total number of groups and coordinates
       do i=1,nneigh
!!$          call pack_int_double(nneighsend(i), 1, neighsendbuf(i)%p(1), neighsendbufpos(i))
          call mpi_pack(nneighsend(i), 1, mpi_integer, neighsendbuf(i)%p, &
               size(neighsendbuf(i)%p)*mpi_real8_size, neighsendbufpos(i), &
               comm_charmm, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_pack in update_group (1)')
!!$          call pack_int_double(ncoord_send(i), 1, neighsendbuf(i)%p(1), neighsendbufpos(i))
          call mpi_pack(ncoord_send(i), 1, mpi_integer, neighsendbuf(i)%p, &
               size(neighsendbuf(i)%p)*mpi_real8_size, neighsendbufpos(i), &
               comm_charmm, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_pack in update_group (1)')
       enddo

       ! neighind_dest(1:zonelist(1)) = destination node for each group

       igroup = 0
       iatom = 0
       do ig=1,zonelist(1)
          i = groupl(ig)
          !call group_out(group(i), is, iq)
          is = iand(group(i), Z'FFFFFFF')
          iq = is + iand(ishft(group(i), -28),7)

          nod = neighind_dest(ig)

          if (nod /= -1) then
             ! Group i has moved outside the box, send it to the correct box which is in node "nod"
             ! Clear homezone value
             homezone(is:iq) = 0
             ! Remove group i from the list
             ! Pack group i index
!!$             call pack_int_double(i, 1, neighsendbuf(nod)%p(1), neighsendbufpos(nod))
             call mpi_pack(i, 1, mpi_integer, neighsendbuf(nod)%p, &
                  size(neighsendbuf(nod)%p)*mpi_real8_size, neighsendbufpos(nod), &
                  comm_charmm, ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec>','Error in mpi_pack in update_group (1)')
             ! Pack group i coordinates
             if (secflag) then
                k = 0
                do j=is,iq
                   packbuf(k+1) = x(j)
                   packbuf(k+2) = y(j)
                   packbuf(k+3) = z(j)
                   packbuf(k+4) = x2(j)
                   packbuf(k+5) = y2(j)
                   packbuf(k+6) = z2(j)
                   k = k + 6
                enddo
             else
                k = 0
                do j=is,iq
                   packbuf(k+1) = x(j)
                   packbuf(k+2) = y(j)
                   packbuf(k+3) = z(j)
                   k = k + 3
                enddo
             endif
!!$             call pack_double_double(packbuf(1), k, neighsendbuf(nod)%p(1), neighsendbufpos(nod))
             call mpi_pack(packbuf, k, mpi_real8, neighsendbuf(nod)%p, &
                  size(neighsendbuf(nod)%p)*mpi_real8_size, neighsendbufpos(nod), &
                  comm_charmm, ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec>','Error in mpi_pack in update_group (2)')
             ! Remove group from list by not putting it in
          else
             ! Keep the group in list
             ! NOTE: No need to check for overflow of group_len because at this point
             !       size of groupl can only stay the same or decrease from original size
             igroup = igroup + 1
             groupl(igroup) = i
             do j=1,iq-is+1
                atoml(iatom+j) = j-1+is
             enddo
             iatom = iatom + (iq-is+1)
          endif
       enddo

       ! Send buffers to neighbors
       do k=1,nneigh
          call mpi_isend(neighsendbuf(k)%p, neighsendbufpos(k), mpi_packed, neighlist(k), &
               coordbuf, comm_charmm, requests(k), ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_isend in update_group')
       enddo

       ! Receive and unpack buffers
       recv_flag(1:nneigh) = .false.
       do k=1,nneigh
          flag = .false.
          ineigh = 0
          do while (.not.flag)
             call mpi_iprobe(neighlist(k), coordbuf, comm_charmm, flag, status, ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec>','Error in mpi_iprobe in update_group')
          enddo
          ineigh = k
          recv_flag(ineigh) = .true.
          call mpi_get_count(status, mpi_packed, neighrecvbufsize, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_get_count in update_group')
          
          ! Resize buffer if needed
          if (neighrecvbufsize > size(neighrecvbuf)*mpi_real8_size) then
             len = int(neighrecvbufsize*1.5/mpi_real8_size)
             call chmdealloc('domdec.src','update_group','neighrecvbuf',&
                  size(neighrecvbuf),mcrlp=neighrecvbuf)
             call chmalloc('domdec.src','update_group','neighrecvbuf',&
                  len,mcrlp=neighrecvbuf)
          endif
          call mpi_recv(neighrecvbuf, neighrecvbufsize, mpi_packed, neighlist(ineigh), &
               coordbuf, comm_charmm, status, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_recv in update_group')

          pos = 0
          ! Unpack number of groups and coordinates
!!$          call unpack_int_double(neighrecvbuf(1), pos, ngroup_recv, 1)
          call mpi_unpack(neighrecvbuf, neighrecvbufsize, pos, &
               ngroup_recv, 1, mpi_integer, comm_charmm, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_unpack in update_group')
!!$          call unpack_int_double(neighrecvbuf(1), pos, ncoord_recv, 1)
          call mpi_unpack(neighrecvbuf, neighrecvbufsize, pos, &
               ncoord_recv, 1, mpi_integer, comm_charmm, ierror)
          if (ierror /= mpi_success) call wrndie(-5, &
               '<domdec>','Error in mpi_unpack in update_group')
          
          if (size(groupl) < igroup + ngroup_recv) then
             call chmrealloc('domdec.src','update_groupl','groupl',&
                  int((igroup+ngroup_recv)*1.5),intg=groupl)
          endif

          if (size(atoml) < iatom + ncoord_recv) then
             call chmrealloc('domdec.src','update_groupl','atoml',&
                  int(1.5*(iatom+ncoord_recv)),intg=atoml)
          endif

          do while (pos < neighrecvbufsize)
             ! Unpack group index igrp
!!$             call unpack_int_double(neighrecvbuf(1), pos, igrp, 1)
             call mpi_unpack(neighrecvbuf, neighrecvbufsize, pos, &
                  igrp, 1, mpi_integer, comm_charmm, ierror)
             if (ierror /= mpi_success) call wrndie(-5, &
                  '<domdec>','Error in mpi_unpack in update_group')
             if (igrp < 0 .or. igrp > ngroup) then
                call wrndie(-5,'<domdec>','Error in group index in update_group')
             endif
             !call group_out(group(igrp), is, iq)
             is = iand(group(igrp), Z'FFFFFFF')
             iq = is + iand(ishft(group(igrp), -28),7)
             igroup = igroup + 1
             groupl(igroup) = igrp
             if (secflag) then
!!$                call unpack_double_double(neighrecvbuf(1), pos, packbuf(1), (iq-is+1)*6)
                call mpi_unpack(neighrecvbuf, neighrecvbufsize, pos, &
                     packbuf, (iq-is+1)*6, mpi_real8, comm_charmm, ierror)
                if (ierror /= mpi_success) call wrndie(-5, &
                     '<domdec>','Error in mpi_unpack in update_group')
                j = 0
                do i = is,iq
                   x(i) = packbuf(j+1)
                   y(i) = packbuf(j+2)
                   z(i) = packbuf(j+3)
                   x2(i) = packbuf(j+4)
                   y2(i) = packbuf(j+5)
                   z2(i) = packbuf(j+6)
                   j = j + 6
                enddo
             else
!!$                call unpack_double_double(neighrecvbuf(1), pos, packbuf(1), (iq-is+1)*3)
                call mpi_unpack(neighrecvbuf, neighrecvbufsize, pos, &
                     packbuf, (iq-is+1)*3, mpi_real8, comm_charmm, ierror)
                if (ierror /= mpi_success) call wrndie(-5, &
                     '<domdec>','Error in mpi_unpack in update_group')
                j = 0
                do i = is,iq
                   x(i) = packbuf(j+1)
                   y(i) = packbuf(j+2)
                   z(i) = packbuf(j+3)
                   j = j + 3
                enddo
             endif
             ! Copy coordinates to local array
             do j=1,iq-is+1
                atoml(iatom+j) = j-1+is
             enddo
             iatom = iatom + (iq-is+1)
          enddo
       enddo

       ! Store the end index of groups and atoms for home box
       zonelist(1) = igroup
       zonelist_atom(1) = iatom
       natoml = iatom
       
       ! Set homezone for groups in home box
       ! NOTE: rest of the homezone values are set in transfer_coord -subroutine
!$omp parallel do schedule(static) private(j, i, is, iq)
       do j=1,igroup
          i = groupl(j)
          !call group_out(group(i), is, iq)
          is = iand(group(i), Z'FFFFFFF')
          iq = is + iand(ishft(group(i), -28),7)
          homezone(is:iq) = 1
       enddo
!$omp end parallel do

       call mpi_waitall(nneigh, requests, mpi_statuses_ignore, ierror)
       if (ierror /= mpi_success) call wrndie(-5, &
            '<domdec>','Error in mpi_waitall in update_group')

#if KEY_LONEPAIR==1
       if (q_lonepr) then
          call setup_lonepr_comm(numlp, lpnhost, lphptr, lphost)
       endif
#endif

    endif

    if (q_test) then
       call test_update_groupl()
    endif

    return

  contains
    subroutine alloc_realloc()
      implicit none
      integer len

      if (allocated(neighind_dest) .and. size(neighind_dest) < zonelist(1)) then
         call chmdealloc('domdec.src','update_groupl','neighind_dest',&
              size(neighind_dest),intg=neighind_dest)
      endif
      
      if (.not.allocated(neighind_dest)) then
         len = int(zonelist(1)*1.5)
         call chmalloc('domdec.src','update_groupl','neighind_dest',len,intg=neighind_dest)
      endif

      return
    end subroutine alloc_realloc

  end subroutine update_groupl

  ! *
  ! * Test for update_groupl()
  ! *
  subroutine test_update_groupl()
    use stream,only:outu,prnlev
    implicit none

    if (test_groupl_atoml()) then
       if (prnlev > 2) then
          write (outu,'(a)') 'test_update_groupl OK'
       endif
    else
       call wrndie(-5,'<domdec>','test_update_groupl FAILED')
    endif

    return
  end subroutine test_update_groupl

  ! *
  ! * Copies coordinates to root
  ! *
  ! *
  subroutine copy_to_root(x, y, z)
    use mpi,only:mpi_integer, mpi_real8
    use parallel,only:mynod, COMM_CHARMM
    use psf,only:natom
    use domdec_dr_common,only:q_direct_node, q_recip_node, comm_direct
    use domdec_common
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    ! Variables
    integer i, j, n, sum_nrecv, ierror
    real(chm_real) r

    if (.not.q_direct_node .and. q_recip_node) then
       ! Pure reciprocal nodes exit here
       return
    endif

    call allocate_copy()

    if (mynod /= 0) then
       n = 0
       do j=1,natoml
          i = atoml(j)
          n = n + 1
          send_xbuf(n) = x(i)
          send_ybuf(n) = y(i)
          send_zbuf(n) = z(i)
          send_indbuf(n) = i
       enddo
    else
       recvcount(1:ndirect) = 1
       do i=1,ndirect
          disp(i) = i-1
       enddo
       n = natoml
    endif

    call mpi_gatherv(n, 1, mpi_integer, nrecv, recvcount, disp, mpi_integer, &
         0, comm_direct, ierror)

    ! nrecv contains the number of particles on each node
    if (mynod == 0) then
       sum_nrecv = sum(nrecv(1:ndirect))
       if (sum_nrecv /= natom) then
          call wrndie(-5,'<domdec_d2d_comm>','Error in sum_nrecv (4)')
       endif
       disp2(1) = 0
       do i=2,ndirect
          disp2(i) = disp2(i-1) + nrecv(i-1)
       enddo
    endif

    call mpi_gatherv(send_xbuf, n, mpi_real8, recv_xbuf, nrecv, disp2, mpi_real8, &
         0, comm_direct, ierror)

    call mpi_gatherv(send_ybuf, n, mpi_real8, recv_ybuf, nrecv, disp2, mpi_real8, &
         0, comm_direct, ierror)

    call mpi_gatherv(send_zbuf, n, mpi_real8, recv_zbuf, nrecv, disp2, mpi_real8, &
         0, comm_direct, ierror)

    call mpi_gatherv(send_indbuf, n, mpi_integer, recv_indbuf, nrecv, disp2, &
         0, mpi_integer, comm_direct, ierror)

    if (mynod == 0) then
       do i=1,natom
          j = recv_indbuf(i)
          x(j) = recv_xbuf(i)
          y(j) = recv_ybuf(i)
          z(j) = recv_zbuf(i)
       enddo
    endif

    return
  end subroutine copy_to_root

  ! *
  ! * Copies coordinates to all nodes
  ! *
  subroutine copy_to_all1(x)
    use mpi,only:mpi_integer, mpi_real8
    use parallel,only:COMM_CHARMM
    use psf,only:natom
    use domdec_dr_common,only:q_direct_node, q_recip_node, comm_direct
    use domdec_common
    implicit none
    ! Input / Output
    real(chm_real) x(*)
    ! Variables
    integer i, j, n, sum_nrecv, ierror
    real(chm_real) r

    if (.not.q_direct_node .and. q_recip_node) then
       ! Reciprocal nodes exit here
       return
    endif

    call allocate_copy()

    recvcount(1:ndirect) = 1
    do i=1,ndirect
       disp(i) = i-1
    enddo

    n = 0
    do j=1,natoml
       i = atoml(j)
       n = n + 1
       send_xbuf(n) = x(i)
       send_indbuf(n) = i
    end do

    call mpi_allgatherv(n, 1, mpi_integer, nrecv, recvcount, disp, mpi_integer, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_d2d_comm>',&
         'Error in mpi_allgatherv (1)')
    sum_nrecv = sum(nrecv(1:ndirect))
    if (sum_nrecv /= natom) then
       call wrndie(-5,'<domdec_d2d_comm>','Error in sum_nrecv (1)')
    endif
       
    ! nrecv contains the number of particles on each node
    disp2(1) = 0
    do i=2,ndirect
       disp2(i) = disp2(i-1) + nrecv(i-1)
    enddo
    
    call mpi_allgatherv(send_xbuf, n, mpi_real8, recv_xbuf, nrecv, disp2, mpi_real8, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (2)')
        
    call mpi_allgatherv(send_indbuf, n, mpi_integer, recv_indbuf, nrecv, disp2, &
         mpi_integer, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (5)')
    
    do i=1,natom
       j = recv_indbuf(i)
       x(j) = recv_xbuf(i)
    enddo

    return
  end subroutine copy_to_all1

  ! *
  ! * Copies (x, y, z) to all (direct) nodes
  ! *
  subroutine copy_to_all3(x, y, z)
    use mpi,only:mpi_integer, mpi_real8
    use parallel,only:comm_charmm
    use psf,only:natom
    use domdec_dr_common,only:q_direct_node, q_recip_node, comm_direct
    use domdec_common
    implicit none
    ! Input / Output
    real(chm_real) x(*), y(*), z(*)
    ! Variables
    integer i, j, n, sum_nrecv, ierror
    real(chm_real) r

    if (.not.q_direct_node .and. q_recip_node) then
       ! Reciprocal nodes exit here
       return
    endif

    call allocate_copy()

    recvcount(1:ndirect) = 1
    do i=1,ndirect
       disp(i) = i-1
    enddo
    
    n = 0
    do j=1,natoml
       i = atoml(j)
       n = n + 1
       send_xbuf(n) = x(i)
       send_ybuf(n) = y(i)
       send_zbuf(n) = z(i)
       send_indbuf(n) = i
    end do

    call mpi_allgatherv(n, 1, mpi_integer, nrecv, recvcount, disp, mpi_integer, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5,'<domdec_d2d_comm>',&
         'Error in mpi_allgatherv (1)')
    sum_nrecv = sum(nrecv(1:ndirect))
    if (sum_nrecv /= natom) then
       call wrndie(-5,'<domdec_d2d_comm>','Error in sum_nrecv (3)')
    endif

    ! nrecv contains the number of particles on each node
    disp2(1) = 0
    do i=2,ndirect
       disp2(i) = disp2(i-1) + nrecv(i-1)
    enddo
    
    call mpi_allgatherv(send_xbuf, n, mpi_real8, recv_xbuf, nrecv, disp2, mpi_real8, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (2)')
  
    call mpi_allgatherv(send_ybuf, n, mpi_real8, recv_ybuf, nrecv, disp2, mpi_real8, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (3)')
    
    call mpi_allgatherv(send_zbuf, n, mpi_real8, recv_zbuf, nrecv, disp2, mpi_real8, &
         comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (4)')
    
    call mpi_allgatherv(send_indbuf, n, mpi_integer, recv_indbuf, nrecv, disp2, &
         mpi_integer, comm_direct, ierror)
    if (ierror /= mpi_success) call wrndie(-5, &
         '<domdec_d2d_comm>','Error in mpi_allgatherv (5)')

    do i=1,natom
       j = recv_indbuf(i)
       x(j) = recv_xbuf(i)
       y(j) = recv_ybuf(i)
       z(j) = recv_zbuf(i)
    enddo

    return
  end subroutine copy_to_all3

  ! *
  ! * Tests atom and group lists, makes sure all atoms are the list and only once.
  ! * Returns .true. if group and atom lists are OK
  ! *
  logical function test_groupl_atoml()
    use stream,only:outu
    use psf,only:natom
    use parallel,only:mynod
    use memory,only:chmalloc, chmdealloc
    use mpi,only:mpi_success, mpi_integer, mpi_sum, mpi_logical, mpi_land
    use domdec_dr_common,only:comm_direct
    use domdec_common,only:natoml, atoml
    implicit none
    ! Variables
    integer i, natom_tmp
    integer, allocatable, dimension(:) :: tmp, res
    integer ierror
    logical flag, resflag

    flag = .true.
    
    call chmalloc('domdec_d2d_comm.src','test_groupl_atoml','tmp',natom,intg=tmp)
    if (mynod == 0) then
       call chmalloc('domdec_d2d_comm.src','test_groupl_atoml','res',natom,intg=res)
       res(1:natom) = 0
    endif

    tmp(1:natom) = 0
    do i=1,natoml
       if (atoml(i) <= 0 .or. atoml(i) > natom) then
          write (outu,'(a,i12)') 'test_groupl_atoml, invalid value in atoml=',atoml(i)
          flag = .false.
       elseif (tmp(atoml(i)) /= 0) then
          write (outu,'(a,i3,2i8)') 'test_groupl_atoml, duplicate atom,mynod,i,atoml(i)=',&
               mynod,i,atoml(i)
          flag = .false.
       endif
       tmp(atoml(i)) = 1
    enddo

    call mpi_reduce(natoml, natom_tmp, 1, mpi_integer, mpi_sum, 0, comm_direct, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2d_comm>','Error in mpi_reduce in test_groupl_atoml')
    endif

    if (mynod == 0) then
       if (natom_tmp /= natom) then
          write (outu,'(a,i12)') 'test_groupl_atoml, incorrect total number of atoms=',natom_tmp
          flag = .false.
       endif
    endif

    call mpi_reduce(tmp, res, natom, mpi_integer, mpi_sum, 0, comm_direct, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2d_comm>','Error in mpi_reduce in test_groupl_atoml')
    endif

    if (mynod == 0) then
       do i=1,natom
          if (res(i) < 1) then
             write (outu,'(a,i8)') 'test_groupl_atoml, missing atom i=',i
             flag = .false.
          elseif (res(i) > 1) then
             write (outu,'(a,i8)') 'test_groupl_atoml, double atom i=',i
             flag = .false.
          endif
       enddo
    endif

    call mpi_allreduce(flag, resflag, 1, mpi_logical, mpi_land, comm_direct, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2d_comm>','Error in mpi_allreduce in test_groupl_atoml')
    endif

    call chmdealloc('domdec_d2d_comm.src','test_groupl_atoml','tmp',natom,intg=tmp)
    if (mynod == 0) then
       call chmdealloc('domdec_d2d_comm.src','test_groupl_atoml','res',natom,intg=res)
    endif

    test_groupl_atoml = resflag

    return
  end function test_groupl_atoml

  ! *
  ! * Allocate buffers for the copy_to_* - routines
  ! *
  subroutine allocate_copy()
    use psf,only:natom
    use memory
    use domdec_common
    implicit none
    integer n

    n = min(max(1,int(natoml*2)),natom)

    if (allocated(nrecv)) then
       if (size(nrecv) < ndirect) then
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','nrecv',ndirect,intg=nrecv)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','disp',ndirect,intg=disp)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','disp2',ndirect,intg=disp2)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','recvcount',ndirect,intg=recvcount)
       endif
    else
       call chmalloc('domdec_d2d_comm.src','allocate_copy','nrecv',ndirect,intg=nrecv)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','disp',ndirect,intg=disp)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','disp2',ndirect,intg=disp2)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','recvcount',ndirect,intg=recvcount)
    endif

    if (allocated(send_indbuf)) then
       if (size(send_indbuf) < natoml) then
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','send_indbuf',n,intg=send_indbuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','send_xbuf',n,crl=send_xbuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','send_ybuf',n,crl=send_ybuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','send_zbuf',n,crl=send_zbuf)
       endif
       if (size(recv_indbuf) < natom) then
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','recv_indbuf',natom,intg=recv_indbuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','recv_xbuf',natom,crl=recv_xbuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','recv_ybuf',natom,crl=recv_ybuf)
          call chmrealloc('domdec_d2d_comm.src','allocate_copy','recv_zbuf',natom,crl=recv_zbuf)
       endif
    else
       call chmalloc('domdec_d2d_comm.src','allocate_copy','send_indbuf',n,intg=send_indbuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','send_xbuf',n,crl=send_xbuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','send_ybuf',n,crl=send_ybuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','send_zbuf',n,crl=send_zbuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','recv_indbuf',natom,intg=recv_indbuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','recv_xbuf',natom,crl=recv_xbuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','recv_ybuf',natom,crl=recv_ybuf)
       call chmalloc('domdec_d2d_comm.src','allocate_copy','recv_zbuf',natom,crl=recv_zbuf)
    endif

    return
  end subroutine allocate_copy

  ! *
  ! * Deallocate buffers for the copy_to_* - routines
  ! *
  subroutine deallocate_copy()
    use memory,only:chmdealloc
    implicit none

    if (allocated(nrecv)) then
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','nrecv',size(nrecv),intg=nrecv)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','disp',size(disp),intg=disp)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','disp2',size(disp2),intg=disp2)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','recvcount',size(recvcount),&
            intg=recvcount)
    endif
    
    if (allocated(send_indbuf)) then
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','send_indbuf',size(send_indbuf),&
            intg=send_indbuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','send_xbuf',size(send_xbuf),&
            crl=send_xbuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','send_ybuf',size(send_ybuf),&
            crl=send_ybuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','send_zbuf',size(send_zbuf),&
            crl=send_zbuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','recv_indbuf',size(recv_indbuf),&
            intg=recv_indbuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','recv_xbuf',size(recv_xbuf),&
            crl=recv_xbuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','recv_ybuf',size(recv_ybuf),&
            crl=recv_ybuf)
       call chmdealloc('domdec_d2d_comm.src','deallocate_copy','recv_zbuf',size(recv_zbuf),&
            crl=recv_zbuf)
    endif

    return
  end subroutine deallocate_copy

#ifdef __PGI
  logical function isnan(a) result(f)
    real(chm_real),intent(in) :: a
    f = a /= a
    return
  end function isnan
#endif /* __PGI */

!#endif /* (not_cmpi)*/
#endif /* (parallel)*/
#endif /* (domdec_main)*/
end module domdec_d2d_comm

