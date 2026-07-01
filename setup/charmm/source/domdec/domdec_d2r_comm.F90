module domdec_d2r_comm

  ! *
  ! * Domain decomposition direct-to-reciprocal communication
  ! * NOTE: Only direct nodes call subroutines in this module
  ! *

#if KEY_DOMDEC==1 /*domdec_main*/
#if KEY_PARALLEL==1 /*parallel*/
!#if KEY_CMPI==0 /*not_cmpi*/
  use chm_kinds
  use dimens_fcm
  use mpi,only:MPI_STATUS_SIZE
  implicit none
  private

  ! Public subroutines
  public send_coord_to_recip
  public wait_results_from_recip, unpack_results_from_recip, probe_results_from_recip
  public send_stop_recip
  !public comm_auxdata_d2r
  public uninit_d2r_comm

contains

  ! *
  ! * Send stop_recip signal to recip CPUs
  ! *
  subroutine send_stop_recip()
    use pack_mod,only:pack_double_byte, pack_int_byte
    use mpi
    use parallel,only:mpi_real8_size
    use reawri,only:qcnstp
    use number,only:minone
    use domdec_common,only:ndirect
    use domdec_dr_common,only:commbuffer, commbuffersize, COORDBUFLEN, comm_direct_recip, &
         commnode, reqbuffer, COORDBUF, STOP_BIT
    implicit none
    integer ierror
    real(chm_real) tmp(6)

    commbuffersize(1) = 0
    if (qcnstp) then
       tmp(1:6) = minone
       call pack_double_byte(tmp(1), 6, commbuffer(1), commbuffersize(1))
    else
       call pack_int_byte(STOP_BIT, 1, commbuffer(1), commbuffersize(1))
    endif
    call mpi_isend(commbuffersize(1), 1, MPI_INTEGER, &
         commnode(1), COORDBUFLEN, comm_direct_recip, &
         reqbuffer(1), ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2r_comm>','error in mpi_isend buffer')
    endif

    call mpi_isend(commbuffer, commbuffersize(1), MPI_BYTE, &
         commnode(1), COORDBUF, comm_direct_recip, &
         reqbuffer(2), ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2r_comm>','error in mpi_isend buffer')
    endif

    call mpi_waitall(2, reqbuffer, mpi_statuses_ignore, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2r_comm>','error in mpi_waitall')
    endif

    return
  end subroutine send_stop_recip

  ! *
  ! * Waits and receives forces, energies, and virial from recip nodes
  ! *
  subroutine wait_results_from_recip()
    use mpi,only:mpi_success, mpi_status_ignore
    use domdec_common,only:natoml
    use new_timer,only:timer_start, timer_stop, T_fcomm2wait  
    use domdec_dr_common,only:reqbuffer, mynod_split
    implicit none
    ! Variables
    integer ierror

    ! #### DIRECT ####

    ! NOTE: mpi_irecv's have been posted at the end of send_coord_to_recip
    
    call timer_start(T_fcomm2wait)  

    ! Catch mpi_irecv commbuffer (forces)
    if (natoml > 0) then
       call mpi_wait(reqbuffer(3), mpi_status_ignore, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5, '<domdec_d2r_comm>','error in wait')
       endif
    endif

    call timer_stop(T_fcomm2wait)  

    return
  end subroutine wait_results_from_recip

  ! *
  ! * Probes for results from recip nodes
  ! * Returns .true. if results have arrived, .false. otherwise
  ! *
  logical function probe_results_from_recip()
    use mpi,only:mpi_success, mpi_status_ignore
    use domdec_common,only:natoml
    use domdec_dr_common,only:reqbuffer
    implicit none
    ! Variables
    integer ierror
    logical flag

    probe_results_from_recip = .true.

    if (natoml > 0) then
       call mpi_test(reqbuffer(3), flag, mpi_status_ignore, ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5, '<domdec_d2r_comm>','probe_results_from_recip: error in mpi_test')
       endif
       probe_results_from_recip = flag
    endif

    return
  end function probe_results_from_recip

  ! *
  ! * Unpacks results from recip nodes
  ! *
  subroutine unpack_results_from_recip(forcex, forcey, forcez, auxdata, nauxdata)
    use mpi,only:mpi_success, mpi_real8
    use domdec_common,only:natoml, atoml
    use pack_mod,only:unpack_force, unpack_double_byte
    use domdec_dr_common,only:commbuffer, commbuffersize, comm_direct
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: forcex(*), forcey(*), forcez(*)
    real(chm_real), intent(inout) :: auxdata(:)
    integer, intent(in) :: nauxdata
    ! Variables
    integer ierror, k

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('Unpack force')
#endif
    ! Unpack force data
    k = 0
    call unpack_force(natoml, atoml, k, commbuffer(1), forcex, forcey, forcez)
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif    

    ! Unpack auxdata(1:nauxdata)
    call unpack_double_byte(commbuffer(1), k, auxdata(1), nauxdata)

    return
  end subroutine unpack_results_from_recip

  ! *
  ! * Sends coordinates to reciprocal nodes
  ! * If q_send_atoml = .true., send the atomlist along with the coordinates
  ! * (this is needed after neighborlist update)
  ! *
  subroutine send_coord_to_recip(x, y, z, q_send_atoml)
    use pack_mod,only:pack_double_byte, pack_int_byte, pack_xyz_atom_byte
    use mpi,only:mpi_byte, mpi_integer, mpi_success, MPI_STATUSES_IGNORE
    use domdec_common,only:natoml, atoml
    use parallel,only:mpi_real8_size
    use reawri,only:qcnstp
    use image,only:xtlabc
    use domdec_dr_common,only:q_direct_node, ncomm, commnode, mynod_split, get_recip_node, &
         commbuffer, commbuffersize, commbuffersize, COORDBUF, COORDBUFLEN, FORCEBUF, &
         comm_direct_recip, reqbuffer, nreqbuffer, nrecip, ATOML_BIT
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif
    use energy_util,only:get_nauxdata
    implicit none
    ! Input
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    logical, intent(in) :: q_send_atoml
    ! Variables
    real(chm_real) xyz(3)
    integer ierror, nauxdata
    integer i, j

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('send_coord_to_recip')
#endif

    ! Little sanity check to start with
    if (.not.q_direct_node) then
       call wrndie(-5,'<domdec_d2r_comm>','send_coord_to_recip can only be called by a direct node')
    endif

    ! Each direct node communicates with a single reciprocal node
    ncomm = 1

    nauxdata = get_nauxdata()

    ! Allocate & reallocate memory (commnode, commbuffersize, commbuffer, reqbuffer)
    call alloc_realloc(natoml, ncomm, nauxdata)

    ! Determine the recip node for this direct node
    commnode(1) = get_recip_node(mynod_split)

    ! Zero buffer size before starting to pack
    commbuffersize(1) = 0
    
    ! In case of constant pressure, pack in new box sizes
    if (qcnstp) then
       call pack_double_byte(xtlabc(1), 6, commbuffer(1), commbuffersize(1))
    endif
    
    ! Pack in number of atoms and atom indices
    ! In case atoml is sent over, sets the appropriate bit 
    if (q_send_atoml) then
       call pack_int_byte(ior(ishft(natoml, 2), ATOML_BIT), 1, commbuffer(1), commbuffersize(1))
       call pack_int_byte(atoml(1), natoml, commbuffer(1), commbuffersize(1))
    else
       call pack_int_byte(ishft(natoml, 2), 1, commbuffer(1), commbuffersize(1))
    endif

    ! Pack in coordinates
    call pack_xyz_atom_byte(natoml, atoml, commbuffer(1), commbuffersize(1), x, y, z)
    
    if (commbuffersize(1) > size(commbuffer)) then
       call wrndie(-5,'<domdec_d2r_comm>','commbuffer exceeded')
    endif

    ! Send buffer size
    nreqbuffer = 1
    call mpi_isend(commbuffersize(1), 1, mpi_integer, &
         commnode(1), COORDBUFLEN, comm_direct_recip, &
         reqbuffer(nreqbuffer), ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2r_comm>','Error in mpi_isend buffer length')
    endif
    
    ! Send buffer
    if (commbuffersize(1) > 0) then
       nreqbuffer = nreqbuffer + 1
       call mpi_isend(commbuffer, commbuffersize(1), mpi_byte, &
            commnode(1), COORDBUF, comm_direct_recip, &
            reqbuffer(nreqbuffer), ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_d2r_comm>','Error in mpi_isend buffer')
       endif
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_start('mpi_waitall')
#endif
    call mpi_waitall(nreqbuffer, reqbuffer, MPI_STATUSES_IGNORE, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_d2r_comm>','Error in mpi_waitall')
    endif
#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! Post mpi_irecv for forces & auxdata from recip CPUs
    if (natoml > 0) then
       call mpi_irecv(commbuffer, (3*natoml + nauxdata)*mpi_real8_size, mpi_byte, &
            commnode(1), FORCEBUF, comm_direct_recip, &
            reqbuffer(3), ierror)
       if (ierror /= mpi_success) then
          call wrndie(-5,'<domdec_d2r_comm>','Error in mpi_irecv buffer')
       endif
    endif

#if KEY_DOMDEC_GPU==1
    if (q_gpu) call range_stop()
#endif

    ! NOTE: the mpi_irecv above is caught with mpi_wait in recv_results_from_recip() -routine
    
    return
  contains

    subroutine alloc_realloc(natoml, ncomm, nauxdata)
      use memory
      use mpi,only:mpi_status_size
      use parallel,only:mpi_integer_size, mpi_real8_size
      use domdec_dr_common,only:alloc_realloc_commbuffer
      implicit none
      ! Input
      integer, intent(in) :: natoml, ncomm, nauxdata
      ! Variables
      integer ierror
      integer req_commbufferlen

      ! commnode
      if (allocated(commnode)) then
         if (size(commnode) < ncomm) then
            call chmdealloc('domdec_d2r_comm.src','send_coord_to_recip','commnode',&
                 size(commnode),intg=commnode)
         endif
      endif

      if (.not.allocated(commnode)) then
         call chmalloc('domdec_d2r_comm.src','send_coord_to_recip','commnode',ncomm,intg=commnode)
      endif

      ! commbuffersize
      if (allocated(commbuffersize)) then
         if (size(commbuffersize) < ncomm) then
            call chmdealloc('domdec_d2r_comm.src','send_coord_to_recip','commbuffersize',&
                 size(commbuffersize),intg=commbuffersize)
         endif
      endif

      if (.not.allocated(commbuffersize)) then
         call chmalloc('domdec_d2r_comm.src','send_coord_to_recip','commbuffersize',&
              ncomm,intg=commbuffersize)
      endif

      ! commbuffer
      req_commbufferlen = (natoml+1)*mpi_integer_size + (3*natoml+6+nauxdata)*mpi_real8_size
      call alloc_realloc_commbuffer(commbuffer, req_commbufferlen)

      ! reqbuffer
      if (allocated(reqbuffer)) then
         if (size(reqbuffer) < max(nrecip,4)) then
            call chmdealloc('domdec_d2r_comm.src','send_coord_to_recip','reqbuffer',&
                 size(reqbuffer),intg=reqbuffer)
         endif
      endif
      
      if (.not.allocated(reqbuffer)) then
         call chmalloc('domdec_d2r_comm.src','send_coord_to_recip','reqbuffer',&
              max(nrecip,4),intg=reqbuffer)
      endif

      return
    end subroutine alloc_realloc
  end subroutine send_coord_to_recip

  ! *
  ! * De-allocates memory for this module
  ! *
  subroutine uninit_d2r_comm()
    implicit none

    return
  end subroutine uninit_d2r_comm

!#endif /* (not_cmpi)*/
#endif /* (parallel)*/
#endif /* (domdec_main)*/
end module domdec_d2r_comm

