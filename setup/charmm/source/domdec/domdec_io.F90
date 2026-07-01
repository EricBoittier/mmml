module domdec_io
#if KEY_DOMDEC==1
  ! *
  ! * Domdec I/O routines, used e.g. to send string data from constraint nodes to the root node
  ! * Antti-Pekka Hynninen
  ! *
  implicit none
  private

  ! Parameters
  integer, parameter :: nstr_tag=0, str_tag=1

  ! Public subroutines
  public send_str_to_root, send_nostr_to_root, recv_str_from_node

contains

  ! *
  ! * Send character array to root node
  ! *
  subroutine send_str_to_root(str)
    use mpi,only:mpi_integer, mpi_character, mpi_success
    use parallel,only:comm_charmm, mynod
    implicit none
    ! Input / Output
    character(*), intent(in) :: str
    ! Variables
    integer nstr
    integer ierror

    ! Send length (including the end-of-string character)
    nstr = len(str)
    call mpi_send(nstr, 1, mpi_integer, 0, nstr_tag, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_io.src>','Error sending string length')
    endif

    ! Send string
    call mpi_send(str, nstr, mpi_character, 0, str_tag, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_io.src>','Error sending string')
    endif

    return
  end subroutine send_str_to_root

  ! *
  ! * Send "no-string" command to root
  ! *
  subroutine send_nostr_to_root()
    use mpi,only:mpi_integer, mpi_success
    use parallel,only:comm_charmm,mynod
    implicit none
    ! Variables
    integer ierror

    call mpi_send(0, 1, mpi_integer, 0, nstr_tag, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_io.src>','Error sending string length 0')
    endif
    
    return
  end subroutine send_nostr_to_root

  ! *
  ! * Receive character array from a node
  ! *
  subroutine recv_str_from_node(str, nstr, src_node)
    use mpi,only:mpi_integer, mpi_character, mpi_success, mpi_status_ignore
    use parallel,only:comm_charmm
    implicit none
    ! Input / Output
    character(*), intent(out) :: str
    integer, intent(out) :: nstr
    integer, intent(in) :: src_node
    ! Variables
    integer ierror

    ! Receive length (including the end-of-string character)
    call mpi_recv(nstr, 1, mpi_integer, src_node, nstr_tag, comm_charmm, mpi_status_ignore, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_io.src>','Error receiving string length')
    endif
    if (nstr <= 0) return

    ! Receive string
    call mpi_recv(str, nstr, mpi_character, src_node, str_tag, comm_charmm, mpi_status_ignore,&
         ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_io.src>','Error receiving string')
    endif

    return
  end subroutine recv_str_from_node


#endif
end module domdec_io
