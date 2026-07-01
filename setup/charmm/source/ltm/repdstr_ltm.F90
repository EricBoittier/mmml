module repdstr
  use chm_kinds
  use dimens_fcm
  use parallel, only: maxnode
  implicit none

#if KEY_REPDSTR==0 /* repdstr_fcm */
      logical, parameter :: QREPDSTR = .false.
#else /* (repdstr_fcm)*/
#if KEY_PARALLEL==1 /*check_dependency*/
!-----------------------------------------------------------------------
!
!     QREPDSTR         are distributed replicas in use at present ?
!     QRDQTT           flag for use of RDCMND (true=no global broadcast)
!     QWRQTT           flag for use of OUTU (true=every mynod=0 writes)
!     NREPDSTR         number of distributed replicas in use at present ?
!     MAXREPDSTR       maximum number of distributed replicas
!     QREXCHG          flag if this is replica exchange run
!     QREXCHGL         flag if this is lambda replica exchange run for free
!                      energy perturbation or umbrella sampling
!     QEXPT            flag if this is parallel tempering replica exchange
!                      for internal barrier of protein side chain or 
!                      backbone(?)
!     QEXBK            flag if replica exchange blocker is applied.
!     IRBK             IREPDSTR of replica exchange blocker   
!
!-----------------------------------------------------------------------
      LOGICAL QREPDSTR,QRDQTT,QWRQTT,QREXCHG,QREXCHGL,QEXPT,QEX2D,QEXBK
      logical :: qrepioset, qrepdnames = .true.
      INTEGER NREPDSTR,IREPDSTR,IREXFQ,COMM_UNIVERSE,COMM_RPG
      INTEGER ITDRB,REPSEED,IOLORIG,PRNLORIG,WRNLORIG,IOON,IUNREX,NATREPCMD
      integer NREPT,NREPX,IRBK,EWRITU
      integer,parameter  :: NWREX=1000
      integer,parameter  :: MAXREPDSTR=64

      integer, save :: repd_inside_if_block               ! > 0 if parser is currently inside an 'if' block
      integer, parameter :: max_repd_if_block = 128
      ! Used to determine if OK to execute commands in 'if' block
      !   1 = OK to execute.
      !   0 = Not OK to execute; inside nested 'if' block with stat 0
      !  -1 = Not OK to execute.
      integer, save, dimension(max_repd_if_block) :: repd_if_block_stat
      logical, save      :: repd_if_block_enabled

      real(chm_real) TEMPRX(MAXNODE)

      logical,save :: qrepmaster         ! True if this thread is master of this replica group
      integer,save :: comm_rep_master    ! MPI comm for all COMM_RPG masters
#endif /* (check_dependency)*/

contains
  subroutine repdstr_iniall
    use mpi
    implicit none
    ioon=0
    qrdqtt=.false.
    qwrqtt=.false.
    qrepioset=.false.
    qrepdstr = .false.
    nrepdstr = 0
    irepdstr = 0
    repd_inside_if_block = 0
    repd_if_block_stat(:) = -1
    repd_if_block_enabled = .true.
    qrepmaster = .false.
    comm_rep_master = MPI_COMM_NULL
    comm_rpg = MPI_COMM_NULL
    return
  end subroutine repdstr_iniall

!*******************************************************************************
!> Set the parallel information from global values, ie all system
!<
  SUBROUTINE PSETGLOB

  use parallel
    !
    integer:: i
    MYNOD=MYNODG
    NUMNOD=NUMNODG
    MYNODP=MYNOD+1
    NODDIM=NPTWO()
    CALL CUBE(MYNOD,NUMNOD,IPPMAP)
    !
    !     Lets do the INODE array, too:
    !
    DO I=1,MAXNODE
       INODE(I)=MOD(I,NUMNOD)
    ENDDO
    !
    COMM_CHARMM=COMM_UNIVERSE
    RETURN
  END SUBROUTINE PSETGLOB

!*******************************************************************************
!> Set the local parallel information.
!<
! TODO Put PSETLOC,PSETGLOB into paral1.src ??
  SUBROUTINE PSETLOC

  use parallel
    !
    INTEGER I
    !
    NUMNOD=NUMNODG/NREPDSTR
    MYNOD=MOD(MYNODG,NUMNOD)
    MYNODP=MYNOD+1
    NODDIM=NPTWO()
    DO I = 0, NUMNOD
       IPPMAP(I)=I+NUMNOD*IREPDSTR
    ENDDO
    !
    !     Lets do the INODE array, too:
    !
    DO I=1,MAXNODE
       INODE(I)=MOD(I,NUMNOD)
    ENDDO
    !
    !     Also iparpt(), nparpt() have to be localized??? !!!! Pressure problems???
    !     Maybe not so trivial??? Better save
    !     both cases when generated and then copy them as needed ???
    !  IPARPT and NPARPT are more tricky, since they depend on
    !  groups. It can't be dealt generally since when repd is started we
    !  might don't have the PSF yet. It is maybe better to use separate routine
    !  and call it before vdgsum & vdgbr, or whatever routines use them.
    !  or use nbonds command in the repd script after PSFs are set up.
    !
    COMM_CHARMM=COMM_RPG
    RETURN
  END SUBROUTINE PSETLOC

#endif /* (repdstr_fcm)*/

end module repdstr

