!-----------------------------------------------------------------------
!     Generate replicas: one per process or group of processes,
!     and maybe many replicas per process.
!     By Paul Maragakis and Milan Hodoscek, 2005
!     Support for parallel/parallel and I/O setup: Milan Hodoscek, May 2007
!
!     Milan Hodoscek, April 2009:
!      - replica exchange printout and fixes
!      - SGLD replica exchange
!      - support for equal temperatures of selected/all replicas
!      - initial implementation of TIGER method.
!      TIGER is based on the code from Satoru Itoh
!
!     Daniel R. Roe, 2016-2018:
!      - Refactor REPD routines to use MPI (remove dependence on CMPI).
!      - Make REPD work in conjunction with DOMDEC.
!      - Support for N-dimensional exchange.
!      - Add pressure-volume correction to exchange calc during NPT dynamics.
!      - Bath temperatures, pH, SGLD parameters all automatically assigned.
!
MODULE REPDSTRMOD
  use chm_kinds
# if KEY_PARALLEL==1
# if KEY_DEBUGREPD == 1
  use parallel, only: maxnode,comm_charmm,mynodg ! DBG mynodg
# else
  use parallel, only: maxnode,comm_charmm
# endif
# endif /* KEY_PARALLEL */
#if KEY_REPDSTR == 1
  use repdstr, only: comm_universe, comm_rep_master, qrepmaster
#endif
#if KEY_OPENMM == 1
  use omm_glblopts, only : qtor_repex, torsion_lambda
#endif


  implicit none
# if KEY_REPDSTR==1
  ! ----- VARIABLES FOR EXCHANGE_CALC ----------------------
  integer,save :: reptag             ! Current replica tag
  integer,save :: rep_oldtag         ! Replica tag at beginning of exchange
  logical,save :: q_replica_exchange ! True if we are doing replica exchange
  logical,save :: q_use_new_exchange ! True if using new exchange method
  logical,save :: q_use_pv           ! If true, use the pressure/volume correction for exchange calc
  logical,save :: q_no_pv            ! If true, do not use pressure/volume correction even when NPT
  logical,save :: allow_multi_exchg  ! If true allow multiple dimensions to swap in same step.
  logical      :: qcontrol_exchg     ! True if this rep controls exchange (ODD reps 1D, LOWER reps 2D)
  logical      :: q_exch_attempted   ! True if an exchange was attempted this step (for output)
# if KEY_DOMDEC==1
  ! True if using DOMDEC and node is pure reciprocal
  logical,save :: q_repd_domdec_purerecip = .false.
# endif
  ! True if memory allocated by repdstr_allocate_dims
  logical,save :: qdim_mem_allocd = .false.
  ! True if memory allocated by repdstr_allocate_hremd
  logical,save :: qhremd_mem_allocd = .false.
  ! Set to true if BLaDE is currently in use
  logical,save :: q_repd_want_blade = .false.

  ! SGLD - Hold SGLD parameters
  real(chm_real), dimension(10) :: our_sgarray, nbr_sgarray

  ! HREMD
  ! Save original coords and box
  real(chm_real), dimension(:), allocatable :: x0,y0,z0,wmain0
  real(chm_real), dimension(6)              :: xtlabc0

  ! N-D replica
  integer                               :: currdim       ! Current dimension during exchange TODO need save?
  integer                               :: exchange_dir  ! Current exchange direction in this dimension during exchange TODO need save?
  integer,save                          :: tdim_idx      ! Index of the temperature dimension (if any)
  integer,save                          :: phdim_idx     ! Index of the pH dimension (if any)
  integer,save                          :: sgdim_idx     ! Index of the SGLD dimension (if any)
  integer,save                          :: ndims         ! Number of dimensions
  integer,save,allocatable,dimension(:) :: exchange_type ! The type of exchange for each dimension [ndims]
  integer,save,allocatable,dimension(:) :: dimsize       ! The size of each dimension [ndims]
  integer,save,allocatable,dimension(:) :: mydims        ! My current position in each dimension [ndims]
  integer,save,allocatable,dimension(:) :: neighbor_up   ! My "up" neighbor in each dimension [ndims] 
  integer,save,allocatable,dimension(:) :: neighbor_down ! My "down" neighbor in each dimension [ndims] 
  integer,save,allocatable,dimension(:) :: exch_freq     ! Exchange frequency in each dimension [ndims]
  logical,allocatable,dimension(:)      :: dim_exchange  ! Set to T if dim will exchange this step [ndims]

  ! N-D log variables
  integer,save,allocatable,dimension(:) :: nopp_up       ! Number of up exchange opportunities in each dim [ndims]
  integer,save,allocatable,dimension(:) :: nopp_down     ! Number of down exchange opportunities in each dim [ndims]
  integer,save,allocatable,dimension(:) :: nattempts     ! Number exchange attempts in each dim [ndims]
  integer,save,allocatable,dimension(:) :: nsuc_up       ! Number up exchange attempts succeeded in each dim [ndims]
  integer,save,allocatable,dimension(:) :: nsuc_down     ! Number up exchange attempts succeeded in each dim [ndims]
  integer,save,allocatable,dimension(:) :: nsuccess      ! Number of exchange successes in each dim [ndims]
  real(chm_real),save,allocatable,dimension(:) :: rex_prob ! Track exchange probabilities

  ! Exchange types
  integer, parameter :: ex_temperature = 0
  integer, parameter :: ex_hamiltonian = 1
  integer, parameter :: ex_phrex       = 2
  integer, parameter :: ex_sgld        = 3

  ! Neighbor types
  integer, parameter :: nbrtype_none = -1
  integer, parameter :: nbrtype_rsv0 = -2 ! "Low" reservoir
  integer, parameter :: nbrtype_rsv1 = -3 ! "High" reservoir

  integer, save :: iunrxc                                ! Unit to write post-exchange coords to ('RXCR')
  ! --------------------------------------------------------
  !
  ! NB curtemps maps indices -> temperatures,
  real(chm_real),save                                  :: tempcurrent               ! for fast repexch
  real(chm_real),allocatable,dimension(:)              :: curtemps                  ! also for fast repexch
  logical                                              :: ffrun                     ! fast repexch logical for first run
  integer,save                                         :: loglevel                  ! fastrepexch how many exchanges be written
  logical, save                                        :: repd_update_temps         ! Will be set to true if temperatures should be
                                                                                    !   updated outside the exchange calculation.
  
  real(chm_real),allocatable,dimension(:),save,public  :: sgtemprx,sgftrx
  real(chm_real),allocatable,dimension(:),save         :: rhener, rlener            ! energies of what is in the reservoir
  real(chm_real4),allocatable,dimension(:),save        :: rescrdx,rescrdy,rescrdz   ! coordinates from reservoir
                                                                                    ! traj stuff in single prec

  logical,save                                         :: qecor

  ! Track and export exchange probabilities
  real(chm_real):: EXRATUP,EXRATDN,RERAT,REPROB
  integer,save  :: noppup,noppdn,nsucup,nsucdn

  !
  integer :: repdsynccount = 0
  real(chm_real),save :: rhtemp, rltemp ! temperatures of the reservoirs
  logical,save,public :: qsump,qrxsgld,qrxtham,qreservoir,qreshigh,qreslow,qfastrepdstr
#if KEY_PHMD==1
  logical,save,public :: qphrx 
#endif
#if KEY_BLOCK==1
  logical,save,public :: QMSPHRX    ! GG MSLD-compatibility
#endif

  ! discrete-state constant pH related variables
  logical,save,public :: qphrex
  real(chm_real),save,public  :: phrx(maxnode)
  real(chm_real)              :: rhph,rlph      ! pH values of the reservoir
  integer,allocatable,target,dimension(:)   :: nprothigh,nprotlow ! store number of protonated groups in each reservoir entry
  integer,allocatable,target,dimension(:,:) :: reshtstate,resltstate
  real(chm_real4),allocatable,dimension(:)  :: rescg

  logical,save,public :: qresboltz,qresnobo ! what exchange criterion to use for reservoir
  integer,save,public :: irex = 0     ! number of exchanges
  integer,save,public :: isuc = 0     ! number of successful exchanges
  integer,save,public :: iresexch = 0 ! number of reservoir exchanges
  integer,save,public :: rhunit,rlunit ! units for high and low reservoirs
  integer,save,public :: highressz,lowressz,maxressz ! current and maximum size(s) of the 
                                                     ! reservoir(s) in adaptive calculations
  integer,save,public :: repdid     ! replica id at current replica
  integer,save,public :: nrepeat    ! number of times to repeat exchange procedure
  integer,save        :: rsvrskip                    ! skip reservoir exchanges sometimes
  !
  !     TIGER stuff:
  !     (Temperature Intervals with Global Energy Reassignment)
  ! 
  !                     __,,,,_
  !          _ __..-;''`--/'/ /.',-`-.
  !      (`/' ` |  \ \ \\ / / / / .-'/`,_
  !     /'`\ \   |  \ | \| // // / -.,/_,'-,
  !    /<7' ;  \ \  | ; ||/ /| | \/    |`-/,/-.,_,/')
  !   /  _.-, `,-\,__|  _-| / \ \/|_/  |    '-/.;.\'
  !   `-`  f/ ;      / __/ \__ `/ |__/ |
  !        `-'      |  -| =|\_  \  |-' |
  !              __/   /_..-' `  ),'  //
  !             ((__.-'((___..-'' \__.'
  !
  !
  !     qrxtiger  - logical: are we using TIGER
  !     qpxtiger  - logical: do we need more dynamics before exchange
  !     qrxtmin   - logical: do we need to perform minimization at this step
  !     tigergr   - real: gradient tolerance for the minimization step
  !     tigerit   - integer: how many mini&equil cycles are needed
  !     tigeriti  - integer: how many cycles already performed
  !     tigernm   - integer: number of minimization steps
  !     tigerneq  - integer: number of equilibration steps
  !
  logical,save,public :: qrxtiger,qpxtiger=.false.,qrxtmin=.false.
  real(chm_real),save,public :: TIGERGR
  integer,save,public :: TIGERIT,TIGERNM,TIGERNEQ,TIGERITI=0
# endif /* KEY_REPDSTR */
  !
CONTAINS
  !
#if KEY_REPDSTR==1 /* repdstr_main */
!*******************************************************************************
!> Replica exchange 'REPD' command. Perform setup etc. 
!<
  SUBROUTINE REPDSTRMAIN
    !-----------------------------------------------------------------------
    !
#if KEY_PARALLEL==1 /* pll */
    !-----------------------------------------------------------------------
  use number
  use dimens_fcm
  use comand
  use cstuff, only: getpid
  use psf
  use coord
  use parallel
  use mpi
  use repdstr
  use stream
  use string
  use memory
  use deriv
  use block_ltm
  use lambdam    !GG MSLD-compatibility
  use consta     !GG MSLD-compatibility
  use param_store, only: set_param
# if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
# endif

  implicit none

    logical :: want_openmm
    LOGICAL QFIN,QFINSYNC,QFINONE,QSYNC
    !
    real(chm_real) STEMP,DTEMP,MTEMP,SGTEMP,DSGTEMP,MSGTEMP,SGFT,DSGFT, ttemp
    !
    character(len=100) FNAME
    INTEGER J !GG New MSLD pH-REX commands added
    INTEGER I,IOERR,MYPID,ENEUN,NAT3,RDIM,IERR
    INTEGER IUNPHN,IUNPHS
    !
    QFIN=(INDXA(COMLYN,COMLEN,'RESE').GT.0)
    !
    IF(QFIN)THEN
       ! If repd not previously called, no need to continue.
       if (.not.qrepdstr) return
       !        Must be called before qrepdstr=.false.
       !        Restore the I/O; can be called individually! No global comm.!
       CALL DREPRESIO(IOLEV,PRNLEV,WRNLEV)
       QREPDSTR = .FALSE.
       QRDQTT = .FALSE.
       QWRQTT = .FALSE.
       !!         IF(MYNODG.EQ.0)QRDQTT=.TRUE.
       !
       ! is iparpt OK?
       ! is ippmap OK?
       ! is INODE() OK?
       !
       !!QFINSYNC=(INDXA(COMLYN,COMLEN,'SYNC').GT.0)
       !!IF(QFINSYNC)CALL PSYNC()

       QFINONE=(INDXA(COMLYN,COMLEN,'PONE').GT.0)
       IF(QFINONE)THEN
          !           IF(MYNODG.EQ.0)THEN
          MYNOD=0
          NUMNOD=1
          CALL CUBE(mynod,numnod,ippmap)
          !           ENDIF
       ENDIF
       !!         write(outu,'(A,L)')'REPDSTR>after sync:qfinsync=',qfinsync

       IF(QFASTREPDSTR) &
          call chmdealloc('repdstr.src','REPDSTRMAIN','curtemps',NREPDSTR,crl=curtemps)
       IF(QRESERVOIR) THEN
          call chmdealloc('repdstr.src','REPDSTRMAIN','rescrdx',NATOM,cr4=rescrdx)
          call chmdealloc('repdstr.src','REPDSTRMAIN','rescrdy',NATOM,cr4=rescrdy)
          call chmdealloc('repdstr.src','REPDSTRMAIN','rescrdz',NATOM,cr4=rescrdz)
          call chmdealloc('repdstr.src','REPDSTRMAIN','rescg',NATOM,cr4=rescg)
       ENDIF

       call repdstr_deallocate

       ! Ditch sub-communicator and put everything back in one big tent
       CALL MPI_BARRIER(COMM_CHARMM,IERR)
       CALL MPI_BARRIER(COMM_UNIVERSE,IERR)

!       WRITE(OUTU,'(A,3I3)') 'REPDSTR> RESETTING COMM_CHARMM COMM_UNIVERSE MPI_COMM_NULL = ',COMM_CHARMM,COMM_UNIVERSE,MPI_COMM_NULL
       if (prnlev.gt.2) write(outu,'(A,I6)') 'REPDSTR> RESETTING COMMUNICATORS'
       CALL FLUSH(OUTU)

       CALL PSETGLOB
       
       CALL MPI_COMM_RANK(COMM_CHARMM,J,IERR)
       if (prnlev.gt.2) WRITE(OUTU,'(A,I3)') 'REPDSTR> RANK ON COMM_CHARMM = ',J
       CALL FLUSH(OUTU)
#      if KEY_DOMDEC==1
       ! Warn that this will invalidate DOMDEC communicators TODO only when split?
       if (q_domdec) then
         write(outu,'(a)') 'REPDSTR> Warning: DOMDEC is active.'
         write(outu,'(a)') 'REPDSTR> Warning: If DOMDEC was called after REPD it should be turned'
         write(outu,'(a)') 'REPDSTR> Warning:   off and set back up before DYNA is called again.'
       endif
#      endif
       ! Free the master REPD communicator since only REPD uses it.
       if (qrepmaster) call mpi_comm_free(comm_rep_master, ierr)
       ! NOTE: Do NOT free COMM_RPG here since other comms (e.g. in domdec)
       !       may have been set up based on it. COMM_RPG will be freed
       !       if REPD is set up again by setup_repd_comms.
       !CALL MPI_COMM_FREE(COMM_RPG,IERR)
       qrepmaster = .false.
!       WRITE(OUTU,'(A,I3)') 'REPDSTR> COMM_CHARMM HAS BEEN RESET TO ',COMM_CHARMM
!       CALL FLUSH(OUTU)
#      if KEY_DEBUGREPD==1
       write(100+mynodg,'(a,i10)') 'DBG: comm_charmm is ', comm_charmm
       flush(100+mynodg)
#      endif
       RETURN
    ENDIF

    !
    !
    !     This is global operation so every input script
    !     should have MATCHING repd sync!!!
    !
    QSYNC=(INDXA(COMLYN,COMLEN,'SYNC').GT.0)

    IF(QSYNC)THEN
       repdsynccount=repdsynccount+1
       if(prnlev.gt.2) then
          write(outu,'(a,i5)')'REPDSTR>sync count=',repdsynccount
       endif
       CALL MPI_BARRIER(COMM_CHARMM,IERR)
       CALL MPI_BARRIER(COMM_UNIVERSE,IERR)
       RETURN
    ENDIF

    !   sometimes we want this in our scripts :-)
    if (indxa(comlyn,comlen,'IORES').gt.0) then
       qrepioset = .false.
       return
    endif

    if (indxa(comlyn,comlen,'IOSET').gt.0) then
       qrepioset = .true.
       return
    endif
    !
    NREPDSTR = GTRMI(COMLYN,COMLEN,'NREP',1)
    !
    NATREPCMD = GTRMI(COMLYN,COMLEN,'NATR',-1)
    !
    QREPDSTR = .TRUE.
    !
    QFASTREPDSTR = (INDXA(COMLYN,COMLEN,'FAST').GT.0)
    IF(QFASTREPDSTR) THEN
       CALL CHMALLOC('repdstr.src','REPDSTRMAIN','CURTEMPS',NREPDSTR,crl=curtemps)
       FFRUN=.TRUE.

       LOGLEVEL=GTRMI(COMLYN,COMLEN,'LOGL',1)
    ENDIF

    qrepioset = .true.
    !
    !     This is the default situation. This changes when stream is called!
    QRDQTT = .FALSE.
    !     This is the default situation. This changes when outu is called!
    QWRQTT = .FALSE.

    NREPEAT=GTRMI(COMLYN,COMLEN,'REPE',1)
    IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I3)') 'REPDSTR> EXCHANGE REPEAT = ', NREPEAT

    q_use_new_exchange=.false.
    q_replica_exchange = .false.
    q_use_pv = .false.
    q_no_pv = (indxa(comlyn,comlen,'NOPV').gt.0)
    if (q_no_pv.and.prnlev.gt.2) write(outu,'(a)') 'REPD> Not using pressure/volume correction for NPT ensemble.'
    allow_multi_exchg = .false.
    ndims = gtrmi(comlyn,comlen,'NDIM',0)
    if (ndims .gt. 1) then
      call setup_nd(comlyn, comlen)
      return
    else if (indxa(comlyn,comlen,'TWOD').gt.0) then
       ndims = -2 ! -2 lets setup_nd know we are using original syntax
       call setup_nd(comlyn, comlen)
       return
    endif

    ! Tim Miller: June, 2011
    ! Made default feature: October, 2014
    ! Decide if we need to do discrete state PH based replica exchange
    QPHREX=(INDXA(COMLYN,COMLEN,'PHRE').GT.0)

    IF(QPHREX) THEN
       !    repd nrep <int> phrex freq <int> phval <real> phval <real> ...

       IF(QFASTREPDSTR) &
          CALL WRNDIE(-4,'<REPDSTR>','FAST PH REX IS NOT SUPPORTED')

       REPSEED=123+IREPDSTR
       IUNRXC=GTRMI(COMLYN,COMLEN,'RXCR',0)
       IUNREX=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       IREXFQ=GTRMI(COMLYN,COMLEN,'FREQ',1)
      
       DO I=1,NREPDSTR
          PHRX(I)=GTRMF(COMLYN,COMLEN,'PHVA',-ONE)
          !! Fix for Ana -- allow negative pH values
          !!IF(PHRX(I).LT.0) &
          !!   CALL WRNDIE(-5,'<REPDSTR>','NEGATIVE PH VALUES ARE NOT ALLOWED!')
       ENDDO
    ENDIF

    QREXCHG = (INDXA(COMLYN,COMLEN,'EXCH').GT.0)
    QREXCHGL = (INDXA(COMLYN,COMLEN,'EXLM').GT.0)
    ! Trap options that are unique to EXCH
    QRXTHAM=(INDXA(COMLYN,COMLEN,'THAM').GT.0)
    if (.not.QREXCHG.and.QRXTHAM) &
      call wrndie(-4,'<REPDSTR>','EXCH must be specified if THAM specified')

    ! Tim Miller: June, 2011
    ! This code takes care of reservoirs for Asim Okur's reservoir relica
    ! exchange method
    RSVRSKIP=1 ! defensive programming in-action
    QRESERVOIR=(INDXA(COMLYN,COMLEN,'RSRV').GT.0)
    QRESHIGH=(INDXA(COMLYN,COMLEN,'RESH').GT.0)
    QRESLOW=(INDXA(COMLYN,COMLEN,'RESL').GT.0)
    IF(QRESERVOIR) THEN

       IF(QFASTREPDSTR) &
          CALL WRNDIE(-4,'<REPDSTR>','FAST RESERVOIR REX IS NOT SUPPORTED')

       RSVRSKIP=GTRMI(COMLYN,COMLEN,'DAMJ',1)
       QRESBOLTZ=(INDXA(COMLYN,COMLEN,'BOLT').GT.0)
       QRESNOBO=(INDXA(COMLYN,COMLEN,'NOBO').GT.0)
       IF(.not. (QRESHIGH .or. QRESLOW)) &
          CALL WRNDIE(-3,'<REPDSTRMAIN>', 'RESERVOIR NEEDS RESHIGH OR RESLOW')

       IF(QREXCHGL) THEN
          ! Only allow "Boltzmann" for Hamiltonian REX
          QRESNOBO=.FALSE.
          QRESBOLTZ=.TRUE.
          IF(PRNLEV.GE.3) WRITE(OUTU,'(A)') &
             'REPDSTRMAIN> HAMILTONIAN REX IN USE. BOLTZMANN EXCHANGE CRITERION ACTIVATED.'
       ELSE
          ! check to make sure there aren't multiple specs
          IF(QRESBOLTZ .and. QRESNOBO) THEN
             CALL WRNDIE(-4,'<REPDSTRMAIN>', 'CONFLICTING EXCHANGE CRITERIA SET FOR RESERVOIR REX')
          ENDIF
          IF(.not. ( QRESBOLTZ .or. QRESNOBO )) THEN
             CALL WRNDIE(0,'<REPDSTRMAIN>', &
                         'NO EXCHANGE CRITERIA SET: RESERVOIR REX WILL USE BOLTZMANN CRITERION')
             QRESBOLTZ=.true.
          ENDIF
       ENDIF
       IF(QRESBOLTZ.OR.QRESNOBO) THEN
          QECOR=(INDXA(COMLYN,COMLEN,'ECOR').GT.0)
       ELSE
          QECOR=.FALSE.
       ENDIF

       call chmalloc('repdstr.src','REPDSTRMAIN','rescrdx',NATOM,cr4=rescrdx)
       call chmalloc('repdstr.src','REPDSTRMAIN','rescrdy',NATOM,cr4=rescrdy)
       call chmalloc('repdstr.src','REPDSTRMAIN','rescrdz',NATOM,cr4=rescrdz)
       call chmalloc('repdstr.src','REPDSTRMAIN','rescg',NATOM,cr4=rescg)

       ! Get info on the trajectory files containing the reservoirs      
       IF(QRESHIGH) THEN
          HIGHRESSZ=GTRMI(COMLYN,COMLEN,'RHSZ',-1)
          RHUNIT=GTRMI(COMLYN,COMLEN,'RHUN',-1)
          IF(RHUNIT.LT.0) &
             CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD UNIT FOR TOP RESERVOIR')

          IF(HIGHRESSZ.LT.0) CALL WRNDIE(-3,'<REPDSTRMAIN>','MUST SPECIFY A VALID HIGH RESERVOIR SIZE')

          IF(QREXCHGL) THEN
             RHTEMP=GTRMF(COMLYN,COMLEN,'RHTE',-1.0)
             IF(RHTEMP.LE.0) &
                CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD BOLTZMANN TEMP FOR HIGH RESERVOIR')
             ENEUN=GTRMI(COMLYN,COMLEN,'FHEN',-1)
             IF(ENEUN.LT.1) &
                CALL WRNDIE(-3,'<REPDSTRMAIN>','RESERVOIR H-REX CANNOT PRECALC ENERGIES.')
             CALL GET_ENE_VAL(ENEUN,QECOR,RLTEMP,.TRUE.)
          ELSE
             RHTEMP=GTRMF(COMLYN,COMLEN,'RHTE',-1.0)
             IF(QRESBOLTZ) THEN
                IF(QPHREX) THEN
                   ! handle boltzmann pH rex, hoorah
                   RHPH=GTRMF(COMLYN,COMLEN,'RHPH',-1.0)
                ELSE
                   ! Get temp of high reservoir
                   IF(RHTEMP.LE.0) &
                      CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD BOLTZMANN TEMP FOR HIGH RESERVOIR')
                ENDIF 
             ENDIF

             IF(QPHREX) THEN
                IUNPHN=GTRMI(COMLYN,COMLEN,'IUNH',-1)
                IUNPHS=GTRMI(COMLYN,COMLEN,'PHSH',-1)
             ELSE
                IF(QRESBOLTZ.OR.QRESNOBO) THEN
                   ENEUN=GTRMI(COMLYN,COMLEN,'FHEN',-1)
                   IF(ENEUN.GT.0) THEN
                      CALL GET_ENE_VAL(ENEUN,QECOR,RHTEMP,.TRUE.)
                   ELSE
                      CALL PRECALCENE(.TRUE.,X,Y,Z,QECOR,RHTEMP)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDIF
       IF(QRESLOW) THEN
          LOWRESSZ=GTRMI(COMLYN,COMLEN,'RLSZ',-1)
          RLUNIT=GTRMI(COMLYN,COMLEN,'RLUN',-1)          
          IF(RLUNIT.LT.0) &
             CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD UNIT FOR BOTTOM RESERVOIR')

          IF(LOWRESSZ.LT.0) CALL WRNDIE(-3,'<REPDSTRMAIN>','MUST SPECIFY A VALID LOW RESERVOIR SIZE')

          IF(QREXCHGL) THEN
             RLTEMP=GTRMF(COMLYN,COMLEN,'RLTE',-1.0)
             IF(RLTEMP.LE.0) &
                CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD BOLTZMANN TEMP FOR LOW RESERVOIR')
             ENEUN=GTRMI(COMLYN,COMLEN,'FLEN',-1)
             IF(ENEUN.LT.1) &
                CALL WRNDIE(-3,'<REPDSTRMAIN>','RESERVOIR H-REX CANNOT PRECALC ENERGIES.')
             CALL GET_ENE_VAL(ENEUN,QECOR,RLTEMP,.FALSE.)
          ELSE
             RLTEMP=GTRMF(COMLYN,COMLEN,'RLTE',-1.0)
             IF(QRESBOLTZ) THEN
                ! Get temp of high reservoir
                IF(QPHREX) THEN
                   ! handle boltzmann pH rex, hoorah
                   RLPH=GTRMF(COMLYN,COMLEN,'RLPH',-1.0)
                ELSE
                   IF(RLTEMP.LE.0) &
                      CALL WRNDIE(-3,'<REPDSTRMAIN>','BAD BOLTZMANN TEMP FOR LOW RESERVOIR')
                ENDIF
             ENDIF


             IF(QPHREX) THEN
                IF(QFASTREPDSTR) &
                   CALL WRNDIE(-4,'<REPDSTR>','FAST PH REX IS NOT SUPPORTED')

                IUNPHN=GTRMI(COMLYN,COMLEN,'IUNL',-1)
                IUNPHS=GTRMI(COMLYN,COMLEN,'PHSL',-1)

             ELSE
                IF(QRESBOLTZ.OR.QRESNOBO) THEN
                   ENEUN=GTRMI(COMLYN,COMLEN,'FLEN',-1)
                   IF(ENEUN.GT.0) THEN
                      CALL GET_ENE_VAL(ENEUN,QECOR,RLTEMP,.FALSE.)
                   ELSE
                      CALL PRECALCENE(.FALSE.,X,Y,Z,QECOR,RLTEMP)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDIF
    ELSE
       QRESBOLTZ=.false.
       QRESNOBO=.false.
    ENDIF

    !OLD:C
    !OLD:C     Redistribute the parallel setup
    !OLD:C     Current limitation (number of replica)/(number of processes) = 1
    !OLD:C     this way we know which replica we are dealing with, without changing
    !OLD:C     charmm PSF, COOR etc structures
    !OLD:C     If you change this rule do it also elsewhere like for example here:
    !OLD:C        - VOPEN() in machio.src
    !
    !     This is initialization. Use PSETLOC(), PSETGLOB() elsewhere
    !     This needs to be generalized!!! Also first two lines are now in paral1.src!!!
    !     No need to be here... IREPDSTR must be calculated from them
    !
    !      MYNODG=MYNOD
    !      NUMNODG=NUMNOD

    ! DRR - New replica setup based on initial work from BTM.
    !       This divides all threads among replicas and sets up a
    !       replica master comm.
    call setup_repd_comms( nrepdstr )

    !
    !      write(70+mynodg,'(a,4i7)')'REPDSTR-1>me,meg,np,npg=',
    !     $     mynod,mynodg,numnod,numnodg
    !      write(70+mynodg,'(a,4i7)')'REPDSTR-2>irepdstr,iolev,prnlev=',
    !     $     irepdstr,iolev,prnlev
    !
    !     Maybe here we could reassign mynode,numnode, too!?
    CALL set_param('MYREP',IREPDSTR)
    CALL set_param('NREP',NREPDSTR)
    !
    !     This is old, now we hope we deal with everything!!!
    !     At this point we could redefine IOLEV but there are case that it
    !     wouldn't work, so we deal with them individualy and currently it
    !     works for:
    !     1. OPEN      (machio.src: maybe we want also: VINQRE, VCLOSE)
    !     2. READ COOR (coorio.src: binary??)
    !     3. WRITE COOR
    !     4. TRAJIO
    !
    !     define the I/O flags:
    CALL DREPSETIO(IOLEV,PRNLEV,WRNLEV)
    !
    !     Replica exchange data: get the temperatures...
    QEXPT = (INDXA(COMLYN,COMLEN,'EXPT').GT.0)
    QEX2D = (INDXA(COMLYN,COMLEN,'EX2D').GT.0)
    QEXBK = (INDXA(COMLYN,COMLEN,'EXBK').GT.0)

    IF(QREXCHG) THEN
       !
       !     repd nrep <int> exch freq <int> temp <real> temp <real> ...
       !
       !     could be also:
       !     repd nrep <int> exch stemp <real> dtemp <real>
       !     where stemp is starting temperature and dtemp is the
       !     interval between the temperatures.
       !
       !         REPSEED=123+MYNODG
       REPSEED=123+IREPDSTR
       QSUMP=(INDXA(COMLYN,COMLEN,'SUMP').GT.0)
       QRXSGLD=(INDXA(COMLYN,COMLEN,'SGLD').GT.0)
       QRXTIGER=(INDXA(COMLYN,COMLEN,'TIGE').GT.0)
#if KEY_PHMD==1
       QPHRX=(INDXA(COMLYN,COMLEN,'PHMD').GT.0) !JAW. Flag for using PHMD and PH exchange
#endif
       IUNRXC=GTRMI(COMLYN,COMLEN,'RXCR',0)
       IUNREX=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       if (QRXTHAM) & 
         EWRITU = GTRMI(COMLYN, COMLEN, 'UEWR', -1)
       IREXFQ=GTRMI(COMLYN,COMLEN,'FREQ',1)
#if KEY_OPENMM==1
       want_openmm = (indxa(comlyn,comlen,'OMM')>0)
       qtor_repex = (indxa(comlyn,comlen,'TORS')>0)
       TTEMP=GTRMF(COMLYN,COMLEN,'TTEM',-ONE)
#endif
       STEMP=GTRMF(COMLYN,COMLEN,'STEM',-ONE)
       !!IF(QRXTHAM.AND.QRXSGLD) &
       !!   CALL WRNDIE(-4,'<REPDSTR>', &
       !!   'TEMP-HAMILTONIAN REPLICA EXCHANGE IS INCOMPATIBLE WITH SGLD REM')
       IF(STEMP>ZERO) THEN
          MTEMP=GTRMF(COMLYN,COMLEN,'MTEM',-ONE)
          IF(MTEMP>ZERO)THEN
             DTEMP=EXP(DLOG(MTEMP/STEMP)/(NREPDSTR-1))
             DO I=1,NREPDSTR
                TEMPRX(I)=STEMP*DTEMP**(I-1)
                IF(QFASTREPDSTR) CURTEMPS(I)=TEMPRX(I)
          ENDDO
          ELSE
             DTEMP=GTRMF(COMLYN,COMLEN,'DTEM',-ONE)
             IF(DTEMP.LT.ZERO) CALL WRNDIE(-5,'<REPDSTR>', &
                  'replica EXCHange needs interval between temperatures.')
             DO I=1,NREPDSTR
                TEMPRX(I)=STEMP+(I-1)*DTEMP
                IF(QFASTREPDSTR) CURTEMPS(I)=TEMPRX(I)
             ENDDO
          ENDIF
#if KEY_OPENMM==1
       else if(ttemp<=0) then
#else
       ELSE
#endif
          DO I=1,NREPDSTR
             TEMPRX(I)=GTRMF(COMLYN,COMLEN,'TEMP',-ONE)
             IF(TEMPRX(I).LT.ZERO) THEN
                write(outu,'(a,i4,a,i4,a)')'Number of specifed temperatures ',i, &
                     ', but ',nrepdstr,' are needed'
                CALL WRNDIE(-5,'<REPDSTR>', &
                     'replica EXCHange needs all temperatures.')
             ENDIF
             IF(QFASTREPDSTR) CURTEMPS(I)=TEMPRX(I)
          ENDDO
       ENDIF
#if KEY_OPENMM==1
       if(qtor_repex) then
          if(ttemp <=0) call wrndie(-5,'<REPDSTR>', &
               'Torsion space replica exchange needs varible TTEM set.')
          DO I=1,NREPDSTR
             TEMPRX(I)=GTRMF(COMLYN,COMLEN,'PHIS',-ONE)
             IF(TEMPRX(I)<=0) THEN
                write(outu,'(a,i4,a,i4,a)')'Number of specifed torsion scaling ',i, &
                     ', but ',nrepdstr,' are needed'
                CALL WRNDIE(-5,'<REPDSTR>', &
                     'replica EXCHange needs all torsion scalings (PHIS).')
             ENDIF
             IF(QFASTREPDSTR) CURTEMPS(I)=TEMPRX(I)
          enddo
            temprx(1:nrepdstr) = ttemp
        if(prnlev>=2) then
             write(outu,'(a,i4,a)') &
                  ' Using OpenMM to run Replica Exchance in torsion space using ', nrepdstr,' replicas'
             write(outu,'(a,f6.2)') ' Dynamics will be run at single temperature ', ttemp
             write(outu,'(a,20(f6.2))') ' Each replica dihedral potentials scaled by', (curtemps(i), i=1, nrepdstr)
          endif
       endif
#endif
       IF(QRXSGLD)THEN
          IF(QFASTREPDSTR) &
             CALL WRNDIE(-4,'<REPDSTR>','FAST SGLD REX IS NOT SUPPORTED')

          call setup_sgld( comlyn, comlen, nrepdstr )
       ENDIF
       !
       !     TIGER parameters parsed here:
       !
       IF(QRXTIGER)THEN
          TIGERGR=GTRMF(COMLYN,COMLEN,'TOLG',ZERO)
          TIGERIT=GTRMI(COMLYN,COMLEN,'ITER',1)
          TIGERNM=GTRMI(COMLYN,COMLEN,'NMIN',100)
          TIGERNEQ=GTRMI(COMLYN,COMLEN,'NEQU',1000)
       ENDIF
    ENDIF

    IF(QREXCHGL)THEN
       IF(QFASTREPDSTR) &
          CALL WRNDIE(-4,'<REPDSTR>','FAST LAMBDA REX IS NOT SUPPORTED')

       QSUMP=(INDXA(COMLYN,COMLEN,'SUMP').GT.0)

       REPSEED=123+MYNODG
       IUNREX=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       IREXFQ=GTRMI(COMLYN,COMLEN,'FREQ',1)
       EWRITU=GTRMI(COMLYN,COMLEN,'UEWR',IUNREX)
       !!IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I5)') 'TIM DEBUG> IREXFQ = ',IREXFQ

       IF((.NOT.QEXPT).AND.(.NOT.QEX2D)) THEN
          !     repd nrep <int> exlm freq <int>
          IUNRXC=GTRMI(COMLYN,COMLEN,'RXCR',0)
#if KEY_PHMD==1
          QPHRX=(INDXA(COMLYN,COMLEN,'PHMD').GT.0) !JAW. Flag for using PHMD and PH exchange
#endif
#if KEY_BLOCK==1
          QMSPHRX=(INDXA(COMLYN,COMLEN,'MSPH').GT.0) !GG: Flag for using pH-REX in CPHMD^MSLD
          IF(QMSPHRX)THEN
             !GG: Usage example "repd nrep <int> exlm freq <int> msph sph <int> mph <int>"
             IF (NREPDSTR .EQ. NREPLICA) THEN        !GG: Check number of replicas are the same in BLOCK and REPDSTR
                call msld_phrex(comlyn,comlen,nrepdstr,1,nblock)
             ELSE
                CALL WRNDIE(-5,'<REPDSTR>', &
                'Number of replicas declared in BLOCK and REPDSTR do not match!')
             ENDIF        !GG: For "NREPDSTR .EQ. NREPLICA" loop
          ENDIF           !GG: For "QMSPHRX" loop
#endif

          IF(QEXBK)IRBK=GTRMI(COMLYN,COMLEN,'REBK',1)
       ELSE IF (QEXPT) THEN
          !     repd nrep <int> exlm expt nrpt <int> freq <int>
          NREPT=GTRMI(COMLYN,COMLEN,'NRPT',1)
          IF(QEXBK)IRBK=GTRMI(COMLYN,COMLEN,'REBK',1)
       ELSE
          ! repd nrep <int> exlm ex2d nrpx <int> freq <int>
          NREPX=GTRMI(COMLYN,COMLEN,'NRPX',1)
          IF(QEXBK)IRBK=GTRMI(COMLYN,COMLEN,'REBK',1)
       ENDIF
    ENDIF

    ! Decide if exchange method should be used, and if so whether
    ! we are using the new method or the original one.
    if (qrexchg.or.qphrex.or.qrexchgl) then
      q_replica_exchange = .true.
      if (qrxtiger) then
        q_use_new_exchange = .false.
      else
        q_use_new_exchange = .true.
      endif
    endif
!    if (QREXCHG.or.QPHREX.or.QREXCHGL) then
!      if (&
!          .not.qmsphrx.and.&
!          .not.qphrx.and.&
!          .not.qreservoir.and.&
!          .not.qfastrepdstr.and.&
!#         if KEY_OPENMM==1
!          .not.qtor_repex.and.&
!#         endif
!          .not.qrxtiger &
!         ) q_use_new_exchange=.true.
!    endif

    ! initialize the starting temperature of this replica
    IF(QFASTREPDSTR) THEN
       TEMPCURRENT=CURTEMPS(IREPDSTR+1)
#if KEY_OPENMM==1
       if(qtor_repex) torsion_lambda = tempcurrent
#endif
       IF(PRNLEV.GE.6) &
          WRITE(OUTU,'(A,I3,A,F10.3)') 'FREX DEBUG> REPL ', IREPDSTR, ' INIT TEMPCURRENT = ', TEMPCURRENT
    ENDIF
    NOPPUP=0
    NOPPDN=0
    NSUCUP=0
    NSUCDN=0
    REPROB=0.0

    ! Initialize in case someone decides to try to use these"
    call set_param('REPROB',REPROB)
    call set_param('EXRUP',ZERO)
    call set_param('EXRDN',ZERO)

    ! Always set up tdim_idx, mydims(1), etc since things like 
    ! my_replica_temperature depend on it.
    tdim_idx = 1
    phdim_idx = 1
    sgdim_idx = 1
    if (.not.q_use_new_exchange) then
      allocate(mydims(1))
      mydims(1) = irepdstr 
    else
      ! Set up for 1 dimension
      ndims = 1
      ! Memory allocation for exchange calcs
      call repdstr_allocate_dims
      if (qphrex) then
        exchange_type(1) = ex_phrex
      else if (qrxtham) then
        call repdstr_allocate_hremd
        exchange_type(1) = ex_hamiltonian
      else if (qrxsgld) then
        exchange_type(1) = ex_sgld
      else
        exchange_type(1) = ex_temperature
      endif
      dimsize(1) = nrepdstr
      mydims(1) = irepdstr
      neighbor_up(1) = irepdstr + 1
      if (neighbor_up(1).ge.nrepdstr) then
        if (qreservoir.and.qreshigh) then
          neighbor_up(1) = nbrtype_rsv1
        else
          neighbor_up(1) = nbrtype_none
        endif
      endif
      neighbor_down(1) = irepdstr - 1
      if (neighbor_down(1).lt.0) then
        if (qreservoir.and.qreslow) then
          neighbor_down(1) = nbrtype_rsv0
        else
          neighbor_down(1) = nbrtype_none
        endif
      endif
      exch_freq(1) = irexfq 
    endif

    CALL FLUSH(OUTU)
    mypid = getpid()
    CALL PSYNC()

#else /* pll */
    CALL WRNDIE(-5,'<REPDSTR>', &
         'REPlica DiSTRibuted runs only in parallel.')
#endif /* pll */
    RETURN
  END SUBROUTINE REPDSTRMAIN
  !

!*******************************************************************************
!> Determine if pressure/volume correction should be used.
!> NOTE: Assuming every replica has qcnstp set to same value.
!<
subroutine setup_pv_correction(qcnstp)
  ! USE STATEMENTS
  use stream, only: outu, prnlev
  ! ARGUMENTS
  implicit none
  logical, intent(in) :: qcnstp ! If true constant pressure enabled.

  if (qcnstp) then
    if (.not.q_no_pv) then
      q_use_pv = .true.
      if (qrepmaster.and.prnlev.gt.0) &
        write(outu,'(a)') 'REPD> Using pressure/volume correction for exchange calc.'
    endif
  endif
end subroutine setup_pv_correction

!*******************************************************************************
!> Setup replica temperatures.
!<
subroutine setup_temperature(comlyn, comlen, n_temp_replicas)
  ! USE STATEMENTS
  use repdstr, only: temprx
  use stream,  only: outu, prnlev
  use string
  ! ARGUMENTS
  implicit none
  character(len=*)      :: comlyn
  integer               :: comlen
  integer, intent(in)   :: n_temp_replicas ! Number of replicas in temperature dimension
  ! LOCAL VARIABLES
  real(chm_real) :: stemp, mtemp, dtemp
  integer i

  ! Lowest temperature
  stemp = gtrmf(comlyn,comlen,'STEM',-1.d0)
  if (stemp > 0.d0) then
    ! Max temperature
    mtemp = gtrmf(comlyn,comlen,'MTEM',-1.d0)
    if (mtemp > 0.d0) then
      dtemp = exp(dlog(mtemp/stemp)/(n_temp_replicas-1))
      do i = 1,n_temp_replicas
        temprx(i) = stemp * dtemp**(i-1)
      enddo
    else
      dtemp = gtrmf(comlyn,comlen,'DTEM',-1.d0)
      if (dtemp.lt.0.d0) &
        call wrndie(-5,'<REPDSTR>', &
                    'Need either max temperature MTEM or temperature interval DTEM with STEM.')
      do i = 1,n_temp_replicas
        temprx(i) = stemp + (i-1)*dtemp
      enddo
    endif
  else 
    do i = 1, n_temp_replicas
      temprx(i) = gtrmf(comlyn,comlen,'TEMP',-1.0)
      if (temprx(i) .lt. 0.d0) &
        call wrndie(-5,'<REPDSTR>',&
                    'Ran out of temperatures for temperature dimension.')
    enddo
  endif
  do i = 1, n_temp_replicas
    if (qfastrepdstr) curtemps(i) = temprx(i)
      if (prnlev.gt.2) &
        write(outu,'(8x,a,i8,a,f10.4)') 'Index ', i-1, ' temperature ', temprx(i)
  enddo

end subroutine setup_temperature

!*******************************************************************************
!> Setup SGLD replica exchange run.
!<
subroutine setup_sgld(comlyn, comlen, n_sg_replicas)
  ! USE STATEMENTS
  use repdstr, only: temprx
  use stream,  only: outu, prnlev
  use string
  ! ARGUMENTS
  implicit none
  character(len=*)      :: comlyn
  integer               :: comlen
  integer, intent(in)   :: n_sg_replicas ! Number of replicas in SGLD dimension
  ! LOCAL VARIABLES
  real(chm_real) :: sgtemp, sgft, dsgft, msgtemp, dsgtemp
  integer i

  ! Sanity check
  if (.not.qrxsgld) return

  allocate(sgtemprx(n_sg_replicas))
  allocate(sgftrx(n_sg_replicas))
  sgtemp = gtrmf(comlyn,comlen,'SGTE',0.d0)
  sgft = gtrmf(comlyn,comlen,'SGFT',0.d0)
  dsgft = gtrmf(comlyn,comlen,'DSGF',0.d0)
  do i=1,n_sg_replicas
    sgftrx(i) = sgft + (i-1)*dsgft
  enddo
  msgtemp = gtrmf(comlyn,comlen,'MSGT',-1.d0)
  if (sgtemp > 0.d0) then
    if (msgtemp > 0.d0) then
      dsgtemp = exp(dlog(msgtemp/sgtemp)/(n_sg_replicas-1))
      do i=1,n_sg_replicas
        sgtemprx(i) = sgtemp * dsgtemp**(i-1)
      enddo
    else
      dsgtemp = gtrmf(comlyn,comlen,'DSGT',0.d0)
      do i=1,n_sg_replicas
        sgtemprx(i) = sgtemp + (i-1)*dsgtemp
      enddo
    endif
  else
    do i=1,n_sg_replicas
      sgtemprx(i) = gtrmf(comlyn,comlen,'SGTT',-1.d0)
      if(sgtemprx(i).lt.0.d0) then
        ! FIXME: will break in ND if size of T dim less than SG dim
        if(prnlev>2)write(outu,'(a,i4,a)')'No SGTT input on stage ',i, &
             ', set to 0'
        sgtemprx(i) = 0.0d0 
      endif
    enddo
  endif
  if (prnlev.gt.2) then
    do i=1,n_sg_replicas
      write(outu,'(4x,a,i8,a,f10.4,a,f10.4,a,f8.4)') 'Replica ', i-1, ' Temp ',temprx(i),&
               ' TempSG ', sgtemprx(i),' sgft ',sgftrx(i)
    enddo
  endif
end subroutine setup_sgld

!*******************************************************************************
!> \return exchange type base on given key.
!> Also set temperature, SG, or pH dim index as appropriate. Check that only
!! one of these are set at a time.
!<
function exch_type_from_key(keyin, dimin)
  ! ARGUMENTS
  implicit none
  integer exch_type_from_key
  character(len=4), intent(in) :: keyin ! Key specifying exchange type
  integer, intent(in)          :: dimin ! Current dimension being set

  exch_type_from_key = -1
  if (keyin == 'TEMP') then
    exch_type_from_key = ex_temperature
    if (tdim_idx .gt. 0) &
      call wrndie(-5,'<SETUP_ND>','ONLY ONE DIMENSION CAN USE TEMPERATURE EXCHANGE!')
    tdim_idx = dimin
  else if (keyin == 'HAM') then
    exch_type_from_key = ex_hamiltonian
    qrxtham = .true. ! TODO remove dependency on this?
  else if (keyin == 'SGLD') then
    exch_type_from_key = ex_sgld
    if (sgdim_idx .gt. 0) &
      call wrndie(-5,'<SETUP_ND>','ONLY ONE DIMENSION CAN USE SGLD EXCHANGE!')
    sgdim_idx = dimin
    qrxsgld = .true. ! TODO remove dependency on this?
  else if (keyin == 'PH') then
    exch_type_from_key = ex_phrex
    if (phdim_idx .gt. 0) &
      call wrndie(-5,'<SETUP_ND>','ONLY ONE DIMENSION CAN USE PH EXCHANGE!')
    phdim_idx = dimin
    qphrex = .true. ! TODO remove dependency on this? 
  else
    call wrndie(-5,'<SETUP_ND>','UNRECOGNIZED EXCHANGE CRITERIA FOR DIMENSION.')
  endif
end function exch_type_from_key

!*******************************************************************************
!> Find replica neighbor in given dimension and direction 
!<
function find_neighbor(idx, dir, indices, nbr_idx)
  ! USE STATEMENTS
  use repdstr, only: irepdstr, nrepdstr
  ! ARGUMENTS
  implicit none
  integer find_neighbor
  integer, intent(in) :: idx                      ! Dimension to search
  integer, intent(in) :: dir                      ! Direction to search
  integer, intent(in), dimension(:,:) :: indices  ! Indices for each replica
  integer, intent(inout), dimension(:) :: nbr_idx ! Temp. array for holding neighbor indices
  ! LOCAL VARS
  logical search, ismatch
  integer rep, jdx

  find_neighbor = nbrtype_none
  ! Target neighbor will be +dir in this dimension, +0 elsewhere.
  search = .true.
  do jdx = 1, ndims
    if (jdx .eq. idx) then
      nbr_idx(jdx) = mydims(jdx) + dir 
      if (nbr_idx(jdx) .eq. dimsize(jdx) .or. nbr_idx(jdx) .lt. 0) then
        search = .false.
        exit
      endif
    else
      nbr_idx(jdx) = mydims(jdx)
    endif
  enddo
  if (search) then
    do rep = 1, nrepdstr
      ismatch = .true.
      do jdx = 1, ndims
        if (nbr_idx(jdx) .ne. indices(rep, jdx)) then
          ismatch = .false.
          exit
        endif
      enddo
      if (ismatch) then
        find_neighbor = rep - 1
        exit
      endif
    enddo
  endif ! search
end function find_neighbor

!*******************************************************************************
!> Set up my replica indices and neighbor replica indices for each dimension. 
!<
subroutine setup_remd_indices()
  ! USE STATEMENTS
  use repdstr,     only: irepdstr, nrepdstr
  use stream,      only: outu !, iolev, prnlev
  ! ARGUMENTS
  implicit none
  ! LOCAL VARS
  integer, allocatable :: indices(:,:) ! Replica, dimension indices
  integer, allocatable :: nbr_idx(:)   ! Temp array for neighbor indices
  integer rep, idx
  logical doincrement
  integer jdx

  ! First determine everyones own indices
  allocate( indices(nrepdstr, ndims), nbr_idx(ndims) )
  ! Initial indices for replica 0
  do idx = 1, ndims
    indices(1, idx) = 0
  enddo
  do rep = 2, nrepdstr
    ! Increment indices
    doincrement = .true. 
    do idx = 1, ndims
      if (doincrement) then
        ! Increment this index
        indices(rep, idx) = indices(rep-1, idx) + 1
        ! Check if index has cycled
        if (indices(rep, idx) .eq. dimsize(idx)) then
          indices(rep, idx) = 0
        else
          doincrement = .false.
        endif
      else
        ! Previous index did not cycle. use value from previous replica
        indices(rep, idx) = indices(rep-1, idx)
      endif
    enddo
  enddo
  ! DEBUG: Write indices
  if (irepdstr.eq.0.and.qrepmaster) then
    write(outu,'(a)') 'REPD> Replica Indices:'
    do rep = 1, nrepdstr
      write(outu,'(a,i6,a)', advance='no') 'Rep ', rep-1, ' : '
      do idx = 1, ndims
        write(outu,'(i6)', advance='no') indices(rep,idx)
      enddo
      write(outu,'(a)') ' '
    enddo
  endif
  ! Set my indices
  do idx = 1, ndims
    mydims(idx) = indices(irepdstr+1, idx)
  enddo
  ! Find my neighbors in each dimension
  do idx = 1, ndims
    jdx = find_neighbor(idx, 1, indices, nbr_idx)
    !write(outu, '(3(a,i6))') 'DBG: Replica ', irepdstr, ' dim ', idx, ' up nbr is ', jdx
    neighbor_up(idx) = jdx
    jdx = find_neighbor(idx, -1, indices, nbr_idx)
    !write(outu, '(3(a,i6))') 'DBG: Replica ', irepdstr, ' dim ', idx, ' dn nbr is ', jdx
    neighbor_down(idx) = jdx
  enddo ! neighbor search in each dimension

  ! Free up memory
  deallocate( indices )
  deallocate( nbr_idx )
end subroutine setup_remd_indices

!*******************************************************************************
!> Set parameter in dimension N. For backwards compat. do not append number when
!> idx is 1 for real values.
!<
subroutine set_remd_param(param, idx, dval, ival)
  use string
  use param_store, only: set_param
  implicit none
  ! ARGUMENTS
  character(len=*), intent(in) :: param
  integer, intent(in)          :: idx    ! DIMENSION
  real(chm_real), intent(in), optional :: dval    ! Parameter to set
  integer, intent(in), optional        :: ival    ! Parameter to set
  ! LOCAL VARIABLES
  character(len=16) :: parstr
  character(len=10) :: dimstr
  integer ldim, lpar, lparmax
  integer ptype
  
  ptype=0                              !  0 is nothing or both present
  if (present(dval)) ptype = ptype + 1 !  1 is real
  if (present(ival)) ptype = ptype - 1 ! -1 is int
  if (ptype.eq.0) &
    call wrndie(-6,'<SET_REMD_PARAM>','INTERNAL ERROR: MUST SPECIFY EITHER DVAL OR IVAL')
  parstr = param
  ! If integer always append. If real, only append for dim > 1
  if (ptype.eq.-1 .or. idx.gt.1) then
    ! 10 is definitely overkill...
    write(dimstr,'(i10)') idx
    ldim=10
    call trima(dimstr, ldim)
    ! Parameter
    lpar = 16 
    call trima(parstr, lpar)
    call addst(parstr, 16, lpar, dimstr, ldim)
  endif
  if (ptype .eq. 1) then
    call set_param(trim(parstr), dval)
  else
    call set_param(trim(parstr), ival)
  endif
end subroutine

!*******************************************************************************
!> Setup N-dimensional replica exchange run.
!<
subroutine setup_nd(comlyn, comlen)
  ! USE STATEMENTS
  use string
  use stream,      only: outu, iolev, prnlev
  use repdstr,     only: iunrex, irexfq, qrepdstr, qrexchg, temprx, nrepdstr, irepdstr, comm_rpg
  use param_store, only: set_param
  use mpi
  ! ARGUMENTS
  implicit none
  character(len=*)      :: comlyn
  integer               :: comlen
  ! LOCAL VARIABLES
  character(len=4)      :: scratch ! For getting replica criteria
  real(chm_real)        :: tempnd  ! N-D temperature when no T dim
  integer i, j, ierr
  logical original_syntax
  integer numreps

  q_replica_exchange = .true.
  q_use_new_exchange = .true.
  numreps = 0

  if (ndims .eq. -2) then
    original_syntax = .true.
    ndims = 2
  else
    original_syntax = .false.
  endif

  ! Allocate memory
  call repdstr_allocate_dims

  ! Exchange log unit
  iunrex = gtrmi(comlyn,comlen,'UNIT',-1)

  ! Check if multiple dimensions can exchange in the same attempt
  allow_multi_exchg = (indxa(comlyn,comlen,'MULT').gt.0)

  ! Get dimension-related variables
  tdim_idx = 0
  phdim_idx = 0
  sgdim_idx = 0
  if (original_syntax) then
    ! Get size in each dimension
    dimsize(1) = gtrmi(comlyn,comlen,'DIM1',0)
    dimsize(2) = gtrmi(comlyn,comlen,'DIM2',0)
    ! Get exchange frequency in each dimension
    exch_freq(1) = gtrmi(comlyn,comlen,'D1FR',-1)
    exch_freq(2) = gtrmi(comlyn,comlen,'D2FR',-1)
    ! Get dimension types.
    scratch = gtrma(comlyn,comlen,'D1CR')
    exchange_type(1) = exch_type_from_key( scratch, 1 )
    scratch = gtrma(comlyn,comlen,'D2CR')
    exchange_type(2) = exch_type_from_key( scratch, 2 )
    ! Total number of replicas
    numreps = dimsize(1) * dimsize(2)
  else
    do i = 1, ndims
      ! Get size in this dimension
      dimsize(i) = gtrmi(comlyn, comlen,'DIMS', 0) ! DIMSize 
      if (dimsize(i) .le. 0) &
        call wrndie(-5,'<SETUP_ND>','MUST SPECIFY VALID SIZE FOR EACH DIMENSION')
      ! Get exchange frequency in each dimension
      exch_freq(i) = gtrmi(comlyn, comlen, 'DFRE', -1) ! DFREquency
      if (exch_freq(i) .lt. 1) &
        call wrndie(-5,'<SETUP_ND>','MUST SPECIFY VALID EXCHANGE FREQ. FOR EACH DIMENSION')
      ! Get dimension type
      scratch = gtrma(comlyn, comlen, 'DCRI') ! DCRIteria
      exchange_type(i) = exch_type_from_key(scratch, i)
      ! Total number of replicas
      if (numreps .eq. 0) then
        numreps = dimsize(i)
      else
        numreps = numreps * dimsize(i)
      endif
    enddo
  endif

  ! Check input variables
  do i = 1,ndims
    if (prnlev.gt.2) &
      write(outu,'(3(a,i8))') 'REPD> SETUP_ND: Dim ', i, ' size= ', dimsize(i), ', freq= ', exch_freq(i)
    if (dimsize(i) <= 0) call wrndie(-5,'<SETUP_ND>','DIMENSION SIZE MUST BE > 0')
    if (exch_freq(i) < 1) call wrndie(-5,'<SETUP_ND>','EXCHANGE FREQUENCY MUST BE > 0')
  enddo
  if (allow_multi_exchg.and.prnlev.gt.2) &
    write(outu,'(a)') 'REPD> Allowing multiple dimensions to exchange in same attempt.'

  ! FIXME are these three statements still needed?
  irexfq = minval(exch_freq)
  qrepdstr = .true.
  qrexchg = .true.

  if (qrxtham) call repdstr_allocate_hremd ! TODO remove dependency on this var?

  ! If a temperature dimension is defined set it up first since other
  ! dimensions may depend on temperature. Otherwise the HTEM
  ! keyword is needed.
  if (tdim_idx .gt. 0) then
    if (prnlev.gt.2) &
      write(outu,'(2(a,i8))') 'REPD> Set up temperature dimension ', tdim_idx, ', size ', dimsize(tdim_idx)
    call setup_temperature(comlyn, comlen, dimsize(tdim_idx))
  else
    ! No temperature dimension assigned. Get temperature for all dims.
    tempnd = gtrmf(comlyn,comlen,'HTEM',-1.0)
    if (tempnd .lt. 0.0) &
      call wrndie(-5,'<SETUP_ND>',&
        'NO TEMPERATURE DIMENSION. MUST SPECIFY TEMPERATURE > 0.0 WITH HTEM <TEMP>')
    if (prnlev.gt.2) &
      write(outu, '(a,f10.4)') 'REPD> All replicas will be at temperature ', tempnd
    do i = 1, maxval(dimsize)
      temprx(i) = tempnd
    enddo
    tdim_idx = 1 ! Does not matter what this gets set to, all temps same.
  endif

  ! Set up remaining dimensions
  do i = 1, ndims
    if (exchange_type(i) .eq. ex_sgld) then
      if (prnlev.gt.2) &
        write(outu,'(2(a,i8))') 'REPD> Setting up SGLD dimension ', i, ', size ', dimsize(i) 
      call setup_sgld( comlyn, comlen, dimsize(i) )
    else if (exchange_type(i) .eq. ex_phrex) then
      if (prnlev.gt.2) &
        write(outu, '(2(a,i8))') 'REPD> Set up pH dimension ', i,', size ', dimsize(i)
      do j = 1, dimsize(i)
        phrx(j) = gtrmf(comlyn,comlen,'PHVA',-1.0)
        if (phrx(j) .lt. 0.0) &
           call wrndie(-5,'<SETUP_ND>', 'RAN OUT OF PH VALUES FOR PH DIMENSION.')
        if (prnlev.gt.2) &
          write(outu,'(8x,a,i8,a,f10.4)') 'Index ', j-1, ' pH ', phrx(j)
      enddo
    else if (exchange_type(i) .eq. ex_hamiltonian) then
      if (prnlev.gt.2) &
        write(outu, '(2(a,i8))') 'REPD> Set up Hamiltonian dimension ', i,', size ', dimsize(i)
    endif
  enddo

  ! Set up replica communicators
  call setup_repd_comms(numreps)
  call drepsetio(iolev,prnlev,wrnlev)

  ! Set my replica indices
  call setup_remd_indices

  if (qrepmaster) then
    if (irepdstr.eq.0) write(outu,'(a)') 'REPD>    Dim NbrDwn MyIndx  NbrUp'
    do j = 1, nrepdstr
      if (j-1.eq.irepdstr) then
        do i = 1, ndims
          write(outu,'(a,4(x,i6))') 'REPD>', i, neighbor_down(i), irepdstr, neighbor_up(i)
        enddo
      endif
    enddo
  endif
  call mpi_barrier(comm_rpg, ierr)

  ! Set parameters 
  call set_param('MYREP',irepdstr)
  call set_param('NREP',nrepdstr)
  ! Set dimension-related parameters
  do i = 1, ndims
    call set_remd_param('MYREPD', i, ival=mydims(i))
    call set_remd_param('NREPD', i, ival=dimsize(i))
    ! Probabilities
    call set_remd_param('REPROB', i, dval=0.d0)
    call set_remd_param('EXRUP', i, dval=0.d0)
    call set_remd_param('EXRDN', i, dval=0.d0)
  enddo

end subroutine setup_nd

!*******************************************************************************
!> Allocate memory for HREMD exchange calcs.
!<
subroutine repdstr_allocate_hremd
  ! USE STATEMENTS
  use psf, only: natom
  use memory
  ! LOCAL VARIABLES
  implicit none
  if (q_use_new_exchange) then
    ! Sanity check
    if (qhremd_mem_allocd) then
      ! For now do not allow new repd without reset of old repd
      call wrndie(-5,'<REPDSTR_ALLOCATE_HREMD>',&
                  'Must call REPD RESEt before new invocation of REPD.') 
    else
      call chmalloc('repdstr.src','repdstr_allocate_hremd','X0',    natom,crl=x0)
      call chmalloc('repdstr.src','repdstr_allocate_hremd','Y0',    natom,crl=y0)
      call chmalloc('repdstr.src','repdstr_allocate_hremd','Z0',    natom,crl=z0)
      call chmalloc('repdstr.src','repdstr_allocate_hremd','WMAIN0',natom,crl=wmain0)
      qhremd_mem_allocd = .true.
    endif
  endif
end subroutine repdstr_allocate_hremd

!*******************************************************************************
!> Allocate memory for exchange calc dimensions.
!<
subroutine repdstr_allocate_dims
  ! USE STATEMENTS
  use memory
  ! LOCAL VARIABLES
  implicit none
  integer cdim
  ! Memory allocation for exchange calcs
  if (q_use_new_exchange) then
    ! Sanity check
    if (qdim_mem_allocd) then
      ! For now do not allow new repd without reset of old repd
      call wrndie(-5,'<REPDSTR_ALLOCATE_DIMS>',&
                  'Must call REPD RESEt before new invocation of REPD.') 
    else
      if (ndims.gt.0) then
        ! NOTE: Using allocate for mydims to be consistent with the 1D case
        allocate(mydims(ndims))
        call chmalloc('repdstr.src','repdstr_allocate_dims','exchange_type',ndims,intg=exchange_type)
        call chmalloc('repdstr.src','repdstr_allocate_dims','dimsize',      ndims,intg=dimsize)
!        call chmalloc('repdstr.src','repdstr_allocate_dims','mydims',       ndims,intg=mydims)
        call chmalloc('repdstr.src','repdstr_allocate_dims','neighbor_up',  ndims,intg=neighbor_up)
        call chmalloc('repdstr.src','repdstr_allocate_dims','neighbor_down',ndims,intg=neighbor_down)
        call chmalloc('repdstr.src','repdstr_allocate_dims','exch_freq',    ndims,intg=exch_freq)
        call chmalloc('repdstr.src','repdstr_allocate_dims','dim_exchange', ndims,log=dim_exchange)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nopp_up',      ndims,intg=nopp_up)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nopp_down',    ndims,intg=nopp_down)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nattempts',    ndims,intg=nattempts)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nsuc_up',      ndims,intg=nsuc_up)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nsuc_down',    ndims,intg=nsuc_down)
        call chmalloc('repdstr.src','repdstr_allocate_dims','nsuccess',     ndims,intg=nsuccess)
        call chmalloc('repdstr.src','repdstr_allocate_dims','rex_prob',     ndims,crl=rex_prob)
        do cdim = 1, ndims
          nopp_up(cdim) = 0
          nopp_down(cdim) = 0
          nattempts(cdim) = 0
          nsuc_up(cdim) = 0
          nsuc_down(cdim) = 0
          nsuccess(cdim) = 0
          rex_prob(cdim) = 0.d0
        enddo
        qdim_mem_allocd = .true.
      endif
    endif
  endif
end subroutine repdstr_allocate_dims

!*******************************************************************************
!> Deallocate any memory allocated during repdstrmain 
!<
subroutine repdstr_deallocate
  use mpi
  use memory
  use psf, only: natom
  implicit none
  integer ierr

  ! NOTE: The following are allocated for repd with allocate()
  if (allocated(mydims))   deallocate(mydims)
  if (allocated(sgtemprx)) deallocate(sgtemprx)
  if (allocated(sgftrx))   deallocate(sgftrx)
  if (qdim_mem_allocd) then
    call chmdealloc('repdstr.src','repdstr_deallocate','exchange_type',ndims,intg=exchange_type)
    call chmdealloc('repdstr.src','repdstr_deallocate','dimsize',      ndims,intg=dimsize)
!    call chmdealloc('repdstr.src','repdstr_deallocate','mydims',       ndims,intg=mydims)
    call chmdealloc('repdstr.src','repdstr_deallocate','neighbor_up',  ndims,intg=neighbor_up)
    call chmdealloc('repdstr.src','repdstr_deallocate','neighbor_down',ndims,intg=neighbor_down)
    call chmdealloc('repdstr.src','repdstr_deallocate','exch_freq',    ndims,intg=exch_freq)
    call chmdealloc('repdstr.src','repdstr_deallocate','dim_exchange', ndims,log=dim_exchange)
    call chmdealloc('repdstr.src','repdstr_deallocate','nopp_up',      ndims,intg=nopp_up)
    call chmdealloc('repdstr.src','repdstr_deallocate','nopp_down',    ndims,intg=nopp_down)
    call chmdealloc('repdstr.src','repdstr_deallocate','nattempts',    ndims,intg=nattempts)
    call chmdealloc('repdstr.src','repdstr_deallocate','nsuc_up',      ndims,intg=nsuc_up)
    call chmdealloc('repdstr.src','repdstr_deallocate','nsuc_down',    ndims,intg=nsuc_down)
    call chmdealloc('repdstr.src','repdstr_deallocate','nsuccess',     ndims,intg=nsuccess)
    call chmdealloc('repdstr.src','repdstr_deallocate','rex_prob',     ndims,crl=rex_prob)
    qdim_mem_allocd = .false.
  endif
  if (qhremd_mem_allocd) then
    call chmdealloc('repdstr.src','repdstr_deallocate','X0',    natom,crl=x0)
    call chmdealloc('repdstr.src','repdstr_deallocate','Y0',    natom,crl=y0)
    call chmdealloc('repdstr.src','repdstr_deallocate','Z0',    natom,crl=z0)
    call chmdealloc('repdstr.src','repdstr_deallocate','WMAIN0',natom,crl=wmain0)
    qhremd_mem_allocd = .false.
  endif 
end subroutine repdstr_deallocate

!*******************************************************************************
!> Preliminary setup for 'FAST' version of replica exchange calc. 
!<
  SUBROUTINE FASTREPEXCHG(WMAIN,EPOT,ISTEP,JHSTRT,IGVOPT,VX,VY,VZ,XOLD,YOLD,ZOLD &
#if KEY_TSM==1
                          ,BACKLS &
#endif
                         )
     use mpi
     use memory
     use parallel
     use repdstr
     use stream
     use number
     use consta
     use clcg_mod,only: random

     ! Arguments
     REAL(CHM_REAL),INTENT(IN)    :: WMAIN(*),EPOT
     REAL(CHM_REAL),INTENT(INOUT) :: VX(*),VY(*),VZ(*),XOLD(*),YOLD(*),ZOLD(*)
     INTEGER,INTENT(IN)           :: ISTEP
     INTEGER,INTENT(OUT)          :: JHSTRT
     ! stupidva
     INTEGER,INTENT(INOUT)        :: IGVOPT
#if KEY_TSM==1
     INTEGER BACKLS(*)
#endif

     ! Local variables
     INTEGER,PARAMETER                       :: TREMFLAG=1,HREMFLAG=2

     INTEGER                                 :: I,J,X,IERR,REP1,REP2,TMPUN,EXTYPE,ME,STEP,NPERREP
     INTEGER                                 :: OURSELVES,NEIGHBOR,CNEIGHBOR,REPORDER(MAXREPDSTR)
     LOGICAL                                 :: QEXC,QMASTER,QDOLOG
     REAL(CHM_REAL)                          :: EPOTARR(MAXREPDSTR),OURTEMP
     REAL(CHM_REAL)                          :: PROBARR(MAXREPDSTR)
     REAL(CHM_REAL)                          :: NBRTEMP,P,PROB,LOWTEMP,LASTLOW,REXP
     REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: SCARR

     IF(MOD(ISTEP-1,IREXFQ).NE.0) RETURN
     ! DRR - I am removing this call to PSETGLOB and the call to PSETLOC
     !       below. Currently only seems to matter for SWAP_CRD below,
     !       which is currently never executed (see additional note below).
     !       All other MPI calls are to COMM_UNIVERSE. Requires modifying
     !       MYNOD=>MYNODG, NUMNOD=>NUMNODG, etc.
     !CALL PSETGLOB()

     EXTYPE=TREMFLAG
    
     IF(PRNLEV.GE.6) WRITE(OUTU,'(A,I6)') 'FREX DEBUG> IN FASTREPEXCHG AT STEP ', ISTEP
     IF(NREPDSTR.GT.MAXREPDSTR) &
        CALL WRNDIE(-5,'<FASTREPEXCHG>','TOO MANY REPLICAS FOR FAST EXCHANGING')
     DO I=1,MAXREPDSTR
        QEXC=.FALSE.
     ENDDO
     PROBARR(1:NREPDSTR)=ZERO

     NPERREP=NUMNODG/NREPDSTR
     CALL CHMALLOC('repdstr.src','FASTREPEXCHG','scarr',NUMNODG,crl=SCARR)

     CALL MPI_GATHER(EPOT,1,MPI_REAL8,SCARR,1,MPI_REAL8,0,COMM_UNIVERSE,IERR)
     IF(IERR.NE.MPI_SUCCESS) &
        CALL WRNDIE(-5,'<FASTREPEXCHG>','BUNGLED MPI COMMUNICATION')
     IF(MYNODG.EQ.0) THEN
        J=1
        DO I=1,NREPDSTR
           EPOTARR(I)=SCARR(J)
           J=J+NPERREP
        ENDDO
     ENDIF

     ! We need to actually figure out how the exchanges are going to
     ! happen, to do so, apply the formula and temps to the energies.
     DO X=1,NREPEAT
        ! DRR - Currently there is no way EXTYPE will ever equal HREMFLAG.
        !       Unless/until that changes, can safely ignore this block
        !       of code.
        IF(EXTYPE.EQ.HREMFLAG) THEN
           ! We need to send our coordinates to our neighbors and do a test exchange

           ! Figure out who our neighbor is
           STEP=MOD(IREX,2)
           ME=MOD(IREPDSTR,2)

           IF(STEP.EQ.1)THEN
              NEIGHBOR=IREPDSTR+1
              IF(ME.NE.0)NEIGHBOR=IREPDSTR-1
           ELSE
              NEIGHBOR=IREPDSTR-1
              IF(ME.NE.0)NEIGHBOR=IREPDSTR+1
           ENDIF

           IF(NEIGHBOR.LT.1.OR.NEIGHBOR.GT.NREPDSTR) THEN
              NEIGHBOR=-1
              CNEIGHBOR=-1
           ELSE
              CNEIGHBOR=NEIGHBOR*NUMNODG
              CALL SWAP_CRD(CNEIGHBOR,VX,VY,VZ)
           ENDIF
        ENDIF

        IREX=IREX+1
        IF(MYNODG.EQ.0) THEN

           IF(FFRUN) THEN
              WRITE(IUNREX,'(A)') '# replica temp. ener. neighbor ntemp nene prob p success? newrep'
              FFRUN=.FALSE.
           ENDIF

           IF(X.EQ.1.OR.X.EQ.NREPEAT) THEN
              QDOLOG=.TRUE.
           ELSE IF(LOGLEVEL.EQ.0) THEN
              QDOLOG=.FALSE.
           ELSE IF(MOD(X,LOGLEVEL).EQ.0) THEN
              QDOLOG=.TRUE.
           ELSE
              QDOLOG=.FALSE.
           ENDIF

           IF(QDOLOG) WRITE(IUNREX,'(A,I15,A,I12,A,I5)') '# Exchange ', IREX, ': STEP ', ISTEP-1, ': REPEAT ', X

           ! put the reservoirs in order of temperature ... a bit of a hacky
           ! bubble sort for now.
           LASTLOW=9999.0
           DO I=1,NREPDSTR
              LOWTEMP=9999.0
              DO J=1,NREPDSTR
                 IF(I.GT.1) THEN
                    IF(CURTEMPS(J).GT.LASTLOW) THEN
                       IF(CURTEMPS(J).LT.LOWTEMP) THEN
                          REPORDER(I)=J
                          LOWTEMP=CURTEMPS(J)
                       ENDIF
                    ENDIF
                 ELSE
                    IF(CURTEMPS(J).LT.LOWTEMP) THEN
                       REPORDER(I)=J
                       LOWTEMP=CURTEMPS(J)
                    ENDIF
                 ENDIF
              ENDDO
              LASTLOW=LOWTEMP
           ENDDO

           QMASTER=(MOD(IREX,2).EQ.0)
           DO I=1,NREPDSTR
              QEXC=.FALSE.
              OURSELVES=REPORDER(I)
              OURTEMP=CURTEMPS(OURSELVES)
              
              ! find our neighbor (next highest temperature replica)
              ! NOTE: in the 2D case, finding our neighbor is going to
              ! be trickier. We have to figure out how to do that.
              IF(QMASTER) THEN
                 IF(I.LE.NREPDSTR-1) THEN
                    NEIGHBOR=REPORDER(I+1)
                    NBRTEMP=CURTEMPS(NEIGHBOR)
                 ELSE 
                    NEIGHBOR=-ONE
                    NBRTEMP=9999.0
                 ENDIF
              ELSE
                 ! Special case to make sure that there's a print out for the first
                 ! replica.
                 IF(I.EQ.1) THEN
                    IF(QDOLOG) WRITE(IUNREX,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L1,x,I2)') &
                               OURSELVES,OURTEMP,EPOTARR(OURSELVES),-1,9999.0,0.00,0.00,0.00,.FALSE.,OURSELVES
                 ENDIF
                 QMASTER=.NOT.QMASTER
                 CYCLE
              ENDIF

              IF(NEIGHBOR.GT.0) THEN
                 ! We control the exchange; our neighbor has the next
                 ! highest temperature.

                 ! Here is where we want something like a mix-in (using function pointers)
                 ! to implement various exchange criteria (temperature, hamiltonian, pH)...

                 IF(OURTEMP.EQ.NBRTEMP) THEN
                    PROB=ONE
                 ELSE
#if KEY_OPENMM==1
                    if(qtor_repex) then
                       prob = min(one, exp(-(one/(kboltz*temprx(1)))       &
                            * ( (nbrtemp/ourtemp-one)*epotarr(ourselves)   & 
                            +   (ourtemp/nbrtemp-one)*epotarr(neighbor) )  &
                            ))
                       p = -(one/(kboltz*temprx(1)))       &
                            * ( (nbrtemp/ourtemp-one)*epotarr(ourselves)   & 
                            +   (ourtemp/nbrtemp-one)*epotarr(neighbor) )  

                    else
#endif
                       PROB=MIN(ONE,EXP(-(ONE/(KBOLTZ*OURTEMP) &
                            -ONE/(KBOLTZ*NBRTEMP))*(EPOTARR(NEIGHBOR)-EPOTARR(OURSELVES))))
#if KEY_OPENMM==1
                    endif
#endif
                 ENDIF
                 P=RANDOM(REPSEED)
                
                 IF(P.LE.PROB) THEN
                    QEXC=.TRUE.
                    CURTEMPS(NEIGHBOR)=OURTEMP
                    CURTEMPS(OURSELVES)=NBRTEMP
                    REP1=NEIGHBOR
                    REP2=OURSELVES
                 ELSE
                    QEXC=.FALSE.
                    REP1=OURSELVES
                    REP2=NEIGHBOR
                 ENDIF
                 PROBARR(OURSELVES)=(((X-1)*PROBARR(OURSELVES))+MIN(PROB,ONE))/FLOAT(X)
                 PROBARR(NEIGHBOR)=(((X-1)*PROBARR(NEIGHBOR))+MIN(PROB,ONE))/FLOAT(X)
                 !!!WRITE(IUNREX,'(A,2I3)') 'UPDATING PROBARR OF REPLICAS ',OURSELVES,NEIGHBOR
                 IF(QDOLOG) WRITE(IUNREX,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L1,x,I2)') &
                            OURSELVES,OURTEMP,EPOTARR(OURSELVES),NEIGHBOR,NBRTEMP,EPOTARR(NEIGHBOR),PROB,P,QEXC,REP1

                 ! write out a line for the neighboring replica, as well
                 IF(QDOLOG) WRITE(IUNREX,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L1,x,I2)') &
                            NEIGHBOR,NBRTEMP,EPOTARR(NEIGHBOR),OURSELVES,OURTEMP,EPOTARR(OURSELVES),PROB,P,QEXC,REP2

              ELSE
                 ! Give this a 0 exchange probability for records-keeping purposes
                 PROBARR(OURSELVES)=((X-1)*PROBARR(OURSELVES))/FLOAT(X)
                 !!!WRITE(IUNREX,'(A,I2,A,F6.4)') 'PROBDEBUG> NO EXCHANGE FOR ',OURSELVES,' PROBARR = ',PROBARR(OURSELVES)

                 IF(QDOLOG) WRITE(IUNREX,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L1,x,I2)') &
                            OURSELVES,OURTEMP,EPOTARR(OURSELVES),-1,9999.0,0.00,0.00,0.00,.FALSE.,OURSELVES

              ENDIF ! END IF(OURSELVES.GT.0)
              QMASTER=.NOT.QMASTER
           ENDDO
          ! ----
          ! Copy exchange probability data to a point where it can be blasted out
          ! one per node. Mike C's COMM_MASTER will remove the need for hijinks
          ! like this.
          ! ----
          WRITE(IUNREX,'(A)',ADVANCE='NO') 'PROBARR> '
          DO I=1,nrepdstr
             WRITE(IUNREX,'(A,I1,A,F6.4,X)',ADVANCE='NO') 'PROBARR(', I, ') =', PROBARR(I)
             SCARR(I)=PROBARR(I) ! TODO is SCARR even necessary anymore?
          ENDDO
          WRITE(IUNREX,'(A)') ' '

        ENDIF ! END part only executed on processor 0

     ENDDO

     ! Now that we have figured out the final temperatures at each state,
     ! broadcast curtemps and call dofastexchg to actually make the exchange.
     ! FIXME should this be using comm_universe instead? They should be 
     !       equivalent but still...
     CALL MPI_BCAST(CURTEMPS,NREPDSTR,MPI_REAL8,0,COMM_UNIVERSE,IERR)
     CALL MPI_BCAST(IREX,1,MPI_INTEGER,0,COMM_UNIVERSE,IERR)
     CALL MPI_BCAST(NREPEAT,1,MPI_INTEGER,0,COMM_UNIVERSE,IERR)
     CALL MPI_SCATTER(SCARR,1,MPI_REAL8,REXP,1,MPI_REAL8,0,COMM_UNIVERSE,IERR)

     CALL CHMDEALLOC('repdstr.src','FASTREPEXCHG','scarr',NUMNODG,crl=SCARR)
     ! DRR - Disabling this call to PSETLOC since PSETGLOB is disabled
     !       above - see notes above for details.
     !CALL PSETLOC()
     CALL DOFASTEXCHG(CURTEMPS(IREPDSTR+1),JHSTRT,IGVOPT,VX,VY,VZ,XOLD,YOLD,ZOLD,REXP &
#if KEY_TSM==1
                      ,BACKLS &
#endif
                     )
     CALL FLUSH(OUTU)
  END SUBROUTINE FASTREPEXCHG

!*******************************************************************************
!> Actually perform 'FAST' version of replica exchange calc. 
!<
  SUBROUTINE DOFASTEXCHG(TEMPNEW,JHSTRT,IGVOPT,VX,VY,VZ,XOLD,YOLD,ZOLD,REXP &
#if KEY_TSM==1
                         ,BACKLS &
#endif
                        )
     use mpi
     use psf
     use parallel
     use stream
     use repdstr
     use coord,only: x,y,z
     use param_store, only: set_param
!     use omm_main, only : omm_change_lambda
  
     ! Arguments
     REAL(CHM_REAL),INTENT(IN)    :: TEMPNEW,REXP
     REAL(CHM_REAL),INTENT(INOUT) :: VX(*),VY(*),VZ(*),XOLD(*),YOLD(*),ZOLD(*)
     INTEGER,INTENT(OUT)          :: JHSTRT
     ! stupidva
     INTEGER,INTENT(INOUT)        :: IGVOPT
#if KEY_TSM==1
     INTEGER BACKLS(*) 
#endif    

     ! Local variables
     LOGICAL        :: QUPVELOC
     REAL(CHM_REAL) :: SCALEF
     INTEGER        :: I,IERR

     IF(MYNOD.EQ.0) THEN
        REPROB=(((IREX-NREPEAT)*REPROB)+(NREPEAT*REXP))/FLOAT(IREX)
        call set_param('REPROB',REPROB)

        ! In FAST replica exchange, we don't track things explicitly
        ! because all decisions are made on the main processor and it
        ! would be a bit costly to send them to each replica; so up and
        ! down are rather arbitrary in this context...
        IF(MOD(IREX,2)==0) THEN
           NOPPDN=NOPPDN+1
        ELSE
           NOPPUP=NOPPUP+1
        ENDIF
        IF(PRNLEV.GT.5) THEN
           WRITE(IUNREX,'(A,2I5)')  'FREX PROB> NOPPUP,NOPPDN = ',NOPPUP,NOPPDN
           WRITE(IUNREX,'(A,F8.6)') 'FREX PROB> REPROB = ',REPROB
        ENDIF

        IF(PRNLEV.GT.3) &
           WRITE(OUTU,'(A,2F10.3)') 'FREX DEBUG> FORMER AND NEW TEMPS: ', TEMPCURRENT, TEMPNEW
        IF(TEMPCURRENT.NE.TEMPNEW) THEN

           IF(TEMPCURRENT.GT.TEMPNEW) THEN
              NSUCUP=NSUCUP+1
              call set_param('EXRUP', REAL(NSUCUP, chm_real) / REAL(NOPPUP, chm_real))
           ELSE
              NSUCDN=NSUCDN+1
              call set_param('EXRDN', REAL(NSUCDN, chm_real) / REAL(NOPPDN, chm_real))
           ENDIF
           IF(PRNLEV.GT.5) THEN
              WRITE(IUNREX,'(A,2I5)')   'FREX PROB> NSUCUP,NSUCDN = ',NSUCUP,NSUCDN
              WRITE(IUNREX,'(A,2F9.6)') 'FREX PROB> EXRUP,EXRDN = ', &
                  REAL(NSUCUP, chm_real) / REAL(NOPPUP, chm_real), &
                  REAL(NSUCDN, chm_real) / REAL(NOPPDN, chm_real)
           ENDIF

#if KEY_OPENMM==1
           if(.not.qtor_repex) then
#endif
              QUPVELOC=.TRUE.
              SCALEF=SQRT(TEMPNEW/TEMPCURRENT)
              IF(PRNLEV.GE.6) WRITE(OUTU,'(A,F7.4)') 'FREX DEBUG> SCALEF: ', SCALEF
              DO I=1,NATOM
                 VX(I)=VX(I)*SCALEF
                 VY(I)=VY(I)*SCALEF
                 VZ(I)=VZ(I)*SCALEF
                 XOLD(I)=XOLD(I)*SCALEF
                 YOLD(I)=YOLD(I)*SCALEF
                 ZOLD(I)=ZOLD(I)*SCALEF
              ENDDO
              TEMPCURRENT=TEMPNEW
#if KEY_OPENMM==1
           else
              tempcurrent = tempnew
              qupveloc = .false.
           endif
#endif
        ELSE
           QUPVELOC=.FALSE.
        ENDIF
     ENDIF
 
     call mpi_bcast(qupveloc, 1, MPI_LOGICAL, 0, comm_charmm, ierr)
     call mpi_bcast(tempcurrent, 1, MPI_REAL8, 0, comm_charmm, ierr)

#if KEY_OPENMM==1
     if(qtor_repex) then
        torsion_lambda = tempcurrent
!        call omm_change_lambda(tempcurrent)
     endif
#endif
     IF(QUPVELOC) THEN
        JHSTRT=0
        IGVOPT=2
        
        ! BTM: I am not sure if we still need to send xold, yold, and zold,
        ! but I am going to do it anyway to be safe.
        if (prnlev.gt.2) WRITE(OUTU,'(A)') 'DOFASTEXCHG> UPDATING MY VELOC'
        call mpi_bcast(igvopt, 1, MPI_INTEGER, 0, comm_charmm, ierr)
        call mpi_bcast(jhstrt, 1, MPI_INTEGER, 0, comm_charmm, ierr)
        call mpi_bcast(vx, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(vy, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(vz, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(xold, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(yold, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(zold, natom, MPI_REAL8, 0, comm_charmm, ierr)
     ENDIF               
  END SUBROUTINE DOFASTEXCHG

!*******************************************************************************
!> Swap coordinates and velocities between here and RANK. 
!<
  ! Eventually, I want swap_crd to handle all times when we swap coordinates
  ! back and forth...
  !
  ! BTM 2014-12-16 -- we might need to deal with XOLD, YOLD, and ZOLD here to
  ! be complete.
  SUBROUTINE SWAP_CRD(RANK,VX,VY,VZ)
     use psf
     use mpi
     use stream
     use parallel
     use memory
     use bases_fcm
     use deriv
     use image
     use energym, only: energy,eprop,eterm,lenent,epot
     use imgup,only: upimag,upimag0
     use coord
     use comand
     use number
#if KEY_DOMDEC==1
     use domdec,only:init_domdec1,init_domdec2
     use domdec_common,only:q_domdec,q_split
     use domdec_dr_common,only:q_direct_node
     use domdec_d2d_comm,only:transfer_coord,transfer_force,copy_to_all
     use domdec_local,only:update_local_coord
#endif

     ! Passed in variables
     integer, intent(in)            :: rank
     real(chm_real), intent(inout)  :: vx(*),vy(*),vz(*)

     ! Local variables
     real(chm_real),allocatable,dimension(:)     :: w
     real(chm_real)                              :: oldxtlabc(6),oldxucell(6),totvx,totvy,totvz
     logical                                     :: qcrys
     integer                                     :: ierr,status(mpi_status_size),mr,sz,i
     real(chm_real),allocatable,dimension(:,:,:) :: transf

     qcrys = (xtltyp.ne.'    ')
     call chmalloc('repdstr.src','swap_crd','w',natom,crl=w)
     oldxtlabc(1:6)=xtlabc(1:6)
     oldxucell(1:6)=xucell(1:6)

#if KEY_DEBUGREPD==1
     totvx = zero
     totvy = zero
     totvz = zero
     do i=1,natom
        totvx = totvx + abs(vx(i))
        totvy = totvy + abs(vy(i))
        totvz = totvz + abs(vz(i))
        !!write(outu,'(a,i5,a,4f12.4)') 'SWAP DBG> VX,VY,VZ(',i,') and TOTVY = ',vx(i),vy(i),vz(i),totvy
     enddo
     write(outu,'(a,3f20.6)') 'SWAP DBG> PRESWAP TOT ABS VX,VY,VZ = ',totvx,totvy,totvz
     write(outu,'(a,3f20.6)') 'SWAP DBG> PRESWAP AVG ABS VX,VY,VZ = ',totvx/natom,totvy/natom,totvz/natom

     call mpi_comm_rank(comm_charmm,mr,ierr)
     call mpi_comm_size(comm_charmm,sz,ierr)
     write(outu,'(a,2i5)') 'SWAP_CRD DBG> RANK AND SIZE = ',mr,sz
     call flush(outu)

#if KEY_DOMDEC==1
     if(q_domdec) call transfer_coord(x,y,z,.true.)
#endif
     !!call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
     CALL PRINTE(OUTU, EPROP, ETERM, 'SDBG', 'ENR', .TRUE., 0, 0, 0, .TRUE.)
#endif

     if(mynod == 0) then

        if(rank > mynodg) then
           call mpi_recv(w,natom,mpi_real8,rank,3,comm_universe,status,ierr)
           call mpi_send(x,natom,mpi_real8,rank,7,comm_universe,ierr)
           x(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,4,comm_universe,status,ierr)
           call mpi_send(y,natom,mpi_real8,rank,8,comm_universe,ierr)
           y(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,5,comm_universe,status,ierr)
           call mpi_send(z,natom,mpi_real8,rank,9,comm_universe,ierr)
           z(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,5,comm_universe,status,ierr)
           call mpi_send(vx,natom,mpi_real8,rank,9,comm_universe,ierr)
           vx(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,5,comm_universe,status,ierr)
           call mpi_send(vy,natom,mpi_real8,rank,9,comm_universe,ierr)
           vy(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,5,comm_universe,status,ierr)
           call mpi_send(vz,natom,mpi_real8,rank,9,comm_universe,ierr)
           vz(1:natom)=w(1:natom)

           call mpi_recv(w,natom,mpi_real8,rank,6,comm_universe,status,ierr)
           call mpi_send(wmain,natom,mpi_real8,rank,10,comm_universe,ierr)
           wmain(1:natom)=w(1:natom)

           if(qcrys.and.xdim.gt.0) then
              call mpi_recv(w,6,mpi_real8,rank,12,comm_universe,status,ierr)
              call mpi_send(xtlabc,6,mpi_real8,rank,11,comm_universe,ierr)
              xtlabc(1:6)=w(1:6)
           endif
        else
           call mpi_send(x,natom,mpi_real8,rank,3,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,7,comm_universe,status,ierr)
           x(1:natom)=w(1:natom)

           call mpi_send(y,natom,mpi_real8,rank,4,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,8,comm_universe,status,ierr)
           y(1:natom)=w(1:natom)

           call mpi_send(z,natom,mpi_real8,rank,5,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,9,comm_universe,status,ierr)
           z(1:natom)=w(1:natom)

           call mpi_send(vx,natom,mpi_real8,rank,5,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,9,comm_universe,status,ierr)
           vx(1:natom)=w(1:natom)

           call mpi_send(vy,natom,mpi_real8,rank,5,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,9,comm_universe,status,ierr)
           vy(1:natom)=w(1:natom)

           call mpi_send(vz,natom,mpi_real8,rank,5,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,9,comm_universe,status,ierr)
           vz(1:natom)=w(1:natom)

           call mpi_send(wmain,natom,mpi_real8,rank,6,comm_universe,ierr)
           call mpi_recv(w,natom,mpi_real8,rank,10,comm_universe,status,ierr)
           wmain(1:natom)=w(1:natom)

           if(qcrys.and.xdim.gt.0) then
              call mpi_send(xtlabc,6,mpi_real8,rank,12,comm_universe,ierr)
              call mpi_recv(w,6,mpi_real8,rank,11,comm_universe,status,ierr)
              xtlabc(1:6)=w(1:6)
           endif
        endif

     endif

     ! The following code is executed on all processors

     ! distribute data within the replica group
     ! it 
     call mpi_bcast(x,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(y,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(z,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(vx,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(vy,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(vz,natom,mpi_real8,0,comm_charmm,ierr)
     call mpi_bcast(wmain,natom,mpi_real8,0,comm_charmm,ierr)
     !call mpi_bcast(xtlabc,6,mpi_real8,0,comm_charmm,ierr)

     if(qcrys) then
        call mpi_bcast(xtlabc,6,mpi_real8,0,comm_charmm,ierr)
        call xtllat(xucell,xtlabc)
        call mpi_bcast(xucell,6,mpi_real8,0,comm_charmm,ierr)

        call chmalloc('repdstr.src','REPXCHG','TRANSF',3,4,XNSYMM,crl=transf)
        call imfill(transf,.false.)
        call chmdealloc('repdstr.src','REPXCHGL','TRANSF',3,4,XNSYMM,crl=transf)

        call upimag0(x,y,z,wmain,0)
     endif

     !!eprop(:)=zero
     !!eterm(:)=zero
     !!dx(:)=zero
     !!dy(:)=zero
     !!dz(:)=zero

#if KEY_DOMDEC==1
     if(q_domdec) then
        call copy_to_all(x,y,z)
        call copy_to_all(vx,vy,vz)
     endif
#endif

     ! Note: this routine calls init_domdec1 and init_domdec2 for us...
     call update(comlyn,comlen,x,y,z,wmain,.false.,.false.,.true., &
          .false.,.true.,0,0,0,0,vx,vy,vz)

     !! Let's let the caller handle if they want energy called; no
     !! sense doing extra work.

#if KEY_DEBUGREPD==1
     call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
     CALL PRINTE(OUTU, EPROP, ETERM, 'SWAP', 'ENR', .TRUE., 0, 0, 0, .TRUE.)
     call flush(outu)
#endif

     call chmdealloc('repdstr.src','swap_crd','w',natom,crl=w)

  END SUBROUTINE SWAP_CRD

! ==============================================================================
#if KEY_DEBUGREPD==1
! DEBUG - Use this to test if communicator has become invalidated
subroutine check_bcast
  use parallel
  use mpi
  use repdstr
  implicit none
  integer        :: testrank, itest, ierr
  real(chm_real) :: RTEST, rdiff
  logical        :: LTEST 

  call mpi_comm_rank(comm_charmm, testrank, ierr)

  if (testrank == 0) then
    itest = 42
    LTEST = .true.
    RTEST = 3.14159d0
  else
    itest = 0
    LTEST = .false.
    RTEST = 0.d0
  endif
  write(100+mynodg,'(a,i6,a,l3,a,f10.4)') 'DBG: Before broadcast, itest is ', itest, ', ltest is ', ltest, ', rtest is ', rtest
  call mpi_bcast(itest, 1, MPI_INTEGER, 0, comm_charmm, ierr)
  call mpi_bcast(ltest, 1, MPI_LOGICAL, 0, comm_charmm, ierr)
  call mpi_bcast(rtest, 1, MPI_REAL8,   0, comm_charmm, ierr)
  write(100+mynodg,'(a,i6,a,l3,a,f10.4)') 'DBG: After broadcast, itest is ', itest, ', ltest is ', ltest, ', rtest is ', rtest
  if (itest /= 42)       write(100+mynodg,'(a)') 'PROBLEM WITH INTEGER BROADCAST'
  if (ltest.eqv..false.) write(100+mynodg,'(a)') 'PROBLEM WITH LOGICAL BROADCAST'
  rdiff = abs(rtest - 3.14159d0)
  if (rdiff > 0.0000001) write(100+mynodg,'(a)') 'PROBLEM WITH REAL BROADCAST'
  call mpi_comm_compare(comm_charmm, comm_rpg, itest, ierr)
  if (itest == MPI_IDENT) then
    write(100+mynodg,'(a)') 'DBG: comm_charmm and comm_rpg are identical'
  else
    write(100+mynodg,'(a)') 'DBG: comm_charmm and comm_rpg differ'
  endif
  call mpi_barrier(comm_universe, ierr)
end subroutine check_bcast
#endif /* KEY_DEBUGREPD */

!*******************************************************************************
!> Set up communicators for replica exchange. Divide available number of threads
!! among specified number of replicas. COMM_RPG (COMM_CHARMM) will be the
!! communicator between threads within the same replica, and COMM_REP_MASTER
!! will be the communicator between the master threads of each replica (i.e.
!! COMM_CHARMM rank == 0). Should only need to be called once per REPD
!! invocation.
!! @param numreps Number of replicas to set up.
!<
subroutine setup_repd_comms(numreps)
  ! USE STATEMENTS
  use mpi
  use repdstr
  use stream, only: prnlev, outu
  use parallel, only: numnod, numnodg, mynodg, mynod
  ! ARGUMENTS
  implicit none
  integer, intent(in) :: numreps
  ! LOCAL VARIABLES
  integer mycolor, ident, ierr, istat(mpi_status_size)
# if KEY_DEBUGREPD==1
  integer testrank, testsize ! DEBUG
# endif
  ! Sanity check: Make sure comm_charmm is not already comm_rpg
!  call mpi_comm_compare(comm_charmm, comm_rpg, ident, ierr)
!  if (ident == MPI_IDENT) &
!    call wrndie(-5,'<REPDSTR>','Internal Error: COMM_CHARMM is already COMM_RPG')
  ! NOTE: setting nrepdstr here can be redundant but is harmless
  nrepdstr = numreps
  numnod = numnodg / nrepdstr
  if (prnlev.gt.2) &
    write(outu, '(2(a,i6),a)') 'REPD> Dividing ', numnodg, ' threads among ',&
                      nrepdstr, ' replicas.'
  if (numnod * nrepdstr /= numnodg) &
    call wrndie(-5,'<REPDSTR>','Wrong combination of NREP and number of threads.')
# if KEY_CMPI==1
  ! CMPI does not work with > 1 thread per replica.
  if (numnod .gt. 1) &
    call wrndie(-5,'<REPDSTR>', 'Cannot use > 1 thread per replica with CMPI.')
# endif
  irepdstr = mynodg / numnod
  repdid = irepdstr
  reptag = irepdstr
  repseed = irepdstr + 123

  ! Free comm_rpg if not null
  if (comm_rpg /= MPI_COMM_NULL) then
    call mpi_comm_free(comm_rpg, ierr)
  endif
  ! Split comm_universe into nrepdstr comms, comm_rpg == comm_charmm
  comm_universe = comm_charmm
  call mpi_comm_split(comm_charmm, irepdstr, mynodg, comm_rpg, ierr)
  if (ierr /= mpi_success) &
    call wrndie(-4,'<REPDSTR>','MPI_COMM_SPLIT for COMM_RPG failed.')
  comm_charmm = comm_rpg
# if KEY_DEBUGREPD==1
  write(100+mynodg,'(2(a,i10))') 'DBG: comm_charmm= ', comm_charmm, ' comm_rpg= ', comm_rpg
  flush(100+mynodg)
# endif

  ! FIXME numnod and mynod should really be set by mpi_comm_size etc
  call psetloc
  qrepmaster = (mynod .eq. 0)
  ! TODO call drepsetio here?

  ! Create communicator containing only master from each comm_charmm
  if (qrepmaster) then
    mycolor = 0
  else
    mycolor = mpi_undefined
  endif
  call mpi_comm_split(comm_universe, mycolor, irepdstr, comm_rep_master, ierr)
  if (ierr /= mpi_success) &
    call wrndie(-4,'<REPDSTR>','MPI_COMM_SPLIT for COMM_REP_MASTER failed.')
# if KEY_DEBUGREPD==1
  ! DEBUG
  if (qrepmaster) then
    call mpi_comm_size(comm_rep_master, testsize, ierr)
    call mpi_comm_rank(comm_rep_master, testrank, ierr)
    write(100+mynodg,'(a,2I6)') 'DBG: My comm_rep_master rank and size: ', testrank, testsize
  endif
# endif

end subroutine setup_repd_comms

!*******************************************************************************
!> \return My SGLD temperature.
!> Implemented as a separate function so it can be called externally.
!<
function my_sg_temperature()
  ! ARGUMENTS
  implicit none
  real(chm_real) :: my_sg_temperature
  ! LOCAL VARIABLES
  integer :: mysgidx ! My index in the SGLD dimension

  mysgidx = mydims(sgdim_idx)
  my_sg_temperature = sgtemprx(mysgidx+1)
end function my_sg_temperature

!*******************************************************************************
!> \return My SGLD FT.
!> Implemented as a separate function so it can be called externally.
!<
function my_sg_ft()
  ! ARGUMENTS
  implicit none
  real(chm_real) :: my_sg_ft
  ! LOCAL VARIABLES
  integer :: mysgidx ! My index in the SGLD dimension

  mysgidx = mydims(sgdim_idx)
  my_sg_ft = sgftrx(mysgidx+1)
end function my_sg_ft

!*******************************************************************************
!> Set my SGLD temperature.
!> This lets DCNTRL set the SGLD temperature.
!<
subroutine set_my_sg_temperature(tempin)
  ! ARGUMENTS
  implicit none
  real(chm_real) :: tempin ! Input SGLD temperature
  ! LOCAL VARIABLES
  integer :: mysgidx ! My index in the SGLD dimension

  mysgidx = mydims(sgdim_idx)
  sgtemprx(mysgidx+1) = tempin
end subroutine set_my_sg_temperature

!*******************************************************************************
!> Set my SGLD FT.
!> This lets DCNTRL set the SGLD FT.
!<
subroutine set_my_sg_ft(ftin)
  ! ARGUMENTS
  implicit none
  real(chm_real) :: ftin ! Input SGLD FT 
  ! LOCAL VARIABLES
  integer :: mysgidx ! My index in the SGLD dimension

  mysgidx = mydims(sgdim_idx)
  sgftrx(mysgidx+1) = ftin
end subroutine set_my_sg_ft

!*******************************************************************************
!> Set my SGLD temperature and neighbors SGLD temperature.
!> NOTE: Only needed for replica log output.
!<
subroutine set_sg_temperatures(neighbor, mytemp, nbrtemp)
  use mpi
  ! ARGUMENTS
  implicit none
  integer, intent(in)         :: neighbor
  real(chm_real), intent(out) :: mytemp
  real(chm_real), intent(out) :: nbrtemp
  ! LOCAL VARIABLES
  integer ierr, istat(mpi_status_size)

  ! Get my SG temperature
  mytemp = my_sg_temperature()
  nbrtemp = 0.d0
  if (neighbor .gt. -1 .and. qrepmaster) then
    if (ndims .gt. 1) then
      ! Send my SG temperature, get neighbors temperature
      call mpi_sendrecv(mytemp,  1, mpi_real8, neighbor, 51, &
                        nbrtemp, 1, mpi_real8, neighbor, 51, &
                        comm_rep_master, istat, ierr)
    else
      ! In one dimension just get neighbors temperature from array.
      nbrtemp = sgtemprx(neighbor+1)
    endif
  endif
end subroutine set_sg_temperatures

!*******************************************************************************
!> Print Results. Only masters will perform actual write.
!> NOTE: The reason this is not implemented in each exchange function is to
!!       preserve the order of the previous output as closely as possible.
!<
subroutine write_log(qexc, istep, neighbor, exprob, ttemp, mytemp, emine, &
                     nbrtemp, eneigh, scaled, rnum, &
                     nbrpoti, ourpotj, hrexarg, &      ! HREMD
                     ph_l, ph_m, iproto, jproto, &     ! pH REMD
                     scalesg, &                        ! SGLD
                     res_struct &                      ! Reservoir REMD
                    )
  ! USE STATEMENTS
  use mpi
  use repdstr,     only: iunrex, irepdstr, nrepdstr, ewritu
  use stream,      only: prnlev
  ! ARGUMENTS
  implicit none
  logical, intent(in)           :: qexc
  integer, intent(in)           :: istep ! istart - 1 
  integer, intent(in)           :: neighbor
  real(chm_real), intent(in)    :: exprob, ttemp, mytemp, emine, nbrtemp, eneigh
  real(chm_real), intent(in)    :: scaled, rnum
  ! hremd only
  real(chm_real), intent(in)    :: nbrpoti, ourpotj, hrexarg
  ! ph remd only
  real(chm_real), intent(in)    :: ph_l, ph_m
  integer, intent(in)           :: iproto, jproto
  ! SGLD only
  real(chm_real), intent(in)    :: scalesg
  ! R-REMD
  integer, intent(in), optional :: res_struct
  ! LOCAL VARIABLES
  real(chm_real)               :: srate, mysgtemp, nbrsgtemp
  real(chm_real)               :: exch_prob_up, exch_prob_dn
  real(chm_real), dimension(6) :: comar
  integer NUP, NDN, NAT, NTOT
  integer i, j
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  NUP = nopp_up(currdim)
  NDN = nopp_down(currdim)
  NAT = nattempts(currdim)
  NTOT = NUP + NDN
!  if (qrepmaster) write(iunrex,'(4(a,i6))') 'DBG: NUP=', NUP, ' NDN=', NDN, ' NAT=', NAT, ' NTOT=', NTOT
  if (qexc) then
    nsuccess(currdim) = nsuccess(currdim) + 1
    if (exchange_dir.eq.1) then
      nsuc_up(currdim)   = nsuc_up(currdim)   + 1
    else if (exchange_dir.eq.-1) then
      nsuc_down(currdim) = nsuc_down(currdim) + 1
    endif
  endif
  ! Calculate exchange success rates
  ! TODO - can param set be generalized to N dims?
  if (NUP > 0) then
    exch_prob_up = dble(nsuc_up(currdim)) / dble(NUP)
    call set_remd_param('EXRUP', currdim, dval=exch_prob_up)
  endif
  if (NDN > 0) then
    exch_prob_dn = dble(nsuc_down(currdim)) / dble(NDN)
    call set_remd_param('EXRDN', currdim, dval=exch_prob_dn)
  endif
  srate = real(nsuccess(currdim)) / real(NAT)
  if (q_exch_attempted.and.NTOT.gt.0) then ! TODO need to check NTOT?
    rex_prob(currdim) = (min(EXPROB,1.d0)+(dble(NTOT-1)*rex_prob(currdim))) / dble(NTOT)
    call set_remd_param('REPROB', currdim, dval=rex_prob(currdim))
    if (prnlev > 5 .and. qrepmaster) then
      write(iunrex,'(a,4i3)'  ) 'REPEXCH> noppup, noppdn, nsucup, nsucdn = ',NUP,NDN,nsuc_up(currdim),nsuc_down(currdim)
      write(iunrex,'(a,f10.5)') 'REPEXCH> P = ',min(EXPROB,1.d0)
      write(iunrex,'(a,f10.5)') 'REPEXCH> NUMERATOR = ',(min(EXPROB,1.d0)+(dble(NTOT-1)*rex_prob(currdim)))
      write(iunrex,'(a,f10.5)') 'REPEXCH> DENOMINATOR = ',dble(NTOT)
      write(iunrex,'(a,3f6.3)') 'REPEXCH> EXRUP EXRDN REPROB = ',exch_prob_up,exch_prob_dn,rex_prob(currdim)
    endif
  endif
  if (qrepmaster) then
    if (ndims .gt. 1) then
      ! N dimension exchange
      write(iunrex,'(a)') '==================================================================================='
      select case (exchange_type(currdim))
        case (ex_temperature)
          write(iunrex,'(a,i10,a,f7.2,a,i3,a,f7.2,a,f8.2,a,f8.2,a,f5.3,a,f5.3,a,l1)') &
                       'T-REX>',istep,' TEMP ',mytemp,' NBR ',neighbor,' NBRTMP ', &
                       nbrtemp, ' OURENE ',emine,' NBRENE ',eneigh,' P ',exprob, &
                       ' RN ', rnum,' SUC? ',qexc
        case (ex_hamiltonian)
          write(iunrex,'(a,i10,2(a,i3,2x))') & 
                       'H-REX>', istep, ' NBR ', neighbor , 'EXCHDIM', currdim
          write(iunrex,'(a,10x,2(1x,a,1x,f14.5))') &
                       'H-REX>', 'EPOT1(P)', emine, 'EPOT1(Q)', ourpotj
          write(iunrex,'(a,10x,2(1x,a,1x,f14.5))') &
                       'H-REX>', 'EPOT2(P)', nbrpoti, 'EPOT2(Q)', eneigh
          write(iunrex,'(a,10x,2(1x,a,1x,f7.3))') &
                       'H-REX>', 'MYTEMP', mytemp, 'NBRTEMP', nbrtemp
          write(iunrex,'(a,10x,2(a,1x,f5.3,1x),1x,a,l1,1x,a,1x,f6.3)') &
                       'H-REX>', 'P', exprob, ' RN ',rnum , ' SUC? ', qexc, 'ACC_RATIO', srate 
          write(iunrex,'(a,10x,2(a,1x,i5,1x))') &
                     'H-REX>', 'ORTAG', rep_oldtag, 'NWTAG', reptag
        case (ex_phrex)
          write(iunrex,'(a,i10,a,f5.2,a,i3,a,f5.2,a,f5.3,a,f5.3,a,l1)') &
                       'PHREX>',istep,' PH ',ph_l,' NBR ',neighbor,' NBRPH ', &
                       ph_m,' P ',exprob,' RN ',rnum,' SUC? ',qexc
        case (ex_sgld)
          call set_sg_temperatures(neighbor, mysgtemp, nbrsgtemp)
          write(iunrex,'(a,i10,2(a,i3,2x))') &
                       'SGREX>', istep,' NBR ', neighbor, 'EXCHDIM', currdim
          write(iunrex,'(a,11x,4(a,2x,f7.2,2x))') &
                       'SGREX>', 'OURTMP', mytemp, 'NBRTMP', nbrtemp, &
                       'OURSGT', mysgtemp, 'NBRSGT', nbrsgtemp
          write(iunrex,'(a,11x,2(a,f14.5,2x))') &
                       'SGREX>', 'OURENE', emine,' NBRENE', eneigh
          write(iunrex,'(a,11x,2(a,2x,f5.3,2x),a,2x,l1,2x,a,2x,f6.3)') &
                       'SGREX>', 'P', exprob, 'RN', rnum, 'SUC?', qexc, 'ACC_RATIO', srate
          write(iunrex,'(a,11x,2(a,2x,i5,1x))') &
                       'SGREX>', 'ORTAG', rep_oldtag, 'NWTAG', reptag
      end select
    else !if (prnlev.ge.2) then
      ! DRR do this no matter what the print level. Do not want to lose this
      ! information but more importantly if prnlev is above/below 2 on
      ! different qrepmaster nodes this will cause qsump to hang forever.
      ! 1 dimension only
      if (qphrex) then
        write(iunrex,'(a)') '------------- pH Replica Exchange ------------'
        write(iunrex,'(a,i10,a,i10)') 'REX>EXCHANGE = ', NAT, '  Step =', istep
        write(iunrex,'(a,i5,a,f7.3,a,i5)') 'REX>REPL     = ',irepdstr, &
                                           '  pH = ', ph_l, &
                                           ' nproto = ', iproto
        write(iunrex,'(a,i5,a,f7.3,a,i5)') 'REX>NEIGHBOR = ',neighbor, &
                                           '  pH = ', ph_m,     &
                                           ' nproto = ', jproto
      else
        write(iunrex,'(a)') '------------- Replica Exchange ------------'
        write(iunrex,'(a,i10,a,i10)') 'REX>EXCHANGE = ', NAT, '  Step =', istep
        write(iunrex,'(a,i5,a,f7.3,a,f20.8)') 'REX>REPL     = ',irepdstr,      &
                                              '  Temp = ', mytemp, &
                                              '  Epot = ', emine
        write(iunrex,'(a,i5,a,f7.3,a,f20.8)') 'REX>NEIGHBOR = ',neighbor,      &
                                              '  Temp = ', nbrtemp,            &
                                              '  Epot = ', eneigh
        if(qrxsgld) then
          !write(iunrex,'(a6,i6,1x,f12.2,f12.2,f6.3,f6.3,f6.3,1x,f7.3)') &
          !    'RXSG> ',NAT,emine,eneigh,scaled,scalsg,srate,sgldarg
          write(iunrex,'(a6,i6,1x,i10,i4,i4,i4,f12.2,f12.2,f6.3,f6.3,f6.3,1x,l1)') &
                       'RXSG> ',NAT,istep,irepdstr,neighbor,reptag,  &
                       emine,eneigh,scaled,scalesg,srate,qexc
        endif
        if(qrxtham) then
          write(iunrex,'(a6,4f12.2,x,f11.3)') 'THAM> ',eneigh,nbrpoti,emine,ourpotj,hrexarg
          if (ewritu .ne. -1) &
            write(ewritu,'(i8,4(x,es25.16))') istep, emine, eneigh, ourpotj, nbrpoti
        endif
      endif
      write(iunrex,'(a,i5,a,i5)') 'REX>ORIGINAL TAG ',rep_oldtag,' NEW TAG ',reptag
      if (present(res_struct)) write(iunrex,'(a,i7)') 'REX>RESERVOIR STRUCT = ', res_struct
      write(iunrex,'(a,f8.5,a,f8.5,a,f7.4,a,l1)') &
            'REX>PROB     = ', exprob, ' Rand = ', rnum,  &
            ' Tscale = ', scaled, ' Success = ', qexc

      ! Printout the summary results if sump flag specified:
      if (qsump) then
        comar(1)=scaled
        comar(2)=srate
        comar(3)=mytemp
        comar(4)=nbrtemp
        comar(5)=ttemp
        comar(6)=emine
        write(iunrex,'(a,a,a)')'#REXSUM Rep# Tscale', &
                               '  Sratio   Temp      NewTemp', &
                               '  CurrTemp       Epot'
        write(iunrex,'(a,i3,2f8.3,3f10.3,f15.4)')'REXSUM> ', &
                     irepdstr,(comar(j),j=1,6)
        ! Lets do some extra communication for a nice rexsum> printout
        if (irepdstr.eq.0)then
          do i = 1, nrepdstr-1
            call mpi_recv(comar,6,mpi_real8,i,5,comm_rep_master,istat,ierr)
            write(iunrex,'(a,i3,2f8.3,3f10.3,f15.4)')'REXSUM> ', &
                         i,(comar(j),j=1,6)
          enddo
        else
          call mpi_send(comar,6,mpi_real8,0,5,comm_rep_master,ierr)
        endif
      endif ! qsump

      if (.not.qrxsgld.and..not.qphrex) then
        write(iunrex,'(a)') '------------- Replica Exchange End --------'
      else if (qphrex) then
        write(iunrex,'(a)') '------------- pH Replica Exchange End --------'
      endif
    endif ! END if prnlev >= 2
  endif ! qrepmaster

end subroutine write_log

!*******************************************************************************
!> Write coords to IUNRXC 
!<
subroutine write_post_swap_crds(istep, x, y, z)
  ! USE STATEMENTS
  use psf,     only: natom
  use stream,  only: prnlev
  ! ARGUMENTS
  implicit none
  integer, intent(in)                      :: istep
  real(chm_real), dimension(:), intent(in) :: x, y, z
  ! LOCAL VARIABLES
  integer i

  if (iunrxc.gt.0.and.prnlev.gt.0.and.qrepmaster) then
    write(iunrxc,'(a,i9,a)') '--- REX STEP ',istep,' POST-SWAP ATTEMPT CRDS ---'
    do i=1,natom
      write(iunrxc,'(i5,x,3f9.3)') i,x(i),y(i),z(i)
    enddo
  endif
end subroutine write_post_swap_crds

!*******************************************************************************
!> Perform actions necessary for telling charmm that coords have changed.
!! Note that the DOMDEC coords update is in repd_energy, not here, since
!! the recip/direct split would not be active at this point. It in fact
!! cannot be active in order to call update().
!<
subroutine coords_update(x,y,z,wmain,vx,vy,vz)
  ! USE STATEMENTS
  use comand, only: comlyn, comlen
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz

  ! NOTE: Update will make certain *everything* related to coords will be
  !       updated (probably) but it is a big stick.
  ! Note: this routine calls init_domdec1 and init_domdec2 for us...
  call update(comlyn, comlen, x, y, z,wmain,.false.,.false.,.true., &
              .false.,.true.,0,0,0,0,vx,vy,vz)
  ! Re-generate the pair list
  !call nbonds(x, y, z, bnbnd, bimag) ! FIXME necessary?
end subroutine coords_update

!*******************************************************************************
!> Wrapper around the energy routine. For domdec split it is assumed that pure
!! recip nodes are currently with us and need to enter energy_recip. The
!! coords_update routine should have been called prior.
!<
subroutine repd_energy(x,y,z)
  ! USE STATEMENTS
  use bases_fcm,only: bnbnd,bimag
  use deriv,    only: dx,dy,dz
  use energym,  only: energy
# if KEY_DOMDEC==1
  use domdec_common,   only: q_domdec, q_split
  use domdec_dr_common,only: start_split_direct_recip, stop_split_direct_recip,&
                             q_direct_node, q_recip_node
  use domdec_d2r_comm, only: send_stop_recip, send_coord_to_recip
  use domdec_d2d_comm, only: transfer_coord
  use domdec_local,    only: update_local_coord
  use energy_util,     only: energy_recip
# endif
#if KEY_BLADE == 1
  use blade_main, only: blade_energy, blade_repd_energy
#endif /* KEY_BLADE */
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z

# if KEY_DOMDEC==1
  if (q_domdec) then
    ! domdec - may need to do split
    call start_split_direct_recip()
    if (.not.q_direct_node .and. q_recip_node) then
      call energy_recip(x,y,z,dx,dy,dz,.false.)
    else
      call transfer_coord(x,y,z,.false.)
      call update_local_coord(x,y,z)
      if (q_split) call send_coord_to_recip(x,y,z,.true.)
      call energy(x, y, z, dx, dy, dz, bnbnd, bimag, 0)
      if (q_split) call send_stop_recip()
    endif
    call stop_split_direct_recip()
  else ! q_domdec
# endif
    ! Not DOMDEC
#if KEY_BLADE==1
  if (q_repd_want_blade) then
    ! call blade_energy(x, y, z)
    call blade_repd_energy(x, y, z) ! Special subroutine to avoid system teardown/setup overhead
  else
#endif
    call energy(x, y, z, dx, dy, dz, bnbnd, bimag, 0)
#if KEY_BLADE==1
  endif
#endif
# if KEY_DOMDEC==1
  endif ! q_domdec
# endif
end subroutine repd_energy

!*******************************************************************************
!> Restart dynamics after an exchange. All threads should already have
!! all necessary info broadcast to them. 
!<
subroutine repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz
  integer                      :: jhstrt, igvopt

  ! Restart dynamics
  jhstrt=0
  igvopt=2
  ! This will update anything that needs updating for swapped coords
  call coords_update(x,y,z,wmain,vx,vy,vz)
  ! This energy call is needed so that DOMDEC nodes properly receive coords.
  ! TODO Only for DOMDEC?
  call repd_energy(x, y, z)
end subroutine repd_restart_dynamics

!*******************************************************************************
!> Broadcast unit cell params to all comm_charmm threads and set up cell.
!! Assumes X, Y, Z, and WMAIN have already been broadcast. 
!<
subroutine repd_bcast_cell(x,y,z,wmain)
  ! USE STATEMENTS
  use mpi
  use memory
  use image,    only: xtlabc, xucell, xnsymm
  use imgup,    only: upimag0
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain
  ! LOCAL VARIABLES
  real(chm_real),allocatable,dimension(:,:,:) :: transf ! For call to imfill
  integer ierr

  call mpi_bcast(xtlabc, 6, mpi_real8, 0, comm_charmm, ierr)
  call xtllat(xucell,xtlabc)
  call chmalloc('repdstr.src','repd_bcast_cell','TRANSF',3,4,XNSYMM,crl=transf)
  call imfill(transf,.false.)
  call chmdealloc('repdstr.src','repd_bcast_cell','TRANSF',3,4,XNSYMM,crl=transf)
  call upimag0(x,y,z,wmain,0) ! Do even if not qcrys?
end subroutine repd_bcast_cell


!*******************************************************************************
!> Swap all coordinates, velocities, etc between this and neighbor
!! in comm_rep_master 
!<
subroutine repd_swap_all(neighbor,qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
  ! USE STATEMENTS
  use mpi
  use psf,   only: natom
  use image, only: xtlabc
  ! ARGUMENTS
  implicit none
  integer, intent(in)          :: neighbor
  logical, intent(in)          :: qcrys
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  ! Exchange was accepted. Swap tags.
  call mpi_sendrecv_replace(reptag, 1, mpi_int, neighbor, 5, neighbor, 5, comm_rep_master, istat, ierr) 
  ! Now swap coordinates etc FIXME xold etc needed since restarting dynamics?
  call mpi_sendrecv_replace(x,     natom, mpi_real8, neighbor, 6,  neighbor, 6,  comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(y,     natom, mpi_real8, neighbor, 7,  neighbor, 7,  comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(z,     natom, mpi_real8, neighbor, 8,  neighbor, 8,  comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(vx,    natom, mpi_real8, neighbor, 9,  neighbor, 9,  comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(vy,    natom, mpi_real8, neighbor, 10, neighbor, 10, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(vz,    natom, mpi_real8, neighbor, 11, neighbor, 11, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(xold,  natom, mpi_real8, neighbor, 12, neighbor, 12, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(yold,  natom, mpi_real8, neighbor, 13, neighbor, 13, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(zold,  natom, mpi_real8, neighbor, 14, neighbor, 14, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(wmain, natom, mpi_real8, neighbor, 15, neighbor, 15, comm_rep_master, istat, ierr)
  if (qcrys) then
    call mpi_sendrecv_replace(xtlabc, 6, mpi_real8, neighbor, 16, neighbor, 16, comm_rep_master, istat, ierr)
  endif
end subroutine repd_swap_all

!*******************************************************************************
!> Send my SGLD info to neighbor in comm_rep_master, receive neighbors 
!! SGLD info. 
!<
subroutine send_sgld_info(eneigh, mytemp, neighbor)
  ! USE STATEMENTS
  use mpi
  use consta, only: kboltz
  use sgld,   only: sgffi,psgldg,epotlf,epothf
  ! ARGUMENTS
  implicit none
  real(chm_real), intent(out) :: eneigh   ! Output: neighbors energy
  real(chm_real), intent(in)  :: mytemp   ! My temperature
  integer, intent(in)         :: neighbor ! Neighbors comm_rep_master rank
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  ! Send all SGLD info to neighbor
  our_sgarray(1) = eneigh  ! pass in myepot
  our_sgarray(2) = epotlf-epothf
  our_sgarray(3) = 1.0d0 / (kboltz*mytemp)
  our_sgarray(4) = (sgffi-psgldg) / (kboltz*mytemp)
  our_sgarray(5) = repdid ! FIXME seems redundant since we have reptags now
  our_sgarray(6) = epotlf
  our_sgarray(7) = epothf
  our_sgarray(8) = mytemp
  call mpi_sendrecv(our_sgarray, 8, mpi_real8, neighbor, 30,&
                    nbr_sgarray, 8, mpi_real8, neighbor, 30,&
                    comm_rep_master, istat, ierr)
  eneigh = nbr_sgarray(1)
end subroutine send_sgld_info

!*******************************************************************************
!> Swap all SGLD-related variables between this and neighbor in comm_rep_master.
!! Perform any needed scaling of SGLD variables.
!! The sgarrays need to have been filled and swapped prior to this.
!<
subroutine repd_swap_sgld(scalsg, neighbor, estourlf)
  ! USE STATEMENTS
  use mpi
  use number, only: rsmall
  use psf,     only: natom, amass
  use sgld,    only: sgvx,sgvy,sgvz,sgkx,sgky,sgkz,epotlf,epothf
  use reawri,  only: delta ! Dynamics time step in AKMA units
  use consta,  only: timfac
  use cnst_fcm,only: fbeta
  ! ARGUMENTS
  implicit none
  integer, intent(in)         :: neighbor
  real(chm_real), intent(in)  :: estourlf ! Used to calculate low freq. scaling factor when HREMD 
  real(chm_real), intent(out) :: scalsg   ! Output: SGLD scaling factor
  ! LOCAL VARIABLES
  real(chm_real) :: myvv2,mykk2,nbrvv2,nbrkk2
  real(chm_real) :: fact, dteflf, dtcflf, elfscale,scalrd
  integer i
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)
 
  ! calcule magnitudes
    myvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
    mykk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)
  call mpi_sendrecv_replace(sgvx, natom, mpi_real8, neighbor, 36, neighbor, 36, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(sgvy, natom, mpi_real8, neighbor, 37, neighbor, 37, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(sgvz, natom, mpi_real8, neighbor, 38, neighbor, 38, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(sgkx, natom, mpi_real8, neighbor, 39, neighbor, 39, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(sgky, natom, mpi_real8, neighbor, 40, neighbor, 40, comm_rep_master, istat, ierr)
  call mpi_sendrecv_replace(sgkz, natom, mpi_real8, neighbor, 41, neighbor, 41, comm_rep_master, istat, ierr)
    nbrvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
    nbrkk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)

  scalsg=1.0d0
  scalrd=1.0d0
  if(nbrvv2>rsmall)scalsg = sqrt(myvv2/nbrvv2)
  if(nbrkk2>rsmall)scalrd = sqrt(mykk2/nbrkk2)
  !if (qrxtham) then ! TODO use exchange type in current dimension?
  !  elfscale = estourlf / epotlf
  !  epotlf = elfscale * epotlf
  !else
  !  elfscale = 1.d0
  !  epotlf = nbr_sgarray(1)
  !endif
  !dteflf = our_sgarray(7)-1.d0-(nbr_sgarray(7)-1.d0)*scalsg
  !dtcflf = our_sgarray(8)-1.d0-(nbr_sgarray(8)-1.d0)*scalsg
  do i = 1, natom
     sgvx(i) = sgvx(i)*scalsg
     sgvy(i) = sgvy(i)*scalsg
     sgvz(i) = sgvz(i)*scalsg
     sgkx(i) = sgkx(i)*scalrd
     sgky(i) = sgky(i)*scalrd
     sgkz(i) = sgkz(i)*scalrd
  enddo
end subroutine repd_swap_sgld

!*******************************************************************************
!> Broadcast all SGLD vars to threads in comm_charmm
!<
subroutine repd_bcast_sgld()
  ! USE STATEMENTS
  use mpi
  use psf,  only: natom
  use sgld, only: epotlf,epothf,sgvx,sgvy,sgvz,sgkx,sgky,sgkz
  ! ARGUMENTS
  implicit none
  ! MPI VARIABLES
  integer ierr

  call mpi_bcast(epotlf, 1, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(epothf, 1, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgvx, natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgvy, natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgvz, natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgkx, natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgky, natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(sgkz, natom, mpi_real8, 0, comm_charmm, ierr)
end subroutine repd_bcast_sgld

!*******************************************************************************
!> Broadcast all coordinates, velocities, etc to threads in comm_charmm
!<
subroutine repd_bcast_all(qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
  ! USE STATEMENTS
  use mpi
  use psf, only: natom
  ! ARGUMENTS
  implicit none
  logical, intent(in)          :: qcrys
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  ! MPI VARIABLES
  integer ierr

  call mpi_bcast(reptag, 1,     mpi_int,   0, comm_charmm, ierr)
  call mpi_bcast(x,      natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(y,      natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(z,      natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(vx,     natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(vy,     natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(vz,     natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(xold,   natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(yold,   natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(zold,   natom, mpi_real8, 0, comm_charmm, ierr)
  call mpi_bcast(wmain,  natom, mpi_real8, 0, comm_charmm, ierr)
  if (qcrys) call repd_bcast_cell(x,y,z,wmain) 
end subroutine repd_bcast_all

!*******************************************************************************
!> Scale velocities according to the square root of ratio of input temperatures 
!<
subroutine scale_velocities(scaled,mytemp,nbrtemp,vx,vy,vz,xold,yold,zold)
  ! USE STATEMENTS
  use psf, only: natom
  ! ARGUMENTS
  implicit none
  real(chm_real), intent(out)  :: scaled
  real(chm_real), intent(in)   :: mytemp, nbrtemp
  real(chm_real), dimension(*) :: vx, vy, vz, xold, yold, zold
  ! LOCAL VARIABLES
  integer i

  ! Now scale velocities
  scaled = sqrt(mytemp / nbrtemp)
  do i = 1, natom
    vx(i) = vx(i)*scaled
    vy(i) = vy(i)*scaled
    vz(i) = vz(i)*scaled
    xold(i) = xold(i)*scaled
    yold(i) = yold(i)*scaled
    zold(i) = zold(i)*scaled
  enddo
end subroutine scale_velocities


!*******************************************************************************
!> pH exchange calc
!<
subroutine phrex_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold, &
                          ididphrex, &
                          istep,jhstrt,igvopt,neighbor,qcrys,rnum)
  ! USE STATEMENTS
  use mpi
  use psf,      only: natom, nres, cg
  use consph,   only: tstate ! NOTE: tstate only allocated on rep masters
  use sgld,     only: qsgld
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  integer, intent(out)         :: ididphrex
  integer                      :: jhstrt, igvopt
  integer, intent(in)          :: istep, neighbor
  logical, intent(in)          :: qcrys
  real(chm_real), intent(in)   :: rnum           ! Random number
  ! LOCAL VARIABLES
  real(chm_real) :: ph_l     ! My pH
  real(chm_real) :: ph_m     ! Neighbor pH
  real(chm_real) :: ph_delta
  real(chm_real) :: exprob   ! Exchange probability
  logical        :: qexc     ! True if exchange accepted
  integer        :: iproto   ! Sum of my residues protonation states
  integer        :: jproto   ! Sum of neighbors residues protonation states
  real(chm_real) :: eneigh,mytemp,scalesg ! SGLD only
  ! loop var
  integer i
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  qexc = .false.
  ph_delta = 0.d0
  ph_l = my_replica_ph()
  ph_m = 0.d0
  exprob = 0.d0
  iproto = 0
  jproto = 0
  ididphrex = 0

  if (neighbor.gt.-1) then
    ididphrex = 1
    if (qrepmaster) then
      if (ndims .gt. 1) then
        ! Send my pH, get neighbors pH.
        call mpi_sendrecv(ph_l, 1, mpi_real8, neighbor, 61, &
                          ph_m, 1, mpi_real8, neighbor, 61, &
                          comm_rep_master, istat, ierr)
      else
        ! In one dimension just get neighbors pH from array.
        ph_m = phrx(neighbor+1)
      endif
      ! We are state i and have a pH value of pH_l, our neighbor
      ! is state j and has a pH value of pH_m.
      ! Count number of prototnated residues (in state 1) and swap with
      ! the neighbor.
      do i = 1, nres
         if (tstate(i).eq.1) iproto = iproto + 1 
      enddo
      call mpi_sendrecv(iproto, 1, mpi_integer, neighbor, 1, &
                        jproto, 1, mpi_integer, neighbor, 1, comm_rep_master, istat, ierr)

      ph_delta = log(10.0)*(ph_m - ph_l)*(iproto - jproto)

      if (qcontrol_exchg) then
        ! I control the exchange. Calculate acceptance prob.
        if (ph_delta.le.0.d0) then
           exprob = 1.d0
        else
          exprob=min(1.d0,exp(-ph_delta))
        endif
        ! Determine if exchange was accepted
        qexc = (exprob .gt. rnum)
        ! Send the result from odd to even replica
        call mpi_send(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, ierr)
        call mpi_send(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, ierr)
        call mpi_send(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, ierr)
      else
        ! Neighbor controls exchange. Receive results of exchange calc.
        call mpi_recv(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, istat, ierr)
        call mpi_recv(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, istat, ierr)
        call mpi_recv(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, istat, ierr)
      endif
#     if KEY_DEBUGREPD==1
      write(100+mynodg,'(2(a,f10.4),2(a,i6),a,l3)') 'DBG: prob= ', exprob, ' rn= ', rnum, ' iproto= ', iproto, ' jproto= ', jproto, ' qexc= ', qexc
#     endif

      if (qexc) then
        ! Exchange was accepted. Swap coords/velo/tags/box
        call repd_swap_all(neighbor,qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
        ! For ph replica need to swap cg and tstate as well
        call mpi_sendrecv_replace(cg, natom, mpi_real8, neighbor, 17, neighbor, 17, comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(tstate, nres, mpi_integer, neighbor, 18, neighbor, 18, comm_rep_master, istat, ierr)
        if (qsgld) then
          ! If running SGLD, swap those variables as well.
          mytemp = my_replica_temperature()
          call send_sgld_info(eneigh, mytemp, neighbor)
          ! Swap SGLD stuff. Last arg only important when HREMD
          call repd_swap_sgld(scalesg, neighbor, 0.d0)
        endif
      endif ! Exchange accepted (rep master)
    endif ! qrepmaster
  endif ! neighbor > -1

  ! Broadcast results of exchange calc to replica group
  call mpi_bcast(qexc,   1, mpi_logical, 0, comm_charmm, ierr)

  if (qexc) then
    ! Exchange was accepted, so coords were swapped. Master needs to broadcast
    ! new data to the rest of the replica group
    call repd_bcast_all(qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
    ! Broadcast cg; tstate only exists on masters (i.e. mynod .eq. 0)
    call mpi_bcast(cg, natom, mpi_real8, 0, comm_charmm, ierr)
    ! Broadcast sgld stuff
    if (qsgld) call repd_bcast_sgld
    ! Restart dynamics
    call repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  endif ! Exchange happened (all)

  ! Print Results. Only masters will perform actual write.
  ! NOTE: Not writing ttemp, mytemp, emine, nbrtemp, eneigh, and scaled
  call write_log(qexc, istep, neighbor, exprob, 0.d0, 0.d0, 0.d0, &
                 0.d0, 0.d0, 1.d0, rnum, &
                 0.d0, 0.d0, 0.d0, &
                 ph_l, ph_m, iproto, jproto, &
                 scalesg)

end subroutine phrex_exchange

!*******************************************************************************
!> SGLD exchange calc.
!<
subroutine sgld_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
  ! USE STATEMENTS
  use mpi
  use repdstr,only: temprx
  use number, only: rsmall
  use stream, only: outu
  !use sgld,   only: treflf,trxlf,avgtlf
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  real(chm_real)               :: myepot, ttemp
  integer                      :: jhstrt, igvopt
  integer, intent(in)          :: istep, neighbor
  logical, intent(in)          :: qcrys
  real(chm_real), intent(in)   :: rnum           ! Random number
  ! LOCAL VARIABLES
  real(chm_real) :: mytemp         ! My temperature
  real(chm_real) :: nbrtemp        ! My neighbors temperature
  real(chm_real) :: emine          ! My potential energy; myepot changes after exchange
  real(chm_real) :: eneigh         ! Neighbors potential energy
  real(chm_real) :: sgldarg        ! Temp var for SGLD exchange prob calc
  real(chm_real) :: exprob         ! SGLD exchange probability
  real(chm_real) :: scaled         ! Velocity scaling factor
  real(chm_real) :: scalesg        ! SGLD scaling factor
  logical        :: qexc           ! True if exchange is accepted
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  mytemp = my_replica_temperature()
  nbrtemp = 0.d0
  emine = myepot
  eneigh = myepot
  sgldarg = 0.d0
  exprob = 0.d0
  qexc = .false.
  scaled = 1.d0
  scalesg = 1.d0


  if (neighbor.gt.-1) then
    if (qrepmaster) then
      nbrtemp = nbr_temperature(mytemp, neighbor)
      ! Send all SGLD info to neighbor
      call send_sgld_info(eneigh, mytemp, neighbor)

      if (qcontrol_exchg) then
        ! Calculate exchange probability
        sgldarg = -((our_sgarray(3)-nbr_sgarray(3))*(nbr_sgarray(1)-our_sgarray(1)) &
                   -(our_sgarray(4)-nbr_sgarray(4))*(nbr_sgarray(2)-our_sgarray(2)))
        exprob = min(1.d0, exp(sgldarg))
        ! Determine if exchange was accepted
        qexc = (exprob .gt. rnum)
        ! Send the result from odd to even replica
        call mpi_send(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, ierr)
        call mpi_send(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, ierr)
        call mpi_send(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, ierr)
        call mpi_send(sgldarg,1, mpi_real8,   neighbor,15, comm_rep_master, ierr)
      else
        ! Neighbor controls exchange. Receive results of exchange calc.
        call mpi_recv(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, istat, ierr)
        call mpi_recv(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, istat, ierr)
        call mpi_recv(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, istat, ierr)
        call mpi_recv(sgldarg,1, mpi_real8,   neighbor,15, comm_rep_master, istat, ierr)
      endif
      if (qexc) then
        ! Exchange was accepted. Swap coords/velo/tags/box
        call repd_swap_all(neighbor,qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
        ! Swap SGLD stuff. Last arg only important when HREMD
        call repd_swap_sgld(scalesg, neighbor, 0.d0)
        ! Now scale velocities
        call scale_velocities(scaled,mytemp,nbrtemp,vx,vy,vz,xold,yold,zold)
      endif ! Exchange accepted (rep master)
    endif ! Replica master
  endif ! Has neighbor

  ! Broadcast results of exchange calc to replica group
  call mpi_bcast(qexc,   1, mpi_logical, 0, comm_charmm, ierr)

  if (qexc) then
    ! Exchange was accepted, so coords were swapped. Master needs to broadcast
    ! new data to the rest of the replica group
    call repd_bcast_all(qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
    ! Broadcast sgld stuff
    call repd_bcast_sgld
    ! Restart dynamics
    call repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  endif ! Exchange happened (all)

  ! Print Results. Only masters will perform actual write.
  call write_log(qexc, istep, neighbor, exprob, ttemp, mytemp, emine, &
                 nbrtemp, eneigh, scaled, rnum, &
                 0.d0, 0.d0, 0.d0, &
                 0.d0, 0.d0, 0, 0, &
                 scalesg)

end subroutine sgld_exchange

!*******************************************************************************
!> Hamiltonian exchange calc, optionally with SGLD. 
!<
subroutine hamiltonian_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
  ! USE STATEMENTS
  use mpi
  use psf,      only: natom
  use consta,   only: kboltz
  use image,    only: xtlabc
  use stream,   only: outu ! DBG for SGLD
  use sgld,     only: qsgld
# if KEY_DEBUGREPD==1
  use energym
# else
  use energym,  only: energy, eprop, EPOT ! For energy calc
# endif
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  real(chm_real)               :: myepot, ttemp
  integer                      :: jhstrt, igvopt
  integer, intent(in)          :: istep, neighbor
  logical, intent(in)          :: qcrys
  real(chm_real), intent(in)   :: rnum           ! Random number
  ! LOCAL VARIABLES
  real(chm_real) :: mytemp         ! My temperature
  real(chm_real) :: nbrtemp        ! My neighbors temperature
  real(chm_real) :: ourpoti        ! My potential energy
  real(chm_real) :: nbrpotj        ! Neighbor potential energy
  real(chm_real) :: ourpotj        ! My PE with neighbor coords
  real(chm_real) :: nbrpoti        ! Neighbor PE with my coords 
  real(chm_real) :: exprob         ! Exchange probability
  logical        :: qexc           ! True if exchange accepted
  real(chm_real) :: scaled         ! Velocity scaling factor
  real(chm_real) :: hrexarg        ! Temp var for HREX prob calc
  real(chm_real) :: pvterm         ! Pressure/volume correction for NPT
  real(chm_real) :: estourlf, estnbrlf, scalesg
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  mytemp = my_replica_temperature()
  nbrtemp = 0.d0
  ourpoti = myepot
  nbrpoti = myepot
  ourpotj = 0.d0
  nbrpotj = 0.d0
  exprob = 0.d0
  pvterm = 0.d0
  qexc = .false.
  scaled = 1.d0
  scalesg = 1.d0
  hrexarg = 0.d0

  ! For Hamiltonian we need to do another energy eval. This means all
  ! threads must be involved.
  if (neighbor.gt.-1) then
    ! Now need to get my energy with neighbor coords and vice versa.
    if (qrepmaster) then
      nbrtemp = nbr_temperature(mytemp, neighbor)
      ! If SGLD exchange, need to send all SGLD info to neighbor.
      if (qrxsgld) call send_sgld_info(nbrpotj, mytemp, neighbor)
      ! Exchange potential energies
      call mpi_sendrecv(ourpoti, 1, mpi_real8, neighbor, 1, &
                        nbrpotj, 1, mpi_real8, neighbor, 1, comm_rep_master, istat, ierr)
      if (q_use_pv) pvterm = pv_correction(mytemp, nbrtemp, neighbor)
      ! First backup original coords.
      ! TODO for domdec may need to gather coords to master thread
      x0(1:natom) = x(1:natom)
      y0(1:natom) = y(1:natom)
      z0(1:natom) = z(1:natom)
      wmain0(1:natom) = wmain(1:natom)
      ! Now swap coords with neighbor.
      call mpi_sendrecv_replace(x,    natom,mpi_real8,neighbor,100,neighbor,100,comm_rep_master,istat,ierr)
      call mpi_sendrecv_replace(y,    natom,mpi_real8,neighbor,101,neighbor,101,comm_rep_master,istat,ierr)
      call mpi_sendrecv_replace(z,    natom,mpi_real8,neighbor,102,neighbor,102,comm_rep_master,istat,ierr)
      call mpi_sendrecv_replace(wmain,natom,mpi_real8,neighbor,103,neighbor,103,comm_rep_master,istat,ierr)
      if (qcrys) then
        ! Save/swap box
        xtlabc0(1:6) = xtlabc(1:6)
        call mpi_sendrecv_replace(xtlabc,6,mpi_real8,neighbor,104,neighbor,104,comm_rep_master,istat,ierr)
      endif
    endif ! qrepmaster
    ! Ensure coords are properly broadcast
    call mpi_bcast(x, natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(y, natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(z, natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(wmain, natom, mpi_real8, 0, comm_charmm, ierr)
    if (qcrys) call repd_bcast_cell(x,y,z,wmain)
    ! Let charmm know coordinates have changed
    call coords_update(x,y,z,wmain,vx,vy,vz)
    ! Calculate new energy with neighbors coords TODO do SGLD arrays need to be swapped?
    call repd_energy(x,y,z)
    ourpotj = eprop(EPOT)
#   if KEY_DEBUGREPD==1
    write(100+mynodg,'(a,7(F16.4))') 'DBG: ', EPROP(EPOT), ETERM(BOND),ETERM(ANGLE),ETERM(DIHE)+ETERM(CMAP),ETERM(IMDIHE),ETERM(VDW),ETERM(ELEC)
#   endif
    if (qrepmaster) then
      ! Let neighbor know what that energy was
      call mpi_sendrecv(ourpotj, 1, mpi_real8, neighbor, 1, &
                        nbrpoti, 1, mpi_real8, neighbor, 1, comm_rep_master, istat, ierr)
      if (qrxsgld) then
        ! Calculate these SGLD factors here since both replicas will need them
        write(outu,'(a,f10.6)') 'REPD> THAM+SGLD: RATIO OF ENERGY ARRAYS IS ',ourpotj/nbrpotj
        estourlf=(ourpotj/nbrpotj)*nbr_sgarray(1) ! LF if we had run the neighbor's hamiltonian
        estnbrlf=(nbrpoti/ourpoti)*our_sgarray(1) ! LF if the neighbor had run with our hamiltonian
      endif
      if (qcontrol_exchg) then
        ! Calculate exchange probability.
        ! NOTE: If using generalize SGLD use the normal exchange criterion
        if (qrxsgld.and..not.qsgld)then
          ! SGLD: We need to use a modified exchange potentials
          !mu_m    = our_sgarray(4)
          !mu_n    = nbr_sgarray(4)
          !mu_mtld = our_sgarray(3)
          !mu_ntld = nbr_sgarray(3)
          !ourlf   = our_sgarray(1)
          !nbrlf   = nbr_sgarray(1)
          hrexarg = -our_sgarray(3)*(estourlf-our_sgarray(1)) - our_sgarray(4)*(ourpotj-ourpoti) &
                    -nbr_sgarray(3)*(estnbrlf-nbr_sgarray(1)) - nbr_sgarray(4)*(nbrpoti-nbrpotj)
        else
          ! Normal HREMD exchange calc
          hrexarg = -((1.d0/(kboltz*mytemp))*(ourpotj-ourpoti) &
                    -(1.d0/(kboltz*nbrtemp))*(nbrpotj-nbrpoti))
        endif
        exprob = min(1.d0, exp(hrexarg+pvterm))
        ! Determine if exchange was accepted
        qexc = (exprob .gt. rnum)
        ! Send the result from odd to even replica
        call mpi_send(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, ierr)
        call mpi_send(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, ierr)
        call mpi_send(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, ierr)
        call mpi_send(hrexarg,1, mpi_real8,   neighbor,15, comm_rep_master, ierr)
      else
        ! Neighbor controls exchange. Receive results of exchange calc.
        call mpi_recv(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, istat, ierr)
        call mpi_recv(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, istat, ierr)
        call mpi_recv(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, istat, ierr)
        call mpi_recv(hrexarg,1, mpi_real8,   neighbor,15, comm_rep_master, istat, ierr)
      endif
#     if KEY_DEBUGREPD==1
      write(100+mynodg,'(2(a,f10.4),a,l3)') 'DBG: prob= ', exprob, ' rn= ', rnum, ' qexc= ', qexc
      write(100+mynodg,'(5(a,f16.4))') 'DBG: myepot= ', myepot, ' ourpoti= ', ourpoti,&
                                       ' ourpotj= ', ourpotj, ' nbrpoti= ', nbrpoti,&
                                       ' nbrpotj= ', nbrpotj
#     endif
      if (qexc) then
        ! Exchange was accepted. Swap tags.
        call mpi_sendrecv_replace(reptag, 1, mpi_int, neighbor, 5, neighbor, 5, comm_rep_master, istat, ierr) 
        ! Coords/box have already been swapped. Now swap velocities etc. 
        call mpi_sendrecv_replace(vx,    natom, mpi_real8, neighbor, 9,  neighbor, 9,  comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(vy,    natom, mpi_real8, neighbor, 10, neighbor, 10, comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(vz,    natom, mpi_real8, neighbor, 11, neighbor, 11, comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(xold,  natom, mpi_real8, neighbor, 12, neighbor, 12, comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(yold,  natom, mpi_real8, neighbor, 13, neighbor, 13, comm_rep_master, istat, ierr)
        call mpi_sendrecv_replace(zold,  natom, mpi_real8, neighbor, 14, neighbor, 14, comm_rep_master, istat, ierr)
        ! Swap SGLD stuff. This will update epotlf. estourlf used to calc LF scaling factor.
        if (qrxsgld) call repd_swap_sgld(scalesg, neighbor, estourlf)
        ! Scale velocities
        call scale_velocities(scaled,mytemp,nbrtemp,vx,vy,vz,xold,yold,zold)
      else
        ! Exchange rejected. Restore original coords
        x(1:natom) = x0(1:natom)
        y(1:natom) = y0(1:natom)
        z(1:natom) = z0(1:natom)
        wmain(1:natom) = wmain0(1:natom)
        if (qcrys) xtlabc(1:6) = xtlabc0(1:6)
      endif ! Exchange on master
    endif ! qrepmaster
  endif ! Has neighbor
 
  ! Broadcast results of exchange calc to replica group
  call mpi_bcast(qexc,   1, mpi_logical, 0, comm_charmm, ierr)

  if (qexc) then
    ! Exchange accepted. Coords/box already bcast
    call mpi_bcast(reptag, 1,     mpi_int,   0, comm_charmm, ierr)
    call mpi_bcast(vx,     natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(vy,     natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(vz,     natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(xold,   natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(yold,   natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(zold,   natom, mpi_real8, 0, comm_charmm, ierr)
    ! Broadcast sgld stuff
    if (qrxsgld) call repd_bcast_sgld
    ! Restart dynamics
    call repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  else if (neighbor .gt. -1) then
    ! Exchange rejected or no exchange attempted.
    ! Bcast original coords/box only if they were previously swapped.
    call mpi_bcast(x,      natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(y,      natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(z,      natom, mpi_real8, 0, comm_charmm, ierr)
    call mpi_bcast(wmain,  natom, mpi_real8, 0, comm_charmm, ierr)
    if (qcrys) call repd_bcast_cell(x,y,z,wmain)
    ! Let charmm know coords have changed back
    call coords_update(x,y,z,wmain,vx,vy,vz)
    call repd_energy(x,y,z) ! FIXME is this call necessary?
  endif

  ! Print Results. Only masters will perform actual write.
  call write_log(qexc, istep, neighbor, exprob, ttemp, mytemp, ourpoti, &
                 nbrtemp, nbrpotj, scaled, rnum, &
                 nbrpoti, ourpotj, hrexarg, &
                 0.d0, 0.d0, 0, 0, &
                 scalesg)

end subroutine hamiltonian_exchange

!*******************************************************************************
!> Temperature exchange calc.
!<
subroutine temperature_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
  ! USE STATEMENTS
  use mpi
  use consta,   only: kboltz
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  real(chm_real)               :: myepot, ttemp
  integer                      :: jhstrt, igvopt
  integer, intent(in)          :: istep, neighbor
  logical, intent(in)          :: qcrys
  real(chm_real), intent(in)   :: rnum           ! Random number
  ! LOCAL VARIABLES
  real(chm_real) :: mytemp         ! My temperature
  real(chm_real) :: nbrtemp        ! My neighbors temperature
  real(chm_real) :: emine          ! My potential energy; myepot changes after exchange
  real(chm_real) :: eneigh         ! Neighbor potential energy
  real(chm_real) :: exprob         ! Exchange probability
  real(chm_real) :: delta          ! Exchange delta
  real(chm_real) :: pvterm         ! Pressure/volume correction for NPT
  logical        :: qexc           ! True if exchange accepted
  real(chm_real) :: scaled         ! Velocity scaling factor
  ! MPI VARIABLES
  integer ierr, istat(mpi_status_size)

  mytemp = my_replica_temperature()
  nbrtemp = 0.d0
  emine = myepot
  eneigh = 0.d0
  exprob = 0.d0
  pvterm = 0.d0
  scaled = 1.d0
  qexc = .false.

  ! Only replica masters will calculate exchange probability
  ! NOTE According to original functionality, ODD replica controls exchange
  if (neighbor.gt.-1) then
    if (qrepmaster) then
      ! Get neighbors temperature
      nbrtemp = nbr_temperature(mytemp, neighbor)
      ! Exchange potential energies and calculate the exchange probability
      ! NOTE: For correct log printout, swap energies even though I dont need it
      call mpi_sendrecv(emine,  1, mpi_real8, neighbor, 1, &
                        eneigh, 1, mpi_real8, neighbor, 1, comm_rep_master, istat, ierr)
      if (q_use_pv) pvterm = pv_correction(mytemp, nbrtemp, neighbor)
      if (qcontrol_exchg) then
        ! I control the exchange. Calculate acceptance prob.
        if(mytemp == nbrtemp) then
          exprob=1.d0
        else
          delta = ((1.d0/(kboltz*nbrtemp)) - (1.d0/(kboltz*mytemp))) * (eneigh - emine)
          exprob = min(1.d0, exp(delta + pvterm))
#         if KEY_DEBUGREPD==1
          write(100+mynodg,'(a,E16.8,E16.8)') 'DBG: exp(delta) and exp(delta+pv)', exp(delta), exp(delta+pvterm)
#         endif
          !exprob=min(1.d0,exp(-(1.d0/(kboltz*mytemp) &
          !                     -1.d0/(kboltz*nbrtemp))*(eneigh-emine)))
        endif
        ! Determine if exchange was accepted
        qexc = (exprob .gt. rnum)
        ! Send the result from odd to even replica
        call mpi_send(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, ierr)
        call mpi_send(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, ierr)
        call mpi_send(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, ierr)
      else
        ! Neighbor controls exchange. Receive results of exchange calc.
        call mpi_recv(qexc,   1, mpi_logical, neighbor, 2, comm_rep_master, istat, ierr)
        call mpi_recv(rnum,   1, mpi_real8,   neighbor, 3, comm_rep_master, istat, ierr)
        call mpi_recv(exprob, 1, mpi_real8,   neighbor, 4, comm_rep_master, istat, ierr)
      endif
#     if KEY_DEBUGREPD==1
      write(100+mynodg,'(3(a,f10.4),a,l3)') 'DBG: prob= ', exprob, ' rn= ', rnum, ' myepot= ', myepot, ' qexc= ', qexc
#     endif

      if (qexc) then
        ! Exchange was accepted. Swap coords/velo/tags/box
        call repd_swap_all(neighbor,qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
        ! Now scale velocities
        call scale_velocities(scaled,mytemp,nbrtemp,vx,vy,vz,xold,yold,zold)
      endif ! Exchange accepted (rep master)
    endif ! Replica master
  endif ! Has neighbor

  ! Broadcast results of exchange calc to replica group
  call mpi_bcast(qexc,   1, mpi_logical, 0, comm_charmm, ierr)

  if (qexc) then
    ! Exchange was accepted, so coords were swapped. Master needs to broadcast
    ! new data to the rest of the replica group
    call repd_bcast_all(qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
    ! Restart dynamics
    call repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  endif ! Exchange happened (all)

  ! Print Results. Only masters will perform actual write.
  call write_log(qexc, istep, neighbor, exprob, ttemp, mytemp, emine, &
                 nbrtemp, eneigh, scaled, rnum, &
                 0.d0, 0.d0, 0.d0, &
                 0.d0, 0.d0, 0, 0, &
                 0.d0)

end subroutine temperature_exchange

!*******************************************************************************
!> \return my replica temperature.
!> NOTE: Implemented as a separate function so it can be called internally.
!<
function my_replica_temperature()
  ! USE STATEMENTS
  use repdstr, only: temprx
  ! ARGUMENTS
  implicit none
  real(chm_real) my_replica_temperature
  ! LOCAL VARIABLES
  integer :: mytidx
  ! Determine my index in the temperature dimension
  mytidx = mydims(tdim_idx)
  my_replica_temperature = temprx(mytidx+1)
end function my_replica_temperature

!*******************************************************************************
!> Get my neighbors temperature. Must ONLY be called by qrepmaster when
!! neighbor is not -1.
!<
function nbr_temperature(ourtemp, neighbor)
  ! USE STATEMENTS
  use mpi
  use repdstr, only: temprx
  ! ARGUMENTS
  implicit none
  real(chm_real)             :: nbr_temperature
  real(chm_real), intent(in) :: ourtemp
  integer, intent(in)        :: neighbor
  ! LOCAL VARIABLES
  real(chm_real) :: nbrtemp
  integer ierr, istat(mpi_status_size)

  if (ndims .gt. 1) then
    ! Send my temperature, get neighbors temperature.
    call mpi_sendrecv(ourtemp, 1, mpi_real8, neighbor, 21, &
                      nbrtemp, 1, mpi_real8, neighbor, 21, &
                      comm_rep_master, istat, ierr)
  else
    ! In one dimension just get neighbors temperature from array.
    nbrtemp = temprx(neighbor+1)
  endif
  nbr_temperature = nbrtemp
end function nbr_temperature

!*******************************************************************************
!> Compute pressure/volume correction to exchange delta; should be added on. 
!! Must ONLY be called by qrepmaster when neighbor is not -1.
!<
function pv_correction(ourtemp, nbrtemp, neighbor)
  ! USE STATEMENTS
  use mpi
  use energym, only: eprop,volume
  use consta,  only: kboltz
  use reawri,  only: refp, rpxx, rpyy, rpzz
  ! ARGUMENTS
  implicit none
  real(chm_real) pv_correction
  real(chm_real), intent(in)  :: ourtemp, nbrtemp
  integer, intent(in)         :: neighbor
  ! LOCAL VARIABLES
  real(chm_real) :: ourpress, nbrpress, ourvol, nbrvol
  integer ierr, istat(mpi_status_size)

  ourpress = (refp(rpxx) + refp(rpyy) + refp(rpzz)) / 3.d0
  ourvol   = eprop(volume)
  !write(100+mynodg,'(2(a,f16.8),a)') 'DBG: OurPressure= ', ourpress, ' atm, OurVolume= ', ourvol, ' Ang^3'
  call mpi_sendrecv(ourpress, 1, mpi_real8, neighbor, 52, &
                    nbrpress, 1, mpi_real8, neighbor, 52, &
                    comm_rep_master, istat, ierr)
  call mpi_sendrecv(ourvol, 1, mpi_real8, neighbor, 53, &
                    nbrvol, 1, mpi_real8, neighbor, 53, &
                    comm_rep_master, istat, ierr)
  pv_correction = (((1.d0/(kboltz*nbrtemp))*nbrpress) - ((1.d0/(kboltz*ourtemp))*ourpress)) &
                  * (nbrvol - ourvol)
  ! Need to convert from units of atm*Ang^3 to kcal/mol
  ! 1 atm*Ang^3 * 101325 Pa/atm * E-30 m^3/Ang^3 * J/Pa*m^3 * kcal/4184 J * 6.02E23/mol
  pv_correction = pv_correction * 1.4584E-5
# if KEY_DEBUGREPD==1
  write(100+mynodg,'(a,E16.8)') 'DBG: PvCorrection= ', pv_correction
# endif
end function pv_correction

!*******************************************************************************
!> \return my replica pH
!<
function my_replica_ph()
  ! ARGUMENTS
  implicit none
  real(chm_real) :: my_replica_ph
  ! LOCAL VARIABLES
  integer :: myphidx
  myphidx = mydims(phdim_idx)
  my_replica_ph = phrx(myphidx+1)
end function my_replica_ph

!*******************************************************************************
!> Ensure that coordinates and velocities are properly gathered and distributed
!> before an exchange attempt.
!<
subroutine distribute_coords_and_velocities(x, y, z, vx, vy, vz, xold, yold, zold)
  ! USE STATEMENTS
# if KEY_DOMDEC==1
  use domdec_common, only: q_domdec, q_split
  use domdec_d2d_comm, only: copy_to_all
  use domdec_dr_common,only: copy_to_recip
# endif
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: vx, vy, vz, xold, yold, zold
  ! LOCAL VARS

# if KEY_DOMDEC==1
  if (q_domdec) then
    ! Get coordinates
    ! DRR - I think we can assume coords/forces already transferred at the end of dynamics
    !call transfer_coord(x,y,z,.true.) TODO Check
    !call transfer_force(dx,dy,dz)
    call copy_to_all(x,y,z)
    call copy_to_all(vx,vy,vz)
    call copy_to_all(xold,yold,zold)
    if (q_split) call copy_to_recip(x,y,z)
  else
# endif
  ! Ensure velocities have been properly distributed
    call vdgbr(vx,vy,vz,1)
# if KEY_DOMDEC==1
  endif
# endif
end subroutine distribute_coords_and_velocities

!*******************************************************************************
subroutine reservoir_exchange(rsv_id, x, y, z, wmain, vx, vy, vz, xold, yold, zold, &
                              ididphrex, &
                              myepot, ttemp, istep, jhstrt, iseed, iasvel, igvopt, qcrys)
  ! USE STATEMENTS
  use mpi
# if KEY_TSM==1
  use tsmh, only     : backls
# endif
  use psf,  only     : nres   ! for phrex
  use consph,   only : tstate ! for phrex, NOTE: tstate only allocated on rep masters
  ! ARGUMENTS
  implicit none
  integer, intent(in)          :: rsv_id ! ID of reservoir to exchange with.
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  integer                      :: ididphrex
  real(chm_real)               :: myepot, ttemp
  integer                      :: istep, jhstrt, iseed, iasvel, igvopt
  logical, intent(in)          :: qcrys ! True if pbc present
  ! LOCAL VARS
  logical        :: qhigh       ! If true, exchanging with high reservoir.
  real(chm_real) :: mytemp      ! My temperature
  real(chm_real) :: emine       ! My potential energy; myepot changes after exchange
  real(chm_real) :: exprob      ! Calculated exchange prob with reservoir
  real(chm_real) :: rn          ! Random number used with reservoir. TODO use rnum?
  real(chm_real) :: eneigh      ! Energy of reservoir structure
  integer        :: res_struct  ! ID of structure used from reservoir
  logical        :: qexc        ! True if structure was exchanged with reservoir.
  integer        :: iproto      ! Number of protonated residues on this replica.
  integer        :: jproto      ! Number of protonated residues on reservoir replica.
  real(chm_real) :: ph_l        ! My pH
  real(chm_real) :: ph_m        ! Reservoir pH
  integer ires, ierr

  mytemp = my_replica_temperature()
  emine  = myepot
  iproto = 0
  ph_l   = 0.d0
  if (qphrex) ididphrex = 1
  ! TODO modify if more than low and high reservoirs present
  qhigh = (rsv_id.eq.nbrtype_rsv1)
  ! TODO does reservoir exchange calc need PV correction?
  if (qrepmaster) then
    if (qphrex) then
      ph_l = my_replica_ph()
      ! Count number of prototnated residues (in state 1) and swap with
      ! the neighbor.
      do ires = 1, nres
         if (tstate(ires).eq.1) iproto = iproto + 1 
      enddo
      call resexch_ph(ttemp, qhigh, x, y, z, vx, vy, vz, ph_l, iproto, &
                      iseed, iasvel, igvopt, exprob, rn, jproto, ph_m, qexc, res_struct &
#                     if KEY_TSM==1
                      ,backls &
#                     endif
                     )

    else
      call resexch(qhigh, x, y, z, vx, vy, vz, qexc, mytemp, myepot, iseed, iasvel, igvopt, &
                   exprob, rn, eneigh, res_struct &
#                  if KEY_TSM==1
                   ,backls & 
#                  endif
                  )
    endif
  endif ! qrepmaster

  ! Broadcast results of exchange calc to replica group
  call mpi_bcast(qexc,   1, mpi_logical, 0, comm_charmm, ierr)

  if (qexc) then
    ! Exchange was accepted, so coords were swapped. Master needs to broadcast
    ! new data to the rest of the replica group
    call repd_bcast_all(qcrys,x,y,z,wmain,vx,vy,vz,xold,yold,zold)
    ! Restart dynamics
    call repd_restart_dynamics(x,y,z,wmain,vx,vy,vz,jhstrt,igvopt)
  endif ! Exchange accepted

  ! NOTE: Passing in scaled = 1.0, neighbor = -1, and nbrtemp = 0.0 for backwards compat.
  call write_log(qexc, istep, -1, exprob, ttemp, mytemp, emine, &
                 0.d0, eneigh, 1.d0, rn, &
                 0.d0, 0.d0, 0.d0, &
                 ph_l, ph_m, iproto, jproto, &
                 0.d0, res_struct)

end subroutine reservoir_exchange

!*******************************************************************************
!> New exchange calc. Can handle exchanges from 1 to N dimensions.
!<
subroutine exchange_calc_ndims(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                               ttemp,istep,jhstrt,iseed,iasvel,igvopt,ididphrex)
  ! USE STATEMENTS
  use repdstr,  only: irepdstr, repseed, temprx, comm_rpg
  use stream,   only: prnlev, outu
  use image,    only: xtltyp
  use clcg_mod, only: random
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  real(chm_real)               :: myepot, ttemp
  integer                      :: istep,jhstrt,iseed,iasvel,igvopt, ididphrex
  ! LOCAL VARS
  real(chm_real) :: rnum              ! Random number
  integer        :: neighbor          ! Replica index of neighbor
  logical        :: qcrys             ! True if PBC present
  logical        :: dim_will_exchange ! True if any dimension will exchange
  logical        :: rep_is_odd        ! True if index in this dimension is odd
  integer        :: myidx             ! My index in the current dimension
  integer        :: nbridx            ! Neighbors index in the current dimension
  integer        :: attempts_offset   ! For consistency with previous results set to 1 for ndims > 1
  logical        :: qexch_w_reservoir ! If true, attempting an exchange with a reservoir.
  integer cdim 

  ! Determine which dimension(s) will exchange.
  ! NOTE: For now, to be consistent with previous 2D results, if both
  !       dimensions would be active the highest one takes precedence.
  !       May want to change this in the future.
  currdim = 0
  dim_will_exchange=.false. 
  do cdim = 1, ndims
    if (mod(istep,exch_freq(cdim)).eq.0) then
      dim_exchange(cdim) = .true.
      dim_will_exchange = .true.
      currdim = cdim
    else
      dim_exchange(cdim) = .false.
    endif
  enddo
  ! Leave if no dimension will exchange
  if (.not.dim_will_exchange) then
    ! DOMDEC will be restarted outside this subroutine
    return
  endif

  ! NOTE - for consistency with previous 2D results
  if (ndims.gt.1) then
    attempts_offset = 1
    if (.not.allow_multi_exchg) then
      ! If we are not allowing multiple exchanges set all dimensions except
      ! currdim to false.
      dim_exchange(1:ndims) = .false.
      dim_exchange(currdim) = .true.
    endif
  else
    attempts_offset = 0
  endif
  ididphrex = 0
  ! Save original tag
  rep_oldtag = reptag
  ! Determine if PBC exist
  qcrys = (xtltyp.ne.'    ')

  ! Ensure coords/velocities properly gathered/distributed before exchange attempt(s)
  call distribute_coords_and_velocities(x, y, z, vx, vy, vz, xold, yold, zold)

  ! NOTE : currdim is a module variable because write_log depends on it
  do currdim = 1, ndims
    if (dim_exchange(currdim)) then
      if (qrepmaster.and.prnlev.gt.2) write(outu,'(a,i6)') 'REPD> New exchange calculation in use, dimension ', currdim
#     if KEY_DEBUGREPD==1
      write(100+mynodg, '(3(a,i6))') 'DBG: Entering REPEXCHG on step ', istep, ', exch ', nattempts(currdim), ', dim ', currdim
      !write(100+mynodg,'(a,Z10)') 'DBG: comm_charmm= ', comm_charmm
#     endif

      ! Init
      ! FIXME should have diff seed for each dim or only each exch?
      rnum = random(repseed)
      ! TODO add +1 to myidx?
      myidx = mydims(currdim)
      ! NOTE: calculate rep_is_odd based off dimension index to be consistent
      !       with previous 2D rep calc.
      rep_is_odd = (mod(myidx,2).eq.1)

      ! Determine who the neighbor is. Will either be an index or
      ! one of the nbrtype_X variables.
!     write(100+mynodg,'(4(a,i4))') 'DBG: rep ', irepdstr, ', myidx= ', myidx, ', nbrup= ', neighbor_up(currdim), ' nbrdn= ', neighbor_down(currdim)
      neighbor = nbrtype_none
      exchange_dir = 0
      q_exch_attempted = .false.
      if (mod(nattempts(currdim)+attempts_offset,2) .eq. 1) then
        ! On odd steps: even reps exch up, odd reps exch down
        if (rep_is_odd) then
          exchange_dir = -1 
        else
          exchange_dir = 1 
        endif
      else
        ! On even steps: even reps exch down, odd reps exch up
        if (rep_is_odd) then
          exchange_dir = 1 
        else
          exchange_dir = -1 
        endif
      endif
      ! Set absolute neighbor index and neighbor in this dimension
      if (exchange_dir .eq. 1) then
        neighbor = neighbor_up(currdim)
        if (neighbor.ne.nbrtype_none) nopp_up(currdim) = nopp_up(currdim) + 1
      else
        neighbor = neighbor_down(currdim)
        if (neighbor.ne.nbrtype_none) nopp_down(currdim) = nopp_down(currdim) + 1
      endif
      nbridx = myidx + exchange_dir
      ! NOTE - To be consistent with the previous incarnation of BTMs 2D
      !        replica code, lower absolute replica needs to control
      !        the calc.
      qexch_w_reservoir = .false.
      if (neighbor.eq.nbrtype_rsv0.or.neighbor.eq.nbrtype_rsv1) then
        qcontrol_exchg = .true.
        qexch_w_reservoir = .true.
      else if (ndims.gt.1 .and. neighbor.gt.nbrtype_none) then
        qcontrol_exchg = (irepdstr.lt.neighbor)
      else
        qcontrol_exchg = (mod(irepdstr,2) .eq. 1)
      endif

      ! TODO only when neighbor .ne. -1?
      nattempts(currdim) = nattempts(currdim) + 1

      ! As long as I have a neighbor, exchange will be attempted.
      q_exch_attempted = (neighbor.ne.nbrtype_none)

      ! NOTE: ODD replica controls the calc
#     if KEY_DEBUGREPD==1
      write(100+mynodg, '(2(a,i3,a,l3),2(a,i3))') &
        'DBG: Rep= ', irepdstr, ' I control= ', qcontrol_exchg, &
        ' neighbor= ', neighbor, ' I am master= ', qrepmaster, &
        ' myidx= ', myidx, ' nbridx= ', nbridx
#     endif
      if (qexch_w_reservoir) then
        ! Exchange with reservoir
        call reservoir_exchange(neighbor, x, y, z, wmain, vx, vy, vz, xold, yold, zold, &
                                ididphrex, &
                                myepot, ttemp, istep, jhstrt, iseed, iasvel, igvopt, qcrys)
      else
        ! Normal exchange
        select case (exchange_type(currdim))
          case (ex_phrex)
            call phrex_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold, &
                                ididphrex, &
                                istep,jhstrt,igvopt,neighbor,qcrys,rnum)
          case (ex_hamiltonian)
            call hamiltonian_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                      ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
          case (ex_sgld)
            call sgld_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                      ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
          case (ex_temperature)
            call temperature_exchange(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                                      ttemp,istep,jhstrt,igvopt,neighbor,qcrys,rnum)
        end select
      endif
    endif ! exchanging this dimension
  enddo ! Loop over dimensions

end subroutine exchange_calc_ndims

!*******************************************************************************
!> Exchange calc entry.
!<
subroutine exchange_calc(x,y,z,wmain,vx,vy,vz,xold,yold,zold,myepot,&
                         ttemp,temnew,istart,jhstrt,iseed,iasvel,igvopt &
#                        if KEY_TSM==1
                         ,BACKLS &
#                        endif
                         ,IDIDPHREX)
  ! USE STATEMENTS
  use mpi
  use repdstr,  only: comm_rpg, qrexchgl
  use stream,   only: prnlev, outu
# if KEY_DOMDEC==1
  use domdec_common,   only: q_domdec, q_split
  use domdec_d2d_comm, only: copy_to_all
  use domdec_d2r_comm, only: send_stop_recip
  use domdec_dr_common,only: copy_to_recip, q_direct_node, q_recip_node,&
                             start_split_direct_recip, stop_split_direct_recip
# endif
# if KEY_OPENMM==1
  use omm_main, only : omm_change_lambda
  use energym,  only : eterm, DIHE
# endif
  ! ARGUMENTS
  implicit none
  real(chm_real), dimension(:) :: x, y, z
  real(chm_real), dimension(*) :: wmain, vx, vy, vz, xold, yold, zold
  real(chm_real)               :: myepot, ttemp, temnew
  integer                      :: istart, jhstrt,iseed,iasvel,igvopt
  integer, intent(out)         :: ididphrex
# if KEY_TSM==1
  integer, dimension(*)        :: backls
# endif
  ! LOCAL VARIABLES
  integer        :: istep             ! Actual step
  integer        :: rpt               ! Current repeat
# if KEY_DOMDEC==1
  logical,save   :: q_repd_domdec_split ! True if using DOMDEC with direct/recip split
# endif
  integer ierr 

  repd_update_temps = .false.
# if KEY_DOMDEC==1
  q_repd_domdec_split = .false.
  if (q_domdec.and.q_split) then
    q_repd_domdec_split = .true.
    ! Right now any pure recip nodes are waiting inside an energy_recip
    ! call invoked from DCNTRL. Tell the pure recip nodes we need them,
    ! otherwise we will not be able to update coordinates.
    if (q_direct_node .or. .not.q_recip_node) then
      call send_stop_recip
    endif
    call stop_split_direct_recip
    ! Wait for everyone
    call mpi_barrier( comm_universe, ierr )
    ! Since recip nodes have been doing nothing they will not have the 
    ! correct step. They will need to to determine whether exchange
    ! occurs below *and* whether REPD is over in DCNTRL.
    ! NOTE: Assuming rank 0 is NEVER a pure recip node!
    call mpi_bcast(istart, 1, mpi_integer, 0, comm_rpg, ierr)
  endif
# endif
  if (qrepmaster.and.prnlev.gt.2) write(outu,'(a)') 'REPD> Using new exchange calculation.'

  if (qfastrepdstr) then
    ! "Fast" replica exchange.
#   if KEY_OPENMM==1
    if (qtor_repex) then
      call fastrepexchg(wmain, eterm(DIHE), istart, jhstrt, igvopt, vx, vy, vz, xold, yold, zold &
#                       if KEY_TSM==1
                        , backls &
#                       endif
                       )
      call omm_change_lambda(torsion_lambda)
    else
#   endif
      call fastrepexchg(wmain, myepot, istart, jhstrt, igvopt, vx, vy, vz, xold, yold, zold &
#                       if KEY_TSM==1
                        , backls &
#                       endif
                       )
#   if KEY_OPENMM==1
    endif
#   endif
    ! NOTE: Only really need to update temps if exchange happened, but do it
    !       each time "fast" exchange performed to be consistent with previous
    !       behavior.
    repd_update_temps = .true.
  else
    ! "Normal" replica exchange.
    istep = istart - 1

    ! Loop over repeats
    do rpt = 1, nrepeat
      if (prnlev.gt.2.and.nrepeat.gt.1) &
        write(outu,'(a,i8)') 'REPD> Exchange repeat ', rpt
      if (.not.q_use_new_exchange) then
        ! "Old" exchange calc - currently only for TIGER
        call repexchg(x, y, z, wmain, vx, vy, vz, xold, yold, zold, myepot, ttemp, &
                      istart, jhstrt, iseed, iasvel, igvopt &
#                      if KEY_TSM==1
                       , backls &
#                      endif
                       , ididphrex)
      else if (qrexchgl) then
        ! Lambda Hamiltonian exchange
        call repexchgl(x, y, z, wmain, vx, vy, vz, xold, yold, zold, myepot, temnew, &
                       istart, iseed, iasvel, igvopt, jhstrt &
#                      if KEY_TSM==1
                       , backls &
#                      endif
                      )
      else
        ! All other cases
        call exchange_calc_ndims(x, y, z, wmain, vx, vy, vz, xold, yold, zold, myepot, &
                                 ttemp, istep, jhstrt, iseed, iasvel, igvopt, ididphrex)
      endif
    enddo ! END loop over repeats
  endif

  call write_post_swap_crds(istep, x, y, z)
# if KEY_DEBUGREPD==1
  flush(100+mynodg)
# endif
  ! NOTE calls to upimag0, nbonds, and energy originally always happened
  ! here even when exchange failed
# if KEY_DOMDEC==1
  ! Restart split if needed for DOMDEC
  ! TODO If DOMDEC has not been re-initialized, original direct/recip split 
  !      should be valid. How best to check this?
  ! TODO Should split never be stopped if no dimension exchanging?
  !      Would need to make certain recip nodes are stopped in the event
  !      an exchange does not happen.
  if (q_repd_domdec_split) call start_split_direct_recip()
# endif
end subroutine exchange_calc

! ==============================================================================

!*******************************************************************************
!> Original replica exchange calculation. 
!<
  SUBROUTINE REPEXCHG(X,Y,Z,WMAIN,VX,VY,VZ,XOLD,YOLD,ZOLD,MYEPOT,TTEMP,ISTART,JHSTRT, &
                      ISEED,IASVEL,IGVOPT &
#if KEY_TSM==1
                      ,BACKLS &
#endif
                      ,IDIDPHREX &
                     )

    !-----------------------------------------------------------------------
    !     Perform necessary communication and calculate new data...
    !
  use number
  use consta
  use comand
  use dimens_fcm
  use psf
  use stream
  use parallel
  use mpi
  use repdstr
  use memory
  use phmd !JAW
  use bases_fcm
  use deriv
  use image
  use energym, only: energy,eprop,eterm,lenent,epot
  use clcg_mod,only: random
  use cnst_fcm,only: fbeta
  use reawri,only: delta
  use imgup,only: upimag, upimag0
  use sgld,only: SGFFI,PSGLDG,EPOTLF,EPOTHF, &
        SGX1,SGY1,SGZ1,SGX2,SGY2,SGZ2,SGVX,SGVY,SGVZ,SGKX,SGKY,SGKZ,QSGLD,QSGMD
  use consph,only: tstate
  use block_ltm
  use lambdam    !GG: MSLD-compatibility
  use param_store, only: set_param

  implicit none

    real(chm_real) :: X(:), Y(:), Z(:),MYEPOT,VX(*),VY(*),VZ(*),WMAIN(*),TTEMP
    real(chm_real) XOLD(*),YOLD(*),ZOLD(*)

    INTEGER ISTART,JHSTRT,ISEED,IASVEL
#if KEY_TSM==1
    INTEGER BACKLS(*) 
#endif

    !
    LOGICAL QEXC,QCRYS,LUSED,LEXATTEMPT
    INTEGER STEP,IDECIDE,ME,NEIGHBOR,IGVOPT,I,J,CNEIGHBOR,FNEIGH
    real(chm_real) ENEIGH,SCALED,EXPROB,srate,rn,ttx,ttsgx,tfsgx,sgldarg,hrexarg
    REAL(CHM_REAL),allocatable,dimension(:) :: w,comar,dd,timx,timy,timz
    REAL(CHM_REAL),allocatable,dimension(:) :: oldwmain
    logical qhes, qdidrsvr
    real(chm_real) SGARRAY(10),SGARRAYN(10),oldxtlabc(6)
    real(chm_real) SCALSG,SCALRD,FACT
    real(chm_real) :: myvv2,mykk2,nbrvv2,nbrkk2
    real(chm_real) ecsum,ourpoti,ourpotj,nbrpoti,nbrpotj
    real(chm_real) estnbrlf,estourlf,mu_m,mu_n,mu_mtld,mu_ntld,ourlf,nbrlf
    real(chm_real) elfscale
    real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
#if KEY_PHMD==1
    real(chm_real) l(ntitr) !JAW
#endif
#if KEY_BLOCK==1
    integer K                                      !GG: MSLD-compatibility
    real(chm_real) n(nsitemld,nblock)              !GG: MSLD-compatibility
    real(chm_real) m(nsitemld*nblock)              !GG: MSLD-compatibility
    real(chm_real) THETAVMLDS(nsitemld*nblock)     !GG: MSLD-compatibility
    real(chm_real) THETAMLDS(nsitemld*nblock)      !GG: MSLD-compatibility
    real(chm_real) THETAMLDOLDS(nsitemld*nblock)   !GG: MSLD-compatibility
    real(chm_real) THETAFMLDS(nsitemld*nblock)     !GG: MSLD-compatibility
#endif
    integer n6,ical,oldrep
    integer ierr,status(mpi_status_size)

    integer                           :: iproto, jproto, ididphrex, phresstruct
    real(chm_real)                    :: ph_l, ph_m, ph_delta

    if (q_use_new_exchange) then
      ! Exchange calculation has already been done
      return
    endif

    ididphrex = 0

    lexattempt=.false.
    oldrep=reptag

    ! Initialize some log vars
    ph_l = 0.d0
    ph_m = 0.d0
    iproto = 0
    jproto = 0

    !
    !     When qpxtiger is on we are in a tiger pre-exchange state
    IF(QPXTIGER)THEN
       IF(MOD(ISTART-1,TIGERNEQ).NE.0) RETURN
       qrxtmin=.true.   ! we can do mini first now
    ELSE
       IF(MOD(ISTART-1,IREXFQ).NE.0) RETURN
    ENDIF
    QCRYS = (XTLTYP.NE.'    ')

    !
    !     If we are about to do the exchange in the case of TIGER method
    !     we need some preparation:
    !       1. run number of iteration cycles of
    !          the pair of minimizer and equlibration
    !       2. then try for the exchange
    !
    !
    IF(QRXTIGER)THEN
       IF(QRXTMIN)THEN
          write(comlyn, '(''mini abnr nstep '',i6,'' tolg '',f12.8,'' nprint 100'')') tigernm,tigergr
          comlen=51
          call maincomx(comlyn,comlen,lused)
          qrxtmin=.false.
       ENDIF
       IF(QPXTIGER)THEN
          TIGERITI=TIGERITI+1
          IF(TIGERITI.LT.TIGERIT)THEN
             return  ! maybe
          ELSE
             qpxtiger=.false.
             tigeriti=0
          ENDIF
       ENDIF
    ENDIF
    !
    !     Prepare the variable NEIGHBOR.
    !     NEIGHBOR=-1 if no communication is needed on this process

    STEP=MOD(IREX,2)
    ME=MOD(IREPDSTR,2)

    IF(STEP.EQ.1)THEN
       NEIGHBOR=IREPDSTR+1
       IF(ME.NE.0)NEIGHBOR=IREPDSTR-1
    ELSE
       NEIGHBOR=IREPDSTR-1
       IF(ME.NE.0)NEIGHBOR=IREPDSTR+1
    ENDIF
    IF(NEIGHBOR.GE.NREPDSTR)NEIGHBOR=-1

    if(neighbor >= 0) then
       if(neighbor > irepdstr) then
          noppup=noppup+1
       else
          noppdn=noppdn+1
       endif
    endif

    eneigh=zero                ! printout looks better this way
    ourpoti=zero
    ourpotj=zero
    nbrpoti=zero
    nbrpotj=zero
    qdidrsvr=.false.
    EXPROB=0.d0
    QEXC=.FALSE.
    RN=RANDOM(REPSEED)
    scaled=one
    scalsg=one

    IF(NEIGHBOR.GE.0) THEN

      LEXATTEMPT=.TRUE.
      CNEIGHBOR=NEIGHBOR*NUMNOD
      IF(QRXSGLD.AND.MYNOD.EQ.0)THEN
          ! Let's take care of SGLD before we start mucking
          ! about with the energies in the H-REX code below.

         write(outu,'(a,1x,i8)')   'NEIGH:   ', neighbor
         write(outu,'(a,1x,i8)')   'CNEIGH:  ', cneighbor
         !write(outu,'(a,1x,f8.3)') 'NBRTEMP: ', nbrtemp
         !write(outu,'(a,1x,f8.3)') 'OURTEMP: ', ourtemp

          SGARRAY(1)=MYEPOT
          SGARRAY(2)=EPOTLF-EPOTHF
          SGARRAY(3)=(SGFFI-PSGLDG)/(kboltz*temprx(irepdstr+1))
          SGARRAY(4)=ONE/(kboltz*temprx(irepdstr+1))
          SGARRAY(5)=REPDID
          SGARRAY(6)=EPOTLF
          SGARRAY(7)=EPOTHF
          SGARRAY(8)=temprx(irepdstr+1)
          CALL REPD_RCVSEND(CNEIGHBOR,1,SGARRAYN,8,SGARRAY,8)
          ENEIGH=SGARRAYN(1)

          !! ORIGINAL CRITERIA
          !sgldarg = -(SGARRAY(3)-SGARRAYN(3))*(SGARRAYN(1)-SGARRAY(1))  &
          ! -(SGARRAY(4)-SGARRAYN(4))*(SGARRAYN(2)-SGARRAY(2))  &
          ! -(SGARRAY(3)*SGARRAYN(7)-SGARRAYN(3)*SGARRAY(7))*(SGARRAY(6)-SGARRAYN(6))

          !p=min(one,exp(-(SGARRAY(3)-SGARRAYN(3))*(SGARRAYN(1)-SGARRAY(1))  &
          ! -(SGARRAY(4)-SGARRAYN(4))*(SGARRAYN(2)-SGARRAY(2))))
          
          !if(.not.QRXTHAM) then
          !   sgldarg = -((SGARRAY(3)-SGARRAYN(3))*(SGARRAYN(1)-SGARRAY(1)) &
          !             -(SGARRAY(4)-SGARRAYN(4))*(SGARRAYN(2)-SGARRAY(2)))
          !endif

       ENDIF
      
       IF(QRXTHAM) THEN

          ! ouch -- perform a hamiltonian coordinate swap INSIDE temperature REX
          !
          ! We're doing this up here because we need to perform the test swap and
          ! energy eval BEFORE telling other processors to get lost.
          if(mynod.eq.0) then
             call mpi_sendrecv(myepot,1,mpi_real8,cneighbor,1,eneigh,1,mpi_real8,cneighbor,1,comm_universe,status,ierr)
             ourpoti=myepot
             nbrpotj=eneigh

             !     Perform a test coordinate exchange and calculate the new energies
             call chmalloc('repdstr.src','REPXCHG','W',NATOM,crl=w)
             call chmalloc('repdstr.src','REPXCHG','TIMX',NATOM,crl=timx)
             call chmalloc('repdstr.src','REPXCHG','TIMY',NATOM,crl=timy)
             call chmalloc('repdstr.src','REPXCHG','TIMZ',NATOM,crl=timz)
             call chmalloc('repdstr.src','REPXCHG','OLDWMAIN',NATOM,crl=oldwmain)

             timx(1:natom)=x(1:natom)
             timy(1:natom)=y(1:natom)
             timz(1:natom)=z(1:natom)
             oldwmain(1:natom)=wmain(1:natom)

             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,X,NATOM)
             x(1:natom) = w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,Y,NATOM)
             y(1:natom) = w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,Z,NATOM)
             z(1:natom) = w(1:natom)
             if(qcrys.and.xdim.gt.0) then
                call repd_rcvsend(cneighbor,4,W,6,XTLABC,6)
                XTLABC(1:6) = W(1:6)
             endif
             call mpi_sendrecv(wmain,natom,mpi_real8,cneighbor,11,w,natom,mpi_real8,cneighbor,11,comm_universe,status,ierr)
             wmain(1:natom) = w(1:natom)
          endif

          call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(wmain, natom, MPI_REAL8, 0, comm_charmm, ierr)

          if(qcrys) then
             oldxtlabc(1:6)=xtlabc(1:6)
             call mpi_bcast(xtlabc, 6, MPI_REAL8, 0, comm_charmm, ierr)
             call xtllat(xucell,xtlabc) ! TODO bcast xucell as well?
             call chmalloc('repdstr.src','REPXCHG','TRANSF',3,4,XNSYMM,crl=transf)
             call imfill(transf,.false.)
             call chmdealloc('repdstr.src','REPXCHGL','TRANSF',3,4,XNSYMM,crl=transf)
             call upimag0(x,y,z,wmain,0)
          endif
          call nbonds(x,y,z,bnbnd,bimag)
          call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
          ecsum=zero
          do i=1,lenent
             ecsum=eterm(i)+ecsum
          enddo
          eprop(epot)=ecsum

          if(mynod.eq.0) then

             call mpi_sendrecv(ecsum,1,mpi_real8,cneighbor,1,eneigh,1,mpi_real8,cneighbor,1,comm_universe,status,ierr)
             ourpotj=ecsum
             nbrpoti=eneigh

             ! calculate ham REX argument taking different temperatures
             ! into account, from Sugita, Kitao, and Okamoto, J. Chem.
             ! Phys. 113, 6042 (2000).

             write(outu,'(a,4f12.4)') 'DD> NBRPOTJ,NBRPOTI,OURPOTJ,OURPOTI = ',nbrpotj,nbrpoti,ourpotj,ourpoti
             write(outu,'(a,f11.4)')  'DD> k  = ',kboltz
             write(outu,'(a,f11.4)')  'DD> T (us)    = ',temprx(irepdstr+1)
             write(outu,'(a,f11.4)')  'DD> T (them)  = ',temprx(neighbor+1)
             write(outu,'(a,f11.4)')  'DD> 1/kT (us) = ',(one/(kboltz*temprx(irepdstr+1)))
             write(outu,'(a,f11.4)')  'DD> 1/kT (them) = ',(one/(kboltz*temprx(neighbor+1)))
             !hrexarg=-((one/(kboltz*temprx(irepdstr+1)))*(nbrpotj-nbrpoti) &
             !         -(one/(kboltz*temprx(neighbor+1)))*(ourpotj-ourpoti))
             !write(outu,'(a,f11.4)') 'hrexarg = ',hrexarg

             hrexarg=-((one/(kboltz*temprx(irepdstr+1)))*(ourpotj-ourpoti) &
                      -(one/(kboltz*temprx(neighbor+1)))*(nbrpotj-nbrpoti))

             ! We have no clue whether this will succeed or fail, so for now, just put
             ! things back the way that they were and let the exchange code down
             ! below take care of things.

             x(1:natom)=timx(1:natom)
             y(1:natom)=timy(1:natom)
             z(1:natom)=timz(1:natom)
             wmain(1:natom)=oldwmain(1:natom)

             call chmdealloc('repdstr.src','REPXCHG','W',NATOM,crl=w)
             call chmdealloc('repdstr.src','REPXCHG','TIMX',NATOM,crl=timx)
             call chmdealloc('repdstr.src','REPXCHG','TIMY',NATOM,crl=timy)
             call chmdealloc('repdstr.src','REPXCHG','TIMZ',NATOM,crl=timz)
             call chmdealloc('repdstr.src','REPXCHG','OLDWMAIN',NATOM,crl=oldwmain)

             ! Now that we've calculated the TOTAL energies under both hamiltonians,
             ! we can, with some guesswork and duct tape, write out a proper(?) SGLD
             ! multi-hamiltonian exchange probability.
             if(qrxsgld) then
                ! We need to use a modified exchange potentials with tildas
                ! NOTE: this code is horribly redundant on purpose! I'll optimize it
                ! once the method is working. Right now, I'm optimizing for readability...

                ! note that all the sgarray terms got filled in conveniently for us above

                mu_m=sgarray(4)
                mu_n=sgarrayn(4)
                mu_mtld=sgarray(3)
                mu_ntld=sgarrayn(3)
                ourlf=sgarray(1)
                nbrlf=sgarrayn(1)

                ! These next two terms assume that the ratio of the low frequency
                ! contributions is the same as the ratio of the total energies. This
                ! seems reasonable for now, but we should print these terms out. If
                ! They remain fairly consistent accross the simulation, then we
                ! may be OK. If not, I have my doubts

                !write(outu,'(a,f10.6)') 'REPEXCHG> RATIO OF ENERGY ARRAYS IS ',ourpoti/nbrpotj
                write(outu,'(a,f10.6)') 'REPEXCHG> RATIO OF ENERGY ARRAYS IS ',ourpotj/nbrpotj
                estourlf=(ourpotj/nbrpotj)*nbrlf ! LF if we had run the neighbor's hamiltonian
                estnbrlf=(nbrpoti/ourpoti)*ourlf ! LF if the neighbor had run with our hamiltonian

                sgldarg=-mu_mtld*(estourlf-ourlf)-mu_m*(ourpotj-ourpoti) &
                        -mu_ntld*(estnbrlf-nbrlf)-mu_n*(nbrpoti-nbrpotj)
             endif
          endif
          call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(wmain, natom, MPI_REAL8, 0, comm_charmm, ierr)

          if(qcrys) then
             xtlabc(1:6)=oldxtlabc(1:6)
             call mpi_bcast(xtlabc, 6, MPI_REAL8, 0, comm_charmm, ierr) ! TODO check that this needs to happen
             call xtllat(xucell,xtlabc)
             call chmalloc('repdstr.src','REPXCHG','TRANSF',3,4,XNSYMM,crl=transf)
             call imfill(transf,.false.)
             call chmdealloc('repdstr.src','REPXCHG','TRANSF',3,4,XNSYMM,crl=transf)
             call upimag0(x,y,z,wmain,0)
          endif
          call nbonds(x,y,z,bnbnd,bimag)
          call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
          ecsum=zero
          do i=1,lenent
             ecsum=eterm(i)+ecsum
          enddo
          eprop(epot)=ecsum

       ELSE IF(QRXSGLD) THEN
          ! This is an approximation of the SGLD exchange criteria which is
          ! valid when the two Hamiltonians are identical.

          sgldarg = -((SGARRAY(3)-SGARRAYN(3))*(SGARRAYN(1)-SGARRAY(1)) &
                    -(SGARRAY(4)-SGARRAYN(4))*(SGARRAYN(2)-SGARRAY(2)))

       ENDIF
    ENDIF

    IF (MYNOD.NE.0) GOTO 99

#if KEY_BLOCK==1
    call msld_checkvariables(1) !GG Variables Before MC Exchange
#endif 
    !
    !     FIXME: Maybe for protection we can call psetloc() here
    !            This code relies on the fact that psetloc()
    !            values are in NUMNOD and MYNOD. It is important
    !            that every psetglob() is immediately followed
    !            by psetloc() in any part of the code!
    !
    !

    IF(NEIGHBOR.EQ.-1) THEN
       IRESEXCH=IRESEXCH+1
       IF(QRESERVOIR.AND.MOD(IRESEXCH,RSVRSKIP).EQ.0) THEN

          IF((IREPDSTR.EQ.0).AND.QRESLOW) THEN
             QDIDRSVR=.TRUE.
             LEXATTEMPT=.TRUE.
             NOPPDN=NOPPDN+1
             IF(PRNLEV.GE.6) &
                WRITE(IUNREX,*) 'REPEXCH> BOT REPL WOULD EXCH W/ RESERVOIR'
             IF(QPHREX) THEN
                IDIDPHREX=1 
                IPROTO=0
                DO I=1,NRES
                   IF(TSTATE(I).EQ.1) IPROTO=IPROTO+1
                ENDDO
                CALL RESEXCH_PH(TTEMP,.FALSE.,X,Y,Z,VX,VY,VZ,PHRX(IREPDSTR+1),IPROTO, &
                                ISEED,IASVEL,IGVOPT,EXPROB,RN,JPROTO,PH_M,QEXC,FNEIGH &
#if KEY_TSM==1
                                ,BACKLS &
#endif
                               )
             ELSE
                CALL RESEXCH(.FALSE.,X,Y,Z,VX,VY,VZ,QEXC,TEMPRX(IREPDSTR+1),MYEPOT, &
                             ISEED,IASVEL,IGVOPT,EXPROB,RN,ENEIGH,FNEIGH &
#if KEY_TSM==1
                             ,BACKLS &
#endif
                            )
             ENDIF
             IF(PRNLEV.GE.6) WRITE(IUNREX,'(A,I6,3F12.6,A,l1)') &
                'REPEXCH> PHREX: FNEIGH,ENEIGH,P,RN = ', FNEIGH, ENEIGH, EXPROB, RN, ' SUCCESS = ', QEXC
          ENDIF
          IF((IREPDSTR.EQ.NREPDSTR-1).AND.QRESHIGH) THEN
             NOPPUP=NOPPUP+1
             LEXATTEMPT=.TRUE.
             QDIDRSVR=.TRUE.
             IF(PRNLEV.GE.6) &
                WRITE(IUNREX,*) 'REPEXCH> TOP REPL WOULD EXCH W/ RESERVOIR'
             IF(QPHREX) THEN
                IDIDPHREX=1
                IPROTO=0
                DO I=1,NRES
                   IF(TSTATE(I).EQ.1) IPROTO=IPROTO+1
                ENDDO
                CALL RESEXCH_PH(TTEMP,.TRUE.,X,Y,Z,VX,VY,VZ,PHRX(IREPDSTR+1),IPROTO, & 
                                ISEED,IASVEL,IGVOPT,EXPROB,RN,JPROTO,PH_M,QEXC,FNEIGH &
#if KEY_TSM==1
                                ,BACKLS & 
#endif
                               )
             ELSE
                CALL RESEXCH(.TRUE.,X,Y,Z,VX,VY,VZ,QEXC,TEMPRX(IREPDSTR+1),MYEPOT, &
                             ISEED,IASVEL,IGVOPT,EXPROB,RN,ENEIGH,FNEIGH & 
#if KEY_TSM==1
                             ,BACKLS &
#endif
                            )
             ENDIF

             IF(PRNLEV.GE.6) WRITE(IUNREX,'(A,I6,3F12.6,A,l1)') &
                'TIM DBG> FNEIGH,ENEIGH,P,RN = ', FNEIGH, ENEIGH, EXPROB, RN, ' SUCCESS = ', QEXC
          ENDIF
       ENDIF
       GOTO 99
    ENDIF

    IF(QPHREX) THEN
      ! we are state i and have a pH value of pH_l, our neighbor
      ! is state j and has a pH value of pH_m
      ! count number of prototnated residues (in state 1) and swap with
      ! the neighbor
      ididphrex = 1
      iproto = 0
      do i=1,nres
         if(tstate(i).eq.1) iproto = iproto + 1 
      enddo
      ! TODO: irepdstr > neighbor not controlling exchange, really doesnt need the info
      if (irepdstr.gt.neighbor) then
        call mpi_recv(jproto, 1, MPI_INT, cneighbor, 5, comm_universe, status, ierr)
        call mpi_send(iproto, 1, MPI_INT, cneighbor, 6, comm_universe, ierr)
      else
        call mpi_send(iproto, 1, MPI_INT, cneighbor, 5, comm_universe, ierr)
        call mpi_recv(jproto, 1, MPI_INT, cneighbor, 6, comm_universe, status, ierr)
      endif
      !write(outu,'(a,6i4)') 'DBG1> STEP,IREPDSTR,NEIGHBOR,CNEIGHBOR,IPROTO,JPROTO = ',istart-1,irepdstr,neighbor,cneighbor,iproto,jproto

      ph_l = phrx(irepdstr+1)
      ph_m = phrx(neighbor+1)

      ph_delta = log(10.0)*(ph_m - ph_l)*(iproto - jproto)
      if(ph_delta.le.zero) then
         EXPROB=one
      else
         EXPROB=min(one,exp(-ph_delta))
      endif

      IF((QSGLD.OR.QSGMD).AND.MYNOD.EQ.0)THEN
        ! we should be swapping SGLD stuffs too...
          SGARRAY(1)=MYEPOT
          SGARRAY(2)=EPOTLF-EPOTHF
          SGARRAY(3)=(SGFFI-PSGLDG)/(kboltz*temprx(irepdstr+1))
          SGARRAY(4)=ONE/(kboltz*temprx(irepdstr+1))
          SGARRAY(5)=REPDID
          SGARRAY(6)=EPOTLF
          SGARRAY(7)=EPOTHF
          SGARRAY(8)=temprx(irepdstr+1)
        CALL REPD_RCVSEND(CNEIGHBOR,1,SGARRAYN,8,SGARRAY,8)
        ENEIGH=SGARRAYN(1)
      ENDIF
    ELSE
       IF(QRXSGLD) THEN
          EXPROB=min(one,exp(sgldarg))
       ELSE
          IF(QRXTHAM) THEN
             EXPROB=min(one,exp(hrexarg))
          ELSE
             call mpi_sendrecv(myepot,1,mpi_real8,cneighbor,1,eneigh,1,mpi_real8,cneighbor,1,comm_universe,status,ierr)
             if(temprx(irepdstr+1).eq.temprx(neighbor+1)) then
                EXPROB=ONE
             else
                EXPROB=MIN(ONE,EXP(-(ONE/(KBOLTZ*TEMPRX(IREPDSTR+1)) &
                                    -ONE/(KBOLTZ*TEMPRX(NEIGHBOR+1)))*(ENEIGH-MYEPOT)))
             endif
          ENDIF
       ENDIF

    ENDIF

    !
    !     QEXC would be the result of the probability test...
    !
    QEXC=EXPROB.GT.RN
    !      if(prnlev.ge.2)write(IUNREX,'(a,i5,l5)')
    !     $     'REPEXCHG>me,qexc=',mynodg,qexc
    !
    IF(MOD(IREPDSTR,2).EQ.0)THEN
       CALL MPI_RECV(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,STATUS,IERR)
       CALL MPI_RECV(RN,1,MPI_REAL8,CNEIGHBOR,4,COMM_UNIVERSE,STATUS,IERR)
    ELSE
       CALL MPI_SEND(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,IERR)
       CALL MPI_SEND(RN,1,MPI_REAL8,CNEIGHBOR,4,COMM_UNIVERSE,IERR)
    ENDIF
    !      if(prnlev.ge.2)write(IUNREX,'(a,i5,l5)')
    !     $     'REPEXCHG>me,qexc=',mynodg,qexc
    !
    !     Perform the exchange of coordinates and velocities:
    IF(QEXC) THEN
       call mpi_sendrecv_replace(reptag, 1, MPI_INT, cneighbor, 1000, cneighbor, &
                                 1000, COMM_UNIVERSE, status, ierr)
       if(irepdstr.gt.neighbor) then
          nsucdn=nsucdn+1
       else
          nsucup=nsucup+1
       endif
       !
       call chmalloc('repdstr.src','REPXCHG','W',NATOM,crl=w)
       if(qphrex) then
          scaled=one
       else
          scaled=sqrt(temprx(irepdstr+1)/temprx(neighbor+1))
       endif

       call swap_crd(cneighbor,vx,vy,vz)
       
#if KEY_PHMD==1
       IF(QPHRX)THEN !JAW - if exchange then swap theta and velocities
         CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,PH_THETA,NTITR)
         PH_THETA(1:ntitr) = l(1:ntitr)
         CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,VPH_THETA,NTITR)
         VPH_THETA(1:ntitr) = l(1:ntitr)
         CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,THETAOLD,NTITR)
         THETAOLD(1:ntitr) = l(1:ntitr)
         IF(PHBETA .gt. ZERO) THEN ! Langevin PHMD
            CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,VPHOLD,NTITR)
            VPHOLD(1:ntitr) = l(1:ntitr)
            CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,DPHOLD,NTITR)
            DPHOLD(1:ntitr) = l(1:ntitr)
         ENDIF
       ENDIF !JAW
#endif
#if KEY_BLOCK==1
       IF (QMSPHRX) THEN
          write(outu,'(a)') 'Transferring Theta variables'
          K = nsitemld*nblock                          !GG: Calc total no. of elements in MSLD array
          THETAMLDS = RESHAPE(THETAMLD,SHAPE(M))       !GG: Reshape array for coordinates to 1D array
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDS,K)    !GG: Transfer array
          THETAMLDS(1:K) = M(1:K)
          THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))       !GG: Reshape array back to nD array
          THETAMLDOLDS = RESHAPE(THETAMLDOLD,SHAPE(M)) !GG: Processing previous theta coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDOLDS,K)
          THETAMLDOLDS(1:K) = M(1:K)
          THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
          THETAVMLDS = RESHAPE(THETAVMLD,SHAPE(M))     !GG: Processing current velocity coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAVMLDS,K)
          THETAVMLDS(1:K) = M(1:K)
          THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
          THETAFMLDS = RESHAPE(THETAFMLD,SHAPE(M))     !GG: Processing current force coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAFMLDS,K)
          THETAFMLDS(1:K) = M(1:K)
          THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
       ENDIF
#endif

       call mpi_sendrecv(xold,natom,mpi_real8,cneighbor,2,w,natom,mpi_real8,cneighbor,2,comm_universe,status,ierr)
       xOLD(1:natom)=w(1:natom)
       call mpi_sendrecv(yold,natom,mpi_real8,cneighbor,2,w,natom,mpi_real8,cneighbor,2,comm_universe,status,ierr)
       yOLD(1:natom)=w(1:natom)
       call mpi_sendrecv(zold,natom,mpi_real8,cneighbor,2,w,natom,mpi_real8,cneighbor,2,comm_universe,status,ierr)
       zOLD(1:natom)=w(1:natom)

       IF(QPHREX) THEN
          call repd_rcvsend(cneighbor,2,w,natom,cg,natom)
          cg(1:natom)=w(1:natom)
          call mpi_sendrecv_replace(tstate, nres, MPI_INT, cneighbor, 6, &
                                    cneighbor, 6, COMM_UNIVERSE, status, ierr)

          IF(QSGLD.OR.QSGMD)THEN
             myvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
             mykk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGX1,NATOM)
             sgx1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGY1,NATOM)
             sgy1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGZ1,NATOM)
             sgz1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGX2,NATOM)
             sgx2(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGY2,NATOM)
             sgy2(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGZ2,NATOM)
             sgz2(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVX,NATOM)
             sgvx(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVY,NATOM)
             sgvy(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVZ,NATOM)
             sgvz(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKX,NATOM)
             sgkx(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKY,NATOM)
             sgky(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKZ,NATOM)
             sgkz(1:natom)=w(1:natom)
             nbrvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
             nbrkk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)

             scalsg=1.0d0
             scalrd=1.0d0
             if(nbrvv2>rsmall)scalsg = sqrt(myvv2/nbrvv2)
             if(nbrkk2>rsmall)scalrd = sqrt(mykk2/nbrkk2)
             REPDID=NINT(SGARRAYN(5))
             EPOTLF=SGARRAYN(6)
             EPOTHF=SGARRAYN(7)
             DO I = 1, NATOM
                SGVX(I)=SGVX(I)*SCALSG
                SGVY(I)=SGVY(I)*SCALSG
                SGVZ(I)=SGVZ(I)*SCALSG
                SGKX(I)=SGKX(I)*SCALRD
                SGKY(I)=SGKY(I)*SCALRD
                SGKZ(I)=SGKZ(I)*SCALRD
             ENDDO
          ENDIF
       ENDIF

       IF(QRXSGLD)THEN
             myvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
             mykk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGX1,NATOM)
             sgx1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGY1,NATOM)
             sgy1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGZ1,NATOM)
             sgz1(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGX2,NATOM)
             sgx2(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGY2,NATOM)
             sgy2(1:natom)=w(1:natom)
             CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGZ2,NATOM)
             sgz2(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVX,NATOM)
          sgvx(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVY,NATOM)
          sgvy(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGVZ,NATOM)
          sgvz(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKX,NATOM)
          sgkx(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKY,NATOM)
          sgky(1:natom)=w(1:natom)
          CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,SGKZ,NATOM)
          sgkz(1:natom)=w(1:natom)
             nbrvv2=SUM(sgvx*sgvx)+SUM(sgvy*sgvy)+SUM(sgvz*sgvz)
             nbrkk2=SUM(sgkx*sgkx)+SUM(sgky*sgky)+SUM(sgkz*sgkz)
       ENDIF
       !
       call chmdealloc('repdstr.src','REPXCHG','W',NATOM,crl=w)
       !
       !         if(prnlev.ge.2)write(IUNREX,'(a,i5,2f20.8)')
       !     $ 'REPEXCHG>me,temps=',mynodg,temprx(irepdstr+1),temprx(neighbor+1)

       DO I = 1, NATOM
          VX(I)=VX(I)*SCALED
          VY(I)=VY(I)*SCALED
          VZ(I)=VZ(I)*SCALED
          XOLD(I)=XOLD(I)*SCALED
          YOLD(I)=YOLD(I)*SCALED
          ZOLD(I)=ZOLD(I)*SCALED
       ENDDO

       IF(QRXSGLD)THEN
             scalsg=1.0d0
             scalrd=1.0d0
             if(nbrvv2>rsmall)scalsg = sqrt(myvv2/nbrvv2)
             if(nbrkk2>rsmall)scalrd = sqrt(mykk2/nbrkk2)
             REPDID=NINT(SGARRAYN(5))
             EPOTLF=SGARRAYN(6)
             EPOTHF=SGARRAYN(7)
          DO I = 1, NATOM
             SGVX(I)=SGVX(I)*SCALSG
             SGVY(I)=SGVY(I)*SCALSG
             SGVZ(I)=SGVZ(I)*SCALSG
             SGKX(I)=SGKX(I)*SCALRD
             SGKY(I)=SGKY(I)*SCALRD
             SGKZ(I)=SGKZ(I)*SCALRD
          ENDDO
       ENDIF
#if KEY_PHMD==1
       IF(QPHRX)THEN
          DO I=1,NTITR           
             VPH_THETA(I)=VPH_THETA(I)*SCALED           
          ENDDO
          IF(PHBETA .gt. ZERO)THEN
             VPHOLD(I)=VPHOLD(I)*SCALED
          ENDIF
       ENDIF
#endif
#if KEY_BLOCK==1
       IF (QMSPHRX) THEN
          K = nsitemld*nblock
          DO I=1,K
             !GG: Scale the 1D array directly, since that is the array that will be broadcasted
             THETAVMLDS(I)=THETAVMLDS(I)*SCALED
             THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
          ENDDO
       ENDIF
#endif
       !ENDIF
       !         write(50+mynodg,'(3f20.10)')(x(i),y(i),z(i),i=1,natom)
       !         close(50+mynodg)
       !         call psync
       !         call mpi_finalize(i)
       !         stop
 

    ENDIF !GG: End loop for QEXC = .TRUE.
    !
    !     We need to broadcast the data within the same replica group
    !
99  CONTINUE

    call mpi_bcast(eneigh, 1, MPI_REAL8, 0, comm_charmm, ierr)
    call mpi_bcast(exprob, 1, MPI_REAL8, 0, comm_charmm, ierr)
    call mpi_bcast(rn, 1, MPI_REAL8, 0, comm_charmm, ierr)
    call mpi_bcast(qexc, 1, MPI_LOGICAL, 0, comm_charmm, ierr)

    IF(QPHREX) THEN
       call mpi_bcast(ididphrex, 1, MPI_INTEGER, 0, comm_charmm, ierr)
       call mpi_bcast(iproto, 1, MPI_INTEGER, 0, comm_charmm, ierr)
       call mpi_bcast(jproto, 1, MPI_INTEGER, 0, comm_charmm, ierr)
    ENDIF

    !     but we need these:
    !

    IF(QEXC)THEN
#if KEY_BLOCK==1
       call msld_checkvariables(2) !GG: Before transmission
#endif

       ! reset the dynamics algorithm if there's an exchange
       ! Do we need to set igvopt to 2 since this is what ASSVEL does?
       ! assvel does
       JHSTRT=0
       IGVOPT=2
       call mpi_bcast(igvopt, 1, MPI_INTEGER, 0, comm_charmm, ierr)
       call mpi_bcast(jhstrt, 1, MPI_INTEGER, 0, comm_charmm, ierr)
       call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(vx, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(vy, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(vz, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(xold, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(yold, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(zold, natom, MPI_REAL8, 0, comm_charmm, ierr)

       IF(QRXTHAM) THEN
          call mpi_bcast(wmain, natom, MPI_REAL8, 0, comm_charmm, ierr)
          if(qcrys) then
             xtlabc(1:6)=oldxtlabc(1:6)
             call mpi_bcast(xtlabc, 6, MPI_REAL8, 0, comm_charmm, ierr)
             call xtllat(xucell,xtlabc)
             call chmalloc('repdstr.src','REPXCHG','TRANSF',3,4,XNSYMM,crl=transf)
             call imfill(transf,.false.)
             call chmdealloc('repdstr.src','REPXCHGL','TRANSF',3,4,XNSYMM,crl=transf)
             call upimag0(x,y,z,wmain,0)
          endif
          ! not sure if we need these calls, but I am leaving them in now
          ! to be safe
          call nbonds(x,y,z,bnbnd,bimag)
          call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
       ENDIF

       IF(QRXSGLD) THEN
          call mpi_bcast(sgvx, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(sgvy, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(sgvz, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(sgkx, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(sgky, natom, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(sgkz, natom, MPI_REAL8, 0, comm_charmm, ierr)
       ENDIF

       IF(QPHREX) call mpi_bcast(cg, natom, MPI_REAL8, 0, comm_charmm, ierr)

#if KEY_PHMD==1
       IF(QPHRX)THEN
          call mpi_bcast(ph_theta, ntitr, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(vph_theta, ntitr, MPI_REAL8, 0, comm_charmm, ierr)
          IF(PHBETA .gt. ZERO)THEN
            call mpi_bcast(vphold, ntitr, MPI_REAL8, 0, comm_charmm, ierr)
            call mpi_bcast(dphold, ntitr, MPI_REAL8, 0, comm_charmm, ierr)
          ENDIF
       ENDIF
#endif

#if KEY_BLOCK==1
       IF (QMSPHRX) THEN   !GG: Broadcast swapped data to other replicas
          ! DRR Hopefully above note refers to threads and not actually
          ! replicas, otherwise comm needs to change to comm_universe.
          K = nsitemld*nblock
          call mpi_bcast(thetamlds,    k, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(thetamldolds, k, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(thetavmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
          call mpi_bcast(thetafmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
          IF (MYNOD.EQ.0) GOTO 991
          THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))
          THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
          THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
          THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
991       CONTINUE
       ENDIF
#endif
    ENDIF  !GG: End loop for QEXC = .TRUE.

    !
    !     Reinitialize the stuff for TIGER
    !
    IF(QRXTIGER)THEN
       qpxtiger=.false.
       qrxtmin=.false.
       tigeriti=0
    ENDIF
    !
    ! Printout the results:
    !
    ! on each exchange during dynamics
    irex=irex+1
 
    if(noppup > 0) then
       exratup=float(nsucup)/float(noppup)
       call set_param('EXRUP',exratup)
    endif
    if(noppdn > 0) then
       exratdn=float(nsucdn)/float(noppdn)
       call set_param('EXRDN',exratdn)
    endif
    if((noppup+noppdn.gt.0).and.lexattempt) then
       reprob=(min(EXPROB,one)+((noppup+noppdn-1)*reprob))/float(noppup+noppdn)
       call set_param('REPROB',reprob)
       if(prnlev > 5) then
          write(iunrex,'(a,4i3)') 'REPEXCH> noppup, noppdn, nsucup, nsucdn = ',noppup,noppdn,nsucup,nsucdn
          write(iunrex,'(a,f10.5)') 'REPEXCH> P = ',min(EXPROB,one)
          write(iunrex,'(a,f10.5)') 'REPEXCH> NUMERATOR = ',(min(EXPROB,one)+((noppup+noppdn-1)*reprob))
          write(iunrex,'(a,f10.5)') 'REPEXCH> DENOMINATOR = ',float(noppup+noppdn)
          write(iunrex,'(a,3f6.3)') 'REPEXCH> EXRUP EXRDN REPROB = ',exratup,exratdn,reprob
       endif
    endif

    if(prnlev.ge.2) then
       !
       if(qexc)isuc=isuc+1
       srate=real(isuc)/real(irex)

       if(qphrex) then
          write(iunrex,'(a)') &
               '------------- pH Replica Exchange ------------'
          write(iunrex,'(a,i10,a,i10)') &
               'REX>EXCHANGE = ', irex, '  Step =', istart-1
          write(iunrex,'(a,i5,a,f7.3,a,i5)') &
               'REX>REPL     = ',irepdstr, &
               '  pH = ', phrx(irepdstr+1), &
               ' nproto = ', iproto
          write(iunrex,'(a,i5,a,f7.3,a,i5)') &           
               'REX>NEIGHBOR = ',neighbor, &
               '  pH = ', ph_m,     &
               ' nproto = ', jproto
       else
          ttx=zero
          if(neighbor.ge.0)ttx=temprx(neighbor+1)
          write(iunrex,'(a)') &
               '------------- Replica Exchange ------------'
          write(iunrex,'(a,i10,a,i10)') &
               'REX>EXCHANGE = ', irex, '  Step =', istart-1
          write(iunrex,'(a,i5,a,f7.3,a,f20.8)') &
               'REX>REPL     = ',irepdstr,      &
               '  Temp = ', temprx(irepdstr+1), &
               '  Epot = ', myepot
          write(iunrex,'(a,i5,a,f7.3,a,f20.8)') &
               'REX>NEIGHBOR = ',neighbor,      &
               '  Temp = ', ttx,                &
               '  Epot = ', eneigh
          if(qrxsgld) then
             !write(iunrex,'(a6,i6,1x,f12.2,f12.2,f6.3,f6.3,f6.3,1x,f7.3)') &
             !    'RXSG> ',irex,myepot,eneigh,scaled,scalsg,srate,sgldarg

             write(iunrex,'(a6,i6,1x,i10,i4,i4,i4,f12.2,f12.2,f6.3,f6.3,f6.3,1x,l1)') &
                  'RXSG> ',irex,istart-1,irepdstr,neighbor,repdid,  &
                   myepot,eneigh,scaled,scalsg,srate,qexc

          endif
          if(qrxtham) then
             write(iunrex,'(a6,4f12.2,x,f11.3)') &
                 'THAM> ',nbrpotj,nbrpoti,ourpoti,ourpotj,hrexarg
          endif
       endif
       write(iunrex,'(a,i5,a,i5)') 'REX>ORIGINAL TAG ',oldrep,' NEW TAG ',reptag

       if(qdidrsvr) &
          write(iunrex,'(a,i7)') 'REX>RESERVOIR STRUCT = ', fneigh
       write(iunrex,'(a,f8.5,a,f8.5,a,f7.4,a,l1)') &
            'REX>PROB     = ', EXPROB, ' Rand = ', rn,  &
            ' Tscale = ', scaled,                  &
            ' Success = ', qexc

       !
       ! Printout the summary results if sump flag specified:
       !
       if(qsump) then
          allocate(comar(6))
          comar(1)=scaled
          comar(2)=srate
          comar(3)=temprx(irepdstr+1)
          comar(4)=temprx(neighbor+1)
          comar(5)=ttemp
          comar(6)=myepot
          write(iunrex,'(a,a,a)')'#REXSUM Rep# Tscale', &
               '  Sratio   Temp      NewTemp',          &
               '  CurrTemp       Epot'
          write(iunrex,'(a,i3,2f8.3,3f10.3,f15.4)')'REXSUM> ', &
               irepdstr,(comar(j),j=1,6)
          !
          ! Lets do some extra communication for a nice rexsum> printout
          !
          if(irepdstr.eq.0)then
             do i = 1, nrepdstr-1
                call mpi_recv(comar,6,mpi_real8,i*numnod,5,comm_universe,status,ierr)
                write(iunrex,'(a,i3,2f8.3,3f10.3,f15.4)')'REXSUM> ', &
                     i,(comar(j),j=1,6)
             enddo
          else
             call mpi_send(comar,6,mpi_real8,0,5,comm_universe,ierr)
          endif
          deallocate(comar)
       endif

       if(.not.qrxsgld.and..not.qphrex) then
          write(iunrex,'(a)') &
            '------------- Replica Exchange End --------'
       else if(qphrex) then
          write(iunrex,'(a)') &
            '------------- pH Replica Exchange End --------'
       endif

    endif

    IF(IUNRXC.GT.0.AND.PRNLEV.GT.0) THEN
       WRITE(IUNRXC,'(A,I9,A)') '--- REX STEP ',ISTART-1,' POST-SWAP ATTEMPT CRDS ---'
       DO I=1,NATOM
          WRITE(IUNRXC,'(I5,X,3F9.3)') I,X(I),Y(I),Z(I)
       ENDDO
    ENDIF

    if(qexc) then
       !
       !     Since we exchanged the coordinates we should perform
       !     non-bond update
       !
       IF(QCRYS) CALL UPIMAG0(X,Y,Z,WMAIN,0)
       call nbonds(x,y,z,bnbnd,bimag)
    endif

    ! Leaving an extra call in for now to solve conflict...
    CALL UPIMAG0(X,Y,Z,WMAIN,0)
    call nbonds(x,y,z,bnbnd,bimag)

#if KEY_PHMD==1
    IF(QPHRX)THEN !JAW
      call UpdatePHMD(1, 1, 0, 1)
    ENDIF
#endif

#if KEY_BLOCK==1
    IF (QMSPHRX) THEN
       call msld_swapcoeff_mod     !GG: Toggle msld_setblcoef_fnexp to use thetamold values
       call msld_checkvariables(3) !GG: Variables after Swap, before Call Energy
    ENDIF
#endif
    CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0) !JAW
#if KEY_BLOCK==1
    IF (QMSPHRX) THEN
       call msld_swapcoeff_norm    !GG: Toggle msld_setblcoef_fnexp to use thetamold values
       call msld_checkvariables(4) !GG: Variables after Swap, after Call Energy
    ENDIF
#endif

    RETURN
  END SUBROUTINE REPEXCHG
  !

!*******************************************************************************
!> !GG:  Subroutine Spacer to get bookmarks working correctly
!<
  SUBROUTINE GARRETTSPACER
    RETURN   
  END SUBROUTINE GARRETTSPACER

!*******************************************************************************
!> Replica exchange lambda (Hamiltonian) calculation.
!<
  SUBROUTINE REPEXCHGL(X,Y,Z,WMAIN,VX,VY,VZ,xold,yold,zold,EPTT,TEMNEW, &
       ISTART,ISEED,IASVEL,IGVOPT,JHSTRT &
#if KEY_TSM==1
       ,BACKLS                    &            
#endif
       )

    !-----------------------------------------------------------------------
    !     Perform necessary communication and calculate new data...
    !
  use chm_types
  use number
  use consta
  use dimens_fcm
  use energym
  use deriv
  use image
  use psf
  use stream
  use parallel
  use mpi
  use repdstr
  use memory
  use phmd !JW
  use bases_fcm
  use imgup
#if KEY_GCMC==1
  use gcmc          
#endif
  use clcg_mod,only: random
  use block_ltm
  use lambdam    !GG: MSLD-compatibility
  use sgld,only: SGX1,SGY1,SGZ1,SGX2,SGY2,SGZ2,SGVX,SGVY,SGVZ,SGKX,SGKY,SGKZ  !GG: SGLD-compatibility
  use param_store, only: set_param

    !
    implicit none

    REAL(chm_real) :: X(:), Y(:), Z(:), WMAIN(*),EPTT,VX(*),VY(*),VZ(*)
    real(chm_real), dimension(*) :: xold, yold, zold
    REAL(chm_real) LAMBDA,LAMBDA2,ECSUM, EMTS(LENENT)
    INTEGER ISTART,OLDREP,J
    REAL(chm_real) TEMNEW,RN,TOTVX,TOTVY,TOTVZ
    INTEGER ISEED,IASVEL,IGVOPT,JHSTRT
    INTEGER IERR,STATUS(MPI_STATUS_SIZE)

#if KEY_TSM==1
    INTEGER BACKLS(*)
#endif
    !
    LOGICAL QEXC,QCRYS,LEXATTEMPT
    INTEGER STEP,IDECIDE,ME,MEG,NEIGHBOR,I,CNEIGHBOR,IREPR,IREPL
    REAL(chm_real) ENEIGH,ENEIGHI,ENEIGHJ,EPOTI,EPOTJ,EXPROB
    real(chm_real),allocatable,dimension(:) :: W
#if KEY_PHMD==1
    real(chm_real) :: L(NTitr) !JW
#endif
#if KEY_BLOCK==1
    integer K                                      !GG: MSLD-compatibility
    real(chm_real) n(nsitemld,nblock)              !GG: MSLD-compatibility
    real(chm_real) m(nsitemld*nblock)              !GG: MSLD-compatibility
    real(chm_real) THETAVMLDS(nsitemld*nblock)     !GG: MSLD-compatibility
    real(chm_real) THETAMLDS(nsitemld*nblock)      !GG: MSLD-compatibility
    real(chm_real) THETAMLDOLDS(nsitemld*nblock)   !GG: MSLD-compatibility
    real(chm_real) THETAFMLDS(nsitemld*nblock)     !GG: MSLD-compatibility
    real(chm_real) EDIFF                           !GG: MSLD-compatibility
#endif
    logical,allocatable,dimension(:) :: WL
    real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
    !

    ! Initialization of log output variables.
    QEXC=.FALSE.
    EXPROB=0.d0
    RN=0.d0
    epoti=0.d0
    epotj=0.d0
    eneighi=0.d0
    eneighj=0.d0
    ! Set cneighbor to -1 here to avoid writing to ewritu when neighbor is -1
    cneighbor=-1
#if KEY_DEBUGREPD==1
    WRITE(OUTU,'(A)') 'REPD DEBUG> START LAMBDA EXCH'
    CALL FLUSH(OUTU)

    CALL MPI_BARRIER(COMM_CHARMM,IERR)
    WRITE(OUTU,'(A)') 'REPD DEBUG> PASS INITIAL BARRIER'
    CALL FLUSH(OUTU)
#endif

    IF(MOD(ISTART-1,IREXFQ).NE.0) RETURN
    OLDREP=REPTAG
    LEXATTEMPT=.FALSE.
#if KEY_DEBUGREPD==1
    WRITE(OUTU,'(A,I5)') 'REPD DEBUG> REPTAG = ',reptag
#endif

  ! Ensure coords/velocities properly gathered/distributed before exchange attempt(s)
  call distribute_coords_and_velocities(x, y, z, vx, vy, vz, xold, yold, zold)

    IF(IUNRXC.GT.0.AND.PRNLEV.GT.0) THEN
       WRITE(IUNRXC,'(A,I9,A)') '--- H-REX STEP ',ISTART-1,' PRE-SWAP CRDS ---'
       DO I=1,NATOM
          WRITE(IUNRXC,'(I5,X,3F9.3)') I,X(I),Y(I),Z(I)
       ENDDO
    ENDIF

    !
    !     Prepare the variable NEIGHBOR.
    !     NEIGHBOR=-1 if no communication is needed on this process
    IF((.NOT.QEXPT).AND.(.NOT.QEX2D))THEN !GG: CPHMD^MSLD uses this protocol (EXLM keyword only)
       STEP=MOD(ISTART/IREXFQ,2)          !GG: What is the current MC step
       ME=MOD(IREPDSTR,2)                 !GG: Is the current replica no. even (STEP=0) or odd (STEP=1)?
       IF(STEP.EQ.1)THEN
          NEIGHBOR=IREPDSTR+1
          IF(ME.NE.0)NEIGHBOR=IREPDSTR-1
       ELSE
          NEIGHBOR=IREPDSTR-1
          IF(ME.NE.0)NEIGHBOR=IREPDSTR+1
       ENDIF
    ELSE IF (QEXPT)THEN
       STEP=MOD(ISTART/IREXFQ,4)
       ME=MOD(IREPDSTR,2)
       MEG=MOD(IREPDSTR/NREPT,2)
       IF(STEP.EQ.1)THEN
          NEIGHBOR=IREPDSTR+1
          IF(ME.NE.0)NEIGHBOR=IREPDSTR-1
          IF((ME.EQ.0).AND.(MOD(NEIGHBOR,NREPT).EQ.0))NEIGHBOR=-1
          IF((ME.NE.0).AND.(MOD((NEIGHBOR+1),NREPT).EQ.0))NEIGHBOR=-1
       ELSE IF(STEP.EQ.2)THEN
          NEIGHBOR=IREPDSTR-1
          IF(ME.NE.0)NEIGHBOR=IREPDSTR+1
          IF((ME.NE.0).AND.(MOD(NEIGHBOR,NREPT).EQ.0))NEIGHBOR=-1
          IF((ME.EQ.0).AND.(MOD((NEIGHBOR+1),NREPT).EQ.0))NEIGHBOR=-1
       ELSE IF(STEP.EQ.3)THEN
          IF(MOD(IREPDSTR,NREPT).EQ.0)THEN
             NEIGHBOR=IREPDSTR+NREPT
             IF(MEG.NE.0)NEIGHBOR=IREPDSTR-NREPT
          ELSE
             NEIGHBOR=-1
          ENDIF
       ELSE
          IF(MOD(IREPDSTR,NREPT).EQ.0)THEN
             NEIGHBOR=IREPDSTR-NREPT
             IF(MEG.NE.0)NEIGHBOR=IREPDSTR+NREPT
          ELSE
             NEIGHBOR=-1
          ENDIF
       ENDIF
    ELSE                              !GG: EX2D uses this???
      STEP=MOD(ISTART/IREXFQ,4)       !GG: What is the current MC step
      ME=MOD(IREPDSTR,2)              !GG: Is the current replica even (STEP=0) or odd (STEP=1)?
      MEG=MOD(IREPDSTR/NREPX,2)
      IF(STEP.EQ.1)THEN
         NEIGHBOR=IREPDSTR+1
         IF(ME.NE.0)NEIGHBOR=IREPDSTR-1
         IF((ME.EQ.0).AND.(MOD(NEIGHBOR,NREPX).EQ.0))NEIGHBOR=-1
         IF((ME.NE.0).AND.(MOD((NEIGHBOR+1),NREPX).EQ.0))NEIGHBOR=-1
      ELSE IF(STEP.EQ.2)THEN
         NEIGHBOR=IREPDSTR-1
         IF(ME.NE.0)NEIGHBOR=IREPDSTR+1
         IF((ME.NE.0).AND.(MOD(NEIGHBOR,NREPX).EQ.0))NEIGHBOR=-1
         IF((ME.EQ.0).AND.(MOD((NEIGHBOR+1),NREPX).EQ.0))NEIGHBOR=-1
      ELSE IF(STEP.EQ.3)THEN
         NEIGHBOR=IREPDSTR+NREPX
         IF(MEG.NE.0)NEIGHBOR=IREPDSTR-NREPX
      ELSE
         NEIGHBOR=IREPDSTR-NREPX
         IF(MEG.NE.0)NEIGHBOR=IREPDSTR+NREPX
      ENDIF
    ENDIF
    IF(NEIGHBOR.LT.0)NEIGHBOR=-1
    IF(NEIGHBOR.GE.NREPDSTR)NEIGHBOR=-1
    IF(QEXBK)THEN
       IREPR=IREPDSTR+1
       IREPL=IREPDSTR-1
       IF((IREPR.EQ.IRBK).AND.(NEIGHBOR.EQ.IRBK))NEIGHBOR=-1
       IF((IREPDSTR.EQ.IRBK).AND.(NEIGHBOR.EQ.IREPL))NEIGHBOR=-1
    ENDIF

    IF(NEIGHBOR.EQ.-1) THEN
       IF(QRESERVOIR) THEN
          IF((IREPDSTR.EQ.0).AND.QRESLOW) THEN
             NOPPDN=NOPPDN+1
             LEXATTEMPT=.TRUE.
             CALL RESEXCHL(.FALSE.,X,Y,Z,VX,VY,VZ,TEMNEW,RHTEMP,EPTT, &
                           ISEED,IASVEL,IGVOPT,EXPROB,ISTART-1,JHSTRT &
#if KEY_TSM==1
                           ,BACKLS & 
#endif
                          )
          ENDIF
          IF((IREPDSTR.EQ.NREPDSTR-1).AND.QRESHIGH) THEN
             NOPPUP=NOPPUP+1
             LEXATTEMPT=.TRUE.
             CALL RESEXCHL(.TRUE.,X,Y,Z,VX,VY,VZ,TEMNEW,RLTEMP,EPTT, &
                           ISEED,IASVEL,IGVOPT,EXPROB,ISTART-1,JHSTRT &
#if KEY_TSM==1
                           ,BACKLS &
#endif
                          )
          ENDIF
       ENDIF
       GOTO 222
    ENDIF

    IF(IREXFQ .GE. NWREX)THEN
#if KEY_PHMD==1
       IF(.NOT. QPHRX)THEN
#endif
       !!write(IUNREX,*)'REPEXCHGL>me,irep,istart,step,neighbor=', &
       !!     mynodg,irepdstr,istart-1,step,neighbor
#if KEY_PHMD==1
       ENDIF
#endif
    ENDIF

    QCRYS = (XTLTYP.NE.'    ')
    !
    !     FIXME: Maybe for protection we can call psetloc() here
    CNEIGHBOR=NEIGHBOR*NUMNOD
    IF(prnlev.ge.7) write(outu,'(a5, i2)') 'MYNOD', MYNOD !GG: Prints out current node

    !     Perform a test coordinate exchange and calculate the new energies


    ! BTM, move this ABOVE mynod .ne. 0 part -- maybe the various PHRX and GCMC
    ! stuff ought to be moved into SWAP_CRD??! Does it matter?
    IF(MYNOD.EQ.0) THEN
       call mpi_sendrecv(eptt,1,mpi_real8,cneighbor,1,eneigh,1,mpi_real8,cneighbor,1,comm_universe,status,ierr)
       ENEIGHI = ENEIGH
       EPOTI = EPTT
#if KEY_DEBUGREPD==1
       WRITE(OUTU,'(A,F20.6)') 'REPD DEBUG> OUR ENERGY = ',EPROP(EPOT)
       WRITE(OUTU,'(A,2F20.6)') 'REPD DEBUG> EPOT1(P),EPOT2(Q) = ',ENEIGHI,EPOTI
       CALL FLUSH(OUTU)
#endif
    ENDIF
    !! TIMDO: can this be removed?
    CALL MPI_BCAST(ENEIGHI,1,MPI_REAL8,0,COMM_CHARMM,IERR)
    CALL MPI_BCAST(EPOTI,1,MPI_REAL8,0,COMM_CHARMM,IERR)

    LEXATTEMPT=.TRUE.
    call swap_crd(cneighbor,vx,vy,vz) ! FIXME why are velocities swapped here?
    IF(IUNRXC.GT.0.AND.PRNLEV.GT.0) THEN
       WRITE(IUNRXC,'(A,I9,A)') '--- H-REX STEP ',ISTART-1,' POST-SWAP CRDS ---'
       DO I=1,NATOM
          WRITE(IUNRXC,'(I5,X,3F9.3)') I,X(I),Y(I),Z(I)
       ENDDO
    ENDIF


    ! -----------------------------GG: Start of MYNOD .EQ. 0 processes -----------------------------------------

    IF (MYNOD.NE.0) GOTO 99

#if KEY_BLOCK==1
    call msld_checkvariables(1) !GG Variables Before MC Exchange
#endif
#if KEY_PHMD==1
    IF (QPHRX) THEN ! JAW. EXCHANGE THETA VALUES BEFORE TESTING EXCHANGE  
       CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,PH_THETA,NTITR)
       PH_THETA(1:ntitr) = l(1:ntitr)
    ENDIF
#endif
#if KEY_BLOCK==1
    IF (QMSPHRX) THEN
       write(outu,'(a)') 'Transferring Theta variables'
       K = nsitemld*nblock                          !GG: Calc total no. of elements in MSLD array
       THETAMLDS = RESHAPE(THETAMLD,SHAPE(M))       !GG: Reshape array for coordinates to 1D array
       CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDS,K)    !GG: Transfer array
       THETAMLDS(1:K) = M(1:K)
       THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))       !GG: Reshape array back to nD array
       THETAMLDOLDS = RESHAPE(THETAMLDOLD,SHAPE(M)) !GG: Processing previous theta coordinates
       CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDOLDS,K)
       THETAMLDOLDS(1:K) = M(1:K)
       THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
       THETAVMLDS = RESHAPE(THETAVMLD,SHAPE(M))     !GG: Processing current velocity coordinates
       CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAVMLDS,K)
       THETAVMLDS(1:K) = M(1:K)
       THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
       THETAFMLDS = RESHAPE(THETAFMLD,SHAPE(M))     !GG: Processing current force coordinates
       CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAFMLDS,K)
       THETAFMLDS(1:K) = M(1:K)
       THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
    ENDIF
#endif
#if KEY_GCMC==1
    IF (QGCMC) THEN
       call chmalloc('repdstr.src','REPXCHGL','WL',MAXA,log=wl)
       call mpi_sendrecv(gcmcon,maxa,mpi_logical,cneighbor,9,wl,maxa,mpi_logical,cneighbor,9,comm_universe,status,ierr)
       gcmcon(1:maxa) = WL(1:MAXA)
       call chmdealloc('repdstr.src','REPXCHGL','WL',MAXA,log=wl)
    ENDIF
#endif
    ! -----------------------------GG: Start of MYNOD .EQ. ALL processes -----------------------------------------
99  CONTINUE

#if KEY_BLOCK==1
    call msld_checkvariables(2) !GG: Variables Before Transmission
#endif
#if KEY_BLOCK==1
      IF (QMSPHRX) THEN   !GG: Broadcast swapped data to other replicas
         K = nsitemld*nblock
         call mpi_bcast(thetamlds,    k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetamldolds, k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetavmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetafmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
         IF (MYNOD.EQ.0) GOTO 991
         THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))
         THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
         THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
         THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
991      CONTINUE
      ENDIF
#endif
#if KEY_GCMC==1
    IF (QGCMC) THEN
       call mpi_bcast(gcmcon, maxa, MPI_LOGICAL, 0, comm_charmm, ierr)
    ENDIF
#endif
#if KEY_PHMD==1
    IF(QPHRX)THEN
       CALL UpdatePHMD(1, 1, 0, 1) ! JAW. After Exchanging Coordinates and Lambda it is necesarry to update charges and
                                   ! total system charge so energy calculations are correct. 
    ENDIF
#endif
    !GG No need for BLOCK version of charge update since charges are stored individually
!    CALL NBONDS(X,Y,Z,BNBND,BIMAG)  ! LNBND,BIMAG,LIMAG)
#if KEY_BLOCK==1
    IF (QMSPHRX) THEN
       call msld_swapcoeff_mod      !GG: Toggle msld_setblcoef_fnexp to use thetamold values
       call msld_checkvariables(3) !GG: Variables After Swap, Before Call Energy
    ENDIF
#endif
    !CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    call repd_energy(x, y, z)

    IF (QMSPHRX) THEN
#if KEY_BLOCK==1
       call msld_swapcoeff_norm       !GG: Toggle msld_setblcoef_fnexp to use thetam values
       call msld_checkvariables(4) !GG: Variables  After Swap, After Call Energy
#endif
       !GG: Not summing ETERM(I), since PE from MSLD biases have no individual ETERM(I) value
       IF(prnlev.ge.8) THEN
          write(outu,'(a)') 'Using MSLD-PHREX potential energy calculations'
       ENDIF
    ELSE
       !GG: Jana's implementation of traditional pH-REX CPHMD
       ECSUM=ZERO
       DO I=1,LENENT
          ECSUM=ETERM(I)+ECSUM
       ENDDO
       EPROP(EPOT)=ECSUM
    ENDIF

    IF(MYNOD.EQ.0) THEN
       call mpi_sendrecv(eprop(epot),1,mpi_real8,cneighbor,1,eneigh,1,mpi_real8,cneighbor,1,comm_universe,status,ierr)
       ENEIGHJ = ENEIGH
       EPOTJ = EPROP(EPOT)

#if KEY_DEBUGREPD==1
       WRITE(OUTU,'(A,2F20.6)') 'REPD DEBUG> EPOT1(Q),EPOT2(P) = ',EPROP(EPOT),ENEIGHJ
       CALL FLUSH(OUTU)
#endif
    ENDIF
    CALL MPI_BCAST(ENEIGHJ,1,MPI_REAL8,0,COMM_CHARMM,IERR)
    CALL MPI_BCAST(EPOTJ,1,MPI_REAL8,0,COMM_CHARMM,IERR)

    IF(MYNOD.NE.0) GOTO 999

    !
    IF (QMSPHRX) THEN
       !GG: Somehow TEMNEW is not parsed in correctly in MSLD-CPHMD, using MSLD temp instead
       !GG: I/J here refer to before/after swap
       EXPROB=MIN(ONE,EXP(-(ONE/(KBOLTZ*TBLD))*(ENEIGHJ+EPOTJ-ENEIGHI-EPOTI)))
       !P=ZERO
    ELSE
       EXPROB=MIN(ONE,EXP(-(ONE/(KBOLTZ*TEMNEW))*(ENEIGHJ+EPOTJ-ENEIGHI-EPOTI)))
    ENDIF
    RN=RANDOM(REPSEED)
    QEXC=EXPROB.GT.RN

#if KEY_PHMD==1
    IF (QPHRX) THEN
111 format(a16,a4,i5,a9,i5,a5,i10) 
112 format(a15,f16.4,a15,f16.4)
113 format(a15,f16.4,a15,f16.4)
         IF (QEXC) THEN
            write(iunrex,'(a)') &
            '-------------------------------------------------------------'
            write(iunrex,111) &
            ' PH-REX> ACCEPT ','REP',ME,' NEIGHBOR',CNEIGHBOR,'STEP',ISTART-1
         ELSE 
            write(iunrex,'(a)') &
            '-------------------------------------------------------------'
            write(iunrex,111) &
            ' PH-REX> REJECT ','REP',ME,'NEIGHBOR',CNEIGHBOR,'STEP',ISTART-1
         ENDIF
         write(iunrex,112) ' Ei(REP)=',EPOTI,' Ej(Neighbor)=',ENEIGHI
         write(iunrex,113) ' Ei(Neighbor)=',EPOTJ,'Ej(REP)=',ENEIGHJ       
     ELSEIF ( .NOT. QMSPHRX) THEN !GG: CPHMD^MSLD PH-REX printout is written out later
     ELSE
#endif
1111     format('H-REX> REPL ',I3,' NEIGHBOR ',I3,' STEP ',I8)
1112     format('H-REX> EPOT1(P) ',f12.2,' EPOT1(Q) ',f12.2,' EPOT2(Q) ',f12.2,' EPOT2(P) ',f12.2)
         if(prnlev.gt.0) then
            write(IUNREX,'(A)') '------------------'
            write(IUNREX,1111) irepdstr,neighbor,istart-1
            write(IUNREX,1112) epoti,epotj,eneighi,eneighj
         endif

#if KEY_PHMD==1
    ENDIF
#endif
    !
    !     QEXC would be the result of the probability test...
    IF(QEXPT.OR.QEX2D)THEN
       IF((STEP.EQ.1).OR.(STEP.EQ.2))THEN
          IF(MOD(IREPDSTR,2).EQ.0)THEN
             CALL MPI_RECV(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,STATUS,IERR)
          ELSE
             CALL MPI_SEND(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,IERR)
          ENDIF
       ELSE
          !         ALLOCATE(W(1))
          !         CALL REPD_RCVSEND(CNEIGHBOR,9,W,1,QEXC,1)
          IF(MEG.EQ.0)THEN
             CALL MPI_RECV(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,STATUS,IERR)
          ELSE
             CALL MPI_SEND(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,IERR)
          ENDIF
       ENDIF
    ELSE
       IF(MOD(IREPDSTR,2).EQ.0)THEN
          CALL MPI_RECV(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,STATUS,IERR)
          CALL MPI_RECV(RN,1,MPI_REAL8,CNEIGHBOR,9,COMM_UNIVERSE,STATUS,IERR)
       ELSE
          CALL MPI_SEND(QEXC,1,MPI_LOGICAL,CNEIGHBOR,3,COMM_UNIVERSE,IERR)
          CALL MPI_SEND(RN,1,MPI_REAL8,CNEIGHBOR,9,COMM_UNIVERSE,IERR)
       ENDIF
    ENDIF

    IF(NEIGHBOR.GT.IREPDSTR) THEN
       NOPPUP=NOPPUP+1
       IF(QEXC) NSUCUP=NSUCUP+1
    ELSE
       NOPPDN=NOPPDN+1
       IF(QEXC) NSUCDN=NSUCDN+1
    ENDIF

#if KEY_BLOCK==1
    !GG: CPHMD^MSLD PH-REX printout done after the Call MPI_REC/GSEN(CNEIGHBOR,3,QEXC,4) commands
    !GG  If done before, QEXC will not be consistent since not updated?
    IF (QMSPHRX) THEN
711 format(a16,a4,i5,a9,i5,a5,i10)     !!GG: writes out exchange results
712 format(a15,f16.4,a15,f16.4)
713 format(a15,f16.4,a15,f16.4)
714 format(a8,f18.14)
         IF (QEXC) THEN
            write(outu,'(a)') &
            '------------Replica Exchange (PH-REX) for CPHMD^MSLD------------'
            write(outu,711) &
            ' PH-REXv2> ACCEPT ','REP', IREPDSTR,' NEIGHBOR',NEIGHBOR,'STEP',ISTART-1
         ELSE
            write(outu,'(a)') &
            '------------Replica Exchange (PH-REX) for CPHMD^MSLD------------'
            write(outu,711) &
            ' PH-REXv2> REJECT ','REP', IREPDSTR,' NEIGHBOR',NEIGHBOR,'STEP',ISTART-1
         ENDIF
         write(outu,712) ' Ei(REP)=',EPOTI,' Ej(Neighbor)=',ENEIGHI  !GG: I/J here refers to pH
         write(outu,713) ' Ei(Neighbor)=',EPOTJ,'Ej(REP)=',ENEIGHJ
         write(outu,714) ' Ediff', EDIFF
         write(outu,714) ' Prob', EXPROB
         write(outu,*) ' Success', QEXC
    ENDIF
#endif

    !
    !     If acceptance, keep the test exchange of coordinate;if rejection,
    !     perform the exchange again(go back to the configuration before exchange):
999 CONTINUE
    call mpi_bcast(qexc, 1, mpi_logical, 0, comm_charmm, ierr)
    call mpi_bcast(noppup, 1, mpi_integer, 0, comm_charmm, ierr)
    call mpi_bcast(noppdn, 1, mpi_integer, 0, comm_charmm, ierr)
    call mpi_bcast(nsucup, 1, mpi_integer, 0, comm_charmm, ierr)
    call mpi_bcast(nsucdn, 1, mpi_integer, 0, comm_charmm, ierr)
    call mpi_bcast(exprob, 1, mpi_real8, 0, comm_charmm, ierr)

1113    format('H-REX> PROB ',f6.4,' EXCH ',L1)

    !!if(prnlev.gt.0) then
    !!   write(IUNREX,1113) p,qexc
    !!   call flush(iunrex)
    !!endif

    IF(QEXC)THEN   !GG: ------------------- IF EXCHANGE WAS ACCEPTED -------------------------
       !     exchange velocities - more stable than assigning new velocities

#if KEY_DEBUGREPD==1
       WRITE(OUTU,'(A)') 'REPD DEBUG> EXCH ACCEPT'
       CALL FLUSH(OUTU)
#endif

       ! BTM -- this code is a gory mess, but since we've exchanged velocities, we need
       ! to restart the dynamics algorithm
       JHSTRT=0
       IGVOPT=2
       IF(MYNOD.NE.0) GOTO 888

       !! BTM -- SWAP_CRD also swaps velocities, so doing it again here is wrong ... I think.

       !!call chmalloc('repdstr.src','REPXCHGL','W',NATOM,crl=w)
       !!CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,VX,NATOM)
       !!vx(1:natom)=w(1:natom)
       !!CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,VY,NATOM)
       !!vy(1:natom)=w(1:natom)
       !!CALL REPD_RCVSEND(CNEIGHBOR,2,W,NATOM,VZ,NATOM)
       !!vz(1:natom)=w(1:natom)
       !!call chmdealloc('repdstr.src','REPXCHGL','W',NATOM,crl=w) 

       !BTM FOOBAT
       call mpi_sendrecv_replace(reptag, 1, MPI_INT, cneighbor, 1000, cneighbor, &
                                 1000, COMM_UNIVERSE, status, ierr)
!       write(outu,'(a,2i4)') 'TIM DEBUG> reptag, oldrep = ',reptag,oldrep
#if KEY_PHMD==1
       IF(QPHRX)THEN ! JAW. EXCHANGE THETA VELOCITY TERMS, EXCHANGE WAS ACCEPTED
          CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,VPH_THETA,NTITR)
          VPH_THETA(1:ntitr) = l(1:ntitr)
          CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,THETAOLD,NTITR)
          THETAOLD(1:ntitr) = l(1:ntitr)
          IF(PHBETA .gt. ZERO) THEN ! Langevin PHMD
             CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,VPHOLD,NTITR)
             VPHOLD(1:ntitr) = l(1:ntitr)
             CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,DPHOLD,NTITR)
             DPHOLD(1:ntitr) = l(1:ntitr)
          ENDIF
       ENDIF
#endif
#if KEY_BLOCK==1
       IF (QMSPHRX) THEN
          IF(prnlev.ge.6) THEN
             write(outu,'(a)') "EXCHANGE ACCEPTED, not transferring theta variables back" !GG: All theta variables prev transferred
          ENDIF
       ENDIF
       !
706 format(a2,i4,a5,f20.8)
#endif
888    CONTINUE

       !! see above, this was done in swap_crd
       !!CALL PSND8(VX,NATOM)
       !!CALL PSND8(VY,NATOM)
       !!CALL PSND8(VZ,NATOM)
       call mpi_bcast(reptag, 1, MPI_INTEGER, 0, comm_charmm, ierr)
    ELSE   !GG: ------------------- IF EXCHANGE WAS REJECTED -------------------------

#if KEY_PHMD==1
       !JAW - Exchange was rejected--> we need to reset the non-bond lists. also an energy call resets GB. THIS IS NECESSARY!!! 
       IF(QPHRX)THEN
          CALL UpdatePHMD(1, 1, 0, 1) ! After Exchanging Coordinates and Lambda it is necesarry to update charges and
                                      ! total system charge so energy calculations are correct. JW

          !???cb3 are the next two calls supposed to be inside te if(qphrx?)
          !???btm I believe that they are...
!          CALL NBONDS(X,Y,Z,BNBND,BIMAG)  ! LNBND,BIMAG,LIMAG)
          !CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
          call repd_energy(x, y, z)
       ENDIF
#endif

       !JAW
       CALL SWAP_CRD(CNEIGHBOR,VX,VY,VZ)
       IF (MYNOD.NE.0) GOTO 9999

#if KEY_PHMD==1
       IF(QPHRX)THEN ! JW: SWITCH THETA VALUES BACK, EXCHANGE WAS REJECTED 
          CALL REPD_RCVSEND(CNEIGHBOR,2,L,NTITR,PH_THETA,NTITR)
          PH_THETA(1:ntitr) = l(1:ntitr)
       ENDIF 
#endif
#if KEY_BLOCK==1
       IF (QMSPHRX) THEN
          IF(prnlev.ge.6) THEN
             write(outu,'(a)') "EXCHANGE REJECTED, transferring theta variables back"
          ENDIF
          K = nsitemld*nblock                          !GG: Calc total no. of elements in MSLD array
          THETAMLDS = RESHAPE(THETAMLD,SHAPE(M))       !GG: Reshape array for coordinates to 1D array
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDS,K)    !GG: Transfer array
          THETAMLDS(1:K) = M(1:K)
          THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))       !GG: Reshape array back to nD array
          THETAMLDOLDS = RESHAPE(THETAMLDOLD,SHAPE(M)) !GG: Processing previous theta coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAMLDOLDS,K)
          THETAMLDOLDS(1:K) = M(1:K)
          THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
          THETAVMLDS = RESHAPE(THETAVMLD,SHAPE(M))     !GG: Processing current velocity coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAVMLDS,K)
          THETAVMLDS(1:K) = M(1:K)
          THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
          THETAFMLDS = RESHAPE(THETAFMLD,SHAPE(M))     !GG: Processing current force coordinates
          CALL REPD_RCVSEND(CNEIGHBOR,2,M,K,THETAFMLDS,K)
          THETAFMLDS(1:K) = M(1:K)
          THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
       ENDIF
#endif

#if KEY_GCMC==1
       IF (QGCMC) THEN
          call chmalloc('repdstr.src','REPXCHGL','WL',MAXA,log=wl)
          call mpi_sendrecv(gcmcon,maxa,mpi_logical,cneighbor,9,w,maxa,mpi_logical,cneighbor,9,comm_universe,status,ierr)
          gcmcon(1:maxa) = WL(1:MAXA)
          call chmdealloc('repdstr.src','REPXCHGL','WL',MAXA,log=wl)
       ENDIF
#endif
9999   CONTINUE
#if KEY_BLOCK==1
       call msld_checkvariables(2) !GG: Before transmission
#endif
#if KEY_GCMC==1
       IF (QGCMC) THEN
          call mpi_bcast(gcmcon, maxa, MPI_LOGICAL, 0, comm_charmm, ierr)
       ENDIF
#endif
#if KEY_BLOCK==1
      IF (QMSPHRX) THEN   !GG: Broadcast swapped data to other replicas
         K = nsitemld*nblock
         call mpi_bcast(thetamlds,    k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetamldolds, k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetavmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
         call mpi_bcast(thetafmlds,   k, MPI_REAL8, 0, comm_charmm, ierr)
         IF (MYNOD.EQ.0) GOTO 992
         THETAMLD = RESHAPE(THETAMLDS,SHAPE(N))
         THETAMLDOLD = RESHAPE(THETAMLDOLDS,SHAPE(N))
         THETAVMLD = RESHAPE(THETAVMLDS,SHAPE(N))
         THETAFMLD = RESHAPE(THETAFMLDS,SHAPE(N))
992      CONTINUE
     ENDIF
     IF (QMSPHRX) THEN
         call msld_swapcoeff_mod      !GG: Toggle msld_setblcoef_fnexp to use thetamold values
         call msld_checkvariables(3) !GG: Variables  After Swap, Before Call Energy
     ENDIF
     !CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
     call repd_energy(x, y, z)
     IF (QMSPHRX) THEN
        call msld_swapcoeff_norm      !GG: Toggle msld_setblcoef_fnexp to use thetamold values
        call msld_checkvariables(4) !GG: Variables  After Swap, After Call Energy
     ENDIF
#endif

      !!CALL NBONDS(X,Y,Z,BNBND,bimag)   !LNBND,BIMAG,LIMAG)

    ENDIF

    IF(QCRYS) CALL UPIMAG0(X,Y,Z,WMAIN,0)
!    CALL NBONDS(X,Y,Z,BNBND,bimag)
    !CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    call repd_energy(x, y, z)
#if KEY_BLOCK==1
    IF (QMSPHRX) THEN
       call msld_checkvariables(5) !GG: Variables  After Swap, After MC Exchange
    ENDIF
#endif
222 CONTINUE

1114    format('SUMHREX> ',I8,' NBR ',I2,' ORTAG ',I2,' NWTAG ',I2,' epot1(p) ',F12.2,' epot1(q) ',F12.2, &
               ' epot2(q) ',F12.2,' epot2(p) ',f12.2,' prob ',f5.3,' rn ',f5.3,' exc ',L1)
1115    format('HREX UP> OPPORTUNITIES ',I8,' SUCCESSES ',I8,' RATIO ',F6.4)
1116    format('HREX DN> OPPORTUNITIES ',I8,' SUCCESSES ',I8,' RATIO ',F6.4)
1117    format(I8,x,ES25.16,x,ES25.16,x,ES25.16,x,ES25.16)

    if(noppup > 0) exratup=real(nsucup)/real(noppup)
    if(noppdn > 0) exratdn=real(nsucdn)/real(noppdn)
    if((noppup+noppdn.gt.0).and.lexattempt) then
       reprob=(min(EXPROB,one)+((noppup+noppdn-1)*reprob))/float(noppup+noppdn)
       if(prnlev > 5) then
          write(iunrex,'(a,4i3)') 'REPEXCHL> noppup, noppdn, nsucup, nsucdn = ',noppup,noppdn,nsucup,nsucdn
          write(iunrex,'(a,f10.5)') 'REPEXCHL> P = ',min(EXPROB,one)
          write(iunrex,'(a,3f6.3)') 'REPEXCHL> EXRUP EXRDN REPROB = ',exratup,exratdn,reprob
       endif
    endif

    call set_param('EXRUP',exratup)
    call set_param('EXRDN',exratdn)
    call set_param('REPROB',reprob)

    if(prnlev.gt.0) then
       if(cneighbor > mynodg .and. ewritu > 0) &
          write(EWRITU,1117) istart-1,epoti,eneighj,epotj,eneighi

       write(IUNREX,1114) istart-1,neighbor,oldrep,reptag,epoti,epotj,eneighi,eneighj,EXPROB,rn,qexc
       if(qsump) then
          if(noppup.gt.0) then
             write(IUNREX,1115) noppup,nsucup,exratup
          endif
          if(noppdn.gt.0) then 
             write(IUNREX,1116) noppdn,nsucdn,exratdn
          endif
          if(nsucup.eq.0.and.nsucdn.eq.0) then
             write(IUNREX,'(a)') 'WARNING! NO SUCCESSFUL EXCHANGES'
          else if(nsucup.gt.0) then
             rerat=exratdn/exratup
             if(noppdn.gt.0.and.(rerat.gt.2.0.or.rerat.lt.0.5)) then
                write(IUNREX,'(a)') 'WARNING! UNBALANCED EXCHANGE RATIO; CONSIDER MOVING REPLICA OVERLAPS!'
             endif
          else if(nsucdn.gt.0) then ! defensive programming, FTW!
             rerat=exratup/exratdn
             if(noppup.gt.0.and.(rerat.gt.2.0.or.rerat.lt.0.5)) then
                write(IUNREX,'(a)') 'WARNING! UNBALANCED EXCHANGE RATIO; CONSIDER MOVING REPLICA OVERLAPS!'
             endif
          endif
       endif
    endif
    ! FIXME Should we be restarting dynamics here?
!#if KEY_DOMDEC==1
!    ! unconditionally restart the dynamics
!#if KEY_DEBUGREPD==1
!    write(outu,'(a)') 'REPD DEBUG> DYNAMICS BEING RESTARTED'
!#endif
!    if(q_domdec) then
!       JHSTRT=0
!       IGVOPT=2
!
!        if(q_split) then
!           call start_split_direct_recip()
!           if(q_recip_node) then
!              call energy_recip(x, y, z, dx, dy, dz, .false.)
!              call stop_split_direct_recip()
!              return
!           endif
!        endif
!    endif
!#endif

#if KEY_DEBUGREPD==1
       WRITE(OUTU,'(A)') 'REPD DEBUG> DO FINAL COMM_CHARMM BARRIER!!'
       CALL FLUSH(OUTU)
       CALL MPI_BARRIER(COMM_CHARMM,IERR)

       WRITE(OUTU,'(A)') 'REPD DEBUG> DO FINAL COMM_UNIVERSE BARRIER!!'
       CALL FLUSH(OUTU)
       CALL MPI_BARRIER(COMM_UNIVERSE,IERR)

       WRITE(OUTU,'(A)') 'REPD DEBUG> REPEXCHGL FINISHED!!'
       CALL FLUSH(OUTU)
#endif

    RETURN
  END SUBROUTINE REPEXCHGL

!*******************************************************************************
!> Set the IOLEV information from global values, ie all system.
!<
  SUBROUTINE DREPSETIO(IOL,PRNL,WRNL)

  use parallel
  use repdstr
    !
    INTEGER IOL,PRNL,WRNL
    !
    !
    !??? not used here anymore      IF (IOON.EQ.1) RETURN
    !
    !     this must be here since it is easier to protect it here then
    !     in the calling routines
    !
    IF (.NOT.QREPDSTR) RETURN
    !
    IOON=1
    !
    IOLORIG=IOL
    PRNLORIG=PRNL
    WRNLORIG=WRNL
    !      write(70+mynodg,'(a,5i6)')'DREPSETIO-0>me,meg,iolev=',
    !     $     mynod,mynodg,iol,prnl
    !
    !     There is no problem here with IOLEV, ie always like this
    !     even when QRDQTT=.TRUE., since we deal with this separately
    IF(MYNOD.EQ.0)IOL=1
    !
    !     For QWRQTT=.TRUE. we can deal with it here:
    IF(QWRQTT.AND.(MYNOD.EQ.0))PRNL=5
    IF(QWRQTT.AND.(MYNOD.EQ.0))WRNL=5
    !      
    !      write(70+mynodg,'(a,5i6)')'DREPSETIO-1>me,meg,iolev=',
    !     $     mynod,mynodg,iol,prnl
    !
    RETURN
  END SUBROUTINE DREPSETIO

!*******************************************************************************
!> Restores the IOLEV information from global values, ie all system.
!<
    !     Generalize this too!!!
  SUBROUTINE DREPRESIO(IOL,PRNL,WRNL)

  use parallel
  use repdstr
    !
    INTEGER IOL,PRNL,WRNL
    !
    !! not used anymore:       IF (IOON.EQ.0) RETURN
    IF (.NOT.QREPDSTR) RETURN
    !
    IOON=0
    !      write(50+mynodg,*)'DREPRESIO-0>me,meg,iolev=',
    !     $     mynod,mynodg,iol,prnl
    IOL=IOLORIG
    PRNL=PRNLORIG
    WRNL=WRNLORIG
    !      write(50+mynodg,*)'DREPRESIO-1>me,meg,iolev=',
    !     $     mynod,mynodg,iol,prnl
    !
    RETURN
  END SUBROUTINE DREPRESIO

!*******************************************************************************
!> Reservoir exchange.
!<
  SUBROUTINE RESEXCH(QHIGH,X,Y,Z,VX,VY,VZ,QEXC,CURTMP,CURENE,ISEED,IASVEL,IGVOPT, &
                     P,RN,ENEIGH,FNEIGH &
#if KEY_TSM==1
                     ,BACKLS & 
#endif
                    )
     use psf
     use clcg_mod,only: random
     use stream
     use consta,only: kboltz
     use number,only: one
     use repdstr
     use energym
     use bases_fcm
     use memory
     use dynutil, only: assvel

     LOGICAL, INTENT(IN)             :: QHIGH
     LOGICAL, INTENT(OUT)            :: QEXC
     INTEGER, INTENT(OUT)            :: FNEIGH
     REAL(CHM_REAL), INTENT(OUT)     :: P,RN,ENEIGH
     REAL(CHM_REAL), INTENT(INOUT)   :: X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
     REAL(CHM_REAL), INTENT(INOUT)   :: CURTMP,CURENE
     INTEGER,INTENT(INOUT)           :: ISEED,IASVEL,IGVOPT
#if KEY_TSM==1
     INTEGER,INTENT(INOUT)           :: BACKLS(*) 
#endif

     INTEGER                                 :: UNUM,RESSZ,TRGT,I
     REAL(CHM_REAL)                          :: C,S,NEWE,NEWT,TMP,PCUR,PGEN,DCUR,DGEN

     IF(QHIGH) THEN
        UNUM=RHUNIT
        RESSZ=HIGHRESSZ
     ELSE
        UNUM=RLUNIT
        RESSZ=LOWRESSZ
     ENDIF

     TRGT=CEILING(RANDOM(REPSEED)*RESSZ)
     IF(TRGT.EQ.0) TRGT=1 ! Just in case the RNG returns 0
     FNEIGH=TRGT

     ! Decide if we want to swap with the reservoir
     QEXC = .FALSE.
     IF(QRESBOLTZ) THEN
        IF(QHIGH) THEN
           NEWT=RHTEMP        
           NEWE=RHENER(TRGT)
        ELSE
           NEWT=RLTEMP
           NEWE=RLENER(TRGT)
        ENDIF

        IF(CURTMP.EQ.NEWT) THEN
           P=MIN(ONE,EXP(-(NEWE-CURENE)/(KBOLTZ*CURTMP)))
        ELSE
           P=MIN(ONE,EXP(-(ONE/(KBOLTZ*CURTMP) &
            -ONE/(KBOLTZ*NEWT))*(NEWE-CURENE)))
        ENDIF
        RN=RANDOM(REPSEED)
        QEXC=RN.LT.P
        ENEIGH=NEWE
     ELSE IF(QRESNOBO) THEN
        IF(QHIGH) THEN
           NEWE=RHENER(TRGT)
        ELSE
           NEWE=RLENER(TRGT)
        ENDIF
        P=MIN(ONE,EXP(-(ONE/(KBOLTZ*CURTMP))*(NEWE-CURENE)))
        RN=RANDOM(REPSEED)
        QEXC=RN.LT.P
        ENEIGH=NEWE
     ENDIF

     ! go ahead and swap in the new atomic coordinates, saving
     ! the old coordinates in rescrdx,y,z for possible adding to 
     ! the reservoir
     IF(QEXC) THEN

        IF(QHIGH) THEN
           NSUCUP=NSUCUP+1
        ELSE
           NSUCDN=NSUCDN+1
        ENDIF

        ! actually read the new coordinates from the file
        I=((TRGT-1)*3)+1
        READ(UNUM,REC=I)   RESCRDX
        READ(UNUM,REC=I+1) RESCRDY
        READ(UNUM,REC=I+2) RESCRDZ

        IF(PRNLEV.GE.6) &
          WRITE(IUNREX,'(A,I9,A)') '-- RESEXCH: SWAPPED IN ELT ', TRGT, ' OF THE RESERVOIR --'
        DO I=1,NATOM
           IF(PRNLEV.GE.7) &
              WRITE(IUNREX,'(A,I5,A,3F10.6)') 'RESEXCH> CRD ', I, ' = ', RESCRDX(I), RESCRDY(I), RESCRDZ(I)
           ! X
           TMP=X(I)
           X(I)=RESCRDX(I)
           RESCRDX(I)=TMP
           ! Y
           TMP=Y(I)
           Y(I)=RESCRDY(I)
           RESCRDY(I)=TMP
           ! Z
           TMP=Z(I)
           Z(I)=RESCRDZ(I)
           RESCRDZ(I)=TMP
        ENDDO

        ! Since we've reassigned the coordinates, we need to call assvel to
        ! reassign the velocities
        CALL ASSVEL(CURTMP,X,Y,Z,VX,VY,VZ,AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
                    ,BACKLS & 
#endif
                   )
     ELSE IF(PRNLEV.GE.6) THEN
        WRITE(IUNREX,'(A)') '-- RESEXCH: SWAP WITH RESERVOIR NOT ACCEPTED --'
     ENDIF

  END SUBROUTINE RESEXCH

!*******************************************************************************
!> Reservoir exchange, Hamiltonian.
!<
  SUBROUTINE RESEXCHL(QHIGH,X,Y,Z,VX,VY,VZ,TEMP,RESTEMP,OUREPOT,ISEED,IASVEL,IGVOPT,P,STEP,JHSTRT &
#if KEY_TSM==1
                             ,BACKLS & 
#endif
                     )

     use psf
     use consta
     use number
     use repdstr
     use parallel
     use stream
     use clcg_mod,only: random
     use memory
     use dynutil, only: assvel
     use mpi

     ! passed in variables
     logical, intent(in)          :: qhigh
     integer, intent(in)          :: iseed,iasvel,step
     integer, intent(inout)       :: igvopt,jhstrt,backls(*)
     real(chm_real),intent(in)    :: temp,restemp,ourepot
     real(chm_real),intent(inout) :: x(*),y(*),z(*),vx(*),vy(*),vz(*)
     real(chm_real),intent(out)   :: p

     ! local variables
     integer                                 :: unum,ressz,trgt,i,ierr
     logical                                 :: qexc
     real(chm_real)                          :: resepot,rn

     if(mynod.eq.0) then
        ! OK, we have a problem here. If we're replica I w/ cords Q and the reservoir is replica
        ! J w/ coords Q', it's easy to calculate E_I(Q) and E_I(Q'), but it's not so easy to
        ! calculate E_J(Q) and E_J(Q'), since we don't have an explicit replica to ship the coordinates
        ! off to. One idea is to treat the reservoir as Boltzmann, which is what I've implemented.

        if(qhigh) then
           unum=rhunit
           ressz=highressz
        else
           unum=rlunit   
           ressz=lowressz
        endif
        trgt=ceiling(random(iseed)*ressz)
        if(trgt.eq.0) trgt=1 ! Just in case the RNG returns 0
        if(prnlev.gt.3) write(iunrex,'(a,i6)') 'RESEXCHL> TRGT = ', trgt
        call flush(iunrex)
        if(qhigh) then
           resepot=rhener(trgt)
        else
           resepot=rlener(trgt)
        endif

        ! Decide if we want to swap with the reservoir
        rn=random(repseed)
        if(temp.eq.restemp) then
           p=one
        else
           p=min(one,exp(-(one/(kboltz*temp) &
                -one/(kboltz*restemp))*(ourepot-resepot)))
        endif
        qexc=p.gt.rn

        if(qexc) then
           ! do the swap
           i=(3*trgt)-2

           read(unum,rec=i)   rescrdx
           read(unum,rec=i+1) rescrdy
           read(unum,rec=i+2) rescrdz
           x(1:natom)=rescrdx(1:natom)
           y(1:natom)=rescrdy(1:natom)
           z(1:natom)=rescrdz(1:natom)
        endif
     endif

     call mpi_bcast(qexc, 1, MPI_LOGICAL, 0, comm_charmm, ierr)
     if(qexc) then     
        if(qhigh) then
           nsucup=nsucup+1
        else
           nsucdn=nsucdn+1
        endif

       jhstrt=0
       igvopt=2
       
       call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
       call assvel(temp,x,y,z,vx,vy,vz,amass,iseed,iasvel,igvopt,natom,imove &
#if KEY_TSM==1
                   ,backls & 
#endif
                   )
     endif

     ! almost done -- report output to the user
123  format('H-R-REX> ',i8,' REP ENE ',f10.4,' RES ENE ',f10.4,' PROB ',f6.4,' RN ',f6.4,' SUCCESS? ',L1)
     if(prnlev.gt.3) then
        write(iunrex,123) step,ourepot,resepot,p,rn,qexc
        call flush(iunrex)
     endif

  END SUBROUTINE RESEXCHL

!*******************************************************************************
!> Reservoir exchange, PH 
!<
  SUBROUTINE RESEXCH_PH(CURTMP,QHIGH,X,Y,Z,VX,VY,VZ,OURPH,IPROTO, &
                        ISEED,IASVEL,IGVOPT,P,RN,JPROTO,PH_M,QEXC,RESNUM &
#if KEY_TSM==1
                        ,BACKLS & 
#endif
                       )

     use psf
     use clcg_mod,only: random
     use stream
     use consta,only: kboltz
     use number,only: one
     use consph,only: tstate 
     use repdstr
     use image
     use bases_fcm
     use dynutil, only: assvel

     LOGICAL, INTENT(IN)            :: QHIGH
     LOGICAL, INTENT(OUT)           :: QEXC
     INTEGER, INTENT(IN)            :: IPROTO
     INTEGER, INTENT(OUT)           :: RESNUM,JPROTO
     REAL(CHM_REAL), INTENT(OUT)    :: P,RN,PH_M
     REAL(CHM_REAL), INTENT(INOUT)  :: X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
     INTEGER,INTENT(INOUT)          :: ISEED,IASVEL,IGVOPT
#if KEY_TSM==1
     INTEGER,INTENT(INOUT)          :: BACKLS(*) 
#endif
     REAL(CHM_REAL), INTENT(IN)     :: CURTMP,OURPH

     INTEGER                        :: UNUM,RESSZ,TRGT,I,J,K,PIDX
     REAL(CHM_REAL)                 :: PH_DELTA,TMP
     INTEGER,DIMENSION(NATOM)       :: TMPTSTATE
     INTEGER,DIMENSION(:,:),POINTER :: RESTSTATE

     IF(QHIGH) THEN
        UNUM=RHUNIT
        RESSZ=HIGHRESSZ
        PH_M=RHPH
        RESTSTATE=>RESHTSTATE
     ELSE
        UNUM=RLUNIT
        RESSZ=LOWRESSZ
        PH_M=RLPH
        RESTSTATE=>RESLTSTATE
     ENDIF
     TRGT=CEILING(RANDOM(REPSEED)*RESSZ)
     IF(TRGT.EQ.0) TRGT=1 ! Just in case the RNG returns 0
     RESNUM=TRGT

     ! Decide if we want to swap with the reservoir
     QEXC = .FALSE.

     ! For the Boltzmann case at least, we need to read in the reservoir file
     ! so we can calculate JPROTO.
     I=((TRGT-1)*5)+1
     READ(UNUM,REC=I)   RESCRDX
     READ(UNUM,REC=I+1) RESCRDY
     READ(UNUM,REC=I+2) RESCRDZ
     READ(UNUM,REC=I+3) RESCG
     READ(UNUM,REC=I+4) TMPTSTATE
     JPROTO=0
     DO J=1,NRES
        IF(TMPTSTATE(J).EQ.1) JPROTO=JPROTO+1
     ENDDO
     !!!WRITE(OUTU,'(A,I5)') 'TIM DEBUG> COORDINATES OF RESERVOIR STRUCTURE ',TRGT
     !!!DO J=1,NATOM
     !!!   WRITE(OUTU,'(3F8.3)') RESCRDX(J),RESCRDY(J),RESCRDZ(J)
     !!!ENDDO
!     WRITE(OUTU,'(A,I5)') 'TIM DEBUG> PROTONATION COUNT OF RESERVOIR STRUCT = ', JPROTO
     CALL FLUSH(OUTU)

     IF(QRESBOLTZ) THEN
        PH_DELTA=LOG(10.0)*(PH_M-OURPH)*(IPROTO-JPROTO)
        IF(PH_DELTA.LE.0) THEN
           P=ONE
        ELSE
           P=MIN(ONE,EXP(-PH_DELTA))
        ENDIF
     ELSE IF(QRESNOBO) THEN
        CALL WRNDIE(-3,'<RESEXCH_PH>','NONBOLTZMANN REX NOT IMPLEMENTED FOR PH')
     ENDIF
     RN=RANDOM(REPSEED)
     QEXC=RN.LT.P

     IF(QEXC) THEN
        IF(PRNLEV.GE.6) &
          WRITE(IUNREX,'(A,I9,A)') '-- RESEXCH: SWAP IN ELT ', TRGT, ' OF THE RESERVOIR --'


        IF(QHIGH) THEN
           NSUCUP=NSUCUP+1
        ELSE
           NSUCDN=NSUCDN+1
        ENDIF

        DO I=1,NATOM
           !!WRITE(IUNREX,'(A,I5,A,4XF10.3)') 'RESEXCH> CRD ', I, ' = ', RESCRDX(I), RESCRDY(I), RESCRDZ(I), RESCG(I)
           ! X
           TMP=X(I)
           X(I)=RESCRDX(I)
           RESCRDX(I)=TMP
           ! Y
           TMP=Y(I)
           Y(I)=RESCRDY(I)
           RESCRDY(I)=TMP
           ! Z
           TMP=Z(I)
           Z(I)=RESCRDZ(I)
           RESCRDZ(I)=TMP
           ! CHARGE
           TMP=CG(I)
           CG(I)=RESCG(I)
           RESCG(I)=TMP

           IF(CG(I).NE.RESCG(I)) THEN
              if(ntrans > 0) then
                 do k=natom+1,natim
                    pidx=bimag%IMATTR(k)
                    if(pidx > 0) then
                       cg(k)=cg(i)
                    endif
                 enddo
              endif
           ENDIF
        ENDDO
        ! now we need to set the protonation states of each of the residues
        ! (via the tstate array)
        DO I=1,NRES
           TSTATE(I)=TMPTSTATE(I)
           !!WRITE(IUNREX,'(A,I5,A,I3)') 'RESEXCH> SET TSTATE(', I, ') = ', TSTATE(I)
        ENDDO

        ! Since we've reassigned the coordinates, we need to call assvel to
        ! reassign the velocities
        CALL ASSVEL(CURTMP,X,Y,Z,VX,VY,VZ,AMASS,ISEED,IASVEL,IGVOPT,NATOM,IMOVE &
#if KEY_TSM==1
                    ,BACKLS & 
#endif
                   )
     ENDIF

  END SUBROUTINE RESEXCH_PH

!*******************************************************************************
!> <Enter description here!> 
!<
  SUBROUTINE PRECALCENE(QHIGH,X,Y,Z,QECOR,ECORTEMP)
     use psf
     use number
     use memory
     use stream
     use bases_fcm
     use energym
     use deriv
     use shake
     use consta
     use parallel
     use mpi

     LOGICAL,INTENT(IN)           :: QHIGH,QECOR
     REAL(CHM_REAL),INTENT(IN)    :: ECORTEMP
     REAL(CHM_REAL),INTENT(INOUT) :: X(NATOM),Y(NATOM),Z(NATOM)
     INTEGER                      :: UNUM,RESSZ,I,J,R,ierr
     
     REAL(CHM_REAL), ALLOCATABLE, DIMENSION(:) :: TMPX, TMPY, TMPZ
     REAL(CHM_REAL)                            :: CORRECTION
     INTEGER                                   :: NDEGF

     IF(QECOR) THEN
        IF(PRNLEV.GT.1) WRITE(OUTU,'(A,F10.4)') 'PRECALCENE> ADDING ENERGY CORRECTION TERM AT TEMP ', ECORTEMP
        CALL GET_ECOR_ADJUST(CORRECTION,ECORTEMP)
        IF(PRNLEV.GT.1) WRITE(OUTU,'(A,F10.4)') 'PRECALCENE> ENERGY CORRECTION = ', CORRECTION
     ELSE
        NDEGF=-1
        CORRECTION=ZERO
     ENDIF

     IF(QHIGH) THEN
        UNUM=RHUNIT
        RESSZ=HIGHRESSZ
        call chmalloc('repdstr.src','PRECALCENE','rhener',RESSZ,crl=RHENER)
     ELSE
        UNUM=RLUNIT
        RESSZ=LOWRESSZ
        call chmalloc('repdstr.src','PRECALCENE','rlener',RESSZ,crl=RLENER)
     ENDIF

     call chmalloc('repdstr.src','PRECALCENE','tmpx',NATOM,crl=TMPX)
     call chmalloc('repdstr.src','PRECALCENE','tmpy',NATOM,crl=TMPY)
     call chmalloc('repdstr.src','PRECALCENE','tmpz',NATOM,crl=TMPZ)

     DO I=1,NATOM
        TMPX(I)=X(I)
        TMPY(I)=Y(I)
        TMPZ(I)=Z(I)
     ENDDO
     DO I=1,RESSZ
        ! NB, we have to read these into rescrd{x,y,z} because the sdcd
        ! stores them as 4 byte floats, not eight byte...

        RESCRDX(:)=ZERO
        RESCRDY(:)=ZERO
        RESCRDZ(:)=ZERO

        IF(MYNOD.EQ.0) THEN
           R=(I*3)-2
           READ(UNUM,REC=R)   RESCRDX
           READ(UNUM,REC=R+1) RESCRDY
           READ(UNUM,REC=R+2) RESCRDZ

           DO J=1,NATOM
              X(J)=RESCRDX(J)
              Y(J)=RESCRDY(J)
              Z(J)=RESCRDZ(J)
           ENDDO
        ENDIF

        ! Update coordinates and non-bond lists
        ! FIXME: We should call UPDATE here instead
        call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
        call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
        CALL NBONDS(X,Y,Z,BNBND,BIMAG)

        ! Get the energy
        ! The potential energy will be stored in the third element
        ! of the ETERM array
        !CALL GETE(X,Y,Z,X,Y,Z,0)
        CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
        IF(PRNLEV.GT.3) WRITE(OUTU,'(A,I6,A,F12.6)') 'PRECALCENE> POTENTIAL OF STRUCT ', &
                                     I, ' = ', EPROP(EPOT)
        !CALL PRINTE(OUTU, EPROP, ETERM, 'ENER', 'ENR', .TRUE., 0, 0, 0, .TRUE.)       

        IF(QECOR) THEN
           IF(PRNLEV.GT.3) WRITE(OUTU,'(A,F12.6)') 'PRECALCENE> CORRECTED ENE = ', EPROP(EPOT)+CORRECTION
           IF(QHIGH) THEN
              RHENER(I)=EPROP(EPOT)+CORRECTION
           ELSE
              RLENER(I)=EPROP(EPOT)+CORRECTION
           ENDIF
        ELSE
           IF(QHIGH) THEN
              RHENER(I)=EPROP(EPOT)
           ELSE
              RLENER(I)=EPROP(EPOT)
           ENDIF
        ENDIF
     ENDDO
     ! Restore original coordinates
     DO I=1,NATOM
        X(I)=TMPX(I)
        Y(I)=TMPY(I)
        Z(I)=TMPZ(I)
     ENDDO
     call chmdealloc('repdstr.src','PRECALCENE','tmpx',NATOM,crl=TMPX)
     call chmdealloc('repdstr.src','PRECALCENE','tmpy',NATOM,crl=TMPY)
     call chmdealloc('repdstr.src','PRECALCENE','tmpz',NATOM,crl=TMPZ)
 
     ! reset the energy and force arrays to whatever they were before
     ! I'm not sure if this matters, but I'm doing it to be safe...
     call mpi_bcast(x, natom, MPI_REAL8, 0, comm_charmm, ierr)
     call mpi_bcast(y, natom, MPI_REAL8, 0, comm_charmm, ierr)
     call mpi_bcast(z, natom, MPI_REAL8, 0, comm_charmm, ierr)
     CALL NBONDS(X,Y,Z,BNBND,BIMAG)
     !CALL GETE(X,Y,Z,X,Y,Z,0)
     CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)

  END SUBROUTINE PRECALCENE

!*******************************************************************************
!> <Enter description here!> 
!<
  SUBROUTINE GET_ENE_VAL(ENEUN,QECOR,TEMP,QHIGH)
     use psf
     use shake
     use consta
     use number
     use memory
     use stream

     integer, intent(in)       :: ENEUN
     logical, intent(in)       :: QHIGH,QECOR
     real(chm_real),intent(in) :: TEMP
     integer                   :: i,sz,ndegf
     real(chm_real)            :: correction

     IF(QECOR) THEN
        IF(PRNLEV.GT.1) WRITE(OUTU,'(A,F10.4)') 'GET_ENE_VAL> ADDING ENERGY CORRECTION TERM AT TEMP ', TEMP
        CALL GET_ECOR_ADJUST(CORRECTION,TEMP)
        IF(PRNLEV.GT.4) WRITE(OUTU,'(A,F10.4)') 'GET_ENE_VAL> ENERGY CORRECTION = ', CORRECTION
     ELSE
        NDEGF=-1
        CORRECTION=ZERO
     ENDIF

     if(qhigh) then
        sz=highressz
        call chmalloc('repdstr.src','GET_ENE_VAL','rhener',sz,crl=RHENER)
     else
        sz=lowressz
        call chmalloc('repdstr.src','GET_ENE_VAL','rlener',sz,crl=RLENER)
     endif
     do i=1,sz
        if(qhigh) then
           read(eneun,'(f9.4)') rhener(i)
           rhener(i)=rhener(i)+correction
           if(prnlev.gt.3) write(outu,'(a,i4,a,f10.4)') 'GET_ENE_VAL> ENERGY OF HIGH RES STRUCT ', i, ' = ', rhener(i)
        else
           read(eneun,'(f9.4)') rlener(i)
           rlener(i)=rlener(i)+correction
           if(prnlev.gt.3) write(outu,'(a,i4,a,f10.4)') 'GET_ENE_VAL> ENERGY OF LOW RES STRUCT ', i, ' = ', rlener(i)
        endif
     enddo

  END SUBROUTINE GET_ENE_VAL

!*******************************************************************************
!> <Enter description here!> 
!<
  SUBROUTINE READ_RES_PROB(UNUM,NTOR,PROB)
     INTEGER,INTENT(IN)                      :: UNUM,NTOR
     REAL(CHM_REAL),DIMENSION(:),INTENT(OUT) :: PROB

     INTEGER                    :: I
     REAL(CHM_REAL)             :: PSUM,T
     CHARACTER(LEN=100)         :: ERR

     PSUM=0.0
     DO I=1,NTOR
        READ(UNUM,'(F8.6)') T
        PROB(I)=T
        PSUM=PSUM+T
     ENDDO
     IF(PSUM.LT.0.9999.OR.PSUM.GT.1.0001) THEN
        WRITE(ERR,'(A,F9.4)') 'SUM OF PROBABILITIES IS ', PSUM
        CALL WRNDIE(-3,'<READ_RES_PROB>',ERR)
     ENDIF

  END SUBROUTINE READ_RES_PROB

!*******************************************************************************
!> <Enter description here!> 
!<
  SUBROUTINE GET_ECOR_ADJUST(C,TEMP)
     USE PSF
     USE SHAKE
     USE CONSTA,ONLY:KBOLTZ
     USE STREAM

     REAL(CHM_REAL),INTENT(OUT) :: C
     REAL(CHM_REAL),INTENT(IN)  :: TEMP
     INTEGER                    :: NDEGF,I

     NDEGF=0 
     DO I=1,NATOM
        IF(IMOVE(I).EQ.0) NDEGF=NDEGF+3
     ENDDO
     IF(QSHAKE) THEN
       DO I=1,NCONST
          IF((IMOVE(SHKAPR(1,I)) == 0).OR.(IMOVE(SHKAPR(2,I)) == 0)) THEN
             NDEGF=NDEGF-1  
          ENDIF
       ENDDO
     ENDIF  
     IF(PRNLEV.GT.1) WRITE(OUTU,'(A,I4)') 'GET_ECOR_ADJUST> NUMBER OF DEGREES OF FREEDOM = ', NDEGF

     IF(TEMP.LE.0) &
        CALL WRNDIE(-3,'<GET_ECOR_ADJUST>','ENERGY CORRECTION REQUESTED WITH INVALID TEMPERATURE')
     C=(KBOLTZ*TEMP*NDEGF)/2.0
     IF(PRNLEV.GT.4) WRITE(OUTU,'(A,F10.4)') 'GET_ENE_VAL> ENERGY CORRECTION = ', C

  END SUBROUTINE GET_ECOR_ADJUST

!*******************************************************************************
!> Swap two arrays of CHM_REAL between this and target rank in comm_universe.
!! @param rank rank in comm_universe to send to/receive from.
!! @param tag unique tag for message.
!! @param rbuff receive buffer - should not overlap with sbuff.
!! @param rblen length of receive buffer.
!! @param sbuff send buffer - should not overlap with rbuff.
!! @param sblen send buffer length.
!<
subroutine repd_rcvsend(rank, tag, rbuff, rblen, sbuff, sblen)
   use mpi
   ! Subroutine variables
   real(chm_real), intent(inout), dimension(:) :: rbuff, sbuff
   integer, intent(in)                         :: rank, tag, rblen, sblen
   ! Local variables
   integer :: ierr, status(mpi_status_size)

   call mpi_sendrecv( sbuff, sblen, MPI_REAL8, rank, tag, rbuff, rblen, &
                      MPI_REAL8, rank, tag, comm_universe, status, ierr )
   if (ierr /= mpi_success) &
      call wrndie(-4, '<REPDSTR>', 'MPI send/receive failed.')
end subroutine repd_rcvsend

#else /* repdstr_main */
!*******************************************************************************
  SUBROUTINE REPDSTRMAIN
    CALL WRNDIE(-1, &
         '<REPDSTR>','REPlica DiSTRibuted code not compiled.')
    RETURN
  END SUBROUTINE REPDSTRmain
#endif /* repdstr_main */

END MODULE REPDSTRMOD
