module quantm
  use chm_kinds
  use dimens_fcm, only : MAXA
  implicit none
  character(len=*),private,parameter :: file_name   ="quantum_ltm.src"
!CHARMM Element source/fcm/quantm.fcm 1.1
  logical :: qmused_quantum = .false. ! this is OK to be outside the KEY...
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 /*quantm_fcm*/
#if KEY_QUANTUM==1 /*quantm_only*/
!
!     Defines the data necessary for a QM calculation on a system.
!         
!  
!     Variable  Index    Purpose 
! 
!     ALPMM              Parameter for MM atoms for interactions with
!                        QM atoms 
!     ATHEAT             Constant: sum of heats of formation of
!                        quantum mechanical atoms 
!     CHARGE             Total charge on QM group of atoms
!     FRACT              ? something to do with partially occupied
!                        orbitals? see MOLDAT? 
!     KEYWRD             What is left after CHARMM commands are passed
!                        from input line (MOPAC keywords) 
!     LENQMI             Length of common block QMDATA without CHARGE
!                        and FRACT 
!     MAXAIM             Maximum number of atoms allowed (including IMAGEs)
!     MAXQM1             Maximum number of QM atoms allowed
!     MXORB1             Maximum number of orbitals allowed
!     N1ST               Indexes first orbital on a particular atom in
!                        orbital list (length NATOM) 
!     NALPHA             Number of electrons with alpha spin
!     NAT                Atomic number of QM atoms
!     NATQM              Number of QM atoms
!     NBETA              Number of electrons with beta spin
!     NCLOSE             Number of doubly-occupied orbitals
!     NELECS             Total number of electrons
!     NLAST              Indexes last orbital on a particular atom in
!                        orbital list (length NATOM) 
!     NMIDLE             Index of first d orbital for a particular
!                        atom; if no d orbitals, same as NLAST (or NLAST+1?) 
!     NOPEN              Number of electronic orbitals with occupation
!                        level of less than 2 
!     NORBS              Number of occupied orbitals (?) (or total #
!                        basis fns?) 
!     QATLAB             Labels QM atoms: 0=link, -1=MM, +=QM element
!     RHOOMM             Parameter for MM atoms for interactions with
!                        QM atoms 
!     USPD               Diagonal elements for the one-electron matrix
!                        (constants) 
!     QEQTRM             Array of flags for subcomponents of the QMEL
!                        energy term 
!     CEQTRM             Names of the various subcomponents of the
!                        QMEL energy term 
!     LENQEQ             Number of subcomponents of the QMEL energy term
!     QMCUTF             Flag indicating whether to use standard nonbond
!                        cutoffs or to interact with all MM atoms (as with
!                        GAMESS/CADPAC)
!     NEWGS              Number of steps to generate new initial guess
!                        during dynamics (Currently, not compatable with GHO)
!                        n > 0: number of steps
!                          <=0: do not generate initial guess again (default)
!     NSTPMD             The step number of dynamics used with NEWGS.
!  JG 5/2002   
!     QTOTAL             Atomic Charge on a QM atom at each step of a MD run
!     QTOT2              Summ of the atomic charge QTOTAL along a MD run
!     QTOTALG            Atomic Charge on a QM atom in gas phase (deco 
!                         calculation)
!     QTOT2G             Summ of the atomic charge on a QM atom in gas phase
!     CHDYn              To average atomic QM charges during MD.
!                          
      integer, parameter :: MAXQM1 = 150, MXORB1 = 400, &   ! MAXQM1 = 100, MXORB1 = 300,
                            LENQMI = 4*MAXQM1+7, LENQEQ=15
!
      logical, save :: QMCUTF                                             ! COMMON /QMDATA/ (logical)
      real(chm_real),save :: FRACT, CHARGE, &                             ! COMMON /QMDATA/ (real)
                             QTOTAL(MAXQM1),QTOT2(MAXQM1),QTOTALG(MAXQM1), & !JG 5/2002
                             QTOT2G(MAXQM1)  
      integer, save :: NATQM, NAT(MAXQM1), &                              ! COMMON /QMDATA/ (integer)
                       N1ST(MAXQM1),NMIDLE(MAXQM1), NLAST(MAXQM1), &
                       NORBS,NELECS,NALPHA,NBETA,NCLOSE,NOPEN 

      integer, save :: IFRAC                          ! COMMON /FRAC/ IFRAC
      INTEGER, save :: NEWGS,NSTPMD                   ! COMMON /QNEWGS/
      real(chm_real), save :: ATHEAT                  ! COMMON /MTHEAT/
      real(chm_real),save :: USPD(MXORB1)             ! COMMON /NOLORB/
      INTEGER ,allocatable,dimension(:), save :: QATLAB        ! COMMON /QUANTMc/
      character(len=80), save :: KEYWRD               ! COMMON /MEYWRD/
      real(chm_real),save::  ALPMM, RHO0MM, SFT123    ! COMMON /FIXMM/

!
      real(chm_real),dimension(60,6),save :: C_tempm,Z_tempm  ! COMMON /TEMPM/
      integer,                       save :: NPNTS_fit        ! COMMON/FIT/
      real(chm_real),                save :: XLOW_fit,XHIGH_fit, &
                                             XMIN_fit,EMIN_fit,DEMIN_fit, &
                                             X_fit(12),F_fit(12),DF_fit(12)

!
      real(chm_real),                save :: CSS1,CSP1,CPPS1,CPPP1,CSS2,CSP2,CPPS2,CPPP2  ! COMMON /ROTDMM/
      real(chm_real),                save :: X_rotdm(3),Y_rotdm(3),Z_rotdm(3)             ! COMMON /ROTDM2?

!     for the qm/mm energy/gradient routine, local scratch arrays.
      real(chm_real), save :: elect_qm,enuclr_qm
      integer, save :: natqm_check=0 ! local check purpose
      real(chm_real),allocatable,dimension(:,:),save :: coorqm
      real(chm_real),allocatable,dimension(:,:),save :: gradqm
#endif /* (quantm_only)*/
      integer, save :: natom_check=0         ! local check purpose.
      real(chm_real),allocatable,dimension(:),save :: dxm_qmmm
      real(chm_real),allocatable,dimension(:),save :: xim,yim,zim  ! for swapxyz_image
#if KEY_QUANTUM==1 /*quantm_only*/

!     this is for working/scratch arrays.
      real(chm_real),allocatable,dimension(:),save :: local_scratch
      integer,       allocatable,dimension(:),save :: local_iscratch

!***********************************************************************
!     COMMON BLOCKS 'OWNED' BY REST OF PROGRAM.
      real(chm_real),                         save :: F03_anal(100),  &     ! COMMON /TWOEM3/
                                                      ALP3_anal(153), &     ! COMMON /ALPHM3/
                                                      BETA3_anal(153)       ! COMMON /BETA3M/
      integer,                                save :: NZTYPE_anal(100), &   ! COMMON /MATYPE/
                                                      MTYPE_anal(10),   &   ! ... for analytical gradient
                                                      LTYPE_anal           
      real(chm_real),allocatable,dimension(:),save :: WJ_anal,WK_anal     ! COMMON /WMATRX/

!     COMMON BLOCKS 'OWNED' BY ANT
      real(chm_real), save :: DS_ant(16),DG_ant(22),DR_ant(100), &          ! COMMON /DERIVM/
                              TDX_ant(3),TDY_ant(3),TDZ_ant(3) 
      real(chm_real), save :: G_ant(22), TX_ant(3),TY_ant(3), TZ_ant(3)     ! COMMON /EXTRAM/


!==== I may not need these..
!====      integer,        save :: IDMY_ant(5),I3N_ant,IX_ant                    ! COMMON /FORCEM/

      real(chm_real), save :: SUMPSV,SUMPSVA,SUMPSVB                        ! COMMON /SUMPSV/

!***********************************************************************

!***********************************************************************
      integer,       save :: ISP_int,IPS_int                                ! COMMON /SETCM/
      real(chm_real),save :: A_int(7),B_int(7),SA_int,SB_int  ! ,FACTOR_int !

!***********************************************************************

!***********************************************************************
!     Moved from sizes_ltm.src
!     for short version use line with nmeci=1, for long version use line with nmeci=10
      integer,parameter :: NMECI=10      ! old, NMECI=1

      integer,dimension(NMECI**2,NMECI),save :: ISPQR_meci
      integer,                          save :: IS_meci,ILOOP_meci,JLOOP_meci
      real(chm_real),dimension(NMECI,NMECI,NMECI,NMECI),save :: XY_meci
      real(chm_real),dimension(NMECI),save   :: OCCA_meci

!***********************************************************************

!***********************************************************************
! These are defined in ENERIN
!   CEQTRM(QQCC) = 'QQCC'
!   CEQTRM(QQEL) = 'QQEL'
!   CEQTRM(QMEE) = 'QMEE'
!   CEQTRM(QMCH) = 'QMCH'
!   CEQTRM(QATH) = 'QATH'
!   CEQTRM(QPT1) = 'QPT1'
!   CEQTRM(QPT2) = 'QPT2'
!   CEQTRM(QVER) = 'QVER'
!   CEQTRM(QSTA) = 'QSTA'
!   CEQTRM(QDIS) = 'QDIS'
!   CEQTRM(QPOL) = 'QPOL'
!   CEQTRM(QGAS) = 'QGAS'
!***********************************************************************
! paramters:
      integer, parameter :: QQCC=1,QQEL=2,QMEE=3,QMCH=4,QATH=5 , &
                            QPT1=6,QPT2=7,QVER=8,QSTA=9,QDIS=10, &
                            QPOL=11,QGAS=12

! QM energy logical arrays
      logical, save :: QEQTRM(LENQEQ),QMPERT,QDECOM,CHDYN   ! COMMON /QEQL/

! QM energy character arrays
      character(len=4), save :: CEQTRM(LENQEQ)              ! COMMON /QEQC/

! QM energy term arrays
      real(chm_real),   save :: EEQTRM(LENQEQ)              ! COMMON /QEQENE/

! QM/MM perturbation temporary integral files 
      integer,          save :: LINQLK                      ! COMMON /QEQPTR/

!!!!     moved to scfblk_ltm.src:
!!!!     real(chm_real),dimension(:),allocatable,save :: H1PERT,H2PERT,H0GAS,PGAS,PENZYM ! COMMON /QEQPTR/

! QM energy accumulation arrays for FEP and decomposition from dynamics
      real(chm_real),   save :: EQPRA(LENQEQ),EQPR2A(LENQEQ), &              ! COMMON /QEQAVE/ 
                                EQPRP(LENQEQ),EQPR2P(LENQEQ), &
                                RLAMB0,RLAMB1,RLAMB2,FACTP1,FACTP2, &
                                ENUGAS,ENUCP1,ENUCP2,TEMPERATURE,BETA_QMMM, &
                                RLAMBF,  &
                                QEQENE
!***********************************************************************

!***********************************************************************
! For the QM/MM electrostatics with Ewald Sum method
!    It is only work with Group based Cutoff option
!     LQMEWD        : Logical to Ewald or PME with QM/MM (Use Mulliken charges)
!     EWMODE        : The mode of Ewald in QM/MM SCF procedure
!                     1: MM from Ewald Sum polarize QM atom in SCF,
!                        and MM atoms in cutoff polarize QM atom based on
!                        M.J.Field et al.
!     MMINB(MAXA)   : The atom discription for the mapping of atom number
!                     '+' : Atoms that are within cutoff (including QM and MM atoms)
!                     '-' : Atoms that are outof cutoff
!     QMINB(MAXQM1) : The QM atom number for Ewald iteration
!
!     IMATTQ(MAXAIM): Mapping index for Images into Primary image and vice versa.
!                     1      ~NATOM : mapped into images (image transformation number)
!                   : <== This has moved to nbndqm_ltm.src
!
!     EMPOT(MAXQM1) : Potential from Ewald Sum on QM atom sites
!     ESLF          : Self-correction term in Ewald potential
! (IF(QMPERT)
!     QCGSC         : Logical to use charge scaling in QMPERT with Ewald
!     MAXCGS        : Maximum number of counter ions to be scaled
!     NUMCGS        : Number of counter ions to be scaled
!     CGSCLE(MAXCGS): Original charges
!     NATMCGS(MAXCGS): Atom number to be scaled
!
! Refer ewald.fcm
!
      integer, parameter :: MAXCGS=10
      
      logical, save :: LQMEWD,QSETUPKQ,QCGSC                             ! COMMON/QMEWAL0/
      logical, save :: QFIRSTD                                           ! COMMON/QMEWAL7/
      integer,allocatable,dimension(:), save :: MMINB
      integer, save :: EWMODE,QMINB(MAXQM1) ! ,IMATTQ(MAXA)     ! COMMON/QMEWAL1/
      real(chm_real),save :: CHAGTOT,CHAG(MAXQM1), &                     ! COMMON/QMEWAL2/
                             EMPOT(MAXQM1),ESLF(MAXQM1),CGSCLE(MAXCGS)  
      integer, save :: TOTKQ,MAXKVQ,KMAXQ,KSQMAXQ, &                     ! COMMON/QMEWAL3/
                       KMAXXQ,KMAXYQ,KMAXZQ, & 
                       OKMAXQ,OKSQMAXQ,OKMAXXQ,OKMAXYQ,OKMAXZQ
!
      real(chm_real),allocatable,dimension(:),save :: PKVECQ             ! COMMON/QMEWAL3/
      real(chm_real),allocatable,dimension(:),save :: PKTABXCQ, &        ! COMMON/QMEWAL4/ 
                                                      PKTABXSQ, &
                                                      PKTABYCQ, &
                                                      PKTABYSQ, &
                                                      PKTABZCQ, &
                                                      PKTABZSQ
!     gradient from Kspace sum
      real(chm_real),allocatable,dimension(:,:),save :: PDXYZ
!
      logical, save :: NOQMIM                                            ! COMMON/QMEWAL5/ (logical)
      integer, save :: NUMCGS,NATMCGS(MAXCGS)                            ! COMMON/QMEWAL5/ (integer)

!
! For background charge correction
!  E_background = half * q_tot^2 * (- pi / kappa*2 / volume)
!  delE_bakcground = E_1_background - E_2_background
!                  = half*(q_tot_1^2-q_tot_2^2)*(- pi / kappa*2 / volume)
!
!     VOLME   : volume of periodic box
!     BCKCHG(3)  : background total charge for
!                  reference/first perturbed/second perturbed system
!     EBKCHG(3)  : back ground charge correction term
!                  half*q_tot^2*(- pi / kappa*2 / volume)
!
      real(chm_real),save :: VOLME,BCKCHG(3),EBKCHG(3)                   ! COMMON/QMEWAL6/
!--------------------------------------------------------------------
!
!
contains
  subroutine quantum_init
    natqm=0
    lqmewd=.false.
    maxkvq=0
    qsetupkq=.false.
    return
  end subroutine quantum_init
  
  subroutine allocate_quantum()
    use memory
    character(len=*),parameter :: routine_name="allocate_quantum"
    logical,save :: q_first=.true.
    ! deallocate first.
    if(.not.q_first) then
       call chmdealloc(file_name,routine_name,'mminb ',maxa,intg=mminb)
       call chmdealloc(file_name,routine_name,'qatlab',maxa,intg=qatlab)
    end if

    ! followed by new allocation.
    call chmalloc(file_name,routine_name,'mminb ',maxa,intg=mminb)
    call chmalloc(file_name,routine_name,'qatlab',maxa,intg=qatlab)

    q_first=.false.
    return
  end subroutine allocate_quantum

#endif /* (quantm_only)*/
#endif /* (quantm_fcm)*/
end module quantm

