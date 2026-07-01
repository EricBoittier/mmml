module block_ltm
  use chm_kinds
  implicit none
  logical :: NOFORC

! Variables moved from block_fcm to break dependency cycles - mkg 2010
#if KEY_BLOCK==0
  logical, parameter :: QBLOCK = .false.
#else /**/
  !     QBLOCK    flag for whether blocks are in use
  !     NBLOCK    number of blocks total (number of interactions
  !                    is equal to (NBLOCK*(NBLOCK+1))/2)
  !     NBLCKCALLS number of times block called, used in OpenMM
  !     QSOFTOMM  softcore potential for use in OpenMM Block calculations
  !     NBSM1     if non-zero, specifies type of nonbond smoothing
  !     NOFORC    if true, no forces are calculated in fast routines
  !     QPRNTV    flag for whether energy file is printed out or not
!  logical :: QBLOCK, NOFORC, QPRNTV
  logical :: QBLOCK, QPRNTV, QSOFT_OMM
  integer :: NBLOCK, NBSM1, NINTER, NBLCKCALLS
  integer :: NREPLICA

  !     IBLCKP    the block number of each atom
  integer, allocatable, dimension(:) :: IBLCKP

  !     BLCOEP    default coefficent for each interaction
  !     BLCOEB, BLCOEA, BLCOED, BLCOEE, BLCOEV, BLCOEVR, BLCOEVA
  !               the individual coefficents for each energy term
  !               (BOND, ANGL/UREY, DIHE/IMPR, ELEC/PELE/EXTE, VDW/PVDW,
  !               VDW-repulsive, VDW-attractive) in each interaction
  real(chm_real), allocatable, dimension(:) :: BLCOEP
  real(chm_real), allocatable, dimension(:) :: BLCOEB, BLCOEA, BLCOED, &
       BLCOEE, BLCOEV, BLCOEVR, BLCOEVA, BLCOEC

  !     VBBOND    bond energy in each block
  !     VBANG     angle energy in each block
  !     VBTORS    proper torsion energy in each block
  !     VBIMPR    improper torsion energy in each block
  !     VBCMAP    cross-term energy in each block
  !     VBGENB    geneneralized born enery in each block
  !     VBELEC    electostatic energy in each block
  !     VBVDW     van der Waals energy in each block
  real(chm_real), allocatable, dimension(:) :: &
       VBBOND, VBANG, VBTORS, VBIMPR, VBGENB, VBELEC, VBVDW
#if KEY_CMAP==1
  real(chm_real), allocatable, dimension(:) :: VBCMAP  
#endif

  !.ab.For HybH. (see AB.J.Comp.Chem2004,25,985-993).
  !     QHYBH     logical variable stating whether HybH is in use.
  !     IHYBH     integer stating energy term assignment
  logical :: QHYBH
  integer :: IHYBH
  real(chm_real), save :: HYBHLB ! Moved from block.src -- Y Huang 2017
contains

  SUBROUTINE LOADL(LAMBDA)
    !-----------------------------------------------------------------------
    !     PROCEDURE LOAD-OLDL & LOAD-NEWL
    use chm_kinds
    use number

    implicit none
    real(chm_real) LAMBDA,RL
    !
    RL=ONE-LAMBDA
    BLCOEP(1) = ONE
    BLCOEB(1) = ONE
    BLCOEA(1) = ONE
    BLCOED(1) = ONE
    BLCOEC(1) = ONE
    BLCOEV(1) = ONE
    BLCOEE(1) = ONE
    BLCOEP(2) = RL
    BLCOEB(2) = RL
    BLCOEA(2) = RL
    BLCOED(2) = RL
    BLCOEC(2) = RL
    BLCOEV(2) = RL
    BLCOEE(2) = RL
    BLCOEP(3) = RL
    BLCOEB(3) = RL
    BLCOEA(3) = RL
    BLCOED(3) = RL
    BLCOEC(3) = RL
    BLCOEV(3) = RL
    BLCOEE(3) = RL
    BLCOEP(4) = LAMBDA
    BLCOEB(4) = LAMBDA
    BLCOEA(4) = LAMBDA
    BLCOED(4) = LAMBDA
    BLCOEC(4) = LAMBDA
    BLCOEV(4) = LAMBDA
    BLCOEE(4) = LAMBDA
    BLCOEP(5) = ZERO
    BLCOEB(5) = ZERO
    BLCOEA(5) = ZERO
    BLCOED(5) = ZERO
    BLCOEC(5) = ZERO
    BLCOEV(5) = ZERO
    BLCOEE(5) = ZERO
    BLCOEP(6) = LAMBDA
    BLCOEB(6) = LAMBDA
    BLCOEA(6) = LAMBDA
    BLCOED(6) = LAMBDA
    BLCOEC(6) = LAMBDA
    BLCOEV(6) = LAMBDA
    BLCOEE(6) = LAMBDA
    
  END subroutine loadl

#endif 
end module block_ltm

