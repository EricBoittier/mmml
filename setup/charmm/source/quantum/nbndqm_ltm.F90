module nbndqm_mod
  use chm_kinds
  use chm_types
  use dimens_fcm

  implicit none
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
  !
  !     The information required to calculate non-bonded interactions is
  !     stored in this data structure. All arrays are stored on the HEAP.
  !     They are defined by an two element integer pointer which gives
  !     the starting address of the array on the heap and the length of
  !     the allocated array.
  !
  !     Description of variables:
  !
  !     --------------
  !     Non-bond lists
  !     --------------
  !
  !     Nnbgrp     The total number of group/group non-bonded interactions
  !                for the molecular mechanics atoms.
  !     Inbgrp     Index array to Jnbgrp giving the number of interactions
  !                per group.
  !     Jnbgrp     List containing the interactions.
  !
  !     Ngrpex     The number of group/group exclusions.
  !     Igrpex     Index array to Jgrpex giving the number of exclusions
  !                per group.
  !     Jgrpex     List containing the exclusions.
  !
  !     Natmex     The number of atom/atom exclusions.
  !     Iatmex     Index array to Jatmex giving the number of exclusions
  !                per atom.
  !     Jatmex     List containing the exclusions.
  !
  !     Nqmgpv     The total number of group/group interactions for calculating
  !                the quantum mechanical/molecular mechanical van der Waal's
  !                energies and forces.
  !     Iqmgpv     Index array to Jqmgpv giving the number of interactions
  !                per group.
  !     Jqmgpv     List containing the interactions.
  !
  !     Nqmgpe     The total number of group/group interactions for calculating
  !                the quantum mechanical/molecular mechanical electrostatic
  !                energies and forces.
  !     Iqmgpe     Index array to Jqmgpe giving the number of interactions
  !                per group.
  !     Jqmgpe     List containing the interactions.
  !
  !     Nqmgex     The total number of group/group exclusions used for
  !                calculating the quantum mechanical/molecular mechanical
  !                van der Waal's energies and forces.
  !     Iqmgex     Index array to Jqmgex giving the number of exclusions
  !                per group.
  !     Jqmgex     List containing the exclusions.
  !     Mnbgrp     The total number of group/group non-bonded interactions
  !                for the molecular mechanics atoms in image.
  !     Imbgrp     Index array to Imjnbg giving the number of interactions
  !                per group.
  !     Jmbgrp     List containing the interactions.
  !
  !     ----------------
  !     Non-bond options
  !     ----------------
  !
  !     Ctofnb     Switching function and shifting turn off cutoff.
  !     Ctonnb     Switching function turn on cutoff.
  !     Cutnb      Electrostatics list cutoff.
  !     E14fac     Factor to multiply 1-4 electrostatic  energies by.
  !     Eps        The dielectric constant.
  !     Nupfrqq    The non-bond update frequency.
  !
  !     QNBGRP     Allowing QM molecule to be partitioned into
  !                groups for construction of QM/MM nonbonded list,
  !                but include interactions of the entire QM molecule
  !                with all unique MM groups within all QM group
  !                cutoff range.   JG 3/21/01
  !
  !     -------------------------
  !     Data structure definition
  !     -------------------------
  !
  !
  integer,save :: nnbgrp, ngrpex, natmex, nqmgpv, nqmgpe, nqmgex, nupfrqq, mnbgrp

  type(chm_iptr),save :: inbgrp, jnbgrp, igrpex, jgrpex, iatmex, jatmex, &
        imbgrp, jmbgrp

  integer,save,pointer,dimension(:) :: iqmgpe,jqmgpe,iqmgex,jqmgex, iqmgpv, jqmgpv

  integer,save :: ngrp_old=0          ! also initializes
  integer,save :: nqmlen_nbnd=0
  integer,save :: natc_old=0
  real(chm_real),save,allocatable,dimension(:,:):: xyzg
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  real(chm_real),save,allocatable,dimension(:,:):: xyzg_sq  
#endif
  real(chm_real),save,allocatable,dimension(:)  :: qg_qmmm
  integer,save,  allocatable,dimension(:)  :: ioff_qmmm
  integer,save,  allocatable,dimension(:)  :: igrpg
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  integer,save,  allocatable,dimension(:)  :: igrpg2 
#endif
  logical,save,  allocatable,dimension(:)  :: qgrpmm,qgrpqm
  logical,save,  allocatable,dimension(:)  :: qatmmm ! mm atom flag
  integer,save,  allocatable,dimension(:)  :: inblos
  integer,save,  allocatable,dimension(:)  :: jnbs
  integer,save,  allocatable,dimension(:)  :: inbloq
  integer,save,  allocatable,dimension(:)  :: jnbq
  integer,save,  allocatable,dimension(:)  :: IMATTQ

  ! size of qm groups
  integer, parameter :: Maxqmg=200

  !
  LOGICAL QNBGRP
#if KEY_MNDO97==1
  logical,save :: q_cut_by_group=.false.
  logical,save :: qmswtch_qmmm  =.false.
  integer,save :: num_mm_group=0
  integer,save :: num_qm_group=0
  integer,save :: num_mmatm_in_list=0
  integer,save,pointer :: map_mmatom_to_group(:)=>Null()   ! map mm atom to its corresponding group in ngrp (igpbs). 
  integer,save,pointer :: map_qmatom_to_group(:)=>Null()   ! map qm atom to its corresponding group in NQMGRP(1).
  integer,save,pointer :: map_allatm_to_group(:)=>Null()   ! map each atom to its group in ngrp (igpbs).
  integer,save,pointer :: map_mmgrp_to_group(:)=>Null()    ! map group in the list to the ngrp.

  ! memory: mapping boxes to each group.
  integer, pointer,save :: i_map_box(:,:)  =>Null()
#endif
  !
#endif 
end module nbndqm_mod

