!************************************************************************
!                              AMBER                                   **
!                                                                      **
!                        Copyright (c) 2003                            **
!                Regents of the University of California               **
!                       All Rights Reserved.                           **
!                                                                      **
!  This software provided pursuant to a license agreement containing   **
!  restrictions on its disclosure, duplication, and use. This software **
!  contains confidential and proprietary information, and may not be   **
!  extracted or distributed, in whole or in part, for any purpose      **
!  whatsoever, without the express written permission of the authors.  **
!  This notice, and the associated author list, must be attached to    **
!  all copies, or extracts, of this software. Any additional           **
!  restrictions set forth in the license agreement also apply to this  **
!  software.                                                           **
!************************************************************************

!+++++++++++++++++++++++++++++++++++++++++++++
!This module contains allocatable arrays
!and variables used for coupled potential
!qmmm calculations. If you need access to one
!of the public variables from module you should
!include it in your code with
!use qmmm_module
!
!Main Authors:
!     Ross Walker
!     Mike Crowley
!+++++++++++++++++++++++++++++++++++++++++++++

      MODULE QMMM_MODULE
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1 /*mainsquatn*/
  use chm_kinds
  use qm2_double
  use qm2_constants
  use qm2_conversions
  use qm2_array_locations
  use qm2_parameters, only:nelements
      implicit none


!!    PRIVATE

      PUBLIC :: GetQMMMType, PutQMMMType,   &
                Get_Type_qmmm_nml, Get_Type_qmmm_struct, &
                Get_Type_qm2_struct, Get_Type_qm2_params, &
                Get_Type_qm2_rij_eqns, Get_Type_qmmm_switch, &
                Get_Type_qm2_ghos, Get_Type_qmmm_ewald, &
                qmsort, adjust_iqmatoms_for_neb, get_atomic_number,   &
                reallocate_iqm_atno_array, allocate_qmmm,   &
                allocate_qmmm_pair_list,   &
                qm2_allocate_params1, qm2_allocate_params2,   &
                qm2_allocate_qmqm_e_repul,   &
                qm2_allocate_qmmm_e_repul,   &
                qm2_allocate_qm2_qmqm_rij_eqns,   &
                qm2_allocate_qm2_qmmm_rij_eqns, deallocate_qmmm


! Note: ***modify also "qmmm_mpi_setup" below if you modify this structure***
! 

      TYPE, PUBLIC :: qmmm_namelist
                                         ! Contains QMMM variables and should be same for all cpus
      logical   :: ifqnt                 ! Flag for whether QMMM is on (True) or off (False)
      real(chm_real)  :: qmcut                 ! Cutoff in angstroms for QM-MM electrostatics
                                         ! Default : same as MM-MM cutoff
      real(chm_real)  :: qmcut2                ! qmcut*qmcut
      real(chm_real)  :: scfconv               ! SCF Convergence criteria
                                         ! = 1.0D-8 (default)
      real(chm_real)  :: pseudo_diag_criteria  ! Criteria for the maximum change in the density matrix
                                         ! between two scf iterations before allowing
                                         ! pseudo diagonalisations (fast diagonalizations ?).
                                         ! = 0.05 (default)
      integer   :: qmtheory              ! Semiempirical QM methods
                                         ! = 1 PM3
                                         ! = 2 AM1 (default)
                                         ! = 3 MNDO
                                         ! = 4 PDDG/PM3
                                         ! = 5 PDDG/MNDO
      integer   :: qmcharge              ! Total charges on QM region in eletronic units
                                         ! It should be Integer
                                         ! = 0 (default)
      integer   :: spin                  ! Spin state of QM wave function
                                         ! = 1 (singlet)
                                         !   valid 1 thru 6
      integer   :: verbosity             ! Controls to output print for QM/MM calculations
                                         ! = 0 (default)
      integer   :: itrmax                ! Maximum number of SCF cycles before assuming 
                                         ! convergence failure
                                         ! = 1000 (default)
      integer, pointer :: iqmatoms(:) => NULL()
                                         ! Integer list of atom numbers of the qm atoms
                                         ! The size should be iqmatoms(nquant)
      logical   :: qmqm_analyt           ! Flag for analytical vs numerical qm-qm derivates in 
                                         ! qm2 routines
                                         ! =.TRUE. (default) use analytical gradient
                                         ! =.FALSE.          use numerical gradient
      logical   :: tight_p_conv          ! Flag to control convergence criteria for density matrix
                                         ! =.FALSE. (default) density converges to 
                                         !                    to 0.05 * Sqrt(SCFCONV)
                                         ! =.TRUE.            to SCFCONV
      logical   :: qmmm_erep_incore      ! Control the storage of qm-qm 1-e repulsion integrals 
                                         ! in memory
                                         ! =.TRUE. (default)
                                         ! =.FALSE. compute on the fly
      logical   :: qmqm_erep_incore      ! Control the storage of qm-mm 1-e repulsion integrals 
                                         ! in memory
                                         ! =.TRUE. (default)
      logical   :: allow_pseudo_diag     ! Control the use of Pseudo-diagonalizations in SCF 
                                         ! when it is possible
                                         ! =.TRUE. (default)
      logical   :: qmqmrij_incore        ! Control the storage qm-qm R_ij and related equations 
                                         ! in memory
                                         ! =.TRUE. (default)
      logical   :: qmmmrij_incore        ! Control the storage qm-mm R_ij and related equations 
                                         ! in memory
                                         ! =.TRUE. (default)
      logical   :: QMMM_Gauss            ! Flag for use QM-MM Gaussian-Gaussian core-core term.
                                         ! =.TRUE.  if AM1/PM3/PDDGPM3/PM3CARB1 (default)
                                         ! =.FALSE. not use them, by user input.
      logical   :: Q_Diis                ! Flag to use DIIS converger.
                                         ! =.TRUE.  use this (default)  / =.FALSE. not use.

      logical   :: tr_bomd               ! turn on time-reversible Born-Oppenheimer MD (TR-BOMD)
      integer   :: num_scf_cycle         ! number of SCF cycle in TR-BOMD.

! To be cleaned up
      real(chm_real)  :: lnk_dis               ! Distance in angstroms between QM atom and link atom
      integer   :: lnk_atomic_no         ! Atomic number of link atom
                                         ! = 1 (Hydrogen, default)
      integer   :: qmgb                  ! Method for GB in QMMM (not yet implemented)
                                         ! = 0 use 0(zero) QM charges (default)
                                         ! = 1 use RESP charges for QMMM charges
                                         ! = 2 use CM1A or Mulliken charges for QM atoms
      integer   :: qmshake               ! Whether to shake qm atoms if ntc>1
                                         !                <==I don't understand this
                                         ! = 1-shake QM atoms
                                         ! = 0 do not shake
      logical   :: printcharges          ! Flag to control the printing of mulliken, 
                                         ! cm1a and cm2a charges
                                         ! =.TRUE.  print every energy evaluation
                                         ! =.FALSE. No printing
      logical   :: peptide_corr          ! Flag to control peptide bond MM correction
                                         ! =.FALSE. (default)

      END TYPE qmmm_namelist


      TYPE, PUBLIC :: qmmm_structure
      real(chm_real)  :: enuclr_qmqm, enuclr_qmmm
                                        ! Core-Core energy for QM-QM and QM-MM integrations (eV)
      real(chm_real)  :: elec_eng             ! Electronic energy (eV)


      real(chm_real), POINTER :: qm_coords(:,:) => NULL()
                                        ! Cartesian coordinates of QM atoms [3*NATQM long]
      real(chm_real), POINTER :: scaled_mm_charges(:) => NULL()
                                        ! MM charges copied from CHARMM main array (EU)
      real(chm_real), POINTER :: dxyzqm(:,:)    => NULL()
      real(chm_real), POINTER :: dxyzcl(:,:)    => NULL()
                                        ! Save gradients for QM and MM atoms 
                                        ! before passed into CHARMM
      real(chm_real), POINTER :: qm_xcrd(:,:)   => NULL()
                                        ! Contains imaged mm coordinates and mm charges


      integer   :: nquant               ! Total number of QM atoms
      integer   :: nlink                ! Total number of Link-H atoms 
                                        ! (that is just kept here) = 0
      integer   :: nquant_nlink         ! Total number of "QM" atoms (nquant+nlink)
      integer   :: qm_ntypes            ! The number of atom types present
      integer   :: dgfree_qm            ! QM degrees of freedom (That may not be used)
      integer   :: dgfree_mm            ! MM degrees of freedom (That may not be used)
      integer   :: qm_type_id(nelements)! The ID of each atom type
                                        ! Essentially the atomic number of each atom type. eg 8
      integer   :: qm_mm_pairs          ! Number of pairs per QM atom. Length of pair_list
      integer, POINTER :: qm_atom_type(:) => NULL()
                                        ! The type of each QM atom, essentially a re-basing of the
                                        ! atomic numbers to minimized memory usage
      integer, POINTER :: iqm_atomic_numbers(:) => NULL()
                                        ! Integer list of atomic numbers for the qm atoms
      integer, POINTER :: qm_mm_pair_list(:) => NULL()
                                        ! Non bond pair list for each QM atom
      logical, POINTER :: atom_mask(:) => NULL()
                                        ! Flag for specifying QM atom
                                        ! =.TRUE.  if QM atom
                                        ! =.FALSE. if MM atom

! To be cleaned up
      real(chm_real)  :: eke_qm, eke_mm       ! QM and MM components of the kinetic energy
                                        ! -Total kinetic energy eke = eke_qm + eke_mm
      real(chm_real)  :: fac_qm, fac_mm       ! conversion factor to temperature of QM and MM degrees 
                                        ! of freedom
                                        ! -Multiply by eke to get temperature
      integer, POINTER :: link_pairs(:,:) => NULL()
                                        ! List of MM-QM atoms for each link atoms added.
                                        ! Position 3,x of link_pairs hole MM atom's in the 
                                        !qm pair list

! Scratch space
! These arrays are available for any subroutine to use.
! They are allocated by allocate_qmmm.
! Each routine that uses one of these arrays should assume that the contents of
! these arrays will not be used
      real(chm_real), POINTER :: qm_real_scratch(:) => NULL()
                                        ! Real scratch array (3*natom)
      real(chm_real), POINTER :: qm_int_scratch(:) => NULL()
                                        ! Integer scratch array (3*natom)

      END TYPE qmmm_structure


      TYPE, PUBLIC :: qm2_structure
! Variables that are specific to qm_routine=2 (qm2)
      real(chm_real), POINTER :: pseudo_diag_matrix(:) => NULL()
                                        ! Scratch space for Pseudo-diagonalizaitons. 
                                        ! noccupied*(norbs-noccupied)
      real(chm_real), POINTER :: den_matrix(:) => NULL()
                                        ! Total density matrix
      real(chm_real), POINTER :: old_den_matrix(:) => NULL()
                                        ! Old total density matrix from previous step
                                        ! Allocated in qm2_load_params on first call
                                        ! Deallocated by deallocate_qmmm
      real(chm_real), POINTER :: old2_den_matrix(:) => NULL()
                                        ! Used by qm2_cnvg as workspace
      real(chm_real), POINTER :: fock_matrix(:) => NULL()
                                        ! Fock matrix
      real(chm_real), POINTER :: qm_mm_e_repul(:,:) => NULL()
                                        ! Array contains the QM-MM 1-e repulsion integrals
      real(chm_real), POINTER :: qm_qm_2e_repul(:) => NULL()
                                        ! Array contains the QM-QM 2-e replusion integrals. 
                                        ! N2el long
                                        ! Allocated by qm_mm and deallocated by deallocate_qmmm.
      real(chm_real), POINTER :: hmatrix(:) => NULL()
                                        ! Core H matrix. Used by routines called from qm2_energy
                                        ! Allocated in qm_mm on first call
                                        ! Deallocated by deallocate_qmmm
      real(chm_real), POINTER :: qm_qm_e_repul(:,:) => NULL()
                                        ! The QM-QM elecron repulsion integrals.
                                        ! Allocated in qm_mm on first call
                                        ! Dealocated by deallocate_qmmm
      real(chm_real), POINTER :: fock2_ptot2(:,:) => NULL()
                                        ! Used in qm2_fock2. (Nquant_nlink, 16)
                                        ! Allocated in qm2_load_params 
                                        ! Deallocated in deallocate_qmmm
      real(chm_real), POINTER :: mat_diag_workspace(:,:) => NULL()
                                        ! Matrix diagonialization workspace
                                        ! Allocated in qm2_load_params
      real(chm_real), POINTER :: eigen_vectors(:,:) => NULL()
                                        ! Eigen vectors during SCF
                                        ! Allocated in qm2_load_params

      integer   :: matsize              ! Size of packed symmetrix matrix. (norbs*(norbs+1)/2)
      integer   :: n2el                 ! Total number of 2-e repulsion integrals.
                                        ! Calculated by moldat as
                                   ! 50*nheavy(nheavy-1)+10*nheavy*nlight+(nlight*(nlight-1))/2
      integer   :: norbs                ! Total number of atomic orbitals
      integer   :: nclosed              ! Number of doubly occupied orbitals
      integer   :: nopenclosed          ! Number of doubly and singly occupied orbitals
      integer   :: qm_mm_e_repul_allocated
                                        ! Used in energy and derivative routines, which has been
                                        ! stored in memory.
                                        ! Allocated in qm_mm on first call or if pair list 
                                        !           changes too much
                                        ! Deallocated by deallocate_qmmm

! To be cleaned up
      integer   :: n_peptide_links      ! Number of peptide linkages in QM region to apply 
                                        ! MM correction to
      integer, POINTER :: peptide_links(:,:) => NULL()
                                        ! Identify of peptide linkages 4, n_peptide_linkages.
                                        ! 1 tp 4 = H-N-C-O atom numbers.

! for diis converger
      real(chm_real), pointer :: diis_fppf(:,:)=>null() ! fppf(linear,6)
      real(chm_real), pointer :: diis_fock(:,:)=>null() ! fock(linear,6)
      real(chm_real)          :: diis_emat(20,20)       ! emat(20,20)

      integer           :: diis_norbs             !
      integer           :: diis_linear            ! linear=norbs*(norbs+1)/2
      integer           :: diis_lfock             ! 1-6
      integer           :: diis_nfock             ! 1-6
      logical           :: diis_start             ! 

! for time-reversible Born-Oppenheimer MD (TR-BOMD), see qmmm_namelist%tr_bomd
!     auxiliary density matrices.
      logical                 :: q_apply_tr_bomd  !
      real(chm_real), POINTER :: density_p_new(:) => NULL()
      real(chm_real), POINTER :: density_p_old(:) => NULL()

      END TYPE qm2_structure


      TYPE, PUBLIC :: qm2_params_structure
! Parameters for each atom in the qm region
      real(chm_real)  :: tot_heat_form        ! Calculated in qm2_load_params
                                        ! Independent of structure, so constant
      real(chm_real), POINTER :: core_chg(:) => NULL()
                                        ! Core charge on each atom
                                        ! Allocated as nquant_nlink long by qm2_load_params
                                        ! Deallocated by deallocate_qmmm
      real(chm_real), POINTER :: orb_elec_ke(:,:) => NULL()
                                        ! Orbital electron kinetic energy integrals 
                                        ! (2,nquant_nlink)
                                        ! 1=2, 2=p
                                        ! Allocated in qm2_load_params
                                        ! Deallocated by deallocate_qmmm
      real(chm_real), POINTER :: betasas(:,:) => NULL()
                                        ! betas(n_i)+betas(n_j) qm_ntypes x qm_ntypes
      real(chm_real), POINTER :: betasap(:,:) => NULL()
                                        ! betas(n_i)+betap(n_j) qm_ntypes x qm_ntypes
      real(chm_real), POINTER :: betapap(:,:) => NULL()
                                        ! betap(n_i)+betap(n_j) qm_ntypes x qm_ntypes
      real(chm_real), POINTER :: FN1(:,:) => NULL()
                                        ! AM1/PM3 core-core Gaussian parameters
      real(chm_real), POINTER :: FN2(:,:) => NULL()
                                        ! AM1/PM3 core-core Gaussian parameters
      real(chm_real), POINTER :: FN3(:,:) => NULL()
                                        ! AM1/PM3 core-core Gaussian parameters
      real(chm_real), POINTER :: onec2elec_params(:,:) => NULL()
                                        ! One center two-e integral parameters 
                                        ! (Coulomb and exchange terms).
                                        ! Allocated as nquant_nlink long in qm2_load_params
                                        ! Deallocated in deallocate_qmmm
      real(chm_real), POINTER :: multip_2c_elec_params(:,:) => NULL()
                                        ! Parameters for the multipole expansions for 
                                        ! two center two-e integrals
                                        ! 9, nquant_nlink in order of DD,QQ,AM,AD,AQ,AM2,AD2,AQ2
                                        ! Allocated in qm2_load_params
      real(chm_real), POINTER :: cc_exp_params(:) => NULL()
                                        ! Exponents for core-core repulsions
      real(chm_real), POINTER :: s_orb_exp_by_type(:) => NULL()
                                        ! S orbital expansion coefficients for STO expansion
      real(chm_real), POINTER :: p_orb_exp_by_type(:) => NULL()
                                        ! P orbital expansion coefficients for STO expansion
                                        ! Both are allocated as qm_ntypes long by qm2_load_params
                                        ! Deallocated by qm2_setup_orb_exp

! Parameters for PDDG Hamiltonians
      real(chm_real), POINTER :: pddge1(:) => NULL()
      real(chm_real), POINTER :: pddge2(:) => NULL()

! Pre-computed Orbital interactions
      real(chm_real), POINTER :: atom_orb_zz_sxs_over_sas(:,:,:,:) => NULL()
                                        ! atom_orb_zz_s_x_s/atom_orb_zz_one_s_a_s
      real(chm_real), POINTER :: atom_orb_zz_sxp_over_sap(:,:,:,:) => NULL()
                                        ! atom_orb_zz_s_x_p/atom_orb_zz_one_s_a_p
      real(chm_real), POINTER :: atom_orb_zz_pxp_over_pap(:,:,:,:) => NULL()
                                        ! atom_orb_zz_p_x_p/atom_orb_zz_one_p_a_p
      real(chm_real), POINTER :: atom_orb_ss_eqn(:,:,:,:) => NULL()
                                        ! sqrt((two*sqrt(atom_orb_zz_s_x_s)*
                                        !           atom_orb_zz_one_s_a_s)**3
                                        ! *atom_orb_cc_s_x_s
      real(chm_real), POINTER :: atom_orb_sp_ovlp(:,:,:,:) => NULL()
                                        ! 2.0D2(:,:,:,:) => NULL()
                                        ! 2.0D0*atom_orb_pp_eqn*
                                        ! sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p
                                        ! SQRT(qm2_params%atom_orb_zz_p_by_type(L,qmjtype))*
                                        ! atom_orb_zz_one_s_a_p(k,l,qmitype,qmjtype)
                                        ! *atom_orb_sp_eqn
                                        ! Used in gover for Si-Pj overlap energy and 
                                        !                  -Sj-Pi overlap.
      real(chm_real), POINTER :: atom_orb_pp_ovlp_inj(:,:,:,:) => NULL()
                                        ! -4.0D0*sqrt(atom_orb_zz_p_x_p)*
                                        ! atom_orb_zz_one_p_a_p*
                                        ! qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)*
                                        ! atom_orb_pp_eqn
                                        ! Used in gover for Pi-Pj overlap energy when i /= j
      real(chm_real), POINTER :: atom_orb_pp_ovlp_ieqj1(:,:,:,:) => NULL()
                                        ! -4.0D0*atom_orb_pp_eqn*
                                        ! sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p*
                                        ! qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)
                                        ! Used in gover for Pi-Pj overlap energy when i == j
      real(chm_real), POINTER :: atom_orb_pp_ovlp_ieqj2(:,:,:,:) => NULL()
                                        ! 2.0D0*atom_orb_pp_eqn*
                                        ! sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p
      real(chm_real), POINTER :: atom_orb_ss_eqn_adb(:,:,:,:) => NULL()
                                    ! -2.0D0*A2_TO_BOHRS2*
                                    ! qm2_params%atom_orb_ss_eqn_adb(i,j,qmitype,qmjtype)*
                                    ! qm2_params%atom_orb_zz_sxs_over_sas(i,j,qmitype,qmjtype)
                                    ! Used for S-S overlap in QM-QM derivatives
      real(chm_real), POINTER :: atom_orb_sp_eqn_xy(:,:,:,:) => NULL()
                                    ! -4.0D0*A3_TO_BOHRS3*
                                    ! qm2_params%atom_orb_zz_sxp_over_sap(i,j,qmitype,qmjtype)**2*
                                    ! (one/(SQRT(qm2_params%atom_orb_zz_p_by_type(J,qmjtype))))*
                                    ! atom_orb_sp_eqn
                                    ! Used for S-P overlap in QM-QM derivatives, where P... /= axis
      real(chm_real), POINTER :: atom_orb_sp_eqn_xx1(:,:,:,:) => NULL()
                                    ! Used for S-P overlap in QM-QM derivatives, where P... == axis
      real(chm_real), POINTER :: atom_orb_sp_eqn_xx2(:,:,:,:) => NULL()
      real(chm_real), POINTER :: atom_orb_pp_eqn_xxy1(:,:,:,:) => NULL()
                                    ! -four*A2_TO_BOHRS2*(ADB_array(inner_index)**2)*
                                    ! (one/(SQRT(atom_orb_zz_p_x_p)))*atom_orb_pp_eqn
                                    ! Used for P-P overlap in QM-QM derivatives, 
                                    ! where P... = P... /= axis
      real(chm_real), POINTER :: atom_orb_pp_eqn_xxy2(:,:,:,:) => NULL()
                                        ! eight*A2_TO_BOHRS2*A2_TO_BOHRS2*
                                        ! (ADB_array(inner_index)**3)*
                                        ! (one/(SQRT(atom_orb_zz_p_x_p)))*atom_orb_pp_eqn

! Pre-computed PDDG parameters
      real(chm_real), POINTER :: pddg_term1(:,:) => NULL()
      real(chm_real), POINTER :: pddg_term2(:,:) => NULL()
      real(chm_real), POINTER :: pddg_term3(:,:) => NULL()
      real(chm_real), POINTER :: pddg_term4(:,:) => NULL()
                                        ! Ntypes*ntypes
                                        ! Stores the pre-exponential part of the PDDG eqns
                                        ! Allocated in qm2_load_params if PDDG is in use
      integer, POINTER :: natomic_orbs(:) => NULL()
                                        ! Number of atomic orbitals on atom
      integer, POINTER :: orb_loc(:,:) => NULL()
                                        ! Locations of orbitals. 2,nquant_nlink.
                                        ! 1, x gives beginning of orbitals on atom x
                                        ! 2, x gives last orbital on atom x
      integer, POINTER :: pascal_tri1(:) => NULL()
                                        ! Lower half triangle indices (pascal's triangle)
      integer, POINTER :: pascal_tri2(:) => NULL()
                                        ! Allocated in load_params
      integer, POINTER :: NUM_FN(:) => NULL()
                                        ! Number of FNX terms (first dimension> that are not zero.

      END TYPE qm2_params_structure


      TYPE, PUBLIC :: qm2_rij_eqns_structure
! Store R_ij infor for each QM-QM pair and QM-MM pair and their related equations
      integer   :: qmmmrij_allocated    ! Holds amount of memory allocated for qmmmrijdata
      integer   :: n1_qmqm,n2_qmqm,   & ! n1 and n2 dimensions of qmqmrijdata array
                   n1_qmmm,n2_qmmm      ! n1 and n2 dimensions of qmmmrijdata

      real(chm_real), POINTER :: qmqmrijdata(:,:) => NULL()
      real(chm_real), POINTER :: qmmmrijdata(:,:) => NULL()
                                        ! Store QM-QM and QM-MM R_ij values and related equations.
                                        ! Refer array_locations.h for the first dimension
                                        ! offsets.

      END TYPE qm2_rij_eqns_structure

      TYPE, PUBLIC :: qmmm_cut_switch
! Store switching function related values in computing energy and gradient
      logical            :: qswitch=.FALSE.
                                          ! Logical flag to use Switching function at cutoff region
      logical            :: qalloc_mem=.FALSE.
                                          ! Logical flag to deallocate and allocate memory again
                                          ! during minimization/dynamics
      integer            :: nswitch_atom  ! number of atoms to be switched
      integer            :: nswitch_atom_old
                                          ! number of atoms to be swithed+50,
                                          ! keep track of them to check whether memory allocatation
                                          ! is enough
      integer            :: natom         ! total number of atoms in the system
      integer            :: natqm         ! total number of qm atom
      integer            :: nqmgrp        ! total number of qm group
      integer            :: nmmgrp        ! total number of mm group
      integer, POINTER   :: iqmatm(:) => NULL()
                                          ! index for QM group information 
                                          ! (natqm) (each qm atom -> group position in iqmgrp)
      integer, POINTER   :: iqmgrp(:) => NULL()
                                          ! index for QM group information 
                                          ! (each qm group -> ngrp position)
                                          ! (nqmgrp) 
      integer, POINTER   :: immatm(:) => NULL()
                                          ! index for MM group information 
                                          ! (each atom -> mm group position)
                                          ! (natom)
      integer, POINTER   :: immgrp(:,:) => NULL() 
                                          ! index for MM group information 
                                          ! (each group -> mm atom position)
                                          ! (3,nmmgrp) 1: QM group position in iqmgrp
                                          !            2: start atom in ngrp / 3: last atom in ngrp
      logical, POINTER   :: scmask(:) => NULL()
                                          ! flag to mask the atoms is to be switched (natom)
                                          ! allocate/deallocate in routine QMMM_Nbnd_Switch
      real(chm_real),POINTER   :: scale(:) => NULL()
                                          ! scaled values of charge that to be swithed (natom)
                                          ! allocate/deallocate in routine QMMM_Nbnd_Switch
      real(chm_real),POINTER   :: dxqmmm(:,:) => NULL()
                                          ! gradient contribution from the dradient of switching
                                          ! (6,natom): 1~3 for MM atom (x,y,z), 
                                          !            4~6 for QM atom(x,y,z)
                                          ! allocate/deallocate in routine QMMM_Nbnd_Switch
      real(chm_real),POINTER   :: dxmm_wrk(:,:) => NULL()
                                          ! working arry for MM gradient from switching portion 
                                          ! (to be copied into dxyzmm array).
                                          ! (3,nswitch_atom_old)
      real(chm_real),POINTER   :: dxyz_qm(:,:) => NULL()
                                          ! working array for QM gradient from switching portion
                                          ! (3,nqmgrp)
      real(chm_real),POINTER   :: dxqmmm_elec(:,:,:)=> NULL()
                                          ! electrostatic/core contribution of QM-MM interaction
                                          ! energy that will be multiplied by dxqmmm to come up 
                                          ! with gradients of switching for QM-MM interaction 
                                          ! (11,nswitch_atom_old,nquant_nlink)
                                          ! 1~10 for orbital pairs, 11 for core part
                                          ! whith should be changed when you implement d-orbital 
                                          ! methods
                                          ! allocate/deallocate in routine qm_fill_mm_coords

      END TYPE qmmm_cut_switch

      TYPE, PUBLIC :: qm2_gho_structure
! Store GHO related information 
      logical            :: QGHO=.FALSE.  ! Logical flag to use GHO atoms
      logical            :: UHFGHO=.FALSE.
                                          ! Logical flag to use UHF/GHO
      integer            :: mqm16=16
      integer            :: natqm=0       ! number of qm atoms
      integer            :: nqmlnk=0      ! number of GHO atoms
      integer            :: norbsgho=0    ! number of AOS, same as qm2_struct%norbs
      integer, POINTER   :: IQLINK(:)   => NULL()   
                                          ! Pointer for GHO atom
      integer, POINTER   :: JQLINK(:,:) => NULL()  
                                          ! Pointer for MM atoms connected to GHO atom (3,nqmlnk)
      integer, POINTER   :: KQLINK(:)   => NULL()  
                                          ! Pointer for QM atom  connected to GHO atom
                                          ! Sizes are (nqmlnk)
      real(chm_real),POINTER   :: QMATMQ(:)   => NULL()  
                                          ! Size  is  (nqmlnk) 
      real(chm_real),POINTER   :: BT(:)       => NULL()  
                                          ! Size  is  (nqmlnk*mqm16)    C' = BT C
      real(chm_real),POINTER   :: BTM(:)      => NULL()  
                                          ! Size  is  (nqmlnk*mqm16)    C  = BTM C'
      real(chm_real),POINTER   :: DBTMMM(:,:,:) => NULL()
                                          ! Size  is  (3,3,nqmlnk*mqm16)
      real(chm_real),POINTER   :: PHO(:)      => NULL()  
                                          ! density matrix for GHO, size is (norbs*(norbs+1)/2)
      real(chm_real),POINTER   :: PBHO(:)     => NULL()
                                          ! density matrix for GHO, size is (norbs*(norbs+1)/2)
      real(chm_real),POINTER   :: FAOA(:)     => NULL()
                                          ! density matrix for GHO, size is (norbs*(norbs+1)/2)
      real(chm_real),POINTER   :: FAOB(:)     => NULL()  
                                          ! density matrix for GHO, 
                                          ! Size is same as density matrix, 
                                          ! which is (norbs*(norbs+1)/2)

! Local varibles only at qm2_scf and etc.
      real(chm_real), POINTER :: CAHB(:) =>NULL() ! Size is norbs*norbs
      real(chm_real), POINTER :: CBHB(:) =>NULL() !         norbs*norbs
      real(chm_real), POINTER :: DAHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: DBHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: FAHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: FBHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: PAHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: PBHB(:) =>NULL() !         norbs*(norbs*+1)/2
      real(chm_real), POINTER :: PAOLD(:)=>NULL() !         norbs*(norbs*+1)/2, used for GHO-DIIS
      real(chm_real), POINTER :: PBOLD(:)=>NULL() !         norbs*(norbs*+1)/2, used for GHO-DIIS
     
!***Note***
! FAOA and FAOB may not be needed..check later..
! -namkh

      END TYPE qm2_gho_structure

      TYPE, PUBLIC :: qmmm_ewald_stucture
! Store information related with QM/MM-Ewald sum calculations.
      logical            :: QEwald=.FALSE. ! Logical flag to use QM/MM-Ewald
      logical            :: QNoPMEwald=.FALSE.   ! Logical Flag not to use QM/MM-PMEwald.
      logical            :: ewald_startup  ! .True. if this is the very first MD step and 
                                           !        we are doing qmewald
      logical            :: erfcx_incore   ! .True. if qmqmrij_incore=.true.

      logical            :: qpka_calc_on   ! .True. if FEP-QM/MM for pKa calculations.
 
      integer            :: Ewmode = 1     ! Ewald mode options
                                           !     0 : No use of QM/MM-Ewald routine
                                           !     1 : QM/MM-Ewald such that MM in cutoff do 
                                           !         interact with QM density as regular
                                           !         QM/MM potential
                                           !(not yet)-1 : Same as 1, use input MM charges on 
                                           ! QM atom
      integer            :: Erfmod         ! Mode for Error function evaluation in CHARMM
                                           ! It is just copied here to use later in routines.
      integer            :: kmaxqx,         & ! Maximu K space vector to be summed in x-dir
                            kmaxqy,         & ! Maximu K space vector to be summed in y-dir
                            kmaxqz,         & ! Maximu K space vector to be summed in z-dir
                            ksqmaxq        ! Maximu K space vector square for spherical cutoff 
                                           !          in K space
      integer            :: totkq          ! Total number of K space vectors summed
      integer            :: natom          ! Total number of Namtom, copied here by 
                                           ! qm_mm convenience

      integer            :: iastrt         ! to be used for parallel calculations.
      integer            :: iafinl         ! numnod=1; iastrt=1, and iafinl=natom
      integer            :: iatotl         !           iatotl=natom
                                           ! numnod=n; iastrt: starting of natom loop.
                                           !           iafinl: ending of natom loop.
                                           !           iatotl: total term in mynod.

      integer            :: natqm_fix      ! Total number of QM atoms in 1st qm selection
                                           ! for FEP-QM/MM
      integer            :: nexl_atm       ! Total number of MM excluded atoms from 
                                           ! QM-MM interaction, such as MM atoms
                                           ! connected to QM atoms (IGMSEL(i)=5)
      integer, POINTER   :: nexl_index(:) => NULL()
                                           ! Index of pointing MM position in CHARMM main 
                                           ! coordinates for excluded MM atoms from 
                                           ! QM-MM interaction

      real(chm_real)           :: kappa                   ! Kappa in CHARMM for width of gaussians
      real(chm_real)           :: Volume                  ! Volume of system
      real(chm_real)           :: Recip(6)                ! Symmetric Reciprocal space Lattice vector
      real(chm_real)           :: ewald_core              ! Ewald potential with QM core charges 
                                                    ! - energy in eV.
      real(chm_real), POINTER  :: exl_xyz(:,:)=>NULL()    ! Coordinates for MM atoms excluded from
                                                    ! QM-MM interactions (4,:) 
                                                    !             in x,y,z,cg,x,y,z,cg...
      real(chm_real), POINTER  :: dexl_xyz(:,:)=>NULL()   ! Save gradient contributions from 
                                                    ! excluded MM atoms
      real(chm_real), POINTER  :: scf_mchg(:)=> NULL()    ! Mulliken charge of QM atoms
      real(chm_real), POINTER  :: scf_mchg_2(:)=>NULL()   ! Mulliken charge of QM atoms for gradient.
      real(chm_real), POINTER  :: mm_chg_fix_qm(:)=>NULL()! MM charges used for FEP-QM/MM in pKa.
      real(chm_real), POINTER  :: Kvec(:)=> NULL()        ! Array storing K vectors (totkq long)
      real(chm_real), POINTER  :: Ktable(:,:,:)=>NULL()   ! Table for storing complex exp(ik,r[j])
                                                    ! dimension = 6, natom, totkq
                                                    ! 1,x,y = x_cos      2,x,y = x_sin
                                                    ! 3,x,y = y_cos      4,x,y = y_sin
                                                    ! 5,x,y = z_cos      6,x,y = z_sin
      real(chm_real), POINTER  :: qmktable(:,:,:)=>NULL() ! As Ktable but stores the qmatom copies 
                                                    ! a linear 1->nquant fashion
      real(chm_real), POINTER  :: structfac_mm(:,:)=>Null() ! Sin and Cos functions in the structure
                                                    ! factor from MM chages; 2,totkq
      real(chm_real), POINTER  :: empot(:)=>NULL()        ! Nquant long, stores the potential at each 
                                                    ! QM atom due to the MM field.
      real(chm_real), POINTER  :: eslf(:,:)=>NULL()       ! Stores the self energy of the QM atoms to 
                                                    ! avoid double counting.
                                                    ! Nquant,Nquant matrix
      real(chm_real), POINTER  :: d_ewald_mm(:,:)=>NULL() ! Gradients on MM atoms due to reciprocal
                                                    ! QM-MM interactions for Virial correction. 
                                                    ! (3,natom)
      real(chm_real), POINTER  :: qmqmerfcx_data(:)=>NULL() 
                                                    ! Stores (one-erfcx)/rij 
                                                    ! when qmqmrij_incore=.true., allocate same 
                                                    ! as qmqm_rij_data array
      END TYPE qmmm_ewald_stucture

!****
!**** 
!!! QMMM_MODULE variables
      TYPE(qmmm_namelist),          SAVE :: qmmm_nml_r       ! for 1st qm
      TYPE(qmmm_structure),         SAVE :: qmmm_struct_r    ! (main qm)
      TYPE(qm2_structure),          SAVE :: qm2_struct_r
      TYPE(qm2_params_structure),   SAVE :: qm2_params_r
      TYPE(qm2_rij_eqns_structure), SAVE :: qm2_rij_eqns_r
      TYPE(qmmm_cut_switch),        SAVE :: qmmm_switch_r
      TYPE(qm2_gho_structure),      SAVE :: qm2_ghos_r
      TYPE(qmmm_ewald_stucture),    SAVE :: qmmm_ewald_r

      TYPE(qmmm_namelist),          SAVE :: qmmm_nml_p       ! for 2nd qm
      TYPE(qmmm_structure),         SAVE :: qmmm_struct_p    ! (small or
      TYPE(qm2_structure),          SAVE :: qm2_struct_p     !  perturbed
      TYPE(qm2_params_structure),   SAVE :: qm2_params_p     !  qm)
      TYPE(qm2_rij_eqns_structure), SAVE :: qm2_rij_eqns_p
      TYPE(qmmm_cut_switch),        SAVE :: qmmm_switch_p
      TYPE(qm2_gho_structure),      SAVE :: qm2_ghos_p
      TYPE(qmmm_ewald_stucture),    SAVE :: qmmm_ewald_p

      TYPE(qmmm_namelist),          SAVE :: qmmm_nml         ! local working
      TYPE(qmmm_structure),         SAVE :: qmmm_struct      ! qm..
      TYPE(qm2_structure),          SAVE :: qm2_struct 
      TYPE(qm2_params_structure),   SAVE :: qm2_params
      TYPE(qm2_rij_eqns_structure), SAVE :: qm2_rij_eqns
      TYPE(qmmm_cut_switch),        SAVE :: qmmm_switch
      TYPE(qm2_gho_structure),      SAVE :: qm2_ghos
      TYPE(qmmm_ewald_stucture),    SAVE :: qmmm_ewald


!BEGIN SUBROUTINES
      CONTAINS

      SUBROUTINE GetQMMMType(qproduct)
      logical                      :: qproduct

      if(qproduct) then                  ! get xxx_p
         qmmm_nml     = qmmm_nml_p
         qmmm_struct  = qmmm_struct_p
         qm2_struct   = qm2_struct_p
         qm2_params   = qm2_params_p
         qm2_rij_eqns = qm2_rij_eqns_p
         qmmm_switch  = qmmm_switch_p
         qm2_ghos     = qm2_ghos_p
         qmmm_ewald   = qmmm_ewald_p
      else                               ! get xxx_r
         qmmm_nml     = qmmm_nml_r
         qmmm_struct  = qmmm_struct_r
         qm2_struct   = qm2_struct_r
         qm2_params   = qm2_params_r
         qm2_rij_eqns = qm2_rij_eqns_r
         qmmm_switch  = qmmm_switch_r
         qm2_ghos     = qm2_ghos_r
         qmmm_ewald   = qmmm_ewald_r
      end if

      RETURN
      END SUBROUTINE GetQMMMType


      SUBROUTINE PutQMMMType(qproduct)
      logical                      :: qproduct

      if(qproduct) then                  ! put xxx_p
         qmmm_nml_p     = qmmm_nml
         qmmm_struct_p  = qmmm_struct
         qm2_struct_p   = qm2_struct
         qm2_params_p   = qm2_params
         qm2_rij_eqns_p = qm2_rij_eqns
         qmmm_switch_p  = qmmm_switch
         qm2_ghos_p     = qm2_ghos
         qmmm_ewald_p   = qmmm_ewald
      else                               ! put xxx_r
         qmmm_nml_r     = qmmm_nml
         qmmm_struct_r  = qmmm_struct
         qm2_struct_r   = qm2_struct
         qm2_params_r   = qm2_params
         qm2_rij_eqns_r = qm2_rij_eqns
         qmmm_switch_r  = qmmm_switch
         qm2_ghos_r     = qm2_ghos
         qmmm_ewald_r   = qmmm_ewald
      end if

      RETURN
      END SUBROUTINE PutQMMMType


      SUBROUTINE Get_Type_qmmm_nml(qmmm_nml_out,qmmm_nml_in)
      TYPE(qmmm_namelist)          :: qmmm_nml_out,qmmm_nml_in
      qmmm_nml_out = qmmm_nml_in

      return
      END SUBROUTINE Get_Type_qmmm_nml


      SUBROUTINE Get_Type_qmmm_struct(qmmm_struct_out,qmmm_struct_in)
      TYPE(qmmm_structure)         :: qmmm_struct_out,qmmm_struct_in
      qmmm_struct_out = qmmm_struct_in

      return
      END SUBROUTINE Get_Type_qmmm_struct


      SUBROUTINE Get_Type_qm2_struct(qm2_struct_out,qm2_struct_in)
      TYPE(qm2_structure)          :: qm2_struct_out,qm2_struct_in
      qm2_struct_out = qm2_struct_in

      return
      END SUBROUTINE Get_Type_qm2_struct


      SUBROUTINE Get_Type_qm2_params(qm2_params_out,qm2_params_in)
      TYPE(qm2_params_structure)   :: qm2_params_out,qm2_params_in
      qm2_params_out = qm2_params_in

      return
      END SUBROUTINE Get_Type_qm2_params


      SUBROUTINE Get_Type_qm2_rij_eqns(qm2_rij_eqns_out,qm2_rij_eqns_in)
      TYPE(qm2_rij_eqns_structure) :: qm2_rij_eqns_out,qm2_rij_eqns_in
      qm2_rij_eqns_out = qm2_rij_eqns_in

      return
      END SUBROUTINE Get_Type_qm2_rij_eqns


      SUBROUTINE Get_Type_qmmm_switch(qmmm_switch_out,qmmm_switch_in)
      TYPE(qmmm_cut_switch)        :: qmmm_switch_out,qmmm_switch_in
      qmmm_switch_out = qmmm_switch_in

      return
      END SUBROUTINE Get_Type_qmmm_switch


      SUBROUTINE Get_Type_qm2_ghos(qm2_ghos_out,qm2_ghos_in)
      TYPE(qm2_gho_structure)      :: qm2_ghos_out,qm2_ghos_in
      qm2_ghos_out = qm2_ghos_in

      return
      END SUBROUTINE Get_Type_qm2_ghos


      SUBROUTINE Get_Type_qmmm_ewald(qmmm_ewald_out,qmmm_ewald_in)
      TYPE(qmmm_ewald_stucture)    :: qmmm_ewald_out,qmmm_ewald_in
      qmmm_ewald_out = qmmm_ewald_in

      return
      END SUBROUTINE Get_Type_qmmm_ewald


!!#ifdef MPI
!!!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!!    SUBROUTINE qmmm_mpi_setup( master, natom, neb )
!!! 
!!! Does QMMM specific mpi setup and broadcasts
!!!
!!! Do we need this process? We should think about this.
!!! -namkh 
!!!
!!  use chm_kinds
!!  use qm2_double
#if KEY_PARALLEL==1
!!  use parallel  
#endif
!!  use mpi
!!    implicit none
!!
!!!Passed in
!!    logical, intent(in) :: master
!!    integer, intent(in) :: natom
!!    integer, intent(in) :: neb
!!
!!!Local
!!    integer :: replicates ! natom/neb  (1 if no neb)
!!    integer :: ier=0
!!
!!    if (neb > 0) then
!!      replicates = natom / neb
!!    else
!!      replicates = 1
!!    end if
!!
!!! Broadcast all of the info in qmmm_nml
!!! I don't think we can assume the structure is linear in memory 
!!! so we have to broadcast each seperately
!!    call mpi_bcast(qmmm_nml%qmcut, 1, MPI_DOUBLE_PRECISION, 0,  
!!   *               commsander, ier)
!!    call mpi_bcast(qmmm_nml%qmcut2, 1, MPI_DOUBLE_PRECISION, 0,  
!!   *               commsander, ier)
!!    call mpi_bcast(qmmm_nml%lnk_dis, 1, MPI_DOUBLE_PRECISION, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%scfconv, 1, MPI_DOUBLE_PRECISION, 0,   
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%pseudo_diag_criteria, 1,  
!!   *               MPI_DOUBLE_PRECISION, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%lnk_atomic_no, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmgb, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmtheory, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmcharge, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%spin, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%verbosity, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%itrmax, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmshake, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmqm_analyt, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%tight_p_conv, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%printcharges, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%peptide_corr, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%allow_pseudo_diag, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmmm_erep_incore, 1,mpi_logical, 0,  
!!   *               commsander, ier)
!!    call mpi_bcast(qmmm_nml%qmqm_erep_incore, 1,mpi_logical, 0,  
!!   *               commsander, ier)
!!    call mpi_bcast(qmmm_nml%qmqmrij_incore, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_nml%qmmmrij_incore, 1, mpi_logical, 0,  
!!   *               commsander, ier) 
!!
!!    call mpi_bcast(qmmm_struct%nquant, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_struct%nlink, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_struct%nquant_nlink, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_struct%qm_ntypes, 1, mpi_integer, 0,  
!!   *               commsander, ier) 
!!    call mpi_bcast(qmmm_struct%qm_type_id, nelements, mpi_integer,  
!!   *               0, commsander, ier) 
!!
!!! Now we do the arrays, note we have to allocate on all the 
!!! non-master nodes
!!! Note: we don't need to allocate the pair list as this is done 
!!!       in qm_mm routine.
!!!       It is also filled here so we don't need to broadcast it 
!!!       at this point.
!!
!!    if ( .not. master ) then
!!       call allocate_qmmm( natom, replicates )  <- old format
!!       allocate (qmmm_struct%link_pairs(3,qmmm_struct%nlink),  
!!   *             stat=ier)   
!!       if(ier.ne.0) call Aass(1,'qmmm_mpi_setup','link_pairs')
!!       allocate (qmmm_struct%qm_atom_type(qmmm_struct%nquant_nlink), 
!!   *             stat=ier)
!!       if(ier.ne.0) call Aass(1,'qmmm_mpi_setup','qm_atom_type')
!!    end if
!!
!!!Now we can broadcast each of the arrays
!!    call mpi_bcast(qmmm_struct%qm_atom_type,  
!!   *               qmmm_struct%nquant_nlink, MPI_INTEGER,  
!!   *               0, commsander, ier)
!!!All nodes now zero the charges and copy them.
!!    call mpi_bcast(qmmm_nml%iqmatoms, qmmm_struct%nquant,  
!!   *               mpi_integer, 0, commsander, ier)
!!    call mpi_bcast(qmmm_struct%link_pairs, 3*qmmm_struct%nlink,  
!!   *               mpi_integer,0, commsander, ier) 
!!    call mpi_bcast(qmmm_struct%iqm_atomic_numbers,  
!!   *               qmmm_struct%nquant_nlink/replicates,  
!!   *               mpi_integer, 0, commsander, ier)
!!    call mpi_bcast(qmmm_struct%atom_mask, natom, mpi_logical,  
!!   *               0, commsander, ier)
!!    call mpi_bcast(qmmm_struct%scaled_mm_charges, natom,  
!!   *               MPI_DOUBLE_PRECISION, 0, commsander, ier)
!!
!!    END SUBROUTINE qmmm_mpi_setup
!!!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!!#endif

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+Sorts the array iqmatoms into numerical order
      SUBROUTINE qmsort( iqmatoms )

  use chm_kinds
      implicit none

      integer, intent(inout) :: iqmatoms(*)

! Local
      integer i,j,lcurrent

!sort array iqmatoms in ascending order
      do i = 1,qmmm_struct%nquant
        lcurrent = iqmatoms(i)
        do j = (i+1),qmmm_struct%nquant
           if (lcurrent.gt.iqmatoms(j)) then
              iqmatoms(i) = iqmatoms(j)
              iqmatoms(j) = lcurrent
              lcurrent = iqmatoms(i)
           endif
        end do
      end do
      return
      END SUBROUTINE qmsort
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+Adjusts iqmatoms array for neb calculations
      SUBROUTINE adjust_iqmatoms_for_neb(iqmatoms, neb, replicates)
  use chm_kinds
      implicit none

      integer, intent(in) :: iqmatoms(*)
      integer, intent(in) :: neb
      integer, intent(in) :: replicates

      return
      END SUBROUTINE adjust_iqmatoms_for_neb
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+Identify qm atom elements.
      SUBROUTINE get_atomic_number(name,atomic_number)
  use chm_kinds
      implicit none

      integer, intent(in) :: atomic_number
      character(len=4)    :: name

      return
      END SUBROUTINE get_atomic_number
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!!!*********************************************************************
!!!
!!!Memory Allocation/Deallocation/Reallocation routines
!!!

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE reallocate_iqm_atno_array(orig_size,new_size)
!
! reallocation of integer array
!
! This routine reallocates iqm_atomic_numbers which was
! originally allocated as orig_size as new_size array ends
! up being padded with zeros
!
! This routine has been called in "identify_link_atoms". So
! this routine may not be used in CHARMM case.

  use chm_kinds
  use number, only : zero
      implicit none

!Passed in
      integer, intent(in) :: orig_size, new_size

!Local
      integer, dimension(:), pointer :: temp_array
      integer :: i
      integer :: ier=0

      allocate(temp_array(new_size),stat=ier)
      if(ier.ne.0) call Aass(1,'reallocate_iqm_atno_array','temp_array')

      temp_array = zero 

      do i=1, orig_size   !Copy out the data to a temporary array
         temp_array(i) = qmmm_struct%iqm_atomic_numbers(i)
      end do

! Deallocate and allocate it as bigger
      deallocate ( qmmm_struct%iqm_atomic_numbers, stat = ier)
      if(ier.ne.0) call Aass(0,'reallocate_iqm_atno_array', &
                            'iqm_atomic_numbers')

      allocate ( qmmm_struct%iqm_atomic_numbers(new_size), stat=ier )
      if(ier.ne.0) call Aass(1,'reallocate_iqm_atno_array', &
                            'iqm_atomic_numbers')

! copy in the data
      do i=1, new_size
         qmmm_struct%iqm_atomic_numbers(i) = temp_array(i) 
      end do

      deallocate(temp_array,stat=ier)
      if(ier.ne.0) call Aass(0,'reallocate_iqm_atno_array', &
                            'iqm_atomic_numbers')

      return
      END SUBROUTINE reallocate_iqm_atno_array
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE allocate_qmmm( natom )
!
! allocates space for qmmm variables and arrays that depend
! only on nquant or natom
!

  use chm_kinds
      implicit none

!passed in
      integer , intent(in) :: natom
!Local
      integer :: ier=0

! changed from "allocate ( qmmm_nml%iqmatoms(qmmm_struct%nquant), 
!            *            stat=ier )"
!             allocate ( qmmm_nml%iqmatoms(qmmm_struct%nquant_nlink),
!            *            stat=ier )
      allocate ( qmmm_nml%iqmatoms(qmmm_struct%nquant), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','iqmatoms')

! iqm_atomic_numbers is allocated here as nquant+nlink long but
! initially nlink is zero so it only gets allocated as nquant long 
! on the master thread. It is then resized when nlink are known 
! about. When all other mpi threads call this allocation routine
! nquant+nlink should include both qm and link atoms.
      allocate ( qmmm_struct%iqm_atomic_numbers((qmmm_struct%nquant +  &
                 qmmm_struct%nlink)), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','iqm_atomic_numbers')

      allocate ( qmmm_struct%atom_mask(natom), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','atom_mask')

      allocate (qmmm_struct%scaled_mm_charges(natom), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','scaled_mm_charges')

      allocate (qmmm_struct%qm_real_scratch(3*natom), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','qm_real_scratch')

      allocate (qmmm_struct%qm_int_scratch(3*natom), stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm','qm_int_scratch')

      return
      END SUBROUTINE allocate_qmmm
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+ allocates space for qmmm pair list
      SUBROUTINE allocate_qmmm_pair_list( natom )

  use chm_kinds
      implicit none

!Passed in
      integer, intent(in) :: natom  !Total number of REAL atoms in sim - MM + QM

!Local
      integer :: ier=0

! NOTE we are overestimating the memory needed here.

      allocate ( qmmm_struct%qm_mm_pair_list( (natom -  &
                         qmmm_struct%nquant + 1) ),     &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'allocate_qmmm_pair_list', &
                             'qm_mm_pair_list')

      return
      END SUBROUTINE allocate_qmmm_pair_list
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Allocate memory for parameter arrays
      SUBROUTINE qm2_allocate_params1

  use chm_kinds
      implicit none

! local variables
      integer :: ier = 0

!Memory allocation
      allocate ( qm2_params%core_chg(qmmm_struct%nquant_nlink),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','core_chg')

      allocate ( qm2_params%natomic_orbs(qmmm_struct%nquant_nlink),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','natomic_orbs')

      allocate ( qm2_params%orb_loc(2,qmmm_struct%nquant_nlink),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','orb_loc')

      allocate ( qm2_params%onec2elec_params(5,   &
                      qmmm_struct%nquant_nlink),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1', &
                             'onec2elec_params')
      allocate ( qm2_params%multip_2c_elec_params(8,  &
                      qmmm_struct%nquant_nlink),  &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1', &
                             'multip_2c_elec_params')
      allocate ( qm2_params%cc_exp_params(qmmm_struct%nquant_nlink),  &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','cc_exp_params')
      allocate (qm2_params%orb_elec_ke(2,qmmm_struct%nquant_nlink),  &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','orb_elec_ke')
      allocate ( qm2_params%betasas(qmmm_struct%qm_ntypes,   &
                                    qmmm_struct%qm_ntypes),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','betasas')
      allocate ( qm2_params%betasap(qmmm_struct%qm_ntypes,   &
                                    qmmm_struct%qm_ntypes),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','betasap')
      allocate ( qm2_params%betapap(qmmm_struct%qm_ntypes,   &
                                    qmmm_struct%qm_ntypes),   &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1','betapap')


      if (qmmm_nml%qmtheory.EQ.AM1 .OR.   &
          qmmm_nml%qmtheory.EQ.PM3 .OR.   &
          qmmm_nml%qmtheory.EQ.PDDGPM3 .OR.   &
          qmmm_nml%qmtheory.EQ.PM3CARB1 ) then
         allocate ( qm2_params%FN1(4,qmmm_struct%qm_ntypes),stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','FN1')

         allocate ( qm2_params%FN2(4,qmmm_struct%qm_ntypes),stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','FN2')

         allocate ( qm2_params%FN3(4,qmmm_struct%qm_ntypes),stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','FN3')

         allocate ( qm2_params%NUM_FN(qmmm_struct%qm_ntypes),stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','NUM_FN')
      end if

      if ( (qmmm_nml%qmtheory.EQ.PDDGPM3) .OR.   &
           (qmmm_nml%qmtheory.EQ.PDDGMNDO) ) then
         allocate ( qm2_params%pddge1(qmmm_struct%nquant_nlink),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddge1')
         allocate ( qm2_params%pddge2(qmmm_struct%nquant_nlink),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddge2')
         allocate ( qm2_params%pddg_term1(qmmm_struct%qm_ntypes,   &
                                          qmmm_struct%qm_ntypes),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddg_term1')
         allocate ( qm2_params%pddg_term2(qmmm_struct%qm_ntypes,   &
                                          qmmm_struct%qm_ntypes),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddg_term2')
         allocate ( qm2_params%pddg_term3(qmmm_struct%qm_ntypes,   &
                                          qmmm_struct%qm_ntypes),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddg_term3')
         allocate ( qm2_params%pddg_term4(qmmm_struct%qm_ntypes,   &
                                          qmmm_struct%qm_ntypes),   &
                    stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params1','pddg_term4')
      end if

!Note: s_orb_exp and p_orb_exp are only needed 
!      on the first call to fill qm2_setup_orb_exp,
!      so after they are used in QMMM_PARM_LOAD/qm2_setup_orb_exp
!      they are deallocated (in qm2_setup_orb_exp).
      allocate (qm2_params%s_orb_exp_by_type(nelements), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1', &
                             's_orb_exp_by_type')

      allocate (qm2_params%p_orb_exp_by_type(nelements), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params1', &
                             'p_orb_exp_by_type')

      return
      END SUBROUTINE qm2_allocate_params1
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Allocate memory for parameter arrays
      SUBROUTINE qm2_allocate_params2

  use chm_kinds
      implicit none

! local variables
      integer :: ier = 0

!---------- ALLOCATE PARAMETER SPACE ---------

!Allocate things that depend on norbs:
      allocate (qm2_params%pascal_tri1(qm2_struct%norbs), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','pascal_tri1')

      allocate (qm2_params%pascal_tri2(qm2_struct%norbs), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','pascal_tri2')

!Allocate items and scratch space in the qm2_struct array:
      allocate (qm2_struct%mat_diag_workspace(qm2_struct%norbs,6),   &
                stat=ier)
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2', &
                             'mat_diag_workspace')

      allocate (qm2_struct%eigen_vectors(qm2_struct%norbs,   &
                                         qm2_struct%norbs),   &
                stat=ier)
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2', &
                             'eigen_vectors')

      qm2_struct%matsize =   &
                 ishft(qm2_struct%norbs*(qm2_struct%norbs+1),-1)
!                !   ishift(x,-1) = integer divide by 2
      allocate ( qm2_struct%den_matrix(qm2_struct%matsize), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','den_matrix')

      allocate ( qm2_struct%old_den_matrix(qm2_struct%matsize),  &
                 stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','old_den_matrix')

      allocate ( qm2_struct%old2_den_matrix(qm2_struct%matsize),  &
                 stat=ier ) ! used by qm2_cnvg as scratch
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','old2_den_matrix')

      allocate ( qm2_struct%fock_matrix(qm2_struct%matsize), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','fock_matrix')

      allocate ( qm2_struct%hmatrix(qm2_struct%matsize), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','hmatrix')

      allocate (qm2_struct%fock2_ptot2(qmmm_struct%nquant_nlink,16),  &
                stat=ier)
      if(ier.ne.0) call Aass(1,'qm2_allocate_params2','fock2_ptot2')

!Allocate scratch space for pseudo diagonalisations
!Will need to be noccupied*(norbs-noccupied)
      if (qmmm_nml%allow_pseudo_diag) then
        allocate (qm2_struct%pseudo_diag_matrix(qm2_struct%nopenclosed*  &
                  (qm2_struct%norbs-qm2_struct%nopenclosed)), stat=ier )
        if(ier.ne.0) call Aass(1,'qm2_allocate_params2', &
                               'pseudo_diag_matrix')
      end if

! for DIIS converger
      if (qmmm_nml%Q_Diis) then
         if((qm2_ghos%QGHO).and. (qm2_ghos%nqmlnk.gt.0)) then
            qm2_struct%diis_norbs = qm2_struct%norbs-3*qm2_ghos%nqmlnk
            qm2_struct%diis_linear= (qm2_struct%diis_norbs* &
                                     (qm2_struct%diis_norbs+1))/2
         else
            qm2_struct%diis_norbs = qm2_struct%norbs
            qm2_struct%diis_linear= qm2_struct%matsize
         end if
         qm2_struct%diis_start =.TRUE.
         qm2_struct%diis_lfock = 0
         qm2_struct%diis_nfock = 0
         allocate(qm2_struct%diis_fppf(qm2_struct%diis_linear,6), &
                  stat=ier)
         if(ier.ne.0) call Aass(1,'qm2_allocate_params2','diis_fppf')
         allocate(qm2_struct%diis_fock(qm2_struct%diis_linear,6), &
                  stat=ier)
         if(ier.ne.0) call Aass(1,'qm2_allocate_params2','diis_fock')
      end if

! for TR-BOMD methods.
      if(qmmm_nml%tr_bomd) then
         allocate(qm2_struct%density_p_new(qm2_struct%matsize), stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params2','density_p_new')

         allocate(qm2_struct%density_p_old(qm2_struct%matsize), stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_params2','density_p_old')
      end if

      return
      END SUBROUTINE qm2_allocate_params2
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE qm2_allocate_qmqm_e_repul(n2el)
!
! Allocates memory for 1-electron repulsion integrals
!
! This routine allocates space for the QM-QM 2 electron repulsion
! integrals and optionally the one electron repulsion integrals.
! It should only be called once at the setup for QM_MM.
!

  use chm_kinds
      implicit none

!Passed in
      integer, intent(in) :: n2el

!Local
      integer :: ier=0

      allocate ( qm2_struct%qm_qm_2e_repul(n2el), stat=ier )
      if(ier.ne.0) call Aass(1,'qm2_allocate_qmqm_e_repul', &
                             'qm_qm_2e_repul')

!only need the QM-QM electron repulsion integrals stored if we
!are doing analytical QM-QM derivatives and
! qmqm_erep_incore = true.
! qmqm_e_repul needs to be nquant_nlink*(nquant_nlink-1)/2 x 22
! long
      if (qmmm_nml%qmqm_erep_incore) then
         allocate ( qm2_struct%qm_qm_e_repul(22,   &
                    (qmmm_struct%nquant_nlink *    &
                     (qmmm_struct%nquant_nlink-1)/2)),   &
                  stat=ier )
         if(ier.ne.0) call Aass(1,'qm2_allocate_qmqm_e_repul', &
                                'qm_qm_e_repul')
      end if

      return
      END SUBROUTINE qm2_allocate_qmqm_e_repul
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE qm2_allocate_qmmm_e_repul(natom,npairs,first_call)
!
! Allocates space for the QM-MM 1 electron repulsion integrals. 
! Note this should be called whenever the array needs to be 
! resized - e.g. npairs gets too big. Must have deallocated 
! the array first.
!
  use chm_kinds
  use stream

      implicit none

!Passed in
      integer, intent(in) :: natom, npairs
      logical, intent(in) :: first_call

!Local
      integer :: ier=0
      integer :: nqatm, nmatm, array_size, array_check
      logical :: qprint

      qprint=.false.
      if(Prnlev.ge.2) qprint=.true.

!We will allocate the qm_mm_e_repul array enough to hold the npairs
!or qmmm_struct%nquant * (natom - qmmm_struct%nquant)
!whichever is smaller. We then store the amount we allocated
!in qm2_rij_eqns%qmmmrij_allocated. In this way when the number
!of pairs changes this value can be checked and the array 
!reallocated large if necessary.

      nqatm      = qmmm_struct%nquant_nlink
      nmatm      = natom - nqatm
      array_check= npairs*nqatm

      array_size = array_check + npairs
      array_size = min(array_size,nqatm*nmatm)

      if (first_call) then
         allocate(qm2_struct%qm_mm_e_repul(4,array_size),stat=ier)
         if(ier.ne.0) call Aass(1,'qm2_allocate_qmmm_e_repul', &
                                'qm_mm_e_repul')
         qm2_struct%qm_mm_e_repul_allocated=array_size
         if (qmmm_nml%verbosity .GT. 1 .and. qprint) then
           write(6,*) 'QMMM: Allocating qm_mm_e_repul array as 4 x ',  &
                       array_size
           write(6,*) 'QMMM: to hold min npairs of ',npairs,  &
                      ' and nquant of ',  &
                       nqatm
         end if
      else if (array_check .GT.  &
               qm2_struct%qm_mm_e_repul_allocated) then
         deallocate(qm2_struct%qm_mm_e_repul,stat=ier)
         if(ier.ne.0) call Aass(0,'qm2_allocate_qmmm_e_repul', &
                                'qm_mm_e_repul')

         allocate(qm2_struct%qm_mm_e_repul(4,array_size),stat=ier)
         if(ier.ne.0) call Aass(1,'qm2_allocate_qmmm_e_repul', &
                                'qm_mm_e_repul')

         qm2_struct%qm_mm_e_repul_allocated=array_size
         if (qmmm_nml%verbosity .GT. 1 .and. qprint) then
           write(6,*) 'QMMM: Allocating qm_mm_e_repul array as 4 x ',  &
                       array_size
           write(6,*) 'QMMM: to hold min npairs of ',npairs,  &
                      ' and nquant of ',  &
                       nqatm
         end if
      end if

      return
      END SUBROUTINE qm2_allocate_qmmm_e_repul
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE qm2_allocate_qm2_qmqm_rij_eqns(PDDG_IN_USE)
!
! Allocates memory for qm2_rij_eqns structure.
!

  use chm_kinds
      implicit none

!Passed in
      logical, intent(in) :: PDDG_IN_USE

!Local variables
      integer :: ier=0
      integer :: array_size, array_check

      array_check = qmmm_struct%nquant_nlink
      if (qmmm_nml%qmqmrij_incore) then
         array_size = ishft(array_check*array_check,-1)

         if (PDDG_IN_USE) then
            qm2_rij_eqns%n1_qmqm   = QMQMNORIJ+QMQMNOPDDG
            qm2_rij_eqns%n2_qmqm   = array_size
            allocate(qm2_rij_eqns%qmqmrijdata(QMQMNORIJ+QMQMNOPDDG,   &
                                              array_size),stat=ier)
            if(ier.ne.0) call Aass(1,'qm2_allocate_qm2_qmqm_rij_eqns', &
                                   'qmqmrijdata')
         else
            qm2_rij_eqns%n1_qmqm   = QMQMNORIJ
            qm2_rij_eqns%n2_qmqm   = array_size
            allocate(qm2_rij_eqns%qmqmrijdata(QMQMNORIJ,array_size),  &
                     stat=ier)
            if(ier.ne.0) call Aass(1,'qm2_allocate_qm2_qmqm_rij_eqns', &
                                   'qmqmrijdata')
         end if
      end if
         
      return
      END SUBROUTINE qm2_allocate_qm2_qmqm_rij_eqns
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE qm2_allocate_qm2_qmmm_rij_eqns(natom,npairs)
!
! Allocate qmmmrijdata array.
!

  use chm_kinds
      implicit none

!Passed in
      integer, intent(in) :: natom, npairs

!Local variables
      integer :: ier=0
      integer :: array_size, array_check

      array_check = qmmm_struct%nquant_nlink

! We will allocate the qmmmrijdata array enough to hold the 
! npairs+a bit or
!  qmmm_struct%nquant * (natom - qmmm_struct%nquant)
! whichever is smaller. We then store the amount we allocated
! in qm2_rij_eqns%qmmmrij_allocated. In this way when the number
! of pairs changes this value can be checked and the array 
! reallocated large if necessary.

      if (qmmm_nml%qmmmrij_incore) then
         array_size = npairs*(array_check+1)
         array_size = min(array_size,array_check*(natom-array_check))
         qm2_rij_eqns%n1_qmmm    = QMMMNORIJ
         qm2_rij_eqns%n2_qmmm    = array_size
         allocate(qm2_rij_eqns%qmmmrijdata(QMMMNORIJ,array_size),   &
                  stat=ier)
         if(ier.ne.0) call Aass(1,'qm2_allocate_qm2_qmmm_rij_eqns', &
                                'qmmmrijdata')

         qm2_rij_eqns%qmmmrij_allocated=array_size
         if (qmmm_nml%verbosity .GT. 1) then
            write(6,*) 'QMMM: Allocating qmmmrijdata array as ',  &
                        QMMMNORIJ,' x ',array_size
            write(6,*) 'QMMM: to hold min npairs of ',npairs,  &
                       ' and nquant of ',array_check
         end if
      end if

      return
      END SUBROUTINE qm2_allocate_qm2_qmmm_rij_eqns
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE deallocate_qmmm( master, neb )
!
! deallocates space for qmmm variables and arrays
!

  use chm_kinds
      implicit none

!Passed in
      logical, intent(in) :: master
      integer, intent(in) :: neb

!Local
      integer :: ier=0

!If this is a parallel run the non master threads will only have
!allocated this memory if neb is on since otherwise QM calc is
!currently only done on master thread.

      if (master .OR. neb.GT.0) then

!Deallocate pointers
        if (qmmm_nml%qmqm_erep_incore) then
          deallocate ( qm2_struct%qm_qm_e_repul, stat = ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_qm_e_repul')
        end if

        if (qm2_struct%n_peptide_links.GT.0) then
          deallocate ( qm2_struct%peptide_links, stat=ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm','peptide_links')
        end if

        deallocate ( qm2_struct%eigen_vectors, stat=ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','eigen_vectors')

        deallocate ( qm2_struct%mat_diag_workspace, stat=ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','mat_diag_workspace')

        if (qmmm_nml%allow_pseudo_diag) then
          deallocate ( qm2_struct%pseudo_diag_matrix, stat=ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                                 'pseudo_diag_matrix')
        end if

        if (qmmm_nml%Q_Diis) then
          deallocate ( qm2_struct%diis_fppf, stat=ier )
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','diis_fppf')
          deallocate ( qm2_struct%diis_fock, stat=ier )
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','diis_fock')
        end if

        if(qmmm_nml%tr_bomd) then
           deallocate ( qm2_struct%density_p_new, stat=ier )
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','density_p_new')

           deallocate ( qm2_struct%density_p_old, stat=ier )
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','density_p_old')
        end if

        deallocate ( qm2_struct%fock2_ptot2, stat=ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','fock2_ptot2')

        deallocate ( qm2_struct%hmatrix, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','hmatrix')

        deallocate (qm2_struct%qm_qm_2e_repul, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_qm_2e_repul')

        if (qmmm_nml%qmmm_erep_incore) then
          deallocate ( qm2_struct%qm_mm_e_repul, stat = ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_mm_e_repul')
        end if

        deallocate ( qm2_struct%fock_matrix, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','fock_matrix')

        deallocate ( qm2_struct%old2_den_matrix, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','old2_den_matrix')

        deallocate ( qm2_struct%old_den_matrix, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','old_den_matrix')

        deallocate ( qm2_struct%den_matrix, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','den_matrix')


!parameter array
        deallocate (qm2_params%atom_orb_pp_eqn_xxy2, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_pp_eqn_xxy2')

        deallocate (qm2_params%atom_orb_pp_eqn_xxy1, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_pp_eqn_xxy1')

        deallocate (qm2_params%atom_orb_sp_eqn_xx2, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_sp_eqn_xx2')

        deallocate (qm2_params%atom_orb_sp_eqn_xx1, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_sp_eqn_xx1')

        deallocate (qm2_params%atom_orb_sp_eqn_xy, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','atom_orb_sp_eqn_xy')

        deallocate (qm2_params%atom_orb_ss_eqn_adb, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_ss_eqn_adb')

        deallocate (qm2_params%atom_orb_pp_ovlp_ieqj2, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_pp_ovlp_ieqj2')

        deallocate (qm2_params%atom_orb_pp_ovlp_ieqj1, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_pp_ovlp_ieqj1')

        deallocate (qm2_params%atom_orb_pp_ovlp_inj, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_pp_ovlp_inj')

        deallocate (qm2_params%atom_orb_sp_ovlp, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','atom_orb_sp_ovlp')

        deallocate (qm2_params%atom_orb_ss_eqn, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','atom_orb_ss_eqn')

        deallocate (qm2_params%atom_orb_zz_pxp_over_pap, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_zz_pxp_over_pap')

        deallocate (qm2_params%atom_orb_zz_sxp_over_sap, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_zz_sxp_over_sap')

        deallocate (qm2_params%atom_orb_zz_sxs_over_sas, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'atom_orb_zz_sxs_over_sas')

        if (qmmm_nml%qmtheory.EQ.AM1 .OR.   &
            qmmm_nml%qmtheory.EQ.PM3 .OR.   &
            qmmm_nml%qmtheory.EQ.PDDGPM3 .OR.   &
            qmmm_nml%qmtheory.EQ.PM3CARB1) then
           deallocate (qm2_params%NUM_FN, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','NUM_FN')

           deallocate (qm2_params%FN3, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','FN3')

           deallocate (qm2_params%FN2, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','FN2')

           deallocate (qm2_params%FN1, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','FN1')

        end if

        if ( (qmmm_nml%qmtheory.EQ.PDDGPM3) .OR.   &
             (qmmm_nml%qmtheory.EQ.PDDGMNDO) ) then
           deallocate (qm2_params%pddge1, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddge1')

           deallocate (qm2_params%pddge2, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddge2')

           deallocate (qm2_params%pddg_term1, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddg_term1')

           deallocate (qm2_params%pddg_term2, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddg_term2')

           deallocate (qm2_params%pddg_term3, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddg_term3')

           deallocate (qm2_params%pddg_term4, stat = ier)
           if(ier.ne.0) call Aass(0,'deallocate_qmmm','pddg_term4')
        end if

        deallocate (qm2_params%cc_exp_params, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','cc_exp_params')

        deallocate (qm2_params%multip_2c_elec_params, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm', &
                               'multip_2c_elec_params')

        deallocate (qm2_params%onec2elec_params, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','onec2elec_params')

        deallocate (qm2_params%orb_elec_ke, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','orb_elec_ke')

        deallocate (qm2_params%betasas, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','betasas')

        deallocate (qm2_params%betasap, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','betasap')

        deallocate (qm2_params%betapap, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','betapap')

        deallocate (qm2_params%orb_loc, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','orb_loc')

        deallocate (qm2_params%natomic_orbs, stat = ier)
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','natomic_orbs')

        deallocate (qm2_params%core_chg, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','core_chg')

        deallocate (qm2_params%pascal_tri1, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','pascal_tri1')

        deallocate (qm2_params%pascal_tri2, stat = ier )
        if(ier.ne.0) call Aass(0,'deallocate_qmmm','pascal_tri2')

        if (qmmm_nml%qmqmrij_incore) then
          deallocate (qm2_rij_eqns%qmqmrijdata, stat = ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm','qmqmrijdata')
        end if

        if (qmmm_nml%qmmmrij_incore) then
          deallocate (qm2_rij_eqns%qmmmrijdata, stat = ier )
          if(ier.ne.0) call Aass(0,'deallocate_qmmm','qmmmrijdata')
        end if
      end if


      if (master .OR. neb.GT.0) then
       deallocate ( qmmm_struct%qm_coords, stat = ier)
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_coords')

       deallocate ( qmmm_struct%qm_xcrd, stat = ier )
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_xcrd')

       deallocate ( qmmm_struct%dxyzcl, stat = ier )
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','dxyzcl')

       deallocate ( qmmm_struct%dxyzqm, stat = ier )
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','dxyzqm')

       deallocate ( qmmm_struct%qm_int_scratch, stat = ier )
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_int_scratch')

       deallocate ( qmmm_struct%qm_real_scratch, stat = ier )
       if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_real_scratch')
      end if

      deallocate ( qmmm_struct%scaled_mm_charges, stat = ier )
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','scaled_mm_charges')

      deallocate ( qmmm_struct%atom_mask, stat = ier)
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','atom_mask')

      deallocate ( qmmm_nml%iqmatoms, stat = ier)
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','iqmatoms')

      deallocate ( qmmm_struct%iqm_atomic_numbers, stat = ier)
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','iqm_atomic_numbers')

      deallocate (qmmm_struct%qm_atom_type, stat = ier)
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','qm_atom_type')

      deallocate ( qmmm_struct%link_pairs, stat = ier)
      if(ier.ne.0) call Aass(0,'deallocate_qmmm','link_pairs')


      if (master .OR. neb.GT.0) then
       deallocate ( qmmm_struct%qm_mm_pair_list, stat = ier)
       if(ier.ne.0) call Aass(0,'qm_mm_pair_list','qm_mm_pair_list')
      end if

      return
      END SUBROUTINE deallocate_qmmm
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!!!End of Memory Allocation/Deallocation/Reallocation routines
!!!
!!!*********************************************************************


!END SUBROUTINES

#endif /* (mainsquatn)*/
      END MODULE QMMM_MODULE

