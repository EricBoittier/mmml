!---------------------------------------------------------
! Main Module for QMMM - Written by Ross Walker (SDSC)
! and Michael Crowley (NREL), Oct 2008.
!
! This implementation is based on the AMBER QM/MM
! implementation written by Ross Walker, Michael Crowley
! and David Case and described in: 
!
! Walker, R.C., Crowley, M.F., Case, D.A., "The Implementation
! of a Fast and Accurate QM/MM Potential Method in AMBER."
! J. Comp. Chem. 2008, 29, 1019-1031
!
! Please cite this paper if you make use of this QMMM support
! in your calculations / Work.
!---------------------------------------------------------

module qmmmsemi 
#if KEY_QMMMSEMI==1 /*qmmmsemi*/
  use dimens_fcm
  use chm_kinds
  !--------- QMMM SPECIFIC CONSTANTS --------
  real(chm_real), parameter :: BOHRS_TO_A = 0.529177249D0
  ! Bohrs * this = angstroms - Same constants as used in dynamo v2.

  !real(chm_real), parameter :: BOHRS_TO_A = 0.52917706D0
  ! Bohrs * this = angstroms - Same constants as used in Gaussian 98

  !real(chm_real), parameter :: BOHRS_TO_A = 0.529177D0
  ! Bohrs * this = angstroms - Same constants as used in Mopac6 hcore.f

  !real(chm_real), parameter :: BOHRS_TO_A = 0.529167D0 
  ! as used in Mopac6 repp.f

  real(chm_real), parameter :: A_TO_BOHRS = 1.0d0 / BOHRS_TO_A

  !real(chm_real), parameter :: A_TO_BOHRS = 1.88976D0 ! Same constants as used in Mopac6 gover.f

  real(chm_real), parameter :: A2_TO_BOHRS2 = A_TO_BOHRS * A_TO_BOHRS
  ! Mopac6 uses 3.5711928576D0 in gover.f for this.

  real(chm_real), parameter :: A3_TO_BOHRS3 = A2_TO_BOHRS2 * A_TO_BOHRS
  real(chm_real), parameter :: A4_TO_BOHRS4 = A2_TO_BOHRS2 * A2_TO_BOHRS2

  !--------- CUTOFF PARAMETERS ---------------

  ! Exponential decay factor for the MM atom in the core-core interaction
  real(chm_real), parameter ::  ALPH_MM = 5.0d0

  real(chm_real), parameter :: AXIS_TOL = 1.0d-8  !Tolerance at which to define a vector is along the axis.
  real(chm_real), parameter :: OVERLAP_CUTOFF = 100.0d0*A2_TO_BOHRS2 !Distance^2 in bohrs at which to assume
  !Gaussian overlap is zero.
  real(chm_real), parameter :: EXPONENTIAL_CUTOFF = 30.0d0 !Value of x at which to assume Exp(-x) = zero.


  ! ---- Common constants and definitions ----

  integer, parameter :: PM3 = 1
  integer, parameter :: AM1 = 2
  integer, parameter :: MNDO = 3
  integer, parameter :: PDDGPM3 = 4
  integer, parameter :: PDDGMNDO = 5
  integer, parameter :: PM3CARB1 = 6
  integer, parameter :: DFTB = 7
  integer, parameter :: RM1 = 8
  integer, parameter :: PDDGPM3_08 = 9

  !-------------------------------------------

  !---------------- ELEMENTS -----------------

  !Total number of elements
  INTEGER, PARAMETER :: nelements = 86  !see also parameter in parameters.h

  ! . The element symbols.
  CHARACTER ( LEN = 2 ), DIMENSION(1:nelements), PARAMETER :: ELEMENT_SYM = (/ &
       'H ', 'He', &
       'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne',  &
       'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',  &
       'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  &
       'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', &
       'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', &
       'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', &
       'Cs', 'Ba', 'La', &
       'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', &
       'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  &
       'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'/)

  !-------------------------------------------


  !--------- SIZE / ARRAY LIMITATIONS -----------------------

  ! Locks maximum valence orbitals at 4 = S,P (no D or F - code is not present)
  integer, parameter :: MAX_VALENCE_ORBITALS =4
  integer, parameter :: MAX_VALENCE_DIMENSION = MAX_VALENCE_ORBITALS*(MAX_VALENCE_ORBITALS+1)/2

  !----------------------------------------------------------

  !GLOBAL VARIABLES / ARRAYS - AVAILABLE TO ANY CODE THAT USES THIS MODULE


  ! Remember to modify qmmm_mpi_setup() below if you modify this structure
  ! Contains QMMM namelist variables - all cpus should have this info available.
  type qmmm_namelist
     !--- flag for qmmmsemi is on or off
     logical :: ifqnt

     !--- qm atom numbers as listed in the psf (initially nquant)
     !    later redimensioned to append link atoms on end
     !    The nlink atoms are the mm atoms linked, called mm link pair atoms,
     !                                                     to the qm region.
     integer,dimension(:),allocatable :: iqmatoms

     real(chm_real) :: qmcut     ! Cutoff in angstroms for QM-MM electrostatics -
     ! default = same as MM-MM cutoff

     real(chm_real) :: qmcut2    ! Cutoff^2 - calculated from qmcut.

     real(chm_real) :: lnk_dis   ! Distance in angstroms between QM atom and link atom
     ! A value of <0.0 means the link atom gets placed on 
     ! the MM link pair atom's coordinates on every MD step.

     real(chm_real) :: scfconv   ! SCF Convergence criteria for SCF routine 
     ! - Minimum (tightest conv) = 1.0D-12, max = 1.0D0
     ! Default = 1.0D-8

     real(chm_real) :: density_conv ! Convergence criteria for density matrix. 
     ! If tight_p_conv = true then = scfconv
     ! else it is 0.5*sqrt(scf_conv)

     real(chm_real)  :: pseudo_diag_criteria   ! Criteria for the maximum change in the
     ! density matrix between two scf iterations
     ! before allowing pseudo diagonalisations.
     ! Default = 0.05.

     integer :: lnk_atomic_no ! Atomic number of link atom.

     integer :: lnk_method    ! This defines how classical valence terms that cross 
     ! the QM/MM boundary are dealt with.
     ! 1 = (Default) in this case any bond, angle or dihedral 
     !     that involves at least one
     !     MM atom, including the MM link pair atom is included. 
     !     This means the following
     !     where QM = QM atom, MM = MM atom, MML = MM link pair atom.
     !     Bonds = MM-MM, MM-MML, MML-QM
     !    Angles = MM-MM-MM, MM-MM-MML, MM-MML-QM, MML-QM-QM
     ! Dihedrals = MM-MM-MM-MM, MM-MM-MM-MML, MM-MM-MM-MML-QM, 
     !            MM-MML-QM-QM, MML-QM-QM-QM
     ! 2 = Only include valence terms that include a full MM atom. 
     !     I.e count the MM link pair atom as effectively being a 
     !     QM atom. This therefore gives
     !     Bonds = MM-MM, MM-MML
     !    Angles = MM-MM-MM, MM-MM-MML, MM-MML-QM
     ! Dihedrals = MM-MM-MM-MM, MM-MM-MM-MML, MM-MM-MML-QM, MM-MML-QM-QM

     integer :: qmgb  ! Method for GB in QMMM (when igb /= 0 and /= 6) 
     !  - 0 (default) leave QM charges at 0
     !      effectively doing gas phase QM (EGB(MM-MM only).
     !  - 1 use resp charges from prmtop for qmmm charges.
     !  - 2 use mulliken charges for QM atoms consistent with GB field, modified
     !      fock matrix.
     !  - 3 (FOR_DEBUG_ ONLY) use mulliken charges from gas phase QMMM calc,
     !      not consistent with GB field and so gradients will not be accurate.

     integer :: qmtheory  ! Level of theory to use for QM region (Hamiltonian)
     !           1 = PM3, 2 = AM1, 3 = MNDO (Default = PM3)
     !           4 = PDDG/PM3, 5 = PDDG/MNDO, 6 = PM3CARB1
     !           7 = DFTB,     8 = RM1

     integer :: qmcharge  ! Charge of the QM region in electron units.
     ! Must be an integer charge. (Default = 0).

     integer :: spin      ! Spin state - default = 1 (singlet).
     ! Current Valid values = 1 (alternate options not currently available).

     integer :: verbosity ! Controls amount of info printed about qm part of calc.
     ! (Default = 0).

     integer :: itrmax    ! Maximum number of SCF cycles to conduct before assuming
     ! convergence has failed. (Default = 1000).

     integer :: qmshake   ! Whether to shake qm atoms if ntc>1 (default = 1 - shake QM atoms).
     ! 0 = do not shake.

     integer :: kmaxqx, kmaxqy, kmaxqz ! Used for qmewald - Maximum K space vectors.

     integer :: ksqmaxq                ! Used for qmewald - Maximum K squared values
     !                    for spherical cutoff in k space.

     integer :: qm_ewald ! Flag for doing an ewald sum for periodic QM-QM interactions.
     ! Default = 1.

     integer :: qmmm_int  ! Flag controlling the way QM-MM interactions are handled:
     !   0 = mechanical embedding. No electrostatic QM-MM is calculated.
     !       Only interactions come from VDW and bonds, angles and dihedrals
     !       that cross the boundary.
     !   1 = Full QM-MM interaction is calculated by placing a 1S gaussian
     !       orbital on the MM atom and calculating the full multipole
     !       interaction (Default).
     !   2 = As 1 but also include extra Gaussian core-core terms if AM1,
     !       PM3 or derivative type hamiltonians are in used.

     integer :: adjust_q  ! Flag for whether to adjust the charges of certain MM atoms so
     ! that charge is conserved.
     !   0 = No charge adjustment
     !   1 = Missing charge is distributed over the nearest atom to
     !       each MM link pair.
     !   2 = Missing charge is distributed evenly over all MM atoms.
     !       Excluding MM link pairs. (default)

     integer :: diag_routine ! Flag controlling which diagonalization routine to use when
     ! doing full diag in SCF.
     !  0 = Automatically pick fastest routine.
     !  1 = Use internal diagonalization routine. (default)
     !  2 = Use lapack dspev.
     !  3 = Use lapack dspevd.
     !  4 = Use lapack dspevx.
     !  5 = Use lapack dsyev.
     !  6 = Use lapack dsyevd.
     !  7 = Use lapack dsyevr. (not currently implemented)
     !  8 = Use lapack dsyevx. (not currently implemented)

     integer :: density_predict ! Flag controlling the way in which the density matrix for
     ! MD step t+dt is predicted based on previous MD steps.
     ! 0 = Use final converged density matrix from previous
     !     MD step. (default)
     ! 1 = Use predictive algorithm based on Phys. Rev. Lett.,
     !     2006, 97, 123001. Here Pguess(t) = 2Pconv(t-dt) - Pguess(t-2dt)

     integer :: nearest_qm_solvent ! Do we continually update the nearest solvent molecules 
     ! (e.g. Resname WAT) to keep as QM.
     ! Default = 0 do not process nearest solvent molecules.
     ! >0 = keep this many nearest solvent molecules as QM.

     integer :: nearest_qm_solvent_fq ! Frequency with which to check for nearest solvent when
     ! nearest_qm_solvent > 0
     ! Default = 1 on every step
     ! Must be > 0.

     character(len=4) :: nearest_qm_solvent_resname ! The residue name of the solvent molecules
     ! you want considered as QM.
     ! e.g. = WAT

     integer :: fock_predict ! Flag controlling the way in which the fock matrix for MD step
     ! t+dt is predicted based on previous MD steps.
     ! 0 = Do not attempt to predict the Fock matrix. (default)
     ! 1 = Use predictive algorithm based on a modification of Chem.
     !     Phys. Lett., 2004, 386, 272
     !     by Ross Walker and Gustavo Seabra to only extrapolate the
     !     electronic part of the Fock matrix. Incompatible with
     !     density_predict > 0.

     real(chm_real) :: fockp_d1
     real(chm_real) :: fockp_d2
     real(chm_real) :: fockp_d3
     real(chm_real) :: fockp_d4 ! Prefactor to multiply each final stored fock matrix by when
     ! building the new predicted fock matrix.

     ! DFTB Specific options
     real(chm_real)  :: chg_lambda ! Charge scaling for Free Energy calculation with DFTB

     integer :: dftb_maxiter    ! Maximum number of SCC iterations before resetting Broyden
     ! (default: 70)

     integer :: dftb_disper     ! Use dispersion correction (default: 0 = false).

     integer :: dftb_chg        ! DFTB Charge output: (default: 0 = Mulliken, 1 = CM3)

     real(chm_real)  :: dftb_telec ! Electronic temperature, in Kelvins. Default: 0.0K

     real(chm_real)  :: dftb_telec_step ! Step size for automatic telec increse, for
     ! convergence. Default = 0.0K
     !End DFT specific options

     logical :: qmqm_analyt ! Flag for analytical vs (pseudo) numerical qm-qm derivates in
     ! qm2 routines. (Default = true, analytical).

     logical :: tight_p_conv ! Flag to control convergence criteria for density matrix. If
     ! 0 SCF routine will converge energy to SCFCRT (def = 1*10^-8)
     ! and density to 0.05 * Sqrt(SCFCONV).
     ! If 1 then both energy and density will be converged to SCFCONV.

     logical :: printcharges ! Flag to control the printing of mulliken, cm1a and cm2a charges.
     ! If set to true (1) then charges are calculated and printed on
     ! every step. Otherwise no charges are printed.

     logical :: peptide_corr ! Default = .false. - add correction to peptide linkages.

     logical :: qmqm_erep_incore ! Store qmqm 1-electron repulsion integrals in memory.

     logical :: allow_pseudo_diag ! Whether or not to allow pseudo diagonalisations in SCF
     ! when possible.

     logical :: qmqmrij_incore ! Flag to store qmqm rij and related equations in memory.
     ! Default = true.

     logical :: qmmmrij_incore ! Flag to store qmmm rij and related equations in memory.
     ! Default = true.

     logical :: writepdb ! If true then a crude pdb of the qm system will be written to
     ! file qmmm_region.pdb

     logical :: qm_pme   ! Flag for using PME instead of regular ewald for QM-MM interactions
     ! when qm_ewald>0. Default = true


  end type qmmm_namelist

  type(qmmm_namelist),save :: qmmm_nml

  type qmmm_structure
     real(chm_real) :: enuclr_qmqm, enuclr_qmmm ! Calculated Core-Core energy for QM-QM 
     ! nuclei-nuclei and QM nuclei - MM charge
     ! interactions in (eV).

     real(chm_real) :: elec_eng                 ! Electronic energy (in eV).

     real(chm_real), dimension(:), pointer :: qm_resp_charges  ! Nquant long - contains the
     ! original resp charges for the
     ! QM atoms as read from the prmtop
     ! file.

     real(chm_real) :: qm_resp_charge_sum ! Sum of the resp charges making up the quantum
     ! region. - In AMBERELECTROSTATIC UNITS.

     real(chm_real), dimension(:), pointer :: mm_link_pair_resp_charges

     real(chm_real), dimension(:,:), pointer :: mm_link_pair_saved_coords ! The coordinates of
     ! the mm link pair atoms as they would be for
     ! a given MD step in amber's unimaged coordinate
     ! array. Extracted by position link atoms and
     ! restored back into the x array by
     ! restore_mm_link_pair_coords.

     real(chm_real), dimension(:,:), pointer :: qm_coords ! Cartesian coordinates of ALL 
     ! (real+link) qm atoms [3*(nquant+nlink)
     ! long]

     real(chm_real), dimension(:), pointer :: scaled_mm_charges ! MM charges scaled by one scale
     ! to make them electron units.

     real(chm_real), dimension(:,:), pointer :: dxyzqm, dxyzcl  ! Used to store the forces generated
     ! by qm_mm before adding them to the
     ! main f array.

     real(chm_real), dimension(:,:), pointer :: qm_xcrd         ! Contains imaged mm coordinates and
     ! scaled mm charges.

     integer :: nquant    ! Total number of quantum atoms (excluding link atoms).

     integer :: nlink     ! Total number of link atoms.

     integer :: nquant_nlink ! Total number of quantum atoms = nquant+nlink

     integer :: qm_ntypes    ! The number of atom types present.

     integer, dimension(nelements) :: qm_type_id  ! The id of each type, essentially the atomic
     ! number of that type, e.g. type 1 may have
     ! atomic number = 8.

     integer, dimension(:), pointer :: qm_atom_type ! The type of each qm atom, essentially a
     ! re-basing of the atomic numbers to minimise
     ! memory usage.

     integer, dimension(:,:), pointer :: link_pairs ! list of MM-QM atoms for which link atoms
     ! were added.

     integer, dimension(:), pointer :: iqm_atomic_numbers ! integer list of atomic numbers for
     ! the qm atoms.

     integer                        :: qm_mm_pairs  ! Number of pairs per QM atom. - length
     ! of pair_list.

     integer, dimension(:), pointer :: qm_mm_pair_list ! Non bond pair list for each QM atom.

     integer :: num_qmmm_calls ! Counter of the number of times qm_mm has been called
     ! - effectively nstep.

     integer :: prmtop_numbnd ! Number of bond types as read from the prmtop before any QM
     ! modifications are made - needed to be able to call setbon
     ! multiple times and not have new QM-H atom types keep being created.

     logical, dimension(:), pointer :: atom_mask ! True / false mask specifying if atom is a QM atom.
     ! True = QM atom (natom long).

     logical, dimension(:), pointer :: mm_link_mask ! True / false mask specifying if atom is a MM
     ! link pair atom. True = MM link pair atom
     ! (natom long)

     logical :: qm_mm_first_call ! Set to true at beginning of sander subroutine and then set to
     ! false at the end of qm_mm. Used for allocation purposes.

     logical :: fock_first_call ! Set to true at beginning of sander subroutine and then set to
     ! false at the end of qm_mm. Used for allocation purposes.

     logical :: fock2_2atm_first_call
     logical :: qm2_allocate_e_repul_first_call
     logical :: qm2_calc_rij_eqns_first_call
     logical :: qm2_scf_first_call
     logical :: zero_link_charges_first_call
     logical :: adj_mm_link_pair_crd_first_call

     logical :: AM1_OR_PM3 ! Set to True if theory is AM1, PM3, PDDG/PM3, PM3CARB1, RM1
     logical :: PDDG_IN_USE ! Set to True if theory is PDDG/PM3 or PDDG/MNDO

     logical :: mmcoords_contains_lnk_coords ! Set to true if the coordinates of the MM link pair
     ! atoms in amber's main coordinate array have been set
     ! to the link atom coordinates. This acts as a safety
     ! to ensure that the adj link atoms is not called
     ! without having restored them.

  end type qmmm_structure

  type(qmmm_structure) :: qmmm_struct

  type qm2_structure  ! Variables that are specific to qm_routine=2 (qm2)
     real(chm_real), dimension(:), pointer :: den_matrix  ! The total density matrix.

     real(chm_real), dimension(:), pointer :: old_den_matrix ! The old total density matrix from
     ! preious step allocated in
     ! qm2_load_params on first call - 
     ! deallocated by deallocate_qmmm.

     real(chm_real), dimension(:), pointer :: old2_density ! Used by qm2_cnvg as workspace, norbs.

     real(chm_real), dimension(:), pointer :: md_den_mat_guess1 ! These two guesses are only used
     ! when density_predict=1.

     real(chm_real), dimension(:), pointer :: md_den_mat_guess2 ! They contain Pguess(t-1) and
     ! Pguess(t-2).

     real(chm_real), dimension(:), pointer :: fock_mat_final4
     real(chm_real), dimension(:), pointer :: fock_mat_final3
     real(chm_real), dimension(:), pointer :: fock_mat_final2
     real(chm_real), dimension(:), pointer :: fock_mat_final1 ! These contain the final Fock matrices
     ! from previous MD steps in the case of
     ! fock_predict=1 it contains the
     ! previous 4 MD step fock matrices.
     ! F4 = t-4, F3 = t-3, F2 = t-2, F1 = t-1.

     real(chm_real), dimension(:), pointer :: fock_matrix ! Fock matrix.

     real(chm_real), dimension(:,:), pointer :: qm_mm_e_repul ! Array containing the QM-MM electron
     ! repulsion integrals.

     real(chm_real), dimension(:), pointer :: qm_qm_2e_repul  ! Array containing the QM-QM 2-electron
     ! repulsion integrals.
     ! This is a big array, allocated by qm_mm and deallocated
     ! by deallocate_qmmm - it needs to be a total of n2el long.

     real(chm_real), dimension(:), pointer :: hmatrix ! The 1-electron matrix - used by routines
     ! called from qm2_energy allocated in qm_mm on
     ! first call - deallocated by  deallocate_qmmm.

     real(chm_real), dimension(:,:), pointer :: qm_qm_e_repul ! Array containing the QM-QM electron
     ! repulsion integrals.
     ! This was originally written as a file to disk in the energy
     ! routines and then re-read in the derivative routines. Now
     ! it is stored in memory. Allocated in qm_mm on first call
     ! - deallocated by deallocate_qmmm

     real(chm_real), dimension(:,:), pointer :: fock2_ptot2 ! Used in qm2_fock2 routine
     ! 16,nquant_nlink allocated in qm2_load_params
     ! deallocated in deallocate_qmmm.

     real(chm_real), dimension(:,:), pointer :: eigen_vectors ! Holds the eigen vectors during the
     ! SCF - allocated in qm2_load_params.

     real(chm_real), dimension(:), pointer :: scf_mchg ! Hold the mulliken charges at each scf step
     ! if calc_mchg_scf is true.
     ! Otherwise only do it at the end of the scf.

     integer :: matsize          ! Size of the various packed symmetric matrices. (norbs(norbs+1)/2)

     integer :: n2el             ! Number of 2 electron repulsion integrals, calculated by
     ! moldat = 50*nheavy(nheavy-1)+10*nheavy*nlight+(nlight*(nlight-1))/2

     integer :: norbs  ! Total Number of atomic orbitals.

     integer :: nclosed ! Number of doubly occupied orbitals.

     integer :: nopenclosed ! Number of doubly occupied and singly occupied orbitals.

     integer :: qm_mm_e_repul_allocated ! This was originally written as a file to disk in the energy
     ! routines and then re-read in the derivative routines. Now
     ! it is stored in memory. Allocated in qm_mm on first call
     ! or if pair list changes too much. See qm_mm_e_repul array
     ! above - deallocated by deallocate_qmmm.

     integer :: n_peptide_links ! Number of peptide linkages in QM region to apply MM correction to.

     integer, dimension(:,:), pointer :: peptide_links ! Identity of peptide linkages
     ! 4,n_peptide_linkages. 1 to 4 = H-N-C-O atom
     ! numbers.

     logical :: calc_mchg_scf ! If set to true the mulliken charges will be calculated on each
     ! SCF iteration.

  end type qm2_structure

  type ( qm2_structure ) qm2_struct

  type qm2_params_structure ! Parameters for each atom in the qm region.
     real(chm_real) :: tot_heat_form ! Calculated in qm2_load_params - independent of structure so
     ! constant for a sander run.

     real(chm_real), dimension(:), pointer :: core_chg ! The core charge on each atom as seen by
     ! the electrons allocated as nquant_nlink
     ! long by qm2_load_params - deallocated by
     ! deallocate_qmmm.

     real(chm_real), dimension(:,:), pointer :: orb_elec_ke ! Orbital electron kinetic energy
     ! integrals (2,nquant_nlink) - allocated in
     ! qm2_load_params. Deallocated by deallocate
     ! qmmm (1=s, 2=p)

     real(chm_real), dimension(:,:), pointer :: betasas ! betas(ni)+betas(nj) qm_ntypes x qm_ntypes.

     real(chm_real), dimension(:,:), pointer :: betasap ! betas(ni)+betap(nj) qm_ntypes x qm_ntypes.

     real(chm_real), dimension(:,:), pointer :: betapap ! betap(ni)+betap(nj) qm_ntypes x qm_ntypes.

     real(chm_real), dimension(:,:), pointer :: FN1 ! PM3 / AM1 specific parameters for
     ! core-core repulsions.
     real(chm_real), dimension(:,:), pointer :: FN2
     real(chm_real), dimension(:,:), pointer :: FN3

     real(chm_real), dimension(:,:), pointer :: onec2elec_params
     ! Coulomb and exchange one centre-two electron integral params.
     ! Allocated as nquant_nlink long in qm2_load_params. Deallocated
     ! in deallocate QMMM.

     real(chm_real), dimension(:,:), pointer :: multip_2c_elec_params ! Parameters for the multipole
     ! expansion of the 2 centre 2 electron
     ! integerals. 9, nquant_nlink in order
     ! DD,QQ,AM,AD,AQ,AM2,AD2,AQ2 allocated in
     ! qm2_load_params.

     real(chm_real), dimension(:), pointer :: cc_exp_params ! Exponents for core core repulsions.

     real(chm_real), dimension(:), pointer :: s_orb_exp_by_type ! S orbital expansion coefficients
     ! for the Slater orbital expansion.

     real(chm_real), dimension(:), pointer :: p_orb_exp_by_type ! P orbital expansion coefficients
     ! for the Slater orbital expansion
     ! both are allocated as qm_ntypes long by qm2_load_params
     ! deallocated by qm2_setup_orb_exp as it is not needed after this
     ! is called.

     ! Arrays for PDDG Hamiltonians
     real(chm_real), dimension(:), pointer :: pddge1, pddge2

     ! Arrays for pre-computed orbital interactions
     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_zz_sxs_over_sas 
     ! atom_orb_zz_s_x_s/atom_orb_zz_one_s_a_s

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_zz_sxp_over_sap
     ! atom_orb_zz_s_x_p/atom_orb_zz_one_s_a_p

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_zz_pxp_over_pap
     ! atom_orb_zz_p_x_p/atom_orb_zz_one_p_a_p

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_ss_eqn 
     ! sqrt((two*sqrt(atom_orb_zz_s_x_s)*atom_orb_zz_one_s_a_s)**3
     ! *atom_orb_cc_s_x_s

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_sp_ovlp
     ! 2.0D0 * qm2_params%atom_orb_zz_s_by_type(K,qmitype)*
     ! SQRT(qm2_params%atom_orb_zz_p_by_type(L,qmjtype))*
     ! atom_orb_zz_one_s_a_p(k,l,qmitype,qmjtype)
     ! *atom_orb_sp_eqn
     ! Used in gover for Si-Pj overlap energy and -Sj-Pi overlap.

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_pp_ovlp_inj 
     ! -4.0D0*sqrt(atom_orb_zz_p_x_p)*
     ! atom_orb_zz_one_p_a_p*
     ! qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)*
     ! atom_orb_pp_eqn
     ! Used in gover for Pi-Pj overlap energy when i!=j

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_pp_ovlp_ieqj1 
     ! -4.0D0*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)*
     ! atom_orb_zz_one_p_a_p*
     ! qm2_params%atom_orb_zz_pxp_over_pap(k,l,qmitype,qmjtype)
     ! Used in gover for Pi-Pj overlap energy when i==j.

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_pp_ovlp_ieqj2
     ! 2.0D0*atom_orb_pp_eqn*sqrt(atom_orb_zz_p_x_p)*atom_orb_zz_one_p_a_p

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_ss_eqn_adb
     ! -2.0D0*A2_TO_BOHRS2*qm2_params%atom_orb_ss_eqn_adb(i,j,qmitype,qmjtype)*
     ! qm2_params%atom_orb_zz_sxs_over_sas(i,j,qmitype,qmjtype)
     ! Used for S-S overlap in QM-QM derivatives

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_sp_eqn_xy 
     ! -four*A3_TO_BOHRS3*qm2_params%atom_orb_zz_sxp_over_sap(i,j,qmitype,qmjtype)**2* &
     ! (one/(SQRT(qm2_params%atom_orb_zz_p_by_type(J,qmjtype))))*atom_orb_sp_eqn
     ! Used for S-P overlap in QM-QM derivatives where P... /= axis

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_sp_eqn_xx1
     ! Used for S-P overlap in QM-QM derivatives where P... == axis

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_sp_eqn_xx2

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_pp_eqn_xxy1
     ! -four*A2_TO_BOHRS2*(ADB_array(inner_index)**2)*
     ! (one/(SQRT(atom_orb_zz_p_x_p)))*atom_orb_pp_eqn
     ! Used for P-P overlap in QM-QM derivatives where P... = P... /= axis.

     real(chm_real), dimension(:,:,:,:), pointer :: atom_orb_pp_eqn_xxy2
     ! eight*A2_TO_BOHRS2*A2_TO_BOHRS2*(ADB_array(inner_index)**3)*
     ! (one/(SQRT(atom_orb_zz_p_x_p)))*atom_orb_pp_eqn

     ! End arrays for pre-computed orbital interactions

     ! Pre-computed PDDG parameters if PDDG Hamiltonian is in use.
     real(chm_real), dimension(:,:), pointer :: pddg_term1, pddg_term2, pddg_term3, pddg_term4
     ! Ntypes*ntypes - allocated in qm2_load_params if PDDG is in use. Stores the
     ! pre-exponential part of the PDDG equation.

     integer, dimension(:), pointer :: natomic_orbs ! Number of atomic orbitals on atom.

     integer, dimension(:,:), pointer :: orb_loc    ! Locations of orbitals. 2,nquant_nlink.
     ! 1,x gives beginning of orbitals on atom x.
     ! 2,x gives last orbital on atom x.

     integer, dimension(:), pointer :: pascal_tri1 ! Lower half triangle indices (pascal's triangle)

     integer, dimension(:), pointer :: pascal_tri2 ! Allocated in load params.

     integer, dimension(:), pointer :: NUM_FN ! Number of FNX terms (first dimension) that are not
     ! zero.

  end type qm2_params_structure

  type (qm2_params_structure) qm2_params

  type  qm2_rij_eqns_structure ! This structure is used to store RIJ info for each
     ! QM-QM pair and related equations
     ! QM-QM
     real(chm_real), dimension(:,:), pointer :: qmqmrijdata ! These two arrays store QMQM and QMMM RIJ
     ! values and related equations. See
     ! array_locations.h for the first
     ! dimension.

     real(chm_real), dimension(:,:), pointer ::  qmmmrijdata ! offsets.

     integer :: qmmmrij_allocated

  end type qm2_rij_eqns_structure

  type (qm2_rij_eqns_structure) qm2_rij_eqns

  ! QMEwald specific structure
  type qm_ewald_structure
     real(chm_real), dimension(:), pointer :: kvec ! Array storing K vectors (totkq long)

     real(chm_real), dimension(:,:), pointer :: dkvec ! Array storing derivative K vectors
     ! (3,totkq long)

     real(chm_real), dimension(:,:), pointer :: dmkv ! used for calculating the ktable

     real(chm_real), dimension(:,:,:), pointer :: ktable ! Table for storing complex exp(ik,r[j])
     ! dimensions = 6,natom,totkq
     ! 1,x,y = x_cos
     ! 2,x,y = x_sin
     ! 3,x,y = y_cos
     ! 4,x,y = y_sin
     ! 5,x,y = z_cos
     ! 6,x,y = z_sin

     real(chm_real), dimension(:,:,:), pointer :: qmktable ! As Ktable but stores the qmatom copies
     ! in a linear 1->nquant fashion.

     real(chm_real), dimension(:), pointer :: mmpot ! Nquant long, stores the potential at each QM
     ! atom due to the MM field.

     real(chm_real), dimension(:), pointer :: qmpot ! Nquant long, stores the self energy of the QM
     ! atoms to avoid double counting.

     real(chm_real), dimension(:,:), pointer :: d_ewald_mm ! 3,natom long stores gradients on MM
     ! atoms due to QM-MM Ewald field.
     ! Reciprocal forces.

     real(chm_real) :: ewald_core ! Ewald Potential with QM CORE charges - energy in eV.

     real(chm_real) :: mm_recip_e ! Reciprocal energy from MM atoms - qm_pme.

     integer :: totkq  !Total number of kspace vectors.

     integer :: natom  !Same as sander's natom, copied here by qm_mm for convenience.

     logical :: ewald_startup !True if this is the very first MD step and we are doing qmewald.
  end type qm_ewald_structure

  type (qm_ewald_structure) qmewald

  type qm_gb_structure
     real(chm_real), dimension(:), pointer :: qmqm_onefij ! Stores the 1.0/fij equations for qm-qm
     ! pairs. Since these values only depend
     ! on Rij and the effective radii
     ! they remain fixed during the SCF and
     ! so are calculated once outside the SCF
     ! and then reused inside.

     real(chm_real), dimension(:), pointer :: qmqm_kappafij ! Stores exp(-kappa*fij) - These are
     ! calculated outside of the scf and
     ! reused inside. Only allocated and
     ! calculated if kappa/=0.0d0 in other
     ! words saltcon /= 0.0d0.

     real(chm_real), dimension(:), pointer :: gb_mmpot ! Nquant long, stores GB potential at each QM
     ! atom due to MM atoms.

     real(chm_real), dimension(:), pointer :: gb_qmpot ! Nquant long, stores GB potential at each QM
     ! atom due to QM atoms.

     real(chm_real) :: intdieli, extdieli ! 1.0d0/intdiel and extdiel respectively.

     real(chm_real) :: kappa ! Debye-Huckel kappa (A**-1) from salt concentration (M), assuming:
     ! T = 298.15, epsext=78.5, kappa = sqrt( 0.10806d0 * saltcon )
     ! scaled kappa by 0.73 to account(?) for lack of ion exlcusions.

     real(chm_real) :: mmcut2   ! cut^2 in angstroms^2 from cntrl namelist.

     real(chm_real) :: one_Arad_beta ! alpb_beta/Arad when alpb/=0.

     integer, dimension(:,:), pointer :: qmqm_gb_list ! 1+nquant,nquant - list of qm atoms that
     ! interact with current qm atom.
     ! +1 because the first entry stores the number
     ! of interactions for QM atom y.

     logical :: saltcon_on ! True if saltcon /= 0.0d0
     logical :: alpb_on ! True if alpb = 1
  end type qm_gb_structure

  type (qm_gb_structure) qm_gb

  type qmmm_mpi_structure
     integer :: commqmmm ! Communications within a given set of QMMM threads potentially
     ! a subset of processors of commsander.

     integer :: numthreads ! Number of threads in commqmmm.

     integer :: mytaskid   ! Task id of this thread in commqmmm.

     integer :: openmp_numthreads ! When openmp is in use this will contain how many threads
     ! would be spawned by the master.

     integer :: natom_start ! Where this thread should start for 1,natom loops.

     integer :: natom_end   ! Where this thread should end for 1,natom loops.

     integer :: nquant_nlink_start ! Where this thread should start for 1,nquant_nlink loops.

     integer :: nquant_nlink_end ! Where this thread should end for 1,nquant_nlink loops.

     integer :: totkq_count

     integer :: kvec_start ! Where this thread starts and ends in 1 to totkq loops.

     integer :: kvec_end

     integer :: two_e_offset ! Offset into two electron matrix for this thread

     !Below are for matrix type double loop load balancing
     integer ::                        nquant_nlink_istart ! These three control load balancing
     integer ::                        nquant_nlink_iend
     integer :: nquant_nlink_loop_extent_begin 
     integer :: nquant_nlink_loop_extent_end 

     integer, dimension(:,:), pointer :: nquant_nlink_jrange ! within a do i = 1, nquant_nlink
     !           do j=1,i-1
     ! loop. Basically istart and iend define
     ! the extent of the outer loop and then
     ! (1,i) and (2,i) of jrange define the
     ! limits of the inner loop for this
     ! processor.

     logical :: commqmmm_master ! True if master thread of commqmmm

  end type qmmm_mpi_structure

  type (qmmm_mpi_structure) qmmm_mpi

  type qmmm_scratch_structure
     ! Various scratch arrays used as part of QMMM - one should typically assume that upon leaving a
     ! routine the contents of these arrays can be assumed to be junk.

     real(chm_real), dimension(:), pointer  :: matsize_red_scratch !Allocated as qm2_struct%matsize
     ! when doing MPI during the
     ! load parameters routine. ONLY ALLOCATED IF WE CAN'T
     ! DO MPI_IN_PLACE.
     ! +1 in size so we can pack extra energies on the end etc.

     real(chm_real), dimension(:), pointer :: qm_pme_scratch ! natom long scratch for qm_pme - only
     ! allocated if doing qm_pme.

     real(chm_real), dimension(:,:), pointer :: mat_diag_workspace ! Matrix diagonalisation workspace
     ! - allocated in qm2_load_params
     ! norbs,6 for internal diag
     ! norbs,1 if lapack diag.

     ! The Pseudo diagonalizer needs a total of 5 real scratch arrays and 1 integer scratch array.
     ! Passed in Scratch Arrays - these are all allocated in qm2_load_params and only if
     ! allow_pseudo_diag is true. ONLY ALLOCATED ON COMMQMMM MASTER THREAD
     real(chm_real), dimension(:,:), pointer :: pdiag_scr_norbs_norbs ! (norbs,norbs)
     real(chm_real), dimension(:,:), pointer :: pdiag_scr_noccupied_norbs ! (noccupied,norbs)
     real(chm_real), dimension(:), pointer :: pdiag_vectmp1           ! (noccupied*(norbs-noccupied))
     real(chm_real), dimension(:), pointer :: pdiag_vectmp2           ! (noccupied*(norbs-noccupied))
     real(chm_real), dimension(:), pointer :: pdiag_vectmp3           ! (noccupied*(norbs-noccupied))
     integer, dimension(:,:), pointer :: pdiag_vecjs              ! (2,noccupied*(norbs-noccupied))

     real(chm_real), dimension(:), pointer :: lapack_dc_real_scr
     real(chm_real), dimension(:), pointer :: lapack_dc_int_scr
     !END ONLY ALLOCATED ON COMMQMMM MASTER THREAD

     ! Scratch Arrays - These arrays are available for any subroutine to use - they are allocated
     ! by allocate_qmmm. Each routine that uses one of these arrays for temporary storage should
     ! assume that the contents of such array will be garbage once the routine is left or another
     ! routine is called.

     real(chm_real), dimension(:), pointer :: qm_real_scratch ! Real scratch array - 4*natom.

     integer, dimension(:), pointer :: qm_int_scratch ! Integer scratch array - 3*natom.

     integer :: lapack_dc_real_scr_aloc ! Number of reals allocated for lapack_dc_real_scr.

     integer :: lapack_dc_int_scr_aloc ! Number of ints allocated for lapack_dc_int_scr_aloc.

     integer :: qm_mm_pairs_allocated ! Size of expected qm_mm_pairs that scratch arrays and
     ! calc_rij_array was allocated to.
  end type qmmm_scratch_structure

  type (qmmm_scratch_structure) qmmm_scratch

  type qmmm_vsolv_structure
     ! Various arrays related to having a variable solvent region.
     integer :: fixed_nquant       ! Stores the number of fixed quantum atoms (excluding link atoms).
     integer :: nsolv_res          ! total number of solvent molecules in simulation.
     integer :: natom_solv_res     ! number of atoms per solvent residue.
     integer, dimension(:),pointer :: fixed_iqmatoms ! stores the original sorted iqmatoms array for
     ! the fixed QM region, orig_nquant long.

     integer, dimension(:),pointer :: solvent_pointers ! Stores the first atom of solvent residue 1
     ! to nsolv_res.

     integer, dimension(:),pointer :: nearest_solvent_pointers ! Stores the first atom of solvent
     ! residue 1 to nearest_qm_solvent.
     ! I.e. the first atom number of the
     ! nearest residues to the QM region
     ! that should be treated as QM.

  end type qmmm_vsolv_structure

  type (qmmm_vsolv_structure) qmmm_vsolv

  !END GLOBALS

  !BEGIN SUBROUTINES

  !-------------------------------------------------------------------------------------
contains
  !-------------------------------------------------------------------------------------

  !-------------------------------------------------
  !             STARTUP
  !-------------------------------------------------
  subroutine qmmm_startup(comlyn,comlen,natom,x,y,z)

    use select
    use stream

    implicit none

    ! Passed in
    character(len=*) :: comlyn
    integer :: comlen
    integer,intent(in) :: natom
    real(chm_real),dimension(natom) :: x,y,z,wmain

    ! Local
    integer :: i,n,alloc_err
    integer, allocatable :: qm1_select(:)

    ! ==== Initialise first_call flags for QMMM ====
    qmmm_struct%qm_mm_first_call = .true.
    qmmm_struct%fock_first_call = .true.
    qmmm_struct%fock2_2atm_first_call = .true.
    qmmm_struct%qm2_allocate_e_repul_first_call = .true.
    qmmm_struct%qm2_calc_rij_eqns_first_call = .true.
    qmmm_struct%qm2_scf_first_call = .true.
    qmmm_struct%zero_link_charges_first_call = .true.
    qmmm_struct%adj_mm_link_pair_crd_first_call = .true.
    qmmm_struct%num_qmmm_calls = 0

    qmmm_nml%ifqnt = .true.

    allocate(qm1_select(natom),stat=alloc_err)
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>qmmm_startup', &
         'unable to allocate qm1_select')

    call selcta(comlyn,comlen,qm1_select,x,y,z,wmain,.true.)
    qmmm_struct%nquant = sum(qm1_select)

    if (prnlev > 7) &
         write(outu,'(a,i6)') "QMMM: Selection routine found quantum atoms ",qmmm_struct%nquant

    allocate(qmmm_nml%iqmatoms(qmmm_struct%nquant),stat=alloc_err)
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>iqmatoms', &
         'unable to allocate qm1_select')
    n=0
    do i=1,natom
       if(qm1_select(i) == 1 ) then
          n=n+1
          qmmm_nml%iqmatoms(n) = i
       endif
    enddo

    if (prnlev > 7) then 
       write(outu,'(a)') "QMMM: QM atom list is: "
       write(outu,'(a,8i7)') "QMMM: ",qmmm_nml%iqmatoms(1:qmmm_struct%nquant) 
    end if

    deallocate(qm1_select, stat=alloc_err)
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>qmmm_startup', &
         'unable to deallocate qm1_select')

    !Initialize nlink to 0
    qmmm_struct%nlink = 0
    qmmm_struct%nquant_nlink = qmmm_struct%nquant

    !Check that the QM atom list is valid.
    call validate_qm_atoms(qmmm_nml%iqmatoms,qmmm_struct%nquant,natom)

    !Sort atoms into numerical order
    call qmsort(qmmm_nml%iqmatoms)

    !Set default values for all the run options and make sure they
    !are within the limits.
    call fill_qm_run_parameters(comlyn,comlen)

    !At this point we know nquant and natom so we can allocate our arrays that depend on nquant or natom
    !Note non master mpi threads need to call this allocation routine manually themselves.
    call allocate_qmmm( natom )

    !Work out what the atomic numbers are of each QM atom.
    call assign_qm_atomic_numbers()

    !Check for specific option incompatibilties or code limitations.
    call qm_check_limitations()

    !Print a summary of all the options.
    if (prnlev > 2) call qm_print_option_summary()

    return

  end subroutine qmmm_startup

  !------------------------------------------------------------------
  !               validate_qm_atoms
  !------------------------------------------------------------------
  subroutine validate_qm_atoms(iqmatoms, nquant, natom)
    ! This routine will check the list of atoms numbers stored in iqmatoms
    ! and check the following:
    !
    ! 1) All are >= 1 .and. <= natoms
    ! 2) All are unique integer numbers
    !
    ! Written by Ross Walker, TSRI, 2004
    !
    !=======================================================
    use stream, only : outu

    implicit none

    !Passed in
    integer, intent(in) :: nquant, natom
    integer, intent(in) :: iqmatoms(nquant)

    !Local
    integer :: icount1, icount2, iatom, ier

    ! Sanity check 1, ensure nquant isn't bigger than natom (it can't be)
    if ((nquant < 1) .OR. (nquant > natom)) then
       write (outu,'("QMMM: QM ATOM VALIDATION: &
            &nquant has a value of ",i8)') nquant
       write (outu,'("QMMM: which is bigger than natom of ",i8,". &
            &Need 0 < nquant <= natom.")') natom
       call wrndie(-4,'validate_qm_atoms nquant illegal', &
            &'Need 0 < nquant <= natom')
    end if

    ! Check 2 - loop over nquant atoms and check it is > 1 and <= natom and it is unique
    do icount1=1,nquant
       !check atom number is legal
       iatom = iqmatoms(icount1)
       if ( (iatom > 0) .AND. (iatom <= natom)  ) then
          !QM atom ID is valid - check it is unique
          do icount2=(icount1+1),nquant
             if ( iatom == iqmatoms(icount2) ) then
                write (outu,'("QMMM: QM ATOM VALIDATION: &
                     &qm atom ",i8," is not unique.")') iatom
                call wrndie(-4,'<qmmmsemi.src>validate_qm_atoms',&
                     'QM atoms specified with iqmatoms do not &
                     &form a unique set. &
                     & Require a unique list of qm atom numbers')
             end if
          end do
       else
          !it is not legal
          write (outu,'("QMMM: QM ATOM VALIDATION: iqmatom ID number of &
               &",i8," is not valid.")') iatom
          call wrndie(-4,'<qmmmsemi.src>validate_qm_atoms: &
               &invalid QM atom ID', &
               'Need 0 < QM atom ID <= natom')
       end if
    end do

    return
  end subroutine validate_qm_atoms

  !------------------------------------------------------------------
  !             qmsort
  !------------------------------------------------------------------
  !Sorts the array iqmatoms into numerical order.
  subroutine qmsort(iqmatoms)

    implicit none
    integer, intent(inout) :: iqmatoms(*)

    ! Local
    integer i,j,lcurrent

    !   sort array iqmatoms in ascending order

    do i = 1, qmmm_struct%nquant
       lcurrent = iqmatoms(i)
       do j = i+1,qmmm_struct%nquant
          if (lcurrent.gt.iqmatoms(j)) then
             iqmatoms(i) = iqmatoms(j)
             iqmatoms(j) = lcurrent
             lcurrent = iqmatoms(i)
          endif
       end do
    end do

    return

  end subroutine qmsort


  !------------------------------------------------------------------
  !             SetQMTheory
  !------------------------------------------------------------------
  subroutine setqmtheory(qm_theory, qmtheory)

    implicit none

    character(len=12),intent(in) :: qm_theory
    integer, intent(out) :: qmtheory

    if ( qm_theory == 'PM3' ) then
       qmtheory = PM3
    else if ( qm_theory == 'AM1' ) then
       qmtheory = AM1
    else if ( qm_theory == 'MNDO' ) then
       qmtheory = MNDO
    else if ( qm_theory == 'PM3-PDDG' .OR. qm_theory == 'PM3PDDG' &
         .OR. qm_theory == 'PM3_PDDG' &
         .OR. qm_theory == 'PDDG-PM3' .OR. qm_theory == 'PDDGPM3' &
         .OR. qm_theory == 'PDDG_PM3' ) then
       qmtheory = PDDGPM3
    else if ( qm_theory == 'MNDO-PDDG' .OR. qm_theory == 'MNDOPDDG' &
         .OR. qm_theory == 'MNDO_PDDG' &
         .OR. qm_theory == 'PDDG-MNDO' .OR. qm_theory == 'PDDGMNDO' &
         .OR. qm_theory == 'PDDG_MNDO' ) then
       qmtheory = PDDGMNDO
    else if ( qm_theory == 'PM3-CARB1' .OR. qm_theory == 'PM3CARB1' &
         .OR. qm_theory == 'PM3_CARB1' ) then
       qmtheory = PM3CARB1
    else if ( qm_theory == 'DFTB' .OR. qm_theory == 'SCCDFTB' .OR. &
         qm_theory == 'SCC-DFTB' .OR. &
         qm_theory == 'SCC_DFTB' ) then
       qmtheory = DFTB
    else if ( qm_theory == 'RM1' ) then
       qmtheory = RM1
    else if ( qm_theory == 'PM3-PDDG08' &
         .OR. qm_theory == 'PM3PDDG08' &
         .OR. qm_theory == 'PM3_PDDG08' &
         .OR. qm_theory == 'PDDG-PM308' &
         .OR. qm_theory == 'PDDGPM308' &
         .OR. qm_theory == 'PDDG_PM308' &
         .OR. qm_theory == 'PM3-PDDG-08' &
         .OR. qm_theory == 'PM3-PDDG08' &
         .OR. qm_theory == 'PDDGPM3_08' &
         .OR. qm_theory == 'PDDG_PM3_08' &
         .OR. qm_theory == 'PM3-PDDG_08' ) then
       qmtheory = PDDGPM3_08
    else
       call wrndie(-4,'<qmmmsemi.src>setqmtheory Unknown &
            &method specified for qm_theory', &
            'Valid options are: PM3, AM1, RM1, MNDO, &
            &PM3-PDDG, PM3-PDDG_08, MNDO-PDDG, PM3-CARB1 &
            &and DFTB')
    endif

    return

  end subroutine setqmtheory

  !------------------------------------------------------------------
  !             FILL_QM_RUN_PARAMETERS
  !------------------------------------------------------------------
  subroutine fill_qm_run_parameters(comlyn,comlen)

    use dimens_fcm
    use psf
    use stream, only : prnlev, outu
    use string

    implicit none
    
    !Passed in
    character(len=*) :: comlyn
    integer :: comlen

    !Local
    integer, parameter :: max_quantum_atoms = 10000
    character(len=12) :: qm_theory
    integer :: clen, qmqmdx,tight_p_conv,printcharges,peptide_corr, &
         qmqmrij_incore,qmmmrij_incore,qmqm_erep_incore, &
         pseudo_diag,qm_pme,writepdb

    !--- Read in all options, setting defaults as needed.
    !    Later check for legal values

    !--- Options=PM3,AM1,MNDO,PDDG-PM3,PM3PDDG,PDDG-MNDO, 
    !            PDDGMNDO,PM3-CARB1,PM3CARB1,DFTB,SCC-DFTB, RM1

    qm_theory = "PM3"
    call gtrmwd(comlyn,comlen,"QM_THEORY",9,qm_theory,12,clen)
    call setqmtheory(qm_theory,qmmm_nml%qmtheory)

    if (prnlev > 7) write(outu,'(a,i4)') "QMMM: Using qm_theory = ",qmmm_nml%qmtheory

    !   qmcut = cut (default to 8.0d0 for now).
    qmmm_nml%qmcut = gtrmf(comlyn,comlen,'QMCUT',8.0d0)
    qmmm_nml%qmcut2 = qmmm_nml%qmcut*qmmm_nml%qmcut2

    !   lnk_dis=1.09d0  !Methyl C-H distance
    qmmm_nml%lnk_dis = gtrmf(comlyn,comlen,'LNK_DIS',1.09d0)

    !   lnk_atomic_no=1 !Hydrogen
    qmmm_nml%lnk_atomic_no = gtrmi(comlyn,comlen,'LNK_ATOMIC_NO',1)

    !   lnk_method=1 !treat MMLink as being MM atom.
    qmmm_nml%lnk_method = gtrmi(comlyn,comlen,'LNK_METHOD',1)

    !   qmgb = 2 !Gets set to zero if igb==6 or igb==0.
    qmmm_nml%qmgb = gtrmi(comlyn,comlen,'QMGB',0)

    !   qmcharge = 0
    qmmm_nml%qmcharge = gtrmi(comlyn,comlen,'QMCHARGE',0)

    !   spin = 1
    qmmm_nml%spin = gtrmi(comlyn,comlen,'SPIN',1)    

    !-------qmqmdx is local here but gets copied later to qmqm_analyt
    qmqmdx = gtrmi(comlyn,comlen,'QMQMDX',1)    

    !   verbosity = 0 - Should ultimately be set based on prnlev.
    qmmm_nml%verbosity = gtrmi(comlyn,comlen,'VERBOSITY',0)

    !   tight_p_conv = 0
    tight_p_conv = gtrmi(comlyn,comlen,'TIGHT_P_CONV',0)

    !   scfconv = 1.0D-8
    qmmm_nml%scfconv = gtrmf(comlyn,comlen,'SCFCONV',1.0d-8)

    !   printcharges = 0
    printcharges = gtrmi(comlyn,comlen,'PRINTCHARGES',0)

    !   peptide_corr = 0
    peptide_corr = gtrmi(comlyn,comlen,'PEPTIDE_CORR',0)

    !   itrmax = 1000
    qmmm_nml%itrmax = gtrmi(comlyn,comlen,'ITRMAX',1000)

    !   qmshake = 1
    qmmm_nml%qmshake = gtrmi(comlyn,comlen,'QMSHAKE',1)

    !   qmqmrij_incore = 1
    qmqmrij_incore = gtrmi(comlyn,comlen,'QMQMRIJ_INCORE',1)

    !   qmmmrij_incore = 1
    qmmmrij_incore = gtrmi(comlyn,comlen,'QMMMRIJ_INCORE',1)

    !   qmqm_erep_incore = 1
    qmqm_erep_incore = gtrmi(comlyn,comlen,'QMQM_EREP_INCORE',1)

    !   pseudo_diag = 1
    pseudo_diag = gtrmi(comlyn,comlen,'PSEUDO_DIAG',1)

    !   pseudo_diag_criteria = 0.05
    qmmm_nml%pseudo_diag_criteria = &
         gtrmf(comlyn,comlen,'PSEUDO_DIAG_CRITER',0.05d0)

    !   qm_ewald=1 !Default is to do QMEwald, with varying charges, 
    !  if ntb=0 or use_pme=0 then this will get turned off
    qmmm_nml%qm_ewald = gtrmi(comlyn,comlen,'QM_EWALD',1)

    !   qm_pme = 1 !use pme for QM-MM
    qm_pme = gtrmi(comlyn,comlen,'QM_PME',1)

    !   kmaxqx=5; kmaxqy=5; kmaxqz=5    !Maximum K space vectors
    qmmm_nml%kmaxqx = gtrmi(comlyn,comlen,'KMAXQX',5)
    qmmm_nml%kmaxqy = gtrmi(comlyn,comlen,'KMAXQY',5)
    qmmm_nml%kmaxqz = gtrmi(comlyn,comlen,'KMAXQZ',5)

    !   ksqmaxq=27 !Maximum K squared values for spherical cutoff in k space.
    qmmm_nml%ksqmaxq = gtrmi(comlyn,comlen,'KSQMAXQ',27)

    !   writepdb = 0 !Set to 1 to write a pdb on the first step 
    !     with just the QM region in it.
    writepdb = gtrmi(comlyn,comlen,'WRITEPDB',0)

    !   qmmm_int = 1 !Default, do full interaction without extra Gaussian terms 
    !            for PM3 / AM1 etc.
    qmmm_nml%qmmm_int = gtrmi(comlyn,comlen,'QMMM_INT',1)

    !   adjust_q = 2 !Default adjust q over all atoms.
    qmmm_nml%adjust_q = gtrmi(comlyn,comlen,'ADJUST_Q',2)

    !   diag_routine = 1 !Use default internal diagonalizer.
    qmmm_nml%diag_routine = &
         gtrmi(comlyn,comlen,'DIAG_ROUTINE',1)

    !   density_predict = 0 !Use density matrix from previous MD step.
    qmmm_nml%density_predict = gtrmi(comlyn,comlen,'density_predict',0)

    !   fock_predict = 0 !Do not attempt to predict the Fock matrix.
    qmmm_nml%fock_predict = gtrmi(comlyn,comlen,'fock_predict',0)

    !   fockp_d1 = 2.4d0
    qmmm_nml%fockp_d1 = gtrmf(comlyn,comlen,'fockp_d1',2.4d0)

    !   fockp_d2 = -1.2d0
    qmmm_nml%fockp_d2 = gtrmf(comlyn,comlen,'fockp_d2',-1.2d0)

    !   fockp_d3 = -0.8d0
    qmmm_nml%fockp_d3 = gtrmf(comlyn,comlen,'fockp_d3',-0.8d0)

    !   fockp_d4 = 0.6d0
    qmmm_nml%fockp_d4 = gtrmf(comlyn,comlen,'fockp_d4',0.6d0)

    !   Nearest solvent options.
    !   nearest_qm_solvent = 0 !Do not change the QM region to keep the nearest solvents.
    qmmm_nml%nearest_qm_solvent = gtrmi(comlyn,comlen,'nearest_qm_solvent',0)

    !   nearest_qm_solvent_fq = 1 !Do update of QM solvent on every step if nearest_qm_solvent > 0
    qmmm_nml%nearest_qm_solvent_fq = gtrmi(comlyn,comlen,'nearest_qm_solvent_fq',1)

    !   nearest_qm_solvent_resname='WAT ' !by default assume solvent is resname WAT
    call gtrmwd(comlyn,comlen,"NEAREST_QM_SOLVENT_RESNAME",26,&
         qmmm_nml%nearest_qm_solvent_resname,4,clen)

    !DFTB
    qmmm_nml%dftb_maxiter     = gtrmi(comlyn,comlen,'DFTB_MAXITER',70)   
    qmmm_nml%dftb_disper      = gtrmi(comlyn,comlen,'DFTB_DISPER',0)
    qmmm_nml%dftb_chg         = gtrmi(comlyn,comlen,'DFTB_CHG',0)
    qmmm_nml%dftb_telec       = gtrmf(comlyn,comlen,'DFTB_TELEC',0.0d0)
    qmmm_nml%dftb_telec_step  = gtrmf(comlyn,comlen,'DFTB_TELEC_STEP',0.0d0)
    qmmm_nml%chg_lambda  = gtrmf(comlyn,comlen,'CHG_LAMBDA',1.0d0)

    ! Next check these options are all valid.

    !  check we don't bust our statically allocated max_quantum_atoms
    call int_legal_range('QMMM: (number of quantum atoms) ', &
         qmmm_struct%nquant, 1, max_quantum_atoms )

    ! --- Variable QM water region - has to very early here because we
    !   will be changing nquant and iqmatoms.  ---
    call int_legal_range('QMMM: (QM-MM nearest_qm_solvent) ', &
         qmmm_nml%nearest_qm_solvent,0,nres)
    if (qmmm_nml%nearest_qm_solvent > 0 ) then
       call int_legal_range('QMMM: (QM-MM nearest_qm_solvent_fq) ',&
            qmmm_nml%nearest_qm_solvent_fq,1,99999999)
    end if
    if (qmmm_nml%nearest_qm_solvent > 0) then
       ! We need to work out how many atoms nearest_qm_solvent * natoms_per_solvent_residue equals.
       ! We then need to find the nearest atoms and update nquant and iqmatoms respectively.
       call wrndie(-1,'<qmmmsemi.src>qmmm_startup', &
            'Variable solvent not currently implemented.')
       !      call qm2_setup_vsolv(qmmm_struct%nquant, max_quantum_atoms, iqmatoms, &
       !                           nres, ih(m02), ix(i02),ix(i70), natom)
       ! check again that we don't bust our statically allocated max_quantum_atoms
       call int_legal_range('QMMM: (number of quantum atoms) ', &
            qmmm_struct%nquant, 1, max_quantum_atoms )
    end if
    ! --- End Variable QM water region ---

    call float_legal_range('QMMM: (QM-MM Cutoff) ', qmmm_nml%qmcut,0.0D0,1.0D30)
    call int_legal_range('QMMM: (QM GB Method) ', qmmm_nml%qmgb,0,3)
    call int_legal_range('QMMM: (QM Theory) ', qmmm_nml%qmtheory,1,9)
    call int_legal_range('QMMM: (QM-QM Derivatives) ', qmqmdx,1,2)
    call int_legal_range('QMMM: (Verbosity) ', qmmm_nml%verbosity,0,5)
    call int_legal_range('QMMM: (Max SCF Iterations) ', qmmm_nml%itrmax,1,10000000)
    call int_legal_range('QMMM: (Shake on QM atoms) ', qmmm_nml%qmshake,0,1)
    call int_legal_range('QMMM: (Density Matrix Convergence) ', tight_p_conv,0,1)
    call float_legal_range('QMMM: (SCF Convergence) ', qmmm_nml%scfconv,1.0D-16,1.0D0)
    call int_legal_range('QMMM: (PRINT CHARGES) ', printcharges,0,1)
    call int_legal_range('QMMM: (Spin State) ', qmmm_nml%spin,1,1)
    !RCW: Currently limit spin state to singlets only since the code for spin>1 does not exist / work at present.
    !     WARNING - IF WE LATER ALLOW SPIN>1 qm2_densit will need updating.
    call int_legal_range('QMMM: (Peptide Correction) ',peptide_corr,0,1)
    call int_legal_range('QMMM: (QM-QM RIJ in Core) ',qmqmrij_incore,0,1)
    call int_legal_range('QMMM: (QM-MM RIJ in Core) ',qmmmrij_incore,0,1)
    call int_legal_range('QMMM: (QM-QM E-Rep in Core) ',qmqm_erep_incore,0,1)
    call int_legal_range('QMMM: (Link Atomic Number) ',qmmm_nml%lnk_atomic_no,1,nelements)
    call int_legal_range('QMMM: (QM-MM Link Method) ',qmmm_nml%lnk_method,1,2)
    call int_legal_range('QMMM: (Pseudo Diag) ',pseudo_diag,0,1)
    call float_legal_range('QMMM: (Pseudo Diag Criteria) ',qmmm_nml%pseudo_diag_criteria,1.0D-12,1.0D0)
    call int_legal_range('QMMM: (QM Ewald) ',qmmm_nml%qm_ewald,0,2)
    call int_legal_range('QMMM: (QM PME) ',qm_pme,0,1)
    call int_legal_range('QMMM: (QM Ewald kmaxqx) ',qmmm_nml%kmaxqx,1,99999999)
    call int_legal_range('QMMM: (QM Ewald kmaxqy) ',qmmm_nml%kmaxqy,1,99999999)
    call int_legal_range('QMMM: (QM Ewald kmaxqz) ',qmmm_nml%kmaxqz,1,99999999)
    call int_legal_range('QMMM: (QM Ewald ksqmaxq) ',qmmm_nml%ksqmaxq,1, &
         qmmm_nml%kmaxqx*qmmm_nml%kmaxqy*qmmm_nml%kmaxqz)
    call int_legal_range('QMMM: (QM-MM qmmm_int) ',qmmm_nml%qmmm_int,0,2)
    call int_legal_range('QMMM: (QM-MM adjust_q) ',qmmm_nml%adjust_q,0,2)
    call int_legal_range('QMMM: (QM-MM diag_routine) ',qmmm_nml%diag_routine,0,7)
    call int_legal_range('QMMM: (QM-MM density_predict) ',qmmm_nml%density_predict,0,1)
    call int_legal_range('QMMM: (QM-MM fock_predict) ',qmmm_nml%fock_predict,0,1)
    if (qmmm_nml%lnk_dis>0.0d0) then
       !if lnk_dis is less than 0.0d0 then the link atom is just placed on top of
       !the MM link pair atom.
       call float_legal_range('QMMM: (Link Atom Distance) ',qmmm_nml%lnk_dis,0.7D0,4.0D0)
    endif

    call float_legal_range('QMMM: (QM-MM chg_lambda)', qmmm_nml%chg_lambda, 0.0D0, 1.0D0)
    call int_legal_range('QMMM: (QM-MM dftb_maxiter)', qmmm_nml%dftb_maxiter, 1, 10000)
    call int_legal_range('QMMM: (QM-MM dftb_disper)', qmmm_nml%dftb_disper, 0, 1)
    call int_legal_range('QMMM: (QM-MM dftb_chg)', qmmm_nml%dftb_chg, 0, 1)
    call float_legal_range('QMMM: (QM-MM dftb_telec)', qmmm_nml%dftb_telec, 0.0D0, 1.0D4)

    !Specific setting of some qmmm_nml values based on the value of other
    !options.
    if (qmmm_nml%dftb_chg > 0) printcharges=1

    ! DFTB Limitations - current things not supported in DFTB
    ! These are silent limitations that are non fatal - just to
    ! avoid problems with the default. Fatal errors are handled
    ! later in this routine.
    if (qmmm_nml%qmtheory == DFTB) then
       qmqmrij_incore=0
       qmmmrij_incore=0
       qmqm_erep_incore=0
       pseudo_diag=0
       qmqmdx=1
    end if

    !qmgb values:
    ! 0 - do GB but leave QM charges as zero and add nothing to Fock matrix. This is like a vacuum
    !     QM molecule in a solvated MM system.
    ! 1 - do GB using the prmtop fixed resp charges for the GB calculation.
    ! 2 - do GB using Mulliken charges that are consistent with the GB field by modifying the fock
    !     matrix at every SCF step. (default)
    ! 3 - do GB using QM gas phase Mulliken charges - This is really a debugging option since the charges
    !     will not be consistent with the GB field since the fock matrix is not modified. This similarly
    !     means that the gradients will not be accurate. A warning will be printed at every QM call if this
    !     option is selected.

    !Make sure igb in &cntrl namelist is compatible with qmgb setting.
    !    if (igb==0 .or. igb==6) then
    !       !no qmgb available
    !       qmgb = 0
    !    end if
    !Print warning about qmgb being for debugging only.
    if (qmmm_nml%qmgb==3) then
       write(outu,*) "QMMM: ------------------------------ WARNING --------------------------------"
       write(outu,*) "QMMM: qmgb = 3 is designed for debugging purposes only. It gives GB"
       write(outu,*) "QMMM:          energies based on gas phase QM Mulliken charges. These charges"
       write(outu,*) "QMMM:          are NOT consistent with the GB field felt by the QM region and"
       write(outu,*) "QMMM:          so any gradients calculated using this approach will"
       write(outu,*) "QMMM:          NOT BE ACCURATE."
       write(outu,*) "QMMM:          This option is really designed for:"
       write(outu,*) "QMMM:                SINGLE POINT ENERGY EVALUATIONS ONLY"
       write(outu,*) "QMMM: ------------------------------ WARNING --------------------------------"
    end if

    qmmm_struct%AM1_OR_PM3 = (qmmm_nml%qmtheory == AM1 .OR. qmmm_nml%qmtheory == PM3 &
         .OR. qmmm_nml%qmtheory == PDDGPM3 .OR. qmmm_nml%qmtheory == PM3CARB1 .OR. &
         qmmm_nml%qmtheory == RM1 .OR. qmmm_nml%qmtheory == PDDGPM3_08)
    qmmm_struct%PDDG_IN_USE = (qmmm_nml%qmtheory == PDDGPM3 .OR. qmmm_nml%qmtheory == PDDGMNDO  &
         .OR. qmmm_nml%qmtheory == PDDGPM3_08 )

    if (qmqmdx /= 1) then
       qmmm_nml%qmqm_analyt = .false. !Do numerical QM-QM derivatives in qm2
    else
       qmmm_nml%qmqm_analyt = .true.  !Do analytical QM-QM dericatives in qm2
    end if
    if (tight_p_conv /= 1) then
       qmmm_nml%tight_p_conv = .false. !Loose density matrix convergence (0.05*sqrt(SCFCRT))
    else
       qmmm_nml%tight_p_conv = .true.  !Tight density matrix convergence (SCFCRT)
    end if
    !Write a warning about excessively tight convergence requests.
    if ( qmmm_nml%scfconv < 1.0D-12 ) then
       write(outu,'("QMMM: WARNING - SCF Conv = ",G8.2)') qmmm_nml%scfconv
       write(outu,*) "QMMM:           There is a risk of convergence problems when the"
       write(outu,*) "QMMM:           requested convergence is less that 1.0D-12 kcal/mol."
    end if

    !How tight do we want the density convergence?
    if (qmmm_nml%tight_p_conv) then
       qmmm_nml%density_conv = qmmm_nml%scfconv
    else
       qmmm_nml%density_conv = 0.05D0 * sqrt(qmmm_nml%scfconv)
    end if
    if ( printcharges /= 1) then
       qmmm_nml%printcharges=.false.
    else
       qmmm_nml%printcharges=.true.
    end if
    if ( peptide_corr == 0) then
       qmmm_nml%peptide_corr = .false.
    else
       qmmm_nml%peptide_corr =  .true.
    end if
    if ( qmqmrij_incore == 0 .or. qmqmdx == 2 ) then
       qmmm_nml%qmqmrij_incore = .false.
    else
       !Only available with analytical derivatives
       qmmm_nml%qmqmrij_incore = .true.
    end if

    if ( qmmmrij_incore == 0 .or. qmmm_nml%qmmm_int==0 ) then
       qmmm_nml%qmmmrij_incore = .false.
    else
       qmmm_nml%qmmmrij_incore = .true. !Only available with qmmm_int=1 or qmmm_int=2
    end if
    if ( qmqm_erep_incore == 0 .or. qmqmdx == 2 ) then
       qmmm_nml%qmqm_erep_incore = .false.
    else
       !Only available with analytical derivatives.
       qmmm_nml%qmqm_erep_incore = .true.
    end if
    if ( pseudo_diag == 1 ) then
       qmmm_nml%allow_pseudo_diag = .true.
    else
       qmmm_nml%allow_pseudo_diag = .false.
    end if

    !If ntb=0 or use_pme =0 then we can't do qm_ewald so overide what the user may
    !have put in the namelist and set the value to false.
    !    if (ntb==0 .or. use_pme==0) then
    !      qmmm_nml%qm_ewald = 0
    !      qmmm_nml%qm_pme = .false.
    !    end if
    if (qmmm_nml%qm_ewald>0 .and. qm_pme>0) then
       qmmm_nml%qm_pme=.true.
    else
       qmmm_nml%qm_pme=.false.
    end if

    if ( writepdb == 0 ) then
       qmmm_nml%writepdb=.false.
    else
       qmmm_nml%writepdb=.true.
    end if

    !Setup some specific calculation flags that depend on namelist variables.
    !Need to make sure these get copied to other threads in an MPI run.

    !Will we be calculating the Mulliken charges on every SCF iteration?
    !Default is no. Will be set to true in a bit if certain options, such as qm_ewald
    !require it.
    if (qmmm_nml%qm_ewald==1 .or. qmmm_nml%qmgb>1) then
       !We will be needing the mulliken charges on every SCF iteration
       qm2_struct%calc_mchg_scf = .true.
    else
       qm2_struct%calc_mchg_scf = .false.
    end if

    !DFTB Calculates Mulliken charges anyway so we might as well store them in the correct place.
    if (qmmm_nml%qmtheory == DFTB) qm2_struct%calc_mchg_scf = .true.

    return

  end subroutine fill_qm_run_parameters

  !-------------------------------------------------
  !     --- FLOAT_LEGAL_RANGE ---
  !-------------------------------------------------
  !+ Check the range of a float; abort on illegal values.
  subroutine float_legal_range(string,param,lo,hi)

    use stream, only : outu

    implicit none

    real(chm_real) param,lo,hi
    character(len=*)string

    if ( param < lo .or. param > hi )then
       write(outu,59)
       write(outu,60)string,param
       write(outu,61)
       write(outu,62)lo,hi
       call wrndie(-4,"<qmmmsemi.src>float_legal_range ", &
            " illegal parameter value")
    end if
59  format(/,1x,'QMMM PARAMETER RANGE CHECKING: ')
60  format(1x,'parameter ',a,' has value ',e12.5)
61  format(1x,'This is outside the legal range')
62  format(1x,'Lower limit: ',e12.5,' Upper limit: ',e12.5)
    return
  end subroutine float_legal_range

  !-------------------------------------------------
  !     --- FLOAT_LEGAL_RANGE ---
  !-------------------------------------------------
  !+ Check the range of an integer; abort on illegal values.
  subroutine int_legal_range(string,param,lo,hi)

    use stream, only : outu

    implicit none
    integer param,lo,hi
    character(len=*)string

    if ( param < lo .or. param > hi )then
       write(outu,59)
       write(outu,60)string,param
       write(outu,61)
       write(outu,62)lo,hi
       call wrndie(-4,"<qmmmsemi.src>int_legal_range ", &
            " illegal parameter value")
    end if
59  format(/,1x,'QMMM PARAMETER RANGE CHECKING: ')
60  format(1x,'parameter ',a,' has value ',i8)
61  format(1x,'This is outside the legal range')
62  format(1x,'Lower limit: ',i8,' Upper limit: ',i8)
    return
  end subroutine int_legal_range

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+ allocates space for qmmm variables and arrays that depend only on nquant or natom
  subroutine allocate_qmmm( natm )
    implicit none
    !passed in
    integer , intent(in) :: natm
    !Local
    integer :: alloc_err=0

    allocate ( qmmm_struct%qm_resp_charges(qmmm_struct%nquant), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%qm_resp_charges')

    !iqmatoms and iqm_atomic_numbers are allocated here as nquant+nlink long but
    !initially nlink is zero so it only gets allocated as nquant long on the master
    !thread. It is then resized when nlink are known about. When all other mpi threads
    !call this allocation routine nquant+nlink should include both qm and link atoms.

    !Note on the master it was allocated in the startup above as nquant so we only want
    !to do the allocation if it isn't allocated.
    if (.not. allocated(qmmm_nml%iqmatoms)) then
       allocate ( qmmm_nml%iqmatoms(qmmm_struct%nquant+qmmm_struct%nlink), stat=alloc_err )
       if (alloc_err /= 0) &
            call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
            'unable to allocate qmmm_struct%iqmatoms')
    end if
    allocate ( qmmm_struct%iqm_atomic_numbers((qmmm_struct%nquant+qmmm_struct%nlink)), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%iqm_atomic_numbers')
    allocate ( qmmm_struct%atom_mask(natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%atom_mask')
    allocate ( qmmm_struct%mm_link_mask(natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%mm_link_mask')
    allocate (qmmm_struct%scaled_mm_charges(natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%scale_mm_charges')
    allocate (qmmm_scratch%qm_real_scratch(4*natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_scratch%qm_real_scratch')
    allocate (qmmm_scratch%qm_int_scratch(3*natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_scratch%qm_int_scratch')

    !dxyzcl array only actually needs to be 3,qm_mm_pairs long..
    if (qmmm_nml%qmmm_int /= 0) then
       allocate ( qmmm_struct%dxyzcl(3,natm), stat=alloc_err )
       if (alloc_err /= 0) &
            call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
            'unable to allocate qmmm_struct%dxyzcl')
    end if
    !qm_xcrd only actually needs to be 4,qm_mm_pairs long...
    allocate (qmmm_struct%qm_xcrd(4,natm), stat=alloc_err )
    if (alloc_err /= 0) &
         call wrndie(-1,'<qmmmsemi.src>allocate_qmmm', &
         'unable to allocate qmmm_struct%qm_xcrd')

    return
  end subroutine allocate_qmmm
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+ Works out the atomic numbers for the QM atoms and sets up some
  !+ additional arrays.
  subroutine assign_qm_atomic_numbers()

    use psf, only : atype

    implicit none

    ! Local
    integer :: i, j

    do i = 1, qmmm_struct%nquant
       !Get the atomic numbers (used to be done in rdparm2...)
       j = qmmm_nml%iqmatoms(i)
       call get_atomic_number(atype(j), qmmm_struct%iqm_atomic_numbers(i))
    end do

    !Now we have a list of atom numbers for QM atoms we can build a true false (natom long) list
    !specifying what the quantum atoms are. Useful for doing quick .OR. operations against other
    !lists.
    qmmm_struct%atom_mask = .false. !Note, sets entire natom long array to false

    do i = 1, qmmm_struct%nquant
       qmmm_struct%atom_mask(qmmm_nml%iqmatoms(i)) = .true.
    end do

    return

  end subroutine assign_qm_atomic_numbers
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+ Attempts to work out what the atomic number is of each QM atom based
  !+ on the atom name.
  subroutine get_atomic_number(atom_name,atomic_number)
    !
    !
    !     This subroutine assigns atomic number based upon the first
    !     letter of the atom symbol read in from the topology file.  Because
    !     this symbol is set in link, this procedure is ripe for misreading
    !     errors, but is the best way to go about it right now.

    !NOTE: THIS ROUTINE NEEDS COMPLETELY RE-WRITING (ROSS WALKER)...
    !
    !     atom_name ==>   character containing the atomic name
    !     iqm_atomic_numbers ==>   integer array of atomic numbers assigned to atoms
    !

    use stream, only : outu

    implicit none

    !passed in
    integer :: atomic_number
    character(len=8) :: atom_name

    !local
    integer :: i, j
    character(len=2) :: elemnt(107)
    data (elemnt(i),i=1,107)/'H','XX', &
         'Li','XX','B','C','N','O','F','XX', &
         'XX','Mg','Al','XX','P','S','XX','XX', &
         'XX','XX','XX','XX','XX','XX','XX','XX','XX','XX','XX', &
         'Z','Ga','XX','XX','XX','XX','XX', &
         'XX','XX','XX','XX','XX','XX','XX','XX','XX','XX','XX', &
         'XX','XX','XX','XX','XX','I','XX', &
         'XX','XX','XX','XX','XX','XX','XX','XX','XX','XX','XX','XX', &
         'XX','XX','XX','XX','XX','XX','XX','W','XX','XX','XX','XX', &
         'XX','XX','XX','XX','XX','XX','XX','XX', &
         'XX','XX','XX','XX','XX','U','XX','XX','XX','XX','XX','XX','XX',&
         'XX','XX','XX','++','+','--','-','TV'/
    !
    !      compare values in atom_name to those in elemnt and assign atomic
    !      numbers accordingly
    !
    do j=1,107
       if(atom_name(1:1) .eq. elemnt(j)(1:1)) then
          if(j.eq.6) then
             if(atom_name(2:2).eq.'L' .OR. atom_name(2:2).eq.'l') then
                atomic_number = 17
                go to 25
             else
                atomic_number =  6
                go to 25
             end if
          elseif(j.eq.5)then
             if(atom_name(2:2).eq.'R' .OR. atom_name(2:2).eq.'r')then
                atomic_number = 35
                go to 25
             else if(atom_name(2:2).eq.'E' .OR. atom_name(2:2).eq.'e')then
                atomic_number = 4
                go to 25
             else
                atomic_number = 5
                go to 25
             endif
          elseif(j.eq.16) then
             if (atom_name(2:2) .eq. 'I' .OR. atom_name(2:2) .eq. 'i') then
                atomic_number = 14 !Silicon
                go to 25
             else
                atomic_number = 16 !Sulphur
                go to 25
             end if
          elseif (j.eq.31) then
             if (atom_name(2:2) .eq. 'E' .OR. atom_name(2:2) .eq. 'e') then
                atomic_number = 32 !Germanium
                go to 25
             else
                atomic_number = 31 !Gallium
                go to 25
             end if
          elseif (j.eq.13) then
             if (atom_name(2:2) .eq. 'S' .OR. atom_name(2:2) .eq. 's') then
                atomic_number = 33 !As
                go to 25
             else
                atomic_number = 13
                go to 25
             end if
          else
             atomic_number = j
             go to 25
          end if
       end if
    end do
    write(outu,*) 'QMMM: Unable to correctly identify element ', atom_name
    call wrndie(-1,'<qmmmsemi.src>get_atomic_number', &
         'Unable to correctly identify element.')
25  continue
    return

  end subroutine get_atomic_number
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+ Checks for options which conflict or exceed the limitations of the code.
  subroutine qm_check_limitations()

    use stream, only : outu

    implicit none

    !--- You cannot mix Fock prediction with density prediction. ---
    if (qmmm_nml%fock_predict > 0 .and. qmmm_nml%density_predict > 0) then
       call wrndie(-1,'<qmmmsemi.src>qm_check_limitations',&
            'Fock matrix and Density matrix prediction are mutually exclusive. &
            & Cannot have fock_predict > 0 and density_predict > 0')
    end if

    !--- For Fock prediction the 4 pre-factors must sum to 1.0d0 ---
    if (qmmm_nml%fock_predict == 1) then
       if (abs(1.0d0-(qmmm_nml%fockp_d1 + qmmm_nml%fockp_d2 + qmmm_nml%fockp_d3 + qmmm_nml%fockp_d4)) > 1.0d-6) then
          write(outu,*) 'QMMM: Failure, fockp_d1 to d4 must sum to 1.0d0 - current sum is', &
               (qmmm_nml%fockp_d1 + qmmm_nml%fockp_d2 + qmmm_nml%fockp_d3 + qmmm_nml%fockp_d4)
          call wrndie(-1,'<qmmmsemi.src>qm_check_limitations',&
               'Fock matrix prediction coefficients do not sum to 1.0. &
               & Adjust fockp_d1 to fockp_d4 so that they sum to 1.0.')
       end if
    end if

    !--- You cannot use variable solvent with GB calculations.
    if (qmmm_nml%qmgb > 0 .and. qmmm_nml%nearest_qm_solvent > 0) then
       call wrndie(-1,'<qmmmsemi.src>qm_check_limitations',&
            'Nearest QM solvent and qmgb are mutually exclusive. &
            & Cannot have nearest_qm_solvent > 0 and qmgb > 0')
    end if

    !--- DFTB LIMITATIONS ---
    if (qmmm_nml%qmtheory == DFTB ) then
       if (qmmm_nml%peptide_corr) then
          call wrndie(-1,'<qmmmsemi.src>qm_check_limitations',&
               'QM_theory=DFTB but peptide_corr /= 0. &
               & Peptide correction is not available, or required for  DFTB. &
               & Set peptide_corr = 0 to proceed.')
       end if
    end if
    !--- END DFTB LIMITATIONS ---

    return

  end subroutine qm_check_limitations
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+ Prints out a summary of all the QMMM options that are set.
  subroutine qm_print_option_summary()

    use stream, only : outu

    implicit none
    !---- QMMM Options ----

    write(outu,'(/a)') 'QMMM: -------------------------------------------------'
    write(outu,'(a)')  'QMMM:      QMMM SEMI Enabled - QMMM Option Summary:'
    write(outu,'(a)')  'QMMM: -------------------------------------------------'

    write(outu, '("QMMM:           ifqnt = True       nquant = ",i8)') &
         qmmm_struct%nquant
    write(outu, '("QMMM:            qmgb = ",i8,"  qmcharge = ",i8,"   adjust_q = ",i8)') &
         qmmm_nml%qmgb, qmmm_nml%qmcharge, qmmm_nml%adjust_q
    write(outu, '("QMMM:            spin = ",i8,"     qmcut = ",f8.4, "    qmshake = ",i8)') &
         qmmm_nml%spin,  qmmm_nml%qmcut, qmmm_nml%qmshake
    write(outu, '("QMMM:        qmmm_int = ",i8)') qmmm_nml%qmmm_int
    write(outu, '("QMMM:   lnk_atomic_no = ",i8,"   lnk_dis = ",f8.4," lnk_method = ",i8)') &
         qmmm_nml%lnk_atomic_no,qmmm_nml%lnk_dis, qmmm_nml%lnk_method
    if ( qmmm_nml%qmtheory == PM3 ) then
       write(outu, '("QMMM:       qm_theory =     PM3")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == AM1 ) then
       write(outu, '("QMMM:       qm_theory =     AM1")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == MNDO ) then
       write(outu, '("QMMM:       qm_theory =    MNDO")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == PDDGPM3 ) then
       write(outu, '("QMMM:       qm_theory = PDDGPM3")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == PDDGMNDO ) then
       write(outu, '("QMMM:       qm_theory =PDDGMNDO")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == PM3CARB1 ) then
       write(outu, '("QMMM:       qm_theory =PM3CARB1")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == DFTB ) then
       write(outu, '("QMMM:       qm_theory =    DFTB")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == RM1 ) then
       write(outu, '("QMMM:       qm_theory =     RM1")',ADVANCE='NO')
    else if ( qmmm_nml%qmtheory == PDDGPM3_08 ) then
       write(outu, '("QMMM:       qm_theory = PDDGPM3_08")',ADVANCE='NO')
    else
       write(outu, '("QMMM:       qm_theory = UNKNOWN!")',ADVANCE='NO')
    end if
    write (outu, '("      verbosity = ",i8)') qmmm_nml%verbosity
    if (qmmm_nml%qmqm_analyt) then
       write (outu, '("QMMM:          qmqmdx = Analytical")')
    else
       write (outu, '("QMMM:          qmqmdx = Numerical")')
    end if
    if (qmmm_nml%tight_p_conv) then
       write (outu, '("QMMM:    tight_p_conv = True (converge density to SCFCRT)")')
    else
       write (outu, '("QMMM:    tight_p_conv = False (converge density to 0.05xSqrt[SCFCRT])")')
    end if
    write (outu, '("QMMM:         scfconv = ",e9.3,"  itrmax = ",i8)') qmmm_nml%scfconv, qmmm_nml%itrmax
    if (qmmm_nml%printcharges) then
       write(outu, '("QMMM:    printcharges = True ")',ADVANCE='NO')
    else
       write(outu, '("QMMM:    printcharges = False")',ADVANCE='NO')
    end if
    if (qmmm_nml%peptide_corr) then
       write(outu, '("        peptide_corr = True")')
    else
       write(outu, '("        peptide_corr = False")')
    end if
    if (qmmm_nml%qmqmrij_incore) then
       write(outu, '("QMMM: qmqmrij_incore = True ")',ADVANCE='NO')
    else
       write(outu, '("QMMM: qmqmrij_incore = False")',ADVANCE='NO')
    end if
    if (qmmm_nml%qmmmrij_incore) then
       write(outu, '("     qmmmrij_incore = True ")')
    else
       write(outu, '("     qmmmrij_incore = False")')
    end if
    if (qmmm_nml%qmqm_erep_incore) then
       write(outu, '("QMMM:qmqm_erep_incore = True ")')
    else
       write(outu, '("QMMM:qmqm_erep_incore = False")')
    end if
    if (qmmm_nml%allow_pseudo_diag) then
       write(outu, '("QMMM:     pseudo_diag = True ")',ADVANCE='NO')
       write(outu, '("pseudo_diag_criteria = ",f8.4)') qmmm_nml%pseudo_diag_criteria
    else
       write(outu, '("QMMM:     pseudo_diag = False")')
    end if
    write(outu, '("QMMM:    diag_routine = ",i8)') qmmm_nml%diag_routine
    !If ntb=0 or use_pme =0 then we can't do qm_ewald so overide what the user may
    !have put in the namelist and set the value to false.
    if (qmmm_nml%qm_ewald>0) then
       if (qmmm_nml%qm_pme) then
          write(outu, '("QMMM:          qm_ewald = ",i8, " qm_pme = True ")') qmmm_nml%qm_ewald
       else
          write(outu, '("QMMM:          qm_ewald = ",i8, " qm_pme = False ")') qmmm_nml%qm_ewald
       end if
       write(outu, '("QMMM:          kmaxqx = ",i4," kmaxqy = ",i4," kmaxqz = ",i4," ksqmaxq = ",i4)') &
            qmmm_nml%kmaxqx, qmmm_nml%kmaxqy, qmmm_nml%kmaxqz, qmmm_nml%ksqmaxq
    else
       write(outu, '("QMMM:        qm_ewald = ",i8, " qm_pme = False ")') qmmm_nml%qm_ewald
    end if
    !Print the fock matrix prediction params if it is in use.
    if (qmmm_nml%fock_predict>0) then
       write(outu, '("QMMM:    fock_predict = ",i4)') qmmm_nml%fock_predict
       write(outu, '("QMMM:        fockp_d1 = ",f8.4," fockp_d2 = ",f8.4)') qmmm_nml%fockp_d1, qmmm_nml%fockp_d2
       write(outu, '("QMMM:        fockp_d2 = ",f8.4," fockp_d4 = ",f8.4)') qmmm_nml%fockp_d3, qmmm_nml%fockp_d4
    end if

    !If nearest_qm_solvent > 1 then use of nearest solvent in QM routine is running so print details here.
    if (qmmm_nml%nearest_qm_solvent > 0) then
       write(outu, '("QMMM:nearest_qm_solvent = ",i4,"    nearest_qm_solvent_fq = ",i4)') qmmm_nml%nearest_qm_solvent, &
            qmmm_nml%nearest_qm_solvent_fq
       write(outu, '("QMMM:nearest_qm_solvent_resname = ",a)') qmmm_nml%nearest_qm_solvent_resname
    end if

    write(outu,'(a)')  'QMMM: -------------------------------------------------'
    write(outu,'(a)')  'QMMM:              End QMMM Option Summary'
    write(outu,'(a)')  'QMMM: -------------------------------------------------'

    return

  end subroutine qm_print_option_summary
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


  ! NOTES ABOUT THINGS WE HAVE TO CHANGE LATER
  ! 1) Change int and float legal range to use outu for unit.
  ! 2) Fix equivalences in free format psf.fcm
  ! 3) Fix setting of qmgb (including default) when igb is on.
  ! 4) Fix setting of qmewald/qm_pme when not periodic etc.
  ! 5) Update verbosity to be based on prnlev
  !

#endif /* (qmmmsemi)*/
end module qmmmsemi

