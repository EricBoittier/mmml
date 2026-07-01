module squantm
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1 /*sqnt_module*/
  !
  !     Defines the data necessary for a New Modified MOPAC calculation on a system.
  !
  !     Variable  Index    Purpose 
  ! 
  !     NQMAX              Maximum number of QM atom
  !     IGMSEL             Selection for QM selection
  !     MAXCEN             Maximum number of atoms that CADPAC works with
  !
  ! 
  INTEGER,parameter :: NQMAX=500

  !     NATQM             Number of QM atoms
  !     NQMTHEORY         QM theory   AM1 (2)/PM3 (1)/MNDO (3)/PDDG-PM3 (4)/PDDG-MNDO(5)
  !                                   PDP3 = PDDG-PM3  /  PDMN = PDDG-MNDO
  !     NQMCHARGE         QM charge
  !     QMCHARGE          QM charge (real value)
  !     NSPIN             QM electronic spin
  !     SCFCONV           SCF convergence criteria
  !
  !     CLINK             Logical to use Connection atoms (Not supporting yet)
  !     NCATM             Number of connection atoms
  !
  !     QLINK             GHO atom (supporting 2 qm regions for FEP)
  !     NumGHO            Number of GHO atoms.
  !
  !     QGRAD             Logical to compute gradients
  !     
  logical, save           :: QGRAD                                       ! COMMON /CGRAD /
  logical, save           :: CLINK,QLINK(2)                              ! COMMON /LINKA / (logical)
  integer, save           :: NCATM,NumGHO(2)                             ! COMMON /LINKA / (integer)
  integer, save           :: NATQM(2), nqmtheory, NQMCHARGE(2), NSPIN(2) ! COMMON /ATOMSQ/
  real(chm_real),save     :: SCFCONV,   QMCHARGE(2)                      ! COMMON /QMDATA/
  CHARACTER(len=80), save :: KEYWRD                                      ! COMMON /NETWRD/
  !

  !----------------------------------------------------------------------
  !     For Time-reversible Born-Oppenheimer MD simulations (TR-BOMD).
  !     Ref:  Niklasson, AMN, et al., Phys. Rev. Lett. 96, 123001 (2006).
  !
  !     P_guess(n+1)=2*P_scf(n) - P_guess(n-1) : n: old step, n+/-1: current/past.
  !
  !     LTRBOMD           Logical to turn on the TR-BOMD.
  !     q_apply_tr_bomd   Logical to apply TR-BOMD during dynamics.
  !     N_scf_cycle       The number of SCF cycles to do TR-BOMD.
  !     i_md_step         Current MD step.
  !     i_md_step_old     Old MD step, for the purpose of checking.
  !
  !
  logical, save           :: LTRBOMD=.false.
  logical, save           :: q_apply_tr_bomd=.false.
  integer, save           :: N_scf_cycle
  integer, save           :: i_md_step=0
  integer, save           :: i_md_step_old=0

  !----------------------------------------------------------------------
  ! For dual quantum region: use in FEP and Multi-layered QM/MM.
  ! QDUALQM : main flag
  ! QMFEP   : use qm/mm free energy perturbation 
  !           for pka calc.: it can be done this way.
  ! QpKa_on : turn on pka calc. in QMFEP
  !           for solvation: it need to use single coordinates.
  ! QMLAY   : use multi-layered QM/MM (always dual qm selection).
  ! QMSOLV  : solvation free energy (not supporting yet)
  !
  ! PreFact_dual    : prefactor for mixing different potential.
  !
  ! (pointer array)
  ! QM_Grp_dual     : logical array for qm group. (1st and 2nd qm region.)
  ! IGMSEL_dual     : IGMSEL array for 2nd qm region
  ! MMINB1_dual(*,2): MMINB1 array for 1st and 2nd selection.
  !                   (contain pointer information in x/y/z/cg array.)
  !                   (initial NATQM are for qm atoms.)
  ! MMINB2_dual(*,2): MMINB2 array for 1st and 2nd selection.
  !                   (contain pointer information for MMINB1_dual array.)
  ! QMINB1_dual     : qminb array for 1st qm region. (pointer for x/y/z/cg.)
  ! QMINB2_dual     : qminb array for 2nd qm region. (pointer for x/y/z/cg.)
  ! cginb           : mm charge for 1st/2nd qm region.
  ! cginb_pka       : original mm charge for 1st qm region, and used in the
  !                   QM/MM-Ewald sum.
  ! QCG_corr        : do background charge correction in QM/MM-Ewald sum.
  ! tot_cg_fep      : total charge for 1st and 2nd qm/mm regions.
  ! ener_bg_cg_corr : energy from background charge correction. 
  ! 
  !
  ! (in QMFEP and QpKa_on)
  ! IpKa_Hatm      : pointer for h atom in cg/igmsel array.
  ! IpKa_Hatm_indx : pointer for h atom in cginb array. 
  ! lambda_qm      : lambda value for hybrid Hamiltonian.
  !                  lambda_qm(1)   : reference lambda value
  !                  lambda_qm(2-3) : Perturbed lambda 1 and 2.
  ! temperature_fep: temperature for the dynamics.
  ! beta_qmmm_fep  : 1/(k_b*Temp) value.
  ! 
  !
  ! (in QMLAY)
  ! HRCUT          : Cut-off distance for High QM/MM
  !                 (default 10.0 A)
  ! QINP           : Use input MM charges for Low QM region in High QM
  !                  calculation.
  ! HIGHL          : do high level (ab initio calculation).
  ! NHSTP          : Step number for updating High QM/MM correction
  !                  of energy and gradient  (not used yet!)
  !                 (default 1, don't use more than 5)
  ! AZNUC_high     : nuclear charge for 1st qm region.
  ! CAATOM_high    : element name for 1st qm region.
  !
  ! MAXCHL         : maximum number of H-link atoms in 2nd qm regions.
  ! NHLink         : number of H-link atoms in 2nd qm regions.
  ! IHOSTGUEST(2,*): 1: h-link host qm atom in the 2nd qm region.
  !                  2: h-link guest mm atom (or in the 1st qm atom). 
  ! R_H_ref        : Reference qm-host to mm-guest distance for H-link atom.
  ! xyz_hlink      : working array for h-link atoms. (self-explanatory)
  ! dxyz_hlink     : working array for h-link atoms. (self-explanatory)
  !
  integer,PARAMETER :: MAXCHL=20
  integer,PARAMETER :: LENQEQ=2,QPT1=1,QPT2=2

  logical :: QDUALQM, QMFEP, QMLAY, QMSOLV, QH_Link, QpKa_on, QCG_corr(2)
  logical, allocatable :: QM_Grp_dual(:,:)
  integer, allocatable :: IGMSEL_dual(:), MMINB1_dual(:,:), MMINB2_dual(:,:)
  integer :: NATMM_dual(2), NHLink, IpKa_Hatm, IpKa_Hatm_indx, &
       QMINB1_dual(NQMAX), QMINB2_dual(NQMAX), IHOSTGUEST(2,MAXCHL)
  real(chm_real),save :: cginb(NQMAX),cginb_pka(NQMAX,2),PreFact_dual(2), &            ! COMMON /DUALQM3/
                         R_H_ref(MAXCHL)
  real(chm_real),save :: lambda_qm(3),temperature_fep,beta_qmmm_fep, &                 ! COMMON /QMFEP1 /
                         tot_cg_fep(2),ener_bg_cg_corr(2)

  ! for statistical averaging.
  ! LENQEQ    : Number of subcomponents of the QMEL energy term
  ! EEQTRM    : potential energy for (perturbed) states.

  real(chm_real),dimension(lenqeq),save ::  EEQTRM,EQPRA,EQPR2A,EQPRP,EQPR2P           ! COMMON /QMFEP2 /


  LOGICAL,save        :: QINP,HIGHL                                                    ! COMMON /DMLAY0 /
  integer, save       :: NHSTP,NMDSTP,LPLEV                                            ! COMMON /DMLAY1 /
  real(chm_real),save :: AZNUC_high(NQMAX),xyz_hlink(6,MAXCHL), &                      ! COMMON /DMLAY2 /
                         dxyz_hlink(6,MAXCHL),HRCUT
  CHARACTER(len=10),save :: CAATOM_high(NQMAX)                                         ! COMMON /DMLAY3 /


  ! (information for MM correction in dummy H atom for pKa calculation.)
  ! NUMBND  Number of MM bonds connecting H-atom
  ! NUMANG  Number of MM angles connecting H-atom
  ! NUMDIH  Number of MM dihedrals connecting H-atom
  ! NUMIMP  Number of MM improper conecting H-atom
  ! HBND()  Position in NBOND
  ! HANG()  Position in NTHETA
  ! HDIH()  Position in NPHI
  ! HIMP()  Position in NIMPHI
  ! RBND    parameter values for bond.
  ! RANG                         angle.
  ! RDIH                         dihedral.
  ! RIMPQ                        improper angle.

  !---mfc--- Needs allocation routines

  integer,parameter :: MAXBNDQ=2,MAXANGQ=5,MAXDIHQ=30

  integer, save       :: NUMBND,NUMANG,NUMDIH,NUMIMP, &                                ! COMMON/PKAINT/
                         HBND(MAXBNDQ),HANG(MAXANGQ),HDIH(MAXDIHQ),HIMP(MAXDIHQ)
  real(chm_real),save :: RBND(2,MAXBNDQ),RANG(3,MAXANGQ), &                            ! COMMON/PKAPRM/
                         RDIH(3,MAXDIHQ),RIMPQ(4,MAXDIHQ)
  integer, save       :: ICBQ(MAXBNDQ),ICTQ(MAXANGQ),ICPQ(MAXDIHQ), &                  ! COMMON/PKARM2/
                         ICIQ(MAXDIHQ),IDIH(MAXDIHQ),IIMP(MAXDIHQ)

  !----------------------------------------------------------------------
  !
  ! ***Important Here***
  !    MAXAIM (maximum atoms including image) = 2*SIZE
  !    SIZE   = 360,000 for  XXLARGE version
  !=================================================================================
  ! namkh 02/14/04
  ! For the QM/MM electrostatics with Cutoff distance and Ewald Sum method
  !     It is only work with Group based Cutoff option
  !     LQMEWD        : Logical to Ewald or PME with QM/MM
  !     MMINB2        : Atom number for mapping image atoms in main array
  !                     Note: special use of MMINB/MMINB2, refer CH2SQMIG/CH2SQMI
  !                           and SwapXYZ_image routines. More likely used as 
  !                           scractch indexing arrays
  !     IMATTQ(MAXA)  : Mapping index for Images into Primary image and vice versa.
  !                     1      ~NATOM : mapped into images (image transformation number)
  !                     refer "nbndqm_ltm.src"
  !
  ! Refer ewald.fcm
  !
  integer,PARAMETER :: NMAXGRP=201 ! 31
  
  LOGICAL, save       :: QMSWTCH                                                       ! COMMON/QMMCUT0/
  integer, save       :: NQMGRP(nmaxgrp),NCUTOFF(2)   !!!,IMATTQ(maxa)                 ! COMMON/QMMCUT1/ and /QMMCUT2/, note IMATTQ

!!!  these are defined in quantm_ltm.src
!!!  integer, save :: PNATM                                              ! COMMON/QMMCUT3/ (integer)
!!!  real(chm_real),allocatable,dimension(:),save :: PXIM, PYIM, PZIM    ! COMMON/QMMCUT3/ (real)

  LOGICAL, save       :: LQMEWD                                                        ! COMMON/QMEWAL1/ (logical)
  integer, save       :: EWMODE                                                        ! COMMON/QMEWAL1/ (integer)
  integer, save       :: MAXKVQ,KMAXQ,KSQMAXQ,KMAXXQ,KMAXYQ,KMAXZQ                     ! COMMON/QMEWAL2/
  real(chm_real),save :: CGMM                                                          ! COMMON/QMEWAL3/

  !     gradient from Kspace sum
  logical, save       :: QFIRSTD                                                       ! COMMON/QMEWAL4/
  character(len=*), parameter, private :: fname = 'squantm_ltm.src'

contains

  subroutine allocate_squantm()
    use memory
    character(len=*), parameter :: proc = 'allocate_squantm'
    call chmalloc(fname,proc,'QM_Grp_dual',MAXGRP,2,log=QM_Grp_dual)
    call chmalloc(fname,proc,'IGMSEL_dual',MAXA,intg=IGMSEL_dual)
    call chmalloc(fname,proc,'MMINB1_dual',MAXA,2,intg=MMINB1_dual)
    call chmalloc(fname,proc,'MMINB2_dual',MAXA,2,intg=MMINB2_dual)
  end subroutine allocate_squantm

  subroutine squantm_iniall()
    natqm(1) = 0
    natqm(2) = 0
    lqmewd = .false.
    maxkvq = 0
    qgrad = .true.
    qdualqm = .false.
    qmfep = .false.
    qmlay = .false.
    qmsolv = .false.
  end subroutine squantm_iniall

#endif /* (sqnt_module)*/
end module squantm

