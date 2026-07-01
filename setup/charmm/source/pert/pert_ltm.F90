module pert
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/pert.fcm 1.1
  !     The Perturbed Protein Structure File (PSF)
#if KEY_PERT==1 /*pert_fcm*/
  !
  ! This common file contains the lambda=0 data for free energy
  ! perturbation simulations and slow growth homology modelling.
  !
  !

  ! Temporary setup for data structures
  !
  ! General pert information and arrays
  integer,allocatable,dimension(:) :: PPIATOM
  real(chm_real),allocatable,dimension(:) :: cglig


  integer,allocatable,dimension(:)        :: PERTIP     !(MAXAIM)
  integer,allocatable,dimension(:)        :: PERTIG     !(MAXGRP)
  real(chm_real),allocatable,dimension(:) :: PERTDX     !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PERTDY     !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PERTDZ     !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PERTEP1    !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PERTEP2    !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PERTEPC    !(MAXAIM)
  integer,allocatable,dimension(:)        :: PERTCONST  !(MAXSHK)
  real(chm_real),allocatable,dimension(:) :: PERTSHAUX  !(2*MAXSHK)

  !
  ! the pert PSF data structure

  integer,dimension(30)                   :: PPNUMS     !
  real(chm_real),dimension(10)            :: PPREALS    !
  integer,allocatable,dimension(:)        :: PPNICTT    !(NSEGT+1)
  integer,allocatable,dimension(:)        :: PPIBASE    !(MAXRES)
  integer,allocatable,dimension(:)        :: PPIGPBS    !(MAXGRP)
  integer,allocatable,dimension(:)        :: PPIGPTP    !(MAXGRP)
  integer,allocatable,dimension(:)        :: PPIAC      !(MAXAIM)
  integer,allocatable,dimension(:)        :: PPIB       !(NBONDT)
  integer,allocatable,dimension(:)        :: PPJB       !(NBONDT)
  integer,allocatable,dimension(:)        :: PPIT       !(NTHETT)
  integer,allocatable,dimension(:)        :: PPJT       !(NTHETT)
  integer,allocatable,dimension(:)        :: PPKT       !(NTHETT)
  integer,allocatable,dimension(:)        :: PPLT       !(NTHETT)
  integer,allocatable,dimension(:,:)        :: PPSTRBL    !(NTHETT*2)
  integer,allocatable,dimension(:)        :: PPIP       !(NPHIT)
  integer,allocatable,dimension(:)        :: PPJP       !(NPHIT)
  integer,allocatable,dimension(:)        :: PPKP       !(NPHIT)
  integer,allocatable,dimension(:)        :: PPLP       !(NPHIT)
  integer,allocatable,dimension(:)        :: PPIM       !(NIMPHT)
  integer,allocatable,dimension(:)        :: PPJM       !(NIMPHT)
  integer,allocatable,dimension(:)        :: PPKM       !(NIMPHT)
  integer,allocatable,dimension(:)        :: PPLM       !(NIMPHT)
#if KEY_CMAP==1
  integer,allocatable,dimension(:)        :: PPI1CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPJ1CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPK1CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPL1CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPI2CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPJ2CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPK2CT     !(NCRTT)
  integer,allocatable,dimension(:)        :: PPL2CT     !(NCRTT)
#endif 
  integer,allocatable,dimension(:)        :: PPIDON     !(NDON)
  integer,allocatable,dimension(:)        :: PPIHD1     !(NDON)
  integer,allocatable,dimension(:)        :: PPIACC     !(NACC)
  integer,allocatable,dimension(:)        :: PPIAC1     !(NACC)
  integer,allocatable,dimension(:)        :: PPINB      !(NNB)
  integer,allocatable,dimension(:)        :: PPIBLO     !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PPCG       !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PPALPHA    !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PPTHOLE    !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PPAMASS    !(MAXAIM)
  real(chm_real),allocatable,dimension(:) :: PPRSCLF    !(MAXAIM)
#if KEY_WCA==1
  real(chm_real),allocatable,dimension(:) :: PPWCA      !(MAXAIM)
#endif
  integer,allocatable,dimension(:)        :: PPICB      !(NBONDT)
  integer,allocatable,dimension(:)        :: PPICT      !(NTHETT)
  integer,allocatable,dimension(:)        :: PPICP      !(NPHIT)
  integer,allocatable,dimension(:)        :: PPICI      !(NIMPHT)
#if KEY_CMAP==1
  integer,allocatable,dimension(:)        :: PPICCT     !(NCRTT)
#endif
  integer,allocatable,dimension(:)        :: PPICOOP    !(NTHETT)
  integer,allocatable,dimension(:)        :: PPICSTBN   !(NTHETT)
  integer,allocatable,dimension(:)        :: PPIACNB    !(MAXAIM)


  !
  ! The pert restraint data structure

  integer,dimension(10)                   :: PRNUMS      !
  real(chm_real),dimension(10)            :: PRREALS     !
  logical,dimension(10)                   :: PRFLAGS     !
  integer,allocatable,dimension(:)        :: PRKCEXP     !(NUMHSETS)
  integer,allocatable,dimension(:)        :: PRICS       !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRJCS       !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRKCS       !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRLCS       !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRICCS      !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRCCSD      !(NCSPHI)
  integer,allocatable,dimension(:)        :: PRTYPEH     !(NUMHSETS)
  integer,allocatable,dimension(:)        :: PRIHSET     !(NATOM)
  !data structures for pca restraints (numpcap,natom)
  real(chm_real),allocatable,dimension(:,:) :: prpcax, prpcay,prpcaz 
  real(chm_real),allocatable,dimension(:) :: PRKCNST     !(NATOM)
  real(chm_real),allocatable,dimension(:) :: PRREFX      !(NATOM)
  real(chm_real),allocatable,dimension(:) :: PRREFY      !(NATOM)
  real(chm_real),allocatable,dimension(:) :: PRREFZ      !(NATOM)
  real(chm_real),allocatable,dimension(:) :: PRCCSC      !(NCSPHI)
  real(chm_real),allocatable,dimension(:) :: PRCCSB      !(NCSPHI)
  real(chm_real),allocatable,dimension(:) :: PRCCSW      !(NCSPHI)
  real(chm_real),allocatable,dimension(:) :: PRCCSCOS    !(NCSPHI)
  real(chm_real),allocatable,dimension(:) :: PRCCSSIN    !(NCSPHI)
  real(chm_real),allocatable,dimension(:) :: PRXHSCAL    !(NUMHSETS)
  real(chm_real),allocatable,dimension(:) :: PRYHSCAL    !(NUMHSETS)
  real(chm_real),allocatable,dimension(:) :: PRZHSCAL    !(NUMHSETS)
  logical,allocatable,dimension(:)        :: PRQHNORT    !(NUMHSETS)
  logical,allocatable,dimension(:)        :: PRQHNOTR    !(NUMHSETS)
  integer,allocatable,dimension(:)        :: PRNEIPT     !(NOENUM)
  integer,allocatable,dimension(:)        :: PRNEJPT     !(NOENUM)
  integer,allocatable,dimension(:)        :: PRNEINM     !(NOENUM)
  integer,allocatable,dimension(:)        :: PRNEJNM     !(NOENUM)
  integer,allocatable,dimension(:)        :: PRNELIS     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNERMN     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNEKMN     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNERMX     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNEKMX     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNEFMX     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNETCN     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNEAVE     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNEEXP     !(NOENUM)
  logical,allocatable,dimension(:)        :: PRNEMIN     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNERSW     !(NOENUM)
  real(chm_real),allocatable,dimension(:) :: PRNESEX     !(NOENUM)
  integer,allocatable,dimension(:)        :: PRNERAM     !(NOENUM)


  !ML----------------------------------------------------------------
  !  PERT - MMFP-restraint data structure
  !  These arrays (and variables) are lambda=0 - copies of the original MMFP-arrays
  !  The logicals QMMFPE and QZEGEO decide how the MMFP terms are handled in PERT
  !  IF (QMMFPE)  -  MMFP-restraints are part of the alchemical mutation
  !  IF (QZEGEO)  -  MMFP-restraints are present at lambda=0, important for epert.src
  !                  if restraints (and hence QGEO=.false.) are turned off at
  !                  lambda=1
  !  PMXGEO  -  Auxillary variable for FREHP
  INTEGER PMXGEO, PNTGEO
  integer,allocatable,dimension(:) :: &
       pIGEO,  pJGEO, plsgeo,pngeo,piugeo
  real(chm_real),allocatable,dimension(:) ::  &
       pXRGEO,  pYRGEO, pZRGEO, pTRGEO, &
       pXDGEO,  pYDGEO, pZDGEO,  &
       pDRGEO,  pDTGEO, &
       pFCGEO,  pP1GEO, pP2GEO, pP3GEO

  LOGICAL QMMFPE, QZEGEO 
  !ML----------------------------------------------------------------
  !sbcp
#if KEY_CHEMPERT==1
  !     add. stuff for chempert; essentially an array containing only the
  !     ligand charges. Not clear where to put this, so we put it here ...
  !     cglig      -  for array allocated on heap, charges of pert affected
  !                   region, all others are zero
  !     cgligt     -  sum of charges in cglig
  !
  real(chm_real) cgligt
#endif 
  !sbcp
  !
  !
  !  GENERAL INFORMATION:
  !
  !  QPERT    - Is perturbation or umbrella sampling being used?
  !  QPWIND   - Windowing method flag
  !  QACCUM   - Flag to indicate that current EPERT call is accumulated.
  !  QPAVER   - The LAMBDA vaule is the average of LSTART and LSTOP.
  !  PTERMN   - PERT section is done and current command should terminate.
  !
  !  PUNIT   - Unit number to get perturbation commands during dynamics.
  !  IWHAM   - Unit number to write output for WHAM (used in epert.src)
  !  IPSTRT  - Step number(-1) to start free energy accumulation
  !  IPSTP   - Step number to stop free energy accumulation
  !  IPNTOT  - total number of steps (IPNTOT=IPSTOP-IPSTRT)
  !  IPINCR  - Step number increment between windows.
  !  IPEQUI  - Number of equilibration steps between windows.
  !  IPNCAL  - number of active energy calls (accumulating).
  !  NIPERT  - Number of atoms being treated as changing.
  !  IPERT(NATOM) - which atoms change (IPERT(I)=1)
  !               - atoms that dont change (IPERT(I)=0)
  !  IGPERT(NGRP) - which groups change (IGPERT(I)=1)
  !               - groups that dont change (IGPERT(I)=0)
  !
  !  LAMSTA  - First lambda value
  !  LAMSTOP - Last lambda value
  !  LAMDAI  - Initial lambda value
  !  LAMDAF  - Final lambda value
  !  LAMDAP  - Parsed lambda value (otherewise computed)
  !  LAMDA   - Current lambda value
  !  LAMDAM  - one minus lambda
  !  LAMDEL  - LAMDEL=LAMDAF-LAMDAI for windowing
  !          - LAMDEL=(LAMDAF-LAMDAI)/(IPSTOP-IPSTRT) for slow growth.
  !  DELDRT  - DELDRT=LAMDEL/(FINALT*KBOLTZ)
  !  LAMINC  - Lambda value increment between windows.
  !
  !  LAMEXP  - Lambda value increment exponential constant.
  !
  !  EP????  - Energy terms for perturbation
  !  EPREF   - Reference energy for windowing
  !  EPREFF  - Reference energy for windowing Forward
  !  EPREFB  - Reference energy for windowing Backward
  !  EPRTOT  - Total free energy results for all runs since reset.
  !  EPRTTW  - Total free energy results for TP.
  !
  !pssp
  ! for support of PSSP (soft core/separation shifted scaling):
  !
  !  QPSSP   - are soft core pots. in use
  !  TQPSSP  - do we need special PSSP routines for this energy call
  !  LAPSSP  - lambda value needed in energy routines
  !  DLAMBD  - parameter d for modified lj-pot
  !  ALAMBD  - parameter alfpha for modified coulomb-pot
  !  EPSSLJ  - d U_lj/d lambda 
  !  EPSSCO  - d U_coulomb/d lambda 
  !  ECODLM  - d U_coulomb/d lambda for lambda = 0
  !  ELJDLM  - d U_lj/d lambda for lambda = 0
  !  ECODLL  - d U_coulomb/d lambda for lambda = 1
  !  ELJDLL  - d U_lj/d lambda for lambda = 1
  !  DCOTOT  - sum of d U_coulomb/d lambda for all runs since reset.
  !  DLJTOT  - Total d U_lj/d lambda for all runs since reset.
  !c PERTLAM - lamda factor (LAMDA or LAMDAM) associate to current state
  !  LAORDER - Power of lamda for DBEXP potential softcore.
  !  NLAMEXP - Number of exponential lamda windows
  !  ILAMEXP - Index of current exponential lamda windows
 !pssp
  !sbcp
  !  qchemp  - are we using CHEM PERT (default is off)
  !sbcp
  !
  ! integers
  INTEGER PUNIT,IPSTRT,IPSTP,IPNTOT,IPINCR,IPEQUI,IPNCAL
  INTEGER NIPERT,IWHAM
  INTEGER LAORDER,NLAMEXP,ILAMEXP

  !  reals
  real(chm_real)  LAMDAI,LAMDAF,LAMDAP,LAMDA,LAMDAM,LAMDEL,DELDRT, &
       LAMINC,LAMEXP,LAMSTA,LAMSTOP
  real(chm_real)  EPDIFF,EPTOT1F,EPTOT1B,EPTOT2F,EPTOT2B,EPTOT3, &
       EPTOT4
  real(chm_real)  EPREF,EPRTOT,EPRTTW,EPREFF,EPREFB
  real(chm_real)  EPSSBPLRCSQ,EPSSBPLRC,EPLRCTMP
#if KEY_WCA==1
  real(chm_real)  SCCUTR0, SCCUTR1                     
#endif
  real(chm_real)  PERTLAM  !Cc New PBLOCK

  real(chm_real)  DLAMBD,ALAMBD, EPSSLJ,EPSSCO,ECODLM,ELJDLM,ECODLL, &
       ELJDLL,DCOTOT,DLJTOT,LAPSSP
  ! logicals
  LOGICAL QPERT, QPWIND, QACCUM, QPAVER, PTERMN
#if KEY_WCA==1
  LOGICAL LSOFTCORE0, LSOFTCORE1               
#endif
  LOGICAL QPSSP,TQPSSP     

#if KEY_CHEMPERT==1
  logical qchemp                               
#endif
  !
  !====================================================================
  !  PSF RELATED INFORMATION
  !
  INTEGER NATOMP, NRESP, NSEGP, NBONDP, NTHETP, NPHIP, NIMPHP, &
#if KEY_CMAP==1
       NCRTP, &   
#endif
       NGRPP, NATMTP, NRESTP, NSEGTP, NBONTP, NTHTTP, NPHITP, &
       NIMPTP, NGRPTP, NNBP, NDONP, NACCP, NST2P
  ! reals
  real(chm_real) CGTOTP
  !====================================================================
  ! HARMONIC AND DIHEDRAL RESTRAINT INFORMATION
  !
  !  DIHEDRAL CONSTRAINTS
  !     NCSPHP     Number of dihedral constraints
  !     CCSBP      Zero energy dihedral angle
  !     CCSWP      Half width of flat bottom harmonic potential
  !     CCSCOSP    Cos(CCSBP) for CCSDP=0, Cos(CCSDP*CCSBP+PI) for CCSDP /= 0
  !     CCSSINP    Sin(CCSBP) for CCSDP=0, Sin(CCSDP*CCSBP+PI) for CCSDP /= 0
  !     CCSCP      Force constant for dihedral
  !     CCSDP      Periodicity for constrained dihedral
  !     ICCSP      Fake index into CCSB and CCSC for energy routines.
  !                ICCS(I)=I
  !     ICSP       First atom of dihedral constraint
  !     JCSP       Second atom of dihedral constraint
  !     KCSP       Third atom of dihedral constraint
  !     LCSP       Fourth atom of dihedral constraint
  !
  !  HARMONIC CONSTRAINTS
  !     QCNSRP     Flag meaning some atom has a non-zero constraint
  !     KCNSTP     Force constant of atomic constraint
  !     KCEXPP     Exponent of the constraint (default 2)
  !     REFXP      X component of reference coordinates
  !     REFYP      Y component of reference coordinates
  !     REFZP      Z component of reference coordinates
  !     NUMHSETP   The total number of active restraint sets
  !     TYPHSETP   The type of each restraint set
  !                   0 - direct with reference (no best fit)
  !                   1 - best fit rotation and translation with reference
  !                  >1 - best fit with alternate set (other than 1)
  !     IHSETP     The set number for each atom
  !     QHNORTP    Flag to suppress bestfit rotation
  !     QHNOTRP    Flag to suppress bestfit translation
  !
  ! integers
  INTEGER   NCSPHP, NUMHSETP
  integer numpcap ! number of pca restraint sets
  ! logicals
  LOGICAL   QCNSRP
  !====================================================================
  ! NOE RESTRAINT DATA
  !
  ! actual number of NOE constraints:
  INTEGER NENUMP
  ! number of noe atoms on list
  INTEGER NENM2P
  ! scale factor for NOE energies and forces
  real(chm_real) NESCAP
  !
  !====================================================================
  ! Stuff for Quantum (w/Q-Chem) Pert.
  !
  !     QMSTATE Specifies which state is being computed in QM/MM 
  !             PERT calculations 

  INTEGER QMSTATE
  LOGICAL QQMPERT

  ! -------------- NKB ----------------------------------------------
  !  To make PERT compatible with extended electrostatics, addition
  ! of following terms: (NKB, Wonpil Im, April,2001)
  !
  !     ATPOT0            The potential at each atom due to groups outside
  !                       the cutoff (for lambda=0)
  !     ATFX0, ATFY0,     The field at each atom (for lambda=0)
  !     ATFZ0
  !     ATGXX0, ATGYY0,   The field gradient at each atom (for lambda=0)
  !     ATGZZ0, ATGXY0,
  !     ATGYZ0, ATGZX0
  !
  !     RSPOT0            The group potential due to groups outside
  !                       the cutoff (for lambda=0)
  !     RSFX0, RSFY0,     The field at each atom (for lambda=0)
  !     RSFZ0
  !     RSGXX0, RSGYY0,   The field gradient at each atom (for lambda=0)
  !     RSGZZ0, RSGXY0,
  !     RSGYZ0, RSGZX0
  !
  !     RSQ0              All these used in the calculation of RSPOT etc in
  !                       nbonds/nbondg.src
  !     RSDX0, RSDY0,
  !     RSDZ0
  !     RSQXX0, RSQYY0,
  !     RSQZZ0, RSQXY0,
  !     RSQYZ0, RSQZX0
  !
  real(chm_real),allocatable,dimension(:) :: &
       ATPOT0,ATFX0,ATFY0,ATFZ0,ATGXX0, &
       ATGYY0,ATGZZ0,ATGXY0,ATGYZ0,ATGZX0, &
       RSPOT0,RSFX0,RSFY0,RSFZ0,RSGXX0, &
       RSGYY0,RSGZZ0,RSGXY0,RSGYZ0,RSGZX0, &
       RSQ0,RSDX0,RSDY0,RSDZ0,RSQXX0, &
       RSQYY0,RSQZZ0,RSQXY0,RSQYZ0,RSQZX0
  !--------- NKB ---------------------------------------------------------
  !====================================================================
contains

  subroutine pert_iniall()
    use number
    implicit none
    qpert=.false.
    eprtot=0.0
#if KEY_WCA==1
    ! shift softcore
    lsoftcore0 = .false.
    sccutr0 = one
    lsoftcore1 = .false.
    sccutr1 = one
#endif 
    !ML   two new logicals for MMFP in PERT
    qmmfpe=.false.
    qzegeo=.false.
    dcotot=zero
    dljtot=zero
    qpssp=.false.
    dlambd=five
    alambd=five
    ! DBEXP parameters
    laorder=3
    LAMEXP=ZERO
    NLAMEXP=0
    ILAMEXP=0
#if KEY_CHEMPERT==1
    !sbcp   disable chempert by default
    qchemp=.false.
#endif 
    return
  end subroutine pert_iniall


#endif /* (pert_fcm)  PERT*/
  !
end module pert

