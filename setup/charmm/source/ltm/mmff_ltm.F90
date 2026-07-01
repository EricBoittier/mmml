module mmffm
  use chm_kinds
  use dimens_fcm

  ! replaces old EQUIVALENCE
  use psf, only: &
       mtype => IAC, &   ! numeric atom types
       PartlQ => CG, &   ! partial atomic charge
       AtNames => ATYPE  ! atom names

  ! replaces old EQUIVALENCE    (mfc---only slightly)
  use param, only: &
       BondEq => CBB, &  ! Equilibrium bond distance
       BondFC => CBC, &  ! bond-stretching force constant
       AnglEq => CTB, &  ! angle equilibrium value
       AnglFC => CTC, &  ! bending force constant
       nvdwm => NCN, &
       TorPr => CPD, &   ! torsional periodicity
       TorPh => CPB, &   ! torsional phase
       rstar => CNBA, &  ! Value of SIGMA**2 for van der Waal terms
       estar => CNBB     ! Value of epsilon for van der Waal terms

  !
  !   estar(i,j)    van-der-Waals well depth (kcal/mol) for the vdW
  !                 interaction between atoms of MMFF atom types i
  !                 and j
  !
  !   rstar(i,j)    van-der-Waals minimum-energy separation (A) for the
  !                 vdW interaction between atoms of MMFF atom types i
  !                 and j
  !
  !      Note:  i and j here label MMFF atoms types (and not atom
  !             numbers). ESTAR and RSTAR are constructed directly from
  !             the data in mmffvdw.par or in the supplementary
  !             parameter file by subroutine rdvdwm, and are used by
  !             the routines which compute vdW interactions
  !

#if KEY_MMFF==1 /*mmff_fcm*/

  implicit none
  character(len=*),private,parameter :: file_name   ="mmff_ltm.src"

#if KEY_MMFF==1
    !
    !       ADDITIONAL PARAMETERS FOR RING_FIND ROUTINES
    !       JAY BANKS, 24 SEPT 1993.
    !
    !       MAX_RINGSIZE, MAX_EACH_SIZE, and MAXPATHS are array dimensions.
    integer,parameter :: MAX_RINGSIZE = 20, MAX_EACH_SIZE = 1000
    integer,parameter :: MAXPATHS = 8000
    !       MAX_TO_SEARCH is passed as an argument to ring_find, telling it the
    !       largest size ring it should look for.
    integer,parameter :: MAX_TO_SEARCH = 6
#endif 

  !
  ! This common file contains information needed by MMFF code.
  !            Created by Jay Banks on Friday, the 13th of October 1995
  !
  ! 08 Nov 95: Changed PTYPE (parent atom type) to PATYPE to avoid conflict
  ! with "pressure type" (external or internal) in reawri.fcm.  JLB
  !
  !
  ! 01 Dec 95: JLB Added MDSTBN (Stretch-Bend force field class) array to
  ! common block PSFI_MM.
  !
  !-----------------------------------------------------------------------
  !CHARMM Element source/fcm/mmffarom.fcm 1.1
  !
  !   MMFFAROM.INC - HOLDS INFORMATION USED TO RELATE AROMATIC
  !   SYMBOLIC TYPES TO THE ORIGINALLY ASSIGNED SYMBOLIC TYPES
  !   NTRAROM = NUMBER OF ENTRIES USED IN TRANSLATING AROMATIC TYPES
  !
  !   IANUM   = ATOMIC NUMBERS
  !   IRSIZE  = RING SIZE (EITHER 5 OR 6)
  !   L5POS   = FOR A 5-RING, THE POSITION RELATIVE TO THE UNIQUE
  !             LONE-PAIR HETEROATOM (1=ALPHA, 2=BETA, 4= )
  !   LIMCAT  = 1 IF THIS ATOM TYPE CAN OCCUR IN AN IMADIZOLIUM OR
  !             OTHER 5-RING CATION, 0 OTHERWISE
  !   LN5AN   = 1 IF THIS ATOM TYPE CAN OCCUR IN A TETRAZOLE OR
  !             OTHER 5-RING ANION, 0 OTHERWISE
  !   SYMOLD  = ORIGINAL (INPUT) ATOM TYPE
  !   SYMAROM = NEW AROMATIC ATOM TYPE INTO WHICH SYMOLD IS MAPPED
  !
  INTEGER NTRAROM
  !integer, IANUM(MAXATC),IRSIZE(MAXATC),L5POS(MAXATC),&
  !     LIMCAT(MAXATC),LN5AN(MAXATC)
  integer, allocatable, dimension(:)  :: IANUM,IRSIZE,L5POS,&
       LIMCAT,LN5AN
  !
  !CHARACTER(len=4) SYMOLD(MAXATC),SYMAROM(MAXATC)
  CHARACTER(len=4), allocatable, dimension(:) :: SYMOLD,SYMAROM
  !
  !CHARMM Element source/fcm/mmffhdef.fcm 1.1
  !   MMFFHDEF.INC - RELATES SYMBOLIC MMFF ATOM TYPES FOR HYDROGENS
  !   TO THOSE OF THE PARENT ATOM TO WHICH EACH IS ATTACHED
  !
  !   NTRHYD  = NUMBER OF PATYPE, HDTYPE PAIRS READ IN
  !   PATYPE   = MMFF SYMBOLIC ATOM TYPE FOR THE PARENT ATOM
  !   HDTYPE  = MMFF SYMBOLIC ATOM TYPE TO BE ASSIGNED TO AN
  !             ATTACHED HYDROGEN
  !
  !CHARACTER(len=4) PATYPE(MAXATC), HDTYPE(MAXATC)
  CHARACTER(len=4), allocatable, dimension(:) :: PATYPE, HDTYPE
  INTEGER NTRHYD
  !
  !CHARMM Element source/fcm/mmffprop.fcm 1.1
  !   MMFFPROP.INC - specifies "properties" of the MMFF atom types
  !
  !  MSPEC   = ATOMIC NUMBER OF THE NTH MMFF NUMERIC ATOM TYPE
  !            (removed: replaced by AtNumT in rtf.fcm)
  !  MCOORD  = NUMBER OF ATTACHED ATOMS
  !  MVALMIN = MINIMUM NUMBER OF BONDS PERMITTED TO ATTACHED ATOMS
  !  MVALMAX = MAXIMUM NUMBER OF BONDS PERMITTED TO ATTACHED ATOMS
  !  MPILP   = 1 IF THIS ATOM TYPE HAS ONE OR MORE LONE PAIRS
  !            WHICH MIGHT INTERACT THROUGH RESONANCE WITH ATTACHED
  !            CENTERS
  !  MLTBND  = 2 IF THIS ATOM TYPE CAN PARTICIPATE IN DOUBLE BONDS
  !  MLTBND  = 3 IF THIS ATOM TYPE CAN PARTICIPATE IN TRIPLE BONDS
  !  MLTBND  = 1 IF THIS ATOM TYPE CAN PARTICIPATE IN ESPECIALLY
  !            STRONGLY DELOCALIZED "SINGLE" BONDS
  !  MLTBND  = 0 OTHERWISE
  !  MAROM   = 1 IF THIS IS AN AROMATIC ATOM TYPE, 0 OTHERWISE
  !  MLINBA  = 1 OF THIS ATOM TYPE FORMS NOMINALLY LINEAR BOND ANGLES
  !  MSP2SGL = 1 IF THIS ATOM TYPE IS SP2-HYDRIDIZED AND CAN FORM
  !            FORMALLY SINGLE BONDS WITHAN  ATTACHED SP2 ATOM
  !
  !INTEGER MCOORD(MAXDEFI),MVALMIN(MAXDEFI), &
  !     MVALMAX(MAXDEFI),MPILP(MAXDEFI),MLTBND(MAXDEFI),MAROM(MAXDEFI), &
  !     MLINBA(MAXDEFI),MSP2SGL(MAXDEFI)
  INTEGER, allocatable, dimension(:) :: MCOORD,MVALMIN, &
       MVALMAX,MPILP,MLTBND,MAROM, &
       MLINBA,MSP2SGL
  !
  !CHARMM Element source/fcm/bondstr.fcm 1.1
  !   BONDSTR.INC - HOLDS QUANTITIES USED IN ASSIGNING DEFAULT
  !   VALUES FOR BOND STRETCHING FORCE CONSTANTS VIA AN INVERSE
  !   6TH POWER RULE
  !
  !   BL1  = REFERENCE (USUALLY SINGLE) BOND LENGTH IN ANGSTROMS
  !   BK1  = BOND STRETCHING FORCE CONSTANT IN MDYNES/ANGSTROM
  !          FOR BOND LENGTH BL1
  !   IA   = ATOMIC NUMBERS FOR THE TWO BONDED ATOMS
  !
  !real(chm_real) BL1(MAXCB), BK1(MAXCB)
  !INTEGER IA(2,MAXCB)
  real(chm_real), allocatable, dimension(:) :: BL1, BK1
  INTEGER, allocatable, dimension(:,:) :: IA
  integer NBNDCON
  !
  !CHARMM Element source/fcm/equiv.fcm 1.1
  ! mdef is a 5 x nmmff array of numerical atom type equivalences
  ! allocated in mmff/readpar.src subroutine RDDEFI
   integer, allocatable, dimension(:,:) ::  mdef

 !     CHARACTER(len=4) CHDEF     !??? MMFF symbolic atom types (unused)

  !CHARMM Element source/fcm/qdef.fcm 1.1
  !   QDEF.INC - HOLDS "PARTIAL BOND CHARGE INCREMENTS" USED IN
  !   ASSIGNING DEFAULT VALUES FOR MISSING BOND CHARGE INCREMENTS
  !   AND FORMAL-CHARGE ADJUSTMENT FACTORS
  !
  !   PBCI  =  PARTIAL BOND CHARGE INCREMENT FOR THE NTH NUMERIC
  !            MMFF ATOM TYPE
  !   FCADJ =  FACTOR INDICATING WHAT PART OF ANY FORMAL CHARGE
  !            ASSIGNED TO AN ATOM OF ATOM TYPE NTH IS TO BE
  !            TRANSFERED TO EACH OF THE ATTACHED ATOMS BEFORE THE
  !            BOND CHARGE INCREMENTS ARE USED TO DEFINE THE FINAL
  !            CHARGE DISTRIBUTION
  !   LPBCI =  .TRUE. IF A PARTIAL BOND CHARGE INCREMENT HAS BEEN
  !            READ IN FOR THE NTH NUMERIC ATOM TYPE
  !
  !real(chm_real) PBCI(MAXDEFI), FCADJ(MAXDEFI)
  !LOGICAL LPBCI(MAXDEFI)
  real(chm_real), allocatable, dimension(:) :: PBCI, FCADJ
  LOGICAL, allocatable, dimension(:) ::  LPBCI
  !
  !CHARMM Element source/fcm/vector.fcm 1.1
  real(chm_real) eij(4), eik(4), eji(4), ejk(4), ejl(4), &
       eki(4), ekj(4), ekl(4), elj(4), elk(4)
  ! MMFF-specific parameters from param.fcm (those not equivalenced to others).

  INTEGER, allocatable, dimension(:) :: KCOOP, KCSTBN
  integer NCBT,NCQ,NCQT,NCTT,NCOOP,NCOOPT,NCPT, &
       NCSB,NCSBT,NCDFSB
  !
  !real(chm_real) OoplFC(MAXCT) ! out-of-plane bending force constant
  !real(chm_real) AuxPar(MAXAUX) ! auxiliary parameters
  real(chm_real), allocatable, dimension(:) :: OoplFC ! out-of-plane bending force constant
  real(chm_real), allocatable, dimension(:) :: AuxPar ! auxiliary parameters
  !
  !CHARMM Element source/fcm/auxpar.fcm 1.1
  ! Keeps pointers to AuxPar array
  integer, parameter :: cstr=1   ! coefficient of MMFF cubic stretch terms
  integer, parameter :: cbnd=2   ! coefficient of MMFF cubic BENDING terms
  integer, parameter :: THRESH=3 ! treshold for energy printout

  !   b_source(n)  text (comment/history) field read from mmffbond.par
  !
  !character(len=36) b_source(MAXCB)
  character(len=36), allocatable,dimension(:) :: b_source
  ! mmff's ptheta.INC
  !
  !
  !                    Note: the out-of-plane interaction is
  !                    COMPUTED ONLY IF LTHETA(N) IS NONZERO.
  !                    The reference out-of-plane angle is
  !                    assumed to be zero.
  !
  !     equivalence(kvndx(1),kcp(1)) ! index for thetas
  !cc      character(len=36) a_source,o_source
  ! mmff's ptor.INC
  !
  !character(len=36) t_source(MAXCP)
  character(len=36), allocatable, dimension(:) :: t_source
  !
  !   KCP(=kvndx)  the packed canonical index (see subroutine omegcon)
  !              for the nth i-j-k-l torsion parameter set read from
  !              the mmfftor.par or supplementary parameters files
  !
  !   v1(n)      1-fold torsion parameter in kcal/mol for the nth
  !              torsion parameter set
  !
  !   v2(n)      2-fold torsion parameter in kcal/mol for the nth
  !              torsion parameter set
  !
  !   v3(n)      2-fold torsion parameter in kcal/mol for the nth
  !              torsion parameter set
  !
  !   t_source(n)  text (comment/history) field for the nth torsion
  !                parameter set read from mmfftor.par
  !
  ! mmff's  pstbn.INC
  !
  !real(chm_real) stbnp(2,MAXCT)
  !character(len=36) ba_source(MAXCT)
  real(chm_real), allocatable, dimension(:,:) :: stbnp
  character(len=36), allocatable, dimension(:) :: ba_source
  !
  ! KCSTBN(=kstbn) the packed canonical index (see subroutine stbncon) for
  !                the nth stretch-bend interaction read from mmffstbn.par
  !
  !   stbnp(l,n)  the stretch-bend force constants (mdynes/rad) for the
  !               i-j/i-j-k and j-k/i-j-k interactions for the nth
  !               stretch-bend parameter
  !
  !   ba_source   text (comment/history) field read for the nth stretch-
  !               bend parameter read from mmffstbn.par
  !
  ! pq.INC
  !
  !integer ichgndx(MAXCB)
  !real(chm_real) bci(MAXCB)
  !
  !character(len=36) bci_source(MAXCB)
  integer, allocatable, dimension(:) :: ichgndx
  real(chm_real), allocatable, dimension(:) :: bci
  !
  character(len=36), allocatable, dimension(:) :: bci_source
  !
  !   ichgndx(n)   the packed canonical index (see subroutine bondcon)
  !                for the nth bond-charge-increment parameter read
  !                from the mmffchg.par or supplementary parameter files
  !
  !   bci(n)       the bond charge increment for the nth parameter
  !
  !   bci_source(n)  text (comment/history) field read from mmffchg.par
  !
  !   default stretch-bend parameters
  !
  real(chm_real) DEFSBK(0:4,1:4,0:4)
  !
  ! MMFF-specific parameter array pointers, from code.fcm in MSI version
  !     ICOOP      Pointer for out-of-plane parameters (for MMFF)
  !     ICSTBN     Pointer for stretch-bend parameters
  INTEGER,allocatable,dimension(:) :: ICOOP, ICSTBN
  !
  ! stbnk.INC
  !
  integer,allocatable,dimension(:,:) :: StrbList ! Stretch-Bend coupling list
  !
  ! StrbList(na,ib) the two values of ib identify the numbers of the
  !                 i-j and j-k bonds in which comprise the i-j-k angle
  !                 DEFINED IB AnglList(NA,). THESE VALUES OF IB INDEX
  !                 entries in the BondList array ; na
  !                 indexes entries in the AnglList array (see thetak.INC)
  !
  !====================================================================
  ! MMFF-SPECIFIC, NON-EQUIVALENCED DATA FROM PSF.FCM
  !     BondType  Bond     Bond Type (1,2 or 3) (for MMFF)
  !     MDBOND    Bond     the "force field class" (MMFF)
  !
  !     LTHETA    Angle    Fourth atom of bond angle (for MMFF oopl)
  !     MDTHET    Angle    "force field class" (MMFF)
  !
  !     MDOMEG    Phi      "force field class" (MMFF)
  !
  !     MDSTBN    Str-Bnd  "force field class" (MMFF)
  !
  INTEGER,allocatable,dimension(:) :: BondType, MDBOND, LTHETA, MDTHET, MDOMEG, MDSTBN
  integer,allocatable,dimension(:) ::  AtNum ! atomic numbers for MMFF
  integer,allocatable,dimension(:,:) ::  ITAB !
  !     ITAB(1..5,i) : list of atoms bonded to atom "i" ended by -1
  !     ITAB(6,i)    : number of atoms bonded to given atom
  integer,allocatable,dimension(:) ::  locat  !?
  !
  ! T.A. Halgren change
  !    ARBOND(2,) Bond   Atom indices for bonds in aromatic rings
  !    N_ARBOND          Number of aromatic bonds
  INTEGER,allocatable,dimension(:,:) ::  AR_BOND
  integer :: N_ARBOND
  !
  ! end T.A. Halgren change
  !
  ! mmff's anames.INC
  !
  CHARACTER(len=4),allocatable,dimension(:) :: RESNAME ! residue name ~RESID (in CHARMM)
  CHARACTER(len=4),allocatable,dimension(:) :: KRES    ! residue type (e.g. ALA, GLU etc...)
  !
  ! mmff's atmprop.INC
  !
  integer,allocatable,dimension(:) :: nichg
  character(len=4),allocatable,dimension(:) :: symb
  logical,allocatable,dimension(:) :: ifarom
  !
  !   nichg(i)     index defining the purely integral formal atomic charge
  !                on atom i as recorded in the Merck MOLEDIT file format
  !
  !   symb(i)      mmff sybmolic atom type for atom i
  !
  !   ifarom(i)    .true. if i is an atom in an aromatic ring system
  !
  ! =====================================================================
  integer,allocatable,dimension(:) :: INRING   ! N if atom is in (nested) ring N, 0 otherwise
  integer IRINGS           ! number of nested ring systems
  ! =====================================================================
  logical,allocatable,dimension(:) :: IFNPDP   ! .true. if quaternary (positively charged)
  !                                nitrogen in a pyridine-type ring

#if KEY_TSM==1
  ! VPRTRR: REACTANT PERTURBATION POTENTIAL ENERGY VDW REPULSIVE
  ! VPRTRP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  ! VPRTAR: REACTANT PERTURBATION POTENTIAL ENERGY VDW ATTRACTIVE
  ! VPRTAP: PRODUCT PERTURBATION POTENTIAL ENERGY "
  real(chm_real) VPRTRR,VPRTRP,VPRTAR,VPRTAP
#endif 
  !

contains
  subroutine allocate_mmff()
    use memory
    character(len=*),parameter :: routine_name="allocate_mmff"

    call chmalloc(file_name,routine_name,'ianum',maxatc,intg=ianum)
    call chmalloc(file_name,routine_name,'irsize',maxatc,intg=irsize)
    call chmalloc(file_name,routine_name,'l5pos',maxatc,intg=l5pos)
    call chmalloc(file_name,routine_name,'limcat',maxatc,intg=limcat)
    call chmalloc(file_name,routine_name,'ln5an',maxatc,intg=ln5an)
    
    call chmalloc(file_name,routine_name,'symold',maxatc,ch4=symold)
    call chmalloc(file_name,routine_name,'symarom',maxatc,ch4=symarom)
    
    call chmalloc(file_name,routine_name,'patype',maxatc,ch4=patype)
    call chmalloc(file_name,routine_name,'hdtype',maxatc,ch4=hdtype)
    
    call chmalloc(file_name,routine_name,'mcoord',maxdefi,intg=mcoord)
    call chmalloc(file_name,routine_name,'mvalmin',maxdefi,intg=mvalmin)
    call chmalloc(file_name,routine_name,'mvalmax',maxdefi,intg=mvalmax)
    call chmalloc(file_name,routine_name,'mpilp',maxdefi,intg=mpilp)
    call chmalloc(file_name,routine_name,'mltbnd',maxdefi,intg=mltbnd)
    call chmalloc(file_name,routine_name,'marom',maxdefi,intg=marom)
    call chmalloc(file_name,routine_name,'mlinba',maxdefi,intg=mlinba)
    call chmalloc(file_name,routine_name,'msp2sgl',maxdefi,intg=msp2sgl)

    marom(1:maxdefi) = 0
    mltbnd(1:maxdefi) = 0
    msp2sgl(1:maxdefi) = 0

    call chmalloc(file_name,routine_name,'bl1',maxcb,crl=bl1)
    call chmalloc(file_name,routine_name,'bk1',maxcb,crl=bk1)
    
    call chmalloc(file_name,routine_name,'ia',2,maxcb,intg=ia)
    
    call chmalloc(file_name,routine_name,'pbci',maxdefi,crl=pbci)
    call chmalloc(file_name,routine_name,' fcadj',maxdefi,crl=fcadj)
    
    call chmalloc(file_name,routine_name,'lpbci',maxdefi,log=lpbci)
    
    call chmalloc(file_name,routine_name,'kcoop',maxct,intg=kcoop)
    call chmalloc(file_name,routine_name,' kcstbn',maxct,intg= kcstbn)
    
    call chmalloc(file_name,routine_name,'ooplfc',maxct,crl=ooplfc)
    call chmalloc(file_name,routine_name,'auxpar',maxaux,crl=auxpar)
    
    call chmalloc(file_name,routine_name,'b_source',maxcb,ch36=b_source)
    call chmalloc(file_name,routine_name,'t_source',maxcp,ch36=t_source)
    
    call chmalloc(file_name,routine_name,'stbnp',2,maxct,crl=stbnp)
    call chmalloc(file_name,routine_name,'ba_source',maxct,ch36=ba_source)
    
    call chmalloc(file_name,routine_name,'ichgndx',maxcb,intg=ichgndx)
    call chmalloc(file_name,routine_name,'bci',maxcb,crl=bci)
    call chmalloc(file_name,routine_name,'bci_source',maxcb,ch36=bci_source)

    call chmalloc(file_name,routine_name,'bondtype',maxb  ,intg=bondtype)
    call chmalloc(file_name,routine_name,'mdbond  ',maxb  ,intg=mdbond)
    call chmalloc(file_name,routine_name,'ltheta  ',maxb  ,intg=ltheta)
    call chmalloc(file_name,routine_name,'mdthet  ',maxt  ,intg=mdthet)
    call chmalloc(file_name,routine_name,'mdomeg  ',maxp  ,intg=mdomeg)
    call chmalloc(file_name,routine_name,'mdstbn  ',maxt  ,intg=mdstbn)
    call chmalloc(file_name,routine_name,'icoop   ',maxt  ,intg=icoop)
    call chmalloc(file_name,routine_name,'icstbn  ',maxt  ,intg=icstbn)
    call chmalloc(file_name,routine_name,'strblist',2,maxt,intg=strblist)
    call chmalloc(file_name,routine_name,'atnum   ',maxaim,intg=atnum)
    call chmalloc(file_name,routine_name,'itab    ',6,maxaim,intg=itab)
    call chmalloc(file_name,routine_name,'locat   ',maxaim,intg=locat)
    call chmalloc(file_name,routine_name,'ar_bond ',2,maxb,intg=ar_bond)
    call chmalloc(file_name,routine_name,'resname ',maxaim,ch4=resname)
    call chmalloc(file_name,routine_name,'kres    ',maxaim,ch4=kres)
    call chmalloc(file_name,routine_name,'symb    ',maxaim,ch4=symb)
    call chmalloc(file_name,routine_name,'ifarom  ',maxaim,log=ifarom)
    call chmalloc(file_name,routine_name,'nichg   ',maxaim,intg=nichg)
    call chmalloc(file_name,routine_name,'inring  ',maxaim,intg=inring)
    call chmalloc(file_name,routine_name,'ifnpdp  ',maxaim,log=ifnpdp)
    mdbond=0
    mdthet=0
    irings=-1
    ltheta = 0
    icoop = 0
    ifarom = .false.
    nichg = 0

    return
  end subroutine allocate_mmff
#endif /* (mmff_fcm)*/

end module mmffm

