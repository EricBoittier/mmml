module param
  use chm_kinds
  use chm_types
  use dimens_fcm
  character(len=*),private,parameter :: file_name   ="param_ltm.src"

  !
  !     The Parameters
  ! 
  !     Purpose:
  !
  !     To store the list of parameters used in evaluating the energy of
  !     the system.
  !
  !     I/O: Formatted and unformatted read: PARRDR in [MK.PROT]PSFRES.FOR
  !     Print and unformatted print: PARWTR in [MK.PROT]PSFRES.FOR
  !
  !     Notes: The Index column in the table below gives the domain of
  !     array. The number of entries that each array spans is specified by
  !     the number of parameters defined for that type. For example,
  !     "Atom" means the number of parameter atom types as defined by the
  !     residue topology file. "Bond" is the number of bond parameters,
  !     etc.
  !
  !     Variable  Index  Purpose
  !
  !     NATC             Number of atom types
  !     ATC       Atom   Atom Type Code (name of each atom type)
  !     ALP       Atom   Polarizability of each atom
  !     EFF       Atom   Effective number of electrons
  !     VDWR      Atom   Van der Waals radius
  !     C6DEF     Atom   Long range C6 coefficient for dispersion PME
  !                         Coding for C6DEF values: 
  !                         negative = specified value for long-range C6
  !                         zero = exclude this atom from long range
  !                         positive = derived from ALPHA,EFF, and VDWR
  !     ITC       Atom   As each atom type is read in, a counter is kept
  !                      of the atoms read in. ITC is the value of the
  !                      counter when the atom type was read in residue
  !                      topology file.
  !     NATVDW           number of different vdw types
  !     ATCCNT    Atom   Count of atom type use
#if KEY_ACE==1 /*ace_comments*/
  !     EFVOL     Atom   Effective volume
  !     CHRAD     Atom   Charge radii
  !     ACEHYD    Atom   Hydrophobic parameter (kcal/mol/A^2) relating
  !                      hydroph. energy to solvent-acc. surface area
  !     ACEA0     Atom   Solvation radius correction, parameter a0
  !     ACEA1     Atom   Solvation radius correction, parameter a1
  !     ACEA2     Atom   Solvation radius correction, parameter a2
#endif /*  (ace_comments)*/
#if KEY_FLUCQ==1 /*flucq_comments*/
  !     FQCHI     Atom   Fluctuating charge electronegativity
  !     FQZETA    Atom   Slater orbital exponent
  !     FQPRIN    Atom   Slater orbital principal quantum number
  !     FQJZ      Atom   Coulombic self-interaction term
  !     FQCHMA    Atom   Charge mass (in kcal/mol (ps/e)**2 )
#endif /* (flucq_comments)*/
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     ACTEQVM   Atom,Ateqv Array mapping atom type to equiv groups
  !     NACTEQV              Number of atom type equivalence groups
  !     ACTEQV    Ateqv      Name of each equivalence group
  !     IACTEQV   Ateqv      Code for each equivalence group
  !
#endif /* (flexparm_comments)*/
  !
  !     NCB              Number of bonds
  !     CBB       Bond   Equilibrium bond distance
  !     CBC       Bond   Force constant of a bond
  !     KCB       Bond   Key for searching for particular bonds. (Sorted)
  !     ICBCNT    Bond   Count of bond use
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     CBAI      Bond   First atom chemical type
  !     CBAJ      Bond   Second atom chemical type
#endif /* (flexparm_comments)*/
  !
  !     NCT              Number of bond angles
  !     CTB       Angle  Equilibrium bond angle (radians)
  !     CTC       Angle  Force constant for a bond angle
  !     CTUB      Angle  Urey-Bradley "term" (equilibrium 1-3 length)
  !     CTUC      Angle  Urey-Bradley "term" (1-3 bond force constant)
  !     KCT       Angle  Key for searching for particular bond angles
  !     ICTCNT    Angle  Count of angle use
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     CTAI      Angle  First atom chemical type
  !     CTAJ      Angle  Second atom chemical type
  !     CTAK      Angle  Third atom chemical type
#endif /* (flexparm_comments)*/
  !
  !     NCP              Number of torsions
  !     CPB       Phi    Phase shift (delta) for the dihedral (radians)
  !     CPCOS     Phi    Cos(CPB)
  !     CPSIN     Phi    Sin(CPB)
  !     CPC       Phi    Force constant for the dihedral energy
  !     CPD       Phi    Periodicity of the dihedral energy
  !     KCP       Phi    Key for searching for particular torsions
  !     ICPCNT    Phi    Count of torsion use
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     CPAI      Phi    First atom chemical type
  !     CPAJ      Phi    Second atom chemical type
  !     CPAK      Phi    Third atom chemical type
  !     CPAL      Phi    Fourth atom chemical type
#endif /* (flexparm_comments)*/
  !
  !     NCI       Imphi  Number of improper torsions
  !     CIB       Imphi  Equilibrium improper torsion angle (radians)
  !     CICOS     Imphi  Cos(CIB)
  !     CISIN     Imphi  Sin(CIB)
  !     CIC       Imphi  Force constant for the improper torsion
  !     CID       Imphi  Periodicity for improper torsion (a value of
  !                      zero will use harmonic potential).
  !     KCI       Imphi  Key for searching for particular improper
  !                      torsions. (Sorted)
  !     ICICNT    Imphi  Count of improper torsion use
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     CIAI      Imphi    First atom chemical type
  !     CIAJ      Imphi    Second atom chemical type
  !     CIAK      Imphi    Third atom chemical type
  !     CIAL      Imphi    Fourth atom chemical type
#endif /* (flexparm_comments)*/
  !
  !     KCH       HBond  Key for searching for particular hydrogen
  !                      bonds. (Sorted)
  !     NCH       HBond  Number of hydrogen bonds
  !     CHBA      HBond  Coefficient of r**-12 term in the hydrogen bond
  !                      potential
  !     CHBB      HBond  Coefficient of r**-10 term in the hydrogen bond
  !                      potential
  !     HBEXPN    4      Exponents for the hydrogen bond function.
  !                      (1) - repulsive term
  !                      (2) - attractive term
  !                      (3) - A-H-D cos exponent
  !                      (4) - AA-A-H cos exponent
  !                      Note, entries must be positive even integers.
  !     ICHCNT    Phi    Count of hydrogen bond use
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     CHAD      HBond  Hbond donor chemical type
  !     CHAA      HBond  Hbond acceptor chemical type
#endif /* (flexparm_comments)*/
  !
  !     NCN       NBond  Number of non-bonded interactions.
  !     CNBA      NBond  Value of SIGMA**2 for van der Waal terms; 1-4 terms offset by MAXCN
  !     CNBB      NBond  Value of epsilon for van der Waal terms; 1-4 terms offset by MAXCN
  !     KCN       Nbond  Key for non-bonded interactions. Unused.
  !
  !     NBFIXN             Number of nonbond fixes
  !     NBFIXI    2,NBFIXN Integers for NBFIX
  !     NBFIXR    4,NBFIXN Reals for NBFIX
  !     QNBFIX             Flag for NBFIX that allows READ PARAm APPEnd
  !
  !     QFLXPARM         Logical indicating flexible paramter reading
  !     QGEOMVDW         Logical indicating OPLS type VDW combining rules
  !
  !     Other parameters used in the common file (from fcm/dimens.fcm):
  !
  !     MAXATC           Maximum number of chemical types
  !     MAXCB            Maximum number of bonds
  !     MAXCH            Maximum number of hydrogen bonds
  !     MAXCI            Maximum number of improper torsions
  !     MAXCN            Maximum number of non-bonded interactions.
  !     MAXCP            Maximum number of torsions
  !     MAXCT            Maximum number of bond angles
  !     MAXNBF           Maximum number of nonbond fixes
#if KEY_FLEXPARM==1 /*flexparm_comments*/
  !     MAXACTEQV        Maximum number of atom equivalences
#endif /* (flexparm_comments)*/
  !
  !-----------------------------------------------------------------------
  ! integers

  integer, allocatable,dimension(:,:) :: nbfixi
  INTEGER      :: NATC, NCB, NCT, NCP, NCI, NCN, NCH, NBFIXN, &
  !    NBFIXI(2,MAXNBF), NATVDW, &
       NATVDW, &
       CPD(MAXCP), CID(MAXCI), HBEXPN(4), ITC(MAXATC), &
       KCB(MAXCB), KCT(MAXCT), &
#if KEY_FLUCQ==1
       FQPRIN(MAXATC)    ,                                  & 
#endif
       KCH(MAXCH) 

#if KEY_FLEXPARM==1 /*integer_decl2*/
  integer  NACTEQV, ACTEQVM(MAXATC,MAXACTEQV) &
       , IACTEQV(MAXACTEQV), CBAI(MAXCB), CBAJ(MAXCB) &
       , CTAI(MAXCT), CTAJ(MAXCT), CTAK(MAXCT) &
       , CPAI(MAXCP), CPAJ(MAXCP), CPAK(MAXCP), CPAL(MAXCP) &
       , CIAI(MAXCI), CIAJ(MAXCI), CIAK(MAXCI), CIAL(MAXCI) &
       , CHAD(MAXCH), CHAA(MAXCH)
#endif /* (integer_decl2)*/
  integer ATCCNT(MAXATC), ICBCNT(MAXCB), ICTCNT(MAXCT), &
       ICPCNT(MAXCP), ICICNT(MAXCI), ICHCNT(MAXCH), &
  ! M. G. Lerner, B. T. Miller, QANGTYPE determines if GROMACS-style
  ! angle param is used. -1 means GROMACS only, 1 means CHARMM only, 0
  ! means mixed
       QANGTYPE

  ! chm_int8 arrays
  integer(chm_int8) :: kcp(maxcp), kci(maxci)

  ! Logicals
  LOGICAL QNBFIX, QGEOMVDW, QFLXPARM

  ! reals
  real(chm_real), allocatable, dimension(:,:) :: nbfixr
  real(chm_real), allocatable, dimension(:) :: cnba, cnbb
  real(chm_real) :: CBC(MAXCB), CBB(MAXCB), &
       CTC(MAXCT), CTB(MAXCT), CTUC(MAXCT), CTUB(MAXCT), &
       CPC(MAXCP), CPB(MAXCP), CPCOS(MAXCP),CPSIN(MAXCP), &
       CIC(MAXCI), CIB(MAXCI), CICOS(MAXCI),CISIN(MAXCI), &
!       CNBA(2*MAXCN), CNBB(2*MAXCN) &
       CHBA(MAXCH), CHBB(MAXCH), ALP(MAXATC*2) &
#if KEY_ACE==1
       , EFVOL(MAXATC), CHRAD(MAXATC), ACEHYD(MAXATC)       & 
#endif
#if KEY_ACE==1
       , ACEA0(MAXATC), ACEA1(MAXATC), ACEA2(MAXATC)        & 
#endif
#if KEY_FLUCQ==1
       , FQCHI(MAXATC), FQZETA(MAXATC)                      & 
#endif
#if KEY_FLUCQ==1
       , FQJZ(MAXATC), FQCHMA(MAXATC)                       & 
#endif
  !    , EFF(MAXATC*2), VDWR(MAXATC*2), NBFIXR(4,MAXNBF)
       , EFF(MAXATC*2), VDWR(MAXATC*2), C6DEF(MAXATC)

  !
  ! characters
  !            ;
  CHARACTER(LEN=8) ATC(MAXATC) &
#if KEY_FLEXPARM==1
       , ACTEQV(MAXACTEQV) &  
#endif
       ;

contains

  subroutine param_iniall()
    use number
    implicit none
    natc=0
    ncb=0
    nct=0
    ncp=0
    nci=0
    nch=0
    ncn=0
    nbfixn=0
    qangtype=1

#if KEY_FLEXPARM==1 /*flexparm_init*/
    qflxparm=.false.
    nacteqv=0
#endif /* (flexparm_init)*/
    !
#if KEY_ACE==1 /*ace_par_init*/
    efvol (1:maxatc) = minone
    chrad (1:maxatc) = zero
    acehyd(1:maxatc) = zero
    acea0 (1:maxatc) = one
    acea1 (1:maxatc) = zero
    acea2 (1:maxatc) = zero
#endif /* (ace_par_init)*/
#if KEY_FLUCQ==1 /*flucq_par_init*/
    fqchi (1:maxatc) = zero
    fqzeta(1:maxatc) = zero
    fqjz  (1:maxatc) = zero
#endif /* (flucq_par_init)*/

    return
  end subroutine param_iniall
  
  subroutine allocate_param_ltm()
    use memory
    use dimens_fcm, only: maxcn

    implicit none
    
    character(len=*),parameter :: routine_name="allocate_param_ltm"
    integer :: maxcn2

    maxcn2 = 2 * maxcn
    
    call chmalloc(file_name,routine_name,'nbfixi',2,maxnbf,intg=nbfixi)
    call chmalloc(file_name,routine_name,'nbfixr',4,maxnbf,crl=nbfixr)
    call chmalloc(file_name,routine_name,'cnba', maxcn2, crl=cnba)
    call chmalloc(file_name,routine_name,'nbfixr', maxcn2, crl=cnbb)

    cnba(1:maxcn2) = 0.0
    cnbb(1:maxcn2) = 0.0
    
    return
  end subroutine allocate_param_ltm
end module param

