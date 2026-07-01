module mc
  use chm_kinds
  use chm_types
  use dimens_fcm

  implicit none

#if KEY_MC==1 /*mc_fcm*/
  !
  !     MOVEAD establishes each of the following pointers for all move types.
  !     Each is a pointer to a dynamically allocated array that is NMVATM 
  !     elements long, where NMVATM is equal to the number of move instances 
  !     in that group.  In all cases, if the array does not apply to a 
  !     particular move, it is not allocated.
  !    
  !     MDXP      This array contains the information about the limits of the
  !               move.  For isotropic or one-dimensional moves, it is simply
  !               an NMVATM-long array of reals containing the maximum
  !               displacement.  If the displacements are to be drawn from an 
  !               anisotropic volume, the array is a list of pointers, each of 
  !               which points to an array of 9 reals that make up the matrix 
  !               that transforms the unit sphere into the appropriate ellipsoid.
  !
  !     IBLSTP    A list of NMVATM pointers, each of which points to
  !               a list of bonded terms changing under that move instance.  
  !               For each element in the four element array QBND (bonds=1, 
  !               angles=2, dihedrals=3, impropers=4, path integral springs=5) 
  !               that is true, there is an element listing the index of the 
  !               final element containing indices of that bonded term type 
  !               followed by the list of terms themselves.  This list is then 
  !               followed by a similar one for the next bonded term type with 
  !               QBND(i) set to true.  
  !
  !               For example, if bonds 3, 8, and 10 and angles 16 and 17 
  !               were changing, the QBND array would contain (T T F F F) and the 
  !               list would contain (4 3 8 10 7 16 17).
  !
  !     IPIVTP    This array keeps track of any pivot or special atoms.
  !               If there is only one pivot atom, then it is stored in the
  !               array.  If there are multiple (e.g., 2 for a TORS move
  !               and 14 for a CROT move), the list stores a pointer to 
  !               a list containing the pivot atoms.
  !
  !     IMVNGP    This array contains a compact list of the moving atoms.
  !               Each element contains a pointer to a list of the following
  !               form.  The first element in the list is 1 more than the 
  !               number of rigid groups (NG).  Elements 2 to NG contain the
  !               index of the last array element with information about that
  !               rigid group.  The atoms in a rigid group are stored as 
  !               the first and last atoms in a contiguous range of atom indices.
  !
  !     QBND      See IBLSTP above.  MBONDT is the number of different bonded 
  !               terms.  For now, MBONDT = 5 (again, see above).
  !
  !     MVTYPE    An integer associated with the type of move.  See MOVEAD.
  !
  !
  !     In addition, MOVEAD associates with each group of move instances 
  !     several parameters. 
  !
  !     WEIGHT      The relative weight of that group of move instances in 
  !                 the complete move set.  The probability of picking a 
  !                 group of move instances with weight w_i is w_i/(sum_j w_j)
  !                 where (sum_j w_j) is the total of all the WEIGht values.
  !
  !     TFACT       A scale factor for the temperature used in MCACPT. 
  !
  !     ARMP        ARM target probability of move instance acceptance.
  !
  !     ARMA, ARMB  Parameters to avoid taking the logarithm of zero in ARM:
  !       
  !                 DMAX(new) = DMAX(old)*ln(ARMA*ARMP+ARMB)/ln(ARMA*obsP+ARMB)
  !
  !                 where obsP is the observed probability of accepting that
  !                 move instance.  
  !      
  !     ARMLIM      Whether there is a limit to the size of the move --- 
  !                 for example, a rotation cannot be bigger than 180 due to the
  !                 periodicity.
  !
  !     ARMMAX      The limit if ARMLIM is .TRUE.
  !    
  !     DOMCF       The F factor in DOMC:
  !
  !                 DMAX(new) = DOMCF*SQRT[(d2ave*TEMP)/Eave]
  !
  !                 where d2ave is the observed average square of the
  !                 displacement and Eave is the observed average change in 
  !                 energy (both averages are done over all moves, not just those
  !                 accepted).  DOMCF is used for the anisotropic version of
  !                 this equation as well.   In the event that the square 
  !                 root of a negative number must be taken, the routine 
  !                 branches to ARM optimization, so ARMA, ARMB, and ARMP 
  !                 should be set even if one plans on using DOMC.
  !
  !    ANISO        DOMC anisotropic optimization of the volume from which the 
  !                 moves are chosen.  If ANISotropic is 0, it is off (isotropic)
  !                 and, if ANISotropic is non-zero, it is on.  At present, 
  !                 only 3D Cartesian moves (RTRN and CART) allow anisotropic 
  !                 optimization.
  !
  !    NLIMIT       Number of dihedrals limited by MDXP in CROT (0 or 1).
  !
  !    MVLABL       An optional tag for the group of move instances.
  !                 Only the first four characters are retained.  All sets of
  !                 move instances are also given an integer index which can
  !                 be used instead.
  !
  !    The following variables are for MOVE LINKing.
  !     
  !    NXTMVG      Next move group in chain of linked moves.
  !    IACMVG      List of active (primary) move groups.
  !    NACMVG      Number of active move groups.
  !    ILNMVP      IMVNGP type list for a chain of linked moves.
  !    ILNBDP      IBLSTP type list for a chain of linked moves.
  !    QLNBND      QBND   type list for a chain of linked moves.
  !
  !    IGCMVG      GCMC move group associated with a move.  The purpose of this
  !                array is to avoid picking non-active atoms for displacement.
  !
  !    The variables not associated with the move set that are defined in
  !    this file are:
  !
  !    ISDMC        The seed for the random number generator.  It is common so
  !                 that it can perpetuate from one call of MC to another.  This
  !                 allows one to seed at the top of the script (with a zero 
  !                 step MC call) and then call MC repeatedly (without explicitly
  !                 re-seeding) and have it produce different results each time.
  !
  !    RVOLMC       Volume for constant pressure simulations.
  !
  !    MCMBND       Maximum number of bonds     for a single atom.
  !    MCMTHT       Maximum number of angles    for a single atom.
  !    MCMPHI       Maximum number of dihedrals for a single atom.
  !    MCMIMP       Maximum number of impropers for a single atom.
  !
  !    MCMINN       Minimization:  number of steps.
  !    MCMTYP       Minimization:  algorithm (0 = SD, 1 = CG)
  !    RMCSTP       Minimization:  initial step size.
  !    RMCMNF       Minimization:  tolerance for the energy change.
  !    RMCMNG       Minimization:  tolerance for the gradient.
  !    RMCMNS       Minimization:  tolerance for the step size.
  !    

  integer,PARAMETER :: MMVTYP = 50, MBONDT = 5
#if KEY_PATHINT==1
  integer,PARAMETER :: MCMBND = 10, MCMTHT = 100, MCMPHI = 200, MCMIMP = 100
#else /**/
  integer,PARAMETER :: MCMBND = 10, MCMTHT =  20, MCMPHI =  35, MCMIMP =  20
#endif 


  !     MCPOIN Common Block

  type(chm_ptr),dimension(mmvtyp),save :: MDXP
  type(iptr_ptr),dimension(mmvtyp),save :: IMVNGP, IBLSTP, IPIVTP
  INTEGER,dimension(mmvtyp) :: NLIMIT
  real(chm_real),dimension(mmvtyp) ::   WEIGHT, TFACT
  CHARACTER(len=4) :: MVLABL(MMVTYP)
  LOGICAL :: QBND(MBONDT,MMVTYP)

  !     MCLIMS Common Block

  INTEGER NMVTYP, NMVATM(mmvtyp), MVTYPE(mmvtyp)

  !     MCOPTS Common Block

  real(chm_real),dimension(mmvtyp) :: ARMP, ARMA, ARMB, ARMMAX, DOMCF
  LOGICAL,dimension(mmvtyp) :: ARMLIM, ANISO

  !     MCPARM Common Block

  INTEGER ISDMC
  real(chm_real)  RVOLMC


  !     MCMINI Common Block

  INTEGER,dimension(mmvtyp) :: MCMINN, MCMTYP
  real(chm_real),dimension(mmvtyp) :: RMCSTP, RMCMNF, RMCMNG, RMCMNS

  !     MCLINK Common Block

  type(iptr_ptr),dimension(mmvtyp),save :: ILNMVP, ILNBDP
  INTEGER,dimension(mmvtyp) :: NXTMVG, IACMVG
  integer :: NACMVG
  LOGICAL,dimension(mbondt,mmvtyp) :: QLNBND

  !     MCGCMC Common Block

#if KEY_GCMC==1
  INTEGER,dimension(mmvtyp) :: IGCMVG        
#endif

contains
  subroutine mc_init
    use mccent, only: lcentr
    use number,only:zero
    nmvtyp = 0
    nacmvg = 0
    isdmc  = 190872
    rvolmc = zero
    lcentr = 0
    return
  end subroutine mc_init

#endif /* (mc_fcm)*/
  !
end module mc

