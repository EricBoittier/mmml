module sasa
  use chm_kinds
  use dimens_fcm

  !----------------------------------------------------------------------
  !     What follows applies only to the code concerning SASA.
  !        Any variable name that begins with 'P' is a pointer to the
  !     heap. The string following 'P' is the name used in any subroutine
  !     throughout the code to access the corresponding space on
  !     the heap.
  !        Any variable name that ends with 'BL' or 'NB' belongs to a
  !     pair list that is indexed in the same way as is, for instance,
  !     the fixed exclusion pair list (IBLO and INB), the nonbond
  !     exclusion pair list (IBLO14 and INB14), or the nonbond pair list
  !     itself (INBLO and JNB). The indexing is explained in more detail
  !     below.
  !        The following note applies only if the nonbond exclusion mode
  !     is equal to 5. In this case the number of the second atom of each
  !     1-4 pair is entered negatively in the nonbond exclusion pair list
  !     if this atom pair is not in the fixed exclusion pair list. These
  !     1-4 pairs are the so called special 1-4 pairs. The number of the
  !     second atom of each 1-4 pair is entered positively in the nonbond
  !     exclusion pair list if this atom pair is also in the fixed
  !     exclusion pair list. Pairs with a negative entry of the number of
  !     their second atom in the nonbond exclusion pair list are actually
  !     added to the nonbond pair list contrary to pairs with a positive
  !     entry of the number of their second atom in the nonbond exclusion
  !     pair list. The bottom line is that 1-4 pairs that are in the fixed
  !     exclusion pair list are not in the nonbond pair list whereas 1-4
  !     pairs that are not in the fixed exclusion pair list are in the
  !     nonbond pair list although in both cases the 1-4 pairs are in the
  !     nonbond exclusion pair list. Note that the fixed exclusion pair
  !     list has only positive entries.
  !
  !     Author: Urs Haberthuer.
  !
  !     SASNPR  - Number of CHARMM atom types supported by SASA.
  !
  !     SASRSP  - Radius of the solvent probe.
  !     SASFCO  - Connectivity parameter for 1-2 pairs.
  !     SASNCO  - Connectivity parameter for 1-n pairs with n greater
  !               than 2.
  !     SASSIG  - One dimensional array for the surface-tension like
  !               solvation parameters.
  !     SASROV  - One dimensional array for the radii.
  !     SASPAT  - One dimensional array for the probabilistic parameters.
  !     SASRSS  - One dimensional array for the solvation shell radii.
  !     SASASS  - One dimensional array for the solvation shell areas.
  !     SASSUM  - Two dimensional array for the sums of the elements of
  !               the SASRSS array (look-up table).
  !     SASDIF  - Two dimensional array for the differences of the
  !               elements of the SASROV array (look-up table).
  !     SASPRD  - Two dimensional array for the products of the elements
  !               of the SASSUM and SASDIF arrays (look-up table).
  !
  !     SASNAT  - Number of atoms.
  !     SASSNW  - Number of elements in the SASWNB array after cleaning.
  !     SASSNI  - Number of elements in the SASINB array after cleaning.
  !     PSASWBL - Pointer to the one dimensional array SASWBL on the heap.
  !     PSASWNB - Pointer to the one dimensional array SASWNB on the heap.
  !                  The arrays SASWBL and SASWNB make up the
  !               SASWBL/SASWNB pair list. They are defined as follows.
  !               The array SASWNB stores the number of the second atom
  !               of each pair in the SASWBL/SASWNB pair list. All pairs
  !               in the SASWBL/SASWNB pair list for which atom I is the
  !               first atom are given by (I,SASWNB(SASWBL(I-1)+1) up to
  !               (I,SASWNB(SASWBL(I)) where SASWBL(0) is assumed to be
  !               0.
  !                  The SASWBL/SASWNB pair list is the list of all
  !               1-2 pairs.
  !     PSASIBL - Analogous to PSASWBL.
  !     PSASINB - Analogous to PSASWNB.
  !                  The SASIBL/SASINB pair list stores all pairs that
  !               SASA needs in addition to the pairs in the nonbond pair
  !               list. The SASIBL/SASINB pair list is henceforth called
  !               the SASA pair list.
  !     PSASLCT - Pointer to the one dimensional array SASLCT on the heap.
  !                  The array SASLCT describes for each atom whether or
  !               not it has been selected for SASA by the user.
  !     PSASIDX - Pointer to the one dimensional array SASIDX on the heap.
  !                  The array SASIDX stores the index for the SASA
  !               parameters of each atom.
  !     PSASACS - Pointer to the one dimensional array SASACS on the heap.
  !                  The array SASACS stores the solvent accessible
  !               surface area of each atom.
  !     QSASA   - Flag that determines whether or not SASA is used. If it
  !               is set to TRUE, SASA is used. If it is set to FALSE,
  !               SASA is not used. The default is FALSE.
  !     QINFX   - Flag that determines whether or not the fixed exclusion
  !               pairs should be included to or excluded from the SASA
  !               pair list. If it set to TRUE, the fixed exclusion pairs
  !               are included. If it set to FALSE, they are excluded.
  !               The default is FALSE. The default corresponds to the
  !               original Ferrara setup.
  !     QSRFC   - Flag that determines whether or not the solvent
  !               accessible surface areas of all atoms (approximated by
  !               the Hasel formula) are stored in the array WMAIN. If it
  !               is set to TRUE, the solvent accessible surface areas of
  !               all atoms are stored in WMAIN. If it set to FALSE, they
  !               are not stored. The default is FALSE.
  !     QNEWP   - Flag that determines whether the original (Hasel and
  !               Still) or the new (Haberthuer 2002) radii,
  !               probabilistic parameters, and connectivity parameters
  !               to calculate the solvent accessible surface areas
  !               should be used. If it is set to TRUE, the new parameter
  !               set is used. If it set to FALSE, the old one is used.
  !               The default is FALSE.

  INTEGER       SASNPR
  PARAMETER    (SASNPR=31)

  integer,allocatable,dimension(:) :: PSASLCT
  integer,allocatable,dimension(:) :: PSASIDX
  real(chm_real),allocatable,dimension(:) :: PSASACS
  integer,allocatable,dimension(:) :: PSASWBL
  integer,allocatable,dimension(:) :: PSASWNB
  integer,allocatable,dimension(:) :: PSASIBL
  integer,allocatable,dimension(:) :: PSASINB

  real(chm_real)        SASRSP
  real(chm_real)        SASFCO,SASNCO
  real(chm_real)        SASSIG,SASROV,SASPAT
  real(chm_real)        SASRSS,SASASS
  real(chm_real)        SASSUM
  real(chm_real)        SASDIF
  real(chm_real)        SASPRD

  INTEGER       SASNAT
  INTEGER       SASSNW,SASSNI

  LOGICAL       QSASA
  LOGICAL       QINFX
  LOGICAL       QSRFC
  LOGICAL       QNEWP

  COMMON /SASAc/ SASRSP, &
       SASFCO,SASNCO, &
       SASSIG(SASNPR),SASROV(SASNPR),SASPAT(SASNPR), &
       SASRSS(SASNPR),SASASS(SASNPR), &
       SASSUM(SASNPR,SASNPR), &
       SASDIF(SASNPR,SASNPR), &
       SASPRD(SASNPR,SASNPR), &
       
       SASNAT, &
       SASSNW,SASSNI, &
       
       QSASA, &
       QINFX, &
       QSRFC, &
       QNEWP
  !
contains
  subroutine sasa_init
    qsasa=.false.
    return
  end subroutine sasa_init
end module sasa

