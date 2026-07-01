module image
  use chm_kinds
  use dimens_fcm
  !
  !     The Image data for crystal symmetry
  !
  !     See also: chm_types_ltm
  !
  !     Variable  Type      Dimension   Purpose
  !
  !     NTRANS    INTEGER*4             Number of transformations
  !     IMNAME    CHARACTER(NTRANS)     Name of transformation
  !     NIMRES    INTEGER  (NTRANS)     Group index pointer for each trans
  !     IMINV     INTEGER  (NTRANS)     Inverse transformation number
  !
  !     NATIM     INTEGER               Number of total atoms (primary
  !                                     and image)
  !     NIMGRP    INTEGER               Number of total groups
  !     NIMHB     INTEGER               Number of image H-bonds
  !     NIMBON    INTEGER               Number of image-primary bonds
  !     NIMANG    INTEGER               Number of image-primary angles
  !     NIMDIH    INTEGER               Number of image-primary dihedrals
  !     NIMAUTO  INTEGER               Number of image-primary autogen angles and dihedrals
  !     NIMIMP    INTEGER               Number of image-primary impropers
#if KEY_CMAP==1
  !     NIMCRT    INTEGER               Number of image-primary cross-terms
#endif 

  CHARACTER(len=8) IMNAME(MAXTRN)
  INTEGER   NTRANS, NIMRES(MAXTRN), IMINV(MAXTRN), &
       NATIM, NIMGRP, NIMHB, NIMBON, &
       NIMANG, NIMDIH, NIMAUTO, NIMIMP, IMGFRQ, IXTFRQ
#if KEY_CMAP==1
  INTEGER NIMCRT
#endif 

  !     IMFACT    real(chm_real)   (3*NTRANS)   Unit cell distance factor for
  !                                     transformation
  !     IMTRNS    real(chm_real)   (12*NTRANS)  Transformation matrix (3x3 rot
  !                                     and 3 trans)
  !     IMFORC    real(chm_real) (3*NTRANS)     Total vector force of image onto
  !                                     primary atoms.
  !     IMTORQ    real(chm_real) (3*NTRANS)     Total vector torque of image onto
  !                                     primary atoms (centered at origin).
  !     LIMINV    LOGICAL               Flag to request suppresion of
  !                                     image atoms of a higher inverse
  !                                     transformation (more efficient)
  !     LIMALL    LOGICAL               Flag to request all image atoms
  !                                     within cutoff to be selected.
  !     CUTIM     real(chm_real)                Image update cutoff distance
  !     CUTXTL    real(chm_real)                cutoff distance for building crystal
  !     NOROT     LOGICAL               Flag to determine if no rotations are used
  !     LIMCEN    LOGICAL               Centering flag for image update
  !     IMXCEN,IMYCEN,IMZCEN real(chm_real)     Position of center for centering
  !
  !     IMGFRQ    The image update frequency.
  !     IXTFRQ    The crystal update frequency.
  !
  !     LMKMA1    LOGICAL                Flag to request old image listbuilder

  real(chm_real)  CUTIM, CUTXTL, IMFACT(3*MAXTRN), &
       IMTRNS(12*MAXTRN), IMFORC(3*MAXTRN), &
       IMTORQ(3*MAXTRN), IMXCEN, IMYCEN, IMZCEN
  LOGICAL NOROT, LIMCEN, LIMINV, LIMALL
  ! flag for determining fast/slow image listbuilder 
  LOGICAL :: LMKMA1=.false. 
  ! flag for determining naming of images in crystal build 
  LOGICAL :: QXTLNAMETYPE=.true. 

! Arrays to be used to create the nonbond exlcusion lists on image updates.
!                   This is the image part of the PSF stored in a different format 
!                   to facilitate remapping at image updates.
!  integer   NIMEXCL            ! The number of image exclusions
!  integer   IMEXCLI(MAXIMEX)   ! primary atom of exclusion (not an image)
!  integer   IMEXCLJ(MAXIMEX)   ! image atom of exclusion -- must reorder on image update
!  integer   IMEXCLT(MAXIMEX)   ! exclusion type (2,3,4)  -- must reorder on image update
!  integer   MAXIMEX            ! maximum number of image exclusions

  integer NIMEXCL,MAXIMEX
  integer,allocatable,dimension(:) :: IMEXCLI
  integer,allocatable,dimension(:) :: IMEXCLJ
  integer,allocatable,dimension(:) :: IMEXCLT
! 

  !
  !     Crystal symmetry and transformation elements:
  !
  !     MAXSYM    INTEGER               The maximum number of symmetry
  !                                     operations allowed.
  !
  !     XDIM      INTEGER               The number of independent crystal
  !                                     variables.
  !
  !     XNA,XNB,XNC INTEGER MAXTRN      The lattice translation defining the
  !                                     image transformation.
  !
  !     XNOP      INTEGER MAXTRN        The symmetry operation defining
  !                                     the image transformation.
  !
  !     XNSYMM    INTEGER               The number of symmetry operations with
  !                                     the identity included. This number is
  !                                     0 if no crystal system is being studied.
  !
  !     XSYMOP    INTEGER 12*MAXSYM     The definition of the symmetry
  !                                     operations in terms of the fractional
  !                                     coordinate axes.
  !
  !     XTLTYP    CHARACTER(len=4)      The lattice type. Seven types other than
  !                                     '    ' are allowed: 'TRIC','MONO','RECT'
  !                                     'ORTH','RHOM','HEXA','TETR' and 'CUBI'
  !                                     'OCTA','RHDO'.
  !
  !     XTLREF    real(chm_real)    3           Original a,b,c values (for 'RECT').
  !
  !     XUCELL    real(chm_real)    6           The lattice parameters in the order
  !                                     a, b, c, alpha, beta, gamma. Units
  !                                     are Angstroms and degrees.
  !     XUCOLD    real(chm_real)    6           Previous XUCELL
  !
  !     XTLABC    real(chm_real)    6           Lower triangle of
  !                                     symmetric shape matrix.
  !
  !     DXTL      real(chm_real)    6           The corresponding crystal first
  !                                     derivatives.
  !
  !     HDOT      real(chm_real)    6           The time derivative of XTLABC
  !                                         (in units of step size)
  !     PNH       real(chm_real)    1
  !     PNHV      real(chm_real)    1           Thermal piston position,velocity,force
  !     PNHF      real(chm_real)    1           for use with constant pressure
  !     QORTH     logical                       Is the box orthogonal?
  !     XTLROT    real(chm_real)    3,3         Rotation matrix from standard
  !                                     symmetric charmm box orientation to
  !                                     accelerator orientation in XTLACC.
  !                                     For accelerator box vectors (OpenMM,
  !                                     domdec, BLaDE, etc...) first row is a
  !                                     pointed along x axis, second row is b in
  !                                     xy plane, third row is c
  !
  !     Image and crystal non-bond list information:
  !
  !     XATIM     INTEGER                The total number of atoms in the
  !                                      image system.
  !
  !     XNNNB     INTEGER                The total number of non-bonded
  !                                      interactions.
  !
  !     XINBLO    INTEGER                A heap pointer to the expanded
  !                                      IMBLO array.
  !
  !     XJNB      INTEGER                A heap pointer to the expanded
  !                                      IMJNB array.
  !
  !     XMATPT    INTEGER                A heap pointer to the expanded
  !                                      IMATPT array.
  !
  !     XMATTR    INTEGER                A heap pointer to the expanded
  !                                      IMATTR array.
  !
  !     Image and crystal second derivative information:
  !
  !     NFREQX    INTEGER                The number of crystal frequencies.
  !     XDDF      INTEGER                A pointer to the crystal second
  !                                      derivative matrix.
  !     XFREQ     INTEGER                A pointer to the crystal
  !                                      frequencies (wavenumbers).
  !     XEVAL     INTEGER                A pointer to the SD matrix eigenvalues.
  !     XEVEC     INTEGER                A pointer to the SD matrix eigenvectors.
  !
  !     NKPTS     INTEGER                The number of values of the
  !                                      k-vector for which the
  !                                      phonons are to be constructed.
  !     KSTART    real(chm_real)                 The starting value of the k-vector.
  !     KSTEP     real(chm_real)                 The step in the k-vector between points.
  !     NPHONS    INTEGER                The number of phonons.
  !     XDDFP     INTEGER                A pointer to the phonon second
  !                                      derivative matrix.
  !     XFREQP    INTEGER                A pointer to the phonon
  !                                      frequencies (wavenumbers).
  !     XEVALP    INTEGER                A pointer to the phonon SD
  !                                      matrix eigenvalues.
  !     XEVECP    INTEGER                A pointer to the phonon SD
  !                                      matrix eigenvectors.

  real(chm_real),allocatable,dimension(:) :: XFREQP
  real(chm_real),allocatable,dimension(:) :: XEVALP
  complex(chm_cmpx),allocatable,dimension(:) :: XDDFP
  real(chm_real),allocatable,dimension(:) :: XDDF
  real(chm_real),allocatable,dimension(:) :: XFREQ
  real(chm_real),allocatable,dimension(:) :: XEVEC
  real(chm_real),allocatable,dimension(:) :: XEVAL
  complex(chm_cmpx),allocatable,dimension(:) :: XEVECP
  integer         XDIM, XNSYMM, XNA(MAXTRN), XNB(MAXTRN), &
       XNC(MAXTRN), XNOP(MAXTRN), XSYMOP(3,4,MAXSYM), &
       NFREQX, NKPTS, &
       NPHONS
  real(chm_real)  XTLABC(6), XUCELL(6), XUCOLD(6), DXTL(6), &
       KSTART(3), KSTEP(3), XTLREF(6), HDOT(6), &
       PNH, PNHV, PNHF
  logical QORTH
  real(chm_real)  XTLROT(3,3)

  CHARACTER(len=4) XTLTYP

  INTEGER XATIM, XNNNB
  integer,allocatable,dimension(:) :: XINBLO
  integer,allocatable,dimension(:) :: XJNB
  integer,allocatable,dimension(:) :: XMATPT
  integer,allocatable,dimension(:) :: XMATTR

  integer,save        :: NumLattice,ILATT
  real(chm_real),save :: lattice_vector(3,3)

contains
  subroutine image_init
    use number,only:fmark
    imgfrq=0
    ixtfrq=0
    NumLattice=0
    cutim = 13.5
    cutxtl=fmark
    return
  end subroutine image_init
end module image

