module chm_types
  use chm_kinds
  implicit none

  type chm_array
     real(chm_real),allocatable,dimension(:) :: a
     integer :: len = 0
  end type chm_array

  type chm_array_2d
     real(chm_real),allocatable,dimension(:,:) :: a
     integer :: len1 = 0, len2 = 0
  end type chm_array_2d

  type chm_iarray
     integer,allocatable,dimension(:) :: a
     integer :: len = 0
  end type chm_iarray

  type chm_ptr
     real(chm_real),pointer,dimension(:) :: a => null()
     integer :: len = 0
  end type chm_ptr

  type chm_ptr4
     real(chm_real4),pointer,dimension(:) :: a => null()
     integer :: len = 0
  end type chm_ptr4

  type chm_ptr_2d
     real(chm_real),pointer,dimension(:,:) :: a => null()
     integer :: len1 = 0, len2 = 0
  end type chm_ptr_2d

  type chm_xyz
     real(chm_real),allocatable,dimension(:) :: x,y,z
     integer :: len = 0
  end type chm_xyz

  type chm_iptr
     integer,pointer,dimension(:) :: a => null()
     integer :: len = 0
  end type chm_iptr

  type chm_lptr
     logical,pointer,dimension(:) :: a => null()
     ! len can be computed if needed
  end type chm_lptr

  type iptr_ptr
     type(chm_iptr),pointer,dimension(:) :: a => null()
  end type iptr_ptr

        type arofar
         real(chm_real),allocatable,dimension(:) :: B
      end type arofar
      type arofar_i
         integer,allocatable,dimension(:) :: B
      end type arofar_i
      type arofar_i4
         integer(chm_int4),allocatable,dimension(:) :: B
      end type arofar_i4
      type arofar_i8
         integer(chm_int8),allocatable,dimension(:) :: B
      end type arofar_i8

  type internalCoordinate
     !     LENIC   Number of internal coordinates in the
     !             current instance of internalCoordinate
     !
     !     INTLEN  Number of availible places in the
     !             current instance of internalCoordinate
     !             for extra internals coordinates.
     !             This is essentially a buffer to prevent
     !             repeated calls to allocate() and is
     !             generally maniplulated via reintc_new()
     ! LENIC,INTLEN are historical names that are being used
     ! here.
     integer :: lenic,intlen
     !
     !                                  L
     !                                 /
     !                                / b2(A)
     !                        (deg)t2/
     !                 J------------K
     !                / t1(deg)
     !         b1(A) /
     !              /
     !             I
     !                   p(degrees)
     !
     ! All following arrays are of size INTLEN,
     ! but may only be populated to LENIC
     !
     !     IAR     First atom of internal coordinate
     !     JAR     Second atom of internal coordinate
     !     KAR     Third atom of internal coordinate
     !     LAR     Fourth atom of internal coordinate
     !     B1IC    Bond length between first two atoms
     !     B2IC    Bond length between last two atoms
     !     T1IC    Bond angle between first three atoms
     !     T2IC    Bond angle between last three atoms
     !     PIC     Torsion angle made by the four atoms (degree)
     !     TAR     Flag indicating that this is an improper torsion

     integer, dimension(:), allocatable  :: &
          iar,jar,kar,lar

     real(chm_real), dimension(:), allocatable  ::  &
          b1ic,b2ic,t1ic,t2ic,pic

     logical, dimension(:), allocatable  :: &
          tar
  end type internalCoordinate

!  NONBONDED DATA STRUCTURE:
!
!     Stores the information needed to calculate the Van der Waals and
!     electrostatic interactions in a structure in an efficient manner
!     as possible.
!
!     See also: inbnd_ltm, datstr, nbutil
!
!  Variable  Type  Dimension  Purpose
!
!  INBLO     INT   (NATOM)  - Number of entries below of JNB entries
!                  (NATOM*2)   for IMCUBES enabled
!                              INBLO(natim+i)+1 is the index of the
!                              first nb pair for atom i, much like
!                              INBLO(i-1) for other list builders.
!  INBLOG    INT   (NGRP)   - Number of entries below of JNBG entries
!
!  LNBOPT, NBDIST, NBINTS   - nonbond calculation settings,
!                             see ltm/inbnd_ltm.src for details
!
!  INB14     INT   (NNB14)  - The second atom of nonbond exclusion pair
!  IBLO14    INT   (NATOM)  - Number of entries below of INB14 entries
!  ING14     INT   (NNG14)  - The second group of nonbond exclusion pair
!  IGLO14    INT   (NGRP)   - Number of entries below of ING14 entries
!  LASTX,LASTY,LASTZ (NATOM)- Coordinates of the last nonbond update
!  JNBG      INT   (NNNBG)  - The second group of nonbond group pair
!  JNB       INT   (NNNB)   - The second atom of nonbond atom pair

  integer,parameter :: NNBOPT=40, NNBDIS=40, NNBINT=10

  type nonbondDataStructure

     integer, dimension(:), pointer        :: INBLO => null()  
     integer, dimension(:), pointer        :: INBLOG => null() 
     logical, dimension(NNBOPT)            :: LNBOPT   
     real(chm_real), dimension(NNBDIS)     :: NBDIST   
     integer, dimension(NNBINT)            :: NBINTS   
     integer, dimension(:), pointer        :: INB14 => null()  
     integer, dimension(:), pointer        :: IBLO14 => null() 
     integer, dimension(:), pointer        :: ING14 => null()  
     integer, dimension(:), pointer        :: IGLO14 => null() 
     real(chm_real), dimension(:), pointer :: LASTX => null()  
     real(chm_real), dimension(:), pointer :: LASTY => null()  
     real(chm_real), dimension(:), pointer :: LASTZ => null()  
     integer, dimension(:), pointer        :: JNBG => null()   
     integer, dimension(:), pointer        :: JNB => null()    
     logical                               :: IS_ALIAS = .false.

  end type nonbondDataStructure

!  IMAGE DATA STRUCTURE:
!
!     Defines relative positions and orientations of any images
!     relative to the primary molecule(s). Also included is nonbonded,
!     H-bonds arrays defining interactions between the primary
!     and image atoms.
!
!     See also: image_module
!
!     IMATPT    INTEGER   (NTRANS)    Pointer to last atom of
!                                     transformation
!     IMATTR    INTEGER   (NATIM)     Corresponding primary atom for each
!                                     image atom.
!     NIMINB    INTEGER               Number of image-primary exclusions
!     IMINB     INTEGER(NIMINB)       Image "INB" array
!     IMIBLO    INTEGER(NATIM)        Image "IBLO" array
!     NIMING    INTEGER               Number of image group exclusions
!     IMING     INTEGER(NGRPT)        Group exclusion list pointer array
!     IMIGLO    INTEGER(NIMING)       Image group exclusion list
!     NIMNB     INTEGER               Number of image nonbonded
!                                     interactions
!     IMJNB     INTEGER   (NIMNB)     "JNB" array for images
!     IMBLO     INTEGER   (NATIM)     "INBLO" array for images
!                         (NATIM * 2 ) for IMCUBES enabled
!                                      IMBLO(natim+i)+1 is the index of the
!                                     first nb pair for atom i, much like
!                                     IMBLO(i-1) for other list builders.
!     NIMNBS    INTEGER               Number of image nonbonded
!                                     self interactions
!     IMJNBS    INTEGER   (NIMNB)     "JNB" array for image self terms
!     IMBLOS    INTEGER   (NATIM)     "INBLO" array for image self terms
!     NIMNBG    INTEGER               Number of image nonbonded
!                                     group interactions
!     IMJNBG    INTEGER   (NIMNB)     "JNBG" array for images
!     IMBLOG    INTEGER   (NATIM)     "INBLOG" array for images
!     NIMNBX    INTEGER               Number of image nonbonded
!                                     self term group interactions
!     IMJNBX    INTEGER   (NIMNB)     "JNBG" array for image self terms
!     IMBLOX    INTEGER   (NATIM)     "INBLOG" array for image self terms
!     IMCENF    INTEGER   (NATIM)     Atom flag for centering disposition
!                                     0-No centering, 1-by segment
!                                     2-by residue,   3-by group
!                                     4-by atom.
  type imageDataStructure

     integer, dimension(:), allocatable  :: IMATPT    
     integer, dimension(:), allocatable  :: IMATTR    
     integer                             :: NIMINB
     integer, dimension(:), allocatable  :: IMINB     
     integer, dimension(:), allocatable  :: IMIBLO    
     integer                             :: NIMING
     integer, dimension(:), allocatable  :: IMING     
     integer, dimension(:), allocatable  :: IMIGLO    
     integer                             :: NIMNB
     integer, dimension(:), allocatable  :: IMJNB     
     integer, dimension(:), allocatable  :: IMBLO     
     integer                             :: NIMNBS
     integer, dimension(:), allocatable  :: IMJNBS    
     integer, dimension(:), allocatable  :: IMBLOS    
     integer                             :: NIMNBG
     integer, dimension(:), allocatable  :: IMJNBG    
     integer, dimension(:), allocatable  :: IMBLOG    
     integer                             :: NIMNBX
     integer, dimension(:), allocatable  :: IMJNBX    
     integer, dimension(:), allocatable  :: IMBLOX    
     integer, dimension(:), allocatable  :: IMCENF    

  end type imageDataStructure

end module chm_types

