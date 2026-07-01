module nblist_types

  !
  ! Type definitions for neighborlists
  !

  use chm_kinds
  use dimens_fcm
  use iso_c_binding
  implicit none
  private

  ! Tile sizes and number of exclusion masks
  integer, parameter :: tilesize = 32
  integer, parameter :: num_excl = (tilesize*tilesize-1)/32 + 1

  ! Neighbor list type:
  integer, parameter :: TYPE_NONE = 0, TYPE_PAIR = 1, TYPE_TILEX = 2

  ! uu = solute - solute
  ! uv = solute - solvent
  ! vv = solvent - solvent
  integer, parameter :: uu=1, uv=2, vv=3

  type intarray_t
     integer, allocatable, dimension(:) :: array
  end type intarray_t

  type intarrayp_t
     integer, pointer, dimension(:) :: array
  end type intarrayp_t

  type intarray2_t
     integer, allocatable, dimension(:,:) :: array
  end type intarray2_t

  type intarray2p_t
     integer, pointer, dimension(:,:) :: array
  end type intarray2p_t

  type cr4array_t
     real(chm_real4), allocatable, dimension(:) :: array
  end type cr4array_t

  ! *
  ! * Non-bonded pair interaction definition
  ! *
  type nb_pair_t
     logical lcoul, lvdw                      ! Flags for Coulomb and VdW
     integer type14                           ! 1-4 type: 0 = none, 1 = interaction, 2 = exclusion
     real(chm_real),  allocatable, dimension(:) :: pftable_dp  ! potential-force lookup table
     real(chm_real4), allocatable, dimension(:) :: pftable_sp  ! potential-force lookup table
     real(chm_real) h                         ! Spacing of lookup table     
     integer n                                ! Number of entries in lookup table
     real(chm_real) ctofnb, ctonnb            ! cut-off and switch-on distances
     integer vdw_type                         ! 1=lookup, 2=vsh, 3=vsw, 4=vfsw
  end type nb_pair_t

  ! *
  ! * Non-bonded list definition
  ! *
  type nblist_pair_t
     logical lcoul, lvdw                          ! Flags for Coulomb and VdW
     logical lblock                               ! Flag for block
     integer type14                            ! 1-4 type: 0 = none, 1 = interaction, 2 = exclusion
     integer itype                                ! Type: uu, uv, vv
     integer ni, nj                               ! Number of i and j atoms
     integer ni_max, nj_max                       ! Maximum number of i and j atoms
     integer np                                   ! Total number of pairs for this neighbor list
     integer, allocatable, dimension(:) :: indi   ! Index of i atoms
     integer, allocatable, dimension(:) :: indj   ! Index of j atoms
     integer, allocatable, dimension(:) :: startj ! Start of j atoms list
     integer, allocatable, dimension(:) :: iscoord         ! Shift coordinates index
     integer nb_pair_ind                          ! Index to nb_pair()
     character(40) name                           ! Name of this neighbor list
  end type nblist_pair_t

  type, bind(C) :: ientry_t
     integer(c_int) indi, ish, startj, endj
  end type ientry_t

  type, bind(C) :: tile_excl_t
     integer(c_int) excl(NUM_EXCL)
  end type tile_excl_t

  type nblist_tilex_t
     integer tilesize
     integer ni, nj
     integer ni_max, nj_max
     type(ientry_t), pointer, dimension(:) :: ientry
     !
     ! For q_skiptile = .false.
     ! tile(1,j)              = indj(1)
     ! tile(2:2+num_excl-1,j) = excl(1:num_excl)
     !
     ! For q_skiptile = .true.
     ! tile(1:tilesize,j)     = indj(1:tilesize)
     ! tile(tilesize+1:tilesize+num_excl,j) = excl(1:num_excl)
     !

     ! tile(1:num_excl,j) = exclusion mask

     integer, allocatable, dimension(:,:) :: tile
  end type nblist_tilex_t

  type nblist_t
     logical q_pack                ! .true. for packed coordinate arrays
     integer type                  ! List type: TYPE_PAIR, TYPE_TILEX
     integer n                     ! Number of lists
     type(nblist_pair_t), allocatable, dimension(:) :: pair
     type(nblist_tilex_t), allocatable, dimension(:) :: tilex1
     type(nblist_tilex_t), allocatable, dimension(:) :: tilex2
  end type nblist_t

  ! Definition of double/single precision arrays:
  ! dp = double precision
  ! sp = single precision
  type dpsp_t
     real(chm_real), allocatable, dimension(:) :: dp
     real(chm_real4), allocatable, dimension(:) :: sp
  end type dpsp_t

  type(nblist_t) nblist_cluster

  ! Non-bonded pair interactions
  type(nb_pair_t) nb_pair(4)

  ! 1-4 interactions & exclusions
  type(nb_pair_t) inex14_pair(2)
  
  ! Non-bonded neighbor lists
  type(nblist_t) nblist

  ! Public variables
  public nblist_pair_t, nblist_t, nblist, nb_pair, inex14_pair, &
       TYPE_NONE, TYPE_PAIR, TYPE_TILEX, &
       uu, uv, vv, &
       dpsp_t, nb_pair_t,&
       nblist_cluster, &
       nblist_tilex_t, ientry_t, tile_excl_t, &
       intarray_t, intarray2_t, intarray2p_t, cr4array_t, &
       tilesize, num_excl
  public intarrayp_t

end module nblist_types

