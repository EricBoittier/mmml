module domdec_bonded_block

#if KEY_BLOCK==1 && KEY_DOMDEC==1
  use chm_kinds
  use nblist_types,only:intarray_t
  !-------------------------------------------------------------------
  ! Following are for the subroutines in domdec_bonded_block_kernel.inc
  !-------------------------------------------------------------------
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  use block_ltm,only:nblock
  use domdec_bonded_types,only:bondlist_t, anglelist_t, dihelist_t, list14_t
  use domdec_bonded,only:build_bondlist_pd, build_bondlist_ps, &
       build_anglelist_pd, build_anglelist_ps, &
       build_dihelist_pd, build_dihelist_ps, &
       build_xx14list_pd, build_xx14list_ps
  !-------------------------------------------------------------------
  implicit none
  private

  ! Bonds
  integer, allocatable, dimension(:) :: nbondtbl_block, bondtbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: bondtbl_block

  ! Urey-Bradley
  integer, allocatable, dimension(:) :: nureybtbl_block, ureybtbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: ureybtbl_block  

  ! Angles
  integer, allocatable, dimension(:) :: nangletbl_block, angletbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: angletbl_block

  ! Dihedrals
  integer, allocatable, dimension(:) :: ndihetbl_block, dihetbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: dihetbl_block

  ! Improper dihedrals
  integer, allocatable, dimension(:) :: nimdihetbl_block, imdihetbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: imdihetbl_block

  ! 1-4 interactions & exclusions
  integer, allocatable, dimension(:) :: nin14tbl_block, in14tbl_block_pos
  integer, allocatable, dimension(:) :: nex14tbl_block, ex14tbl_block_pos
  type(intarray_t), allocatable, dimension(:) :: xx14tbl_block

  interface calc_ntbl
     module procedure calc_ntbl_ilist_jlist
     module procedure calc_ntbl_list
  end interface

  interface block_split_tbl
     module procedure block_split_tbl_ilist_jlist
     module procedure block_split_tbl_list
  end interface

#endif

  ! Public variables
  public nbondtbl_block, bondtbl_block_pos, bondtbl_block
  public nureybtbl_block, ureybtbl_block_pos, ureybtbl_block
  public nangletbl_block, angletbl_block_pos, angletbl_block
  public ndihetbl_block, dihetbl_block_pos, dihetbl_block
  public nimdihetbl_block, imdihetbl_block_pos, imdihetbl_block
  public nin14tbl_block, in14tbl_block_pos
  public nex14tbl_block, ex14tbl_block_pos

  ! Public functions
  public lower_triangle_index

  ! Public subroutines
  public init_bonded_block, uninit_bonded_block
  public build_bondlist_block_ps, build_bondlist_block_pd
  public build_xx14list_block_ps, build_xx14list_block_pd
  public build_anglelist_block_ps, build_anglelist_block_pd
  public build_dihelist_block_ps, build_dihelist_block_pd

#if KEY_BLOCK==1 && KEY_DOMDEC==1

contains

  ! *
  ! * Allocates memory for bonded block tables
  ! *
  subroutine init_bonded_block()
    use memory
    use block_ltm,only:nblock
    use lambdam,only:qsobo
    implicit none
    ! Variables
    integer nint
    integer, allocatable, dimension(:) :: nxx14tbl_block

    ! Number of interactions
    nint = nblock*(nblock+1)/2

    if (.not.qsobo) then

       ! nbondtbl_block
       call alloc_realloc_ntbl_block(nint, nbondtbl_block)

       ! bondtbl_block_pos
       call alloc_realloc_ntbl_block(nint+1, bondtbl_block_pos)

       ! nureybtbl_block
       call alloc_realloc_ntbl_block(nint, nureybtbl_block)

       ! ureybtbl_block_pos
       call alloc_realloc_ntbl_block(nint+1, ureybtbl_block_pos)

       ! nangletbl_block
       call alloc_realloc_ntbl_block(nint, nangletbl_block)

       ! angletbl_block_pos
       call alloc_realloc_ntbl_block(nint+1, angletbl_block_pos)

       ! ndihetbl_block
       call alloc_realloc_ntbl_block(nint, ndihetbl_block)

       ! dihetbl_block_pos
       call alloc_realloc_ntbl_block(nint+1, dihetbl_block_pos)

       ! nimdihetbl_block
       call alloc_realloc_ntbl_block(nint, nimdihetbl_block)

       ! imdihetbl_block_pos
       call alloc_realloc_ntbl_block(nint+1, imdihetbl_block_pos)

    else
       call alloc_realloc_ntbl_block(2*nint, nbondtbl_block)
       call alloc_realloc_ntbl_block(2*nint+1, bondtbl_block_pos)
       call alloc_realloc_ntbl_block(2*nint, nureybtbl_block)
       call alloc_realloc_ntbl_block(2*nint+1, ureybtbl_block_pos)
       call alloc_realloc_ntbl_block(2*nint, nangletbl_block)
       call alloc_realloc_ntbl_block(2*nint+1, angletbl_block_pos)
       call alloc_realloc_ntbl_block(2*nint, ndihetbl_block)
       call alloc_realloc_ntbl_block(2*nint+1, dihetbl_block_pos)
       call alloc_realloc_ntbl_block(2*nint, nimdihetbl_block)
       call alloc_realloc_ntbl_block(2*nint+1, imdihetbl_block_pos)
    endif

    ! nin14tbl_block
    call alloc_realloc_ntbl_block(nint, nin14tbl_block)

    ! in14tbl_block_pos
    call alloc_realloc_ntbl_block(nint+1, in14tbl_block_pos)

    ! nex14tbl_block
    call alloc_realloc_ntbl_block(nint, nex14tbl_block)

    ! ex14tbl_block_pos
    call alloc_realloc_ntbl_block(nint+1, ex14tbl_block_pos)

    ! Calculate (nbondtbl_block, nureybtbl_block, nangletbl_block, ndihetbl_block
    !            nimdihetbl_block, nxx14tbl_block)
    call alloc_realloc_ntbl_block(nint, nxx14tbl_block)
    call calc_block_sizes(nblock, nbondtbl_block, nureybtbl_block, nangletbl_block, &
         ndihetbl_block, nimdihetbl_block, nxx14tbl_block)

    if (.not.qsobo) then

       ! bondtbl_block
       call alloc_realloc_tbl_block(nint, nbondtbl_block, bondtbl_block)

       ! ureybtbl_block
       call alloc_realloc_tbl_block(nint, nureybtbl_block, ureybtbl_block)

       ! angletbl_block
       call alloc_realloc_tbl_block(nint, nangletbl_block, angletbl_block)

       ! dihetbl_block
       call alloc_realloc_tbl_block(nint, ndihetbl_block, dihetbl_block)

       ! imdihetbl_block
       call alloc_realloc_tbl_block(nint, nimdihetbl_block, imdihetbl_block)

    else
       call alloc_realloc_tbl_block(2*nint, nbondtbl_block, bondtbl_block)
       call alloc_realloc_tbl_block(2*nint, nureybtbl_block, ureybtbl_block)
       call alloc_realloc_tbl_block(2*nint, nangletbl_block, angletbl_block)
       call alloc_realloc_tbl_block(2*nint, ndihetbl_block, dihetbl_block)
       call alloc_realloc_tbl_block(2*nint, nimdihetbl_block, imdihetbl_block)
    endif

    ! xx14tbl_block
    call alloc_realloc_tbl_block(nint, nxx14tbl_block, xx14tbl_block)

    call dealloc_ntbl_block(nxx14tbl_block)

    return
  end subroutine init_bonded_block

  ! *
  ! * Allocates & Reallocates (nbondtbl_block)
  ! *
  subroutine alloc_realloc_ntbl_block(nint, ntbl_block)
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: nint
    integer, allocatable, dimension(:), intent(inout) :: ntbl_block

    if (allocated(ntbl_block)) then
       if (size(ntbl_block) < nint) then
          call chmdealloc('domdec_bonded_block.src','alloc_realloc_ntbl_block','ntbl_block',&
               size(ntbl_block),intg=ntbl_block)
       endif
    endif

    if (.not.allocated(ntbl_block)) then
       call chmalloc('domdec_bonded_block.src','alloc_realloc_ntbl_block','ntbl_block',&
            nint,intg=ntbl_block)
    endif

    return
  end subroutine alloc_realloc_ntbl_block

  ! *
  ! * Deallocates (nbondtbl_block)
  ! *
  subroutine dealloc_ntbl_block(ntbl_block)
    use memory,only:chmdealloc
    implicit none
    ! Input / Output
    integer, allocatable, dimension(:), intent(inout) :: ntbl_block

    if (allocated(ntbl_block)) then
       call chmdealloc('domdec_bonded_block.src','dealloc_ntbl_block','ntbl_block',&
            size(ntbl_block),intg=ntbl_block)
    endif

    return
  end subroutine dealloc_ntbl_block

  ! *
  ! * Allocates & Reallocates (bondtbl_block)
  ! *
  subroutine alloc_realloc_tbl_block(nint, ntbl_block, tbl_block)
    use memory
    implicit none
    ! Input / Output
    integer, intent(in) :: nint, ntbl_block(:)
    type(intarray_t), allocatable, dimension(:), intent(inout) :: tbl_block
    ! Variables
    integer i

    if (allocated(tbl_block)) then
       if (size(tbl_block) < nint) then
          call dealloc_tbl_block(tbl_block)
       endif
    endif

    if (.not.allocated(tbl_block)) then
       allocate(tbl_block(nint))
    endif

    do i=1,nint
       if (allocated(tbl_block(i)%array)) then
          if (size(tbl_block(i)%array) < ntbl_block(i)) then
             call chmdealloc('domdec_bonded_block.src','alloc_realloc_tbl_block',&
                  'tbl_block(i)%array',&
                  size(tbl_block(i)%array),intg=tbl_block(i)%array)
          endif
       endif
       if (.not.allocated(tbl_block(i)%array)) then
          call chmalloc('domdec_bonded_block.src','alloc_realloc_tbl_block',&
               'tbl_block(i)%array',ntbl_block(i),intg=tbl_block(i)%array)
       endif
    enddo

    return
  end subroutine alloc_realloc_tbl_block

  ! *
  ! * Deallocates (bondtbl_block)
  ! *
  subroutine dealloc_tbl_block(tbl_block)
    use memory,only:chmdealloc
    implicit none
    ! Input / Output
    type(intarray_t), allocatable, dimension(:), intent(inout) :: tbl_block
    ! Variables
    integer i

    if (allocated(tbl_block)) then
       do i=1,size(tbl_block)
          if (allocated(tbl_block(i)%array)) then
             call chmdealloc('domdec_bonded_block.src','dealloc_tbl_block',&
                  'tbl_block(i)%array',size(tbl_block(i)%array),intg=tbl_block(i)%array)
          endif
       enddo
       deallocate(tbl_block)
    endif

    return
  end subroutine dealloc_tbl_block

  ! *
  ! * Deallocates memory for bonded block tables
  ! *
  subroutine uninit_bonded_block()
    implicit none

    ! nbondtbl_block
    call dealloc_ntbl_block(nbondtbl_block)

    ! bondtbl_block_pos
    call dealloc_ntbl_block(bondtbl_block_pos)

    ! bondtbl_block
    call dealloc_tbl_block(bondtbl_block)

    ! nangletbl_block
    call dealloc_ntbl_block(nangletbl_block)

    ! angletbl_block_pos
    call dealloc_ntbl_block(angletbl_block_pos)

    ! angletbl_block
    call dealloc_tbl_block(angletbl_block)

    ! ndihetbl_block
    call dealloc_ntbl_block(ndihetbl_block)

    ! dihetbl_block_pos
    call dealloc_ntbl_block(dihetbl_block_pos)

    ! dihetbl_block
    call dealloc_tbl_block(dihetbl_block)

    ! nimdihetbl_block
    call dealloc_ntbl_block(nimdihetbl_block)

    ! imdihetbl_block_pos
    call dealloc_ntbl_block(imdihetbl_block_pos)

    ! imdihetbl_block
    call dealloc_tbl_block(imdihetbl_block)

    ! nureybtbl_block
    call dealloc_ntbl_block(nureybtbl_block)

    ! ureybtbl_block_pos
    call dealloc_ntbl_block(ureybtbl_block_pos)

    ! ureybtbl_block
    call dealloc_tbl_block(ureybtbl_block)

    ! nin14tbl_block
    call dealloc_ntbl_block(nin14tbl_block)

    ! in14tbl_block_pos
    call dealloc_ntbl_block(in14tbl_block_pos)

    ! nex14tbl_block
    call dealloc_ntbl_block(nex14tbl_block)

    ! ex14tbl_block_pos
    call dealloc_ntbl_block(ex14tbl_block_pos)

    return
  end subroutine uninit_bonded_block

  ! *
  ! * Calculates how many bonds belong to each block to 
  ! * nbondtbl_block(:)
  ! * This helps us to allocate correct size arrays in the initialization
  ! *
  subroutine calc_block_sizes(nblock_in, nbondtbl_block, nureybtbl_block, nangletbl_block, &
       ndihetbl_block, nimdihetbl_block, nxx14tbl_block)
    use block_ltm,only:iblckp
    use lambdam,only:mldbondi, mldbondj, mldureyi, mldureyj, mldangi, mldangj, &
         mlddihi, mlddihj, mldimpi, mldimpj, &
         qsobo, mldbonds, mldureys, mldangs, mlddihs, mldimps
    use domdec_bonded,only:nin14, in14i, in14j, nex14, ex14i, ex14j
    use psf,only:nbond, ntheta, nphi, nimphi, ncrterm
    implicit none
    ! Input / Output
    integer, intent(in) :: nblock_in
    integer, intent(inout) :: nbondtbl_block(:), nureybtbl_block(:), nangletbl_block(:)
    integer, intent(inout) :: ndihetbl_block(:), nimdihetbl_block(:), nxx14tbl_block(:)
    ! Variables
    integer nint, itbl
    integer, allocatable, dimension(:) :: ntmp

    ! Number of interactions
    nint = nblock_in*(nblock_in+1)/2

    if (.not.qsobo) then

      ! Bonds
      call calc_ntbl(nint, nbond, mldbondi, mldbondj, nbondtbl_block)

      ! Urey-Bradley
      call calc_ntbl(nint, ntheta, mldureyi, mldureyj, nureybtbl_block)

      ! Angles
      call calc_ntbl(nint, ntheta, mldangi, mldangj, nangletbl_block)

      ! Dihedrals
      call calc_ntbl(nint, nphi, mlddihi, mlddihj, ndihetbl_block)

      ! Improper dihedrals
      call calc_ntbl(nint, nimphi, mldimpi, mldimpj, nimdihetbl_block)

    else
      call calc_ntbl_soft(nint, nbond, mldbondi, mldbondj, mldbonds, nbondtbl_block)
      call calc_ntbl_soft(nint, ntheta, mldureyi, mldureyj, mldureys, nureybtbl_block)
      call calc_ntbl_soft(nint, ntheta, mldangi, mldangj, mldangs, nangletbl_block)
      call calc_ntbl_soft(nint, nphi, mlddihi, mlddihj, mlddihs, ndihetbl_block)
      call calc_ntbl_soft(nint, nimphi, mldimpi, mldimpj, mldimps, nimdihetbl_block)
    endif

    ! 1-4 interactions & exclusions
    call alloc_realloc_ntbl_block(nint, ntmp)

    call calc_ntbl(nint, nin14, in14i, in14j, iblckp, nxx14tbl_block)
    call calc_ntbl(nint, nex14, ex14i, ex14j, iblckp, ntmp)
    do itbl=1,nint
       nxx14tbl_block(itbl) = max(nxx14tbl_block(itbl), ntmp(itbl))
    enddo

    call dealloc_ntbl_block(ntmp)

    ! Fix ntbl to make sure all entries are at least 1
    call fix_ntbl(nint, nbondtbl_block)
    call fix_ntbl(nint, nureybtbl_block)
    call fix_ntbl(nint, nangletbl_block)
    call fix_ntbl(nint, ndihetbl_block)
    call fix_ntbl(nint, nimdihetbl_block)
    call fix_ntbl(nint, nxx14tbl_block)

    return
  end subroutine calc_block_sizes

  ! *
  ! * Calculates nXXXtbl_block(1:nint)
  ! *
  subroutine calc_ntbl_ilist_jlist(nint, nlist, ilist, jlist, iblock, ntbl)
    implicit none
    ! Input / Output
    integer, intent(in) :: nint, nlist, ilist(:), jlist(:), iblock(:)
    integer, intent(inout) :: ntbl(:)
    ! Variables
    integer i, ibl, jbl, itbl

    ntbl(1:nint) = 0
    do i=1,nlist
       ibl = iblock(ilist(i))
       jbl = iblock(jlist(i))
       itbl = lower_triangle_index(ibl, jbl)
       ntbl(itbl) = ntbl(itbl) + 1
    enddo

    return
  end subroutine calc_ntbl_ilist_jlist
  
  ! *
  ! * Calculates nXXXtbl_block(1:nint)
  ! *
  subroutine calc_ntbl_list(nint, nlist, iblock, jblock, ntbl)
    implicit none
    ! Input / Output
    integer, intent(in) :: nint, nlist, iblock(:), jblock(:)
    integer, intent(inout) :: ntbl(:)
    ! Variables
    integer i, ibl, jbl, itbl

    ntbl(1:nint) = 0
    do i=1,nlist
       ibl = iblock(i)
       jbl = jblock(i)
       itbl = lower_triangle_index(ibl, jbl)
       ntbl(itbl) = ntbl(itbl) + 1
    enddo

    return
  end subroutine calc_ntbl_list

  ! *
  ! * Calculates nXXXtbl_block(1:nint) for soft interactions
  ! *
  subroutine calc_ntbl_soft(nint, nlist, iblock, jblock, sblock, ntbl)
    implicit none
    ! Input / Output
    integer, intent(in) :: nint, nlist, iblock(:), jblock(:), sblock(:)
    integer, intent(inout) :: ntbl(:)
    ! Variables
    integer i, ibl, jbl, itbl

    ntbl(1:2*nint) = 0
    do i=1,nlist
       ibl = iblock(i)
       jbl = jblock(i)
       itbl = lower_triangle_index(ibl, jbl) + nint*sblock(i)
       ntbl(itbl) = ntbl(itbl) + 1
    enddo

    return
  end subroutine calc_ntbl_soft

  ! *
  ! * Splits table into nblock*(nblock+1)/2 bond tables
  ! *
  subroutine block_split_tbl_ilist_jlist(nblock_in, ntbl, tbl, ilist, jlist, iblock, &
       ntbl_block, tbl_block)
    implicit none
    ! Input / Output
    integer, intent(in) :: nblock_in, ntbl, tbl(:), ilist(:), jlist(:), iblock(:)
    integer, intent(inout) :: ntbl_block(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    ! Variables
    integer nint
    integer i, j, ibl, jbl, itbl

    ! Number of interactions
    nint = nblock_in*(nblock_in+1)/2

    ntbl_block(1:nint) = 0

    do i=1,ntbl
       j = tbl(i)
       ibl = iblock(ilist(j))
       jbl = iblock(jlist(j))
       itbl = lower_triangle_index(ibl, jbl)
       ntbl_block(itbl) = ntbl_block(itbl) + 1
       tbl_block(itbl)%array(ntbl_block(itbl)) = j
    enddo

    return
  end subroutine block_split_tbl_ilist_jlist

  ! *
  ! * Splits table into nblock*(nblock+1)/2 bond tables
  ! *
  subroutine block_split_tbl_list(nblock_in, ntbl, tbl, iblock, jblock, ntbl_block, tbl_block)
    implicit none
    ! Input / Output
    integer, intent(in) :: nblock_in, ntbl, tbl(:), iblock(:), jblock(:)
    integer, intent(inout) :: ntbl_block(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    ! Variables
    integer nint
    integer i, j, ibl, jbl, itbl

    ! Number of interactions
    nint = nblock_in*(nblock_in+1)/2

    ntbl_block(1:nint) = 0

    do i=1,ntbl
       j = tbl(i)
       ibl = iblock(j)
       jbl = jblock(j)
       itbl = lower_triangle_index(ibl, jbl)
       ntbl_block(itbl) = ntbl_block(itbl) + 1
       tbl_block(itbl)%array(ntbl_block(itbl)) = j
    enddo

    return
  end subroutine block_split_tbl_list

  ! *
  ! * Splits table into 2*nblock*(nblock+1)/2 bond tables, hard followed by soft
  ! *
  subroutine block_split_tbl_soft(nblock_in, ntbl, tbl, iblock, jblock, sblock, ntbl_block, tbl_block)
    implicit none
    ! Input / Output
    integer, intent(in) :: nblock_in, ntbl, tbl(:), iblock(:), jblock(:), sblock(:)
    integer, intent(inout) :: ntbl_block(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    ! Variables
    integer nint
    integer i, j, ibl, jbl, itbl

    ! Number of interactions
    nint = nblock_in*(nblock_in+1)/2

    ntbl_block(1:2*nint) = 0

    do i=1,ntbl
       j = tbl(i)
       ibl = iblock(j)
       jbl = jblock(j)
       itbl = lower_triangle_index(ibl, jbl) + nint*sblock(j)
       ntbl_block(itbl) = ntbl_block(itbl) + 1
       tbl_block(itbl)%array(ntbl_block(itbl)) = j
    enddo

    return
  end subroutine block_split_tbl_soft

  ! *
  ! * Sets each entry in ntbl(1:nint) to be at least 1
  ! *
  subroutine fix_ntbl(nint, ntbl)
    implicit none
    ! Input / Output
    integer, intent(in) :: nint
    integer, intent(inout) :: ntbl(:)
    ! Variables
    integer i

    do i=1,nint
       ntbl(i) = max(1, ntbl(i))
    enddo

    return
  end subroutine fix_ntbl

  ! *
  ! * Returns lower triangle matrix index
  ! *
  integer function lower_triangle_index(ibl, jbl)
    implicit none
    ! Input
    integer, intent(in) :: ibl, jbl
    ! Variables
    integer row, col

    col = min(ibl, jbl)
    row = max(ibl, jbl)
    lower_triangle_index = col + row*(row-1)/2

    return
  end function lower_triangle_index

! -------------- double precision -------------

#define BONDLIST
#define DOUBLE_PREC
#define KERNEL_NAME build_bondlist_block_pd
#include "domdec_bonded_block_kernel.inc"
#undef BONDLIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define ANGLELIST
#define DOUBLE_PREC
#define KERNEL_NAME build_anglelist_block_pd
#include "domdec_bonded_block_kernel.inc"
#undef ANGLELIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define DIHELIST
#define DOUBLE_PREC
#define KERNEL_NAME build_dihelist_block_pd
#include "domdec_bonded_block_kernel.inc"
#undef DIHELIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define XX14LIST
#define DOUBLE_PREC
#define KERNEL_NAME build_xx14list_block_pd
#include "domdec_bonded_block_kernel.inc"
#undef XX14LIST
#undef DOUBLE_PREC
#undef KERNEL_NAME

! -------------- single precision -------------

#define BONDLIST
#define SINGLE_PREC
#define KERNEL_NAME build_bondlist_block_ps
#include "domdec_bonded_block_kernel.inc"
#undef BONDLIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define ANGLELIST
#define SINGLE_PREC
#define KERNEL_NAME build_anglelist_block_ps
#include "domdec_bonded_block_kernel.inc"
#undef ANGLELIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define DIHELIST
#define SINGLE_PREC
#define KERNEL_NAME build_dihelist_block_ps
#include "domdec_bonded_block_kernel.inc"
#undef DIHELIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#define XX14LIST
#define SINGLE_PREC
#define KERNEL_NAME build_xx14list_block_ps
#include "domdec_bonded_block_kernel.inc"
#undef XX14LIST
#undef SINGLE_PREC
#undef KERNEL_NAME

#endif

end module domdec_bonded_block
