module nblist_pair

  !
  ! Neighbor list subroutines and data structures for pair list
  !
#if KEY_DOMDEC==1
  use chm_kinds
  use dimens_fcm
  !-----------------------------------------------------
  ! These are for nblist_pair_kernel.inc
  use groupxfast,only:group_out
  use nblist_types,only:nblist_pair_t
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  !-----------------------------------------------------
  implicit none
  private

  ! Indices to nblist:
  ! iuu    = solute - solute with VdW
  ! iuuc   = solute - solute with Coulomb and VdW
  ! iuv    = solute - solvent with VdW
  ! iuvc   = solute - solvent with Coulomb and VdW
  ! ivv    = solvent -solvent
  !
  ! _block are for BLOCK
  ! 
  integer iuu, iuuc, iuv, iuvc, ivv
#if KEY_BLOCK==1
  integer iuuc_block
#endif

  ! Public variables
  public iuu, iuuc, iuv, iuvc, ivv
#if KEY_BLOCK==1
  public iuuc_block
#endif

  ! Public subroutines
  public flush_pair_ps
  public flush_pair_pd
#if KEY_BLOCK==1
  public flush_pair_block_ps, flush_pair_block_pd
#endif
#endif

contains

#if KEY_DOMDEC==1
  ! *
  ! * Start a new i-list
  ! *
  subroutine add_i_nblist(ni, nj, ni_max, indi, startj, iscoord, ish, i)
    use memory
    implicit none
    ! Input / Output
    integer, intent(inout) :: ni, ni_max
    integer, intent(inout), allocatable, dimension(:) :: indi, iscoord, startj
    integer, intent(in) :: ish, i, nj

    ni = ni + 1
    if (ni > ni_max) then
       ni_max = int(ni*1.5)
       ! NOTE: startj size is ni_max+1 to fit the stop list
       call chmrealloc('nblist_pair.src','add_i_nblist','indi',ni_max,intg=indi)
       call chmrealloc('nblist_pair.src','add_i_nblist','startj',ni_max+1,intg=startj)
       call chmrealloc('nblist_pair.src','add_i_nblist','scoord',ni_max,intg=iscoord)
    endif
    ! Store atom/solvent group index
    indi(ni) = i
    ! Store coordinate shift
    iscoord(ni) = ish

    startj(ni) = nj + 1

    return
  end subroutine add_i_nblist

  ! *
  ! * Stop i-list
  ! *
  subroutine stop_i_nblist(ni, nj, startj)
    implicit none
    ! Input / Output
    integer, intent(in) :: nj
    integer, intent(inout) :: ni
    integer, intent(in) :: startj(:)

    if (startj(ni) == nj + 1) then
       ! No j groups/atoms has been added to this list => remove the entry i
       ni = ni - 1
    endif

    return
  end subroutine stop_i_nblist

  ! *
  ! * Add j to nblist
  ! *
  subroutine add_j_nblist(nj, nj_max, indj, j)
    use memory

    implicit none

    ! Input / Output
    integer, intent(inout) :: nj, nj_max
    integer, intent(inout), allocatable, dimension(:) :: indj
    integer, intent(in) :: j

    ! local vars
    integer, allocatable, dimension(:) :: temp
    integer :: new_size

    nj = nj + 1
    if (nj > nj_max) then
       ! removed call to chmrealloc to
       ! avoid segfault with intel 16.x, openmpi 3 and domdec_gpu
       ! nj_max = int(nj*1.5)
       ! call chmrealloc('nblist_pair.src','add_j_nblist','indj',nj_max,intg=indj)
       new_size = int(nj*1.5)
       call chmalloc('nblist_pair.src', 'add_j_nblist', 'temp', &
            new_size, intg = temp)
       temp(1:nj_max) = indj(1:nj_max)
       call move_alloc(temp, indj)
       nj_max = new_size
    endif
    indj(nj) = j

    return
  end subroutine add_j_nblist

#if KEY_BLOCK==1
  logical function q_block_in_use_i(is, iq, is_unpack)
    use number,only:one
    use block_ltm,only:iblckp
    use lambdam,only:fullblcoep
    implicit none
    ! Input
    integer, intent(in) :: is, iq, is_unpack
    ! Variables
    real(chm_real) scale
    integer ii, jj, ii_unpack, jj_unpack

    q_block_in_use_i = .false.

    ii_unpack = is_unpack
    do ii=is,iq-1
       jj_unpack = ii_unpack + 1
       do jj=ii+1,iq
          scale = fullblcoep(iblckp(ii_unpack),iblckp(jj_unpack))
          if (scale /= one) then
             q_block_in_use_i = .true.
             return
          endif
          jj_unpack = jj_unpack + 1
       enddo
       ii_unpack = ii_unpack + 1
    enddo

    return
  end function q_block_in_use_i

  ! *
  ! * Check if any of the atom is in block
  ! *
  logical function q_block_in_use_ij(is, iq, is_unpack, ntmpl, tmpl, group, group_unpack)
    use number,only:one
    use groupxfast,only:group_out
    use block_ltm,only:iblckp
    use lambdam,only:fullblcoep
    implicit none
    ! Input
    integer, intent(in) :: is, iq, is_unpack, ntmpl, tmpl(2,*), group(:), group_unpack(:)
    ! Variables
    real(chm_real) scale
    integer ii, jj, js, jq, j, jg, ii_unpack, jj_unpack, ibl, jbl, dummy

    q_block_in_use_ij = .false.

    ii_unpack = is_unpack
    do ii=is,iq
       do j=1,ntmpl
          jg = tmpl(1,j)
          call group_out(group(jg), js, jq, dummy)
          jg = tmpl(2,j)
          call group_out(group_unpack(jg), jj_unpack, dummy)
          do jj=js,jq
             ibl = iblckp(ii_unpack)
             jbl = iblckp(jj_unpack)
             scale = fullblcoep(ibl,jbl)
             ! Block is in use if
             ! (a) scale /= 1.0
             !       OR
             ! (b) atom i or j belong to other blocks than block 1 (=environment)
             if (scale /= one .or. ibl /= 1 .or. jbl /= 1) then
                q_block_in_use_ij = .true.
                return
             endif
             jj_unpack = jj_unpack + 1
          enddo
       enddo
       ii_unpack = ii_unpack + 1
    enddo

    return
  end function q_block_in_use_ij

#define BLOCK_ON
#define DOUBLE_PREC
#define KERNEL_NAME flush_pair_block_pd
#include "nblist_pair_kernel.inc"
#undef BLOCK_ON
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define BLOCK_ON
#define SINGLE_PREC
#define KERNEL_NAME flush_pair_block_ps
#include "nblist_pair_kernel.inc"
#undef BLOCK_ON
#undef SINGLE_PREC
#undef KERNEL_NAME

#endif

#define GROUP_ON
#define DOUBLE_PREC
#define KERNEL_NAME flush_pair_pd
#include "nblist_pair_kernel.inc"
#undef GROUP_ON
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define GROUP_ON
#define SINGLE_PREC
#define KERNEL_NAME flush_pair_ps
#include "nblist_pair_kernel.inc"
#undef GROUP_ON
#undef SINGLE_PREC
#undef KERNEL_NAME

  ! *
  ! * Writes neighbor list into a unit
  ! *
  subroutine write_nblist(unit, lnblist, ind)
    use nblist_types,only:nblist_pair_t
    implicit none
    ! Input / Output
    type(nblist_pair_t) lnblist(:)
    integer unit, ind
    ! Variables
    integer i, j, j0, j1, ii(3), jj

    do i=1,lnblist(ind)%ni
       j0 = lnblist(ind)%startj(i)
       j1 = lnblist(ind)%startj(i+1) - 1
       ii(1) = lnblist(ind)%indi(i)
       ii(2) = ii(1) + 1
       ii(3) = ii(1) + 2
       do j=j0,j1
          jj = lnblist(ind)%indj(j)
          if (ind == iuvc) then
             write (unit,'(2i6)') min(ii(1),jj), max(ii(1),jj)
             write (unit,'(2i6)') min(ii(2),jj), max(ii(2),jj)
             write (unit,'(2i6)') min(ii(3),jj), max(ii(3),jj)
          else
             write (unit,'(2i6)') min(ii(1),jj), max(ii(1),jj)
          endif
       enddo
    enddo

    return
  end subroutine write_nblist

#endif

end module nblist_pair
