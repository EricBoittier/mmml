#ifdef BONDLIST
  ! *
  ! * Build bondlists [nblock*(nblock+1)/2] for block
  ! *
  subroutine KERNEL_NAME(ntbl, tbl, ilist, jlist, icb, iblock, jblock, &
       sblock, qsobo, &
       glo2loc_ind, xyzq, list, ntbl_block, tbl_block_pos, tbl_block)
    implicit none
    ! Input / Output
    integer, intent(in) :: ntbl, tbl(:), ilist(:), jlist(:), icb(:)
    integer, intent(in) :: iblock(:), jblock(:), sblock(:), glo2loc_ind(:)
    logical, intent(in) :: qsobo
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(bondlist_t), intent(inout) :: list(:)
    integer, intent(inout) :: ntbl_block(:), tbl_block_pos(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    ! Variables
    integer i, pos
    integer nint

    nint = nblock*(nblock+1)/2
    if (qsobo) then
       call block_split_tbl_soft(nblock, ntbl, tbl, iblock, jblock, sblock, ntbl_block, tbl_block)
       nint = 2*nint
    else
       call block_split_tbl(nblock, ntbl, tbl, iblock, jblock, ntbl_block, tbl_block)
    endif

    pos = 1
    do i=1,nint
       tbl_block_pos(i) = pos
#ifdef DOUBLE_PREC
       call build_bondlist_pd(ntbl_block(i), tbl_block(i)%array, ilist, jlist, icb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)))
#else
       call build_bondlist_ps(ntbl_block(i), tbl_block(i)%array, ilist, jlist, icb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)))
#endif
       pos = pos + ntbl_block(i)
    enddo
    tbl_block_pos(i) = pos

    return
  end subroutine KERNEL_NAME
#endif

#ifdef ANGLELIST
  ! *
  ! * Build anglelists [nblock*(nblock+1)/2] for block
  ! *
  subroutine KERNEL_NAME(ntbl, tbl, ilist, jlist, klist, icb, ctb, iblock, jblock, &
       sblock, qsobo, &
       glo2loc_ind, xyzq, list, ntbl_block, tbl_block_pos, tbl_block, &
       q_gromacs_style, q_charmm_style)
    implicit none
    ! Input / Output
    integer, intent(in) :: ntbl, tbl(:), ilist(:), jlist(:), klist(:), icb(:)
    real(chm_real), intent(in) :: ctb(:)
    integer, intent(in) :: iblock(:), jblock(:), sblock(:), glo2loc_ind(:)
    logical, intent(in) :: qsobo
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(anglelist_t), intent(inout) :: list(:)
    integer, intent(inout) :: ntbl_block(:), tbl_block_pos(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    logical, intent(out) :: q_gromacs_style, q_charmm_style
    ! Variables
    integer i, pos
    integer nint
    logical q_gromacs_style_tmp, q_charmm_style_tmp

    nint = nblock*(nblock+1)/2
    if (qsobo) then
       call block_split_tbl_soft(nblock, ntbl, tbl, iblock, jblock, sblock, ntbl_block, tbl_block)
       nint = 2*nint
    else
       call block_split_tbl(nblock, ntbl, tbl, iblock, jblock, ntbl_block, tbl_block)
    endif

    q_gromacs_style = .false.
    q_charmm_style = .false.

    pos = 1
    do i=1,nint
       tbl_block_pos(i) = pos
#ifdef DOUBLE_PREC
       call build_anglelist_pd(ntbl_block(i), tbl_block(i)%array, ilist, jlist, klist, icb, ctb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)), q_gromacs_style_tmp, q_charmm_style_tmp)
#else
       call build_anglelist_ps(ntbl_block(i), tbl_block(i)%array, ilist, jlist, klist, icb, ctb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)), q_gromacs_style_tmp, q_charmm_style_tmp)
#endif
       pos = pos + ntbl_block(i)
       q_gromacs_style = q_gromacs_style .or. q_gromacs_style_tmp
       q_charmm_style = q_charmm_style .or. q_charmm_style_tmp
    enddo
    tbl_block_pos(i) = pos

    return
  end subroutine KERNEL_NAME
#endif

#ifdef DIHELIST
  ! *
  ! * Build dihelists [nblock*(nblock+1)/2] for block
  ! *
  subroutine KERNEL_NAME(ntbl, tbl, ilist, jlist, klist, llist, &
       icb, iblock, jblock, sblock, qsobo, &
       glo2loc_ind, xyzq, list, ntbl_block, tbl_block_pos, tbl_block)
    implicit none
    ! Input / Output
    integer, intent(in) :: ntbl, tbl(:), ilist(:), jlist(:), klist(:), llist(:), icb(:)
    integer, intent(in) :: iblock(:), jblock(:), sblock(:), glo2loc_ind(:)
    logical, intent(in) :: qsobo
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(dihelist_t), intent(inout) :: list(:)
    integer, intent(inout) :: ntbl_block(:), tbl_block_pos(:)
    type(intarray_t), intent(inout) :: tbl_block(:)
    ! Variables
    integer i, pos
    integer nint

    nint = nblock*(nblock+1)/2
    if (qsobo) then
       call block_split_tbl_soft(nblock, ntbl, tbl, iblock, jblock, sblock, ntbl_block, tbl_block)
       nint = 2*nint
    else
       call block_split_tbl(nblock, ntbl, tbl, iblock, jblock, ntbl_block, tbl_block)
    endif

    pos = 1
    do i=1,nint
       tbl_block_pos(i) = pos
#ifdef DOUBLE_PREC
       call build_dihelist_pd(ntbl_block(i), tbl_block(i)%array, &
            ilist, jlist, klist, llist, icb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)))
#else
       call build_dihelist_ps(ntbl_block(i), tbl_block(i)%array, &
            ilist, jlist, klist, llist, icb, &
            glo2loc_ind, xyzq, list(pos:pos+ntbl_block(i)))
#endif
       pos = pos + ntbl_block(i)
    enddo
    tbl_block_pos(i) = pos

    return
  end subroutine KERNEL_NAME
#endif

#ifdef XX14LIST
  ! *
  ! * Build in14list/ex14list [nblock*(nblock+1)/2] for block
  ! *
  subroutine KERNEL_NAME(nxx14tbl, xx14tbl, xx14i, xx14j, iblock, &
       glo2loc_ind, xyzq, xx14list, nxx14tbl_block, xx14tbl_block_pos)
    implicit none
    ! Input / Output
    integer, intent(in) :: nxx14tbl, xx14tbl(:), xx14i(:), xx14j(:), iblock(:), glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(list14_t), intent(inout) :: xx14list(:)
    integer, intent(inout) :: nxx14tbl_block(:), xx14tbl_block_pos(:)
    ! Variables
    integer i, j, pos

    call block_split_tbl(nblock, nxx14tbl, xx14tbl, xx14i, xx14j, iblock, &
         nxx14tbl_block, xx14tbl_block)

    pos = 1
    do i=1,nblock*(nblock+1)/2
       xx14tbl_block_pos(i) = pos
#ifdef DOUBLE_PREC
       call build_xx14list_pd(nxx14tbl_block(i), xx14tbl_block(i)%array, xx14i, xx14j, &
            glo2loc_ind, xyzq, xx14list(pos:pos+nxx14tbl_block(i)))
#else
       call build_xx14list_ps(nxx14tbl_block(i), xx14tbl_block(i)%array, xx14i, xx14j, &
            glo2loc_ind, xyzq, xx14list(pos:pos+nxx14tbl_block(i)))
#endif
       pos = pos + nxx14tbl_block(i)
    enddo
    xx14tbl_block_pos(i) = pos

    return
  end subroutine KERNEL_NAME
#endif
  
