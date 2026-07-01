module nblist_tilex

#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
  !
  ! Tilex neighbor search
  ! NOTE: all calculations are done in single precision
  
  use, intrinsic :: iso_c_binding
  use chm_kinds
  use dimens_fcm
  use nblist_types,only:tilesize, num_excl, intarray_t, intarray2_t, cr4array_t, &
       ientry_t, tile_excl_t
  implicit none
  private

  type bb_t
     sequence
     real(chm_real4) x0, y0, z0   ! center
     real(chm_real4) xw, yw, zw   ! (half) width
  end type bb_t

  ! Exclusion mask
  type(tile_excl_t), allocatable, dimension(:) :: tile_excl1, tile_excl2
  integer, allocatable, dimension(:) :: tiles1, tiles2

  ! Final exclusion with correct topological exclusions
  integer, pointer, dimension(:) :: tile_ind_top1, tile_ind_top2
  type(tile_excl_t), pointer, dimension(:) :: tile_excl_top1, tile_excl_top2

  integer, allocatable, dimension(:) :: tile_ind_shift1, tile_ind_shift2

  integer, allocatable, dimension(:) :: num_tiles_table1, num_tiles_table2
  integer, allocatable, dimension(:) :: pos_shift1, pos_shift2

  ! Cells
  logical(kind=1), allocatable, dimension(:) :: sort_cell

  ! Arrays for ns_tilex
  type(cr4array_t), allocatable, dimension(:) :: xdist, ydist, zdist

  type(intarray_t), allocatable, dimension(:) :: jlist1, jlist2
  type(bb_t), allocatable, dimension(:) :: cell_bb
  integer, allocatable, dimension(:) :: ex14_len
  type(intarray2_t), allocatable, dimension(:) :: ex14    ! Temporary 1-4 list

  integer, allocatable, dimension(:) :: n_ijlist1, ijlist1_len
  integer, allocatable, dimension(:) :: n_ijlist2, ijlist2_len
  ! ijlist for each thread
  type(intarray2_t), allocatable, dimension(:) :: ijlist1, ijlist2
  ! Total ijlist(1:3,:)
  integer, pointer, dimension(:,:) :: ijlist1_tot, ijlist2_tot
  
  ! Total ientry
  type(ientry_t), pointer, dimension(:) :: ientry1_tot, ientry2_tot

  type(ientry_t), allocatable, dimension(:) :: ientry_tmp

  ! Array cumsum_th(0:nthread) is used for cumulative sums over threads
  integer, allocatable, dimension(:) :: cumsum_th

  ! Sort lists for sort_ientry
  integer, allocatable, dimension(:) :: sortlist1, sortlist2
  integer, allocatable, dimension(:,:) :: bucket

  ! Public subroutines
  public ns_tilex, print_info_tilex, write_nj
  public uninit_nblist_tilex, uninit_pinned_nblist_tilex

  ! CUDA routines, these are in nblist_tilex_gpu.cu
  interface

     subroutine copy_ijlist_to_gpu(h_whichlist, h_n_ijlist, h_ijlist) &
          bind(C,name="copy_ijlist_to_gpu")
       import
       integer(c_int), intent(in) :: h_whichlist, h_n_ijlist
       integer(c_int), intent(in) :: h_ijlist(*)
     end subroutine copy_ijlist_to_gpu

     subroutine build_excl_dist_index(h_whichlist, h_boxx, h_boxy, h_boxz, h_roff) &
          bind(C,name="build_excl_dist_index")
       import
       integer(c_int), intent(in) :: h_whichlist
       real(c_double), intent(in) :: h_boxx, h_boxy, h_boxz, h_roff
     end subroutine build_excl_dist_index

     subroutine copy_ientry_to_gpu(h_whichlist, h_ni, h_ientry) &
          bind(C,name="copy_ientry_to_gpu")
       use nblist_types,only:ientry_t
       import
       integer(c_int), intent(in) :: h_whichlist, h_ni
       type(ientry_t), intent(in) :: h_ientry(*)
     end subroutine copy_ientry_to_gpu

     subroutine check_tilex(h_tilex_cpu) &
          bind(C,name="check_tilex")
       import
       integer(c_int), intent(in) :: h_tilex_cpu(*)
     end subroutine check_tilex

     subroutine combine_tile_top_to_gpu(h_whichlist, h_ntile_top, h_tile_ind_top, h_tile_excl_top) &
          bind(C,name='combine_tile_top_to_gpu')
       import
       integer(c_int), intent(in) :: h_whichlist, h_ntile_top
       integer(c_int), intent(in) :: h_tile_ind_top(*)
       type(tile_excl_t), intent(in) :: h_tile_excl_top(*)
     end subroutine combine_tile_top_to_gpu

  end interface

contains

  ! *
  ! * Merges ijlist(0:nthread-1) into ijlist_tot()
  ! *
  subroutine merge_ijlist(n_ijlist, ijlist, n_ijlist_tot, ijlist_tot)
    use domdec_common,only:nthread
    use nblist_util,only:init_array_gpu
    implicit none
    ! Input / Output
    integer, intent(in) :: n_ijlist(0:nthread-1)
    type(intarray2_t), intent(in) :: ijlist(0:nthread-1)
    integer, intent(out) :: n_ijlist_tot
    integer, pointer, dimension(:,:), intent(inout) :: ijlist_tot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num 
#endif
    ! Variables
    integer i
    integer tid, istart, iend

    ! Calculates cumulative sum of n_ijlist(0:nthread-1) to cumsum_th(0:nthread)
    cumsum_th(0) = 0
    do i=1,nthread
       cumsum_th(i) = cumsum_th(i-1) + n_ijlist(i-1)
    enddo

    n_ijlist_tot = cumsum_th(nthread)

    ! Allocate pinned CPU memory for ijlist_tot
    call init_array_gpu(ijlist_tot, 3, max(1,n_ijlist_tot), 1.4_chm_real)

    ! Copy ijlist(0:nthread-1)%array to ijlist_tot
!$omp parallel private(tid, istart, iend)
#ifndef _OPENMP
    tid = 0                     
#else
    tid = omp_get_thread_num()  
#endif
    istart = cumsum_th(tid) + 1
    iend = cumsum_th(tid+1)
    ijlist_tot(1:3,istart:iend) = ijlist(tid)%array(1:3,1:n_ijlist(tid))
!$omp end parallel

    return
  end subroutine merge_ijlist

  ! *
  ! * Copies nblist to GPU
  ! *
  subroutine copy_nblist_to_gpu(whichlist, nblist, ientry_tot)
    use nblist_types,only:nblist_tilex_t
    use domdec_common,only:nthread
    use nblist_util,only:alloc_gpu, dealloc_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
    implicit none
    ! Input
    integer, intent(in) :: whichlist
    type(nblist_tilex_t), intent(in), allocatable, dimension(:) :: nblist
    type(ientry_t), intent(inout), pointer, dimension(:) :: ientry_tot
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num 
#endif
    ! Variables
    integer i, ni_tot, ni_split
    integer tid, istart, iend

    ! Calculates cumulative sum of nblist(0:nthread-1)%ni to cumsum_th(0:nthread)
    cumsum_th(0) = 0
    do i=1,nthread
       cumsum_th(i) = cumsum_th(i-1) + nblist(i-1)%ni
    enddo

    ni_tot = cumsum_th(nthread)

    ! Allocate pinned CPU memory for ientry_tot
    if (associated(ientry_tot)) then
       if (size(ientry_tot) < ni_tot*2) then
          call dealloc_gpu(ientry_tot)
       endif
    endif
    if (.not.associated(ientry_tot)) then
       call alloc_gpu(ientry_tot, max(1,int(ni_tot*2*1.4)))
    endif

    if (allocated(ientry_tmp)) then
       if (size(ientry_tmp) < ni_tot*2) then
          deallocate(ientry_tmp)
       endif
    endif
    if (.not.allocated(ientry_tmp)) then
       allocate(ientry_tmp(int(ni_tot*2*1.4)))
    endif

    ! Copy nblist(0:nthread)%ientry to ientry_tot
!$omp parallel private(tid, istart, iend)
#ifndef _OPENMP
    tid = 0                     
#endif
#ifdef _OPENMP
    tid = omp_get_thread_num()  
#endif
    istart = cumsum_th(tid) + 1
    iend = cumsum_th(tid+1)
    ientry_tmp(istart:iend) = nblist(tid)%ientry(1:nblist(tid)%ni)
!$omp end parallel

!!$    call range_start('split_ientry')
!!$    call split_ientry(ni_tot, ientry_tot, 30, ni_split, ientry_tmp)
!!$    call range_stop()
!!$    ni_tot = ni_split

    ! Sort ientry_tmp -> ientry_tot
    call range_start('sort_ientry')
    call sort_ientry(ni_tot, ientry_tmp, ientry_tot, .true.)
    call range_stop()
!!$    ientry_tot(1:ni_tot) = ientry_tmp(1:ni_tot)

    ! Copy ientry_tot to GPU
    call copy_ientry_to_gpu(whichlist, ni_tot, ientry_tot)

    return
  end subroutine copy_nblist_to_gpu

  ! *
  ! * Sorts ientry according to the number of j-entries
  ! *
  subroutine sort_ientry(n_in, ientry_in, ientry_out, inv_sort)
    use memory
    use nblist_util,only:bucket_sort
!    use exfunc,only:order5
    implicit none
!    ! External subroutines
!    external exch5
    ! Input / Output
    integer, intent(in) :: n_in
    type(ientry_t), intent(in) :: ientry_in(:)
    type(ientry_t), intent(out) :: ientry_out(:)
    logical, intent(in) :: inv_sort
    ! Variables
    integer i, j, num, min_num, max_num

    if (allocated(sortlist1)) then
       if (size(sortlist1) < n_in) then
          call chmdealloc('nblist_tilex.src','sort_ientry','sortlist1',&
               size(sortlist1),intg=sortlist1)
          call chmdealloc('nblist_tilex.src','sort_ientry','sortlist2',&
               size(sortlist2),intg=sortlist2)
       endif
    endif
    if (.not.allocated(sortlist1)) then
       call chmalloc('nblist_tilex.src','sort_ientry','sortlist1',&
            int(n_in*1.4),intg=sortlist1)
       call chmalloc('nblist_tilex.src','sort_ientry','sortlist2',&
            int(n_in*1.4),intg=sortlist2)
    endif

    min_num = 100000000
    max_num = 0
!$omp parallel do private(i, num) reduction(min:min_num) reduction(max:max_num)
    do i=1,n_in
       num = ientry_in(i)%endj - ientry_in(i)%startj + 1
       sortlist1(i) = num
       min_num = min(num, min_num)
       max_num = max(num, max_num)
    enddo
!$omp end parallel do

    ! Sort keys given in sortlist1, new order is given by sortlist2
    ! NOTE: Only sortlist2 is created
    call bucket_sort(n_in, sortlist1, sortlist2, bucket, 1.4_chm_real, min_num, max_num)

!    call sort(n_in, exch5, order5, sortlist1, sortlist2, 0, 0, 0, 0, 0, 2)

    if (inv_sort) then
!$omp parallel do private(i, j) schedule(static)
       do i=1,n_in
          j = sortlist2(i)
          ientry_out(n_in-i+1)%indi = ientry_in(j)%indi
          ientry_out(n_in-i+1)%ish = ientry_in(j)%ish
          ientry_out(n_in-i+1)%startj = ientry_in(j)%startj
          ientry_out(n_in-i+1)%endj = ientry_in(j)%endj
       enddo
!$omp end parallel do
    else
!$omp parallel do private(i, j) schedule(static)
       do i=1,n_in
          j = sortlist2(i)
          ientry_out(i)%indi = ientry_in(j)%indi
          ientry_out(i)%ish = ientry_in(j)%ish
          ientry_out(i)%startj = ientry_in(j)%startj
          ientry_out(i)%endj = ientry_in(j)%endj
       enddo
!$omp end parallel do
    endif

    return
  end subroutine sort_ientry

  ! *
  ! * Splits ientry
  ! *
  subroutine split_ientry(n_in, ientry_in, level, n_out, ientry_out)
    implicit none
    ! Input / Output
    integer, intent(in) :: n_in
    type(ientry_t), intent(in) :: ientry_in(:)
    integer, intent(in) :: level
    integer, intent(out) :: n_out
    type(ientry_t), intent(out) :: ientry_out(:)
    ! Variables
    integer i, num

    n_out = 0
    do i=1,n_in
       num = ientry_in(i)%endj - ientry_in(i)%startj + 1
       if (num > level) then
          n_out = n_out + 1
          ientry_out(n_out)%indi = ientry_in(i)%indi
          ientry_out(n_out)%ish = ientry_in(i)%ish
          ientry_out(n_out)%startj = ientry_in(i)%startj
          ientry_out(n_out)%endj = ientry_in(i)%startj + num/2
          n_out = n_out + 1
          ientry_out(n_out)%indi = ientry_in(i)%indi
          ientry_out(n_out)%ish = ientry_in(i)%ish
          ientry_out(n_out)%startj = ientry_in(i)%startj + num/2 + 1
          ientry_out(n_out)%endj = ientry_in(i)%endj
       else
          n_out = n_out + 1
          ientry_out(n_out)%indi = ientry_in(i)%indi
          ientry_out(n_out)%ish = ientry_in(i)%ish
          ientry_out(n_out)%startj = ientry_in(i)%startj
          ientry_out(n_out)%endj = ientry_in(i)%endj
       endif
    enddo

    return
  end subroutine split_ientry

  ! *
  ! * Neighborlist search
  ! *
  subroutine ns_tilex(nblist1, nblist2, xyzq, loc2glo_ind, glo2loc_ind, ncoord, &
       top_excl_pos, top_excl)
    use memory
    use domdec_common,only:zonelist, boxx, boxy, boxz, nx, ny, nz, nthread, q_test
    use nblist_types,only:nblist_tilex_t, nblist_pair_t
    use domdec_local_types,only:xyzq_sp_t
    use inbnd,only:cutnb, ctofnb
    use domdec_util_gpu_mod,only:range_start, range_stop
    use nblist_util,only:alloc_gpu, dealloc_gpu
    use nblist_tilex_sort,only:ncellx, ncelly, ncellz, max_ncellz, ncell, &
         n_int_zone, int_zone, cellbx, cellby, cellbz, cell_start, cell_index, icellxyz, &
         zone_cell_start, startcol_zone, max_ncellz, startcell_col
    implicit none
    ! Input / Output
    type(nblist_tilex_t), intent(inout), allocatable, dimension(:) :: nblist1, nblist2
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:), glo2loc_ind(:)
    integer, intent(in) :: ncoord
    integer, intent(in) :: top_excl_pos(:), top_excl(:)
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num  
#endif
    ! Variables
    real(chm_real4) cut, cutsq
    real(chm_real4) boxxf, boxyf, boxzf
    integer tid
    integer imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    integer i, tile_start, tile_end
    integer num_tiles1, num_tiles2
    integer ntile1_tot, n_ijlist1_tot, ntile2_tot, n_ijlist2_tot

    boxxf = boxx
    boxyf = boxy
    boxzf = boxz

    cut = cutnb
    cutsq = cut*cut

    ! Allocate & re-allocate memory for this subroutine
    ! (xdist, ydist, zdist, jlist1, cumsum_th, ijlist, ijlist_len, n_ijlist, ijlist, cell_bb,
    !  nblist, ex14, ex14_len, sort_cell, tile_ind_shift, num_tiles_table, pos_shift)
    call alloc_realloc(ncellx, ncelly, max_ncellz, ncell, ncoord)

    ! Set image loop boundaries
    imx_lo = 0
    imx_hi = 0
    imy_lo = 0
    imy_hi = 0
    imz_lo = 0
    imz_hi = 0
    if (nx == 1) then
       imx_lo = -1
       imx_hi = 1
    endif
    if (ny == 1) then
       imy_lo = -1
       imy_hi = 1
    endif
    if (nz == 1) then
       imz_lo = -1
       imz_hi = 1
    endif

    !    call print_iexcl_ibit()

    ! Calculate bounding boxes
    call range_start('calc_bounding_box')
    call calc_bounding_box(xyzq, ncell, cell_start)
    call range_stop()

    call range_start('ns_tilex_kernel1')
!$omp parallel private(tid)
#ifndef _OPENMP
    tid = 0                     
#else
    tid = omp_get_thread_num()  
#endif
    ! Do the neighbor search 1
    call ns_tilex_kernel1(zone_cell_start(2)-1, ncellx, ncelly, ncellz, &
         cell_start, icellxyz, startcol_zone, max_ncellz, startcell_col, &
         cellbx, cellby, cellbz, &
         boxxf, boxyf, boxzf, cut, cutsq, &
         imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi, &
         xdist(tid)%array, ydist(tid)%array, zdist(tid)%array, &
         jlist1(tid)%array, nblist1(tid), ijlist1(tid)%array, n_ijlist1(tid), ijlist1_len(tid))
!$omp end parallel
    call range_stop()

    ! Copy ijlist1 to GPU
    call range_start('merge/copy_ijlist_to_gpu 1')
    call merge_ijlist(n_ijlist1, ijlist1, n_ijlist1_tot, ijlist1_tot)
    call copy_ijlist_to_gpu(1, n_ijlist1_tot, ijlist1_tot)
    call range_stop()

    ! Build exclusion mask on GPU based on distance and index (uses ijlist1_tot)
    call build_excl_dist_index(1, boxx, boxy, boxz, cutnb)

    call range_start('ns_tilex_kernel2')
!$omp parallel private(tid)
#ifndef _OPENMP
    tid = 0                     
#else
    tid = omp_get_thread_num()  
#endif
    ! Do the neighbor search 2
    call ns_tilex_kernel2(zone_cell_start(2), ncell, ncellx, ncelly, ncellz, &
         cell_start, icellxyz, zone_cell_start, startcol_zone, max_ncellz, startcell_col, &
         cellbx, cellby, cellbz, &
         n_int_zone, int_zone, &
         boxxf, boxyf, boxzf, cut, cutsq, &
         imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi, &
         xdist(tid)%array, ydist(tid)%array, zdist(tid)%array, &
         jlist2(tid)%array, nblist2(tid), ijlist2(tid)%array, n_ijlist2(tid), ijlist2_len(tid))
!$omp end parallel
    call range_stop()

    ! Copy ijlist2 to GPU
    call range_start('merge/copy_ijlist_to_gpu 2')
    call merge_ijlist(n_ijlist2, ijlist2, n_ijlist2_tot, ijlist2_tot)
    call copy_ijlist_to_gpu(2, n_ijlist2_tot, ijlist2_tot)
    call range_stop()

    ! Build exclusion mask on GPU based on distance and index (uses ijlist2_tot)
    call build_excl_dist_index(2, boxx, boxy, boxz, cutnb)

    ! Reduce startj and endj counts
    call range_start('reduce_nblist_startj_endj')
    call reduce_nblist_startj_endj(nblist1, ntile1_tot, tile_ind_shift1)
    call reduce_nblist_startj_endj(nblist2, ntile2_tot, tile_ind_shift2)
    call range_stop()

    ! Copy nblist to GPU
    call range_start('copy_nblist_to_gpu 1')
    call copy_nblist_to_gpu(1, nblist1, ientry1_tot)
    call range_stop()

    ! Copy nblist to GPU
    call range_start('copy_nblist_to_gpu 2')
    call copy_nblist_to_gpu(2, nblist2, ientry2_tot)
    call range_stop()

!    ! Test neighborlist
!    if (q_test) then
!       call test1_ns_tilex(ncoord, xyzq, boxxf, boxyf, boxzf, cutsq, &
!            zone_cell_start, ncell, cell_start, &
!            n_ijlist_tot, ijlist_tot)
!    endif
    
    ! Allocate & reallocate tile_excl
    if (allocated(tile_excl1)) then
       if (size(tile_excl1) < ntile1_tot) then
          deallocate(tile_excl1)
       endif
    endif
    if (.not.allocated(tile_excl1)) then
       allocate(tile_excl1(int(ntile1_tot*1.4)))
    endif

    if (allocated(tile_excl2)) then
       if (size(tile_excl2) < ntile2_tot) then
          deallocate(tile_excl2)
       endif
    endif
    if (.not.allocated(tile_excl2)) then
       allocate(tile_excl2(int(ntile2_tot*1.4)))
    endif
    
    ! Allocate & reallocate tiles
    if (allocated(tiles1)) then
       if (size(tiles1) < ntile1_tot) then
          call chmdealloc('nblist_tilex.src','ns_tilex','tiles1',&
               size(tiles1),intg=tiles1)
       endif
    endif
    if (.not.allocated(tiles1)) then
       call chmalloc('nblist_tilex.src','ns_tilex','tiles1',&
            int(ntile1_tot*1.4),intg=tiles1)
    endif

    if (allocated(tiles2)) then
       if (size(tiles2) < ntile2_tot) then
          call chmdealloc('nblist_tilex.src','ns_tilex','tiles2',&
               size(tiles2),intg=tiles2)
       endif
    endif
    if (.not.allocated(tiles2)) then
       call chmalloc('nblist_tilex.src','ns_tilex','tiles2',&
            int(ntile2_tot*1.4),intg=tiles2)
    endif

    ! Build exclusion mask on CPU based on topology
    call range_start('build_excl_topology')
!$omp parallel private(tid)
#ifndef _OPENMP
    tid = 0
#else
    tid = omp_get_thread_num()
#endif
!$omp do schedule(static) private(i)
    do i=1,ntile1_tot
       tiles1(i) = 0
    enddo
!$omp end do
!$omp do schedule(static) private(i)
    do i=1,ntile2_tot
       tiles2(i) = 0
    enddo
!$omp end do
    call build_excl_topology(nblist1(tid), ijlist1(tid)%array, loc2glo_ind, glo2loc_ind, &
         cell_start, cell_index, tiles1, tile_excl1, tile_ind_shift1(tid), top_excl_pos, top_excl)
    call build_excl_topology(nblist2(tid), ijlist2(tid)%array, loc2glo_ind, glo2loc_ind, &
         cell_start, cell_index, tiles2, tile_excl2, tile_ind_shift2(tid), top_excl_pos, top_excl)
!$omp end parallel
    call range_stop()

    ! Reduce tile_excl to tile_excl_top, which only contains entries that are non-zero
    call range_start('reduce_tile_excl')

    ! Calculate the number of tiles that have a non-zero topological exclusion mask
    num_tiles1 = 0
    num_tiles2 = 0
!$omp parallel private(tid, i, tile_start, tile_end) reduction(+:num_tiles1, num_tiles2)
#ifndef _OPENMP
    tid = 0                     
#else
    tid = omp_get_thread_num()  
#endif
    tile_start = tid*(ntile1_tot)/nthread + 1
    tile_end = (tid+1)*(ntile1_tot)/nthread
    do i=tile_start,tile_end
       num_tiles1 = num_tiles1 + tiles1(i)
    enddo
    num_tiles_table1(tid) = num_tiles1

    tile_start = tid*(ntile2_tot)/nthread + 1
    tile_end = (tid+1)*(ntile2_tot)/nthread
    do i=tile_start,tile_end
       num_tiles2 = num_tiles2 + tiles2(i)
    enddo
    num_tiles_table2(tid) = num_tiles2
!$omp end parallel

    ! Allocate & reallocate tile_excl_top
    if (associated(tile_excl_top1)) then
       if (size(tile_excl_top1) < num_tiles1) then
          call dealloc_gpu(tile_excl_top1)
       endif
    endif
    if (.not.associated(tile_excl_top1) .and. num_tiles1 > 0) then
       call alloc_gpu(tile_excl_top1, int(num_tiles1*1.4))
    endif

    if (associated(tile_excl_top2)) then
       if (size(tile_excl_top2) < num_tiles2) then
          call dealloc_gpu(tile_excl_top2)
       endif
    endif
    if (.not.associated(tile_excl_top2) .and. num_tiles2 > 0) then
       call alloc_gpu(tile_excl_top2, int(num_tiles2*1.4))
    endif

    ! Allocate & reallocate tile_ind_top
    if (associated(tile_ind_top1)) then
       if (size(tile_ind_top1) < num_tiles1) then
          call dealloc_gpu(tile_ind_top1)
       endif
    endif
    if (.not.associated(tile_ind_top1) .and. num_tiles1 > 0) then
       call alloc_gpu(tile_ind_top1, int(num_tiles1*1.4))
    endif
    
    if (associated(tile_ind_top2)) then
       if (size(tile_ind_top2) < num_tiles2) then
          call dealloc_gpu(tile_ind_top2)
       endif
    endif
    if (.not.associated(tile_ind_top2) .and. num_tiles2 > 0) then
       call alloc_gpu(tile_ind_top2, int(num_tiles2*1.4))
    endif

    ! Calculate position shift where each thread starts writing tile_excl_top
    pos_shift1(0) = 0
    do i=1,nthread-1
       pos_shift1(i) = pos_shift1(i-1) + num_tiles_table1(i-1)
    enddo

    pos_shift2(0) = 0
    do i=1,nthread-1
       pos_shift2(i) = pos_shift2(i-1) + num_tiles_table2(i-1)
    enddo

!$omp parallel private(tid, tile_start, tile_end)
#ifndef _OPENMP
    tid = 0                     
#else
    tid = omp_get_thread_num()  
#endif
    tile_start = tid*ntile1_tot/nthread + 1
    tile_end = (tid+1)*ntile1_tot/nthread
    call reduce_to_tile_top(tile_excl1, tile_ind_top1, tile_excl_top1, &
         tiles1, tile_start, tile_end, pos_shift1(tid) + 1)
    tile_start = tid*ntile2_tot/nthread + 1
    tile_end = (tid+1)*ntile2_tot/nthread
    call reduce_to_tile_top(tile_excl2, tile_ind_top2, tile_excl_top2, &
         tiles2, tile_start, tile_end, pos_shift2(tid) + 1)
!$omp end parallel

    call range_stop()

    ! Combine (tile_ind_top, tile_excl_top) on GPU
    call combine_tile_top_to_gpu(1, num_tiles1, tile_ind_top1, tile_excl_top1)
    call combine_tile_top_to_gpu(2, num_tiles2, tile_ind_top2, tile_excl_top2)

    return
    
  contains
    subroutine alloc_realloc(ncellx, ncelly, max_ncellz, ncell, ncoord)
      use memory
      use consta,only:pi
      use domdec_common,only:nthread
      use nblist_util,only:init_array
      implicit none
      ! Input
      integer, intent(in) :: ncellx(8), ncelly(8), max_ncellz(8), ncell, ncoord
      ! Variables
      integer i, len, num_j_per_i

      ! NOTE: +128 is added for padding for xdist, ydist, and zdist. This makes openmp run faster

      ! xdist
      call init_array(xdist, nthread, sum(ncellx), int(sum(ncellx)*1.5) + 128)

      ! ydist
      call init_array(ydist, nthread, sum(ncelly), int(sum(ncelly)*1.5) + 128)

      ! zdist
      call init_array(zdist, nthread, maxval(max_ncellz), int(maxval(max_ncellz)*1.5) + 128)

      ! jlist
      call init_array(jlist1, nthread, 1024)
      call init_array(jlist2, nthread, 1024)

      ! cumsum_th
      if (allocated(cumsum_th)) then
         if (size(cumsum_th) < nthread+1) then
            call chmdealloc('nblist_tilex.src','ns_tilex','cumsum_th',&
                 size(cumsum_th),intg=cumsum_th)
         endif
      endif
      if (.not.allocated(cumsum_th)) then
         call chmalloc('nblist_tilex.src','ns_tilex','cumsum_th',&
              nthread+1,lbou=0,intg=cumsum_th)
      endif

      ! ijlist_len
      if (allocated(ijlist1_len)) then
         if (size(ijlist1_len) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','ijlist1_len',&
                 size(ijlist1_len),intg=ijlist1_len)
         endif
      endif
      if (.not.allocated(ijlist1_len)) then
         call chmalloc('nblist_tilex.src','ns_tilex','ijlist1_len',&
              nthread,lbou=0,intg=ijlist1_len)
      endif

      if (allocated(ijlist2_len)) then
         if (size(ijlist2_len) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','ijlist2_len',&
                 size(ijlist2_len),intg=ijlist2_len)
         endif
      endif
      if (.not.allocated(ijlist2_len)) then
         call chmalloc('nblist_tilex.src','ns_tilex','ijlist2_len',&
              nthread,lbou=0,intg=ijlist2_len)
      endif

      ! n_ijlist
      if (allocated(n_ijlist1)) then
         if (size(n_ijlist1) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','n_ijlist1',&
                 size(n_ijlist1),intg=n_ijlist1)
         endif
      endif
      if (.not.allocated(n_ijlist1)) then
         call chmalloc('nblist_tilex.src','ns_tilex','n_ijlist1',&
              nthread,lbou=0,intg=n_ijlist1)
      endif

      if (allocated(n_ijlist2)) then
         if (size(n_ijlist2) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','n_ijlist2',&
                 size(n_ijlist2),intg=n_ijlist2)
         endif
      endif
      if (.not.allocated(n_ijlist2)) then
         call chmalloc('nblist_tilex.src','ns_tilex','n_ijlist2',&
              nthread,lbou=0,intg=n_ijlist2)
      endif

      ! ijlist
      call init_array(ijlist1, nthread, 3, ncell*32/nthread)
      do i=0,nthread-1
         ijlist1_len(i) = size(ijlist1(i)%array,2)
      enddo
      call init_array(ijlist2, nthread, 3, ncell*32/nthread)
      do i=0,nthread-1
         ijlist2_len(i) = size(ijlist2(i)%array,2)
      enddo

      ! cell_bb
      if (allocated(cell_bb) .and. size(cell_bb) < ncell) then
         deallocate(cell_bb)
      endif

      if (.not.allocated(cell_bb)) then
         len = int(ncell*1.5)
         allocate(cell_bb(len))
      endif

      ! Sanity check
      if (.not.allocated(nblist1)) then
         call wrndie(-5,'<nblist_tilex>','ns_tilex: nblist1 not allocated')
      endif
      if (.not.allocated(nblist2)) then
         call wrndie(-5,'<nblist_tilex>','ns_tilex: nblist2 not allocated')
      endif

      ! nblist%ientry
      ! NOTE: re-allocation done in start_ientry
      do i=0,nthread-1
         if (.not.associated(nblist1(i)%ientry)) then
            nblist1(i)%ni_max = ncoord*2/nthread
            allocate(nblist1(i)%ientry(nblist1(i)%ni_max))
         endif
      enddo
      do i=0,nthread-1
         if (.not.associated(nblist2(i)%ientry)) then
            nblist2(i)%ni_max = ncoord*2/nthread
            allocate(nblist2(i)%ientry(nblist2(i)%ni_max))
         endif
      enddo

      ! nblist%tile
      do i=0,nthread-1
         if (.not.allocated(nblist1(i)%tile)) then
            ! Estimate the number of j-atoms per i-atom
            num_j_per_i = int(0.5*4.0/3.0*pi*cut**3*real(ncoord)/(boxx*boxy*boxz)) + 1
            nblist1(i)%nj_max = int(num_j_per_i*ncoord/tilesize*1.5)
            call chmalloc('nblist_tilex.src','ns_tilex','tile',&
                 num_excl, nblist1(i)%nj_max, intg=nblist1(i)%tile)
         endif
      enddo
      do i=0,nthread-1
         if (.not.allocated(nblist2(i)%tile)) then
            ! Estimate the number of j-atoms per i-atom
            num_j_per_i = int(0.5*4.0/3.0*pi*cut**3*real(ncoord)/(boxx*boxy*boxz)) + 1
            nblist2(i)%nj_max = int(num_j_per_i*ncoord/tilesize*1.5)
            call chmalloc('nblist_tilex.src','ns_tilex','tile',&
                 num_excl, nblist2(i)%nj_max, intg=nblist2(i)%tile)
         endif
      enddo

      ! ex14_len
      if (.not.allocated(ex14_len)) then
         call chmalloc('nblist_tilex.src','ns_tilex','ex14_len',&
              nthread,lbou=0,intg=ex14_len)
      endif

      ! ex14
      call init_array(ex14, nthread, 1024, tilesize, tilesize)
      do i=0,nthread-1
         ex14_len(i) = size(ex14(i)%array,1)
      enddo

      ! sort_cell
      if (allocated(sort_cell)) then
         if (size(sort_cell) < ncell) then
            deallocate(sort_cell)
         endif
      endif

      if (.not.allocated(sort_cell)) then
         len = int(ncell*1.5)
         allocate(sort_cell(len))
      endif

      ! tile_ind_shift
      if (allocated(tile_ind_shift1)) then
         if (size(tile_ind_shift1) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','tile_ind_shift1',&
                 size(tile_ind_shift1),intg=tile_ind_shift1)
         endif
      endif
      if (.not.allocated(tile_ind_shift1)) then
         call chmalloc('nblist_tilex.src','ns_tilex','tile_ind_shift1',&
              nthread,lbou=0,intg=tile_ind_shift1)
      endif

      if (allocated(tile_ind_shift2)) then
         if (size(tile_ind_shift2) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','tile_ind_shift2',&
                 size(tile_ind_shift2),intg=tile_ind_shift2)
         endif
      endif
      if (.not.allocated(tile_ind_shift2)) then
         call chmalloc('nblist_tilex.src','ns_tilex','tile_ind_shift2',&
              nthread,lbou=0,intg=tile_ind_shift2)
      endif

      ! num_tiles_table
      if (allocated(num_tiles_table1)) then
         if (size(num_tiles_table1) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','num_tiles_table1',&
                 size(num_tiles_table1),intg=num_tiles_table1)
         endif
      endif
      if (.not.allocated(num_tiles_table1)) then
         call chmalloc('nblist_tilex.src','ns_tilex','num_tiles_table1',&
              nthread,lbou=0,intg=num_tiles_table1)
      endif

      if (allocated(num_tiles_table2)) then
         if (size(num_tiles_table2) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','num_tiles_table2',&
                 size(num_tiles_table2),intg=num_tiles_table2)
         endif
      endif
      if (.not.allocated(num_tiles_table2)) then
         call chmalloc('nblist_tilex.src','ns_tilex','num_tiles_table2',&
              nthread,lbou=0,intg=num_tiles_table2)
      endif

      ! pos_shift
      if (allocated(pos_shift1)) then
         if (size(pos_shift1) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','pos_shift1',&
                 size(pos_shift1),intg=pos_shift1)
         endif
      endif
      if (.not.allocated(pos_shift1)) then
         call chmalloc('nblist_tilex.src','ns_tilex','pos_shift1',&
              nthread,lbou=0,intg=pos_shift1)
      endif

      if (allocated(pos_shift2)) then
         if (size(pos_shift2) < nthread) then
            call chmdealloc('nblist_tilex.src','ns_tilex','pos_shift2',&
                 size(pos_shift2),intg=pos_shift2)
         endif
      endif
      if (.not.allocated(pos_shift2)) then
         call chmalloc('nblist_tilex.src','ns_tilex','pos_shift2',&
              nthread,lbou=0,intg=pos_shift2)
      endif

      return
    end subroutine alloc_realloc
    
  end subroutine ns_tilex

  ! *
  ! * Tests that all cell-cell interactions were found correctly
  ! * Uses a simple N^2 algorithm
  ! * I,FZ,FY,EX,FX,EZ,EY,C = 1,...8
  ! *
  subroutine test1_ns_tilex(ncoord, xyzq, boxx, boxy, boxz, cutsq, &
       zone_cell_start, ncell, cell_start, &
       n_ijlist_tot, ijlist_tot)
    use stream,only:outu
    use memory
    use domdec_local_types,only:xyzq_sp_t
    use parallel,only:mynod
    implicit none
    ! Input / Output
    integer, intent(in) :: ncoord
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: boxx, boxy, boxz, cutsq
    integer, intent(in) :: zone_cell_start(9), ncell, cell_start(:)
    integer, intent(in) :: n_ijlist_tot, ijlist_tot(:,:)
    ! Variables
    real(chm_real4) hboxx, hboxy, hboxz
    real(chm_real4) xi_min, yi_min, zi_min, xi_max, yi_max, zi_max
    real(chm_real4) xj_min, yj_min, zj_min, xj_max, yj_max, zj_max
    real(chm_real4) xi0, yi0, zi0, xiw, yiw, ziw
    real(chm_real4) xj0, yj0, zj0, xjw, yjw, zjw
    real(chm_real4) dx, dy, dz, rsq
    logical q_found, q_missing
    integer ncellpair, cellpair_len, nprint
    integer, allocatable, dimension(:,:) :: cellpair
    real(chm_real4), allocatable, dimension(:) :: cellrsq
    integer icell, jcell, izone, jzone, prev_izone, prev_jzone
    integer i, j, istart, iend, jstart, jend
    integer n_not_really_missing

    cellpair_len = (ncoord/tilesize)**2
    call chmalloc('nblist_tilex.src','test1_ns_tilex','cellpair',2,cellpair_len,intg=cellpair)
    call chmalloc('nblist_tilex.src','test1_ns_tilex','cellrsq',cellpair_len,cr4=cellrsq)

    hboxx = 0.5_chm_real4*boxx
    hboxy = 0.5_chm_real4*boxy
    hboxz = 0.5_chm_real4*boxz

    ncellpair = 0
    do icell=1,ncell
       prev_izone = 1
       izone = get_zone(zone_cell_start, icell, prev_izone)
       ! Only izone = I,FX,FY,FZ allowed
       if (izone /= 1 .and. izone /= 2 .and. izone /= 3 .and. izone /= 5) cycle
       istart = cell_start(icell)
       iend   = cell_start(icell + 1) - 1
       if (istart == iend + 1) cycle
       if (istart > ncoord .or. iend > ncoord) then
          call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: ncoord exceeded')
       endif
       if (iend < istart) then
          write (outu,'(a,3i10)') 'icell,istart,iend=',icell,istart,iend
          call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: iend < istart')
       endif
       xi_min = minval(xyzq(istart:iend)%x)
       yi_min = minval(xyzq(istart:iend)%y)
       zi_min = minval(xyzq(istart:iend)%z)
       xi_max = maxval(xyzq(istart:iend)%x)
       yi_max = maxval(xyzq(istart:iend)%y)
       zi_max = maxval(xyzq(istart:iend)%z)
       xi0 = 0.5_chm_real4*(xi_min + xi_max)
       yi0 = 0.5_chm_real4*(yi_min + yi_max)
       zi0 = 0.5_chm_real4*(zi_min + zi_max)
       xiw = xi_max - xi0
       yiw = yi_max - yi0
       ziw = zi_max - zi0
       do jcell=1,ncell
          prev_jzone = 1
          jzone = get_zone(zone_cell_start, jcell, prev_jzone)

          if (izone == 1 .and. izone == jzone .and. icell > jcell) cycle

          if (izone == 2 .and. jzone /= 6) cycle

          if (izone == 3 .and. jzone /= 2 .and. jzone /= 7) cycle

          if (izone == 5 .and. jzone /= 2 .and. jzone /= 3 .and. jzone /= 4) cycle

          jstart = cell_start(jcell)
          jend   = cell_start(jcell + 1) - 1

          if (jstart == jend + 1) cycle

          xj_min = minval(xyzq(jstart:jend)%x)
          yj_min = minval(xyzq(jstart:jend)%y)
          zj_min = minval(xyzq(jstart:jend)%z)
          xj_max = maxval(xyzq(jstart:jend)%x)
          yj_max = maxval(xyzq(jstart:jend)%y)
          zj_max = maxval(xyzq(jstart:jend)%z)

          xj0 = 0.5_chm_real4*(xj_min + xj_max)
          yj0 = 0.5_chm_real4*(yj_min + yj_max)
          zj0 = 0.5_chm_real4*(zj_min + zj_max)
          xjw = xj_max - xj0
          yjw = yj_max - yj0
          zjw = zj_max - zj0

          dx = abs(xi0 - xj0)
          dy = abs(yi0 - yj0)
          dz = abs(zi0 - zj0)
          do while (dx > hboxx)
             dx = dx - boxx
          enddo
          do while (dy > hboxy)
             dy = dy - boxy
          enddo
          do while (dz > hboxz)
             dz = dz - boxz
          enddo
          dx = max(0.0_chm_real4, abs(dx) - xiw - xjw)
          dy = max(0.0_chm_real4, abs(dy) - yiw - yjw)
          dz = max(0.0_chm_real4, abs(dz) - ziw - zjw)
          rsq = dx*dx + dy*dy + dz*dz

          if (rsq < cutsq) then
             ncellpair = ncellpair + 1
             if (ncellpair > cellpair_len) then
                cellpair_len = int(ncellpair*1.5)
                call chmrealloc('nblist_tilex.src','test1_ns_tilex','cellpair',&
                     2,cellpair_len,intg=cellpair)
                call chmrealloc('nblist_tilex.src','test1_ns_tilex','cellrsq',&
                     cellpair_len,cr4=cellrsq)
             endif
             cellpair(1:2,ncellpair) = (/ icell, jcell /)
             cellrsq(ncellpair) = rsq
          endif

       enddo
    enddo

    if (ncellpair /= n_ijlist_tot) then
       write (outu,'(a,2i10)') 'ncellpair,n_ijlist_tot=',ncellpair,n_ijlist_tot
    endif

    ! Loop through GPU cellpairs, look for missing CPU pairs
    nprint = 0
    q_missing = .false.
    do i=1,n_ijlist_tot
       icell = ijlist_tot(1,i)
       jcell = ijlist_tot(3,i)
       q_found = .false.
       do j=1,ncellpair
          if ((icell == cellpair(1,j) .and. jcell == cellpair(2,j)) .or. &
               (icell == cellpair(2,j) .and. jcell == cellpair(1,j))) then
             q_found = .true.
             exit
          endif
       enddo
       if (.not.q_found) then
          q_missing = .true.
          if (nprint <= 10) then
             nprint = nprint + 1
             prev_jzone = 1
             izone = get_zone(zone_cell_start, icell, prev_jzone)
             prev_jzone = 1
             jzone = get_zone(zone_cell_start, jcell, prev_jzone)
             write (outu,'(a,3i8,3i3)') 'missing(1): i,icell,jcell,izone,jzone,mynod=',&
                  i,icell,jcell,izone,jzone,mynod
          else
             exit
          endif
       endif
    enddo
    if (q_missing) then
       call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: Missing cell-cell pairs')
    endif

    ! Loop through CPU cellpairs, look for missing GPU pairs
    nprint = 0
    n_not_really_missing = 0
    q_missing = .false.
    do i=1,ncellpair
       icell = cellpair(1,i)
       jcell = cellpair(2,i)
       q_found = .false.
       do j=1,n_ijlist_tot
          if ((icell == ijlist_tot(1,j) .and. jcell == ijlist_tot(3,j)) .or. &
               (icell == ijlist_tot(3,j) .and. jcell == ijlist_tot(1,j))) then
             q_found = .true.
             exit
          endif
       enddo
       if (.not.q_found) then
          if (abs(cellrsq(i)-cutsq) > 5.0e-5_chm_real4) then
             q_missing = .true.
             if (nprint <= 10) then
                nprint = nprint + 1
                prev_jzone = 1
                izone = get_zone(zone_cell_start, icell, prev_jzone)
                prev_jzone = 1
                jzone = get_zone(zone_cell_start, jcell, prev_jzone)
                write (outu,'(a,3i8,3i3)') 'missing(2): i,icell,jcell,izone,jzone,mynod=',&
                     i,icell,jcell,izone,jzone,mynod
                write (outu,'(a,f14.8,g12.3)') 'rsq=',cellrsq(i),abs(cellrsq(i)-cutsq)
             else
                exit
             endif
          else
             ! Not really missing
             n_not_really_missing = n_not_really_missing + 1
          endif
       endif
    enddo

    call chmdealloc('nblist_tilex.src','test1_ns_tilex','cellpair',2,size(cellpair,2),intg=cellpair)
    call chmdealloc('nblist_tilex.src','test1_ns_tilex','cellrsq',size(cellrsq),cr4=cellrsq)

    if (q_missing) then
       call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: Missing cell-cell pairs')
    endif

    ! Too many GPU pairs
    if (ncellpair < n_ijlist_tot) then
       call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: Invalid number of cell-cell pairs (1)')
    endif

    ! Missing GPU pairs
    if (ncellpair > n_ijlist_tot + n_not_really_missing) then
       call wrndie(-5,'<nblist_tilex>','test1_ns_tilex: Invalid number of cell-cell pairs (2)')
    endif

    write (outu,'(a)') 'test1_ns_tilex OK'

    return
  end subroutine test1_ns_tilex

  ! *
  ! * Returns zone index for cell icell assuming we are traversing cells in increasing order
  ! *
  integer function get_zone(zone_cell_start, icell, prev_zone)
    ! Input
    integer, intent(in) :: zone_cell_start(9)
    integer, intent(in) :: icell
    integer, intent(inout) :: prev_zone

    get_zone = prev_zone
    
    do while (icell >= zone_cell_start(get_zone+1))
       get_zone = get_zone + 1
    enddo
    
    return
  end function get_zone

  ! *
  ! * Neighbor search loop kernel: I vs. I
  ! *
  subroutine ns_tilex_kernel1(ncell, ncellx, ncelly, ncellz, &
       cell_start, icellxyz, startcol_zone, max_ncellz, startcell_col, &
       cellbx, cellby, cellbz, &
       boxx, boxy, boxz, cut, cutsq, &
       imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi, &
       xdist, ydist, zdist, &
       jlist, nblist, ijlist, n_ijlist, ijlist_len)
    use nblist_types,only:nblist_tilex_t
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: ncell, ncellx(8), ncelly(8), ncellz(:)
    integer, intent(in) :: cell_start(:), icellxyz(:,:), startcol_zone(8)
    integer, intent(in) :: max_ncellz(8), startcell_col(:)
    type(cr4array_t), intent(in) :: cellbx(8), cellby(8), cellbz(8)
    real(chm_real4), intent(in) :: boxx, boxy, boxz, cut, cutsq
    integer, intent(in) :: imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    real(chm_real4), intent(inout) :: xdist(:), ydist(:), zdist(:)
    integer, allocatable, dimension(:), intent(inout) :: jlist
    type(nblist_tilex_t), intent(inout) :: nblist
    integer, allocatable, dimension(:,:), intent(inout) :: ijlist
    integer, intent(inout) :: n_ijlist, ijlist_len
    ! Variables
    real(chm_real4) ibbx0, ibby0, ibbz0, ibbxw, ibbyw, ibbzw
    real(chm_real4) imbbx0, imbby0, imbbz0
    real(chm_real4) celldist1, celldist2
    real(chm_real4) jbbx0, jbby0, jbbz0, jbbxw, jbbyw, jbbzw
    real(chm_real4) bbxdist, bbydist, bbzdist
    integer imx, imy, imz
    integer jcellx0(8), jcellx1(8), jcelly0(8), jcelly1(8), jcellz0, jcellz1
    integer jcellx, jcelly, jcellz, jcell
    integer icellx, icelly, icellz, icell
    integer ish, pos_xy, pos_cellbz, pos_ncellz
    integer n_jlist, jlist_len
    integer i
    integer n_jcellx, n_jcelly, icellz_im
    integer xdist_pos, ydist_pos
    integer jcellx0_t

    ! Clear list counters
    nblist%ni = 0
    nblist%nj = 0

    n_ijlist = 0
    jlist_len = size(jlist)

!$omp do schedule(dynamic)
    do icell=1,ncell

       icellx = icellxyz(1,icell)
       icelly = icellxyz(2,icell)
       icellz = icellxyz(3,icell)

       ! Read bounding box:
       ibbx0 = cell_bb(icell)%x0
       ibby0 = cell_bb(icell)%y0
       ibbz0 = cell_bb(icell)%z0
       ibbxw = cell_bb(icell)%xw
       ibbyw = cell_bb(icell)%yw
       ibbzw = cell_bb(icell)%zw

       ! cell index is at: icellx, icelly, icellz

       ! Loop over images, max. number of images = 3*3*3=27.
       ! X
       do imx=imx_lo,imx_hi
          imbbx0 = ibbx0 + imx*boxx
          n_jcellx = 0
          xdist_pos = 1
          call get_cell_bounds1(icellx + imx*ncellx(1), &
               ncellx(1), imbbx0-ibbxw, imbbx0+ibbxw, cellbx(1)%array, &
               cut, jcellx0(1), jcellx1(1), xdist(xdist_pos:))
          n_jcellx = n_jcellx + max(0, jcellx1(1) - jcellx0(1) + 1)
          xdist_pos = xdist_pos + ncellx(1)
          if (n_jcellx == 0) cycle
          ! Y
          do imy=imy_lo,imy_hi
             imbby0 = ibby0 + imy*boxy
             n_jcelly = 0
             ydist_pos = 1
             call get_cell_bounds1(icelly + imy*ncelly(1), &
                  ncelly(1), imbby0-ibbyw, imbby0+ibbyw, cellby(1)%array, &
                  cut, jcelly0(1), jcelly1(1), ydist(ydist_pos:))
             jcelly0(1) = max(icelly, jcelly0(1))
             n_jcelly = n_jcelly + max(0,jcelly1(1) - jcelly0(1) + 1)
             ydist_pos = ydist_pos + ncelly(1)
             if (n_jcelly == 0) cycle
             ! Z
             do imz=imz_lo,imz_hi
                imbbz0 = ibbz0 + imz*boxz
                
                ! Clear j-list
                n_jlist = 0

                ! Compute shift index: 1...26*3+1
                ish = imx+1 + 3*(imy+1 + 3*(imz+1))

                xdist_pos = 0
                ydist_pos = 0

                if (jcelly1(1) >= jcelly0(1) .and. jcellx1(1) >= jcellx0(1)) then
                   ! Loop over j-cells
                   ! NOTE: we do this in order y, x, z so that the resulting tile list
                   !       is ordered
                   do jcelly=jcelly0(1),jcelly1(1)
                      celldist1 = ydist(ydist_pos + jcelly)**2
                      if (icelly == jcelly) then
                         jcellx0_t = max(icellx, jcellx0(1))
                      else
                         jcellx0_t = jcellx0(1)
                      endif
                      do jcellx=jcellx0_t,jcellx1(1)
                         celldist2 = celldist1 + xdist(xdist_pos + jcellx)**2
                         if (celldist2 > cutsq) cycle
                         ! Get jcellz limits (jcellz0, jcellz1)
                         pos_xy = jcellx + (jcelly-1)*ncellx(1)
                         pos_cellbz = (max_ncellz(1)+1)*(pos_xy - 1)
                         pos_ncellz = pos_xy + startcol_zone(1)
                         if (ncellz(pos_ncellz) == 0) cycle
                         ! NOTE: icellz_im is only used in "get_cell_bounds" when 
                         !       izone == jzone, therefore, no value need to be set when
                         !       izone /= jzone
                         icellz_im = icellz + imz*ncellz(pos_ncellz)
                         call get_cell_bounds1(icellz_im, &
                              ncellz(pos_ncellz), imbbz0-ibbzw, imbbz0+ibbzw, &
                              cellbz(1)%array(pos_cellbz:), cut, jcellz0, jcellz1, zdist)
                         if (icellx == jcellx .and. icelly == jcelly) then
                            ! icell and jcell are within zone I and within the same xy-column
                            ! In this case, we can restrict jcellz0 >= icellz
                            jcellz0 = max(icellz, jcellz0)
                            !jcellz0 = max(icell - startcell_col(pos_ncellz), jcellz0)
                         endif
                         do jcellz=jcellz0,jcellz1
                            if (celldist2 + zdist(jcellz)**2 > cutsq) cycle
                            ! j-cell index is calculated as jcellz + start of the column cells
                            jcell = jcellz + startcell_col(pos_ncellz)
                            
                            if (icell > jcell) cycle
                            
                            ! Read bounding box for j-cell
                            jbbx0 = cell_bb(jcell)%x0
                            jbby0 = cell_bb(jcell)%y0
                            jbbz0 = cell_bb(jcell)%z0
                            jbbxw = cell_bb(jcell)%xw
                            jbbyw = cell_bb(jcell)%yw
                            jbbzw = cell_bb(jcell)%zw
                            
                            ! Calculate distance between i- and j-cell bounding boxes
                            bbxdist = max(0.0_chm_real4, abs(imbbx0 - jbbx0) - ibbxw - jbbxw)
                            bbydist = max(0.0_chm_real4, abs(imbby0 - jbby0) - ibbyw - jbbyw)
                            bbzdist = max(0.0_chm_real4, abs(imbbz0 - jbbz0) - ibbzw - jbbzw)
                            
                            if (bbxdist**2 + bbydist**2 + bbzdist**2 < cutsq) then
                               ! At least one atom pair is within the cutoff =>
                               ! add jcell to the neighbor list
                               n_jlist = n_jlist + 1
                               if (n_jlist > jlist_len) then
                                  jlist_len = int(1.5*n_jlist)
                                  call chmrealloc('nblist_tilex.src','ns_tilex_kernel1',&
                                       'jlist',jlist_len,intg=jlist)
                               endif
                               jlist(n_jlist) = jcell
                            endif
                            
                         enddo ! jcellz=jcellz0,jcellz1
                      enddo ! jcelly=jcelly0,jcelly1
                   enddo ! jcellx=jcellx0,jcellx1
                endif
                xdist_pos = xdist_pos + ncellx(1)
                ydist_pos = ydist_pos + ncelly(1)

                ! Flush jlist
                if (n_jlist > 0) then
                   call flush_to_ijlist(nblist, ijlist, n_ijlist, &
                        ijlist_len, icell, jlist, n_jlist, ish, cell_start)
                endif

             enddo ! imz=imz_lo,imz_hi
          enddo ! imy=imy_lo,imy_hi
       enddo ! imx=imx_lo,imx_hi
       
    enddo ! do icell=1,ncell
!$omp end do

    return
  end subroutine ns_tilex_kernel1

  ! *
  ! * Neighbor search loop kernel: Rest
  ! *
  subroutine ns_tilex_kernel2(icell_start, icell_stop, ncellx, ncelly, ncellz, &
       cell_start, icellxyz, zone_cell_start, startcol_zone, max_ncellz, startcell_col, &
       cellbx, cellby, cellbz, &
       n_int_zone, int_zone, &
       boxx, boxy, boxz, cut, cutsq, &
       imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi, &
       xdist, ydist, zdist, &
       jlist, nblist, ijlist, n_ijlist, ijlist_len)
    use nblist_types,only:nblist_tilex_t
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: icell_start, icell_stop, ncellx(8), ncelly(8), ncellz(:)
    integer, intent(in) :: cell_start(:), icellxyz(:,:), zone_cell_start(9), startcol_zone(8)
    integer, intent(in) :: max_ncellz(8), startcell_col(:)
    type(cr4array_t), intent(in) :: cellbx(8), cellby(8), cellbz(8)
    integer, intent(in) :: n_int_zone(8), int_zone(8,8)
    real(chm_real4), intent(in) :: boxx, boxy, boxz, cut, cutsq
    integer, intent(in) :: imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    real(chm_real4), intent(inout) :: xdist(:), ydist(:), zdist(:)
    integer, allocatable, dimension(:), intent(inout) :: jlist
    type(nblist_tilex_t), intent(inout) :: nblist
    integer, allocatable, dimension(:,:), intent(inout) :: ijlist
    integer, intent(inout) :: n_ijlist, ijlist_len
    ! Variables
    real(chm_real4) ibbx0, ibby0, ibbz0, ibbxw, ibbyw, ibbzw
    real(chm_real4) imbbx0, imbby0, imbbz0
    real(chm_real4) celldist1, celldist2
    real(chm_real4) jbbx0, jbby0, jbbz0, jbbxw, jbbyw, jbbzw
    real(chm_real4) bbxdist, bbydist, bbzdist
    integer imx, imy, imz
    integer jcellx0(8), jcellx1(8), jcelly0(8), jcelly1(8), jcellz0, jcellz1
    integer jcellx, jcelly, jcellz, jcell
    integer icellx, icelly, icellz, icell
    integer ish, pos_xy, pos_cellbz, pos_ncellz
    integer n_jlist, jlist_len
    integer i, prev_zone
    integer izone, n_jzone, jjzone, jzone, n_jcellx, n_jcelly, icellz_im
    integer xdist_pos, ydist_pos
    integer jcellx0_t

    ! Clear list counters
    nblist%ni = 0
    nblist%nj = 0

    n_ijlist = 0
    jlist_len = size(jlist)

    prev_zone = 1

!$omp do schedule(dynamic)
    do icell=icell_start,icell_stop

       ! Get zone index for this cell
       izone = get_zone(zone_cell_start, icell, prev_zone)

       ! Get number of zones this zone interacts with
       ! NOTE: the interacting zones are given by int_zone(1:nzone,izone)
       n_jzone = n_int_zone(izone)
       if (n_jzone == 0) cycle

       icellx = icellxyz(1,icell)
       icelly = icellxyz(2,icell)
       icellz = icellxyz(3,icell)

       ! Read bounding box:
       ibbx0 = cell_bb(icell)%x0
       ibby0 = cell_bb(icell)%y0
       ibbz0 = cell_bb(icell)%z0
       ibbxw = cell_bb(icell)%xw
       ibbyw = cell_bb(icell)%yw
       ibbzw = cell_bb(icell)%zw

       ! cell index is at: icellx, icelly, icellz

       ! Loop over images, max. number of images = 3*3*3=27.
       ! X
       do imx=imx_lo,imx_hi
          imbbx0 = ibbx0 + imx*boxx
          n_jcellx = 0
          xdist_pos = 1
          do jjzone=1,n_jzone
             jzone = int_zone(jjzone,izone)
             call get_cell_bounds2(izone, jzone, icellx + imx*ncellx(izone), &
                  ncellx(jzone), imbbx0-ibbxw, imbbx0+ibbxw, cellbx(jzone)%array, &
                  cut, jcellx0(jzone), jcellx1(jzone), xdist(xdist_pos:))
             n_jcellx = n_jcellx + max(0, jcellx1(jzone) - jcellx0(jzone) + 1)
             xdist_pos = xdist_pos + ncellx(jzone)
          enddo
          if (n_jcellx == 0) cycle
          ! Y
          do imy=imy_lo,imy_hi
             imbby0 = ibby0 + imy*boxy
             n_jcelly = 0
             ydist_pos = 1
             do jjzone=1,n_jzone
                jzone = int_zone(jjzone,izone)
                call get_cell_bounds2(izone, jzone, icelly + imy*ncelly(izone), &
                     ncelly(jzone), imbby0-ibbyw, imbby0+ibbyw, cellby(jzone)%array, &
                     cut, jcelly0(jzone), jcelly1(jzone), ydist(ydist_pos:))
                n_jcelly = n_jcelly + max(0,jcelly1(jzone) - jcelly0(jzone) + 1)
                ydist_pos = ydist_pos + ncelly(jzone)
             enddo
             if (n_jcelly == 0) cycle
             ! Z
             do imz=imz_lo,imz_hi
                imbbz0 = ibbz0 + imz*boxz

                ! Clear j-list
                n_jlist = 0

                ! Compute shift index: 1...26*3+1
                ish = imx+1 + 3*(imy+1 + 3*(imz+1))

                xdist_pos = 0
                ydist_pos = 0
                ! Loop over zones
                do jjzone=1,n_jzone
                   jzone = int_zone(jjzone,izone)
                   if (jcelly1(jzone) >= jcelly0(jzone) .and. jcellx1(jzone) >= jcellx0(jzone)) then
                      ! Loop over j-cells
                      ! NOTE: we do this in order y, x, z so that the resulting tile list
                      !       is ordered
                      do jcelly=jcelly0(jzone),jcelly1(jzone)
                         celldist1 = ydist(ydist_pos + jcelly)**2
                         jcellx0_t = jcellx0(jzone)
                         do jcellx=jcellx0_t,jcellx1(jzone)
                            celldist2 = celldist1 + xdist(xdist_pos + jcellx)**2
                            if (celldist2 > cutsq) cycle
                            ! Get jcellz limits (jcellz0, jcellz1)
                            pos_xy = jcellx + (jcelly-1)*ncellx(jzone)
                            pos_cellbz = (max_ncellz(jzone)+1)*(pos_xy - 1)
                            pos_ncellz = pos_xy + startcol_zone(jzone)
                            if (ncellz(pos_ncellz) == 0) cycle
                            call get_cell_bounds2(izone, jzone, icellz_im, &
                                 ncellz(pos_ncellz), imbbz0-ibbzw, imbbz0+ibbzw, &
                                 cellbz(jzone)%array(pos_cellbz:), cut, jcellz0, jcellz1, zdist)
                            do jcellz=jcellz0,jcellz1
                               if (celldist2 + zdist(jcellz)**2 > cutsq) cycle
                               ! j-cell index is calculated as jcellz + start of the column cells
                               jcell = jcellz + startcell_col(pos_ncellz)

                               ! Read bounding box for j-cell
                               jbbx0 = cell_bb(jcell)%x0
                               jbby0 = cell_bb(jcell)%y0
                               jbbz0 = cell_bb(jcell)%z0
                               jbbxw = cell_bb(jcell)%xw
                               jbbyw = cell_bb(jcell)%yw
                               jbbzw = cell_bb(jcell)%zw
                               
                               ! Calculate distance between i- and j-cell bounding boxes
                               bbxdist = max(0.0_chm_real4, abs(imbbx0 - jbbx0) - ibbxw - jbbxw)
                               bbydist = max(0.0_chm_real4, abs(imbby0 - jbby0) - ibbyw - jbbyw)
                               bbzdist = max(0.0_chm_real4, abs(imbbz0 - jbbz0) - ibbzw - jbbzw)

                               if (bbxdist**2 + bbydist**2 + bbzdist**2 < cutsq) then
                                  ! At least one atom pair is within the cutoff =>
                                  ! add jcell to the neighbor list
                                  n_jlist = n_jlist + 1
                                  if (n_jlist > jlist_len) then
                                     jlist_len = int(1.5*n_jlist)
                                     call chmrealloc('nblist_tilex.src','ns_tilex_kernel2',&
                                          'jlist',jlist_len,intg=jlist)
                                  endif
                                  jlist(n_jlist) = jcell
                               endif
                               
                            enddo ! jcellz=jcellz0,jcellz1
                         enddo ! jcelly=jcelly0,jcelly1
                      enddo ! jcellx=jcellx0,jcellx1
                   endif
                   xdist_pos = xdist_pos + ncellx(jzone)
                   ydist_pos = ydist_pos + ncelly(jzone)
                enddo ! do jjzone=1,n_jzone

                ! Flush jlist
                if (n_jlist > 0) then
                   call flush_to_ijlist(nblist, ijlist, n_ijlist, &
                        ijlist_len, icell, jlist, n_jlist, ish, cell_start)
                endif

             enddo ! imz=imz_lo,imz_hi
          enddo ! imy=imy_lo,imy_hi
       enddo ! imx=imx_lo,imx_hi
       
    enddo ! do icell=1,ncell
!$omp end do

    return
  end subroutine ns_tilex_kernel2

  ! ------------------------------- OLD KERNEL ------------------------------------------

  ! *
  ! * Computes cell bounds (icell0, icell1)
  ! * bx(0:ncell) is the right cell boundary, for x (and y) direction:
  ! * bx(0) = 0
  ! * bx(1) = dx
  ! * bx(2) = dx*2
  ! *
  ! * Returns:
  ! * jcell0:jcell1 = range of cells within cut-off
  ! * dist          = distance for each cell
  ! *
  subroutine get_cell_bounds(izone, jzone, icell, ncell, x0, x1, bx, cut, jcell0, jcell1, dist)
    implicit none
    ! Input / Output
    integer, intent(in) :: izone, jzone, icell, ncell
    real(chm_real4), intent(in) :: x0, x1, bx(0:*), cut
    integer, intent(out) :: jcell0, jcell1
    real(chm_real4), intent(out) :: dist(:)
    ! Variables
    real(chm_real4) d
    integer jcell_start_left, jcell_start_right
    integer j

    if (izone == jzone) then
       ! Search within a single zone (I)
       if (icell < 1) then
          ! This is one of the image cells on the left =>
          ! set the left cell boundary (jcell0) to 1 and start looking for the right
          ! boundary from 1
          jcell_start_left = 0          ! with this value, we don't look for cells on the left
          jcell_start_right = 1         ! start looking for cells at right from 1
          jcell0 = 1                    ! left boundary set to minimum value
          jcell1 = 0                    ! set to "no cells" value
          dist(1) = 0.0_chm_real4
       elseif (icell > ncell) then
          ! This is one of the image cells on the right =>
          ! set the right cell boundary (icell1) to ncell and start looking for the left
          ! boundary from ncell
          jcell_start_left = ncell      ! start looking for cells at left from ncell
          jcell_start_right = ncell + 1 ! with this value, we don't look for cells on the right
          jcell0 = ncell + 1            ! set to "no cells" value
          jcell1 = ncell                ! right boundary set to maximum value
          dist(ncell) = 0.0_chm_real4
       else
          jcell_start_left = icell - 1
          jcell_start_right = icell + 1
          jcell0 = icell
          jcell1 = icell
          dist(icell) = 0.0_chm_real4
       endif
    else
       ! Search between two zones
       if (bx(0) >= x1 .or. (bx(0) < x1 .and. bx(0) > x0)) then
          ! j-zone is to the right of i-zone
          ! => no left search, start right search from 1
          jcell_start_left = 0
          jcell_start_right = 1
          jcell0 = 1
          jcell1 = 0
       elseif (bx(ncell) <= x0 .or. (bx(ncell) > x0 .and. bx(ncell) < x1)) then
          ! j-zone is to the left of i-zone
          ! => no right search, start left search from ncell
          jcell_start_left = ncell
          jcell_start_right = ncell + 1
          jcell0 = ncell + 1
          jcell1 = ncell
       else
          ! i-zone is between j-zones
          ! => safe choice is to search the entire range
          jcell_start_left = ncell
          jcell_start_right = 1
          jcell0 = ncell
          jcell1 = 1
       endif

    endif

    ! Check cells at left, stop once the distance to the cell right boundary 
    ! is greater than the cutoff.
    !
    ! Cell right boundary is at bx(i)
    do j=jcell_start_left,1,-1
       d = x0 - bx(j)
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell0 = j
    enddo

    ! Check cells at right, stop once the distance to the cell left boundary
    ! is greater than the cutoff.
    !
    ! Cell left boundary is at bx(i-1)
    
    do j=jcell_start_right,ncell
       d = bx(j-1) - x1
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell1 = j
    enddo

    ! Cell bounds are jcell0:jcell1

    return
  end subroutine get_cell_bounds

  ! *
  ! * Neighbor search loop kernel
  ! *
  subroutine ns_tilex_kernel(ncell, ncellx, ncelly, ncellz, &
       cell_start, icellxyz, zone_cell_start, startcol_zone, max_ncellz, startcell_col, &
       cellbx, cellby, cellbz, &
       n_int_zone, int_zone, &
       boxx, boxy, boxz, cut, cutsq, &
       imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi, &
       xdist, ydist, zdist, &
       jlist1, nblist, ijlist, n_ijlist, ijlist_len)
    use nblist_types,only:nblist_tilex_t
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: ncell, ncellx(8), ncelly(8), ncellz(:)
    integer, intent(in) :: cell_start(:), icellxyz(:,:), zone_cell_start(9), startcol_zone(8)
    integer, intent(in) :: max_ncellz(8), startcell_col(:)
    type(cr4array_t), intent(in) :: cellbx(8), cellby(8), cellbz(8)
    integer, intent(in) :: n_int_zone(8), int_zone(8,8)
    real(chm_real4), intent(in) :: boxx, boxy, boxz, cut, cutsq
    integer, intent(in) :: imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    real(chm_real4), intent(inout) :: xdist(:), ydist(:), zdist(:)
    integer, allocatable, dimension(:), intent(inout) :: jlist1
    type(nblist_tilex_t), intent(inout) :: nblist
    integer, allocatable, dimension(:,:), intent(inout) :: ijlist
    integer, intent(inout) :: n_ijlist, ijlist_len
    ! Variables
    real(chm_real4) ibbx0, ibby0, ibbz0, ibbxw, ibbyw, ibbzw
    real(chm_real4) imbbx0, imbby0, imbbz0
    real(chm_real4) celldist1, celldist2
    real(chm_real4) jbbx0, jbby0, jbbz0, jbbxw, jbbyw, jbbzw
    real(chm_real4) bbxdist, bbydist, bbzdist
    integer imx, imy, imz
    integer jcellx0(8), jcellx1(8), jcelly0(8), jcelly1(8), jcellz0, jcellz1
    integer jcellx, jcelly, jcellz, jcell
    integer icellx, icelly, icellz, icell
    integer ish, pos_xy, pos_cellbz, pos_ncellz
    integer n_jlist1, jlist1_len
    integer i, prev_zone
    integer izone, n_jzone, jjzone, jzone, n_jcellx, n_jcelly, icellz_im
    integer xdist_pos, ydist_pos
    integer jcellx0_t

    ! Clear list counters
    nblist%ni = 0
    nblist%nj = 0

    n_ijlist = 0
    jlist1_len = size(jlist1)

    prev_zone = 1

!$omp do schedule(dynamic)
    do icell=1,ncell

       ! Get zone index for this cell
       izone = get_zone(zone_cell_start, icell, prev_zone)

       ! Get number of zones this zone interacts with
       ! NOTE: the interacting zones are given by int_zone(1:nzone,izone)
       n_jzone = n_int_zone(izone)
       if (n_jzone == 0) cycle

       icellx = icellxyz(1,icell)
       icelly = icellxyz(2,icell)
       icellz = icellxyz(3,icell)

       ! Read bounding box:
       ibbx0 = cell_bb(icell)%x0
       ibby0 = cell_bb(icell)%y0
       ibbz0 = cell_bb(icell)%z0
       ibbxw = cell_bb(icell)%xw
       ibbyw = cell_bb(icell)%yw
       ibbzw = cell_bb(icell)%zw

       ! cell index is at: icellx, icelly, icellz

       ! Loop over images, max. number of images = 3*3*3=27.
       ! X
       do imx=imx_lo,imx_hi
          imbbx0 = ibbx0 + imx*boxx
          n_jcellx = 0
          xdist_pos = 1
          do jjzone=1,n_jzone
             jzone = int_zone(jjzone,izone)
             call get_cell_bounds(izone, jzone, icellx + imx*ncellx(izone), &
                  ncellx(jzone), imbbx0-ibbxw, imbbx0+ibbxw, cellbx(jzone)%array, &
                  cut, jcellx0(jzone), jcellx1(jzone), xdist(xdist_pos:))
             n_jcellx = n_jcellx + max(0, jcellx1(jzone) - jcellx0(jzone) + 1)
             xdist_pos = xdist_pos + ncellx(jzone)
          enddo
          if (n_jcellx == 0) cycle
          ! Y
          do imy=imy_lo,imy_hi
             imbby0 = ibby0 + imy*boxy
             n_jcelly = 0
             ydist_pos = 1
             do jjzone=1,n_jzone
                jzone = int_zone(jjzone,izone)
                call get_cell_bounds(izone, jzone, icelly + imy*ncelly(izone), &
                     ncelly(jzone), imbby0-ibbyw, imbby0+ibbyw, cellby(jzone)%array, &
                     cut, jcelly0(jzone), jcelly1(jzone), ydist(ydist_pos:))
                if (izone == jzone) jcelly0(jzone) = max(icelly, jcelly0(jzone))
                n_jcelly = n_jcelly + max(0,jcelly1(jzone) - jcelly0(jzone) + 1)
                ydist_pos = ydist_pos + ncelly(jzone)
             enddo
             if (n_jcelly == 0) cycle
             ! Z
             do imz=imz_lo,imz_hi
                imbbz0 = ibbz0 + imz*boxz

                ! Clear j-list
                n_jlist1 = 0

                ! Compute shift index: 1...26*3+1
                ish = imx+1 + 3*(imy+1 + 3*(imz+1))

                xdist_pos = 0
                ydist_pos = 0
                ! Loop over zones
                do jjzone=1,n_jzone
                   jzone = int_zone(jjzone,izone)
                   if (jcelly1(jzone) >= jcelly0(jzone) .and. jcellx1(jzone) >= jcellx0(jzone)) then
                      ! Loop over j-cells
                      ! NOTE: we do this in order y, x, z so that the resulting tile list
                      !       is ordered
                      do jcelly=jcelly0(jzone),jcelly1(jzone)
                         celldist1 = ydist(ydist_pos + jcelly)**2
                         if (izone == jzone .and. icelly == jcelly) then
                            jcellx0_t = max(icellx, jcellx0(jzone))
                         else
                            jcellx0_t = jcellx0(jzone)
                         endif
                         do jcellx=jcellx0_t,jcellx1(jzone)
                            celldist2 = celldist1 + xdist(xdist_pos + jcellx)**2
                            if (celldist2 > cutsq) cycle
                            ! Get jcellz limits (jcellz0, jcellz1)
                            pos_xy = jcellx + (jcelly-1)*ncellx(jzone)
                            pos_cellbz = (max_ncellz(jzone)+1)*(pos_xy - 1)
                            pos_ncellz = pos_xy + startcol_zone(jzone)
                            if (izone == jzone) then
                               ! NOTE: icellz_im is only used in "get_cell_bounds" when 
                               !       izone == jzone, therefore, no value need to be set when
                               !       izone /= jzone
                               icellz_im = icellz + imz*ncellz(pos_ncellz)
                            endif
                            call get_cell_bounds(izone, jzone, icellz_im, &
                                 ncellz(pos_ncellz), imbbz0-ibbzw, imbbz0+ibbzw, &
                                 cellbz(jzone)%array(pos_cellbz:), cut, jcellz0, jcellz1, zdist)
                            if (izone == jzone .and. icellx == jcellx .and. icelly == jcelly) then
                               ! icell and jcell are within zone I and within the same xy-column
                               ! In this case, we can restrict jcellz0 >= icellz
                               jcellz0 = max(icellz, jcellz0)
                               !jcellz0 = max(icell - startcell_col(pos_ncellz), jcellz0)
                            endif
                            do jcellz=jcellz0,jcellz1
                               if (celldist2 + zdist(jcellz)**2 > cutsq) cycle
                               ! j-cell index is calculated as jcellz + start of the column cells
                               jcell = jcellz + startcell_col(pos_ncellz)
                               
                               if (izone == jzone .and. icell > jcell) cycle
                               
                               ! Read bounding box for j-cell
                               jbbx0 = cell_bb(jcell)%x0
                               jbby0 = cell_bb(jcell)%y0
                               jbbz0 = cell_bb(jcell)%z0
                               jbbxw = cell_bb(jcell)%xw
                               jbbyw = cell_bb(jcell)%yw
                               jbbzw = cell_bb(jcell)%zw
                               
                               ! Calculate distance between i- and j-cell bounding boxes
                               bbxdist = max(0.0_chm_real4, abs(imbbx0 - jbbx0) - ibbxw - jbbxw)
                               bbydist = max(0.0_chm_real4, abs(imbby0 - jbby0) - ibbyw - jbbyw)
                               bbzdist = max(0.0_chm_real4, abs(imbbz0 - jbbz0) - ibbzw - jbbzw)

                               if (bbxdist**2 + bbydist**2 + bbzdist**2 < cutsq) then
                                  ! At least one atom pair is within the cutoff =>
                                  ! add jcell to the neighbor list
                                  n_jlist1 = n_jlist1 + 1
                                  if (n_jlist1 > jlist1_len) then
                                     jlist1_len = int(1.5*n_jlist1)
                                     call chmrealloc('nblist_tilex.src','ns_tilex_kernel',&
                                          'jlist1',jlist1_len,intg=jlist1)
                                  endif
                                  jlist1(n_jlist1) = jcell
                               endif
                               
                            enddo ! jcellz=jcellz0,jcellz1
                         enddo ! jcelly=jcelly0,jcelly1
                      enddo ! jcellx=jcellx0,jcellx1
                   endif
                   xdist_pos = xdist_pos + ncellx(jzone)
                   ydist_pos = ydist_pos + ncelly(jzone)
                enddo ! do jjzone=1,n_jzone

                ! Flush jlist1
                if (n_jlist1 > 0) then
                   call flush_to_ijlist(nblist, ijlist, n_ijlist, &
                        ijlist_len, icell, jlist1, n_jlist1, ish, cell_start)
                endif

             enddo ! imz=imz_lo,imz_hi
          enddo ! imy=imy_lo,imy_hi
       enddo ! imx=imx_lo,imx_hi
       
    enddo ! do icell=1,ncell
!$omp end do

    return
  end subroutine ns_tilex_kernel
  ! ------------------------------- OLD KERNEL ------------------------------------------

  ! *
  ! * Reduces tile_excl => tile_excl_top
  ! * tile_excl_top only contains tiles that have non-zero topological exclusion masks
  ! *
  subroutine reduce_to_tile_top(tile_excl, tile_ind_top, tile_excl_top, tiles, &
       tile_start, tile_end, pos0)
    implicit none

    ! Input / Output
    type(tile_excl_t), intent(in), dimension(:) :: tile_excl
    integer, pointer, dimension(:) :: tile_ind_top
    type(tile_excl_t), pointer, dimension(:) :: tile_excl_top
    integer, intent(in), dimension(:) :: tiles
    integer, intent(in) :: tile_start, tile_end, pos0

    ! Variables
    integer i, pos

    pos = pos0

    do i=tile_start,tile_end
       if (tiles(i) == 1) then
          ! Set j-atom start index
          tile_ind_top(pos) = i - 1
          ! Set tile topological exclusion mask
          tile_excl_top(pos)%excl = tile_excl(i)%excl
          pos = pos + 1
       endif
    enddo

    return
  end subroutine reduce_to_tile_top

  ! *
  ! * Builds exclusion mask based on topology
  ! * cell_index() = cell index for each sorted atom
  ! *
  subroutine build_excl_topology(nblist, ijlist, loc2glo_ind, glo2loc_ind, &
       cell_start, cell_index, tiles, tile_excl, tile_ind_shift, top_excl_pos, top_excl)
    use parallel,only:mynod
    use nblist_types,only:nblist_tilex_t
    implicit none
    ! Input / Output
    type(nblist_tilex_t), intent(inout) :: nblist
    integer, intent(in), allocatable, dimension(:,:) :: ijlist
    integer, intent(in), dimension(:) :: loc2glo_ind, glo2loc_ind, cell_start, cell_index
    integer, intent(inout) :: tiles(:)
    type(tile_excl_t), intent(inout), dimension(:) :: tile_excl
    integer, intent(in) :: tile_ind_shift
    integer, intent(in) :: top_excl_pos(:), top_excl(:)
    ! Variables
    integer ic, istart, iend, icell
    integer excl_i, excl_start, excl_end
    integer jcell, jcell_min, jcell_max
    integer i, j, is, js, jstart, iexcl, ibit
    integer tile_ind, tile_start, tile_end
    integer tile_ind0, tile_ind1, jcell_tmp

    do ic=1,nblist%ni
       ! cell i consists of atoms istart:iend
       istart = nblist%ientry(ic)%indi + 1
       icell = cell_index(istart)
       iend = cell_start(icell+1) - 1
       ! icell entry consists of tiles tile_start:tile_end
       tile_start = nblist%ientry(ic)%startj + 1
       tile_end = nblist%ientry(ic)%endj + 1
       ! i-cell has neighbor j-cells at jcell_min:jcell_max
       jcell_min = ijlist(3,tile_start - tile_ind_shift)
       jcell_max   = ijlist(3,tile_end - tile_ind_shift)

       ! Loop through atoms in cell i
       do is=istart,iend
          i = loc2glo_ind(is)+1

          excl_start = top_excl_pos(i)
          excl_end = top_excl_pos(i+1) - 1

          ! Loop through topological exclusions:
          ! Atom i has exclusions top_excl(excl_start:excl_end)
          do excl_i=excl_start,excl_end
             ! j = j atom index in global list
             ! js = j atom index in sorted list
             ! jcell = j atom cell index
             j = top_excl(excl_i)
             js = glo2loc_ind(j)+1
             if (js == 0) cycle        ! This atom is not on this node
             jcell = cell_index(js)

             ! Atom j is in jcell

             ! Cycle loop because is >= js is excluded by the GPU routine
             if (icell == jcell .and. is >= js) cycle

             if (jcell >= jcell_min .and. jcell <= jcell_max) then
                ! Atom j is in this j-list
                ! Now find the tile index where this i-j interaction resides

                ! NOTE: Linear search same speed as binary search

!!$                do tile_ind=tile_start,tile_end
!!$                   if (ijlist(3,tile_ind-tile_ind_shift) == jcell) exit
!!$                enddo
                tile_ind0 = tile_start
                tile_ind1 = tile_end
                jcell_tmp = 0
                do while (jcell_tmp /= jcell .and. tile_ind0 <= tile_ind1)
                   tile_ind = (tile_ind0 + tile_ind1)/2
                   jcell_tmp = ijlist(3,tile_ind-tile_ind_shift)
                   if (jcell_tmp < jcell) then
                      ! jcell_tmp < jcell => jcell is in the upper part
                      tile_ind0 = tile_ind + 1
                   else if (jcell_tmp > jcell) then
                      ! jcell_tmp > jcell => jcell is in the lower part
                      tile_ind1 = tile_ind - 1
                   endif
                enddo

                if (jcell_tmp == jcell) then
!!$                if (tile_ind <= tile_end) then
                   jstart = cell_start(jcell)
                   call calc_iexcl_ibit(is-istart, js-jstart, iexcl, ibit)
                   if (tiles(tile_ind) == 0) then
                      ! New tile
                      tile_excl(tile_ind)%excl(1:num_excl) = 0
                      tile_excl(tile_ind)%excl(iexcl) = ibit
                      tiles(tile_ind) = 1
                   else
                      ! Existing tile
                      tile_excl(tile_ind)%excl(iexcl) = ior(tile_excl(tile_ind)%excl(iexcl), ibit)
                   endif
                endif

             endif
          enddo
       enddo

    enddo

    return
  end subroutine build_excl_topology

  ! *
  ! * Start a new i entry
  ! *
  subroutine start_ientry(ni, nj, ni_max, ientry, ish, i)
    use nblist_types,only:ientry_t
    use memory
    implicit none
    ! Input / Output
    integer, intent(inout) :: ni, ni_max
    integer, intent(in) :: nj
    type(ientry_t), intent(inout), pointer, dimension(:) :: ientry
    integer, intent(in) :: ish, i
    ! Variables
    type(ientry_t), allocatable, dimension(:) :: ientry_tmp
    integer ni_max_new

    ni = ni + 1
    if (ni > ni_max) then
       ni_max_new = int(ni*1.5)
       ! Allocate temporary ientry_tmp
       allocate(ientry_tmp(ni_max))
       ! Copy ientry -> ientry_tmp
       ientry_tmp(1:ni_max) = ientry(1:ni_max)
       ! Deallocate ientry
       deallocate(ientry)
       ! Allocate new ientry
       allocate(ientry(ni_max_new))
       ientry(1:ni_max) = ientry_tmp(1:ni_max)
       ! Deallocate temporary ientry_tmp
       deallocate(ientry_tmp)
       ni_max = ni_max_new
    endif
    ! Store atom index
    ientry(ni)%indi = i - 1
    ! Store shift index
    ientry(ni)%ish = ish
    ! Store start index of j-list
    ientry(ni)%startj = nj + 1 - 1

    return
  end subroutine start_ientry

  ! *
  ! * Stop an i entry
  ! *
  subroutine stop_ientry(ni, nj, ientry)
    use nblist_types,only:ientry_t
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, nj
    type(ientry_t), intent(inout), pointer, dimension(:) :: ientry

    ientry(ni)%endj = nj - 1

    return
  end subroutine stop_ientry

  ! *
  ! * Reduce ientry%startj and ientry%endj
  ! * After the reduction, ientry()%%startj and ientry()%%endj give the correct global
  ! * tile indices
  ! *
  subroutine reduce_nblist_startj_endj(nblist, ntile_tot, tile_ind_shift)
    use nblist_types,only:nblist_tilex_t
    use domdec_common,only:nthread
    implicit none
    ! Input / Output
    type(nblist_tilex_t), intent(inout), allocatable, dimension(:) :: nblist
    integer, intent(out) :: ntile_tot
    integer, intent(out) :: tile_ind_shift(0:nthread-1)
    ! Variables
    integer i, ii, nj_start

    nj_start = nblist(0)%nj

    tile_ind_shift(0) = 0

    if (nthread > 1) then
       do i=1,nthread-1
!$omp parallel do schedule(static) private(ii)
          do ii=1,nblist(i)%ni
             nblist(i)%ientry(ii)%startj = nblist(i)%ientry(ii)%startj + nj_start
             nblist(i)%ientry(ii)%endj = nblist(i)%ientry(ii)%endj + nj_start
          enddo
!$omp end parallel do
          tile_ind_shift(i) = nj_start
          nj_start = nj_start + nblist(i)%nj
       enddo
    endif

    ntile_tot = nj_start

    return
  end subroutine reduce_nblist_startj_endj

  ! *
  ! * Add j to nblist
  ! *
  subroutine add_jentry(nj, nj_max, tile, j, excl)
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(inout) :: nj, nj_max
    integer, intent(inout), allocatable, dimension(:,:) :: tile
    integer, intent(in) :: j, excl(num_excl)

    nj = nj + 1
    if (nj > nj_max) then
       nj_max = int(nj*1.5)
       call chmrealloc('nblist_tilex.src','add_jentry','tile',1+num_excl,nj_max,intg=tile)
    endif
    ! Index of atom j
    tile(1,nj) = j - 1
    ! ij exclusion mask
    tile(2:2+num_excl-1,nj) = excl(1:num_excl)

    return
  end subroutine add_jentry

  ! *
  ! * Flushes to ijlist
  ! *
  subroutine flush_to_ijlist(nblist, ijlist_loc, n_ijlist_loc, ijlist_len_loc, &
       icell, jlist, n_jlist, ish, cell_start)
    use memory,only:chmrealloc
    use nblist_types,only:nblist_tilex_t
    implicit none
    ! Input / Output
    type(nblist_tilex_t), intent(inout) :: nblist
    integer, intent(inout), allocatable, dimension(:,:) :: ijlist_loc
    integer, intent(inout) :: n_ijlist_loc, ijlist_len_loc
    integer, intent(in) :: icell, jlist(:), n_jlist, ish, cell_start(:)
    ! Variables
    integer istart, i

    istart = cell_start(icell)
    call start_ientry(nblist%ni, nblist%nj, nblist%ni_max, nblist%ientry, ish, istart)
    nblist%nj = nblist%nj + n_jlist
    call stop_ientry(nblist%ni, nblist%nj, nblist%ientry)

    if (n_ijlist_loc + n_jlist > ijlist_len_loc) then
       ! Re-allocate
       ijlist_len_loc = int((n_ijlist_loc + n_jlist)*1.5)
       call chmrealloc('nblist_tilex.src','flush_to_ijlist','ijlist_loc',&
            3,ijlist_len_loc,intg=ijlist_loc)
    endif

    do i=1,n_jlist
       ijlist_loc(1:3,n_ijlist_loc+i) = (/ icell, ish, jlist(i) /)
    enddo
    n_ijlist_loc = n_ijlist_loc + n_jlist

    return
  end subroutine flush_to_ijlist

  ! *
  ! * Calculates iexcl and ibit from i and j
  ! *
  ! * iexcl = index of the exclusion mask 1...num_excl
  ! * ibit  = exclusion bit
  ! *
  ! * NOTE: i,j = 0...tilesize-1 are the indices on the tile
  ! *
  subroutine calc_iexcl_ibit(i, j, iexcl, ibit)
    implicit none
    ! Input / Output
    integer, intent(in) :: i, j
    integer, intent(out) :: iexcl, ibit
    ! Variables
    integer ij, tmp, tid

    if (tilesize == 32) then
       ij = i + j*tilesize - j
       iexcl = j + 1
       ibit = ishft(1, mod(ij, 32))
    else
       tid = mod(i+j,2)*16 + j
       iexcl = tid/4 + 1
       tmp = i + 1 + j*15
       ibit = ishft(1, mod(tmp/2,8) + mod(j,4)*8)
    endif

    return
  end subroutine calc_iexcl_ibit

  subroutine print_iexcl_ibit()
    use stream,only:outu
    implicit none
    integer i, j, iexcl, ibit
    integer row(0:32-1)

    write (outu,'(a)') 'iexcl='
    do j=0,tilesize-1
       do i=0,tilesize-1
          call calc_iexcl_ibit(i, j, iexcl, ibit)
          row(i) = iexcl - 1
       enddo
       write (outu,'(32i3)') row(0:tilesize-1)
    enddo

    write (outu,'(a)') 'ibit='
    do j=0,tilesize-1
       do i=0,tilesize-1
          call calc_iexcl_ibit(i, j, iexcl, ibit)
          row(i) = log2(ibit)
       enddo
       write (outu,'(32i3)') row(0:tilesize-1)
    enddo

    return

  contains
    integer function log2(x)
      implicit none
      integer, intent(in) :: x
      integer t

      log2 = -1
      t = x
      do while (t /= 0)
         t = ishft(t,-1)
         log2 = log2 + 1
      enddo

      return
    end function log2

  end subroutine print_iexcl_ibit

  ! *
  ! * Checks if particles i and j are within the cut-off
  ! *
  subroutine check_cut(i, j, xsh, ysh, zsh, xyzq, cutsq, q_incut, rsq_out)
    use domdec_common,only:hboxx, hboxy, hboxz
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: i, j
    real(chm_real4), intent(in) :: xsh, ysh, zsh
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: cutsq
    logical, intent(out) :: q_incut
    real(chm_real4), intent(out), optional :: rsq_out
    ! Variables
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq

    dx = abs(xyzq(i)%x + xsh - xyzq(j)%x)
    dy = abs(xyzq(i)%y + ysh - xyzq(j)%y)
    dz = abs(xyzq(i)%z + zsh - xyzq(j)%z)

    if (dx > hboxx .or. dy > hboxy .or. dz > hboxz) then
       write (*,'(a,3f10.3)') 'dx,dy,dz=',dx,dy,dz
       write (*,'(a,3f10.3)') 'xsh,ysh,zsh=',xsh,ysh,zsh
       write (*,'(a,3f10.3)') 'xyzi=',xyzq(i)%x,xyzq(i)%y,xyzq(i)%z
       write (*,'(a,3f10.3)') 'xyzj=',xyzq(j)%x,xyzq(j)%y,xyzq(j)%z
       stop
    endif

    rsq = dx*dx + dy*dy + dz*dz
    q_incut = .false.
    if (rsq < cutsq) q_incut = .true.

    if (present(rsq_out)) rsq_out = rsq

    return
  end subroutine check_cut

  ! *
  ! * Computes cell bounds (icell0, icell1)
  ! * bx(0:ncell) is the right cell boundary, for x (and y) direction:
  ! * bx(0) = 0
  ! * bx(1) = dx
  ! * bx(2) = dx*2
  ! *
  ! * Returns:
  ! * jcell0:jcell1 = range of cells within cut-off
  ! * dist          = distance for each cell
  ! *
  subroutine get_cell_bounds1(icell, ncell, x0, x1, bx, cut, jcell0, jcell1, dist)
    implicit none
    ! Input / Output
    integer, intent(in) :: icell, ncell
    real(chm_real4), intent(in) :: x0, x1, bx(0:*), cut
    integer, intent(out) :: jcell0, jcell1
    real(chm_real4), intent(out) :: dist(:)
    ! Variables
    real(chm_real4) d
    integer jcell_start_left, jcell_start_right
    integer j

    ! Search within a single zone (I)
    if (icell < 1) then
       ! This is one of the image cells on the left =>
       ! set the left cell boundary (jcell0) to 1 and start looking for the right
       ! boundary from 1
       jcell_start_left = 0          ! with this value, we don't look for cells on the left
       jcell_start_right = 1         ! start looking for cells at right from 1
       jcell0 = 1                    ! left boundary set to minimum value
       jcell1 = 0                    ! set to "no cells" value
       dist(1) = 0.0_chm_real4
    elseif (icell > ncell) then
       ! This is one of the image cells on the right =>
       ! set the right cell boundary (icell1) to ncell and start looking for the left
       ! boundary from ncell
       jcell_start_left = ncell      ! start looking for cells at left from ncell
       jcell_start_right = ncell + 1 ! with this value, we don't look for cells on the right
       jcell0 = ncell + 1            ! set to "no cells" value
       jcell1 = ncell                ! right boundary set to maximum value
       dist(ncell) = 0.0_chm_real4
    else
       jcell_start_left = icell - 1
       jcell_start_right = icell + 1
       jcell0 = icell
       jcell1 = icell
       dist(icell) = 0.0_chm_real4
    endif

    ! Check cells at left, stop once the distance to the cell right boundary 
    ! is greater than the cutoff.
    !
    ! Cell right boundary is at bx(i)
    do j=jcell_start_left,1,-1
       d = x0 - bx(j)
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell0 = j
    enddo

    ! Check cells at right, stop once the distance to the cell left boundary
    ! is greater than the cutoff.
    !
    ! Cell left boundary is at bx(i-1)
    
    do j=jcell_start_right,ncell
       d = bx(j-1) - x1
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell1 = j
    enddo

    ! Cell bounds are jcell0:jcell1

    return
  end subroutine get_cell_bounds1

  ! *
  ! * Computes cell bounds (icell0, icell1)
  ! * bx(0:ncell) is the right cell boundary, for x (and y) direction:
  ! * bx(0) = 0
  ! * bx(1) = dx
  ! * bx(2) = dx*2
  ! *
  ! * Returns:
  ! * jcell0:jcell1 = range of cells within cut-off
  ! * dist          = distance for each cell
  ! *
  subroutine get_cell_bounds2(izone, jzone, icell, ncell, x0, x1, bx, cut, jcell0, jcell1, dist)
    implicit none
    ! Input / Output
    integer, intent(in) :: izone, jzone, icell, ncell
    real(chm_real4), intent(in) :: x0, x1, bx(0:*), cut
    integer, intent(out) :: jcell0, jcell1
    real(chm_real4), intent(out) :: dist(:)
    ! Variables
    real(chm_real4) d
    integer jcell_start_left, jcell_start_right
    integer j

    ! Search between two zones
    if (bx(0) >= x1 .or. (bx(0) < x1 .and. bx(0) > x0)) then
       ! j-zone is to the right of i-zone
       ! => no left search, start right search from 1
       jcell_start_left = 0
       jcell_start_right = 1
       jcell0 = 1
       jcell1 = 0
    elseif (bx(ncell) <= x0 .or. (bx(ncell) > x0 .and. bx(ncell) < x1)) then
       ! j-zone is to the left of i-zone
       ! => no right search, start left search from ncell
       jcell_start_left = ncell
       jcell_start_right = ncell + 1
       jcell0 = ncell + 1
       jcell1 = ncell
    else
       ! i-zone is between j-zones
       ! => safe choice is to search the entire range
       jcell_start_left = ncell
       jcell_start_right = 1
       jcell0 = ncell
       jcell1 = 1
    endif

    ! Check cells at left, stop once the distance to the cell right boundary 
    ! is greater than the cutoff.
    !
    ! Cell right boundary is at bx(i)
    do j=jcell_start_left,1,-1
       d = x0 - bx(j)
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell0 = j
    enddo

    ! Check cells at right, stop once the distance to the cell left boundary
    ! is greater than the cutoff.
    !
    ! Cell left boundary is at bx(i-1)
    
    do j=jcell_start_right,ncell
       d = bx(j-1) - x1
       if (d > cut) exit
       dist(j) = max(0.0_chm_real4, d)
       jcell1 = j
    enddo

    ! Cell bounds are jcell0:jcell1

    return
  end subroutine get_cell_bounds2

  ! *
  ! * Calculates bounding boxes for each cell
  ! *
  subroutine calc_bounding_box(xyzq, ncell, cell_start)
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: ncell, cell_start(:)
    ! Variables
    real(chm_real4) x0, x1, y0, y1, z0, z1
    integer i, ii, istart, iend

!$omp parallel do schedule(static) private(i, istart, iend, x0, y0, z0, x1, y1, z1, ii)
    do i=1,ncell
       istart = cell_start(i)
       iend = cell_start(i+1) - 1
       x0 = xyzq(istart)%x
       y0 = xyzq(istart)%y
       z0 = xyzq(istart)%z
       x1 = x0
       y1 = y0
       z1 = z0
       do ii=istart+1,iend
          x0 = min(x0, xyzq(ii)%x)
          y0 = min(y0, xyzq(ii)%y)
          z0 = min(z0, xyzq(ii)%z)
          x1 = max(x1, xyzq(ii)%x)
          y1 = max(y1, xyzq(ii)%y)
          z1 = max(z1, xyzq(ii)%z)
       enddo
       cell_bb(i)%x0 = 0.5*(x0 + x1)
       cell_bb(i)%y0 = 0.5*(y0 + y1)
       cell_bb(i)%z0 = 0.5*(z0 + z1)
       cell_bb(i)%xw = 0.5*(x1 - x0)
       cell_bb(i)%yw = 0.5*(y1 - y0)
       cell_bb(i)%zw = 0.5*(z1 - z0)
    enddo
!$omp end parallel do

    return
  end subroutine calc_bounding_box

  ! *
  ! * Sort j-tiles
  ! * NOTE: this is a temporary testing routine, should not be used in release version!
  ! *
  subroutine sort_jtiles(nblist)
    use memory
    use nblist_types,only:nblist_tilex_t
    use exfunc,only:order5
    implicit none
    ! External subroutines
    external exch5
    ! Input / Output
    type(nblist_tilex_t), intent(inout) :: nblist
    ! Variables
    integer, allocatable, dimension(:) :: tmp1, tmp2
    integer, allocatable, dimension(:,:) :: tmpexcl
    integer startj, endj, n
    integer i, j

    call chmalloc('nblist_tilex.src','sort_jtiles','tmp1',500,intg=tmp1)
    call chmalloc('nblist_tilex.src','sort_jtiles','tmp2',500,intg=tmp2)
    call chmalloc('nblist_tilex.src','sort_jtiles','tmpexcl',num_excl,500,intg=tmpexcl)

    do i=1,nblist%ni
       startj = nblist%ientry(i)%startj ! NOTE: startj starts from 0
       endj = nblist%ientry(i)%endj

       n = endj-startj+1

       if (size(tmp1) < n) then
          call chmdealloc('nblist_tilex.src','sort_jtiles','tmp1',size(tmp1),intg=tmp1)
          call chmdealloc('nblist_tilex.src','sort_jtiles','tmp2',size(tmp2),intg=tmp2)
          call chmdealloc('nblist_tilex.src','sort_jtiles','tmpexcl',&
               size(tmpexcl,1),size(tmpexcl,2),intg=tmpexcl)
          call chmalloc('nblist_tilex.src','sort_jtiles','tmp1',int(n*1.5),intg=tmp1)
          call chmalloc('nblist_tilex.src','sort_jtiles','tmp2',int(n*1.5),intg=tmp2)
          call chmalloc('nblist_tilex.src','sort_jtiles','tmpexcl',&
               num_excl,int(n*1.5),intg=tmpexcl)
       endif

       do j=1,n
          tmp1(j) = nblist%tile(1, startj+j)
          tmp2(j) = j
          tmpexcl(1:num_excl,j) = nblist%tile(2:2+num_excl-1, startj+j)
       enddo

       call sort(n, exch5, order5, tmp1, tmp2, 0, 0, 0, 0, 0, 2)

       ! tmp1(1:n) = sorted indj
       ! tmp2(1:n) = new indices
       do j=1,n
          nblist%tile(1, startj+j) = tmp1(j)
          nblist%tile(2:2+num_excl-1, startj+j) = tmpexcl(1:num_excl, tmp2(j))
       enddo

    enddo

    call chmdealloc('nblist_tilex.src','sort_jtiles','tmp1',size(tmp1),intg=tmp1)
    call chmdealloc('nblist_tilex.src','sort_jtiles','tmp2',size(tmp2),intg=tmp2)
    call chmdealloc('nblist_tilex.src','sort_jtiles','tmpexcl',&
         size(tmpexcl,1),size(tmpexcl,2),intg=tmpexcl)

    return
  end subroutine sort_jtiles

  ! *
  ! * For debugging: Print info about the neighbor list
  ! *
  subroutine print_info_tilex(nblist, xyzq, atom_index, ncoord, inb14, iblo14)
    use stream,only:outu
    use nblist_types,only:nblist_tilex_t
    use domdec_local_types,only:xyzq_sp_t
    use inbnd,only:cutnb, ctofnb
    use nblist_util,only:check_excl
    implicit none
    ! Input
    type(nblist_tilex_t), intent(in) :: nblist
    type(xyzq_sp_t), optional, intent(in) :: xyzq(:)
    integer, optional, intent(in) :: atom_index(:), ncoord
    integer, optional, intent(in) :: inb14(*), iblo14(*)
    ! Variables
    integer starti
    integer startj, endj
    integer excl(num_excl)
    integer npairs, ntiles, nlines
    integer npairs_cutnb, npairs_ctofnb
    integer iexcl, ibit
    integer i, j, k
    integer ii, jj
    integer ix14

    write (*,'(a,i3)') 'tilesize=',tilesize
    write (*,'(a,2i8)') 'ni,ni_max=',nblist%ni,nblist%ni_max
    write (*,'(a,2i8)') 'nj,nj_max=',nblist%nj,nblist%nj_max

    call check_excl(1, 2, inb14, iblo14, ix14)
    write (outu,*) '1 vs. 2',ix14
    call check_excl(1, 3, inb14, iblo14, ix14)
    write (outu,*) '1 vs. 3',ix14
    call check_excl(1, 4, inb14, iblo14, ix14)
    write (outu,*) '1 vs. 4',ix14

    ntiles = nblist%nj
    write (outu,'(a,i12,i8,i10)') 'npairs,ntiles,nlines=',npairs,ntiles,nlines

    if (present(xyzq)) then
       call calc_npairs(xyzq, atom_index, ncoord, inb14, iblo14, &
            cutnb, npairs_cutnb)
       write (*,'(a,i12,f8.2)') 'npairs_cutnb,cutnb=',npairs_cutnb,cutnb
       call calc_npairs(xyzq, atom_index, ncoord, inb14, iblo14, &
            ctofnb, npairs_ctofnb)
       write (*,'(a,i12,f8.2)') 'npairs_ctofnb,ctofnb=',npairs_ctofnb,ctofnb
    endif

!    call write_nj(nblist)    

    return
  end subroutine print_info_tilex

!!$  ! *
!!$  ! * Finds the cell index of atom i
!!$  ! *
!!$  integer function find_icell(i)
!!$    implicit none
!!$    ! Input
!!$    integer, intent(in) :: i
!!$
!!$    do find_icell=1,ncell
!!$       if (i >= cell_start(find_icell) .and. i < cell_start(find_icell+1)) return
!!$    enddo
!!$
!!$  end function find_icell

  ! *
  ! * Counts the number of 1 bits in an integer
  ! *
  integer function countbits(a)
    implicit none
    ! Input
    integer, intent(in) :: a
    ! Variables
    integer b, i

    b = a
    countbits = 0
    do i=1,32
       countbits = countbits + iand(b,1)
       b = ishft(b,-1)
    enddo

    return
  end function countbits

  ! *
  ! * Calculates number of pairs using a simple N^2 method
  ! *
  subroutine calc_npairs(xyzq, atom_index, ncoord, inb14, iblo14, cut, npairs)
    use domdec_common,only:boxx, boxy, boxz, hboxx, hboxy, hboxz
    use domdec_local_types,only:xyzq_sp_t
    use nblist_util,only:check_excl
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: atom_index(:)
    integer, intent(in) :: ncoord
    integer, intent(in) :: inb14(*), iblo14(*)
    real(chm_real), intent(in) :: cut
    integer, intent(out) :: npairs
    ! Variables
    real(chm_real4) cutsq
    real(chm_real4) boxxf, boxyf, boxzf
    real(chm_real4) hboxxf, hboxyf, hboxzf
    real(chm_real4) xi, yi, zi
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq
    integer ix14
    integer i, j
    
    boxxf = boxx
    boxyf = boxy
    boxzf = boxz
    hboxxf = hboxx
    hboxyf = hboxy
    hboxzf = hboxz
    cutsq = cut*cut

    open(121,file='simple.txt')

    npairs = 0
    do i=1,ncoord-1
       xi = xyzq(i)%x
       yi = xyzq(i)%y
       zi = xyzq(i)%z
       do j=i+1,ncoord
          call check_excl(atom_index(i), atom_index(j), inb14, iblo14, ix14)
          if (ix14 > 0) then
             dx = abs(xi - xyzq(j)%x)
             dy = abs(yi - xyzq(j)%y)
             dz = abs(zi - xyzq(j)%z)
             if (dx > hboxxf) dx = dx - boxxf
             if (dy > hboxyf) dy = dy - boxyf
             if (dz > hboxzf) dz = dz - boxzf
             rsq = dx**2 + dy**2 + dz**2
             if (rsq < cutsq) then
                write (121,'(2i6)') i,j
                npairs = npairs + 1
             endif
          endif
       enddo
    enddo

    close(121)

    return
  end subroutine calc_npairs

  ! *
  ! * For debugging: Writes xyzq to file
  ! *
  subroutine write_xyzq(xyzq, ncoord)
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: ncoord
    ! Variables
    integer i

    open(121,file='xyzq.txt')

    do i=1,ncoord
       write (121,'(3f10.4)') xyzq(i)%x, xyzq(i)%y, xyzq(i)%z
    enddo

    close(121)

    return
  end subroutine write_xyzq

  ! *
  ! * For debugging: Writes the distribution of neighbor list to file
  ! *
  subroutine write_nj(nblist)
    use domdec_common,only:nthread
    use nblist_types,only:nblist_tilex_t
    implicit none
    ! Input
    type(nblist_tilex_t), intent(in) :: nblist(:)
    ! Variables
    integer i, nt

    open(121,file='nj.txt')

    do nt=0,nthread-1
       do i=1,nblist(nt)%ni
          write (121,'(i6)') nblist(nt)%ientry(i)%endj - nblist(nt)%ientry(i)%startj + 1
       enddo
    enddo

    close(121)

    return
  end subroutine write_nj

  ! *
  ! * For debugging: Writes the distribution of neighbor list to file
  ! *
  subroutine write_ientry(n, ientry)
    use nblist_types,only:ientry_t
    implicit none
    ! Input
    integer, intent(in) :: n
    type(ientry_t), intent(in) :: ientry(:)
    ! Variables
    integer i

    open(124,file='nj.txt')

    do i=1,n
       write (124,'(i6)') ientry(i)%endj - ientry(i)%startj + 1
    enddo

    close(124)

    return
  end subroutine write_ientry

  ! *
  ! * Deallocates all data for this module
  ! *
  subroutine uninit_nblist_tilex(nblist1, nblist2)
    use nblist_types,only:nblist_tilex_t
    use nblist_util,only:uninit_array
    use memory,only:chmdealloc
    use domdec_common,only:nthread
    implicit none
    ! Input / Output
    type(nblist_tilex_t), intent(inout), allocatable, dimension(:) :: nblist1, nblist2
    ! Variables
    integer i

    call uninit_pinned_nblist_tilex()
    
    call uninit_array(xdist)

    call uninit_array(ydist)

    call uninit_array(zdist)

    call uninit_array(jlist1)
    call uninit_array(jlist2)

    call uninit_array(ijlist1)
    call uninit_array(ijlist2)

    if (allocated(cumsum_th)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','cumsum_th',&
            size(cumsum_th),intg=cumsum_th)
    endif

    if (allocated(ijlist1_len)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','ijlist1_len',&
            size(ijlist1_len),intg=ijlist1_len)
    endif
    if (allocated(ijlist2_len)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','ijlist2_len',&
            size(ijlist2_len),intg=ijlist2_len)
    endif

    if (allocated(n_ijlist1)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','n_ijlist1',&
            size(n_ijlist1),intg=n_ijlist1)
    endif
    if (allocated(n_ijlist2)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','n_ijlist2',&
            size(n_ijlist2),intg=n_ijlist2)
    endif

    if (allocated(nblist1)) then
       do i=0,nthread-1
          if (associated(nblist1(i)%ientry)) then
             deallocate(nblist1(i)%ientry)
          endif
          if (allocated(nblist1(i)%tile)) then
             call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','tile',&
                  size(nblist1(i)%tile,1), size(nblist1(i)%tile,2), intg=nblist1(i)%tile)
          endif
       enddo
    endif
    if (allocated(nblist2)) then
       do i=0,nthread-1
          if (associated(nblist2(i)%ientry)) then
             deallocate(nblist2(i)%ientry)
          endif
          if (allocated(nblist2(i)%tile)) then
             call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','tile',&
                  size(nblist2(i)%tile,1), size(nblist2(i)%tile,2), intg=nblist2(i)%tile)
          endif
       enddo
    endif

    if (allocated(cell_bb)) then
       deallocate(cell_bb)
    endif

    call uninit_array(ex14)

    if (allocated(ex14_len)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','ex14_len',&
            nthread,intg=ex14_len)
    endif

    if (allocated(sort_cell)) then
       deallocate(sort_cell)
    endif

    if (allocated(ientry_tmp)) then
       deallocate(ientry_tmp)
    endif

    if (allocated(sortlist1)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','sortlist1',&
            size(sortlist1),intg=sortlist1)
    endif

    if (allocated(sortlist2)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','sortlist2',&
            size(sortlist2),intg=sortlist2)
    endif

    if (allocated(bucket)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','bucket',&
            size(bucket,1),size(bucket,2),intg=bucket)
    endif

    if (allocated(tile_ind_shift1)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','tile_ind_shift1',&
            size(tile_ind_shift1),intg=tile_ind_shift1)
    endif
    if (allocated(tile_ind_shift2)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','tile_ind_shift2',&
            size(tile_ind_shift2),intg=tile_ind_shift2)
    endif

    if (allocated(num_tiles_table1)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','num_tiles_table1',&
            size(num_tiles_table1),intg=num_tiles_table1)
    endif
    if (allocated(num_tiles_table2)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','num_tiles_table2',&
            size(num_tiles_table2),intg=num_tiles_table2)
    endif

    if (allocated(pos_shift1)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','pos_shift1',&
            size(pos_shift1),intg=pos_shift1)
    endif
    if (allocated(pos_shift2)) then
       call chmdealloc('nblist_tilex.src','dealloc_nblist_tilex','pos_shift2',&
            size(pos_shift2),intg=pos_shift2)
    endif

    if (allocated(tile_excl1)) then
       deallocate(tile_excl1)
    endif

    if (allocated(tile_excl2)) then
       deallocate(tile_excl2)
    endif
    
    if (allocated(tiles1)) then
       call chmdealloc('nblist_tilex.src','uninit_nblist_tilex','tiles1',&
            size(tiles1),intg=tiles1)
    endif

    if (allocated(tiles2)) then
       call chmdealloc('nblist_tilex.src','uninit_nblist_tilex','tiles2',&
            size(tiles2),intg=tiles2)
    endif

    return
  end subroutine uninit_nblist_tilex

  ! *
  ! * De-allocates all pinned memory
  ! *
  subroutine uninit_pinned_nblist_tilex()
    use nblist_util,only:dealloc_gpu

    if (associated(ijlist1_tot)) then
       call dealloc_gpu(ijlist1_tot)
    endif
    if (associated(ijlist2_tot)) then
       call dealloc_gpu(ijlist2_tot)
    endif

    if (associated(ientry1_tot)) then
       call dealloc_gpu(ientry1_tot)
    endif
    if (associated(ientry2_tot)) then
       call dealloc_gpu(ientry2_tot)
    endif    

    if (associated(tile_excl_top1)) then
       call dealloc_gpu(tile_excl_top1)
    endif

    if (associated(tile_excl_top2)) then
       call dealloc_gpu(tile_excl_top2)
    endif

    if (associated(tile_ind_top1)) then
       call dealloc_gpu(tile_ind_top1)
    endif

    if (associated(tile_ind_top2)) then
       call dealloc_gpu(tile_ind_top2)
    endif

    return
  end subroutine uninit_pinned_nblist_tilex
  
#endif /* (domdec_gpu)*/

end module nblist_tilex

