module nblist_tilex_sort

#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
  !
  ! Tilex neighbor search sorting
  !
  use, intrinsic :: iso_c_binding
  use chm_kinds
  use dimens_fcm
  use nblist_types,only:cr4array_t
  implicit none
  private

  real(chm_real4), parameter :: bucket_scale = 2.0d0

  ! Number of cells for each zone
  integer ncellx(8), ncelly(8)
  integer, allocatable, dimension(:) :: ncellz
  integer max_ncellz(8)                   ! = maxval(ncellz)
  integer ncell                           ! = total number of cells
  ! Cell boundaries
  type(cr4array_t) cellbx(8), cellby(8), cellbz(8)
  ! Cell x and y sizes
  real(chm_real4) celldx(8), celldy(8)

  integer startcol_zone(8)

  ! Interaction zones:
  ! zone "izone" interacts with zones int_zone(1:n_int_zone(izone),izone)
  integer n_int_zone(8), int_zone(8,8)

  ! Defines the starting cell index for each zone
  ! zone_cell_start(1) = 1 = start for zone I
  ! zone_cell_start(2) = start for zone FZ
  ! ...
  ! zone_cell_start(9) = ncell+1
  ! zone ordering is: I,FZ,FY,EX,FX,EZ,EY,C = 1,...8
  integer zone_cell_start(9)

  ! atom_cc_ind(1:ncoord) = index of columns or cells each each atom
  integer, allocatable, dimension(:) :: atom_cc_ind, atom_order

  ! Arrays for sort_xy_columns
  integer, allocatable, dimension(:) :: startatom_col
  integer, allocatable, dimension(:) :: startcell_col
  integer, allocatable, dimension(:,:) :: col_n
  integer, allocatable, dimension(:,:) :: icolxy

  ! Arrays for zsort_tilex
  integer, allocatable, dimension(:,:) :: bucket_n, bucket_start
  real(chm_real4), allocatable, dimension(:) :: zbuf

  ! Cells
  integer, allocatable, dimension(:) :: cell_start, cell_index
  integer, allocatable, dimension(:,:) :: icellxyz

  ! Public variables
  public ncellx, ncelly, ncellz, max_ncellz, ncell, cellbx, cellby, cellbz
  public n_int_zone, int_zone, cell_start, cell_index, icellxyz
  public zone_cell_start, startcell_col, startcol_zone

  ! Public subroutines
  public sort_tilex, copy_cell_start_to_gpu, uninit_nblist_tilex_sort

  ! CUDA routines, these are in nblist_tilex_gpu.cu
  interface
     subroutine copy_cell_start_to_gpuB(h_ncell, h_cell_start) &
          bind(C,name="copy_cell_start_to_gpuB")
       import
       integer(c_int), intent(in) :: h_ncell
       integer(c_int), intent(in) :: h_cell_start(*)
     end subroutine copy_cell_start_to_gpuB
  end interface

contains

  ! *
  ! * Copies cell_start(1:ncell) to GPU
  ! *
  subroutine copy_cell_start_to_gpu()
    implicit none

    call copy_cell_start_to_gpuB(ncell+1, cell_start)

    return
  end subroutine copy_cell_start_to_gpu

  ! *
  ! * Sort atoms for tilex
  ! * NOTE: xyzq_in is the unsorted array
  ! *       xyzq_out is the sorted array
  ! *
  subroutine sort_tilex(zonelist_atom, xyzq_in, xyzq_out, loc2glo_ind_in, loc2glo_ind_out, &
       ncoord, natom)
    use memory
    use domdec_common,only:boxz, hboxz, q_test, min_xyz, max_xyz, nthread
    use domdec_local_types,only:xyzq_sp_t

    use nblist_util,only:cumsum_exclusive  

    use domdec_util_gpu_mod,only:range_start, range_stop
    use nblist_types,only:tilesize
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist_atom(8)
    type(xyzq_sp_t), intent(inout) :: xyzq_in(:)
    type(xyzq_sp_t), intent(out) :: xyzq_out(:)
    integer, intent(inout) :: loc2glo_ind_in(:)
    integer, intent(out) :: loc2glo_ind_out(:)
    integer, intent(in) :: ncoord, natom
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num  
#endif
    ! Variables
!    real(chm_real4) hboxzf
    integer tid
    integer max_col_n(8), ncol_tot, col_start, col_end
    integer icellx, icelly    
    integer ind, nbucket, nbucket_max, len
    integer icell, ncellz_t
    integer i, k, pos, kstart, kend
    integer izone
    integer, allocatable, dimension(:,:) :: atom_order_loc

!    hboxzf = hboxz

    ! Setup interaction zones
    call set_int_zone(zonelist_atom, n_int_zone, int_zone)

    ! Allocate memory for atom_cc_ind
    if (allocated(atom_cc_ind)) then
       if (size(atom_cc_ind) < ncoord) then
          call chmdealloc('nblist_tilex_sort.src','sort_tilex','atom_cc_ind',&
               size(atom_cc_ind),intg=atom_cc_ind)
       endif
    endif
    
    if (.not.allocated(atom_cc_ind)) then
       len = int(ncoord*1.5)
       call chmalloc('nblist_tilex_sort.src','sort_tilex','atom_cc_ind',len,intg=atom_cc_ind)
    endif

    ! Sort atoms into xy-columns
    !
    ! Outputs:
    ! max_col_n(1:8)            = maximum number of atoms in the xy-column per zone
    ! ncol_tot                  = total number of xy-columns
    ! startcol_zone(1:8)        = start of xy-columns for each zone
    ! startatom_col(1:ncol_tot) = start of atom indices for each xy-column
    ! col_n(1:ncol_tot,0)       = number of atoms for each xy-column
    ! xyzq_out(1:ncoord)        = sorted coordinate array
    !
    call sort_xy_columns(zonelist_atom, xyzq_in, loc2glo_ind_in, ncoord, &
         max_col_n, ncol_tot, startcol_zone, atom_cc_ind, xyzq_out, loc2glo_ind_out)

    if (q_test) then
       call test_sort_xy_columns(ncoord, xyzq_out, &
            ncol_tot, startatom_col, startcol_zone, col_n(:,0))
    endif

    ! max_ncellz(1:8) = maximum number of z-cells for each zone
    max_ncellz(1:8) = (max_col_n(1:8)-1)/tilesize + 1

    nbucket_max = int(maxval(max_col_n(1:8))*bucket_scale)

    ! Allocate & reallocate:
    ! (zbuf, bucket_n, bucket_start, cell_start, cell_index, startcell_col, icellxyz, ncellz,
    !  cellbx, cellby, cellbz)
    call alloc_realloc(ncoord, nbucket_max, ncellx, ncelly, max_ncellz, natom, ncol_tot)

    ! Calculate ncellz(1:ncol_tot) = number of cells in each xy-column    
!$omp parallel do schedule(static) private(i)
    do i=1,ncol_tot
       if (col_n(i,0) >= 1) then
          ncellz(i) = (col_n(i,0)-1)/tilesize + 1
       else
          ncellz(i) = 0
       endif
    enddo
!$omp end parallel do

    ! Calculate startcell_col(1:ncol_tot) = cumsum(ncellz(1:ncol_tot))
    ! startcell_col(i) = starting cell index for xy-column "i"
    call cumsum_exclusive(ncellz, ncol_tot, startcell_col)

    ! Set the total number of cells
    ncell = startcell_col(ncol_tot) + ncellz(ncol_tot)

    call range_start('sort_columns_kernel')

    ! Sorts xy-columns according to z coordinate
    ! Outputs:
    ! cell_start(1:ncell) = starting atom index for each cell
    ! xyz_in(1:ncoord)    = sorted coordinate array
!$omp parallel private(tid, izone, nbucket, col_start, col_end)
#ifndef _OPENMP
    tid = 0
#else
    tid = omp_get_thread_num()
#endif
    do izone=1,8
       nbucket = int(max_col_n(izone)*bucket_scale)
       col_start = startcol_zone(izone) + 1
       if (izone < 8) then
          col_end = startcol_zone(izone+1)
       else
          col_end = ncol_tot
       endif
       if (col_start > col_end) cycle
       call sort_columns_kernel(xyzq_out, xyzq_in, loc2glo_ind_out, loc2glo_ind_in, &
!!$            min_xyz(3,izone) + hboxzf, &
!!$            max_xyz(3,izone) + hboxzf, &
            min_xyz(3,izone), &
            max_xyz(3,izone), &
            col_start, col_end, col_n(:,0), &
            nbucket, bucket_n(:,tid), bucket_start(:,tid), atom_cc_ind, atom_order)
    enddo
!$omp end parallel

    call range_stop()

    ! Close cell_start
    cell_start(ncell+1) = ncoord + 1

    call range_start('Calculate cellb')

    ! Calculate cellbx, cellby, cellbz
    call calc_cellb(xyzq_in, real(boxz,kind=chm_real4), startcol_zone)

    ! Set zone_cell_start(1:9)
    do izone=1,8
       ! ind = start index of columns for this zone
       ind = startcol_zone(izone) + 1
       if (ind <= ncol_tot) then
          zone_cell_start(izone) = startcell_col(ind) + 1
       else
          zone_cell_start(izone) = ncell + 1
       endif
    enddo
    zone_cell_start(9) = ncell + 1

    call range_stop()

    ! Write atoms once more to xyzq_out
    ! NOTE: I should implement a spatial sort here for better cache use

!$omp parallel do schedule(static) private(k)
    do k=1,ncoord
       xyzq_out(k) = xyzq_in(k)
    enddo
!$omp end parallel do

!$omp parallel do schedule(static) private(k)
    do k=1,ncoord
       loc2glo_ind_out(k) = loc2glo_ind_in(k)
    enddo
!$omp end parallel do

    if (q_test) then
       call test_sort_tilex(xyzq_out, loc2glo_ind_out, ncoord, natom, zone_cell_start, &
            startcol_zone, ncol_tot)
    endif

    !call write_xyzq_order(ncoord, xyzq_out, loc2glo_ind_out)

    return

  contains
    subroutine alloc_realloc(ncoord, nbucket, ncellx, ncelly, max_ncellz, natom, ncol_tot)
      use memory
      use domdec_common,only:nthread
      implicit none
      ! Input
      integer, intent(in) :: ncoord, nbucket, ncellx(8), ncelly(8), max_ncellz(8), natom, &
           ncol_tot
      ! Variables
      integer len, max_ncellxyz, izone, tmp

      max_ncellxyz = 0
      do izone=1,8
         max_ncellxyz = max_ncellxyz + ncellx(izone)*ncelly(izone)*max_ncellz(izone)
      enddo

      ! zbuf
      if (allocated(zbuf) .and. size(zbuf) < ncoord) then
         call chmdealloc('nblist_tilex_sort.src','sort_tilex','zbuf',&
              size(zbuf),cr4=zbuf)
      endif

      if (.not.allocated(zbuf)) then
         len = int(ncoord*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_tilex','zbuf',&
              len,cr4=zbuf)
      endif

      ! bucket_n, bucket_start
      if (allocated(bucket_n) .and. &
           (size(bucket_n,1) < nbucket .or. size(bucket_n,2) < nthread)) then
         call chmdealloc('nblist_tilex_sort.src','sort_tilex','bucket_n',&
              size(bucket_n,1),size(bucket_n,2),intg=bucket_n)
         call chmdealloc('nblist_tilex_sort.src','sort_tilex','bucket_start',&
              size(bucket_start,1),size(bucket_start,2),intg=bucket_start)
      endif

      if (.not.allocated(bucket_n)) then
         len = int(nbucket*1.2) + 128
         call chmalloc('nblist_tilex_sort.src','sort_tilex','bucket_n',&
              len,nthread,lbou2=0,intg=bucket_n)
         call chmalloc('nblist_tilex_sort.src','sort_tilex','bucket_start',&
              len+1,nthread,lbou2=0,intg=bucket_start)
      endif

      ! cell_start
      if (allocated(cell_start)) then
         if (size(cell_start) < max_ncellxyz + 1) then
            call chmdealloc('nblist_tilex_sort.src','sort_tilex','cell_start',&
                 size(cell_start),intg=cell_start)
         endif
      endif

      if (.not.allocated(cell_start)) then
         len = int(max_ncellxyz*1.5) + 1
         call chmalloc('nblist_tilex_sort.src','sort_tilex','cell_start',&
              len,intg=cell_start)
      endif

      ! cell_index
      if (allocated(cell_index)) then
         if (size(cell_index) < ncoord) then
            call chmdealloc('nblist_tilex_sort.src','sort_tilex','cell_index',&
                 size(cell_index),intg=cell_index)
         endif
      endif

      if (.not.allocated(cell_index)) then
         len = int(ncoord*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_tilex','cell_index',&
              len,intg=cell_index)
      endif

      ! startcell_col
      if (allocated(startcell_col)) then
         if (size(startcell_col) < ncol_tot) then
            call chmdealloc('nblist_tilex_sort.src','sort_tilex','startcell_col',&
                 size(startcell_col),intg=startcell_col)
         endif
      endif

      if (.not.allocated(startcell_col)) then
         len = int(ncol_tot*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_tilex','startcell_col',&
              len,intg=startcell_col)
      endif

      ! icellxyz
      if (allocated(icellxyz)) then
         if (size(icellxyz,2) < max_ncellxyz) then
            call chmdealloc('nblist_tilex_sort.src','sort_tilex','icellxyz',&
                 3,size(icellxyz,2),intg=icellxyz)
         endif
      endif

      if (.not.allocated(icellxyz)) then
         len = int(max_ncellxyz*1.5) + 1
         call chmalloc('nblist_tilex_sort.src','sort_tilex','icellxyz',&
              3,len,intg=icellxyz)
      endif

      ! ncellz
      if (allocated(ncellz)) then
         if (size(ncellz) < ncol_tot) then
            call chmdealloc('nblist_tilex_sort.src','sort_tilex','ncellz',size(ncellz),intg=ncellz)
         endif
      endif

      if (.not.allocated(ncellz)) then
         len = int(ncol_tot*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_tilex','ncellz',len,intg=ncellz)
      endif

      ! cellbx
      do izone=1,8
         if (allocated(cellbx(izone)%array)) then
            if (size(cellbx(izone)%array) < ncellx(izone)+1) then
               call chmdealloc('nblist_tilex_sort.src','sort_tilex','cellbx(izone)%array',&
                    size(cellbx(izone)%array),cr4=cellbx(izone)%array)
            endif
         endif
         if (.not.allocated(cellbx(izone)%array)) then
            len = int(ncellx(izone)*1.5) + 1
            call chmalloc('nblist_tilex_sort.src','sort_tilex','cellbx(izone)%array',&
                 len,lbou=0,cr4=cellbx(izone)%array)
         endif
      enddo

      ! cellby
      do izone=1,8
         if (allocated(cellby(izone)%array)) then
            if (size(cellby(izone)%array) < ncelly(izone)+1) then
               call chmdealloc('nblist_tilex_sort.src','sort_tilex','cellby(izone)%array',&
                    size(cellby(izone)%array),cr4=cellby(izone)%array)
            endif
         endif
         if (.not.allocated(cellby(izone)%array)) then
            len = int(ncelly(izone)*1.5) + 1
            call chmalloc('nblist_tilex_sort.src','sort_tilex','cellby(izone)%array',&
                 len,lbou=0,cr4=cellby(izone)%array)
         endif
      enddo

      ! cellbz
      do izone=1,8
         tmp = ncellx(izone)*ncelly(izone)*(max_ncellz(izone) + 1)
         if (allocated(cellbz(izone)%array)) then
            if (size(cellbz(izone)%array) < tmp) then
               call chmdealloc('nblist_tilex_sort.src','sort_tilex','cellbz(izone)%array',&
                    size(cellbz(izone)%array),cr4=cellbz(izone)%array)
            endif
         endif
         if (.not.allocated(cellbz(izone)%array)) then
            len = int(tmp*1.5)
            call chmalloc('nblist_tilex_sort.src','sort_tilex','cellbz(izone)%array',&
                 len,lbou=0,cr4=cellbz(izone)%array)
         endif
      enddo

      return
    end subroutine alloc_realloc

  end subroutine sort_tilex

  ! *
  ! * Writes xyzq and loc2glo into a file. For debugging purposes only!
  ! *
  subroutine write_xyzq_order(ncoord, xyzq, loc2glo_ind)
    use parallel,only:mynod
    use domdec_local_types,only:xyzq_sp_t
    use domdec_common,only:nthread
    implicit none
    ! Input
    integer, intent(in) :: ncoord
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    ! Variables
    character(40) filename
    integer i

    write (filename,'(a,i1,a,i1,a)') 'xyzq_order_nod',mynod,'_nt',nthread,'.txt'

    open(121,file=trim(filename))
    
    do i=1,ncoord
       write (121,'(i6,3f12.4)') loc2glo_ind(i)+1, xyzq(i)%x, xyzq(i)%y, xyzq(i)%z
    enddo

    close(121)

    return
  end subroutine write_xyzq_order

  ! *
  ! * Calculates cellbx, cellby, cellbz
  ! *
  subroutine calc_cellb(xyzq, boxz, startcol_zone)
    use domdec_common,only:min_xyz, max_xyz !, hboxx, hboxy, hboxz
    use domdec_local_types,only:xyzq_sp_t
    use nblist_types,only:cr4array_t
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: boxz
    integer, intent(in) :: startcol_zone(8)
    ! Variables
!    real(chm_real4) hboxxf, hboxyf, hboxzf
    real(chm_real4) botval, topval
    integer izone, i, ind, icell, ncellz_t, pos, k

!    hboxxf = hboxx
!    hboxyf = hboxy
!    hboxzf = hboxz

!$omp parallel private(izone, i, botval, topval)
    do izone=1,8

       botval = min_xyz(1,izone)! + hboxxf
!$omp do schedule(static)
       do i=0,ncellx(izone)
          cellbx(izone)%array(i) = botval + i*celldx(izone)
       enddo
!$omp end do nowait

       botval = min_xyz(2,izone)! + hboxyf
!$omp do schedule(static)
       do i=0,ncelly(izone)
          cellby(izone)%array(i) = botval + i*celldy(izone)
       enddo
!$omp end do nowait

       botval = min_xyz(3,izone)! + hboxzf
       topval = max_xyz(3,izone)! + hboxzf
!$omp do schedule(static) private(ind, icell, ncellz_t, pos, k)
       do i=1,ncellx(izone)*ncelly(izone)
          ind = i + startcol_zone(izone)
          icell = startcell_col(ind) + 1
          ncellz_t = ncellz(ind)
          pos = (max_ncellz(izone)+1)*(i-1)
          cellbz(izone)%array(pos) = botval
          do k=icell,icell+ncellz_t-2
             ! Set the cell z boundary equal to the z-coordinate of the last atom in the cell:
             pos = pos + 1
             cellbz(izone)%array(pos) = xyzq(cell_start(k+1)-1)%z
          enddo
          ! Last cell boundary = box boundary
          pos = pos + 1
          cellbz(izone)%array(pos) = topval
       enddo
!$omp end do nowait

    enddo
!$omp end parallel

    return
  end subroutine calc_cellb

  ! *
  ! * Sorts columns kernel
  ! *
  subroutine sort_columns_kernel(xyzq_A, xyzq_B, loc2glo_ind_A, loc2glo_ind_B, &
       min_z, max_z, col_start, col_end, col_n, nbucket, bucket_n, bucket_start, &
       cell_ind, atom_order)
    use nblist_types,only:tilesize
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(inout) :: xyzq_A(:), xyzq_B(:)
    integer, intent(in) :: loc2glo_ind_A(:)
    integer, intent(out) :: loc2glo_ind_B(:)
    real(chm_real4), intent(in) :: min_z, max_z
    integer, intent(in) :: col_start, col_end, nbucket, col_n(:)
    integer, intent(inout) :: bucket_n(:), bucket_start(:)
    integer, intent(out) :: cell_ind(:), atom_order(:)
    ! Functions
    integer omp_get_thread_num
    ! Variables
    real(chm_real4) cellz0, cellz1
    integer ind, istart, iend, ncellz_t, icell, i, k, kstart, kend, pos
    integer icellx, icelly

!$omp barrier
!$omp do schedule(static)
    do ind=col_start,col_end
       ! Number of cells in this z-column
       ncellz_t = ncellz(ind)
       if (ncellz_t == 0) cycle
       ! Sort atoms within a z-column, smallest z first
       istart = startatom_col(ind) + 1
       iend = istart + col_n(ind) - 1
       call zsort_tilex(xyzq_A, istart, iend, min_z, max_z, &
            nbucket, bucket_n, bucket_start, cell_ind, atom_order)
       ! atom_order(istart:iend) contains the new ordering
       xyzq_B(istart:iend) = xyzq_A(atom_order(istart:iend))
       loc2glo_ind_B(istart:iend) = loc2glo_ind_A(atom_order(istart:iend))
       ! Set the start of the z-column
       icell = startcell_col(ind) + 1
       ! Set cell_start for this z-coumn
       cell_start(icell:icell+ncellz_t-1)=(/ (i, i=istart,istart+(ncellz_t-1)*tilesize,tilesize) /)
       do k=icell,icell+ncellz_t-2
          ! Set cell_index(istart:iend) = cell index for each atom
          kstart = cell_start(k)
          kend   = cell_start(k+1) - 1
          cell_index(kstart:kend) = k
          ! Set icellxyz(1:3,icell) = cell x,y,z coordinate for each cell
          icellxyz(1,k) = icolxy(1, ind)
          icellxyz(2,k) = icolxy(2, ind)
          icellxyz(3,k) = k - icell + 1
       enddo
       kstart = cell_start(k)
       kend   = iend
       cell_index(kstart:kend) = k
       icellxyz(1,k) = icolxy(1, ind)
       icellxyz(2,k) = icolxy(2, ind)
       icellxyz(3,k) = k - icell + 1
    enddo
!$omp end do

    return
  end subroutine sort_columns_kernel
  
  ! *
  ! * Testing routine for sort_tilex
  ! *
  subroutine test_sort_tilex(xyzq, loc2glo_ind, ncoord, natom, zone_cell_start, &
       startcol_zone, ncol_tot)
    use memory
    use stream,only:outu
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: loc2glo_ind(:)
    integer, intent(in) :: ncoord, natom
    integer, intent(in) :: zone_cell_start(9), startcol_zone(8), ncol_tot
    ! Parameters
    real(chm_real4), parameter :: tol = 1.0e-5
    ! Variables
    real(chm_real4) prev_z
    real(chm_real4) xlo, ylo, zlo, xhi, yhi, zhi
    integer icellx, icelly, icellz
    integer prev_icellx, prev_icelly, prev_zone
    integer izone, i, j, icell, pos, pos2
    logical q_coordmap_ok
    integer, allocatable, dimension(:) :: coordmap
    integer istart, iend, n, ncoord_tmp

    if (ncoord > natom) then
       call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Cannot have ncoord > natom')
    endif

    ! Test zone_cell_start
    if (zone_cell_start(1) /= 1) then
       call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid zone_cell_start(1)')
    endif

    do izone=1,8
       if (zone_cell_start(izone) > zone_cell_start(izone+1)) then
          call wrndie(-5,'<nblist_tilex_sort>',&
               'test_sort_tilex: zone_cell_start is setup incorrectly')
       endif
    enddo

    if (zone_cell_start(9) /= ncell+1) then
       call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid zone_cell_start(9)')
    endif

    ! Test to make sure z-coordinates are ordered correctly
    prev_icellx = 0
    prev_icelly = 0
    do i=1,ncoord
       icell = cell_index(i)
       if (icell <= 0 .or. icell > ncell) then
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: icell out of bounds')
       endif
       icellx = icellxyz(1, icell)
       icelly = icellxyz(2, icell)
       icellz = icellxyz(3, icell)
       prev_zone = 1
       izone = get_zone_safe(zone_cell_start, icell, prev_zone)
       if (izone < 1 .or. izone > 8) then
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid izone value')
       endif
       pos = icellx + (icelly-1)*ncellx(izone) + startcol_zone(izone)
       if (pos == 0 .or. pos > ncol_tot) then
          write (outu,'(a,7i6)') 'izone,pos,ncol_tot,icellx,icelly,ncellx,startcol_zone=',&
               izone,pos,ncol_tot,&
               icellx,icelly,ncellx(izone),startcol_zone(izone)
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid pos value')
       endif
       if (icellx <= 0 .or. icelly <= 0 .or. icellz <= 0 .or. &
            icellx > ncellx(izone) .or. icelly > ncelly(izone) .or. icellz > ncellz(pos)) then
          write (outu,'(a,3i6)') 'izone,pos,zone_cell_start=',izone,pos,zone_cell_start(izone)
          write (outu,'(a,3i6)') 'icellx, icelly, icellz=',icellx, icelly, icellz
          write (outu,'(a,3i6)') 'ncellx, ncelly, ncellz=',ncellx(izone), ncelly(izone), ncellz(pos)
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid icellxyz value')
       endif
       if (prev_icellx /= icellx .or. prev_icelly /= icelly) then
          prev_z = -100000.0_chm_real4
       endif
       if (xyzq(i)%z < prev_z) then
          write (outu,'(a,2f12.4)') 'xyzq(i)%z,prev_z=',xyzq(i)%z,prev_z
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: Invalid z-ordering')
       endif
       prev_z = xyzq(i)%z
       prev_icellx = icellx
       prev_icelly = icelly
    enddo

    ! Test to make sure cell_start is setup correctly
    call chmalloc('nblist_tilex_sort.src','test_sort_tilex','coordmap',natom,intg=coordmap)
    coordmap(1:ncoord) = 0
    ncoord_tmp = 0
    do icell=1,ncell
       istart = cell_start(icell)
       iend = cell_start(icell+1) - 1
       if (istart > ncoord+1 .or. iend > ncoord) then
          write (outu,'(a,5i8)') 'icell,ncell,istart,iend,ncoord=',icell,ncell,istart,iend,ncoord
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: ncoord exceeded')
       endif
       n = iend - istart + 1
       if (n < 0) then
          write (outu,'(a,3i10)') 'icell,istart,iend=',icell,istart,iend
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: iend < istart')
       endif
       ncoord_tmp = ncoord_tmp + n
       if (n > 0) then
          coordmap(istart:iend) = coordmap(istart:iend) + 1
          ! Test cellbx, cellby, cellbz
          prev_zone = 1
          izone = get_zone(zone_cell_start, icell, prev_zone)
          icellx = icellxyz(1, icell)
          icelly = icellxyz(2, icell)
          icellz = icellxyz(3, icell)
          xlo = minval(xyzq(istart:iend)%x)
          ylo = minval(xyzq(istart:iend)%y)
          zlo = minval(xyzq(istart:iend)%z)
          xhi = maxval(xyzq(istart:iend)%x)
          yhi = maxval(xyzq(istart:iend)%y)
          zhi = maxval(xyzq(istart:iend)%z)
          pos = (max_ncellz(izone) + 1)*(icellx-1 + (icelly-1)*ncellx(izone)) + icellz
          if (xlo + tol < cellbx(izone)%array(icellx-1) .or. &
               xhi - tol > cellbx(izone)%array(icellx)) then
             write (outu,*) xlo-cellbx(izone)%array(icellx-1)
             write (outu,'(a,2f10.3)') 'xlo,cellbx=',xlo,cellbx(izone)%array(icellx-1)
             write (outu,'(a,2f10.3)') 'xhi,cellbx=',xhi,cellbx(izone)%array(icellx)
             call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: cellbx incorrectly set')
          endif
          if (ylo + tol < cellby(izone)%array(icelly-1) .or. &
               yhi - tol > cellby(izone)%array(icelly)) then
             call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: cellby incorrectly set')
          endif
          if (zlo + tol < cellbz(izone)%array(pos-1) .or. &
               zhi - tol > cellbz(izone)%array(pos)) then
             pos2 = icellx + (icelly-1)*ncellx(izone) + startcol_zone(izone)
             write (outu,'(a,i3,2i6)') 'izone,pos2,max_ncellz(izone)=',izone,pos2,&
                  max_ncellz(izone)
             write (outu,'(a,3i6)') 'icellx,icelly,icellz=',icellx,icelly,icellz
             write (outu,'(a,3i6)') 'ncellx,ncelly,ncellz=',ncellx(izone),ncelly(izone),&
                  ncellz(pos2)
             write (outu,'(a,2i6,4f10.3)') 'icell,pos,zlo,zhi,cellbz',&
                  icell,pos, zlo,zhi,cellbz(izone)%array(pos-1),cellbz(izone)%array(pos)
             call wrndie(-5,'<nblist_tilex_sort>','test_sort_tilex: cellbz incorrectly set')
          endif
       endif
    enddo

    if (ncoord_tmp /= ncoord) then
       write (outu,'(a,2i10)') 'ncoord_tmp,ncoord=',ncoord_tmp,ncoord
       call wrndie(-5,'<nblist_tilex_sort>',&
            'test_sort_tilex: Invalid number of coordinates in cell_start')
    endif

    q_coordmap_ok = (sum(coordmap(1:ncoord)) == ncoord)
    do i=1,ncoord
       if (coordmap(i) /= 1) q_coordmap_ok = .false.
    enddo
    if (.not.q_coordmap_ok) then
       call wrndie(-5,'<nblist_tilex_sort>',&
            'test_sort_tilex: Invalid cell_start')
    endif

    ! Test loc2glo_ind for negative values and values that are >= natom
    if (minval(loc2glo_ind(1:ncoord)) < 0 .or. maxval(loc2glo_ind(1:ncoord)) >= natom) then
       call wrndie(-5,'<nblist_tilex_sort>',&
            'test_sort_tilex: loc2glo_ind has invalid values')
    endif

    ! Make sure loc2glo_ind only contains each index once
    coordmap(1:natom) = 0
    do i=1,ncoord
       j = loc2glo_ind(i)+1
       if (coordmap(j) /= 0) then
          call wrndie(-5,'<nblist_tilex_sort>',&
               'test_sort_tilex: loc2glo_ind has duplicate values')
       endif
       coordmap(j) = 1
    enddo

    call chmdealloc('nblist_tilex_sort.src','test_sort_tilex','coordmap',natom,intg=coordmap)

    write (outu,'(a)') 'test_sort_tilex OK'

    return
  end subroutine test_sort_tilex

  ! *
  ! * Sorts atoms along the z coordinate
  ! *
  subroutine zsort_tilex(xyzq, istart, iend, min_z, max_z, &
       nbucket, bucket_n, bucket_start, cell_ind, atom_order)
!    use domdec_common,only:hboxz
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: istart, iend, nbucket
    real(chm_real4), intent(in) :: min_z, max_z
    integer, intent(inout) :: bucket_n(:), bucket_start(:)
    integer, intent(out) :: cell_ind(:), atom_order(:)
    ! Variables
    real(chm_real4) boxzf !, hboxzf
    real(chm_real4) z, inv_dz, z0
    integer iz, n, start0, ind, ind1, ind2
    integer order_loc(6)
    logical q_swap
    integer max_n
    integer i, j, m

!    hboxzf = hboxz

    inv_dz = real(nbucket)/(max_z - min_z + 0.001_chm_real4)
    
    bucket_n(1:nbucket) = 0

    do i=istart,iend
       z = xyzq(i)%z! + hboxzf
       ! Save z value
       !xyzq(i)%z = z
       iz = int((z-min_z)*inv_dz) + 1
       bucket_n(iz) = bucket_n(iz) + 1
       cell_ind(i) = iz
    enddo

    n = 1
    do iz=1,nbucket
       bucket_start(iz) = istart + n - 1
       n = n + bucket_n(iz)
       bucket_n(iz) = 0
    enddo
    bucket_start(nbucket+1) = n

    max_n = 0
    do i=istart,iend
       iz = cell_ind(i)
       start0 = bucket_start(iz)
       n = bucket_n(iz)

       max_n = max(max_n, n)

!!$       if (n >= 1) then
!!$          ! bucket already contains atoms => insert at correct location
!!$          ! atoms are listed in: atom_order(start0:start0+n-1)
!!$          z0 = xyzq(i)%z
!!$          !z0 = zbuf(i)
!!$          do j=1,n
!!$             ind = atom_order(start0+j-1)
!!$             if (xyzq(ind)%z > z0) exit
!!$             !if (zbuf(ind) > z0) exit
!!$          enddo
!!$          ! Shift atoms from j
!!$          atom_order(start0+j:start0+n) = atom_order(start0+j-1:start0+n-1)
!!$          ! Insert before j
!!$          atom_order(start0+j-1) = i
!!$       else
!!$          atom_order(start0) = i
!!$       endif

       atom_order(start0+n) = i

       n = n + 1
       bucket_n(iz) = n
    enddo

    if (max_n >= 1) then
       do iz=1,nbucket
          n = bucket_n(iz)
          if (n >= 1) then
             ! atom_order(start0:start0+n-1) must be re-sorted
             ! atom_order(start0) has smallest z
             start0 = bucket_start(iz)
!             order_loc(1:n) = atom_order(start0:start0+n-1)
             q_swap = .true.
             m = n
             do while (q_swap)
                q_swap = .false.
                do j=0,m-2
!                   ind1 = order_loc(1+j)
!                   ind2 = order_loc(1+j+1)
                   ind1 = atom_order(start0+j)
                   ind2 = atom_order(start0+j+1)
                   if (xyzq(ind1)%z > xyzq(ind2)%z) then
!                      order_loc(1+j) = ind2
!                      order_loc(1+j+1) = ind1
                      atom_order(start0+j) = ind2
                      atom_order(start0+j+1) = ind1
                      q_swap = .true.
                   endif
                enddo
                m = m - 1
             enddo
!             atom_order(start0:start0+n-1) = order_loc(1:n)
          endif
       enddo
    endif

    ! bucket_n(1:nbucket) = number of atoms in each bucket
    ! atom_order(istart:iend) = new order of atoms for this z-column

    return
  end subroutine zsort_tilex

  ! *
  ! * Sets cell sizes for each zone
  ! *
  subroutine set_cell_sizes(zonelist_atom, ncellx, ncelly, celldx, celldy)
    use nblist_types,only:tilesize
!    use domdec_dlb,only:get_zone_corners
    use domdec_common,only:min_xyz, max_xyz
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist_atom(8)
    integer, intent(out) :: ncellx(8), ncelly(8)
    real(chm_real4), intent(out) :: celldx(8), celldy(8)
    ! Variables
    integer izone, nstart, ncoord_zone
    real(chm_real4) xsize, ysize, zsize
    real(chm_real4) delta
    
    do izone=1,8
       if (izone > 1) then
          nstart = zonelist_atom(izone-1) + 1
       else
          nstart = 1
       endif
       ncoord_zone = zonelist_atom(izone) - nstart + 1
       ! ncoord_zone = number of atoms in this zone
       if (ncoord_zone > 0) then
          ! NOTE: we increase the cell sizes here by 0.001 to make sure no atoms drop outside cells
          xsize = max_xyz(1,izone) - min_xyz(1,izone) + 0.001_chm_real4
          ysize = max_xyz(2,izone) - min_xyz(2,izone) + 0.001_chm_real4
          zsize = max_xyz(3,izone) - min_xyz(3,izone) + 0.001_chm_real4
          delta = (xsize*ysize*zsize*tilesize/real(ncoord_zone))**(1.0/3.0)
          ncellx(izone) = max(1, int(xsize/delta))
          ncelly(izone) = max(1, int(ysize/delta))
          celldx(izone) = xsize/real(ncellx(izone))
          celldy(izone) = ysize/real(ncelly(izone))
          ! Increase ncellx and ncelly by one to account for bonded atoms outside the box
!          ncellx(izone) = ncellx(izone) ! + 1
!          ncelly(izone) = ncelly(izone) ! + 1
       else
          ncellx(izone) = 0
          ncelly(izone) = 0
          celldx(izone) = 1.0_chm_real4
          celldy(izone) = 1.0_chm_real4
       endif

    enddo

    return
  end subroutine set_cell_sizes

  ! *
  ! * Setup n_int_zone(1:8) and int_zone(1:8,1:8)
  ! * zone ordering is: I,FZ,FY,EX,FX,EZ,EY,C = 1,...8
  ! *
  subroutine set_int_zone(zonelist_atom, n_int_zone, int_zone)
    implicit none
    ! Input
    integer, intent(in) :: zonelist_atom(8)
    integer, intent(out) :: n_int_zone(8), int_zone(8,8)
    ! Parameters
    integer, parameter :: I=1,FZ=2,FY=3,EX=4,FX=5,EZ=6,EY=7,C=8
    ! Variables
    integer izone, j, nstart
    integer ncoord_zone(8), zones(5, 8)

    ! ncoord_zone(izone) = number of atoms in zone "izone"
    do izone=1,8
       if (izone > 1) then
          nstart = zonelist_atom(izone-1) + 1
       else
          nstart = 1
       endif
       ncoord_zone(izone) = zonelist_atom(izone) - nstart + 1
    enddo

    ! Setup interaction order that maximizes communication-computation overlap
    zones(1:2,I)  = (/ I, 0 /)             ! I-I
    zones(1:2,FZ) = (/ I, 0 /)             ! I-FZ
    zones(1:3,FY) = (/ I, FZ, 0 /)         ! I-FY, FZ-FY
    zones(1:2,EX) = (/ I, 0 /)             ! I-EX
    zones(1:5,FX) = (/ I, FZ, FY, EX, 0 /) ! I-FX, FZ-FX, FY-FX, EX-FX
    zones(1:3,EZ) = (/ I, FZ, 0/)          ! I-EZ, FZ-EZ
    zones(1:3,EY) = (/ I, FY, 0/)          ! I-EY, FY-EY
    zones(1:2,C)  = (/ I, 0 /)             ! I-C

    do izone=1,8
       n_int_zone(izone) = 0
       j = 1
       do while (zones(j,izone) > 0)
          if (ncoord_zone(zones(j,izone)) > 0) then
             n_int_zone(izone) = n_int_zone(izone) + 1
             int_zone(n_int_zone(izone),izone) = zones(j,izone)
          endif
          j = j + 1
       enddo
    enddo

    return
  end subroutine set_int_zone

  ! *
  ! * Sorts atoms into x-y columns
  ! * col_n(ind)   = number of atoms in column "ind"
  ! * atom_order(i) = index for atom i in the new sorted array
  ! *
  subroutine sort_xy_columns(zonelist_atom, xyzq_in, loc2glo_ind_in, ncoord, &
       max_col_n, ncol_tot, startcol_zone, col_ind, xyzq_out, loc2glo_ind_out)
    use domdec_common,only:boxx, boxy, boxz, hboxx, hboxy, hboxz, nthread, nx, ny, nz, &
         min_xyz, max_xyz
    use domdec_local_types,only:xyzq_sp_t
#ifdef _OPENMP
    use nblist_util,only:cumsum_exclusive  
#endif
    use domdec_util_gpu_mod,only:range_start, range_stop
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist_atom(8)
    type(xyzq_sp_t), intent(in) :: xyzq_in(:)
    integer, intent(in) :: loc2glo_ind_in(:)
    integer, intent(in) :: ncoord
    integer, intent(out) :: max_col_n(8), ncol_tot, startcol_zone(8)
    integer, intent(out) :: col_ind(:)
    type(xyzq_sp_t), intent(out) :: xyzq_out(:)
    integer, intent(out) :: loc2glo_ind_out(:)
    ! Functions
#ifdef _OPENMP
    integer omp_get_thread_num  
#endif
    ! Variables
    real(chm_real4) boxxf, boxyf !, hboxxf, hboxyf
    real(chm_real4) x, y, x0f, y0f
    real(chm_real4) inv_dx, inv_dy
    real(chm_real) x0, y0, z0, x1, y1, z1
#ifdef _OPENMP
    integer ind_start, ind_end  
#endif
    integer max_tmp
    integer tid
    integer ix, iy
    integer ind, n, start
    integer i, izone, k, istart, iend

    boxxf = boxx
    boxyf = boxy
!    hboxxf = hboxx
!    hboxyf = hboxy

    ! Setup ncellx(1:8), ncelly(1:8), ncellz(1:8), celldx(1:8), celldy(1:8)
    call set_cell_sizes(zonelist_atom, ncellx, ncelly, celldx, celldy)

    ! ncol_tot = total number of xy-columns
    ! startcol_zone(1:8) = starting position of xy-columns for each zone
    ncol_tot = 0
    do izone=1,8
       startcol_zone(izone) = ncol_tot
       ncol_tot = ncol_tot + ncellx(izone)*ncelly(izone)
    enddo

    ! Allocate & reallocate arrays used in this subroutine
    ! (startatom_col, atom_order, col_n, icolxy)
    call alloc_realloc(ncoord, ncol_tot)

    call range_start('loop1')

!$omp parallel private(tid, izone, istart, iend, x0, y0, z0, x1, y1, z1, x0f, y0f, inv_dx, inv_dy)
#ifndef _OPENMP
    tid = 0                     
#endif
#ifdef _OPENMP
    tid = omp_get_thread_num()  
#endif
    col_n(1:ncol_tot,tid) = 0
    do izone=1,8
       if (izone > 1) then
          istart = zonelist_atom(izone-1) + 1
       else
          istart = 1
       endif
       iend = zonelist_atom(izone)
       x0f = min_xyz(1,izone)! + hboxxf
       y0f = min_xyz(2,izone)! + hboxyf
       inv_dx = 1.0_chm_real4/celldx(izone)
       inv_dy = 1.0_chm_real4/celldy(izone)
       if (iend >= istart) then
!$omp barrier 
!$omp do schedule(static) private(i, x, y, ix, iy, ind)
          do i=istart,iend
             x = xyzq_in(i)%x! + hboxxf
             y = xyzq_in(i)%y! + hboxyf
             !xyzq_in(i)%x = x
             !xyzq_in(i)%y = y
             ix = int((x - x0f)*inv_dx)
             iy = int((y - y0f)*inv_dy)
             ind = 1 + ix + iy*ncellx(izone) + startcol_zone(izone)
             col_n(ind,tid) = col_n(ind,tid) + 1
             col_ind(i) = ind
          enddo
!$omp end do
       endif
    enddo

!$omp master
    call range_stop()
!$omp end master
!$omp master
    call range_start('loop2')
!$omp end master

    ! Reduce col_n(:,1:nthread-1) to col_n(:,0)
#ifdef _OPENMP
!$omp do schedule(static) private(ind)
    do ind=1,ncol_tot
       col_n(ind,0) = col_n(ind,0) + sum(col_n(ind,1:nthread-1))
    enddo
!$omp end do
#endif 
!$omp end parallel

    call range_stop()

    ! NOTE: col_n(1:ncol_tot,0) gives the number of atoms in each xy column

#ifdef _OPENMP
    ! startatom_col(1:ncol_tot) = atom index starting position for each column
    call range_start('cumsum_exclusive')
    call cumsum_exclusive(col_n(:,0), ncol_tot, startatom_col)
    call range_stop()

    call range_start('find max')
    ! Find maximum from col_n() separately for each zone
    do izone=1,8
       max_tmp = 0
!$omp parallel do schedule(static) private(i, ind, ix, iy) reduction(max:max_tmp)
       do i=1,ncellx(izone)*ncelly(izone)
          ind = i + startcol_zone(izone)
          max_tmp = max(max_tmp, col_n(ind,0))
          col_n(ind,0) = 0
          ! Set icolxy(1:2,ind) = column x and y coordinates
          iy = (i-1)/ncellx(izone)
          ix = i - iy*ncellx(izone)
          icolxy(1, ind) = ix
          icolxy(2, ind) = iy + 1
       enddo
!$omp end parallel do
       max_col_n(izone) = max_tmp
    enddo
    call range_stop()
#else /**/
    n = 0
    do izone=1,8
       max_col_n(izone) = 0
       do i=1,ncellx(izone)*ncelly(izone)
          ind = i + startcol_zone(izone)
          n = n + col_n(ind,0)
          startatom_col(ind) = n
          max_col_n(izone) = max(max_col_n(izone), col_n(ind,0))
          col_n(ind,0) = 0
       enddo
    enddo
#endif 

#ifdef _OPENMP
    ! NOTE: this last loop is not parallelized
    call range_start('last loop')
    do i=1,ncoord
       ind = col_ind(i)
       start = startatom_col(ind) + 1
       n = col_n(ind,0)
       atom_order(start + n) = i
       n = n + 1
       col_n(ind,0) = n
    enddo
    call range_stop()
#else /*    */
    do i=1,ncoord
       ind = col_ind(i)
       start = startatom_col(ind) + 1
       n = col_n(ind,0)
       ! atom is displaced: i -> start+n
       atom_order(start + n) = i
       n = n + 1
       col_n(ind,0) = n
    enddo
#endif 

    ! xyzq_out(1:ncoord)        = atoms sorted into xy-columns
    ! loc2glo_ind_out(1:ncoord) = local->global mapping for this sorting
!$omp parallel do schedule(static) private(i, k)
    do i=1,ncoord
       k = atom_order(i)
       xyzq_out(i) = xyzq_in(k)
       loc2glo_ind_out(i) = loc2glo_ind_in(k)
    enddo
!$omp end parallel do

    return
    
  contains

    subroutine alloc_realloc(ncoord, ncol_tot)
      use memory
#ifdef _OPENMP
      use domdec_common,only:nthread  
#endif
      implicit none
      ! Input
      integer, intent(in) :: ncoord, ncol_tot
      ! Variables
      integer len
      
      ! startatom_col
      if (allocated(startatom_col)) then
         if (size(startatom_col) < ncol_tot) then
            call chmdealloc('nblist_tilex_sort.src','sort_xy_columns','startatom_col',&
                 size(startatom_col),intg=startatom_col)
         endif
      endif

      if (.not.allocated(startatom_col)) then
         len = int(ncol_tot*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_xy_columns','startatom_col',&
              len,intg=startatom_col)
      endif

      ! atom_order
      if (allocated(atom_order)) then
         if (size(atom_order) < ncoord) then
            call chmdealloc('nblist_tilex_sort.src','sort_xy_columns','atom_order',&
                 size(atom_order),intg=atom_order)
         endif
      endif

      if (.not.allocated(atom_order)) then
         len = int(ncoord*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_xy_columns','atom_order',len,intg=atom_order)
      endif

      ! col_n
      if (allocated(col_n)) then
         if (size(col_n,1) < ncol_tot .or. size(col_n,2) < nthread) then
            call chmdealloc('nblist_tilex_sort.src','sort_xy_columns','col_n',&
                 size(col_n,1),size(col_n,2),intg=col_n)
         endif
      endif

      if (.not.allocated(col_n)) then
         len = int(ncol_tot*1.2)
         call chmalloc('nblist_tilex_sort.src','sort_xy_columns','col_n',&
              len,nthread,lbou2=0,intg=col_n)
      endif

      ! icolxy
      if (allocated(icolxy)) then
         if (size(icolxy) < ncol_tot) then
            call chmdealloc('nblist_tilex_sort.src','sort_xy_columns','icolxy',&
                 2,size(icolxy),intg=icolxy)
         endif
      endif

      if (.not.allocated(icolxy)) then
         len = int(ncol_tot*1.5)
         call chmalloc('nblist_tilex_sort.src','sort_xy_columns','icolxy',2,len,intg=icolxy)
      endif

      return
    end subroutine alloc_realloc

  end subroutine sort_xy_columns  

  ! *
  ! * Tests sort_xy_columns
  ! *
  subroutine test_sort_xy_columns(ncoord, xyzq, ncol_tot, startatom_col, startcol_zone, col_n)
    use stream,only:outu
!    use domdec_dlb,only:get_zone_corners
    use domdec_common,only:min_xyz!, hboxx, hboxy
    use domdec_local_types,only:xyzq_sp_t
    implicit none
    ! Input / Output
    integer, intent(in) :: ncoord
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    integer, intent(in) :: ncol_tot, startatom_col(:), startcol_zone(8), col_n(:)
    ! Variables
    real(chm_real) x0, y0, z0, x1, y1, z1
    real(chm_real4) x0f, y0f, inv_dx, inv_dy, x, y!, hboxxf, hboxyf
    integer ix, iy, ind
    integer i, icol, izone
    integer istart, iend

    ! max_col_n(1:8)            = maximum number of atoms in the xy-column per zone
    ! ncol_tot                  = total number of xy-columns
    ! startcol_zone(1:8)        = start of xy-columns for each zone
    ! startatom_col(1:ncol_tot) = start of atom indices for each xy-column
    ! col_n(1:ncol_tot)         = number of atoms for each xy-column
    ! xyzq_out(1:ncoord)        = sorted coordinate array

!    hboxxf = hboxx
!    hboxyf = hboxy

    icol = 1
    izone = 1
    ! Test to make sure atoms are in the correct xy-columns
    do i=1,ncoord
       ! Determine column index "icol" for atom "i"
       do while (i > startatom_col(icol+1) .and. icol < ncol_tot)
          icol = icol + 1
       enddo
       if (icol > ncol_tot) then
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: icol exceeded ncol_tot')
       endif
       ! Determine zone index "izone" for column "icol"
       do while (icol > startcol_zone(izone+1) .and. izone < 8)
          izone = izone + 1
       enddo
       if (izone > 8) then
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: izone exceeded 8')
       endif
       ! Check atom coordinates
       x0f = min_xyz(1,izone)! + hboxxf
       y0f = min_xyz(2,izone)! + hboxyf
       inv_dx = 1.0_chm_real4/celldx(izone)
       inv_dy = 1.0_chm_real4/celldy(izone)
       x = xyzq(i)%x! + hboxxf
       y = xyzq(i)%y! + hboxyf
       ix = int((x - x0f)*inv_dx)
       iy = int((y - y0f)*inv_dy)
       if (ix < 0 .or. ix >= ncellx(izone)) then
          write (outu,'(a,2i4,3f12.4)') 'ix,ncellx,x,x0f,celldx=',&
               ix,ncellx(izone),x,x0f,celldx(izone)
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Invalid ix')
       endif
       if (iy < 0 .or. iy >= ncelly(izone)) then
          write (outu,'(a,2i4,3f12.4)') 'iy,ncelly,y,y0f,celldy=',&
               iy,ncelly(izone),y,y0f,celldy(izone)
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Invalid iy')
       endif
       ind = 1 + ix + iy*ncellx(izone) + startcol_zone(izone)
       if (ind /= icol) then
          write (outu,*) 'i,ind,icol=',i,ind,icol
          write (outu,*) 'ix,iy=',ix,iy
          write (outu,*) 'x,y=',x,y
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Incorrect column index')
       endif
    enddo

    ! Test to make sure startatom_col is setup correctly
    do ind=1,ncol_tot
       istart = startatom_col(ind) + 1
       iend = istart + col_n(ind) - 1
       if (istart > ncoord+1 .or. iend > ncoord .or. istart <= 0 .or. iend-istart+1 < 0) then
          write (outu,'(a,3i8)') 'ind,istart,iend=',ind,istart,iend
          call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Invalid startatom_col/col_n')
       endif
    enddo

    ! Test icolxy
    do izone=1,8
       do i=1,ncellx(izone)*ncelly(izone)
          ind = i + startcol_zone(izone)
          ix = icolxy(1,ind)
          iy = icolxy(2,ind)
          if (ix <= 0 .or. ix > ncellx(izone)) then
             write (outu,'(a,4i4)') 'izone,i,ix,ncellx=',izone,i,ix,ncellx(izone)
             call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Invalid icolxy(1,ind)')
          endif
          if (iy <= 0 .or. iy > ncelly(izone)) then
             call wrndie(-5,'<nblist_tilex_sort>','test_sort_xy_columns: Invalid icolxy(2,ind)')
          endif
       enddo
    enddo    

    write (outu,'(a)') 'test_sort_xy_columns OK'

    return
  end subroutine test_sort_xy_columns

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
  ! * Returns zone index for cell icell assuming we are traversing cells in increasing order
  ! * NOTE: Safe slow version
  ! *
  integer function get_zone_safe(zone_cell_start, icell, prev_zone)
    use stream,only:outu
    ! Input
    integer, intent(in) :: zone_cell_start(9)
    integer, intent(in) :: icell
    integer, intent(inout) :: prev_zone

    if (prev_zone <= 0 .or. prev_zone > 8) then
       call wrndie(-5,'<nblist_tilex_sort>','get_zone_safe: prev_zone out of bounds')
    endif

    get_zone_safe = prev_zone
    
    do while (icell >= zone_cell_start(get_zone_safe+1))
       get_zone_safe = get_zone_safe + 1
       if (get_zone_safe > 8) then
          write (outu,'(a,10i4)') 'icell,zone_cell_start=',icell,zone_cell_start
          call wrndie(-5,'<nblist_tilex_sort>','get_zone_safe: zone out of bounds')
       endif
    enddo
    
    return
  end function get_zone_safe

  ! *
  ! * Deallocates all data for this module
  ! *
  subroutine uninit_nblist_tilex_sort()
    use memory,only:chmdealloc
    implicit none
    ! Variables
    integer i

    ! cellbx
    do i=1,8
       if (allocated(cellbx(i)%array)) then
          call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','cellbx(i)%array',&
               size(cellbx(i)%array),cr4=cellbx(i)%array)
       endif
    enddo

    ! cellby
    do i=1,8
       if (allocated(cellby(i)%array)) then
          call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','cellby(i)%array',&
               size(cellby(i)%array),cr4=cellby(i)%array)
       endif
    enddo

    ! cellbz
    do i=1,8
       if (allocated(cellbz(i)%array)) then
          call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','cellbz(i)%array',&
               size(cellbz(i)%array),cr4=cellbz(i)%array)
       endif
    enddo

    ! atom_cc_ind
    if (allocated(atom_cc_ind)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','atom_cc_ind',&
            size(atom_cc_ind),intg=atom_cc_ind)
    endif

    ! col_n
    if (allocated(col_n)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','col_n',&
            size(col_n,1),size(col_n,2),intg=col_n)
    endif

    if (allocated(startatom_col)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','startatom_col',&
            size(startatom_col),intg=startatom_col)
    endif

    if (allocated(atom_order)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','atom_order',&
            size(atom_order),intg=atom_order)
    endif

    if (allocated(startcell_col)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','startcell_col',&
            size(startcell_col),intg=startcell_col)
    endif

    if (allocated(bucket_n)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','bucket_n',&
            size(bucket_n,1),size(bucket_n,2),intg=bucket_n)
    endif

    if (allocated(bucket_start)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','bucket_start',&
            size(bucket_start,1),size(bucket_start,2),intg=bucket_start)
    endif

    if (allocated(zbuf)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','zbuf',&
            size(zbuf),cr4=zbuf)
    endif

    if (allocated(cell_start)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','cell_start',&
            size(cell_start),intg=cell_start)
    endif

    if (allocated(cell_index)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','cell_index',&
            size(cell_index),intg=cell_index)
    endif

    if (allocated(icellxyz)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','icellxyz',&
            3,size(icellxyz,2),intg=icellxyz)
    endif

    ! icolxy
    if (allocated(icolxy)) then
       call chmdealloc('nblist_tilex_sort.src','uninit_nblist_tilex_sort','icolxy',&
            2,size(icolxy),intg=icolxy)
    endif

    return
  end subroutine uninit_nblist_tilex_sort

#endif /* (domdec_gpu)*/

end module nblist_tilex_sort

