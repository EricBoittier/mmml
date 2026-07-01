!
! Builds non-bonded lists using a grid search algorithm
!
module nblist_builder

#if KEY_DOMDEC==1

  use chm_kinds
  use dimens_fcm
#if KEY_DOMDEC_GPU==1
  use domdec_local_types,only:xyzq_sp_t
#endif
  !-------------------------------------------------------------------------
  ! These are for nblist_builder_kernel.inc
  !-------------------------------------------------------------------------
  use memory,only:chmrealloc
  use number,only:zero, two
  use domdec_common,only:boxx, boxy, boxz
  use nblist_types,only:nblist_t, TYPE_PAIR
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  use nblist_pair,only:flush_pair_ps, flush_pair_pd!, flush_pair_atom_ps, flush_pair_atom_pd
#if KEY_BLOCK==1
  use nblist_pair,only:flush_pair_block_ps, flush_pair_block_pd
#endif
  use groupxfast,only:maxbox, group_out
  use stream,only:outu
  use domdec_dlb,only:topx, topy, topz
  !-------------------------------------------------------------------------
  implicit none
  private
  
  ! Debug flag: if .true. prints info about the nonbond list.
  ! NOTE: Affects performance, should be kept at .false. except for debugging!
  logical, parameter :: debug_ns = .false.

  ! Type definition for grid routiness
  type grid_t
     ! Cut-off
     real(chm_real) cutgrp
     integer, allocatable, dimension(:) :: grid_n, grid_start
     integer grid_nx, grid_ny, grid_nz, grid_nxy, grid_nxyz
     integer grid_nxlocal, grid_nylocal, grid_nzlocal
     real(chm_real) grid_frx, grid_fry, grid_frz
     real(chm_real), allocatable, dimension(:) :: grid_dx, grid_dy, grid_dz
     real(chm_real), allocatable, dimension(:) :: grid_bx, grid_by, grid_bz
     real(chm_real) grid_origx, grid_origy, grid_origz
     real(chm_real), allocatable, dimension(:) :: grid_xdist, grid_ydist, grid_zdist
     integer, allocatable, dimension(:) :: grid_ind
     ! grid_grps constains the list of groups for each grid entry
     ! For grid cell (ix,iy,iz) at ind = ix + iy*grid_nx + iz*grid_nxy,
     ! groups are listed in:
     ! grid_grps(grid_start(ind):grid_start(ind+1)-1)
     integer, allocatable, dimension(:) :: grid_grps
  end type grid_t

  type(grid_t) main_grid
  type(grid_t) thole_grid

  ! List of different thole IAC types
  integer ntholetypes
  integer, allocatable, dimension(:) :: tholetypes

  ! Thole type interaction matrix
  integer, allocatable, dimension(:) :: tholeij

  ! Thole non-bonded type: 1:nbthol, or 0 if not part of Thole nb
  integer, allocatable, dimension(:) :: tholenbtype

  ! List of thole atoms tholenblist(1:ntholenblist)
  integer ntholenblist
  integer, allocatable, dimension(:) :: tholenblist

  ! Local group positions and bounding boxes
  real(chm_real), allocatable, dimension(:,:) :: grouppos_loc, groupbox_loc

  ! Local group definitions
  integer, allocatable, dimension(:) :: group_loc

  ! Temporary lists for ns_grid() -subroutines
  integer, allocatable, dimension(:,:) :: tmpl1, tmpl2
  integer :: tmpl1_len = 512, tmpl2_len = 512

  ! Full topological exclusion tables
  integer, allocatable, dimension(:) :: top_excl_pos
  integer, allocatable, dimension(:) :: top_excl

  ! Public subroutines
  public uninit_nsxfast, init_nsxfast
  public ns_xfast

#endif

contains

#if KEY_DOMDEC==1

  ! *
  ! * Initializes
  ! *
  subroutine init_nsxfast()
    use memory,only:chmalloc, chmdealloc
    use psf,only:natom, qdrude, isdrude, iac
    use nbthole,only:nbthol, nbtholij
    use domdec_common,only:ndirect
    implicit none
    integer i, j, k, ij, n, len
    logical found

    ! Allocate temporary lists (tmpl1, tmpl2)
    if (.not.allocated(tmpl1)) then
       tmpl1_len = 512
       call chmalloc('nblist_builder.src','init_nsxfast','tmpl1',2,tmpl1_len,intg=tmpl1)
    endif
    if (.not.allocated(tmpl2)) then
       tmpl2_len = 512
       call chmalloc('nblist_builder.src','init_nsxfast','tmpl2',2,tmpl2_len,intg=tmpl2)
    endif

    ! tholenbtype
    if (qdrude .and. nbthol > 0) then
       ! Allocata tholenbtype
       if (allocated(tholenbtype)) then
          if (size(tholenbtype) /= natom) then
             call chmdealloc('nblist_builder.src','init_nsxfast','tholenbtype',&
                  size(tholenbtype),intg=tholenbtype)
          endif
       endif
       if (.not.allocated(tholenbtype)) then
          call chmalloc('nblist_builder.src','init_nsxfast','tholenbtype',natom,intg=tholenbtype)
       endif
       ! Count the number of different IAC in nbtholij(1:nbthol)
       if (allocated(tholetypes)) then
          if (size(tholetypes) /= nbthol) then
             call chmdealloc('nblist_builder.src','init_nsxfast','tholetypes',&
                  size(tholetypes),intg=tholetypes)
          endif
       endif
       if (.not.allocated(tholetypes)) then
          call chmalloc('nblist_builder.src','init_nsxfast','tholetypes',nbthol,intg=tholetypes)
       endif
       ntholetypes = 0
       do i=1,nbthol
          do k=1,2
             found = .false.
             do j=1,ntholetypes
                if (nbtholij(k,i) == tholetypes(j)) then
                   found = .true.
                   exit
                endif
             enddo
             if (.not.found) then
                ntholetypes = ntholetypes + 1
                tholetypes(ntholetypes) = nbtholij(k,i)
             endif
          enddo
       enddo
       ! Build interaction matrix for tholetypes
       if (allocated(tholeij)) then
          if (size(tholeij) /= ntholetypes*(ntholetypes+1)/2) then
             call chmdealloc('nblist_builder.src','init_nsxfast','tholeij',&
                  size(tholeij),intg=tholeij)
          endif
       endif
       if (.not.allocated(tholeij)) then
          call chmalloc('nblist_builder.src','init_nsxfast','tholeij',&
               ntholetypes*(ntholetypes+1)/2,intg=tholeij)
       endif
       do i=1,ntholetypes
          do j=i,ntholetypes
             ij = j*(j-3)/2 + i + j
             tholeij(ij) = 0
             do k=1,nbthol
                if ((nbtholij(1,k) == tholetypes(i) .and. nbtholij(2,k) == tholetypes(j)) .or. &
                     (nbtholij(2,k) == tholetypes(i) .and. nbtholij(1,k) == tholetypes(j))) then
                   tholeij(ij) = k
                   exit
                endif
             enddo
          enddo
       enddo
       ! build tholenbtype(1:natom)
       n = 0
       do i=1,natom
          tholenbtype(i) = 0
          if (.not.isdrude(i)) then
             do j=1,ntholetypes
                if (tholetypes(j) == iac(i)) then
                   tholenbtype(i) = j
                   n = n + 1
                   exit
                endif
             enddo
          endif
       enddo
       len = max(100, min(n, int(2*n/ndirect)))
       if (allocated(tholenblist)) then
          if (size(tholenblist) < len) then
             call chmdealloc('nblist_builder.src','init_nsxfast','tholenblist',&
                  size(tholenblist),intg=tholenblist)
          endif
       endif
       if (.not.allocated(tholenblist)) then
          call chmalloc('nblist_builder.src','init_nsxfast','tholenblist',&
               len,intg=tholenblist)
       endif

    endif

    return
  end subroutine init_nsxfast

  ! *
  ! * Initialize grid for nbonds_grid -routine
  ! *
  subroutine init_grid(rcutoff, grid)
    use memory,only:chmalloc, chmdealloc
    use number,only:half
    use stream,only:prnlev, outu
    use domdec_dlb,only:get_home_fr
    use domdec_common,only:boxx, boxy, boxz
    use groupxfast,only:ngroup
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: rcutoff
    type(grid_t), intent(inout) :: grid
    ! Variables
    real(chm_real) frxl, fryl, frzl
    integer grid_nx, grid_ny, grid_nz
    integer grid_nxlocal, grid_nylocal, grid_nzlocal
    logical print_info

    grid_nx = grid%grid_nx
    grid_ny = grid%grid_ny
    grid_nz = grid%grid_nz
    grid_nxlocal = grid%grid_nxlocal
    grid_nylocal = grid%grid_nylocal
    grid_nzlocal = grid%grid_nzlocal

    grid%cutgrp = rcutoff

    ! grid_nx = number of cells for the entire import volume
    ! grid_nxlocal = number of cells for the homebox
    call get_home_fr(frxl, fryl, frzl)
    grid%grid_nx = max(1,int(( frxl*boxx  + grid%cutgrp)/(half*grid%cutgrp)))
    grid%grid_ny = max(1,int(( fryl*boxy  + grid%cutgrp)/(half*grid%cutgrp)))
    grid%grid_nz = max(1,int(( frzl*boxz  + grid%cutgrp)/(half*grid%cutgrp)))
    grid%grid_nxlocal = max(1,int(( frxl*boxx )/(half*grid%cutgrp)))
    grid%grid_nylocal = max(1,int(( fryl*boxy )/(half*grid%cutgrp)))
    grid%grid_nzlocal = max(1,int(( frzl*boxz )/(half*grid%cutgrp)))
    if (grid%grid_nx == grid%grid_nxlocal .or. grid%grid_ny == grid%grid_nylocal .or. &
         grid%grid_nz == grid%grid_nzlocal) then
       call wrndie(-5,'<nblist_builder>','Invalid grid_nx / grid_nxlocal')
    endif

    print_info = (grid_nx /= grid%grid_nx) .or. (grid_ny /= grid%grid_ny) .or. &
         (grid_nz /= grid%grid_nz) .or. (grid_nxlocal /= grid%grid_nxlocal) .or. &
         (grid_nylocal /= grid%grid_nylocal) .or. (grid_nzlocal /= grid%grid_nzlocal)

    ! (grid_dx, grid_dy, grid_dz)
    if (allocated(grid%grid_dx)) then
       if (size(grid%grid_dx) < grid%grid_nx) then
          call chmdealloc('nblist_builder.src','init_grid','grid_dx',&
               size(grid%grid_dx),crl=grid%grid_dx)
       endif
    endif
    if (allocated(grid%grid_dy)) then
       if (size(grid%grid_dy) < grid%grid_ny) then
          call chmdealloc('nblist_builder.src','init_grid','grid_dy',&
               size(grid%grid_dy),crl=grid%grid_dy)
       endif
    endif
    if (allocated(grid%grid_dz)) then
       if (size(grid%grid_dz) < grid%grid_nz) then
          call chmdealloc('nblist_builder.src','init_grid','grid_dz',&
               size(grid%grid_dz),crl=grid%grid_dz)
       endif
    endif
    if (.not.allocated(grid%grid_dx)) then
       call chmalloc('nblist_builder.src','init_grid','grid_dx',grid%grid_nx,crl=grid%grid_dx)
    endif
    if (.not.allocated(grid%grid_dy)) then
       call chmalloc('nblist_builder.src','init_grid','grid_dy',grid%grid_ny,crl=grid%grid_dy)
    endif
    if (.not.allocated(grid%grid_dz)) then
       call chmalloc('nblist_builder.src','init_grid','grid_dz',grid%grid_nz,crl=grid%grid_dz)
    endif

    ! (grid_bx, grid_by, grid_bz)
    if (allocated(grid%grid_bx)) then
       if (size(grid%grid_bx) < grid%grid_nx+1) then
          call chmdealloc('nblist_builder.src','init_grid','grid_bx',&
               size(grid%grid_bx),crl=grid%grid_bx)
       endif
    endif
    if (allocated(grid%grid_by)) then
       if (size(grid%grid_by) < grid%grid_ny+1) then
          call chmdealloc('nblist_builder.src','init_grid','grid_by',&
               size(grid%grid_by),crl=grid%grid_by)
       endif
    endif
    if (allocated(grid%grid_bz)) then
       if (size(grid%grid_bz) < grid%grid_nz+1) then
          call chmdealloc('nblist_builder.src','init_grid','grid_bz',&
               size(grid%grid_bz),crl=grid%grid_bz)
       endif
    endif

   if (.not.allocated(grid%grid_bx)) then
       call chmalloc('nblist_builder.src','init_grid','grid_bx',&
            grid%grid_nx+1,lbou=0,crl=grid%grid_bx)
    endif
    if (.not.allocated(grid%grid_by)) then
       call chmalloc('nblist_builder.src','init_grid','grid_by',&
            grid%grid_ny+1,lbou=0,crl=grid%grid_by)
    endif
    if (.not.allocated(grid%grid_bz)) then
       call chmalloc('nblist_builder.src','init_grid','grid_bz',&
            grid%grid_nz+1,lbou=0,crl=grid%grid_bz)
    endif

    grid%grid_nxy = grid%grid_nx*grid%grid_ny
    grid%grid_nxyz = grid%grid_nxy*grid%grid_nz

    if (print_info .and. prnlev > 2) then
       write(outu,'(a,6i3)') ' NBLIST_BUILDER Allocating grid, nx,ny,nz, local nx,ny,nz=',&
            grid%grid_nx,grid%grid_ny,grid%grid_nz,&
            grid%grid_nxlocal,grid%grid_nylocal,grid%grid_nzlocal
    endif

    if (allocated(grid%grid_xdist)) then
       if (size(grid%grid_xdist) < grid%grid_nx) then
          call chmdealloc('nblist_builder.src','init_grid','grid_xdist',&
               size(grid%grid_xdist),crl=grid%grid_xdist)
       endif
    endif
    if (.not.allocated(grid%grid_xdist)) then
       call chmalloc('nblist_builder.src','init_grid','grid_xdist',grid%grid_nx,crl=grid%grid_xdist)
    endif

    if (allocated(grid%grid_ydist)) then
       if (size(grid%grid_ydist) < grid%grid_ny) then
          call chmdealloc('nblist_builder.src','init_grid','grid_ydist',&
               size(grid%grid_ydist),crl=grid%grid_ydist)
       endif
    endif
    if (.not.allocated(grid%grid_ydist)) then
       call chmalloc('nblist_builder.src','init_grid','grid_ydist',grid%grid_ny,crl=grid%grid_ydist)
    endif

    if (allocated(grid%grid_zdist)) then
       if (size(grid%grid_zdist) < grid%grid_nz) then
          call chmdealloc('nblist_builder.src','init_grid','grid_zdist',&
               size(grid%grid_zdist),crl=grid%grid_zdist)
       endif
    endif
    if (.not.allocated(grid%grid_zdist)) then
       call chmalloc('nblist_builder.src','init_grid','grid_zdist',grid%grid_nz,crl=grid%grid_zdist)
    endif

    if (allocated(grid%grid_n)) then
       if (size(grid%grid_n) < grid%grid_nxyz) then
          call chmdealloc('nblist_builder.src','init_grid','grid_n',&
               size(grid%grid_n),intg=grid%grid_n)
       endif
    endif
    if (allocated(grid%grid_start)) then
       if (size(grid%grid_start) < grid%grid_nxyz+1) then
          call chmdealloc('nblist_builder.src','init_grid','grid_start',&
               size(grid%grid_start),intg=grid%grid_start)
       endif
    endif
    if (.not.allocated(grid%grid_n)) then
       call chmalloc('nblist_builder.src','init_grid','grid_n',grid%grid_nxyz,intg=grid%grid_n)
    endif
    if (.not.allocated(grid%grid_start)) then
       call chmalloc('nblist_builder.src','init_grid','grid_start',&
            grid%grid_nxyz+1,intg=grid%grid_start)
    endif

    return
  end subroutine init_grid

  ! *
  ! * Deallocate memory allocated in init_grid
  ! *
  subroutine uninit_grid(grid)
    use memory,only:chmdealloc
    implicit none
    ! Input / Output
    type(grid_t), intent(inout) :: grid

    if (allocated(grid%grid_dx)) then
       call chmdealloc('nblist_builder.src','uninit_grid','grid_dx',&
            size(grid%grid_dx),crl=grid%grid_dx)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_dy',&
            size(grid%grid_dy),crl=grid%grid_dy)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_dz',&
            size(grid%grid_dz),crl=grid%grid_dz)
    endif

    if (allocated(grid%grid_bx)) then
       call chmdealloc('nblist_builder.src','uninit_grid','grid_bx',&
            size(grid%grid_bx),crl=grid%grid_bx)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_by',&
            size(grid%grid_by),crl=grid%grid_by)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_bz',&
            size(grid%grid_bz),crl=grid%grid_bz)
    endif

    if (allocated(grid%grid_xdist)) then
       call chmdealloc('nblist_builder.src','uninit_grid','grid_xdist',size(grid%grid_xdist),&
            crl=grid%grid_xdist)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_ydist',size(grid%grid_ydist),&
            crl=grid%grid_ydist)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_zdist',size(grid%grid_zdist),&
            crl=grid%grid_zdist)
    endif

    if (allocated(grid%grid_n)) then
       call chmdealloc('nblist_builder.src','uninit_grid','grid_n',&
            size(grid%grid_n),intg=grid%grid_n)
       call chmdealloc('nblist_builder.src','uninit_grid','grid_start',size(grid%grid_start),&
            intg=grid%grid_start)
    endif

    return
  end subroutine uninit_grid

  ! *
  ! * Reset grid
  ! *
  subroutine reset_grid(n, grid)
    use memory,only:chmalloc, chmdealloc
    use domdec_common,only:boxx, boxy, boxz, frx, fry, frz
    use domdec_dlb,only:q_load_balance, topx, topy, topz, get_home_orig
    implicit none
    ! Input / Output
    integer, intent(in) :: n
    type(grid_t), intent(inout) :: grid
    ! Variables
    integer i

    call get_home_orig(grid%grid_origx, grid%grid_origy, grid%grid_origz)

    ! Calculate grid sizes
    grid%grid_dx(1:grid%grid_nxlocal) = (topx - grid%grid_origx)/real(grid%grid_nxlocal)
    grid%grid_dy(1:grid%grid_nylocal) = (topy - grid%grid_origy)/real(grid%grid_nylocal)
    grid%grid_dz(1:grid%grid_nzlocal) = (topz - grid%grid_origz)/real(grid%grid_nzlocal)
    grid%grid_dx(grid%grid_nxlocal+1:grid%grid_nx) = &
         grid%cutgrp/real(grid%grid_nx-grid%grid_nxlocal)
    grid%grid_dy(grid%grid_nylocal+1:grid%grid_ny) = &
         grid%cutgrp/real(grid%grid_ny-grid%grid_nylocal)
    grid%grid_dz(grid%grid_nzlocal+1:grid%grid_nz) = &
         grid%cutgrp/real(grid%grid_nz-grid%grid_nzlocal)

    grid%grid_bx(0) = zero
    do i=1,grid%grid_nx
       grid%grid_bx(i) = grid%grid_bx(i-1) + grid%grid_dx(i)
    enddo

    grid%grid_by(0) = zero
    do i=1,grid%grid_ny
       grid%grid_by(i) = grid%grid_by(i-1) + grid%grid_dy(i)
    enddo

    grid%grid_bz(0) = zero
    do i=1,grid%grid_nz
       grid%grid_bz(i) = grid%grid_bz(i-1) + grid%grid_dz(i)
    enddo

    ! grid_ind
    if (allocated(grid%grid_ind)) then
       if (size(grid%grid_ind) < n) then
          call chmdealloc('nblist_builder.src','reset_grid','grid_ind',&
               size(grid%grid_ind),intg=grid%grid_ind)
       endif
    endif
    if (.not.allocated(grid%grid_ind)) then
       call chmalloc('nblist_builder.src','reset_grid','grid_ind',int(n*1.5),intg=grid%grid_ind)
    endif

    ! grid_grps
    if (allocated(grid%grid_grps)) then
       if (size(grid%grid_grps) < n) then
          call chmdealloc('nblist_builder.src','reset_grid','grid_grps',&
               size(grid%grid_grps),intg=grid%grid_grps)
       endif
    endif
    if (.not.allocated(grid%grid_grps)) then
       call chmalloc('nblist_builder.src','reset_grid','grid_grps',int(n*1.5),intg=grid%grid_grps)
    endif

    ! Null group counter
    grid%grid_n(1:grid%grid_nxyz) = 0
    
    return
  end subroutine reset_grid

  ! *
  ! * Builds grid_grps
  ! *
  subroutine build_grid_grps(n, grid)
    implicit none
    ! Input / Output
    integer, intent(in) :: n
    type(grid_t), intent(inout) :: grid
    ! Variables
    integer i, ix, iy, iz, ind, nt, start

    ind = 0
    nt = 1
    do iz=1,grid%grid_nz
       do iy=1,grid%grid_ny
          do ix=1,grid%grid_nx
             ind = ind + 1
             grid%grid_start(ind) = nt
             nt = nt + grid%grid_n(ind)
             grid%grid_n(ind) = 0
          enddo
       enddo
    enddo

    if (ind /= grid%grid_nxyz) call wrndie(-5,'<nblist_builder>','Invalid ind')
    if (nt-1 /= n) call wrndie(-5,'<nblist_builder>','Invalid n')

    grid%grid_start(ind+1) = nt

    do i=1,n
       ind = grid%grid_ind(i)
       start = grid%grid_start(ind)
       nt = grid%grid_n(ind)
       grid%grid_grps(start+nt) = i
       nt = nt + 1
       grid%grid_n(ind) = nt
    enddo

    return
  end subroutine build_grid_grps

  ! *
  ! * Builds the grid for groups
  ! *
  subroutine build_group_grid(ngroup, groupcenter, grid)
    use number,only:zero, two
    use stream,only:outu
    use domdec_dlb,only:topx, topy, topz
    implicit none
    ! Input / Output
    integer, intent(in) :: ngroup
    real(chm_real), intent(in) :: groupcenter(3,*)
    type(grid_t), intent(inout) :: grid
    ! Variables
    real(chm_real) cmx, cmy, cmz
    integer i, ix, iy, iz, ind

    call reset_grid(ngroup, grid)

    ! Assign groups to grid
    do i=1,ngroup
       cmx = groupcenter(1,i) - grid%grid_origx
       cmy = groupcenter(2,i) - grid%grid_origy
       cmz = groupcenter(3,i) - grid%grid_origz
       ix = 0
       do while (cmx >= zero .and. ix < grid%grid_nx)
          ix = ix + 1
          cmx = cmx - grid%grid_dx(ix)
       enddo
       if (cmx > zero) ix = grid%grid_nx + 1
       
       iy = 0
       do while (cmy >= zero .and. iy < grid%grid_ny)
          iy = iy + 1
          cmy = cmy - grid%grid_dy(iy)
       enddo
       if (cmy > zero) iy = grid%grid_ny + 1
       
       iz = 0
       do while (cmz >= zero .and. iz < grid%grid_nz)
          iz = iz + 1
          cmz = cmz - grid%grid_dz(iz)
       enddo
       if (cmz > zero) iz = grid%grid_nz + 1
       
       if (ix == 0) ix = 1
       if (iy == 0) iy = 1
       if (iz == 0) iz = 1

       if (ix < 1 .or. ix > grid%grid_nx .or. &
            iy < 1 .or. iy > grid%grid_ny .or. &
            iz < 1 .or. iz > grid%grid_nz) then
          write (outu,'(a,6i4)') 'ix,iy,iz=',ix,iy,iz,&
               grid%grid_nx,grid%grid_ny,grid%grid_nz
          write (outu,'(a,6f8.2)') 'cmxyz,topxyz=',cmx,cmy,cmz,topx,topy,topz
          write (outu,'(a,8f7.2)') 'grid_dx=',grid%grid_dx(1:grid%grid_nx)
          write (outu,'(a,8f7.2)') 'grid_dy=',grid%grid_dy(1:grid%grid_ny)
          write (outu,'(a,8f7.2)') 'grid_dz=',grid%grid_dz(1:grid%grid_nz)
          write (outu,'(a,3f8.2)') 'groupcenter=',groupcenter(1:3,i)
          write (outu,'(a,3f8.2)') 'grid_origx,grid_origy,grid_origz=',&
               grid%grid_origx,grid%grid_origy,grid%grid_origz
          call wrndie(-5,'<nblist_builder>','outside the grid')
       endif
       ind = ix + (iy-1)*grid%grid_nx + (iz-1)*grid%grid_nxy
       grid%grid_n(ind) = grid%grid_n(ind) + 1
       grid%grid_ind(i) = ind
    enddo

    call build_grid_grps(ngroup, grid)

    return
  end subroutine build_group_grid

  ! *
  ! * Builds tholenblist(1:ntholenblist) = local atom indices
  ! *
  subroutine build_tholenblist(natoml, atoml, glo2loc_ind, ntholenblist, tholenblist)
    use memory,only:chmrealloc
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml, atoml(:), glo2loc_ind(:)
    integer, intent(inout) :: ntholenblist
    integer, allocatable, dimension(:), intent(inout) :: tholenblist
    ! Variables
    integer ii, i

    ntholenblist = 0
    do ii=1,natoml
       i = atoml(ii)
       if (tholenbtype(i) > 0) then
          ntholenblist = ntholenblist + 1
          if (size(tholenblist) < ntholenblist) then
             call chmrealloc('nblist_builder.src','build_tholenblist','tholenblist',&
                  int(ntholenblist*1.5),intg=tholenblist)
          endif
          tholenblist(ntholenblist) = glo2loc_ind(i)+1
       endif
    enddo

    return
  end subroutine build_tholenblist

  ! *
  ! * Converts index i to grid index (ix, iy, iz)
  ! *
  subroutine index2gridindex(grid, i, ix, iy, iz)
    implicit none
    ! Input / Output
    type(grid_t), intent(in) :: grid
    integer, intent(in) :: i
    integer, intent(out) :: ix, iy, iz
    ! Variables
    integer ind

    ind = grid%grid_ind(i)
    iz = ind/grid%grid_nxy
    ind = ind - iz*grid%grid_nxy
    iy = ind/grid%grid_nx
    ind = ind - iy*grid%grid_nx
    ix = ind
    
    iy = iy + 1
    iz = iz + 1

    return
  end subroutine index2gridindex

!!$  ! *
!!$  ! * Gets cell bounds for neighbor searching
!!$  ! *
!!$  ! * gix = grid index of the group with center at x
!!$  ! * bx = grid boundaries
!!$  ! * nx = dimension of the grid
!!$  ! * rsq = squared cell edge distance from x
!!$  ! *
!!$  subroutine get_cell_bounds(zonei, gix, x, bx, rc, nx, nxlocal, ix0, ix1, rsq)
!!$    use number
!!$    implicit none
!!$    ! Input / Output
!!$    integer, intent(in) :: zonei, gix, nx, nxlocal
!!$    real(chm_real), intent(in) :: x, bx(*), rc
!!$    integer, intent(out) :: ix0, ix1
!!$    real(chm_real), intent(out) :: rsq(*)
!!$    ! Variables
!!$    integer nx_min
!!$    real(chm_real) r
!!$
!!$    if (gix < 1) then
!!$!       ixx0 = 0
!!$!       ixx1 = 1
!!$       nx_min = 1
!!$       ix0 = 1
!!$       ix1 = 0
!!$       rsq(1) = zero
!!$    elseif (gix > nx) then
!!$!       ixx0 = nx
!!$!       ixx1 = nx + 1
!!$       nx_min = 1
!!$       ix0 = nx + 1
!!$       ix1 = nx
!!$       rsq(nx) = zero
!!$    else
!!$       rsq(gix) = zero
!!$       nx_min = 1
!!$       ix0 = gix
!!$       ix1 = gix
!!$!       if (nxlocal == 0) then
!!$!       elseif (zonei /= 1 .and. gix > nxlocal) then
!!$!          ! group is outside home box => only look outside the home box
!!$!          nx_min = nxlocal-1
!!$!       endif
!!$    endif
!!$
!!$    ! ix0, ix1 contain the initial guesses for the lower and higher bounds,
!!$    ! now check them:
!!$
!!$    ! Check cells up: from ix1 to nx
!!$    do while (ix1 < nx)
!!$       r = bx(ix1) - x
!!$       if (r > rc) exit
!!$       ix1 = ix1 + 1
!!$       rsq(ix1) = r
!!$    enddo
!!$
!!$    ! Check cells down: from ix0 to nx_min
!!$    do while (ix0 > nx_min)
!!$       ix0 = ix0 - 1
!!$       r = x - bx(ix0)
!!$       if (r > rc) exit
!!$       rsq(ix0) = r
!!$    enddo
!!$
!!$    return
!!$  end subroutine get_cell_bounds

  ! *
  ! * Gets cell bounds for neighbor searching
  ! * gix = grid index of the group with center at x
  ! * bx = grid boundaries
  ! * nx = dimension of the grid
  ! * rsq = cell edge distance from x
  ! *
  ! * Returns:
  ! * ix0 = low bound
  ! * ix1 = high bound
  ! *
  subroutine get_cell_bounds(gix, x, bx, rc, nx, nxlocal, ix0, ix1, rsq)
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: gix, nx, nxlocal
    real(chm_real), intent(in) :: x, bx(0:*), rc
    integer, intent(out) :: ix0, ix1
    real(chm_real), intent(out) :: rsq(:)
    ! Variables
    integer i, ixx0, ixx1
    real(chm_real) r

    if (gix < 1) then
       ixx0 = 0
       ixx1 = 1
       ix0 = 1
       ix1 = 0
       rsq(1) = zero
    elseif (gix > nx) then
       ixx0 = nx
       ixx1 = nx + 1
       ix0 = nx + 1
       ix1 = nx
       rsq(nx) = zero
    else
       rsq(gix) = zero
!!$       if (nxlocal == 0) then
          ixx0 = gix - 1
          ixx1 = gix + 1
          ix0 = gix
          ix1 = gix
!!$       else
!!$          if (gix <= nxlocal) then
!!$             ! group is within home box grid
!!$             
!!$          else
!!$          endif
!!$       endif
    endif

    do i=ixx1,nx
       r = bx(i-1) - x
       if (r > rc) exit
       rsq(i) = max(zero, r)
       ix1 = i
    enddo

    do i=ixx0,1,-1
       r = x - bx(i)
       if (r > rc) exit
       rsq(i) = max(zero, r)
       ix0 = i
    enddo

    return
  end subroutine get_cell_bounds

  ! *
  ! * Builds a full exclusion table from (inb14, iblo14)
  ! *
  subroutine build_full_excl_table(natom, inb14, iblo14)
    use stream,only:outu
    use memory
    use domdec_common,only:q_inb_changed
    implicit none
    ! Input / Output
    integer, intent(in) :: natom, inb14(:), iblo14(:)
    ! Variables
    integer, allocatable, dimension(:) :: nexcl
    integer i, j, ni, nj, excl_i, excl_start, excl_end
    integer pos, pos_starti, pos_startj

    if (.not.allocated(top_excl_pos)) q_inb_changed = .true.

    if (.not.q_inb_changed) return

    call chmalloc('nblist_builder.src','build_full_excl_table','nexcl',&
         natom,intg=nexcl)

    if (allocated(top_excl_pos)) then
       if (size(top_excl_pos) /= natom+1) then
          call chmdealloc('nblist_builder.src','build_full_excl_table','top_excl_pos',&
               size(top_excl_pos),intg=top_excl_pos)
       endif
    endif
    if (.not.allocated(top_excl_pos)) then
       call chmalloc('nblist_builder.src','build_full_excl_table','top_excl_pos',&
            natom+1,intg=top_excl_pos)
    endif

    ! Count the number of exclusions to nexcl(1:natom)
    nexcl(1:natom) = 0
    do i=1,natom
       if (i > 1) then
          excl_start = iblo14(i-1) + 1
       else
          excl_start = 1
       endif
       excl_end = iblo14(i)
       nexcl(i) = nexcl(i) + excl_end - excl_start + 1
       do excl_i=excl_start,excl_end
          j = iabs(inb14(excl_i))
          ! add i-j exclusion to atom j
          nexcl(j) = nexcl(j) + 1
       enddo
    enddo

    ! Setup top_excl_pos(1:natom+1):
    ! top_excl_pos(i) = starting position of exclusions for atom i
    ! Atom i has exclusions:
    ! top_excl(top_excl_pos(i):top_excl_pos(i+1)-1)
    pos = 1
    do i=1,natom
       top_excl_pos(i) = pos
       pos = pos + nexcl(i)
    enddo
    top_excl_pos(i) = pos

    if (allocated(top_excl)) then
       if (size(top_excl) /= pos) then
          call chmdealloc('nblist_builder.src','build_full_excl_table','top_excl',&
               size(top_excl),intg=top_excl)
       endif
    endif
    if (.not.allocated(top_excl)) then
       call chmalloc('nblist_builder.src','build_full_excl_table','top_excl',&
            pos,intg=top_excl)
    endif

    nexcl(1:natom) = 0

    do i=1,natom

       if (i > 1) then
          excl_start = iblo14(i-1) + 1
       else
          excl_start = 1
       endif
       excl_end = iblo14(i)

       pos_starti = top_excl_pos(i)
       ni = nexcl(i)

       do excl_i=excl_start,excl_end
          j = iabs(inb14(excl_i))
          ! Add i-j exclusion to atom j
          pos_startj = top_excl_pos(j)
          nj = nexcl(j)
          if (pos_startj + nj > top_excl_pos(j+1)-1) then
             write (outu,*) 'overflow in j'
             stop
          endif
          top_excl(pos_startj + nj) = i
          nj = nj + 1
          nexcl(j) = nj
          ! Add i-j exclusion to atom i
          if (pos_starti + ni > top_excl_pos(i+1)-1) then
             write (outu,*) 'overflow in i'
             stop
          endif
          top_excl(pos_starti + ni) = j
          ni = ni + 1
       enddo

       nexcl(i) = ni
       
    enddo

    call chmdealloc('nblist_builder.src','build_full_excl_table','nexcl',&
         natom,intg=nexcl)

    ! Sort exclusion table
    do i=1,natom
       excl_start = top_excl_pos(i)
       excl_end = top_excl_pos(i+1) - 1
       ! Loop through topological exclusions:
       ! Atom i has exclusions top_excl(excl_start:excl_end)
       do excl_i=excl_start,excl_end
          call qsort_ints(top_excl(excl_start:excl_end), excl_end-excl_start+1)
       enddo
    enddo

    q_inb_changed = .false.

    return
  end subroutine build_full_excl_table

  ! *
  ! * Simple wrapping routine for neighbor search
  ! *
  subroutine ns_xfast(nnnb, nblist, x, y, z, inb14, iblo14, cmpltd, atsx, atsy, atsz)
    use domdec_common,only:set_box, zonelist, zonelist_atom, groupl, natoml_tot, atoml, &
         q_sort_groups, q_single, q_test
    use domdec_local,only:glo2loc_ind, loc2glo_ind
    use nblist_types,only:nblist_t, TYPE_PAIR, nblist_cluster
    use domdec_local,only:&
#if KEY_DOMDEC_GPU==1
         loc2glo_ind, glo2loc_ind, &
#endif
         xyzq_loc
    use groupxfast,only:groupcenter, groupsh, groupbox, group, group_out, maxgrp_rad
    use nblist_util,only:pack_grouppos, pack_groups, &
#if KEY_DOMDEC_GPU==1
         init_array_gpu, &
#endif
         init_array
    use psf,only:natom, qdrude
    use nbthole,only:tholcut, nbthol, nbtholij, nbtholp, nbthol1, nbthol2, nbthol3
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu, gpu_code_version
    use nblist_tilex,only:ns_tilex, print_info_tilex
    use domdec_util_gpu_mod,only:range_start, range_stop, build_neighborlist_on_gpu
#endif
    use inbnd,only:cutnb, ctofnb
#if KEY_BLOCK==1
    use lambdam,only:qmld
#endif
    implicit none
    ! Input / Output
    integer, intent(out) :: nnnb
    type(nblist_t), intent(inout) :: nblist
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(out) :: atsx(*), atsy(*), atsz(*)
    integer, intent(in) :: inb14(:), iblo14(:)
    logical, intent(out) :: cmpltd
    ! Variables
    integer i, ig, j, n
    integer is, iq
    logical ok

    cmpltd = .false.

    call set_box()

    ! Save current homebox coordinates to atsx, atsy, atsz
    ! NOTE: these are in unpacked coordinates
!$omp parallel do schedule(static) private(ig, i, is, iq)
    do ig=1,zonelist(1)
       i = groupl(ig)
       call group_out(group(i), is, iq)
       atsx(is:iq) = x(is:iq)
       atsy(is:iq) = y(is:iq)
       atsz(is:iq) = z(is:iq)
    enddo
!$omp end parallel do

    ! Build full exclusion table (only done when inb14 and iblo14 have changed or at first call)
    call build_full_excl_table(natom, inb14, iblo14)

#if KEY_DOMDEC_GPU==1
    if (q_gpu) then
       if (gpu_code_version == 2) then
          call build_neighborlist_on_gpu(1, zonelist_atom, cutnb, boxx, boxy, boxz)
          call build_neighborlist_on_gpu(2, zonelist_atom, cutnb, boxx, boxy, boxz)
       else
          call ns_tilex(nblist%tilex1, nblist%tilex2, xyzq_loc%sp, loc2glo_ind, glo2loc_ind, &
               natoml_tot, top_excl_pos, top_excl)
       endif
    else
#endif

       if (.not.nblist%q_pack) call wrndie(-5,'<nblist_builder>','Must use packed coordinates')

       ! Allocate & reallocate arrays
       call init_array(grouppos_loc, 3, zonelist(8))
       call init_array(groupbox_loc, 3, zonelist(8))
       call init_array(group_loc, zonelist(8))

       ! Pack groups to group_loc
       if (q_single) then
          call pack_groups(zonelist(8), groupl, group, group_loc)
       else
          call pack_groups(zonelist(8), groupl, group, group_loc)
       endif
       
       call init_grid(cutnb+two*maxgrp_rad, main_grid)
       
       ! Pack and add: grouppos_loc = groupcenter + groupsh*box_size
       call pack_grouppos(zonelist(8), groupl, groupcenter, groupsh, &
            groupbox, grouppos_loc, groupbox_loc)

#if KEY_BLOCK==1
       if (qmld) then
          if (q_single) then
             call ns_grid_block_ps(nnnb, nblist, zonelist, grouppos_loc, groupbox_loc, &
                  xyzq_loc%sp, groupl, group_loc, group, zonelist(8), &
                  top_excl_pos, top_excl, cutnb+two*maxgrp_rad, cutnb, ctofnb, main_grid)
          else
             call ns_grid_block_pd(nnnb, nblist, zonelist, grouppos_loc, groupbox_loc, &
                  xyzq_loc%dp, groupl, group_loc, group, zonelist(8), &
                  top_excl_pos, top_excl, cutnb+two*maxgrp_rad, cutnb, ctofnb, main_grid)
          endif
       else
#endif
          if (q_single) then
             call ns_grid_ps(nnnb, nblist, zonelist, grouppos_loc, groupbox_loc, &
                  xyzq_loc%sp, groupl, group_loc, group, zonelist(8), &
                  top_excl_pos, top_excl, cutnb+two*maxgrp_rad, cutnb, ctofnb, main_grid)
          else
             call ns_grid_pd(nnnb, nblist, zonelist, grouppos_loc, groupbox_loc, &
                  xyzq_loc%dp, groupl, group_loc, group, zonelist(8), &
                  top_excl_pos, top_excl, cutnb+two*maxgrp_rad, cutnb, ctofnb, main_grid)
          endif
#if KEY_BLOCK==1
       endif
#endif

       if (qdrude .and. nbthol > 0) then
          call init_grid(sqrt(tholcut)+two*maxgrp_rad, thole_grid)
          if (q_single) then
             call ns_grid_thole_ps(nbtholp, nbthol1, nbthol2, nbthol3, &
                  natoml_tot, atoml, zonelist_atom, glo2loc_ind, loc2glo_ind, &
                  xyzq_loc%sp, sqrt(tholcut)+two*maxgrp_rad, sqrt(tholcut), thole_grid)
          else
             call ns_grid_thole_pd(nbtholp, nbthol1, nbthol2, nbthol3, &
                  natoml_tot, atoml, zonelist_atom, glo2loc_ind, loc2glo_ind, &
                  xyzq_loc%dp, sqrt(tholcut)+two*maxgrp_rad, sqrt(tholcut), thole_grid)
          endif
       endif

       if (debug_ns) call print_debug_info(nblist)

#if KEY_DOMDEC_GPU==1
    endif
#endif

    cmpltd = .true.

    return
  end subroutine ns_xfast

  ! *
  ! * Deallocates memory allocated by ns_xfast
  ! *
  subroutine uninit_nsxfast()
    use memory,only:chmdealloc
#if KEY_DOMDEC_GPU==1
    use nblist_util,only:dealloc_gpu
#endif
    implicit none

    call uninit_grid(main_grid)
    call uninit_grid(thole_grid)

    if (allocated(grouppos_loc)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','grouppos_loc',&
            size(grouppos_loc,1),size(grouppos_loc,2),crl=grouppos_loc)
    endif
    if (allocated(groupbox_loc)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','groupbox_loc',&
            size(groupbox_loc,1),size(groupbox_loc,2),crl=groupbox_loc)
    endif
    if (allocated(group_loc)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','group_loc',&
            size(group_loc),intg=group_loc)
    endif

    if (allocated(tmpl1)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tmpl1',2,size(tmpl1,2),intg=tmpl1)
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tmpl2',2,size(tmpl2,2),intg=tmpl2)
    endif

    if (allocated(top_excl_pos)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','top_excl_pos',&
            size(top_excl_pos),intg=top_excl_pos)
    endif

    if (allocated(top_excl)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','top_excl',&
            size(top_excl),intg=top_excl)
    endif

    if (allocated(tholetypes)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tholetypes',&
            size(tholetypes),intg=tholetypes)
    endif

    if (allocated(tholeij)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tholeij',&
            size(tholeij),intg=tholeij)
    endif

    if (allocated(tholenbtype)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tholenbtype',&
            size(tholenbtype),intg=tholenbtype)
    endif

    if (allocated(tholenblist)) then
       call chmdealloc('nblist_builder.src','uninit_nsxfast','tholenblist',&
            size(tholenblist),intg=tholenblist)
    endif

    return
  end subroutine uninit_nsxfast

  !-------- Double precision -----------
#define NS_GRID
#define DOUBLE_PREC
#define KERNEL_NAME ns_grid_pd
#include "nblist_builder_kernel.inc"
#undef NS_GRID
#undef DOUBLE_PREC
#undef KERNEL_NAME

#define NS_GRID
#define BLOCK_ON
#define DOUBLE_PREC
#define KERNEL_NAME ns_grid_block_pd
#include "nblist_builder_kernel.inc"
#undef NS_GRID
#undef DOUBLE_PREC
#undef KERNEL_NAME
#undef BLOCK_ON

#define NS_GRID_THOLE
#define DOUBLE_PREC
#define KERNEL_NAME ns_grid_thole_pd
#include "nblist_builder_kernel.inc"
#undef NS_GRID_THOLE
#undef DOUBLE_PREC
#undef KERNEL_NAME

  !-------- Single precision -----------
#define NS_GRID
#define SINGLE_PREC
#define KERNEL_NAME ns_grid_ps
#include "nblist_builder_kernel.inc"
#undef NS_GRID
#undef SINGLE_PREC
#undef KERNEL_NAME

#define NS_GRID
#define BLOCK_ON
#define SINGLE_PREC
#define KERNEL_NAME ns_grid_block_ps
#include "nblist_builder_kernel.inc"
#undef NS_GRID
#undef SINGLE_PREC
#undef KERNEL_NAME
#undef BLOCK_ON

#define NS_GRID_THOLE
#define SINGLE_PREC
#define KERNEL_NAME ns_grid_thole_ps
#include "nblist_builder_kernel.inc"
#undef NS_GRID_THOLE
#undef SINGLE_PREC
#undef KERNEL_NAME

!!$  ! *
!!$  ! * Constructs the non-bonded list using the grid algorithm
!!$  ! * NOTE: groupcenter, groupbox, group are in local indices
!!$  ! *       group_glo gives the global group definition
!!$  ! *
!!$##EXPAND B0 block_on block_off .when. EXPAND (expand_block)
!!$##PASS1 B0 block_off
!!$##PASS2 B1 block_on .when. BLOCK
!!$##EXEND
!!$
!!$##EXPAND P0 sp dp .when. EXPAND (expand_precision)
!!$##PASS1 PS sp
!!$##PASS2 PD dp
!!$##EXEND
!!$
!!$##IF DOMDEC (domdec)
!!$  subroutine ns_grid_{B*}_{P*}(nnnb, nblist, groupcenter, groupbox, xyzq, &
!!$       groupl, group, group_glo, ngroup, top_excl_pos, top_excl)
!!$    use memory,only:chmalloc, chmrealloc
!!$    use number,only:zero, two
!!$    use inbnd,only:cutnb, ctofnb
!!$    use domdec_common,only:zonelist, boxx, boxy, boxz, nx, ny, nz
!!$    use nblist_types,only:nblist_t, TYPE_PAIR
!!$    use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
!!$    use nblist_pair,only:flush_pair_{P*}
!!$##IF block_on
!!$    use nblist_pair,only:flush_pair_block_{P*}
!!$##ENDIF
!!$    use groupxfast,only:maxgrp_rad, maxbox, group_out
!!$    implicit none
!!$    ! Input / Output
!!$    integer nnnb
!!$    type(nblist_t) nblist
!!$    real(chm_real), intent(in) :: groupcenter(3,*), groupbox(3,*)
!!$##IF dp
!!$    type(xyzq_dp_t), intent(in) :: xyzq(*)
!!$##ELSE
!!$    type(xyzq_sp_t), intent(in) :: xyzq(*)
!!$##ENDIF
!!$    integer, intent(in) :: groupl(:), group(:), group_glo(:)
!!$    integer, intent(in) :: ngroup, top_excl_pos(:), top_excl(:)
!!$    ! Variables
!!$    integer i, j, k, klo, khi, n, ii, is, iq, itype, ish, is_unpack, iq_unpack
!!$    integer ix0, iy0, iz0, ix1, iy1, iz1
!!$    integer gimx, gimy, gimz, icx, icy, icz, ind, imx, imy, imz
!!$    integer imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
!!$    integer zonei, jg_min, jg_max, jg_lomax, jglo, jghi
!!$    integer nxi, nximax, np
!!$    real(chm_real) xi, yi, zi, xj, yj, zj, xij, yij, zij
!!$    real(chm_real) rsq, cut1sq, cut2sq, cutcell
!!$    real(chm_real) xim, yim, zim, ximmax, yimmax, zimmax
!!$    real(chm_real) shxyz(3)
!!$    real(chm_real) celld1, celld2
!!$    integer ntmpl1, ntmpl2
!!$    !
!!$    integer icx_ind(5), icy_ind(5), icz_ind(5), ic_ind(5*5*5)
!!$    integer itx, ity, itz
!!$    integer ncx, ncy, ncz
!!$
!!$    cutcell = cutnb + two*maxgrp_rad
!!$    ! cut1sq < cut2sq
!!$    cut1sq = ctofnb*ctofnb
!!$    cut2sq = cutnb*cutnb
!!$
!!$    if (nblist%type == TYPE_PAIR) then
!!$       nblist%pair(1:nblist%n)%ni = 0
!!$       nblist%pair(1:nblist%n)%nj = 0
!!$    endif
!!$
!!$    ! Allocate temporary list
!!$    if (.not.allocated(tmpl1)) then
!!$       tmpl1_len = 512
!!$       tmpl2_len = 512
!!$       call chmalloc('nblist_builder.src','ns_grid','tmpl1',2,tmpl1_len,intg=tmpl1)
!!$       call chmalloc('nblist_builder.src','ns_grid','tmpl2',2,tmpl2_len,intg=tmpl2)
!!$    endif
!!$
!!$    imx_lo = 0
!!$    imx_hi = 0
!!$    imy_lo = 0
!!$    imy_hi = 0
!!$    imz_lo = 0
!!$    imz_hi = 0
!!$    if (nx == 1) then
!!$       imx_lo = -1
!!$       imx_hi = 1
!!$    endif
!!$    if (ny == 1) then
!!$       imy_lo = -1
!!$       imy_hi = 1
!!$    endif
!!$    if (nz == 1) then
!!$       imz_lo = -1
!!$       imz_hi = 1
!!$    endif
!!$
!!$    call build_grid(ngroup, groupcenter)
!!$
!!$    if (nblist%type == TYPE_PAIR) then
!!$       nblist%pair(1:nblist%n)%ni = 0
!!$       nblist%pair(1:nblist%n)%nj = 0
!!$       nblist%pair(1:nblist%n)%np = 0
!!$    endif
!!$
!!$    zonei = 1
!!$    do i=1,ngroup
!!$
!!$       call get_j_range(zonei, i, ngroup, jg_min, jg_max, jg_lomax)
!!$       ! .not.(I or FX or FY or FZ)
!!$       if (zonei /= 1 .and. zonei /= 2 .and. zonei /= 3 .and. zonei /= 5) cycle
!!$
!!$       ! Global group is defined in group_glo(groupl(i))
!!$       ! Local group is defined in group(i)
!!$       call group_out(group(i), is, iq, itype)
!!$       call group_out(group_glo(groupl(i)), is_unpack, iq_unpack)
!!$
!!$       ! Convert group index to grid index
!!$       call group2grid(i, gimx, gimy, gimz)
!!$
!!$       ! Get atom group center coordinates
!!$       xi = groupcenter(1,i)
!!$       yi = groupcenter(2,i)
!!$       zi = groupcenter(3,i)
!!$
!!$       ! Loop over images, max. number of images = 3*3*3=27.
!!$       ! For serial: done in every direction
!!$       ! For domdec: done in every direction that has only one sub-box (i.e. where nx,ny,nz = 1)
!!$       do imx=imx_lo,imx_hi
!!$          xim = xi + imx*boxx
!!$          call get_cell_bounds(gimx+imx*grid_nx, xim-grid_origx, &
!!$               grid_bx, cutcell, grid_nx, grid_nxlocal, ix0, ix1, grid_xdist)
!!$          if (ix0 > ix1) cycle
!!$          do imy=imy_lo,imy_hi
!!$             yim = yi + imy*boxy
!!$             call get_cell_bounds(gimy+imy*grid_ny, yim-grid_origy, &
!!$                  grid_by, cutcell, grid_ny, grid_nylocal, iy0, iy1, grid_ydist)
!!$             if (iy0 > iy1) cycle
!!$             do imz=imz_lo,imz_hi
!!$                zim = zi + imz*boxz
!!$                call get_cell_bounds(gimz+imz*grid_nz, zim-grid_origz, &
!!$                     grid_bz, cutcell, grid_nz, grid_nzlocal, iz0, iz1, grid_zdist)
!!$                if (iz0 > iz1) cycle
!!$
!!$                ! ish = shift index = 1...26*3+1
!!$                ish = (imx+1 + (imy+1)*3 + (imz+1)*9 + 1)*3 - 2
!!$                ! Empty temporary list
!!$                ntmpl1 = 0
!!$                ntmpl2 = 0
!!$                ! Loop over the cells
!!$
!!$                do icz=iz0,iz1
!!$                   celld1 = max(grid_zdist(icz) - groupbox(3,i) - maxbox(3),zero)**2
!!$                   do icy=iy0,iy1
!!$                      celld2 = celld1 + max(grid_ydist(icy) - groupbox(2,i) - maxbox(2),zero)**2
!!$                      if (celld2 > cut2sq) cycle
!!$                      do icx=ix0,ix1
!!$                         if (celld2 + max(grid_xdist(icx) - groupbox(1,i) - maxbox(1),zero)**2 > &
!!$                              cut2sq) cycle
!!$                         ind = icx + (icy-1)*grid_nx + (icz-1)*grid_nxy
!!$                         n = grid_n(ind)
!!$                         ! Check wether entire cell is out of range
!!$                         ! grid_grps(grid(ind)%start)     = minimum j-group index
!!$                         ! grid_grps(grid(ind+1)%start-1) = maximum j-group index
!!$                         if (n == 0) cycle
!!$                         klo = grid_start(ind)
!!$                         khi = grid_start(ind+1)-1
!!$                         jglo = grid_grps(klo)
!!$                         jghi = grid_grps(khi)
!!$
!!$                         if ((jghi < jg_min .or. jglo > jg_max) .and. jglo > jg_lomax) cycle
!!$                         do k=klo,khi
!!$                            j = grid_grps(k)
!!$                            if ((j < jg_min .or. j > jg_max) .and. j > jg_lomax) cycle
!!$                            if (zonei == 3) then
!!$                               ! FY
!!$                               if (j > zonelist(2) .and. j <= zonelist(6)) cycle
!!$                            endif
!!$                            xij = abs(xim - groupcenter(1,j))
!!$                            yij = abs(yim - groupcenter(2,j))
!!$                            zij = abs(zim - groupcenter(3,j))
!!$                            xij = max(xij-groupbox(1,i)-groupbox(1,j),zero)
!!$                            yij = max(yij-groupbox(2,i)-groupbox(2,j),zero)
!!$                            zij = max(zij-groupbox(3,i)-groupbox(3,j),zero)
!!$                            rsq = xij*xij + yij*yij + zij*zij
!!$                            ! Add to temporary list, extend temporary list size if neccessary
!!$                            ! tmpl(1,:) = local group index
!!$                            ! tmpl(2,:) = global group index
!!$                            if (rsq < cut1sq) then
!!$                               ntmpl1 = ntmpl1 + 1
!!$                               if (ntmpl1 > tmpl1_len) then
!!$                                  tmpl1_len = int(1.5*tmpl1_len)
!!$                                  call chmrealloc('nblist_builder.src','ns_grid','tmpl1',&
!!$                                       2,tmpl1_len,intg=tmpl1)
!!$                               endif
!!$                               tmpl1(1,ntmpl1) = j
!!$                               tmpl1(2,ntmpl1) = groupl(j)
!!$                            elseif (rsq < cut2sq) then
!!$                               ntmpl2 = ntmpl2 + 1
!!$                               if (ntmpl2 > tmpl2_len) then
!!$                                  tmpl2_len = int(1.5*tmpl2_len)
!!$                                  call chmrealloc('nblist_builder.src','ns_grid','tmpl2',&
!!$                                       2,tmpl2_len,intg=tmpl2)
!!$                               endif
!!$                               tmpl2(1,ntmpl2) = j
!!$                               tmpl2(2,ntmpl2) = groupl(j)
!!$                            endif
!!$                         enddo ! do k=klo,khi
!!$                      enddo  ! do icz=iz0,iz1
!!$                   enddo  ! do icy=iy0,iy1
!!$                enddo  ! do icx=ix0,ix1
!!$                ! Flush the temporary lists
!!$                if (ntmpl1 > 0) then
!!$##IF block_on
!!$                   call flush_pair_block_{P*}(i, xyzq, top_excl_pos, top_excl, is, iq, &
!!$                        itype, is_unpack, ish, ntmpl1, tmpl1, &
!!$                        nblist%pair, group, group_glo)
!!$##ELSE
!!$                   call flush_pair_{P*}(i, xyzq, top_excl_pos, top_excl, is, iq, &
!!$                        itype, is_unpack, ish, ntmpl1, tmpl1, &
!!$                        nblist%pair, group, group_glo)
!!$##ENDIF
!!$                endif
!!$                if (ntmpl2 > 0) then
!!$##IF block_on
!!$                   call flush_pair_block_{P*}(i, xyzq, top_excl_pos, top_excl, is, iq, &
!!$                        itype, is_unpack, ish, ntmpl2, tmpl2, &
!!$                        nblist%pair, group, group_glo)
!!$##ELSE
!!$                   call flush_pair_{P*}(i, xyzq, top_excl_pos, top_excl, is, iq, &
!!$                        itype, is_unpack, ish, ntmpl2, tmpl2, &
!!$                        nblist%pair, group, group_glo)
!!$##ENDIF
!!$                endif
!!$             enddo ! do imz=-1,1
!!$          enddo ! do imy=-1,1
!!$       enddo ! do imx=-1,1
!!$       !
!!$    enddo ! do i=1,ngroup
!!$
!!$    if (nblist%type == TYPE_PAIR) then
!!$       ! Close the nblist
!!$       do i=1,nblist%n
!!$          nblist%pair(i)%startj(nblist%pair(i)%ni+1) = nblist%pair(i)%nj + 1
!!$       enddo
!!$       ! Calculate the total number of pairs
!!$       nnnb = sum(nblist%pair(1:nblist%n)%np)
!!$    endif
!!$
!!$    return
!!$  end subroutine ns_grid_{B*}_{P*}
!!$
!!$##ENDIF (domdec)
!!$
!!$##ENDEX (expand_precision)
!!$##ENDEX (expand_block)

  ! *
  ! * Gets image lows and highs
  ! *
  subroutine get_im_lo_hi(imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi)
    use domdec_common,only:nx, ny, nz
    implicit none
    ! Input / Output
    integer, intent(out) :: imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi

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
    
    return
  end subroutine get_im_lo_hi

  ! *
  ! * Returns the range of j groups that are allowed for this i group
  ! * also returns the zone of group ig
  ! *
  ! * Allowed j groups satisfy:
  ! * if ((jg >= jg_min .and. jg <= jg_max) .or. jg <= jg_lomax)
  ! *
  subroutine get_j_range(zonelist, zonei, ig, jg_min, jg_max, jg_lomax)
    implicit none
    ! Input / Output
    integer, intent(in) :: zonelist(:)
    integer, intent(inout) :: zonei
    integer, intent(in) :: ig
    integer, intent(out) :: jg_min, jg_max, jg_lomax
    ! Variables
    integer zonei0

    zonei0 = zonei
    do while (ig > zonelist(zonei))
       zonei = zonei + 1
    enddo

    if (zonei0 /= zonei) then
       if (zonei == 2) then
          jg_min = zonelist(5) + 1
          jg_max = zonelist(6)
          jg_lomax = 0
       elseif (zonei == 3) then
          jg_min = zonelist(1) + 1
          jg_max = zonelist(7)
          jg_lomax = 0
       elseif (zonei == 5) then
          jg_min = zonelist(1) + 1
          jg_max = zonelist(4)
          jg_lomax = 0
       endif
    elseif (zonei == 1) then
       jg_min = ig + 1
       jg_max = zonelist(8)
       jg_lomax = 0
    endif

    return
  end subroutine get_j_range

  ! *
  ! * Prints debug info about the neighbor list
  ! *
  subroutine print_debug_info(nblist)
    use stream
    use memory
    use mpi
    use parallel
    use domdec_dr_common,only:comm_direct
    use nblist_types,only:nblist_t, TYPE_PAIR, TYPE_TILEX
    implicit none
    ! Input
    type(nblist_t), intent(in) :: nblist
    ! Variables
    integer, allocatable, dimension(:) :: nptot, nptmp
    integer ierror
    integer i

    if (nblist%type == TYPE_PAIR) then
       call chmalloc('nblist_builder.src','ns_xfast','nptot',nblist%n,intg=nptot)
       call chmalloc('nblist_builder.src','ns_xfast','nptmp',nblist%n,intg=nptmp)
       nptmp(1:nblist%n) = nblist%pair(1:nblist%n)%np
       nptot(1:nblist%n) = 0
       call mpi_reduce(nptmp, nptot, nblist%n, MPI_INTEGER, MPI_SUM, 0, COMM_DIRECT, ierror)
       if (ierror /= MPI_SUCCESS) call wrndie(-5,'<NBLIST_BUILDER>','ERROR IN MPI_REDUCE')
       if (prnlev > 2) then
          write (outu,'(a,i12)') 'GRID NP, nnnb=',sum(nptot)
          do i=1,nblist%n
             write (outu,'(a,i3,i12,2a)') 'i,nptot=',i,nptot(i),'   ',trim(nblist%pair(i)%name)
          enddo
       endif

       call chmdealloc('nblist_builder.src','nbaph_ns','nptot',nblist%n,intg=nptot)
       call chmdealloc('nblist_builder.src','nbaph_ns','nptmp',nblist%n,intg=nptmp)
    elseif (nblist%type == TYPE_TILEX) then
!       call print_info_tilex(nblist%tilex, xyzq_gpu, atom_index, natoml_tot, inb14, iblo14)
    endif

    return
  end subroutine print_debug_info

#endif

end module nblist_builder
