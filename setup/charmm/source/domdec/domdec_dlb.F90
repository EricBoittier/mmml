module domdec_dlb

  ! *
  ! * Domain decomposition dynamic load balancing
  ! *

#if KEY_DOMDEC==1 /*domdec_main*/
#if KEY_PARALLEL==1 /*parallel*/
!#if KEY_CMPI==0 /*not_cmpi*/
  use chm_kinds
  use dimens_fcm
  use mpi
  implicit none
  private

  real(chm_real), parameter :: round_up = 1.0001, round_up_small = 1.00005, round_down = 0.9999

  ! *** For load balancing ***
  logical :: q_load_balance = .true.
  logical :: q_load_balance_x = .true.
  logical :: q_load_balance_y = .true.
  logical :: q_load_balance_z = .true.
  real(chm_real), parameter :: dlb_maxratio=0.50d0
  ! Grouped interactions cut-off (set by build_groupl() in domdec.src)
  real(chm_real) rcut_grouped_dlb
  ! Sub-box fractional sizes
  ! nodefrx(1:nx)
  ! nodefry(1:ny,-nx_comm:nx_comm)
  ! nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
  real(chm_real), allocatable, dimension(:) :: nodefrx
  real(chm_real), allocatable, dimension(:,:) :: nodefry
  real(chm_real), allocatable, dimension(:,:,:) :: nodefrz
  ! nodebx(1:nx+1)
  ! nodeby(1:ny+1,-nx_comm:nx_comm)
  ! nodebz(1:nz+1,-ny_comm:ny_comm,-nx_comm:nx_comm)
  real(chm_real), allocatable, dimension(:) :: nodebx
  real(chm_real), allocatable, dimension(:,:) :: nodeby
  real(chm_real), allocatable, dimension(:,:,:) :: nodebz
  real(chm_real) topx, topy, topz

  ! Public subroutines
  public load_balance, get_nodebx, get_nodeby, get_nodebz, &
       nodefrx_pbc, nodefry_pbc, nodefrz_pbc, write_nodeb, &
       init_subboxes, uninit_subboxes, print_dlb_info, calc_min_nodefr, get_zone_corners
  public get_home_fr, get_home_orig

  ! Public variables
  public q_load_balance, q_load_balance_x, q_load_balance_y, q_load_balance_z, &
       topx, topy, topz, nodefrx, nodefry, nodefrz, rcut_grouped_dlb
  public nodebx, nodeby, nodebz

contains

  ! *
  ! * Returns homebox origin
  ! *
  subroutine get_home_orig(origx, origy, origz)
    use number
    use domdec_common,only:homeix, homeiy, homeiz, boxx, boxy, boxz, nx_comm, ny_comm,&
         frx, fry, frz
    implicit none
    ! Output
    real(chm_real), intent(out) :: origx, origy, origz
    ! Variables
    integer ix, iy
    real(chm_real) fr

    if (q_load_balance) then
       origx = zero
       origy = zero
       origz = zero
       ! nodebx(1:nx)
       ! nodeby(1:ny,-nx_comm:-nx_comm)
       ! nodebz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
       if (homeix > 1) origx = get_nodebx(homeix-1)*boxx
       if (homeiy > 1) then
          fr = two
          do ix=-min(1,nx_comm),nx_comm
             fr = min(fr,get_nodeby(homeiy-1,homeix+ix))
          enddo
          origy = fr*boxy
       endif
       if (homeiz > 1) then
          fr = two
          do ix=-min(1,nx_comm),nx_comm
             do iy=-min(1,ny_comm),ny_comm
                fr = min(fr,get_nodebz(homeiz-1,homeiy+iy,homeix+ix))
             enddo
          enddo
          origz = fr*boxz
       endif

       origx = origx - half*boxx
       origy = origy - half*boxy
       origz = origz - half*boxz
    else
       origx = frx*boxx*(homeix-1) - half*boxx
       origy = fry*boxy*(homeiy-1) - half*boxy
       origz = frz*boxz*(homeiz-1) - half*boxz
    endif

    return
  end subroutine get_home_orig

  ! *
  ! * Returns homebox fractional size
  ! *
  subroutine get_home_fr(frx_out, fry_out, frz_out)
    use domdec_common,only:homeix, homeiy, homeiz, frx, fry, frz
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: frx_out, fry_out, frz_out

    if (q_load_balance) then
       frx_out = nodefrx(homeix)
       fry_out = nodefry(homeiy,0)
       frz_out = nodefrz(homeiz,0,0)
    else
       frx_out = frx
       fry_out = fry
       frz_out = frz
    endif

    return
  end subroutine get_home_fr

  ! *
  ! * Returns corners of the zone
  ! * I,FZ,FY,EX,FX,EZ,EY,C = 1,...8
  ! *
  subroutine get_zone_corners(izone, rcut, x0, y0, z0, x1, y1, z1)
    use domdec_common,only:homeix, homeiy, homeiz, frx, fry, frz, boxx, boxy, boxz
    implicit none
    ! Input / Output
    integer, intent(in) :: izone
    real(chm_real), intent(in) :: rcut
    real(chm_real), intent(out) :: x0, y0, z0, x1, y1, z1
    ! Variables
    integer dix, diy, diz

    call get_zone_dix_diy_diz(izone, dix, diy, diz)

    call get_zone_orig(izone, x0, y0, z0)

    if (izone == 1) then
       if (q_load_balance) then
          x1 = x0 + nodefrx(homeix + dix)*boxx
          y1 = y0 + nodefry(homeiy + diy, 0)*boxy
          z1 = z0 + nodefrz(homeiz + diz, 0, 0)*boxz
       else
          x1 = x0 + frx*boxx
          y1 = y0 + fry*boxy
          z1 = z0 + frz*boxz
       endif
    else
       x1 = x0 + dix*rcut
       y1 = y0 + diy*rcut
       z1 = z0 + diz*rcut
    endif

    return
  end subroutine get_zone_corners

  ! *
  ! * Returns difference (dix, diy, diz) for this zone
  ! *
  subroutine get_zone_dix_diy_diz(izone, dix, diy, diz)
    implicit none
    ! Input / Output
    integer, intent(in) :: izone
    integer, intent(out) :: dix, diy, diz

    dix = 0
    diy = 0
    diz = 0

    select case (izone)
       case (2)
          ! FZ
          diz = 1
       case (3)
          ! FY
          diy = 1
       case (4)
          ! EX
          diy = 1
          diz = 1
       case (5)
          ! FX
          dix = 1
       case (6)
          ! EZ
          dix = 1
          diy = 1
       case (7)
          ! EY
          dix = 1
          diz = 1
       case (8)
          ! C
          dix = 1
          diy = 1
          diz = 1
    end select

    return
  end subroutine get_zone_dix_diy_diz

  ! *
  ! * Returns zone origin
  ! *
  ! * I  FZ  FY  EX  FX  EZ  EY  C
  ! * 1   2   3   4   5   6   7  8
  ! *
  subroutine get_zone_orig(izone, x0, y0, z0)
    use number,only:zero,half,two
    use domdec_common,only:boxx, boxy, boxz, nx_comm, ny_comm, nz_comm, frx, fry, frz, &
         homeix, homeiy, homeiz
    implicit none
    integer, intent(in) :: izone
    real(chm_real), intent(out) :: x0, y0, z0
    ! Variables
    integer ixt, iyt
    integer ix, iy, iz
    integer dix, diy, diz
    real(chm_real) fr

    call get_zone_dix_diy_diz(izone, dix, diy, diz)

    ix = homeix + dix
    iy = homeiy + diy
    iz = homeiz + diz

    if (q_load_balance) then
       x0 = zero
       y0 = zero
       z0 = zero
       ! nodebx(1:nx)
       ! nodeby(1:ny,-nx_comm:-nx_comm)
       ! nodebz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
       if (ix > 1) x0 = get_nodebx(ix-1)*boxx
       if (iy > 1) then
          fr = two
          do ixt=0,nx_comm
             fr = min(fr,get_nodeby(iy-1,ix+ixt))
          enddo
          y0 = fr*boxy
       endif
       if (iz > 1) then
          fr = two
          do ixt=0,nx_comm
             do iyt=0,ny_comm
                fr = min(fr,get_nodebz(iz-1,iy+iyt,ix+ixt))
             enddo
          enddo
          z0 = fr*boxz
       endif
    else
       x0 = frx*boxx*(ix-1)
       y0 = fry*boxy*(iy-1)
       z0 = frz*boxz*(iz-1)
    endif

!    x0 = x0 - half*boxx
!    y0 = y0 - half*boxy
!    z0 = z0 - half*boxz

    return
  end subroutine get_zone_orig

  ! *
  ! * Print DLB info
  ! *
  subroutine print_dlb_info()
    use stream
    implicit none

    if (prnlev > 2) then
       if (q_load_balance_x .and. q_load_balance_y .and. q_load_balance_z) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in directions X Y Z'
       elseif (q_load_balance_x .and. q_load_balance_y) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in directions X Y'
       elseif (q_load_balance_x .and. q_load_balance_z) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in directions X Z'
       elseif (q_load_balance_y .and. q_load_balance_y) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in directions Y Z'
       elseif (q_load_balance_x) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in direction X'
       elseif (q_load_balance_y) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in direction Y'
       elseif (q_load_balance_z) then
          write (outu,'(a)') 'Dynamic Load Balancing enabled in direction Z'
       else
          write (outu,'(a)') 'Dynamic Load Balancing disabled'
       endif
    endif

    return
  end subroutine print_dlb_info

  ! *
  ! *
  ! *
  subroutine init_subboxes()
    use number
    use memory
    use inbnd,only:cutnb
    use groupxfast,only:maxgrp_rad
    use domdec_common,only:energy_time,frx,fry,frz,nx,ny,nz,nx_comm,ny_comm,nz_comm,&
         boxx,boxy,boxz, set_box, ndirect, q_split
    use domdec_dr_common,only:nrecip
    use stream
    implicit none
    ! Variables
    real(chm_real) cutgrp
    real(chm_real) min_nodefr_x, min_nodefr_y, min_nodefr_z
    real(chm_real) max_nodefr_x, max_nodefr_y, max_nodefr_z
    real(chm_real) frx_val, fry_val, frz_val

    if (.not.q_split .and. ndirect == nrecip) then
       ! Turn DLB off for the case where direct nodes are also reciprocal nodes
       q_load_balance = .false.
    endif

    ! maxgrp_rad was calculated when make_groups was called
    cutgrp = cutnb + two*maxgrp_rad

    call set_box()

    nx_comm = min(nx,ceiling(cutgrp/(frx*boxx)))
    ny_comm = min(ny,ceiling(cutgrp/(fry*boxy)))
    nz_comm = min(nz,ceiling(cutgrp/(frz*boxz)))

    if (nx == 1) nx_comm = 0
    if (ny == 1) ny_comm = 0
    if (nz == 1) nz_comm = 0

    if (q_load_balance) then
       ! For dynamic load balancing, adjust (nx_comm, ny_comm, nz_comm)
       call increase_nc_comm(nx_comm, nx, q_load_balance_x, cutgrp, boxx, frx)
       call increase_nc_comm(ny_comm, ny, q_load_balance_y, cutgrp, boxy, fry)
       call increase_nc_comm(nz_comm, nz, q_load_balance_z, cutgrp, boxz, frz)

       ! No knowledge of the grouped interactions yet:
       rcut_grouped_dlb = zero

       call calc_min_nodefr(min_nodefr_x, min_nodefr_y, min_nodefr_z)
       call calc_max_nodefr(max_nodefr_x, max_nodefr_y, max_nodefr_z)

       ! If node can increase by less than 5% => switch off load balancing
       if (max_nodefr_x < frx*1.05d0) q_load_balance_x = .false.
       if (max_nodefr_y < fry*1.05d0) q_load_balance_y = .false.
       if (max_nodefr_z < frz*1.05d0) q_load_balance_z = .false.

       ! If minimum node size is larger than the starting node size => switch off load balancing
       if (min_nodefr_x > frx) q_load_balance_x = .false.
       if (min_nodefr_y > fry) q_load_balance_y = .false.
       if (min_nodefr_z > frz) q_load_balance_z = .false.

       q_load_balance = q_load_balance_x .or. q_load_balance_y .or. q_load_balance_z
    else
       ! DLB off
       q_load_balance_x = .false.
       q_load_balance_y = .false.
       q_load_balance_z = .false.
       call calc_max_nodefr(max_nodefr_x, max_nodefr_y, max_nodefr_z)
    endif

    if (q_load_balance) then
       ! nodefrx(1:nx)
       ! nodefry(1:ny,-nx_comm:nx_comm)
       ! nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
       if (allocated(nodefrx)) then
          call chmdealloc('domdec_dlb.src','init_subboxes','nodefrx',&
               size(nodefrx),crl=nodefrx)
          call chmdealloc('domdec_dlb.src','init_subboxes','nodefry',&
               size(nodefry,1),size(nodefry,2),crl=nodefry)
          call chmdealloc('domdec_dlb.src','init_subboxes','nodefrz',&
               size(nodefrz,1),size(nodefrz,2),size(nodefrz,3),crl=nodefrz)
       endif
       
       if (.not.allocated(nodefrx)) then
          call chmalloc('domdec_dlb.src','init_subboxes','nodefrx',nx,crl=nodefrx)
          call chmalloc('domdec_dlb.src','init_subboxes','nodefry',ny,2*nx_comm+1,&
               lbou2=-nx_comm,crl=nodefry)
          call chmalloc('domdec_dlb.src','init_subboxes','nodefrz',nz,2*ny_comm+1,&
               2*nx_comm+1,lbou2=-ny_comm,lbou3=-nx_comm,crl=nodefrz)
       endif

       ! nodebx(1:nx+1)
       ! nodeby(1:ny+1,-nx_comm:nx_comm)
       ! nodebz(1:nz+1,-ny_comm:ny_comm,-nx_comm:nx_comm)
       if (allocated(nodebx)) then
          call chmdealloc('domdec_dlb.src','init_subboxes','nodebx',&
               size(nodebx),crl=nodebx)
          call chmdealloc('domdec_dlb.src','init_subboxes','nodeby',&
               size(nodeby,1),size(nodeby,2),crl=nodeby)
          call chmdealloc('domdec_dlb.src','init_subboxes','nodebz',&
               size(nodebz,1),size(nodebz,2),size(nodebz,3),crl=nodebz)
       endif
       
       if (.not.allocated(nodebx)) then
          call chmalloc('domdec_dlb.src','init_subboxes','nodebx',nx+1,crl=nodebx)
          call chmalloc('domdec_dlb.src','init_subboxes','nodeby',ny+1,2*nx_comm+1,&
               lbou2=-nx_comm,crl=nodeby)
          call chmalloc('domdec_dlb.src','init_subboxes','nodebz',nz+1,2*ny_comm+1,&
               2*nx_comm+1,lbou2=-ny_comm,lbou3=-nx_comm,crl=nodebz)
       endif
       
       ! Start from uniform distribution
       nodefrx(1:nx) = frx
       nodefry(1:ny,-nx_comm:nx_comm) = fry
       nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm) = frz
       call fill_nodeb()

       !       call read_nodeb('nodeb.txt')

       energy_time = minone

       frx_val = maxval(nodefrx(1:nx))
       fry_val = maxval(nodefry(1:ny,0))
       frz_val = maxval(nodefrz(1:nz,0,0))
    else
       frx_val = frx
       fry_val = fry
       frz_val = frz
    endif

    if (nx > 1 .and. frx_val > max_nodefr_x) then
       write (outu,'(a,3i4)') 'NDIR = ',nx,ny,nz
       write (outu,'(a,f10.3)') 'cutoff = cutnb + 2*max_group_radius = ',cutnb + two*maxgrp_rad
       write (outu,'(a,f10.3)') 'cutnb = ',cutnb
       write (outu,'(a,f10.3)') '2*max_group_radius = ',two*maxgrp_rad
       call wrndie(-5,'<domdec_dlb>',&
            'box size too small in x direction, try setting larger ndir value')
    endif
    
    if (ny > 1 .and. fry_val > max_nodefr_y) then
       write (outu,'(a,3i4)') 'NDIR = ',nx,ny,nz
       write (outu,'(a,f10.3)') 'cutoff = cutnb + 2*max_group_radius = ',cutnb + two*maxgrp_rad
       write (outu,'(a,f10.3)') 'cutnb = ',cutnb
       write (outu,'(a,f10.3)') '2*max_group_radius = ',two*maxgrp_rad
       call wrndie(-5,'<domdec_dlb>',&
            'box size too small in y direction, try setting larger ndir value')
    endif
    
    if (nz > 1 .and. frz_val > max_nodefr_z) then
       write (outu,'(a,3i4)') 'NDIR = ',nx,ny,nz
       write (outu,'(a,f10.3)') 'cutoff = cutnb + 2*max_group_radius = ',cutnb + two*maxgrp_rad
       write (outu,'(a,f10.3)') 'cutnb = ',cutnb
       write (outu,'(a,f10.3)') '2*max_group_radius = ',two*maxgrp_rad
       call wrndie(-5,'<domdec_dlb>',&
            'box size too small in z direction, try setting larger ndir value')
    endif

    return
  end subroutine init_subboxes

  ! *
  ! * Increase nc_comm (=nx_comm,ny_comm,nz_comm) to allow more load balancing
  ! *
  subroutine increase_nc_comm(nc_comm, nc, q_load_balance_c, cut, boxsize, frc)
    implicit none
    ! Input / Output
    integer nc_comm
    integer, intent(in) :: nc
    logical q_load_balance_c
    real(chm_real), intent(in) :: cut, boxsize, frc
    ! Variables
    real(chm_real) margin, minsubc
    logical ok

    q_load_balance_c = .true.

    if (nc == 1) then
       nc_comm = 0
       q_load_balance_c = .false.
       return
    endif

    ok = .false.
    do while (.not.ok)
       ok = .true.
       ! margin = maximum possible increase in scaling for the sub-boxes
       minsubc = cut/real(nc_comm)
       margin = (boxsize*frc - minsubc)/minsubc
       if (margin < 1.05d0) then
          ok = .false.
          if (nc_comm < nc) then
             nc_comm = nc_comm + 1
          else
             ! We can't increase nx_comm => switch off load balancing in this direction
             q_load_balance_c = .false.
             return
          endif
       endif
    enddo

    return
  end subroutine increase_nc_comm

  subroutine uninit_subboxes
    use memory
    use domdec_common
    implicit none

    if (q_load_balance) then
       if (allocated(nodefrx)) then
          call chmdealloc('domdec_dlb.src','uninit_subboxes','nodefrx',nx,crl=nodefrx)
       endif

       if (allocated(nodefry)) then
          call chmdealloc('domdec_dlb.src','uninit_subboxes','nodefry',ny,2*nx_comm+1,&
               crl=nodefry)
       endif

       if (allocated(nodefrz)) then
          call chmdealloc('domdec_dlb.src','uninit_subboxes','nodefrz',nz,2*ny_comm+1,&
               2*nx_comm+1,crl=nodefrz)
       endif
    endif

    return
  end subroutine uninit_subboxes

  ! *
  ! * Calculates node z-boundary from nodefrz, where (iz, iy, ix) are in absolute coordinates:
  ! * (iz, iy, ix) in (1:nz, -ny_comm+homeiy:ny_comm+homeiy, -nx_comm+homeix:nx_comm+homeix)
  ! *
  real(chm_real) function get_nodebz(iz,iy,ix)
    use number
    use domdec_common
    implicit none
    ! Input
    integer iz, iy, ix
    ! Variables
    integer izl

    if (iz == 0 .or. iz == nz) then
       get_nodebz = one
    else
       izl = iz
       if (izl < 0) then
          izl = izl + nz
       elseif (izl > nz) then
          izl = izl - nz
       endif
       if (q_load_balance) then
          if (iy-homeiy < -ny_comm .or. iy-homeiy > ny_comm) then
             call wrndie(-5,'<domdec_dlb>','get_nodebz: iy out of range')
          endif
          if (ix-homeix < -nx_comm .or. ix-homeix > nx_comm) then
             call wrndie(-5,'<domdec_dlb>','get_nodebz: ix out of range')
          endif
!          get_nodebz = sum(nodefrz(1:izl,iy-homeiy,ix-homeix))
!          if (get_nodebz /= nodebz(izl+1,iy-homeiy,ix-homeix)) then
!             write (0,'(a,2f7.4,a,6f7.4,a,7f7.4)') 'Z',&
!                  get_nodebz,nodebz(izl+1,iy-homeiy,ix-homeix),'|',&
!                  nodefrz(1:nz,iy-homeiy,ix-homeix),'|',&
!                  nodebz(1:nz+1,iy-homeiy,ix-homeix)
!             stop
!          endif
          get_nodebz = nodebz(izl+1,iy-homeiy,ix-homeix)
       else
          get_nodebz = izl*frz
       endif
    endif
    
    return
  end function get_nodebz

  ! *
  ! * Calculates node y-boundary from nodefry, where (iy, ix) are in absolute coordinates:
  ! * (iy, ix) in (1:ny, -nx_comm+homeix:nx_comm+homeix)
  ! *
  real(chm_real) function get_nodeby(iy,ix)
    use number
    use domdec_common
    implicit none
    ! Input
    integer, intent(in) :: iy, ix
    ! Variables
    integer iyl

    if (iy == 0 .or. iy == ny) then
       get_nodeby = one
    else
       iyl = iy
       if (iyl < 0) then
          iyl = iyl + ny
       elseif (iyl > ny) then
          iyl = iyl - ny
       endif
       if (q_load_balance) then
          if (ix-homeix < -nx_comm .or. ix-homeix > nx_comm) then
             call wrndie(-5,'<domdec_dlb>','get_nodeby: ix out of range')
          endif
!          get_nodeby = sum(nodefry(1:iyl,ix-homeix))
!          if (get_nodeby /= nodeby(iyl+1,ix-homeix)) then
!             write (0,*) 'Y',get_nodeby, nodeby(iyl+1,ix-homeix)
!             stop
!          endif
          get_nodeby = nodeby(iyl+1,ix-homeix)
       else
          get_nodeby = iyl*fry
       endif
    endif
    
    return
  end function get_nodeby

  ! *
  ! * Calculates node x-boundary from nodefrx, where (ix) is in absolute coordinates:
  ! * (ix) in (1:nx)
  ! *
  real(chm_real) function get_nodebx(ix)
    use number
    use domdec_common
    implicit none
    ! Input
    integer, intent(in) :: ix
    ! Variables
    integer ixl

    if (ix == 0 .or. ix == nx) then
       get_nodebx = one
    else
       ixl = ix
       if (ixl < 0) then
          ixl = ixl + nx
       elseif (ixl > nx) then
          ixl = ixl - nx
       endif
       if (q_load_balance) then
!          get_nodebx = sum(nodefrx(1:ixl))
!          if (get_nodebx /= nodebx(ixl+1)) then
!             write (0,*) 'X',get_nodebx, nodebx(ixl+1)
!             stop
!          endif
          get_nodebx = nodebx(ixl+1)
       else
          get_nodebx = ixl*frx
       endif
    endif
    
    return
  end function get_nodebx

  ! *
  ! * Returns the value of nodefrx with periodic wrapping, 
  ! * where (ix) is in absolute coordinates:
  ! * (ix) in (1:nx)
  ! *
  real(chm_real) function nodefrx_pbc(ix)
    use domdec_common
    implicit none
    ! Input
    integer, intent(in) :: ix
    
    if (ix < 1) then
       nodefrx_pbc = nodefrx(ix + nx)
    elseif (ix > nx) then
       nodefrx_pbc = nodefrx(ix - nx)
    else
       nodefrx_pbc = nodefrx(ix)
    endif

    return
  end function nodefrx_pbc

  ! *
  ! * Returns the value of nodefry with periodic wrapping, 
  ! * where (iy, ix) are in absolute coordinates:
  ! * (iy, ix) in (1:ny, -nx_comm+homeix:nx_comm+homeix)
  ! *
  real(chm_real) function nodefry_pbc(iy, ix)
    use domdec_common
    implicit none
    ! Input
    integer, intent(in) :: iy, ix

    if (ix-homeix < -nx_comm .or. ix-homeix > nx_comm) then
       call wrndie(-5,'<domdec_dlb>','nodefry_pbc, ix out of range')
    endif

    if (iy < 1) then
       nodefry_pbc = nodefry(iy + ny, ix-homeix)
    elseif (iy > ny) then
       nodefry_pbc = nodefry(iy - ny, ix-homeix)
    else
       nodefry_pbc = nodefry(iy, ix-homeix)
    endif

    return
  end function nodefry_pbc

  ! *
  ! * Returns the value of nodefrz with periodic wrapping,
  ! * where (iz, iy, ix) are in absolute coordinates:
  ! * (iz, iy, ix) in (1:nz, -ny_comm+homeiy:ny_comm+homeiy, -nx_comm+homeix:nx_comm+homeix)
  ! *
  real(chm_real) function nodefrz_pbc(iz, iy, ix)
    use domdec_common
    implicit none
    ! Input
    integer, intent(in) :: iz, iy, ix

    if (iy-homeiy < -ny_comm .or. iy-homeiy > ny_comm) then
       call wrndie(-5,'<domdec_dlb>','nodefrz_pbc, iy out of range')
    endif

    if (ix-homeix < -nx_comm .or. ix-homeix > nx_comm) then
       call wrndie(-5,'<domdec_dlb>','nodefrz_pbc, ix out of range')
    endif

    if (iz < 1) then
       nodefrz_pbc = nodefrz(iz + nz, iy-homeiy, ix-homeix)
    elseif (iz > nz) then
       nodefrz_pbc = nodefrz(iz - nz, iy-homeiy, ix-homeix)
    else
       nodefrz_pbc = nodefrz(iz, iy-homeiy, ix-homeix)
    endif

    return
  end function nodefrz_pbc

  ! *
  ! * Calculates new node boundaries for load balancing
  ! *
  subroutine calc_nodefr(nodefrc, frc_default, nc, energy_time_c, energy_time_csum, min_nodefr, &
       minup, maxlo)
    use memory
    use number
    use inbnd,only:cutnb
    use parallel,only:mynod
    use groupxfast,only:maxgrp_rad
    use stream,only:outu
    use domdec_common
    implicit none
    ! Input / Output
    real(chm_real) nodefrc(*)
    integer nc
    real(chm_real) energy_time_c(*), energy_time_csum, min_nodefr, frc_default
    real(chm_real), optional :: minup(*), maxlo(*)
    ! Variables
    real(chm_real) limb, factor, scale_max, f, dist, fold, fnew
    real(chm_real) min_energy_time_c
    ! Temporary (work) node sizes
    real(chm_real), save, allocatable, dimension(:) :: nodefrt
    ! low/hi node bounds and new/old node bounds
    real(chm_real), save, allocatable, dimension(:) :: bound_lo, bound_hi, bound_new, bound_old
    logical, save, allocatable, dimension(:) :: adjusted
    logical adjacent
    integer i
    real(chm_real) tmp(100)

    if (.not.allocated(nodefrt)) then
       call chmalloc('domdec_dlb.src','calc_nodefr','nodefrt',max(nx,ny,nz),crl=nodefrt)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_lo',max(nx,ny,nz)+1,crl=bound_lo)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_hi',max(nx,ny,nz)+1,crl=bound_hi)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_new',max(nx,ny,nz)+1,crl=bound_new)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_old',max(nx,ny,nz)+1,crl=bound_old)
       call chmalloc('domdec_dlb.src','calc_nodefr','adjusted',max(nx,ny,nz),log=adjusted)
    elseif (size(nodefrt) < nc) then
       call chmdealloc('domdec_dlb.src','calc_nodefr','nodefrt',size(nodefrt),crl=nodefrt)
       call chmdealloc('domdec_dlb.src','calc_nodefr','bound_lo',size(bound_lo),crl=bound_lo)
       call chmdealloc('domdec_dlb.src','calc_nodefr','bound_hi',size(bound_hi),crl=bound_hi)
       call chmdealloc('domdec_dlb.src','calc_nodefr','bound_new',size(bound_new),crl=bound_new)
       call chmdealloc('domdec_dlb.src','calc_nodefr','bound_old',size(bound_old),crl=bound_old)
       call chmdealloc('domdec_dlb.src','calc_nodefr','adjusted',size(adjusted),log=adjusted)
       call chmalloc('domdec_dlb.src','calc_nodefr','nodefrt',nc,crl=nodefrt)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_lo',nc+1,crl=bound_lo)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_hi',nc+1,crl=bound_hi)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_new',nc+1,crl=bound_new)
       call chmalloc('domdec_dlb.src','calc_nodefr','bound_old',nc+1,crl=bound_old)
       call chmalloc('domdec_dlb.src','calc_nodefr','adjusted',nc,log=adjusted)
    endif

    ! Restrict energy_time_c to positive values
    min_energy_time_c = maxval(energy_time_c(1:nc))
    do i=1,nc
       if (energy_time_c(i) > zero) then
          min_energy_time_c = min(min_energy_time_c, energy_time_c(i))
       endif
    enddo
    if (.not.(min_energy_time_c > zero)) min_energy_time_c = 0.001d0
    do i=1,nc
       if (.not.(energy_time_c(i) > zero)) energy_time_c(i) = min_energy_time_c
    enddo

    ! Calculate the maximum scaling of the nodes
    factor = half
    scale_max = zero
    do i=1,nc
       limb = energy_time_csum/(energy_time_c(i)*real(nc))
       scale_max = max(factor*abs(limb - one), scale_max)
    enddo

    ! Limit the maximum scaling to 5% and then rescale all nodes with this
    ! new factor.
    if (scale_max > 0.05d0) then
       factor = factor*0.05d0/scale_max
    endif

    ! Assign new node sizes
    f = zero
    do i=1,nc
       limb = energy_time_csum/(energy_time_c(i)*real(nc))
       nodefrt(i) = nodefrc(i)*(one - factor*(one - limb))
       f = f + nodefrt(i)
    enddo
    ! Get rid of rounding errors and scale nodefrt such that
    ! sum(nodefr(1:nc)) = 1
    f = one/f
    nodefrt(1:nc) = nodefrt(1:nc)*f

    ! Calculate old and new node boundaries
    ! NOTE:
    ! bound_new(1)    = bound_old(1)    = 0
    ! bound_new(nc+1) = bound_old(nc+1) = 1
    fold = zero
    fnew = zero
    do i=1,nc
       bound_old(i) = fold
       bound_new(i) = fnew
       fold = fold + nodefrc(i)
       fnew = fnew + nodefrt(i)
    enddo
    bound_old(nc+1) = one
    bound_new(nc+1) = one

    ! Check that the nodes have not shifted such that the distance between
    ! adjacent node boundaries is less than the minimum size minsubf
    adjacent = .false.
    if (present(minup) .and. present(maxlo)) then
       !
       ! Calculate lower and upper limits for the node boundaries based
       ! on the adjacent node boundaries and the minimum size of node
       !
       ! At the end, we must have: bound_lo(i) < bound_new(i) < bound_hi(i)
       !
       bound_lo(1) = zero
       bound_hi(1) = zero
       do i=2,nc
          ! bound_lo = low bound for the node boundary
          ! on the one hand the lower boundary is limited by the smallest node size
          ! (min_nodefr):
          bound_lo(i) = maxlo(i-1) + min_nodefr
          ! dist = node boundary - lower node boundary
          dist = bound_old(i) - bound_lo(i)
          if (dist > zero) then
             bound_lo(i) = bound_lo(i) + half*dist
          endif
          ! bound_hi = high bound for the node boundary
          bound_hi(i) = minup(i) - min_nodefr
          dist = bound_hi(i) - bound_old(i)
          if (dist > zero) then
             bound_hi(i) = bound_hi(i) - half*dist
          endif
       enddo
       bound_lo(nc+1) = one
       bound_hi(nc+1) = one
       adjacent = .true.
    endif

    ! save old bound_new
    tmp(1:nc+1) = bound_new(1:nc+1)

    call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, nodefrt, adjusted, &
         1, nc, adjacent)

    ! Calculate new nodefrc and normalize to 1.0
    do i=1,nc
       nodefrc(i) = bound_new(i+1) - bound_new(i)
    enddo
    nodefrc(1:nc) = nodefrc(1:nc)/sum(nodefrc(1:nc))
    ! Calculate new bound_new also (only for testing)
    fnew = zero
    do i=1,nc
       bound_new(i) = fnew
       fnew = fnew + nodefrc(i)
    enddo
    bound_new(nc+1) = one

    if (adjacent) then
       do i=2,nc
          if (bound_new(i) < bound_lo(i) .or. bound_new(i) > bound_hi(i)) then
!!$             write (outu,'(a,2i3)') 'boundary limit violation remains, mynod,i=',mynod,i
!!$             write (outu,'(a,12f8.4)') 'bound_new (orig)=',tmp(1:nc+1)
!!$             write (outu,'(a,12f8.4)') 'bound_new       =',bound_new(1:nc+1)
!!$             write (outu,'(a,12f8.4)') 'bound_lo        =',bound_lo(1:nc+1)
!!$             write (outu,'(a,12f8.4)') 'bound_hi        =',bound_hi(1:nc+1)
!!$             write (outu,'(a,f8.4)') 'min_nodefr=',min_nodefr
             ! NOTE: This really should never happen
             write (outu,'(a)') 'domdec_dlb: boundary limit violation, resetting to default'
             nodefrc(1:nc) = frc_default
!             call wrndie(-5,'<domdec_dlb>','Dynamic Load Balancing: boundary limit violated')
          endif
       enddo
    endif

    return
  end subroutine calc_nodefr

  ! *
  ! * Limit nodefr from ilo to ihi
  ! *
  recursive subroutine limit_nodefr(bound_old, bound_new, &
       bound_lo, bound_hi, min_nodefr, nodefrt, adjusted, ilo, ihi, adjacent)
    use number
    implicit none
    ! Input / Output
    real(chm_real) bound_old(*), bound_new(*), min_nodefr
    real(chm_real) bound_lo(*), bound_hi(*), nodefrt(*)
    integer ilo, ihi
    logical adjusted(*), adjacent
    ! Variables
    logical ok, prev_below, fix_lo, fix_hi
    real(chm_real) f, fsize, fad, fnad, fhalf
    integer i, j, nadjusted, ilo_new, ihi_new

    ! Update node sizes
    do i=ilo,ihi
       nodefrt(i) = bound_new(i+1) - bound_new(i)
    enddo

    ! fsize = size of the node range from ihi to ilo
    fsize = bound_new(ihi+1) - bound_new(ilo)

    ! Limit node sizes such that they're not smaller than the minimum node size min_nodefr
    adjusted(ilo:ihi) = .false.
    nadjusted = 0
    f = one
    ok = .false.
    do while (.not.ok)
       ! fad  = (fractional) room taken by the adjusted nodes
       ! fnad = (fractional) room taken by the non-adjusted nodes
       ! NOTE: fad + fnad = 1
       fad = zero
       fnad = zero
       ok = .true.
       do i=ilo,ihi
          ! Only scale and check nodes that have not yet been adjusted
          if (adjusted(i)) then
             fad = fad + nodefrt(i)  !min_nodefr
          else
             nodefrt(i) = nodefrt(i)*f
             if (nodefrt(i) < min_nodefr) then
                nodefrt(i) = min_nodefr
                adjusted(i) = .true.
                nadjusted = nadjusted + 1
                ok = .false.
                fad = fad + nodefrt(i) !min_nodefr
             else
                fnad = fnad + nodefrt(i)
             endif
          endif
          bound_new(i+1) = bound_new(i) + nodefrt(i)
       enddo
       ! New scaling factor f is calculated by requiring that:
       ! f*fnad + fad = fsize
       f = (fsize - fad)/fnad
    enddo

    ! Check the size of the top node
    nodefrt(ihi) = bound_new(ihi+1) - bound_new(ihi)
    if (nodefrt(ihi) < min_nodefr*round_up_small/round_up) then
       call wrndie(-5,'<domdec_dlb>','Unable to resize node boundaries')
    endif

    ! Limit node boundaries such that they don't shift more than halfway between
    ! old node boundaries 
    do i=ilo+1,ihi
       fhalf = half*(bound_old(i) + bound_old(i-1))
       if (bound_new(i) < fhalf) then
          ! Lift the new boundary up to halfway
          bound_new(i) = fhalf
          ! Move the higher boundaries if needed
          do j=i+1,ihi
             if (bound_new(j) < bound_new(j-1) + min_nodefr) then
                bound_new(j) = bound_new(j-1) + min_nodefr
             endif
          enddo
       endif
       fhalf = half*(bound_old(i) + bound_old(i+1))
       if (bound_new(i) > fhalf) then
          ! Lower the new boundary down to halfway
          bound_new(i) = fhalf
          ! Move the lower boundaries if needed
          do j=i-1,ilo+1,-1
             if (bound_new(j) < bound_new(j+1) - min_nodefr) then
                bound_new(j) = bound_new(j+1) - min_nodefr
             endif
          enddo
       endif
    enddo

    if (adjacent) then
       ilo_new = ilo
       ihi_new = ihi
       ! Limit node boundaries such that they don't violate bounds given by bound_lo and bould_hi
       prev_below = .true.
       fix_lo = .false.
       fix_hi = .false.
       do i=ilo+1,ihi
          if (bound_new(i) < bound_lo(i) .and. bound_new(i) > bound_hi(i)) then
             ! New boundary is outside both low and high limits
             bound_new(i) = half*(bound_lo(i) + bound_hi(i))
             ! Limit nodes ilo to i-1
             ilo_new = ilo
             ihi_new = i-1
             call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
                  nodefrt, adjusted, ilo_new, ihi_new, adjacent)
             ! Limit nodes i to ihi
             ilo_new = i
             ihi_new = ihi
             call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
                  nodefrt, adjusted, ilo_new, ihi_new, adjacent)
             return
          elseif (bound_new(i) < bound_lo(i)) then
             ! New boundary is below low limit
             ! NOTE: this finds the highest low limit violation
             ihi_new = i
             prev_below = .true.
             fix_lo = .true.
          elseif (bound_new(i) > bound_hi(i) .and. prev_below) then
             ! New boundary is above high limit
!             if (ihi_new < ihi) then
             if (fix_lo) then
                ! Fix the highest low limit violation:
                bound_new(ihi_new) = bound_lo(ihi_new)*round_up
                call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr,&
                     nodefrt, adjusted, ilo_new, ihi_new-1, adjacent)
                ilo_new = ihi_new
             endif
             ! Fix the high limit violation:
             bound_new(i) = bound_hi(i)/round_up
             ihi_new = i
             call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
                  nodefrt, adjusted, ilo_new, ihi_new-1, adjacent)
             ilo_new = i
             ihi_new = ihi
             prev_below = .false.
             fix_lo = .false.
             fix_hi = .true.
          endif
       enddo
!       if (ihi_new < ihi) then
       if (fix_lo) then
          bound_new(ihi_new) = bound_lo(ihi_new)*round_up
          call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
               nodefrt, adjusted, ilo_new, ihi_new-1, adjacent)
          ilo_new = ihi_new
          ihi_new = ihi
          call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
               nodefrt, adjusted, ilo_new, ihi_new, adjacent)
       elseif (fix_hi) then
!       elseif (ilo_new > ilo) then
          call limit_nodefr(bound_old, bound_new, bound_lo, bound_hi, min_nodefr, &
               nodefrt, adjusted, ilo_new, ihi_new, adjacent)
       endif
    endif

    return
  end subroutine limit_nodefr

  ! *
  ! * Calculates the maximum lower boundary (nodefrc_maxlo) and
  ! * minimum upper boundary (nodefrc_minup)
  ! *
  subroutine calc_nodefr_minmax(nc, nodefrc, nodefrc_minup, nodefrc_maxlo)
    use number
    implicit none
    ! Input / Output
    integer nc
    real(chm_real) nodefrc(*), nodefrc_minup(*), nodefrc_maxlo(*)
    ! Variables
    real(chm_real) x1
    integer i

    ! By definition:
    ! nodefrc_maxlo(1)  = 0
    ! nodefrc_minup(nc) = 1
    ! therefore, these remain unchanged by this routine

    x1 = zero
    do i=1,nc-1
       ! x1 = current boundary
       x1 = x1 + nodefrc(i)
       if (x1 < nodefrc_minup(i)) nodefrc_minup(i) = x1
       if (x1 > nodefrc_maxlo(i+1)) nodefrc_maxlo(i+1) = x1
    enddo

    return
  end subroutine calc_nodefr_minmax

  ! *
  ! * Performs load balancing based on time in energy_time
  ! *
  subroutine load_balance
    use memory
    use number
    use parallel,only:comm_charmm,mynod,numnod
    use inbnd,only:cutnb
    use groupxfast,only:maxgrp_rad
    use mpi
    use domdec_common,only:energy_time,nx,ny,nz,homeix,homeiy,homeiz,nx_comm,ny_comm,frx,fry,frz,&
         nodeind
    use reawri,only:qcnstp
    use stream,only:outu
    implicit none
    ! Input
    ! Variables
    real(chm_real) energy_time_xsum, energy_time_ysum, energy_time_zsum
    real(chm_real), save, allocatable, dimension(:) :: tmpbuf, nodefr_minup, nodefr_maxlo, &
         energy_time_x, energy_time_y, energy_time_z
    ! nodefrx(1:nx)
    ! nodefry(1:ny,-nx_comm:nx_comm)
    ! nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
    real(chm_real), save, allocatable, dimension(:) :: nodefrx_prev
    real(chm_real), save, allocatable, dimension(:,:) :: nodefry_prev
    real(chm_real), save, allocatable, dimension(:,:,:) :: nodefrz_prev
    real(chm_real) min_nodefr_x, min_nodefr_y, min_nodefr_z
    integer i, j, ihi, jhi
    integer nx_comm0, nx_comm1, ny_comm0, ny_comm1
    integer status(MPI_STATUS_SIZE), ierror
    logical ok, ok_and_all
    !
!!$    integer nseed
!!$    integer, allocatable, dimension(:) :: seed
!!$    integer, save :: ncall = 0
!!$    ncall = ncall + 1

    ! Calculate communication limits for communicating node boundaries
    call calc_nxny_comm(nx_comm0,nx_comm1,ny_comm0,ny_comm1)

!!$    if (ncall == 1) then
!!$       call random_seed(size=nseed)
!!$       allocate(seed(nseed))
!!$       seed(1:nseed) = (mynod+1)
!!$       call random_seed(put=seed)
!!$       deallocate(seed)
!!$    endif
!!$    call random_number(energy_time)

    if (.not.allocated(energy_time_x)) then
       call chmalloc('domdec_dlb.src','load_balance','energy_time_x',nx,crl=energy_time_x)
       call chmalloc('domdec_dlb.src','load_balance','energy_time_y',ny,crl=energy_time_y)
       call chmalloc('domdec_dlb.src','load_balance','energy_time_z',nz,crl=energy_time_z)
       call chmalloc('domdec_dlb.src','load_balance','tmpbuf',nx+ny+nz,crl=tmpbuf)
       call chmalloc('domdec_dlb.src','load_balance','nodefr_minup',max(ny,nz),crl=nodefr_minup)
       call chmalloc('domdec_dlb.src','load_balance','nodefr_maxlo',max(ny,nz),crl=nodefr_maxlo)
       call chmalloc('domdec_dlb.src','load_balance','nodefrx_prev',nx,crl=nodefrx_prev)
       call chmalloc('domdec_dlb.src','load_balance','nodefry_prev',ny,2*nx_comm+1,&
            lbou2=-nx_comm,crl=nodefry_prev)
       call chmalloc('domdec_dlb.src','load_balance','nodefrz_prev',nz,2*ny_comm+1,2*nx_comm+1,&
            lbou2=-ny_comm,lbou3=-nx_comm,crl=nodefrz_prev)
    else
       if (size(energy_time_x) < nx) then
          call chmrealloc('domdec_dlb.src','load_balance','energy_time_x',nx,crl=energy_time_x)
       endif
       if (size(energy_time_y) < ny) then
          call chmrealloc('domdec_dlb.src','load_balance','energy_time_y',ny,crl=energy_time_y)
       endif
       if (size(energy_time_z) < nz) then
          call chmrealloc('domdec_dlb.src','load_balance','energy_time_z',nz,crl=energy_time_z)
       endif
       if (size(tmpbuf) < nx+ny+nz) then
          call chmrealloc('domdec_dlb.src','load_balance','tmpbuf',nx+ny+nz,crl=tmpbuf)
       endif
       if (size(nodefr_minup) < max(ny,nz)) then
          call chmrealloc('domdec_dlb.src','load_balance','nodefr_minup',max(ny,nz),crl=nodefr_minup)
          call chmrealloc('domdec_dlb.src','load_balance','nodefr_maxlo',max(ny,nz),crl=nodefr_maxlo)
       endif

       if (size(nodefrx_prev,1) < nx) then
          call chmdealloc('domdec_dlb.src','load_balance','nodefrx_prev',&
               size(nodefrx_prev),crl=nodefrx_prev)
          call chmalloc('domdec_dlb.src','load_balance','nodefrx_prev',nx,crl=nodefrx_prev)
       endif
       if (size(nodefry_prev,1) < ny .or. size(nodefry_prev,2) < 2*nx_comm+1) then
          call chmdealloc('domdec_dlb.src','load_balance','nodefry_prev',&
               size(nodefry_prev,1),size(nodefry_prev,2),crl=nodefry_prev)
          call chmalloc('domdec_dlb.src','load_balance','nodefry_prev',ny,2*nx_comm+1,&
               lbou2=-nx_comm,crl=nodefry_prev)
       endif
       if (size(nodefrz_prev,1) < nz .or. size(nodefrz_prev,2) < 2*ny_comm+1 .or. &
            size(nodefrz_prev,3) < 2*nx_comm+1) then
          call chmdealloc('domdec_dlb.src','load_balance','nodefrz_prev',&
               size(nodefrz_prev,1),size(nodefrz_prev,2),size(nodefrz_prev,3),&
               crl=nodefrz_prev)
          call chmalloc('domdec_dlb.src','load_balance','nodefrz_prev',nz,2*ny_comm+1,2*nx_comm+1,&
               lbou2=-ny_comm,lbou3=-nx_comm,crl=nodefrz_prev)
       endif
    endif

    ! Save previous configuration
    nodefrx_prev(1:nx) = nodefrx(1:nx)
    nodefry_prev(1:ny,-nx_comm:nx_comm) = nodefry(1:ny,-nx_comm:nx_comm)
    nodefrz_prev(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm) = &
         nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)

    call comm_energy_time(energy_time_xsum, energy_time_ysum, energy_time_zsum, &
         energy_time_x, energy_time_y, energy_time_z)

    call calc_min_nodefr(min_nodefr_x, min_nodefr_y, min_nodefr_z)

    ! Increase min_nodefr by 2% for constant pressure simulations
    if (qcnstp) then
       min_nodefr_x = min_nodefr_x*1.02d0
       min_nodefr_y = min_nodefr_y*1.02d0
       min_nodefr_z = min_nodefr_z*1.02d0
    endif

    min_nodefr_x = min_nodefr_x*round_up
    min_nodefr_y = min_nodefr_y*round_up
    min_nodefr_z = min_nodefr_z*round_up

    if (q_load_balance_x) then
       if (homeiz == 1 .and. homeiy == 1 .and. homeix == 1) then
          ! node 1,1,1 is ready to load balance along the x-axis
          call calc_nodefr(nodefrx, frx, nx, energy_time_x, energy_time_xsum, min_nodefr_x)
          ! Send nodefrx to nodes with homeiy == 1
          do i=2,nx
             call mpi_send(nodefrx, nx, MPI_REAL8, nodeind(i, 1, 1), &
                  1, comm_charmm, ierror)
             if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                  'ERROR IN MPI_SEND (X)')
          enddo
       elseif (homeiz == 1 .and. homeiy == 1) then
          call mpi_recv(nodefrx, nx, MPI_REAL8, nodeind(1, 1, 1), &
               MPI_ANY_TAG, comm_charmm, status, ierror)
          if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
               'ERROR IN MPI_RECV (X)')
       endif
    endif

    if (q_load_balance_y) then
       if (homeiz == 1 .and. homeiy == 1) then
          ! Calculate the maximum lower boundary and the minimum upper boundary
          ! based on the surrounding columns
          if (nx_comm > 0) then
             nodefr_minup(1:ny) = one
             nodefr_maxlo(1:ny) = zero
             ihi = min(1,nx_comm1)  ! take only surrounding columns: ihi = 0 or 1
             do i=-1,ihi
                call calc_nodefr_minmax(ny, nodefry(1:ny,i), nodefr_minup, nodefr_maxlo)
             enddo
             call calc_nodefr(nodefry(1:ny,0), fry, ny, energy_time_y, energy_time_ysum, &
                  min_nodefr_y, nodefr_minup, nodefr_maxlo)
          else
             call calc_nodefr(nodefry(1:ny,0), fry, ny, energy_time_y, energy_time_ysum, &
                  min_nodefr_y)
          endif
          ! Send nodefrx and nodefry
          tmpbuf(1:nx) = nodefrx(1:nx)
          tmpbuf(nx+1:nx+ny) = nodefry(1:ny,0)
          do i=2,ny
             call mpi_send(tmpbuf, nx+ny, MPI_REAL8, nodeind(homeix, i, 1), &
                  1, comm_charmm, ierror)
             if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                  'ERROR IN MPI_SEND (Y)')
          enddo
       elseif (homeiz == 1) then
          ! Receive nodefrx and nodefry
          call mpi_recv(tmpbuf, nx+ny, MPI_REAL8, nodeind(homeix, 1, 1), &
               MPI_ANY_TAG, comm_charmm, status, ierror)
          if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
               'ERROR IN MPI_RECV (Y)')
          nodefrx(1:nx) = tmpbuf(1:nx)
          nodefry(1:ny,0) = tmpbuf(nx+1:nx+ny)
       endif
    endif

    if (q_load_balance_z) then
       if (homeiz == 1) then
          ! Calculate the maximum lower boundary and the minimum upper boundary
          ! based on the surrounding nodes
          if (nx_comm > 0 .or. ny_comm > 0) then
             nodefr_minup(1:nz) = one
             nodefr_maxlo(1:nz) = zero
             ihi = min(1,nx_comm1)   ! take only surrounding columns: ihi = 0 or 1
             jhi = min(1,ny_comm1)   ! take only surrounding columns: jhi = 0 or 1
             do j=-min(1,ny_comm),jhi
                do i=-min(1,nx_comm),ihi
                   call calc_nodefr_minmax(nz, nodefrz(1:nz,j,i), nodefr_minup, nodefr_maxlo)
                enddo
             enddo
             call calc_nodefr(nodefrz(:,0,0), frz, nz, energy_time_z, energy_time_zsum, &
                  min_nodefr_z, nodefr_minup, nodefr_maxlo)
          else
             call calc_nodefr(nodefrz(:,0,0), frz, nz, energy_time_z, energy_time_zsum, &
                  min_nodefr_z)
          endif
          tmpbuf(1:nx) = nodefrx(1:nx)
          tmpbuf(nx+1:nx+ny) = nodefry(1:ny,0)
          tmpbuf(nx+ny+1:nx+ny+nz) = nodefrz(1:nz,0,0)
          do i=2,nz
             call mpi_send(tmpbuf, nx+ny+nz, MPI_REAL8, nodeind(homeix, homeiy, i), &
                  1, comm_charmm, ierror)
             if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                  'ERROR IN MPI_SEND (Z)')
          enddo
       else
          call mpi_recv(tmpbuf, nx+ny+nz, MPI_REAL8, nodeind(homeix, homeiy, 1), &
               MPI_ANY_TAG, comm_charmm, status, ierror)
          if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
               'ERROR IN MPI_RECV (Z)')
          nodefrx(1:nx) = tmpbuf(1:nx)
          nodefry(1:ny,0) = tmpbuf(nx+1:nx+ny)
          nodefrz(1:nz,0,0) = tmpbuf(nx+ny+1:nx+ny+nz)
       endif
    endif

    ! Communicate neighboring node boundaries
    call comm_nodefry(nx_comm0,nx_comm1)
    call comm_nodefrz(ny_comm0,ny_comm1,nx_comm0,nx_comm1)
    call fill_nodefr(nx_comm0,nx_comm1,ny_comm0,ny_comm1)

!    call read_nodeb('nodeb.txt')

    call check_nodefr(ok)

    call mpi_allreduce(ok, ok_and_all, 1, mpi_logical, mpi_land, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<domdec_dlb>','load_balance, mpi_allreduce failed')
    endif
    if (.not.ok_and_all) then
       nodefrx(1:nx) = nodefrx_prev(1:nx)
       nodefry(1:ny,-nx_comm:nx_comm) = nodefry_prev(1:ny,-nx_comm:nx_comm)
       nodefrz(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm) = &
            nodefrz_prev(1:nz,-ny_comm:ny_comm,-nx_comm:nx_comm)
    endif

    call fill_nodeb()

    ! Zero nongrouped energy time counter
    energy_time = zero

    return
  end subroutine load_balance

  ! *
  ! * Calculates minimum nodesize in fractional units
  ! *
  subroutine calc_min_nodefr(min_nodefr_x, min_nodefr_y, min_nodefr_z)
    use number
    use domdec_common,only:nx,ny,nz,nx_comm,ny_comm,nz_comm,invx,invy,invz
    use inbnd,only:cutnb
    use groupxfast,only:maxgrp_rad
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: min_nodefr_x, min_nodefr_y, min_nodefr_z
    ! Variables
    real(chm_real) max_nodefr_x, max_nodefr_y, max_nodefr_z

    ! Calculate minimum node size (in fractional units)
    min_nodefr_x = (cutnb + two*maxgrp_rad)/real(max(1,nx_comm))*invx
    min_nodefr_y = (cutnb + two*maxgrp_rad)/real(max(1,ny_comm))*invy
    min_nodefr_z = (cutnb + two*maxgrp_rad)/real(max(1,nz_comm))*invz

    min_nodefr_x = max(min_nodefr_x,rcut_grouped_dlb*invx)
    min_nodefr_y = max(min_nodefr_y,rcut_grouped_dlb*invy)
    min_nodefr_z = max(min_nodefr_z,rcut_grouped_dlb*invz)

    call calc_max_nodefr(max_nodefr_x, max_nodefr_y, max_nodefr_z)
    
    ! Limit the maximum node size by setting a larger minimum node size:
    if (nx > 1) min_nodefr_x = max(min_nodefr_x, (one - max_nodefr_x)/real(nx-1))
    if (ny > 1) min_nodefr_y = max(min_nodefr_y, (one - max_nodefr_y)/real(ny-1))
    if (nz > 1) min_nodefr_z = max(min_nodefr_z, (one - max_nodefr_z)/real(nz-1))
   
    return   
  end subroutine calc_min_nodefr

  ! *
  ! * Calculates maximum nodesize in fractional units
  ! *
  subroutine calc_max_nodefr(max_nodefr_x, max_nodefr_y, max_nodefr_z)
    use number
    use inbnd,only:cutnb
    use groupxfast,only:maxgrp_rad
    use domdec_common,only:invx,invy,invz
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: max_nodefr_x, max_nodefr_y, max_nodefr_z

    ! Calculate maximum node size (in fractional units)
    max_nodefr_x = one - (two*cutnb + maxgrp_rad)*invx
    max_nodefr_y = one - (two*cutnb + maxgrp_rad)*invy
    max_nodefr_z = one - (two*cutnb + maxgrp_rad)*invz
    
    return   
  end subroutine calc_max_nodefr

  ! *
  ! * Calculate communication limits for communicating node boundaries
  ! *
  subroutine calc_nxny_comm(nx_comm0,nx_comm1,ny_comm0,ny_comm1)
    use domdec_common
    implicit none
    ! Input / Output
    integer nx_comm0, nx_comm1
    integer ny_comm0, ny_comm1

    nx_comm0 = -nx_comm
    nx_comm1 = nx_comm
    if (nx_comm1 - nx_comm0 + 1 > nx) then
       nx_comm0 = -1
       nx_comm1 = nx - 1 + nx_comm0
       ! Adjust boundary such that nx_comm1 <= nx_comm
       if (nx_comm1 > nx_comm) then
          nx_comm0 = nx_comm0 - (nx_comm1-nx_comm)
          nx_comm1 = nx_comm1 - (nx_comm1-nx_comm)
       endif
    endif
    ny_comm0 = -ny_comm
    ny_comm1 = ny_comm
    if (ny_comm1 - ny_comm0 + 1 > ny) then
       ny_comm0 = -1
       ny_comm1 = ny - 1 + ny_comm0
       ! Adjust boundary such that ny_comm1 <= ny_comm
       if (ny_comm1 > ny_comm) then
          ny_comm0 = ny_comm0 - (ny_comm1-ny_comm)
          ny_comm1 = ny_comm1 - (ny_comm1-ny_comm)
       endif
    endif

    return
  end subroutine calc_nxny_comm

  ! *
  ! * Fills up the nodefry and nodefrz to the entire range:
  ! * [-nx_comm:nx_comm] [-ny_comm:ny_comm]
  ! *
  subroutine fill_nodefr(nx_comm0,nx_comm1,ny_comm0,ny_comm1)
    use domdec_common,only:nx_comm, ny_comm, ny, nz
    implicit none
    ! Input / Output
    integer nx_comm0, nx_comm1
    integer ny_comm0, ny_comm1
    ! Variables
    integer i, ii, j, jj

    ! nodefry is correct in the range [-nx_comm0:nx_comm1]
    ! nodefrz is correct in the range [-ny_comm0:ny_comm1] [-nx_comm0:nx_comm1]
    !
    ! Here we copy node sizes to the entire range [-nx_comm:nx_comm] [-ny_comm:ny_comm]
    do i=-nx_comm,nx_comm
       ii = i
       if (ii < nx_comm0) then
          ii = ii + (nx_comm1 - nx_comm0 + 1)
       elseif (ii > nx_comm1) then
          ii = ii - (nx_comm1 - nx_comm0 + 1)
       endif
       if (i /= ii) nodefry(1:ny,i) = nodefry(1:ny,ii)
       do j=-ny_comm,ny_comm
          jj = j
          if (jj < ny_comm0) then
             jj = jj + (ny_comm1 - ny_comm0 + 1)
          elseif (jj > ny_comm1) then
             jj = jj - (ny_comm1 - ny_comm0 + 1)
          endif
          if (i /= ii .or. j /= jj) nodefrz(1:nz,j,i) = nodefrz(1:nz,jj,ii)
       enddo
    enddo

    return
  end subroutine fill_nodefr

  ! *
  ! * Fills nodebx, nodeby, nodebz:
  ! * nodebx(i) = sum(nodefrx(1:i))
  ! *
  subroutine fill_nodeb()
    use number,only:zero, one
    use domdec_common,only:nx, ny, nz, nx_comm, ny_comm
    implicit none
    ! Input / Output
    ! Variables
    integer i, iy, ix

    nodebx(1) = zero
    do i=1,nx
       nodebx(i+1) = nodebx(i) + nodefrx(i)
    enddo
    nodebx(nx+1) = one

    do ix=-nx_comm,nx_comm
       nodeby(1,ix) = zero
       do i=1,ny
          nodeby(i+1,ix) = nodeby(i,ix) + nodefry(i,ix)
       enddo
       nodeby(ny+1,ix) = one
    enddo

    do ix=-nx_comm,nx_comm
       do iy=-ny_comm,ny_comm
          nodebz(1,iy,ix) = zero
          do i=1,nz
             nodebz(i+1,iy,ix) = nodebz(i,iy,ix) + nodefrz(i,iy,ix)
          enddo
          nodebz(nz+1,iy,ix) = one
       enddo
    enddo

    return
  end subroutine fill_nodeb

  ! *
  ! * Check nodefr for cell borders that are more than dlb_maxratio
  ! *
  subroutine check_nodefr(ok)
    use number
    use parallel,only:mynod
    use domdec_common
    implicit none
    ! Input / Output
    logical ok
    ! Variables
    integer i
    real(chm_real) x0, x1, ratio, maxratio
    real(chm_real) min_nodefr_x, min_nodefr_y, min_nodefr_z
    real(chm_real) max_nodefr_x, max_nodefr_y, max_nodefr_z

    maxratio = zero
    ok = .true.

    call set_box()

    ! Check if minimum or maximum node size has been breached:
    call calc_min_nodefr(min_nodefr_x, min_nodefr_y, min_nodefr_z)
    call calc_max_nodefr(max_nodefr_x, max_nodefr_y, max_nodefr_z)

    if (q_load_balance_x) then
       if (minval(nodefrx(1:nx)) < min_nodefr_x) ok = .false.
       if (maxval(nodefrx(1:nx)) > max_nodefr_x) ok = .false.
    endif

    if (q_load_balance_y) then
       if (minval(nodefry(1:ny,0)) < min_nodefr_y) ok = .false.
       if (maxval(nodefry(1:ny,0)) > max_nodefr_y) ok = .false.
    endif

    if (q_load_balance_z) then
       if (minval(nodefrz(1:nz,0,0)) < min_nodefr_z) ok = .false.
       if (maxval(nodefrz(1:nz,0,0)) > max_nodefr_z) ok = .false.
    endif

    ! Check for neighboring node sizes:
    if (q_load_balance_y) then
       if (nx_comm > 0) then
          x0 = zero
          x1 = zero
          do i=1,ny-1
             x0 = x0 + nodefry(i,-1)
             x1 = x1 + nodefry(i,0)
             if (x0 > x1) then
                ratio = (x0 - x1)/nodefry(i,-1)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefry, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefry(:,-1)=',mynod,nodefry(1:ny,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefry(:,0) =',mynod,nodefry(1:ny,0)
                   ok = .false.
                endif
             else
                ratio = (x1 - x0)/nodefry(i,0)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefry, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefry(:,-1)=',mynod,nodefry(1:ny,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefry(:,0) =',mynod,nodefry(1:ny,0)
                   ok = .false.
                endif
             endif
             maxratio = max(maxratio, ratio)
          enddo
       endif
    endif

    if (q_load_balance_z) then
       if (nx_comm > 0 .and. ny_comm > 0) then
          x0 = zero
          x1 = zero
          do i=1,nz-1
             x0 = x0 + nodefrz(i,-1,-1)
             x1 = x1 + nodefrz(i,0,0)
             if (x0 > x1) then
                ratio = (x0 - x1)/nodefrz(i,-1,-1)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,0,-1) =',mynod,nodefrz(1:nz,0,-1)
                   ok = .false.
                endif
             else
                ratio = (x1 - x0)/nodefrz(i,0,0)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,0,-1) =',mynod,nodefrz(1:nz,0,-1)
                   ok = .false.
                endif
             endif
             maxratio = max(maxratio, ratio)
          enddo
       endif
       
       if (nx_comm > 0) then
          x0 = zero
          x1 = zero
          do i=1,nz-1
             x0 = x0 + nodefrz(i,0,-1)
             x1 = x1 + nodefrz(i,0,0)
             if (x0 > x1) then
                ratio = (x0 - x1)/nodefrz(i,0,-1)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,0) =',mynod,nodefrz(1:nz,-1,0)
                   ok = .false.
                endif
             else
                ratio = (x1 - x0)/nodefrz(i,0,0)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,0) =',mynod,nodefrz(1:nz,-1,0)
                   ok = .false.
                endif
             endif
             maxratio = max(maxratio, ratio)
          enddo
       endif

       if (ny_comm > 0) then
          x0 = zero
          x1 = zero
          do i=1,nz-1
             x0 = x0 + nodefrz(i,-1,0)
             x1 = x1 + nodefrz(i,0,0)
             if (x0 > x1) then
                ratio = (x0 - x1)/nodefrz(i,-1,0)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,0,0)  =',mynod,nodefrz(1:nz,0,0)
                   ok = .false.
                endif
             else
                ratio = (x1 - x0)/nodefrz(i,0,0)
                if (ratio > dlb_maxratio) then
!!$             write (*,'(a,2i3,f6.3)') 'cell border over limit, nodefrz, mynod,i,ratio=',&
!!$                  mynod,i,ratio
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,-1,-1)=',mynod,nodefrz(1:nz,-1,-1)
!!$             write (*,'(a,i3,6f6.3)') 'mynod,nodefrz(:,0,0)  =',mynod,nodefrz(1:nz,0,0)
                   ok = .false.
                endif
             endif
             maxratio = max(maxratio, ratio)
          enddo
       endif
    endif

    return
  end subroutine check_nodefr

  ! *
  ! * Communicate node boundaries to neighboring nodes
  ! *
  subroutine comm_nodefry(min_nx, max_nx)
    use parallel,only:comm_charmm
    use mpi
    use memory
    use domdec_common
    implicit none
    ! Input / Output
    integer min_nx, max_nx
    ! Variables
    integer i, node, ierror, ireq
    integer, save, allocatable :: stats(:,:), req(:)

    if (q_load_balance_y) then
       i = (max_nx-min_nx+1)*2
       if (.not.allocated(stats)) then
          call chmalloc('domdec_dlb.src','comm_nodefry','stats',MPI_STATUS_SIZE,i,intg=stats)
          call chmalloc('domdec_dlb.src','comm_nodefry','req',i,intg=req)
       elseif (size(req) < i) then
          call chmrealloc('domdec_dlb.src','comm_nodefry','stats',MPI_STATUS_SIZE,i,intg=stats)
          call chmrealloc('domdec_dlb.src','comm_nodefry','req',i,intg=req)
       endif
    
       ireq = 0
       do i=min_nx,max_nx
          if (i /= 0) then
             node = nodeindfunc(i+homeix,homeiy,homeiz)
             ireq = ireq + 1
             call mpi_irecv(nodefry(:,i), ny, MPI_REAL8, node, &
                  MPI_ANY_TAG, comm_charmm, req(ireq), ierror)
             if (ierror /= MPI_SUCCESS) &
                  call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_IRECV')
          endif
       enddo
       
       do i=min_nx,max_nx
          if (i /= 0) then
             node = nodeindfunc(i+homeix,homeiy,homeiz)
             ireq = ireq + 1
             call mpi_isend(nodefry(:,0), ny, MPI_REAL8, node, &
                  0, comm_charmm, req(ireq), ierror)
             if (ierror /= MPI_SUCCESS) &
                  call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_ISEND (comm_nodefry)')
          endif
       enddo
       if (ireq > size(req)) &
            call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN COMM_NODEFRY')
       
       call mpi_waitall(ireq, req, stats, ierror)
       if (ierror /= MPI_SUCCESS) &
            call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_WAITALL')
    else
       do i=min_nx,max_nx
          nodefry(1:ny,i) = nodefry(1:ny,0)
       enddo
    endif

    return
  end subroutine comm_nodefry

  ! *
  ! * Communicate node boundaries to neighboring nodes
  ! *
  subroutine comm_nodefrz(min_ny, max_ny, min_nx, max_nx)
    use parallel,only:comm_charmm
    use mpi
    use memory
    use domdec_common
    implicit none
    ! Input / Output
    integer min_nx, max_nx, min_ny, max_ny
    ! Variables
    integer i, j, node, ierror, ireq
    integer, save, allocatable :: stats(:,:), req(:)

    if (q_load_balance_z) then
       i = (max_nx-min_nx+1)*(max_ny-min_ny+1)*2
       if (.not.allocated(stats)) then
          call chmalloc('domdec_dlb.src','comm_nodefrz','stats',MPI_STATUS_SIZE,i,intg=stats)
          call chmalloc('domdec_dlb.src','comm_nodefrz','req',i,intg=req)
       elseif (size(req) < i) then
          call chmrealloc('domdec_dlb.src','comm_nodefrz','stats',MPI_STATUS_SIZE,i,intg=stats)
          call chmrealloc('domdec_dlb.src','comm_nodefrz','req',i,intg=req)
       endif

       ireq = 0
       do i=min_nx,max_nx
          do j=min_ny,max_ny
             if (i /= 0 .or. j /= 0) then
                node = nodeindfunc(i+homeix,j+homeiy,homeiz)
                ireq = ireq + 1
                call mpi_irecv(nodefrz(:,j,i), nz, MPI_REAL8, node, &
                     MPI_ANY_TAG, comm_charmm, req(ireq), ierror)
                if (ierror /= MPI_SUCCESS) &
                     call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_IRECV')
             endif
          enddo
       enddo
       
       do i=min_nx,max_nx
          do j=min_ny,max_ny
             if (i /= 0 .or. j /= 0) then
                node = nodeindfunc(i+homeix,j+homeiy,homeiz)
                ireq = ireq + 1
                call mpi_isend(nodefrz(:,0,0), nz, MPI_REAL8, node, &
                     0, comm_charmm, req(ireq), ierror)
                if (ierror /= MPI_SUCCESS) &
                     call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_ISEND  (comm_nodefrz)')
             endif
          enddo
       enddo
       
       if (ireq > size(req)) &
            call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN COMM_NODEFRZ')
       
       call mpi_waitall(ireq, req, stats, ierror)
       if (ierror /= MPI_SUCCESS) &
            call WRNDIE(-5,'<DOMDEC_DLB>','ERROR IN MPI_WAITALL')
    else
       do i=min_nx,max_nx
          do j=min_ny,max_ny
             nodefrz(1:nz,j,i) = nodefrz(1:nz,0,0)
          enddo
       enddo
    endif
    return
  end subroutine comm_nodefrz

  ! *
  ! * Communicate energy_time
  ! *
  subroutine comm_energy_time(energy_time_xsum, energy_time_ysum, energy_time_zsum, &
       energy_time_x, energy_time_y, energy_time_z)
    use parallel,only:comm_charmm
    use number
    use mpi
    use domdec_common,only:energy_time,nx,ny,nz,homeix,homeiy,homeiz,nodeind
    implicit none
    ! Input / Output
    real(chm_real) energy_time_xsum, energy_time_ysum, energy_time_zsum
    real(chm_real) energy_time_x(*), energy_time_y(*), energy_time_z(*)
    ! Variables
    integer i
    integer ierror, status(MPI_STATUS_SIZE)
    real(chm_real) tmp

    if (energy_time < zero) energy_time = one

    if (homeiz == 1) then
       ! Receive energy_time from nodes homeiz > 1
       energy_time_zsum = energy_time
       energy_time_z(1) = energy_time_zsum
       do i=2,nz
          call mpi_recv(energy_time_z(i), 1, MPI_REAL8, nodeind(homeix,homeiy,i), &
               MPI_ANY_TAG, comm_charmm, status, ierror)
          if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
               'ERROR IN MPI_RECV')
          energy_time_zsum = energy_time_zsum + energy_time_z(i)
       enddo
       if (homeiy == 1) then
          energy_time_ysum = energy_time_zsum
          energy_time_y(1) = energy_time_ysum
          do i=2,ny
             call mpi_recv(energy_time_y(i), 1, MPI_REAL8, nodeind(homeix,i,1), &
                  MPI_ANY_TAG, comm_charmm, status, ierror)
             if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                  'ERROR IN MPI_RECV')
             energy_time_ysum = energy_time_ysum + energy_time_y(i)
          enddo
          if (homeix == 1) then
             energy_time_xsum = energy_time_ysum
             energy_time_x(1) = energy_time_xsum
             do i=2,nx
                call mpi_recv(energy_time_x(i), 1, MPI_REAL8, nodeind(i,1,1), &
                     MPI_ANY_TAG, comm_charmm, status, ierror)
                if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                     'ERROR IN MPI_RECV')
                energy_time_xsum = energy_time_xsum + energy_time_x(i)
             enddo
          else
             call mpi_send(energy_time_ysum, 1, MPI_REAL8, nodeind(1,1,1), &
                  1, comm_charmm, ierror)
             if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
                  'ERROR IN MPI_SEND (comm_energy_time y)')
          endif
       else
          call mpi_send(energy_time_zsum, 1, MPI_REAL8, nodeind(homeix,1,1), &
               1, comm_charmm, ierror)
          if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
               'ERROR IN MPI_SEND (comm_energy_time z)')
       endif
    else
       ! Send energy_time to node homeiz = 1
       call mpi_send(energy_time, 1, MPI_REAL8, nodeind(homeix,homeiy,1), &
            1, comm_charmm, ierror)
       if (ierror /= MPI_SUCCESS) call WRNDIE(-5,'<DOMDEC_DLB>',&
            'ERROR IN MPI_SEND (comm_energy_time)')
    endif

  end subroutine comm_energy_time

  ! *
  ! * Writes node boundaries
  ! *
  subroutine write_nodeb()
    use parallel,only:mynod
    use number
    use domdec_common
    implicit none
    ! Variables
    real(chm_real) x0, x1, y0, y1, z0, z1
    character(30) filename

    if (mynod < 10) then
       write (filename,'(a,i1,a)') 'nodeb_',mynod,'.txt'
    elseif (mynod < 100) then
       write (filename,'(a,i2,a)') 'nodeb_',mynod,'.txt'
    elseif (mynod < 1000) then
       write (filename,'(a,i3,a)') 'nodeb_',mynod,'.txt'
    else
       write (filename,'(a,i4,a)') 'nodeb_',mynod,'.txt'
    endif
    open(121,file=filename)

    x0 = zero
    y0 = zero
    z0 = zero
    x1 = get_nodebx(homeix)
    y1 = get_nodeby(homeiy,homeix)
    z1 = get_nodebz(homeiz,homeiy,homeix)
    if (homeix > 1) x0 = get_nodebx(homeix-1)
    if (homeiy > 1) y0 = get_nodeby(homeiy-1,homeix)
    if (homeiz > 1) z0 = get_nodebz(homeiz-1,homeiy,homeix)

    write (121,'(i4,6f11.8)') mynod,x0,x1,y0,y1,z0,z1

    close(121)

    return
  end subroutine write_nodeb

  ! *
  ! * Reads node boundaries from a file
  ! *
  subroutine read_nodeb(filename)
    use memory
    use parallel,only:mynod
    use domdec_common
    implicit none
    ! Input / Output
    character(*) filename
    ! Variables
    integer nx_comm0, nx_comm1
    integer ny_comm0, ny_comm1
    integer i, node, ix, iy, iz
    real(chm_real) x0, x1, y0, y1, z0, z1
    logical node_found

    ! Read file to construct nodefrx(:), nodefry(:,0), nodefrz(:,0,0)
    open(121,file=filename)
    do i=1,ndirect
       read (121,'(i4,6f11.8)') node,x0,x1,y0,y1,z0,z1
       ! find ix,iy,iz
       node_found = .false.
       izloop: do iz=1,nz
          do iy=1,ny
             do ix=1,nx
                if (nodeind(ix,iy,iz) == node) then
                   node_found = .true.
                   exit izloop
                endif
             enddo
          enddo
       enddo izloop
       if (.not.node_found) call wrndie(-5,'<domdec_dlb>','read_nodeb: node not found')
       nodefrx(ix) = x1-x0
       if (ix == homeix) then
          nodefry(iy,0) = y1-y0
          if (iy == homeiy) nodefrz(iz,0,0) = z1-z0
       endif
    enddo
    close(121)

    call calc_nxny_comm(nx_comm0,nx_comm1,ny_comm0,ny_comm1)
    call comm_nodefry(nx_comm0,nx_comm1)
    call comm_nodefrz(ny_comm0,ny_comm1,nx_comm0,nx_comm1)
    call fill_nodefr(nx_comm0,nx_comm1,ny_comm0,ny_comm1)

    return
  end subroutine read_nodeb

!#endif /* (not_cmpi)*/
#endif /* (parallel)*/
#endif /* (domdec_main)*/
end module domdec_dlb

