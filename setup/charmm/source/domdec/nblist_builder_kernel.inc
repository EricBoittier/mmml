
#ifdef NS_GRID
  subroutine KERNEL_NAME(nnnb, nblist, zonelist, groupcenter, groupbox, xyzq, groupl, group, &
       group_glo, ngroup, top_excl_pos, top_excl, cutcell, cutnb, ctofnb, grid)
!    use memory,only:chmalloc, chmrealloc
!    use number,only:zero, two
!    use inbnd,only:cutnb, ctofnb
!    use domdec_common,only:zonelist, boxx, boxy, boxz, nx, ny, nz
!    use nblist_types,only:nblist_t, TYPE_PAIR
!    use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
!    use nblist_pair,only:flush_pair_{P*}
!#ifdef BLOCK_ON
!    use nblist_pair,only:flush_pair_block_{P*}
!#endif
!    use groupxfast,only:maxgrp_rad, maxbox, group_out
    implicit none
    ! Input / Output
    integer nnnb
    type(nblist_t) nblist
    integer, intent(in) :: zonelist(:)
    real(chm_real), intent(in) :: groupcenter(3,*), groupbox(3,*)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    integer, intent(in) :: groupl(:), group(:), group_glo(:)
    integer, intent(in) :: ngroup, top_excl_pos(:), top_excl(:)
    real(chm_real), intent(in) :: cutcell, cutnb, ctofnb
    type(grid_t), intent(inout) :: grid
    ! Variables
    integer i, j, k, klo, khi, n, is, iq, itype, ish, is_unpack, iq_unpack
    integer ix0, iy0, iz0, ix1, iy1, iz1
    integer gimx, gimy, gimz, icx, icy, icz, ind, imx, imy, imz
    integer imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    integer zonei, jg_min, jg_max, jg_lomax, jglo, jghi
    real(chm_real) xi, yi, zi, xj, yj, zj, xij, yij, zij
    real(chm_real) rsq, cut1sq, cut2sq
    real(chm_real) xim, yim, zim
    real(chm_real) celld1, celld2
    integer ntmpl1, ntmpl2

    ! cut1sq < cut2sq
    cut1sq = ctofnb*ctofnb
    cut2sq = cutnb*cutnb

    call get_im_lo_hi(imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi)

    call build_group_grid(ngroup, groupcenter, grid)

    if (nblist%type == TYPE_PAIR) then
       nblist%pair(1:nblist%n)%ni = 0
       nblist%pair(1:nblist%n)%nj = 0
       nblist%pair(1:nblist%n)%np = 0
    endif

    zonei = 1
    do i=1,ngroup

       call get_j_range(zonelist, zonei, i, jg_min, jg_max, jg_lomax)
       ! .not.(I or FX or FY or FZ)
       if (zonei /= 1 .and. zonei /= 2 .and. zonei /= 3 .and. zonei /= 5) cycle

       ! Global group is defined in group_glo(groupl(i))
       ! Local group is defined in group(i)
       call group_out(group(i), is, iq, itype)
       call group_out(group_glo(groupl(i)), is_unpack, iq_unpack)

       ! Convert group index to grid index
       call index2gridindex(grid, i, gimx, gimy, gimz)

       ! Get atom group center coordinates
       xi = groupcenter(1,i)
       yi = groupcenter(2,i)
       zi = groupcenter(3,i)

       ! Loop over images, max. number of images = 3*3*3=27.
       ! For serial: done in every direction
       ! For domdec: done in every direction that has only one sub-box (i.e. where nx,ny,nz = 1)
       do imx=imx_lo,imx_hi
          xim = xi + imx*boxx
          call get_cell_bounds(gimx+imx*grid%grid_nx, xim-grid%grid_origx, &
               grid%grid_bx, cutcell, grid%grid_nx, grid%grid_nxlocal, ix0, ix1, grid%grid_xdist)
          if (ix0 > ix1) cycle
          do imy=imy_lo,imy_hi
             yim = yi + imy*boxy
             call get_cell_bounds(gimy+imy*grid%grid_ny, yim-grid%grid_origy, &
                  grid%grid_by, cutcell, grid%grid_ny, grid%grid_nylocal, iy0, iy1, grid%grid_ydist)
             if (iy0 > iy1) cycle
             do imz=imz_lo,imz_hi
                zim = zi + imz*boxz
                call get_cell_bounds(gimz+imz*grid%grid_nz, zim-grid%grid_origz, &
                     grid%grid_bz, cutcell, grid%grid_nz, grid%grid_nzlocal, iz0, iz1, grid%grid_zdist)
                if (iz0 > iz1) cycle

                ! ish = shift index = 1...26*3+1
                ish = (imx+1 + (imy+1)*3 + (imz+1)*9 + 1)*3 - 2
                ! Empty temporary list
                ntmpl1 = 0
                ntmpl2 = 0
                ! Loop over the cells

                do icz=iz0,iz1
                   celld1 = max(grid%grid_zdist(icz) - groupbox(3,i) - maxbox(3),zero)**2
                   do icy=iy0,iy1
                      celld2 = celld1 + max(grid%grid_ydist(icy) - groupbox(2,i) - maxbox(2),zero)**2
                      if (celld2 > cut2sq) cycle
                      do icx=ix0,ix1
                         if (celld2 + max(grid%grid_xdist(icx) - groupbox(1,i) - maxbox(1),zero)**2 > &
                              cut2sq) cycle
                         ind = icx + (icy-1)*grid%grid_nx + (icz-1)*grid%grid_nxy
                         n = grid%grid_n(ind)
                         ! Check wether entire cell is out of range
                         ! grid_grps(grid(ind)%start)     = minimum j-group index
                         ! grid_grps(grid(ind+1)%start-1) = maximum j-group index
                         if (n == 0) cycle
                         klo = grid%grid_start(ind)
                         khi = grid%grid_start(ind+1)-1
                         jglo = grid%grid_grps(klo)
                         jghi = grid%grid_grps(khi)

                         if ((jghi < jg_min .or. jglo > jg_max) .and. jglo > jg_lomax) cycle
                         do k=klo,khi
                            j = grid%grid_grps(k)
                            if ((j < jg_min .or. j > jg_max) .and. j > jg_lomax) cycle
                            if (zonei == 3) then
                               ! FY
                               if (j > zonelist(2) .and. j <= zonelist(6)) cycle
                            endif
                            xij = abs(xim - groupcenter(1,j))
                            yij = abs(yim - groupcenter(2,j))
                            zij = abs(zim - groupcenter(3,j))
                            xij = max(xij-groupbox(1,i)-groupbox(1,j),zero)
                            yij = max(yij-groupbox(2,i)-groupbox(2,j),zero)
                            zij = max(zij-groupbox(3,i)-groupbox(3,j),zero)
                            rsq = xij*xij + yij*yij + zij*zij
                            ! Add to temporary list, extend temporary list size if neccessary
                            ! tmpl(1,:) = local group index
                            ! tmpl(2,:) = global group index
                            if (rsq < cut1sq) then
                               ntmpl1 = ntmpl1 + 1
                               if (ntmpl1 > tmpl1_len) then
                                  tmpl1_len = int(1.5*tmpl1_len)
                                  call chmrealloc('nblist_builder.src','#KERNEL_NAME','tmpl1',&
                                       2,tmpl1_len,intg=tmpl1)
                               endif
                               tmpl1(1,ntmpl1) = j
                               tmpl1(2,ntmpl1) = groupl(j)
                            elseif (rsq < cut2sq) then
                               ntmpl2 = ntmpl2 + 1
                               if (ntmpl2 > tmpl2_len) then
                                  tmpl2_len = int(1.5*tmpl2_len)
                                  call chmrealloc('nblist_builder.src','#KERNEL_NAME','tmpl2',&
                                       2,tmpl2_len,intg=tmpl2)
                               endif
                               tmpl2(1,ntmpl2) = j
                               tmpl2(2,ntmpl2) = groupl(j)
                            endif
                         enddo ! do k=klo,khi
                      enddo  ! do icz=iz0,iz1
                   enddo  ! do icy=iy0,iy1
                enddo  ! do icx=ix0,ix1
                ! Flush the temporary lists
                if (ntmpl1 > 0) then
#ifdef BLOCK_ON
#ifdef DOUBLE_PREC
                   call flush_pair_block_pd&
#else
                   call flush_pair_block_ps&
#endif
                        (i, xyzq, top_excl_pos, top_excl, is, iq, &
                        itype, is_unpack, ish, ntmpl1, tmpl1, &
                        nblist%pair, group, group_glo)
#else
#ifdef DOUBLE_PREC
                   call flush_pair_pd&
#else
                   call flush_pair_ps&
#endif
                        (i, xyzq, top_excl_pos, top_excl, is, iq, &
                        itype, is_unpack, ish, ntmpl1, tmpl1, &
                        nblist%pair, group, group_glo)
#endif
                endif
                if (ntmpl2 > 0) then
#ifdef BLOCK_ON
#ifdef DOUBLE_PREC
                   call flush_pair_block_pd&
#else
                   call flush_pair_block_ps&
#endif
                        (i, xyzq, top_excl_pos, top_excl, is, iq, &
                        itype, is_unpack, ish, ntmpl2, tmpl2, &
                        nblist%pair, group, group_glo)
#else
#ifdef DOUBLE_PREC
                   call flush_pair_pd&
#else
                   call flush_pair_ps&
#endif
                        (i, xyzq, top_excl_pos, top_excl, is, iq, &
                        itype, is_unpack, ish, ntmpl2, tmpl2, &
                        nblist%pair, group, group_glo)
#endif
                endif
             enddo ! do imz=-1,1
          enddo ! do imy=-1,1
       enddo ! do imx=-1,1
       !
    enddo ! do i=1,ngroup

    if (nblist%type == TYPE_PAIR) then
       ! Close the nblist
       do i=1,nblist%n
          nblist%pair(i)%startj(nblist%pair(i)%ni+1) = nblist%pair(i)%nj + 1
       enddo
       ! Calculate the total number of pairs
       nnnb = sum(nblist%pair(1:nblist%n)%np)
    endif

    return
  end subroutine KERNEL_NAME
#endif

#ifdef NS_GRID_THOLE
  subroutine KERNEL_NAME(nbtholp, nbthol1, nbthol2, nbthol3, natoml, atoml, zonelist, &
       glo2loc_ind, loc2glo_ind, xyzq, cutcell, cutnb, grid)
    implicit none
    ! Input / Output
    integer, intent(inout) :: nbtholp, nbthol1(:), nbthol2(:), nbthol3(:)
    integer, intent(in) :: natoml, atoml(:), zonelist(:), glo2loc_ind(:), loc2glo_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    real(chm_real), intent(in) :: cutcell, cutnb
    type(grid_t), intent(inout) :: grid
    ! Variables
    integer i, j, k, klo, khi, n, ii, jj, ish
    integer ix0, iy0, iz0, ix1, iy1, iz1
    integer iglo, jglo, itype, jtype, aa, ijtype, tholeij_val
    integer gimx, gimy, gimz, icx, icy, icz, ind, imx, imy, imz
    integer imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi
    integer zonei, j_min, j_max, j_lomax
    real(chm_real) xi, yi, zi, xj, yj, zj, xij, yij, zij
    real(chm_real) rsq, cutsq
    real(chm_real) xim, yim, zim
    real(chm_real) celld1, celld2

    cutsq = cutnb*cutnb

    call get_im_lo_hi(imx_lo, imx_hi, imy_lo, imy_hi, imz_lo, imz_hi)
#ifdef DOUBLE_PREC
    call build_thole_grid_pd(natoml, atoml, glo2loc_ind, xyzq, grid)
#else
    call build_thole_grid_ps(natoml, atoml, glo2loc_ind, xyzq, grid)
#endif

    nbtholp = 0
    zonei = 1
    do ii=1,ntholenblist
       i = tholenblist(ii)          ! local index
       iglo = loc2glo_ind(i)+1
       itype = tholenbtype(iglo)

       call get_j_range(zonelist, zonei, i, j_min, j_max, j_lomax)
       ! .not.(I or FX or FY or FZ)
       if (zonei /= 1 .and. zonei /= 2 .and. zonei /= 3 .and. zonei /= 5) cycle

       ! Convert atom index to grid index
       call index2gridindex(grid, ii, gimx, gimy, gimz)

       ! Get atom coordinates
       xi = xyzq(i)%x
       yi = xyzq(i)%y
       zi = xyzq(i)%z

       ! Loop over images, max. number of images = 3*3*3=27.
       ! For serial: done in every direction
       ! For domdec: done in every direction that has only one sub-box (i.e. where nx,ny,nz = 1)
       do imx=imx_lo,imx_hi
          xim = xi + imx*boxx
          call get_cell_bounds(gimx+imx*grid%grid_nx, xim-grid%grid_origx, &
               grid%grid_bx, cutcell, grid%grid_nx, grid%grid_nxlocal, ix0, ix1, grid%grid_xdist)
          if (ix0 > ix1) cycle
          do imy=imy_lo,imy_hi
             yim = yi + imy*boxy
             call get_cell_bounds(gimy+imy*grid%grid_ny, yim-grid%grid_origy, &
                  grid%grid_by, cutcell, grid%grid_ny, grid%grid_nylocal, iy0, iy1, grid%grid_ydist)
             if (iy0 > iy1) cycle
             do imz=imz_lo,imz_hi
                zim = zi + imz*boxz
                call get_cell_bounds(gimz+imz*grid%grid_nz, zim-grid%grid_origz, &
                     grid%grid_bz, cutcell, grid%grid_nz, grid%grid_nzlocal, iz0, iz1, grid%grid_zdist)
                if (iz0 > iz1) cycle
                ! ish = shift index = 1...26*3+1
                ish = (imx+1 + (imy+1)*3 + (imz+1)*9 + 1)*3 - 2
                ! Loop over the cells
                do icz=iz0,iz1
                   celld1 = max(grid%grid_zdist(icz),zero)**2
                   do icy=iy0,iy1
                      celld2 = celld1 + max(grid%grid_ydist(icy),zero)**2
                      if (celld2 > cutsq) cycle
                      do icx=ix0,ix1
                         if (celld2 + max(grid%grid_xdist(icx),zero)**2 > cutsq) cycle
                         ind = icx + (icy-1)*grid%grid_nx + (icz-1)*grid%grid_nxy
                         n = grid%grid_n(ind)

                         ! Check wether entire cell is out of range
                         ! grid_grps(grid(ind)%start)     = minimum j-group index
                         ! grid_grps(grid(ind+1)%start-1) = maximum j-group index
                         if (n == 0) cycle
                         klo = grid%grid_start(ind)
                         khi = grid%grid_start(ind+1)-1

                         do k=klo,khi
                            jj = grid%grid_grps(k)
                            j = tholenblist(jj)

                            if ((j < j_min .or. j > j_max) .and. j > j_lomax) cycle
                            if (zonei == 3) then
                               ! FY
                               if (j > zonelist(2) .and. j <= zonelist(6)) cycle
                            endif

                            jglo = loc2glo_ind(j)+1
                            jtype = tholenbtype(jglo)
                            aa = max(itype, jtype)
                            ijtype = aa*(aa-3)/2 + itype + jtype
                            tholeij_val = tholeij(ijtype)

                            if (tholeij_val > 0) then
                               xj = xyzq(j)%x
                               yj = xyzq(j)%y
                               zj = xyzq(j)%z
                               xij = xim - xj
                               yij = yim - yj
                               zij = zim - zj
                               rsq = xij*xij + yij*yij + zij*zij
                               if (rsq < cutsq) then
                                  nbtholp = nbtholp + 1
                                  nbthol1(nbtholp) = iglo
                                  nbthol2(nbtholp) = jglo
                                  nbthol3(nbtholp) = tholeij_val
                               endif
                            endif

                         enddo ! do k=klo,khi
                      enddo  ! do icz=iz0,iz1
                   enddo  ! do icy=iy0,iy1
                enddo  ! do icx=ix0,ix1
             enddo ! do imz=-1,1
          enddo ! do imy=-1,1
       enddo ! do imx=-1,1
       !
    enddo ! do i=1,ntholenblist

    return
  end subroutine KERNEL_NAME

  ! *
  ! * Builds the grid for atoms
  ! *
#ifdef DOUBLE_PREC
  subroutine build_thole_grid_pd(natoml, atoml, glo2loc_ind, xyzq, grid)
#else
  subroutine build_thole_grid_ps(natoml, atoml, glo2loc_ind, xyzq, grid)
#endif
!    use number,only:zero, two
    implicit none
    ! Input / Output
    integer, intent(in) :: natoml, atoml(:), glo2loc_ind(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
#endif
    type(grid_t), intent(inout) :: grid
    ! Variables
    real(chm_real) cmx, cmy, cmz
    integer i, ii, ix, iy, iz, ind

    call build_tholenblist(natoml, atoml, glo2loc_ind, ntholenblist, tholenblist)
    call reset_grid(ntholenblist, grid)

    ! Assign atoms to grid
    do ii=1,ntholenblist
       i = tholenblist(ii)

       cmx = xyzq(i)%x - grid%grid_origx
       cmy = xyzq(i)%y - grid%grid_origy
       cmz = xyzq(i)%z - grid%grid_origz
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
          write (outu,'(a,80f7.2)') 'grid_dx=',grid%grid_dx(1:grid%grid_nx)
          write (outu,'(a,80f7.2)') 'grid_dy=',grid%grid_dy(1:grid%grid_ny)
          write (outu,'(a,80f7.2)') 'grid_dz=',grid%grid_dz(1:grid%grid_nz)
          write (outu,'(a,3f8.2)') 'xyzq=',xyzq(i)%x,xyzq(i)%y,xyzq(i)%z
          write (outu,'(a,3f8.2)') 'grid_origx,grid_origy,grid_origz=',&
               grid%grid_origx,grid%grid_origy,grid%grid_origz
          call wrndie(-5,'<nblist_builder>','build_thole_grid, outside the grid')
       endif
       ind = ix + (iy-1)*grid%grid_nx + (iz-1)*grid%grid_nxy
       grid%grid_n(ind) = grid%grid_n(ind) + 1
       grid%grid_ind(ii) = ind
    enddo

    call build_grid_grps(ntholenblist, grid)

    return
#ifdef DOUBLE_PREC
  end subroutine build_thole_grid_pd
#else
  end subroutine build_thole_grid_ps
#endif

#endif
