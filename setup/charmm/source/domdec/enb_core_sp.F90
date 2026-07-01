module enb_core_sp
#if KEY_DOMDEC==1 /*domdec_main*/
  use chm_kinds
  use, intrinsic :: iso_c_binding
  use domdec_local_types,only:xyzq_sp_t
  !use enb_core_wrapper_mod,only:invsqrt
  implicit none
  private

  integer, parameter :: VECLEN = 4

  integer niter

  real(chm_real4),parameter :: one_sp = 1.0_CHM_REAL4, two_sp = 2.0_CHM_REAL4, &
       three_sp = 3.0_CHM_REAL4

  public enb_vv_charmm_sp_fortran, enb_uu_coul_vdw_sp_fortran, &
       enb_uu_vdw_sp_fortran, enb_uv_coul_vdw_sp_fortran, enb_uv_vdw_sp_fortran

#if KEY_BLOCK==1
  public enb_coul_vdw_block_sp_fortran, enb_coul_vdw_block_softcore_sp_fortran
#endif
  
  !public enb_uv_coul_vdw_sp_sse
  public enb_vv_charmm_sp_sse_old

  ! defined in enb_core_sse_sp.c
  interface
!!$     subroutine enb_uv_coul_vdw_sp_sse(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
!!$          xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rc) &
!!$          bind(C)
!!$       import
!!$       integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
!!$       real(c_float),  intent(in) :: pftable(*)
!!$       real(c_double), intent(in) :: hinv
!!$       integer(c_int), intent(in) :: vdwtype(*)
!!$       real(c_float),  intent(in) :: vdwparam(*)
!!$       type(xyzq_sp_t),intent(in) :: xyzq(*)
!!$       real(c_double), intent(inout) :: force(*)
!!$       integer(c_int), intent(in) :: iscoord(*)
!!$       real(c_double), intent(inout) :: sforce(*)
!!$       real(c_double), intent(in) :: scoordtab(*)
!!$       real(c_double), intent(inout) :: coulpot, vdwpot
!!$       integer(c_int), intent(inout) :: nnnb
!!$       real(c_double), intent(in) :: rc
!!$     end subroutine enb_uv_coul_vdw_sp_sse
     
     subroutine enb_vv_charmm_sp_sse_old(ni, indi, indj, startj, pftable, hinv, xyzq, &
          force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, &
          c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, qOO, qOH, qHH, rc) &
          bind(C)
       import
       integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
       real(c_float),  intent(in) :: pftable(*)
       real(c_double), intent(in) :: hinv
       type(xyzq_sp_t), intent(in) :: xyzq(*)
       real(c_double), intent(inout) :: force(*)
       integer(c_int), intent(in) :: iscoord(*)
       real(c_double), intent(inout) :: sforce(*)
       real(c_float), intent(in) :: scoordtab(*)
       real(c_double), intent(inout) :: coulpot, vdwpot
       integer(c_int), intent(inout) :: nnnb
       real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
       real(c_double), intent(in) :: qOO, qOH, qHH, rc
     end subroutine enb_vv_charmm_sp_sse_old
     
  end interface
  
contains
  ! ##############################################################################################
  ! ################################# SINGLE PRECISION ###########################################
  ! ##############################################################################################

  subroutine enb_uu_vdw_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, vdwpot, nnnb, rcd)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) :: force(*), sforce(*), vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    ! Single precision
    real(chm_real4) c6, c12, rc, hinv
    real(chm_real4) xi, yi, zi
    real(chm_real4) sx, sy, sz
    real(chm_real4) xj, yj, zj
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
    real(chm_real4) fij, fijx, fijy, fijz
#if KEY_LJPME==1
    real(chm_real4) c6i, c6ij
#endif
    ! Double precision
    real(chm_real) fix, fiy, fiz

    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc
    hinv = real(hinvd,kind=chm_real4)

!$omp barrier
!$omp do schedule(dynamic)
    do i=1,ni
       ! Take periodic shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i index
       ii = indi(i)
       ii3 = ii*3 - 2
       ! Coordinates for atom i
       xi = sx + xyzq(ii)%x
       yi = sy + xyzq(ii)%y
       zi = sz + xyzq(ii)%z
       ! VdW parameters for atom i
       ia = vdwtype(ii)
#if KEY_LJPME==1
       c6i = xyzq(ii)%c6
#endif
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero atom i forces
       fix = zero
       fiy = zero
       fiz = zero
       do j=j0,j1
          jj = indj(j)
          jj3 = jj*3 - 2
          ! Coordinates for atom j
          xj = xyzq(jj)%x
          yj = xyzq(jj)%y
          zj = xyzq(jj)%z
          ! Calculate distance
          dx = xi - xj
          dy = yi - yj
          dz = zi - zj
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! VdW parameters for atom j
             ja = vdwtype(jj)
             aa = max(ja, ia)
             ivdw = aa*(aa-3) + 2*(ja + ia) - 1
             c6 = vdwparam(ivdw)
             c12 = vdwparam(ivdw+1)
             ! Calculate lookup table index t
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             c6ij = c6i*xyzq(jj)%c6
             t = 12*t - 11
#else
             t = 8*t - 7
#endif
             ! VdW attraction lookup
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion lookup
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+8)          ! a0
             a1 = pftable(t+9)          ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             ! Store forces
             fix = fix + fijx
             fiy = fiy + fijy
             fiz = fiz + fijz
             force(jj3)   = force(jj3)   - fijx
             force(jj3+1) = force(jj3+1) - fijy
             force(jj3+2) = force(jj3+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       force(ii3)   = force(ii3)   + fix
       force(ii3+1) = force(ii3+1) + fiy
       force(ii3+2) = force(ii3+2) + fiz
       ! Store forces for periodic shift
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_uu_vdw_sp_fortran

  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! *
  subroutine enb_uu_coul_vdw_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rcd, qscaled)
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) :: force(*), sforce(*), coulpot, vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd, qscaled
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    ! Single precision
    real(chm_real4) qi, qq, c6, c12, qscale, hinv, rc
    real(chm_real4) xi, yi, zi
    real(chm_real4) sx, sy, sz
    real(chm_real4) xj, yj, zj
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
    real(chm_real4) fij, fijx, fijy, fijz
    ! Double precision
    real(chm_real) fix, fiy, fiz
#if KEY_LJPME==1
    real(chm_real4) c6i, c6ij
#endif

    hinv = real(hinvd,kind=chm_real4)
    qscale = real(qscaled,kind=chm_real4)
    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc

!$omp barrier
!$omp do schedule(dynamic)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i index
       ii = indi(i)
       ! Coordinates for atom i
       xi = sx + xyzq(ii)%x
       yi = sy + xyzq(ii)%y
       zi = sz + xyzq(ii)%z
       ! Parameters for atom i
       qi = qscale*xyzq(ii)%q
       ia = vdwtype(ii)
#if KEY_LJPME==1
       c6i = xyzq(ii)%c6
#endif
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero atom i forces
       fix = zero
       fiy = zero
       fiz = zero
       do j=j0,j1
          jj = indj(j)
          ! Coordinates for atom j
          xj = xyzq(jj)%x
          yj = xyzq(jj)%y
          zj = xyzq(jj)%z
          ! Calculate distance
          dx = xi - xj
          dy = yi - yj
          dz = zi - zj
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Parameters for atom j
             qq = qi*xyzq(jj)%q
             ja = vdwtype(jj)
             aa = max(ja, ia)
             ivdw = aa*(aa-3) + 2*(ja + ia) - 1
             c6 = vdwparam(ivdw)
             c12 = vdwparam(ivdw+1)
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             c6ij = c6i*xyzq(jj)%c6
             t = 16*t - 15
#else
             t = 12*t - 11
#endif
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             coulpot = coulpot + qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix = fix + fijx
             fiy = fiy + fijy
             fiz = fiz + fijz
             jj = jj*3 - 2
             force(jj)   = force(jj)   - fijx
             force(jj+1) = force(jj+1) - fijy
             force(jj+2) = force(jj+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       ii = ii*3 - 2       
       force(ii)   = force(ii)   + fix
       force(ii+1) = force(ii+1) + fiy
       force(ii+2) = force(ii+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_uu_coul_vdw_sp_fortran

  subroutine enb_uv_coul_vdw_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rcd, qscaled)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) :: force(*), sforce(*), coulpot, vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd, qscaled
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, iaO, iaH, ja, aa, ivdw, t
    ! Single precision
    real(chm_real4) qO, qH, qj, qq, c6, c12, qscale, hinv, rc
    real(chm_real4) sx, sy, sz
    real(chm_real4) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real4) xj1, yj1, zj1
    real(chm_real4) fij, fijx, fijy, fijz
    real(chm_real4) dx, dy, dz, rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real4) c6O, c6H, c6j, c6ij
#endif
    ! Double precision
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    real(chm_real) fjx1, fjy1, fjz1

    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc
    qscale = real(qscaled,kind=chm_real4)
    hinv = real(hinvd,kind=chm_real4)

    ! All solvent molecules have the same charges and VdW parameters => pre-load them here
    ii = indi(1)
    qO = qscale*xyzq(ii)%q
    qH = qscale*xyzq(ii+1)%q
    iaO = vdwtype(ii)
    iaH = vdwtype(ii+1)
#if KEY_LJPME==1
    c6O = xyzq(ii)%c6
    c6H = xyzq(ii+1)%c6
#endif

!$omp barrier
!$omp do schedule(dynamic)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i index
       ii = indi(i)
       ii3 = ii*3 - 2
       ! Coordinates for solvent i
       xi1 = sx + xyzq(ii)%x
       yi1 = sy + xyzq(ii)%y
       zi1 = sz + xyzq(ii)%z
       xi2 = sx + xyzq(ii+1)%x
       yi2 = sy + xyzq(ii+1)%y
       zi2 = sz + xyzq(ii+1)%z
       xi3 = sx + xyzq(ii+2)%x
       yi3 = sy + xyzq(ii+2)%y
       zi3 = sz + xyzq(ii+2)%z
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero solvent i forces
       fix1 = zero
       fiy1 = zero
       fiz1 = zero
       fix2 = zero
       fiy2 = zero
       fiz2 = zero
       fix3 = zero
       fiy3 = zero
       fiz3 = zero
       do j=j0,j1
          jj = indj(j)
          jj3 = jj*3 - 2
          ! Coordinates for atom j
          xj1 = xyzq(jj)%x
          yj1 = xyzq(jj)%y
          zj1 = xyzq(jj)%z
          qj = xyzq(jj)%q
#if KEY_LJPME==1
          c6j = xyzq(jj)%c6
#endif
          ja = vdwtype(jj)
          fjx1 = zero
          fjy1 = zero
          fjz1 = zero
          ! 11 = O-atom
          dx = xi1 - xj1
          dy = yi1 - yj1
          dz = zi1 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Parameters for O - atom j
             qq = qj*qO
             aa = max(ja, iaO)
             ivdw = aa*(aa-3) + 2*(ja + iaO) - 1
             c6 = vdwparam(ivdw)
             c12 = vdwparam(ivdw+1)
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             c6ij = c6O*c6j
             t = 16*t - 15
#else
             t = 12*t - 11
#endif
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             coulpot = coulpot + qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix1 = fix1 + fijx
             fiy1 = fiy1 + fijy
             fiz1 = fiz1 + fijz
             fjx1 = fijx
             fjy1 = fijy
             fjz1 = fijz
             nnnb = nnnb + 1
          endif

          ! Parameters for H - atom j
          qq = qj*qH
          aa = max(ja, iaH)
          ivdw = aa*(aa-3) + 2*(ja + iaH) - 1
          c6 = vdwparam(ivdw)
          c12 = vdwparam(ivdw+1)
#if KEY_LJPME==1
          c6ij = c6H*c6j
#endif

          ! 21 = H-atom
          dx = xi2 - xj1
          dy = yi2 - yj1
          dz = zi2 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             t = 16*t - 15
#else
             t = 12*t - 11
#endif
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             coulpot = coulpot + qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix2 = fix2 + fijx
             fiy2 = fiy2 + fijy
             fiz2 = fiz2 + fijz
             fjx1 = fjx1 + fijx
             fjy1 = fjy1 + fijy
             fjz1 = fjz1 + fijz
             nnnb = nnnb + 1
          endif

          ! 31 = H-atom
          dx = xi3 - xj1
          dy = yi3 - yj1
          dz = zi3 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             t = 16*t - 15
#else
             t = 12*t - 11
#endif
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             coulpot = coulpot + qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix3 = fix3 + fijx
             fiy3 = fiy3 + fijy
             fiz3 = fiz3 + fijz
             fjx1 = fjx1 + fijx
             fjy1 = fjy1 + fijy
             fjz1 = fjz1 + fijz
             nnnb = nnnb + 1
          endif

          ! Store j forces
          force(jj3)   = force(jj3)   - fjx1
          force(jj3+1) = force(jj3+1) - fjy1
          force(jj3+2) = force(jj3+2) - fjz1
       enddo
       ! Store i forces
       force(ii3)   = force(ii3)   + fix1
       force(ii3+1) = force(ii3+1) + fiy1
       force(ii3+2) = force(ii3+2) + fiz1

       force(ii3+3) = force(ii3+3) + fix2
       force(ii3+4) = force(ii3+4) + fiy2
       force(ii3+5) = force(ii3+5) + fiz2

       force(ii3+6) = force(ii3+6) + fix3
       force(ii3+7) = force(ii3+7) + fiy3
       force(ii3+8) = force(ii3+8) + fiz3
       
       ! Store shifted forces
       sforce(is) = sforce(is) + fix1 + fix2 + fix3
       sforce(is+1) = sforce(is+1) + fiy1 + fiy2 + fiy3
       sforce(is+2) = sforce(is+2) + fiz1 + fiz2 + fiz3
    enddo
!$omp end do

    return
  end subroutine enb_uv_coul_vdw_sp_fortran

  subroutine enb_uv_vdw_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, vdwpot, nnnb, rcd)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) ::  force(*), sforce(*), vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, iaO, iaH, ja, aa, ivdw, t
    ! Single precision
    real(chm_real4) c6, c12, hinv, rc
    real(chm_real4) sx, sy, sz
    real(chm_real4) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real4) xj1, yj1, zj1
    real(chm_real4) fij, fijx, fijy, fijz
    real(chm_real4) dx, dy, dz, rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real4) c6O, c6H, c6j, c6ij
#endif
    ! Double precision
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    real(chm_real) fjx1, fjy1, fjz1

    hinv = real(hinvd,kind=chm_real4)
    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc

    ! All solvent molecules have the VdW parameters => pre-load the types
    ii = indi(1)
    iaO = vdwtype(ii)
    iaH = vdwtype(ii+1)
#if KEY_LJPME==1
    c6O = xyzq(ii)%c6
    c6H = xyzq(ii+1)%c6
#endif

!$omp barrier
!$omp do schedule(dynamic)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i coordinates
       ii = indi(i)
       ii3 = ii*3 - 2
       ! Coordinates for solvent i
       xi1 = sx + xyzq(ii)%x
       yi1 = sy + xyzq(ii)%y
       zi1 = sz + xyzq(ii)%z
       xi2 = sx + xyzq(ii+1)%x
       yi2 = sy + xyzq(ii+1)%y
       zi2 = sz + xyzq(ii+1)%z
       xi3 = sx + xyzq(ii+2)%x
       yi3 = sy + xyzq(ii+2)%y
       zi3 = sz + xyzq(ii+2)%z
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero solvent i forces
       fix1 = zero
       fiy1 = zero
       fiz1 = zero
       fix2 = zero
       fiy2 = zero
       fiz2 = zero
       fix3 = zero
       fiy3 = zero
       fiz3 = zero
       do j=j0,j1
          jj = indj(j)
          jj3 = jj*3 - 2
          ! Coordinates for atom j
          xj1 = xyzq(jj)%x
          yj1 = xyzq(jj)%y
          zj1 = xyzq(jj)%z
          fjx1 = zero
          fjy1 = zero
          fjz1 = zero
          ! Parameters for O - atom j
#if KEY_LJPME==1
          c6j = xyzq(jj)%c6
#endif
          ja = vdwtype(jj)
          aa = max(ja, iaO)
          ivdw = aa*(aa-3) + 2*(ja + iaO) - 1
          c6 = vdwparam(ivdw)
          c12 = vdwparam(ivdw+1)
          ! 11 = O-atom
          dx = xi1 - xj1
          dy = yi1 - yj1
          dz = zi1 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             c6ij = c6O*c6j
             t = 12*t - 11
#else
             t = 8*t - 7
#endif
             ! VdW attraction
             a0 = pftable(t)           ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+8)          ! a0
             a1 = pftable(t+9)          ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix1 = fix1 + fijx
             fiy1 = fiy1 + fijy
             fiz1 = fiz1 + fijz
             fjx1 = fijx
             fjy1 = fijy
             fjz1 = fijz
             nnnb = nnnb + 1
          endif

          ! Parameters for H - atom j
          aa = max(ja, iaH)
          ivdw = aa*(aa-3) + 2*(ja + iaH) - 1
          c6 = vdwparam(ivdw)
          c12 = vdwparam(ivdw+1)

          ! 21 = H-atom
          dx = xi2 - xj1
          dy = yi2 - yj1
          dz = zi2 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
#if KEY_LJPME==1
             t = 12*t - 11
#else
             t = 8*t - 7
#endif
             ! VdW attraction
             a0 = pftable(t)           ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+8)          ! a0
             a1 = pftable(t+9)          ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             vdwpot = vdwpot + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix2 = fix2 + fijx
             fiy2 = fiy2 + fijy
             fiz2 = fiz2 + fijz
             fjx1 = fjx1 + fijx
             fjy1 = fjy1 + fijy
             fjz1 = fjz1 + fijz
             nnnb = nnnb + 1
          endif

          ! 31 = H-atom
          dx = xi3 - xj1
          dy = yi3 - yj1
          dz = zi3 - zj1
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
             t = 8*t - 7
             ! VdW attraction
             a0 = pftable(t)           ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             vdwpot = vdwpot + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
             ! Calculate final force
             fij = fij*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix3 = fix3 + fijx
             fiy3 = fiy3 + fijy
             fiz3 = fiz3 + fijz
             fjx1 = fjx1 + fijx
             fjy1 = fjy1 + fijy
             fjz1 = fjz1 + fijz
             nnnb = nnnb + 1
          endif

          ! Store j forces
          force(jj3)   = force(jj3)   - fjx1
          force(jj3+1) = force(jj3+1) - fjy1
          force(jj3+2) = force(jj3+2) - fjz1
       enddo
       ! Store i forces
       force(ii3)   = force(ii3)   + fix1
       force(ii3+1) = force(ii3+1) + fiy1
       force(ii3+2) = force(ii3+2) + fiz1

       force(ii3+3) = force(ii3+3) + fix2
       force(ii3+4) = force(ii3+4) + fiy2
       force(ii3+5) = force(ii3+5) + fiz2

       force(ii3+6) = force(ii3+6) + fix3
       force(ii3+7) = force(ii3+7) + fiy3
       force(ii3+8) = force(ii3+8) + fiz3

       ! Store shifted forces
       sforce(is) = sforce(is) + fix1 + fix2 + fix3
       sforce(is+1) = sforce(is+1) + fiy1 + fiy2 + fiy3
       sforce(is+2) = sforce(is+2) + fiz1 + fiz2 + fiz3
    enddo
!$omp end do

    return
  end subroutine enb_uv_vdw_sp_fortran

  ! *
  ! * Solvent - Solvent, CHARMM version of TIP3P which has O-H and H-H VdW interactions
  ! *
  subroutine enb_vv_charmm_sp_fortran(ni, indi, indj, startj, pftable, hinvd, xyzq, &
       force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, &
       c6OOd, c12OOd, c6OHd, c12OHd, c6HHd, c12HHd,&
#if KEY_LJPME==1
       c6mult_OOd, c6mult_OHd, c6mult_HHd, &
#endif
       qOOd, qOHd, qHHd, rcd)
    use chm_kinds
    use number
    use consta,only:pi
    use ewald_1m,only:kappa
    use parallel,only:mynod
    implicit none
     ! Input / Output
    integer, intent(in) :: ni, indi(:), startj(:), indj(:)
    real(chm_real4), intent(in) :: pftable(:)
    real(chm_real), intent(in) :: hinvd
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: force(:)
    integer, intent(in) :: iscoord(:)
    real(chm_real), intent(inout) :: sforce(:), coulpot, vdwpot
    real(chm_real), intent(in) :: c6OOd, c12OOd, c6OHd, c12OHd, c6HHd, c12HHd
#if KEY_LJPME==1
    real(chm_real), intent(in) :: c6mult_OOd, c6mult_OHd, c6mult_HHd
    real(chm_real4) :: c6mult_OO, c6mult_OH, c6mult_HH
#endif
    real(chm_real), intent(in) :: qOOd, qOHd, qHHd, rcd
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    ! Single precision
    real(chm_real4) c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
    real(chm_real4) qOO, qOH, qHH, rc, hinv
    real(chm_real4) sx, sy, sz
    real(chm_real4) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real4) xj, yj, zj
    real(chm_real4) rc2
    real(chm_real4) dx1, dy1, dz1
    real(chm_real4) dx2, dy2, dz2
    real(chm_real4) dx3, dy3, dz3
    real(chm_real4) rsq1, rsq2, rsq3

    c6OO  = real(c6OOd,kind=chm_real4)
    c12OO = real(c12OOd,kind=chm_real4)
    c6OH  = real(c6OHd,kind=chm_real4)
    c12OH = real(c12OHd,kind=chm_real4)
    c6HH  = real(c6HHd,kind=chm_real4)
    c12HH = real(c12HHd,kind=chm_real4)
    qOO  = real(qOOd,kind=chm_real4)
    qOH  = real(qOHd,kind=chm_real4)
    qHH  = real(qHHd,kind=chm_real4)
#if KEY_LJPME==1
    c6mult_OO = real(c6mult_OOd, kind=chm_real4)
    c6mult_OH = real(c6mult_OHd, kind=chm_real4)
    c6mult_HH = real(c6mult_HHd, kind=chm_real4)
#endif

    rc = real(rcd,kind=chm_real4)
    hinv = real(hinvd,kind=chm_real4)

    rc2 = rc*rc

!    rc2 = real((rc + 1.0/hinvd)*(rc + 1.0/hinvd),kind=chm_real4)

!$omp barrier
!$omp do schedule(dynamic)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i coordinates
       ii = indi(i)
       ii3 = ii*3 - 2
       ! Coordinates for solvent ia
       xi1 = sx + xyzq(ii)%x
       yi1 = sy + xyzq(ii)%y
       zi1 = sz + xyzq(ii)%z
       xi2 = sx + xyzq(ii+1)%x
       yi2 = sy + xyzq(ii+1)%y
       zi2 = sz + xyzq(ii+1)%z
       xi3 = sx + xyzq(ii+2)%x
       yi3 = sy + xyzq(ii+2)%y
       zi3 = sz + xyzq(ii+2)%z
       ! j solvent loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero solvent i forces
       fix1 = zero
       fiy1 = zero
       fiz1 = zero
       fix2 = zero
       fiy2 = zero
       fiz2 = zero
       fix3 = zero
       fiy3 = zero
       fiz3 = zero
       do j=j0,j1
          jj = indj(j)
          jj3 = jj*3 - 2
          ! j O - i O H1 H2
          xj = xyzq(jj)%x
          yj = xyzq(jj)%y
          zj = xyzq(jj)%z
          dx1 = xi1 - xj
          dy1 = yi1 - yj
          dz1 = zi1 - zj
          rsq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq3 = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq1 < rc2 .or. rsq2 < rc2 .or. rsq3 < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, &
                  rsq1, rsq2, rsq3, hinv, &
                  qOO, qOH, c6OO, c6OH, c12OO, c12OH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, &
#endif
                  pftable, coulpot, vdwpot, &
                  fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, &
                  force(jj3),force(jj3+1),force(jj3+2))
             nnnb = nnnb + 3
          endif

          ! j H1 - i O H1 H2
          xj = xyzq(jj+1)%x
          yj = xyzq(jj+1)%y
          zj = xyzq(jj+1)%z
          dx1 = xi1 - xj
          dy1 = yi1 - yj
          dz1 = zi1 - zj
          rsq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq3 = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq1 < rc2 .or. rsq2 < rc2 .or. rsq3 < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3,&
                  rsq1, rsq2, rsq3, hinv, &
                  qOH, qHH, c6OH, c6HH, c12OH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OH, c6mult_HH, &
#endif
                  pftable, coulpot, vdwpot, &
                  fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, &
                  force(jj3+3),force(jj3+4),force(jj3+5))
             nnnb = nnnb + 3
          endif

          ! j H2 - i O H1 H2
          xj = xyzq(jj+2)%x
          yj = xyzq(jj+2)%y
          zj = xyzq(jj+2)%z
          dx1 = xi1 - xj
          dy1 = yi1 - yj
          dz1 = zi1 - zj
          rsq1 = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq2 = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq3 = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq1 < rc2 .or. rsq2 < rc2 .or. rsq3 < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3,&
                  rsq1, rsq2, rsq3, hinv, &
                  qOH, qHH, c6OH, c6HH, c12OH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OH, c6mult_HH, &
#endif
                  pftable, coulpot, vdwpot, &
                  fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, &
                  force(jj3+6),force(jj3+7),force(jj3+8))
             nnnb = nnnb + 3
          endif

       enddo

       ! Store i forces
       force(ii3)   = force(ii3)   + fix1
       force(ii3+1) = force(ii3+1) + fiy1
       force(ii3+2) = force(ii3+2) + fiz1

       force(ii3+3) = force(ii3+3) + fix2
       force(ii3+4) = force(ii3+4) + fiy2
       force(ii3+5) = force(ii3+5) + fiz2

       force(ii3+6) = force(ii3+6) + fix3
       force(ii3+7) = force(ii3+7) + fiy3
       force(ii3+8) = force(ii3+8) + fiz3

       ! Store shifted forces
       sforce(is) = sforce(is) + fix1 + fix2 + fix3
       sforce(is+1) = sforce(is+1) + fiy1 + fiy2 + fiy3
       sforce(is+2) = sforce(is+2) + fiz1 + fiz2 + fiz3
    enddo
!$omp end do

    return
  end subroutine enb_vv_charmm_sp_fortran

  !###########################################################################################
  !###########################################################################################
  !###########################################################################################

!!$  subroutine force_core_vv_vec(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, rsq1,rsq2,rsq3, hinv, &
!!$       q1, q2, c6_1, c6_2, c12_1, c12_2, pftable, coulpot, vdwpot, &
!!$       fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, forcexjj,forceyjj,forcezjj)
!!$    use chm_kinds
!!$    use number
!!$    implicit none
!!$    ! Input / Output
!!$    real(chm_real4), intent(in) :: dx1(VECLEN), dy1(VECLEN), dz1(VECLEN)
!!$    real(chm_real4), intent(in) :: dx2(VECLEN), dy2(VECLEN), dz2(VECLEN)
!!$    real(chm_real4), intent(in) :: dx3(VECLEN), dy3(VECLEN), dz3(VECLEN)
!!$    real(chm_real4), intent(inout) :: rsq1(VECLEN), rsq2(VECLEN), rsq3(VECLEN)
!!$    real(chm_real4), intent(in) :: hinv(VECLEN), q1(VECLEN), q2(VECLEN)
!!$    real(chm_real4), intent(in) :: c6_1(VECLEN), c6_2(VECLEN), c12_1(VECLEN), c12_2(VECLEN)
!!$    real(chm_real4), intent(in) :: pftable(*)
!!$    real(chm_real) coulpot, vdwpot
!!$    real(chm_real) fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3
!!$    real(chm_real) forcexjj(VECLEN), forceyjj(VECLEN), forcezjj(VECLEN)
!!$    integer ii,jj
!!$    ! Variables
!!$    integer t1, t2, t3, tt1(VECLEN), tt2(VECLEN), tt3(VECLEN)
!!$    ! Single precision
!!$    real(chm_real4) r1(VECLEN), r2(VECLEN), r3(VECLEN)
!!$    real(chm_real4) rs1(VECLEN), rs2(VECLEN), rs3(VECLEN)
!!$    real(chm_real4) ept1(VECLEN), ept2(VECLEN), ept3(VECLEN)
!!$    real(chm_real4) rinv1(VECLEN), rinv2(VECLEN), rinv3(VECLEN)
!!$    real(chm_real4) a0_1(VECLEN), a1_1(VECLEN), a2_1(VECLEN), a3_1(VECLEN)
!!$    real(chm_real4) a0_2(VECLEN), a1_2(VECLEN), a2_2(VECLEN), a3_2(VECLEN)
!!$    real(chm_real4) a0_3(VECLEN), a1_3(VECLEN), a2_3(VECLEN), a3_3(VECLEN)
!!$    real(chm_real4) fij1(VECLEN), fij2(VECLEN), fij3(VECLEN)
!!$    real(chm_real4) fijx(VECLEN), fijy(VECLEN), fijz(VECLEN)
!!$    ! Double precision accumulation variables
!!$    real(chm_real4) fijx_d(VECLEN), fijy_d(VECLEN), fijz_d(VECLEN), pott(VECLEN)
!!$
!!$    rinv1 = one_sp/sqrt(rsq1)
!!$    rinv2 = one_sp/sqrt(rsq2)
!!$    rinv3 = one_sp/sqrt(rsq3)
!!$
!!$    r1 = rinv1*rsq1
!!$    r2 = rinv2*rsq2
!!$    r3 = rinv3*rsq3
!!$
!!$    rs1 = r1*hinv
!!$    tt1 = int(rs1)
!!$    ept1 = rs1 - tt1
!!$    tt1 = 12*tt1 - 11
!!$
!!$    rs2 = r2*hinv
!!$    tt2 = int(rs2)
!!$    ept2 = rs2 - tt2
!!$    tt2 = 12*tt2 - 11
!!$
!!$    rs3 = r3*hinv
!!$    tt3 = int(rs3)
!!$    ept3 = rs3 - tt3
!!$    tt3 = 12*tt3 - 11
!!$
!!$    ! Coulomb interaction
!!$    a0_1 = pftable(tt1)             ! a0
!!$    a0_2 = pftable(tt2)             ! a0
!!$    a0_3 = pftable(tt3)             ! a0
!!$    a1_1 = pftable(tt1+1)           ! a1
!!$    a1_2 = pftable(tt2+1)           ! a1
!!$    a1_3 = pftable(tt3+1)           ! a1
!!$
!!$    a2_1 = pftable(tt1+2)*ept1      ! a2*ep
!!$    a2_2 = pftable(tt2+2)*ept2      ! a2*ep
!!$    a2_3 = pftable(tt3+2)*ept3      ! a2*ep
!!$
!!$    a3_1 = pftable(tt1+3)*ept1*ept1    ! a3*ep^2
!!$    a3_2 = pftable(tt2+3)*ept2*ept2    ! a3*ep^2
!!$    a3_3 = pftable(tt3+3)*ept3*ept3    ! a3*ep^2
!!$
!!$    pott = real(q1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1), kind=chm_real)
!!$    coulpot = coulpot + sum(pott)
!!$    pott = real(q2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2), kind=chm_real)
!!$    coulpot = coulpot + sum(pott)
!!$    pott = real(q2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3), kind=chm_real)
!!$    coulpot = coulpot + sum(pott)
!!$!    coulpot = coulpot + real(sum(q1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1)),kind=chm_real)
!!$!    coulpot = coulpot + real(sum(q2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2)),kind=chm_real)
!!$!    coulpot = coulpot + real(sum(q2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3)),kind=chm_real)
!!$
!!$    fij1 = q1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
!!$    fij2 = q2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
!!$    fij3 = q2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)
!!$
!!$    ! VdW attraction
!!$    a0_1 = pftable(tt1+4)             ! a0
!!$    a0_2 = pftable(tt2+4)             ! a0
!!$    a0_3 = pftable(tt3+4)             ! a0
!!$    a1_1 = pftable(tt1+5)           ! a1
!!$    a1_2 = pftable(tt2+5)           ! a1
!!$    a1_3 = pftable(tt3+5)           ! a1
!!$    
!!$    a2_1 = pftable(tt1+6)*ept1      ! a2*ep
!!$    a2_2 = pftable(tt2+6)*ept2      ! a2*ep
!!$    a2_3 = pftable(tt3+6)*ept3      ! a2*ep
!!$    
!!$    a3_1 = pftable(tt1+7)*ept1*ept1    ! a3*ep^2
!!$    a3_2 = pftable(tt2+7)*ept2*ept2    ! a3*ep^2
!!$    a3_3 = pftable(tt3+7)*ept3*ept3    ! a3*ep^2
!!$    
!!$    pott = real(c6_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$    pott = real(c6_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$    pott = real(c6_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$!    vdwpot = vdwpot + real(sum(c6_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1)),kind=chm_real)
!!$!    vdwpot = vdwpot + real(sum(c6_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2)),kind=chm_real)
!!$!    vdwpot = vdwpot + real(sum(c6_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3)),kind=chm_real)
!!$    
!!$    fij1 = fij1 + c6_1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
!!$    fij2 = fij2 + c6_2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
!!$    fij3 = fij3 + c6_2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)
!!$
!!$    ! VdW repulsion
!!$    a0_1 = pftable(tt1+8)             ! a0
!!$    a0_2 = pftable(tt2+8)             ! a0
!!$    a0_3 = pftable(tt3+8)             ! a0
!!$    a1_1 = pftable(tt1+9)           ! a1
!!$    a1_2 = pftable(tt2+9)           ! a1
!!$    a1_3 = pftable(tt3+9)           ! a1
!!$    
!!$    a2_1 = pftable(tt1+10)*ept1      ! a2*ep
!!$    a2_2 = pftable(tt2+10)*ept2      ! a2*ep
!!$    a2_3 = pftable(tt3+10)*ept3      ! a2*ep
!!$    
!!$    a3_1 = pftable(tt1+11)*ept1*ept1    ! a3*ep^2
!!$    a3_2 = pftable(tt2+11)*ept2*ept2    ! a3*ep^2
!!$    a3_3 = pftable(tt3+11)*ept3*ept3    ! a3*ep^2
!!$    
!!$    pott = real(c12_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$    pott = real(c12_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$    pott = real(c12_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3), kind=chm_real)
!!$    vdwpot = vdwpot + sum(pott)
!!$!    vdwpot = vdwpot + real(sum(c12_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1)),kind=chm_real)
!!$!    vdwpot = vdwpot + real(sum(c12_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2)),kind=chm_real)
!!$!    vdwpot = vdwpot + real(sum(c12_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3)),kind=chm_real)
!!$
!!$    fij1 = fij1 + c12_1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
!!$    fij2 = fij2 + c12_2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
!!$    fij3 = fij3 + c12_2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)
!!$
!!$    ! Calculate final force
!!$    fij1 = fij1*rinv1*hinv
!!$    fij2 = fij2*rinv2*hinv
!!$    fij3 = fij3*rinv3*hinv
!!$    
!!$    ! j X - i O
!!$    fijx = fij1*dx1
!!$    fijy = fij1*dy1
!!$    fijz = fij1*dz1
!!$
!!$    fijx_d = real(fijx, kind=chm_real)
!!$    fijy_d = real(fijy, kind=chm_real)
!!$    fijz_d = real(fijz, kind=chm_real)
!!$    fix1 = fix1 + sum(fijx_d)
!!$    fiy1 = fiy1 + sum(fijy_d)
!!$    fiz1 = fiz1 + sum(fijz_d)
!!$    
!!$    forcexjj = forcexjj - fijx
!!$    forceyjj = forceyjj - fijy
!!$    forcezjj = forcezjj - fijz
!!$    
!!$    ! j X - i H1
!!$    fijx = fij2*dx2
!!$    fijy = fij2*dy2
!!$    fijz = fij2*dz2
!!$    
!!$    fijx_d = real(fijx, kind=chm_real)
!!$    fijy_d = real(fijy, kind=chm_real)
!!$    fijz_d = real(fijz, kind=chm_real)
!!$    fix2 = fix2 + sum(fijx_d)
!!$    fiy2 = fiy2 + sum(fijy_d)
!!$    fiz2 = fiz2 + sum(fijz_d)
!!$    
!!$    forcexjj = forcexjj - fijx
!!$    forceyjj = forceyjj - fijy
!!$    forcezjj = forcezjj - fijz
!!$    
!!$    ! j X - i H2
!!$    fijx = fij3*dx3
!!$    fijy = fij3*dy3
!!$    fijz = fij3*dz3
!!$    
!!$    fijx_d = real(fijx, kind=chm_real)
!!$    fijy_d = real(fijy, kind=chm_real)
!!$    fijz_d = real(fijz, kind=chm_real)
!!$    fix3 = fix3 + sum(fijx_d)
!!$    fiy3 = fiy3 + sum(fijy_d)
!!$    fiz3 = fiz3 + sum(fijz_d)
!!$    
!!$    forcexjj = forcexjj - fijx
!!$    forceyjj = forceyjj - fijy
!!$    forcezjj = forcezjj - fijz
!!$    
!!$    return
!!$  end subroutine force_core_vv_vec

  subroutine force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, rsq1,rsq2,rsq3, hinv, &
       q1, q2, c6_1, c6_2, c12_1, c12_2, &
#if KEY_LJPME==1
       c6mult_1, c6mult_2, &
#endif
       pftable, coulpot, vdwpot, &
       fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, forcexjj,forceyjj,forcezjj)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    real(chm_real4), intent(in) :: dx1, dy1, dz1
    real(chm_real4), intent(in) :: dx2, dy2, dz2
    real(chm_real4), intent(in) :: dx3, dy3, dz3
    real(chm_real4), intent(inout) :: rsq1, rsq2, rsq3
    real(chm_real4), intent(in) :: hinv, q1, q2
    real(chm_real4), intent(in) :: c6_1, c6_2, c12_1, c12_2
#if KEY_LJPME==1
    real(chm_real4), intent(in) :: c6mult_1, c6mult_2
#endif
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real) coulpot, vdwpot
    real(chm_real) fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3
    real(chm_real) forcexjj, forceyjj, forcezjj
    integer ii,jj
    ! Variables
    integer t1, t2, t3, tt1, tt2, tt3
    ! Single precision
    real(chm_real4) r1, r2, r3
    real(chm_real4) rs1, rs2, rs3
    real(chm_real4) ept1, ept2, ept3
    real(chm_real4) rinv1, rinv2, rinv3
    real(chm_real4) a0_1, a1_1, a2_1, a3_1
    real(chm_real4) a0_2, a1_2, a2_2, a3_2
    real(chm_real4) a0_3, a1_3, a2_3, a3_3
    real(chm_real4) fij1, fij2, fij3
    real(chm_real4) fijx, fijy, fijz

    !rinv1 = hinv/sqrt(rsq1)
    !rinv2 = hinv/sqrt(rsq2)
    !rinv3 = hinv/sqrt(rsq3)

    !call invsqrt(rsq1, rinv1)
    !call invsqrt(rsq2, rinv2)
    !call invsqrt(rsq3, rinv3)
    rinv1 = one_sp/sqrt(rsq1)
    rinv2 = one_sp/sqrt(rsq2)
    rinv3 = one_sp/sqrt(rsq3)

    r1 = rinv1*rsq1
    r2 = rinv2*rsq2
    r3 = rinv3*rsq3

    rs1 = r1*hinv
    tt1 = int(rs1)
    ept1 = rs1 - tt1
#if KEY_LJPME==1
    tt1 = 16*tt1 - 15
#else
    tt1 = 12*tt1 - 11
#endif

    rs2 = r2*hinv
    tt2 = int(rs2)
    ept2 = rs2 - tt2
#if KEY_LJPME==1
    tt2 = 16*tt2 - 15
#else
    tt2 = 12*tt2 - 11
#endif

    rs3 = r3*hinv
    tt3 = int(rs3)
    ept3 = rs3 - tt3
#if KEY_LJPME==1
    tt3 = 16*tt3 - 15
#else
    tt3 = 12*tt3 - 11
#endif

    ! Coulomb interaction
    a0_1 = pftable(tt1)             ! a0
    a0_2 = pftable(tt2)             ! a0
    a0_3 = pftable(tt3)             ! a0
    a1_1 = pftable(tt1+1)           ! a1
    a1_2 = pftable(tt2+1)           ! a1
    a1_3 = pftable(tt3+1)           ! a1

    a2_1 = pftable(tt1+2)*ept1      ! a2*ep
    a2_2 = pftable(tt2+2)*ept2      ! a2*ep
    a2_3 = pftable(tt3+2)*ept3      ! a2*ep

    a3_1 = pftable(tt1+3)*ept1*ept1    ! a3*ep^2
    a3_2 = pftable(tt2+3)*ept2*ept2    ! a3*ep^2
    a3_3 = pftable(tt3+3)*ept3*ept3    ! a3*ep^2

    coulpot = coulpot + real(q1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1),kind=chm_real)
    coulpot = coulpot + real(q2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2),kind=chm_real)
    coulpot = coulpot + real(q2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3),kind=chm_real)

    fij1 = q1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
    fij2 = q2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
    fij3 = q2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)

    ! VdW attraction
    a0_1 = pftable(tt1+4)             ! a0
    a0_2 = pftable(tt2+4)             ! a0
    a0_3 = pftable(tt3+4)             ! a0
    a1_1 = pftable(tt1+5)           ! a1
    a1_2 = pftable(tt2+5)           ! a1
    a1_3 = pftable(tt3+5)           ! a1
    
    a2_1 = pftable(tt1+6)*ept1      ! a2*ep
    a2_2 = pftable(tt2+6)*ept2      ! a2*ep
    a2_3 = pftable(tt3+6)*ept3      ! a2*ep
    
    a3_1 = pftable(tt1+7)*ept1*ept1    ! a3*ep^2
    a3_2 = pftable(tt2+7)*ept2*ept2    ! a3*ep^2
    a3_3 = pftable(tt3+7)*ept3*ept3    ! a3*ep^2
    
    vdwpot = vdwpot + real(c6_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1),kind=chm_real)
    vdwpot = vdwpot + real(c6_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2),kind=chm_real)
    vdwpot = vdwpot + real(c6_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3),kind=chm_real)
    
    fij1 = fij1 + c6_1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
    fij2 = fij2 + c6_2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
    fij3 = fij3 + c6_2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)

    ! VdW repulsion
    a0_1 = pftable(tt1+8)             ! a0
    a0_2 = pftable(tt2+8)             ! a0
    a0_3 = pftable(tt3+8)             ! a0
    a1_1 = pftable(tt1+9)           ! a1
    a1_2 = pftable(tt2+9)           ! a1
    a1_3 = pftable(tt3+9)           ! a1
    
    a2_1 = pftable(tt1+10)*ept1      ! a2*ep
    a2_2 = pftable(tt2+10)*ept2      ! a2*ep
    a2_3 = pftable(tt3+10)*ept3      ! a2*ep
    
    a3_1 = pftable(tt1+11)*ept1*ept1    ! a3*ep^2
    a3_2 = pftable(tt2+11)*ept2*ept2    ! a3*ep^2
    a3_3 = pftable(tt3+11)*ept3*ept3    ! a3*ep^2
    
    vdwpot = vdwpot + real(c12_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1),kind=chm_real)
    vdwpot = vdwpot + real(c12_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2),kind=chm_real)
    vdwpot = vdwpot + real(c12_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3),kind=chm_real)
    
    fij1 = fij1 + c12_1*(a1_1 + two_sp*a2_1 + three_sp*a3_1)
    fij2 = fij2 + c12_2*(a1_2 + two_sp*a2_2 + three_sp*a3_2)
    fij3 = fij3 + c12_2*(a1_3 + two_sp*a2_3 + three_sp*a3_3)

#if KEY_LJPME==1
    ! LJPME grid correction, to remove multiplicative terms
    a0_1 = pftable(tt1+12)             ! a0
    a0_2 = pftable(tt2+12)             ! a0
    a0_3 = pftable(tt3+12)             ! a0
    a1_1 = pftable(tt1+13)           ! a1
    a1_2 = pftable(tt2+13)           ! a1
    a1_3 = pftable(tt3+13)           ! a1
    
    a2_1 = pftable(tt1+14)*ept1      ! a2*ep
    a2_2 = pftable(tt2+14)*ept2      ! a2*ep
    a2_3 = pftable(tt3+14)*ept3      ! a2*ep
    
    a3_1 = pftable(tt1+15)*ept1*ept1    ! a3*ep^2
    a3_2 = pftable(tt2+15)*ept2*ept2    ! a3*ep^2
    a3_3 = pftable(tt3+15)*ept3*ept3    ! a3*ep^2

    vdwpot = vdwpot + c6mult_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept1)
    vdwpot = vdwpot + c6mult_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept2)
    vdwpot = vdwpot + c6mult_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept3)
    fij1 = fij1 + c6mult_1*(a1_1 + two*a2_1 + three*a3_1)
    fij2 = fij2 + c6mult_2*(a1_2 + two*a2_2 + three*a3_2)
    fij3 = fij3 + c6mult_2*(a1_3 + two*a2_3 + three*a3_3)
#endif

    ! Calculate final force
    fij1 = fij1*rinv1*hinv
    fij2 = fij2*rinv2*hinv
    fij3 = fij3*rinv3*hinv
    
    ! j X - i O
    fijx = fij1*dx1
    fijy = fij1*dy1
    fijz = fij1*dz1
    
    fix1 = fix1 + fijx
    fiy1 = fiy1 + fijy
    fiz1 = fiz1 + fijz
    
    forcexjj = forcexjj - fijx
    forceyjj = forceyjj - fijy
    forcezjj = forcezjj - fijz
    
    ! j X - i H1
    fijx = fij2*dx2
    fijy = fij2*dy2
    fijz = fij2*dz2
    
    fix2 = fix2 + fijx
    fiy2 = fiy2 + fijy
    fiz2 = fiz2 + fijz
    
    forcexjj = forcexjj - fijx
    forceyjj = forceyjj - fijy
    forcezjj = forcezjj - fijz
    
    ! j X - i H2
    fijx = fij3*dx3
    fijy = fij3*dy3
    fijz = fij3*dz3
    
    fix3 = fix3 + fijx
    fiy3 = fiy3 + fijy
    fiz3 = fiz3 + fijz
    
    forcexjj = forcexjj - fijx
    forceyjj = forceyjj - fijy
    forcezjj = forcezjj - fijz
    
    return
  end subroutine force_core_vv

#if KEY_BLOCK==1
  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! *
  subroutine enb_coul_vdw_block_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rcd, qscaled, loc2glo_ind, iblckp)
    use number
    use lambdam,only:msld_nb_scale_enerforce
    use domdec_block,only:biflam_loc, biflam2_loc
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) :: force(*), sforce(*), coulpot, vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd, qscaled
    integer, intent(inout) :: nnnb
    integer, intent(in) :: loc2glo_ind(:), iblckp(:)
    ! Variables
    integer i, j, ii, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    integer ii_glo, ii_block, jj_glo, jj_block
    ! Single precision
    real(chm_real4) qi, qq, c6, c12, qscale, hinv, rc
    real(chm_real4) xi, yi, zi
    real(chm_real4) sx, sy, sz
    real(chm_real4) xj, yj, zj
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
    ! Double precision
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij_elec, pot_elec, fij_vdw, pot_vdw
    real(chm_real) fij, fijx, fijy, fijz

    hinv = real(hinvd,kind=chm_real4)
    qscale = real(qscaled,kind=chm_real4)
    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc

!$omp barrier
!$omp do schedule(dynamic) reduction(+:biflam_loc, biflam2_loc)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i index
       ii = indi(i)
       ! Atom i global index
       ii_glo = loc2glo_ind(ii)+1
       ! Atom i block index
       ii_block = iblckp(ii_glo)
       ! Coordinates for atom i
       xi = sx + xyzq(ii)%x
       yi = sy + xyzq(ii)%y
       zi = sz + xyzq(ii)%z
       ! Parameters for atom i
       qi = qscale*xyzq(ii)%q
       ia = vdwtype(ii)
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero atom i forces
       fix = zero
       fiy = zero
       fiz = zero
       do j=j0,j1
          jj = indj(j)
          ! Atom j global index
          jj_glo = loc2glo_ind(jj)+1
          ! Atom j block index
          jj_block = iblckp(jj_glo)
          ! Coordinates for atom j
          xj = xyzq(jj)%x
          yj = xyzq(jj)%y
          zj = xyzq(jj)%z
          ! Calculate distance
          dx = xi - xj
          dy = yi - yj
          dz = zi - zj
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Parameters for atom j
             qq = qi*xyzq(jj)%q
             ja = vdwtype(jj)
             aa = max(ja, ia)
             ivdw = aa*(aa-3) + 2*(ja + ia) - 1
             c6 = vdwparam(ivdw)
             c12 = vdwparam(ivdw+1)
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             rs = rsq*rinv*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
             t = 12*t - 11
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             pot_elec = real(qq*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_elec = real(qq*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             pot_vdw = real(c6*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_vdw = real(c6*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + real(c12*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_vdw = fij_vdw + real(c12*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! Calculate BLOCK
             call msld_nb_scale_enerforce(ii_block, jj_block, fij_elec, pot_elec, fij_vdw, pot_vdw, &
                  biflam_loc, biflam2_loc)
             coulpot = coulpot + pot_elec
             vdwpot = vdwpot + pot_vdw
             ! Calculate final force
             fij = (fij_elec + fij_vdw)*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix = fix + fijx
             fiy = fiy + fijy
             fiz = fiz + fijz
             jj = jj*3 - 2
             force(jj)   = force(jj)   - fijx
             force(jj+1) = force(jj+1) - fijy
             force(jj+2) = force(jj+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       ii = ii*3 - 2       
       force(ii)   = force(ii)   + fix
       force(ii+1) = force(ii+1) + fiy
       force(ii+2) = force(ii+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_coul_vdw_block_sp_fortran

  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! * With MSLD soft cores
  ! *
  subroutine enb_coul_vdw_block_softcore_sp_fortran(ni, indi, indj, startj, pftable, hinvd, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rcd, qscaled, loc2glo_ind, iblckp)
    use number
    use lambdam,only:msld_nbsoft_scale_enerforce,fullblcoep
    use domdec_block,only:biflam_loc, biflam2_loc
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(*), indj(*), startj(*), vdwtype(*), iscoord(*)
    real(chm_real), intent(in) :: hinvd
    real(chm_real4), intent(in) :: vdwparam(*)
    type(xyzq_sp_t), intent(in) :: xyzq(*)
    real(chm_real4), intent(in) :: pftable(*)
    real(chm_real), intent(inout) :: force(*), sforce(*), coulpot, vdwpot
    real(chm_real4), intent(in) :: scoordtab(*)
    real(chm_real), intent(in) :: rcd, qscaled
    integer, intent(inout) :: nnnb
    integer, intent(in) :: loc2glo_ind(:), iblckp(:)
    ! Variables
    integer i, j, ii, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    integer ii_glo, ii_block, jj_glo, jj_block
    ! Single precision
    real(chm_real4) qi, qq, c6, c12, qscale, hinv, rc
    real(chm_real4) xi, yi, zi
    real(chm_real4) sx, sy, sz
    real(chm_real4) xj, yj, zj
    real(chm_real4) dx, dy, dz
    real(chm_real4) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
    ! Double precision
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij_elec, pot_elec, fij_vdw, pot_vdw
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real4) r, rp, rcsoft
    real(chm_real) scale, rdivrcs, drpdr, drpds

    hinv = real(hinvd,kind=chm_real4)
    qscale = real(qscaled,kind=chm_real4)
    rc = real(rcd,kind=chm_real4)
    rc2 = rc*rc

!$omp barrier
!$omp do schedule(dynamic) reduction(+:biflam_loc, biflam2_loc)
    do i=1,ni
       ! Shift coordinates
       is = iscoord(i)
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Atom i index
       ii = indi(i)
       ! Atom i global index
       ii_glo = loc2glo_ind(ii)+1
       ! Atom i block index
       ii_block = iblckp(ii_glo)
       ! Coordinates for atom i
       xi = sx + xyzq(ii)%x
       yi = sy + xyzq(ii)%y
       zi = sz + xyzq(ii)%z
       ! Parameters for atom i
       qi = qscale*xyzq(ii)%q
       ia = vdwtype(ii)
       ! j atom loop limits
       j0 = startj(i)
       j1 = startj(i+1) - 1
       ! Zero atom i forces
       fix = zero
       fiy = zero
       fiz = zero
       do j=j0,j1
          jj = indj(j)
          ! Atom j global index
          jj_glo = loc2glo_ind(jj)+1
          ! Atom j block index
          jj_block = iblckp(jj_glo)
          ! Coordinates for atom j
          xj = xyzq(jj)%x
          yj = xyzq(jj)%y
          zj = xyzq(jj)%z
          ! Calculate distance
          dx = xi - xj
          dy = yi - yj
          dz = zi - zj
          rsq = dx*dx + dy*dy + dz*dz
          if (rsq < rc2) then
             ! Parameters for atom j
             qq = qi*xyzq(jj)%q
             ja = vdwtype(jj)
             aa = max(ja, ia)
             ivdw = aa*(aa-3) + 2*(ja + ia) - 1
             c6 = vdwparam(ivdw)
             c12 = vdwparam(ivdw+1)
             ! Calculate table index
             rinv = one_sp/sqrt(rsq)
             !call invsqrt(rsq, rinv)
             scale = fullblcoep(ii_block,jj_block)
             r=sqrt(rsq)
             rcsoft = 2.0*sqrt(4.0)*(1.0-scale)
             if (r < rcsoft) then
                rdivrcs = r / rcsoft
                rp = 1.0 - 0.5 * rdivrcs
                rp = rp*rdivrcs*rdivrcs*rdivrcs + 0.5
                drpdr = 3.0 - 2.0 * rdivrcs
                drpdr = drpdr*rdivrcs*rdivrcs
                drpds = rp - drpdr*rdivrcs
                drpds = (-2.0 * sqrt(4.0)) * drpds
                drpds = drpds * hinv   ! because fij is multiplied by hinv after msld_nbsoft_scale_enerforce
                rp = rp*rcsoft   ! rp = rcsoft*(0.5+rdivrcs**3-0.5*rdivrcs**4)
             else
                drpdr = 1.0
                drpds = 0.0
                rp = r
             endif
             rs = rp*hinv
             t = int(rs)
             ep = rs - t
             ep2 = ep*ep
             t = 12*t - 11
             ! Coulomb interaction
             a0 = pftable(t)             ! a0
             a1 = pftable(t+1)           ! a1
             a2ep = pftable(t+2)*ep      ! a2*ep
             a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
             pot_elec = real(qq*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_elec = real(qq*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             pot_vdw = real(c6*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_vdw = real(c6*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + real(c12*(a0 + (a1 + a2ep + a3ep2)*ep),kind=chm_real)
             fij_vdw = fij_vdw + real(c12*(a1 + two_sp*a2ep + three_sp*a3ep2),kind=chm_real)
             ! Calculate BLOCK
             call msld_nbsoft_scale_enerforce(ii_block, jj_block, fij_elec, pot_elec, fij_vdw, pot_vdw, &
                  drpdr, drpds, &
                  biflam_loc, biflam2_loc)
             coulpot = coulpot + pot_elec
             vdwpot = vdwpot + pot_vdw
             ! Calculate final force
             fij = (fij_elec + fij_vdw)*hinv*rinv
             ! Calculate force components
             fijx = fij*dx
             fijy = fij*dy
             fijz = fij*dz
             fix = fix + fijx
             fiy = fiy + fijy
             fiz = fiz + fijz
             jj = jj*3 - 2
             force(jj)   = force(jj)   - fijx
             force(jj+1) = force(jj+1) - fijy
             force(jj+2) = force(jj+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       ii = ii*3 - 2       
       force(ii)   = force(ii)   + fix
       force(ii+1) = force(ii+1) + fiy
       force(ii+2) = force(ii+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_coul_vdw_block_softcore_sp_fortran
#endif
  
#endif /* (domdec_main)*/
end module enb_core_sp

