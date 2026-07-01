module enb_core

#if KEY_DOMDEC==1 /*domdec_main*/

  use chm_kinds
  use, intrinsic :: iso_c_binding
  use domdec_local_types,only:xyzq_dp_t, xyzq_sp_t
  ! Simple wrapper module for the nonbonded energy core routines
  private

  ! Public subroutines
  public enb_uu_coul_vdw, enb_uu_vdw, enb_uv_coul_vdw, enb_uv_vdw, enb_vv

  ! defined in enb_core_vec.cpp (use of enb_core_sse.c is deprecated)
  interface

    subroutine enb_uu_coul_vdw_sse(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
         xyzq, force, iscoord, sforce, scoordtab, &
         coulpot, vdwpot, nnnb, roff, ron, vdw_type, qscale) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
      real(c_double), intent(in) :: qscale
    end subroutine enb_uu_coul_vdw_sse

    subroutine enb_uu_vdw_sse(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
         xyzq, force, iscoord, sforce, scoordtab, &
         vdwpot, nnnb, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uu_vdw_sse

    subroutine enb_uv_coul_vdw_sse(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
         xyzq, force, iscoord, sforce, scoordtab, &
         coulpot, vdwpot, nnnb, roff, ron, vdw_type, qscale) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
      real(c_double), intent(in) :: qscale
    end subroutine enb_uv_coul_vdw_sse

    subroutine enb_uv_vdw_sse(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
         xyzq, force, iscoord, sforce, scoordtab, &
         vdwpot, nnnb, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      integer(c_int), intent(in) :: vdwtype(*)
      real(c_double), intent(in) :: vdwparam(*)
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_uv_vdw_sse

    subroutine enb_vv_charmm_sse_old(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, qOO, qOH, qHH, roff, ron, vdw_type) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
      real(c_double), intent(in) :: qOO, qOH, qHH
      real(c_double), intent(in) :: roff, ron
      integer(c_int), intent(in) :: vdw_type
    end subroutine enb_vv_charmm_sse_old

    subroutine enb_vv_charmm_avx_old(ni, indi, indj, startj, pftable, hinv, xyzq, &
         force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, &
         c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, qOO, qOH, qHH, rc) &
         bind(C)
      import
      integer(c_int), intent(in) :: ni, indi(*), indj(*), startj(*)
      real(c_double), intent(in) :: pftable(*), hinv
      type(xyzq_dp_t), intent(in) :: xyzq(*)
      real(c_double), intent(inout) :: force(*)
      integer(c_int), intent(in) :: iscoord(*)
      real(c_double), intent(inout) :: sforce(*)
      real(c_double), intent(in) :: scoordtab(*)
      real(c_double), intent(inout) :: coulpot, vdwpot
      integer(c_int), intent(inout) :: nnnb
      real(c_double), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
      real(c_double), intent(in) :: qOO, qOH, qHH, rc
    end subroutine enb_vv_charmm_avx_old

  end interface

contains


  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! *
  subroutine enb_uu_coul_vdw(nblist, nb_pair, vdwtype, vdwparam, xyzq, force, &
       sforce, coulpot, vdwpot, scoordtab, nnnb, q_single)
    use consta,only:ccelec
    use inbnd,only:eps
    use number,only:one
    use nblist_types,only:nblist_pair_t, dpsp_t, nb_pair_t
    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, SIMD_AVX2_FMA
    use domdec_local_types,only:xyzq_dpsp_t
    use enb_core_sp,only:enb_uu_coul_vdw_sp_fortran
    use enb_core_wrapper_mod,only:enb_uu_coul_vdw_dp_sse, enb_uu_coul_vdw_sp_sse, &
         enb_uu_coul_vdw_dp_avx, enb_uu_coul_vdw_sp_avx
#if KEY_BLOCK==1
    use domdec_local,only:loc2glo_ind
    use block_ltm,only:iblckp
    use lambdam,only:iqldm_softcore
    use enb_core_sp,only:enb_coul_vdw_block_sp_fortran, enb_coul_vdw_block_softcore_sp_fortran
#endif 
    implicit none
    ! Input / Output
    type(nblist_pair_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) :: force(:), sforce(:), coulpot, vdwpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) :: nnnb
    logical, intent(in) :: q_single
    ! Variables
    real(chm_real) qscale

    qscale = one !ccelec/eps

#if KEY_BLOCK==1
    if (nblist%lblock) then
       if (q_single) then
          if (iqldm_softcore==0) then
             call enb_coul_vdw_block_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale, loc2glo_ind, iblckp)
          else
             call enb_coul_vdw_block_softcore_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale, loc2glo_ind, iblckp)
          endif
       else
          if (iqldm_softcore==0) then
             call enb_coul_vdw_block_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale, loc2glo_ind, iblckp)
          else
             call enb_coul_vdw_block_softcore_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale, loc2glo_ind, iblckp)
          endif
       endif
    else
#endif 
       if (q_single) then
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
               simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_uu_coul_vdw_sp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_uu_coul_vdw_sp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          else
             call enb_uu_coul_vdw_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale)
          endif
       else
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
               simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_uu_coul_vdw_dp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_uu_coul_vdw_dp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          else
             call enb_uu_coul_vdw_dp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, coulpot, vdwpot, nnnb, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, qscale)
          endif
       endif
#if KEY_BLOCK==1
    endif
#endif 

    return
  end subroutine enb_uu_coul_vdw

  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! *
  subroutine enb_uu_coul_vdw_dp_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, &
       coulpot, vdwpot, nnnb, rc, qscale)
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: coulpot, vdwpot
    real(chm_real), intent(in) :: rc, qscale
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    real(chm_real) qi, qq, c6, c12
    real(chm_real) xi, yi, zi
    real(chm_real) sx, sy, sz
    real(chm_real) xj, yj, zj
    real(chm_real) dx, dy, dz
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6i, c6ij
#endif

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
       ii3 = ii*3-2
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
          jj3 = jj*3-2
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
             rinv = one/sqrt(rsq)
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
             force(jj3)   = force(jj3)   - fijx
             force(jj3+1) = force(jj3+1) - fijy
             force(jj3+2) = force(jj3+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       force(ii3)   = force(ii3)   + fix
       force(ii3+1) = force(ii3+1) + fiy
       force(ii3+2) = force(ii3+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_uu_coul_vdw_dp_fortran

  ! *
  ! * Solute-solute with Vdw interactions
  ! *
  subroutine enb_uu_vdw(nblist, nb_pair, vdwtype, vdwparam, xyzq, force, &
       sforce, vdwpot, scoordtab, nnnb, q_single)
    use number,only:one
    use nblist_types,only:nblist_pair_t, dpsp_t, nb_pair_t
    use domdec_local_types,only:xyzq_dpsp_t
    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, SIMD_AVX2_FMA
    use enb_core_wrapper_mod,only:enb_uu_vdw_dp_sse, enb_uu_vdw_sp_sse, &
         enb_uu_vdw_dp_avx, enb_uu_vdw_sp_avx
    use enb_core_sp,only:enb_uu_vdw_sp_fortran
    implicit none
    ! Input / Output
    type(nblist_pair_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) :: force(:), sforce(:), vdwpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) :: nnnb
    logical, intent(in) :: q_single

#if KEY_BLOCK==1
    if (nblist%lblock) then
       call wrndie(-5,'<enb_core>',&
            'How did we end up here? BLOCK does not have VdW-only kernels')
    else
#endif 
       if (q_single) then
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
               simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_uu_vdw_sp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_uu_vdw_sp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
                  scoordtab%sp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          else
             call enb_uu_vdw_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, &
                  vdwpot, nnnb, nb_pair(nblist%nb_pair_ind)%ctofnb)
          endif
       else
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
               simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_uu_vdw_dp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_uu_vdw_dp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
                  scoordtab%dp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
                  nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
          else
             call enb_uu_vdw_dp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, scoordtab%dp, &
                  vdwpot, nnnb, nb_pair(nblist%nb_pair_ind)%ctofnb)
          endif
       endif
#if KEY_BLOCK==1
    endif
#endif 

    return
  end subroutine enb_uu_vdw

  subroutine enb_uu_vdw_dp_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, vdwpot, nnnb, rc)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: vdwpot
    real(chm_real), intent(in) :: rc
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    real(chm_real) c6, c12
    real(chm_real) xi, yi, zi
    real(chm_real) sx, sy, sz
    real(chm_real) xj, yj, zj
    real(chm_real) dx, dy, dz
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6i, c6ij
#endif

    ! rc2 = square of non-bonded cut-off
    rc2 = rc*rc
    
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
          jj3 = jj*3  - 2
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
             rinv = one/sqrt(rsq)
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
  end subroutine enb_uu_vdw_dp_fortran

  ! *
  ! * Solute - Solvent with Coulomb and Vdw interactions
  ! *
  subroutine enb_uv_coul_vdw(nblist, nb_pair, vdwtype, vdwparam, xyzq, &
       force, sforce, coulpot, vdwpot, scoordtab, nnnb, q_single)
    use consta,only:ccelec
    use inbnd,only:eps
    use number
    use nblist_types,only:nblist_pair_t, dpsp_t, nb_pair_t
    use domdec_local_types,only:xyzq_dpsp_t
    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, SIMD_AVX2_FMA
    use enb_core_wrapper_mod,only:enb_uv_coul_vdw_dp_sse, enb_uv_coul_vdw_sp_sse, &
         enb_uv_coul_vdw_dp_avx, enb_uv_coul_vdw_sp_avx
    use enb_core_sp,only:enb_uv_coul_vdw_sp_fortran
    implicit none
    ! Input / Output
    type(nblist_pair_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) :: force(:), sforce(:), coulpot, vdwpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) :: nnnb
    logical, intent(in) :: q_single
    ! Variables
    real(chm_real) qscale

#if KEY_BLOCK==1
    if (nblist%lblock) then
       call wrndie(-5,'<enb_core>',&
            'How did we end up here? BLOCK does not have Solute-Solvent kernels')
    endif
#endif 

    if (nblist%ni == 0) return

    qscale = one !ccelec/eps

    if (q_single) then
       if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
          call enb_uv_coul_vdw_sp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
               scoordtab%sp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       elseif (simd_version == SIMD_SSE) then
          call enb_uv_coul_vdw_sp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
               scoordtab%sp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       else
          call enb_uv_coul_vdw_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
               scoordtab%sp, coulpot, vdwpot, nnnb, &
               nb_pair(nblist%nb_pair_ind)%ctofnb, qscale)
       endif
    else
       if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
          call enb_uv_coul_vdw_dp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
               scoordtab%dp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       elseif (simd_version == SIMD_SSE) then
          call enb_uv_coul_vdw_dp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
               scoordtab%dp, coulpot, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       else
          call enb_uv_coul_vdw_dp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
               scoordtab%dp, coulpot, vdwpot, nnnb, &
               nb_pair(nblist%nb_pair_ind)%ctofnb, qscale)
       endif
    endif

    return
  end subroutine enb_uv_coul_vdw

  subroutine enb_uv_coul_vdw_dp_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, rc, qscale)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: coulpot, vdwpot
    real(chm_real), intent(in) :: rc, qscale
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, iaO, iaH, ja, aa, ivdw, t
    real(chm_real) qO, qH, qj, qq, c6, c12
    real(chm_real) sx, sy, sz
    real(chm_real) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real) xj1, yj1, zj1
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    real(chm_real) fjx1, fjy1, fjz1
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) dx, dy, dz, rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6O, c6H, c6j, c6ij
#endif

    rc2 = rc*rc

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
             rinv = one/sqrt(rsq)
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
             rinv = one/sqrt(rsq)
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
             rinv = one/sqrt(rsq)
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
  end subroutine enb_uv_coul_vdw_dp_fortran

  ! *
  ! * Solute - Solvent with Vdw interactions
  ! *
  subroutine enb_uv_vdw(nblist, nb_pair, vdwtype, vdwparam, xyzq, &
       force, sforce, vdwpot, scoordtab, nnnb, q_single)
    use chm_kinds
    use number
    use nblist_types,only:nblist_pair_t, dpsp_t, nb_pair_t
    use domdec_local_types,only:xyzq_dpsp_t
    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, SIMD_AVX2_FMA
    use enb_core_wrapper_mod,only:enb_uv_vdw_dp_sse, enb_uv_vdw_sp_sse, &
         enb_uv_vdw_dp_avx, enb_uv_vdw_sp_avx
    use enb_core_sp,only:enb_uv_vdw_sp_fortran
    implicit none
    ! Input / Output
    type(nblist_pair_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) :: force(:), sforce(:), vdwpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) ::  nnnb
    logical, intent(in) :: q_single
    ! Variables

#if KEY_BLOCK==1
    if (nblist%lblock) then
       call wrndie(-5,'<enb_core>',&
            'How did we end up here? BLOCK does not have Solute-Solvent kernels')
    endif
#endif 

    if (nblist%ni == 0) return

    if (q_single) then
       if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
          call enb_uv_vdw_sp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
               scoordtab%sp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       elseif (simd_version == SIMD_SSE) then
          call enb_uv_vdw_sp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, &
               scoordtab%sp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       else
          call enb_uv_vdw_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%sp, xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, &
               vdwpot, nnnb, nb_pair(nblist%nb_pair_ind)%ctofnb)
       endif
    else
       if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
          call enb_uv_vdw_dp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
               scoordtab%dp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       elseif (simd_version == SIMD_SSE) then
          call enb_uv_vdw_dp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, &
               scoordtab%dp, vdwpot, nb_pair(nblist%nb_pair_ind)%ctofnb, &
               nb_pair(nblist%nb_pair_ind)%ctonnb, nb_pair(nblist%nb_pair_ind)%vdw_type)
       else
          call enb_uv_vdw_dp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
               nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
               vdwtype, vdwparam%dp, xyzq%dp, force, nblist%iscoord, sforce, scoordtab%dp, &
               vdwpot, nnnb, nb_pair(nblist%nb_pair_ind)%ctofnb)
       endif
    endif

    return
  end subroutine enb_uv_vdw


  subroutine enb_uv_vdw_dp_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, vdwpot, nnnb, rc)
    use chm_kinds
    use number
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: vdwpot
    real(chm_real), intent(in) :: rc
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, iaO, iaH, ja, aa, ivdw, t
    real(chm_real) c6, c12
    real(chm_real) sx, sy, sz
    real(chm_real) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real) xj1, yj1, zj1
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    real(chm_real) fjx1, fjy1, fjz1
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) dx, dy, dz, rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6O, c6H, c6j, c6ij
#endif

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
             rinv = one/sqrt(rsq)
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
             rinv = one/sqrt(rsq)
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
             rinv = one/sqrt(rsq)
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
  end subroutine enb_uv_vdw_dp_fortran

  ! *
  ! * Solvent - Solvent (with Coulomb and Vdw interactions)
  ! *
  subroutine enb_vv(nblist, nb_pair, vdwtype, vdwparam, xyzq, force, sforce, &
       coulpot, vdwpot, scoordtab, nnnb, q_single)
    use chm_kinds
    use consta,only:ccelec
    use inbnd,only:eps
    use number
    use nblist_types,only:nblist_pair_t, dpsp_t, nb_pair_t
    use domdec_local_types,only:xyzq_dpsp_t
    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, SIMD_AVX2, SIMD_AVX2_FMA
    use enb_core_wrapper_mod,only:enb_vv_charmm_dp_sse, enb_vv_charmm_sp_sse, &
         enb_vv_charmm_dp_avx, enb_vv_charmm_sp_avx
    use enb_core_sp,only:enb_vv_charmm_sp_fortran, enb_vv_charmm_sp_sse_old
#if KEY_LJPME==1
    use pmeutil, only: qljpme
#endif
    implicit none
    ! Input / Output
    type(nblist_pair_t), intent(in) :: nblist
    type(nb_pair_t), intent(in) :: nb_pair(:)
    integer, intent(in) :: vdwtype(:)
    type(dpsp_t), intent(in) :: vdwparam
    type(xyzq_dpsp_t), intent(in) :: xyzq
    real(chm_real), intent(inout) ::  force(:), sforce(:), coulpot, vdwpot
    type(dpsp_t), intent(in) :: scoordtab
    integer, intent(inout) :: nnnb
    logical, intent(in) :: q_single
    ! Variables
    integer ii, ia, ja, aa, ivdw
    real(chm_real) qO, qH, qOO, qOH, qHH, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, qscale
#if KEY_LJPME==1
    real(chm_real) c6mult_O, c6mult_H, c6mult_OO, c6mult_OH, c6mult_HH
#endif

#if KEY_BLOCK==1
    if (nblist%lblock) then
       call wrndie(-5,'<enb_core>',&
            'How did we end up here? BLOCK does not have Solvent-Solvent kernels')
    endif
#endif 

    if (nblist%ni == 0) return

    qscale = one !ccelec/eps
    ! All solvent molecules have the same charges and VdW parameters => pre-load them here
    ii = nblist%indi(1)
    if (q_single) then
       qO = real(xyzq%sp(ii)%q, kind=chm_real)
       qH = real(xyzq%sp(ii+1)%q, kind=chm_real)
    else
       qO = xyzq%dp(ii)%q
       qH = xyzq%dp(ii+1)%q
    endif
    qOO = qscale*qO*qO
    qOH = qscale*qO*qH
    qHH = qscale*qH*qH
    ia = vdwtype(ii)
    ivdw = ia*(ia-3) + 2*(ia + ia) - 1
    if (q_single) then
       c6OO = vdwparam%sp(ivdw)
       c12OO = vdwparam%sp(ivdw+1)
    else
       c6OO = vdwparam%dp(ivdw)
       c12OO = vdwparam%dp(ivdw+1)
    endif

    ia = vdwtype(ii)
    ja = vdwtype(ii+1)
    aa = max(ja, ia)
    ivdw = aa*(aa-3) + 2*(ja + ia) - 1
    if (q_single) then
       c6OH = vdwparam%sp(ivdw)
       c12OH = vdwparam%sp(ivdw+1)
    else
       c6OH = vdwparam%dp(ivdw)
       c12OH = vdwparam%dp(ivdw+1)
    endif

    ia = vdwtype(ii+1)
    ivdw = ia*(ia-3) + 2*(ia + ia) - 1
    if (q_single) then
       c6HH = vdwparam%sp(ivdw)
       c12HH = vdwparam%sp(ivdw+1)
    else
       c6HH = vdwparam%dp(ivdw)
       c12HH = vdwparam%dp(ivdw+1)
    endif

#if KEY_LJPME==1
    if (q_single) then
       c6mult_O = real(xyzq%sp(ii)%c6, kind=chm_real)
       c6mult_H = real(xyzq%sp(ii+1)%c6, kind=chm_real)
    else
       c6mult_O = xyzq%dp(ii)%c6
       c6mult_H = xyzq%dp(ii+1)%c6
    endif
    c6mult_OO = c6mult_O*c6mult_O
    c6mult_OH = c6mult_O*c6mult_H
    c6mult_HH = c6mult_H*c6mult_H
#endif

    ! NOTE: CHARMM modified TIP3P model has VdW interactions between O-H and H-H

    if (c6OH == zero .and. c12OH == zero .and. c6HH == zero .and. c12HH == zero) then
       ! Solvent only has O-O interactions
       ! NOT IMPLEMENTED
       call wrndie(-5,'<enb_core>','This solvent-solvent model not implemented')
    else
       ! CHARMM version of TIP3P
       if (q_single) then
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_vv_charmm_sp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, coulpot, &
                  vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, nb_pair(nblist%nb_pair_ind)%ctonnb, &
                  nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_vv_charmm_sp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, coulpot, &
                  vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, nb_pair(nblist%nb_pair_ind)%ctonnb, &
                  nb_pair(nblist%nb_pair_ind)%vdw_type)
!!$             call enb_vv_charmm_sp_sse_old(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
!!$                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
!!$                  xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, coulpot, &
!!$                  vdwpot, nnnb, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, qOO, qOH, qHH, &
!!$                  nb_pair(nblist%nb_pair_ind)%ctofnb)
          else
             call enb_vv_charmm_sp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_sp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%sp, force, nblist%iscoord, sforce, scoordtab%sp, coulpot, &
                  vdwpot, nnnb, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, nb_pair(nblist%nb_pair_ind)%ctofnb)
          endif
       else
          if (simd_version == SIMD_AVX .or. simd_version == SIMD_AVX_FMA .or. &
            simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
             call enb_vv_charmm_dp_avx(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%dp, force, nblist%iscoord, sforce, scoordtab%dp, coulpot, &
                  vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, nb_pair(nblist%nb_pair_ind)%ctonnb, &
                  nb_pair(nblist%nb_pair_ind)%vdw_type)
          elseif (simd_version == SIMD_SSE) then
             call enb_vv_charmm_dp_sse(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%dp, force, nblist%iscoord, sforce, scoordtab%dp, coulpot, &
                  vdwpot, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, &
                  nb_pair(nblist%nb_pair_ind)%ctofnb, nb_pair(nblist%nb_pair_ind)%ctonnb, &
                  nb_pair(nblist%nb_pair_ind)%vdw_type)
          else
             call enb_vv_charmm_dp_fortran(nblist%ni, nblist%indi, nblist%indj, nblist%startj, &
                  nb_pair(nblist%nb_pair_ind)%pftable_dp, one/nb_pair(nblist%nb_pair_ind)%h, &
                  xyzq%dp, force, nblist%iscoord, sforce, scoordtab%dp, coulpot, &
                  vdwpot, nnnb, c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
                  qOO, qOH, qHH, nb_pair(nblist%nb_pair_ind)%ctofnb)
          endif
       endif
    endif

    return
  end subroutine enb_vv

  subroutine force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3, rsq, hinv, &
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
    real(chm_real), intent(in) :: dx1, dy1, dz1
    real(chm_real), intent(in) :: dx2, dy2, dz2
    real(chm_real), intent(in) :: dx3, dy3, dz3
    real(chm_real), intent(inout) :: rsq(3)
    real(chm_real), intent(in) :: hinv, q1, q2, c6_1, c6_2, c12_1, c12_2, pftable(:)
#if KEY_LJPME==1
    real(chm_real), intent(in) :: c6mult_1, c6mult_2
#endif
    real(chm_real) coulpot, vdwpot
    real(chm_real) fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3
    real(chm_real) forcexjj, forceyjj, forcezjj
    ! Variables
    integer tt(3)
    real(chm_real) rs(3)
    real(chm_real) ept(3)
    real(chm_real) rinv(3)
    real(chm_real) a0_1, a1_1, a2_1, a3_1
    real(chm_real) a0_2, a1_2, a2_2, a3_2
    real(chm_real) a0_3, a1_3, a2_3, a3_3
    real(chm_real) fij1, fij2, fij3
    real(chm_real) fijx, fijy, fijz
    real(chm_real) fjx, fjy, fjz
    integer i

    rinv = hinv/sqrt(rsq)

    rs = rsq*rinv
    tt = int(rs)
    ept = rs - tt
#if KEY_LJPME==1
    tt = 16*tt - 15
#else
    tt = 12*tt - 11
#endif

    ! Coulomb interaction
    a0_1 = pftable(tt(1))             ! a0
    a0_2 = pftable(tt(2))             ! a0
    a0_3 = pftable(tt(3))             ! a0
    a1_1 = pftable(tt(1)+1)           ! a1
    a1_2 = pftable(tt(2)+1)           ! a1
    a1_3 = pftable(tt(3)+1)           ! a1

    a2_1 = pftable(tt(1)+2)*ept(1)      ! a2*ep
    a2_2 = pftable(tt(2)+2)*ept(2)      ! a2*ep
    a2_3 = pftable(tt(3)+2)*ept(3)      ! a2*ep

    a3_1 = pftable(tt(1)+3)*ept(1)*ept(1)    ! a3*ep^2
    a3_2 = pftable(tt(2)+3)*ept(2)*ept(2)    ! a3*ep^2
    a3_3 = pftable(tt(3)+3)*ept(3)*ept(3)    ! a3*ep^2

    coulpot = coulpot + q1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept(1))
    coulpot = coulpot + q2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept(2))
    coulpot = coulpot + q2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept(3))

    fij1 = q1*(a1_1 + two*a2_1 + three*a3_1)
    fij2 = q2*(a1_2 + two*a2_2 + three*a3_2)
    fij3 = q2*(a1_3 + two*a2_3 + three*a3_3)

    ! VdW attraction
    a0_1 = pftable(tt(1)+4)             ! a0
    a0_2 = pftable(tt(2)+4)             ! a0
    a0_3 = pftable(tt(3)+4)             ! a0
    a1_1 = pftable(tt(1)+5)           ! a1
    a1_2 = pftable(tt(2)+5)           ! a1
    a1_3 = pftable(tt(3)+5)           ! a1
    
    a2_1 = pftable(tt(1)+6)*ept(1)      ! a2*ep
    a2_2 = pftable(tt(2)+6)*ept(2)      ! a2*ep
    a2_3 = pftable(tt(3)+6)*ept(3)      ! a2*ep
    
    a3_1 = pftable(tt(1)+7)*ept(1)*ept(1)    ! a3*ep^2
    a3_2 = pftable(tt(2)+7)*ept(2)*ept(2)    ! a3*ep^2
    a3_3 = pftable(tt(3)+7)*ept(3)*ept(3)    ! a3*ep^2
    
    vdwpot = vdwpot + c6_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept(1))
    vdwpot = vdwpot + c6_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept(2))
    vdwpot = vdwpot + c6_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept(3))
    
    fij1 = fij1 + c6_1*(a1_1 + two*a2_1 + three*a3_1)
    fij2 = fij2 + c6_2*(a1_2 + two*a2_2 + three*a3_2)
    fij3 = fij3 + c6_2*(a1_3 + two*a2_3 + three*a3_3)

    ! VdW repulsion
    a0_1 = pftable(tt(1)+8)             ! a0
    a0_2 = pftable(tt(2)+8)             ! a0
    a0_3 = pftable(tt(3)+8)             ! a0
    a1_1 = pftable(tt(1)+9)           ! a1
    a1_2 = pftable(tt(2)+9)           ! a1
    a1_3 = pftable(tt(3)+9)           ! a1
    
    a2_1 = pftable(tt(1)+10)*ept(1)      ! a2*ep
    a2_2 = pftable(tt(2)+10)*ept(2)      ! a2*ep
    a2_3 = pftable(tt(3)+10)*ept(3)      ! a2*ep
    
    a3_1 = pftable(tt(1)+11)*ept(1)*ept(1)    ! a3*ep^2
    a3_2 = pftable(tt(2)+11)*ept(2)*ept(2)    ! a3*ep^2
    a3_3 = pftable(tt(3)+11)*ept(3)*ept(3)    ! a3*ep^2
    
    vdwpot = vdwpot + c12_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept(1))
    vdwpot = vdwpot + c12_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept(2))
    vdwpot = vdwpot + c12_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept(3))

    fij1 = fij1 + c12_1*(a1_1 + two*a2_1 + three*a3_1)
    fij2 = fij2 + c12_2*(a1_2 + two*a2_2 + three*a3_2)
    fij3 = fij3 + c12_2*(a1_3 + two*a2_3 + three*a3_3)
#if KEY_LJPME==1
    ! LJPME grid correction, to remove multiplicative terms
    a0_1 = pftable(tt(1)+12)             ! a0
    a0_2 = pftable(tt(2)+12)             ! a0
    a0_3 = pftable(tt(3)+12)             ! a0
    a1_1 = pftable(tt(1)+13)           ! a1
    a1_2 = pftable(tt(2)+13)           ! a1
    a1_3 = pftable(tt(3)+13)           ! a1
    
    a2_1 = pftable(tt(1)+14)*ept(1)      ! a2*ep
    a2_2 = pftable(tt(2)+14)*ept(2)      ! a2*ep
    a2_3 = pftable(tt(3)+14)*ept(3)      ! a2*ep
    
    a3_1 = pftable(tt(1)+15)*ept(1)*ept(1)    ! a3*ep^2
    a3_2 = pftable(tt(2)+15)*ept(2)*ept(2)    ! a3*ep^2
    a3_3 = pftable(tt(3)+15)*ept(3)*ept(3)    ! a3*ep^2

    vdwpot = vdwpot + c6mult_1*(a0_1 + (a1_1 + a2_1 + a3_1)*ept(1))
    vdwpot = vdwpot + c6mult_2*(a0_2 + (a1_2 + a2_2 + a3_2)*ept(2))
    vdwpot = vdwpot + c6mult_2*(a0_3 + (a1_3 + a2_3 + a3_3)*ept(3))

    fij1 = fij1 + c6mult_1*(a1_1 + two*a2_1 + three*a3_1)
    fij2 = fij2 + c6mult_2*(a1_2 + two*a2_2 + three*a3_2)
    fij3 = fij3 + c6mult_2*(a1_3 + two*a2_3 + three*a3_3)
#endif
    ! Calculate final force
    fij1 = fij1*rinv(1)
    fij2 = fij2*rinv(2)
    fij3 = fij3*rinv(3)
    
    ! j X - i O
    fijx = fij1*dx1
    fijy = fij1*dy1
    fijz = fij1*dz1
    
    fix1 = fix1 + fijx
    fiy1 = fiy1 + fijy
    fiz1 = fiz1 + fijz

    fjx = forcexjj
    fjy = forceyjj
    fjz = forcezjj

    fjx = fjx - fijx
    fjy = fjy - fijy
    fjz = fjz - fijz
    
    ! j X - i H1
    fijx = fij2*dx2
    fijy = fij2*dy2
    fijz = fij2*dz2
    
    fix2 = fix2 + fijx
    fiy2 = fiy2 + fijy
    fiz2 = fiz2 + fijz
    
    fjx = fjx - fijx
    fjy = fjy - fijy
    fjz = fjz - fijz
    
    ! j X - i H2
    fijx = fij3*dx3
    fijy = fij3*dy3
    fijz = fij3*dz3
    
    fix3 = fix3 + fijx
    fiy3 = fiy3 + fijy
    fiz3 = fiz3 + fijz
    
    fjx = fjx - fijx
    fjy = fjy - fijy
    fjz = fjz - fijz
    
    forcexjj = fjx
    forceyjj = fjy
    forcezjj = fjz

    return
  end subroutine force_core_vv

  ! *
  ! * Solvent - Solvent, CHARMM version of TIP3P which has O-H and H-H VdW interactions
  ! *
  subroutine enb_vv_charmm_dp_fortran(ni, indi, indj, startj, pftable, hinv, xyzq, &
       force, iscoord, sforce, scoordtab, coulpot, vdwpot, nnnb, &
       c6OO, c12OO, c6OH, c12OH, c6HH, c12HH, &
#if KEY_LJPME==1
       c6mult_OO, c6mult_OH, c6mult_HH, &
#endif
       qOO, qOH, qHH, rc)
    use chm_kinds
    use number
    use consta,only:pi
    use ewald_1m,only:kappa
    use parallel,only:mynod
    implicit none
     ! Input / Output
    integer, intent(in) :: ni, indi(:), startj(:), indj(:)
    real(chm_real), intent(in) :: pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: force(:)
    integer, intent(in) :: iscoord(:)
    real(chm_real), intent(inout) :: sforce(:), coulpot, vdwpot
    real(chm_real), intent(in) :: c6OO, c12OO, c6OH, c12OH, c6HH, c12HH
#if KEY_LJPME==1
    real(chm_real), intent(in) :: c6mult_OO, c6mult_OH, c6mult_HH
#endif
    real(chm_real), intent(in) :: qOO, qOH, qHH, rc
    integer, intent(inout) :: nnnb
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1
    real(chm_real) sx, sy, sz
    real(chm_real) xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3
    real(chm_real) xj, yj, zj
    real(chm_real) fix1, fiy1, fiz1, fix2, fiy2, fiz2, fix3, fiy3, fiz3
    real(chm_real) dx1, dy1, dz1, rsq(3), rc2
    real(chm_real) dx2, dy2, dz2
    real(chm_real) dx3, dy3, dz3
#if KEY_LJPME==1
    real(chm_real) c6i, c6ij
#endif

    rc2 = rc*rc

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
          rsq(1) = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq(2) = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq(3) = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq(1) < rc2 .or. rsq(2) < rc2 .or. rsq(3) < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3,rsq, hinv, &
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
          rsq(1) = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq(2) = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq(3) = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq(1) < rc2 .or. rsq(2) < rc2 .or. rsq(3) < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3,rsq, hinv, &
                  qOH, qHH, c6OH, c6HH, c12OH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OH, c6mult_HH, &
#endif
                  pftable, coulpot, vdwpot, fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, &
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
          rsq(1) = dx1*dx1 + dy1*dy1 + dz1*dz1
          dx2 = xi2 - xj
          dy2 = yi2 - yj
          dz2 = zi2 - zj
          rsq(2) = dx2*dx2 + dy2*dy2 + dz2*dz2
          dx3 = xi3 - xj
          dy3 = yi3 - yj
          dz3 = zi3 - zj
          rsq(3) = dx3*dx3 + dy3*dy3 + dz3*dz3
          if (rsq(1) < rc2 .or. rsq(2) < rc2 .or. rsq(3) < rc2) then
             call force_core_vv(dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3,rsq, hinv, &
                  qOH, qHH, c6OH, c6HH, c12OH, c12HH, &
#if KEY_LJPME==1
                  c6mult_OH, c6mult_HH, &
#endif
                  pftable, coulpot, vdwpot, fix1,fiy1,fiz1, fix2,fiy2,fiz2, fix3,fiy3,fiz3, &
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
  end subroutine enb_vv_charmm_dp_fortran

#if KEY_BLOCK==1 /*block*/
  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! *
  subroutine enb_coul_vdw_block_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, &
       coulpot, vdwpot, nnnb, rc, qscale, loc2glo_ind, iblckp)
    use number,only:zero, one, two, three
    use lambdam,only:msld_nb_scale_enerforce
    use domdec_block,only:biflam_loc, biflam2_loc
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: coulpot, vdwpot
    integer, intent(inout) :: nnnb
    real(chm_real), intent(in) :: rc, qscale
    integer, intent(in) :: loc2glo_ind(:), iblckp(:)
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    integer ii_glo, ii_block, jj_glo, jj_block
    real(chm_real) qi, qq, c6, c12
    real(chm_real) xi, yi, zi
    real(chm_real) sx, sy, sz
    real(chm_real) xj, yj, zj
    real(chm_real) dx, dy, dz
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6i, c6ij
#endif
    real(chm_real) fij_elec, pot_elec, fij_vdw, pot_vdw

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
       ii3 = ii*3-2
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
          ! Atom j global index
          jj_glo = loc2glo_ind(jj)+1
          ! Atom j block index
          jj_block = iblckp(jj_glo)
          jj3 = jj*3-2
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
             rinv = one/sqrt(rsq)
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
             pot_elec = qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_elec = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             pot_vdw = c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = fij_vdw + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = fij_vdw + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
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
             force(jj3)   = force(jj3)   - fijx
             force(jj3+1) = force(jj3+1) - fijy
             force(jj3+2) = force(jj3+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       force(ii3)   = force(ii3)   + fix
       force(ii3+1) = force(ii3+1) + fiy
       force(ii3+2) = force(ii3+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_coul_vdw_block_fortran

  ! *
  ! * Solute-solute with Coulomb and Vdw interactions
  ! * With MSLD soft cores
  ! *
  subroutine enb_coul_vdw_block_softcore_fortran(ni, indi, indj, startj, pftable, hinv, vdwtype, vdwparam, &
       xyzq, force, iscoord, sforce, scoordtab, &
       coulpot, vdwpot, nnnb, rc, qscale, loc2glo_ind, iblckp)
    use number,only:zero, one, two, three
    use lambdam,only:msld_nbsoft_scale_enerforce,fullblcoep
    use domdec_block,only:biflam_loc, biflam2_loc
    implicit none
    ! Input / Output
    integer, intent(in) :: ni, indi(:), indj(:), startj(:), vdwtype(:), iscoord(:)
    real(chm_real), intent(in) :: vdwparam(:), pftable(:), hinv
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(inout) :: coulpot, vdwpot
    integer, intent(inout) :: nnnb
    real(chm_real), intent(in) :: rc, qscale
    integer, intent(in) :: loc2glo_ind(:), iblckp(:)
    ! Variables
    integer i, j, ii, ii3, is, jj, jj3, j0, j1, ia, ja, aa, ivdw, t
    integer ii_glo, ii_block, jj_glo, jj_block
    real(chm_real) qi, qq, c6, c12
    real(chm_real) xi, yi, zi
    real(chm_real) sx, sy, sz
    real(chm_real) xj, yj, zj
    real(chm_real) dx, dy, dz
    real(chm_real) fix, fiy, fiz
    real(chm_real) fij, fijx, fijy, fijz
    real(chm_real) rsq, rinv, rc2, rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
#if KEY_LJPME==1
    real(chm_real) c6i, c6ij
#endif
    real(chm_real) fij_elec, pot_elec, fij_vdw, pot_vdw
    real(chm_real) r, rp, scale, rdivrcs, drpdr, drpds, rcsoft

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
       ii3 = ii*3-2
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
          ! Atom j global index
          jj_glo = loc2glo_ind(jj)+1
          ! Atom j block index
          jj_block = iblckp(jj_glo)
          jj3 = jj*3-2
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
             rinv = one/sqrt(rsq)
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
             pot_elec = qq*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_elec = qq*(a1 + two*a2ep + three*a3ep2)
             ! VdW attraction
             a0 = pftable(t+4)           ! a0
             a1 = pftable(t+5)           ! a1
             a2ep = pftable(t+6)*ep      ! a2*ep
             a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
             pot_vdw = c6*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = c6*(a1 + two*a2ep + three*a3ep2)
             ! VdW repulsion
             a0 = pftable(t+8)           ! a0
             a1 = pftable(t+9)           ! a1
             a2ep = pftable(t+10)*ep     ! a2*ep
             a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + c12*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = fij_vdw + c12*(a1 + two*a2ep + three*a3ep2)
#if KEY_LJPME==1
             ! LJPME grid correction, to remove multiplicative terms
             a0 = pftable(t+12)          ! a0
             a1 = pftable(t+13)          ! a1
             a2ep = pftable(t+14)*ep     ! a2*ep
             a3ep2 = pftable(t+15)*ep2   ! a3*ep^2
             pot_vdw = pot_vdw + c6ij*(a0 + (a1 + a2ep + a3ep2)*ep)
             fij_vdw = fij_vdw + c6ij*(a1 + two*a2ep + three*a3ep2)
#endif
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
             force(jj3)   = force(jj3)   - fijx
             force(jj3+1) = force(jj3+1) - fijy
             force(jj3+2) = force(jj3+2) - fijz
             nnnb = nnnb + 1
          endif
       enddo
       force(ii3)   = force(ii3)   + fix
       force(ii3+1) = force(ii3+1) + fiy
       force(ii3+2) = force(ii3+2) + fiz
       ! Store shifted forces
       sforce(is) = sforce(is) + fix
       sforce(is+1) = sforce(is+1) + fiy
       sforce(is+2) = sforce(is+2) + fiz
    enddo
!$omp end do

    return
  end subroutine enb_coul_vdw_block_softcore_fortran

#endif /* (block)*/

#endif /* (domdec_main)*/

end module enb_core

