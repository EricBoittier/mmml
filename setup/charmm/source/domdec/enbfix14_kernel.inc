#ifdef THOLE14_CALL
  ! *
  ! * Calculate Thole 1-4 exclusions
  ! *
  subroutine KERNEL_NAME(istart, iend, list, xyzq, scoordtab, force, sforce, coulpot)
    implicit none
    ! Input / Output
    integer, intent(in) :: istart, iend
    type(list14thole_t), intent(in) :: list(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:)
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:)
#endif
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(inout) :: coulpot
    ! Variables
    integer ii, i, j, ith, jth, is
#ifdef DOUBLE_PREC
    real(chm_real) sx, sy, sz
    real(chm_real) xi, yi, zi
    real(chm_real) xj, yj, zj
    real(chm_real) xith, yith, zith
    real(chm_real) xjth, yjth, zjth
    real(chm_real) qq, aa
    real(chm_real) rx, ry, rz
#else
    real(chm_real4) sx, sy, sz
    real(chm_real4) xi, yi, zi
    real(chm_real4) xj, yj, zj
    real(chm_real4) xith, yith, zith
    real(chm_real4) xjth, yjth, zjth
    real(chm_real4) qq, aa
    real(chm_real4) rx, ry, rz
#endif
    real(chm_real) dfaa, dfad, dfda, dfdd
    real(chm_real) dxi, dyi, dzi
    real(chm_real) sfx, sfy, sfz

    do ii=istart,iend

       i = list(ii)%i+1
       j = list(ii)%j+1
       ith = list(ii)%ith+1
       jth = list(ii)%jth+1
       is = list(ii)%ishift*3+1

       ! Get shift coordinate
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)

       aa = list(ii)%aa

       xi = xyzq(i)%x + sx
       yi = xyzq(i)%y + sy
       zi = xyzq(i)%z + sz

       xj = xyzq(j)%x
       yj = xyzq(j)%y
       zj = xyzq(j)%z

       xith = xyzq(ith)%x + sx
       yith = xyzq(ith)%y + sy
       zith = xyzq(ith)%z + sz

       xjth = xyzq(jth)%x
       yjth = xyzq(jth)%y
       zjth = xyzq(jth)%z

       qq = xyzq(ith)%q * xyzq(jth)%q !* ccelec
!       QQ = QD1*QD2 * CCELEC

       !           Atom-Atom
       rx = xi-xj
       ry = yi-yj
       rz = zi-zj
       coulpot = coulpot + e_thole(qq,aa,rx,ry,rz, dfaa)
       !           Atom-Drude
       rx = xi-xjth
       ry = yi-yjth
       rz = zi-zjth
       coulpot = coulpot + e_thole(-qq,aa,rx,ry,rz, dfad)
       !           Drude-Atom
       rx = xith-xj
       ry = yith-yj
       rz = zith-zj
       coulpot = coulpot + e_thole(-qq,aa,rx,ry,rz, dfda)
       !           Drude-Drude
       rx = xith-xjth
       ry = yith-yjth
       rz = zith-zjth
       coulpot = coulpot + e_thole(qq,aa,rx,ry,rz, dfdd)

       i = i*3-2
       j = j*3-2
       ith = ith*3-2
       jth = jth*3-2
       !              Atom-Atom
       rx = xi-xj
       ry = yi-yj
       rz = zi-zj
       dxi = rx*dfaa
       dyi = ry*dfaa
       dzi = rz*dfaa
       force(i)   = force(i)   + dxi 
       force(i+1) = force(i+1) + dyi
       force(i+2) = force(i+2) + dzi
       force(j)   = force(j)   - dxi
       force(j+1) = force(j+1) - dyi
       force(j+2) = force(j+2) - dzi
       sfx = dxi
       sfy = dyi
       sfz = dzi
       !              Atom-Drude
       rx = xi-xjth
       ry = yi-yjth
       rz = zi-zjth
       dxi = rx*dfad
       dyi = ry*dfad
       dzi = rz*dfad       
       force(i)     = force(i)     + dxi
       force(i+1)   = force(i+1)   + dyi
       force(i+2)   = force(i+2)   + dzi
       force(jth)   = force(jth)   - dxi
       force(jth+1) = force(jth+1) - dyi
       force(jth+2) = force(jth+2) - dzi
       sfx = sfx + dxi
       sfy = sfy + dyi
       sfz = sfz + dzi
       !              Drude-Atom
       rx = xith-xj
       ry = yith-yj
       rz = zith-zj
       dxi = rx*dfda
       dyi = ry*dfda
       dzi = rz*dfda       
       force(ith)   = force(ith)   + dxi
       force(ith+1) = force(ith+1) + dyi
       force(ith+2) = force(ith+2) + dzi
       force(j)     = force(j)     - dxi
       force(j+1)   = force(j+1)   - dyi
       force(j+2)   = force(j+2)   - dzi
       sfx = sfx + dxi
       sfy = sfy + dyi
       sfz = sfz + dzi
       !              Drude-Drude
       rx = xith-xjth
       ry = yith-yjth
       rz = zith-zjth
       dxi = rx*dfdd
       dyi = ry*dfdd
       dzi = rz*dfdd       
       force(ith)   = force(ith)   + dxi
       force(ith+1) = force(ith+1) + dyi
       force(ith+2) = force(ith+2) + dzi
       force(jth)   = force(jth)   - dxi
       force(jth+1) = force(jth+1) - dyi
       force(jth+2) = force(jth+2) - dzi
       sfx = sfx + dxi
       sfy = sfy + dyi
       sfz = sfz + dzi

       ! Store shift force
       sforce(is)   = sforce(is)   + sfx
       sforce(is+1) = sforce(is+1) + sfy
       sforce(is+2) = sforce(is+2) + sfz
    enddo

    return

  contains

    function e_thole( qq,aa,rx,ry,rz, df) result(e_thole_1)
      implicit none
      !     
      !     returns the screened coulomb interaction.
      !     computes the force and, in option, the second derivative.
      !     
      !use number
      !     arguments
      real(chm_real) e_thole_1
#ifdef DOUBLE_PREC
      real(chm_real),intent(in) ::  qq,aa             ! input
      real(chm_real),intent(in) ::  rx,ry,rz          ! input
#else
      real(chm_real4),intent(in) ::  qq,aa             ! input
      real(chm_real4),intent(in) ::  rx,ry,rz          ! input
#endif
      real(chm_real),intent(out) ::  df                ! output
      ! constants
#ifdef DOUBLE_PREC
      real(chm_real), parameter :: sixth = 1.0_chm_real/6.0_chm_real, one = 1.0_chm_real, &
           half = 0.5_chm_real
#else
      real(chm_real4), parameter :: sixth = 1.0_chm_real4/6.0_chm_real4, one = 1.0_chm_real4, &
           half = 0.5_chm_real4
#endif
      !     locals
#ifdef DOUBLE_PREC
      real(chm_real)  norm,r,au,expau,polyau
#else
      real(chm_real4)  norm,r,au,expau,polyau
#endif
      
      norm = aa**(-sixth)
      r = sqrt(rx*rx+ry*ry+rz*rz)
      au = norm*r
      expau = exp(-au)
      
      !     u = qq/r [1 - (1+au/2)exp(-au)]
      polyau = one+au*half
      e_thole_1 = qq/r * (one-polyau*expau)
      !     (du/dr)/r = qq/r^3 [(1+au+au^2/2)exp(-au) - 1]
      polyau = one+au*(one+half*au)
      df = qq/r**3 * (polyau*expau-one)
      
      return
    end function e_thole

  end subroutine KERNEL_NAME

#endif

#ifdef LJPME14_CALL

  ! *
  ! * Calculate 1-4 exclusions and interactions on CPU
  ! *
  subroutine KERNEL_NAME(qljpme, qvdw, qljexc, &
       in14_start, in14_end, in14list, &
       ex14_start, ex14_end, ex14list, xyzq, scoordtab, vdwtype, vdwparam, fscale, &
       force, sforce, evdw, edexc)
    implicit none
    ! Input / Output
    logical, intent(in) :: qljpme, qvdw, qljexc
    integer, intent(in) :: in14_start, in14_end
    type(list14_t), intent(in) :: in14list(:)
    integer, intent(in) :: ex14_start, ex14_end
    type(list14_t), intent(in) :: ex14list(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(in) :: vdwparam(:)
    real(chm_real), intent(in) :: fscale
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:)
    real(chm_real4), intent(in) :: vdwparam(:)
    real(chm_real4), intent(in) :: fscale
#endif
    integer, intent(in) :: vdwtype(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(inout) :: edexc, evdw
    ! Variables
    logical use_sse
#ifdef DOUBLE_PREC
    real(chm_real) kappa_pp, ctofnb_pp, ctonnb_pp, e14fac_pp, hinv_pp, qq_scale
#else
    real(chm_real4) kappa_pp, ctofnb_pp, ctonnb_pp, e14fac_pp, hinv_pp, qq_scale
#endif

    ! If neccessary, build lookup tables for 1-4 interactions & exclusions
!    call init_nblist(1)

#ifdef DOUBLE_PREC
    kappa_pp = kappa
    ctofnb_pp = inex14_pair(1)%ctofnb
    ctonnb_pp = inex14_pair(1)%ctonnb
    e14fac_pp = e14fac
#else
    kappa_pp = real(kappa, kind=chm_real4)
    ctofnb_pp = real(inex14_pair(1)%ctofnb, kind=chm_real4)
    ctonnb_pp = real(inex14_pair(1)%ctonnb, kind=chm_real4)
    e14fac_pp = real(e14fac, kind=chm_real4)
#endif

    if (simd_version == SIMD_SSE .or. simd_version == SIMD_AVX .or. &
         simd_version == SIMD_AVX_FMA .or. &
         simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
       use_sse = .true.
    else
       use_sse = .false.
    endif

    ! fscale /= 1.0 only works in the Fortran routines
#ifdef DOUBLE_PREC
    if (fscale /= one) use_sse = .false.
#else
    if (fscale /= 1.0_chm_real4) use_sse = .false.
#endif

    !
    ! Calculate 1-4 exclusions for LJPME
    !
    if (.not.q_gpu) then
       ! CPU version
       ! First, the inclusive terms get added in by calling the LJPME exclusion term
       ! with the IN14 lists, accumulating into evdw
#ifdef DOUBLE_PREC
       hinv_pp = 1.0_chm_real/inex14_pair(2)%h
       qq_scale = 1.0_chm_real
#else
       hinv_pp = real(1.0_chm_real/inex14_pair(2)%h, kind=chm_real4)
       qq_scale = 1.0_chm_real4
#endif
       if (qljpme .and. qvdw) then
#ifdef DOUBLE_PREC
          call enb_fix_14_lookup_ljpme14_pd(in14_start, in14_end, in14list, &
               inex14_pair(1)%pftable_dp, &
#else
          call enb_fix_14_lookup_ljpme14_ps(in14_start, in14_end, in14list, &
               inex14_pair(1)%pftable_sp, &
#endif
               hinv_pp, xyzq, scoordtab, fscale, &
               force, sforce, evdw)
       endif

       ! Now the VDW-excluded terms get added in by calling the same LJPME exclusion code
       ! with the EX14 lists, accumulating into dexc
       if (qljpme .and. qljexc) then
#ifdef DOUBLE_PREC
          call enb_fix_14_lookup_ljpme14_pd(ex14_start, ex14_end, ex14list, &
               inex14_pair(2)%pftable_dp, &
#else
          call enb_fix_14_lookup_ljpme14_ps(ex14_start, ex14_end, ex14list, &
               inex14_pair(2)%pftable_sp, &
#endif
               hinv_pp, xyzq, scoordtab, fscale, &
               force, sforce, edexc)
       endif
    else
       call wrndie(-5,'<enbfix14>','LJPME not implemented for GPU yet')
    endif

    return
  end subroutine KERNEL_NAME

#endif 
!
! End of LJPME calls
!

#ifdef EWALD14_CALL

  ! *
  ! * Calculate 1-4 exclusions and interactions on CPU
  ! *
  subroutine KERNEL_NAME(q_vdw, q_ewald, q_ewexcl, in14_start, in14_end, in14list, &
       ex14_start, ex14_end, ex14list, xyzq, scoordtab, vdwtype, vdwparam, vdwparamf, fscale, fscalex, &
       force, sforce, vdwpot, coulpot, excoulpot &
#ifdef SOFT
       , dpotdfs &
#endif
       )
!!$    use number,only:zero, one
!!$    use nblist_types,only:inex14_pair
!!$    use domdec_common,only:simd_version, SIMD_SSE, SIMD_AVX, SIMD_AVX_FMA, &
!!$         SIMD_AVX2, SIMD_AVX2_FMA, q_gpu
!!$    use ewald_1m,only:kappa
!!$    use enbxfast,only:elecmodel,vdwmodel
!!$    use inbnd,only:e14fac
!!$    use psf,only:qdrude
    implicit none
    ! Input / Output
    logical, intent(in) :: q_vdw, q_ewald, q_ewexcl
    integer, intent(in) :: in14_start, in14_end
    type(list14_t), intent(in) :: in14list(:)
    integer, intent(in) :: ex14_start, ex14_end
    type(list14_t), intent(in) :: ex14list(:)
#ifdef DOUBLE_PREC
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: scoordtab(:)
    real(chm_real), intent(in) :: vdwparam(:),vdwparamf(:)
    real(chm_real), intent(in) :: fscale
    real(chm_real), intent(in) :: fscalex
#else
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: scoordtab(:)
    real(chm_real4), intent(in) :: vdwparam(:),vdwparamf(:)
    real(chm_real4), intent(in) :: fscale
    real(chm_real4), intent(in) :: fscalex
#endif
    integer, intent(in) :: vdwtype(:)
    real(chm_real), intent(inout) :: force(:), sforce(:)
    real(chm_real), intent(inout) :: vdwpot, coulpot, excoulpot
    real(chm_real) ::  ipsvdwpot,ipscoulpot
#ifdef SOFT
    real(chm_real), intent(inout) :: dpotdfs
#endif
    ! Variables
    logical use_sse, q_ips
#ifdef DOUBLE_PREC
    real(chm_real) kappa_pp, ctofnb_pp, ctonnb_pp, e14fac_pp, hinv_pp, qq_scale
#else
    real(chm_real4) kappa_pp, ctofnb_pp, ctonnb_pp, e14fac_pp, hinv_pp, qq_scale
#endif

    ! If neccessary, build lookup tables for 1-4 interactions & exclusions
!    call init_nblist(1)

#ifdef DOUBLE_PREC
    kappa_pp = kappa
    ctofnb_pp = inex14_pair(1)%ctofnb
    ctonnb_pp = inex14_pair(1)%ctonnb
    e14fac_pp = e14fac
#else
    kappa_pp = real(kappa, kind=chm_real4)
    ctofnb_pp = real(inex14_pair(1)%ctofnb, kind=chm_real4)
    ctonnb_pp = real(inex14_pair(1)%ctonnb, kind=chm_real4)
    e14fac_pp = real(e14fac, kind=chm_real4)
#endif

    if (simd_version == SIMD_SSE .or. simd_version == SIMD_AVX .or. &
         simd_version == SIMD_AVX_FMA .or. &
         simd_version == SIMD_AVX2 .or. simd_version == SIMD_AVX2_FMA) then
       use_sse = .true.
    else
       use_sse = .false.
    endif

    ! fscale /= 1.0 only works in the Fortran routines
#ifdef DOUBLE_PREC
    if (fscale /= one) use_sse = .false.
    if (fscalex /= one) use_sse = .false.
#else
    if (fscale /= 1.0_chm_real4) use_sse = .false.
    if (fscalex /= 1.0_chm_real4) use_sse = .false.
#endif

    !
    ! NOTE about passing Fortran pointers to C
    !
    ! In passing the pointer vdwtype to the C -language functions, we can not pass it as
    ! integer(c_int) :: vdwtype(*), but must instead do one of the these:
    !
    ! (1) pass it as c_loc(vdwtype(1)) with interface type(c_ptr),
    !     which then means that the C function reads it as **int
    !
    ! (2) pass it as vdwtype(1) with interface integer(c_int) :: vdwtype
    !     WITHOUT THE (*)  !!! then the C function reads it as *int
    !
    ! I choose to use option (2). It's not pretty but should work OK.
    !
    ! Same goes for xyzq, which is also a pointer
    q_ips=(elecmodel==EIPS .or. vdwmodel==VIPS)
    !
    ! Calculate 1-4 interactions
    !
    if (.not.q_gpu) then
       ! CPU version
       if (q_vdw .or. q_ewald) then
         if(q_ips) then
#ifndef SOFT
#ifdef DOUBLE_PREC
          hinv_pp = 1.0_chm_real/inex14_pair(1)%h
          call enb_fix_14_lookup_in14ips_pd(in14_start, in14_end, in14list, &
               vdwtype, vdwparamf, vdwparam, inex14_pair(1)%pftable_dp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot)
#else
          hinv_pp = real(1.0_chm_real/inex14_pair(1)%h, kind=chm_real4)
          call enb_fix_14_lookup_in14ips_ps(in14_start, in14_end, in14list, &
               vdwtype, vdwparamf, vdwparam, inex14_pair(1)%pftable_sp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot)
#endif
#else
#ifdef DOUBLE_PREC
          call wrndie(-5,'<ewald14_soft_pd>','No soft-core function written for 1-4 ips interactions')
#else
          call wrndie(-5,'<ewald14_soft_ps>','No soft-core function written for 1-4 ips interactions')
#endif
#endif
         else
#ifndef SOFT
#ifdef DOUBLE_PREC
          hinv_pp = 1.0_chm_real/inex14_pair(1)%h
          call enb_fix_14_lookup_in14_pd(in14_start, in14_end, in14list, &
               vdwtype, vdwparam, inex14_pair(1)%pftable_dp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot)
#else
          hinv_pp = real(1.0_chm_real/inex14_pair(1)%h, kind=chm_real4)
          call enb_fix_14_lookup_in14_ps(in14_start, in14_end, in14list, &
               vdwtype, vdwparam, inex14_pair(1)%pftable_sp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot)
#endif
#else
#ifdef DOUBLE_PREC
          hinv_pp = 1.0_chm_real/inex14_pair(1)%h
          call enb_fix_14_lookup_in14_soft_pd(in14_start, in14_end, in14list, &
               vdwtype, vdwparam, inex14_pair(1)%pftable_dp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot, dpotdfs)
#else
          hinv_pp = real(1.0_chm_real/inex14_pair(1)%h, kind=chm_real4)
          call enb_fix_14_lookup_in14_soft_ps(in14_start, in14_end, in14list, &
               vdwtype, vdwparam, inex14_pair(1)%pftable_sp, hinv_pp, &
               xyzq, scoordtab, fscale, force, sforce, &
               vdwpot, coulpot, dpotdfs)
#endif
#endif
         endif
       endif
    else
       !! RLH 2018-08-23 Pretty sure this entire branch is never used anymore
       !! Because this function is only called when g_gpu is false.
       !! Recommend deleting. NOTE2JOSH
       !! Also, I'm highly suspicious since only SINGLE_PREC and not double
       !! are dealt with here. Maybe it was still in the process of being
       !! written?
#ifdef SINGLE_PREC
       ! GPU version
       qq_scale = 332.0716_chm_real4
       if (q_vdw .and. q_ewald) then
          if (vdwmodel == VSH) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vsh_ewald(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vsh_ewald_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          elseif (vdwmodel == VSW) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vsw_ewald(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vsw_ewald_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          elseif (vdwmodel == VFSW) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vfsw_ewald(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vfsw_ewald_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          else
             call wrndie(-5,'<enbfix14>','vdwmodel not implemented (1)')
          endif
       elseif (q_vdw) then
          if (vdwmodel == VSH) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vsh_none(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vsh_none_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          elseif (vdwmodel == VSW) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vsw_none(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vsw_none_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          elseif (vdwmodel == VFSW) then
             if (use_sse) then
                call enb_fix_14_sse_in14_vfsw_none(in14_start, in14_end, in14list, &
                     vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                     vdwpot, coulpot)
             else
                call enb_fix_14_in14_vfsw_none_ps(in14_start, in14_end, in14list, &
                     vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                     kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                     vdwpot, coulpot)
             endif
          else
             call wrndie(-5,'<enbfix14>','vdwmodel not implemented (2)')
          endif
       elseif (q_ewald) then
          if (use_sse) then
             call enb_fix_14_sse_in14_none_ewald(in14_start, in14_end, in14list, &
                  vdwtype(1), vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                  kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                  vdwpot, coulpot)
          else
             call enb_fix_14_in14_none_ewald_ps(in14_start, in14_end, in14list, &
                  vdwtype, vdwparam, ctonnb_pp, ctofnb_pp, e14fac_pp, &
                  kappa_pp, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
                  vdwpot, coulpot)
          endif
       endif
#endif
    endif

    !
    ! Calculate 1-4 exclusions
    !
    if (.not.q_gpu) then
       ! CPU version
             if (q_ips) then
                ipsvdwpot=zero
                ipscoulpot=zero
#ifdef DOUBLE_PREC
                hinv_pp = 1.0_chm_real/inex14_pair(2)%h
                call enb_fix_14_lookup_ex14ips_pd(ex14_start, ex14_end, ex14list, &
                     vdwtype,  vdwparamf, inex14_pair(2)%pftable_dp, hinv_pp, &
                     xyzq, scoordtab, fscalex, force, sforce, &
                     ipsvdwpot, ipscoulpot)
#else
                hinv_pp = real(1.0_chm_real/inex14_pair(2)%h, kind=chm_real4)
                call enb_fix_14_lookup_ex14ips_ps(ex14_start, ex14_end, ex14list, &
                     vdwtype,  vdwparamf, inex14_pair(2)%pftable_sp, hinv_pp, &
                     xyzq, scoordtab, fscalex, force, sforce, &
                     ipsvdwpot, ipscoulpot)
#endif
                 vdwpot=vdwpot+ipsvdwpot
                 coulpot=coulpot+ipscoulpot
              else
       if (q_ewexcl .and. q_ewald) then
          if (qdrude) then
             qq_scale = 1.0_chm_real
#ifdef DOUBLE_PREC
             call enb_fix_14_ex14_ewald_smallr_pd(&
#else
             call enb_fix_14_ex14_ewald_smallr_ps(&
#endif
                  ex14_start, ex14_end, ex14list, &
                  kappa_pp, qq_scale, xyzq, scoordtab, fscalex, force, sforce, &
                  excoulpot)
          else
#ifdef DOUBLE_PREC
                 hinv_pp = 1.0_chm_real/inex14_pair(2)%h
#else
                 hinv_pp = real(1.0_chm_real/inex14_pair(2)%h, kind=chm_real4)
#endif
#ifdef DOUBLE_PREC
                 call enb_fix_14_lookup_ex14_pd(ex14_start, ex14_end, ex14list, &
                      inex14_pair(2)%pftable_dp, &
#else
                 call enb_fix_14_lookup_ex14_ps(ex14_start, ex14_end, ex14list, &
                      inex14_pair(2)%pftable_sp, &
#endif
                 hinv_pp, xyzq, scoordtab, fscalex, &
                 force, sforce, excoulpot)
          endif
       endif
              endif
    else
#ifdef SINGLE_PREC
       ! GPU version
       if (q_ewexcl .and. q_ewald) then
          qq_scale = 332.0716_chm_real4
          if (use_sse) then
             call enb_fix_14_sse_ex14_none_ewald(ex14_start, ex14_end, ex14list, &
                  kappa_pp, qq_scale, xyzq(1), scoordtab, force, sforce, &
                  excoulpot)
          else
#ifdef DOUBLE_PREC
             call enb_fix_14_ex14_ewald_pd(&
#else
             call enb_fix_14_ex14_ewald_ps(&
#endif
                  ex14_start, ex14_end, ex14list, &
                  kappa_pp, qq_scale, xyzq, scoordtab, fscalex, force, sforce, &
                  excoulpot)
          endif
       endif
#endif
    endif
    return
  end subroutine KERNEL_NAME

#endif

#ifdef LOOKUP /* lookup */

  subroutine KERNEL_NAME(ii_start, ii_end, xx14list, &
#ifdef IN14
       vdwtype, vdwparam, &
#endif
#ifdef IPSIN
       vdwparamf, &
#endif
       pftable, hinv, &
       xyzq, scoordtab, fscale, force, sforce, &
#ifdef IN14
       vdwpot, &
#endif
       coulpot &
#ifdef SOFT
       , dpotdfs &
#endif
       )
!!$#ifdef DOUBLE_PREC
!!$    use domdec_local_types,only:xyzq_dp_t
!!$#else
!!$    use domdec_local_types,only:xyzq_sp_t
!!$#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: ii_start, ii_end
    type(list14_t), intent(in) :: xx14list(:)
#ifdef IN14
    integer, intent(in) :: vdwtype(:)
#endif
#ifdef DOUBLE_PREC
#ifdef IN14
    real(chm_real), intent(in) :: vdwparam(:)
#endif
#ifdef IPSIN
    real(chm_real), intent(in) :: vdwparamf(:)
#endif
    real(chm_real), intent(in) :: pftable(*), hinv
    real(chm_real), intent(in) :: scoordtab(:)
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: fscale
#else
#ifdef IN14
    real(chm_real4), intent(in) :: vdwparam(:)
#endif
#ifdef IPSIN
    real(chm_real4), intent(in) :: vdwparamf(:)
#endif
    real(chm_real4), intent(in) :: pftable(*), hinv
    real(chm_real4), intent(in) :: scoordtab(:)
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: fscale
#endif
    real(chm_real), intent(inout) :: force(:), sforce(:)
#ifdef IN14
    real(chm_real), intent(inout) :: vdwpot
#endif
    real(chm_real), intent(inout) :: coulpot
#ifdef SOFT
    real(chm_real), intent(inout) :: dpotdfs
#endif
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: one = 1.0_chm_real, &
         two = 2.0_chm_real, three = 3.0_chm_real, six = 6.0_chm_real, &
         zero = 0.0_chm_real,twelve = 12.0_chm_real
#else
    real(chm_real4), parameter :: one = 1.0_chm_real4, &
         two = 2.0_chm_real4, three = 3.0_chm_real4, six = 6.0_chm_real4, &
         zero = 0.0_chm_real4, twelve = 12.0_chm_real4
#endif
    real(chm_real), parameter :: zerod = 0.0_chm_real
    ! Variables
    integer ii, jj, i, j, jj_start, jj_end, is, t
#ifdef IN14
    integer ia, ja, aa, ivdw
#endif
#ifdef DOUBLE_PREC
    real(chm_real) xi, yi, zi, qi
    real(chm_real) xj, yj, zj, qj
    real(chm_real) sx, sy, sz
    real(chm_real) dx, dy, dz
    real(chm_real) qq
    real(chm_real) rsq, r, rinv
#ifdef SOFT
    real(chm_real) rp, rcsoft
#endif
#ifdef IN14
    real(chm_real) c6, c12
#endif
#ifdef IPSIN
    real(chm_real) c6f, c12f, r6inv
#endif
    real(chm_real) rs, ep, ep2
    real(chm_real) a0, a1, a2ep, a3ep2
    real(chm_real) fij
#else
    real(chm_real4) xi, yi, zi, qi
    real(chm_real4) xj, yj, zj, qj
    real(chm_real4) sx, sy, sz
    real(chm_real4) dx, dy, dz
    real(chm_real4) qq
    real(chm_real4) rsq, r, rinv
#ifdef SOFT
    real(chm_real4) rp, rcsoft
#endif
#ifdef IN14
    real(chm_real4) c6, c12
#endif
#ifdef IPSIN
    real(chm_real4) c6f, c12f, r6inv
#endif
    real(chm_real4) rs, ep, ep2
    real(chm_real4) a0, a1, a2ep, a3ep2
    real(chm_real4) fij
#endif
#ifdef SOFT
    real(chm_real) rdivrcs, drpdr, drpds
#endif
    real(chm_real) fijx, fijy, fijz, fix, fiy, fiz

    do ii=ii_start,ii_end
       i = xx14list(ii)%i+1
       j = xx14list(ii)%j+1
       xi = xyzq(i)%x
       yi = xyzq(i)%y
       zi = xyzq(i)%z
#ifdef DISPERSION_TERM
       qi = xyzq(i)%c6
#else
       qi = xyzq(i)%q
#endif
#ifdef IN14
       ia = vdwtype(i)
#endif

       is = xx14list(ii)%ishift*3+1

       ! Get shift coordinate
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Load j atom coordinates
       xj = xyzq(j)%x
       yj = xyzq(j)%y
       zj = xyzq(j)%z
#ifdef DISPERSION_TERM
       qj = xyzq(j)%c6
#else
       qj = xyzq(j)%q
#endif
       ! Calculate distance
       dx = xi - xj + sx
       dy = yi - yj + sy
       dz = zi - zj + sz
       rsq = dx*dx + dy*dy + dz*dz
       qq = qi*qj
       ! Calculate the interaction
       r = sqrt(rsq)
       rinv = one/r

#ifdef IN14
       ! 1-4 interaction
       ja = vdwtype(j)
       aa = max(ja, ia)
       ivdw = aa*(aa-3) + 2*(ja + ia) - 1
       c6 = vdwparam(ivdw)
       c12 = vdwparam(ivdw+1)
#endif

#ifdef SOFT
       rcsoft = 2.0*sqrt(4.0)*(1.0-fscale)
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
#endif

       ! Calculate table index
       !rinv = one/sqrt(rsq)
#ifdef SOFT
       rs = rp*hinv
#else
       rs = rsq*rinv*hinv
#endif
       t = int(rs)
       ep = rs - t
       ep2 = ep*ep
#if KEY_LJPME==1
       t = 16*t - 15
#else
       t = 12*t - 11
#endif

#ifdef DISPERSION_TERM
       ! Dispersion interaction
       a0 = pftable(t+12)             ! a0
       a1 = pftable(t+13)           ! a1
       a2ep = pftable(t+14)*ep      ! a2*ep
       a3ep2 = pftable(t+15)*ep2    ! a3*ep^2
#else
       ! Coulomb interaction
       a0 = pftable(t)             ! a0
       a1 = pftable(t+1)           ! a1
       a2ep = pftable(t+2)*ep      ! a2*ep
       a3ep2 = pftable(t+3)*ep2    ! a3*ep^2
#endif
       coulpot = coulpot + qq*(a0 + (a1 + a2ep + a3ep2)*ep)
       fij = qq*(a1 + two*a2ep + three*a3ep2)
#ifdef IN14
       ! VdW attraction
       a0 = pftable(t+4)           ! a0
       a1 = pftable(t+5)           ! a1
       a2ep = pftable(t+6)*ep      ! a2*ep
       a3ep2 = pftable(t+7)*ep2    ! a3*ep^2
       vdwpot = vdwpot + real(c6*(a0 + (a1 + a2ep + a3ep2)*ep), kind=chm_real)
       fij = fij + c6*(a1 + two*a2ep + three*a3ep2)
       ! VdW repulsion
       a0 = pftable(t+8)           ! a0
       a1 = pftable(t+9)           ! a1
       a2ep = pftable(t+10)*ep     ! a2*ep
       a3ep2 = pftable(t+11)*ep2   ! a3*ep^2
       vdwpot = vdwpot + real(c12*(a0 + (a1 + a2ep + a3ep2)*ep), kind=chm_real)
       fij = fij + c12*(a1 + two*a2ep + three*a3ep2)
#ifdef IPSIN
       c6f = vdwparamf(ivdw)
       c12f = vdwparamf(ivdw+1)
       r6inv=one/rsq/rsq/rsq
       vdwpot = vdwpot + real((c12f*r6inv-c6f)*r6inv, kind=chm_real)
       fij = fij - six*(two*c12f*r6inv-c6f)*r6inv*rinv/hinv
#endif
#endif
#ifdef SOFT
       dpotdfs = dpotdfs + fscale*fij*drpds
#endif
       ! Calculate final force
       fij = fij*hinv*rinv*fscale
#ifdef SOFT
       fij = fij*drpdr
#endif

       ! Calculate force components
       fijx = real(fij*dx,kind=chm_real)
       fijy = real(fij*dy,kind=chm_real)
       fijz = real(fij*dz,kind=chm_real)

       j = j*3-2

       ! Store forces
       force(j)   = force(j)   - fijx
       force(j+1) = force(j+1) - fijy
       force(j+2) = force(j+2) - fijz

       i = i*3-2
       force(i)   = force(i)   + fijx
       force(i+1) = force(i+1) + fijy
       force(i+2) = force(i+2) + fijz

       ! Store shifted forces
       sforce(is)   = sforce(is)   + fijx
       sforce(is+1) = sforce(is+1) + fijy
       sforce(is+2) = sforce(is+2) + fijz

    enddo

    return
  end subroutine KERNEL_NAME
#endif /* lookup */

#ifdef MATH_EVAL /* math_eval */
  ! *
  ! * Fixes 1-4 interactions / exclusions using an all-pairs list
  ! *
  ! * 1-4 interactions: vdwpot, coulpot
  ! * 1-4 exclusions:   coulpot
  ! * 
  ! * NOTE: 1-4 interactions and exclusions are divided among the nodes in
  ! *       non-bonded list build (see subroutine ns_xfast() in nblist_builder.src)
  ! *
  subroutine KERNEL_NAME(ii_start, ii_end, xx14list, &
#ifdef IN14
       vdwtype, vdwparam, ron, roff, e14fac, &
#endif
       kappa, qq_scale, xyzq, scoordtab, fscale, force, sforce, &
#ifdef IN14
       vdwpot, &
#endif
       coulpot)
!!$#ifdef DOUBLE_PREC
!!$    use domdec_local_types,only:xyzq_dp_t
!!$#else
!!$    use domdec_local_types,only:xyzq_sp_t
!!$#endif
    implicit none
    ! Input / Output
    integer, intent(in) :: ii_start, ii_end
    type(list14_t), intent(in) :: xx14list(:)
#ifdef IN14
    integer, intent(in) :: vdwtype(:)
#endif
#ifdef DOUBLE_PREC
#ifdef IN14
    real(chm_real), intent(in) :: vdwparam(:)
    real(chm_real), intent(in) :: ron, roff, e14fac
#endif
    real(chm_real), intent(in) :: kappa, qq_scale
    real(chm_real), intent(in) :: scoordtab(:)
    type(xyzq_dp_t), intent(in) :: xyzq(:)
    real(chm_real), intent(in) :: fscale
#else
#ifdef IN14
    real(chm_real4), intent(in) :: vdwparam(:)
    real(chm_real4), intent(in) :: ron, roff, e14fac
#endif
    real(chm_real4), intent(in) :: kappa, qq_scale
    real(chm_real4), intent(in) :: scoordtab(:)
    type(xyzq_sp_t), intent(in) :: xyzq(:)
    real(chm_real4), intent(in) :: fscale
#endif
    real(chm_real), intent(inout) :: force(:), sforce(:)
#ifdef IN14
    real(chm_real), intent(inout) :: vdwpot
#endif
    real(chm_real), intent(inout) :: coulpot
    ! Parameters
#ifdef DOUBLE_PREC
    real(chm_real), parameter :: one = 1.0_chm_real, &
         two = 2.0_chm_real, three = 3.0_chm_real, six = 6.0_chm_real, &
         twelve = 12.0_chm_real, two_sqrtpi = 1.12837916709551_chm_real, &
         zero = 0.0_chm_real, min_two = -2.0_chm_real, rsmall = 1.0e-10_chm_real, &
         rsqpi2 = 2.0*5.6418958354775628695e-1_chm_real
#else
    real(chm_real4), parameter :: one = 1.0_chm_real4, &
         two = 2.0_chm_real4, three = 3.0_chm_real4, six = 6.0_chm_real4, &
         twelve = 12.0_chm_real4, two_sqrtpi = 1.12837916709551_chm_real4, &
         zero = 0.0_chm_real4, min_two = -2.0_chm_real4, rsmall = 1.0e-4_chm_real4, &
         rsqpi2 = 2.0*5.6418958354775628695e-1_chm_real4
#endif
    real(chm_real), parameter :: zerod = 0.0_chm_real
    ! Variables
    integer ii, jj, i, j, jj_start, jj_end, is
#ifdef IN14
    integer ia, ja, aa, ivdw
#endif
#ifdef DOUBLE_PREC /* DOUBLE_PREC */
    real(chm_real) xi, yi, zi, qi
    real(chm_real) xj, yj, zj, qj
    real(chm_real) sx, sy, sz
    real(chm_real) dx, dy, dz
    real(chm_real) qq
    real(chm_real) r2, r, rinv
    real(chm_real) erfc_val, exp_val
#ifdef IN14 /* opt14 */
    real(chm_real) rinv2
    real(chm_real) c6, c12
    real(chm_real) e14fac_minone
#ifdef VDW_VSH
    real(chm_real) r6, rinv6, rinv12
    real(chm_real) roff2, roff6, roff12, roff18
#endif
#ifdef VDW_VSW
    real(chm_real) rinv4, rinv6, sw, dsw_6
    real(chm_real) roff2, ron2, inv_roff2_ron2, roff2_r2_sq
#endif
#ifdef VDW_VFSW
    real(chm_real) rinv3, rinv6, rinv3_B6_sq, rinv6_B12_sq
    real(chm_real) ron2, ron3, ron6, roff3, roff6
    real(chm_real) A6, B6, A12, B12, CC6, CC12, k6, k12, dv6, dv12, roffinv3, roffinv6
#endif
#endif /* opt14 */
    real(chm_real) fij, fij_vdw, fij_elec
    real(chm_real) kappasq, kappa_fac
    real(chm_real) qq_efac_rinv
#else /* DOUBLE_PREC */
    real(chm_real4) xi, yi, zi, qi
    real(chm_real4) xj, yj, zj, qj
    real(chm_real4) sx, sy, sz
    real(chm_real4) dx, dy, dz
    real(chm_real4) qq
    real(chm_real4) r2, r, rinv
    real(chm_real4) erfc_val, exp_val
#ifdef IN14 /* opt14 */
    real(chm_real4) rinv2
    real(chm_real4) c6, c12
    real(chm_real4) e14fac_minone
#ifdef VDW_VSH
    real(chm_real4) r6, rinv6, rinv12
    real(chm_real4) roff2, roff6, roff12, roff18
#endif
#ifdef VDW_VSW
    real(chm_real4) rinv4, rinv6, sw, dsw_6
    real(chm_real4) roff2, ron2, inv_roff2_ron2, roff2_r2_sq
#endif
#ifdef VDW_VFSW
    real(chm_real4) rinv3, rinv6, rinv3_B6_sq, rinv6_B12_sq
    real(chm_real4) ron2, ron3, ron6, roff3, roff6
    real(chm_real4) A6, B6, A12, B12, CC6, CC12, k6, k12, dv6, dv12, roffinv3, roffinv6
#endif
#endif /* opt14 */
    real(chm_real4) fij, fij_vdw, fij_elec
    real(chm_real4) kappasq, kappa_fac
    real(chm_real4) qq_efac_rinv
#endif /* DOUBLE_PREC */
    real(chm_real) fijx, fijy, fijz, fix, fiy, fiz

#ifdef IN14
#ifdef VDW_VSH
    roff2 = roff*roff
    roff6 = one/(roff2*roff2*roff2)
    roff12 = roff6*roff6
    roff18 = roff12*roff6
#endif
#ifdef VDW_VSW
    roff2 = roff*roff
    ron2 = ron*ron
    inv_roff2_ron2 = one/((roff2 - ron2)**3)
#endif
#ifdef VDW_VFSW
    ron2 = ron*ron
    ron3 = ron*ron*ron
    ron6 = ron3*ron3
    roff3 = roff*roff*roff
    roff6 = roff3*roff3
    if (ron < roff) then
       k6 = roff3/(roff3 - ron3)
       k12 = roff6/(roff6 - ron6)
       dv6 = -one/(ron3*roff3)
       dv12 = -one/(ron6*roff6)
    else
       k6 = one
       k12 = one
       dv6 = -one/(roff6)
       dv12 = -one/(roff6*roff6)
    endif
    roffinv3 =  one/roff3;
    roffinv6 =  one/roff6;
#endif
#endif

    kappasq = kappa*kappa
    kappa_fac = two_sqrtpi*kappa
#ifdef IN14
    e14fac_minone = e14fac - one
#endif

    do ii=ii_start,ii_end
       i = xx14list(ii)%i+1
       j = xx14list(ii)%j+1
       xi = xyzq(i)%x
       yi = xyzq(i)%y
       zi = xyzq(i)%z
       qi = xyzq(i)%q
#ifdef IN14
       ia = vdwtype(i)
#endif

       is = xx14list(ii)%ishift*3+1

       ! Get shift coordinate
       sx = scoordtab(is)
       sy = scoordtab(is+1)
       sz = scoordtab(is+2)
       ! Load j atom coordinates
       xj = xyzq(j)%x
       yj = xyzq(j)%y
       zj = xyzq(j)%z
       qj = xyzq(j)%q
       ! Calculate distance
       dx = xi - xj + sx
       dy = yi - yj + sy
       dz = zi - zj + sz
       r2 = dx*dx + dy*dy + dz*dz
       qq = qq_scale*qi*qj
       ! Calculate the interaction
       r = sqrt(r2)
       rinv = one/r
       erfc_val = erfc(kappa*r)
       exp_val = exp(-kappasq*r2)

#ifdef IN14 /* opt14 */
       ! 1-4 interaction
       ja = vdwtype(j)
       aa = max(ja, ia)
       ivdw = aa*(aa-3) + 2*(ja + ia) - 1
       c6 = vdwparam(ivdw)
       c12 = vdwparam(ivdw+1)

       rinv2 = rinv*rinv
#ifdef VDW_VSH
       r6 = r2*r2*r2
       rinv6 = rinv2*rinv2*rinv2
       rinv12 = rinv6*rinv6

       vdwpot = vdwpot + real(c12*(rinv12 + r6*two*roff18 - three*roff12) - &
            c6*(rinv6 + r6*roff12 - two*roff6), kind=chm_real)

       fij_vdw = six*c6*(rinv6 - r6*roff12) - twelve*c12*(rinv12 + r6*roff18)
#endif
#ifdef VDW_VSW
       roff2_r2_sq = (roff2 - r2)**2;

       sw = one
       if (r2 > ron2) then
          sw = roff2_r2_sq*(roff2 + two*r2 - three*ron2)*inv_roff2_ron2
       endif
       
       dsw_6 = zero
       if (r2 > ron2) then
          dsw_6 = (roff2 - r2)*(ron2 - r2)*inv_roff2_ron2
       endif
       
       rinv4 = rinv2*rinv2
       rinv6 = rinv2*rinv2*rinv2

       vdwpot = vdwpot + real(sw*rinv6*(c12*rinv6 - c6), kind=chm_real)
       
       fij_vdw = rinv4*( twelve*c12*rinv6*(dsw_6 - sw*rinv2) - six*c6*(two*dsw_6 - sw*rinv2) )

#endif
#ifdef VDW_VFSW
       !call wrndie(-5,'<enbfix14>','vdw_vfsw not implemented yet')

       rinv3 = rinv*rinv2
       rinv6 = rinv3*rinv3

       A6 = one
       if (r2 > ron2) then
          A6 = k6
       endif

       A12 = one
       if (r2 > ron2) then
          A12 = k12
       endif

       B6 = zero
       if (r2 > ron2) then
          B6 = roffinv3
       endif

       B12 = zero
       if (r2 > ron2) then
          B12 = roffinv6
       endif

       fij_vdw = six*c6*A6*(rinv3 - B6)*rinv3 - twelve*c12*A12*(rinv6 - B12)*rinv6

       CC6 = dv6
       if (r2 > ron2) then
          CC6 = zero
       endif

       CC12 = dv12
       if (r2 > ron2) then
          CC12 = zero
       endif

       rinv3_B6_sq = (rinv3 - B6)**2

       rinv6_B12_sq = (rinv6 - B12)**2

       vdwpot = vdwpot + real(c12*(A12*rinv6_B12_sq+CC12) - c6*(A6*rinv3_B6_sq+CC6), kind=chm_real)

#endif
#ifdef VDW_NONE
       fij_vdw = zero
#endif

#ifdef ELEC_EWALD
       qq_efac_rinv = qq*(erfc_val + e14fac_minone)*rinv
       coulpot = coulpot + qq_efac_rinv
       fij_elec = -qq*kappa_fac*exp_val - qq_efac_rinv
#endif
#ifdef ELEC_NONE
       fij_elec = zero
#endif

       fij = (fij_vdw + fij_elec)*rinv2

#else /* opt14 */
       ! 1-4 exclusion
#ifdef SMALLR
       if (r2 < rsmall) then
          qq_efac_rinv = -qq*kappa*rsqpi2
          fij = zero
       else
#endif
          qq_efac_rinv = qq*(erfc_val - one)*rinv
          fij = (-qq*kappa_fac*exp_val - qq_efac_rinv)*rinv*rinv
#ifdef SMALLR
       endif
#endif
       coulpot = coulpot + qq_efac_rinv
#endif /* opt14 */

       fij = fij*fscale

       ! Calculate force components
       fijx = real(fij*dx,kind=chm_real)
       fijy = real(fij*dy,kind=chm_real)
       fijz = real(fij*dz,kind=chm_real)

       j = j*3-2

       ! Store forces
       force(j)   = force(j)   - fijx
       force(j+1) = force(j+1) - fijy
       force(j+2) = force(j+2) - fijz

       i = i*3-2
       force(i)   = force(i)   + fijx
       force(i+1) = force(i+1) + fijy
       force(i+2) = force(i+2) + fijz

       ! Store shifted forces
       sforce(is)   = sforce(is)   + fijx
       sforce(is+1) = sforce(is+1) + fijy
       sforce(is+2) = sforce(is+2) + fijz

    enddo

    return
  end subroutine KERNEL_NAME

#endif /* math_eval */
